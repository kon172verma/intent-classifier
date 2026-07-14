#!/usr/bin/env python3
"""
finetune_LoRA/src/lora_validate.py
====================================
Post-training evaluation of a LoRA adapter on val or test split.

Adapter loading priority
------------------------
1. HuggingFace Hub (default):
     PeftModel.from_pretrained(base, HF_HUB_REPO, subfolder="LoRA/{run_tag}")
2. Local fallback (--local flag):
     PeftModel.from_pretrained(base, finetune_LoRA/adapters/{run_tag}/)

Outputs
-------
  reports_validation/{run_tag}_{split}_{ts}.json
  reports_test/      {run_tag}_{split}_{ts}.json

Usage
-----
    python lora_validate.py --model qwen2.5-0.5b --lora-config B --split val
    python lora_validate.py --model llama3.2-1b  --lora-config C --split test
    python lora_validate.py --model smollm2-360m --lora-config A --split val --local
"""

import argparse
import gc
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_env_file = _REPO_ROOT / ".env"
if _env_file.exists():
    from dotenv import load_dotenv

    load_dotenv(_env_file)

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", message=".*max_new_tokens.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from finetune_lib import (
    FINETUNE_MODEL_REGISTRY,
    LORA_CONFIGS,
    HF_HUB_REPO,
    hf_adapter_subfolder,
    build_chat_messages,
    apply_chat_template_safe,
    load_jsonl,
    extract_prediction,
    compute_accuracy,
    compute_per_tool_metrics,
    resolve_device,
    peak_memory_mb,
)

LORA_DIR = Path(__file__).parent.parent
DEFAULT_DATA_DIR = LORA_DIR / "data"
DEFAULT_ADAPTER_DIR = LORA_DIR / "adapters"
DEFAULT_VAL_DIR = LORA_DIR / "reports_validation"
DEFAULT_TEST_DIR = LORA_DIR / "reports_test"
_TECHNIQUE = "LoRA"


# ── Inference helper ───────────────────────────────────────────────────────────


def run_inference(
    model,
    tokenizer,
    example: dict,
    device: torch.device,
    model_key: str,
) -> tuple[str, float, int]:
    """Single-example greedy inference. Returns (prediction, latency_s, n_tokens)."""
    messages = build_chat_messages(example, include_answer=False)
    text = apply_chat_template_safe(
        tokenizer, messages, model_key, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=int(tokenizer.eos_token_id),
        )
    latency = time.perf_counter() - t0
    new_ids = out[0][inputs["input_ids"].shape[1] :]
    return (
        extract_prediction(tokenizer.decode(new_ids, skip_special_tokens=True)),
        latency,
        len(new_ids),
    )


# ── Argument parsing ─────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a fine-tuned LoRA adapter.")
    p.add_argument(
        "--model",
        choices=list(FINETUNE_MODEL_REGISTRY.keys()),
        default="qwen2.5-0.5b",
    )
    p.add_argument(
        "--lora-config",
        choices=list(LORA_CONFIGS.keys()),
        default="B",
        dest="lora_config",
    )
    p.add_argument(
        "--dataset-size",
        choices=["1k", "10k"],
        default="1k",
        dest="dataset_size",
    )
    p.add_argument(
        "--split",
        choices=["val", "test", "test_anchor"],
        default="val",
        help=(
            "val         : validation set  → reports_validation/\n"
            "test        : full test set   → reports_test/\n"
            "test_anchor : sample_0001 (100 ex) → reports_test/"
        ),
    )
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help="Root dir for local adapters (default: finetune_LoRA/adapters/).",
    )
    p.add_argument("--val-report-dir", type=Path, default=DEFAULT_VAL_DIR)
    p.add_argument("--test-report-dir", type=Path, default=DEFAULT_TEST_DIR)
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N examples (for quick checks).",
    )
    p.add_argument(
        "--local",
        action="store_true",
        help="Load adapter from local adapters/ dir instead of HF Hub.",
    )
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model_id = FINETUNE_MODEL_REGISTRY[args.model]
    run_tag = f"{args.model}_{args.lora_config}_{args.dataset_size}"

    data_dir = (args.data_dir or DEFAULT_DATA_DIR) / args.dataset_size
    adapter_dir = (args.adapter_dir or DEFAULT_ADAPTER_DIR) / run_tag

    is_test = args.split in ("test", "test_anchor")
    out_dir = args.test_report_dir if is_test else args.val_report_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load examples ─────────────────────────────────────────────────────────────
    split_file = {
        "val": data_dir / "val.jsonl",
        "test": data_dir / "test.jsonl",
        "test_anchor": data_dir / "test_anchor.jsonl",
    }[args.split]

    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}\nRun prepare_lora_data.py first."
        )

    examples = load_jsonl(split_file)
    if args.limit:
        examples = examples[: args.limit]

    print(f"\n{'=' * 60}")
    print(f"  LoRA Evaluation — {run_tag}")
    print(f"  Split    : {args.split}  ({len(examples)} examples)")
    print(
        f"  Source   : {'local ' + str(adapter_dir) if args.local else HF_HUB_REPO + '/LoRA/' + run_tag}"
    )
    print(f"  Device   : {device}")
    print(f"{'=' * 60}\n")

    # ── Load model + adapter ─────────────────────────────────────────────────────
    dtype = torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32

    print(f"  Loading base model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=False,
    )

    # Load adapter: HF Hub (default) or local
    hf_sub = hf_adapter_subfolder(
        _TECHNIQUE, args.model, args.lora_config, args.dataset_size
    )
    if args.local:
        print(f"  Loading adapter (local): {adapter_dir}")
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    else:
        print(f"  Loading adapter from HF: {HF_HUB_REPO}/{hf_sub}")
        model = PeftModel.from_pretrained(base_model, HF_HUB_REPO, subfolder=hf_sub)

    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # ── Inference loop ──────────────────────────────────────────────────────────────
    results: list[dict] = []
    latencies: list[float] = []
    tok_counts: list[int] = []

    for i, ex in enumerate(examples, 1):
        pred, latency, n_tok = run_inference(model, tokenizer, ex, device, args.model)
        correct = pred == ex["answer"]
        latencies.append(latency)
        tok_counts.append(n_tok)
        results.append(
            {
                "index": i,
                "answer": ex["answer"],
                "prediction": pred,
                "correct": correct,
                "latency_s": round(latency, 4),
            }
        )
        if i % 20 == 0 or i == len(examples):
            running_acc = sum(r["correct"] for r in results) / len(results)
            print(f"  [{i:>4}/{len(examples)}]  running_acc={running_acc:.3f}")

    # ── Metrics ───────────────────────────────────────────────────────────────────
    predictions = [r["prediction"] for r in results]
    labels = [ex["answer"] for ex in examples]
    accuracy = sum(r["correct"] for r in results) / len(results)

    lat_ms = [l * 1000 for l in latencies]
    avg_lat_ms = float(np.mean(lat_ms))
    p50_lat_ms = float(np.percentile(lat_ms, 50))
    p95_lat_ms = float(np.percentile(lat_ms, 95))
    avg_tps = float(np.mean([n / l for n, l in zip(tok_counts, latencies) if l > 0]))
    mem_mb = peak_memory_mb(device)

    # Anchor accuracy: first 100 of test.jsonl == sample_0001 (for 10k cross-comparison)
    anchor_accuracy = None
    if args.split == "test" and args.dataset_size == "10k":
        n_anchor = sum(r["correct"] for r in results[:100])
        anchor_accuracy = n_anchor / 100
        print(
            f"\n  Anchor accuracy (sample_0001, first 100): {n_anchor}/100  ({anchor_accuracy:.4f})"
        )

    per_tool = compute_per_tool_metrics(predictions, labels) if is_test else {}

    # ── Write report ─────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "model_key": args.model,
        "model_id": model_id,
        "lora_config": args.lora_config,
        "dataset_size": args.dataset_size,
        "technique": _TECHNIQUE,
        "split": args.split,
        "timestamp": ts,
        "n_examples": len(results),
        "n_correct": sum(r["correct"] for r in results),
        "accuracy": round(accuracy, 4),
        "anchor_accuracy": round(anchor_accuracy, 4)
        if anchor_accuracy is not None
        else None,
        "avg_latency_ms": round(avg_lat_ms, 2),
        "p50_latency_ms": round(p50_lat_ms, 2),
        "p95_latency_ms": round(p95_lat_ms, 2),
        "avg_tokens_per_sec": round(avg_tps, 2),
        "peak_memory_mb": round(mem_mb, 1),
        "hf_repo": HF_HUB_REPO,
        "hf_subfolder": hf_sub,
        "per_tool_metrics": per_tool,
        "results": results,
    }

    report_name = f"{run_tag}_{args.split}_{ts}.json"
    report_path = out_dir / report_name
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"\n  Accuracy   : {accuracy:.4f}  ({sum(r['correct'] for r in results)}/{len(results)})"
    )
    print(f"  Avg latency: {avg_lat_ms:.1f} ms")
    print(f"  Peak memory: {mem_mb:.0f} MB")
    print(f"  Report     : {report_path}")

    del model, base_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
