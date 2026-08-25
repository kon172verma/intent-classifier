#!/usr/bin/env python3
"""
finetune_AdaLoRA/src/adalora_validate.py
=========================================
Post-training evaluation of an AdaLoRA adapter on val or test split.

Adapter loading priority
------------------------
1. HuggingFace Hub (default):
     PeftModel.from_pretrained(base, HF_EXPERIMENTS_REPO, subfolder="v1.0/{model}_AdaLoRA_{config}_{size}_{timestamp}")
     The exact timestamp is resolved automatically from EXPERIMENTS.jsonl
     (latest matching run) unless --timestamp is passed explicitly.
2. Local fallback (--local flag):
     PeftModel.from_pretrained(base, finetune_AdaLoRA/adapters/{run_tag}/)

Outputs
-------
  reports_validation/{run_tag}_{split}_{ts}.json
  reports_test/      {run_tag}_{split}_{ts}.json

Usage
-----
    python adalora_validate.py --model qwen2.5-0.5b --adalora-config B --split val
    python adalora_validate.py --model llama3.2-1b  --adalora-config C --split test
    python adalora_validate.py --model smollm2-360m --adalora-config A --split val --local
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
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from finetune_lib import (
    ADALORA_CONFIGS,
    ADALORA_MODEL_REGISTRY,
    CURRENT_VERSION,
    HF_EXPERIMENTS_REPO,
    NO_TOOL_ID,
    PROMPT_FORMAT_VERSION,
    TOOL_IDS,
    answer_to_tool_id,
    apply_chat_template_safe,
    build_chat_messages,
    compute_per_tool_metrics,
    extract_prediction,
    hf_adapter_subfolder,
    hf_report_path,
    load_jsonl,
    peak_memory_mb,
    resolve_device,
    tool_id_to_answer,
    upload_report_to_hf,
)
from finetune_lib.registry import find_latest_experiment

ADALORA_DIR = Path(__file__).parent.parent
DEFAULT_DATA_DIR = ADALORA_DIR / "data"
DEFAULT_ADAPTER_DIR = ADALORA_DIR / "adapters"
DEFAULT_VAL_DIR = ADALORA_DIR / "reports_validation"
DEFAULT_TEST_DIR = ADALORA_DIR / "reports_test"
_TECHNIQUE = "AdaLoRA"


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
    text = apply_chat_template_safe(tokenizer, messages, model_key, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=8,
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
    p = argparse.ArgumentParser(description="Evaluate a fine-tuned AdaLoRA adapter.")
    p.add_argument(
        "--model",
        choices=list(ADALORA_MODEL_REGISTRY.keys()),
        default="qwen3-0.6b",
    )
    p.add_argument(
        "--adalora-config",
        choices=list(ADALORA_CONFIGS.keys()),
        default="B",
        dest="adalora_config",
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
        help="Root dir for local adapters (default: finetune_AdaLoRA/adapters/).",
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
    p.add_argument(
        "--timestamp",
        default=None,
        help=(
            "Exact experiment timestamp (e.g. 20260101-120000) to load from "
            "HF_EXPERIMENTS_REPO. Defaults to the latest matching entry in "
            "EXPERIMENTS.jsonl."
        ),
    )
    p.add_argument(
        "--version",
        default=None,
        help="Experiment version folder to look up (default: CURRENT_VERSION).",
    )
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model_id = ADALORA_MODEL_REGISTRY[args.model]
    run_tag = f"{args.model}_{args.adalora_config}_{args.dataset_size}"

    data_dir = (args.data_dir or DEFAULT_DATA_DIR) / args.dataset_size
    adapter_dir = (args.adapter_dir or DEFAULT_ADAPTER_DIR) / run_tag

    is_test = args.split in ("test", "test_anchor")
    out_dir = args.test_report_dir if is_test else args.val_report_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load examples ─────────────────────────────────────────────────────────
    split_file = {
        "val": data_dir / "val.jsonl",
        "test": data_dir / "test.jsonl",
        "test_anchor": data_dir / "test_anchor.jsonl",
    }[args.split]

    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}\nRun prepare_adalora_data.py first."
        )

    examples = load_jsonl(split_file)
    if args.limit:
        examples = examples[: args.limit]

    version = args.version or CURRENT_VERSION
    hf_sub = ""
    if not args.local:
        entry = find_latest_experiment(
            technique=_TECHNIQUE,
            model_key=args.model,
            lora_config=args.adalora_config,
            dataset_size=args.dataset_size,
            version=version,
        )
        timestamp = args.timestamp or (entry["timestamp"] if entry else None)
        if timestamp is None:
            raise ValueError(
                f"No experiment found in EXPERIMENTS.jsonl for {_TECHNIQUE}/{run_tag} "
                f"(version={version}). Pass --timestamp explicitly or use --local."
            )
        hf_sub = hf_adapter_subfolder(
            _TECHNIQUE,
            args.model,
            args.adalora_config,
            args.dataset_size,
            timestamp=timestamp,
            version=version,
        )

    print(f"\n{'=' * 60}")
    print(f"  AdaLoRA Evaluation — {run_tag}")
    print(f"  Split    : {args.split}  ({len(examples)} examples)")
    print(
        f"  Source   : {'local ' + str(adapter_dir) if args.local else HF_EXPERIMENTS_REPO + '/' + hf_sub}"
    )
    print(f"  Device   : {device}")
    print(f"{'=' * 60}\n")

    # ── Load model + adapter ─────────────────────────────────────────────────
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

    if args.local:
        print(f"  Loading adapter (local): {adapter_dir}")
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    else:
        print(f"  Loading adapter from HF: {HF_EXPERIMENTS_REPO}/{hf_sub}")
        model = PeftModel.from_pretrained(base_model, HF_EXPERIMENTS_REPO, subfolder=hf_sub)

    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # ── Inference loop ────────────────────────────────────────────────────────
    results: list[dict] = []
    latencies: list[float] = []
    tok_counts: list[int] = []

    for i, ex in enumerate(examples, 1):
        pred, latency, n_tok = run_inference(model, tokenizer, ex, device, args.model)
        answer_id = answer_to_tool_id(ex)
        prediction_tool = tool_id_to_answer(ex, pred)
        correct = pred == answer_id
        latencies.append(latency)
        tok_counts.append(n_tok)
        results.append(
            {
                "index": i,
                "answer": ex["answer"],
                "answer_id": answer_id,
                "prediction": pred,
                "prediction_tool": prediction_tool,
                "correct": correct,
                "latency_s": round(latency, 4),
            }
        )
        if i % 20 == 0 or i == len(examples):
            running_acc = sum(r["correct"] for r in results) / len(results)
            print(f"  [{i:>4}/{len(examples)}]  running_acc={running_acc:.3f}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    predictions = [r["prediction_tool"] for r in results]
    labels = [ex["answer"] for ex in examples]
    accuracy = sum(r["correct"] for r in results) / len(results)

    lat_ms = [lat * 1000 for lat in latencies]
    avg_lat_ms = float(np.mean(lat_ms))
    p50_lat_ms = float(np.percentile(lat_ms, 50))
    p95_lat_ms = float(np.percentile(lat_ms, 95))
    avg_tps = float(np.mean([n / lat for n, lat in zip(tok_counts, latencies) if lat > 0]))
    mem_mb = peak_memory_mb(device)

    anchor_accuracy = None
    if args.split == "test" and args.dataset_size == "10k":
        n_anchor = sum(r["correct"] for r in results[:100])
        anchor_accuracy = n_anchor / 100
        print(
            f"\n  Anchor accuracy (sample_0001, first 100): {n_anchor}/100  ({anchor_accuracy:.4f})"
        )

    per_tool = compute_per_tool_metrics(predictions, labels) if is_test else {}

    # ── Write report ──────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "model_key": args.model,
        "model_id": model_id,
        "adalora_config": args.adalora_config,
        # "lora_config" alias for compatibility with shared plot_lib loaders
        "lora_config": args.adalora_config,
        "dataset_size": args.dataset_size,
        "technique": _TECHNIQUE,
        "split": args.split,
        "timestamp": ts,
        "prompt_format": PROMPT_FORMAT_VERSION,
        "tool_id_scheme": "".join(TOOL_IDS),
        "no_tool_id": NO_TOOL_ID,
        "n_examples": len(results),
        "n_correct": sum(r["correct"] for r in results),
        "accuracy": round(accuracy, 4),
        "anchor_accuracy": round(anchor_accuracy, 4) if anchor_accuracy is not None else None,
        "avg_latency_ms": round(avg_lat_ms, 2),
        "p50_latency_ms": round(p50_lat_ms, 2),
        "p95_latency_ms": round(p95_lat_ms, 2),
        "avg_tokens_per_sec": round(avg_tps, 2),
        "peak_memory_mb": round(mem_mb, 1),
        "hf_repo": HF_EXPERIMENTS_REPO if not args.local else None,
        "hf_subfolder": hf_sub if not args.local else None,
        "hf_report_path": None,
        "per_tool_metrics": per_tool,
        "results": results,
    }

    report_name = f"{run_tag}_{args.split}_{ts}.json"
    report_path = out_dir / report_name
    report["hf_report_path"] = (
        hf_report_path(
            report_name=report_path.name,
            version=version,
            technique=_TECHNIQUE,
            report_group=out_dir.name,
        )
        if not args.local and hf_sub
        else None
    )
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.local and hf_sub:
        uploaded_path = upload_report_to_hf(
            report_path=report_path,
            version=version,
            technique=_TECHNIQUE,
            report_group=out_dir.name,
            commit_message=f"Add AdaLoRA {args.split} report: {run_tag} [{ts}]",
        )
        if uploaded_path:
            print(f"  Report pushed  : {HF_EXPERIMENTS_REPO}/{uploaded_path}")

    print(f"\n  Accuracy   : {accuracy:.4f}  ({sum(r['correct'] for r in results)}/{len(results)})")
    print(f"  Avg latency: {avg_lat_ms:.1f} ms")
    print(f"  Peak memory: {mem_mb:.0f} MB")
    print(f"  Report     : {report_path}")

    del model, base_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
