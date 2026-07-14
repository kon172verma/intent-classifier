#!/usr/bin/env python3
"""
Phase 3 – Post-training evaluation of a LoRA adapter on val or test split.

Loads the base model + saved LoRA adapter and runs greedy inference on the
requested split.  Writes a JSON report to reports_validation/ (val / test_anchor)
or reports_test/ (test).

For --split test with --dataset-size 10k, two accuracy numbers are reported:
    anchor_accuracy : accuracy on sample_0001.json only (first 100 of test.jsonl)
    accuracy        : accuracy on the full 1 000-example test set

This enables direct comparison with the 1k test results using the same 100
examples regardless of which dataset-size variant was trained.

Usage
-----
    python lora_validate.py --model qwen2.5-0.5b --lora-config B --split val
    python lora_validate.py --model qwen3-0.6b   --lora-config B --split test --dataset-size 10k
    python lora_validate.py --model qwen2.5-0.5b --lora-config A --split test_anchor --limit 10
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

_env_file = Path(__file__).parent.parent.parent / ".env"
if _env_file.exists():
    from dotenv import load_dotenv

    load_dotenv(_env_file)

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", message=".*max_new_tokens.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    MODEL_REGISTRY,
    LORA_CONFIGS,
    build_chat_messages,
    apply_chat_template_safe,
    extract_prediction,
    compute_accuracy,
    compute_per_tool_metrics,
)

LORA_DIR = Path(__file__).parent.parent
DEFAULT_DATA_DIR = LORA_DIR / "data"
DEFAULT_CKPT_DIR = LORA_DIR / "checkpoints"
DEFAULT_VAL_DIR = LORA_DIR / "reports_validation"
DEFAULT_TEST_DIR = LORA_DIR / "reports_test"


# ── Helpers ───────────────────────────────────────────────────────────────────


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024**2
    if device.type == "mps":
        try:
            return torch.mps.current_allocated_memory() / 1024**2
        except AttributeError:
            return 0.0
    return 0.0


def run_inference(
    model,
    tokenizer,
    example: dict,
    device: torch.device,
    model_key: str,
) -> tuple[str, float, int]:
    """
    Single-example greedy inference.
    Returns (predicted_tool_name, latency_s, n_tokens_generated).
    """
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
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.perf_counter() - t0
    new_ids = out[0][inputs["input_ids"].shape[1] :]
    n_tokens = len(new_ids)
    raw = tokenizer.decode(new_ids, skip_special_tokens=True)
    return extract_prediction(raw), latency, n_tokens


# ── Argument parsing ──────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a fine-tuned LoRA adapter.")
    p.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        default="qwen2.5-0.5b",
    )
    p.add_argument(
        "--lora-config",
        choices=["A", "B", "C", "D"],
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
            "test_anchor : sample_0001 only (100 ex) → reports_test/"
        ),
    )
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--ckpt-dir", type=Path, default=None)
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
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model_id = MODEL_REGISTRY[args.model]
    run_tag = f"{args.model}_config_{args.lora_config}_{args.dataset_size}"

    data_dir = (args.data_dir or DEFAULT_DATA_DIR) / args.dataset_size
    ckpt_dir = (args.ckpt_dir or DEFAULT_CKPT_DIR) / run_tag

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
            f"Split file not found: {split_file}\nRun prepare_lora_data.py first."
        )

    examples = load_jsonl(split_file)
    if args.limit:
        examples = examples[: args.limit]

    print(f"\n{'=' * 60}")
    print(f"  LoRA Evaluation — {run_tag}")
    print(f"  Split    : {args.split}  ({len(examples)} examples)")
    print(f"  Adapter  : {ckpt_dir}")
    print(f"  Device   : {device}")
    print(f"{'=' * 60}\n")

    # ── Load tokenizer, base model, and LoRA adapter ──────────────────────────
    dtype = torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32

    print(f"  Loading tokenizer from adapter dir: {ckpt_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading base model: {model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=False,
    )

    print(f"  Loading LoRA adapter: {ckpt_dir}")
    model = PeftModel.from_pretrained(base_model, str(ckpt_dir))
    model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # ── Inference loop ────────────────────────────────────────────────────────
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

    # ── Compute metrics ───────────────────────────────────────────────────────
    predictions = [r["prediction"] for r in results]
    labels = [ex["answer"] for ex in examples]
    n_correct = sum(r["correct"] for r in results)
    accuracy = n_correct / len(results)

    lat_ms = [l * 1000 for l in latencies]
    avg_latency_ms = float(np.mean(lat_ms))
    p50_latency_ms = float(np.percentile(lat_ms, 50))
    p95_latency_ms = float(np.percentile(lat_ms, 95))
    avg_tps = float(np.mean([n / l for n, l in zip(tok_counts, latencies) if l > 0]))
    mem_mb = peak_memory_mb(device)

    # Anchor accuracy: first 100 examples of test.jsonl are always sample_0001.
    # Only computed when split="test" AND dataset_size="10k" (where test has 1 000 ex).
    anchor_accuracy = None
    n_anchor_correct = None
    if args.split == "test" and args.dataset_size == "10k":
        n_anchor_correct = sum(r["correct"] for r in results[:100])
        anchor_accuracy = n_anchor_correct / 100
        print(
            f"\n  Anchor accuracy (sample_0001, first 100): "
            f"{n_anchor_correct}/100  ({anchor_accuracy:.4f})"
        )

    # Per-tool breakdown only for test splits (too noisy on val).
    per_tool = compute_per_tool_metrics(predictions, labels) if is_test else {}

    # ── Write report ──────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "model_key": args.model,
        "model_id": model_id,
        "lora_config": args.lora_config,
        "dataset_size": args.dataset_size,
        "split": args.split,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "timestamp": ts,
        "n_examples": len(results),
        "n_correct": n_correct,
        "accuracy": round(accuracy, 4),
        "anchor_accuracy": round(anchor_accuracy, 4)
        if anchor_accuracy is not None
        else None,
        "n_anchor_correct": n_anchor_correct,
        "avg_latency_ms": round(avg_latency_ms, 2),
        "p50_latency_ms": round(p50_latency_ms, 2),
        "p95_latency_ms": round(p95_latency_ms, 2),
        "avg_tokens_per_sec": round(avg_tps, 2),
        "peak_memory_mb": round(mem_mb, 1),
        "per_tool_metrics": per_tool,
        "results": results,
    }

    fname = f"{run_tag}_{args.split}_{ts}.json"
    report_path = out_dir / fname
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  EVAL COMPLETE — {run_tag}  [{args.split}]")
    print(f"  Accuracy       : {n_correct}/{len(results)}  ({accuracy:.4f})")
    if anchor_accuracy is not None:
        print(f"  Anchor acc     : {n_anchor_correct}/100  ({anchor_accuracy:.4f})")
    print(f"  Avg latency    : {avg_latency_ms:.1f} ms")
    print(f"  Throughput     : {avg_tps:.1f} tok/s")
    print(f"  Peak memory    : {mem_mb:.0f} MB")
    print(f"  Report         : {report_path}")
    print(f"{'=' * 60}\n")

    # Cleanup
    del model, base_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
