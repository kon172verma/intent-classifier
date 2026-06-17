#!/usr/bin/env python3
"""
Run zero-shot baseline evaluation across all registered models and
print/save a summary accuracy table.

Usage:
    .venv/bin/python baseline_evaluation/src/run_all_baselines.py
    .venv/bin/python baseline_evaluation/src/run_all_baselines.py --models smollm2-135m qwen3-0.6b
    .venv/bin/python baseline_evaluation/src/run_all_baselines.py --skip smollm2-135m
    .venv/bin/python baseline_evaluation/src/run_all_baselines.py --limit 10   # smoke-test
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Load .env from repo root so HF_TOKEN is available to subprocess runs
_env_file = Path(__file__).parent.parent.parent / ".env"
if _env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_file)

ROOT = Path(__file__).parent.parent.parent  # repo root
EVAL_SCRIPT = Path(__file__).parent / "baseline_eval.py"
DEFAULT_DATA = ROOT / "dataset_sample" / "sample.json"
REPORTS_DIR = Path(__file__).parent.parent / "reports_zero_shot"
FEW_SHOT_REPORTS_DIR = Path(__file__).parent.parent / "reports_few_shot"
PYTHON = sys.executable  # use the same venv python that launched this script

# Ordered from smallest to largest so each run finishes quickly
ALL_MODELS = [
    "smollm2-135m",
    "gemma3-270m",
    "smollm2-360m",
    "qwen2.5-0.5b",
    "qwen3-0.6b",
    "tinyllama",
    "gemma3-1b",
    "qwen2.5-1.5b",
    "qwen3-1.7b",
    "smollm3",
    "llama3.2-3b",
    "qwen3-4b",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch zero-shot baseline evaluation across all models."
    )
    p.add_argument(
        "--models", nargs="+", default=None,
        help="Subset of model keys to run (default: all).",
    )
    p.add_argument(
        "--skip", nargs="+", default=None,
        help="Model keys to skip (useful if some are already evaluated).",
    )
    p.add_argument(
        "--data", type=Path, default=DEFAULT_DATA,
        help="Dataset file (.json or .jsonl).",
    )
    p.add_argument(
        "--device", default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    p.add_argument(
        "--mode",
        default="zero_shot",
        choices=["zero_shot", "few_shot"],
        help="Prompt mode passed to baseline_eval.py.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only first N examples per model (for smoke-tests).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory for reports (default: reports/ or few_shot_reports/ depending on mode).",
    )
    return p.parse_args()


def latest_report_for(model_key: str, reports_dir: Path) -> Path | None:
    """Return the most-recent report JSON for a model key, if it exists."""
    matches = sorted(reports_dir.glob(f"{model_key}_*.json"))
    return matches[-1] if matches else None


def run_model(
    model_key: str,
    data: Path,
    device: str,
    limit: int | None,
    out_dir: Path,
    mode: str = "zero_shot",
) -> dict | None:
    """
    Invoke baseline_eval.py as a subprocess for one model.
    Returns the parsed report dict, or None on failure.
    """
    cmd = [
        PYTHON, str(EVAL_SCRIPT),
        "--model", model_key,
        "--data", str(data),
        "--device", device,
        "--mode", mode,
        "--out-dir", str(out_dir),
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]

    print(f"\n{'='*60}")
    print(f"  Running: {model_key}")
    print(f"{'='*60}")

    # Stream subprocess output live
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()

    if proc.returncode != 0:
        print(f"\n  [ERROR] {model_key} exited with code {proc.returncode}")
        return None

    # Find the report that was just written
    report_path = latest_report_for(model_key, out_dir)
    if report_path is None:
        print(f"\n  [ERROR] No report found for {model_key}")
        return None

    return json.loads(report_path.read_text(encoding="utf-8"))


def print_summary(results: list[dict]) -> None:
    if not results:
        print("\nNo results to display.")
        return

    col_key = max(len(r["model_key"]) for r in results) + 2
    mode_label = results[0].get("eval_mode", "zero_shot").upper().replace("_", "-") if results else "ZERO-SHOT"
    header = (
        f"{'Model':<{col_key}}  {'Acc':>7}  {'Correct':>8}  "
        f"{'AvgLat(ms)':>11}  {'Tok/s':>7}  {'Mem(MB)':>8}"
    )
    sep = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print(f"  {mode_label} BASELINE SUMMARY")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)
    for r in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        print(
            f"{r['model_key']:<{col_key}}  "
            f"{r['accuracy']:>6.1%}  "
            f"{r['n_correct']:>4}/{r['n_examples']:<3}  "
            f"{r['avg_latency_ms']:>11.1f}  "
            f"{r['avg_tokens_per_sec']:>7.1f}  "
            f"{r['peak_memory_mb']:>8.1f}"
        )
    print(sep)


def save_summary(results: list[dict], out_dir: Path) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"summary_{ts}.json"
    path.write_text(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in (
                        "model_key", "model_id", "device", "dtype",
                        "timestamp", "n_examples", "n_correct", "accuracy",
                        "avg_latency_ms", "p50_latency_ms", "p95_latency_ms",
                        "avg_tokens_per_sec", "peak_memory_mb",
                    )
                }
                for r in results
            ],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n  Summary saved → {path}")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (FEW_SHOT_REPORTS_DIR if args.mode == "few_shot" else REPORTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = args.models if args.models else ALL_MODELS
    skip_set = set(args.skip or [])
    targets = [m for m in targets if m not in skip_set]

    # Check for already-completed reports so we can reload rather than re-run
    results: list[dict] = []
    to_run: list[str] = []

    for key in targets:
        existing = latest_report_for(key, out_dir)
        if existing:
            print(f"  [CACHED] {key}  →  {existing.name}")
            r = json.loads(existing.read_text(encoding="utf-8"))
            results.append(r)
        else:
            to_run.append(key)

    print(f"\n  Models to run   : {len(to_run)}")
    print(f"  Models cached   : {len(results)}")
    print(f"  Dataset         : {args.data}")
    print(f"  Mode            : {args.mode}")
    print(f"  Device          : {args.device}")
    if args.limit:
        print(f"  Limit           : {args.limit} examples")

    for key in to_run:
        report = run_model(key, args.data, args.device, args.limit, out_dir, mode=args.mode)
        if report:
            results.append(report)

    print_summary(results)
    if results:
        save_summary(results, out_dir)


if __name__ == "__main__":
    main()
