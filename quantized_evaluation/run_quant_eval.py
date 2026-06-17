#!/usr/bin/env python3
"""
Batch runner: evaluates all 5 selected models × 3 quant configs × 2 modes.

Runs one subprocess at a time (one model per process, sequentially) to avoid
OOM. Models are iterated in increasing parameter order.

Usage
-----
# Full run (both zero-shot and few-shot)
python run_quant_eval.py

# Zero-shot only
python run_quant_eval.py --mode zero_shot

# Quick smoke-test (first 5 examples)
python run_quant_eval.py --limit 5

# Only specific models
python run_quant_eval.py --models qwen2.5-0.5b qwen3-0.6b
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).parent
_EVAL_SCRIPT = _THIS_DIR / "quant_eval.py"
_PYTHON = sys.executable

# Models in increasing parameter order
ALL_MODELS: list[str] = [
    "qwen2.5-0.5b",   # ~500M
    "qwen3-0.6b",     # ~600M
    "gemma3-1b",      # ~1.0B
    "qwen2.5-1.5b",   # ~1.5B
    "smollm3",        # ~3.0B
]

ALL_QUANTS: list[str] = ["int8", "nf4", "nf4_dq"]

QUANT_LABELS: dict[str, str] = {
    "int8":   "INT8",
    "nf4":    "NF4",
    "nf4_dq": "NF4+DQ",
}

DEFAULT_DATA = _THIS_DIR.parent / "dataset_sample" / "sample.json"
DEFAULT_OUT_DIR = _THIS_DIR / "reports"


def run_one(
    model: str,
    quant: str,
    mode: str,
    data: Path,
    out_dir: Path,
    device: str,
    limit: int | None,
    max_new_tokens: int,
) -> int:
    """Spawn a subprocess for one (model, quant, mode) combo. Returns exit code."""
    cmd = [
        _PYTHON, str(_EVAL_SCRIPT),
        "--model", model,
        "--quant", quant,
        "--mode", mode,
        "--data", str(data),
        "--out-dir", str(out_dir),
        "--device", device,
        "--max-new-tokens", str(max_new_tokens),
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]

    label = f"{model} [{QUANT_LABELS[quant]}] [{mode}]"
    print(f"\n{'='*64}")
    print(f"  Starting: {label}")
    print(f"{'='*64}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n  *** FAILED: {label} (exit {result.returncode}) — continuing ***\n")
    return result.returncode


def print_summary(out_dir: Path, modes: list[str]) -> None:
    """Print a compact accuracy table from the saved reports."""
    reports: list[dict] = []
    for f in sorted(out_dir.glob("*.json")):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            reports.append(r)
        except Exception:
            continue

    if not reports:
        print("No reports found.")
        return

    for mode in modes:
        subset = [r for r in reports if r.get("eval_mode") == mode]
        if not subset:
            continue
        print(f"\n{'─'*72}")
        print(f"  Summary — {mode.upper()}")
        print(f"{'─'*72}")
        print(f"  {'Model':<20} {'Quant':<10} {'Accuracy':>10} {'Mem (MB)':>12}")
        print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*12}")
        for r in sorted(
            subset,
            key=lambda x: (x.get("model_key", ""), x.get("quant", "")),
        ):
            print(
                f"  {r.get('model_key',''):20s} {r.get('quant',''):10s} "
                f"{r.get('accuracy',0)*100:>9.1f}%  "
                f"{r.get('peak_memory_mb',0):>10.0f} MB"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch quantization evaluation runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+", default=ALL_MODELS,
        choices=ALL_MODELS,
        metavar="MODEL",
        help="Models to evaluate (default: all 5 selected models).",
    )
    parser.add_argument(
        "--quants", nargs="+", default=ALL_QUANTS,
        choices=ALL_QUANTS,
        metavar="QUANT",
        help="Quantization configs to run.",
    )
    parser.add_argument(
        "--mode", type=str, default="both",
        choices=["zero_shot", "few_shot", "both"],
        help="Prompt mode to evaluate.",
    )
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA,
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=32,
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only first N examples (smoke-test).",
    )
    args = parser.parse_args()

    modes = ["zero_shot", "few_shot"] if args.mode == "both" else [args.mode]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    total = len(args.models) * len(args.quants) * len(modes)
    done = 0
    failures: list[str] = []

    for model in args.models:
        for quant in args.quants:
            for mode in modes:
                done += 1
                print(f"\n[{done}/{total}]")
                rc = run_one(
                    model=model,
                    quant=quant,
                    mode=mode,
                    data=args.data,
                    out_dir=args.out_dir,
                    device=args.device,
                    limit=args.limit,
                    max_new_tokens=args.max_new_tokens,
                )
                if rc != 0:
                    failures.append(f"{model}/{quant}/{mode}")

    print_summary(args.out_dir, modes)

    if failures:
        print(f"\n  *** {len(failures)} run(s) failed: {failures} ***")
    else:
        print("\n  All runs completed successfully.")


if __name__ == "__main__":
    main()
