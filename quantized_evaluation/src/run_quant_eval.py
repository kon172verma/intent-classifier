#!/usr/bin/env python3
"""
Batch runner: evaluates all 4 selected models × 4 quant configs × 2 modes.

Runs one subprocess at a time (one model per process, sequentially) to avoid
OOM. Models are iterated in increasing parameter order.

Quantization configs:
  int8    – Classic 8-bit (universal fallback)
  nf4     – 4-bit NF4 (NVIDIA GPUs: H100, A100, L4, T4, G4)
  int4    – 4-bit INT4 (broader hardware: TensorRT, ONNX, edge devices)
  fp8     – 8-bit FP8 (modern GPUs + emerging edge device support)

Usage
-----
# Full run (all 4 models × 4 quants × 2 modes = 32 runs)
python run_quant_eval.py

# Zero-shot only
python run_quant_eval.py --mode zero_shot

# Quick smoke-test (first 5 examples)
python run_quant_eval.py --limit 5

# Only specific models
python run_quant_eval.py --models qwen2.5-0.5b qwen2.5-1.5b

# Only specific quantizations
python run_quant_eval.py --quants int8 int4 fp8
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).parent
_EVAL_SCRIPT = _THIS_DIR / "src" / "quant_eval.py"
_PYTHON = sys.executable

# Models in increasing parameter order (4-model core set)
ALL_MODELS: list[str] = [
    "qwen2.5-0.5b",   # ~500M
    "qwen3-0.6b",     # ~600M
    "qwen2.5-1.5b",   # ~1.5B
    "smollm3",        # ~3.0B
]

# All quantization precisions
ALL_QUANTS: list[str] = ["int8", "nf4", "int4", "fp8"]

QUANT_LABELS: dict[str, str] = {
    "int8": "INT8",
    "nf4":  "NF4",
    "int4": "INT4",
    "fp8":  "FP8",
}

# Map quantization to output directory
QUANT_OUTPUT_DIRS: dict[str, str] = {
    "int8": "reports_int8",
    "nf4": "reports_nf4",

    "int4": "reports_int4",
    "fp8": "reports_fp8",
}

DEFAULT_DATA = _THIS_DIR.parent / "dataset_sample" / "sample.json"


def run_one(
    model: str,
    quant: str,
    mode: str,
    data: Path,
    device: str,
    limit: int | None,
    max_new_tokens: int,
) -> int:
    """Spawn a subprocess for one (model, quant, mode) combo. Returns exit code."""
    # Output directory is determined by quantization type
    out_dir = _THIS_DIR / QUANT_OUTPUT_DIRS[quant]
    
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


def print_summary(modes: list[str]) -> None:
    """Print a compact accuracy table from the saved reports across all quant dirs."""
    reports: list[dict] = []
    
    for quant in ALL_QUANTS:
        out_dir = _THIS_DIR / QUANT_OUTPUT_DIRS[quant]
        if not out_dir.exists():
            continue
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
        print(f"\n{'─'*80}")
        print(f"  Summary — {mode.upper()}")
        print(f"{'─'*80}")
        print(f"  {'Model':<20} {'Quant':<10} {'Accuracy':>10} {'Mem (MB)':>12} {'Latency (ms)':>14}")
        print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*12} {'─'*14}")
        for r in sorted(
            subset,
            key=lambda x: (x.get("model_key", ""), x.get("quant", "")),
        ):
            print(
                f"  {r.get('model_key',''):20s} {r.get('quant',''):10s} "
                f"{r.get('accuracy',0)*100:>9.1f}%  "
                f"{r.get('peak_memory_mb',0):>10.0f} MB  "
                f"{r.get('avg_latency_ms',0):>12.1f} ms"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch quantization evaluation runner for 4 selected models × 5 precisions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+", default=ALL_MODELS,
        choices=ALL_MODELS,
        metavar="MODEL",
        help="Models to evaluate (default: all 4 selected models).",
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
                    device=args.device,
                    limit=args.limit,
                    max_new_tokens=args.max_new_tokens,
                )
                if rc != 0:
                    failures.append(f"{model}/{quant}/{mode}")

    print_summary(modes)

    if failures:
        print(f"\n  *** {len(failures)} run(s) failed: {failures} ***")
    else:
        print("\n  All runs completed successfully.")


if __name__ == "__main__":
    main()
