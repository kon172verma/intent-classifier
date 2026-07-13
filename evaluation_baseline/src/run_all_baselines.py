#!/usr/bin/env python3
"""
Batch runner: evaluate all registered models zero-shot and/or few-shot.

Usage
-----
# Evaluate all models
python run_all_baselines.py

# Subset of models
python run_all_baselines.py --models qwen2.5-0.5b qwen3-0.6b

# Few-shot mode
python run_all_baselines.py --mode few_shot

# Skip models already cached
python run_all_baselines.py --skip gemma3-270m gemma3-1b

# Smoke-test (first 10 examples per model)
python run_all_baselines.py --limit 10
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

_env_file = _REPO_ROOT / ".env"
if _env_file.exists():
    from dotenv import load_dotenv

    load_dotenv(_env_file)

from evaluation_lib import MODEL_REGISTRY  # noqa: E402, F401
from evaluation_lib.config import ALL_MODELS  # noqa: E402

EVAL_SCRIPT = Path(__file__).parent / "baseline_eval.py"
DEFAULT_DATA = _REPO_ROOT / "dataset_sample" / "sample.json"
REPORTS_DIR = Path(__file__).parent.parent / "reports_zero_shot"
FS_REPORTS_DIR = Path(__file__).parent.parent / "reports_few_shot"
PYTHON = sys.executable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch baseline evaluation across all models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of model keys to run (default: all).",
    )
    p.add_argument("--skip", nargs="+", default=None, help="Model keys to skip.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--mode", default="zero_shot", choices=["zero_shot", "few_shot"])
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=32)
    return p.parse_args()


def latest_report_for(model_key: str, reports_dir: Path) -> Path | None:
    matches = sorted(reports_dir.glob(f"{model_key}_*.json"))
    return matches[-1] if matches else None


def run_model(
    model_key: str,
    data: Path,
    device: str,
    limit: int | None,
    out_dir: Path,
    mode: str,
    max_new_tokens: int,
) -> dict | None:
    cmd = [
        PYTHON,
        str(EVAL_SCRIPT),
        "--model",
        model_key,
        "--data",
        str(data),
        "--device",
        device,
        "--mode",
        mode,
        "--out-dir",
        str(out_dir),
        "--max-new-tokens",
        str(max_new_tokens),
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]

    print(f"\n{'=' * 60}\n  Running: {model_key}\n{'=' * 60}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()

    if proc.returncode != 0:
        print(f"\n  [ERROR] {model_key} exited with code {proc.returncode}")
        return None

    report_path = latest_report_for(model_key, out_dir)
    if report_path is None:
        print(f"\n  [ERROR] No report found for {model_key}")
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))


def print_summary(results: list[dict]) -> None:
    if not results:
        return
    col_key = max(len(r["model_key"]) for r in results) + 2
    mode_lbl = results[0].get("eval_mode", "zero_shot").upper().replace("_", "-")
    header = (
        f"{'Model':<{col_key}}  {'Acc':>7}  {'Correct':>8}  "
        f"{'Garbage%':>9}  {'AvgLat(ms)':>11}  {'Tok/s':>7}  {'Mem(MB)':>8}"
    )
    sep = "-" * len(header)
    print(f"\n{'=' * len(header)}\n  {mode_lbl} BASELINE SUMMARY\n{'=' * len(header)}")
    print(header)
    print(sep)
    for r in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        print(
            f"{r['model_key']:<{col_key}}  {r['accuracy']:>6.1%}  "
            f"{r['n_correct']:>4}/{r['n_examples']:<3}  "
            f"{r.get('garbage_pct', 0.0):>8.1f}%  "
            f"{r['avg_latency_ms']:>11.1f}  {r['avg_tokens_per_sec']:>7.1f}  "
            f"{r['peak_memory_mb']:>8.1f}"
        )
    print(sep)


def save_summary(results: list[dict], out_dir: Path) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"summary_{ts}.json"
    keep = (
        "model_key",
        "model_id",
        "device",
        "dtype",
        "timestamp",
        "n_examples",
        "n_correct",
        "accuracy",
        "garbage_pct",
        "avg_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "avg_tokens_per_sec",
        "peak_memory_mb",
    )
    path.write_text(
        json.dumps(
            [{k: r[k] for k in keep if k in r} for r in results],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n  Summary saved → {path}")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (
        FS_REPORTS_DIR if args.mode == "few_shot" else REPORTS_DIR
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = args.models if args.models else ALL_MODELS
    skip_set = set(args.skip or [])
    targets = [m for m in targets if m not in skip_set]

    results: list[dict] = []
    to_run: list[str] = []
    for key in targets:
        existing = latest_report_for(key, out_dir)
        if existing:
            print(f"  [CACHED] {key}  →  {existing.name}")
            results.append(json.loads(existing.read_text(encoding="utf-8")))
        else:
            to_run.append(key)

    print(f"\n  Models to run : {len(to_run)}")
    print(f"  Models cached : {len(results)}")
    print(f"  Dataset       : {args.data}")
    print(f"  Mode          : {args.mode}")
    print(f"  Device        : {args.device}")
    if args.limit:
        print(f"  Limit         : {args.limit} examples")

    for key in to_run:
        r = run_model(
            key,
            args.data,
            args.device,
            args.limit,
            out_dir,
            args.mode,
            args.max_new_tokens,
        )
        if r:
            results.append(r)

    print_summary(results)
    if results:
        save_summary(results, out_dir)


if __name__ == "__main__":
    main()
