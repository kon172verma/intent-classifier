#!/usr/bin/env python3
"""
Batch runner for all QLoRA fine-tuning experiments.

Experiment matrix: 2 models × 4 configs = 8 training runs.
For each run it:
  1. Calls qlora_train.py  (skipped with --skip-training)
  2. Calls qlora_validate.py --split val

Test-set evaluation is intentionally kept separate — run qlora_validate.py
with --split test or --split test_anchor manually after you are satisfied
with the validation results to avoid data leakage from repeated test checks.

Usage
-----
    # Full matrix (1k dataset, all models & configs)
    python run_qlora_experiments.py

    # Specific models / configs
    python run_qlora_experiments.py --models qwen2.5-0.5b --configs A B

    # Scale to 10k dataset
    python run_qlora_experiments.py --dataset-size 10k

    # Validate only (training already done)
    python run_qlora_experiments.py --skip-training

    # Quick pipeline check — 10 steps per run, no checkpoint saved
    python run_qlora_experiments.py --smoke-test
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_env_file = Path(__file__).parent.parent.parent / ".env"
if _env_file.exists():
    from dotenv import load_dotenv

    load_dotenv(_env_file)

sys.path.insert(0, str(Path(__file__).parent))
from common import MODEL_REGISTRY, LORA_CONFIGS

QLORA_DIR = Path(__file__).parent.parent
TRAIN_SCRIPT = Path(__file__).parent / "qlora_train.py"
EVAL_SCRIPT = Path(__file__).parent / "qlora_validate.py"
PYTHON = sys.executable

DEFAULT_MODELS = ["qwen2.5-0.5b", "qwen3-0.6b"]
DEFAULT_CONFIGS = ["A", "B", "C", "D"]


# ── Helpers ───────────────────────────────────────────────────────────────────


def run_subprocess(cmd: list[str], label: str) -> bool:
    """Stream subprocess output and return True on success."""
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        print(f"\n  [ERROR] Exited with code {proc.returncode}")
    return proc.returncode == 0


def latest_val_report(
    model_key: str, lora_config: str, dataset_size: str
) -> Path | None:
    """Return the most recent val report JSON for this run, or None."""
    val_dir = QLORA_DIR / "reports_validation"
    tag = f"{model_key}_config_{lora_config}_{dataset_size}_val_"
    matches = sorted(val_dir.glob(f"{tag}*.json"))
    return matches[-1] if matches else None


# ── Argument parsing ──────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run all QLoRA fine-tuning experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=list(MODEL_REGISTRY.keys()),
    )
    p.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        choices=["A", "B", "C", "D"],
    )
    p.add_argument(
        "--dataset-size",
        choices=["1k", "10k"],
        default="1k",
        dest="dataset_size",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    p.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip qlora_train.py and only run val evaluation on existing checkpoints.",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="10 training steps per run only — validates the pipeline without full training.",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    print(f"\n{'=' * 60}")
    print(f"  QLoRA Experiment Runner")
    print(f"  Models   : {args.models}")
    print(f"  Configs  : {args.configs}")
    print(f"  Dataset  : {args.dataset_size}")
    print(f"  Device   : {args.device}")
    if args.smoke_test:
        print(f"  Mode     : SMOKE TEST (10 steps per run)")
    print(f"{'=' * 60}")

    results: list[dict] = []
    failed: list[str] = []

    for model_key in args.models:
        for cfg in args.configs:
            run_tag = f"{model_key}_config_{cfg}_{args.dataset_size}"

            # ── Training ──────────────────────────────────────────────────────
            if not args.skip_training:
                train_cmd = [
                    PYTHON,
                    str(TRAIN_SCRIPT),
                    "--model",
                    model_key,
                    "--lora-config",
                    cfg,
                    "--dataset-size",
                    args.dataset_size,
                    "--device",
                    args.device,
                ]
                if args.smoke_test:
                    train_cmd.append("--smoke-test")

                ok = run_subprocess(train_cmd, f"TRAIN  {run_tag}")
                if not ok:
                    failed.append(f"train:{run_tag}")
                    continue

                if args.smoke_test:
                    print(f"  [SMOKE] Skipping val eval (no checkpoint saved).")
                    continue

            # ── Validation eval ───────────────────────────────────────────────
            val_cmd = [
                PYTHON,
                str(EVAL_SCRIPT),
                "--model",
                model_key,
                "--lora-config",
                cfg,
                "--dataset-size",
                args.dataset_size,
                "--split",
                "val",
                "--device",
                args.device,
            ]
            ok = run_subprocess(val_cmd, f"EVAL   {run_tag}  [val]")
            if not ok:
                failed.append(f"eval:{run_tag}")
                continue

            rp = latest_val_report(model_key, cfg, args.dataset_size)
            if rp:
                results.append(json.loads(rp.read_text(encoding="utf-8")))

    # ── Summary table ─────────────────────────────────────────────────────────
    if results:
        col = max(len(r["model_key"]) for r in results) + 2
        header = (
            f"{'Run':<{col + 12}}  {'Acc':>7}  {'Correct':>8}  "
            f"{'AvgLat(ms)':>11}  {'Tok/s':>7}  {'Mem(MB)':>8}"
        )
        sep = "─" * len(header)
        print(f"\n{'=' * len(header)}")
        print(f"  VAL RESULTS SUMMARY  (dataset={args.dataset_size})")
        print(f"{'=' * len(header)}")
        print(header)
        print(sep)
        for r in sorted(results, key=lambda x: x["accuracy"], reverse=True):
            tag = f"{r['model_key']} / cfg-{r['lora_config']}"
            print(
                f"{tag:<{col + 12}}  "
                f"{r['accuracy']:>6.1%}  "
                f"{r['n_correct']:>4}/{r['n_examples']:<3}  "
                f"{r['avg_latency_ms']:>11.1f}  "
                f"{r['avg_tokens_per_sec']:>7.1f}  "
                f"{r['peak_memory_mb']:>8.1f}"
            )
        print(sep)

    if failed:
        print(f"\n  [FAILED] {len(failed)} run(s):")
        for f_tag in failed:
            print(f"    {f_tag}")

    print(
        "\n  Tip: run qlora_validate.py --split test (or --split test_anchor) "
        "manually to evaluate the locked test set."
    )


if __name__ == "__main__":
    main()
