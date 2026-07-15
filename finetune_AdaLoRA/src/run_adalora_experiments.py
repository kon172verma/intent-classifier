#!/usr/bin/env python3
"""
finetune_AdaLoRA/src/run_adalora_experiments.py
================================================
Batch runner for all AdaLoRA fine-tuning experiments.

Experiment matrix: 5 models × 4 configs × 2 modes (train + val) = 40 runs.
Models are iterated in increasing parameter order.
For each run:
  1. adalora_train.py    (skipped with --skip-training)
  2. adalora_validate.py --split val

Usage
-----
    # Full matrix (1k dataset)
    python run_adalora_experiments.py

    # Specific models / configs
    python run_adalora_experiments.py --models qwen2.5-0.5b llama3.2-1b --configs A B

    # Validate only (training already done, adapters on HF)
    python run_adalora_experiments.py --skip-training

    # Quick pipeline check (10 steps per run, no adapter saved)
    python run_adalora_experiments.py --smoke-test
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_env_file = _REPO_ROOT / ".env"
if _env_file.exists():
    from dotenv import load_dotenv

    load_dotenv(_env_file)

from finetune_lib import (
    FINETUNE_MODEL_REGISTRY,
    ALL_FINETUNE_MODELS,
    ADALORA_CONFIGS,
    ALL_ADALORA_CONFIGS,
)

ADALORA_DIR = Path(__file__).parent.parent
TRAIN_SCRIPT = Path(__file__).parent / "adalora_train.py"
EVAL_SCRIPT = Path(__file__).parent / "adalora_validate.py"
PYTHON = sys.executable

DEFAULT_MODELS = ALL_FINETUNE_MODELS
DEFAULT_CONFIGS = ALL_ADALORA_CONFIGS


# ── Helpers ───────────────────────────────────────────────────────────────


def run_subprocess(cmd: list[str], label: str) -> bool:
    print(f"\n{'\u2500' * 60}")
    print(f"  {label}")
    print(f"{'\u2500' * 60}")
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


# ── Argument parsing ─────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run all AdaLoRA fine-tuning experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=list(FINETUNE_MODEL_REGISTRY.keys()),
    )
    p.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        choices=list(ADALORA_CONFIGS.keys()),
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
        choices=["auto", "cpu", "cuda", "mps"],
    )
    p.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip adalora_train.py and only run val evaluation on existing adapters.",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="10 training steps per run only — validates the pipeline without full training.",
    )
    p.add_argument(
        "--no-push",
        action="store_true",
        help="Skip HF Hub upload (passed through to adalora_train.py).",
    )
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    total_runs = len(args.models) * len(args.configs)
    print(f"\n{'=' * 60}")
    print(f"  AdaLoRA Experiment Runner")
    print(f"  Models   : {args.models}")
    print(f"  Configs  : {args.configs}")
    print(f"  Dataset  : {args.dataset_size}")
    print(f"  Device   : {args.device}")
    print(f"  Runs     : {total_runs} training + {total_runs} val eval")
    if args.smoke_test:
        print(f"  Mode     : SMOKE TEST (10 steps per run)")
    if args.skip_training:
        print(f"  Training : SKIPPED")
    print(f"{'=' * 60}")

    results: list[dict] = []
    run_num = 0

    for model in args.models:
        for cfg in args.configs:
            run_num += 1
            run_tag = f"{model}_{cfg}_{args.dataset_size}"
            label = f"[{run_num}/{total_runs}] {run_tag}"

            # ── Training ──────────────────────────────────────────────────────
            train_ok = True
            if not args.skip_training:
                train_cmd = [
                    PYTHON,
                    str(TRAIN_SCRIPT),
                    "--model",
                    model,
                    "--adalora-config",
                    cfg,
                    "--dataset-size",
                    args.dataset_size,
                    "--device",
                    args.device,
                ]
                if args.smoke_test:
                    train_cmd.append("--smoke-test")
                if args.no_push:
                    train_cmd.append("--no-push")
                train_ok = run_subprocess(train_cmd, f"TRAIN  {label}")

            # ── Validation ────────────────────────────────────────────────────
            val_ok = False
            if not args.smoke_test:
                val_cmd = [
                    PYTHON,
                    str(EVAL_SCRIPT),
                    "--model",
                    model,
                    "--adalora-config",
                    cfg,
                    "--dataset-size",
                    args.dataset_size,
                    "--split",
                    "val",
                    "--device",
                    args.device,
                ]
                if not args.skip_training:
                    val_cmd.append("--local")
                val_ok = run_subprocess(val_cmd, f"VAL    {label}")

            results.append(
                {
                    "run_tag": run_tag,
                    "train_ok": train_ok,
                    "val_ok": val_ok,
                }
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  EXPERIMENT RUNNER COMPLETE")
    print(f"  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    for r in results:
        t = "OK" if r["train_ok"] else "FAIL"
        v = "OK" if r["val_ok"] else "SKIP" if args.smoke_test else "FAIL"
        print(f"  {r['run_tag']:35s}  train={t}  val={v}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
