#!/usr/bin/env python3
"""
finetune_LoRA/src/run_lora_experiments.py
==========================================
Batch runner for all LoRA fine-tuning experiments.

Experiment matrix: 5 models × 4 configs × 2 modes (train + val) = 40 runs.
Models are iterated in increasing parameter order.
For each run:
  1. lora_train.py    (skipped with --skip-training)
  2. lora_validate.py --split val

Test-set evaluation is intentionally separate: run lora_validate.py
with --split test manually after reviewing validation results to avoid
accidental leakage from repeated test checks.

Usage
-----
    # Full matrix (1k dataset)
    python run_lora_experiments.py

    # Specific models / configs
    python run_lora_experiments.py --models qwen2.5-0.5b llama3.2-1b --configs A B

    # 10k dataset
    python run_lora_experiments.py --dataset-size 10k

    # Validate only (training already done, adapters on HF)
    python run_lora_experiments.py --skip-training

    # Quick pipeline check (10 steps per run, no adapter saved)
    python run_lora_experiments.py --smoke-test
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

from finetune_lib import (  # noqa: E402
    FINETUNE_MODEL_REGISTRY,
    ALL_FINETUNE_MODELS,
    LORA_CONFIGS,
    ALL_CONFIGS,
)

LORA_DIR = Path(__file__).parent.parent
TRAIN_SCRIPT = Path(__file__).parent / "lora_train.py"
EVAL_SCRIPT = Path(__file__).parent / "lora_validate.py"
PYTHON = sys.executable

DEFAULT_MODELS = ALL_FINETUNE_MODELS  # all 5 models
DEFAULT_CONFIGS = ALL_CONFIGS  # A, B, C, D


# ── Helpers ───────────────────────────────────────────────────────────────


def run_subprocess(cmd: list[str], label: str) -> bool:
    print(f"\n{'\u2500' * 60}")
    print(f"  {label}")
    print(f"{'\u2500' * 60}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    assert proc.stdout is not None
    _buf: list[str] = []
    while True:
        ch = proc.stdout.read(1)
        if not ch:
            if _buf:
                line = "".join(_buf)
                if not line.startswith("Failed to load "):
                    print(line, end="", flush=True)
            break
        _buf.append(ch)
        if ch in ("\r", "\n"):
            line = "".join(_buf)
            if not line.startswith("Failed to load "):
                print(line, end="", flush=True)
            _buf = []
    proc.stdout.close()
    proc.wait()
    if proc.returncode != 0:
        print(f"\n  [ERROR] Exited with code {proc.returncode}")
    return proc.returncode == 0


# ── Argument parsing ─────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run all LoRA fine-tuning experiments.",
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
        choices=list(LORA_CONFIGS.keys()),
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
        help="Skip lora_train.py and only run val evaluation on existing adapters.",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="10 training steps per run only — validates the pipeline without full training.",
    )
    p.add_argument(
        "--no-push",
        action="store_true",
        help="Skip HF Hub upload (passed through to lora_train.py).",
    )
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────


def run_experiments_main(
    technique: str = "LoRA",
    train_script: Path | None = None,
    eval_script: Path | None = None,
) -> None:
    if train_script is None:
        train_script = TRAIN_SCRIPT
    if eval_script is None:
        eval_script = EVAL_SCRIPT

    args = parse_args()

    total_runs = len(args.models) * len(args.configs)
    print(f"\n{'=' * 60}")
    print(f"  {technique} Experiment Runner")
    print(f"  Models   : {args.models}")
    print(f"  Configs  : {args.configs}")
    print(f"  Dataset  : {args.dataset_size}")
    print(f"  Device   : {args.device}")
    print(f"  Runs     : {total_runs} training + {total_runs} val eval")
    if args.smoke_test:
        print("  Mode     : SMOKE TEST (10 steps per run)")
    if args.skip_training:
        print("  Training : SKIPPED")
    print(f"{'=' * 60}")

    results: list[dict] = []
    run_num = 0

    for model in args.models:
        for cfg in args.configs:
            run_num += 1
            run_tag = f"{model}_{cfg}_{args.dataset_size}"
            label = f"[{run_num}/{total_runs}] {run_tag}"

            # ── Training ────────────────────────────────────────────────────────────
            train_ok = True
            if not args.skip_training:
                train_cmd = [
                    PYTHON,
                    "-u",
                    str(train_script),
                    "--model",
                    model,
                    "--lora-config",
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

            # ── Validation ───────────────────────────────────────────────────────────
            val_ok = False
            if not args.smoke_test:
                val_cmd = [
                    PYTHON,
                    "-u",
                    str(eval_script),
                    "--model",
                    model,
                    "--lora-config",
                    cfg,
                    "--dataset-size",
                    args.dataset_size,
                    "--split",
                    "val",
                    "--device",
                    args.device,
                ]
                # Load from local adapters/ if training ran (adapters freshly saved there);
                # load from HF Hub if --skip-training (relies on previously uploaded adapters).
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
    print("  EXPERIMENT RUNNER COMPLETE")
    print(f"  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    for r in results:
        t = "OK" if r["train_ok"] else "FAIL"
        v = "OK" if r["val_ok"] else "SKIP" if args.smoke_test else "FAIL"
        print(f"  {r['run_tag']:35s}  train={t}  val={v}")
    print(f"{'=' * 60}\n")


def main() -> None:
    run_experiments_main("LoRA", TRAIN_SCRIPT, EVAL_SCRIPT)


if __name__ == "__main__":
    main()
