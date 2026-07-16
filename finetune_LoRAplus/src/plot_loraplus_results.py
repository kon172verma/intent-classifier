#!/usr/bin/env python3
"""
finetune_LoRAplus/src/plot_loraplus_results.py
===============================================
Generate analysis charts for LoRA+ fine-tuning experiments.

All chart logic lives in finetune_lib/plot_lib.py (shared with LoRA/DoRA/AdaLoRA).

Outputs (saved to finetune_LoRAplus/analysis/)
----------------------------------------------
  loraplus_training_curves.png
  loraplus_combined_train.png
  loraplus_combined_val.png
  loraplus_combined_test.png

Note: only 2 models are plotted (qwen3-0.6b, llama3.2-1b).

Usage
-----
    python plot_loraplus_results.py
    python plot_loraplus_results.py --out-dir ../analysis
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from finetune_lib import ALL_LORAPLUS_MODELS, ALL_LORAPLUS_CONFIGS
from finetune_lib.plot_lib import plot_training_curves, plot_combined_accuracy_memory

LORAPLUS_DIR = Path(__file__).parent.parent
DEFAULT_TRAIN_DIR = LORAPLUS_DIR / "reports_training"
DEFAULT_VAL_DIR = LORAPLUS_DIR / "reports_validation"
DEFAULT_TEST_DIR = LORAPLUS_DIR / "reports_test"
DEFAULT_OUT_DIR = LORAPLUS_DIR / "analysis"
_TECHNIQUE = "LoRA+"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot LoRA+ experiment results.")
    p.add_argument("--train-reports-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    p.add_argument("--val-reports-dir", type=Path, default=DEFAULT_VAL_DIR)
    p.add_argument("--test-reports-dir", type=Path, default=DEFAULT_TEST_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Generating {_TECHNIQUE} plots → {args.out_dir}")
    print(f"  Models             : {ALL_LORAPLUS_MODELS}")
    print(f"  Training reports   : {args.train_reports_dir}")
    print(f"  Validation reports : {args.val_reports_dir}")
    print(f"  Test reports       : {args.test_reports_dir}")
    print()

    print("  [1/4] Training curves (2 models × 4 configs)...")
    plot_training_curves(
        train_reports_dir=args.train_reports_dir,
        all_models=ALL_LORAPLUS_MODELS,
        all_configs=ALL_LORAPLUS_CONFIGS,
        out_dir=args.out_dir,
        technique=_TECHNIQUE,
    )

    print("  [2/4] Combined chart — train split...")
    plot_combined_accuracy_memory(
        reports_dir=args.train_reports_dir,
        all_models=ALL_LORAPLUS_MODELS,
        all_configs=ALL_LORAPLUS_CONFIGS,
        out_dir=args.out_dir,
        split="train",
        technique=_TECHNIQUE,
    )

    print("  [3/4] Combined chart — val split...")
    plot_combined_accuracy_memory(
        reports_dir=args.val_reports_dir,
        all_models=ALL_LORAPLUS_MODELS,
        all_configs=ALL_LORAPLUS_CONFIGS,
        out_dir=args.out_dir,
        split="val",
        technique=_TECHNIQUE,
    )

    print("  [4/4] Combined chart — test split (skips if no reports)...")
    plot_combined_accuracy_memory(
        reports_dir=args.test_reports_dir,
        all_models=ALL_LORAPLUS_MODELS,
        all_configs=ALL_LORAPLUS_CONFIGS,
        out_dir=args.out_dir,
        split="test",
        technique=_TECHNIQUE,
    )

    print(f"\n  Done. All plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()
