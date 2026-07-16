#!/usr/bin/env python3
"""
finetune_DoRA/src/plot_dora_results.py
=======================================
Generate analysis charts for DoRA fine-tuning experiments.

All chart logic lives in finetune_lib/plot_lib.py (shared with LoRA/AdaLoRA).

Outputs (saved to finetune_DoRA/analysis/)
------------------------------------------
  dora_training_curves.png
  dora_combined_train.png
  dora_combined_val.png
  dora_combined_test.png

Usage
-----
    python plot_dora_results.py
    python plot_dora_results.py --out-dir ../analysis
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from finetune_lib import ALL_FINETUNE_MODELS, ALL_CONFIGS
from finetune_lib.plot_lib import plot_training_curves, plot_combined_accuracy_memory

DORA_DIR = Path(__file__).parent.parent
DEFAULT_TRAIN_DIR = DORA_DIR / "reports_training"
DEFAULT_VAL_DIR = DORA_DIR / "reports_validation"
DEFAULT_TEST_DIR = DORA_DIR / "reports_test"
DEFAULT_OUT_DIR = DORA_DIR / "analysis"
_TECHNIQUE = "DoRA"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot DoRA experiment results.")
    p.add_argument("--train-reports-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    p.add_argument("--val-reports-dir", type=Path, default=DEFAULT_VAL_DIR)
    p.add_argument("--test-reports-dir", type=Path, default=DEFAULT_TEST_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Generating {_TECHNIQUE} plots → {args.out_dir}")
    print(f"  Training reports   : {args.train_reports_dir}")
    print(f"  Validation reports : {args.val_reports_dir}")
    print(f"  Test reports       : {args.test_reports_dir}")
    print()

    print("  [1/4] Training curves (5 models × 4 configs)...")
    plot_training_curves(
        train_reports_dir=args.train_reports_dir,
        all_models=ALL_FINETUNE_MODELS,
        all_configs=ALL_CONFIGS,
        out_dir=args.out_dir,
        technique=_TECHNIQUE,
    )

    print("  [2/4] Combined chart — train split...")
    plot_combined_accuracy_memory(
        reports_dir=args.train_reports_dir,
        all_models=ALL_FINETUNE_MODELS,
        all_configs=ALL_CONFIGS,
        out_dir=args.out_dir,
        split="train",
        technique=_TECHNIQUE,
    )

    print("  [3/4] Combined chart — val split...")
    plot_combined_accuracy_memory(
        reports_dir=args.val_reports_dir,
        all_models=ALL_FINETUNE_MODELS,
        all_configs=ALL_CONFIGS,
        out_dir=args.out_dir,
        split="val",
        technique=_TECHNIQUE,
    )

    print("  [4/4] Combined chart — test split (skips if no reports)...")
    plot_combined_accuracy_memory(
        reports_dir=args.test_reports_dir,
        all_models=ALL_FINETUNE_MODELS,
        all_configs=ALL_CONFIGS,
        out_dir=args.out_dir,
        split="test",
        technique=_TECHNIQUE,
    )

    print(f"\n  Done. All plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()
