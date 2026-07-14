#!/usr/bin/env python3
"""
finetune_LoRA/src/plot_lora_results.py
=======================================
Generate analysis charts for LoRA fine-tuning experiments.

All chart logic lives in finetune_lib/plot_lib.py so QLoRA and AdaLoRA
plot scripts can reuse the same functions with a different `technique` label.

Outputs (saved to finetune_LoRA/analysis/)
------------------------------------------
  lora_training_curves.png
      5 rows (one per model) × 4 cols (one per config).
      Each panel: train_loss, val_loss (left y-axis),
                  train_accuracy, val_accuracy (right y-axis).
      ★ marks step-0 (pre-fine-tuning baseline).

  lora_combined.png
      Row 1: final test accuracy (%) per model, 4 grouped bars (one per config).
      Row 2: peak inference memory (MB) per model, same structure.

Usage
-----
    python plot_lora_results.py
    python plot_lora_results.py --split test --out-dir ../analysis
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from finetune_lib import ALL_FINETUNE_MODELS, ALL_CONFIGS
from finetune_lib.plot_lib import plot_training_curves, plot_combined_accuracy_memory

LORA_DIR = Path(__file__).parent.parent
DEFAULT_TRAIN_DIR = LORA_DIR / "reports_training"
DEFAULT_VAL_DIR = LORA_DIR / "reports_validation"
DEFAULT_TEST_DIR = LORA_DIR / "reports_test"
DEFAULT_OUT_DIR = LORA_DIR / "analysis"
_TECHNIQUE = "LoRA"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot LoRA experiment results.")
    p.add_argument(
        "--split",
        default="test",
        choices=["val", "test"],
        help="Eval split for the combined accuracy/memory chart.",
    )
    p.add_argument("--train-reports-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    p.add_argument("--val-reports-dir", type=Path, default=DEFAULT_VAL_DIR)
    p.add_argument("--test-reports-dir", type=Path, default=DEFAULT_TEST_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = args.test_reports_dir if args.split == "test" else args.val_reports_dir

    print(f"\n  Generating {_TECHNIQUE} plots → {args.out_dir}")
    print(f"  Training reports : {args.train_reports_dir}")
    print(f"  Eval reports     : {eval_dir}  (split={args.split})")
    print()

    print("  [1/2] Training curves (5 models × 4 configs)...")
    plot_training_curves(
        train_reports_dir=args.train_reports_dir,
        all_models=ALL_FINETUNE_MODELS,
        all_configs=ALL_CONFIGS,
        out_dir=args.out_dir,
        technique=_TECHNIQUE,
    )

    print("  [2/2] Combined accuracy + memory chart...")
    plot_combined_accuracy_memory(
        test_reports_dir=eval_dir,
        all_models=ALL_FINETUNE_MODELS,
        all_configs=ALL_CONFIGS,
        out_dir=args.out_dir,
        technique=_TECHNIQUE,
    )

    print(f"\n  Done. All plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()
