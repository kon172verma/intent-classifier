#!/usr/bin/env python3
"""
finetune_AdaLoRA/src/plot_adalora_results.py
=============================================
Generate analysis charts for AdaLoRA fine-tuning experiments.

All chart logic lives in finetune_lib/plot_lib.py — this script is a thin
wrapper that passes the AdaLoRA report dirs and technique label.

Outputs (saved to finetune_AdaLoRA/analysis/)
---------------------------------------------
  adalora_training_curves.png
      5 rows (one per model) × 4 cols (one per config).
      Each panel: train_loss, val_loss (left y-axis),
                  train_accuracy, val_accuracy (right y-axis).
      ★ marks step-0 (pre-fine-tuning baseline).

  adalora_combined_train.png   Final train accuracy + peak training memory.
  adalora_combined_val.png     Val accuracy + peak inference memory.
  adalora_combined_test.png    Test accuracy + peak inference memory.

Usage
-----
    python plot_adalora_results.py
    python plot_adalora_results.py --out-dir ../analysis
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from finetune_lib import ALL_ADALORA_CONFIGS, ALL_ADALORA_MODELS
from finetune_lib.wrappers import parse_plot_args, run_plot_pipeline

ADALORA_DIR = Path(__file__).parent.parent
DEFAULT_TRAIN_DIR = ADALORA_DIR / "reports_training"
DEFAULT_VAL_DIR = ADALORA_DIR / "reports_validation"
DEFAULT_TEST_DIR = ADALORA_DIR / "reports_test"
DEFAULT_OUT_DIR = ADALORA_DIR / "analysis"
_TECHNIQUE = "AdaLoRA"


def parse_args() -> argparse.Namespace:
    return parse_plot_args(
        description="Plot AdaLoRA experiment results.",
        train_reports_dir=DEFAULT_TRAIN_DIR,
        val_reports_dir=DEFAULT_VAL_DIR,
        test_reports_dir=DEFAULT_TEST_DIR,
        out_dir=DEFAULT_OUT_DIR,
    )


def main() -> None:
    args = parse_args()
    run_plot_pipeline(
        args=args,
        technique=_TECHNIQUE,
        all_models=ALL_ADALORA_MODELS,
        all_configs=ALL_ADALORA_CONFIGS,
    )


if __name__ == "__main__":
    main()
