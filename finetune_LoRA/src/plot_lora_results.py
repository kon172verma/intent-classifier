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

  lora_combined_train.png   Final train accuracy + peak training memory.
  lora_combined_val.png     Val accuracy + peak inference memory.
  lora_combined_test.png    Test accuracy + peak inference memory.

Usage
-----
    python plot_lora_results.py
    python plot_lora_results.py --out-dir ../analysis
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from finetune_lib import ALL_CONFIGS, ALL_FINETUNE_MODELS
from finetune_lib.wrappers import parse_plot_args, run_plot_pipeline

LORA_DIR = Path(__file__).parent.parent
DEFAULT_TRAIN_DIR = LORA_DIR / "reports_training"
DEFAULT_VAL_DIR = LORA_DIR / "reports_validation"
DEFAULT_TEST_DIR = LORA_DIR / "reports_test"
DEFAULT_OUT_DIR = LORA_DIR / "analysis"
_TECHNIQUE = "LoRA"


def parse_args() -> argparse.Namespace:
    return parse_plot_args(
        description="Plot LoRA experiment results.",
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
        all_models=ALL_FINETUNE_MODELS,
        all_configs=ALL_CONFIGS,
    )


if __name__ == "__main__":
    main()
