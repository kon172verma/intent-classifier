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

Note: all 5 models are plotted — see finetune_lib.ALL_LORAPLUS_MODELS
(aliases finetune_lib.ALL_FINETUNE_MODELS).

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

from finetune_lib import ALL_LORAPLUS_CONFIGS, ALL_LORAPLUS_MODELS
from finetune_lib.wrappers import parse_plot_args, run_plot_pipeline

LORAPLUS_DIR = Path(__file__).parent.parent
DEFAULT_TRAIN_DIR = LORAPLUS_DIR / "reports_training"
DEFAULT_VAL_DIR = LORAPLUS_DIR / "reports_validation"
DEFAULT_TEST_DIR = LORAPLUS_DIR / "reports_test"
DEFAULT_OUT_DIR = LORAPLUS_DIR / "analysis"
_TECHNIQUE = "LoRA+"


def parse_args() -> argparse.Namespace:
    return parse_plot_args(
        description="Plot LoRA+ experiment results.",
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
        all_models=ALL_LORAPLUS_MODELS,
        all_configs=ALL_LORAPLUS_CONFIGS,
    )


if __name__ == "__main__":
    main()
