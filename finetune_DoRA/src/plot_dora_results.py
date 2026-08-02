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

from finetune_lib import ALL_CONFIGS, ALL_FINETUNE_MODELS
from finetune_lib.wrappers import parse_plot_args, run_plot_pipeline

DORA_DIR = Path(__file__).parent.parent
DEFAULT_TRAIN_DIR = DORA_DIR / "reports_training"
DEFAULT_VAL_DIR = DORA_DIR / "reports_validation"
DEFAULT_TEST_DIR = DORA_DIR / "reports_test"
DEFAULT_OUT_DIR = DORA_DIR / "analysis"
_TECHNIQUE = "DoRA"


def parse_args() -> argparse.Namespace:
    return parse_plot_args(
        description="Plot DoRA experiment results.",
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
