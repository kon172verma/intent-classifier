#!/usr/bin/env python3
"""
finetune_DoRAplus/src/plot_doraplus_results.py
================================================
Generate analysis charts for DoRA+ fine-tuning experiments.

All chart logic lives in finetune_lib/plot_lib.py (shared with
LoRA/DoRA/LoRA+/QLoRA/AdaLoRA).

Outputs (saved to finetune_DoRAplus/analysis/)
------------------------------------------------
  doraplus_training_curves.png
  doraplus_combined_train.png
  doraplus_combined_val.png
  doraplus_combined_test.png

Note: all 5 models are plotted — see finetune_lib.ALL_FINETUNE_MODELS.
Configs use LORAPLUS_CONFIGS (same hyperparameters as LoRA+; only the
use_dora flag differs, toggled in doraplus_train.py).

Usage
-----
    python plot_doraplus_results.py
    python plot_doraplus_results.py --out-dir ../analysis
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from finetune_lib import ALL_FINETUNE_MODELS, ALL_LORAPLUS_CONFIGS
from finetune_lib.wrappers import parse_plot_args, run_plot_pipeline

DORAPLUS_DIR = Path(__file__).parent.parent
DEFAULT_TRAIN_DIR = DORAPLUS_DIR / "reports_training"
DEFAULT_VAL_DIR = DORAPLUS_DIR / "reports_validation"
DEFAULT_TEST_DIR = DORAPLUS_DIR / "reports_test"
DEFAULT_OUT_DIR = DORAPLUS_DIR / "analysis"
_TECHNIQUE = "DoRA+"


def parse_args() -> argparse.Namespace:
    return parse_plot_args(
        description="Plot DoRA+ experiment results.",
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
        all_configs=ALL_LORAPLUS_CONFIGS,
    )


if __name__ == "__main__":
    main()
