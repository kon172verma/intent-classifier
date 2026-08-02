#!/usr/bin/env python3
"""
finetune_QLoRA/src/plot_qlora_results.py
==========================================
Generate analysis charts for QLoRA fine-tuning experiments.

All chart logic lives in finetune_lib/plot_lib.py (shared with LoRA/DoRA/LoRA+/AdaLoRA).

Outputs (saved to finetune_QLoRA/analysis/)
--------------------------------------------
  qlora_training_curves.png
  qlora_combined_train.png
  qlora_combined_val.png
  qlora_combined_test.png

Note: only 2 models are plotted (qwen3-0.6b, llama3.2-1b) — see
finetune_lib.ALL_QLORA_MODELS.

Usage
-----
    python plot_qlora_results.py
    python plot_qlora_results.py --out-dir ../analysis
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from finetune_lib import ALL_CONFIGS, ALL_QLORA_MODELS
from finetune_lib.wrappers import parse_plot_args, run_plot_pipeline

QLORA_DIR = Path(__file__).parent.parent
DEFAULT_TRAIN_DIR = QLORA_DIR / "reports_training"
DEFAULT_VAL_DIR = QLORA_DIR / "reports_validation"
DEFAULT_TEST_DIR = QLORA_DIR / "reports_test"
DEFAULT_OUT_DIR = QLORA_DIR / "analysis"
_TECHNIQUE = "QLoRA"


def parse_args() -> argparse.Namespace:
    return parse_plot_args(
        description="Plot QLoRA experiment results.",
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
        all_models=ALL_QLORA_MODELS,
        all_configs=ALL_CONFIGS,
    )


if __name__ == "__main__":
    main()
