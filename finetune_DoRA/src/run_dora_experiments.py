#!/usr/bin/env python3
"""
finetune_DoRA/src/run_dora_experiments.py
==========================================
Batch runner for all DoRA fine-tuning experiments.

All orchestration logic lives in finetune_LoRA/src/run_lora_experiments.py
(run_experiments_main); this script is a thin wrapper that passes the DoRA
train/eval scripts and the "DoRA" technique label.

Usage (identical flags to run_lora_experiments.py):
    python run_dora_experiments.py
    python run_dora_experiments.py --models qwen2.5-0.5b --configs A B
    python run_dora_experiments.py --skip-training
    python run_dora_experiments.py --smoke-test
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from finetune_lib.wrappers import run_experiments_wrapper

if __name__ == "__main__":
    run_experiments_wrapper(
        __file__,
        technique="DoRA",
        train_script_name="dora_train.py",
        eval_script_name="dora_validate.py",
    )
