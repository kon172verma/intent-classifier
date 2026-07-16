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
_LORA_SRC = _REPO_ROOT / "finetune_LoRA" / "src"
for _p in (_REPO_ROOT, _LORA_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_lora_experiments import run_experiments_main  # type: ignore  # noqa: E402

_DORA_SRC = Path(__file__).parent
TRAIN_SCRIPT = _DORA_SRC / "dora_train.py"
EVAL_SCRIPT = _DORA_SRC / "dora_validate.py"

if __name__ == "__main__":
    run_experiments_main(
        technique="DoRA",
        train_script=TRAIN_SCRIPT,
        eval_script=EVAL_SCRIPT,
    )
