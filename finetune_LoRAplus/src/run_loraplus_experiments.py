#!/usr/bin/env python3
"""
finetune_LoRAplus/src/run_loraplus_experiments.py
==================================================
Batch runner for all LoRA+ fine-tuning experiments.

Experiment matrix: 2 models × 4 configs × 2 modes (train + val) = 16 runs.
Models: qwen3-0.6b, llama3.2-1b.

All orchestration logic lives in finetune_LoRA/src/run_lora_experiments.py
(run_experiments_main); this script is a thin wrapper that passes the LoRA+
train/eval scripts, the "LoRA+" technique label, and restricts the default
model list to the LoRA+ subset.

Usage (identical flags to run_lora_experiments.py):
    python run_loraplus_experiments.py
    python run_loraplus_experiments.py --models qwen3-0.6b --configs A B
    python run_loraplus_experiments.py --skip-training
    python run_loraplus_experiments.py --smoke-test
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
_LORA_SRC = _REPO_ROOT / "finetune_LoRA" / "src"
for _p in (_REPO_ROOT, _LORA_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_lora_experiments import run_experiments_main  # type: ignore  # noqa: E402

_LORAPLUS_SRC = Path(__file__).parent
TRAIN_SCRIPT = _LORAPLUS_SRC / "loraplus_train.py"
EVAL_SCRIPT = _LORAPLUS_SRC / "loraplus_validate.py"

if __name__ == "__main__":
    run_experiments_main(
        technique="LoRA+",
        train_script=TRAIN_SCRIPT,
        eval_script=EVAL_SCRIPT,
    )
