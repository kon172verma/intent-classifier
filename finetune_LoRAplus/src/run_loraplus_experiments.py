#!/usr/bin/env python3
"""
finetune_LoRAplus/src/run_loraplus_experiments.py
==================================================
Batch runner for all LoRA+ fine-tuning experiments.

Experiment matrix: 5 models × 4 configs × 2 modes (train + val) = 40 runs.

All orchestration logic lives in finetune_LoRA/src/run_lora_experiments.py
(run_experiments_main); this script is a thin wrapper that passes the LoRA+
train/eval scripts and the "LoRA+" technique label. No model restriction is
applied — LoRA+ runs on the full 5-model matrix.

Usage (identical flags to run_lora_experiments.py):
    python run_loraplus_experiments.py
    python run_loraplus_experiments.py --models qwen3-0.6b --configs A B
    python run_loraplus_experiments.py --skip-training
    python run_loraplus_experiments.py --smoke-test
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
        technique="LoRA+",
        train_script_name="loraplus_train.py",
        eval_script_name="loraplus_validate.py",
    )
