#!/usr/bin/env python3
"""
finetune_DoRAplus/src/run_doraplus_experiments.py
===================================================
Batch runner for all DoRA+ fine-tuning experiments.

Experiment matrix: 5 models × 4 configs × 2 modes (train + val) = 40 runs.

All orchestration logic lives in finetune_LoRA/src/run_lora_experiments.py
(run_experiments_main); this script is a thin wrapper that passes the DoRA+
train/eval scripts and the "DoRA+" technique label. No model restriction is
applied — DoRA+ runs on the full 5-model matrix.

Usage (identical flags to run_lora_experiments.py):
    python run_doraplus_experiments.py
    python run_doraplus_experiments.py --models qwen3-0.6b --configs A B
    python run_doraplus_experiments.py --skip-training
    python run_doraplus_experiments.py --smoke-test
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
_LORA_SRC = _REPO_ROOT / "finetune_LoRA" / "src"
for _p in (_REPO_ROOT, _LORA_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_lora_experiments import run_experiments_main  # type: ignore  # noqa: E402

_DORAPLUS_SRC = Path(__file__).parent
TRAIN_SCRIPT = _DORAPLUS_SRC / "doraplus_train.py"
EVAL_SCRIPT = _DORAPLUS_SRC / "doraplus_validate.py"

if __name__ == "__main__":
    run_experiments_main(
        technique="DoRA+",
        train_script=TRAIN_SCRIPT,
        eval_script=EVAL_SCRIPT,
    )
