#!/usr/bin/env python3
"""
finetune_QLoRA/src/run_qlora_experiments.py
=============================================
Batch runner for all QLoRA fine-tuning experiments.

All orchestration logic lives in finetune_LoRA/src/run_lora_experiments.py
(run_experiments_main); this script is a thin wrapper that passes the QLoRA
train/eval scripts, the "QLoRA" technique label, and restricts the model
matrix to finetune_lib.QLORA_MODEL_REGISTRY (qwen3-0.6b, llama3.2-1b).

Experiment matrix: 2 models × 4 configs × 2 modes (train + val) = 16 runs.

QLoRA requires CUDA (bitsandbytes NF4 has no CPU/MPS kernels); pass
--device cuda explicitly on a GPU host.

Usage (identical flags to run_lora_experiments.py):
    python run_qlora_experiments.py --device cuda
    python run_qlora_experiments.py --models qwen3-0.6b --configs A B --device cuda
    python run_qlora_experiments.py --skip-training
    python run_qlora_experiments.py --smoke-test --device cuda
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from finetune_lib import ALL_QLORA_MODELS, QLORA_MODEL_REGISTRY
from finetune_lib.wrappers import run_experiments_wrapper

if __name__ == "__main__":
    run_experiments_wrapper(
        __file__,
        technique="QLoRA",
        train_script_name="qlora_train.py",
        eval_script_name="qlora_validate.py",
        model_registry=QLORA_MODEL_REGISTRY,
        default_models=ALL_QLORA_MODELS,
    )
