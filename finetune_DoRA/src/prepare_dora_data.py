#!/usr/bin/env python3
"""
finetune_DoRA/src/prepare_dora_data.py
=========================================
Prepare train / val / test JSONL splits for DoRA fine-tuning.

The split logic is identical to LoRA — this script is a thin wrapper around
finetune_LoRA/src/prepare_lora_data.py that redirects the default output
directory to finetune_DoRA/data/.

Usage:
    python prepare_dora_data.py --dataset-size 1k
    python prepare_dora_data.py --dataset-size 10k
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from finetune_lib.wrappers import run_prepare_wrapper

if __name__ == "__main__":
    run_prepare_wrapper(__file__)
