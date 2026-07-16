#!/usr/bin/env python3
"""
finetune_LoRAplus/src/prepare_loraplus_data.py
===============================================
Prepare train / val / test JSONL splits for LoRA+ fine-tuning.

The split logic is identical to LoRA — this script is a thin wrapper around
finetune_LoRA/src/prepare_lora_data.py that redirects the default output
directory to finetune_LoRAplus/data/.

Usage:
    python prepare_loraplus_data.py --dataset-size 1k
    python prepare_loraplus_data.py --dataset-size 10k
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
_LORA_SRC = _REPO_ROOT / "finetune_LoRA" / "src"
for _p in (_REPO_ROOT, _LORA_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import prepare_lora_data as _prep  # type: ignore  # noqa: E402

# Redirect default output to finetune_LoRAplus/data/ instead of finetune_LoRA/data/
_prep.DEFAULT_OUT_DIR = Path(__file__).parent.parent / "data"

if __name__ == "__main__":
    _prep.main()
