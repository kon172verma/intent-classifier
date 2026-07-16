#!/usr/bin/env python3
"""
finetune_DoRA/src/dora_train.py
================================
DoRA fine-tuning for MCP tool-selection (intent classification).

DoRA (Weight-Decomposed Low-Rank Adaptation) is LoRA with use_dora=True in
LoraConfig.  All training logic lives in finetune_LoRA/src/lora_train.py
(train_main); this script is a thin wrapper that:
  1. Adds lora_train's directory to sys.path so it can be imported directly.
  2. Calls train_main(technique="DoRA", use_dora=True, base_dir=<this folder>)
     so adapters and reports land in finetune_DoRA/ instead of finetune_LoRA/.

Usage (identical flags to lora_train.py, --lora-config A/B/C/D reused):
    python dora_train.py --model qwen2.5-0.5b --lora-config B --dataset-size 1k
    python dora_train.py --model smollm2-360m --lora-config A --smoke-test
    python dora_train.py --model llama3.2-1b  --lora-config C --no-push
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
_LORA_SRC = _REPO_ROOT / "finetune_LoRA" / "src"
for _p in (_REPO_ROOT, _LORA_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lora_train import train_main  # type: ignore  # noqa: E402

if __name__ == "__main__":
    train_main(
        technique="DoRA",
        use_dora=True,
        base_dir=Path(__file__).parent.parent,
    )
