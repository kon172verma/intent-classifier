#!/usr/bin/env python3
"""
finetune_LoRAplus/src/loraplus_train.py
========================================
LoRA+ fine-tuning for MCP tool-selection (intent classification).

LoRA+ (Hayou et al., 2024) is LoRA with an asymmetric learning rate:
matrix B is trained at `loraplus_lr_ratio × lr` while matrix A uses the base
learning rate `lr`.  The adapter structure is identical to LoRA, so inference,
validation, and HF Hub upload are unchanged.

All training logic lives in finetune_LoRA/src/lora_train.py (train_main);
this script is a thin wrapper that:
  1. Adds lora_train's directory to sys.path so it can be imported directly.
  2. Reads loraplus_lr_ratio from LORAPLUS_CONFIGS and passes it to train_main.
  3. Redirects adapters and reports to finetune_LoRAplus/ instead of
     finetune_LoRA/.

Supported models (LoRA+ subset):
  qwen3-0.6b   — Qwen/Qwen3-0.6B
  llama3.2-1b  — meta-llama/Llama-3.2-1B-Instruct (gated)

Usage (identical flags to lora_train.py):
    python loraplus_train.py --model qwen3-0.6b  --lora-config C --dataset-size 1k
    python loraplus_train.py --model llama3.2-1b --lora-config A --smoke-test
    python loraplus_train.py --model qwen3-0.6b  --lora-config B --no-push
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
_LORA_SRC = _REPO_ROOT / "finetune_LoRA" / "src"
for _p in (_REPO_ROOT, _LORA_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from finetune_lib import LORAPLUS_CONFIGS  # noqa: E402
from lora_train import train_main, parse_args  # type: ignore  # noqa: E402


def main() -> None:
    # parse_args() reads --lora-config from sys.argv so we can look up the ratio
    # before calling train_main (which also calls parse_args internally).
    _args = parse_args()
    _ratio = LORAPLUS_CONFIGS[_args.lora_config]["loraplus_lr_ratio"]

    train_main(
        technique="LoRA+",
        use_dora=False,
        base_dir=Path(__file__).parent.parent,
        loraplus_lr_ratio=_ratio,
    )


if __name__ == "__main__":
    main()
