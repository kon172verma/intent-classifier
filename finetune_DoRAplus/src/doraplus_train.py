#!/usr/bin/env python3
"""
finetune_DoRAplus/src/doraplus_train.py
=========================================
DoRA+ fine-tuning for MCP tool-selection (intent classification).

DoRA+ combines DoRA (Weight-Decomposed Low-Rank Adaptation — Liu et al. 2024,
use_dora=True) with LoRA+'s asymmetric learning rate (Hayou et al. 2024):
matrix B is trained at `loraplus_lr_ratio × lr` while matrix A and the DoRA
magnitude vector use the base learning rate `lr`. Hyperparameters are shared
with LoRA+ (finetune_lib.LORAPLUS_CONFIGS) — only the use_dora flag differs.

All training logic lives in finetune_LoRA/src/lora_train.py (train_main);
this script is a thin wrapper that:
  1. Adds lora_train's directory to sys.path so it can be imported directly.
  2. Reads loraplus_lr_ratio from LORAPLUS_CONFIGS and passes it to train_main.
  3. Passes use_dora=True.
  4. Redirects adapters and reports to finetune_DoRAplus/ instead of
     finetune_LoRA/.

Supported models: all 5 (see finetune_lib.ALL_FINETUNE_MODELS).

Usage (identical flags to lora_train.py):
    python doraplus_train.py --model qwen3-0.6b  --lora-config C --dataset-size 1k
    python doraplus_train.py --model llama3.2-1b --lora-config A --smoke-test
    python doraplus_train.py --model qwen3-0.6b  --lora-config B --no-push
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
_LORA_SRC = _REPO_ROOT / "finetune_LoRA" / "src"
for _p in (_REPO_ROOT, _LORA_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lora_train import parse_args, train_main

from finetune_lib import LORAPLUS_CONFIGS


def main() -> None:
    # parse_args() reads --lora-config from sys.argv so we can look up the ratio
    # before calling train_main (which also calls parse_args internally).
    _args = parse_args()
    _ratio = LORAPLUS_CONFIGS[_args.lora_config]["loraplus_lr_ratio"]

    train_main(
        technique="DoRA+",
        use_dora=True,
        base_dir=Path(__file__).parent.parent,
        loraplus_lr_ratio=_ratio,
    )


if __name__ == "__main__":
    main()
