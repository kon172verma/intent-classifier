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

Supported models: all 5 (see finetune_lib.ALL_FINETUNE_MODELS).

Usage (identical flags to lora_train.py):
    python loraplus_train.py --model qwen3-0.6b  --lora-config C --dataset-size 1k
    python loraplus_train.py --model llama3.2-1b --lora-config A --smoke-test
    python loraplus_train.py --model qwen3-0.6b  --lora-config B --no-push
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from finetune_lib import LORAPLUS_CONFIGS
from finetune_lib.wrappers import run_train_wrapper


def main() -> None:
    run_train_wrapper(
        __file__,
        technique="LoRA+",
        use_dora=False,
        loraplus_configs=LORAPLUS_CONFIGS,
    )


if __name__ == "__main__":
    main()
