#!/usr/bin/env python3
"""
finetune_LoRAplus/src/loraplus_validate.py
===========================================
Post-training evaluation of a LoRA+ adapter on val or test split.

LoRA+ produces a standard LoRA adapter (use_dora=False); inference is identical
to LoRA.  All evaluation logic lives in finetune_LoRA/src/lora_validate.py
(validate_main); this script is a thin wrapper that redirects paths and the
technique label to LoRA+.

Usage (identical flags to lora_validate.py):
    python loraplus_validate.py --model qwen3-0.6b  --lora-config C --split val
    python loraplus_validate.py --model llama3.2-1b --lora-config A --split test
    python loraplus_validate.py --model qwen3-0.6b  --lora-config B --split val --local
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from finetune_lib.wrappers import run_validate_wrapper

if __name__ == "__main__":
    run_validate_wrapper(
        __file__,
        technique="LoRA+",
    )
