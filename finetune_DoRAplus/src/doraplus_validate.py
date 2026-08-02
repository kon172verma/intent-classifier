#!/usr/bin/env python3
"""
finetune_DoRAplus/src/doraplus_validate.py
============================================
Post-training evaluation of a DoRA+ adapter on val or test split.

DoRA+ produces a weight-decomposed (magnitude + direction) adapter; inference
is identical to DoRA. All evaluation logic lives in
finetune_LoRA/src/lora_validate.py (validate_main); this script is a thin
wrapper that redirects paths and the technique label to DoRA+.

Usage (identical flags to lora_validate.py):
    python doraplus_validate.py --model qwen3-0.6b  --lora-config C --split val
    python doraplus_validate.py --model llama3.2-1b --lora-config A --split test
    python doraplus_validate.py --model qwen3-0.6b  --lora-config B --split val --local
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
        technique="DoRA+",
    )
