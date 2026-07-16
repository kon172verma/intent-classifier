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
_LORA_SRC = _REPO_ROOT / "finetune_LoRA" / "src"
for _p in (_REPO_ROOT, _LORA_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lora_validate import validate_main  # type: ignore  # noqa: E402

if __name__ == "__main__":
    validate_main(
        technique="LoRA+",
        base_dir=Path(__file__).parent.parent,
    )
