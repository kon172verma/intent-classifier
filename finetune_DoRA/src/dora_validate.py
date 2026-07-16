#!/usr/bin/env python3
"""
finetune_DoRA/src/dora_validate.py
=====================================
Post-training evaluation of a DoRA adapter on val or test split.

All evaluation logic lives in finetune_LoRA/src/lora_validate.py
(validate_main); this script is a thin wrapper that redirects paths and the
technique label to DoRA.

Usage (identical flags to lora_validate.py):
    python dora_validate.py --model qwen2.5-0.5b --lora-config B --split val
    python dora_validate.py --model llama3.2-1b  --lora-config C --split test
    python dora_validate.py --model smollm2-360m --lora-config A --split val --local
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
        technique="DoRA",
        base_dir=Path(__file__).parent.parent,
    )
