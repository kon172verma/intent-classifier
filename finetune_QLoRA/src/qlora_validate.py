#!/usr/bin/env python3
"""
finetune_QLoRA/src/qlora_validate.py
======================================
Post-training evaluation of a QLoRA adapter on val or test split.

Loads the base model in the same 4-bit NF4 quantization used during training
(never a full-precision fallback) then evaluates the saved adapter. All
evaluation logic lives in finetune_LoRA/src/lora_validate.py (validate_main);
this script is a thin wrapper that redirects paths and the technique label
to QLoRA and sets quantize_4bit=True.

Usage (identical flags to lora_validate.py):
    python qlora_validate.py --model qwen3-0.6b  --lora-config B --split val --device cuda
    python qlora_validate.py --model llama3.2-1b --lora-config C --split test --device cuda
    python qlora_validate.py --model qwen3-0.6b  --lora-config A --split val --local --device cuda
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
_LORA_SRC = _REPO_ROOT / "finetune_LoRA" / "src"
for _p in (_REPO_ROOT, _LORA_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lora_validate import validate_main  # type: ignore  # noqa: E402
from finetune_lib import QLORA_MODEL_REGISTRY  # noqa: E402

if __name__ == "__main__":
    validate_main(
        technique="QLoRA",
        base_dir=Path(__file__).parent.parent,
        quantize_4bit=True,
        model_registry=QLORA_MODEL_REGISTRY,
    )
