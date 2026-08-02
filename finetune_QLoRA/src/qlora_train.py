#!/usr/bin/env python3
"""
finetune_QLoRA/src/qlora_train.py
===================================
QLoRA fine-tuning for MCP tool-selection (intent classification).

QLoRA (Dettmers et al. 2023) is LoRA with the frozen base model loaded in
4-bit NF4 (Normal Float 4-bit, double-quantized, bfloat16 compute) via
BitsAndBytesConfig — never fp4, and never a full-precision fallback. All
training logic lives in finetune_LoRA/src/lora_train.py (train_main); this
script is a thin wrapper that:
  1. Adds lora_train's directory to sys.path so it can be imported directly.
  2. Calls train_main(technique="QLoRA", quantize_4bit=True, base_dir=<this folder>)
     so adapters and reports land in finetune_QLoRA/ instead of finetune_LoRA/.

train_main() requires a CUDA device whenever quantize_4bit=True — bitsandbytes
has no CPU/MPS 4-bit kernels, so there is no silent full-precision fallback;
running on a non-CUDA host raises a RuntimeError instead of quietly training
without quantization.

Supported models (QLoRA subset — see finetune_lib.QLORA_MODEL_REGISTRY):
  qwen3-0.6b   — Qwen/Qwen3-0.6B
  llama3.2-1b  — meta-llama/Llama-3.2-1B-Instruct (gated)

Usage (identical flags to lora_train.py, --lora-config A/B/C/D reused):
    python qlora_train.py --model qwen3-0.6b  --lora-config B --dataset-size 1k --device cuda
    python qlora_train.py --model llama3.2-1b --lora-config A --smoke-test --device cuda
    python qlora_train.py --model qwen3-0.6b  --lora-config C --no-push --device cuda
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
_LORA_SRC = _REPO_ROOT / "finetune_LoRA" / "src"
for _p in (_REPO_ROOT, _LORA_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lora_train import train_main

from finetune_lib import QLORA_MODEL_REGISTRY

if __name__ == "__main__":
    train_main(
        technique="QLoRA",
        use_dora=False,
        base_dir=Path(__file__).parent.parent,
        quantize_4bit=True,
        model_registry=QLORA_MODEL_REGISTRY,
    )
