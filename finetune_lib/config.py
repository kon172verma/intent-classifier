"""
finetune_lib/config.py
======================
Shared configuration for all fine-tuning experiments (LoRA, QLoRA, AdaLoRA).

Exports
-------
  FINETUNE_MODEL_REGISTRY  – 5 selected models (keys reused from evaluation_lib)
  ALL_FINETUNE_MODELS      – ordered list (smallest → largest)
  QWEN3_FINETUNE_KEYS      – models needing enable_thinking=False
  SYSTEM_PROMPT            – tool-router system prompt (identical to zero-shot eval)
  LORA_CONFIGS             – shared A/B/C/D adapter configs
  MAX_SEQ_LENGTH           – tokenisation context cap
  HF_HUB_REPO              – target HuggingFace repo for all adapter uploads
  hf_adapter_subfolder()   – helper to build the per-run subfolder path
"""

from __future__ import annotations

# Reuse evaluation_lib for shared constants (avoids duplication).
from evaluation_lib.config import (
    MODEL_REGISTRY as _EVAL_REGISTRY,
    SYSTEM_PROMPT_ZERO_SHOT as SYSTEM_PROMPT,  # noqa: F401
)

# ── Fine-tune model subset ─────────────────────────────────────────────────────
# 5 models selected for LoRA / QLoRA / AdaLoRA experiments.
# Must be valid keys in evaluation_lib.config.MODEL_REGISTRY.
_FINETUNE_KEYS: list[str] = [
    "smollm2-360m",  # tiny,   384M instruct,   HuggingFaceTB/SmolLM2-360M-Instruct
    "qwen2.5-0.5b",  # small,  494M instruct,   Qwen/Qwen2.5-0.5B-Instruct
    "qwen3-0.6b",  # small,  600M base+chat,  Qwen/Qwen3-0.6B
    "llama3.2-1b",  # medium, 1.2B instruct,   meta-llama/Llama-3.2-1B-Instruct (gated)
    "smollm2-1.7b",  # medium, 1.7B instruct,   HuggingFaceTB/SmolLM2-1.7B-Instruct
]

FINETUNE_MODEL_REGISTRY: dict[str, str] = {k: _EVAL_REGISTRY[k] for k in _FINETUNE_KEYS}

ALL_FINETUNE_MODELS: list[str] = _FINETUNE_KEYS

# ── Chat-template quirks ───────────────────────────────────────────────────────
# Qwen3 unified base+chat defaults to thinking mode — always disable for
# deterministic, single-token routing output.
QWEN3_FINETUNE_KEYS: frozenset[str] = frozenset({"qwen3-0.6b"})

# ── HuggingFace Hub ────────────────────────────────────────────────────────────
# All adapters are pushed to a single HF model repo using subfolders:
#
#   {HF_HUB_REPO}/{technique}/{model_key}_{lora_config}_{dataset_size}/
#
# Loading a saved adapter:
#   from peft import PeftModel
#   model = PeftModel.from_pretrained(
#       base_model,
#       HF_HUB_REPO,
#       subfolder="LoRA/qwen2.5-0.5b_B_1k",
#   )
#   merged = model.merge_and_unload()
HF_HUB_REPO: str = "kon172verma/intent-classifier"


def hf_adapter_subfolder(
    technique: str,
    model_key: str,
    lora_config: str,
    dataset_size: str,
) -> str:
    """Return the HF subfolder path for a specific adapter.

    Example: hf_adapter_subfolder("LoRA", "qwen2.5-0.5b", "B", "1k")
             → "LoRA/qwen2.5-0.5b_B_1k"
    """
    return f"{technique}/{model_key}_{lora_config}_{dataset_size}"


# ── Tokenisation ───────────────────────────────────────────────────────────────
# Worst-case prompt (30 tools × ~30 tok + system + request) ≈ 975 tokens.
MAX_SEQ_LENGTH: int = 1024

# ── LoRA / QLoRA / AdaLoRA adapter configs ────────────────────────────────────
# Shared across all finetune techniques for direct cross-technique comparisons.
# target_modules tested on: Qwen2.5, Qwen3, SmolLM2 (Llama-2 arch),
# and Llama-3.2 architectures.
#
# Config A — Light:    Q+V only,       rank 8   (fast diagnostic; fewest params)
# Config B — Standard: full attention, rank 16  (recommended baseline)
# Config C — Wide:     attn + MLP,     rank 16  (wider coverage, same rank as B)
# Config D — Heavy:    attn + MLP,     rank 32  (max capacity; may overfit on 1k)
#
# lora_alpha = 2 × rank  (standard scaling convention)
# Effective batch = per_device_train_batch_size × gradient_accumulation_steps = 16
LORA_CONFIGS: dict[str, dict] = {
    "A": {
        "description": "Light — attention Q/V only, rank 8",
        "target_modules": ["q_proj", "v_proj"],
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "learning_rate": 2e-4,
        "num_train_epochs": 3,
    },
    "B": {
        "description": "Standard — full attention, rank 16",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1e-4,
        "num_train_epochs": 3,
    },
    "C": {
        "description": "Wide — full attention + MLP, rank 16",
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1e-4,
        "num_train_epochs": 3,
    },
    "D": {
        "description": "Heavy — full attention + MLP, rank 32",
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5e-5,
        "num_train_epochs": 3,
    },
}

ALL_CONFIGS: list[str] = list(LORA_CONFIGS.keys())
