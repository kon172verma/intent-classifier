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

# ── AdaLoRA adapter configs ────────────────────────────────────────────────────
# AdaLoRA decomposes updates as P·Λ·Q (SVD form) and prunes the least-important
# singular values during training.
#
# Key parameters:
#   init_r     – starting rank (budget before pruning; ~2–4× target_r)
#   target_r   – final rank after pruning (analogous to r in LoRA)
#   lora_alpha – scaling numerator. PEFT computes AdaLoRA scaling as
#                lora_alpha / init_r (NOT / target_r).  We set lora_alpha =
#                2 × init_r so the effective scaling is 2.0, matching every LoRA
#                config.  Leaving it unset falls back to PEFT's default of 8,
#                which yields 0.25–0.67 scaling — a 3–8× weaker adapter that
#                barely shifts greedy predictions (loss drops, accuracy doesn't).
#   beta1/2    – EMA smoothing for importance scores S_t = β·S_{t-1} + (1-β)·|∇Λ|·|Λ|.
#                0.85 ≈ 7-step moving average; balances noise vs. responsiveness.
#   deltaT     – steps between rank reallocation updates.
#                The paper used deltaT=10 on 5k-step GLUE runs (500 updates).
#                deltaT=1 at 300 steps = 300 rank shuffles — far too noisy.
#                deltaT=10 = 20 updates during the pruning phase (sensible).
#   orth_reg_weight – weight of the ||PᵀP−I||² + ||QQᵀ−I||² penalty that keeps
#                     P and Q orthogonal (required by SVD decomposition).
#                     PEFT default 0.5 was calibrated for 5k+ step runs and
#                     drowns task gradient at 300 steps. 0.1 is mild enough.
#   tinit      – warm-up steps: training at full init_r before any pruning.
#                Set to 1 epoch (50 steps) so the model learns the task before
#                AdaLoRA starts reorganising rank budget.
#   tfinal     – fine-tuning steps at the end: rank frozen at target_r.
#                Set to 1 epoch (50 steps) so the model consolidates at the
#                final rank after pruning is done.
#                Pruning phase = total_steps − tinit − tfinal = 200 steps.
#
# total_step must equal the actual number of training steps and is injected at
# runtime in adalora_train.py (it depends on dataset size and batch config).
#
# Configs are designed for direct comparison with the corresponding LoRA config:
#   A — same scope as LoRA-A, init 12 → target 4  (67% pruning, lighter final)
#   B — same scope + same final rank as LoRA-B,    init 24 → target 8
#   C — same scope + same final rank as LoRA-C,    init 32 → target 8  (75% pruning)
#   D — same scope + same final rank as LoRA-D,    init 32 → target 16 (50% pruning)
ADALORA_CONFIGS: dict[str, dict] = {
    "A": {
        "description": "Light adaptive — Q/V only, init 12 → target 4",
        "target_modules": ["q_proj", "v_proj"],
        "init_r": 12,
        "target_r": 4,
        "lora_alpha": 24,
        "beta1": 0.85,
        "beta2": 0.85,
        "orth_reg_weight": 0.1,
        "deltaT": 10,
        "tinit": 50,
        "tfinal": 50,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "learning_rate": 2e-4,
        "num_train_epochs": 6,
    },
    "B": {
        "description": "Standard adaptive — full attention, init 24 → target 8",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "init_r": 24,
        "target_r": 8,
        "lora_alpha": 48,
        "beta1": 0.85,
        "beta2": 0.85,
        "orth_reg_weight": 0.1,
        "deltaT": 10,
        "tinit": 50,
        "tfinal": 50,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "learning_rate": 2e-4,
        "num_train_epochs": 6,
    },
    "C": {
        "description": "Wide adaptive — full attention + MLP, init 32 → target 8",
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "init_r": 32,
        "target_r": 8,
        "lora_alpha": 64,
        "beta1": 0.85,
        "beta2": 0.85,
        "orth_reg_weight": 0.1,
        "deltaT": 10,
        "tinit": 50,
        "tfinal": 50,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "learning_rate": 2e-4,
        "num_train_epochs": 6,
    },
    "D": {
        "description": "Heavy adaptive — full attention + MLP, init 32 → target 16",
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "init_r": 32,
        "target_r": 16,
        "lora_alpha": 64,
        "beta1": 0.85,
        "beta2": 0.85,
        "orth_reg_weight": 0.1,
        "deltaT": 10,
        "tinit": 50,
        "tfinal": 50,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1e-4,
        "num_train_epochs": 6,
    },
}

ALL_ADALORA_CONFIGS: list[str] = list(ADALORA_CONFIGS.keys())
