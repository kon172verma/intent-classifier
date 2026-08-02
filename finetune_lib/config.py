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
  CURRENT_VERSION          – release-in-progress version, read from VERSION file
  HF_EXPERIMENTS_REPO      – HF repo that receives every adapter pushed during experimentation
  HF_RELEASE_REPO          – HF repo that receives only the final merged/GGUF/ONNX release models
  generate_experiment_timestamp() – UTC timestamp used in adapter experiment names
  hf_adapter_subfolder()   – helper to build the per-run adapter subfolder path (experiments repo)
  hf_merged_subfolder()    – helper to build the per-run merged-model subfolder path (release repo)
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

# Reuse evaluation_lib for shared constants (avoids duplication).
from evaluation_lib.config import (
    MODEL_REGISTRY as _EVAL_REGISTRY,
)
from evaluation_lib.config import (
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

# ── Restricted model subset (QLoRA, AdaLoRA) ──────────────────────────────────
# QLoRA and AdaLoRA are limited to 2 models: the small Qwen3 base+chat model
# and the gated Llama-3.2 instruct model. QLoRA needs a CUDA + bitsandbytes
# host to exercise real NF4 quantization; AdaLoRA's rank-reallocation loop is
# expensive, so 2 representative models are enough to compare against LoRA.
_RESTRICTED_KEYS: list[str] = [
    "qwen3-0.6b",  # small,  600M base+chat,  Qwen/Qwen3-0.6B
    "llama3.2-1b",  # medium, 1.2B instruct,   meta-llama/Llama-3.2-1B-Instruct (gated)
]

QLORA_MODEL_REGISTRY: dict[str, str] = {k: _EVAL_REGISTRY[k] for k in _RESTRICTED_KEYS}
ALL_QLORA_MODELS: list[str] = _RESTRICTED_KEYS

ADALORA_MODEL_REGISTRY: dict[str, str] = {k: _EVAL_REGISTRY[k] for k in _RESTRICTED_KEYS}
ALL_ADALORA_MODELS: list[str] = _RESTRICTED_KEYS

# ── Chat-template quirks ───────────────────────────────────────────────────────
# Qwen3 unified base+chat defaults to thinking mode — always disable for
# deterministic, single-token routing output.
QWEN3_FINETUNE_KEYS: frozenset[str] = frozenset({"qwen3-0.6b"})

# ── HuggingFace Hub ────────────────────────────────────────────────────────────
# Two separate HF repos, split by responsibility:
#
#   HF_EXPERIMENTS_REPO — every adapter produced during experimentation.
#     Layout:  {HF_EXPERIMENTS_REPO}/{version}/{model_key}_{technique}_{lora_config}_{dataset_size}_{timestamp}/
#
#   HF_RELEASE_REPO — only the 2 best merged models chosen per version release,
#     plus their GGUF and ONNX exports.
#     Layout:  {HF_RELEASE_REPO}/{technique}_merged/{model_key}_{lora_config}_{dataset_size}/
#
# Loading a saved adapter:
#   from peft import PeftModel
#   model = PeftModel.from_pretrained(
#       base_model,
#       HF_EXPERIMENTS_REPO,
#       subfolder="v1.0/qwen2.5-0.5b_LoRA_B_1k_20260101-120000",
#   )
#   merged = model.merge_and_unload()
HF_EXPERIMENTS_REPO: str = "kon172verma/intent-classifier-experiments"
HF_RELEASE_REPO: str = "kon172verma/intent-classifier"

# ── Version tracking ──────────────────────────────────────────────────────────
# CURRENT_VERSION is the version folder that new experiments are pushed under.
# It is read from the VERSION file at the repo root so bumping the version for
# a new round of experiments is a one-line file edit, not a code change.
_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_VERSION_FILE: Path = _REPO_ROOT / "VERSION"


def _read_current_version() -> str:
    if _VERSION_FILE.exists():
        contents = _VERSION_FILE.read_text(encoding="utf-8").strip()
        if contents:
            return contents
    return "v1.0"


CURRENT_VERSION: str = _read_current_version()


def generate_experiment_timestamp() -> str:
    """Return a UTC timestamp (YYYYMMDD-HHMMSS) for tagging a new experiment."""
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def hf_adapter_subfolder(
    technique: str,
    model_key: str,
    lora_config: str,
    dataset_size: str,
    timestamp: str | None = None,
    version: str | None = None,
) -> str:
    """Return the HF_EXPERIMENTS_REPO subfolder path for a specific adapter run.

    Naming convention: {version}/{model}_{technique}_{config}_{size}_{timestamp}

    Example: hf_adapter_subfolder("LoRA", "qwen2.5-0.5b", "B", "1k", "20260101-120000")
             → "v1.0/qwen2.5-0.5b_LoRA_B_1k_20260101-120000"
    """
    version = version or CURRENT_VERSION
    timestamp = timestamp or generate_experiment_timestamp()
    return f"{version}/{model_key}_{technique}_{lora_config}_{dataset_size}_{timestamp}"


def hf_merged_subfolder(model_key: str) -> str:
    """Return the HF_RELEASE_REPO subfolder path for a merged (adapter-unloaded) model.

    Layout: <model_key>/safetensors/

    Example: hf_merged_subfolder("qwen3-0.6b") → "qwen3-0.6b/safetensors"
    """
    return f"{model_key}/safetensors"


# ── Tokenisation ───────────────────────────────────────────────────────────────
# Worst-case prompt (30 tools × ~30 tok + system + request) ≈ 975 tokens.
MAX_SEQ_LENGTH: int = 1024

# ── Early stopping (patience) ─────────────────────────────────────────────────
# Applies to every technique that shares finetune_LoRA/src/lora_train.py's
# train_main() — LoRA, DoRA, LoRA+, DoRA+, and QLoRA. Patience is counted in
# *evaluation calls* (eval_strategy="steps"), not epochs: with num_train_epochs=4
# and ~2 evals/epoch that's ~8 evals per run, so a patience of 2 stops training
# once a full epoch's worth of evals shows no val_loss improvement.
#
# AdaLoRA intentionally does NOT use early stopping: its rank-reallocation
# schedule (tinit/deltaT/tfinal) makes eval_loss noisy mid-training, and it
# needs to run its full epoch budget to reach target_r and consolidate.
EARLY_STOPPING_PATIENCE: int = 2
EARLY_STOPPING_THRESHOLD: float = 0.001

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
        "num_train_epochs": 4,
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
        "num_train_epochs": 4,
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
        "num_train_epochs": 4,
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
        "num_train_epochs": 4,
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
        "num_train_epochs": 10,
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
        "num_train_epochs": 10,
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
        "num_train_epochs": 10,
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
        "num_train_epochs": 10,
    },
}

ALL_ADALORA_CONFIGS: list[str] = list(ADALORA_CONFIGS.keys())

# ── LoRA+ model subset ────────────────────────────────────────────────────────
# LoRA+ (and DoRA+) run on all 5 fine-tune models — same lineup as LoRA/DoRA —
# so the asymmetric-learning-rate gain can be compared model-for-model.
ALL_LORAPLUS_MODELS: list[str] = ALL_FINETUNE_MODELS

# ── LoRA+ adapter configs ─────────────────────────────────────────────────────
# LoRA+ (Hayou et al., 2024) keeps the same LoraConfig as LoRA but uses an
# asymmetric learning rate: matrix B is trained at a higher rate than matrix A.
#
#   η_B  =  loraplus_lr_ratio × η_A
#
# The original paper reports the sweet spot at ratio 4–16 depending on task.
# We use 16 for Configs A/B/C and 8 for Config D (high-rank adapters are less
# sensitive to the ratio and a large ratio can destabilise training at r=32).
#
# All other hyperparameters are identical to LORA_CONFIGS so results are
# directly comparable.
LORAPLUS_CONFIGS: dict[str, dict] = {
    "A": {
        **LORA_CONFIGS["A"],
        "description": "LoRA+ Light — attention Q/V only, rank 8",
        "loraplus_lr_ratio": 16,
    },
    "B": {
        **LORA_CONFIGS["B"],
        "description": "LoRA+ Standard — full attention, rank 16",
        "loraplus_lr_ratio": 16,
    },
    "C": {
        **LORA_CONFIGS["C"],
        "description": "LoRA+ Wide — full attention + MLP, rank 16",
        "loraplus_lr_ratio": 16,
    },
    "D": {
        **LORA_CONFIGS["D"],
        "description": "LoRA+ Heavy — full attention + MLP, rank 32",
        "loraplus_lr_ratio": 8,
    },
}

ALL_LORAPLUS_CONFIGS: list[str] = list(LORAPLUS_CONFIGS.keys())
