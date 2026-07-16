"""
finetune_lib
============
Shared utilities for fine-tuning experiments (LoRA, QLoRA, AdaLoRA, LoRA+).

Quick imports:
    from finetune_lib import FINETUNE_MODEL_REGISTRY, LORA_CONFIGS, HF_HUB_REPO
    from finetune_lib import ADALORA_CONFIGS, ALL_ADALORA_CONFIGS
    from finetune_lib import LORAPLUS_CONFIGS, ALL_LORAPLUS_CONFIGS, ALL_LORAPLUS_MODELS
    from finetune_lib import build_chat_messages, tokenize_with_labels
    from finetune_lib import TrainValAccuracyCallback, compute_initial_train_loss
    from finetune_lib import resolve_device, peak_memory_mb, extract_prediction
"""

from .config import (  # noqa: F401
    FINETUNE_MODEL_REGISTRY,
    ALL_FINETUNE_MODELS,
    QWEN3_FINETUNE_KEYS,
    SYSTEM_PROMPT,
    LORA_CONFIGS,
    ALL_CONFIGS,
    ADALORA_CONFIGS,
    ALL_ADALORA_CONFIGS,
    LORAPLUS_CONFIGS,
    ALL_LORAPLUS_CONFIGS,
    ALL_LORAPLUS_MODELS,
    MAX_SEQ_LENGTH,
    HF_HUB_REPO,
    hf_adapter_subfolder,
)
from .lib import (  # noqa: F401
    build_chat_messages,
    apply_chat_template_safe,
    tokenize_with_labels,
    load_jsonl,
    TrainValAccuracyCallback,
    compute_initial_train_loss,
    compute_per_tool_metrics,
    # re-exported from evaluation_lib
    extract_prediction,
    compute_accuracy,
    resolve_device,
    peak_memory_mb,
)
