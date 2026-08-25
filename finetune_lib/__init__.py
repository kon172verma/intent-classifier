"""
finetune_lib
============
Shared utilities for fine-tuning experiments (LoRA, DoRA, LoRA+, DoRA+, QLoRA, AdaLoRA).

Quick imports:
    from finetune_lib import FINETUNE_MODEL_REGISTRY, LORA_CONFIGS, HF_RELEASE_REPO
    from finetune_lib import ADALORA_CONFIGS, ALL_ADALORA_CONFIGS, ADALORA_MODEL_REGISTRY
    from finetune_lib import LORAPLUS_CONFIGS, ALL_LORAPLUS_CONFIGS, ALL_LORAPLUS_MODELS
    from finetune_lib import QLORA_MODEL_REGISTRY, ALL_QLORA_MODELS
    from finetune_lib import EARLY_STOPPING_PATIENCE, EARLY_STOPPING_THRESHOLD
    from finetune_lib import build_chat_messages, tokenize_with_labels
    from finetune_lib import TrainValAccuracyCallback, compute_initial_train_loss
    from finetune_lib import resolve_device, peak_memory_mb, extract_prediction
    from finetune_lib import CURRENT_VERSION, HF_EXPERIMENTS_REPO, HF_RELEASE_REPO
    from finetune_lib import generate_experiment_timestamp
    from finetune_lib import hf_adapter_subfolder, hf_merged_subfolder
    from finetune_lib.registry import log_experiment, find_latest_experiment
"""

from evaluation_lib.eval_core import (  # noqa: F401
    NO_TOOL_ID,
    PROMPT_FORMAT_VERSION,
    TOOL_IDS,
    answer_to_tool_id,
    tool_id_to_answer,
)

from .config import (  # noqa: F401
    ADALORA_CONFIGS,
    ADALORA_MODEL_REGISTRY,
    ALL_ADALORA_CONFIGS,
    ALL_ADALORA_MODELS,
    ALL_CONFIGS,
    ALL_FINETUNE_MODELS,
    ALL_LORAPLUS_CONFIGS,
    ALL_LORAPLUS_MODELS,
    ALL_QLORA_MODELS,
    CURRENT_VERSION,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_THRESHOLD,
    FINETUNE_MODEL_REGISTRY,
    HF_EXPERIMENTS_REPO,
    HF_RELEASE_REPO,
    LORA_CONFIGS,
    LORA_GRADIENT_CHECKPOINTING_SKIP_KEYS,
    LORAPLUS_CONFIGS,
    MAX_SEQ_LENGTH,
    QLORA_MODEL_REGISTRY,
    QWEN3_FINETUNE_KEYS,
    SYSTEM_PROMPT,
    generate_experiment_timestamp,
    hf_adapter_subfolder,
    hf_merged_subfolder,
)
from .lib import (  # noqa: F401
    TrainValAccuracyCallback,
    apply_chat_template_safe,
    build_chat_messages,
    build_training_arguments,
    compute_accuracy,
    compute_initial_train_loss,
    compute_per_tool_metrics,
    # re-exported from evaluation_lib
    extract_prediction,
    hf_report_path,
    load_jsonl,
    peak_memory_mb,
    resolve_device,
    tokenize_with_labels,
    upload_report_to_hf,
)
