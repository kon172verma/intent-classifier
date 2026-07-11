"""
evaluation_lib
==============
Shared library for evaluation_baseline and evaluation_quantized.

Re-exports commonly used symbols so callers can do:
    from evaluation_lib import MODEL_REGISTRY, evaluate, ExampleResult
"""

from .config import (
    MODEL_REGISTRY,
    ALL_MODELS,
    QWEN3_KEYS,
    SYSTEM_PROMPT_ZERO_SHOT,
    SYSTEM_PROMPT_FEW_SHOT,
    SYSTEM_PROMPT,
)
from .eval_core import (
    ExampleResult,
    BenchmarkReport,
    build_chat_messages,
    build_raw_prompt,
    extract_prediction,
    run_example,
    evaluate,
    compute_prefix_kv_cache,
)
from .model_utils import (
    resolve_device,
    dtype_for_device,
    load_model_and_tokenizer,
    reset_peak_memory,
    peak_memory_mb,
    free_model_memory,
)
from .model_info import (
    MODEL_PARAMS_B,
    MODEL_CTX_LENGTH,
    MODEL_ARCHITECTURE,
    MODEL_LICENSE,
    MODEL_IS_BASE,
    MODEL_DISPLAY_LABELS,
    model_size_category,
    SIZE_CATEGORY_COLORS,
    display_label,
    print_comparison_table,
)

__all__ = [
    # config
    "MODEL_REGISTRY", "ALL_MODELS", "QWEN3_KEYS",
    "SYSTEM_PROMPT_ZERO_SHOT", "SYSTEM_PROMPT_FEW_SHOT", "SYSTEM_PROMPT",
    # eval_core
    "ExampleResult", "BenchmarkReport",
    "build_chat_messages", "build_raw_prompt", "extract_prediction",
    "run_example", "evaluate", "compute_prefix_kv_cache",
    # model_utils
    "resolve_device", "dtype_for_device", "load_model_and_tokenizer",
    "reset_peak_memory", "peak_memory_mb", "free_model_memory",
    # model_info
    "MODEL_PARAMS_B", "MODEL_CTX_LENGTH", "MODEL_ARCHITECTURE",
    "MODEL_LICENSE", "MODEL_IS_BASE", "MODEL_DISPLAY_LABELS",
    "model_size_category", "SIZE_CATEGORY_COLORS", "display_label",
    "print_comparison_table",
]
