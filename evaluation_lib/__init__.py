"""
evaluation_lib
==============
Shared library for evaluation_baseline and evaluation_quantized.

Re-exports commonly used symbols so callers can do:
    from evaluation_lib import MODEL_REGISTRY, evaluate, ExampleResult
"""

from .config import (
    ALL_MODELS,
    MODEL_REGISTRY,
    QWEN3_KEYS,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_FEW_SHOT,
    SYSTEM_PROMPT_ZERO_SHOT,
)
from .eval_core import (
    NO_TOOL_ID,
    PROMPT_FORMAT_VERSION,
    TOOL_IDS,
    BenchmarkReport,
    ExampleResult,
    answer_to_tool_id,
    build_chat_messages,
    build_raw_prompt,
    compute_prefix_kv_cache,
    evaluate,
    extract_prediction,
    run_example,
    tool_id_to_answer,
)
from .model_info import (
    MODEL_ARCHITECTURE,
    MODEL_CTX_LENGTH,
    MODEL_DISPLAY_LABELS,
    MODEL_IS_BASE,
    MODEL_LICENSE,
    MODEL_PARAMS_B,
    SIZE_CATEGORY_COLORS,
    display_label,
    model_size_category,
    print_comparison_table,
)
from .model_utils import (
    dtype_for_device,
    free_model_memory,
    load_model_and_tokenizer,
    peak_memory_mb,
    reset_peak_memory,
    resolve_device,
)

__all__ = [
    # config
    "MODEL_REGISTRY", "ALL_MODELS", "QWEN3_KEYS",
    "SYSTEM_PROMPT_ZERO_SHOT", "SYSTEM_PROMPT_FEW_SHOT", "SYSTEM_PROMPT",
    # eval_core
    "ExampleResult", "BenchmarkReport", "TOOL_IDS", "NO_TOOL_ID",
    "PROMPT_FORMAT_VERSION",
    "build_chat_messages", "build_raw_prompt", "extract_prediction",
    "answer_to_tool_id", "tool_id_to_answer",
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
