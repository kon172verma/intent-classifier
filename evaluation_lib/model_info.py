"""
evaluation_lib/model_info.py
============================
Static model metadata: parameter counts, context windows, architectures,
licences, and display labels.  Used for comparison tables and plot annotations.

Contents
--------
  MODEL_PARAMS_B          – approximate parameter count (billions)
  MODEL_CTX_LENGTH        – maximum context window (tokens)
  MODEL_ARCHITECTURE      – model family / architecture type
  MODEL_LICENSE           – licence identifier
  MODEL_IS_BASE           – True if this is a base (pretrained-only) checkpoint
  MODEL_DISPLAY_LABELS    – human-readable plot labels
  SIZE_CATEGORY_COLORS    – colour palette keyed by size category
  model_size_category()   – assign "tiny" / "small" / "mid"
  display_label()         – get display label with fallback
  print_comparison_table()– pretty-print full model metadata table
"""

from __future__ import annotations

# ── Parameter counts (approximate, in billions) ────────────────────────────────
MODEL_PARAMS_B: dict[str, float] = {
    "smollm2-135m": 0.135,
    "gemma3-270m":  0.270,
    "smollm2-360m": 0.360,
    "qwen2.5-0.5b": 0.500,
    "qwen3-0.6b":   0.600,
    "gemma3-1b":    1.000,
    "tinyllama":    1.100,
    "qwen2.5-1.5b": 1.540,
    "qwen3-1.7b":   1.700,
    "smollm3":      3.000,
    "llama3.2-3b":  3.210,
    "qwen3-4b":     4.000,   # MoE: 4B total, ~0.9B active per token
}

# ── Context window (max tokens) ────────────────────────────────────────────────
MODEL_CTX_LENGTH: dict[str, int] = {
    "smollm2-135m":   2_048,
    "smollm2-360m":   2_048,
    "smollm3":        8_192,
    "qwen2.5-0.5b":  32_768,
    "qwen2.5-1.5b": 131_072,
    "qwen3-0.6b":    32_768,
    "qwen3-1.7b":    32_768,
    "qwen3-4b":      32_768,
    "tinyllama":      2_048,
    "gemma3-270m":   32_768,
    "gemma3-1b":     32_768,
    "llama3.2-3b":  131_072,
}

# ── Architecture family ────────────────────────────────────────────────────────
MODEL_ARCHITECTURE: dict[str, str] = {
    "smollm2-135m": "Llama-2 (dense)",
    "smollm2-360m": "Llama-2 (dense)",
    "smollm3":      "Llama-3 (dense)",
    "qwen2.5-0.5b": "Qwen2.5 (dense)",
    "qwen2.5-1.5b": "Qwen2.5 (dense)",
    "qwen3-0.6b":   "Qwen3 (dense)",
    "qwen3-1.7b":   "Qwen3 (dense)",
    "qwen3-4b":     "Qwen3 (MoE)",       # 4B total / ~0.9B active
    "tinyllama":    "Llama-2 (dense)",
    "gemma3-270m":  "Gemma-3 (dense)",
    "gemma3-1b":    "Gemma-3 (dense)",
    "llama3.2-3b":  "Llama-3 (dense)",
}

# ── Licence ────────────────────────────────────────────────────────────────────
MODEL_LICENSE: dict[str, str] = {
    "smollm2-135m": "Apache-2.0",
    "smollm2-360m": "Apache-2.0",
    "smollm3":      "Apache-2.0",
    "qwen2.5-0.5b": "Apache-2.0",
    "qwen2.5-1.5b": "Apache-2.0",
    "qwen3-0.6b":   "Apache-2.0",
    "qwen3-1.7b":   "Apache-2.0",
    "qwen3-4b":     "Apache-2.0",
    "tinyllama":    "Apache-2.0",
    "gemma3-270m":  "Gemma (custom)",
    "gemma3-1b":    "Gemma (custom)",
    "llama3.2-3b":  "Llama-3 Community",
}

# ── Base vs chat checkpoint ────────────────────────────────────────────────────
# True  = raw pretrained weights, no RLHF / instruction tuning
# False = instruction-tuned or unified base+chat (Qwen3, SmolLM3)
MODEL_IS_BASE: dict[str, bool] = {
    "smollm2-135m": True,
    "smollm2-360m": True,
    "smollm3":      False,   # released as unified chat model; no separate base
    "qwen2.5-0.5b": True,
    "qwen2.5-1.5b": True,
    "qwen3-0.6b":   False,   # unified base+chat; no separate base released
    "qwen3-1.7b":   False,
    "qwen3-4b":     False,
    "tinyllama":    True,
    "gemma3-270m":  True,
    "gemma3-1b":    True,
    "llama3.2-3b":  True,
}

# ── Display labels for plot axes ───────────────────────────────────────────────
# Only entries that differ from the key itself are needed.
MODEL_DISPLAY_LABELS: dict[str, str] = {
    "tinyllama":   "tinyllama (1.1B)",
    "smollm3":     "smollm3 (3B)",
    "llama3.2-3b": "llama3.2-3b (3.2B)",
    "gemma3-270m": "gemma3-270m (0.27B)",
    "gemma3-1b":   "gemma3-1b (1B)",
}

# ── Size-category colours (used for plot colour-coding) ───────────────────────
# tiny  : < 0.6 B params
# small : 0.6 B – 1.2 B
# mid   : >= 1.2 B
SIZE_CATEGORY_COLORS: dict[str, tuple[str, str]] = {
    "tiny":  ("< 0.6 B",       "#4C72B0"),
    "small": ("0.6 B – 1.2 B", "#55A868"),
    "mid":   (">= 1.2 B",      "#C44E52"),
}


def model_size_category(model_key: str) -> str:
    """Return 'tiny', 'small', or 'mid' based on parameter count."""
    params = MODEL_PARAMS_B.get(model_key, 1.0)
    if params < 0.6:
        return "tiny"
    if params < 1.2:
        return "small"
    return "mid"


def display_label(model_key: str) -> str:
    """Return the plot-friendly display label for a model key."""
    return MODEL_DISPLAY_LABELS.get(model_key, model_key)


# ── Comparison table ───────────────────────────────────────────────────────────

def print_comparison_table(model_keys: list[str] | None = None) -> None:
    """
    Print a formatted comparison table for all (or selected) models.

    Columns: key | HF model ID | Params (B) | Ctx (k tok) | Base? | Architecture | License
    """
    # Lazy import to avoid circular dependency at module load time
    from evaluation_lib.config import MODEL_REGISTRY  # noqa: PLC0415

    keys = model_keys or sorted(MODEL_PARAMS_B, key=lambda k: MODEL_PARAMS_B.get(k, 0.0))

    col_key  = max(len(k) for k in keys) + 2
    col_id   = max(len(MODEL_REGISTRY.get(k, "")) for k in keys) + 2
    col_arch = max(len(MODEL_ARCHITECTURE.get(k, "")) for k in keys) + 2

    header = (
        f"{'Model key':<{col_key}}  {'HF model ID':<{col_id}}"
        f"  {'Params(B)':>9}  {'Ctx(k)':>6}  {'Base?':>5}"
        f"  {'Architecture':<{col_arch}}  License"
    )
    sep = "─" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)
    for k in keys:
        hf_id   = MODEL_REGISTRY.get(k, "")
        params  = MODEL_PARAMS_B.get(k, 0.0)
        ctx_k   = MODEL_CTX_LENGTH.get(k, 0) // 1_000
        is_base = "yes" if MODEL_IS_BASE.get(k, False) else "no"
        arch    = MODEL_ARCHITECTURE.get(k, "")
        lic     = MODEL_LICENSE.get(k, "")
        print(
            f"{k:<{col_key}}  {hf_id:<{col_id}}"
            f"  {params:>9.3f}  {ctx_k:>6}  {is_base:>5}"
            f"  {arch:<{col_arch}}  {lic}"
        )
    print(sep)
    print()
