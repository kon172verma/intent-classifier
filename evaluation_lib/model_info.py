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
    # TINY (<300M)
    "pythia-70m":    0.070,
    "cerebras-111m": 0.111,
    "smollm2-135m":  0.135,
    "gemma3-270m":   0.270,
    # SMALL (<1B)
    "smollm2-360m":  0.360,
    "qwen2.5-0.5b":  0.494,
    "qwen3-0.6b":    0.600,
    # MEDIUM (<2B)
    "gemma3-1b":     1.000,
    "llama3.2-1b":   1.235,
    "qwen3-1.7b":    1.700,
    "smollm2-1.7b":  1.710,
    # LARGE (<=3B)
    "granite3.3-2b": 2.000,
    "gemma2-2b":     2.610,
    "smollm3":       3.000,
    "llama3.2-3b":   3.210,
}

# ── Context window (max tokens) ────────────────────────────────────────────────
MODEL_CTX_LENGTH: dict[str, int] = {
    # TINY
    "pythia-70m":     2_048,
    "cerebras-111m":  2_048,
    "smollm2-135m":   2_048,
    "gemma3-270m":   32_768,
    # SMALL
    "smollm2-360m":   2_048,
    "qwen2.5-0.5b":  32_768,
    "qwen3-0.6b":    32_768,
    # MEDIUM
    "gemma3-1b":     32_768,
    "llama3.2-1b":  131_072,
    "qwen3-1.7b":    32_768,
    "smollm2-1.7b":   8_192,
    # LARGE
    "granite3.3-2b": 131_072,
    "gemma2-2b":      8_192,
    "smollm3":       65_536,
    "llama3.2-3b":  131_072,
}

# ── Architecture family ────────────────────────────────────────────────────────
MODEL_ARCHITECTURE: dict[str, str] = {
    # TINY
    "pythia-70m":    "GPT-NeoX (dense)",
    "cerebras-111m": "GPT-2 (dense)",
    "smollm2-135m":  "Llama-2 (dense)",
    "gemma3-270m":   "Gemma-3 (dense)",
    # SMALL
    "smollm2-360m":  "Llama-2 (dense)",
    "qwen2.5-0.5b":  "Qwen2.5 (dense)",
    "qwen3-0.6b":    "Qwen3 (dense)",
    # MEDIUM
    "gemma3-1b":     "Gemma-3 (dense)",
    "llama3.2-1b":   "Llama-3 (dense)",
    "qwen3-1.7b":    "Qwen3 (dense)",
    "smollm2-1.7b":  "Llama-2 (dense)",
    # LARGE
    "granite3.3-2b": "Granite-3 (dense)",
    "gemma2-2b":     "Gemma-2 (dense)",
    "smollm3":       "Llama-3 (dense)",
    "llama3.2-3b":   "Llama-3 (dense)",
}

# ── Licence ────────────────────────────────────────────────────────────────────
MODEL_LICENSE: dict[str, str] = {
    # TINY
    "pythia-70m":    "Apache-2.0",
    "cerebras-111m": "Apache-2.0",
    "smollm2-135m":  "Apache-2.0",
    "gemma3-270m":   "Gemma (custom)",
    # SMALL
    "smollm2-360m":  "Apache-2.0",
    "qwen2.5-0.5b":  "Apache-2.0",
    "qwen3-0.6b":    "Apache-2.0",
    # MEDIUM
    "gemma3-1b":     "Gemma (custom)",
    "llama3.2-1b":   "Llama-3.2 Community",
    "qwen3-1.7b":    "Apache-2.0",
    "smollm2-1.7b":  "Apache-2.0",
    # LARGE
    "granite3.3-2b": "Apache-2.0",
    "gemma2-2b":     "Gemma (custom)",
    "smollm3":       "Apache-2.0",
    "llama3.2-3b":   "Llama-3.2 Community",
}

# ── Base vs chat checkpoint ────────────────────────────────────────────────────
# True  = raw pretrained weights, no RLHF / instruction tuning
# False = instruction-tuned or unified base+chat (Qwen3, SmolLM3, Granite)
MODEL_IS_BASE: dict[str, bool] = {
    # TINY
    "pythia-70m":    True,    # base only; no instruct variant
    "cerebras-111m": True,    # base only; no instruct variant
    "smollm2-135m":  False,   # instruct (SFT + DPO)
    "gemma3-270m":   False,   # instruct (-it suffix)
    # SMALL
    "smollm2-360m":  False,   # instruct
    "qwen2.5-0.5b":  False,   # instruct
    "qwen3-0.6b":    False,   # unified base+chat; no separate base released
    # MEDIUM
    "gemma3-1b":     False,   # instruct (-it suffix)
    "llama3.2-1b":   False,   # instruct
    "qwen3-1.7b":    False,   # unified base+chat
    "smollm2-1.7b":  False,   # instruct
    # LARGE
    "granite3.3-2b": False,   # instruct (thinking via thinking= kwarg)
    "gemma2-2b":     False,   # instruct (-it suffix)
    "smollm3":       False,   # unified base+chat; no separate base released
    "llama3.2-3b":   False,   # instruct
}

# ── Display labels for plot axes ───────────────────────────────────────────────
# Only entries that differ from the key itself are needed.
MODEL_DISPLAY_LABELS: dict[str, str] = {
    "pythia-70m":    "pythia-70m (0.07B)",
    "cerebras-111m": "cerebras-111m (0.11B)",
    "smollm2-135m":  "smollm2-135m (0.14B)",
    "gemma3-270m":   "gemma3-270m (0.27B)",
    "smollm2-360m":  "smollm2-360m (0.36B)",
    "qwen2.5-0.5b":  "qwen2.5-0.5b (0.49B)",
    "qwen3-0.6b":    "qwen3-0.6b (0.6B)",
    "gemma3-1b":     "gemma3-1b (1B)",
    "llama3.2-1b":   "llama3.2-1b (1.2B)",
    "qwen3-1.7b":    "qwen3-1.7b (1.7B)",
    "smollm2-1.7b":  "smollm2-1.7b (1.7B)",
    "granite3.3-2b": "granite3.3-2b (2B)",
    "gemma2-2b":     "gemma2-2b (2.6B)",
    "smollm3":       "smollm3 (3B)",
    "llama3.2-3b":   "llama3.2-3b (3.2B)",
}

# ── Size-category colours (used for plot colour-coding) ───────────────────────
# tiny   : <300M params
# small  : <1B params
# medium : <2B params
# large  : <=3B params
SIZE_CATEGORY_COLORS: dict[str, tuple[str, str]] = {
    "tiny":   ("< 300 M",  "#4C72B0"),
    "small":  ("< 1 B",    "#55A868"),
    "medium": ("< 2 B",    "#E07B39"),
    "large":  ("<= 3 B",   "#C44E52"),
}


def model_size_category(model_key: str) -> str:
    """Return 'tiny', 'small', 'medium', or 'large' based on parameter count."""
    params = MODEL_PARAMS_B.get(model_key, 1.0)
    if params < 0.3:
        return "tiny"
    if params < 1.0:
        return "small"
    if params < 2.0:
        return "medium"
    return "large"


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
