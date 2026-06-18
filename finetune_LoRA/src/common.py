#!/usr/bin/env python3
"""
Shared utilities for finetune_LoRA.

Contains the model registry, LoRA config definitions, prompt helpers,
prediction extraction, and metric computation.

No torch / transformers imports at module level — safe to import anywhere.
"""

import re

# ── Model registry ────────────────────────────────────────────────────────────
# Only the two models selected for Phase 3 LoRA fine-tuning.
MODEL_REGISTRY: dict[str, str] = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen3-0.6b":   "Qwen/Qwen3-0.6B",
}

ALL_MODELS: list[str] = list(MODEL_REGISTRY.keys())

# Qwen3 tokenizers support enable_thinking= in apply_chat_template.
# We always disable thinking for deterministic single-token routing output.
QWEN3_KEYS: frozenset[str] = frozenset({"qwen3-0.6b"})

# ── LoRA configs ──────────────────────────────────────────────────────────────
# Three configs covering light / standard / wide adapter capacity.
# target_modules are valid for both qwen2 (Qwen2.5) and qwen3 architectures.
#
# Config A — Light: Q+V only, rank 8.  Fast diagnostic run; fewest params.
# Config B — Standard: full attention, rank 16.  Main recommended run.
# Config C — Wide: full attention + MLP, rank 32.  High-capacity; may overfit
#             on 1k data — useful for measuring dataset-size sensitivity.
#
# lora_alpha = 2 × rank  (standard scaling convention).
# Effective batch size = per_device_train_batch_size × gradient_accumulation_steps = 16 for all configs.
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
        "description": "Wide — full attention + MLP, rank 32",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
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

# Worst-case prompt (30 tools × ~30 tok + system + request) ≈ 975 tokens.
MAX_SEQ_LENGTH: int = 1024

# ── System prompt ─────────────────────────────────────────────────────────────
# Kept identical to baseline_evaluation so comparisons are fair.
SYSTEM_PROMPT: str = (
    "You are a tool router.\n\n"
    "Rules:\n"
    "- Return only the tool name.\n"
    '- Return "none" if no tool matches.\n'
    "- Do not explain."
)

# ── Prompt helpers ────────────────────────────────────────────────────────────

def _tool_block(tools: list[dict]) -> str:
    parts: list[str] = []
    for t in tools:
        parts.append(f"Name: {t['name']}")
        parts.append(f"Description: {t['description']}")
        parts.append("")
    return "\n".join(parts)


def build_user_content(example: dict) -> str:
    return (
        f"Available Tools:\n{_tool_block(example['available_tools'])}\n"
        f"User Request:\n{example['user_request']}\n\n"
        "Selected Tool:"
    )


def build_chat_messages(
    example: dict,
    include_answer: bool = False,
) -> list[dict]:
    """
    Build system + user turn (optionally with the ground-truth assistant turn).

    include_answer=True  → used when formatting training data (full conversation).
    include_answer=False → used at inference time (generation prompt follows).
    """
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_user_content(example)},
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": example["answer"]})
    return messages


def apply_chat_template_safe(
    tokenizer,
    messages: list[dict],
    model_key: str,
    add_generation_prompt: bool,
) -> str:
    """
    Apply tokenizer chat template with Qwen3 thinking mode disabled.
    Falls back gracefully if enable_thinking= is not supported by the tokenizer.
    """
    kwargs: dict = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if model_key in QWEN3_KEYS:
        try:
            return tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            pass  # older tokenizer build without enable_thinking
    return tokenizer.apply_chat_template(messages, **kwargs)


# ── Output extraction ─────────────────────────────────────────────────────────
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_prediction(raw_generated: str) -> str:
    """
    Clean model output and extract the tool name.

    1. Strip Qwen3 <think>…</think> blocks.
    2. Take the first non-empty line.
    3. Remove markdown artifacts (* _ ` " ').
    4. Strip any echoed 'Selected Tool:' prefix.
    5. Take only the first whitespace-delimited token.
    """
    text = _THINK_RE.sub("", raw_generated).strip()
    for line in text.splitlines():
        line = line.strip().strip("*_`\"'")
        if line.lower().startswith("selected tool:"):
            line = line[len("selected tool:"):].strip().strip("*_`\"'")
        if line:
            token = line.split()[0].rstrip(".,;:()")
            return token
    return ""


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_accuracy(predictions: list[str], labels: list[str]) -> float:
    if not predictions:
        return 0.0
    return sum(p == l for p, l in zip(predictions, labels)) / len(predictions)


def compute_per_tool_metrics(
    predictions: list[str],
    labels: list[str],
) -> dict[str, dict]:
    """Return per-tool precision, recall, F1, and support counts."""
    tool_set = sorted(set(labels) | set(predictions))
    metrics: dict[str, dict] = {}
    for tool in tool_set:
        tp = sum(p == tool and l == tool for p, l in zip(predictions, labels))
        fp = sum(p == tool and l != tool for p, l in zip(predictions, labels))
        fn = sum(p != tool and l == tool for p, l in zip(predictions, labels))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        metrics[tool] = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "support":   sum(l == tool for l in labels),
        }
    return metrics
