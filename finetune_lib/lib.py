"""
finetune_lib/lib.py
====================
Shared runtime utilities for all fine-tuning experiments.

Provides:
  build_chat_messages()        – system + user turn (optionally with answer)
  apply_chat_template_safe()   – chat template with Qwen3 thinking disabled
  tokenize_with_labels()       – prompt-masked tokenisation (model-agnostic)
  load_jsonl()                 – read a JSONL file
  TrainValAccuracyCallback     – HF Trainer callback measuring train + val accuracy
  compute_initial_train_loss() – forward-pass loss before any gradient update

Re-exports from evaluation_lib (no duplication):
  extract_prediction, compute_accuracy, compute_per_tool_metrics,
  resolve_device, peak_memory_mb
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import (
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

# ── Re-export shared utilities from evaluation_lib ────────────────────────────
from evaluation_lib.eval_core import extract_prediction  # noqa: F401
from evaluation_lib.model_utils import (  # noqa: F401
    peak_memory_mb,
    resolve_device,
)

from .config import QWEN3_FINETUNE_KEYS, SYSTEM_PROMPT, MAX_SEQ_LENGTH


# ── Helpers imported from evaluation_lib ──────────────────────────────────────


def _compute_per_tool_metrics(
    predictions: list[str], labels: list[str]
) -> dict[str, dict]:
    """Per-tool precision / recall / F1 / support."""
    tool_set = sorted(set(labels) | set(predictions))
    metrics: dict[str, dict] = {}
    for tool in tool_set:
        tp = sum(p == tool and lb == tool for p, lb in zip(predictions, labels))
        fp = sum(p == tool and lb != tool for p, lb in zip(predictions, labels))
        fn = sum(p != tool and lb == tool for p, lb in zip(predictions, labels))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        metrics[tool] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(lb == tool for lb in labels),
        }
    return metrics


compute_per_tool_metrics = _compute_per_tool_metrics


def compute_accuracy(predictions: list[str], labels: list[str]) -> float:
    """Fraction of predictions that exactly match the ground-truth label."""
    if not labels:
        return 0.0
    return sum(p == lb for p, lb in zip(predictions, labels)) / len(labels)


# ── Prompt building ───────────────────────────────────────────────────────────


def _tool_block(tools: list[dict]) -> str:
    parts: list[str] = []
    for t in tools:
        parts.append(f"Name: {t['name']}")
        parts.append(f"Description: {t['description']}")
        parts.append("")
    return "\n".join(parts)


def build_chat_messages(
    example: dict,
    include_answer: bool = False,
) -> list[dict]:
    """
    Build [system, user] messages for a single example.

    include_answer=False  → inference / evaluation (add_generation_prompt=True follows)
    include_answer=True   → training (full turn; add_generation_prompt=False)
    """
    user_content = (
        f"Available Tools:\n{_tool_block(example['available_tools'])}\n"
        f"User Request:\n{example['user_request']}\n\n"
        "Selected Tool:"
    )
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": example["answer"]})
    return messages


def apply_chat_template_safe(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict],
    model_key: str,
    add_generation_prompt: bool,
) -> str:
    """Apply chat template with Qwen3 thinking mode disabled."""
    kwargs: dict = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if model_key in QWEN3_FINETUNE_KEYS:
        try:
            result = tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
            return str(result)
        except TypeError:
            pass  # older tokenizer without enable_thinking
    return str(tokenizer.apply_chat_template(messages, **kwargs))


# ── Tokenisation with label masking ──────────────────────────────────────────


def tokenize_with_labels(
    example: dict,
    tokenizer: PreTrainedTokenizerBase,
    model_key: str,
) -> dict:
    """
    Tokenise one training example and mask all non-answer tokens with -100.

    Uses a model-agnostic length-comparison strategy:
      1. Tokenise the prompt only (with generation prompt) → get prompt length N.
      2. Tokenise the full conversation (prompt + answer) → get full input_ids.
      3. Mask the first N positions in labels with -100.

    This approach works across Qwen2.5 (ChatML), Qwen3 (ChatML),
    SmolLM2 (ChatML), and Llama-3.2 (Llama-3 header format).
    """
    # Full conversation (will be the actual input_ids / labels)
    full_text = apply_chat_template_safe(
        tokenizer,
        build_chat_messages(example, include_answer=True),
        model_key,
        add_generation_prompt=False,
    )
    # Prompt only — used to determine where the answer starts
    prompt_text = apply_chat_template_safe(
        tokenizer,
        build_chat_messages(example, include_answer=False),
        model_key,
        add_generation_prompt=True,
    )

    full_enc = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors=None,
    )
    prompt_enc = tokenizer(
        prompt_text,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors=None,
    )

    input_ids: list[int] = [int(x) for x in full_enc["input_ids"]]
    attention_mask: list[int] = [int(x) for x in full_enc["attention_mask"]]
    prompt_len: int = min(len(prompt_enc["input_ids"]), len(input_ids))

    labels: list[int] = [-100] * prompt_len + input_ids[prompt_len:]
    # Trim to input_ids length in case of rounding differences
    labels = labels[: len(input_ids)]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ── Data loading ──────────────────────────────────────────────────────────────


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ── Inference helper (used by accuracy callback) ──────────────────────────────


def _generate_one(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    example: dict,
    device: torch.device,
    model_key: str,
) -> str:
    """Greedy generation on a single example; returns the predicted tool name."""
    messages = build_chat_messages(example, include_answer=False)
    text = apply_chat_template_safe(
        tokenizer, messages, model_key, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    eos_id = tokenizer.eos_token_id
    pad_id = int(eos_id) if isinstance(eos_id, int) else 0
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=pad_id,
        )
    new_ids = out[0][inputs["input_ids"].shape[1] :]
    decoded = tokenizer.decode(new_ids, skip_special_tokens=True)
    return extract_prediction(decoded if isinstance(decoded, str) else decoded[0])


def _compute_sample_accuracy(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    examples: list[dict],
    device: torch.device,
    model_key: str,
) -> float:
    """Greedy accuracy on a list of examples (no gradient)."""
    if not examples:
        return 0.0
    correct = 0
    for ex in examples:
        pred = _generate_one(model, tokenizer, ex, device, model_key)
        if pred == ex["answer"]:
            correct += 1
    return correct / len(examples)


# ── Accuracy callback (train + val) ──────────────────────────────────────────


class TrainValAccuracyCallback(TrainerCallback):
    """
    Measures both train and val accuracy at every evaluation checkpoint.
    Results are attached to the most recent eval-loss log entry so
    plot_lora_results.py can draw training curves with all four signals:
        train_loss, eval_loss, train_accuracy, eval_accuracy.

    Max sample sizes keep the callback fast:
      - train: ≤50 examples  (~2.5 s at ~50 ms/example)
      - val:   ≤100 examples (~5.0 s)
    """

    def __init__(
        self,
        train_examples: list[dict],
        val_examples: list[dict],
        tokenizer: PreTrainedTokenizerBase,
        model_key: str,
        device: torch.device,
        max_train: int = 50,
        max_val: int = 100,
    ) -> None:
        self.train_sample = train_examples[:max_train]
        self.val_sample = val_examples[:max_val]
        self.tokenizer = tokenizer
        self.model_key = model_key
        self.device = device

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        if model is None:
            return

        was_training = model.training
        model.eval()

        train_acc = _compute_sample_accuracy(
            model, self.tokenizer, self.train_sample, self.device, self.model_key
        )
        val_acc = _compute_sample_accuracy(
            model, self.tokenizer, self.val_sample, self.device, self.model_key
        )

        # Attach accuracies to the most recently added eval-loss log entry.
        for entry in reversed(state.log_history):
            if "eval_loss" in entry:
                entry["train_accuracy"] = round(train_acc, 4)
                entry["eval_accuracy"] = round(val_acc, 4)
                break

        print(
            f"\n  [Accuracy] step={state.global_step}"
            f"  train={train_acc:.4f}  val={val_acc:.4f}"
        )

        if was_training:
            model.train()


# ── Initial train-loss helper ─────────────────────────────────────────────────


def compute_initial_train_loss(
    model: Any,
    dataset: Any,
    collator: Any,
    device: torch.device,
    batch_size: int = 8,
) -> float:
    """
    Compute CE loss on a small training batch before any gradient updates.
    The model must already have LoRA attached (freshly initialised weights are
    effectively identity, so this closely approximates the base-model loss).
    """
    sample_size = min(batch_size, len(dataset))
    sample = [dataset[i] for i in range(sample_size)]
    batch = collator(sample)

    model.eval()
    with torch.no_grad():
        loss = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        ).loss
    model.train()
    return float(loss)
