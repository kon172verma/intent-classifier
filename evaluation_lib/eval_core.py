"""
evaluation_lib/eval_core.py
============================
Core evaluation logic shared by evaluation_baseline and evaluation_quantized:

  Prompt building  – build_chat_messages, build_raw_prompt
  Output parsing   – extract_prediction
  Inference        – run_example
  Evaluation loop  – evaluate
  Data structures  – ExampleResult, BenchmarkReport
"""

from __future__ import annotations

import json
import re
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from evaluation_lib.config import (
    QWEN3_KEYS,
    SYSTEM_PROMPT_ZERO_SHOT,
    SYSTEM_PROMPT_FEW_SHOT,
)
from evaluation_lib.model_utils import (
    _NO_KV_CACHE_KEYS,
    dtype_for_device,
    free_model_memory,
    load_model_and_tokenizer,
    peak_memory_mb,
    reset_peak_memory,
)


# ── Prompt builders ────────────────────────────────────────────────────────────
# Prompt order: system-prompt → available-tools → user-request.
# For chat models  : system role holds SYSTEM_PROMPT; user role holds
#                    tool list + user request (in that order).
# For base models  : same order flattened into a single completion string.

def _tool_block(tools: list[dict]) -> str:
    parts: list[str] = []
    for t in tools:
        parts.append(f"Name: {t['name']}")
        parts.append(f"Description: {t['description']}")
        parts.append("")
    return "\n".join(parts)


def build_raw_prompt(
    example: dict,
    system_prompt: str = SYSTEM_PROMPT_ZERO_SHOT,
) -> str:
    """Completion-style prompt for models without a chat template."""
    return (
        f"{system_prompt}\n\n"
        f"Available Tools:\n{_tool_block(example['available_tools'])}\n"
        f"User Request:\n{example['user_request']}\n\n"
        "Selected Tool:\n"
    )


def build_chat_messages(
    example: dict,
    system_prompt: str = SYSTEM_PROMPT_ZERO_SHOT,
) -> list[dict]:
    """
    System + user turn for models with a chat template.
    Within the user turn: available-tools appears before user-request.
    """
    user = (
        f"Available Tools:\n{_tool_block(example['available_tools'])}\n"
        f"User Request:\n{example['user_request']}\n\n"
        "Selected Tool:"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user},
    ]


# ── Prefix KV-cache helpers ────────────────────────────────────────────────────

def _build_prefix_text(
    tokenizer: PreTrainedTokenizerBase,
    model_key: str,
    system_prompt: str,
) -> str:
    """Return the fixed text that forms the cached system-prompt prefix.

    For chat models : formats just the system message via the chat template
                      (no user turn, no generation prompt).
    For base models : the system prompt followed by the two-newline separator
                      that precedes the tool list.
    """
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        messages = [{"role": "system", "content": system_prompt}]
        kwargs: dict = dict(tokenize=False, add_generation_prompt=False)
        if model_key in QWEN3_KEYS:
            kwargs["enable_thinking"] = False
        return str(tokenizer.apply_chat_template(messages, **kwargs))  # type: ignore[union-attr]
    else:
        return f"{system_prompt}\n\n"


def compute_prefix_kv_cache(
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    model_key: str,
    system_prompt: str,
    device: torch.device,
) -> tuple[Any, int]:
    """Pre-compute the KV cache for the fixed system-prompt prefix.

    Returns ``(past_key_values, prefix_token_length)``.

    The cache is stored as the legacy tuple-of-tuples format so that each
    ``model.generate()`` call constructs its own internal ``DynamicCache``
    from it rather than extending our stored tensors in-place.
    """
    prefix_text = _build_prefix_text(tokenizer, model_key, system_prompt)
    inputs = tokenizer(prefix_text, return_tensors="pt").to(device)  # type: ignore[operator]
    prefix_len = int(inputs["input_ids"].shape[-1])
    with torch.inference_mode():
        out = model(**inputs, use_cache=True)  # type: ignore[operator]
    kv: Any = out.past_key_values
    # Convert DynamicCache → immutable legacy tuple so every generate() call
    # builds a fresh internal cache rather than mutating our stored one.
    try:
        from transformers.cache_utils import Cache  # type: ignore[import-untyped]
        if isinstance(kv, Cache):
            to_legacy = getattr(kv, "to_legacy_cache", None)
            if to_legacy is not None:
                kv = to_legacy()
    except (ImportError, AttributeError):
        pass  # Already a tuple in older Transformers versions
    return kv, prefix_len


# ── Output extraction ──────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_prediction(raw_generated: str) -> str:
    """
    Extract the tool name from raw model output.

    Steps:
    1. Strip Qwen3 <think>…</think> blocks.
    2. Take the first non-empty line.
    3. Remove markdown artefacts (* _ ` " ').
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


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ExampleResult:
    index:            int
    user_request:     str
    n_tools:          int
    answer:           str
    prediction:       str
    correct:          bool
    is_garbage:       bool    # pred is non-empty, not "none", and not in the available tool list
    latency_s:        float
    tokens_generated: int


@dataclass
class BenchmarkReport:
    model_key:          str
    model_id:           str
    device:             str
    dtype:              str
    timestamp:          str
    eval_mode:          str     # "zero_shot" | "few_shot"
    n_examples:         int
    n_correct:          int
    accuracy:           float
    garbage_pct:        float   # % of wrong predictions that are hallucinated tool names
    avg_latency_ms:     float
    p50_latency_ms:     float
    p95_latency_ms:     float
    avg_tokens_per_sec: float
    peak_memory_mb:     float
    results:            list[dict] = field(default_factory=list)


# ── Single-example inference ───────────────────────────────────────────────────

def run_example(
    example: dict,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    model_key: str,
    max_new_tokens: int,
    use_cache: bool = True,
    system_prompt: str = SYSTEM_PROMPT_ZERO_SHOT,
    prefix_kv: Any | None = None,
    prefix_len: int = 0,
) -> tuple[str, float, int]:
    """
    Run one example through the model.
    Returns (prediction, latency_seconds, tokens_generated).

    When *prefix_kv* / *prefix_len* are provided (via ``compute_prefix_kv_cache``),
    only the variable part of the prompt (user turn + tools) is fed to the model;
    the cached system-prompt KV avoids re-prefilling the fixed prefix every call.
    """
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        messages = build_chat_messages(example, system_prompt=system_prompt)
        template_kwargs: dict = dict(tokenize=False, add_generation_prompt=True)
        if model_key in QWEN3_KEYS:
            template_kwargs["enable_thinking"] = False
        prompt_text: str = str(tokenizer.apply_chat_template(messages, **template_kwargs))  # type: ignore[union-attr, assignment]
    else:
        prompt_text = build_raw_prompt(example, system_prompt=system_prompt)

    # Tokenise the full prompt once (needed by both paths).
    full_ids: torch.Tensor = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)  # type: ignore[assignment, index]
    full_len = int(full_ids.shape[-1])

    t0 = time.perf_counter()
    if prefix_kv is not None and prefix_len > 0 and use_cache and full_len > prefix_len:
        # ── Prefix-KV path ────────────────────────────────────────────────────
        # Feed only the variable tokens; the prefix cache covers the system prompt.
        # Attention mask must span the entire sequence (prefix + variable).
        variable_ids = full_ids[:, prefix_len:]
        variable_len = int(variable_ids.shape[-1])
        attention_mask = torch.ones(1, full_len, dtype=torch.long, device=device)
        with torch.inference_mode():
            output_ids = model.generate(  # type: ignore[operator]
                input_ids=variable_ids,
                attention_mask=attention_mask,
                past_key_values=prefix_kv,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        new_ids = output_ids[0][variable_len:]
    else:
        # ── Standard path ─────────────────────────────────────────────────────
        attention_mask = torch.ones(1, full_len, dtype=torch.long, device=device)
        with torch.inference_mode():
            output_ids = model.generate(  # type: ignore[operator]
                input_ids=full_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=use_cache,
            )
        new_ids = output_ids[0][full_len:]

    latency = time.perf_counter() - t0
    tokens_generated = new_ids.shape[0]
    raw_output = str(tokenizer.decode(new_ids, skip_special_tokens=True))  # type: ignore[arg-type]
    return extract_prediction(raw_output), latency, tokens_generated


# ── Full evaluation loop ───────────────────────────────────────────────────────

def evaluate(
    model_key: str,
    model_id: str,
    data_path: Path,
    device: torch.device,
    max_new_tokens: int,
    limit: Optional[int] = None,
    system_prompt: str = SYSTEM_PROMPT_ZERO_SHOT,
    eval_mode: str = "zero_shot",
) -> BenchmarkReport:
    # ── Load dataset ─────────────────────────────────────────────────────────
    suffix = data_path.suffix.lower()
    if suffix == ".jsonl":
        examples = [
            json.loads(line)
            for line in data_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        examples = json.loads(data_path.read_text(encoding="utf-8"))

    if limit is not None:
        examples = examples[:limit]

    dtype = dtype_for_device(device)
    reset_peak_memory(device)
    model, tokenizer = load_model_and_tokenizer(model_id, device, dtype, model_key=model_key)
    reset_peak_memory(device)   # reset after load; measure inference memory only

    use_kv = model_key not in _NO_KV_CACHE_KEYS
    prefix_kv: Any | None = None
    prefix_len: int = 0
    if use_kv:
        try:
            prefix_kv, prefix_len = compute_prefix_kv_cache(
                tokenizer, model, model_key, system_prompt, device
            )
            print(f"  Prefix KV cache: {prefix_len} tokens cached\n")
        except Exception:
            traceback.print_exc()
            print("  Prefix KV cache unavailable — falling back to full prefill\n")

    print(f"\n  Running inference on {len(examples)} examples …\n")

    per_results: list[ExampleResult] = []
    for i, ex in enumerate(examples):
        try:
            pred, lat, ntok = run_example(
                ex, model, tokenizer, device, model_key, max_new_tokens,
                use_cache=use_kv,
                system_prompt=system_prompt,
                prefix_kv=prefix_kv,
                prefix_len=prefix_len,
            )
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            pred, lat, ntok = "", 0.0, 0

        correct = pred.strip().lower() == ex["answer"].strip().lower()
        valid_tools: set[str] = {t["name"].strip().lower() for t in ex["available_tools"]} | {"none"}
        pred_lower = pred.strip().lower()
        is_garbage = bool(pred_lower) and pred_lower not in valid_tools
        per_results.append(ExampleResult(
            index=i,
            user_request=ex["user_request"],
            n_tools=len(ex["available_tools"]),
            answer=ex["answer"],
            prediction=pred,
            correct=correct,
            is_garbage=is_garbage,
            latency_s=lat,
            tokens_generated=ntok,
        ))
        status = "✓" if correct else ("?" if is_garbage else "✗")
        print(
            f"  [{i+1:>3}/{len(examples)}] {status} "
            f"pred={pred!r:30s}  ans={ex['answer']!r:30s}  "
            f"{lat*1000:.0f}ms"
        )

    mem_mb = peak_memory_mb(device)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n_correct = sum(r.correct for r in per_results)
    accuracy  = n_correct / len(per_results) if per_results else 0.0
    n_garbage    = sum(r.is_garbage for r in per_results)
    garbage_pct  = round(n_garbage / len(per_results) * 100, 2) if per_results else 0.0

    latencies_ms = sorted(r.latency_s * 1000 for r in per_results if r.latency_s > 0)
    avg_lat = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    p50_lat = latencies_ms[len(latencies_ms) // 2] if latencies_ms else 0.0
    p95_lat = latencies_ms[int(len(latencies_ms) * 0.95)] if latencies_ms else 0.0

    tps_vals = [
        r.tokens_generated / r.latency_s
        for r in per_results
        if r.latency_s > 0 and r.tokens_generated > 0
    ]
    avg_tps = sum(tps_vals) / len(tps_vals) if tps_vals else 0.0

    free_model_memory(model, tokenizer, device)

    return BenchmarkReport(
        model_key=model_key,
        model_id=model_id,
        device=str(device),
        dtype=str(dtype),
        timestamp=datetime.now().isoformat(timespec="seconds"),
        eval_mode=eval_mode,
        n_examples=len(per_results),
        n_correct=n_correct,
        accuracy=round(accuracy, 4),
        garbage_pct=garbage_pct,
        avg_latency_ms=round(avg_lat, 2),
        p50_latency_ms=round(p50_lat, 2),
        p95_latency_ms=round(p95_lat, 2),
        avg_tokens_per_sec=round(avg_tps, 2),
        peak_memory_mb=round(mem_mb, 2),
        results=[asdict(r) for r in per_results],
    )
