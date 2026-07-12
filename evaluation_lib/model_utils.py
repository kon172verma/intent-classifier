"""
evaluation_lib/model_utils.py
==============================
Model loading, device resolution, and GPU/CPU memory utilities.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase

# ── Device resolution ──────────────────────────────────────────────────────────

def resolve_device(requested: str) -> torch.device:
    """Resolve a device string; 'auto' picks CUDA > MPS > CPU."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def dtype_for_device(device: torch.device) -> torch.dtype:
    """Use bfloat16 on accelerators, float32 on CPU."""
    if device.type in ("cuda", "mps"):
        return torch.bfloat16
    return torch.float32


# ── Memory tracking ────────────────────────────────────────────────────────────

def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024 ** 2
    if device.type == "mps":
        try:
            return torch.mps.current_allocated_memory() / 1024 ** 2
        except AttributeError:
            return 0.0
    try:
        import os
        import psutil  # type: ignore[import-untyped]
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    except ImportError:
        return 0.0


def free_model_memory(model: Any, tokenizer: Any, device: torch.device) -> None:
    """Delete model + tokenizer and flush the GPU/MPS allocator cache."""
    del model, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    model_key: str = "",
    extra_kwargs: dict | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a HuggingFace causal LM and its tokenizer onto *device*."""
    print(f"  Loading tokenizer … {model_id}")
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    kwargs: dict = dict(extra_kwargs or {})

    # Attempt Flash Attention 2 on CUDA for faster prefill; fall back silently
    # if the package is absent or the model architecture doesn't support it.
    if device.type == "cuda":
        try:
            print(f"  Loading model     … (device={device}, dtype={dtype}, attn=flash_attention_2)")
            model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=dtype,
                device_map=str(device),
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                **kwargs,
            )
        except (ImportError, ValueError, NotImplementedError):
            print(f"  Loading model     … (device={device}, dtype={dtype}, attn=sdpa [FA2 unavailable])")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=dtype,
                device_map=str(device),
                trust_remote_code=True,
                **kwargs,
            )
    else:
        print(f"  Loading model     … (device={device}, dtype={dtype})")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=str(device),
            trust_remote_code=True,
            **kwargs,
        )
    model.eval()
    return model, tokenizer
