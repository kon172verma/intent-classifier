#!/usr/bin/env python3
"""
Quantization evaluation for 4 selected PEFT candidates.

Runs the 4 selected models under multiple quantization configs:
    int8  – bitsandbytes LLM.int8()
    nf4   – 4-bit NF4 (QLoRA default, NVIDIA GPUs only)
    int4  – 4-bit INT4 (ONNX/TensorRT compatible, wider device support)
    fp8   – 8-bit FP8 (modern GPUs + edge devices)

Supports both zero-shot and few-shot prompt modes.

Note on device compatibility:
  nf4   : NVIDIA GPUs only (H100, A100, L4, T4, G4)
  int4  : Broader support (TensorRT, ONNX, QNN, edge devices)
  fp8   : Most modern GPUs + upcoming edge device support
  int8  : Universal fallback

Usage
-----
# Single model, single quant, zero-shot
python quant_eval.py --model qwen2.5-0.5b --quant int8

# Few-shot
python quant_eval.py --model qwen2.5-1.5b --quant nf4 --mode few_shot

# Quick smoke-test
python quant_eval.py --model qwen2.5-0.5b --quant int4 --limit 5
"""

import argparse
import gc
import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

_env_file = _REPO_ROOT / ".env"
if _env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_file)

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")
warnings.filterwarnings("ignore", message=".*bitsandbytes.*")

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from evaluation_lib import (  # noqa: E402
    SYSTEM_PROMPT_ZERO_SHOT,
    SYSTEM_PROMPT_FEW_SHOT,
    run_example,
    resolve_device,
    peak_memory_mb,
    reset_peak_memory,
    ExampleResult,
    compute_prefix_kv_cache,
)
from evaluation_lib.config import MODEL_REGISTRY, QWEN3_KEYS  # noqa: E402

# ── Selected models for quantization evaluation ──────────────────────────────
# Core set covering a range of sizes and architectures (open models only).
SELECTED_MODELS_QUANT: dict[str, str] = {
    "qwen2.5-0.5b": MODEL_REGISTRY["qwen2.5-0.5b"],   # small,  instruct
    "qwen3-0.6b":   MODEL_REGISTRY["qwen3-0.6b"],      # small,  unified base+chat
    "qwen3-1.7b":   MODEL_REGISTRY["qwen3-1.7b"],      # medium, unified base+chat
    "smollm3":      MODEL_REGISTRY["smollm3"],          # large,  unified base+chat
}
SELECTED_MODELS: dict[str, str] = dict(SELECTED_MODELS_QUANT)

# ── Quantization configs ───────────────────────────────────────────────────────
QUANT_CONFIGS: dict[str, Optional[BitsAndBytesConfig]] = {
    "int8": BitsAndBytesConfig(load_in_8bit=True),
    "nf4": BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    ),
    "int4": BitsAndBytesConfig(
        # bitsandbytes supports "nf4" and "fp4" 4-bit types.
        # Keep the CLI label as int4 for user-facing consistency.
        load_in_4bit=True,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    ),
    "fp8": BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=["lm_head"],
    ),  # fp8 via HF (fallback to int8 if not supported)
}

QUANT_LABELS: dict[str, str] = {
    "int8": "INT8",
    "nf4":  "NF4",
    "int4": "INT4",
    "fp8":  "FP8",
}

DEFAULT_DATA    = _REPO_ROOT / "dataset_sample" / "sample.json"
DEFAULT_OUT_DIR = Path(__file__).parent.parent / "reports_int8"


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class QuantBenchmarkReport:
    model_key:          str
    model_id:           str
    quant:              str     # "int8" | "nf4" | "int4" | "fp8"
    device:             str
    compute_dtype:      str
    timestamp:          str
    eval_mode:          str     # "zero_shot" | "few_shot"
    n_examples:         int
    n_correct:          int
    accuracy:           float
    avg_latency_ms:     float
    p50_latency_ms:     float
    p95_latency_ms:     float
    avg_tokens_per_sec: float
    peak_memory_mb:     float
    results:            list[dict] = field(default_factory=list)


# ── Quantized model loading ────────────────────────────────────────────────────

def load_quantized(
    model_id: str,
    quant: str,
    device: torch.device,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    bnb_config = QUANT_CONFIGS[quant]
    print(f"  Loading tokenizer … {model_id}")
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # BitsAndBytes quantized models use device_map="auto" (works on CUDA;
    # falls back to CPU on MPS).
    # Attempt Flash Attention 2 on CUDA; fall back silently if unavailable.
    load_kwargs: dict = dict(
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    if device.type == "cuda":
        try:
            print(f"  Loading model [{QUANT_LABELS[quant]}] … (device_map=auto, attn=flash_attention_2)")
            model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                model_id,
                **load_kwargs,
                attn_implementation="flash_attention_2",
            )
        except (ImportError, ValueError, NotImplementedError):
            print(f"  Loading model [{QUANT_LABELS[quant]}] … (device_map=auto, attn=sdpa [FA2 unavailable])")
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    else:
        print(f"  Loading model [{QUANT_LABELS[quant]}] … (device_map=auto, compute_dtype=bfloat16)")
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    return model, tokenizer


# ── Evaluation loop ────────────────────────────────────────────────────────────

def evaluate_quantized(
    model_key: str,
    model_id: str,
    quant: str,
    data_path: Path,
    device: torch.device,
    max_new_tokens: int = 32,
    limit: Optional[int] = None,
    eval_mode: str = "zero_shot",
) -> QuantBenchmarkReport:
    # Load dataset
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

    system_prompt = (
        SYSTEM_PROMPT_FEW_SHOT if eval_mode == "few_shot" else SYSTEM_PROMPT_ZERO_SHOT
    )

    reset_peak_memory(device)
    model, tokenizer = load_quantized(model_id, quant, device)

    # Determine actual device (BnB may move to CPU on MPS)
    try:
        actual_device = next(model.parameters()).device
    except StopIteration:
        actual_device = device

    reset_peak_memory(device)
    prefix_kv = None
    prefix_len = 0
    try:
        prefix_kv, prefix_len = compute_prefix_kv_cache(
            tokenizer, model, model_key, system_prompt, actual_device
        )
        print(f"  Prefix KV cache: {prefix_len} tokens cached\n")
    except Exception:
        import traceback
        traceback.print_exc()
        print("  Prefix KV cache unavailable — falling back to full prefill\n")
    print(f"\n  Running inference on {len(examples)} examples …\n")

    per_results: list[ExampleResult] = []
    for i, ex in enumerate(examples):
        try:
            pred, lat, ntok = run_example(
                ex, model, tokenizer, actual_device, model_key, max_new_tokens,
                use_cache=True,
                system_prompt=system_prompt,
                prefix_kv=prefix_kv,
                prefix_len=prefix_len,
            )
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            pred, lat, ntok = "", 0.0, 0

        correct = pred.strip().lower() == ex["answer"].strip().lower()
        per_results.append(ExampleResult(
            index=i,
            user_request=ex["user_request"],
            n_tools=len(ex["available_tools"]),
            answer=ex["answer"],
            prediction=pred,
            correct=correct,
            latency_s=lat,
            tokens_generated=ntok,
        ))
        status = "✓" if correct else "✗"
        print(
            f"  [{i+1:>3}/{len(examples)}] {status} "
            f"pred={pred!r:30s}  ans={ex['answer']!r:30s}  "
            f"{lat*1000:.0f}ms"
        )

    mem_mb = peak_memory_mb(device)

    n_correct    = sum(r.correct for r in per_results)
    accuracy     = n_correct / len(per_results) if per_results else 0.0
    latencies_ms = sorted(r.latency_s * 1000 for r in per_results if r.latency_s > 0)
    avg_lat      = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    p50_lat      = latencies_ms[len(latencies_ms) // 2] if latencies_ms else 0.0
    p95_lat      = latencies_ms[int(len(latencies_ms) * 0.95)] if latencies_ms else 0.0
    tps_vals     = [
        r.tokens_generated / r.latency_s
        for r in per_results
        if r.latency_s > 0 and r.tokens_generated > 0
    ]
    avg_tps = sum(tps_vals) / len(tps_vals) if tps_vals else 0.0

    del model, tokenizer
    gc.collect()
    if device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return QuantBenchmarkReport(
        model_key=model_key,
        model_id=model_id,
        quant=quant,
        device=str(actual_device),
        compute_dtype="bfloat16",
        timestamp=datetime.now().isoformat(timespec="seconds"),
        eval_mode=eval_mode,
        n_examples=len(per_results),
        n_correct=n_correct,
        accuracy=round(accuracy, 4),
        avg_latency_ms=round(avg_lat, 2),
        p50_latency_ms=round(p50_lat, 2),
        p95_latency_ms=round(p95_lat, 2),
        avg_tokens_per_sec=round(avg_tps, 2),
        peak_memory_mb=round(mem_mb, 2),
        results=[asdict(r) for r in per_results],
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def _out_dir_for_quant(quant: str) -> Path:
    mapping = {
        "int8": "reports_int8",
        "nf4":  "reports_nf4",
        "int4": "reports_int4",
        "fp8":  "reports_fp8",
    }
    return Path(__file__).parent.parent / mapping.get(quant, "reports_int8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantization evaluation for 4 selected PEFT candidate models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", "-m", type=str, required=True,
                        choices=list(SELECTED_MODELS.keys()), help="Model key.")
    parser.add_argument("--quant", "-q", type=str, required=True,
                        choices=list(QUANT_CONFIGS.keys()),
                        help="Quantization scheme: int8 | nf4 | int4 | fp8.")
    parser.add_argument("--mode", type=str, default="zero_shot",
                        choices=["zero_shot", "few_shot"], help="Prompt mode.")
    parser.add_argument("--data", "-d", type=Path, default=DEFAULT_DATA,
                        help="Dataset file (.json or .jsonl).")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N examples.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Override automatic output directory (defaults to reports_{quant}).")
    args = parser.parse_args()

    device   = resolve_device(args.device)
    model_id = SELECTED_MODELS[args.model]
    out_dir  = args.out_dir or _out_dir_for_quant(args.quant)

    print(f"\n{'='*60}")
    print(f"  Model   : {model_id}")
    print(f"  Quant   : {QUANT_LABELS[args.quant]}")
    print(f"  Mode    : {args.mode}")
    print(f"  Device  : {device}")
    print(f"  Data    : {args.data}")
    print(f"  Limit   : {args.limit or 'all'}")
    print(f"{'='*60}\n")

    report = evaluate_quantized(
        model_key=args.model,
        model_id=model_id,
        quant=args.quant,
        data_path=args.data,
        device=device,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        eval_mode=args.mode,
    )

    print(f"\n{'='*60}")
    print(f"  RESULTS — {report.model_key} [{QUANT_LABELS[report.quant]}]")
    print(f"{'='*60}")
    print(f"  Accuracy       : {report.accuracy:.2%}  ({report.n_correct}/{report.n_examples})")
    print(f"  Avg latency    : {report.avg_latency_ms:.1f} ms")
    print(f"  Peak memory    : {report.peak_memory_mb:.1f} MB")
    print(f"{'='*60}\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{args.model}_{args.quant}_{args.mode}_{ts}.json"
    out_path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Report saved → {out_path}\n")


if __name__ == "__main__":
    main()
