#!/usr/bin/env python3
"""
Phase 3 – Quantization evaluation for selected PEFT candidates.

Runs the 5 selected models under three quantization configs:
  int8        – bitsandbytes LLM.int8()
  nf4         – 4-bit NF4 (QLoRA default)
  nf4_dq      – 4-bit NF4 + double quantization

Supports both zero-shot and few-shot prompt modes.

Usage
-----
# Single model, single quant, zero-shot
python quant_eval.py --model qwen2.5-0.5b --quant nf4

# Few-shot
python quant_eval.py --model gemma3-1b --quant int8 --mode few_shot

# Quick smoke-test
python quant_eval.py --model qwen2.5-0.5b --quant nf4 --limit 5
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

# ── path bootstrap ──────────────────────────────────────────────────────────
_BASELINE_DIR = Path(__file__).parent.parent / "baseline_evaluation"
sys.path.insert(0, str(_BASELINE_DIR))

# Suppress noisy HF warnings before importing transformers
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")
warnings.filterwarnings("ignore", message=".*bitsandbytes.*")

from dotenv import load_dotenv  # noqa: E402
_env_file = Path(__file__).parent.parent.parent / ".env"
if _env_file.exists():
    load_dotenv(_env_file)

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# Re-use all prompt logic from baseline_eval
from baseline_eval import (  # type: ignore[import]  # noqa: E402
    SYSTEM_PROMPT_ZERO_SHOT,
    SYSTEM_PROMPT_FEW_SHOT,
    _QWEN3_KEYS,
    run_example,
    extract_prediction,
    resolve_device,
    peak_memory_mb,
    reset_peak_memory,
    ExampleResult,
)

# ── Selected models for PEFT ────────────────────────────────────────────────
SELECTED_MODELS: dict[str, str] = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen3-0.6b":   "Qwen/Qwen3-0.6B",
    "gemma3-1b":    "google/gemma-3-1b-it",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "smollm3":      "HuggingFaceTB/SmolLM3-3B",
}

# ── Quantization configs ────────────────────────────────────────────────────
QUANT_CONFIGS: dict[str, Optional[BitsAndBytesConfig]] = {
    "int8": BitsAndBytesConfig(load_in_8bit=True),
    "nf4": BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    ),
    "nf4_dq": BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ),
}

QUANT_LABELS: dict[str, str] = {
    "int8":   "INT8",
    "nf4":    "NF4",
    "nf4_dq": "NF4+DQ",
}

DEFAULT_DATA = Path(__file__).parent.parent / "dataset_sample" / "sample.json"
DEFAULT_OUT_DIR = Path(__file__).parent / "reports"


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class QuantBenchmarkReport:
    model_key: str
    model_id: str
    quant: str          # "int8" | "nf4" | "nf4_dq"
    device: str
    compute_dtype: str
    timestamp: str
    eval_mode: str      # "zero_shot" | "few_shot"
    n_examples: int
    n_correct: int
    accuracy: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    avg_tokens_per_sec: float
    peak_memory_mb: float
    results: list[dict] = field(default_factory=list)


# ── Model loading with quantization ─────────────────────────────────────────

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

    # BitsAndBytes quantized models must be loaded onto CPU/CUDA or via device_map.
    # On MPS bitsandbytes falls back gracefully with device_map="auto" → CPU.
    print(f"  Loading model [{QUANT_LABELS[quant]}] … (device_map=auto, compute_dtype=bfloat16)")
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


# ── Evaluation loop ──────────────────────────────────────────────────────────

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

    # Determine actual device the model landed on (BnB may move to CPU on MPS)
    try:
        actual_device = next(model.parameters()).device
    except StopIteration:
        actual_device = device

    reset_peak_memory(device)
    print(f"\n  Running inference on {len(examples)} examples …\n")

    per_results: list[ExampleResult] = []

    for i, ex in enumerate(examples):
        try:
            pred, lat, ntok = run_example(
                ex, model, tokenizer, actual_device, model_key, max_new_tokens,
                use_cache=True,
                system_prompt=system_prompt,
            )
        except Exception as exc:  # noqa: BLE001
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

    n_correct = sum(r.correct for r in per_results)
    accuracy  = n_correct / len(per_results) if per_results else 0.0
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

    del model, tokenizer
    gc.collect()
    try:
        torch.mps.empty_cache()
    except AttributeError:
        pass
    if torch.cuda.is_available():
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


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantization evaluation for PEFT candidate models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m", type=str, required=True,
        choices=list(SELECTED_MODELS.keys()),
        help="Model key.",
    )
    parser.add_argument(
        "--quant", "-q", type=str, required=True,
        choices=list(QUANT_CONFIGS.keys()),
        help="Quantization scheme: int8 | nf4 | nf4_dq.",
    )
    parser.add_argument(
        "--mode", type=str, default="zero_shot",
        choices=["zero_shot", "few_shot"],
        help="Prompt mode.",
    )
    parser.add_argument(
        "--data", "-d", type=Path, default=DEFAULT_DATA,
        help="Dataset file (.json or .jsonl).",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=32,
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only first N examples.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    model_id = SELECTED_MODELS[args.model]

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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out_dir / f"{args.model}_{args.quant}_{args.mode}_{ts}.json"
    out_path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Report saved → {out_path}\n")


if __name__ == "__main__":
    main()
