#!/usr/bin/env python3
"""
Phase 2 – Zero-shot baseline evaluation for MCP tool selection.

Usage
-----
# List all registered models
python baseline_eval.py --list-models

# Evaluate a single model on sample.json
python baseline_eval.py --model qwen3-0.6b

# Specify data file, device, output directory
python baseline_eval.py --model qwen3-4b --data ../dataset_sample/sample.json --device mps --out-dir reports/

# Quick smoke-test on first 10 examples
python baseline_eval.py --model smollm2-135m --limit 10
"""

import argparse
import gc
import json
import os
import re
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, field

# Load .env from repo root (provides HF_TOKEN for gated models)
from pathlib import Path as _Path
_env_file = _Path(__file__).parent.parent / ".env"
if _env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_file)

# Suppress noisy but harmless transformers generation warnings
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase

# ── Model registry ─────────────────────────────────────────────────────────────
# Maps short CLI keys → HuggingFace model IDs.
# Pass --model with either the key or a full HF model ID (org/name).
MODEL_REGISTRY: dict[str, str] = {
    # ── Open / no-auth models ───────────────────────────────────────────
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "smollm3":      "HuggingFaceTB/SmolLM3-3B",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3-0.6b":   "Qwen/Qwen3-0.6B",
    "qwen3-1.7b":   "Qwen/Qwen3-1.7B",
    "qwen3-4b":     "Qwen/Qwen3-4B",
    "tinyllama":    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # ── Gated models (require HF token + accepted licence) ─────────────
    # Set HF_TOKEN env var or run `huggingface-cli login` before using these.
    "gemma3-270m":  "google/gemma-3-270m-it",
    "gemma3-1b":    "google/gemma-3-1b-it",
    "llama3.2-3b":  "meta-llama/Llama-3.2-3B-Instruct",
}

# Qwen3 tokenizers support enable_thinking= in apply_chat_template.
# We always disable thinking for deterministic single-token routing output.
_QWEN3_KEYS = {"qwen3-0.6b", "qwen3-1.7b", "qwen3-4b"}

# Per-model extra kwargs forwarded to AutoModelForCausalLM.from_pretrained.
_MODEL_EXTRA_LOAD_KWARGS: dict[str, dict] = {}

_ROPE_SCALING_FIX_KEYS: frozenset[str] = frozenset()
_NO_KV_CACHE_KEYS: frozenset[str] = frozenset()

SYSTEM_PROMPT_ZERO_SHOT = (
    "You are a tool router.\n\n"
    "Rules:\n"
    "- Return only the tool name.\n"
    '- Return "none" if no tool matches.\n'
    "- Do not explain."
)

SYSTEM_PROMPT_FEW_SHOT = (
    "You are a tool router.\n\n"
    "Rules:\n"
    "- Return only the tool name.\n"
    '- Return "none" if no tool matches.\n'
    "- Do not explain.\n\n"
    "Examples:\n\n"
    'User Request: "I need directions to the airport."\n'
    "Selected Tool: nav_route_planner\n\n"
    'User Request: "Call my wife."\n'
    "Selected Tool: call_handler\n\n"
    'User Request: "It is too warm in the car."\n'
    "Selected Tool: climate_control\n\n"
    'User Request: "My tyre went flat on the motorway."\n'
    "Selected Tool: roadside_assistance\n\n"
    'User Request: "Read me the latest news headlines."\n'
    "Selected Tool: news_briefing\n\n"
    'User Request: "Lock the car remotely."\n'
    "Selected Tool: remote_vehicle_lock\n\n"
    'User Request: "What is the capital of France?"\n'
    "Selected Tool: none"
)

# Keep SYSTEM_PROMPT as an alias so existing call-sites keep working
SYSTEM_PROMPT = SYSTEM_PROMPT_ZERO_SHOT

DEFAULT_DATA = Path(__file__).parent.parent / "data" / "sample.json"
DEFAULT_OUT_DIR = Path(__file__).parent / "reports"
DEFAULT_FEW_SHOT_OUT_DIR = Path(__file__).parent / "few_shot_reports"


# ── Prompt helpers ─────────────────────────────────────────────────────────────

def _tool_block(tools: list[dict]) -> str:
    parts: list[str] = []
    for t in tools:
        parts.append(f"Name: {t['name']}")
        parts.append(f"Description: {t['description']}")
        parts.append("")
    return "\n".join(parts)


def build_raw_prompt(example: dict, system_prompt: str = SYSTEM_PROMPT_ZERO_SHOT) -> str:
    """Completion-style prompt for models without a chat template."""
    return (
        f"{system_prompt}\n\n"
        f"Available Tools:\n{_tool_block(example['available_tools'])}\n"
        f"User Request:\n{example['user_request']}\n\n"
        "Selected Tool:\n"
    )


def build_chat_messages(example: dict, system_prompt: str = SYSTEM_PROMPT_ZERO_SHOT) -> list[dict]:
    """System + user turn for instruct / chat models."""
    user = (
        f"Available Tools:\n{_tool_block(example['available_tools'])}\n"
        f"User Request:\n{example['user_request']}\n\n"
        "Selected Tool:"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user},
    ]


# ── Output extraction ──────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_prediction(raw_generated: str) -> str:
    """
    Clean and extract the tool name from model output.

    Steps:
    1. Strip Qwen3 <think>…</think> blocks.
    2. Take the first non-empty line.
    3. Remove markdown artifacts (* _ ` " ').
    4. Strip any echoed 'Selected Tool:' prefix.
    """
    text = _THINK_RE.sub("", raw_generated).strip()
    for line in text.splitlines():
        line = line.strip().strip("*_`\"'")
        if line.lower().startswith("selected tool:"):
            line = line[len("selected tool:"):].strip().strip("*_`\"'")
        if line:
            # Take only the first word/token if the model emits extra text
            # e.g. "nav_route_planner (navigation)" → "nav_route_planner"
            token = line.split()[0].rstrip(".,;:()")
            return token
    return ""


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ExampleResult:
    index: int
    user_request: str
    n_tools: int
    answer: str
    prediction: str
    correct: bool
    latency_s: float
    tokens_generated: int


@dataclass
class BenchmarkReport:
    model_key: str
    model_id: str
    device: str
    dtype: str
    timestamp: str
    eval_mode: str          # "zero_shot" or "few_shot"
    n_examples: int
    n_correct: int
    accuracy: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    avg_tokens_per_sec: float
    peak_memory_mb: float
    # Per-example breakdown
    results: list[dict] = field(default_factory=list)


# ── Memory helpers ─────────────────────────────────────────────────────────────

def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "mps":
        # MPS does not expose reset; best-effort read at end
        pass


def peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024**2
    if device.type == "mps":
        try:
            return torch.mps.current_allocated_memory() / 1024**2
        except AttributeError:
            return 0.0
    # CPU: use psutil if available
    try:
        import psutil  # type: ignore[import-untyped]  # optional dependency
        import os
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / 1024**2  # type: ignore[union-attr]
    except ImportError:
        return 0.0


# ── Device selection ───────────────────────────────────────────────────────────

def resolve_device(requested: str) -> torch.device:
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


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    model_key: str = "",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    print(f"  Loading tokenizer … {model_id}")
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    extra_kwargs = dict(_MODEL_EXTRA_LOAD_KWARGS.get(model_key, {}))

    # Some models may need config patching before load (none currently).
    if model_key in _ROPE_SCALING_FIX_KEYS:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        rs = getattr(cfg, "rope_scaling", None)
        if isinstance(rs, dict):
            rope_type = rs.get("rope_type") or rs.get("type", "")
            if rope_type == "default":
                # "default" means standard RoPE (no scaling); the cached
                # modeling_phi3.py only handles None or "longrope", so null it out.
                cfg.rope_scaling = None
            elif "type" not in rs and "rope_type" in rs:
                rs["type"] = rs["rope_type"]
        extra_kwargs["config"] = cfg

    print(f"  Loading model     … (device={device}, dtype={dtype})")
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=str(device),
        trust_remote_code=True,
        **extra_kwargs,
    )
    model.eval()
    return model, tokenizer


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
) -> tuple[str, float, int]:
    """
    Run one example through the model.
    Returns (prediction, latency_seconds, tokens_generated).
    """
    chat_template = getattr(tokenizer, "chat_template", None)
    has_chat_template = bool(chat_template)

    if has_chat_template:
        messages = build_chat_messages(example, system_prompt=system_prompt)
        kwargs: dict = dict(
            tokenize=False,
            add_generation_prompt=True,
        )
        # Qwen3: disable thinking mode for clean single-token output
        if model_key in _QWEN3_KEYS:
            kwargs["enable_thinking"] = False
        prompt_text = tokenizer.apply_chat_template(messages, **kwargs)  # type: ignore[union-attr]
    else:
        prompt_text = build_raw_prompt(example, system_prompt=system_prompt)

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)  # type: ignore[operator]
    input_len = inputs["input_ids"].shape[-1]

    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(  # type: ignore[operator]
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,           # greedy – deterministic
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=use_cache,
        )
    latency = time.perf_counter() - t0

    new_ids = output_ids[0][input_len:]
    tokens_generated = new_ids.shape[0]
    raw_output = str(tokenizer.decode(new_ids, skip_special_tokens=True))  # type: ignore[arg-type]
    prediction = extract_prediction(raw_output)
    return prediction, latency, tokens_generated


# ── Evaluation loop ────────────────────────────────────────────────────────────

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
    # Load dataset
    suffix = data_path.suffix.lower()
    if suffix == ".jsonl":
        examples = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        examples = json.loads(data_path.read_text(encoding="utf-8"))

    if limit is not None:
        examples = examples[:limit]

    dtype = dtype_for_device(device)
    reset_peak_memory(device)

    model, tokenizer = load_model_and_tokenizer(model_id, device, dtype, model_key=model_key)
    reset_peak_memory(device)   # reset after model load to measure only inference

    print(f"\n  Running inference on {len(examples)} examples …\n")

    per_results: list[ExampleResult] = []

    for i, ex in enumerate(examples):
        try:
            pred, lat, ntok = run_example(
                ex, model, tokenizer, device, model_key, max_new_tokens,
                use_cache=(model_key not in _NO_KV_CACHE_KEYS),
                system_prompt=system_prompt,
            )
        except Exception:  # noqa: BLE001
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

    # ── Aggregate metrics ────────────────────────────────────────────────
    n_correct = sum(r.correct for r in per_results)
    accuracy  = n_correct / len(per_results) if per_results else 0.0

    latencies_ms = sorted(r.latency_s * 1000 for r in per_results if r.latency_s > 0)
    avg_lat  = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    p50_lat  = latencies_ms[len(latencies_ms) // 2] if latencies_ms else 0.0
    p95_lat  = latencies_ms[int(len(latencies_ms) * 0.95)] if latencies_ms else 0.0

    tps_vals = [
        r.tokens_generated / r.latency_s
        for r in per_results
        if r.latency_s > 0 and r.tokens_generated > 0
    ]
    avg_tps = sum(tps_vals) / len(tps_vals) if tps_vals else 0.0

    # ── Free model memory immediately so the OS can reclaim it ───────
    del model, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass

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
        avg_latency_ms=round(avg_lat, 2),
        p50_latency_ms=round(p50_lat, 2),
        p95_latency_ms=round(p95_lat, 2),
        avg_tokens_per_sec=round(avg_tps, 2),
        peak_memory_mb=round(mem_mb, 2),
        results=[asdict(r) for r in per_results],
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 zero-shot baseline evaluation for MCP tool routing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        help=(
            "Model key from the registry (e.g. qwen3-0.6b) "
            "or a full HuggingFace model ID (e.g. Qwen/Qwen3-0.6B)."
        ),
    )
    parser.add_argument(
        "--data", "-d",
        type=Path,
        default=DEFAULT_DATA,
        help="Path to the dataset file (.json or .jsonl).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device. 'auto' picks CUDA > MPS > CPU.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum new tokens to generate per example.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="zero_shot",
        choices=["zero_shot", "few_shot"],
        help="Prompt mode: zero_shot uses the bare system prompt; few_shot embeds examples in it.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write the JSON report (default: reports/ for zero_shot, few_shot_reports/ for few_shot).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N examples (for quick smoke-tests).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the model registry and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_models:
        print("\nRegistered models:")
        col = max(len(k) for k in MODEL_REGISTRY)
        for key, mid in MODEL_REGISTRY.items():
            print(f"  {key:<{col}}  →  {mid}")
        print(
            "\nYou may also pass a full HuggingFace model ID "
            "directly to --model, e.g. Qwen/Qwen3-0.6B"
        )
        return

    if args.model is None:
        print("Error: --model is required. Use --list-models to see available keys.")
        raise SystemExit(1)

    # Resolve model key / ID
    model_key = args.model.lower()
    if model_key in MODEL_REGISTRY:
        model_id = MODEL_REGISTRY[model_key]
    else:
        # Treat as a full HF model ID; derive a key from the last path component
        model_id  = args.model
        model_key = args.model.split("/")[-1].lower()

    device = resolve_device(args.device)
    system_prompt = SYSTEM_PROMPT_FEW_SHOT if args.mode == "few_shot" else SYSTEM_PROMPT_ZERO_SHOT
    out_dir = args.out_dir or (DEFAULT_FEW_SHOT_OUT_DIR if args.mode == "few_shot" else DEFAULT_OUT_DIR)

    print(f"\n{'='*60}")
    print(f"  Model   : {model_id}")
    print(f"  Mode    : {args.mode}")
    print(f"  Device  : {device}")
    print(f"  Data    : {args.data}")
    print(f"  Limit   : {args.limit or 'all'}")
    print(f"{'='*60}\n")

    report = evaluate(
        model_key=model_key,
        model_id=model_id,
        data_path=args.data,
        device=device,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        system_prompt=system_prompt,
        eval_mode=args.mode,
    )

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS — {report.model_key}")
    print(f"{'='*60}")
    print(f"  Accuracy          : {report.accuracy:.2%}  ({report.n_correct}/{report.n_examples})")
    print(f"  Avg latency       : {report.avg_latency_ms:.1f} ms")
    print(f"  P50 latency       : {report.p50_latency_ms:.1f} ms")
    print(f"  P95 latency       : {report.p95_latency_ms:.1f} ms")
    print(f"  Avg tokens/sec    : {report.avg_tokens_per_sec:.1f}")
    print(f"  Peak memory       : {report.peak_memory_mb:.1f} MB")
    print(f"{'='*60}\n")

    # ── Save report ──────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{model_key}_{ts}.json"
    out_path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Report saved → {out_path}\n")


if __name__ == "__main__":
    main()
