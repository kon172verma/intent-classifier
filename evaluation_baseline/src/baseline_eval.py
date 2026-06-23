#!/usr/bin/env python3
"""
Baseline evaluation for MCP tool selection.

Usage
-----
# List all registered models
python baseline_eval.py --list-models

# Evaluate a single model (zero-shot)
python baseline_eval.py --model qwen3-0.6b

# Few-shot mode
python baseline_eval.py --model qwen2.5-0.5b --mode few_shot

# Quick smoke-test on first 10 examples
python baseline_eval.py --model smollm2-135m --limit 10

# Custom data file and device
python baseline_eval.py --model qwen3-4b --data ../../dataset_sample/sample.json --device mps
"""

import argparse
import json
import os
import sys
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# ── Path bootstrap ─────────────────────────────────────────────────────────────
# Add repo root to sys.path so evaluation_lib is importable.
_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Load .env for HF_TOKEN (required by gated models: Gemma, Llama)
_env_file = _REPO_ROOT / ".env"
if _env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_file)

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")

from evaluation_lib import (  # noqa: E402
    MODEL_REGISTRY,
    SYSTEM_PROMPT_ZERO_SHOT,
    SYSTEM_PROMPT_FEW_SHOT,
    evaluate,
    resolve_device,
)

DEFAULT_DATA    = _REPO_ROOT / "dataset_sample" / "sample.json"
DEFAULT_OUT_DIR = Path(__file__).parent.parent / "reports_zero_shot"
DEFAULT_FS_DIR  = Path(__file__).parent.parent / "reports_few_shot"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Baseline evaluation for MCP tool routing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", "-m", type=str,
                   help="Model key (see --list-models) or full HF model ID.")
    p.add_argument("--data", "-d", type=Path, default=DEFAULT_DATA)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--mode", type=str, default="zero_shot",
                   choices=["zero_shot", "few_shot"])
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--list-models", action="store_true",
                   help="Print the model registry and exit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_models:
        col = max(len(k) for k in MODEL_REGISTRY)
        print("\nRegistered models:")
        for key, mid in MODEL_REGISTRY.items():
            print(f"  {key:<{col}}  →  {mid}")
        print("\nYou may also pass a full HF model ID directly, e.g. Qwen/Qwen3-0.6B")
        return

    if args.model is None:
        print("Error: --model is required. Use --list-models to see available keys.")
        raise SystemExit(1)

    model_key = args.model.lower()
    if model_key in MODEL_REGISTRY:
        model_id = MODEL_REGISTRY[model_key]
    else:
        model_id  = args.model
        model_key = args.model.split("/")[-1].lower()

    device        = resolve_device(args.device)
    system_prompt = SYSTEM_PROMPT_FEW_SHOT if args.mode == "few_shot" else SYSTEM_PROMPT_ZERO_SHOT
    out_dir       = args.out_dir or (DEFAULT_FS_DIR if args.mode == "few_shot" else DEFAULT_OUT_DIR)

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

    print(f"\n{'='*60}")
    print(f"  RESULTS — {report.model_key}")
    print(f"{'='*60}")
    print(f"  Accuracy       : {report.accuracy:.2%}  ({report.n_correct}/{report.n_examples})")
    print(f"  Avg latency    : {report.avg_latency_ms:.1f} ms")
    print(f"  P50 latency    : {report.p50_latency_ms:.1f} ms")
    print(f"  P95 latency    : {report.p95_latency_ms:.1f} ms")
    print(f"  Avg tokens/sec : {report.avg_tokens_per_sec:.1f}")
    print(f"  Peak memory    : {report.peak_memory_mb:.1f} MB")
    print(f"{'='*60}\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{model_key}_{ts}.json"
    out_path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Report saved → {out_path}\n")


if __name__ == "__main__":
    main()
