"""
release.py
==========
Publish a version release: merge the two best PEFT adapters (sourced from
HF_EXPERIMENTS_REPO) into standalone full-weight models, push the merged
models plus their GGUF and ONNX exports to HF_RELEASE_REPO, then stamp this
GitHub repo with a version tag.

Responsibility split
---------------------
  HF_EXPERIMENTS_REPO (kon172verma/intent-classifier-experiments)
      Every adapter produced during experimentation. This script only reads
      from it (never writes).
  HF_RELEASE_REPO (kon172verma/intent-classifier)
      Only the merged/unloaded models chosen for a release, plus GGUF/ONNX.
      This script is the sole writer for this repo.
  GitHub repo (this repo, kon172verma/intent-classifier)
      Tagged locally with the release version. Tags are NOT pushed
      automatically — review, then `git push origin <version>` yourself.

Each --run entry is "{model_key}_{lora_config}_{dataset_size}[:{timestamp}]",
e.g. "qwen3-0.6b_C_1k" or "qwen3-0.6b_C_1k:20260101-120000". If the timestamp
is omitted, the latest matching entry in EXPERIMENTS.jsonl is used.

Usage
-----
    # Merge + push the two best models for a release
    python release.py --version v1.0 --technique LoRA \\
        --runs qwen3-0.6b_C_1k llama3.2-1b_C_1k

    # Also publish GGUF / ONNX exports produced by intent-classifier-inference
    python release.py --version v1.0 --technique LoRA \\
        --runs qwen3-0.6b_C_1k llama3.2-1b_C_1k \\
        --gguf-dir ../intent-classifier-inference/models/gguf \\
        --onnx-dir ../intent-classifier-inference/models/onnx

    # Only create the local git tag (no merging)
    python release.py --tag-only --version v1.0 --message "LoRA v1.0 release"
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# ── allow running from repo root ──────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT))

from finetune_lib.config import (
    CURRENT_VERSION,
    FINETUNE_MODEL_REGISTRY,
    HF_EXPERIMENTS_REPO,
    HF_RELEASE_REPO,
    hf_adapter_subfolder,
    hf_merged_subfolder,
)
from finetune_lib.registry import find_latest_experiment

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _parse_run(run: str) -> tuple[str, str, str, str | None]:
    """
    Parse "{model_key}_{lora_config}_{dataset_size}[:{timestamp}]".

    The model_key can contain hyphens and dots (e.g. "qwen2.5-0.5b"),
    lora_config is a single uppercase letter (A–D), and dataset_size is
    typically "1k". An optional ":timestamp" suffix pins an exact experiment.

    Returns (model_key, lora_config, dataset_size, timestamp_or_None).
    """
    run_spec, _, timestamp = run.partition(":")
    parts = run_spec.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(
            f"Cannot parse run '{run}'. Expected format: "
            "<model_key>_<config>_<size>[:<timestamp>], e.g. 'qwen3-0.6b_C_1k'."
        )
    model_key, lora_config, dataset_size = parts
    return model_key, lora_config, dataset_size, (timestamp or None)


def _base_model_id(model_key: str) -> str:
    if model_key not in FINETUNE_MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model_key '{model_key}'. Valid keys: {list(FINETUNE_MODEL_REGISTRY)}"
        )
    return FINETUNE_MODEL_REGISTRY[model_key]


def _resolve_adapter_subfolder(
    technique: str,
    model_key: str,
    lora_config: str,
    dataset_size: str,
    timestamp: str | None,
    version: str,
) -> str:
    """Resolve the exact HF_EXPERIMENTS_REPO subfolder for a run."""
    if timestamp is None:
        entry = find_latest_experiment(
            technique=technique,
            model_key=model_key,
            lora_config=lora_config,
            dataset_size=dataset_size,
            version=version,
        )
        if entry is None:
            raise ValueError(
                f"No experiment logged in EXPERIMENTS.jsonl for "
                f"{technique}/{model_key}_{lora_config}_{dataset_size} "
                f"(version={version}). Pass an explicit "
                "'model_config_size:timestamp' run spec instead."
            )
        timestamp = entry["timestamp"]
    return hf_adapter_subfolder(
        technique,
        model_key,
        lora_config,
        dataset_size,
        timestamp=timestamp,
        version=version,
    )


def _merge_one(
    technique: str,
    run: str,
    version: str,
    hf_token: str | None,
    tmp_root: Path,
) -> Path:
    """
    Download adapter from HF_EXPERIMENTS_REPO, merge into base model, save locally.
    Returns the path to the merged model directory.
    """
    # Import heavy libs only when needed (not required at import-time)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_key, lora_config, dataset_size, timestamp = _parse_run(run)
    base_id = _base_model_id(model_key)
    adapter_subfolder = _resolve_adapter_subfolder(
        technique, model_key, lora_config, dataset_size, timestamp, version
    )

    print(f"\n  Loading base model  : {base_id}")
    dtype = torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=dtype,
        device_map="cpu",  # CPU merge — avoids GPU memory limits
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_id, token=hf_token)

    print(f"  Loading adapter     : {HF_EXPERIMENTS_REPO}/{adapter_subfolder}")
    peft_model = PeftModel.from_pretrained(
        base_model,
        HF_EXPERIMENTS_REPO,
        subfolder=adapter_subfolder,
        token=hf_token,
    )

    print("  Merging and unloading...")
    merged = peft_model.merge_and_unload()

    run_tag = f"{model_key}_{lora_config}_{dataset_size}"
    out_dir = tmp_root / f"{technique}_{run_tag}_merged"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Saving to tmp dir   : {out_dir}")
    merged.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    return out_dir


def _push_merged(
    merged_dir: Path,
    technique: str,
    run: str,
    version: str,
    hf_token: str | None,
) -> None:
    from huggingface_hub import HfApi

    model_key, _lora_config, _dataset_size, _timestamp = _parse_run(run)
    sub = hf_merged_subfolder(model_key)

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=HF_RELEASE_REPO, repo_type="model", exist_ok=True, private=True)

    print(f"  Pushing merged model: {HF_RELEASE_REPO}/{sub}")
    api.upload_folder(
        folder_path=str(merged_dir),
        repo_id=HF_RELEASE_REPO,
        path_in_repo=sub,
        commit_message=f"Add merged model {technique}/{model_key} [{version}]",
    )
    print("  Pushed successfully.")


def _push_format_dir(
    local_dir: Path,
    format_name: str,
    model_key: str,
    hf_token: str | None,
) -> None:
    """Push a local GGUF or ONNX directory to HF_RELEASE_REPO/{model_key}/{format_name}/.

    `local_dir` is expected to contain either files or subdirectories whose
    names are prefixed with `model_key` (e.g. GGUF's `qwen3-0.6b-Q4_K_M.gguf`
    or ONNX's `qwen3-0.6b-fp16/`).
    """
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=HF_RELEASE_REPO, repo_type="model", exist_ok=True, private=True)
    sub = f"{model_key}/{format_name.lower()}"
    print(f"  Pushing {format_name}: {HF_RELEASE_REPO}/{sub}")
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=HF_RELEASE_REPO,
        path_in_repo=sub,
        commit_message=f"Add {format_name} export for {model_key}",
        allow_patterns=[f"{model_key}*"],
    )
    print(f"  Pushed {format_name} successfully.")


def _create_hf_tag(version: str, message: str, hf_token: str | None) -> None:
    """Optionally also tag HF_RELEASE_REPO (opt-in via --hf-tag)."""
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    print(f"\n  Creating HF tag '{version}' on {HF_RELEASE_REPO}…")
    api.create_tag(
        repo_id=HF_RELEASE_REPO,
        repo_type="model",
        tag=version,
        tag_message=message,
        token=hf_token,
        exist_ok=True,  # idempotent — re-running won't crash
    )
    print(f"  HF tag '{version}' created.")
    print(f"  View at: https://huggingface.co/{HF_RELEASE_REPO}/tree/{version}")


def _create_github_tag(version: str, message: str) -> None:
    """Create a local, annotated git tag on this repo. Never pushes automatically."""
    print(f"\n  Creating local git tag '{version}'…")
    result = subprocess.run(
        ["git", "tag", "-a", version, "-m", message],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  WARNING: git tag failed: {result.stderr.strip()}")
        return
    print(f"  Local tag '{version}' created.")
    print(f"  Review it, then push with: git push origin {version}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge adapters, publish a release, and tag the repo."
    )
    parser.add_argument("--version", default=CURRENT_VERSION, help="Version tag, e.g. v1.0")
    parser.add_argument(
        "--technique",
        default=None,
        choices=["LoRA", "DoRA", "LoRAplus", "DoRAplus", "QLoRA", "AdaLoRA"],
        help="Technique of the adapters being released (required unless --tag-only).",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=[],
        metavar="MODEL_CFG_SIZE[:TIMESTAMP]",
        help=(
            "Runs to merge, e.g. qwen3-0.6b_C_1k llama3.2-1b_C_1k. "
            "Append ':timestamp' to pin an exact experiment; otherwise the "
            "latest matching EXPERIMENTS.jsonl entry is used. "
            "Omit (or use --tag-only) to skip merging and just create the tag."
        ),
    )
    parser.add_argument(
        "--gguf-dir",
        type=Path,
        default=None,
        help="Local directory of *.gguf files to publish alongside the merged models.",
    )
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=None,
        help="Local directory of ONNX export folders to publish alongside the merged models.",
    )
    parser.add_argument(
        "--tag-only",
        action="store_true",
        help="Skip merging; only create the version tag.",
    )
    parser.add_argument(
        "--message",
        default=None,
        help="Human-readable tag annotation. Defaults to the version string.",
    )
    parser.add_argument(
        "--no-tag",
        action="store_true",
        help="Merge and push models but do NOT create a tag (useful for incremental pushes).",
    )
    parser.add_argument(
        "--hf-tag",
        action="store_true",
        help="Also create a tag on HF_RELEASE_REPO (default: GitHub tag only).",
    )
    args = parser.parse_args()

    hf_token: str | None = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("WARNING: HF_TOKEN not set. Pushes to private repos will fail.")

    tag_message = args.message or args.version

    if not args.tag_only:
        if not args.technique:
            parser.error("--technique is required when merging (omit only with --tag-only).")

        with tempfile.TemporaryDirectory(prefix="ic_merge_") as tmp:
            tmp_root = Path(tmp)
            for run in args.runs:
                print(f"\n── Merging {args.technique}/{run} ──────────────────")
                try:
                    merged_dir = _merge_one(args.technique, run, args.version, hf_token, tmp_root)
                    _push_merged(merged_dir, args.technique, run, args.version, hf_token)
                except Exception as exc:
                    print(f"  ERROR merging {run}: {exc}")
                    print("  Skipping this run; continuing with others.")

            for run in args.runs:
                model_key, _cfg, _size, _ts = _parse_run(run)
                if args.gguf_dir:
                    try:
                        _push_format_dir(args.gguf_dir, "GGUF", model_key, hf_token)
                    except Exception as exc:
                        print(f"  ERROR pushing GGUF for {model_key}: {exc}")
                if args.onnx_dir:
                    try:
                        _push_format_dir(args.onnx_dir, "ONNX", model_key, hf_token)
                    except Exception as exc:
                        print(f"  ERROR pushing ONNX for {model_key}: {exc}")

    if not args.no_tag:
        _create_github_tag(args.version, tag_message)
        if args.hf_tag:
            _create_hf_tag(args.version, tag_message, hf_token)
    else:
        print("\n--no-tag set; skipping tag creation.")

    print("\nDone.")


if __name__ == "__main__":
    main()
