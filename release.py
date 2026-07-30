"""
release.py
==========
Merge the best PEFT adapters into standalone full-weight models, push them to
HuggingFace Hub, then stamp the whole repo with a version tag.

Usage (run on Colab after training is complete, or locally with enough RAM):

    python release.py --version v1.0 --technique LoRA --runs qwen3-0.6b_C_1k llama3.2-1b_C_1k
    python release.py --version v1.0 --technique DoRA --runs qwen3-0.6b_C_1k llama3.2-1b_C_1k
    python release.py --tag-only   --version v1.0 --message "LoRA + DoRA + LoRA+ v1"

Each --run entry is "{model_key}_{lora_config}_{dataset_size}", e.g. "qwen3-0.6b_C_1k".
The adapters are loaded from HF (HF_HUB_REPO/{technique}/{run}), merged, saved to a
tmp dir, then pushed to HF_HUB_REPO/{technique}_merged/{run}.

After all merges, create_tag() stamps the repo.  The tag is visible in the
"Files and versions" tab on the HF model card page.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

# ── allow running from repo root ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from finetune_lib.config import (
    FINETUNE_MODEL_REGISTRY,
    HF_HUB_REPO,
    QWEN3_FINETUNE_KEYS,
    hf_merged_subfolder,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _parse_run(run: str) -> tuple[str, str, str]:
    """
    Parse "{model_key}_{lora_config}_{dataset_size}" into its three parts.

    The model_key can contain hyphens and dots (e.g. "qwen2.5-0.5b"),
    lora_config is a single uppercase letter (A–D), and dataset_size is
    typically "1k".

    Returns (model_key, lora_config, dataset_size).
    """
    # dataset_size is always the last token, lora_config is the one before it.
    parts = run.rsplit("_", 2)
    if len(parts) != 3:  # noqa: PLR2004
        raise ValueError(
            f"Cannot parse run '{run}'. Expected format: <model_key>_<config>_<size>, "
            "e.g. 'qwen3-0.6b_C_1k'."
        )
    model_key, lora_config, dataset_size = parts
    return model_key, lora_config, dataset_size


def _base_model_id(model_key: str) -> str:
    if model_key not in FINETUNE_MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model_key '{model_key}'. "
            f"Valid keys: {list(FINETUNE_MODEL_REGISTRY)}"
        )
    return FINETUNE_MODEL_REGISTRY[model_key]


def _merge_one(
    technique: str,
    run: str,
    hf_token: str | None,
    tmp_root: Path,
) -> Path:
    """
    Download adapter from HF, merge into base model, save locally.
    Returns the path to the merged model directory.
    """
    # Import heavy libs only when needed (not required at import-time)
    import torch
    from peft import PeftModel  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_key, _cfg, _size = _parse_run(run)
    base_id = _base_model_id(model_key)
    adapter_subfolder = f"{technique}/{run}"

    print(f"\n  Loading base model  : {base_id}")
    dtype = torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=dtype,
        device_map="cpu",  # CPU merge — avoids GPU memory limits
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_id, token=hf_token)

    print(f"  Loading adapter     : {HF_HUB_REPO}/{adapter_subfolder}")
    peft_model = PeftModel.from_pretrained(
        base_model,
        HF_HUB_REPO,
        subfolder=adapter_subfolder,
        token=hf_token,
    )

    print("  Merging and unloading...")
    merged = peft_model.merge_and_unload()

    out_dir = tmp_root / f"{technique}_{run}_merged"
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
    from huggingface_hub import HfApi  # type: ignore

    model_key, lora_config, dataset_size = _parse_run(run)
    sub = hf_merged_subfolder(technique, model_key, lora_config, dataset_size)

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=HF_HUB_REPO, repo_type="model", exist_ok=True, private=True)

    print(f"  Pushing merged model: {HF_HUB_REPO}/{sub}")
    api.upload_folder(
        folder_path=str(merged_dir),
        repo_id=HF_HUB_REPO,
        path_in_repo=sub,
        commit_message=f"Add merged model {technique}/{run} [{version}]",
    )
    print("  Pushed successfully.")


def _create_version_tag(version: str, message: str, hf_token: str | None) -> None:
    from huggingface_hub import HfApi  # type: ignore

    api = HfApi(token=hf_token)
    print(f"\n  Creating tag '{version}' on {HF_HUB_REPO}…")
    api.create_tag(
        repo_id=HF_HUB_REPO,
        repo_type="model",
        tag=version,
        tag_message=message,
        token=hf_token,
        exist_ok=True,  # idempotent — re-running won't crash
    )
    print(f"  Tag '{version}' created.")
    print(f"  View at: https://huggingface.co/{HF_HUB_REPO}/tree/{version}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge adapters and release a version tag."
    )
    parser.add_argument("--version", required=True, help="Version tag, e.g. v1.0")
    parser.add_argument(
        "--technique",
        default=None,
        choices=["LoRA", "DoRA", "LoRAplus", "QLoRA", "AdaLoRA"],
        help="Technique folder prefix (required unless --tag-only).",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=[],
        metavar="MODEL_CFG_SIZE",
        help=(
            "Runs to merge, e.g.  qwen3-0.6b_C_1k llama3.2-1b_C_1k. "
            "Omit (or use --tag-only) to skip merging and just create the tag."
        ),
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
    args = parser.parse_args()

    hf_token: str | None = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("WARNING: HF_TOKEN not set. Pushes to private repos will fail.")

    tag_message = args.message or args.version

    if not args.tag_only:
        if not args.technique:
            parser.error(
                "--technique is required when merging (omit only with --tag-only)."
            )

        with tempfile.TemporaryDirectory(prefix="ic_merge_") as tmp:
            tmp_root = Path(tmp)
            for run in args.runs:
                print(f"\n── Merging {args.technique}/{run} ──────────────────")
                try:
                    merged_dir = _merge_one(args.technique, run, hf_token, tmp_root)
                    _push_merged(
                        merged_dir, args.technique, run, args.version, hf_token
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  ERROR merging {run}: {exc}")
                    print("  Skipping this run; continuing with others.")

    if not args.no_tag:
        _create_version_tag(args.version, tag_message, hf_token)
    else:
        print("\n--no-tag set; skipping tag creation.")

    print("\nDone.")


if __name__ == "__main__":
    main()
