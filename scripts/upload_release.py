#!/usr/bin/env python3
"""Upload model folders to the HuggingFace release repo using the HF Hub API.

No git required. Large files are uploaded directly; HF deduplicates on sha256 so
re-uploading the same binary a second time costs zero bandwidth.

Usage
-----
# Upload qwen3-0.6b first (12 GB), then delete local copy to free space:
python scripts/upload_release.py --model qwen3-0.6b

# After verifying the upload on HF, delete the local copy to free ~12 GB:
rm -rf ../intent-classifier-release/qwen3-0.6b

# Then upload llama3.2-1b (26 GB):
python scripts/upload_release.py --model llama3.2-1b

# Remove the old LoRA/DoRA/LoRA+ adapter folders from the remote:
python scripts/upload_release.py --delete-adapters
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "kon172verma/intent-classifier"

# Local clone of the HF release repo, relative to this script's parent directory.
RELEASE_DIR = Path(__file__).resolve().parent.parent.parent / "intent-classifier-release"


def _human_size(path: Path) -> str:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return f"{total / 1e9:.1f} GB"


def upload_model(model_key: str, api: HfApi) -> None:
    local_path = RELEASE_DIR / model_key
    if not local_path.is_dir():
        raise FileNotFoundError(f"Local folder not found: {local_path}")

    size = _human_size(local_path)
    print(f"Uploading {model_key}/ ({size}) → {REPO_ID}/{model_key}/")
    print("  This may take a while for large files …")

    api.upload_folder(
        folder_path=str(local_path),
        repo_id=REPO_ID,
        path_in_repo=model_key,
        repo_type="model",
        commit_message=f"Add {model_key}: safetensors, gguf, onnx [v1.0]",
    )
    print(f"✓  {model_key}/ uploaded successfully.")
    print(f"   You can now delete the local copy to reclaim {size}:")
    print(f"   rm -rf {local_path}")


def delete_adapter_folders(api: HfApi) -> None:
    """Delete the legacy LoRA/DoRA/LoRA+ adapter folders from the remote."""
    old_folders = ["DoRA", "LoRA", "LoRA+"]
    for folder in old_folders:
        print(f"Deleting {folder}/ from remote …")
        try:
            api.delete_folder(
                path_in_repo=folder,
                repo_id=REPO_ID,
                repo_type="model",
                commit_message=f"Remove legacy {folder}/ adapters (migrated to intent-classifier-experiments)",
            )
            print(f"✓  {folder}/ deleted.")
        except Exception as exc:
            print(f"  Warning: could not delete {folder}/: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload release models to HuggingFace.")
    parser.add_argument(
        "--model",
        choices=["qwen3-0.6b", "llama3.2-1b"],
        help="Which model folder to upload.",
    )
    parser.add_argument(
        "--delete-adapters",
        action="store_true",
        help="Delete legacy LoRA/DoRA/LoRA+ folders from the remote repo.",
    )
    args = parser.parse_args()

    if not args.model and not args.delete_adapters:
        parser.error("Specify --model and/or --delete-adapters.")

    api = HfApi()

    if args.model:
        upload_model(args.model, api)

    if args.delete_adapters:
        delete_adapter_folders(api)


if __name__ == "__main__":
    main()
