#!/usr/bin/env python3
"""
scripts/migrate_adapters_to_experiments.py
============================================
One-time migration: move legacy adapters that currently live in
HF_RELEASE_REPO (kon172verma/intent-classifier) into HF_EXPERIMENTS_REPO
(kon172verma/intent-classifier-experiments), renamed to the new naming
convention, under the CURRENT_VERSION folder.

Background
----------
Before intent-classifier-experiments existed, every LoRA/DoRA/LoRA+ adapter
was pushed straight into the release repo under "{technique}/{run_tag}/".
That is the wrong home for in-progress experiments — the release repo should
only ever contain the 2 best merged models (+ GGUF/ONNX) per version.

What this script does
----------------------
1. Reads adapter folders from a local clone of the release repo
   (default: ../intent-classifier-release relative to this repo).
2. For each adapter folder "{technique_dir}/{model}_{config}_{size}/", uses
   `git log` on that path to recover the original commit date as the
   experiment timestamp (so migrated history stays chronologically accurate).
3. Renamed target path (per user convention):
     {version}/{model}_{technique}_{config}_{size}_{timestamp}/
4. Appends one EXPERIMENTS.jsonl entry per migrated adapter.
5. Optionally deletes the original folder from the release repo clone
   (only with --delete-from-release, and only after a successful copy/push).
6. After all deletions, removes empty technique parent dirs (LoRA/, DoRA/, etc.).

Modes
-----
  --local-experiments-dir DIR  Copy into a local git clone (default mode when
                                the flag is present). Does NOT call HF API.
  (no flag)                    Push to HF_EXPERIMENTS_REPO via HF API.

Safety
------
This script is DRY-RUN by default. Nothing is written or deleted until you
pass --execute (and separately --delete-from-release for the cleanup step).
Review the printed plan carefully before running with --execute.

Usage
-----
    # 1. See what would happen (no network calls, no writes)
    python scripts/migrate_adapters_to_experiments.py

    # 2. Copy into a local experiments repo clone (no HF push)
    python scripts/migrate_adapters_to_experiments.py --execute \\
        --local-experiments-dir ../intent-classifier-experiments

    # 3. Also delete the adapters from the release repo clone after copying
    python scripts/migrate_adapters_to_experiments.py --execute \\
        --local-experiments-dir ../intent-classifier-experiments \\
        --delete-from-release

    # 4. Push directly to HF_EXPERIMENTS_REPO (original behaviour)
    python scripts/migrate_adapters_to_experiments.py --execute
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from finetune_lib.config import CURRENT_VERSION, HF_EXPERIMENTS_REPO, hf_adapter_subfolder
from finetune_lib.registry import log_experiment

DEFAULT_SOURCE_DIR = _REPO_ROOT.parent / "intent-classifier-release"

# Local folder name (on disk in the release repo clone) -> canonical technique name.
TECHNIQUE_DIR_MAP: dict[str, str] = {
    "LoRA": "LoRA",
    "DoRA": "DoRA",
    "LoRA+": "LoRAplus",
    "DoRA+": "DoRAplus",
    "AdaLoRA": "AdaLoRA",
    "QLoRA": "QLoRA",
}


@dataclass
class AdapterMigration:
    technique: str
    model_key: str
    lora_config: str
    dataset_size: str
    timestamp: str
    source_path: Path
    hf_subfolder: str


def _git_commit_timestamp(repo_dir: Path, relative_path: str) -> str:
    """Return the first commit date that added `relative_path`, as YYYYMMDD-HHMMSS UTC."""
    result = subprocess.run(
        ["git", "log", "--diff-filter=A", "--follow", "--format=%aI", "--", relative_path],
        cwd=repo_dir,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    if not lines:
        # Fall back to "now" if history lookup fails (e.g. squashed history).
        return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    # Oldest entry is last in `git log` output (newest first) -> take the last line.
    iso_ts = lines[-1]
    dt = datetime.fromisoformat(iso_ts).astimezone(UTC)
    return dt.strftime("%Y%m%d-%H%M%S")


def _parse_run_dir(name: str) -> tuple[str, str, str] | None:
    """Parse "{model_key}_{lora_config}_{dataset_size}" folder name."""
    parts = name.rsplit("_", 2)
    if len(parts) != 3:
        return None
    model_key, lora_config, dataset_size = parts
    return model_key, lora_config, dataset_size


def discover_migrations(source_dir: Path, version: str) -> list[AdapterMigration]:
    migrations: list[AdapterMigration] = []
    for technique_dir, technique in TECHNIQUE_DIR_MAP.items():
        tdir = source_dir / technique_dir
        if not tdir.is_dir():
            continue
        for run_dir in sorted(tdir.iterdir()):
            if not run_dir.is_dir():
                continue
            parsed = _parse_run_dir(run_dir.name)
            if parsed is None:
                print(f"  Skipping unrecognised folder: {run_dir}")
                continue
            model_key, lora_config, dataset_size = parsed
            relative_path = f"{technique_dir}/{run_dir.name}"
            timestamp = _git_commit_timestamp(source_dir, relative_path)
            hf_sub = hf_adapter_subfolder(
                technique,
                model_key,
                lora_config,
                dataset_size,
                timestamp=timestamp,
                version=version,
            )
            migrations.append(
                AdapterMigration(
                    technique=technique,
                    model_key=model_key,
                    lora_config=lora_config,
                    dataset_size=dataset_size,
                    timestamp=timestamp,
                    source_path=run_dir,
                    hf_subfolder=hf_sub,
                )
            )
    return migrations


def copy_migration_local(m: AdapterMigration, experiments_dir: Path) -> None:
    """Copy adapter folder into a local experiments repo clone (no HF API call)."""
    dest = experiments_dir / m.hf_subfolder
    if dest.exists():
        raise FileExistsError(f"Destination already exists: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(str(m.source_path), str(dest))


def push_migration(m: AdapterMigration, hf_token: str | None) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=HF_EXPERIMENTS_REPO, repo_type="model", exist_ok=True, private=True)
    api.upload_folder(
        folder_path=str(m.source_path),
        repo_id=HF_EXPERIMENTS_REPO,
        path_in_repo=m.hf_subfolder,
        commit_message=f"Migrate {m.technique} adapter: {m.model_key}_{m.lora_config}_{m.dataset_size}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate legacy adapters from the release repo into the experiments repo."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--version", default=CURRENT_VERSION)
    parser.add_argument(
        "--local-experiments-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Copy adapters into this local experiments repo clone instead of "
            "pushing to HF_EXPERIMENTS_REPO. No HF API calls are made."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually copy/push adapters (default: dry-run, prints the plan only).",
    )
    parser.add_argument(
        "--delete-from-release",
        action="store_true",
        help=(
            "After a successful copy/push, delete the source folder from the local "
            "release repo clone (--source-dir). You still need to `git commit` "
            "and `git push` in that clone yourself. Requires --execute."
        ),
    )
    args = parser.parse_args()

    if not args.source_dir.is_dir():
        raise FileNotFoundError(
            f"Source dir not found: {args.source_dir}. Pass --source-dir explicitly."
        )

    use_local = args.local_experiments_dir is not None
    if use_local and args.execute and not args.local_experiments_dir.is_dir():
        raise FileNotFoundError(
            f"Local experiments dir not found: {args.local_experiments_dir}. Clone the repo first."
        )

    print(f"Scanning {args.source_dir} for legacy adapters...")
    migrations = discover_migrations(args.source_dir, args.version)

    if not migrations:
        print("No adapters found to migrate.")
        return

    target_label = str(args.local_experiments_dir) if use_local else HF_EXPERIMENTS_REPO
    print(f"\nFound {len(migrations)} adapter(s) to migrate:\n")
    for m in migrations:
        print(
            f"  {m.technique:10s} {m.model_key}_{m.lora_config}_{m.dataset_size:5s} "
            f"-> {target_label}/{m.hf_subfolder}"
        )

    if not args.execute:
        print("\nDRY RUN — nothing was copied or deleted. Re-run with --execute to proceed.")
        return

    hf_token: str | None = None
    if not use_local:
        import os

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("WARNING: HF_TOKEN not set. Pushes to private repos will fail.")

    failed: list[AdapterMigration] = []
    for m in migrations:
        label = f"{m.technique}/{m.model_key}_{m.lora_config}_{m.dataset_size}"
        print(f"\n-- Migrating {label} --")
        try:
            if use_local:
                copy_migration_local(m, args.local_experiments_dir)
                print(f"  Copied to {args.local_experiments_dir / m.hf_subfolder}")
            else:
                push_migration(m, hf_token)
                print("  Pushed successfully.")
            log_experiment(
                version=args.version,
                technique=m.technique,
                model_key=m.model_key,
                base_model_id="",
                lora_config=m.lora_config,
                dataset_size=m.dataset_size,
                timestamp=m.timestamp,
                run_tag=f"{m.model_key}_{m.lora_config}_{m.dataset_size}",
                hf_repo=HF_EXPERIMENTS_REPO,
                hf_subfolder=m.hf_subfolder,
            )
            if args.delete_from_release:
                print(f"  Deleting local copy: {m.source_path}")
                shutil.rmtree(m.source_path)
        except Exception as exc:
            print(f"  ERROR migrating {m.source_path}: {exc}")
            print("  Skipping delete for this adapter; continuing with others.")
            failed.append(m)

    # Remove empty technique parent dirs (LoRA/, DoRA/, LoRA+/, etc.)
    if args.delete_from_release:
        for technique_dir in TECHNIQUE_DIR_MAP:
            tdir = args.source_dir / technique_dir
            if tdir.is_dir():
                remaining = list(tdir.iterdir())
                if not remaining:
                    tdir.rmdir()
                    print(f"\nRemoved empty dir: {tdir}")
                else:
                    print(f"\nNOTE: {tdir} still has {len(remaining)} item(s) — not removed.")
        print(
            "\nLocal folders removed from the release repo clone. "
            "Review with `git status`, then `git add -A && git commit && git push` "
            f"in {args.source_dir} to finalize."
        )

    if failed:
        print(f"\n{len(failed)} migration(s) FAILED — see errors above.")
    else:
        print("\nDone. All adapters migrated successfully.")


if __name__ == "__main__":
    main()
