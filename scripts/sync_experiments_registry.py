#!/usr/bin/env python3
"""
scripts/sync_experiments_registry.py
====================================
Verify and optionally reconcile local EXPERIMENTS.jsonl against the
Hugging Face experiments repository directory structure.

Default mode is verification only.
Use --sync to reconcile the local file to the remote state.
Use --strict to fail on any drift.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

HF_SUBFOLDER_RE = re.compile(
    r"^(?P<version>[^/]+)/"
    r"(?P<model_key>.+?)_"
    r"(?P<technique>LoRAplus|DoRAplus|LoRA\+|DoRA\+|LoRA|DoRA|QLoRA|AdaLoRA)_"
    r"(?P<lora_config>[A-Z])_"
    r"(?P<dataset_size>1k|10k)_"
    r"(?P<timestamp>\d{8}-\d{6})$"
)

TECHNIQUE_CANONICAL: dict[str, str] = {
    "LoRAplus": "LoRA+",
    "DoRAplus": "DoRA+",
}


@dataclass(frozen=True)
class RegistryRow:
    version: str
    technique: str
    model_key: str
    lora_config: str
    dataset_size: str
    timestamp: str
    hf_repo: str
    hf_subfolder: str


def _read_hf_token(required: bool) -> str | None:
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    if required and not token:
        raise RuntimeError(
            "Missing HF token. Set HF_TOKEN (or HUGGINGFACE_TOKEN/HUGGINGFACEHUB_API_TOKEN)."
        )
    return token


def load_local_registry(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def infer_remote_subfolders(repo_id: str, token: str | None) -> tuple[set[str], set[str]]:
    """
    Return (valid_subfolders, malformed_subfolders).

    We infer adapter folders from the first two path components in repo files:
        version/subfolder/...

    Evaluation reports are stored separately under reports/... and are not
    adapter folders, so they are excluded from registry reconciliation.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type="model")

    candidate_subfolders: set[str] = set()
    for p in files:
        parts = p.split("/")
        if parts[0] == "reports":
            continue
        if len(parts) >= 3:
            candidate_subfolders.add(f"{parts[0]}/{parts[1]}")

    valid: set[str] = set()
    malformed: set[str] = set()
    for sub in sorted(candidate_subfolders):
        if HF_SUBFOLDER_RE.match(sub):
            valid.add(sub)
        else:
            malformed.add(sub)
    return valid, malformed


def row_from_subfolder(subfolder: str, repo_id: str, version_fallback: str) -> RegistryRow:
    m = HF_SUBFOLDER_RE.match(subfolder)
    if m is None:
        raise ValueError(f"Invalid experiments subfolder format: {subfolder}")

    model_key = m.group("model_key")
    technique = TECHNIQUE_CANONICAL.get(m.group("technique"), m.group("technique"))
    return RegistryRow(
        version=m.group("version") or version_fallback,
        technique=technique,
        model_key=model_key,
        lora_config=m.group("lora_config"),
        dataset_size=m.group("dataset_size"),
        timestamp=m.group("timestamp"),
        hf_repo=repo_id,
        hf_subfolder=subfolder,
    )


def append_missing_entries(
    path: Path,
    rows: list[RegistryRow],
    known_models: dict[str, str],
) -> None:
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            entry = {
                "version": row.version,
                "technique": row.technique,
                "model_key": row.model_key,
                "base_model_id": known_models.get(row.model_key, ""),
                "lora_config": row.lora_config,
                "dataset_size": row.dataset_size,
                "timestamp": row.timestamp,
                "run_tag": f"{row.model_key}_{row.lora_config}_{row.dataset_size}",
                "hf_repo": row.hf_repo,
                "hf_subfolder": row.hf_subfolder,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def write_registry(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    from finetune_lib.config import (
        CURRENT_VERSION,
        FINETUNE_MODEL_REGISTRY,
        HF_EXPERIMENTS_REPO,
    )
    from finetune_lib.registry import REGISTRY_PATH

    parser = argparse.ArgumentParser(
        description="Verify/sync EXPERIMENTS.jsonl against HF experiments repo folders."
    )
    parser.add_argument("--repo-id", default=HF_EXPERIMENTS_REPO)
    parser.add_argument("--registry-path", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--version-fallback", default=CURRENT_VERSION)
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Reconcile local file to remote state (add missing and remove stale entries).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when drift is found.",
    )
    args = parser.parse_args()

    token = _read_hf_token(required=True)

    local_rows = load_local_registry(args.registry_path)
    local_subfolders = [str(r.get("hf_subfolder", "")) for r in local_rows if r.get("hf_subfolder")]
    local_set = set(local_subfolders)

    duplicates = [k for k, c in Counter(local_subfolders).items() if c > 1]

    remote_valid, remote_malformed = infer_remote_subfolders(args.repo_id, token)

    missing_in_local = sorted(remote_valid - local_set)
    missing_on_remote = sorted(local_set - remote_valid)

    print("Registry verification summary")
    print(f"  local entries           : {len(local_rows)}")
    print(f"  local unique subfolders : {len(local_set)}")
    print(f"  remote valid subfolders : {len(remote_valid)}")
    print(f"  missing in local        : {len(missing_in_local)}")
    print(f"  missing on remote       : {len(missing_on_remote)}")
    print(f"  local duplicates        : {len(duplicates)}")
    print(f"  malformed remote paths  : {len(remote_malformed)}")

    if missing_in_local:
        print("\nMissing in local EXPERIMENTS.jsonl:")
        for s in missing_in_local:
            print(f"  - {s}")

    if missing_on_remote:
        print("\nMissing on remote repo:")
        for s in missing_on_remote:
            print(f"  - {s}")

    if duplicates:
        print("\nDuplicate local hf_subfolder entries:")
        for s in duplicates:
            print(f"  - {s}")

    if remote_malformed:
        print("\nRemote subfolders with unexpected naming:")
        for s in sorted(remote_malformed):
            print(f"  - {s}")

    if args.sync:
        existing_by_subfolder: dict[str, dict[str, Any]] = {}
        for row in local_rows:
            subfolder_raw = row.get("hf_subfolder")
            if not subfolder_raw:
                continue
            subfolder = str(subfolder_raw)
            if subfolder not in existing_by_subfolder:
                existing_by_subfolder[subfolder] = row

        reconciled_rows: list[dict[str, Any]] = []
        added_count = 0
        for subfolder in sorted(remote_valid):
            existing = existing_by_subfolder.get(subfolder)
            if existing is not None:
                reconciled_rows.append(existing)
                continue

            remote_row = row_from_subfolder(subfolder, args.repo_id, args.version_fallback)
            reconciled_rows.append(
                {
                    "version": remote_row.version,
                    "technique": remote_row.technique,
                    "model_key": remote_row.model_key,
                    "base_model_id": FINETUNE_MODEL_REGISTRY.get(remote_row.model_key, ""),
                    "lora_config": remote_row.lora_config,
                    "dataset_size": remote_row.dataset_size,
                    "timestamp": remote_row.timestamp,
                    "run_tag": (
                        f"{remote_row.model_key}_{remote_row.lora_config}_{remote_row.dataset_size}"
                    ),
                    "hf_repo": remote_row.hf_repo,
                    "hf_subfolder": remote_row.hf_subfolder,
                }
            )
            added_count += 1

        write_registry(args.registry_path, reconciled_rows)
        print(f"\nReconciled {args.registry_path}.")
        print(f"  Added from remote : {added_count}")
        print(f"  Removed stale     : {len(missing_on_remote)}")
        print(f"  Removed duplicates: {len(duplicates)}")
        print(f"  Final entries     : {len(reconciled_rows)}")

    has_drift = bool(missing_in_local or missing_on_remote or duplicates or remote_malformed)
    if args.strict and has_drift:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
