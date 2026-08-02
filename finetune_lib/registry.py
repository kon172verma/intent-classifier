"""
finetune_lib/registry.py
========================
Append-only local log of every adapter experiment pushed to HF_EXPERIMENTS_REPO.

The registry lives at EXPERIMENTS.jsonl (repo root) and is committed to git
alongside the code, so "which experiments have we saved on HF" is always
answerable without calling the HF API.

Each line is one JSON object:
    {
        "version": "v1.0",
        "technique": "LoRA",
        "model_key": "qwen3-0.6b",
        "base_model_id": "Qwen/Qwen3-0.6B",
        "lora_config": "C",
        "dataset_size": "1k",
        "timestamp": "20260101-120000",
        "run_tag": "qwen3-0.6b_C_1k",
        "hf_repo": "kon172verma/intent-classifier-experiments",
        "hf_subfolder": "v1.0/qwen3-0.6b_LoRA_C_1k_20260101-120000",
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
REGISTRY_PATH: Path = _REPO_ROOT / "EXPERIMENTS.jsonl"


def log_experiment(
    *,
    version: str,
    technique: str,
    model_key: str,
    base_model_id: str,
    lora_config: str,
    dataset_size: str,
    timestamp: str,
    run_tag: str,
    hf_repo: str,
    hf_subfolder: str,
) -> None:
    """Append one experiment record to EXPERIMENTS.jsonl."""
    entry: dict[str, Any] = {
        "version": version,
        "technique": technique,
        "model_key": model_key,
        "base_model_id": base_model_id,
        "lora_config": lora_config,
        "dataset_size": dataset_size,
        "timestamp": timestamp,
        "run_tag": run_tag,
        "hf_repo": hf_repo,
        "hf_subfolder": hf_subfolder,
    }
    with REGISTRY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_registry() -> list[dict[str, Any]]:
    """Return every logged experiment, oldest first."""
    if not REGISTRY_PATH.exists():
        return []
    return [
        json.loads(line)
        for line in REGISTRY_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def find_latest_experiment(
    *,
    technique: str,
    model_key: str,
    lora_config: str,
    dataset_size: str,
    version: str | None = None,
) -> dict[str, Any] | None:
    """Return the most recent registry entry matching the given run, or None."""
    matches = [
        e
        for e in load_registry()
        if e["technique"] == technique
        and e["model_key"] == model_key
        and e["lora_config"] == lora_config
        and e["dataset_size"] == dataset_size
        and (version is None or e["version"] == version)
    ]
    if not matches:
        return None
    return max(matches, key=lambda e: e["timestamp"])
