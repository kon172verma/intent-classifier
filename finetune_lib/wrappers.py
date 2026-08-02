"""
finetune_lib/wrappers.py
========================
Shared wrapper helpers for technique-specific CLI scripts.

These helpers keep finetune_<technique>/src entrypoints short and readable while
reusing the canonical LoRA implementations where applicable.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, cast

from .plot_lib import plot_combined_accuracy_memory, plot_training_curves


def repo_root_from_script(script_file: str) -> Path:
    """Return repo root for a script at finetune_<technique>/src/*.py."""
    return Path(script_file).resolve().parent.parent.parent


def _ensure_paths_for_lora_reuse(script_file: str) -> Path:
    """Ensure repo root and finetune_LoRA/src are importable for wrapper scripts."""
    repo_root = repo_root_from_script(script_file)
    lora_src = repo_root / "finetune_LoRA" / "src"
    for p in (repo_root, lora_src):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)
    return repo_root


def run_prepare_wrapper(script_file: str, *, output_dir_name: str = "data") -> None:
    """Run prepare_lora_data.main() with DEFAULT_OUT_DIR redirected for this technique."""
    _ensure_paths_for_lora_reuse(script_file)
    prep = importlib.import_module("prepare_lora_data")
    prep_any = cast(Any, prep)
    prep_any.DEFAULT_OUT_DIR = Path(script_file).resolve().parent.parent / output_dir_name
    prep_any.main()


def run_train_wrapper(
    script_file: str,
    *,
    technique: str,
    use_dora: bool,
    quantize_4bit: bool = False,
    model_registry: dict[str, str] | None = None,
    loraplus_configs: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Call lora_train.train_main with per-technique options."""
    _ensure_paths_for_lora_reuse(script_file)
    lora_train = importlib.import_module("lora_train")
    train_main = lora_train.train_main

    kwargs: dict[str, Any] = {
        "technique": technique,
        "use_dora": use_dora,
        "base_dir": Path(script_file).resolve().parent.parent,
        "quantize_4bit": quantize_4bit,
    }

    if model_registry is not None:
        kwargs["model_registry"] = model_registry

    if loraplus_configs is not None:
        parse_args = lora_train.parse_args
        parsed = parse_args(model_registry=model_registry)
        kwargs["loraplus_lr_ratio"] = loraplus_configs[parsed.lora_config]["loraplus_lr_ratio"]

    train_main(**kwargs)


def run_validate_wrapper(
    script_file: str,
    *,
    technique: str,
    quantize_4bit: bool = False,
    model_registry: dict[str, str] | None = None,
) -> None:
    """Call lora_validate.validate_main with per-technique options."""
    _ensure_paths_for_lora_reuse(script_file)
    lora_validate = importlib.import_module("lora_validate")
    kwargs: dict[str, Any] = {
        "technique": technique,
        "base_dir": Path(script_file).resolve().parent.parent,
        "quantize_4bit": quantize_4bit,
    }
    if model_registry is not None:
        kwargs["model_registry"] = model_registry
    lora_validate.validate_main(**kwargs)


def run_experiments_wrapper(
    script_file: str,
    *,
    technique: str,
    train_script_name: str,
    eval_script_name: str,
    model_registry: dict[str, str] | None = None,
    default_models: list[str] | None = None,
) -> None:
    """Call run_lora_experiments.run_experiments_main with script and model overrides."""
    _ensure_paths_for_lora_reuse(script_file)
    runner = importlib.import_module("run_lora_experiments")
    src_dir = Path(script_file).resolve().parent
    runner.run_experiments_main(
        technique=technique,
        train_script=src_dir / train_script_name,
        eval_script=src_dir / eval_script_name,
        model_registry=model_registry,
        default_models=default_models,
    )


def parse_plot_args(
    *,
    description: str,
    train_reports_dir: Path,
    val_reports_dir: Path,
    test_reports_dir: Path,
    out_dir: Path,
) -> argparse.Namespace:
    """Standard argument parser for all finetune plot entrypoints."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--train-reports-dir", type=Path, default=train_reports_dir)
    p.add_argument("--val-reports-dir", type=Path, default=val_reports_dir)
    p.add_argument("--test-reports-dir", type=Path, default=test_reports_dir)
    p.add_argument("--out-dir", type=Path, default=out_dir)
    return p.parse_args()


def run_plot_pipeline(
    *,
    args: argparse.Namespace,
    technique: str,
    all_models: list[str],
    all_configs: list[str],
) -> None:
    """Run the standard 4-step plotting pipeline used by all techniques."""
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Generating {technique} plots -> {args.out_dir}")
    print(f"  Training reports   : {args.train_reports_dir}")
    print(f"  Validation reports : {args.val_reports_dir}")
    print(f"  Test reports       : {args.test_reports_dir}")
    print()

    print(f"  [1/4] Training curves ({len(all_models)} models x {len(all_configs)} configs)...")
    plot_training_curves(
        train_reports_dir=args.train_reports_dir,
        all_models=all_models,
        all_configs=all_configs,
        out_dir=args.out_dir,
        technique=technique,
    )

    print("  [2/4] Combined chart - train split...")
    plot_combined_accuracy_memory(
        reports_dir=args.train_reports_dir,
        all_models=all_models,
        all_configs=all_configs,
        out_dir=args.out_dir,
        split="train",
        technique=technique,
    )

    print("  [3/4] Combined chart - val split...")
    plot_combined_accuracy_memory(
        reports_dir=args.val_reports_dir,
        all_models=all_models,
        all_configs=all_configs,
        out_dir=args.out_dir,
        split="val",
        technique=technique,
    )

    print("  [4/4] Combined chart - test split (skips if no reports)...")
    plot_combined_accuracy_memory(
        reports_dir=args.test_reports_dir,
        all_models=all_models,
        all_configs=all_configs,
        out_dir=args.out_dir,
        split="test",
        technique=technique,
    )

    print(f"\n  Done. All plots saved to {args.out_dir}")
