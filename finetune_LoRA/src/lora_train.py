#!/usr/bin/env python3
"""
finetune_LoRA/src/lora_train.py
================================
LoRA fine-tuning for MCP tool-selection (intent classification).

Key design choices
------------------
* Shared configuration, prompt building, and callbacks live in finetune_lib/
  so QLoRA and AdaLoRA experiments use identical hyperparameters and utilities.
* Gradient checkpointing is always enabled (saves ~30-40% VRAM with minimal
  throughput cost; `use_reentrant=False` avoids the deprecation warning).
* No intermediate checkpoints are saved.  After training completes the final
  adapter is:
    1. Saved locally  → finetune_LoRA/adapters/{model}_{config}_{size}/
    2. Pushed to HF   → {HF_HUB_REPO}/LoRA/{model}_{config}_{size}/
  Loading for inference / merge_and_unload:
    model = PeftModel.from_pretrained(base, HF_HUB_REPO,
                subfolder="LoRA/qwen2.5-0.5b_B_1k")
    merged = model.merge_and_unload()
* The training report includes step-0 baseline metrics (train_loss, val_loss,
  train_accuracy, val_accuracy before any gradient update) so training-curve
  plots clearly show the pre-fine-tuning starting point.

Usage
-----
    # Recommended main run
    python lora_train.py --model qwen2.5-0.5b --lora-config B --dataset-size 1k

    # Smoke-test (10 steps only — validates the whole pipeline quickly)
    python lora_train.py --model smollm2-360m --lora-config A --smoke-test

    # Skip HF push (e.g. no internet on the target machine)
    python lora_train.py --model llama3.2-1b --lora-config C --no-push
"""

import argparse
import gc
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, cast

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_env_file = _REPO_ROOT / ".env"
if _env_file.exists():
    from dotenv import load_dotenv

    load_dotenv(_env_file)

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", message=".*max_new_tokens.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model  # type: ignore  # noqa: E402
from trl import SFTTrainer  # type: ignore  # noqa: E402
from datasets import Dataset  # type: ignore  # noqa: E402
from huggingface_hub import HfApi  # type: ignore  # noqa: E402

from finetune_lib import (  # noqa: E402
    FINETUNE_MODEL_REGISTRY,
    LORA_CONFIGS,
    HF_HUB_REPO,
    hf_adapter_subfolder,
    tokenize_with_labels,
    load_jsonl,
    TrainValAccuracyCallback,
    compute_initial_train_loss,
    resolve_device,
    peak_memory_mb,
)

LORA_DIR = Path(__file__).parent.parent
DEFAULT_DATA_DIR = LORA_DIR / "data"
DEFAULT_ADAPTER_DIR = LORA_DIR / "adapters"
DEFAULT_REPORT_DIR = LORA_DIR / "reports_training"
_TECHNIQUE = "LoRA"


# ── Helpers ───────────────────────────────────────────────────────────────────


def count_trainable(model: Any) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ── Argument parsing ──────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoRA fine-tuning for intent classification."
    )
    p.add_argument(
        "--model",
        choices=list(FINETUNE_MODEL_REGISTRY.keys()),
        default="qwen2.5-0.5b",
    )
    p.add_argument(
        "--lora-config",
        choices=list(LORA_CONFIGS.keys()),
        default="B",
        dest="lora_config",
    )
    p.add_argument(
        "--dataset-size",
        choices=["1k", "10k"],
        default="1k",
        dest="dataset_size",
    )
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help="Root dir for locally saved adapters (default: finetune_LoRA/adapters/).",
    )
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 10 training steps only — validates the full pipeline without committing.",
    )
    p.add_argument(
        "--no-push",
        action="store_true",
        help="Skip pushing the adapter to HuggingFace Hub.",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def train_main(
    technique: str = "LoRA",
    use_dora: bool = False,
    base_dir: Path | None = None,
) -> None:
    if base_dir is None:
        base_dir = Path(__file__).parent.parent  # finetune_LoRA/

    args = parse_args()

    # Override directory defaults when called from a different technique's script
    # (parse_args bakes in LORA_DIR-based defaults at import time)
    if args.report_dir == DEFAULT_REPORT_DIR:
        args.report_dir = base_dir / "reports_training"

    device = resolve_device(args.device)
    lora_cfg = LORA_CONFIGS[args.lora_config]
    model_id = FINETUNE_MODEL_REGISTRY[args.model]
    run_tag = f"{args.model}_{args.lora_config}_{args.dataset_size}"

    data_dir = (args.data_dir or base_dir / "data") / args.dataset_size
    adapter_dir = (args.adapter_dir or base_dir / "adapters") / run_tag
    # TrainingArguments requires an output_dir even when save_strategy="no".
    tmp_dir = base_dir / "tmp" / run_tag
    args.report_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  {technique} Training — {run_tag}")
    print(f"  Model        : {model_id}")
    print(f"  LoRA config  : {args.lora_config} — {lora_cfg['description']}")
    print(f"  Dataset      : {args.dataset_size}")
    print(f"  Device       : {device}")
    print(f"  Adapter dest : {adapter_dir}")
    print(f"  HF repo      : {HF_HUB_REPO}/{technique}/{run_tag}")
    if args.smoke_test:
        print("  Mode         : SMOKE TEST (10 steps only)")
    print(f"{'=' * 60}")

    # ── Load data ─────────────────────────────────────────────────────────────
    train_examples = load_jsonl(data_dir / "train.jsonl")
    val_examples = load_jsonl(data_dir / "val.jsonl")
    print(f"\n  Train : {len(train_examples)} examples")
    print(f"  Val   : {len(val_examples)} examples")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    print(f"\n  Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # required for causal-LM training

    # ── Load base model ───────────────────────────────────────────────────────
    print(f"  Loading model:     {model_id}")
    dtype = torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=False,
    )
    model.enable_input_require_grads()  # gradient flow through frozen base layers
    model.config.use_cache = False  # incompatible with gradient checkpointing

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    lora_peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg["target_modules"],
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias="none",
        use_dora=use_dora,
        inference_mode=False,
    )
    model = cast(Any, get_peft_model(model, lora_peft_config))

    trainable, total = count_trainable(model)
    print(f"\n  Trainable params : {trainable:,}  ({trainable / total * 100:.3f}%)")
    print(f"  Total params     : {total:,}")

    # ── Tokenise datasets ─────────────────────────────────────────────────────
    print("\n  Tokenizing datasets...")
    train_records = [
        tokenize_with_labels(ex, tokenizer, args.model) for ex in train_examples
    ]
    val_records = [
        tokenize_with_labels(ex, tokenizer, args.model) for ex in val_examples
    ]
    train_dataset = Dataset.from_list(train_records)
    val_dataset = Dataset.from_list(val_records)

    # ── Step counts ───────────────────────────────────────────────────────────
    eff_batch = (
        lora_cfg["per_device_train_batch_size"]
        * lora_cfg["gradient_accumulation_steps"]
    )
    steps_per_epoch = max(1, len(train_examples) // eff_batch)
    eval_steps = 10 if args.smoke_test else max(50, steps_per_epoch // 2)
    total_steps = steps_per_epoch * lora_cfg["num_train_epochs"]

    print(f"\n  Effective batch  : {eff_batch}")
    print(f"  Steps / epoch    : {steps_per_epoch}")
    print(
        f"  Total steps      : {total_steps if not args.smoke_test else '10 (smoke)'}"
    )
    print(f"  Eval every       : {eval_steps} steps")

    # ── Training arguments ────────────────────────────────────────────────────
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = device.type == "cuda" and not use_bf16

    training_args = TrainingArguments(
        output_dir=str(tmp_dir),
        num_train_epochs=lora_cfg["num_train_epochs"] if not args.smoke_test else 1,
        max_steps=10 if args.smoke_test else -1,
        per_device_train_batch_size=lora_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=lora_cfg["gradient_accumulation_steps"],
        learning_rate=lora_cfg["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        # No intermediate checkpoints — only the final adapter is saved.
        save_strategy="no",
        report_to="none",
        dataloader_pin_memory=(device.type == "cuda"),
        # Gradient checkpointing: ~30-40% VRAM saving; use_reentrant=False
        # avoids the PyTorch deprecation warning.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
    )

    # ── Data collator ─────────────────────────────────────────────────────────
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )

    # ── Accuracy callback (train + val at every eval checkpoint) ──────────────
    accuracy_cb = TrainValAccuracyCallback(
        train_examples=train_examples,
        val_examples=val_examples,
        tokenizer=tokenizer,
        model_key=args.model,
        device=device,
    )

    # ── Build trainer ─────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[accuracy_cb],
    )

    # ── Step-0 baseline (pre-fine-tuning) ─────────────────────────────────────
    # Calling trainer.evaluate() before trainer.train() captures all four
    # signals at step=0 in log_history: eval_loss, train_accuracy, eval_accuracy.
    # We also compute initial train_loss via a single forward pass.
    print("\n  Computing step-0 baseline (pre-fine-tuning)...")
    initial_train_loss = compute_initial_train_loss(
        model, train_dataset, collator, device
    )
    trainer.evaluate()
    # Patch the step-0 eval log entry with the initial train_loss.
    for entry in trainer.state.log_history:
        if "eval_loss" in entry and entry.get("step", -1) == 0:
            entry["loss"] = round(initial_train_loss, 6)
            break
    step0_eval: dict[str, Any] = cast(
        dict[str, Any],
        next((e for e in trainer.state.log_history if "eval_loss" in e), {}),
    )
    print(
        f"  Step 0 — train_loss={initial_train_loss:.4f}"
        f"  val_loss={step0_eval.get('eval_loss', 'n/a')}"
        f"  train_acc={step0_eval.get('train_accuracy', 'n/a')}"
        f"  val_acc={step0_eval.get('eval_accuracy', 'n/a')}"
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(
        f"\n  Starting training{' (smoke-test: 10 steps)' if args.smoke_test else ''}...\n"
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t_start = time.time()
    train_result = trainer.train()
    t_elapsed = time.time() - t_start
    mem_mb = peak_memory_mb(device)

    print(f"\n  Training complete in {t_elapsed:.1f}s  |  Peak VRAM: {mem_mb:.0f} MB")

    trainer.model.config.use_cache = True  # restore for subsequent inference

    # ── Save final adapter locally ────────────────────────────────────────────
    hf_sub = ""
    if not args.smoke_test:
        print(f"\n  Saving adapter locally → {adapter_dir}")
        trainer.model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        # ── Push to HuggingFace Hub ───────────────────────────────────────────
        hf_sub = hf_adapter_subfolder(
            technique, args.model, args.lora_config, args.dataset_size
        )
        if not args.no_push:
            hf_token = os.environ.get("HF_TOKEN")
            try:
                print(f"  Pushing adapter to HF → {HF_HUB_REPO}/{hf_sub}")
                api = HfApi(token=hf_token)
                api.create_repo(
                    repo_id=HF_HUB_REPO, repo_type="model", exist_ok=True, private=True
                )
                api.upload_folder(
                    folder_path=str(adapter_dir),
                    repo_id=HF_HUB_REPO,
                    path_in_repo=hf_sub,
                    commit_message=f"Add {technique} adapter: {run_tag}",
                )
                print("  Adapter pushed successfully.")
            except Exception as e:
                print(f"  WARNING: HF push failed: {e}")
                print(f"           Adapter saved locally at {adapter_dir}")
        else:
            print("  Skipping HF push (--no-push)")

    # ── Build training report ─────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_train_loss = train_result.training_loss
    eval_entries = [e for e in trainer.state.log_history if "eval_loss" in e]
    # Exclude step-0 (baseline) when reporting "final" end-of-training metrics.
    trained_evals = [e for e in eval_entries if e.get("step", 0) > 0]
    last_eval = (
        trained_evals[-1]
        if trained_evals
        else (eval_entries[-1] if eval_entries else {})
    )

    report = {
        "model_key": args.model,
        "model_id": model_id,
        "lora_config": args.lora_config,
        "lora_config_desc": lora_cfg["description"],
        "dataset_size": args.dataset_size,
        "technique": technique,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "timestamp": ts,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(trainable / total * 100, 4),
        "steps_per_epoch": steps_per_epoch,
        "total_steps_trained": trainer.state.global_step,
        "peak_memory_mb": round(mem_mb, 1),
        "total_training_time_s": round(t_elapsed, 1),
        "final_train_loss": round(final_train_loss, 6),
        "final_eval_loss": round(last_eval.get("eval_loss", 0), 6)
        if "eval_loss" in last_eval
        else None,
        "final_val_accuracy": round(last_eval.get("eval_accuracy", 0), 4)
        if "eval_accuracy" in last_eval
        else None,
        "final_train_accuracy": round(last_eval.get("train_accuracy", 0), 4)
        if "train_accuracy" in last_eval
        else None,
        # HF adapter location (for loading / merge_and_unload)
        "hf_repo": HF_HUB_REPO if not args.smoke_test else None,
        "hf_subfolder": hf_sub if not args.smoke_test else None,
        # Complete step-by-step history including step-0 baseline.
        # train_loss every logging_steps; eval entries (eval_loss, train_accuracy,
        # eval_accuracy) at each eval checkpoint + step 0.
        "log_history": trainer.state.log_history,
    }

    report_path = args.report_dir / f"{run_tag}_{ts}.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE — {run_tag}")
    print(f"  Final train loss : {final_train_loss:.4f}")
    if "eval_loss" in last_eval:
        print(f"  Final val loss   : {last_eval['eval_loss']:.4f}")
    if "eval_accuracy" in last_eval:
        print(f"  Final val acc    : {last_eval['eval_accuracy']:.4f}")
    if "train_accuracy" in last_eval:
        print(f"  Final train acc  : {last_eval['train_accuracy']:.4f}")
    print(f"  Training time    : {t_elapsed:.1f}s")
    print(f"  Peak VRAM        : {mem_mb:.0f} MB")
    print(f"  Training report  : {report_path}")
    if not args.smoke_test:
        print(f"  Local adapter    : {adapter_dir}")
        if not args.no_push:
            print(f"  HF adapter       : {HF_HUB_REPO}/{hf_sub}")
    print(f"{'=' * 60}\n")

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    train_main("LoRA", False, Path(__file__).parent.parent)


if __name__ == "__main__":
    main()
