#!/usr/bin/env python3
"""
finetune_AdaLoRA/src/adalora_train.py
======================================
AdaLoRA fine-tuning for MCP tool-selection (intent classification).

AdaLoRA vs LoRA
---------------
AdaLoRA decomposes the weight update as P·Λ·Q (SVD form) where Λ is a diagonal
matrix of singular values.  During training it starts with `init_r` singular
values per layer and prunes the least-important ones every `deltaT` steps until
reaching `target_r`.  Layers that matter more for the task keep higher rank;
unimportant layers are pruned aggressively.

Key difference from LoRA training: `AdaLoraModel.update_and_allocate(step)` must
be called after every optimizer step.  This is handled by `AdaLoRAUpdateCallback`.

Adapter locations
-----------------
  Local   → finetune_AdaLoRA/adapters/{model}_{config}_{size}/
  HF Hub  → kon172verma/intent-classifier/AdaLoRA/{model}_{config}_{size}/

Usage
-----
    python adalora_train.py --model qwen2.5-0.5b --adalora-config B --dataset-size 1k
    python adalora_train.py --model smollm2-360m --adalora-config A --smoke-test
    python adalora_train.py --model llama3.2-1b  --adalora-config C --no-push
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
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from peft import AdaLoraConfig, TaskType, get_peft_model  # type: ignore  # noqa: E402
from trl import SFTTrainer  # type: ignore  # noqa: E402
from datasets import Dataset  # type: ignore  # noqa: E402
from huggingface_hub import HfApi  # type: ignore  # noqa: E402

from finetune_lib import (  # noqa: E402
    FINETUNE_MODEL_REGISTRY,
    ADALORA_CONFIGS,
    HF_HUB_REPO,
    hf_adapter_subfolder,
    tokenize_with_labels,
    load_jsonl,
    TrainValAccuracyCallback,
    compute_initial_train_loss,
    resolve_device,
    peak_memory_mb,
)

ADALORA_DIR = Path(__file__).parent.parent
DEFAULT_DATA_DIR = ADALORA_DIR / "data"
DEFAULT_ADAPTER_DIR = ADALORA_DIR / "adapters"
DEFAULT_REPORT_DIR = ADALORA_DIR / "reports_training"
_TECHNIQUE = "AdaLoRA"


# ── AdaLoRA rank-update callback ─────────────────────────────────────────────


class AdaLoRAUpdateCallback(TrainerCallback):
    """Calls model.update_and_allocate(step) after every optimizer step.

    This is required for AdaLoRA's rank pruning scheduler to run.
    Without it, no pruning occurs and the model stays at init_r throughout.
    """

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        if model is not None and hasattr(model, "update_and_allocate"):
            _m: Any = model
            _m.update_and_allocate(state.global_step)


# ── Helpers ───────────────────────────────────────────────────────────────────


def count_trainable(model: Any) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ── Argument parsing ──────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AdaLoRA fine-tuning for intent classification."
    )
    p.add_argument(
        "--model",
        choices=list(FINETUNE_MODEL_REGISTRY.keys()),
        default="qwen2.5-0.5b",
    )
    p.add_argument(
        "--adalora-config",
        choices=list(ADALORA_CONFIGS.keys()),
        default="B",
        dest="adalora_config",
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
        help="Root dir for locally saved adapters (default: finetune_AdaLoRA/adapters/).",
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


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    ada_cfg = ADALORA_CONFIGS[args.adalora_config]
    model_id = FINETUNE_MODEL_REGISTRY[args.model]
    run_tag = f"{args.model}_{args.adalora_config}_{args.dataset_size}"

    data_dir = (args.data_dir or DEFAULT_DATA_DIR) / args.dataset_size
    adapter_dir = (args.adapter_dir or DEFAULT_ADAPTER_DIR) / run_tag
    tmp_dir = ADALORA_DIR / "tmp" / run_tag
    args.report_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  AdaLoRA Training — {run_tag}")
    print(f"  Model        : {model_id}")
    print(f"  AdaLoRA cfg  : {args.adalora_config} — {ada_cfg['description']}")
    print(f"  Dataset      : {args.dataset_size}")
    print(f"  Device       : {device}")
    print(f"  Adapter dest : {adapter_dir}")
    print(f"  HF repo      : {HF_HUB_REPO}/AdaLoRA/{run_tag}")
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
    tokenizer.padding_side = "right"

    # ── Load base model ───────────────────────────────────────────────────────
    print(f"  Loading model:     {model_id}")
    dtype = torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=False,
    )
    model.enable_input_require_grads()
    model.config.use_cache = False

    # ── Step counts (needed for AdaLoraConfig.total_step) ────────────────────
    eff_batch = (
        ada_cfg["per_device_train_batch_size"] * ada_cfg["gradient_accumulation_steps"]
    )
    steps_per_epoch = max(1, len(train_examples) // eff_batch)
    total_steps = (
        10 if args.smoke_test else steps_per_epoch * ada_cfg["num_train_epochs"]
    )
    eval_steps = 10 if args.smoke_test else max(50, steps_per_epoch // 2)

    print(f"\n  Effective batch  : {eff_batch}")
    print(f"  Steps / epoch    : {steps_per_epoch}")
    print(
        f"  Total steps      : {total_steps if not args.smoke_test else '10 (smoke)'}"
    )
    print(f"  Eval every       : {eval_steps} steps")

    # ── Apply AdaLoRA ─────────────────────────────────────────────────────────
    # total_step must equal actual training steps for the rank scheduler.
    adalora_peft_config = AdaLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=ada_cfg["target_modules"],
        init_r=ada_cfg["init_r"],
        target_r=ada_cfg["target_r"],
        beta1=ada_cfg["beta1"],
        beta2=ada_cfg["beta2"],
        orth_reg_weight=ada_cfg["orth_reg_weight"],
        deltaT=ada_cfg["deltaT"],
        total_step=total_steps,
        inference_mode=False,
    )
    model = cast(Any, get_peft_model(model, adalora_peft_config))

    trainable, total = count_trainable(model)
    print(
        f"\n  Trainable params (init) : {trainable:,}  ({trainable / total * 100:.3f}%)"
    )
    print(f"  Total params            : {total:,}")
    print(f"  init_r → target_r       : {ada_cfg['init_r']} → {ada_cfg['target_r']}")

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

    # ── Training arguments ────────────────────────────────────────────────────
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = device.type == "cuda" and not use_bf16

    training_args = TrainingArguments(
        output_dir=str(tmp_dir),
        num_train_epochs=ada_cfg["num_train_epochs"] if not args.smoke_test else 1,
        max_steps=10 if args.smoke_test else -1,
        per_device_train_batch_size=ada_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=ada_cfg["gradient_accumulation_steps"],
        learning_rate=ada_cfg["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="no",
        report_to="none",
        dataloader_pin_memory=(device.type == "cuda"),
        # AdaLoRA is incompatible with gradient checkpointing — the rank-update
        # callback requires a full forward/backward pass at each step.
        gradient_checkpointing=False,
        optim="adamw_torch",
    )

    # ── Data collator ─────────────────────────────────────────────────────────
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    # AdaLoRAUpdateCallback must run before accuracy measurement so the rank
    # allocation is up-to-date when we sample train/val examples.
    adalora_update_cb = AdaLoRAUpdateCallback()
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
        callbacks=[adalora_update_cb, accuracy_cb],
    )

    # ── Step-0 baseline (pre-fine-tuning) ─────────────────────────────────────
    print("\n  Computing step-0 baseline (pre-fine-tuning)...")
    initial_train_loss = compute_initial_train_loss(
        model, train_dataset, collator, device
    )
    trainer.evaluate()
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

    trainer.model.config.use_cache = True

    # ── Save final adapter locally ────────────────────────────────────────────
    hf_sub = ""
    if not args.smoke_test:
        print(f"\n  Saving adapter locally → {adapter_dir}")
        trainer.model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        hf_sub = hf_adapter_subfolder(
            _TECHNIQUE, args.model, args.adalora_config, args.dataset_size
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
                    commit_message=f"Add AdaLoRA adapter: {run_tag}",
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
    trained_evals = [e for e in eval_entries if e.get("step", 0) > 0]
    last_eval = (
        trained_evals[-1]
        if trained_evals
        else (eval_entries[-1] if eval_entries else {})
    )

    report = {
        "model_key": args.model,
        "model_id": model_id,
        "adalora_config": args.adalora_config,
        "adalora_config_desc": ada_cfg["description"],
        "init_r": ada_cfg["init_r"],
        "target_r": ada_cfg["target_r"],
        "dataset_size": args.dataset_size,
        "technique": _TECHNIQUE,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "timestamp": ts,
        "trainable_params_init": trainable,
        "total_params": total,
        "trainable_pct_init": round(trainable / total * 100, 4),
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
        "hf_repo": HF_HUB_REPO if not args.smoke_test else None,
        "hf_subfolder": hf_sub if not args.smoke_test else None,
        "log_history": trainer.state.log_history,
    }

    # Use "lora_config" key for compatibility with shared plot_lib loaders
    report["lora_config"] = args.adalora_config

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


if __name__ == "__main__":
    main()
