#!/usr/bin/env python3
"""
finetune_LoRA/src/lora_train.py
================================
LoRA fine-tuning for MCP tool-selection (intent classification).

Key design choices
------------------
* Shared configuration, prompt building, and callbacks live in finetune_lib/
  so DoRA, LoRA+, DoRA+, and QLoRA all share this exact training loop
  (technique differences are toggled via use_dora / loraplus_lr_ratio /
  quantize_4bit) and use identical hyperparameters and utilities.
* Gradient checkpointing is enabled for full-precision runs (saves ~30-40%
  VRAM with minimal throughput cost; `use_reentrant=False` avoids the
  deprecation warning). Disabled automatically when quantize_4bit=True
  (QLoRA) since BNB 4-bit + gradient checkpointing is known to NaN.
* Early stopping: eval/logging both run every `eval_steps` (aligned so
  train_loss and val_loss are always compared at the same checkpoint), and
  training stops once val_loss (eval_loss) fails to improve by at least
  EARLY_STOPPING_THRESHOLD for EARLY_STOPPING_PATIENCE consecutive evals
  (see finetune_lib/config.py). Checkpoints are written to a scratch tmp_dir
  every eval_steps (save_total_limit=2) purely so load_best_model_at_end can
  restore the best weights; the adapter actually shipped is:
    1. Saved locally  → finetune_LoRA/adapters/{model}_{config}_{size}/
    2. Pushed to HF   → {HF_EXPERIMENTS_REPO}/{version}/{model}_{technique}_{config}_{size}_{timestamp}/
  Loading for inference / merge_and_unload:
    model = PeftModel.from_pretrained(base, HF_EXPERIMENTS_REPO,
                subfolder="v1.0/qwen2.5-0.5b_LoRA_B_1k_20260101-120000")
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

import torch
from datasets import Dataset
from huggingface_hub import HfApi
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from peft.optimizers import create_loraplus_optimizer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    get_cosine_schedule_with_warmup,
)
from trl.trainer.sft_trainer import SFTTrainer

from finetune_lib import (
    CURRENT_VERSION,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_THRESHOLD,
    FINETUNE_MODEL_REGISTRY,
    HF_EXPERIMENTS_REPO,
    LORA_CONFIGS,
    NO_TOOL_ID,
    PROMPT_FORMAT_VERSION,
    TOOL_IDS,
    TrainValAccuracyCallback,
    build_training_arguments,
    compute_initial_train_loss,
    generate_experiment_timestamp,
    hf_adapter_subfolder,
    hf_report_path,
    load_jsonl,
    peak_memory_mb,
    resolve_device,
    tokenize_with_labels,
    upload_report_to_hf,
)
from finetune_lib.registry import log_experiment

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


def parse_args(model_registry: dict[str, str] | None = None) -> argparse.Namespace:
    model_registry = model_registry or FINETUNE_MODEL_REGISTRY
    default_model = (
        "qwen2.5-0.5b" if "qwen2.5-0.5b" in model_registry else next(iter(model_registry))
    )
    p = argparse.ArgumentParser(description="LoRA fine-tuning for intent classification.")
    p.add_argument(
        "--model",
        choices=list(model_registry.keys()),
        default=default_model,
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
    p.add_argument(
        "--ckpt-dir",
        type=Path,
        default=None,
        help="Root dir for Trainer checkpoints (default: finetune_<technique>/tmp/).",
    )
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    p.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing during training (enabled by default).",
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
    loraplus_lr_ratio: int | None = None,
    quantize_4bit: bool = False,
    model_registry: dict[str, str] | None = None,
) -> None:
    if base_dir is None:
        base_dir = Path(__file__).parent.parent  # finetune_LoRA/

    model_registry = model_registry or FINETUNE_MODEL_REGISTRY
    args = parse_args(model_registry=model_registry)

    # Override directory defaults when called from a different technique's script
    # (parse_args bakes in LORA_DIR-based defaults at import time)
    if args.report_dir == DEFAULT_REPORT_DIR:
        args.report_dir = base_dir / "reports_training"

    device = resolve_device(args.device)
    if quantize_4bit and device.type != "cuda":
        raise RuntimeError(
            f"{technique} requires CUDA for 4-bit NF4 quantization "
            "(bitsandbytes has no CPU/MPS 4-bit kernels) — "
            f"got device={device}. Re-run on a CUDA host."
        )
    lora_cfg = LORA_CONFIGS[args.lora_config]
    model_id = model_registry[args.model]
    run_tag = f"{args.model}_{args.lora_config}_{args.dataset_size}"

    data_dir = (args.data_dir or base_dir / "data") / args.dataset_size
    adapter_dir = (args.adapter_dir or base_dir / "adapters") / run_tag
    # TrainingArguments requires an output_dir even when save_strategy="no".
    tmp_dir = (args.ckpt_dir or base_dir / "tmp") / run_tag
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
    print(
        f"  HF repo      : {HF_EXPERIMENTS_REPO}/{CURRENT_VERSION}/"
        f"{args.model}_{technique}_{args.lora_config}_{args.dataset_size}_<timestamp>"
    )
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
    bnb_config: BitsAndBytesConfig | None = None
    if quantize_4bit:
        # NF4 (Normal Float 4-bit) is the quantization dtype from the QLoRA
        # paper — never fp4. Double-quant further compresses the quantization
        # constants; compute stays bfloat16 for numerically stable forward passes.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print("  Quantization:      NF4 (4-bit, double-quant, bf16 compute)")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=False,
    )
    use_gradient_checkpointing = args.gradient_checkpointing
    if quantize_4bit:
        # kbit-training prep enables gradient flow through frozen 4-bit layers
        # and configures checkpointing for QLoRA when requested.
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
    else:
        model.enable_input_require_grads()  # gradient flow through frozen base layers
    model.config.use_cache = not use_gradient_checkpointing

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
    print(
        "  Grad checkpoint  : "
        f"{'enabled' if use_gradient_checkpointing else 'disabled'}"
    )

    # ── Tokenise datasets ─────────────────────────────────────────────────────
    print("\n  Tokenizing datasets...")
    train_records = [tokenize_with_labels(ex, tokenizer, args.model) for ex in train_examples]
    val_records = [tokenize_with_labels(ex, tokenizer, args.model) for ex in val_examples]
    train_dataset = Dataset.from_list(train_records)
    val_dataset = Dataset.from_list(val_records)

    # ── Step counts ───────────────────────────────────────────────────────────
    eff_batch = lora_cfg["per_device_train_batch_size"] * lora_cfg["gradient_accumulation_steps"]
    steps_per_epoch = max(1, len(train_examples) // eff_batch)
    eval_steps = 10 if args.smoke_test else max(50, steps_per_epoch // 2)
    total_steps = steps_per_epoch * lora_cfg["num_train_epochs"]

    print(f"\n  Effective batch  : {eff_batch}")
    print(f"  Steps / epoch    : {steps_per_epoch}")
    print(f"  Total steps      : {total_steps if not args.smoke_test else '10 (smoke)'}")
    print(f"  Eval every       : {eval_steps} steps")

    # ── Training arguments ────────────────────────────────────────────────────
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = device.type == "cuda" and not use_bf16

    training_args = build_training_arguments(
        total_training_steps=10 if args.smoke_test else total_steps,
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
        # Train loss (logging) and val loss (eval) are measured at the same
        # cadence so patience is evaluated on aligned, comparable checkpoints.
        logging_steps=eval_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        # Checkpoint every eval — required so load_best_model_at_end can
        # restore the best (lowest val_loss) weights once early stopping
        # or training fires. save_total_limit keeps disk usage bounded to
        # the current + best checkpoint; these live in tmp_dir and are not
        # the adapter we ship (that's saved separately below).
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_pin_memory=(device.type == "cuda"),
        # Gradient checkpointing trades extra compute for lower activation memory.
        gradient_checkpointing=use_gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if use_gradient_checkpointing else None
        ),
        # For LoRA+/DoRA+ we inject a custom optimizer below; QLoRA uses a
        # paged optimizer to save VRAM; plain LoRA/DoRA keep adamw_torch.
        optim="paged_adamw_8bit" if quantize_4bit else "adamw_torch",
    )

    # ── LoRA+ asymmetric optimiser ────────────────────────────────────────────
    # When loraplus_lr_ratio is set, matrix B parameters use a higher LR than
    # matrix A parameters (Hayou et al., 2024).  We build the optimizer and a
    # matching cosine scheduler ourselves and pass them to SFTTrainer so that
    # the Trainer skips its own optimizer construction.
    loraplus_optimizers: tuple | None = None
    if loraplus_lr_ratio is not None:
        _actual_steps = (10 if args.smoke_test else total_steps) // lora_cfg[
            "gradient_accumulation_steps"
        ]
        _warmup_steps = max(1, int(_actual_steps * 0.05))
        _lp_optimizer = create_loraplus_optimizer(
            model=cast(PeftModel, model),
            optimizer_cls=torch.optim.AdamW,
            lr=lora_cfg["learning_rate"],
            loraplus_lr_ratio=loraplus_lr_ratio,
        )
        _lp_scheduler = get_cosine_schedule_with_warmup(
            _lp_optimizer,
            num_warmup_steps=_warmup_steps,
            num_training_steps=_actual_steps,
        )
        loraplus_optimizers = (_lp_optimizer, _lp_scheduler)
        print(
            f"  LoRA+ optimizer  : ratio={loraplus_lr_ratio}  "
            f"lr_A={lora_cfg['learning_rate']:.1e}  "
            f"lr_B={lora_cfg['learning_rate'] * loraplus_lr_ratio:.1e}"
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
        callbacks=[
            accuracy_cb,
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
            ),
        ],
        optimizers=loraplus_optimizers or (None, None),
    )

    # ── Step-0 baseline (pre-fine-tuning) ─────────────────────────────────────
    # Calling trainer.evaluate() before trainer.train() captures all four
    # signals at step=0 in log_history: eval_loss, train_accuracy, eval_accuracy.
    # We also compute initial train_loss via a single forward pass.
    print("\n  Computing step-0 baseline (pre-fine-tuning)...")
    initial_train_loss = compute_initial_train_loss(model, train_dataset, collator, device)
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
    print(f"\n  Starting training{' (smoke-test: 10 steps)' if args.smoke_test else ''}...\n")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t_start = time.time()
    train_result = trainer.train()
    t_elapsed = time.time() - t_start
    mem_mb = peak_memory_mb(device)

    print(f"\n  Training complete in {t_elapsed:.1f}s  |  Peak VRAM: {mem_mb:.0f} MB")

    trained_model = cast(Any, trainer.model)
    if trained_model is None:
        raise RuntimeError("Trainer model is unavailable after training")
    trained_model.config.use_cache = True  # restore for subsequent inference

    # ── Save final adapter locally ────────────────────────────────────────────
    # load_best_model_at_end=True already restored the best (lowest val_loss)
    # checkpoint into trainer.model before we get here, so this saves the
    # best weights, not necessarily the last-step weights.
    hf_sub = ""
    experiment_timestamp = ""
    adapter_pushed = False
    if not args.smoke_test:
        print(f"\n  Saving adapter locally → {adapter_dir}")
        trained_model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        # ── Push to HuggingFace Hub (experiments repo) ────────────────────────
        experiment_timestamp = generate_experiment_timestamp()
        hf_sub = hf_adapter_subfolder(
            technique,
            args.model,
            args.lora_config,
            args.dataset_size,
            timestamp=experiment_timestamp,
        )
        if not args.no_push:
            hf_token = os.environ.get("HF_TOKEN")
            try:
                print(f"  Pushing adapter to HF → {HF_EXPERIMENTS_REPO}/{hf_sub}")
                api = HfApi(token=hf_token)
                api.create_repo(
                    repo_id=HF_EXPERIMENTS_REPO,
                    repo_type="model",
                    exist_ok=True,
                    private=True,
                )
                api.upload_folder(
                    folder_path=str(adapter_dir),
                    repo_id=HF_EXPERIMENTS_REPO,
                    path_in_repo=hf_sub,
                    commit_message=f"Add {technique} adapter: {run_tag} [{experiment_timestamp}]",
                )
                print("  Adapter pushed successfully.")
                log_experiment(
                    version=CURRENT_VERSION,
                    technique=technique,
                    model_key=args.model,
                    base_model_id=model_id,
                    lora_config=args.lora_config,
                    dataset_size=args.dataset_size,
                    timestamp=experiment_timestamp,
                    run_tag=run_tag,
                    hf_repo=HF_EXPERIMENTS_REPO,
                    hf_subfolder=hf_sub,
                )
                adapter_pushed = True
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
    last_eval = trained_evals[-1] if trained_evals else (eval_entries[-1] if eval_entries else {})

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
        "prompt_format": PROMPT_FORMAT_VERSION,
        "tool_id_scheme": "".join(TOOL_IDS),
        "no_tool_id": NO_TOOL_ID,
        "gradient_checkpointing": use_gradient_checkpointing,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(trainable / total * 100, 4),
        "steps_per_epoch": steps_per_epoch,
        "total_steps_trained": trainer.state.global_step,
        "early_stopped": trainer.state.global_step < total_steps,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_threshold": EARLY_STOPPING_THRESHOLD,
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
        "hf_repo": HF_EXPERIMENTS_REPO if not args.smoke_test else None,
        "hf_subfolder": hf_sub if not args.smoke_test else None,
        "version": CURRENT_VERSION if not args.smoke_test else None,
        # Complete step-by-step history including step-0 baseline.
        # train_loss every logging_steps; eval entries (eval_loss, train_accuracy,
        # eval_accuracy) at each eval checkpoint + step 0.
        "log_history": trainer.state.log_history,
    }

    report_path = args.report_dir / f"{run_tag}_{ts}.json"
    report["hf_report_path"] = (
        hf_report_path(
            report_name=report_path.name,
            version=CURRENT_VERSION,
            technique=technique,
            report_group=args.report_dir.name,
        )
        if adapter_pushed
        else None
    )
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if adapter_pushed:
        uploaded_path = upload_report_to_hf(
            report_path=report_path,
            version=CURRENT_VERSION,
            technique=technique,
            report_group=args.report_dir.name,
            commit_message=f"Add {technique} training report: {run_tag} [{ts}]",
        )
        if uploaded_path:
            print(f"  Report pushed  : {HF_EXPERIMENTS_REPO}/{uploaded_path}")

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
            print(f"  HF adapter       : {HF_EXPERIMENTS_REPO}/{hf_sub}")
    print(f"{'=' * 60}\n")

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    train_main("LoRA", False, Path(__file__).parent.parent)


if __name__ == "__main__":
    main()
