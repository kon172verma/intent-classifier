#!/usr/bin/env python3
"""
Phase 3 – QLoRA fine-tuning for MCP tool selection.

Loads the base model in 4-bit NF4 quantization via BitsAndBytesConfig, then
attaches LoRA adapters with PEFT — the QLoRA recipe (Dettmers et al. 2023).
Loss is computed only on the assistant (answer) tokens via pre-tokenized
labels (-100 masking) + DataCollatorForSeq2Seq.
Compatible with TRL 1.x (DataCollatorForCompletionOnlyLM was removed in TRL 1.0).

Compared to finetune_LoRA the key differences are:
  - BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16)
  - prepare_model_for_kbit_training() instead of enable_input_require_grads()
  - gradient_checkpointing=False  (bnb 4-bit + GC can cause NaN losses)
  - VRAM usage is ~50% lower; fp16 not used (bfloat16 only for compute)

Saves
-----
  checkpoints/{model}_{config}_{size}/   — best LoRA adapter (top-2 by eval_loss)
  reports_training/{run_tag}_{ts}.json  — full training log with loss + accuracy history

Usage
-----
    # Recommended main run
    python qlora_train.py --model qwen2.5-0.5b --lora-config B --dataset-size 1k

    # All options
    python qlora_train.py --model qwen3-0.6b --lora-config C --dataset-size 10k --device cuda

    # Smoke-test: 10 steps only — validates the whole pipeline without committing to training
    python qlora_train.py --model qwen2.5-0.5b --lora-config A --smoke-test
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

# Load .env from repo root (provides HF_TOKEN for gated models)
_env_file = Path(__file__).parent.parent.parent / ".env"
if _env_file.exists():
    from dotenv import load_dotenv

    load_dotenv(_env_file)

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", message=".*max_new_tokens.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer  # type: ignore
from datasets import Dataset  # type: ignore

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    MODEL_REGISTRY,
    LORA_CONFIGS,
    QWEN3_KEYS,
    MAX_SEQ_LENGTH,
    build_chat_messages,
    apply_chat_template_safe,
    extract_prediction,
    compute_accuracy,
)

QLORA_DIR = Path(__file__).parent.parent
DEFAULT_DATA_DIR = QLORA_DIR / "data"
DEFAULT_CKPT_DIR = QLORA_DIR / "checkpoints"
DEFAULT_REPORT_DIR = QLORA_DIR / "reports_training"

# Number of val examples used inside the accuracy callback during training.
_CALLBACK_MAX_VAL: int = 100


# ── Inference helper (used by accuracy callback) ──────────────────────────────


def _generate_one(
    model: Any,
    tokenizer,
    example: dict,
    device: torch.device,
    model_key: str,
) -> str:
    """Greedy generation on a single example; returns the predicted tool name."""
    messages = build_chat_messages(example, include_answer=False)
    text = apply_chat_template_safe(
        tokenizer, messages, model_key, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=int(tokenizer.eos_token_id),
        )
    new_ids = out[0][inputs["input_ids"].shape[1] :]
    return extract_prediction(tokenizer.decode(new_ids, skip_special_tokens=True))


# ── Accuracy callback ─────────────────────────────────────────────────────────


class ValAccuracyCallback(TrainerCallback):
    """
    Appends eval_accuracy to the trainer log at each evaluation checkpoint.
    Uses at most _CALLBACK_MAX_VAL examples so the callback stays fast.
    """

    def __init__(
        self,
        val_examples: list[dict],
        tokenizer,
        model_key: str,
        device: torch.device,
    ) -> None:
        self.val_sample = val_examples[:_CALLBACK_MAX_VAL]
        self.tokenizer = tokenizer
        self.model_key = model_key
        self.device = device

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module | None = None,
        **kwargs,
    ) -> None:
        if model is None:
            return

        was_training = model.training
        model.eval()

        predictions = [
            _generate_one(model, self.tokenizer, ex, self.device, self.model_key)
            for ex in self.val_sample
        ]
        labels = [ex["answer"] for ex in self.val_sample]
        acc = compute_accuracy(predictions, labels)
        n_ok = sum(p == l for p, l in zip(predictions, labels))

        # Attach to the most recent eval log entry so it ends up in log_history.
        for entry in reversed(state.log_history):
            if "eval_loss" in entry:
                entry["eval_accuracy"] = round(acc, 4)
                break

        print(
            f"\n  [ValAccuracy] step={state.global_step}  "
            f"acc={acc:.4f}  ({n_ok}/{len(self.val_sample)})"
        )

        if was_training:
            model.train()


# ── Helpers ───────────────────────────────────────────────────────────────────


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def tokenize_with_labels(
    example: dict,
    tokenizer,
    model_key: str,
) -> dict:
    """
    Tokenizes one training example and masks all non-assistant tokens with -100.

    Pre-tokenizing the dataset means TRL skips its own tokenization step, making
    this compatible with TRL 1.x where DataCollatorForCompletionOnlyLM was removed.
    Returns a dict with 'input_ids', 'attention_mask', and 'labels'.
    """
    messages = build_chat_messages(example, include_answer=True)
    text = apply_chat_template_safe(
        tokenizer, messages, model_key, add_generation_prompt=False
    )
    enc = tokenizer(
        text,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors=None,  # plain Python lists
    )
    input_ids: list[int] = [int(x) for x in enc["input_ids"]]
    attention_mask: list[int] = [int(x) for x in enc["attention_mask"]]

    # Find the LAST occurrence of "<|im_start|>assistant\n" token sequence.
    # Everything up to and including that template is masked (-100); only
    # the assistant answer tokens receive gradient signal.
    resp_ids: list[int] = [
        int(x)
        for x in tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    ]
    rlen = len(resp_ids)
    last_pos = -1
    for i in range(len(input_ids) - rlen + 1):
        if input_ids[i : i + rlen] == resp_ids:
            last_pos = i

    labels = list(input_ids)
    mask_end = (last_pos + rlen) if last_pos >= 0 else len(labels)
    for i in range(mask_end):
        labels[i] = -100

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # BitsAndBytes 4-bit only supports CUDA — no MPS fallback for training.
        return torch.device("cpu")
    return torch.device(requested)


def peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024**2
    return 0.0


def count_trainable(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ── Argument parsing ──────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA fine-tuning for MCP tool selection.")
    p.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        default="qwen2.5-0.5b",
    )
    p.add_argument(
        "--lora-config",
        choices=["A", "B", "C", "D"],
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
    p.add_argument("--ckpt-dir", type=Path, default=None)
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="cuda required for 4-bit BitsAndBytes quantization.",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run only 10 training steps to validate the pipeline (no checkpoint saved).",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    # Reduce CUDA allocator fragmentation — especially helpful for 4-bit models
    # where large contiguous allocations are needed mid-training.
    if device.type == "cuda":
        import os

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    lora_cfg = LORA_CONFIGS[args.lora_config]
    model_id = MODEL_REGISTRY[args.model]
    run_tag = f"{args.model}_config_{args.lora_config}_{args.dataset_size}"

    data_dir = (args.data_dir or DEFAULT_DATA_DIR) / args.dataset_size
    ckpt_dir = (args.ckpt_dir or DEFAULT_CKPT_DIR) / run_tag
    args.report_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    use_4bit = device.type == "cuda"

    print(f"\n{'=' * 60}")
    print(f"  QLoRA Training — {run_tag}")
    print(f"  Model      : {model_id}")
    print(f"  LoRA config: {args.lora_config} — {lora_cfg['description']}")
    print(f"  Dataset    : {args.dataset_size}")
    print(f"  Device     : {device}")
    print(f"  4-bit NF4  : {use_4bit}")
    if args.smoke_test:
        print(f"  Mode       : SMOKE TEST (10 steps only)")
    print(f"{'=' * 60}")

    if not use_4bit:
        print(
            "\n  WARNING: 4-bit quantization requires CUDA.  "
            "Running in full precision — memory savings will not apply."
        )

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
    tokenizer.padding_side = "right"  # required for causal LM training

    # ── Build 4-bit quantization config ───────────────────────────────────────
    # NF4 (Normal Float 4-bit) is the quantization dtype from QLoRA paper.
    # Double quantization further compresses the quantization constants.
    # Compute dtype stays bfloat16 for numerical stability during forward passes.
    bnb_config: BitsAndBytesConfig | None = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"  Loading model: {model_id}  (4-bit={use_4bit})")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        # When not quantizing, use bfloat16 on CUDA, float32 on CPU.
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map={"": device},
        trust_remote_code=False,
    )

    # prepare_model_for_kbit_training: enables gradient checkpointing-compatible
    # input gradient hooks for frozen quantized layers.  Replaces the plain
    # enable_input_require_grads() call used in full-precision LoRA.
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=False,  # disabled: BNB 4-bit + GC can NaN
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing

    # ── Apply LoRA adapter ────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg["target_modules"],
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias="none",
        inference_mode=False,
    )
    model = cast(Any, get_peft_model(model, lora_config))

    trainable, total = count_trainable(model)
    print(f"\n  Trainable params : {trainable:,}  ({trainable / total * 100:.3f}%)")
    print(f"  Total params     : {total:,}")

    # ── Tokenize datasets (with prompt masking) ───────────────────────────────
    # Pre-tokenize so TRL skips its own dataset processing step (TRL 1.x compat).
    print("\n  Tokenizing training data...")
    train_records = [
        tokenize_with_labels(ex, tokenizer, args.model) for ex in train_examples
    ]
    val_records = [
        tokenize_with_labels(ex, tokenizer, args.model) for ex in val_examples
    ]
    train_dataset = Dataset.from_list(train_records)
    val_dataset = Dataset.from_list(val_records)

    # ── Step 0 baseline (untrained model) ─────────────────────────────────────
    # Captures val accuracy and loss BEFORE any gradient updates so training
    # curves start at step 0 — makes the improvement from fine-tuning visible.
    #
    # Memory discipline: use inference_mode (no autograd graph), delete every
    # intermediate tensor immediately, then empty_cache so the trainer starts
    # with a clean allocator.  This is critical on 4-bit / QLoRA where the
    # optimizer states will need the headroom freed here.
    print("\n  Computing step 0 baseline (untrained model)...")
    model.eval()
    step0_preds = [
        _generate_one(model, tokenizer, ex, device, args.model)
        for ex in val_examples[:_CALLBACK_MAX_VAL]
    ]
    step0_acc = compute_accuracy(
        step0_preds, [ex["answer"] for ex in val_examples[:_CALLBACK_MAX_VAL]]
    )
    del step0_preds  # free generation outputs before loss loop
    _loss_total = 0.0
    _loss_n = 0
    with torch.inference_mode():
        for record in val_records[:16]:  # 16 samples — enough for a stable estimate
            ids = torch.tensor([record["input_ids"]], device=device)
            lbls = torch.tensor([record["labels"]], device=device)
            mask = torch.tensor([record["attention_mask"]], device=device)
            out = model(input_ids=ids, attention_mask=mask, labels=lbls)
            if not torch.isnan(out.loss):
                _loss_total += out.loss.item()
                _loss_n += 1
            del ids, lbls, mask, out  # release each tensor immediately
    step0_loss = _loss_total / _loss_n if _loss_n > 0 else float("nan")
    step0_log_entry = {
        "step": 0,
        "epoch": 0.0,
        "eval_loss": round(step0_loss, 6),
        "eval_accuracy": round(step0_acc, 4),
    }
    print(f"  [Step 0]  val_loss={step0_loss:.4f}  val_acc={step0_acc:.4f}")
    model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()  # return all freed memory to the allocator

    # ── Compute eval_steps ────────────────────────────────────────────────────
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
    # QLoRA training: bfloat16 compute, no fp16, no gradient checkpointing
    # (BNB 4-bit + GC is known to produce NaN losses in some configurations).
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=lora_cfg["num_train_epochs"] if not args.smoke_test else 1,
        max_steps=10 if args.smoke_test else -1,
        per_device_train_batch_size=lora_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=lora_cfg["gradient_accumulation_steps"],
        learning_rate=lora_cfg["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=use_bf16,
        fp16=False,  # never use fp16 with NF4 quantization
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_pin_memory=(device.type == "cuda"),
        gradient_checkpointing=False,  # disabled for BNB 4-bit compatibility
        optim="paged_adamw_8bit",  # paged optimizer saves VRAM on GPU
    )

    # ── Data collator ─────────────────────────────────────────────────────────
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=3),
        ValAccuracyCallback(val_examples, tokenizer, args.model, device),
    ]

    # ── Build SFTTrainer ──────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(
        f"\n  Starting training"
        f"{' (smoke-test: 10 steps)' if args.smoke_test else ''}...\n"
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t_start = time.time()
    train_result = trainer.train()
    t_elapsed = time.time() - t_start
    mem_mb = peak_memory_mb(device)

    print(f"\n  Training complete in {t_elapsed:.1f}s  |  Peak memory: {mem_mb:.0f} MB")

    # Prepend step 0 entry so training curves show the full improvement arc.
    trainer.state.log_history.insert(0, step0_log_entry)

    # ── Save adapter ──────────────────────────────────────────────────────────
    if not args.smoke_test:
        print(f"\n  Saving best adapter → {ckpt_dir}")
        trainer.model.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))

    trainer.model.config.use_cache = True

    # ── Save training report ──────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_train_loss = train_result.training_loss
    eval_entries = [e for e in trainer.state.log_history if "eval_loss" in e]
    final_eval_loss = eval_entries[-1].get("eval_loss") if eval_entries else None
    final_val_acc = eval_entries[-1].get("eval_accuracy") if eval_entries else None

    report = {
        "model_key": args.model,
        "model_id": model_id,
        "lora_config": args.lora_config,
        "lora_config_desc": lora_cfg["description"],
        "dataset_size": args.dataset_size,
        "quant_method": "nf4_4bit" if use_4bit else "none",
        "device": str(device),
        "timestamp": ts,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(trainable / total * 100, 4),
        "steps_per_epoch": steps_per_epoch,
        "total_steps_trained": trainer.state.global_step,
        "early_stopped": trainer.state.global_step < total_steps,
        "peak_memory_mb": round(mem_mb, 1),
        "total_training_time_s": round(t_elapsed, 1),
        "final_train_loss": round(final_train_loss, 6),
        "final_eval_loss": round(final_eval_loss, 6)
        if final_eval_loss is not None
        else None,
        "final_val_accuracy": round(final_val_acc, 4)
        if final_val_acc is not None
        else None,
        "checkpoint_dir": str(ckpt_dir),
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
    if final_eval_loss is not None:
        print(f"  Final val loss   : {final_eval_loss:.4f}")
    if final_val_acc is not None:
        print(f"  Final val acc    : {final_val_acc:.4f}")
    print(f"  Training time    : {t_elapsed:.1f}s")
    print(f"  Peak memory      : {mem_mb:.0f} MB")
    print(f"  Training report  : {report_path}")
    if not args.smoke_test:
        print(f"  Adapter saved    : {ckpt_dir}")
    print(f"{'=' * 60}\n")

    # Cleanup
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
