# Intent Classifier — Fine-Tuning Experiments

Train a Small Language Model (SLM) to route user requests to the correct tool from a
dynamically provided list. The model receives a user request and a list of available tools
with descriptions, and outputs either the selected tool name or `"none"`.

## Related Repositories

| Repo | Purpose |
| ------ | --------- |
| **This repo** — GitHub [kon172verma/intent-classifier](https://github.com/kon172verma/intent-classifier) | Dataset, baseline evaluation, fine-tuning experiments, and release reports |
| **Experiments** — HF [kon172verma/intent-classifier-experiments](https://huggingface.co/kon172verma/intent-classifier-experiments) | Every adapter produced during experimentation, organized by version folder |
| **Release** — HF [kon172verma/intent-classifier](https://huggingface.co/kon172verma/intent-classifier) | The 2 best models per release (merged, full-weight) |
| **Inference** — GitHub [kon172verma/intent-classifier-inference](https://github.com/kon172verma/intent-classifier-inference) | Downloads release models and benchmarks them across inference engines (llama.cpp, HF Transformers, ONNX, TensorRT-LLM) on edge hardware |

---

## Dataset

A synthetic dataset of tool-routing examples generated from a fixed catalog of 30 tools.

**Schema:**

```json
{
  "user_request": "...",
  "available_tools": [{"name": "...", "description": "..."}],
  "answer": "tool_name_or_none"
}
```

**Key properties:**

- ~20% of examples have `"none"` as the answer
- 5 rare tools appear as the correct answer in ≤ 2–3% of examples
- Available-tools count is distributed across few-tool (1–3), standard (4–19),
  and many-tool (20–30) regimes
- Split: 80% train / 10% validation / 10% test

Dataset lives in `dataset_full/` (full scale) and `dataset_sample/` (100-example sample
used during development).

---

## Baseline Evaluation

Zero-shot accuracy benchmarked across a range of SLMs before any fine-tuning. Results live in `evaluation_baseline/`.

| Key | Model | Size |
| ----- | ------- | ------ |
| pythia-70m | EleutherAI/pythia-70m | 70M |
| cerebras-111m | cerebras/Cerebras-GPT-111M | 111M |
| smollm2-135m | HuggingFaceTB/SmolLM2-135M-Instruct | 135M |
| gemma3-270m | google/gemma-3-270m-it | 270M |
| smollm2-360m | HuggingFaceTB/SmolLM2-360M-Instruct | 360M |
| qwen2.5-0.5b | Qwen/Qwen2.5-0.5B-Instruct | 0.5B |
| qwen3-0.6b | Qwen/Qwen3-0.6B | 0.6B |
| gemma3-1b | google/gemma-3-1b-it | 1B |
| llama3.2-1b | meta-llama/Llama-3.2-1B-Instruct | 1B |
| qwen3-1.7b | Qwen/Qwen3-1.7B | 1.7B |
| smollm2-1.7b | HuggingFaceTB/SmolLM2-1.7B-Instruct | 1.7B |
| granite3.3-2b | ibm-granite/granite-3.3-2b-instruct | 2B |
| gemma2-2b | google/gemma-2-2b-it | 2B |
| smollm3 | HuggingFaceTB/SmolLM3-3B | 3B |
| llama3.2-3b | meta-llama/Llama-3.2-3B-Instruct | 3B |

Metric: Exact Match Accuracy.

---

## Fine-Tuning

Six PEFT techniques are implemented and evaluated, each in its own directory:

| Technique | Directory | Notes |
| ----------- | ----------- | ------- |
| LoRA | `finetune_LoRA/` | rank, alpha, dropout |
| LoRA+ | `finetune_LoRAplus/` | differential learning rates for A/B matrices |
| DoRA | `finetune_DoRA/` | weight decomposition into magnitude + direction |
| DoRA+ | `finetune_DoRAplus/` | DoRA with LoRA+ learning rate schedule |
| AdaLoRA | `finetune_AdaLoRA/` | adaptive rank allocation with SVD pruning |
| QLoRA | `finetune_QLoRA/` | NF4 quantized base + LoRA adapters |

All techniques use HuggingFace PEFT. Shared training utilities live in `finetune_lib/`.
Each training run evaluates the adapter on the validation/test set and logs metrics
alongside the adapter.

Adapters are pushed to the experiments repo under
`{version}/{model}_{technique}_{config}_{dataset_size}_{timestamp}`
(e.g. `v1.0/qwen3-0.6b_LoRA_C_1k_20260715-044041`). `EXPERIMENTS.jsonl` is an
append-only log of every push, written by `finetune_lib/registry.py`.

---

## Release Evaluation

Once all fine-tuning experiments for a version are complete, the 2 best adapters are
selected and subjected to a deeper evaluation in a dedicated `evaluation_release/` folder
(separate from the per-technique evaluations in each `finetune_<technique>/` directory).
This evaluation computes:

- Exact Match Accuracy
- Per-tool Precision, Recall, and F1
- Confusion matrix

Results are used to write the release report documenting the selection rationale,
metrics comparison, and final adapter locations in the release repo.

---

## Versioning and Release

`VERSION` holds the current experiment version (e.g. `v1.0`). Bumping it starts a new
round — all subsequent adapter pushes land in a new folder in the experiments repo.

After the release evaluation is complete, the 2 best adapters are merged and pushed to the release repo.

---

## Project Structure

```text
dataset_sample/        # 100-example sample
dataset_full/          # Full 10k+ example dataset
evaluation_lib/        # Shared evaluation utilities
evaluation_baseline/   # Zero-shot benchmark results
finetune_lib/          # Shared training utilities and registry
finetune_LoRA/         # LoRA experiments
finetune_LoRAplus/     # LoRA+ experiments
finetune_DoRA/         # DoRA experiments
finetune_DoRAplus/     # DoRA+ experiments
finetune_AdaLoRA/      # AdaLoRA experiments
finetune_QLoRA/        # QLoRA experiments
evaluation_release/    # Deep per-tool evaluation for the 2 best models per release
scripts/               # Helper scripts
EXPERIMENTS.jsonl      # Append-only log of all adapter pushes
VERSION                # Current experiment version
```
