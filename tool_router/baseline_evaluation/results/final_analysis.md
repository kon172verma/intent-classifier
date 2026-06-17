# Baseline Evaluation: Final Analysis & PEFT Model Selection

## What We Did

We evaluated 11 open-weight language models on an **edge MCP tool-selection task** — given a natural-language user request, the model must return exactly the right tool name from a catalogue of 30 MCP tools (or `none`).

The evaluation was conducted in two phases:

| Phase | Description | Prompt strategy |
|-------|-------------|-----------------|
| **Zero-shot** | No examples in the prompt | System prompt with tool catalogue only |
| **Few-shot** | 7 labelled examples prepended to the prompt | Same system prompt + 7 input→output demonstrations |

All models were evaluated on the same 100-question benchmark drawn from `data/sample.json`. Metrics collected per model: accuracy (%), peak memory (MB), and average latency (ms).

---

## Results

### Combined Comparison

![Combined bar chart — zero-shot, few-shot, and peak memory per model](/tool_router/baseline_evaluation/results/combined_comparision.png)

### Summary Table

![Combined table with zero-shot accuracy, few-shot accuracy, delta, latency, and memory](/tool_router/baseline_evaluation/results/combined_table.png)

---

## Full Numerical Results

| Model | Params | ZS Acc | FS Acc | Δ | Peak Mem |
|-------|--------|--------|--------|---|----------|
| smollm2-135m | ~135M | 0.0% | 1.0% | +1.0pp | 257 MB |
| gemma3-270m | ~270M | 10.0% | 12.0% | +2.0pp | 511 MB |
| smollm2-360m | ~360M | 13.0% | 18.0% | +5.0pp | 690 MB |
| qwen2.5-0.5b | ~500M | 62.0% | 51.0% | −11.0pp | 942 MB |
| qwen3-0.6b | ~600M | 21.0% | 52.0% | +31.0pp | 1,137 MB |
| gemma3-1b | ~1.0B | 54.0% | 44.0% | −10.0pp | 1,907 MB |
| tinyllama | ~1.1B | 0.0% | 10.0% | +10.0pp | 2,098 MB |
| qwen2.5-1.5b | ~1.5B | 87.0% | 86.0% | −1.0pp | 2,944 MB |
| qwen3-1.7b | ~1.7B | 79.0% | 80.0% | +1.0pp | 3,282 MB |
| smollm3 | ~3.0B | 39.0% | 78.0% | +39.0pp | 5,865 MB |
| qwen3-4b | ~4.0B | 96.0% | 96.0% | 0.0pp | 7,672 MB |

> Note: llama3.2-3B was blocked by Meta's access policy (HTTP 403). phi3-mini was permanently skipped due to stale cached model code with unrecoverable bugs.

---

## Model Selection for PEFT Fine-tuning

We select **5 models** for the next phase of fine-tuning using PEFT (LoRA / QLoRA). The selection prioritises models that are:

1. **Deployable on-device** — fit within ≤ 3 GB memory, making them viable for edge scenarios.
2. **Not already saturated** — models near 96% zero-shot accuracy have little room to improve and are expensive to run at inference time.
3. **Show genuine learning signal** — either reasonable zero-shot accuracy or meaningful few-shot gain, demonstrating the model architecture can follow tool-selection instructions.
4. **Architecturally diverse** — covering different parameter scales and training lineages to avoid redundant experiments.

### Selected Models

| Model | Params | ZS Acc | FS Acc | Rationale |
|-------|--------|--------|--------|-----------|
| **qwen2.5-0.5b** | ~500M | 62% | 51% | Strongest sub-1B model in zero-shot. The few-shot regression suggests the model's instruction following degrades with longer context — exactly the kind of alignment gap PEFT is designed to fix. Low memory footprint (942 MB) makes it the most attractive target for edge deployment post-fine-tuning. |
| **qwen3-0.6b** | ~600M | 21% | 52% | Largest zero-shot-to-few-shot gain (+31pp) among all models. The architecture clearly has the capacity to route correctly when given examples — PEFT should be able to bake that capability in without needing runtime few-shot context. Slightly heavier than qwen2.5-0.5b but still under 1.2 GB. |
| **gemma3-1b** | ~1.0B | 54% | 44% | Strong zero-shot baseline for its size; however, few-shot prompting actively hurts it (−10pp), suggesting prompt-format sensitivity. Fine-tuning on the task format should resolve this regression and push accuracy above its current ceiling. Represents the Gemma architecture lineage. |
| **qwen2.5-1.5b** | ~1.5B | 87% | 86% | Best accuracy-to-memory ratio of all evaluated models. 87% zero-shot with only 2.9 GB peak memory is a compelling baseline. Despite its high starting accuracy, there is still meaningful headroom (13pp) to reach near-perfect routing, and PEFT at this size is computationally tractable. |
| **smollm3** | ~3.0B | 39% | 78% | Largest few-shot gain (+39pp) in the entire benchmark. The zero-shot figure is unexpectedly low given its parameter count — likely due to the model's CoT-oriented training causing it to emit `<think>` tokens and raw reasoning traces rather than a clean tool name. PEFT fine-tuning on the task format should unlock this model's latent capability. |

### Rejected Models

| Model | Reason for rejection |
|-------|----------------------|
| **smollm2-135m** | Near-zero accuracy in both modes (0%, 1%). No usable signal for fine-tuning; the model is too small to follow structured tool-selection instructions. |
| **smollm2-360m** | Only 13%/18% accuracy. While slightly better than the 135M variant, the accuracy is still far too low to serve as a meaningful PEFT starting point. Would require disproportionate training effort for marginal gain. |
| **tinyllama** | 0% zero-shot, 10% few-shot. The TinyLlama-1.1B architecture predates modern instruction-tuning practices and shows no zero-shot task comprehension. Highest latency in the benchmark (avg 2,408 ms) for one of the weakest results. |
| **gemma3-270m** | 10%/12% accuracy; mostly produces empty or `None` outputs. Too small to distinguish between 30 tools reliably. |
| **qwen3-1.7b** | Good performance (79%/80%) but functionally redundant: qwen2.5-1.5b achieves 87%/86% at a comparable parameter count and lower memory usage. Running both would give limited additional insight. |
| **qwen3-4b** | 96% zero-shot accuracy is effectively a ceiling result — there is very little room left to improve. At 7.7 GB peak memory it is also the heaviest model evaluated and the least suited for edge deployment. Better used as a teacher/reference model if knowledge distillation is explored later. |

---

## Next Steps

The 5 selected models will be fine-tuned using **LoRA / QLoRA** (PEFT) on the 100-example benchmark dataset:

1. Define LoRA rank, alpha, and target modules for each architecture.
2. Train with `peft` + `transformers` + `accelerate` on MPS.
3. Evaluate fine-tuned adapters on a held-out test split.
4. Compare fine-tuned accuracy against the zero-shot and few-shot baselines above.
