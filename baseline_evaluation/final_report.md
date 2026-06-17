# Baseline Evaluation: Final Analysis & PEFT Model Selection

## What We Did

We evaluated 11 open-weight language models on an **edge MCP tool-selection task** — given a natural-language user request, the model must return exactly the right tool name from a catalogue of 30 MCP tools (or `none`).

The evaluation was conducted in two phases:

- **Zero-shot**: No examples in the prompt; uses only the system prompt with the tool catalogue.
- **Few-shot**: Prepends 7 labelled demonstrations; uses the same system prompt plus input to output examples.

All models were evaluated on the same 100-question benchmark drawn from `dataset_sample/sample.json`. Metrics collected per model: accuracy (%), peak memory (MB), average latency (ms), and token throughput (tokens/s).

---

## Results

### Combined Comparison

![Combined bar chart - accuracy/peak-memory panel and throughput/latency panel](/baseline_evaluation/analysis/combined_comparision.png)

### Summary Table

![Combined table with zero-shot accuracy, few-shot accuracy, delta, latency, and memory](/baseline_evaluation/analysis/combined_table.png)

---

> Note: llama3.2-3B was blocked by Meta's access policy (HTTP 403). phi3-mini was permanently skipped due to stale cached model code with unrecoverable bugs.

---

## Model Selection for PEFT Fine-tuning

Based on the updated combined comparison graph and latest reruns, the model strategy is now split into:

1. **Primary production candidates (3 models)**
2. **Specialist experimental candidate (1 model)**

### Primary Production Candidates (Definitive)

- **qwen2.5-0.5b** (~500M): Best small-model operating point for edge deployment, with strong routing quality for its size and low resource use.
- **qwen3-0.6b** (~600M): Strong balance of quality and efficiency; useful as a second small-model backbone with different behavior than qwen2.5-0.5b.
- **qwen2.5-1.5b** (~1.5B): Highest-confidence general-purpose option among practical-size models, with still-manageable memory and latency.

### Specialist Experimental Candidate

- **smollm3** (~3.0B): Shows a strong few-shot learning signal and good task understanding under demonstrations, but latency is materially higher. Keep this as an escalation and specialist path rather than the default.

### Decision Summary

- **Default serving pool:** `qwen2.5-0.5b`, `qwen3-0.6b`, `qwen2.5-1.5b`
- **Escalation pool:** `smollm3` (triggered only for hard or low-confidence cases)

### Deprioritised Models

- **smollm2-135m**: Near-zero accuracy in both modes (0%, 1%). No usable signal for fine-tuning; the model is too small to follow structured tool-selection instructions.
- **smollm2-360m**: Only 13%/18% accuracy. While slightly better than the 135M variant, the accuracy is still too low to serve as a meaningful PEFT starting point.
- **tinyllama**: 0% zero-shot and 10% few-shot, plus the highest latency in the benchmark (avg 2,408 ms) among weak performers.
- **gemma3-270m**: 10%/12% accuracy and mostly empty or None outputs; too small to distinguish between 30 tools reliably.
- **gemma3-1b**: Competitive but not selected in the current track because it is outperformed by the chosen Qwen trio on the quality efficiency tradeoff for deployment.
- **qwen3-1.7b**: Decent quality, but redundant versus qwen2.5-1.5b for current goals.
- **qwen3-4b**: 96% zero-shot is near ceiling with little fine-tuning headroom, and peak memory is too high for edge-first deployment.

---

## Next Steps

The next phase should optimize for both quality and real-time serving behavior:

1. Fine-tune the 3 core models first (`qwen2.5-0.5b`, `qwen3-0.6b`, `qwen2.5-1.5b`) using LoRA/QLoRA.
2. Run `smollm3` as a separate experimental track focused on difficult cases and few-shot-style generalization.
