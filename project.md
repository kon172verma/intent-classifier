# Project: Edge Tool Selection using SLM + QLoRA/AdaLoRA

## Project Goal

Build a complete research and engineering project that trains a Small Language Model (SLM) to perform dynamic MCP tool selection on an edge device.

The model receives:

* A user request
* A dynamically changing list of available MCP tools
* Tool descriptions

The model outputs:

* The selected tool name
* OR the string "none" if no available tool can satisfy the request

The model must output only the tool name or "none".

Example:

Input:

Available Tools:

Name: tool_7f2a91
Description: Provides driving analytics and vehicle telemetry.

Name: face_emotion
Description: Analyzes facial expressions and emotions.

User Request:
Show me the average speed and fuel efficiency for the last week.

Selected Tool:

Output:

tool_7f2a91

---

## Long-Term Vision

The final system should support:

* Dynamic MCP server discovery
* Dynamic tool names
* Dynamic tool descriptions
* Edge deployment
* Quantized models
* PEFT fine-tuning
* Benchmarking of multiple SLMs
* Comparison of LoRA, QLoRA, and AdaLoRA

The goal is to determine whether a small edge-deployable language model can reliably perform tool routing with high accuracy while consuming minimal memory.

---

## Phase 1: Dataset Generation

Create a synthetic dataset generator.

Dataset schema:

{
"user_request": "...",
"available_tools": [
{
"name": "...",
"description": "..."
}
],
"answer": "tool_name_or_none"
}

### Tool Catalog

A fixed reference catalog of 30 tools is maintained at:

dataset_sample/tools_reference.json

Tools have stable names and descriptions. The catalog is the single source of truth used by the generator.
Tool names intentionally mix naming styles (kebab-case corp names, snake_case) to mimic realistic MCP server registries.

Five tools in the catalog are designated **rare**: they appear as the correct answer in fewer than 2-3% of total examples, but they still appear in the available_tools list of other examples so the model learns their descriptions. This forces the model to generalise intent classification rather than memorise frequent patterns.

Rare tools: emergency_sos, roadside_assistance, insurance_claims, home_automation_bridge, corp-fleet-manager.

### Available-Tools Count Distribution

Each example draws a random subset of tools from the catalog. The count distribution is:

* 10% of examples: 1-3 tools (few-tool regime — tests disambiguation with minimal context)
* 10% of examples: 20-30 tools (many-tool regime — tests selection under noise)
* 80% of examples: 4-19 tools (standard regime)

### Answer Distribution

* ~20% of examples: answer is "none" (no available tool satisfies the request)
* ~80% of examples: answer is a valid tool name from the available_tools list

### Requirements

* Generate at least 10,000 examples total
* Support configurable dataset size via CLI argument
* Randomise tool ordering within each example
* Correct tool is always included in available_tools for non-none examples
* Rare tools appear as the correct answer in ≤ 2-3% of examples
* "none" examples must include plausible-but-wrong tools in the available_tools list

### Generation Milestones

1. Generate 100 sample examples first for manual review and prompt debugging
2. Scale to 10,000 examples after validation

### Dataset Split

80% train
10% validation
10% test

Export:

train.jsonl
validation.jsonl
test.jsonl

---

## Phase 2: Baseline Evaluation

Implement inference benchmarking for:

* Qwen3-0.6B
* Qwen3-1.7B
* Phi-3 Mini
* SmolLM3

| Model        | Parameters |
| ------------ | ---------- |
| SmolLM2-135M | 135M       |
| Qwen2.5-0.5B | 0.5B       |
| Gemma 3 270M | 270M       |
| SmolLM2-360M | 360M       |
| TinyLlama    | 1.1B       |
| Qwen2.5-1.5B | 1.5B       |
| Gemma 3 1B   | 1B         |
| Phi-3 Mini   | 3.8B       |
| Llama 3.2 3B | 3B         |
| Qwen3-4B     | 4B         |
| Qwen3-0.6B   | 0.6B       |
| Qwen3-1.7B   | 1.7B       |
| SmolLM3      | 1.7B       |

Use zero-shot prompting.

Prompt template:

You are a tool router.

Rules:

* Return only the tool name.
* Return "none" if no tool matches.
* Do not explain.

Available Tools:
...

User Request:
...

Selected Tool:

Metrics:

* Exact Match Accuracy
* Latency
* Tokens/sec
* Memory Usage

Generate benchmark reports.

---

## Phase 3: Fine-Tuning

Implement support for:

### LoRA

Parameters:

* rank
* alpha
* dropout

### QLoRA

Parameters:

* NF4 quantization
* double quantization
* bfloat16 compute

### AdaLoRA

Parameters:

* target rank
* initial rank
* warmup schedule
* pruning schedule

Use HuggingFace PEFT.

Implement configuration files.

Example:

configs/lora.yaml
configs/qlora.yaml
configs/adalora.yaml

---

## Phase 4: Experiment Tracking

Use:

* Weights & Biases or MLflow

Track:

* training loss
* validation loss
* accuracy
* memory usage
* trainable parameters
* training time

Save experiment metadata.

---

## Phase 5: Evaluation

Compute:

Exact Match Accuracy

Confusion Matrix

Per-tool Precision

Per-tool Recall

Per-tool F1

Generate plots.

Compare:

* Zero-shot
* LoRA
* QLoRA
* AdaLoRA

---

## Phase 6: Edge Deployment

Support:

* GGUF export
* llama.cpp inference
* ONNX export

Benchmark:

* RAM usage
* CPU usage
* latency

Target devices:

* Raspberry Pi
* Jetson
* Automotive edge computer
* Laptop CPU

---

## Phase 7: MCP Integration

Create a runtime service.

Input:

{
"user_request": "...",
"tools": [...]
}

Output:

{
"selected_tool": "..."
}

The runtime should:

1. Discover MCP servers
2. Build prompt
3. Query model
4. Return tool name

No tool execution yet.

Only routing.

---

## Project Structure

intent-classifier/
├── data/
├── datasets/
├── models/
├── training/
├── evaluation/
├── deployment/
├── inference/
├── configs/
├── experiments/
├── notebooks/
└── tests/

---

## Coding Requirements

* Python 3.12
* HuggingFace Transformers
* PEFT
* Datasets
* Accelerate
* BitsAndBytes
* PyTorch

Use type hints everywhere.

Use dataclasses where appropriate.

Write production-quality code.

Add unit tests.

Add CLI commands.

Support GPU and CPU execution.

---

## Deliverables

1. Dataset generator
2. Baseline evaluation framework
3. LoRA training pipeline
4. QLoRA training pipeline
5. AdaLoRA training pipeline
6. Evaluation suite
7. Edge deployment benchmarks
8. MCP router service
9. Documentation
10. Reproducible experiments

Implement the project incrementally, beginning with Phase 1 (dataset generation) and Phase 2 (baseline evaluation).
