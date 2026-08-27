# LoRA v2.0 Final Report

## Scope

This report covers the completed v2.0 LoRA experiment matrix: five base models,
four adapter configurations, and the 1k dataset split (800 train / 100
validation / 100 test examples). In v2.0 the model predicts the positional tool
ID assigned in each prompt, not a tool name. The on-disk dataset still stores
the readable tool name; the shared prompt renderer maps it to the expected ID.

All accuracies below are exact-match ID accuracy. Each validation or test point
is one of 100 examples, so a one-point percentage difference is one example.

## Configuration

LoRA freezes the base model and trains low-rank adapters. Configurations A-D
range from Q/V-only rank 8 adapters to full-attention-plus-MLP rank 32 adapters.
The current definitions, including learning rates and batch settings, live in
[`finetune_lib/config.py`](../finetune_lib/config.py).

## Results

Values are **validation / test accuracy (%)**.

| Model | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| smollm2-360m | 50 / 51 | 51 / 52 | 58 / 57 | 55 / 57 |
| qwen2.5-0.5b | 64 / 68 | 69 / 71 | 74 / 77 | 72 / 78 |
| qwen3-0.6b | 96 / 99 | 97 / 97 | 96 / 99 | 98 / 99 |
| llama3.2-1b | 95 / 98 | 97 / 97 | 96 / 97 | 96 / 94 |
| smollm2-1.7b | 93 / 97 | 94 / 97 | 94 / 98 | 95 / 97 |

## Findings

- Qwen3-0.6B is the most efficient high-performing option: configurations A,
  C, and D each reached 99% test accuracy using 1.26-1.35 GB peak inference
  memory.
- Llama3.2-1B reached 98% test accuracy with config A, but the heavier
  configurations did not improve this result.
- Config C or D helps Qwen2.5-0.5B, while the 360M SmolLM model remains well
  below the larger models even with the wider adapters.

## Charts And Artefacts

- [Training curves](analysis/lora_training_curves.png)
- [Training summary](analysis/lora_combined_train.png)
- [Validation summary](analysis/lora_combined_val.png)
- [Test summary](analysis/lora_combined_test.png)
- JSON reports: `reports_training/`, `reports_validation/`, and `reports_test/`
