# LoRA+ v2.0 Final Report

## Scope

This report covers the completed v2.0 LoRA+ matrix: five base models, four
configurations, and the 1k split (800 train / 100 validation / 100 test). The
task is dynamic tool routing: each model emits the positional tool ID in the
prompt rather than a tool name.

All values are exact-match ID accuracy. A percentage point represents one
example on the 100-example validation or test split.

## Configuration

LoRA+ uses the same adapter scopes as LoRA, but gives matrix B a higher learning
rate than matrix A. The LoRA+ ratios are 16 for configurations A-C and 8 for D;
the shared configuration definitions are in
[`finetune_lib/config.py`](../finetune_lib/config.py).

## Results

Values are **validation / test accuracy (%)**.

| Model | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| smollm2-360m | 65 / 67 | 64 / 72 | 80 / 83 | 63 / 71 |
| qwen2.5-0.5b | 85 / 83 | 76 / 82 | 76 / 86 | 82 / 92 |
| qwen3-0.6b | 98 / 100 | 96 / 95 | 97 / 97 | 95 / 97 |
| llama3.2-1b | 97 / 100 | 97 / 99 | 98 / 98 | 98 / 98 |
| smollm2-1.7b | 96 / 97 | 94 / 96 | 97 / 99 | 95 / 97 |

## Findings

- LoRA+ gives Qwen3-0.6B config A and Llama3.2-1B config A perfect 100% test
  accuracy. Those are the lightest adapters in their model rows.
- Config D is particularly effective for Qwen2.5-0.5B, reaching 92% test
  accuracy, its best LoRA+ result.
- Config C is the clear choice for the 360M and 1.7B SmolLM models in this
  technique, reaching 83% and 99% test accuracy respectively.

## Charts And Artefacts

- [Training curves](analysis/lora+_training_curves.png)
- [Training summary](analysis/lora+_combined_train.png)
- [Validation summary](analysis/lora+_combined_val.png)
- [Test summary](analysis/lora+_combined_test.png)
- JSON reports: `reports_training/`, `reports_validation/`, and `reports_test/`
