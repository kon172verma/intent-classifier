# DoRA+ v2.0 Final Report

## Scope

This report covers the completed v2.0 DoRA+ matrix: five base models, four
configurations, and the 1k split (800 train / 100 validation / 100 test). The
models predict the dynamic positional tool ID rendered in each prompt, rather
than predicting a fixed tool name.

All values are exact-match ID accuracy. The validation and test sets each have
100 examples, so small percentage differences should be interpreted cautiously.

## Configuration

DoRA+ combines DoRA's weight decomposition with LoRA+'s asymmetric matrix
learning rates. It uses the shared A-D adapter definitions in
[`finetune_lib/config.py`](../finetune_lib/config.py).

## Results

Values are **validation / test accuracy (%)**.

| Model | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| smollm2-360m | 62 / 72 | 63 / 71 | 86 / 89 | 62 / 68 |
| qwen2.5-0.5b | 85 / 86 | 88 / 91 | 83 / 88 | 81 / 86 |
| qwen3-0.6b | 99 / 99 | 97 / 99 | 97 / 100 | 99 / 100 |
| llama3.2-1b | 96 / 97 | 97 / 97 | 96 / 100 | 94 / 99 |
| smollm2-1.7b | 96 / 98 | 95 / 97 | 97 / 99 | 96 / 99 |

## Findings

- DoRA+ is the strongest overall technique in this matrix. It reaches 100%
  test accuracy for Qwen3-0.6B configurations C and D, and Llama3.2-1B
  configuration C.
- Qwen3-0.6B config D is the most robust high-performing run: 99% validation,
  100% test, and 1.40 GB peak inference memory.
- Config C substantially improves the 360M model to 89% test accuracy, but its
  larger models also come with higher inference memory and latency.

## Charts And Artefacts

- [Training curves](analysis/dora+_training_curves.png)
- [Training summary](analysis/dora+_combined_train.png)
- [Validation summary](analysis/dora+_combined_val.png)
- [Test summary](analysis/dora+_combined_test.png)
- JSON reports: `reports_training/`, `reports_validation/`, and `reports_test/`
