# AdaLoRA v2.0 Final Report

## Scope

This report covers the completed v2.0 AdaLoRA experiments for Qwen3-0.6B and
Llama3.2-1B on the 1k split (800 train / 100 validation / 100 test). The task
uses dynamic positional tool IDs: the model receives an example-specific tool
map and emits the correct ID instead of a tool name.

All values are exact-match ID accuracy. Each percentage point is one example on
the 100-example validation or test split.

## Configuration

AdaLoRA begins at a larger rank and adaptively prunes toward its target rank.
Configurations A-D cover Q/V-only through full-attention-plus-MLP adapters,
with final ranks 4, 8, 8, and 16 respectively. The adaptive rank schedule and
training settings are defined in
[`finetune_lib/config.py`](../finetune_lib/config.py).

## Results

Values are **validation / test accuracy (%)**.

| Model | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| qwen3-0.6b | 93 / 96 | 94 / 98 | 97 / 97 | 93 / 95 |
| llama3.2-1b | 93 / 96 | 94 / 96 | 92 / 97 | 94 / 96 |

## Findings

- Qwen3-0.6B config B has the best AdaLoRA test result at 98%; config C has
  the best validation result at 97%.
- Llama3.2-1B config C reaches 97% test accuracy, but no configuration exceeds
  that mark in this adaptive-rank experiment.
- The results are tightly clustered, indicating that the smaller adaptive
  target ranks preserve most of the task performance for these two models.

## Charts And Artefacts

- [Training curves](analysis/adalora_training_curves.png)
- [Training summary](analysis/adalora_combined_train.png)
- [Validation summary](analysis/adalora_combined_val.png)
- [Test summary](analysis/adalora_combined_test.png)
- JSON reports: `reports_training/`, `reports_validation/`, and `reports_test/`
