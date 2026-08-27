# QLoRA v2.0 Final Report

## Scope

This report covers the completed v2.0 QLoRA experiments for Qwen3-0.6B and
Llama3.2-1B on the 1k split (800 train / 100 validation / 100 test). QLoRA
uses the same dynamic positional-ID task as the other experiments: it trains
and evaluates the adapter on the rendered tool ID rather than the tool name.

All values are exact-match ID accuracy. One percentage point represents one
example on either 100-example held-out split.

## Configuration

QLoRA loads the frozen base model in 4-bit NF4 and trains LoRA adapters. The
completed matrix contains configurations A-C; configuration D was not run.
The adapter definitions are shared with LoRA and live in
[`finetune_lib/config.py`](../finetune_lib/config.py).

## Results

Values are **validation / test accuracy (%)**.

| Model | A | B | C |
| --- | ---: | ---: | ---: |
| qwen3-0.6b | 95 / 96 | 99 / 96 | 99 / 97 |
| llama3.2-1b | 95 / 97 | 97 / 97 | 95 / 96 |

## Findings

- Llama3.2-1B configurations A and B both reach 97% test accuracy. Config B
  also has the strongest validation result for that model at 97%.
- Qwen3-0.6B config C is the strongest QLoRA test run at 97%, while B and C
  each reach 99% validation accuracy.
- QLoRA delivers these results with much lower inference memory than the
  full-precision adapter variants: about 638-688 MB for Qwen3-0.6B and
  1.09-1.15 GB for Llama3.2-1B.

## Charts And Artefacts

- [Training curves](analysis/qlora_training_curves.png)
- [Training summary](analysis/qlora_combined_train.png)
- [Validation summary](analysis/qlora_combined_val.png)
- [Test summary](analysis/qlora_combined_test.png)
- JSON reports: `reports_training/`, `reports_validation/`, and `reports_test/`
