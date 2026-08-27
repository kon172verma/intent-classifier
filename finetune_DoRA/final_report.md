# DoRA v2.0 Final Report

## Scope

This report covers the completed v2.0 DoRA matrix: five base models, four
configurations, and the 1k split (800 train / 100 validation / 100 test). The
shared v2.0 renderer dynamically maps the correct tool name to a positional ID;
the adapter is trained and evaluated on that ID output.

All values are exact-match ID accuracy. A one-point percentage difference is
one example on the held-out 100-example splits.

## Configuration

DoRA applies weight-decomposed low-rank adaptation (`use_dora=True`) with the
same A-D adapter scopes as LoRA. The shared configuration definitions are in
[`finetune_lib/config.py`](../finetune_lib/config.py).

## Results

Values are **validation / test accuracy (%)**.

| Model | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| smollm2-360m | 49 / 46 | 51 / 53 | 58 / 56 | 56 / 55 |
| qwen2.5-0.5b | 63 / 66 | 68 / 72 | 73 / 76 | 72 / 77 |
| qwen3-0.6b | 97 / 98 | 97 / 98 | 96 / 98 | 98 / 98 |
| llama3.2-1b | 93 / 93 | 95 / 99 | 97 / 99 | 96 / 99 |
| smollm2-1.7b | 93 / 97 | 94 / 96 | 95 / 100 | 94 / 96 |

## Findings

- DoRA is highly competitive for the larger models: Llama3.2-1B reaches 99%
  with configurations B-D, and SmolLM2-1.7B config C reaches 100%.
- Qwen3-0.6B is stable at 98% across all four configurations, so adapter scope
  has little effect for that model in this 1k experiment.
- The 360M model remains capacity-limited; config C is its best DoRA result at
  56% test accuracy.

## Charts And Artefacts

- [Training curves](analysis/dora_training_curves.png)
- [Training summary](analysis/dora_combined_train.png)
- [Validation summary](analysis/dora_combined_val.png)
- [Test summary](analysis/dora_combined_test.png)
- JSON reports: `reports_training/`, `reports_validation/`, and `reports_test/`
