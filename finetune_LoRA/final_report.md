# PEFT (LoRA) Evaluation Report

Date: 2026-06-18
Status: Initialized

## Dataset Subset for PEFT

Using the first 10 files from `dataset_full/`:

- `sample_0001.json` to `sample_0010.json`
- Each file has 100 examples
- Total examples: 1000

Split policy (80-10-10):

- Test: `sample_0001.json` (100 examples)
- Validation: `sample_0010.json` (100 examples)
- Train: `sample_0002.json` to `sample_0009.json` (800 examples)

## Planned Outputs

- Validation reports in `reports_validation/`
- Test reports in `reports_test/`
- Analysis charts in `analysis/`

## Notes

- This phase focuses on PEFT with LoRA.
- Follow the same prompt schema used in baseline and quantized evaluation for fair comparison.
