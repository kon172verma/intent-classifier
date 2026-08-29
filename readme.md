# Intent Classifier

This project fine-tunes small language models (SLMs) to route a natural-language
request to the correct tool from the tools available in that request. It contains
the synthetic dataset, baseline evaluations, PEFT fine-tuning experiments, and
the final reports for each experiment version.

## Repositories

| Repository | Purpose |
| --- | --- |
| [Source](https://github.com/kon172verma/intent-classifier) | Dataset, evaluations, fine-tuning code, reports, and experiment registry |
| [Experiments](https://huggingface.co/kon172verma/intent-classifier-experiments) | Versioned adapter artifacts and uploaded JSON reports |
| [Release](https://huggingface.co/kon172verma/intent-classifier) | Selected merged models for release |
| [Inference](https://github.com/kon172verma/intent-classifier-inference) | Downloads and benchmarks released models on inference engines and edge hardware |

## The Tool-Routing Task

Each example provides a user query and a dynamic, ordered list of available
tools. The model must select the one tool that matches the request, or return
"-" when no listed tool applies. Tool availability and ordering vary by example,
so a model cannot rely on a fixed tool position.

The on-disk record retains the readable answer name:

~~~json
{
  "user_request": "Call my wife.",
  "available_tools": [
    {
      "name": "health_monitor",
      "description": "Monitors driver biometrics such as heart rate and fatigue."
    },
    {
      "name": "insurance_claims",
      "description": "Initiates and tracks vehicle insurance claims."
    },
    {
      "name": "nav_route_planner",
      "description": "Plans routes and navigation."
    },
    {
      "name": "call_handler",
      "description": "Initiates, receives, holds, and terminates hands-free phone calls."
    }
  ],
  "answer": "call_handler"
}
~~~

In the v2 prompt format, the same example is rendered for the model as:

~~~text
You are a tool router.

Available Tools:
ID | Name | Description
a | health_monitor | Monitors driver biometrics such as heart rate and fatigue.
b | insurance_claims | Initiates and tracks vehicle insurance claims.
c | nav_route_planner | Plans routes and navigation.
d | call_handler | Initiates, receives, holds, and terminates hands-free phone calls.

User Request:
Call my wife.

Selected Tool:
~~~

The expected output is "d". IDs are assigned from the displayed tool order
(a-z, then A-Z); "-" is reserved for no matching tool. The production prompt
also instructs the model to return only the ID and not an explanation.

## Dataset

The dataset is synthetic and built from a fixed catalog of 30 automotive,
navigation, communication, personal-assistance, and corporate tools. Each
record has user_request, available_tools, and answer fields.

- dataset_sample/ contains 100 development examples and the canonical tool
  catalog.
- dataset_full/ contains 10,000 globally unique examples in 100 JSON files;
  sample_0001.json is identical to the development sample.
- The full-data experiments use an 80/10/10 split: 8,000 train, 1,000
  validation, and 1,000 test examples.
- About 20% of examples are no-tool cases. Tool-list sizes span 1-3, 4-19, and
  20-30 tools; five tools are deliberately rare answers (about 2-3% each).

Dataset generation and other maintenance commands are documented in
[dev_notes.md](dev_notes.md).

## Experiment Versions And Recommendations

All reported scores are exact-match routing accuracy. v1.0 predicts a readable
tool name; v2.0 and v2.1 use the dynamic positional tool-ID format shown above.

| Version | Scope | Best adapters / completed models |
| --- | --- | --- |
| v2.1 | Full 10k positional-ID DoRA+ experiments, configuration D | Qwen3-0.6B: 99.8% test; Qwen2.5-0.5B: 99.9%; SmolLM2-360M: 99.7%. **Recommended: Qwen2.5-0.5B**, the highest-scoring and fastest of the three (253 ms P50). |
| v2.0 | 1k positional-ID experiment matrix (800/100/100) | **DoRA+ Qwen3-0.6B D** (99% validation, 100% test) and **DoRA+ Llama3.2-1B C** (96% validation, 100% test) are the recommended pair. |
| v1.0 | Initial 1k name-output experiments | **LoRA Qwen3-0.6B C** (99% test) and **LoRA Llama3.2-1B C** (100% test) were the two selected release adapters. |

---

The v2.1 models were evaluated on the shared 1,000-example test split. Their
test scores, on a fixed 100-example subset, are 99.8% for Qwen3-0.6B,
99.9% for Qwen2.5-0.5B, and 99.7% for SmolLM2-360M.

## Fine-Tuning And Final Reports

### v2.1 final report

- [Full-dataset DoRA+ fine-tuning](full_dataset_finetune/final_report.md)

The 1k matrix evaluates six PEFT techniques with shared utilities in
finetune_lib/. Each report includes results, findings, charts, and links to
the underlying training, validation, and test JSON reports.

### v2.0 final reports

- [LoRA](finetune_LoRA/final_report.md)
- [LoRA+](finetune_LoRAplus/final_report.md)
- [DoRA](finetune_DoRA/final_report.md)
- [DoRA+](finetune_DoRAplus/final_report.md)
- [AdaLoRA](finetune_AdaLoRA/final_report.md)
- [QLoRA](finetune_QLoRA/final_report.md)

The baseline shortlist is documented separately in the
[baseline final report](evaluation_baseline/final_report.md).

## Versioning, Artifacts, And Releases

VERSION identifies the active experiment version. Each adapter push is logged
in the append-only EXPERIMENTS.jsonl registry and stored in the experiments
repository at:

~~~text
{version}/{model}_{technique}_{config}_{dataset_size}_{timestamp}
~~~

For example:

~~~text
v2.1/qwen2.5-0.5b_DoRA+_D_10k_20260828-134739
~~~

Training, validation, and test reports are saved next to their experiments in
reports_training/, reports_validation/, and reports_test/. Non-smoke runs also
upload reports to the experiments repository under a versioned reports/ tree.
Every report records the adapter's hf_subfolder, preserving the link between a
metric and the exact adapter that produced it.

A release selects the best adapters for that version, merges each adapter into
its base model, and publishes the resulting full-weight model to the release
repository. The final reports above provide the selection rationale and
evaluation evidence. Operational details, including registry synchronization,
are in [dev_notes.md](dev_notes.md).

## Project Structure

~~~text
dataset_sample/             # 100-example sample, tool catalog, and generator
dataset_full/               # 10,000-example dataset and full-data generator

evaluation_lib/             # Shared prompt rendering, parsing, and evaluation
evaluation_baseline/        # Zero-shot/few-shot baselines and final report

finetune_lib/               # Shared PEFT configuration, training, and registry
finetune_LoRA/              # LoRA 1k experiments and final report
finetune_LoRAplus/          # LoRA+ 1k experiments and final report
finetune_DoRA/              # DoRA 1k experiments and final report
finetune_DoRAplus/          # DoRA+ 1k experiments and final report
finetune_AdaLoRA/           # AdaLoRA 1k experiments and final report
finetune_QLoRA/             # QLoRA 1k experiments and final report
full_dataset_finetune/      # v2.1 10k DoRA+ experiments, charts, final report

scripts/                    # Repository maintenance helpers
EXPERIMENTS.jsonl           # Append-only adapter registry
VERSION                     # Active experiment version
dev_notes.md                # Development and maintenance procedures
~~~
