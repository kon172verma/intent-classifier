# Intent Classifier: Fine-Tuning Experiments

Train a Small Language Model (SLM) to route a user request to the right tool
from a dynamic list of available tools.

In v2.0 the dataset schema stays the same, but prompts are rendered with
positional tool IDs. The model now predicts the tool ID, not the tool name. If
no tool matches, it predicts `-`.

## Related Repositories

- This repo: `github.com/kon172verma/intent-classifier`
- Experiments repo: `huggingface.co/kon172verma/intent-classifier-experiments`
- Release repo: `huggingface.co/kon172verma/intent-classifier`
- Inference repo: `github.com/kon172verma/intent-classifier-inference`

## Version 2.0

v2.0 keeps the same examples and answer names in `dataset_sample/` and
`dataset_full/`, but changes how examples are presented to the model.

Why v2.0 exists:

- Shorter outputs: `a` is cheaper to generate and parse than `call_handler`.
- Cleaner labels: the model only has to emit one of `a-z`, `A-Z`, or `-`.
- Dynamic tools still work: IDs are assigned per example from the listed order.
- Smaller benchmark decode budget: evaluation and validation use fewer tokens.

The current ID scheme supports up to 52 tools per example:

```text
a-z, A-Z
```

`-` is reserved for "no valid tool".

## Dataset

A synthetic dataset of tool-routing examples generated from a fixed catalog of
30 tools.

Schema:

```json
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
```

Important detail: `answer` remains the readable tool name or `none` on disk.
During training and inference, the prompt renderer maps it to the positional ID
shown in that example.

Key properties:

- About 20% of examples have `none` as the answer.
- Five rare tools appear as the correct answer in only 2-3% of examples.
- Available-tools count spans few-tool, standard, and many-tool regimes.
- Split: 80% train, 10% validation, 10% test.

Dataset folders:

- `dataset_sample/`: 100-example sample used during development.
- `dataset_full/`: full-scale dataset.

## Prompt Format

Prompt order is always:

```text
system prompt -> available tools -> user request -> selected tool
```

System prompt:

```text
You are a tool router.

Available tools are listed with id, name, and description.

Rules:
- Return only the tool id.
- Use the id from the available tools list.
- Return "-" if no tool matches.
- Do not explain.
```

User message body:

```text
Available Tools:
ID | Name | Description
a | call_handler | Makes hands-free phone calls.
b | nav_route_planner | Plans routes and navigation.
c | sms_messenger | Reads and sends short SMS messages.

User Request:
Call my wife.

Selected Tool:
```

The expected assistant output for this example is:

```text
a
```

## Qwen3-0.6B Template Example

Before `apply_chat_template`, a training example is represented as messages:

```python
[
    {
        "role": "system",
        "content": (
            "You are a tool router.\n\n"
            "Available tools are listed with id, name, and description.\n\n"
            "Rules:\n"
            "- Return only the tool id.\n"
            "- Use the id from the available tools list.\n"
            "- Return \"-\" if no tool matches.\n"
            "- Do not explain."
        ),
    },
    {
        "role": "user",
        "content": (
            "Available Tools:\n"
            "ID | Name | Description\n"
            "a | call_handler | Makes hands-free phone calls.\n"
            "b | nav_route_planner | Plans routes and navigation.\n"
            "c | sms_messenger | Reads and sends short SMS messages.\n\n"
            "User Request:\n"
            "Call my wife.\n\n"
            "Selected Tool:"
        ),
    },
    {
        "role": "assistant",
        "content": "a",
    },
]
```

After Qwen3-0.6B `apply_chat_template(..., enable_thinking=False)`:

```text
<|im_start|>system
You are a tool router.

Available tools are listed with id, name, and description.

Rules:
- Return only the tool id.
- Use the id from the available tools list.
- Return "-" if no tool matches.
- Do not explain.<|im_end|>
<|im_start|>user
Available Tools:
ID | Name | Description
a | call_handler | Makes hands-free phone calls.
b | nav_route_planner | Plans routes and navigation.
c | sms_messenger | Reads and sends short SMS messages.

User Request:
Call my wife.

Selected Tool:<|im_end|>
<|im_start|>assistant
<think>

</think>

a<|im_end|>
```

For inference, the same system and user messages are rendered with
`add_generation_prompt=True`; the model then generates the final ID.

## Baseline Evaluation

Zero-shot and few-shot baselines live in `evaluation_baseline/`. Shared prompt
construction, parsing, and report schemas live in `evaluation_lib/`.

Metric: exact-match accuracy on the predicted tool ID.

The baseline matrix includes models from 70M to 3B parameters. The final
fine-tuning shortlist is documented in `evaluation_baseline/final_report.md`.

## Fine-Tuning

Six PEFT techniques are implemented:

- LoRA: `finetune_LoRA/`
- LoRA+: `finetune_LoRAplus/`
- DoRA: `finetune_DoRA/`
- DoRA+: `finetune_DoRAplus/`
- AdaLoRA: `finetune_AdaLoRA/`
- QLoRA: `finetune_QLoRA/`

Shared fine-tuning utilities live in `finetune_lib/`. All PEFT training paths
use the same v2 prompt renderer and train on the ID label.

Adapters are pushed to the experiments repo under:

```text
{version}/{model}_{technique}_{config}_{dataset_size}_{timestamp}
```

Example:

```text
v2.0/qwen3-0.6b_LoRA_C_1k_20260715-044041
```

`EXPERIMENTS.jsonl` is the append-only index of adapter pushes.

## Checkpoints And Reports

LoRA-family training writes Hugging Face Trainer checkpoints locally under:

```text
finetune_<technique>/tmp/{model}_{config}_{dataset_size}/checkpoint-*
```

The trainer keeps disk usage bounded with `save_total_limit=2`. With
`load_best_model_at_end=True`, the saved and pushed adapter is the best
validation-loss adapter restored at the end, not every intermediate checkpoint.

Training, validation, and test JSON reports are written locally under each
technique folder:

```text
reports_training/
reports_validation/
reports_test/
```

For non-smoke runs, scripts also upload JSON reports to a central reports tree
in the experiments repo. Adapter files stay in their adapter run folders, while
reports are grouped by version, technique, and report type:

```text
reports/
  v2.0/
    dora/
      reports_training/
      reports_validation/
      reports_test/
    loraplus/
      reports_training/
      reports_validation/
      reports_test/
```

Each JSON report still includes `hf_subfolder`, which points back to the exact
adapter run that produced it. This keeps reports easy to fetch after a Colab
disconnect without mixing them into adapter artifact folders.

## Versioning And Release

`VERSION` holds the current experiment version. Bumping it starts a new round;
subsequent adapter pushes land in the matching version folder.

After release evaluation, the best adapters are merged and pushed to the
release repo.

## Project Structure

```text
dataset_sample/        # 100-example sample
dataset_full/          # Full dataset
evaluation_lib/        # Shared evaluation utilities
evaluation_baseline/   # Zero-shot and few-shot baseline results
finetune_lib/          # Shared training utilities and registry
finetune_LoRA/         # LoRA experiments
finetune_LoRAplus/     # LoRA+ experiments
finetune_DoRA/         # DoRA experiments
finetune_DoRAplus/     # DoRA+ experiments
finetune_AdaLoRA/      # AdaLoRA experiments
finetune_QLoRA/        # QLoRA experiments
evaluation_release/    # Deep evaluation for release candidates
scripts/               # Helper scripts
EXPERIMENTS.jsonl      # Append-only log of adapter pushes
VERSION                # Current experiment version
```
