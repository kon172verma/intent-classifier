# Development Notes

This document contains repository-maintenance procedures that are not required
to understand or use the project. For the task, dataset, results, and final
reports, see the [README](readme.md).

## Generate The Full Dataset

Regenerate the 10,000-example dataset (100 files of 100 examples) with the
default deterministic seed:

~~~bash
python dataset_full/generate_dataset_full.py
~~~

To write a generated dataset elsewhere, choose an output directory explicitly:

~~~bash
python dataset_full/generate_dataset_full.py --seed 42 --out-dir dataset_full
~~~

## Synchronize The Experiment Registry

The registry synchronization script compares EXPERIMENTS.jsonl with adapter
folders in the Hugging Face experiments repository. It intentionally ignores
the remote reports/ tree, because those are evaluation artifacts rather than
adapter folders. Set HF_TOKEN before using either command.

Verify that the local registry and remote adapter folders agree:

~~~bash
./scripts/sync_experiments_registry.py --strict
~~~

Add remote adapter folders that are absent from EXPERIMENTS.jsonl and remove
stale or duplicate local registry entries:

~~~bash
./scripts/sync_experiments_registry.py --sync
~~~

Run the strict check again after a synchronization.

## Checkpoints And Reports

LoRA-family training keeps Hugging Face Trainer checkpoints under:

~~~text
finetune_<technique>/tmp/{model}_{config}_{dataset_size}/checkpoint-*
~~~

save_total_limit=2 bounds checkpoint disk use. When load_best_model_at_end=True,
the adapter ultimately saved and pushed is the best validation-loss checkpoint
restored at the end of training, not every intermediate checkpoint.

Local reports are written to the owning experiment's reports_training/,
reports_validation/, and reports_test/ directories. Use the final report in
that directory as the human-readable summary of the corresponding JSON data.
