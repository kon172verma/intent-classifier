#!/usr/bin/env python3
"""Generate charts for the full-dataset DoRA+ experiments.

Outputs
-------
full_dataset_combined_metrics.png
    One grouped-bar chart with train/validation/test/anchor accuracy and peak
    memory for Qwen3-0.6B, Qwen2.5-0.5B, and SmolLM2-360M.

full_dataset_training_curves.png
    One row of training-curve panels, one per model.

The script reads only report files stored under full_dataset_finetune/. It is
deliberately independent of the technique-level plotting scripts because this
directory contains one selected configuration per model rather than a grid of
every model/configuration pair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RUN_DIR = Path(__file__).parent
DEFAULT_TRAIN_DIR = RUN_DIR / "reports_training"
DEFAULT_VAL_DIR = RUN_DIR / "reports_validation"
DEFAULT_TEST_DIR = RUN_DIR / "reports_test"
DEFAULT_OUT_DIR = RUN_DIR / "analysis"

MODELS = ["qwen3-0.6b", "qwen2.5-0.5b", "smollm2-360m"]
MODEL_LABELS = {
    "qwen3-0.6b": "Qwen3-0.6B",
    "qwen2.5-0.5b": "Qwen2.5-0.5B",
    "smollm2-360m": "SmolLM2-360M",
}

ACCURACY_SERIES = [
    ("train", "Train accuracy", "#4C72B0"),
    ("val", "Validation accuracy", "#55A868"),
    ("test", "Test accuracy", "#DD8452"),
    ("test_anchor", "Test-anchor accuracy", "#8172B2"),
]
MEMORY_COLOR = "#B8D5E8"
MEMORY_HATCH = "///"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-reports-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--val-reports-dir", type=Path, default=DEFAULT_VAL_DIR)
    parser.add_argument("--test-reports-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _read_reports(reports_dir: Path) -> list[dict[str, Any]]:
    """Read valid 10k DoRA+ report JSONs, ignoring unrelated files."""
    reports: list[dict[str, Any]] = []
    for path in sorted(reports_dir.glob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if report.get("dataset_size") == "10k" and report.get("technique") == "DoRA+":
            reports.append(report)
    return reports


def _latest_by_model(reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the newest report for each selected model."""
    latest: dict[str, dict[str, Any]] = {}
    for report in reports:
        model = report.get("model_key")
        if model not in MODELS:
            continue
        if model not in latest or str(report.get("timestamp", "")) > str(
            latest[model].get("timestamp", "")
        ):
            latest[model] = report
    return latest


def load_reports(
    train_reports_dir: Path,
    val_reports_dir: Path,
    test_reports_dir: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Load the latest report of each required split for every selected model."""
    train = _latest_by_model(_read_reports(train_reports_dir))
    val = _latest_by_model(_read_reports(val_reports_dir))

    test_reports = _read_reports(test_reports_dir)
    test = _latest_by_model([r for r in test_reports if r.get("split") == "test"])
    anchor = _latest_by_model(
        [r for r in test_reports if r.get("split") == "test_anchor"]
    )

    return {
        model: {
            split: reports[model]
            for split, reports in {
                "train": train,
                "val": val,
                "test": test,
                "test_anchor": anchor,
            }.items()
            if model in reports
        }
        for model in MODELS
    }


def _accuracy(report: dict[str, Any] | None, split: str) -> float | None:
    if report is None:
        return None
    value = report.get("final_train_accuracy") if split == "train" else report.get("accuracy")
    return float(value) * 100 if value is not None else None


def _max_memory(reports: dict[str, dict[str, Any]]) -> float | None:
    values = [
        float(report["peak_memory_mb"])
        for report in reports.values()
        if report.get("peak_memory_mb") is not None
    ]
    return max(values) if values else None


def _label_bar(axis: Any, bar: Any, value: float, text: str) -> None:
    """Place a compact value label immediately above one bar."""
    axis.annotate(
        text,
        xy=(bar.get_x() + bar.get_width() / 2, value),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7,
        fontweight="bold",
    )


def plot_combined_metrics(
    reports_by_model: dict[str, dict[str, dict[str, Any]]], out_dir: Path
) -> Path:
    """Plot four solid accuracy bars and one hatched peak-memory bar per model."""
    x = np.arange(len(MODELS))
    bar_width = 0.14
    offsets = np.array([-2, -1, 0, 1, 2]) * bar_width
    memory_offset = offsets[-1] + 0.07

    fig, ax_accuracy = plt.subplots(figsize=(15, 7))
    ax_memory = ax_accuracy.twinx()

    for index, (split, label, color) in enumerate(ACCURACY_SERIES):
        values = [_accuracy(reports_by_model[model].get(split), split) for model in MODELS]
        bars = ax_accuracy.bar(
            x + offsets[index],
            [value if value is not None else 0 for value in values],
            bar_width,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        for bar, value in zip(bars, values):
            if value is not None:
                _label_bar(ax_accuracy, bar, value, f"{value:.1f}%")

    memory_values = [_max_memory(reports_by_model[model]) for model in MODELS]
    memory_bars = ax_memory.bar(
        x + memory_offset,
        [value if value is not None else 0 for value in memory_values],
        bar_width,
        color=MEMORY_COLOR,
        edgecolor="#4B6B82",
        linewidth=0.8,
        hatch=MEMORY_HATCH,
        zorder=3,
    )
    for bar, value in zip(memory_bars, memory_values):
        if value is not None:
            _label_bar(ax_memory, bar, value, f"{value:,.0f} MB")

    for model_index, model in enumerate(MODELS):
        if not reports_by_model[model]:
            ax_accuracy.text(
                x[model_index],
                3,
                "Awaiting reports",
                ha="center",
                va="bottom",
                fontsize=8,
                style="italic",
                color="#666666",
            )

    ax_accuracy.set_xticks(x)
    ax_accuracy.set_xticklabels([MODEL_LABELS[model] for model in MODELS], fontsize=10)
    ax_accuracy.set_xlabel("Model", fontsize=11)
    ax_accuracy.set_ylabel("Accuracy (%)", fontsize=11)
    ax_accuracy.set_ylim(0, 110)
    ax_accuracy.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax_accuracy.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax_accuracy.set_axisbelow(True)
    ax_accuracy.spines["top"].set_visible(False)

    max_memory = max((value or 0 for value in memory_values), default=0)
    ax_memory.set_ylabel("Peak memory (MB)", fontsize=11)
    # A 100% accuracy bar occupies 100 / 110 of the left axis. Give the
    # highest memory bar a little more visual height while retaining room for
    # its value label above the bar.
    ax_memory.set_ylim(0, max(1000, max_memory / 0.95))
    ax_memory.spines["top"].set_visible(False)

    legend_handles = [
        Patch(facecolor=color, label=label) for _, label, color in ACCURACY_SERIES
    ] + [
        Patch(
            facecolor=MEMORY_COLOR,
            edgecolor="#4B6B82",
            hatch=MEMORY_HATCH,
            label="Peak memory (max across train/evaluation)",
        )
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=5,
        fontsize=9,
        frameon=True,
        framealpha=0.95,
    )
    fig.suptitle(
        "Full 10k DoRA+ Results — Accuracy and Peak Memory",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82))

    out_path = out_dir / "full_dataset_combined_metrics.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_training_curves(
    reports_by_model: dict[str, dict[str, dict[str, Any]]], out_dir: Path
) -> Path:
    """Plot a one-row, three-model loss-and-accuracy training curve figure."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(18, 5.5), squeeze=False)

    for axis, model in zip(axes[0], MODELS):
        report = reports_by_model[model].get("train")
        axis.set_title(MODEL_LABELS[model], fontsize=11, fontweight="bold", pad=8)
        if report is None:
            axis.text(
                0.5,
                0.5,
                "Awaiting training report",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                style="italic",
                color="#666666",
            )
            axis.set_axis_off()
            continue

        history = report.get("log_history", [])
        train_entries = [entry for entry in history if "loss" in entry and "eval_loss" not in entry]
        eval_entries = [entry for entry in history if "eval_loss" in entry]

        if train_entries:
            axis.plot(
                [entry["step"] for entry in train_entries],
                [entry["loss"] for entry in train_entries],
                color="#4C72B0",
                alpha=0.75,
                linewidth=1.4,
            )
        if eval_entries:
            eval_steps = [entry["step"] for entry in eval_entries]
            axis.plot(
                eval_steps,
                [entry["eval_loss"] for entry in eval_entries],
                color="#C44E52",
                marker="o",
                markersize=4,
                linewidth=1.7,
            )
            if eval_entries[0].get("step") == 0:
                axis.plot(
                    0,
                    eval_entries[0]["eval_loss"],
                    marker="*",
                    color="#C44E52",
                    markersize=10,
                    zorder=5,
                )
        else:
            eval_steps = []

        accuracy_axis = axis.twinx()
        train_accuracy = [
            (entry["step"], entry["train_accuracy"])
            for entry in eval_entries
            if entry.get("train_accuracy") is not None
        ]
        val_accuracy = [
            (entry["step"], entry["eval_accuracy"])
            for entry in eval_entries
            if entry.get("eval_accuracy") is not None
        ]
        if train_accuracy:
            steps, values = zip(*train_accuracy)
            accuracy_axis.plot(
                steps,
                [value * 100 for value in values],
                color="#4C72B0",
                linestyle="--",
                marker="s",
                markersize=3.5,
                linewidth=1.4,
            )
        if val_accuracy:
            steps, values = zip(*val_accuracy)
            accuracy_axis.plot(
                steps,
                [value * 100 for value in values],
                color="#55A868",
                linestyle="--",
                marker="^",
                markersize=3.5,
                linewidth=1.4,
            )

        config = report.get("lora_config", "?")
        axis.set_title(
            f"{MODEL_LABELS[model]} — config {config}",
            fontsize=11,
            fontweight="bold",
            pad=8,
        )
        axis.set_xlabel("Step", fontsize=9)
        axis.set_ylabel("Loss", fontsize=9)
        accuracy_axis.set_ylabel("Accuracy (%)", fontsize=9)
        accuracy_axis.set_ylim(0, 110)
        axis.grid(True, axis="y", linestyle="--", alpha=0.35)
        axis.spines["top"].set_visible(False)
        accuracy_axis.spines["top"].set_visible(False)
        axis.tick_params(axis="both", labelsize=8)
        accuracy_axis.tick_params(axis="y", labelsize=8)

    legend_handles = [
        Line2D([0], [0], color="#4C72B0", linewidth=1.5, label="Train loss"),
        Line2D(
            [0], [0], color="#C44E52", marker="o", markersize=4, linewidth=1.5, label="Validation loss"
        ),
        Line2D(
            [0], [0], color="#4C72B0", linestyle="--", marker="s", markersize=3.5, label="Train accuracy (%)"
        ),
        Line2D(
            [0], [0], color="#55A868", linestyle="--", marker="^", markersize=3.5, label="Validation accuracy (%)"
        ),
        Line2D(
            [0], [0], color="#C44E52", marker="*", linestyle="None", markersize=9, label="Step-0 baseline"
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=5,
        fontsize=8.5,
        frameon=True,
        framealpha=0.95,
    )
    fig.suptitle(
        "Full 10k DoRA+ Training Curves — Loss and Accuracy",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.80))

    out_path = out_dir / "full_dataset_training_curves.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reports_by_model = load_reports(
        args.train_reports_dir,
        args.val_reports_dir,
        args.test_reports_dir,
    )

    completed = [MODEL_LABELS[model] for model in MODELS if reports_by_model[model]]
    pending = [MODEL_LABELS[model] for model in MODELS if not reports_by_model[model]]
    print(f"Completed models: {', '.join(completed) or 'none'}")
    if pending:
        print(f"Awaiting reports: {', '.join(pending)}")

    print(f"Saved → {plot_combined_metrics(reports_by_model, args.out_dir)}")
    print(f"Saved → {plot_training_curves(reports_by_model, args.out_dir)}")


if __name__ == "__main__":
    main()
