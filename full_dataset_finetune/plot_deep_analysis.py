#!/usr/bin/env python3
"""Create deep-analysis charts from full-dataset DoRA+ test reports.

Outputs
-------
deep_per_tool_f1_heatmap.png
    Per-tool F1 across the three models, with test support beside each tool.
deep_error_flows.png
    Expected-to-predicted error pairs, separated into test-anchor and other
    full-test examples.
deep_latency_ecdf.png
    Test-prediction latency distributions with P50 and P95 markers.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RUN_DIR = Path(__file__).parent
DEFAULT_TEST_DIR = RUN_DIR / "reports_test"
DEFAULT_OUT_DIR = RUN_DIR / "analysis"

MODELS = ["qwen3-0.6b", "qwen2.5-0.5b", "smollm2-360m"]
MODEL_LABELS = {
    "qwen3-0.6b": "Qwen3-0.6B",
    "qwen2.5-0.5b": "Qwen2.5-0.5B",
    "smollm2-360m": "SmolLM2-360M",
}
MODEL_COLORS = {
    "qwen3-0.6b": "#4C72B0",
    "qwen2.5-0.5b": "#DD8452",
    "smollm2-360m": "#55A868",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-reports-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _read_reports(reports_dir: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in sorted(reports_dir.glob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if report.get("dataset_size") == "10k" and report.get("technique") == "DoRA+":
            reports.append(report)
    return reports


def _latest_by_model(
    reports: list[dict[str, Any]], split: str
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for report in reports:
        model = report.get("model_key")
        if report.get("split") != split or model not in MODELS:
            continue
        if model not in latest or str(report.get("timestamp", "")) > str(
            latest[model].get("timestamp", "")
        ):
            latest[model] = report
    return latest


def load_test_reports(
    reports_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    reports = _read_reports(reports_dir)
    return _latest_by_model(reports, "test"), _latest_by_model(reports, "test_anchor")


def _tool_order(full_reports: dict[str, dict[str, Any]]) -> list[str]:
    tools = sorted(
        {
            tool
            for report in full_reports.values()
            for tool in report.get("per_tool_metrics", {})
        }
    )

    def sort_key(tool: str) -> tuple[float, int, str]:
        metrics = [
            report["per_tool_metrics"][tool]
            for report in full_reports.values()
            if tool in report.get("per_tool_metrics", {})
        ]
        return (min(metric["f1"] for metric in metrics), -max(metric["support"] for metric in metrics), tool)

    return sorted(tools, key=sort_key)


def plot_per_tool_f1(full_reports: dict[str, dict[str, Any]], out_dir: Path) -> Path:
    """Render annotated per-tool F1 values plus the common test support."""
    tools = _tool_order(full_reports)
    if not tools:
        raise ValueError("No per-tool metrics found in the full-test reports.")

    values = np.full((len(tools), len(MODELS)), np.nan)
    support = np.zeros(len(tools), dtype=int)
    for row, tool in enumerate(tools):
        for col, model in enumerate(MODELS):
            metric = full_reports.get(model, {}).get("per_tool_metrics", {}).get(tool)
            if metric is not None:
                values[row, col] = float(metric["f1"])
                support[row] = max(support[row], int(metric["support"]))

    figure_height = max(10, len(tools) * 0.34 + 2.2)
    fig, (ax_support, ax_heatmap) = plt.subplots(
        1,
        2,
        figsize=(12, figure_height),
        gridspec_kw={"width_ratios": [1.25, 4.75], "wspace": 0.04},
        sharey=True,
    )
    y = np.arange(len(tools))
    ax_support.barh(y, support, color="#B7C7D6", edgecolor="white", linewidth=0.5)
    ax_support.set_yticks(y)
    ax_support.set_yticklabels(tools, fontsize=8)
    ax_support.invert_yaxis()
    ax_support.set_xlabel("Test support", fontsize=9)
    ax_support.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax_support.set_axisbelow(True)
    ax_support.spines[["top", "right"]].set_visible(False)

    image = ax_heatmap.imshow(
        values,
        cmap="RdYlGn",
        norm=Normalize(vmin=0.80, vmax=1.00, clip=True),
        aspect="auto",
    )
    ax_heatmap.set_xticks(np.arange(len(MODELS)))
    ax_heatmap.set_xticklabels([MODEL_LABELS[model] for model in MODELS], fontsize=10)
    ax_heatmap.tick_params(axis="y", left=False, labelleft=False)
    ax_heatmap.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False)
    ax_heatmap.set_xticks(np.arange(-0.5, len(MODELS), 1), minor=True)
    ax_heatmap.set_yticks(np.arange(-0.5, len(tools), 1), minor=True)
    ax_heatmap.grid(which="minor", color="white", linewidth=0.8)
    ax_heatmap.tick_params(which="minor", bottom=False, left=False)

    for row in range(len(tools)):
        for col in range(len(MODELS)):
            if not np.isnan(values[row, col]):
                color = "white" if values[row, col] < 0.90 else "#202020"
                ax_heatmap.text(
                    col,
                    row,
                    f"{values[row, col] * 100:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    fontweight="bold",
                    color=color,
                )

    colorbar = fig.colorbar(image, ax=ax_heatmap, fraction=0.035, pad=0.02)
    colorbar.set_label("F1 score", fontsize=9)
    fig.suptitle(
        "Full 10k Test — Per-Tool F1 by Model",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.006,
        "Rows are sorted by the lowest F1 across models. Support is the number of expected examples in the shared 1,000-example test set.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    fig.subplots_adjust(left=0.20, right=0.91, top=0.88, bottom=0.055, wspace=0.04)

    out_path = out_dir / "deep_per_tool_f1_heatmap.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _error_counts(report: dict[str, Any]) -> Counter[tuple[str, str]]:
    return Counter(
        (result["answer"], result["prediction_tool"])
        for result in report.get("results", [])
        if not result.get("correct", False)
    )


def plot_error_flows(
    full_reports: dict[str, dict[str, Any]],
    anchor_reports: dict[str, dict[str, Any]],
    out_dir: Path,
) -> Path:
    """Plot error pairs and identify the subset reproduced by test_anchor."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(17, 5.8), squeeze=False)
    max_error_count = 1
    per_model: dict[str, tuple[Counter[tuple[str, str]], Counter[tuple[str, str]]]] = {}
    for model in MODELS:
        full_errors = _error_counts(full_reports.get(model, {}))
        anchor_errors = _error_counts(anchor_reports.get(model, {}))
        per_model[model] = (full_errors, anchor_errors)
        if full_errors:
            max_error_count = max(max_error_count, max(full_errors.values()))

    for axis, model in zip(axes[0], MODELS):
        full_errors, anchor_errors = per_model[model]
        axis.set_title(MODEL_LABELS[model], fontsize=11, fontweight="bold", pad=8)
        if not full_errors:
            axis.text(
                0.5,
                0.5,
                "No full-test errors",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                style="italic",
                color="#666666",
            )
            axis.set_axis_off()
            continue

        pairs = sorted(full_errors, key=lambda pair: (-full_errors[pair], pair))
        labels = [f"{expected}\n→ {predicted}" for expected, predicted in pairs]
        anchor_values = np.array([anchor_errors[pair] for pair in pairs])
        other_values = np.array([full_errors[pair] - anchor_errors[pair] for pair in pairs])
        y = np.arange(len(pairs))
        other_bars = axis.barh(y, other_values, color="#DD8452", label="Other test cases")
        anchor_bars = axis.barh(
            y,
            anchor_values,
            left=other_values,
            color="#8172B2",
            hatch="///",
            edgecolor="#5F5387",
            label="Test-anchor cases",
        )
        for bar, other, anchor in zip(other_bars, other_values, anchor_values):
            total = int(other + anchor)
            axis.text(total + 0.03, bar.get_y() + bar.get_height() / 2, str(total), va="center", fontsize=9, fontweight="bold")

        axis.set_yticks(y)
        axis.set_yticklabels(labels, fontsize=8)
        axis.invert_yaxis()
        axis.set_xlim(0, max_error_count + 0.45)
        axis.set_xticks(range(max_error_count + 1))
        axis.set_xlabel("Incorrect predictions", fontsize=9)
        axis.xaxis.grid(True, linestyle="--", alpha=0.3)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)

    fig.legend(
        handles=[
            Patch(facecolor="#DD8452", label="Other full-test cases"),
            Patch(facecolor="#8172B2", edgecolor="#5F5387", hatch="///", label="Test-anchor cases"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=2,
        fontsize=9,
        frameon=True,
    )
    fig.suptitle(
        "Full 10k Test — Error Flows by Model",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82))

    out_path = out_dir / "deep_error_flows.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_latency_ecdf(full_reports: dict[str, dict[str, Any]], out_dir: Path) -> Path:
    """Plot the full-test latency CDF, including P50 and P95 markers."""
    fig, axis = plt.subplots(figsize=(11.5, 6.5))
    for model in MODELS:
        report = full_reports.get(model)
        if report is None:
            continue
        latencies = np.sort(
            [float(result["latency_s"]) * 1000 for result in report.get("results", [])]
        )
        if not len(latencies):
            continue
        cumulative = np.arange(1, len(latencies) + 1) / len(latencies) * 100
        color = MODEL_COLORS[model]
        p50, p95 = np.percentile(latencies, [50, 95])
        axis.step(
            latencies,
            cumulative,
            where="post",
            color=color,
            linewidth=2,
            label=f"{MODEL_LABELS[model]}  (P50 {p50:.0f} ms, P95 {p95:.0f} ms)",
        )
        axis.scatter([p50], [50], color=color, marker="o", s=38, zorder=5)
        axis.scatter([p95], [95], color=color, marker="^", s=44, zorder=5)

    axis.set_xlabel("Per-prediction latency (ms)", fontsize=11)
    axis.set_ylabel("Cumulative test predictions (%)", fontsize=11)
    axis.set_ylim(0, 102)
    axis.set_yticks(np.arange(0, 101, 10))
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.spines[["top", "right"]].set_visible(False)
    model_legend = axis.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.95)
    axis.add_artist(model_legend)
    axis.legend(
        handles=[
            Line2D([0], [0], color="#555555", marker="o", linestyle="None", label="P50"),
            Line2D([0], [0], color="#555555", marker="^", linestyle="None", label="P95"),
        ],
        loc="upper left",
        fontsize=9,
        frameon=True,
    )
    fig.suptitle(
        "Full 10k Test — Prediction Latency Distribution",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()

    out_path = out_dir / "deep_latency_ecdf.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    full_reports, anchor_reports = load_test_reports(args.test_reports_dir)
    missing = [MODEL_LABELS[model] for model in MODELS if model not in full_reports]
    if missing:
        raise ValueError(f"Missing full-test reports for: {', '.join(missing)}")

    print(f"Saved → {plot_per_tool_f1(full_reports, args.out_dir)}")
    print(f"Saved → {plot_error_flows(full_reports, anchor_reports, args.out_dir)}")
    print(f"Saved → {plot_latency_ecdf(full_reports, args.out_dir)}")


if __name__ == "__main__":
    main()
