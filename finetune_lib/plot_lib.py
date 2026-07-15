"""
finetune_lib/plot_lib.py
========================
Shared plotting functions for all fine-tuning experiments.

Called by each technique's plot script (e.g. plot_lora_results.py,
plot_qlora_results.py) with the `technique` name for titles and filenames.

Figures produced:
  {technique_lower}_training_curves.png
      5 rows (one per model) × 4 cols (one per config).
      Each panel: train_loss, val_loss (left y-axis),
                  train_accuracy, val_accuracy (right y-axis).
      Step 0 marks the pre-finetuning baseline.

  {technique_lower}_combined.png
      2 rows: final test accuracy (%) and peak inference memory (MB).
      X-axis: all 5 models; 4 grouped bars per model (one per config).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import Patch

# ── Colour palette ────────────────────────────────────────────────────────────
CONFIG_COLORS: dict[str, str] = {
    "A": "#4C72B0",  # blue
    "B": "#55A868",  # green
    "C": "#C44E52",  # red
    "D": "#8172B2",  # purple
}
CONFIG_LABELS: dict[str, str] = {
    "A": "Config A (light, Q+V, r=8)",
    "B": "Config B (standard, full-attn, r=16)",
    "C": "Config C (wide, attn+MLP, r=16)",
    "D": "Config D (heavy, attn+MLP, r=32)",
}

# ── Report loaders ────────────────────────────────────────────────────────────


def _load_latest(reports_dir: Path, key_fn) -> dict[str, dict]:
    """Load the most-recent JSON per key returned by key_fn(report_dict)."""
    latest: dict[str, dict] = {}
    for f in sorted(reports_dir.glob("*.json")):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            k = key_fn(r)
            if k not in latest or r.get("timestamp", "") > latest[k].get(
                "timestamp", ""
            ):
                latest[k] = r
        except Exception:
            continue
    return latest


def load_train_reports(reports_dir: Path) -> dict[tuple[str, str], dict]:
    """Return {(model_key, lora_config): report} for training reports."""
    raw = _load_latest(
        reports_dir,
        lambda r: (
            f"{r.get('model_key', '?')}_{r.get('lora_config', '?')}_{r.get('dataset_size', '?')}"
        ),
    )
    out: dict[tuple[str, str], dict] = {}
    for r in raw.values():
        out[(r["model_key"], r["lora_config"])] = r
    return out


def load_eval_reports(reports_dir: Path) -> dict[tuple[str, str], dict]:
    """Return {(model_key, lora_config): report} for eval reports."""
    raw = _load_latest(
        reports_dir,
        lambda r: (
            f"{r.get('model_key', '?')}_{r.get('lora_config', '?')}_{r.get('dataset_size', '?')}"
        ),
    )
    out: dict[tuple[str, str], dict] = {}
    for r in raw.values():
        out[(r["model_key"], r["lora_config"])] = r
    return out


# ── Plot 1: Training curves (5 rows × 4 cols) ─────────────────────────────────


def plot_training_curves(
    train_reports_dir: Path,
    all_models: list[str],
    all_configs: list[str],
    out_dir: Path,
    technique: str = "LoRA",
) -> None:
    """
    5 rows (models) × 4 cols (configs) grid.

    Each panel shows 4 curves:
      Left y-axis  (solid)  : train_loss (blue), val_loss (red)
      Right y-axis (dashed) : train_accuracy (blue), val_accuracy (green)

    Step 0 marks the initial model state before any gradient updates.
    Empty panels (no report available) are hidden.
    """
    train_reports = load_train_reports(train_reports_dir)

    n_rows = len(all_models)
    n_cols = len(all_configs)
    fig_w = max(5 * n_cols, 20)
    fig_h = max(4 * n_rows, 16)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
    )
    if n_rows == 1:
        axes = [axes]  # ensure always 2-D indexable
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    for row_idx, model_key in enumerate(all_models):
        for col_idx, cfg in enumerate(all_configs):
            ax = axes[row_idx][col_idx]
            report = train_reports.get((model_key, cfg))

            if report is None:
                ax.set_visible(False)
                continue

            history = report.get("log_history", [])

            # Split log history into training steps and eval checkpoints
            train_entries = [h for h in history if "loss" in h and "eval_loss" not in h]
            eval_entries = [h for h in history if "eval_loss" in h]

            train_steps = [h["step"] for h in train_entries]
            train_losses = [h["loss"] for h in train_entries]
            eval_steps = [h["step"] for h in eval_entries]
            eval_losses = [h["eval_loss"] for h in eval_entries]
            train_accs = [h.get("train_accuracy") for h in eval_entries]
            val_accs = [h.get("eval_accuracy") for h in eval_entries]

            ax2 = ax.twinx()

            # --- Loss curves ---
            if train_steps:
                ax.plot(
                    train_steps,
                    train_losses,
                    color="#4C72B0",
                    alpha=0.7,
                    linewidth=1.2,
                    label="Train loss",
                )
            if eval_steps:
                ax.plot(
                    eval_steps,
                    eval_losses,
                    color="#C44E52",
                    linewidth=1.8,
                    marker="o",
                    markersize=3.5,
                    label="Val loss",
                )

            # Mark step-0 baseline with a distinct marker
            if eval_entries and eval_entries[0].get("step", -1) == 0:
                ax.plot(
                    0,
                    eval_losses[0],
                    marker="*",
                    color="#C44E52",
                    markersize=8,
                    zorder=5,
                )

            # --- Accuracy curves ---
            valid_train = [
                (s, a) for s, a in zip(eval_steps, train_accs) if a is not None
            ]
            valid_val = [(s, a) for s, a in zip(eval_steps, val_accs) if a is not None]

            if valid_train:
                s_t, a_t = zip(*valid_train)
                ax2.plot(
                    s_t,
                    [a * 100 for a in a_t],
                    color="#4C72B0",
                    linewidth=1.5,
                    linestyle="--",
                    marker="s",
                    markersize=3,
                    label="Train acc (%)",
                )
                if valid_train[0][0] == 0:
                    ax2.plot(
                        0,
                        valid_train[0][1] * 100,
                        marker="*",
                        color="#4C72B0",
                        markersize=8,
                        zorder=5,
                    )

            if valid_val:
                s_v, a_v = zip(*valid_val)
                ax2.plot(
                    s_v,
                    [a * 100 for a in a_v],
                    color="#55A868",
                    linewidth=1.5,
                    linestyle="--",
                    marker="^",
                    markersize=3,
                    label="Val acc (%)",
                )
                if valid_val[0][0] == 0:
                    ax2.plot(
                        0,
                        valid_val[0][1] * 100,
                        marker="*",
                        color="#55A868",
                        markersize=8,
                        zorder=5,
                    )

            # Axis labels and formatting
            ax.set_xlabel("Step", fontsize=8)
            ax.set_ylabel("Loss", fontsize=8)
            ax2.set_ylabel("Accuracy (%)", fontsize=8, color="#555555")
            ax2.set_ylim(0, 110)
            ax2.tick_params(axis="y", labelsize=7)
            ax.tick_params(axis="both", labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.set_title(
                f"{model_key} / cfg-{cfg}", fontsize=8.5, fontweight="bold", pad=5
            )

            # Suppress per-panel legend; one shared legend is added below.
            ax.legend().set_visible(False)
            ax2.legend().set_visible(False)

        # Row label (model name)
        axes[row_idx][0].set_ylabel(f"{model_key}\nLoss", fontsize=8, fontweight="bold")

    # Reserve top 10% of the figure for title + legend, then lay out panels.
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    fig.suptitle(
        f"{technique} Training Curves — Loss & Accuracy per Model and Config\n"
        "★ = initial model (step 0, before fine-tuning)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color="#4C72B0", linewidth=1.5, label="Train loss"),
        Line2D(
            [0],
            [0],
            color="#C44E52",
            linewidth=1.5,
            marker="o",
            markersize=4,
            label="Val loss",
        ),
        Line2D(
            [0],
            [0],
            color="#4C72B0",
            linewidth=1.5,
            linestyle="--",
            marker="s",
            markersize=3,
            label="Train acc (%)",
        ),
        Line2D(
            [0],
            [0],
            color="#55A868",
            linewidth=1.5,
            linestyle="--",
            marker="^",
            markersize=3,
            label="Val acc (%)",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="grey",
            linestyle="None",
            markersize=8,
            label="Step-0 baseline (★)",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=5,
        fontsize=8,
        frameon=True,
        framealpha=0.9,
    )

    out_path = out_dir / f"{technique.lower()}_training_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Plot 2: Combined test accuracy + peak memory ───────────────────────────────


def plot_combined_accuracy_memory(
    test_reports_dir: Path,
    all_models: list[str],
    all_configs: list[str],
    out_dir: Path,
    technique: str = "LoRA",
) -> None:
    """
    2 rows: test accuracy (%) and peak inference memory (MB).
    X-axis: all models. 4 grouped bars per model, one per config.
    Bar values are annotated on top.
    """
    eval_reports = load_eval_reports(test_reports_dir)
    if not eval_reports:
        print("  [combined] No test reports found — skipping.")
        return

    n_models = len(all_models)
    n_configs = len(all_configs)
    bar_w = 0.18
    offsets = np.linspace(
        -(n_configs - 1) * bar_w / 2,
        (n_configs - 1) * bar_w / 2,
        n_configs,
    )
    x = np.arange(n_models)

    fig, axes = plt.subplots(2, 1, figsize=(max(12, n_models * 2.5), 10))

    for ax, (metric_key, metric_label, pct) in zip(
        axes,
        [
            ("accuracy", "Test Accuracy (%)", True),
            ("peak_memory_mb", "Peak Inference Memory (MB)", False),
        ],
    ):
        for ci, cfg in enumerate(all_configs):
            vals = []
            for model in all_models:
                r = eval_reports.get((model, cfg))
                v = r[metric_key] if r and metric_key in r else 0.0
                vals.append(v * 100 if pct else v)

            bars = ax.bar(
                x + offsets[ci],
                vals,
                bar_w,
                color=CONFIG_COLORS.get(cfg, "#888888"),
                label=CONFIG_LABELS.get(cfg, f"Config {cfg}"),
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
            for bar, val in zip(bars, vals):
                if val > 0:
                    fmt = f"{val:.0f}%" if pct else f"{val:.0f}"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vals) * 0.012,
                        fmt,
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                        fontweight="bold",
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(all_models, fontsize=9)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if pct:
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.legend().set_visible(False)
        ax.set_title(metric_label, fontsize=10, fontweight="bold")

    # Reserve top 12% for title + legend.
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.subplots_adjust(hspace=0.35)

    fig.suptitle(
        f"{technique} — Final Test Accuracy and Peak Inference Memory by Model & Config",
        fontsize=12,
        fontweight="bold",
        y=0.97,
    )

    fig.legend(
        handles=[
            Patch(facecolor=CONFIG_COLORS[c], label=CONFIG_LABELS[c])
            for c in all_configs
            if c in CONFIG_COLORS
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=len(all_configs),
        fontsize=9,
        frameon=True,
        framealpha=0.9,
    )

    out_path = out_dir / f"{technique.lower()}_combined.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
