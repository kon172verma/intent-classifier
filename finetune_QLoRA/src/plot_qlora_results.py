#!/usr/bin/env python3
"""
Phase 3 – Plot QLoRA experiment results.

Reads training and evaluation reports to produce three figures:

  qlora_accuracy_comparison.png  — bar chart of accuracy by (model, config)
  qlora_training_curves.png      — loss + val-accuracy curves per run
  qlora_memory_comparison.png    — peak inference memory by run

Usage
-----
    python plot_qlora_results.py
    python plot_qlora_results.py --split test --out-dir ../analysis
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

QLORA_DIR         = Path(__file__).parent.parent
DEFAULT_TRAIN_DIR = QLORA_DIR / "reports_training"
DEFAULT_VAL_DIR   = QLORA_DIR / "reports_validation"
DEFAULT_TEST_DIR  = QLORA_DIR / "reports_test"
DEFAULT_OUT_DIR   = QLORA_DIR / "analysis"

CONFIG_COLORS: dict[str, str] = {
    "A": "#4C72B0",
    "B": "#55A868",
    "C": "#C44E52",
}
CONFIG_LABELS: dict[str, str] = {
    "A": "Config A (light, Q+V, r=8)",
    "B": "Config B (standard, full-attn, r=16)",
    "C": "Config C (wide, attn+MLP, r=32)",
}


# ── Report loaders ────────────────────────────────────────────────────────────

def load_eval_reports(reports_dir: Path) -> list[dict]:
    """Latest eval report per (model_key, lora_config, split)."""
    latest: dict[str, dict] = {}
    for f in sorted(reports_dir.glob("*.json")):
        if "_train" in f.stem:
            continue
        try:
            r   = json.loads(f.read_text(encoding="utf-8"))
            key = f"{r['model_key']}_{r['lora_config']}_{r.get('split', 'val')}"
            if key not in latest or r["timestamp"] > latest[key]["timestamp"]:
                latest[key] = r
        except Exception:
            continue
    return list(latest.values())


def load_train_reports(reports_dir: Path) -> list[dict]:
    """Latest training report per (model_key, lora_config, dataset_size)."""
    latest: dict[str, dict] = {}
    for f in sorted(reports_dir.glob("*.json")):
        try:
            r   = json.loads(f.read_text(encoding="utf-8"))
            key = f"{r['model_key']}_{r['lora_config']}_{r.get('dataset_size', '1k')}"
            if key not in latest or r["timestamp"] > latest[key]["timestamp"]:
                latest[key] = r
        except Exception:
            continue
    return list(latest.values())


# ── Plot 1: Accuracy comparison ───────────────────────────────────────────────

def plot_accuracy_comparison(
    eval_reports: list[dict],
    out_dir: Path,
    split_label: str = "Validation",
) -> None:
    if not eval_reports:
        print("  [accuracy_comparison] No eval reports — skipping.")
        return

    models  = sorted({r["model_key"]   for r in eval_reports})
    configs = sorted({r["lora_config"] for r in eval_reports})
    n_models  = len(models)
    n_configs = len(configs)

    x       = np.arange(n_models)
    width   = 0.22
    offsets = np.linspace(
        -(n_configs - 1) * width / 2,
         (n_configs - 1) * width / 2,
        n_configs,
    )

    fig, ax = plt.subplots(
        figsize=(max(8, n_models * 2.5), 5), constrained_layout=True
    )

    for i, cfg in enumerate(configs):
        accs = []
        for model in models:
            match = [
                r for r in eval_reports
                if r["model_key"] == model and r["lora_config"] == cfg
            ]
            accs.append(match[-1]["accuracy"] * 100 if match else 0.0)

        bars = ax.bar(
            x + offsets[i], accs, width,
            color=CONFIG_COLORS.get(cfg, "#888888"),
            label=CONFIG_LABELS.get(cfg, f"Config {cfg}"),
            edgecolor="white", linewidth=0.6, zorder=3,
        )
        for bar, val in zip(bars, accs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{val:.0f}%",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title(
        f"QLoRA Accuracy by Model & Config  ({split_label})",
        fontsize=12, fontweight="bold",
    )

    out_path = out_dir / "qlora_accuracy_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Plot 2: Training curves ───────────────────────────────────────────────────

def plot_training_curves(train_reports: list[dict], out_dir: Path) -> None:
    if not train_reports:
        print("  [training_curves] No training reports — skipping.")
        return

    reports = sorted(train_reports, key=lambda r: f"{r['model_key']}_{r['lora_config']}")
    n = len(reports)

    # Fixed 2-row × 3-col grid for the standard 2-model × 3-config matrix.
    # Gracefully handles fewer runs by hiding unused axes.
    n_cols = 3
    n_rows = max(2, (n + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 4.5 * n_rows),
        constrained_layout=True,
    )
    ax_flat = axes.flatten()

    for idx, report in enumerate(reports):
        ax  = ax_flat[idx]
        ax2 = ax.twinx()

        history = report.get("log_history", [])

        train_steps  = [h["step"] for h in history if "loss"      in h and "eval_loss" not in h]
        train_losses = [h["loss"] for h in history if "loss"      in h and "eval_loss" not in h]
        eval_steps   = [h["step"] for h in history if "eval_loss" in h]
        eval_losses  = [h["eval_loss"] for h in history if "eval_loss" in h]
        eval_accs    = [h.get("eval_accuracy") for h in history if "eval_loss" in h]

        if train_steps:
            ax.plot(
                train_steps, train_losses,
                color="#4C72B0", alpha=0.7, linewidth=1.2, label="Train loss",
            )
        if eval_steps:
            ax.plot(
                eval_steps, eval_losses,
                color="#C44E52", linewidth=1.8,
                marker="o", markersize=4, label="Val loss",
            )
        valid_accs = [(s, a) for s, a in zip(eval_steps, eval_accs) if a is not None]
        if valid_accs:
            s_vals, a_vals = zip(*valid_accs)
            ax2.plot(
                s_vals, [a * 100 for a in a_vals],
                color="#55A868", linewidth=1.8, linestyle="--",
                marker="s", markersize=4, label="Val acc (%)",
            )
            ax2.set_ylabel("Val Accuracy (%)", fontsize=9, color="#55A868")
            ax2.tick_params(axis="y", colors="#55A868")
            ax2.set_ylim(0, 110)

        tag = f"{report['model_key']} / cfg-{report['lora_config']} / {report.get('dataset_size','?')}"
        ax.set_title(tag, fontsize=9, fontweight="bold")
        ax.set_xlabel("Step", fontsize=9)
        ax.set_ylabel("Loss",  fontsize=9)
        ax.spines["top"].set_visible(False)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    # Hide unused subplot slots
    for idx in range(len(reports), n_rows * n_cols):
        ax_flat[idx].set_visible(False)

    out_path = out_dir / "qlora_training_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Plot 3: Memory comparison ─────────────────────────────────────────────────

def plot_memory_comparison(eval_reports: list[dict], out_dir: Path) -> None:
    if not eval_reports:
        return

    reports = sorted(
        eval_reports,
        key=lambda r: (r["model_key"], r["lora_config"]),
    )
    labels = [
        f"{r['model_key']}\ncfg-{r['lora_config']}" for r in reports
    ]
    mems   = [r["peak_memory_mb"] for r in reports]
    colors = [CONFIG_COLORS.get(r["lora_config"], "#888888") for r in reports]

    fig, ax = plt.subplots(
        figsize=(max(8, len(labels) * 1.4), 4), constrained_layout=True
    )
    bars = ax.bar(
        range(len(labels)), mems,
        color=colors, edgecolor="white", linewidth=0.6, zorder=3,
    )
    for bar, val in zip(bars, mems):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(mems) * 0.012,
            f"{val:.0f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Peak Memory (MB)", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Peak Inference Memory by Run  (QLoRA 4-bit)", fontsize=12, fontweight="bold")

    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=CONFIG_COLORS[cfg], label=CONFIG_LABELS[cfg])
        for cfg in sorted(CONFIG_COLORS)
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="upper right")

    out_path = out_dir / "qlora_memory_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot QLoRA experiment results.")
    p.add_argument(
        "--split", default="val", choices=["val", "test"],
        help="Which eval split to use for the accuracy chart.",
    )
    p.add_argument("--train-reports-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    p.add_argument("--val-reports-dir",   type=Path, default=DEFAULT_VAL_DIR)
    p.add_argument("--test-reports-dir",  type=Path, default=DEFAULT_TEST_DIR)
    p.add_argument("--out-dir",           type=Path, default=DEFAULT_OUT_DIR)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reports_dir   = args.test_reports_dir if args.split == "test" else args.val_reports_dir
    eval_reports  = load_eval_reports(reports_dir)
    train_reports = load_train_reports(args.train_reports_dir)

    split_label = args.split.replace("_", " ").title()
    print(f"\n  Loaded {len(eval_reports)} eval reports ({args.split})")
    print(f"  Loaded {len(train_reports)} training reports\n")

    plot_accuracy_comparison(eval_reports, args.out_dir, split_label)
    plot_training_curves(train_reports, args.out_dir)
    plot_memory_comparison(eval_reports, args.out_dir)

    print(f"\n  All plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()
