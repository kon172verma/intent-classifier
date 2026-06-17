#!/usr/bin/env python3
"""
Phase 2 - Baseline comparison plot.

Reads all JSON reports from the reports/ directory, picks the newest report
per model, then generates a single grouped vertical bar chart showing
accuracy (%) and peak memory (MB) for every model, sorted by accuracy.

Models are colour-coded by size category:
  tiny  : < 0.6 B params  (blue)
  small : 0.6 B to 1.2 B  (green)
  mid   : >= 1.2 B params  (red)

Usage
-----
    .venv/bin/python evaluation/plot_baselines.py
    .venv/bin/python evaluation/plot_baselines.py --out-dir figures/
    .venv/bin/python evaluation/plot_baselines.py --reports-dir path/to/reports
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless backend - no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import Patch

# -- Model -> approximate param count (billions) --------------------------------
MODEL_PARAMS_B: dict[str, float] = {
    "smollm2-135m": 0.135,
    "gemma3-270m":  0.270,
    "smollm2-360m": 0.360,
    "qwen2.5-0.5b": 0.500,
    "qwen3-0.6b":   0.600,
    "gemma3-1b":    1.000,
    "tinyllama":    1.100,
    "qwen2.5-1.5b": 1.500,
    "qwen3-1.7b":   1.700,
    "smollm3":      3.000,
    "llama3.2-3b":  3.200,
    "qwen3-4b":     4.000,
}

# Display labels for models whose name doesn't already include param count
MODEL_DISPLAY_LABELS: dict[str, str] = {
    "tinyllama":  "tinyllama (1.1B)",
    "smollm3":    "smollm3 (3B)",
    "llama3.2-3b": "llama3.2-3b (3.2B)",
    "gemma3-270m": "gemma3-270m (0.27B)",
    "gemma3-1b":   "gemma3-1b (1B)",
}


def display_label(model_key: str) -> str:
    return MODEL_DISPLAY_LABELS.get(model_key, model_key)

# Category label and base colour
CATEGORIES: dict[str, tuple[str, str]] = {
    "tiny":  ("< 0.6 B",       "#4C72B0"),
    "small": ("0.6 B - 1.2 B", "#55A868"),
    "mid":   (">= 1.2 B",      "#C44E52"),
}


def assign_category(model_key: str) -> str:
    params = MODEL_PARAMS_B.get(model_key, 1.0)
    if params < 0.6:
        return "tiny"
    if params < 1.2:
        return "small"
    return "mid"


def load_latest_reports(reports_dir: Path) -> dict[str, dict]:
    """Return {model_key: report_dict} keeping only the newest file per model."""
    latest: dict[str, dict] = {}
    for f in reports_dir.glob("*.json"):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            key = r.get("model_key")
            if not key:
                continue
            if key not in latest or r["timestamp"] > latest[key]["timestamp"]:
                latest[key] = r
        except Exception:
            continue
    return latest


def lighten(hex_color: str, factor: float = 0.45) -> str:
    """Return a lighter tint of a hex colour."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def plot_overview(reports: list[dict], out_dir: Path) -> None:
    """
    Single figure: grouped vertical bars per model.
    Left  bar  = Accuracy (%)        - solid category colour
    Right bar  = Peak Memory (MB)    - lighter hatch, secondary Y axis
    Models sorted by parameter count ascending.
    """
    reports = sorted(reports, key=lambda r: MODEL_PARAMS_B.get(r["model_key"], 0.0))

    model_keys   = [r["model_key"] for r in reports]
    x_labels     = [display_label(k) for k in model_keys]
    accuracies   = [r["accuracy"] * 100 for r in reports]
    memories_raw = [r["peak_memory_mb"] for r in reports]

    n = len(model_keys)
    x = np.arange(n)
    width = 0.35

    cats        = [assign_category(k) for k in model_keys]
    colors      = [CATEGORIES[c][1] for c in cats]
    light_colors = [lighten(c) for c in colors]

    fig, ax1 = plt.subplots(figsize=(max(12, n * 1.2), 7), constrained_layout=True)

    # -- Accuracy bars (left axis) ----------------------------------------------
    bars_acc = ax1.bar(
        x - width / 2, accuracies, width,
        color=colors, edgecolor="white", linewidth=0.6,
        zorder=3,
    )
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_ylim(0, 118)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    # -- Peak memory bars (right axis) ------------------------------------------
    ax2 = ax1.twinx()
    bars_mem = ax2.bar(
        x + width / 2, memories_raw, width,
        color=light_colors, edgecolor="white", linewidth=0.6,
        hatch="//",
        zorder=3,
    )
    mem_max = max(memories_raw) if memories_raw else 1000
    ax2.set_ylabel("Peak Memory (MB)", fontsize=11)
    ax2.set_ylim(0, mem_max * 1.3)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    # -- Value labels -----------------------------------------------------------
    for bar, val in zip(bars_acc, accuracies):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.0f}%",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold",
        )

    for bar, val in zip(bars_mem, memories_raw):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + mem_max * 0.012,
            f"{val:.0f}",
            ha="center", va="bottom", fontsize=7, color="#555555",
        )

    # -- X axis -----------------------------------------------------------------
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=9)
    ax1.tick_params(axis="x", length=0)

    # -- Grid -------------------------------------------------------------------
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax1.set_axisbelow(True)

    # -- Spines -----------------------------------------------------------------
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # -- Legend -----------------------------------------------------------------
    size_patches = [
        Patch(facecolor=CATEGORIES[k][1], label=f"{CATEGORIES[k][0]} params")
        for k in ("tiny", "small", "mid")
    ]
    metric_patches = [
        Patch(facecolor="#888888", label="Accuracy (%)"),
        Patch(facecolor="#cccccc", hatch="//", label="Peak Memory (MB)"),
    ]
    leg1 = ax1.legend(
        handles=size_patches, title="Model size",
        fontsize=8, title_fontsize=9,
        loc="upper right", bbox_to_anchor=(1.13, 1.0),
    )
    ax1.add_artist(leg1)
    ax1.legend(
        handles=metric_patches,
        fontsize=8,
        loc="upper right", bbox_to_anchor=(1.13, 0.78),
    )

    ax1.set_title(
        "Zero-Shot Baseline: Accuracy & Peak Memory  (all models, sorted by parameter count)",
        fontsize=12, fontweight="bold", pad=12,
    )

    out_path = out_dir / "baseline_overview.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot baseline evaluation results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reports-dir", type=Path,
        default=Path(__file__).parent / "reports",
        help="Directory containing JSON report files.",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).parent / "figures",
        help="Directory to write PNG figures.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    reports_by_key = load_latest_reports(args.reports_dir)
    if not reports_by_key:
        print(f"No reports found in {args.reports_dir}")
        raise SystemExit(1)

    print(f"\nLoaded {len(reports_by_key)} model report(s):\n")
    for key, r in sorted(
        reports_by_key.items(), key=lambda kv: MODEL_PARAMS_B.get(kv[0], 0.0)
    ):
        cat = assign_category(key)
        print(
            f"  {key:20s}  {CATEGORIES[cat][0]:14s}  "
            f"acc={r['accuracy']*100:5.1f}%  "
            f"mem={r['peak_memory_mb']:6.0f}MB"
        )

    print(f"\nGenerating figure -> {args.out_dir}/\n")
    plot_overview(list(reports_by_key.values()), args.out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
