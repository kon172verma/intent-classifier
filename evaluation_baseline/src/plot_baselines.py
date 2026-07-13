#!/usr/bin/env python3
"""
Baseline comparison plots.

Reads JSON reports from reports_zero_shot/ and reports_few_shot/ and saves
three PNG figures to the analysis/ directory:

  graph1_accuracy_garbage.png  – Accuracy & Garbage% per model (ZS + FS)
                                  Split into 2 rows: tiny+small | medium+large
  graph2_performance.png       – Peak memory, Avg latency, Token throughput
                                  Split into 2 rows: tiny+small | medium+large
  table_combined.png           – Full summary table for all models

Usage
-----
python plot_baselines.py
python plot_baselines.py \\
    --reports-dir ../reports_zero_shot \\
    --few-shot-reports-dir ../reports_few_shot \\
    --out-dir ../analysis
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import Patch

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evaluation_lib.model_info import (  # noqa: E402
    MODEL_PARAMS_B,
    SIZE_CATEGORY_COLORS as CAT_COLORS,
    model_size_category,
    display_label,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def lighten(hex_color: str, factor: float = 0.45) -> str:
    """Return hex_color blended toward white by `factor` (0 = unchanged, 1 = white)."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def load_latest_reports(reports_dir: Path) -> dict[str, dict]:
    """Load the most-recent JSON report per model_key from reports_dir."""
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


def _split_by_size(keys: list[str]) -> tuple[list[str], list[str]]:
    """Split model keys into (tiny+small, medium+large), sorted by param count."""
    top = sorted(
        [k for k in keys if model_size_category(k) in ("tiny", "small")],
        key=lambda k: MODEL_PARAMS_B.get(k, 0.0),
    )
    bot = sorted(
        [k for k in keys if model_size_category(k) in ("medium", "large")],
        key=lambda k: MODEL_PARAMS_B.get(k, 0.0),
    )
    return top, bot


# ── Graph 1 — Accuracy & Garbage Rate ─────────────────────────────────────────

def _acc_garbage_row(
    ax: plt.Axes,
    keys: list[str],
    zs_by_key: dict,
    fs_by_key: dict,
    row_title: str,
) -> None:
    """Draw 4 grouped bars per model: ZS-acc, ZS-garbage%, FS-acc, FS-garbage%."""
    n = len(keys)
    if n == 0:
        ax.set_visible(False)
        return

    bar_w   = 0.18
    offsets = [-1.5 * bar_w, -0.5 * bar_w, 0.5 * bar_w, 1.5 * bar_w]
    x       = np.arange(n)

    for i, key in enumerate(keys):
        cat   = model_size_category(key)
        solid = CAT_COLORS[cat][1]
        light = lighten(solid, 0.50)

        zs = zs_by_key.get(key, {})
        fs = fs_by_key.get(key, {})

        # accuracy is 0-1 fraction; garbage_pct is already in %
        vals    = [
            zs.get("accuracy", 0.0) * 100,
            zs.get("garbage_pct", 0.0),
            fs.get("accuracy", 0.0) * 100,
            fs.get("garbage_pct", 0.0),
        ]
        colors  = [solid, solid, light, light]
        hatches = ["", "//", "", "//"]

        for val, col, hatch, off in zip(vals, colors, hatches, offsets):
            ax.bar(
                i + off, val, bar_w,
                color=col, hatch=hatch, edgecolor="white",
                linewidth=0.5, zorder=3,
            )
            ax.text(
                i + off, val + 1.0, f"{val:.0f}%",
                ha="center", va="bottom", fontsize=6.5, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [display_label(k) for k in keys],
        rotation=30, ha="right", fontsize=8.5,
    )
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 120)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_ylabel("Percentage (%)", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(row_title, fontsize=10, fontweight="bold", pad=8)


def plot_accuracy_garbage(
    zs_by_key: dict, fs_by_key: dict, out_dir: Path
) -> None:
    """Graph 1: Accuracy & Garbage% — ZS vs FS, 2 rows by model size."""
    common = sorted(
        zs_by_key.keys() & fs_by_key.keys(),
        key=lambda k: MODEL_PARAMS_B.get(k, 0.0),
    )
    top_keys, bot_keys = _split_by_size(common)

    fig_w = max(13, max(len(top_keys), len(bot_keys)) * 1.9 + 2)
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, 13))
    fig.subplots_adjust(hspace=0.55, bottom=0.15)

    _acc_garbage_row(axes[0], top_keys, zs_by_key, fs_by_key,
                     "Tiny & Small Models  (< 1 B params)")
    _acc_garbage_row(axes[1], bot_keys, zs_by_key, fs_by_key,
                     "Medium & Large Models  (1 B – 3.2 B params)")

    # Legend
    cat_handles = [
        Patch(facecolor=CAT_COLORS[k][1], label=f"{CAT_COLORS[k][0]} ({k})")
        for k in ("tiny", "small", "medium", "large")
    ]
    metric_handles = [
        Patch(facecolor="#666666", hatch="",   label="Zero-shot accuracy"),
        Patch(facecolor="#666666", hatch="//", label="Zero-shot garbage %"),
        Patch(facecolor="#bbbbbb", hatch="",   label="Few-shot accuracy"),
        Patch(facecolor="#bbbbbb", hatch="//", label="Few-shot garbage %"),
    ]
    fig.legend(
        handles=cat_handles + metric_handles,
        ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.01),
        fontsize=8.5, frameon=True,
        title="Color = model-size category  |  Shade + hatch = metric type",
        title_fontsize=8,
    )

    fig.suptitle(
        "Accuracy & Garbage Rate — Zero-Shot vs Few-Shot\n"
        "Bars (left→right) per model: ZS accuracy, ZS garbage%, FS accuracy, FS garbage%",
        fontsize=12, fontweight="bold", y=1.01,
    )

    out_path = out_dir / "graph1_accuracy_garbage.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ── Graph 2 — Performance Metrics ─────────────────────────────────────────────

def _perf_row(
    ax: plt.Axes,
    keys: list[str],
    data_by_key: dict,
    row_title: str,
) -> None:
    """
    Draw 3 grouped bars per model: peak memory (MB), avg latency (ms), tok/s.
    Each metric is normalized to the row's maximum so all three share one y-axis
    [0–100%]; actual values are annotated on top of each bar.
    """
    n = len(keys)
    if n == 0:
        ax.set_visible(False)
        return

    mem_vals = [data_by_key[k]["peak_memory_mb"]    for k in keys]
    lat_vals = [data_by_key[k]["avg_latency_ms"]     for k in keys]
    tps_vals = [data_by_key[k]["avg_tokens_per_sec"] for k in keys]

    max_mem = max(mem_vals) or 1.0
    max_lat = max(lat_vals) or 1.0
    max_tps = max(tps_vals) or 1.0

    bar_w   = 0.22
    offsets = [-bar_w, 0.0, bar_w]
    x       = np.arange(n)

    for i, key in enumerate(keys):
        cat   = model_size_category(key)
        solid = CAT_COLORS[cat][1]
        mid   = lighten(solid, 0.30)
        light = lighten(solid, 0.60)

        norm_mem = mem_vals[i] / max_mem * 100
        norm_lat = lat_vals[i] / max_lat * 100
        norm_tps = tps_vals[i] / max_tps * 100

        configs = [
            (norm_mem, f"{mem_vals[i]:.0f} MB",  solid, ""),
            (norm_lat, f"{lat_vals[i]:.0f} ms",  mid,   "//"),
            (norm_tps, f"{tps_vals[i]:.1f} t/s", light, "xx"),
        ]

        for (norm_val, label, col, hatch), off in zip(configs, offsets):
            ax.bar(
                i + off, norm_val, bar_w,
                color=col, hatch=hatch, edgecolor="white",
                linewidth=0.5, zorder=3,
            )
            ax.text(
                i + off, norm_val + 1.0, label,
                ha="center", va="bottom", fontsize=6.0,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [display_label(k) for k in keys],
        rotation=30, ha="right", fontsize=8.5,
    )
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 125)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_ylabel("Normalized value\n(% of row maximum)", fontsize=8.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(row_title, fontsize=10, fontweight="bold", pad=8)


def plot_performance(
    zs_by_key: dict, fs_by_key: dict, out_dir: Path
) -> None:
    """
    Graph 2: Performance metrics — peak memory, avg latency, token throughput.
    Latency & throughput are averaged over ZS + FS; peak memory uses max of both.
    """
    common = sorted(
        zs_by_key.keys() & fs_by_key.keys(),
        key=lambda k: MODEL_PARAMS_B.get(k, 0.0),
    )

    merged: dict[str, dict] = {}
    for k in common:
        zs = zs_by_key[k]
        fs = fs_by_key[k]
        merged[k] = {
            "peak_memory_mb":    max(zs["peak_memory_mb"], fs["peak_memory_mb"]),
            "avg_latency_ms":    (zs["avg_latency_ms"] + fs["avg_latency_ms"]) / 2,
            "avg_tokens_per_sec":(zs["avg_tokens_per_sec"] + fs["avg_tokens_per_sec"]) / 2,
        }

    top_keys, bot_keys = _split_by_size(common)

    fig_w = max(13, max(len(top_keys), len(bot_keys)) * 1.9 + 2)
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, 12))
    fig.subplots_adjust(hspace=0.55, bottom=0.15)

    _perf_row(axes[0], top_keys, merged, "Tiny & Small Models  (< 1 B params)")
    _perf_row(axes[1], bot_keys, merged, "Medium & Large Models  (1 B – 3.2 B params)")

    # Legend
    cat_handles = [
        Patch(facecolor=CAT_COLORS[k][1], label=f"{CAT_COLORS[k][0]} ({k})")
        for k in ("tiny", "small", "medium", "large")
    ]
    metric_handles = [
        Patch(facecolor="#555555", hatch="",   label="Peak memory (MB)"),
        Patch(facecolor="#999999", hatch="//", label="Avg latency (ms)"),
        Patch(facecolor="#cccccc", hatch="xx", label="Token throughput (tok/s)"),
    ]
    fig.legend(
        handles=cat_handles + metric_handles,
        ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.01),
        fontsize=8.5, frameon=True,
        title=(
            "Color = model-size category  |  Hatch = metric type  |  "
            "Bar height normalized to row max; actual values labeled"
        ),
        title_fontsize=8,
    )

    fig.suptitle(
        "Performance Metrics — Peak Memory, Avg Latency & Token Throughput\n"
        "Latency & throughput: average of ZS + FS runs.  Memory: max of the two runs.",
        fontsize=12, fontweight="bold", y=1.01,
    )

    out_path = out_dir / "graph2_performance.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ── Table — Combined Summary ───────────────────────────────────────────────────

def plot_table_combined(
    zs_by_key: dict, fs_by_key: dict, out_dir: Path
) -> None:
    """Styled summary table saved as table_combined.png."""
    common = sorted(
        zs_by_key.keys() & fs_by_key.keys(),
        key=lambda k: MODEL_PARAMS_B.get(k, 0.0),
    )
    if not common:
        print("  [table] No models common to ZS and FS — skipping.")
        return

    col_headers = [
        "Model",
        "Params\n(B)",
        "ZS Acc\n(%)",
        "ZS Garbage\n(%)",
        "FS Acc\n(%)",
        "FS Garbage\n(%)",
        "Avg Latency\n(ms)",
        "Throughput\n(tok/s)",
        "Peak Memory\n(MB)",
    ]

    rows = []
    for key in common:
        zs = zs_by_key[key]
        fs = fs_by_key[key]
        rows.append([
            display_label(key),
            f"{MODEL_PARAMS_B.get(key, 0.0):.3f}",
            f"{zs['accuracy'] * 100:.1f}",
            f"{zs.get('garbage_pct', 0.0):.1f}",
            f"{fs['accuracy'] * 100:.1f}",
            f"{fs.get('garbage_pct', 0.0):.1f}",
            f"{(zs['avg_latency_ms']     + fs['avg_latency_ms'])     / 2:.1f}",
            f"{(zs['avg_tokens_per_sec'] + fs['avg_tokens_per_sec']) / 2:.1f}",
            f"{max(zs['peak_memory_mb'], fs['peak_memory_mb']):.1f}",
        ])

    n_rows = len(rows)
    n_cols = len(col_headers)

    fig, ax = plt.subplots(figsize=(17, max(4, 0.55 * (n_rows + 3))))
    ax.axis("off")
    ax.set_title(
        "Baseline Evaluation Summary  —  Zero-Shot vs Few-Shot"
        "  (sorted by parameter count)",
        fontsize=12, fontweight="bold", pad=14,
    )

    tbl = ax.table(
        cellText=rows, colLabels=col_headers,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.8)

    # Header row
    for col_idx in range(n_cols):
        cell = tbl[0, col_idx]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Data rows: pastel tints by category, alternating lighter / darker shade
    _CAT_LIGHT = {"tiny": "#dde8f7", "small": "#d8f0e0",
                  "medium": "#fde9d8", "large": "#fdeaea"}
    _CAT_DARK  = {"tiny": "#c8d8ef", "small": "#c5e4d0",
                  "medium": "#f5d9c5", "large": "#f5d8d8"}

    for row_idx, key in enumerate(common, start=1):
        cat   = model_size_category(key)
        shade = _CAT_DARK[cat] if row_idx % 2 == 0 else _CAT_LIGHT[cat]
        for col_idx in range(n_cols):
            tbl[row_idx, col_idx].set_facecolor(shade)

    tbl.auto_set_column_width(list(range(n_cols)))
    fig.tight_layout()

    out_path = out_dir / "table_combined.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot baseline evaluation results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reports-dir", type=Path,
        default=Path(__file__).parent.parent / "reports_zero_shot",
    )
    parser.add_argument(
        "--few-shot-reports-dir", type=Path,
        default=Path(__file__).parent.parent / "reports_few_shot",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).parent.parent / "analysis",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    zs_by_key = load_latest_reports(args.reports_dir)
    if not zs_by_key:
        print(f"No zero-shot reports found in {args.reports_dir}")
        raise SystemExit(1)

    fs_by_key: dict[str, dict] = {}
    if args.few_shot_reports_dir.exists():
        fs_by_key = load_latest_reports(args.few_shot_reports_dir)

    if not fs_by_key:
        print(
            f"Few-shot reports not found in {args.few_shot_reports_dir}. "
            "All three figures require both ZS and FS data. Exiting."
        )
        raise SystemExit(1)

    common = sorted(
        zs_by_key.keys() & fs_by_key.keys(),
        key=lambda k: MODEL_PARAMS_B.get(k, 0.0),
    )
    print(
        f"\nLoaded {len(zs_by_key)} ZS + {len(fs_by_key)} FS reports, "
        f"{len(common)} in common:\n"
    )
    for k in common:
        zs  = zs_by_key[k]
        fs  = fs_by_key[k]
        cat = model_size_category(k)
        print(
            f"  {k:20s}  {CAT_COLORS[cat][0]:10s}  "
            f"ZS acc={zs['accuracy']*100:5.1f}%  "
            f"FS acc={fs['accuracy']*100:5.1f}%  "
            f"mem={max(zs['peak_memory_mb'], fs['peak_memory_mb']):6.0f} MB"
        )

    print("\nGenerating graph1_accuracy_garbage.png ...")
    plot_accuracy_garbage(zs_by_key, fs_by_key, args.out_dir)

    print("Generating graph2_performance.png ...")
    plot_performance(zs_by_key, fs_by_key, args.out_dir)

    print("Generating table_combined.png ...")
    plot_table_combined(zs_by_key, fs_by_key, args.out_dir)

    print(f"\nDone. Outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
