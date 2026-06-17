#!/usr/bin/env python3
"""
Phase 2 - Baseline comparison plots.

Reads JSON reports from the zero-shot reports/ directory (and optionally
few_shot_reports/) to produce:

  baseline_overview.png  - Original 2-bar figure: zero-shot accuracy + peak memory.
  combined_overview.png  - 3-bar figure: zero-shot accuracy, few-shot accuracy,
                           peak memory, for every model present in both runs.

Models are colour-coded by size category:
  tiny  : < 0.6 B params  (blue)
  small : 0.6 B to 1.2 B  (green)
  mid   : >= 1.2 B params  (red)

Usage
-----
    .venv/bin/python baseline_evaluation/src/plot_baselines.py
    .venv/bin/python baseline_evaluation/src/plot_baselines.py \\
        --few-shot-reports-dir baseline_evaluation/reports_few_shot
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


def plot_overview(reports: list[dict], out_dir: Path, mode: str = "zero_shot") -> None:
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

    mode_label = "Zero-Shot" if mode == "zero_shot" else "Few-Shot"
    ax1.set_title(
        f"{mode_label} Baseline: Accuracy & Peak Memory  (all models, sorted by parameter count)",
        fontsize=12, fontweight="bold", pad=12,
    )

    out_path = out_dir / f"{mode}_comparision.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def plot_combined(zs_reports: list[dict], fs_reports: list[dict], out_dir: Path) -> None:
    """
        Two-panel figure per model:
            Panel 1 - Zero-shot accuracy, few-shot accuracy, peak memory (max of the two)
            Panel 2 - Averaged token throughput and averaged latency
    Only models present in BOTH zero-shot and few-shot reports are shown.
    Sorted by parameter count ascending.
    """
    zs_by_key = {r["model_key"]: r for r in zs_reports}
    fs_by_key = {r["model_key"]: r for r in fs_reports}
    common_keys = sorted(
        zs_by_key.keys() & fs_by_key.keys(),
        key=lambda k: MODEL_PARAMS_B.get(k, 0.0),
    )
    if not common_keys:
        print("  [combined] No common models between zero-shot and few-shot reports — skipping.")
        return

    x_labels    = [display_label(k) for k in common_keys]
    zs_acc      = [zs_by_key[k]["accuracy"] * 100 for k in common_keys]
    fs_acc      = [fs_by_key[k]["accuracy"] * 100 for k in common_keys]
    peak_mem    = [max(zs_by_key[k]["peak_memory_mb"], fs_by_key[k]["peak_memory_mb"]) for k in common_keys]
    avg_lat     = [(zs_by_key[k]["avg_latency_ms"] + fs_by_key[k]["avg_latency_ms"]) / 2 for k in common_keys]
    avg_tps     = [(zs_by_key[k]["avg_tokens_per_sec"] + fs_by_key[k]["avg_tokens_per_sec"]) / 2 for k in common_keys]

    n = len(common_keys)
    x = np.arange(n)
    width = 0.24

    cats         = [assign_category(k) for k in common_keys]
    solid_colors = [CATEGORIES[c][1] for c in cats]
    mid_colors   = [lighten(c, 0.25) for c in solid_colors]   # medium tint for few-shot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(16, n * 1.6), 12), constrained_layout=True)

    # -- Panel 1: accuracy + peak memory --------------------------------------
    bars_zs = ax1.bar(
        x - width, zs_acc, width,
        color=solid_colors, edgecolor="white", linewidth=0.6, zorder=3,
    )
    bars_fs = ax1.bar(
        x, fs_acc, width,
        color=mid_colors, edgecolor="white", linewidth=0.6, zorder=3,
    )
    ax1_mem = ax1.twinx()
    bars_mem = ax1_mem.bar(
        x + width, peak_mem, width,
        color=[lighten(c, 0.55) for c in solid_colors],
        edgecolor="white", linewidth=0.6, hatch="//", zorder=3,
    )

    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_ylim(0, 122)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    mem_max = max(peak_mem) if peak_mem else 1000
    ax1_mem.set_ylabel("Peak Memory (MB)", fontsize=11)
    ax1_mem.set_ylim(0, mem_max * 1.3)
    ax1_mem.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    for bar, val in zip(bars_zs, zs_acc):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
            f"{val:.0f}%", ha="center", va="bottom", fontsize=7, fontweight="bold",
        )
    for bar, val in zip(bars_fs, fs_acc):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
            f"{val:.0f}%", ha="center", va="bottom", fontsize=7, fontweight="bold",
        )
    for bar, val in zip(bars_mem, peak_mem):
        ax1_mem.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + mem_max * 0.012,
            f"{val / 1024:.1f} GB", ha="center", va="bottom", fontsize=6.8, color="#555555",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=8)
    ax1.tick_params(axis="x", length=0)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1_mem.spines["top"].set_visible(False)

    size_patches = [
        Patch(facecolor=CATEGORIES[k][1], label=f"{CATEGORIES[k][0]} params")
        for k in ("tiny", "small", "mid")
    ]
    metric_patches = [
        Patch(facecolor="#555555", label="Zero-shot accuracy"),
        Patch(facecolor="#999999", label="Few-shot accuracy"),
        Patch(facecolor="#cccccc", hatch="//", label="Peak memory (max)"),
    ]
    leg1 = ax1.legend(
        handles=size_patches, title="Model size",
        fontsize=8, title_fontsize=9,
        loc="upper left", bbox_to_anchor=(0.0, 1.15),
    )
    ax1.add_artist(leg1)
    ax1.legend(
        handles=metric_patches,
        fontsize=8,
        loc="upper left", bbox_to_anchor=(0.0, 0.95),
    )
    ax1.set_title(
        "Zero-Shot vs Few-Shot Accuracy & Peak Memory (peak memory uses the larger of the two runs)",
        fontsize=12, fontweight="bold", pad=12,
    )

    # -- Panel 2: averaged throughput + latency --------------------------------
    ax2_lat = ax2.twinx()
    bars_tps = ax2.bar(
        x - width / 2, avg_tps, width,
        color=solid_colors, edgecolor="white", linewidth=0.6, zorder=3,
    )
    bars_lat = ax2_lat.bar(
        x + width / 2, avg_lat, width,
        color=[lighten(c, 0.55) for c in solid_colors],
        edgecolor="white", linewidth=0.6, hatch="//", zorder=3,
    )

    tps_max = max(avg_tps) if avg_tps else 100.0
    lat_max = max(avg_lat) if avg_lat else 100.0
    ax2.set_ylabel("Avg token throughput", fontsize=11)
    ax2.set_ylim(0, tps_max * 1.3)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax2_lat.set_ylabel("Avg latency (ms)", fontsize=11)
    ax2_lat.set_ylim(0, lat_max * 1.3)
    ax2_lat.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax2.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2_lat.spines["top"].set_visible(False)

    for bar, val in zip(bars_tps, avg_tps):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + tps_max * 0.02,
            f"{val:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold",
        )
    for bar, val in zip(bars_lat, avg_lat):
        ax2_lat.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + lat_max * 0.02,
            f"{val:.0f}", ha="center", va="bottom", fontsize=7, fontweight="bold",
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=8)
    ax2.tick_params(axis="x", length=0)

    metric_patches_2 = [
        Patch(facecolor="#555555", label="Avg token throughput"),
        Patch(facecolor="#999999", label="Avg latency"),
    ]
    ax2.legend(handles=metric_patches_2, fontsize=8, loc="upper left")
    ax2.set_title(
        "Averaged Token Throughput & Latency (zero-shot and few-shot averaged per model)",
        fontsize=12, fontweight="bold", pad=12,
    )

    fig.suptitle(
        "Zero-Shot vs Few-Shot Combined Comparison",
        fontsize=13, fontweight="bold",
    )

    out_path = out_dir / "combined_comparision.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def plot_table(reports: list[dict], out_dir: Path, mode: str = "zero_shot") -> None:
    """
    Render a styled summary table as a PNG image.
    Columns: Model | Params (B) | Accuracy | Correct | Avg Lat (ms) | Tok/s | Mem (MB)
    Rows sorted by parameter count ascending.
    """
    reports = sorted(reports, key=lambda r: MODEL_PARAMS_B.get(r["model_key"], 0.0))

    col_headers = ["Model", "Params (B)", "Accuracy", "Correct", "Avg Lat (ms)", "Tok/s", "Mem (MB)"]
    rows = []
    for r in reports:
        key = r["model_key"]
        params = MODEL_PARAMS_B.get(key, 0.0)
        rows.append([
            display_label(key),
            f"{params:.3f}",
            f"{r['accuracy']*100:.1f}%",
            f"{r['n_correct']}/{r['n_examples']}",
            f"{r['avg_latency_ms']:.1f}",
            f"{r['avg_tokens_per_sec']:.1f}",
            f"{r['peak_memory_mb']:.1f}",
        ])

    n_rows = len(rows)
    fig_h = 0.45 * (n_rows + 2.5)   # scale height to row count
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.axis("off")

    title_label = "Zero-Shot" if mode == "zero_shot" else "Few-Shot"
    ax.set_title(
        f"{title_label} Baseline Summary  —  models sorted by parameter count (ascending)",
        fontsize=12, fontweight="bold", pad=10,
    )

    tbl = ax.table(
        cellText=rows,
        colLabels=col_headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.55)

    # Style header row
    for col_idx in range(len(col_headers)):
        cell = tbl[0, col_idx]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Style data rows: alternate bg + colour-code by size category
    cat_colors = {"tiny": "#d6e4f7", "small": "#d6f0e0", "mid": "#fde8e8"}
    for row_idx, r in enumerate(reports, start=1):
        key = r["model_key"]
        cat = assign_category(key)
        base_color = cat_colors[cat]
        # slightly lighter on even rows for zebra effect
        alt = 0.06 if row_idx % 2 == 0 else 0.0
        import colorsys
        rgb = tuple(int(base_color[i:i+2], 16) / 255 for i in (1, 3, 5))
        h, s, v = colorsys.rgb_to_hsv(*rgb)
        lighter = colorsys.hsv_to_rgb(h, max(0, s - alt), min(1, v + alt))
        cell_color = "#{:02x}{:02x}{:02x}".format(*[int(c * 255) for c in lighter])
        for col_idx in range(len(col_headers)):
            tbl[row_idx, col_idx].set_facecolor(cell_color)

    # Widen the Model column
    tbl.auto_set_column_width(list(range(len(col_headers))))

    fig.tight_layout()
    fname = f"{mode}_table.png"
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def plot_combined_table(zs_reports: list[dict], fs_reports: list[dict], out_dir: Path) -> None:
    """
    Combined table: Model | Params | ZS Acc | FS Acc | Delta | Avg Lat (ms) | Mem (MB)
    Only models present in both runs; sorted by parameter count ascending.
    """
    zs_by_key = {r["model_key"]: r for r in zs_reports}
    fs_by_key = {r["model_key"]: r for r in fs_reports}
    common_keys = sorted(
        zs_by_key.keys() & fs_by_key.keys(),
        key=lambda k: MODEL_PARAMS_B.get(k, 0.0),
    )
    if not common_keys:
        print("  [combined_table] No common models -- skipping.")
        return

    col_headers = ["Model", "Params (B)", "ZS Acc", "FS Acc", "Delta", "Avg Lat (ms)", "Mem (MB)"]
    rows = []
    for key in common_keys:
        zs = zs_by_key[key]
        fs = fs_by_key[key]
        params = MODEL_PARAMS_B.get(key, 0.0)
        zs_acc = zs["accuracy"] * 100
        fs_acc = fs["accuracy"] * 100
        delta  = fs_acc - zs_acc
        sign   = "+" if delta >= 0 else ""
        rows.append([
            display_label(key),
            f"{params:.3f}",
            f"{zs_acc:.1f}%",
            f"{fs_acc:.1f}%",
            f"{sign}{delta:.1f}pp",
            f"{zs['avg_latency_ms']:.1f}",
            f"{zs['peak_memory_mb']:.1f}",
        ])

    n_rows = len(rows)
    fig_h = 0.45 * (n_rows + 2.5)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.axis("off")
    ax.set_title(
        "Combined Summary: Zero-Shot vs Few-Shot  --  sorted by parameter count (ascending)",
        fontsize=12, fontweight="bold", pad=10,
    )

    tbl = ax.table(
        cellText=rows,
        colLabels=col_headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.55)

    # Header
    for col_idx in range(len(col_headers)):
        cell = tbl[0, col_idx]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    import colorsys
    cat_colors = {"tiny": "#d6e4f7", "small": "#d6f0e0", "mid": "#fde8e8"}
    for row_idx, key in enumerate(common_keys, start=1):
        cat = assign_category(key)
        base_color = cat_colors[cat]
        alt = 0.06 if row_idx % 2 == 0 else 0.0
        rgb = tuple(int(base_color[i:i+2], 16) / 255 for i in (1, 3, 5))
        h, s, v = colorsys.rgb_to_hsv(*rgb)
        lighter = colorsys.hsv_to_rgb(h, max(0, s - alt), min(1, v + alt))
        cell_color = "#{:02x}{:02x}{:02x}".format(*[int(c * 255) for c in lighter])
        for col_idx in range(len(col_headers)):
            tbl[row_idx, col_idx].set_facecolor(cell_color)
        # Colour Delta cell: green if positive, red if negative
        delta_val = float(rows[row_idx - 1][4].replace("pp", "").replace("+", ""))
        delta_color = "#1a7a3c" if delta_val > 0 else ("#c0392b" if delta_val < 0 else "#555555")
        tbl[row_idx, 4].set_text_props(color=delta_color, fontweight="bold")

    tbl.auto_set_column_width(list(range(len(col_headers))))
    fig.tight_layout()
    out_path = out_dir / "combined_table.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot baseline evaluation results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--few-shot-reports-dir", type=Path,
        default=Path(__file__).parent.parent / "reports_few_shot",
        help="Directory containing few-shot JSON report files (for combined figure).",
    )
    parser.add_argument(
        "--reports-dir", type=Path,
        default=Path(__file__).parent.parent / "reports_zero_shot",
        help="Directory containing JSON report files.",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).parent.parent / "analysis",
        help="Directory to write PNG figures.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    reports_by_key = load_latest_reports(args.reports_dir)
    if not reports_by_key:
        print(f"No reports found in {args.reports_dir}")
        raise SystemExit(1)

    print(f"\nLoaded {len(reports_by_key)} zero-shot report(s):\n")
    for key, r in sorted(
        reports_by_key.items(), key=lambda kv: MODEL_PARAMS_B.get(kv[0], 0.0)
    ):
        cat = assign_category(key)
        print(
            f"  {key:20s}  {CATEGORIES[cat][0]:14s}  "
            f"acc={r['accuracy']*100:5.1f}%  "
            f"mem={r['peak_memory_mb']:6.0f}MB"
        )

    print(f"\nGenerating zero_shot_comparision.png -> {args.out_dir}/\n")
    plot_overview(list(reports_by_key.values()), args.out_dir, mode="zero_shot")

    print(f"\nGenerating zero_shot_table.png -> {args.out_dir}/\n")
    plot_table(list(reports_by_key.values()), args.out_dir, mode="zero_shot")

    # -- Combined 3-bar figure + few-shot table ---------------------------------
    fs_by_key: dict[str, dict] = {}
    if args.few_shot_reports_dir.exists():
        fs_by_key = load_latest_reports(args.few_shot_reports_dir)
        print(f"\nLoaded {len(fs_by_key)} few-shot report(s).")
    else:
        print(f"\nFew-shot reports dir not found: {args.few_shot_reports_dir} — skipping combined figure.")

    if fs_by_key:
        print(f"\nGenerating few_shot_comparision.png -> {args.out_dir}/\n")
        plot_overview(list(fs_by_key.values()), args.out_dir, mode="few_shot")

        print(f"\nGenerating combined_comparision.png -> {args.out_dir}/\n")
        plot_combined(list(reports_by_key.values()), list(fs_by_key.values()), args.out_dir)

        print(f"\nGenerating few_shot_table.png -> {args.out_dir}/\n")
        plot_table(list(fs_by_key.values()), args.out_dir, mode="few_shot")

        print(f"\nGenerating combined_table.png -> {args.out_dir}/\n")
        plot_combined_table(list(reports_by_key.values()), list(fs_by_key.values()), args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
