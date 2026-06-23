#!/usr/bin/env python3
"""
Generate one combined 3-row comparison figure for quantized evaluation reports.

Row 1 – Accuracy (%):         grouped by model, 10 bars per group
Row 2 – Peak memory (MB):     same grouping
Row 3 – Throughput + latency: throughput bars + latency dot markers

Bar order per model group:
  fp16-ZS  fp16-FS | fp8-ZS  fp8-FS | int8-ZS  int8-FS | nf4-ZS  nf4-FS | int4-ZS  int4-FS

Zero-shot bars: solid colour.
Few-shot bars:  lighter tint + diagonal /// hatch.

FP16 values are sourced from evaluation_baseline reports.
Quantized precision values are sourced from evaluation_quantized report folders.

Usage
-----
python plot_quant.py
python plot_quant.py --out-dir ../analysis
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import Patch

import sys
_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evaluation_lib.config import SELECTED_MODELS_QUANT  # noqa: E402
from evaluation_lib.model_info import MODEL_DISPLAY_LABELS as _MDISPLAY  # noqa: E402

# ── Layout constants ───────────────────────────────────────────────────────────

# Ordered by parameter size (smallest first)
SELECTED_MODELS: list[str] = [
    "qwen2.5-0.5b",
    "qwen3-0.6b",
    "qwen2.5-1.5b",
    "smollm3",
]

MODEL_DISPLAY: dict[str, str] = {k: _MDISPLAY.get(k, k) for k in SELECTED_MODELS}

PRECISIONS: list[str] = ["fp16", "fp8", "int8", "nf4", "int4"]

PREC_LABELS: dict[str, str] = {
    "fp16": "FP16",
    "fp8":  "FP8",
    "int8": "INT8",
    "nf4":  "NF4",
    "int4": "INT4",
}

PREC_COLORS: dict[str, str] = {
    "fp16": "#4C72B0",   # blue
    "fp8":  "#CCBB44",   # gold
    "int8": "#DD8452",   # orange
    "nf4":  "#55A868",   # green
    "int4": "#9467BD",   # purple
}

BAR_SLOTS: list[tuple[str, str]] = [
    (prec, mode)
    for prec in PRECISIONS
    for mode in ("zero_shot", "few_shot")
]

# ── Default paths ──────────────────────────────────────────────────────────────

_QUANT_DIR  = Path(__file__).parent.parent
_BASELINE_DIR = _REPO_ROOT / "evaluation_baseline"

DEFAULT_QUANT_INT8_REPORTS = _QUANT_DIR / "reports_int8"
DEFAULT_QUANT_NF4_REPORTS  = _QUANT_DIR / "reports_nf4"
DEFAULT_QUANT_INT4_REPORTS = _QUANT_DIR / "reports_int4"
DEFAULT_QUANT_FP8_REPORTS  = _QUANT_DIR / "reports_fp8"

DEFAULT_ZS_BASELINE = _BASELINE_DIR / "reports_zero_shot"
DEFAULT_FS_BASELINE = _BASELINE_DIR / "reports_few_shot"

DEFAULT_OUT_DIR = _QUANT_DIR / "analysis"


# ── Helpers ────────────────────────────────────────────────────────────────────

def lighten(hex_color: str, factor: float = 0.40) -> str:
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def _safe_load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _iter_report_objects(payload) -> list[dict]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    return []


# ── Loading ────────────────────────────────────────────────────────────────────

def load_quant_reports(
    int8_dir: Path,
    nf4_dir: Path,
    int4_dir: Path,
    fp8_dir: Path,
) -> dict:
    """Returns {(model_key, quant, eval_mode): latest_report_dict}."""
    latest: dict = {}
    for report_dir in [int8_dir, nf4_dir, int4_dir, fp8_dir]:
        if not report_dir.exists():
            continue
        for f in report_dir.glob("*.json"):
            for r in _iter_report_objects(_safe_load_json(f)):
                key = (r.get("model_key"), r.get("quant"), r.get("eval_mode"))
                if None in key:
                    continue
                if key not in latest or r.get("timestamp", "") > latest[key].get("timestamp", ""):
                    latest[key] = r
    return latest


def load_baseline_reports(reports_dir: Path) -> dict:
    """Returns {(model_key, eval_mode): latest_report_dict}."""
    latest: dict = {}
    if not reports_dir.exists():
        return latest
    for f in reports_dir.glob("*.json"):
        for r in _iter_report_objects(_safe_load_json(f)):
            model = r.get("model_key")
            mode  = r.get("eval_mode")
            if not model or not mode:
                continue
            key = (model, mode)
            if key not in latest or r.get("timestamp", "") > latest[key].get("timestamp", ""):
                latest[key] = r
    return latest


def get_metrics_for_mode(
    model: str,
    precision: str,
    mode: str,
    quant_reports: dict,
    zs_base: dict,
    fs_base: dict,
):
    """Return raw metrics dict for a single (model, precision, mode), or None."""
    if precision == "fp16":
        src = zs_base if mode == "zero_shot" else fs_base
        r = src.get((model, mode))
    else:
        r = quant_reports.get((model, precision, mode))
    if r is None:
        return None
    return {
        "accuracy":    r.get("accuracy", 0.0) * 100,
        "latency":     r.get("avg_latency_ms", 0.0),
        "throughput":  r.get("avg_tokens_per_sec", 0.0),
        "peak_memory": r.get("peak_memory_mb", 0.0),
    }


def _bar_style(prec: str, mode: str) -> dict:
    base = PREC_COLORS[prec]
    if mode == "zero_shot":
        return dict(color=base,               hatch=None,  edgecolor="white",   linewidth=0.4)
    else:
        return dict(color=lighten(base, 0.40), hatch="///", edgecolor="#aaaaaa", linewidth=0.4)


# ── Legend ─────────────────────────────────────────────────────────────────────

def _build_legend_handles() -> list:
    _zs_label = Patch(facecolor="none", edgecolor="none", label="Zero-Shot  →")
    _fs_label = Patch(facecolor="none", edgecolor="none", label="Few-Shot   →")
    zs_patches = [
        Patch(facecolor=PREC_COLORS[p], edgecolor="#555", linewidth=0.5, label=PREC_LABELS[p])
        for p in PRECISIONS
    ]
    fs_patches = [
        Patch(
            facecolor=lighten(PREC_COLORS[p], 0.40),
            hatch="///", edgecolor="#555", linewidth=0.5,
            label=PREC_LABELS[p] + " ",
        )
        for p in PRECISIONS
    ]
    handles = [_zs_label, _fs_label]
    for zp, fp in zip(zs_patches, fs_patches):
        handles.append(zp)
        handles.append(fp)
    return handles


def _attach_legend(fig) -> None:
    fig.legend(
        handles=_build_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=6,
        framealpha=0.92,
        fontsize=9.5,
        handlelength=1.8,
        handleheight=1.1,
        columnspacing=1.2,
        borderpad=0.9,
    )


# ── Bar-drawing helper ─────────────────────────────────────────────────────────

def _draw_bars(ax, data_slice: dict, metric: str, label_fn, x_center, bar_w, n_slots) -> float:
    all_vals = [
        v for m in SELECTED_MODELS for slot in BAR_SLOTS
        if (v := data_slice[m].get(slot)) is not None
    ]
    max_val = max(all_vals) if all_vals else 1.0

    for si, (prec, mode) in enumerate(BAR_SLOTS):
        offsets = x_center + (si - (n_slots - 1) / 2) * bar_w
        style = _bar_style(prec, mode)
        for xi, model in enumerate(SELECTED_MODELS):
            bx  = offsets[xi]
            val = data_slice[model].get((prec, mode))
            if val is None:
                ax.bar(bx, 0, bar_w * 0.70, color="#eeeeee", edgecolor="white", linewidth=0.3)
            else:
                ax.bar(
                    bx, val, bar_w * 0.70,
                    color=style["color"],
                    hatch=style["hatch"],
                    edgecolor=style["edgecolor"],
                    linewidth=style["linewidth"],
                )
                ax.text(
                    bx, val + max_val * 0.015,
                    label_fn(val),
                    ha="center", va="bottom",
                    fontsize=8, color="#333333",
                    rotation=90,
                )
    return max_val


def _style_ax(ax, ylabel: str, x_center, ylim_max: float, yformatter=None) -> None:
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(x_center)
    ax.set_xticklabels([MODEL_DISPLAY[m] for m in SELECTED_MODELS], fontsize=10)
    ax.set_ylim(0, ylim_max)
    if yformatter:
        ax.yaxis.set_major_formatter(yformatter)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Figure 1: Accuracy + Peak Memory ──────────────────────────────────────────

def plot_acc_mem(
    quant_reports: dict,
    zs_base: dict,
    fs_base: dict,
    out_dir: Path,
) -> Path:
    METRICS = ("accuracy", "peak_memory")
    data: dict = {met: {m: {} for m in SELECTED_MODELS} for met in METRICS}
    for model in SELECTED_MODELS:
        for prec, mode in BAR_SLOTS:
            row = get_metrics_for_mode(model, prec, mode, quant_reports, zs_base, fs_base)
            for met in METRICS:
                data[met][model][(prec, mode)] = row[met] if row else None

    n_models = len(SELECTED_MODELS)
    n_slots  = len(BAR_SLOTS)
    group_w  = 0.88
    bar_w    = group_w / n_slots
    x_center = np.arange(n_models, dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(24, n_models * 5.5), 16))
    fig.subplots_adjust(top=0.91, hspace=0.50)

    max_acc = _draw_bars(ax1, data["accuracy"], "accuracy",
                         lambda v: f"{v:.0f}%", x_center, bar_w, n_slots)
    ax1.set_title(
        "Accuracy Comparison  (All Models × All Precisions × Zero-Shot & Few-Shot)",
        fontsize=12, fontweight="bold", pad=8,
    )
    _style_ax(ax1, "Accuracy (%)", x_center, min(max_acc * 1.30, 112),
              yformatter=mticker.FormatStrFormatter("%.0f%%"))

    max_mem = _draw_bars(ax2, data["peak_memory"], "peak_memory",
                         lambda v: f"{v:.0f}", x_center, bar_w, n_slots)
    ax2.set_title(
        "Peak Memory Comparison  (All Models × All Precisions × Zero-Shot & Few-Shot)",
        fontsize=12, fontweight="bold", pad=8,
    )
    _style_ax(ax2, "Peak Memory (MB)", x_center, max_mem * 1.30)

    _attach_legend(fig)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "combined_quant_comparision.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Figure 2: Throughput + Latency ────────────────────────────────────────────

def plot_throughput_latency(
    quant_reports: dict,
    zs_base: dict,
    fs_base: dict,
    out_dir: Path,
) -> Path:
    METRICS = ("throughput", "latency")
    data: dict = {met: {m: {} for m in SELECTED_MODELS} for met in METRICS}
    for model in SELECTED_MODELS:
        for prec, mode in BAR_SLOTS:
            row = get_metrics_for_mode(model, prec, mode, quant_reports, zs_base, fs_base)
            for met in METRICS:
                data[met][model][(prec, mode)] = row[met] if row else None

    n_models = len(SELECTED_MODELS)
    n_slots  = len(BAR_SLOTS)
    group_w  = 0.88
    bar_w    = group_w / n_slots
    x_center = np.arange(n_models, dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(24, n_models * 5.5), 16))
    fig.subplots_adjust(top=0.91, hspace=0.50)

    max_tps = _draw_bars(ax1, data["throughput"], "throughput",
                         lambda v: f"{v:.0f}", x_center, bar_w, n_slots)
    ax1.set_title(
        "Avg Throughput  (All Models × All Precisions × Zero-Shot & Few-Shot)",
        fontsize=12, fontweight="bold", pad=8,
    )
    _style_ax(ax1, "Avg Tokens / Sec", x_center, max_tps * 1.30)

    max_lat = _draw_bars(ax2, data["latency"], "latency",
                         lambda v: f"{v:.0f}", x_center, bar_w, n_slots)
    ax2.set_title(
        "Avg Latency  (All Models × All Precisions × Zero-Shot & Few-Shot)",
        fontsize=12, fontweight="bold", pad=8,
    )
    _style_ax(ax2, "Avg Latency (ms)", x_center, max_lat * 1.30)

    _attach_legend(fig)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "throughput_latency_comparision.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate quantization comparison plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--int8-reports-dir", type=Path, default=DEFAULT_QUANT_INT8_REPORTS)
    parser.add_argument("--nf4-reports-dir",  type=Path, default=DEFAULT_QUANT_NF4_REPORTS)
    parser.add_argument("--int4-reports-dir", type=Path, default=DEFAULT_QUANT_INT4_REPORTS)
    parser.add_argument("--fp8-reports-dir",  type=Path, default=DEFAULT_QUANT_FP8_REPORTS)
    parser.add_argument("--zs-baseline-dir",  type=Path, default=DEFAULT_ZS_BASELINE)
    parser.add_argument("--fs-baseline-dir",  type=Path, default=DEFAULT_FS_BASELINE)
    parser.add_argument("--out-dir",          type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    print("Loading quantized reports …")
    quant_reports = load_quant_reports(
        args.int8_reports_dir,
        args.nf4_reports_dir,
        args.int4_reports_dir,
        args.fp8_reports_dir,
    )
    print(f"  Found {len(quant_reports)} latest quantized (model, quant, mode) entries.")

    print("Loading baseline reports for FP16 source …")
    zs_base = load_baseline_reports(args.zs_baseline_dir)
    fs_base = load_baseline_reports(args.fs_baseline_dir)
    print(f"  Found {len(zs_base)} zero-shot and {len(fs_base)} few-shot baseline entries.")

    p1 = plot_acc_mem(quant_reports, zs_base, fs_base, args.out_dir)
    p2 = plot_throughput_latency(quant_reports, zs_base, fs_base, args.out_dir)
    print(f"\nSaved → {p1}")
    print(f"Saved → {p2}")


if __name__ == "__main__":
    main()
