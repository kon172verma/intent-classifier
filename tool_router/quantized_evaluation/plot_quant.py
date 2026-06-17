#!/usr/bin/env python3
"""
Quantization comparison plots.

Reads:
  quantized_evaluation/reports/*.json     – quant evaluation reports
  baseline_evaluation/reports/*.json      – zero-shot BF16 baseline
  baseline_evaluation/few_shot_reports/   – few-shot BF16 baseline

Produces (in quantized_evaluation/results/):
  zero_shot_quant_comparision.png
  few_shot_quant_comparision.png

Each figure shows one grouped-bar cluster per model (5 models × 4 precisions).
Bars = Accuracy (%).
Below each bar: peak memory in MB as a text annotation.

Usage
-----
    python plot_quant.py
    python plot_quant.py --out-dir quantized_evaluation/results
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

# ── Layout constants ─────────────────────────────────────────────────────────

# Models in ascending parameter order
SELECTED_MODELS: list[str] = [
    "qwen2.5-0.5b",
    "qwen3-0.6b",
    "gemma3-1b",
    "qwen2.5-1.5b",
    "smollm3",
]

MODEL_DISPLAY: dict[str, str] = {
    "qwen2.5-0.5b": "Qwen2.5-0.5B",
    "qwen3-0.6b":   "Qwen3-0.6B",
    "gemma3-1b":    "Gemma3-1B",
    "qwen2.5-1.5b": "Qwen2.5-1.5B",
    "smollm3":      "SmolLM3-3B",
}

PRECISIONS: list[str] = ["bf16", "int8", "nf4", "nf4_dq"]

PREC_LABELS: dict[str, str] = {
    "bf16":   "BF16",
    "int8":   "INT8",
    "nf4":    "NF4",
    "nf4_dq": "NF4+DQ",
}

# One distinct colour per precision
PREC_COLORS: dict[str, str] = {
    "bf16":   "#4C72B0",   # blue
    "int8":   "#DD8452",   # orange
    "nf4":    "#55A868",   # green
    "nf4_dq": "#C44E52",   # red
}

# ── Default paths ────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).parent
_BASELINE_DIR = _THIS_DIR.parent / "baseline_evaluation"

DEFAULT_QUANT_REPORTS = _THIS_DIR / "reports"
DEFAULT_ZS_BASELINE   = _BASELINE_DIR / "reports"
DEFAULT_FS_BASELINE   = _BASELINE_DIR / "few_shot_reports"
DEFAULT_OUT_DIR       = _THIS_DIR / "results"


# ── Report loading ────────────────────────────────────────────────────────────

def load_quant_reports(reports_dir: Path) -> dict[tuple[str, str, str], dict]:
    """
    Returns {(model_key, quant, eval_mode): report_dict}.
    Keeps only the newest file per (model, quant, mode) triple.
    """
    latest: dict[tuple[str, str, str], dict] = {}
    for f in reports_dir.glob("*.json"):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            key = (r["model_key"], r["quant"], r["eval_mode"])
            if key not in latest or r["timestamp"] > latest[key]["timestamp"]:
                latest[key] = r
        except Exception:
            continue
    return latest


def load_baseline_reports(reports_dir: Path) -> dict[str, dict]:
    """Returns {model_key: report_dict} keeping only the newest file per model."""
    latest: dict[str, dict] = {}
    for f in reports_dir.glob("*.json"):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            key = r.get("model_key", "")
            if not key:
                continue
            if key not in latest or r["timestamp"] > latest[key]["timestamp"]:
                latest[key] = r
        except Exception:
            continue
    return latest


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_quant_comparison(
    mode: str,
    quant_reports: dict[tuple[str, str, str], dict],
    zs_baseline: dict[str, dict],
    fs_baseline: dict[str, dict],
    out_dir: Path,
) -> None:
    """
    One figure per mode. Groups = models, bars = precision variants.
    Accuracy on Y-axis; peak memory shown as text below each bar.
    """
    baseline = fs_baseline if mode == "few_shot" else zs_baseline
    mode_label = "Few-Shot" if mode == "few_shot" else "Zero-Shot"

    # Build data matrix: models × precisions
    # acc[model][prec]  = accuracy (0–100) or None
    # mem[model][prec]  = peak_memory_mb or None
    acc: dict[str, dict[str, float | None]] = {m: {} for m in SELECTED_MODELS}
    mem: dict[str, dict[str, float | None]] = {m: {} for m in SELECTED_MODELS}

    # BF16 baseline
    for model in SELECTED_MODELS:
        r = baseline.get(model)
        if r is not None:
            acc[model]["bf16"] = r["accuracy"] * 100
            mem[model]["bf16"] = r["peak_memory_mb"]
        else:
            acc[model]["bf16"] = None
            mem[model]["bf16"] = None

    # Quantized runs
    for model in SELECTED_MODELS:
        for quant in ["int8", "nf4", "nf4_dq"]:
            r = quant_reports.get((model, quant, mode))
            if r is not None:
                acc[model][quant] = r["accuracy"] * 100
                mem[model][quant] = r["peak_memory_mb"]
            else:
                acc[model][quant] = None
                mem[model][quant] = None

    n_models = len(SELECTED_MODELS)
    n_prec   = len(PRECISIONS)
    group_w  = 0.8
    bar_w    = group_w / n_prec
    x_center = np.arange(n_models, dtype=float)

    fig, ax = plt.subplots(figsize=(max(14, n_models * 2.8), 7), constrained_layout=True)

    for pi, prec in enumerate(PRECISIONS):
        offsets = x_center + (pi - (n_prec - 1) / 2) * bar_w
        values  = [acc[m].get(prec) for m in SELECTED_MODELS]
        mem_vals = [mem[m].get(prec) for m in SELECTED_MODELS]
        color   = PREC_COLORS[prec]

        for xi, (val, mval) in enumerate(zip(values, mem_vals)):
            bx = offsets[xi]
            if val is None:
                # No data – draw a faint placeholder
                ax.bar(bx, 0, bar_w * 0.92, color="#e0e0e0", edgecolor="white", zorder=3)
                ax.text(bx, 2, "N/A", ha="center", va="bottom", fontsize=6.5,
                        color="#aaaaaa", rotation=90)
                continue

            bar = ax.bar(bx, val, bar_w * 0.92, color=color,
                         edgecolor="white", linewidth=0.5, zorder=3)

            # Accuracy label on top of bar
            ax.text(
                bx, val + 0.8,
                f"{val:.0f}%",
                ha="center", va="bottom",
                fontsize=7, fontweight="bold", color="#222222",
            )

            # Memory label below the bar (below x-axis line)
            if mval is not None:
                mem_str = f"{mval:.0f}MB"
                ax.text(
                    bx, -3.5,
                    mem_str,
                    ha="center", va="top",
                    fontsize=6, color="#555555",
                    rotation=90,
                    clip_on=False,
                )

    # Axes
    ax.set_xticks(x_center)
    ax.set_xticklabels(
        [MODEL_DISPLAY.get(m, m) for m in SELECTED_MODELS],
        fontsize=10,
    )
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(-18, 115)          # headroom below x-axis for memory labels
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0)

    # Draw a subtle line at y=0 to visually separate bars from memory annotations
    ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=2)

    # Legend
    legend_handles = [
        Patch(facecolor=PREC_COLORS[p], label=PREC_LABELS[p])
        for p in PRECISIONS
    ]
    ax.legend(
        handles=legend_handles,
        title="Precision",
        fontsize=9,
        title_fontsize=10,
        loc="upper left",
        framealpha=0.85,
    )

    ax.set_title(
        f"{mode_label}: Accuracy by Quantization Precision — selected PEFT models\n"
        "(memory footprint shown below each bar)",
        fontsize=12, fontweight="bold", pad=10,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{mode}_quant_comparision.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot quantization accuracy comparison figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--quant-reports-dir", type=Path, default=DEFAULT_QUANT_REPORTS)
    parser.add_argument("--zs-baseline-dir",   type=Path, default=DEFAULT_ZS_BASELINE)
    parser.add_argument("--fs-baseline-dir",   type=Path, default=DEFAULT_FS_BASELINE)
    parser.add_argument("--out-dir",            type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    print("Loading quant reports …")
    quant_reports = load_quant_reports(args.quant_reports_dir)
    print(f"  {len(quant_reports)} quant report(s) found.")

    print("Loading BF16 baselines …")
    zs_baseline = load_baseline_reports(args.zs_baseline_dir)
    fs_baseline = load_baseline_reports(args.fs_baseline_dir)
    print(f"  {len(zs_baseline)} zero-shot, {len(fs_baseline)} few-shot baseline(s).")

    for mode in ("zero_shot", "few_shot"):
        print(f"\nGenerating {mode}_quant_comparision.png …")
        plot_quant_comparison(
            mode=mode,
            quant_reports=quant_reports,
            zs_baseline=zs_baseline,
            fs_baseline=fs_baseline,
            out_dir=args.out_dir,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
