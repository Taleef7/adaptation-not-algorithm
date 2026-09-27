#!/usr/bin/env python3
"""Figure 1: black-box vs white-box aggregate delta-ASR."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent))
from figure_style import EVAL_COLORS, despine, set_paper_style


ROOT = Path(__file__).resolve().parents[3]
OUTDIR = ROOT / "artifacts" / "revision" / "figures"
CLAIM1 = ROOT / "artifacts" / "revision" / "claim1_table1.csv"
DELTAS = ROOT / "artifacts" / "peft_deltas.csv"

CONFIG_ORDER = ["base_4bit", "lora", "qlora", "fft"]
CONFIG_LABELS = {
    "base_4bit": "Base\n4-bit",
    "lora": "LoRA",
    "qlora": "QLoRA",
    "fft": "FFT",
}
EVAL_ORDER = ["HB-Cls", "GPT-4o-mini", "LlamaGuard-3"]
EVAL_DISPLAY = {
    "Llama-2-13b-cls": "HB-Cls",
    "GPT-4o-mini": "GPT-4o-mini",
    "LlamaGuard-3": "LlamaGuard-3",
}


def bootstrap_ci(values: np.ndarray, seed: int = 20260520, n_boot: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    means = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def build_white_box() -> pd.DataFrame:
    deltas = pd.read_csv(DELTAS)
    wb = deltas[deltas["attack"].eq("AutoDAN")].copy()
    wb["evaluator_display"] = wb["evaluator"].map(EVAL_DISPLAY)
    rows = []
    for evaluator in EVAL_ORDER:
        for config in CONFIG_ORDER:
            vals = wb[(wb["evaluator_display"].eq(evaluator)) & (wb["config"].eq(config))]["delta_ASR"].to_numpy()
            lo, hi = bootstrap_ci(vals)
            rows.append(
                {
                    "evaluator_display": evaluator,
                    "config": config,
                    "mean_delta_asr_pp": vals.mean(),
                    "ci95_low_pp": lo,
                    "ci95_high_pp": hi,
                }
            )
    return pd.DataFrame(rows)


def draw_panel(ax, df: pd.DataFrame, title: str) -> None:
    x = np.arange(len(CONFIG_ORDER))
    width = 0.23
    offsets = np.linspace(-width, width, len(EVAL_ORDER))
    for offset, evaluator in zip(offsets, EVAL_ORDER):
        sub = df[df["evaluator_display"].eq(evaluator)].set_index("config").loc[CONFIG_ORDER]
        y = sub["mean_delta_asr_pp"].to_numpy()
        lo = y - sub["ci95_low_pp"].to_numpy()
        hi = sub["ci95_high_pp"].to_numpy() - y
        ax.bar(
            x + offset,
            y,
            width=width,
            label=evaluator,
            color=EVAL_COLORS[evaluator],
            edgecolor="#222222",
            linewidth=0.35,
        )
        ax.errorbar(
            x + offset,
            y,
            yerr=np.vstack([lo, hi]),
            fmt="none",
            ecolor="#222222",
            elinewidth=0.7,
            capsize=1.8,
            capthick=0.7,
        )
    ax.axhline(0, color="#333333", linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in CONFIG_ORDER])
    ax.set_ylabel(r"$\Delta$ASR vs. Base FP16 (pp)")
    despine(ax)


def main() -> None:
    set_paper_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    bb = pd.read_csv(CLAIM1)
    wb = build_white_box()

    # sharey=False: white-box CIs span several times the black-box range, so a shared
    # axis either clips them (misleading) or flattens the black-box panel.
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.3), sharey=False)
    draw_panel(axes[0], bb, "Black-box attacks")
    draw_panel(axes[1], wb, "White-box AutoDAN")
    axes[1].set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.52, 1.04))
    fig.tight_layout(w_pad=1.1)

    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUTDIR / f"fig1_delta_asr_comparison.{ext}", bbox_inches="tight")
    print(OUTDIR / "fig1_delta_asr_comparison.pdf")


if __name__ == "__main__":
    main()
