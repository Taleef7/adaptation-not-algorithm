#!/usr/bin/env python3
"""Figure 3: ASR vs OR-Bench Hard-1K rejection."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parent))
from figure_style import FAMILY_COLORS, FAMILY_LABELS, FAMILY_MARKERS, despine, set_paper_style


ROOT = Path(__file__).resolve().parents[3]
OUTDIR = ROOT / "artifacts" / "revision" / "figures"
ASR = ROOT / "artifacts" / "master_asr_long.csv"
# Genuine 4-bit OR-Bench cells (see analysis/revision/build_orbench_genuine4bit.py).
ORBENCH = ROOT / "artifacts" / "revision" / "or_bench_vs_legacy_orr_genuine4bit.csv"

EVALS = [("Llama-2-13b-cls", "HB-Cls"), ("GPT-4o-mini", "GPT-4o-mini")]


def format_p_value(p: float) -> str:
    if p < 0.001:
        return "$p$ < 0.001"
    return f"$p$ = {p:.3f}"


def main() -> None:
    set_paper_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    asr = pd.read_csv(ASR)
    bb = asr[asr["attack"].isin(["PAIR", "DeepInception", "ArtPrompt"])]
    means = bb.groupby(["evaluator", "model_family", "config"], as_index=False)["ASR"].mean()
    orbench = pd.read_csv(ORBENCH)[
        ["model_family", "config", "hard_1k_rejection_rate_pct"]
    ]
    df = means.merge(orbench, on=["model_family", "config"], how="inner")

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.4), sharex=True, sharey=True)
    for ax, (eval_key, eval_label) in zip(axes, EVALS):
        sub = df[df["evaluator"].eq(eval_key)].copy()
        rho, p = stats.spearmanr(sub["ASR"], sub["hard_1k_rejection_rate_pct"])
        for family, fam_df in sub.groupby("model_family"):
            ax.scatter(
                fam_df["hard_1k_rejection_rate_pct"],
                fam_df["ASR"],
                s=30,
                color=FAMILY_COLORS[family],
                marker=FAMILY_MARKERS[family],
                edgecolor="#222222",
                linewidth=0.35,
                alpha=0.9,
                label=FAMILY_LABELS[family],
            )
        ax.set_title(eval_label, loc="left", fontweight="bold")
        ax.text(
            0.04,
            0.94,
            f"Spearman $\\rho$ = {rho:.3f}\n{format_p_value(p)}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#DDDDDD", "boxstyle": "round,pad=0.25", "linewidth": 0.5},
        )
        ax.set_xlabel("OR-Bench Hard-1K rejection rate (%)")
        ax.set_ylabel("Black-box ASR (%)")
        ax.set_xlim(18, 88)
        ax.set_ylim(-2, 66)
        despine(ax)

    axes[1].set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.52, 1.05))
    fig.tight_layout(w_pad=1.0)

    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUTDIR / f"fig3_asr_orr_scatter.{ext}")
    print(OUTDIR / "fig3_asr_orr_scatter.pdf")


if __name__ == "__main__":
    main()
