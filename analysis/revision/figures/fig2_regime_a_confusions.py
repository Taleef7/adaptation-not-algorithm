#!/usr/bin/env python3
"""Figure 2: Regime A evaluator confusion matrices."""

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

CONFUSIONS = [
    ("HB-Cls", ROOT / "artifacts" / "revision" / "regime_a_confusion_hbcls_asr.csv"),
    ("LlamaGuard-3", ROOT / "artifacts" / "revision" / "regime_a_confusion_llamaguard3_asr.csv"),
    ("GPT-4o-mini", ROOT / "artifacts" / "revision" / "regime_a_confusion_gpt4omini_asr.csv"),
]


def load_matrix(path: Path) -> np.ndarray:
    df = pd.read_csv(path, index_col=0)
    return df.loc[["actual_negative", "actual_positive"], ["pred_negative", "pred_positive"]].to_numpy()


def main() -> None:
    set_paper_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.1), constrained_layout=True)
    for ax, (name, path) in zip(axes, CONFUSIONS):
        mat = load_matrix(path)
        pct = mat / mat.sum()
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=0.68)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]}\n{pct[i, j]*100:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#111111" if pct[i, j] < 0.45 else "white",
                    fontweight="bold" if i == j else "normal",
                )
        ax.set_title(name, color=EVAL_COLORS.get(name, "#222222"), fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred.\nsafe", "Pred.\nharmful"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Human\nsafe", "Human\nharmful"])
        ax.tick_params(length=0)
        despine(ax)
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUTDIR / f"fig2_regime_a_confusions.{ext}")
    print(OUTDIR / "fig2_regime_a_confusions.pdf")


if __name__ == "__main__":
    main()

