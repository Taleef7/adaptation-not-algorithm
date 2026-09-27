"""Shared plotting style for EMNLP revision figures."""

from __future__ import annotations

import matplotlib.pyplot as plt


OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "pink": "#CC79A7",
    "black": "#000000",
}

EVAL_COLORS = {
    "HB-Cls": OKABE_ITO["blue"],
    "GPT-4o-mini": OKABE_ITO["orange"],
    "LlamaGuard-3": OKABE_ITO["green"],
}

FAMILY_COLORS = {
    "gemma2": OKABE_ITO["sky"],
    "llama31": OKABE_ITO["blue"],
    "qwen3": OKABE_ITO["green"],
    "phi4": OKABE_ITO["vermillion"],
    "qwen25": OKABE_ITO["orange"],
}

FAMILY_MARKERS = {
    "gemma2": "o",
    "llama31": "s",
    "qwen3": "^",
    "phi4": "D",
    "qwen25": "v",
}

FAMILY_LABELS = {
    "gemma2": "Gemma-2",
    "llama31": "Llama-3.1",
    "qwen3": "Qwen-3",
    "phi4": "Phi-4",
    "qwen25": "Qwen-2.5",
}


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Times New Roman", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=3, color="#444444")

