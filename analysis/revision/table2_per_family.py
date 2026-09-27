#!/usr/bin/env python3
"""Build canonical Table 2 per-family black-box delta-ASR statistics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
IN = REPO / "artifacts" / "revision" / "per_family_per_config_summary.csv"
OUT = REPO / "artifacts" / "revision" / "table2_per_family.csv"

FAMILIES = ("gemma2", "llama31", "qwen3", "phi4", "qwen25")
CONFIGS = ("lora", "qlora", "fft")
DISPLAY_FAMILY = {
    "gemma2": "Gemma-2",
    "llama31": "Llama-3.1",
    "qwen3": "Qwen-3",
    "phi4": "Phi-4",
    "qwen25": "Qwen-2.5",
}
DISPLAY_CONFIG = {"lora": "LoRA", "qlora": "QLoRA", "fft": "FFT"}


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        raw = (m - rank) * p_values[idx]
        running_max = max(running_max, raw)
        adjusted[idx] = min(1.0, running_max)
    return adjusted.tolist()


def main() -> None:
    df = pd.read_csv(IN)
    bb = df[
        (df["evaluator"] == "Llama-2-13b-cls")
        & (df["attack"].isin(["PAIR", "DeepInception", "ArtPrompt"]))
        & (df["benchmark"].isin(["HarmBench", "JBB"]))
        & (df["config"].isin(CONFIGS))
    ].copy()

    rows = []
    for family in FAMILIES:
        for config in CONFIGS:
            values = bb[
                (bb["model_family"] == family) & (bb["config"] == config)
            ]["delta_ASR"].to_numpy(dtype=float)
            if len(values) == 0:
                raise ValueError(f"missing values for {family=} {config=}")
            try:
                stat, p_value = stats.wilcoxon(values, alternative="two-sided")
            except ValueError:
                stat, p_value = float("nan"), float("nan")
            sd = values.std(ddof=1)
            rows.append(
                {
                    "family": family,
                    "family_display": DISPLAY_FAMILY[family],
                    "config": config,
                    "config_display": DISPLAY_CONFIG[config],
                    "delta_asr": values.mean(),
                    "cohens_d": values.mean() / sd if sd > 0 else np.nan,
                    "wilcoxon_p": p_value,
                    "wilcoxon_W": stat,
                    "n_pairs": len(values),
                }
            )

    out = pd.DataFrame(rows)
    out["p_holm_15_tests"] = holm_adjust(out["wilcoxon_p"].tolist())
    out["significant_holm"] = out["p_holm_15_tests"] < 0.05
    out.to_csv(OUT, index=False, float_format="%.6f")
    print(f"Wrote {OUT} ({len(out)} rows)")


if __name__ == "__main__":
    main()
