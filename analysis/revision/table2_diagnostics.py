#!/usr/bin/env python3
"""Diagnostics for canonical Table 2 narrative shifts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "artifacts"
OUT = ART / "revision"

BB_ATTACKS = ("PAIR", "DeepInception", "ArtPrompt")
FAMILIES = ("gemma2", "llama31", "qwen3", "phi4", "qwen25")
CONFIGS = ("base_fp16", "base_4bit", "lora", "qlora", "fft")
DISPLAY_FAMILY = {
    "gemma2": "Gemma-2",
    "llama31": "Llama-3.1",
    "qwen3": "Qwen-3",
    "phi4": "Phi-4",
    "qwen25": "Qwen-2.5",
}


def paired_deltas(master: pd.DataFrame, *, exclude_di_jbb: bool = False) -> pd.DataFrame:
    df = master[
        (master["evaluator"] == "Llama-2-13b-cls")
        & (master["attack"].isin(BB_ATTACKS))
    ].copy()
    if exclude_di_jbb:
        df = df[~((df["attack"] == "DeepInception") & (df["dataset"] == "JBB"))]

    key_cols = ["model_family", "attack", "dataset"]
    base = (
        df[df["config"] == "base_fp16"][key_cols + ["ASR"]]
        .rename(columns={"ASR": "base_ASR"})
    )
    rows = []
    for config in CONFIGS:
        cur = (
            df[df["config"] == config][key_cols + ["ASR"]]
            .rename(columns={"ASR": "config_ASR"})
        )
        merged = base.merge(cur, on=key_cols, how="inner")
        merged["config"] = config
        merged["delta_ASR"] = merged["config_ASR"] - merged["base_ASR"]
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def di_jbb_exclusion(master: pd.DataFrame) -> pd.DataFrame:
    full = paired_deltas(master, exclude_di_jbb=False)
    excl = paired_deltas(master, exclude_di_jbb=True)
    rows = []
    for family in ("qwen25", "llama31"):
        for source, frame in (("full", full), ("exclude_deepinception_jbb", excl)):
            values = frame[
                (frame["model_family"] == family) & (frame["config"] == "fft")
            ]["delta_ASR"]
            rows.append(
                {
                    "family": family,
                    "family_display": DISPLAY_FAMILY[family],
                    "config": "fft",
                    "source": source,
                    "n_pairs": len(values),
                    "mean_delta_asr_pp": values.mean(),
                }
            )
    return pd.DataFrame(rows)


def kruskal_wallis(master: pd.DataFrame) -> pd.DataFrame:
    deltas = paired_deltas(master, exclude_di_jbb=False)
    rows = []
    for family in FAMILIES:
        groups = [
            deltas[
                (deltas["model_family"] == family) & (deltas["config"] == config)
            ]["config_ASR"].to_numpy(dtype=float)
            for config in CONFIGS
        ]
        h_stat, p_value = stats.kruskal(*groups)
        rows.append(
            {
                "family": family,
                "family_display": DISPLAY_FAMILY[family],
                "kruskal_H": h_stat,
                "p_value": p_value,
                "n_per_config": len(groups[0]),
                "significant_0_05": p_value < 0.05,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    master = pd.read_csv(ART / "master_asr_long.csv")
    di = di_jbb_exclusion(master)
    kw = kruskal_wallis(master)
    di.to_csv(OUT / "table2_di_jbb_exclusion_diagnostic.csv", index=False, float_format="%.6f")
    kw.to_csv(OUT / "table2_kruskal_wallis.csv", index=False, float_format="%.6f")
    print("DeepInception/JBB exclusion diagnostic")
    print(di.to_string(index=False))
    print("\nKruskal-Wallis")
    print(kw.to_string(index=False))


if __name__ == "__main__":
    main()
