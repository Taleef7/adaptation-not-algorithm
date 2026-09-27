#!/usr/bin/env python3
"""
Build the Claim 1 aggregate Delta-ASR table (paper Table 2) and the covariate-adjusted OLS.

Inputs:
  artifacts/master_asr_long.csv

Output:
  artifacts/revision/claim1_table1.csv
  artifacts/revision/claim1_table_a1_ols.csv

This script intentionally derives deltas from master_asr_long.csv rather than
reading peft_deltas.csv, so Table 1 can be regenerated directly from the
canonical ASR table shipped with the paper.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "artifacts"
OUT = ART / "revision"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260520
BOOT_N = 10_000
BB_ATTACKS = ("PAIR", "ArtPrompt", "DeepInception")
CONFIGS = ("base_4bit", "lora", "qlora", "fft")
EVALUATORS = ("Llama-2-13b-cls", "GPT-4o-mini", "LlamaGuard-3")
EVALUATOR_DISPLAY = {
    "Llama-2-13b-cls": "HB-Cls",
    "GPT-4o-mini": "GPT-4o-mini",
    "LlamaGuard-3": "LlamaGuard-3",
}
CONFIG_DISPLAY = {
    "base_4bit": "Base 4-bit",
    "lora": "LoRA",
    "qlora": "QLoRA",
    "fft": "FFT",
}


def bootstrap_ci(values: np.ndarray, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(BOOT_N, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_delta_table(asr: pd.DataFrame) -> pd.DataFrame:
    bb = asr[asr["attack"].isin(BB_ATTACKS)].copy()
    key_cols = ["attack", "dataset", "evaluator", "model_family"]
    base = (
        bb[bb["config"] == "base_fp16"][key_cols + ["ASR"]]
        .rename(columns={"ASR": "base_ASR"})
    )
    rows = []
    for cfg in CONFIGS:
        cur = (
            bb[bb["config"] == cfg][key_cols + ["ASR"]]
            .rename(columns={"ASR": "config_ASR"})
        )
        merged = base.merge(cur, on=key_cols, how="inner")
        merged["config"] = cfg
        merged["delta_ASR"] = merged["config_ASR"] - merged["base_ASR"]
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


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


def run_ols(asr: pd.DataFrame) -> pd.DataFrame:
    import statsmodels.formula.api as smf

    bb = asr[asr["attack"].isin(BB_ATTACKS)].copy()
    bb["config"] = pd.Categorical(
        bb["config"],
        categories=["base_fp16", "base_4bit", "lora", "qlora", "fft"],
        ordered=False,
    )
    rows = []
    formula = (
        "ASR ~ C(config, Treatment(reference='base_fp16'))"
        " + C(model_family) + C(attack) + C(dataset)"
    )
    for evaluator in EVALUATORS:
        ev = bb[bb["evaluator"] == evaluator].copy()
        model = smf.ols(formula, data=ev).fit()
        conf = model.conf_int()
        for param, coef in model.params.items():
            if "C(config" not in param:
                continue
            cfg = param.split("[T.", 1)[1].rstrip("]")
            rows.append({
                "evaluator": evaluator,
                "evaluator_display": EVALUATOR_DISPLAY[evaluator],
                "config": cfg,
                "config_display": CONFIG_DISPLAY[cfg],
                "coefficient_pp": coef,
                "p_value": model.pvalues[param],
                "ci95_low_pp": conf.loc[param, 0],
                "ci95_high_pp": conf.loc[param, 1],
                "r_squared": model.rsquared,
                "adj_r_squared": model.rsquared_adj,
                "n_obs": int(model.nobs),
                "formula": formula,
            })
    return pd.DataFrame(rows)


def main() -> None:
    asr = pd.read_csv(ART / "master_asr_long.csv")
    deltas = paired_delta_table(asr)

    rows: list[dict] = []
    for evaluator in EVALUATORS:
        for cfg in CONFIGS:
            values = deltas[
                (deltas["evaluator"] == evaluator) & (deltas["config"] == cfg)
            ]["delta_ASR"].to_numpy(dtype=float)
            if len(values) == 0:
                raise ValueError(f"No paired deltas for {evaluator=} {cfg=}")
            try:
                # method pinned: with ties in |d| the exact null distribution does not hold, and
                # scipy's method="auto" silently switched exact->asymptotic between versions.
                w_stat, p_value = stats.wilcoxon(
                    values, alternative="two-sided", method="asymptotic"
                )
            except ValueError:
                w_stat, p_value = float("nan"), float("nan")
            ci_low, ci_high = bootstrap_ci(values)
            sd = values.std(ddof=1)
            rows.append({
                "evaluator": evaluator,
                "evaluator_display": EVALUATOR_DISPLAY[evaluator],
                "config": cfg,
                "config_display": CONFIG_DISPLAY[cfg],
                "attack_subset": "black_box",
                "attacks": "+".join(BB_ATTACKS),
                "n_pairs": len(values),
                "mean_delta_asr_pp": values.mean(),
                "median_delta_asr_pp": np.median(values),
                "wilcoxon_W": w_stat,
                "p_value": p_value,
                "cohens_d_paired_delta": values.mean() / sd if sd > 0 else np.nan,
                "bootstrap_seed": SEED,
                "bootstrap_n": BOOT_N,
                "ci95_low_pp": ci_low,
                "ci95_high_pp": ci_high,
            })

    out = pd.DataFrame(rows)
    out["p_holm_12_tests"] = holm_adjust(out["p_value"].tolist())
    out = out[[
        "evaluator",
        "evaluator_display",
        "config",
        "config_display",
        "attack_subset",
        "attacks",
        "n_pairs",
        "mean_delta_asr_pp",
        "median_delta_asr_pp",
        "wilcoxon_W",
        "p_value",
        "p_holm_12_tests",
        "cohens_d_paired_delta",
        "bootstrap_seed",
        "bootstrap_n",
        "ci95_low_pp",
        "ci95_high_pp",
    ]]
    out.to_csv(OUT / "claim1_table1.csv", index=False, float_format="%.6f")
    ols = run_ols(asr)
    ols_path = OUT / "claim1_table_a1_ols.csv"
    ols.to_csv(ols_path, index=False, float_format="%.6f")
    print(f"Wrote {OUT / 'claim1_table1.csv'} ({len(out)} rows)")
    print(f"Wrote {ols_path} ({len(ols)} rows)")


if __name__ == "__main__":
    main()
