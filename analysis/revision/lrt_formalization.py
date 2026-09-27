#!/usr/bin/env python3
"""Formal BB/WB interaction test (appendix: LRT for BB/WB Divergence).

The target is a likelihood-ratio test for whether the effect of
fine-tuning configuration depends on attack paradigm (black-box vs white-box).
The specified mixed model includes both C(model_family) fixed effects and
a random intercept grouped by model_family. In statsmodels this is redundant:
the family random intercept is estimated on the boundary with zero variance and
infinite log-likelihood. We record that diagnostic and provide a finite mixed
model with family as the random intercept and evaluator as a fixed effect.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "artifacts" / "master_asr_long.csv"
OUT = ROOT / "artifacts" / "revision" / "lrt_results.csv"

BB_ATTACKS = {"PAIR", "DeepInception", "ArtPrompt"}
WB_ATTACKS = {"AutoDAN"}
CONFIG_ORDER = ["base_fp16", "base_4bit", "lora", "qlora", "fft"]


def p_value_text(p: float) -> str:
    if np.isnan(p):
        return "nan"
    return "<0.001" if p < 0.001 else f"{p:.6f}"


def fit_mixedlm(formula: str, data: pd.DataFrame, group_col: str, method: str = "lbfgs"):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = smf.mixedlm(formula, data=data, groups=data[group_col]).fit(
            reml=False, method=method, maxiter=1000, disp=False
        )
    warning_text = " | ".join(str(w.message) for w in caught)
    return result, warning_text


def fit_ols(formula: str, data: pd.DataFrame):
    return smf.ols(formula, data=data).fit()


def lrt(null_result, alt_result, df_diff: int) -> tuple[float, float, float]:
    if not (np.isfinite(null_result.llf) and np.isfinite(alt_result.llf)):
        return np.nan, np.nan, np.nan
    delta_ll = alt_result.llf - null_result.llf
    chi2 = 2 * delta_ll
    p = stats.chi2.sf(chi2, df=df_diff) if np.isfinite(chi2) else np.nan
    return delta_ll, chi2, p


def interaction_row(model_name: str, result, term: str, extra: dict) -> dict:
    ci = result.conf_int().loc[term]
    return {
        "model": model_name,
        "row_type": "coefficient",
        "term": "LoRA x white_box",
        "coef_pp": result.params[term],
        "ci_low_pp": ci.iloc[0],
        "ci_high_pp": ci.iloc[1],
        "p_value": result.pvalues[term],
        **extra,
    }


def summary_row(
    model_name: str,
    null_result,
    alt_result,
    df_diff: int,
    null_formula: str,
    alt_formula: str,
    notes: str,
    warnings_text: str = "",
) -> dict:
    delta_ll, chi2, p = lrt(null_result, alt_result, df_diff)
    null_group_var = np.nan
    alt_group_var = np.nan
    if hasattr(null_result, "cov_re") and len(null_result.cov_re):
        null_group_var = float(null_result.cov_re.iloc[0, 0])
    if hasattr(alt_result, "cov_re") and len(alt_result.cov_re):
        alt_group_var = float(alt_result.cov_re.iloc[0, 0])
    return {
        "model": model_name,
        "row_type": "lrt",
        "term": "config x attack_type",
        "loglik_null": null_result.llf,
        "loglik_alt": alt_result.llf,
        "delta_loglik": delta_ll,
        "df": df_diff,
        "chi2": chi2,
        "p_value": p,
        "null_group_var": null_group_var,
        "alt_group_var": alt_group_var,
        "null_formula": null_formula,
        "alt_formula": alt_formula,
        "notes": notes,
        "warnings": warnings_text,
    }


def main() -> None:
    df = pd.read_csv(MASTER)
    df = df[df["attack"].isin(BB_ATTACKS | WB_ATTACKS)].copy()
    df["attack_type"] = np.where(df["attack"].isin(WB_ATTACKS), "white_box", "black_box")
    df["config"] = pd.Categorical(df["config"], categories=CONFIG_ORDER)
    df["attack_type"] = pd.Categorical(df["attack_type"], categories=["black_box", "white_box"])

    rows: list[dict] = []

    requested_null = (
        "ASR ~ C(config, Treatment('base_fp16')) + "
        "C(attack_type, Treatment('black_box')) + C(model_family) + C(dataset)"
    )
    requested_alt = (
        "ASR ~ C(config, Treatment('base_fp16')) * "
        "C(attack_type, Treatment('black_box')) + C(model_family) + C(dataset)"
    )
    finite_mixed_null = (
        "ASR ~ C(config, Treatment('base_fp16')) + "
        "C(attack_type, Treatment('black_box')) + C(dataset) + C(evaluator)"
    )
    finite_mixed_alt = (
        "ASR ~ C(config, Treatment('base_fp16')) * "
        "C(attack_type, Treatment('black_box')) + C(dataset) + C(evaluator)"
    )

    interaction_term = (
        "C(config, Treatment('base_fp16'))[T.lora]:"
        "C(attack_type, Treatment('black_box'))[T.white_box]"
    )

    # Exact requested model: record the boundary diagnostic for review.
    req_null, req_null_warn = fit_mixedlm(requested_null, df, "model_family")
    req_alt, req_alt_warn = fit_mixedlm(requested_alt, df, "model_family")
    rows.append(
        summary_row(
            "requested_mixedlm_family_fixed_plus_family_random",
            req_null,
            req_alt,
            len(req_alt.fe_params) - len(req_null.fe_params),
            requested_null + " + (1 | model_family)",
            requested_alt + " + (1 | model_family)",
            "Non-identifiable in statsmodels: family is both fixed and random; "
            "random-intercept variance is estimated on the boundary.",
            " || ".join(x for x in [req_null_warn, req_alt_warn] if x),
        )
    )
    rows.append(
        interaction_row(
            "requested_mixedlm_family_fixed_plus_family_random",
            req_alt,
            interaction_term,
            {
                "notes": "Coefficient is estimable, but LRT log-likelihood is not finite.",
                "null_formula": requested_null + " + (1 | model_family)",
                "alt_formula": requested_alt + " + (1 | model_family)",
            },
        )
    )

    # Finite mixed model: family random intercept, evaluator fixed effect.
    mix_null, mix_null_warn = fit_mixedlm(finite_mixed_null, df, "model_family", method="powell")
    mix_alt, mix_alt_warn = fit_mixedlm(finite_mixed_alt, df, "model_family", method="powell")
    rows.append(
        summary_row(
            "mixedlm_family_random_evaluator_fixed",
            mix_null,
            mix_alt,
            len(mix_alt.fe_params) - len(mix_null.fe_params),
            finite_mixed_null + " + (1 | model_family)",
            finite_mixed_alt + " + (1 | model_family)",
            "Finite mixed-effects variant: family random intercept, evaluator fixed effect.",
            " || ".join(x for x in [mix_null_warn, mix_alt_warn] if x),
        )
    )
    rows.append(
        interaction_row(
            "mixedlm_family_random_evaluator_fixed",
            mix_alt,
            interaction_term,
            {
                "notes": "Recommended finite mixed model for text propagation if approved.",
                "null_formula": finite_mixed_null + " + (1 | model_family)",
                "alt_formula": finite_mixed_alt + " + (1 | model_family)",
            },
        )
    )

    # Fixed-effects equivalent for sensitivity.
    ols_null = fit_ols(requested_null, df)
    ols_alt = fit_ols(requested_alt, df)
    rows.append(
        summary_row(
            "ols_family_fixed_requested_formula",
            ols_null,
            ols_alt,
            len(ols_alt.params) - len(ols_null.params),
            requested_null,
            requested_alt,
            "Fixed-effects sensitivity matching the requested fixed-effect structure.",
        )
    )
    rows.append(
        interaction_row(
            "ols_family_fixed_requested_formula",
            ols_alt,
            interaction_term,
            {
                "notes": "Sensitivity check without random intercept.",
                "null_formula": requested_null,
                "alt_formula": requested_alt,
            },
        )
    )

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    display_cols = [
        "model",
        "row_type",
        "term",
        "loglik_null",
        "loglik_alt",
        "delta_loglik",
        "df",
        "chi2",
        "p_value",
        "coef_pp",
        "ci_low_pp",
        "ci_high_pp",
    ]
    print(f"Wrote {OUT}")
    print(out.reindex(columns=display_cols).to_string(index=False))
    finite_lrt = out[(out["model"] == "mixedlm_family_random_evaluator_fixed") & (out["row_type"] == "lrt")].iloc[0]
    finite_coef = out[(out["model"] == "mixedlm_family_random_evaluator_fixed") & (out["row_type"] == "coefficient")].iloc[0]
    print()
    print(
        "Recommended finite mixed model: "
        f"delta_loglik={finite_lrt['delta_loglik']:.3f}, "
        f"df={int(finite_lrt['df'])}, chi2={finite_lrt['chi2']:.3f}, "
        f"p={p_value_text(float(finite_lrt['p_value']))}; "
        f"LoRA x white_box={finite_coef['coef_pp']:.2f} pp "
        f"[{finite_coef['ci_low_pp']:.2f}, {finite_coef['ci_high_pp']:.2f}], "
        f"p={p_value_text(float(finite_coef['p_value']))}"
    )


if __name__ == "__main__":
    main()
