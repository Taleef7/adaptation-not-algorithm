#!/usr/bin/env python3
"""
Supplementary statistics on the original grid: R1 TOST between training methods,
R7 seed robustness (black-box vs white-box), R8 Wilson CI for evaluator FPR.
Reads only shipped tables; writes under artifacts/revision_stats/.
Run: python analysis/audits/revision_stats.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "artifacts"
OUT = REPO / "artifacts" / "revision_stats"
OUT.mkdir(parents=True, exist_ok=True)

BB = ["PAIR", "DeepInception", "ArtPrompt"]
HBCLS = "Llama-2-13b-cls"


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / den
    return (centre - half, centre + half)


# --------------------------------------------------------------------------
# R1: TOST equivalence between training methods (black-box, HB-Cls).
# Matched per (family, attack, dataset): d = ASR_A - ASR_B (= deltaA - deltaB).
# --------------------------------------------------------------------------
def tost_paired(d, margin):
    d = np.asarray(d, float)
    n = len(d)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    # H1: -margin < mean < margin
    t_lower = (m - (-margin)) / se          # test mean > -margin
    p_lower = stats.t.sf(t_lower, df=n - 1)  # P(T > t_lower)
    t_upper = (m - margin) / se             # test mean < margin
    p_upper = stats.t.cdf(t_upper, df=n - 1)
    p_tost = max(p_lower, p_upper)
    return dict(n=n, mean_diff=m, se=se, ci_low=m - 1.96 * se, ci_high=m + 1.96 * se,
                margin=margin, p_tost=p_tost, equivalent=bool(p_tost < 0.05))


def r1_tost():
    a = pd.read_csv(ART / "master_asr_long.csv")
    bb = a[(a.attack.isin(BB)) & (a.evaluator == HBCLS)]
    key = ["model_family", "attack", "dataset"]
    wide = bb.pivot_table(index=key, columns="config", values="ASR").reset_index()
    rows = []
    for A, B in [("lora", "fft"), ("qlora", "fft"), ("lora", "qlora")]:
        sub = wide.dropna(subset=[A, B])
        d = sub[A].to_numpy() - sub[B].to_numpy()
        # Wilcoxon difference test (is there ANY difference?)
        try:
            w, p_diff = stats.wilcoxon(d, alternative="two-sided")
        except ValueError:
            w, p_diff = float("nan"), float("nan")
        for margin in (3.0, 5.0):
            r = tost_paired(d, margin)
            r.update(pair=f"{A}_vs_{B}", wilcoxon_diff_W=w, wilcoxon_diff_p=p_diff)
            rows.append(r)
    df = pd.DataFrame(rows)[["pair", "n", "mean_diff", "ci_low", "ci_high",
                             "wilcoxon_diff_p", "margin", "p_tost", "equivalent"]]
    df.to_csv(OUT / "r1_tost_method_equivalence.csv", index=False, float_format="%.4f")
    print("\n=== R1: TOST method equivalence (black-box, HB-Cls) ===")
    print(df.to_string(index=False))


# --------------------------------------------------------------------------
# R7: seed robustness, black-box vs white-box (Llama-3.1 LoRA, HB-Cls).
# --------------------------------------------------------------------------
def r7_seed():
    a = pd.read_csv(ART / "master_asr_long.csv")
    s = pd.read_csv(ART / "step3a2_seed_asr_summary.csv")
    s = s.copy()
    s["seed"] = s["model"].str.extract(r"seed(\d+)")
    s["asr_pct"] = s["llama2_asr"] * 100.0
    rows = []
    # black-box cells with seed variants
    for attack, dataset in [("PAIR", "JBB"), ("DeepInception", "HarmBench"), ("ArtPrompt", "HarmBench")]:
        orig = a[(a.model_family == "llama31") & (a.config == "lora") &
                 (a.evaluator == HBCLS) & (a.attack == attack) & (a.dataset == dataset)]["ASR"]
        vals = list(orig.to_numpy())
        sv = s[(s.attack == attack) & (s.dataset == dataset) &
               s.model.str.contains("llama31_lora")]["asr_pct"].to_numpy()
        vals += list(sv)
        vals = [float(v) for v in vals]
        if len(vals) >= 2:
            rows.append(dict(paradigm="black-box", cell=f"{attack}/{dataset}",
                             n_seeds=len(vals), asr_values=";".join(f"{v:.1f}" for v in vals),
                             mean=np.mean(vals), sd=np.std(vals, ddof=1), rng=max(vals) - min(vals)))
    # white-box AutoDAN (Llama-3.1 LoRA) from cross_seed_results
    cs = pd.read_csv(ART / "cross_seed_results.csv")
    wb = cs[(cs.method == "lora") & (cs.attack == "AutoDAN") & (cs.evaluator == HBCLS)]["ASR"].to_numpy()
    wb = [float(v) for v in wb]
    rows.append(dict(paradigm="white-box", cell="AutoDAN/HarmBench",
                     n_seeds=len(wb), asr_values=";".join(f"{v:.1f}" for v in wb),
                     mean=np.mean(wb), sd=np.std(wb, ddof=1), rng=max(wb) - min(wb)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "r7_seed_robustness_bb_vs_wb.csv", index=False, float_format="%.2f")
    print("\n=== R7: seed robustness, Llama-3.1 LoRA (HB-Cls) ===")
    print(df.to_string(index=False))


# --------------------------------------------------------------------------
# R8: FPR Wilson 95% CIs from regime_a_calibration counts.
# --------------------------------------------------------------------------
def r8_fpr_ci():
    c = pd.read_csv(ART / "revision" / "regime_a_calibration.csv")
    asr = c[c.task == "asr"]
    rows = []
    for _, r in asr.iterrows():
        fp, tn = int(r.fp), int(r.tn)
        n_safe = fp + tn
        lo, hi = wilson_ci(fp, n_safe)
        rows.append(dict(evaluator=r.evaluator, fp=fp, n_safe=n_safe,
                         fpr=100 * fp / n_safe, ci_low=100 * lo, ci_high=100 * hi))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "r8_fpr_wilson_ci.csv", index=False, float_format="%.2f")
    print("\n=== R8: ASR FPR Wilson 95% CIs (clean consensus, n_safe per eval) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    r1_tost()
    r7_seed()
    r8_fpr_ci()
    print(f"\nWrote outputs to {OUT}")
