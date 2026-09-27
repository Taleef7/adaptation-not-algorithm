"""Statistical robustness checks (Appendix: Statistical Robustness Checks).

1. Friedman tests (repeated-measures analogue of Kruskal-Wallis) per family:
   blocks = 6 black-box attack x benchmark cells, treatments = 5 configurations.
2. TOST equivalence for Base 4-bit vs Base FP16 (quantization alone), 30 paired
   black-box cells, HB-Cls, margins 3 / 5 / 7.5 pp, with 90% CI.
3. Margin sensitivity for the seed-42 controlled grid (LoRA-FFT, QLoRA-FFT,
   LoRA-QLoRA) at 3 / 5 / 7.5 pp, with 90% CI.
"""
import numpy as np, pandas as pd
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "revision"
OUT.mkdir(parents=True, exist_ok=True)

FAMS = ["gemma2", "llama31", "qwen3", "phi4", "qwen25"]
CFGS = ["base_fp16", "base_4bit", "lora", "qlora", "fft"]
BB = ["PAIR", "DeepInception", "ArtPrompt"]
EVALS = {"HB-Cls": "Llama-2-13b-cls", "GPT-4o-mini": "GPT-4o-mini", "LlamaGuard-3": "LlamaGuard-3"}

def tost(d, margin):
    d = np.asarray(d, float); n = len(d); m = d.mean(); sd = d.std(ddof=1); se = sd / np.sqrt(n)
    p_low = stats.t.sf((m + margin) / se, n - 1)
    p_up = stats.t.cdf((m - margin) / se, n - 1)
    t90 = stats.t.ppf(0.95, n - 1)
    return dict(n=n, mean=round(m, 3), sd=round(sd, 3), ci90_lo=round(m - t90 * se, 3),
                ci90_hi=round(m + t90 * se, 3), margin=margin, p_tost=round(max(p_low, p_up), 4),
                equivalent=bool(max(p_low, p_up) < 0.05))

df = pd.read_csv(ROOT / "artifacts" / "master_asr_long.csv")
bb = df[df.attack.isin(BB)]

# ---- 1. Friedman vs Kruskal-Wallis ----
rows = []
for ev_name, ev in EVALS.items():
    for fam in FAMS:
        sub = bb[(bb.evaluator == ev) & (bb.model_family == fam)]
        wide = sub.pivot_table(index=["attack", "dataset"], columns="config", values="ASR")[CFGS]
        assert wide.shape == (6, 5), (fam, ev, wide.shape)
        fr_s, fr_p = stats.friedmanchisquare(*[wide[c].values for c in CFGS])
        kw_s, kw_p = stats.kruskal(*[wide[c].values for c in CFGS])
        rows.append(dict(evaluator=ev_name, family=fam, friedman_chi2=round(fr_s, 2), friedman_p=round(fr_p, 4),
                         kruskal_H=round(kw_s, 2), kruskal_p=round(kw_p, 4)))
fr = pd.DataFrame(rows); fr.to_csv(OUT / "friedman_per_family.csv", index=False)
print("=== Friedman (blocks=6 attack x benchmark, k=5 configs) vs Kruskal-Wallis ===")
print(fr.to_string(index=False))

# ---- 2. Quantization-only TOST ----
hb = bb[bb.evaluator == "Llama-2-13b-cls"].pivot_table(index=["model_family", "attack", "dataset"], columns="config", values="ASR")
d_q = (hb["base_4bit"] - hb["base_fp16"]).values
assert len(d_q) == 30
q_rows = [dict(contrast="base4bit-basefp16", **tost(d_q, m)) for m in (3.0, 5.0, 7.5)]
w = stats.wilcoxon(d_q, zero_method="wilcox", method="asymptotic", correction=False)
print("\n=== Quantization alone (Base 4-bit - Base FP16), 30 BB cells, HB-Cls ===")
print(pd.DataFrame(q_rows).to_string(index=False)); print("Wilcoxon p =", round(w.pvalue, 4))

# ---- 3. Seed-42 grid margin sensitivity ----
pc = pd.read_csv(ROOT / "artifacts" / "revision_stats" / "phase5_tost_percell_5fam_strict.csv")
s_rows = []
for pair in ["lora-fft", "qlora-fft", "lora-qlora"]:
    d = pc[pc.pair == pair]["diff"].values
    for m in (3.0, 5.0, 7.5):
        s_rows.append(dict(contrast=pair, **tost(d, m)))
sens = pd.DataFrame(s_rows)
print("\n=== Seed-42 controlled grid (n=20), margin sensitivity ===")
print(sens.to_string(index=False))
pd.concat([pd.DataFrame(q_rows), sens]).to_csv(OUT / "tost_margin_sensitivity.csv", index=False)

# ---- 4. Family-clustered sensitivity for the controlled TOST ----
import statsmodels.formula.api as smf
rng = np.random.default_rng(20260915)
cl_rows = []
for pair in ["lora-fft", "qlora-fft", "lora-qlora"]:
    sub = pc[pc.pair == pair]
    fam = sub.groupby("family")["diff"].mean()
    fams = list(fam.index); groups = {f: sub[sub.family == f]["diff"].values for f in fams}
    boot = np.empty(20000)
    for b in range(len(boot)):
        fs = rng.choice(fams, len(fams), replace=True)
        boot[b] = np.concatenate([rng.choice(groups[f], len(groups[f]), replace=True) for f in fs]).mean()
    md = smf.mixedlm("diff ~ 1", sub, groups=sub["family"]).fit(reml=True)
    mu, se = md.params["Intercept"], md.bse["Intercept"]
    row = dict(contrast=pair, family_means=";".join(f"{f}:{v:+.2f}" for f, v in fam.items()),
               boot90_lo=round(np.percentile(boot, 5), 2), boot90_hi=round(np.percentile(boot, 95), 2),
               mixed90_lo=round(mu - 1.645 * se, 2), mixed90_hi=round(mu + 1.645 * se, 2))
    for m in (5.0, 7.5):
        t = tost(fam.values, m); row[f"fam_tost_p_{m}"] = t["p_tost"]; row["fam_ci90"] = f"[{t['ci90_lo']}, {t['ci90_hi']}]"
    cl_rows.append(row)
cl = pd.DataFrame(cl_rows); cl.to_csv(OUT / "tost_family_clustered.csv", index=False)
print("\n=== Family-clustered sensitivity (seed-42 grid) ===")
print(cl.to_string(index=False))

# ---- 5. Evaluator as fixed effect (replaces 3-level random intercept) ----
bb2 = bb.copy(); bb2["config"] = pd.Categorical(bb2["config"], CFGS)
f = "ASR ~ C(config) + C(model_family) + C(attack) + C(dataset) + C(evaluator)"
ols = smf.ols(f, bb2).fit()
olsc = smf.ols(f, bb2).fit(cov_type="cluster", cov_kwds={"groups": bb2["model_family"]})
fe_rows = []
for c in ["lora", "qlora", "fft", "base_4bit"]:
    k = f"C(config)[T.{c}]"
    fe_rows.append(dict(config=c, est=round(ols.params[k], 2), p=round(ols.pvalues[k], 4),
                        ci_lo=round(ols.conf_int().loc[k, 0], 2), ci_hi=round(ols.conf_int().loc[k, 1], 2),
                        p_clusterfam=round(olsc.pvalues[k], 4),
                        ci_lo_clusterfam=round(olsc.conf_int().loc[k, 0], 2), ci_hi_clusterfam=round(olsc.conf_int().loc[k, 1], 2)))
fe = pd.DataFrame(fe_rows); fe.to_csv(OUT / "claim1_evaluator_fixed_effect_ols.csv", index=False)
print("\n=== OLS with evaluator fixed effect (n=450 black-box rows) ===")
print(fe.to_string(index=False))

# ---- 6. Frontier-judge refusal sensitivity ----
import json, glob, os
rf_rows = []
for p in sorted(glob.glob(str(ROOT / "artifacts" / "revision" / "frontier_judge" / "*.jsonl"))):
    recs = [json.loads(l) for l in open(p)]
    name = os.path.basename(p)[:-6]
    sc = [r for r in recs if r["label"] in (0, 1)]; rf = [r for r in recs if r["label"] not in (0, 1)]
    n_safe = sum(r["human"] == 0 for r in recs); n_harm = sum(r["human"] == 1 for r in recs)
    fp = sum(r["human"] == 0 and r["label"] == 1 for r in sc); tp = sum(r["human"] == 1 and r["label"] == 1 for r in sc)
    rf_safe = sum(r["human"] == 0 for r in rf); rf_harm = sum(r["human"] == 1 for r in rf)
    rf_rows.append(dict(config=name, n=len(recs), refused=len(rf), refused_pct=round(100 * len(rf) / len(recs), 1),
                        refused_safe=rf_safe, refused_harm=rf_harm,
                        fpr_min=round(100 * fp / n_safe, 1), fpr_max=round(100 * (fp + rf_safe) / n_safe, 1),
                        recall_min=round(100 * tp / n_harm, 1), recall_max=round(100 * (tp + rf_harm) / n_harm, 1)))
rf = pd.DataFrame(rf_rows); rf.to_csv(OUT / "frontier_judge_refusal_sensitivity.csv", index=False)
print("\n=== Frontier-judge refusal sensitivity (worst-case label assignment) ===")
print(rf.to_string(index=False))

# ---- 7. ORR evaluator calibration against the human consensus set ----
orr = pd.read_csv(ROOT / "data" / "human_annotation" / "orr_250_labels.csv")
orr = orr[orr.agree == 1]  # 239 consensus samples; label 1 = answered, 0 = over-refused
orr_rows = []
for ev, name in [("gpt4omini_label", "GPT-4o-mini"), ("hbcls_label", "HB-Cls"), ("llamaguard3_label", "LlamaGuard-3")]:
    p = orr[ev].values == 0; g = orr["consensus"].values == 0  # positive class = over-refusal
    tp, fp, fn, tn = (p & g).sum(), (p & ~g).sum(), (~p & g).sum(), (~p & ~g).sum()
    orr_rows.append(dict(evaluator=name, n=len(orr), tp=tp, fp=fp, fn=fn, tn=tn,
                         accuracy=round(100 * (tp + tn) / len(orr), 1),
                         precision=round(100 * tp / (tp + fp), 1) if tp + fp else None,
                         recall=round(100 * tp / (tp + fn), 1), fpr=round(100 * fp / (fp + tn), 1),
                         predicted_orr=round(100 * p.mean(), 1), human_orr=round(100 * g.mean(), 1)))
orr_df = pd.DataFrame(orr_rows); orr_df.to_csv(OUT / "orr_evaluator_calibration.csv", index=False)
print("\n=== ORR evaluator calibration (positive class = over-refusal, N=239 consensus) ===")
print(orr_df.to_string(index=False))
