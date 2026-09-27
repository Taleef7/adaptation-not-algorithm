"""Controlled-grid robustness checks (adaptation shifts, three-judge TOST, AutoDAN/GCG).

T1. Adaptation effect (Base FP16 -> LoRA / QLoRA / FFT) on the 20 seed-42 controlled
    cells (5 families x {DeepInception, ArtPrompt} x {JBB, HarmBench}), with paired-t and
    bootstrap 95% CIs, Wilcoxon p, a one-sided test of "shift > 5 pp", and per-family means.
    Base models are not retrained, so the base ASR is the original-grid base_fp16 cell from
    artifacts/master_asr_long.csv (the source of Table A.1).
T2. The same controlled-grid TOST (LoRA-FFT, QLoRA-FFT, LoRA-QLoRA; +-3 / 5 / 7.5 pp,
    t-based 90% CI; identical function to camera_ready_stats.py) under GPT-4o-mini and
    LlamaGuard-3, plus the T1 shifts under each judge.
T3. Restricted TOST (Llama-3.1, Gemma-2, Qwen-3; n=12), i.e. excluding the two families
    whose FFT cells used paged_adamw_8bit.
T4. AutoDAN (HarmBench / JBB / mean) and GCG (JBB-100) per-family ASR, HB-Cls, with exact
    McNemar tests for GCG base-vs-LoRA and LoRA-vs-FFT.
T5. SafeLoRA audited cell (Llama-3.1, AutoDAN, HarmBench-400): base vs LoRA vs SafeLoRA.

Outputs (artifacts/revision/):
  r6_controlled_cells_3judge.csv     per-cell ASR, all configs, all three judges
  r6_adaptation_shift.csv            T1/T2 aggregate shifts per judge
  r6_adaptation_shift_family.csv     per-family shifts per judge
  r6_tost_3judge.csv                 T2/T3 TOST per judge, full (n=20) and restricted (n=12)
  r6_autodan_gcg_family.csv          T4 per-family ASR
  r6_gcg_mcnemar.csv                 T4 GCG exact McNemar tests
"""
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "revision"
HB = ROOT / "toolkits" / "HarmBench"
SCORED = HB / "results_phase5" / "scored"
# Behavior-level labels shipped with the release; used when the raw scored files are absent.
LABELS = ROOT / "data" / "evaluator_labels"
RAW = SCORED.is_dir()

FAMS = ["gemma2", "llama31", "phi4", "qwen25", "qwen3"]
RESTRICT = ["gemma2", "llama31", "qwen3"]
METHODS = ["lora", "qlora", "fft"]
JUDGES = {"hbcls": "Llama-2-13b-cls", "gpt4omini": "GPT-4o-mini", "llamaguard": "LlamaGuard-3"}
JNAME = {"hbcls": "HB-Cls", "gpt4omini": "GPT-4o-mini", "llamaguard": "LlamaGuard-3"}
DS = {"jbb": "JBB", "harmbench": "HarmBench"}
RNG_SEED = 20260520


def tost(d, margin):
    """Identical to camera_ready_stats.py::tost (Schuirmann, paired, t-based 90% CI)."""
    d = np.asarray(d, float); n = len(d); m = d.mean(); sd = d.std(ddof=1); se = sd / np.sqrt(n)
    p_low = stats.t.sf((m + margin) / se, n - 1)
    p_up = stats.t.cdf((m - margin) / se, n - 1)
    t90 = stats.t.ppf(0.95, n - 1)
    return dict(n=n, mean=round(m, 3), sd=round(sd, 3), ci90_lo=round(m - t90 * se, 3),
                ci90_hi=round(m + t90 * se, 3), margin=margin, p_tost=round(max(p_low, p_up), 4),
                equivalent=bool(max(p_low, p_up) < 0.05))


def boot_ci(d, B=10000, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    bs = rng.choice(d, (B, len(d)), replace=True).mean(1)
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def shift_stats(d):
    d = np.asarray(d, float); n = len(d); m = d.mean(); sd = d.std(ddof=1); se = sd / np.sqrt(n)
    t95 = stats.t.ppf(0.975, n - 1)
    blo, bhi = boot_ci(d)
    try:
        wp = stats.wilcoxon(d, alternative="two-sided").pvalue
    except ValueError:
        wp = np.nan
    # one-sided: H0 mean <= 5 vs H1 mean > 5 (is the effect clearly larger than the margin?)
    p_gt5 = stats.t.sf((m - 5.0) / se, n - 1)
    return dict(n=n, mean=round(m, 3), sd=round(sd, 3), dz=round(m / sd, 3),
                t_ci95_lo=round(m - t95 * se, 3), t_ci95_hi=round(m + t95 * se, 3),
                t_p=round(stats.ttest_1samp(d, 0).pvalue, 5),
                boot_ci95_lo=round(blo, 3), boot_ci95_hi=round(bhi, 3),
                wilcoxon_p=round(wp, 5), p_one_sided_mean_gt_5=round(p_gt5, 4),
                n_cells_positive=int((d > 0).sum()))


# ---------------------------------------------------------------- load seed-42 cells
def parse(fname, judge):
    base = re.sub(r"\.jsonl$", "", fname)
    if not base.endswith("_" + judge):
        return None
    base = base[: -len(judge) - 1]
    attack = "ArtPrompt" if base.endswith("_artprompt") else "DeepInception"
    base = re.sub(r"_artprompt$", "", base).replace("_seed42", "")
    m = re.match(r"(llama31|gemma2|qwen3|phi4|qwen25)_(lora|qlora|fft)_(jbb|harmbench)$", base)
    return None if not m else (m.group(1), m.group(2), DS[m.group(3)], attack)


def load_seed42(judge):
    if not RAW:
        lab = pd.read_csv(LABELS / "seed42_grid.csv.gz")
        lab = lab[lab.evaluator == JNAME[judge]]
        g = lab.groupby(["family", "config", "benchmark", "attack"]).label.agg(["size", "sum"]).reset_index()
        return pd.DataFrame(dict(model_family=g.family, config=g.config, dataset=g.benchmark, attack=g.attack,
                                 n=g["size"], ASR=100.0 * g["sum"] / g["size"]))
    rows = []
    for p in sorted(SCORED.glob(f"*seed42*_{judge}.jsonl")):
        info = parse(p.name, judge)
        if not info:
            continue
        recs = [json.loads(l) for l in open(p) if l.strip()]
        jb = sum(int(r.get("jailbroken", 0)) for r in recs)
        rows.append(dict(model_family=info[0], config=info[1], dataset=info[2], attack=info[3],
                         n=len(recs), ASR=100.0 * jb / len(recs)))
    return pd.DataFrame(rows)


master = pd.read_csv(ROOT / "artifacts" / "master_asr_long.csv")
cells = []
for jk, ev in JUDGES.items():
    s42 = load_seed42(jk)
    assert len(s42) == 60, (jk, len(s42))
    assert s42.groupby("dataset").n.unique().map(lambda v: len(v) == 1).all(), jk
    base = master[(master.evaluator == ev) & master.attack.isin(["DeepInception", "ArtPrompt"])
                  & master.config.isin(["base_fp16", "base_4bit"])]
    orig = master[(master.evaluator == ev) & master.attack.isin(["DeepInception", "ArtPrompt"])
                  & master.config.isin(METHODS)].assign(config=lambda x: x.config + "_orig")
    allc = pd.concat([s42, base[s42.columns], orig[s42.columns]])
    W = allc.pivot_table(index=["model_family", "attack", "dataset"], columns="config", values="ASR").reset_index()
    W.insert(0, "judge", JNAME[jk])
    assert len(W) == 20 and W[["base_fp16", "base_4bit"] + METHODS].notna().all().all(), jk
    cells.append(W)
C = pd.concat(cells, ignore_index=True)
C.to_csv(OUT / "r6_controlled_cells_3judge.csv", index=False, float_format="%.2f")

# Sanity: Table A.1 base_fp16 means come from master_asr_long (6 BB cells, HB-Cls)
a1 = master[(master.evaluator == "Llama-2-13b-cls") & master.attack.isin(["PAIR", "DeepInception", "ArtPrompt"])
            & (master.config == "base_fp16")].groupby("model_family").ASR.agg(["mean", "std"]).round(1)
print("=== Table A.1 base_fp16 check (mean/std over 6 BB cells, HB-Cls) ===")
print(a1.to_string())

# ---------------------------------------------------------------- T1 / T2 shifts
CONTRASTS = [("lora", "base_fp16"), ("qlora", "base_fp16"), ("fft", "base_fp16"),
             ("base_4bit", "base_fp16"), ("lora", "fft"), ("qlora", "fft"), ("lora", "qlora"),
             ("lora_orig", "base_fp16"), ("qlora_orig", "base_fp16"), ("fft_orig", "base_fp16")]
agg, fam_rows = [], []
for jn, W in C.groupby("judge", sort=False):
    for a, b in CONTRASTS:
        for scope, sub in [("all5", W), ("restricted3", W[W.model_family.isin(RESTRICT)])]:
            d = (sub[a] - sub[b]).values
            agg.append(dict(judge=jn, contrast=f"{a}-{b}", scope=scope, **shift_stats(d)))
        for fam in FAMS:
            sub = W[W.model_family == fam]
            d = (sub[a] - sub[b]).values
            fam_rows.append(dict(judge=jn, contrast=f"{a}-{b}", family=fam, n=len(d),
                                 mean=round(d.mean(), 2), min=round(d.min(), 2), max=round(d.max(), 2)))
A = pd.DataFrame(agg); A.to_csv(OUT / "r6_adaptation_shift.csv", index=False)
F = pd.DataFrame(fam_rows); F.to_csv(OUT / "r6_adaptation_shift_family.csv", index=False)
print("\n=== T1/T2 adaptation shifts (20 controlled cells; *_orig = original-grid checkpoints) ===")
print(A[A.scope == "all5"].drop(columns="scope").to_string(index=False))
print("\n=== Per-family mean shift (n=4 cells each) ===")
print(F.pivot_table(index=["judge", "contrast"], columns="family", values="mean", sort=False).to_string())

# ---------------------------------------------------------------- T2 / T3 TOST
trows = []
for jn, W in C.groupby("judge", sort=False):
    for a, b in [("lora", "fft"), ("qlora", "fft"), ("lora", "qlora")]:
        for scope, sub in [("all5", W), ("restricted3", W[W.model_family.isin(RESTRICT)])]:
            d = (sub[a] - sub[b]).values
            wp = stats.wilcoxon(d).pvalue
            for mg in (3.0, 5.0, 7.5):
                trows.append(dict(judge=jn, scope=scope, contrast=f"{a}-{b}", **tost(d, mg), wilcoxon_p=round(wp, 4)))
T = pd.DataFrame(trows); T.to_csv(OUT / "r6_tost_3judge.csv", index=False)
print("\n=== T2/T3 TOST per judge ===")
print(T.to_string(index=False))

# cross-check against the published HB-Cls per-cell file
pc = pd.read_csv(ROOT / "artifacts" / "revision_stats" / "phase5_tost_percell_5fam_strict.csv")
hb = C[C.judge == "HB-Cls"]
for pair in ["lora-fft", "qlora-fft", "lora-qlora"]:
    a, b = pair.split("-")
    mine = np.sort((hb[a] - hb[b]).round(3).values); pub = np.sort(pc[pc.pair == pair]["diff"].round(3).values)
    assert np.allclose(mine, pub, atol=1e-3), pair
print("HB-Cls per-cell differences match phase5_tost_percell_5fam_strict.csv for all three contrasts.")

# ---------------------------------------------------------------- T4 AutoDAN + GCG
ad = master[(master.attack == "AutoDAN") & (master.evaluator == "Llama-2-13b-cls")]
adw = ad.pivot_table(index=["model_family", "config"], columns="dataset", values="ASR")
adw["mean_HB_JBB"] = adw[["HarmBench", "JBB"]].mean(1)
adw = adw.reset_index().assign(attack="AutoDAN")
gcg_rows, mc_rows = [], []
GCGCELL = {"qwen25": "qwen25_14b"}
lab = {}
if not RAW:
    gl = pd.read_csv(LABELS / "original_grid.csv.gz")
    gl = gl[(gl.attack == "GCG") & (gl.evaluator == "HB-Cls")]
for fam in FAMS:
    for cfg in ["base_fp16", "lora", "fft"]:
        if RAW:
            cell = f"{GCGCELL.get(fam, fam)}_{cfg}_custom"
            r = json.load(open(HB / "results_jbb" / "GCG" / cell / "results" / f"{cell}.json"))
            lab[(fam, cfg)] = {k: int(v[0]["label"]) for k, v in r.items()}
        else:
            sub = gl[(gl.family == fam) & (gl.config == cfg)]
            assert len(sub) == 100, (fam, cfg, len(sub))
            lab[(fam, cfg)] = dict(zip(sub.behavior_id, sub.label.astype(int)))
        gcg_rows.append(dict(model_family=fam, config=cfg, attack="GCG", JBB=100.0 * np.mean(list(lab[(fam, cfg)].values()))))
    for a, b in [("lora", "base_fp16"), ("fft", "lora"), ("fft", "base_fp16")]:
        keys = sorted(set(lab[(fam, a)]) & set(lab[(fam, b)]))
        x = np.array([lab[(fam, a)][k] for k in keys]); y = np.array([lab[(fam, b)][k] for k in keys])
        n10, n01 = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
        p = stats.binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        mc_rows.append(dict(family=fam, contrast=f"{a}-{b}", n=len(keys), delta_pp=100 * (x.mean() - y.mean()),
                            only_A=n10, only_B=n01, mcnemar_exact_p=round(p, 4)))
G = pd.DataFrame(gcg_rows)
pd.concat([adw, G], ignore_index=True).to_csv(OUT / "r6_autodan_gcg_family.csv", index=False, float_format="%.2f")
M = pd.DataFrame(mc_rows); M.to_csv(OUT / "r6_gcg_mcnemar.csv", index=False)
print("\n=== AutoDAN HB-Cls per family ===")
print(adw.round(2).to_string(index=False))
b = adw[adw.config == "base_fp16"].set_index("model_family")["mean_HB_JBB"]
for cfg in ["base_4bit", "lora", "qlora", "fft"]:
    s = adw[adw.config == cfg].set_index("model_family")["mean_HB_JBB"] - b
    dd = (ad[ad.config == cfg].set_index(["model_family", "dataset"]).ASR
          - ad[ad.config == "base_fp16"].set_index(["model_family", "dataset"]).ASR)
    print(f"  AutoDAN {cfg:9s} dASR per family: " + ", ".join(f"{k} {v:+.1f}" for k, v in s.items())
          + f" | aggregate (n=10) {dd.mean():+.2f}")
print("\n=== GCG JBB-100 HB-Cls ===")
print(G.pivot_table(index="model_family", columns="config", values="JBB").to_string())
print(M.to_string(index=False))

# ---------------------------------------------------------------- template side check (ArtPrompt)
# The two side checks below read raw scored files that are not part of the release.
if not RAW:
    print("\n(ArtPrompt template check skipped: raw scored files not present)")
if RAW:
    tk = {}
    for c in ["base", "lora", "fft"]:
        recs = [json.loads(l) for l in open(SCORED / f"llama31_{c}_jbb_artprompt_tokchat_hbcls.jsonl") if l.strip()]
        tk[c] = 100 * np.mean([int(r["jailbroken"]) for r in recs])
    g = C[(C.judge == "HB-Cls") & (C.model_family == "llama31") & (C.attack == "ArtPrompt") & (C.dataset == "JBB")].iloc[0]
    print("\n=== ArtPrompt template check, Llama-3.1 JBB-100, HB-Cls ===")
    print(f"  tokenizer template: base {tk['base']:.1f}  lora {tk['lora']:.1f}  fft {tk['fft']:.1f}")
    print(f"  grid cells        : base_fp16 {g.base_fp16:.1f} (original grid)  lora {g.lora:.1f}  fft {g.fft:.1f} (seed-42 grid)")

# ---------------------------------------------------------------- attack split of T1 shifts
# The seed-42 ArtPrompt cells were generated with the Alpaca prompt template, whereas the original-grid base cells used the tokenizer chat template.
# DeepInception used the tokenizer template in both pipelines, so the DI-only split is the
# format-matched base -> adapted comparison.
srows = []
for jn, W in C.groupby("judge", sort=False):
    for att in ["DeepInception", "ArtPrompt"]:
        sub = W[W.attack == att]
        for a, b in [("lora", "base_fp16"), ("qlora", "base_fp16"), ("fft", "base_fp16"), ("lora", "fft")]:
            srows.append(dict(judge=jn, attack=att, contrast=f"{a}-{b}", **shift_stats((sub[a] - sub[b]).values)))
S = pd.DataFrame(srows); S.to_csv(OUT / "r6_adaptation_shift_attacksplit.csv", index=False)
print("\n=== Adaptation shifts split by attack (n=10 each) ===")
print(S[["judge", "attack", "contrast", "n", "mean", "t_ci95_lo", "t_ci95_hi", "boot_ci95_lo", "boot_ci95_hi",
         "wilcoxon_p", "p_one_sided_mean_gt_5"]].to_string(index=False))

if RAW:
    # DeepInception base pipeline check: phase-5 driver on the untuned Llama-3.1 vs original-grid base cell
    recs = [json.loads(l) for l in open(SCORED / "llama31_base_jbb_hbcls.jsonl") if l.strip()]
    g = C[(C.judge == "HB-Cls") & (C.model_family == "llama31") & (C.attack == "DeepInception") & (C.dataset == "JBB")].iloc[0]
    print(f"\nDeepInception base check, Llama-3.1 JBB-100, HB-Cls: phase-5 driver {100*np.mean([int(r['jailbroken']) for r in recs]):.1f}"
          f" vs original-grid base_fp16 {g.base_fp16:.1f}")
