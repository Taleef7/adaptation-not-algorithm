#!/usr/bin/env python3
"""
Seed-42 controlled-grid TOST (Finding: LoRA ~= FFT). Paper appendix "TOST Equivalence Test".
Mirrors analysis/audits/tost_dump.py exactly (Schuirmann's TOST, paired config diffs, +-3/+-5 pp,
contrasts LoRA-FFT, QLoRA-FFT, LoRA-QLoRA), but on the seed-42 re-trained cells:
  5 families x 2 attacks {DeepInception, ArtPrompt} x 2 datasets {JBB-100, HarmBench-400},
  evaluator = HB-Cls (Llama-2-13b-cls).

Reports:
  (1) the aggregate TOST (n=20 paired cells per contrast);
  (2) per-family TOST (n=4 per contrast, underpowered, descriptive only);
  (3) the per-cell HB-Cls change between the original-grid and seed-42 checkpoints;
  (4) the margin rationale (training-seed noise and DeepInception sampling noise).
Note: the 95% CI printed here is normal-approximation; the paper's 90% t-based CI is
computed by analysis/revision/camera_ready_stats.py from the per-cell file written here.
"""
import json, glob, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
SCORED = REPO / "toolkits/HarmBench/results_phase5/scored"
SCORES = REPO / "toolkits/HarmBench/results_phase5/scores"
ART = REPO / "artifacts"
OUT = REPO / "artifacts" / "revision_stats"
OUT.mkdir(parents=True, exist_ok=True)

HBCLS = "Llama-2-13b-cls"
PAIRS = [("lora", "fft"), ("qlora", "fft"), ("lora", "qlora")]
DS = {"jbb": "JBB", "harmbench": "HarmBench"}
FAMS = ["llama31", "gemma2", "qwen3", "phi4", "qwen25"]
METHODS = ["lora", "qlora", "fft"]


def asr_of(path):
    n = jb = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n += 1
            jb += int(r.get("jailbroken", 0))
    return n, jb, (100.0 * jb / n if n else float("nan"))


def parse(fname):
    """-> (family, method, dataset_key, attack) or None."""
    base = re.sub(r"\.jsonl$", "", fname)
    attack = "ArtPrompt" if "artprompt" in base else "DeepInception"
    for suf in ("_artprompt_hbcls", "_hbcls", "_labels"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    base = base.replace("_seed42", "")
    m = re.match(r"(llama31|gemma2|qwen3|phi4|qwen25)_(lora|qlora|fft)_(jbb|harmbench)$", base)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3), attack


LABELS = REPO / "data" / "evaluator_labels" / "seed42_grid.csv.gz"


def load_seed42():
    if not SCORED.is_dir():
        # Release mode: behavior-level HB-Cls labels shipped in data/evaluator_labels/.
        lab = pd.read_csv(LABELS)
        lab = lab[lab.evaluator == "HB-Cls"]
        g = lab.groupby(["family", "config", "benchmark", "attack"]).label.agg(["size", "sum"]).reset_index()
        return pd.DataFrame(dict(model_family=g.family, config=g.config, dataset=g.benchmark, attack=g.attack,
                                 n=g["size"], successes=g["sum"], ASR=100.0 * g["sum"] / g["size"],
                                 file=LABELS.name))
    rows = []
    files = list(SCORED.glob("*seed42*.jsonl")) + list(SCORES.glob("*seed42*.jsonl"))
    for p in files:
        info = parse(p.name)
        if not info:
            print(f"  WARN unparsed: {p.name}")
            continue
        fam, method, dskey, attack = info
        n, jb, asr = asr_of(p)
        rows.append(dict(model_family=fam, config=method, dataset=DS[dskey],
                         attack=attack, n=n, successes=jb, ASR=asr, file=p.name))
    df = pd.DataFrame(rows)
    # guard: expect 60 unique (family, config, dataset, attack) cells
    key = ["model_family", "config", "dataset", "attack"]
    dup = df[df.duplicated(key, keep=False)]
    if len(dup):
        print("  WARN duplicate cells:\n", dup[key + ["file"]].to_string(index=False))
    df = df.drop_duplicates(key)
    return df


def tost(d, margin):
    d = np.asarray(d, float)
    n = len(d); m = d.mean(); sd = d.std(ddof=1); se = sd / np.sqrt(n)
    t_low = (m + margin) / se;  p_low = stats.t.sf(t_low, n - 1)
    t_up = (m - margin) / se;   p_up = stats.t.cdf(t_up, n - 1)
    return dict(n=n, mean=m, sd=sd, se=se, ci_lo=m - 1.96 * se, ci_hi=m + 1.96 * se,
                margin=margin, p_tost=max(p_low, p_up),
                equivalent=bool(max(p_low, p_up) < 0.05))


def wide(df):
    return df.pivot_table(index=["model_family", "attack", "dataset"],
                          columns="config", values="ASR").reset_index()


def run_tost(W, label):
    dumps = []
    for A, B in PAIRS:
        sub = W.dropna(subset=[A, B]).copy()
        sub["diff"] = sub[A] - sub[B]
        for _, r in sub.iterrows():
            dumps.append(dict(pair=f"{A}-{B}", family=r.model_family, attack=r.attack,
                              dataset=r.dataset, asr_A=r[A], asr_B=r[B], diff=r["diff"]))
    dd = pd.DataFrame(dumps)
    dd.to_csv(OUT / f"phase5_tost_percell_{label}.csv", index=False, float_format="%.3f")

    agg = []
    for A, B in PAIRS:
        d = dd[dd.pair == f"{A}-{B}"]["diff"].to_numpy()
        try:
            _, pdiff = stats.wilcoxon(d, alternative="two-sided")
        except ValueError:
            pdiff = np.nan
        for mg in (3.0, 5.0):
            row = tost(d, mg); row.update(pair=f"{A}-{B}", scope="aggregate",
                                          family="ALL", wilcoxon_p=pdiff)
            agg.append(row)
    for fam in FAMS:
        for A, B in PAIRS:
            sub = W[W.model_family == fam].dropna(subset=[A, B])
            d = (sub[A] - sub[B]).to_numpy()
            if len(d) < 2:
                continue
            for mg in (3.0, 5.0):
                row = tost(d, mg); row.update(pair=f"{A}-{B}", scope="per-family",
                                              family=fam, wilcoxon_p=np.nan)
                agg.append(row)
    T = pd.DataFrame(agg)[["scope", "family", "pair", "n", "mean", "sd", "se",
                           "ci_lo", "ci_hi", "margin", "p_tost", "equivalent", "wilcoxon_p"]]
    T.to_csv(OUT / f"phase5_tost_full_{label}.csv", index=False, float_format="%.4f")
    return dd, T


def old_vs_new(seed42):
    """Original-grid vs seed-42 HB-Cls delta per cell; qwen3-QLoRA shown separately."""
    a = pd.read_csv(ART / "master_asr_long.csv")
    old = a[(a.evaluator == HBCLS) & (a.attack.isin(["DeepInception", "ArtPrompt"]))]
    old = old[old.model_family.isin(FAMS)][["model_family", "config", "dataset", "attack", "ASR"]]
    old = old.rename(columns={"ASR": "ASR_old"})
    new = seed42[["model_family", "config", "dataset", "attack", "ASR"]].rename(columns={"ASR": "ASR_new"})
    j = new.merge(old, on=["model_family", "config", "dataset", "attack"], how="left")
    j["delta"] = j["ASR_new"] - j["ASR_old"]
    j = j.sort_values(["model_family", "config", "attack", "dataset"])
    j.to_csv(OUT / "phase5_old_vs_new_hbcls_5fam.csv", index=False, float_format="%.2f")
    return j


def main():
    print("=== Loading seed-42 scored cells ===")
    df = load_seed42()
    print(f"  loaded {len(df)} cells (expect 60)")
    grid = df.pivot_table(index=["model_family", "config"], columns=["attack", "dataset"], values="ASR")
    print("\n=== Seed-42 ASR grid (HB-Cls) ===")
    print(grid.round(2).to_string())

    W = wide(df)
    dd, T = run_tost(W, "5fam_strict")

    print("\n=== Per-cell PEFT-vs-FFT differences (aggregate n per contrast) ===")
    for A, B in PAIRS:
        d = dd[dd.pair == f"{A}-{B}"]
        vals = ", ".join(f"{x:+.1f}" for x in d["diff"])
        print(f"  {A}-{B}: mean={d['diff'].mean():+.2f}  SD={d['diff'].std(ddof=1):.2f}  n={len(d)}  [{vals}]")

    print("\n=== AGGREGATE TOST (n=20 paired cells per contrast) ===")
    print(T[T.scope == "aggregate"][["pair", "n", "mean", "ci_lo", "ci_hi", "margin", "p_tost", "equivalent"]]
          .to_string(index=False))
    print("\n=== PER-FAMILY TOST  (n=4 per contrast, underpowered) ===")
    print(T[(T.scope == "per-family") & (T.margin == 5.0)]
          [["family", "pair", "n", "mean", "sd", "p_tost", "equivalent"]].to_string(index=False))

    print("\n=== Original grid vs seed-42 HB-Cls delta ===")
    j = old_vs_new(df)
    q = j[(j.model_family == "qwen3") & (j.config == "qlora")]
    print("  -- qwen3 QLoRA (highlighted) --")
    print(q[["attack", "dataset", "ASR_old", "ASR_new", "delta"]].to_string(index=False))
    print(f"  qwen3-QLoRA mean |delta| = {q['delta'].abs().mean():.2f} pp over {len(q)} cells")
    print(f"  all-cell mean |delta| = {j['delta'].abs().mean():.2f} pp (n={j['delta'].notna().sum()})")

    print("\n=== Margin rationale ===")
    print("  +-5pp equivalence bound is calibrated against TWO noise sources:")
    print("   (a) training-seed noise: <=5.2pp (prior multi-seed spread, paper Sec multi-seed);")
    print("   (b) attack-sampling noise: DeepInception is do_sample (no RNG seed) -> per-cell ASR")
    print("       carries sampling noise; ArtPrompt is greedy/deterministic -> contributes none.")
    print("  The bound must exceed BOTH; +-5pp does. (We also report the tighter +-3pp.)")

    print(f"\nWrote: {OUT}/phase5_tost_full_5fam_strict.csv, phase5_tost_percell_5fam_strict.csv, phase5_old_vs_new_hbcls.csv")


if __name__ == "__main__":
    main()
