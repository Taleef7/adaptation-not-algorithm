#!/usr/bin/env python3
"""
PAIR three-way TOST (adaptive attacker, seed-42 controlled subset).
Mirrors phase5_tost_5fam.py's Schuirmann TOST EXACTLY (paired config diffs, +-3/+-5 pp,
contrasts LoRA-FFT, QLoRA-FFT, LoRA-QLoRA), but on the black-box PAIR grid:
  3 families {llama31, gemma2, qwen3} x 1 attacker {PAIR (Mistral-7B-Instruct-v0.3,
  n_streams20 x steps3, cutoff10)} x 2 datasets {JBB-100, HB-400}
  evaluator = HB-Cls (Llama-2-13b-cls), reading results/<key>_seed42_custom.json.

Paired diffs are taken over (family x dataset) => n=6 per contrast.
Reported in the appendix paragraph 'PAIR under the controlled recipe'.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
PAIR_JBB = REPO / "toolkits/HarmBench/results_phase5/PAIR_jbb"
PAIR_HB = REPO / "toolkits/HarmBench/results_phase5/PAIR_hb"
OUT = REPO / "artifacts" / "revision_stats"
OUT.mkdir(parents=True, exist_ok=True)

PAIRS = [("lora", "fft"), ("qlora", "fft"), ("lora", "qlora")]
FAMS = ["llama31", "gemma2", "qwen3"]
METHODS = ["lora", "qlora", "fft"]


def asr_of(path, n_expected):
    """HarmBench results json: {behavior_id: [ {label:0/1, ...}, ... ]}."""
    d = json.load(open(path))
    n = jb = 0
    for _, items in d.items():
        for it in items:
            n += 1
            jb += 1 if it.get("label") == 1 else 0
    assert n == n_expected, f"{path}: got n={n}, expected {n_expected}"
    return n, jb, 100.0 * jb / n


def load_pair():
    rows = []
    for fam in FAMS:
        for method in METHODS:
            key = f"{fam}_{method}"
            jf = PAIR_JBB / f"{key}_seed42_pair_experiment" / "results" / f"{key}_seed42_custom.json"
            hf = PAIR_HB / f"test_{key}_seed42_custom" / "results" / f"{key}_seed42_custom.json"
            n, jb, asr = asr_of(jf, 100)
            rows.append(dict(model_family=fam, config=method, dataset="JBB", n=n, successes=jb, ASR=asr))
            n, jb, asr = asr_of(hf, 400)
            rows.append(dict(model_family=fam, config=method, dataset="HarmBench", n=n, successes=jb, ASR=asr))
    return pd.DataFrame(rows)


def tost(d, margin):
    d = np.asarray(d, float)
    n = len(d); m = d.mean(); sd = d.std(ddof=1); se = sd / np.sqrt(n)
    t_low = (m + margin) / se;  p_low = stats.t.sf(t_low, n - 1)
    t_up = (m - margin) / se;   p_up = stats.t.cdf(t_up, n - 1)
    return dict(n=n, mean=m, sd=sd, se=se, ci_lo=m - 1.96 * se, ci_hi=m + 1.96 * se,
                margin=margin, p_tost=max(p_low, p_up),
                equivalent=bool(max(p_low, p_up) < 0.05))


def wide(df):
    return df.pivot_table(index=["model_family", "dataset"], columns="config", values="ASR").reset_index()


def run_tost(W, label):
    dumps = []
    for A, B in PAIRS:
        sub = W.dropna(subset=[A, B]).copy()
        sub["diff"] = sub[A] - sub[B]
        for _, r in sub.iterrows():
            dumps.append(dict(pair=f"{A}-{B}", family=r.model_family, dataset=r.dataset,
                              asr_A=r[A], asr_B=r[B], diff=r["diff"]))
    dd = pd.DataFrame(dumps)
    dd.to_csv(OUT / f"phase5_pair_tost_percell_{label}.csv", index=False, float_format="%.3f")

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
    T.to_csv(OUT / f"phase5_pair_tost_full_{label}.csv", index=False, float_format="%.4f")
    return dd, T


def main():
    print("=== Loading PAIR (adaptive-attacker) seed-42 cells ===")
    df = load_pair()
    print(f"  loaded {len(df)} cells (expect 18)")
    grid = df.pivot_table(index=["model_family", "config"], columns="dataset", values="ASR")
    print("\n=== PAIR ASR grid (HB-Cls) ===")
    print(grid.round(2).to_string())

    W = wide(df)
    dd, T = run_tost(W, "strict")

    print("\n=== Per-cell PEFT-vs-FFT differences (n per contrast = 3 fam x 2 datasets = 6) ===")
    for A, B in PAIRS:
        d = dd[dd.pair == f"{A}-{B}"]
        vals = ", ".join(f"{x:+.1f}" for x in d["diff"])
        print(f"  {A}-{B}: mean={d['diff'].mean():+.2f}  SD={d['diff'].std(ddof=1):.2f}  n={len(d)}  [{vals}]")

    print("\n=== AGGREGATE TOST (the headline; per-cell table is NOT the test) ===")
    print(T[T.scope == "aggregate"][["pair", "n", "mean", "ci_lo", "ci_hi", "margin", "p_tost", "equivalent"]]
          .to_string(index=False))
    print("\n=== PER-FAMILY TOST (n=2 per contrast, UNDERPOWERED — descriptive only) ===")
    print(T[(T.scope == "per-family") & (T.margin == 5.0)]
          [["family", "pair", "n", "mean", "sd", "p_tost", "equivalent"]].to_string(index=False))

    print(f"\nWrote: {OUT}/phase5_pair_tost_full_strict.csv, phase5_pair_tost_percell_strict.csv")


if __name__ == "__main__":
    main()
