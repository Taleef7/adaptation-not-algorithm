#!/usr/bin/env python3
"""
TOST provenance on the original grid: per-cell differences, aggregate and per-family TOST.
Black-box, HB-Cls. Reads artifacts/master_asr_long.csv only.
Writes per-cell difference dumps + aggregate and per-family TOST tables.
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
PAIRS = [("lora", "fft"), ("qlora", "fft"), ("lora", "qlora")]


def tost(d, margin):
    d = np.asarray(d, float)
    n = len(d); m = d.mean(); sd = d.std(ddof=1); se = sd / np.sqrt(n)
    t_low = (m + margin) / se;  p_low = stats.t.sf(t_low, n - 1)   # mean > -margin
    t_up = (m - margin) / se;   p_up = stats.t.cdf(t_up, n - 1)    # mean <  margin
    return dict(n=n, mean=m, sd=sd, se=se, ci_lo=m - 1.96 * se, ci_hi=m + 1.96 * se,
                margin=margin, t_lower=t_low, t_upper=t_up, p_tost=max(p_low, p_up),
                equivalent=bool(max(p_low, p_up) < 0.05))


def wide(df):
    bb = df[(df.attack.isin(BB)) & (df.evaluator == HBCLS)]
    return bb.pivot_table(index=["model_family", "attack", "dataset"],
                          columns="config", values="ASR").reset_index()


def main():
    a = pd.read_csv(ART / "master_asr_long.csv")
    W = wide(a)

    # ---- per-cell difference dump (aggregate, n=30) ----
    dumps = []
    for A, B in PAIRS:
        sub = W.dropna(subset=[A, B]).copy()
        sub["diff"] = sub[A] - sub[B]
        for _, r in sub.iterrows():
            dumps.append(dict(pair=f"{A}-{B}", family=r.model_family, attack=r.attack,
                              dataset=r.dataset, asr_A=r[A], asr_B=r[B], diff=r["diff"]))
    dd = pd.DataFrame(dumps)
    dd.to_csv(OUT / "tost_percell_differences.csv", index=False, float_format="%.3f")

    # ---- aggregate TOST ----
    agg = []
    for A, B in PAIRS:
        d = dd[dd.pair == f"{A}-{B}"]["diff"].to_numpy()
        try:
            w, pdiff = stats.wilcoxon(d, alternative="two-sided")
        except ValueError:
            w, pdiff = np.nan, np.nan
        for mg in (3.0, 5.0):
            row = tost(d, mg); row.update(pair=f"{A}-{B}", scope="aggregate",
                                          family="ALL", wilcoxon_p=pdiff)
            agg.append(row)
    # ---- per-family TOST (n=6 per cell) ----
    for fam in sorted(W.model_family.unique()):
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
    T.to_csv(OUT / "tost_full.csv", index=False, float_format="%.4f")

    print("=== Per-cell differences (aggregate, n=30 each) ===")
    for A, B in PAIRS:
        d = dd[dd.pair == f"{A}-{B}"]
        vals = ", ".join(f"{x:+.2f}" for x in d["diff"])
        print(f"\n{A}-{B}: mean={d['diff'].mean():+.3f}  SD={d['diff'].std(ddof=1):.3f}  "
              f"n={len(d)}\n  diffs: {vals}")
    print("\n=== Aggregate TOST ===")
    print(T[T.scope == "aggregate"][["pair", "mean", "sd", "margin", "p_tost", "equivalent"]]
          .to_string(index=False))
    print("\n=== Per-family TOST (n=6 per cell; expect underpowered) ===")
    print(T[(T.scope == "per-family") & (T.margin == 5.0)]
          [["family", "pair", "mean", "sd", "p_tost", "equivalent"]].to_string(index=False))
    print(f"\nWrote {OUT/'tost_percell_differences.csv'} and {OUT/'tost_full.csv'}")


if __name__ == "__main__":
    main()
