#!/usr/bin/env python3
"""Attack-split TOST on the 5-family seed-42 grid (HB-Cls).
Separates method effect from attack-sampling noise: ArtPrompt is greedy/deterministic,
DeepInception is do_sample (no RNG seed). Reads the per-cell dump written by phase5_tost_5fam.py.
Writes artifacts/revision_stats/phase5_tost_attacksplit_5fam.csv.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parents[2] / "artifacts" / "revision_stats"
dd = pd.read_csv(OUT / "phase5_tost_percell_5fam_strict.csv")

def tost(d, mg=5.0):
    d = np.asarray(d, float); n = len(d); m = d.mean(); sd = d.std(ddof=1); se = sd/np.sqrt(n)
    p = max(stats.t.sf((m+mg)/se, n-1), stats.t.cdf((m-mg)/se, n-1))
    try: _, w = stats.wilcoxon(d, alternative="two-sided")
    except ValueError: w = np.nan
    return dict(n=n, mean=round(m,4), sd=round(sd,4), tost_p=round(p,4),
               equivalent=bool(p < 0.05), wilcoxon_p=round(w,4))

rows = []
for pair in ["lora-fft", "qlora-fft", "lora-qlora"]:
    s = dd[dd.pair == pair]
    subsets = {
        "all_DI+AP": s,
        "ArtPrompt_only_deterministic": s[s.attack == "ArtPrompt"],
        "DeepInception_only_stochastic": s[s.attack == "DeepInception"],
        "all_minus_qwen25": s[s.family != "qwen25"],
    }
    for label, sub in subsets.items():
        r = tost(sub["diff"]); r.update(pair=pair, subset=label); rows.append(r)

T = pd.DataFrame(rows)[["pair","subset","n","mean","sd","tost_p","equivalent","wilcoxon_p"]]
T.to_csv(OUT / "phase5_tost_attacksplit_5fam.csv", index=False)
print(T.to_string(index=False))
print(f"\nWrote {OUT}/phase5_tost_attacksplit_5fam.csv")
