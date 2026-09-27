#!/usr/bin/env python3
"""Track 2 -- does LoRA-approx-FFT survive a second dataset and a second recipe?

Tests whether the equivalence result is an artifact of the single training
recipe used in the main grid (Alpaca-cleaned, 1 epoch). We trained LoRA and FFT cells under a factorial set of recipe perturbations and
re-ran the same TOST equivalence test used for the canonical grid.

Design (all seed 42, canonical recipe otherwise, HB-Cls, DeepInception+ArtPrompt
x JBB-100 + HarmBench-400):

  arm                 dataset   epochs   isolates
  ------------------  --------  -------  --------------------------------------
  canonical           Alpaca    1        (reference: the main-paper grid)
  dataset-isolated    Dolly     1        dataset alone
  epochs-isolated     Alpaca    3        training length alone
  step-matched        Dolly     ~1*      dataset alone, optimizer steps matched
  combined            Dolly     3        both at once

* step-matched holds the number of optimizer steps equal to the Alpaca-1-epoch
  budget, so it separates "different data" from "more gradient steps."

Outputs artifacts/revision/track2_equivalence.csv and prints a report.
"""

import csv
import os

import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(REPO, "artifacts")
OUT = os.path.join(ART, "revision")

ARMS = [
    ("dataset-isolated", "track2_dataset_isolation_dolly1ep.csv", "Dolly", "1"),
    ("step-matched",     "track2_dollystep_stepmatched.csv",      "Dolly", "1 (step-matched)"),
    ("epochs-isolated",  "track2_alpaca3ep_epochs_isolation.csv", "Alpaca", "3"),
    ("combined",         "track2_dolly3ep_3fam.csv",              "Dolly", "3"),
]

BOUND = 5.0  # pp, same equivalence bound as the canonical grid


def tost(diffs, bound=BOUND):
    """Schuirmann's TOST for paired differences against +/- bound."""
    d = np.asarray(diffs, float)
    n = len(d)
    m, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    if se == 0:
        return m, sd, (0.0 if abs(m) < bound else 1.0), n
    t_lo = (m - (-bound)) / se          # H0: diff <= -bound
    t_hi = (m - bound) / se             # H0: diff >= +bound
    p_lo = stats.t.sf(t_lo, n - 1)
    p_hi = stats.t.cdf(t_hi, n - 1)
    return m, sd, max(p_lo, p_hi), n


def load_canonical():
    """Canonical per-cell LoRA-FFT differences, keyed by (family, attack, bench)."""
    path = os.path.join(REPO, "artifacts/revision_stats/phase5_tost_percell_5fam_strict.csv")
    out = {}
    for r in csv.DictReader(open(path)):
        if r["pair"] != "lora-fft":
            continue
        bench = "harmbench" if r["dataset"].lower().startswith("harm") else "jbb"
        out[(r["family"], r["attack"], bench)] = float(r["diff"])
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    canon = load_canonical()
    rows = []

    print("=" * 84)
    print("TRACK 2 -- LoRA vs FFT equivalence under recipe perturbations (HB-Cls, +/-5pp)")
    print("=" * 84)
    hdr = (f"{'arm':<18}{'data':<7}{'ep':<18}{'n':>3}{'mean':>8}{'sd':>7}"
           f"{'TOST p':>9}{'equiv':>7}{'Wilcox':>8}")
    print(hdr)
    print("-" * len(hdr))

    # reference row: full canonical grid
    cdiffs_all = list(canon.values())
    m, sd, p, n = tost(cdiffs_all)
    w = stats.wilcoxon(cdiffs_all).pvalue
    print(f"{'canonical':<18}{'Alpaca':<7}{'1':<18}{n:>3}{m:>+8.2f}{sd:>7.2f}"
          f"{p:>9.4f}{str(p < 0.05):>7}{w:>8.3f}")
    rows.append(dict(arm="canonical", dataset="Alpaca", epochs="1", n=n, mean=m,
                     sd=sd, tost_p=p, equivalent=p < 0.05, wilcoxon_p=w,
                     families="all 5"))

    for arm, fname, ds, ep in ARMS:
        path = os.path.join(ART, fname)
        recs = list(csv.DictReader(open(path)))
        diffs = [float(r["delta_lora_minus_fft"]) for r in recs]
        fams = sorted({r["family"] for r in recs})
        m, sd, p, n = tost(diffs)
        w = stats.wilcoxon(diffs).pvalue
        print(f"{arm:<18}{ds:<7}{ep:<18}{n:>3}{m:>+8.2f}{sd:>7.2f}"
              f"{p:>9.4f}{str(p < 0.05):>7}{w:>8.3f}")
        rows.append(dict(arm=arm, dataset=ds, epochs=ep, n=n, mean=m, sd=sd,
                         tost_p=p, equivalent=p < 0.05, wilcoxon_p=w,
                         families=",".join(fams)))

        # Like-for-like: restrict the canonical grid to this arm's own cells.
        matched = [canon[k] for k in
                   ((r["family"], r["attack"], r["benchmark"]) for r in recs)
                   if k in canon]
        if len(matched) == len(diffs) and len(matched) > 2:
            m2, sd2, p2, n2 = tost(matched)
            print(f"{'  (canonical, same cells)':<43}{n2:>3}{m2:>+8.2f}{sd2:>7.2f}"
                  f"{p2:>9.4f}{str(p2 < 0.05):>7}")
            rows.append(dict(arm=f"canonical-matched-to-{arm}", dataset="Alpaca",
                             epochs="1", n=n2, mean=m2, sd=sd2, tost_p=p2,
                             equivalent=p2 < 0.05, wilcoxon_p="",
                             families=",".join(fams)))

    # Does recipe perturbation shift the MEAN, or just inflate the VARIANCE of the
    # per-cell gaps? Compare each arm against the canonical cells it matches.
    print("\nVariance of per-cell LoRA-FFT gaps vs the matched canonical cells\n")
    print(f"{'arm':<18}{'sd_arm':>8}{'sd_canon':>10}{'F':>7}{'Levene p':>10}{'mean shift':>12}")
    for arm, fname, _, _ in ARMS:
        recs = list(csv.DictReader(open(os.path.join(ART, fname))))
        d = np.array([float(r["delta_lora_minus_fft"]) for r in recs])
        c = np.array([canon[k] for k in
                      ((r["family"], r["attack"], r["benchmark"]) for r in recs)
                      if k in canon])
        if len(c) != len(d):
            continue
        lev = stats.levene(d, c, center="median")
        F = d.var(ddof=1) / c.var(ddof=1)
        print(f"{arm:<18}{d.std(ddof=1):>8.2f}{c.std(ddof=1):>10.2f}{F:>7.2f}"
              f"{lev.pvalue:>10.4f}{d.mean() - c.mean():>+12.2f}")
        for r in rows:
            if r["arm"] == arm:
                r["sd_canonical_matched"] = c.std(ddof=1)
                r["variance_ratio_F"] = F
                r["levene_p"] = lev.pvalue
                r["mean_shift_vs_canonical"] = d.mean() - c.mean()

    # Per-family direction check within each arm: is any separation consistent?
    print("\nPer-family mean LoRA-FFT (pp) by arm -- consistency of direction\n")
    fams_all = sorted({r["family"] for _, f, _, _ in ARMS
                       for r in csv.DictReader(open(os.path.join(ART, f)))})
    print(f"{'arm':<18}" + "".join(f"{f:>12}" for f in fams_all))
    for arm, fname, _, _ in ARMS:
        recs = list(csv.DictReader(open(os.path.join(ART, fname))))
        line = f"{arm:<18}"
        for f in fams_all:
            v = [float(r["delta_lora_minus_fft"]) for r in recs if r["family"] == f]
            line += f"{np.mean(v):>+12.2f}" if v else f"{'--':>12}"
        print(line)

    path = os.path.join(OUT, "track2_equivalence.csv")
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=keys)
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"\nwrote {os.path.relpath(path, REPO)}")


if __name__ == "__main__":
    main()
