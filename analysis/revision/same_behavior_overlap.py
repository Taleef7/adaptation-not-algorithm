#!/usr/bin/env python3
"""Do LoRA and FFT jailbreak the SAME behaviors?

Claim 1 says LoRA and FFT degrade safety by an equivalent aggregate magnitude.
That leaves open a sharper question: is it the same underlying vulnerability?
Two possibilities with identical aggregate ASR:

  (a) shared mechanism -- both methods break the same behaviors, so the overlap
      should approach the smaller of the two success sets;
  (b) method-specific -- each method opens a different subset, and equal
      magnitude is a coincidence of counts rather than of content.

Raw Jaccard alone cannot distinguish these, because behaviors differ in intrinsic
difficulty: some are easy for everything, which forces overlap upward regardless
of mechanism. We therefore compare the observed overlap against the overlap
expected if the two success sets were drawn independently at the same rates,
using Fisher's exact test per cell and a Mantel-Haenszel pooled odds ratio.

Input : artifacts/transfer_same_behavior_alpaca.csv
        (fam, attack, ds, nL, nF, both, either, jaccard, agree)
Output: artifacts/revision/same_behavior_overlap.csv
"""

import csv
import os

import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO, "artifacts", "transfer_same_behavior_alpaca.csv")
OUT = os.path.join(REPO, "artifacts", "revision", "same_behavior_overlap.csv")

N_BEHAVIORS = {"jbb": 100, "harmbench": 400}


def main():
    rows = []
    tables = []

    print("=" * 92)
    print("SAME-BEHAVIOR OVERLAP: LoRA vs FFT jailbreak sets (Alpaca canonical, HB-Cls)")
    print("=" * 92)
    hdr = (f"{'family':<9}{'attack':<15}{'bench':<10}{'N':>4}{'nL':>5}{'nF':>5}"
           f"{'both':>6}{'exp':>7}{'Jacc':>7}{'OR':>8}{'Fisher p':>10}")
    print(hdr)
    print("-" * len(hdr))

    for r in csv.DictReader(open(SRC)):
        N = N_BEHAVIORS[r["ds"]]
        nL, nF, both = int(r["nL"]), int(r["nF"]), int(r["both"])
        # 2x2 over all N behaviors: rows = jailbroken under LoRA or not,
        # cols = jailbroken under FFT or not.
        a = both                 # both
        b = nL - both            # LoRA only
        c = nF - both            # FFT only
        d = N - a - b - c        # neither
        expected = nL * nF / N   # overlap under independence
        odds, p = stats.fisher_exact([[a, b], [c, d]])
        jacc = float(r["jaccard"])
        print(f"{r['fam']:<9}{r['attack']:<15}{r['ds']:<10}{N:>4}{nL:>5}{nF:>5}"
              f"{both:>6}{expected:>7.1f}{jacc:>7.2f}{odds:>8.2f}{p:>10.2g}")
        tables.append((a, b, c, d))
        rows.append(dict(family=r["fam"], attack=r["attack"], bench=r["ds"], N=N,
                         n_lora=nL, n_fft=nF, both=both, expected_by_chance=expected,
                         jaccard=jacc, odds_ratio=odds, fisher_p=p,
                         agreement=float(r["agree"])))

    # ---- pooled ------------------------------------------------------------
    jaccs = np.array([x["jaccard"] for x in rows])
    obs = sum(t[0] for t in tables)
    exp = sum(x["expected_by_chance"] for x in rows)
    num = sum(a * d / (a + b + c + d) for a, b, c, d in tables)
    den = sum(b * c / (a + b + c + d) for a, b, c, d in tables)
    or_mh = num / den
    # MH chi-square
    e_sum = v_sum = a_sum = 0.0
    for a, b, c, d in tables:
        n = a + b + c + d
        a_sum += a
        e_sum += (a + b) * (a + c) / n
        v_sum += ((a + b) * (c + d) * (a + c) * (b + d)) / (n * n * (n - 1))
    chi2 = (abs(a_sum - e_sum) - 0.5) ** 2 / v_sum
    p_mh = stats.chi2.sf(chi2, 1)

    print("\nPOOLED")
    print(f"  mean Jaccard            = {jaccs.mean():.3f}  (median {np.median(jaccs):.3f}, "
          f"range {jaccs.min():.2f}-{jaccs.max():.2f})")
    print(f"  observed co-jailbreaks  = {obs}")
    print(f"  expected if independent = {exp:.1f}")
    print(f"  ratio observed/expected = {obs / exp:.2f}x")
    print(f"  Mantel-Haenszel OR      = {or_mh:.2f}  chi2={chi2:.1f}  p={p_mh:.3g}")
    n_sig = sum(1 for x in rows if x["fisher_p"] < 0.05 and x["odds_ratio"] > 1)
    print(f"  cells with significant positive association: {n_sig}/{len(rows)}")

    print("\nINTERPRETATION")
    print("  Overlap is well ABOVE chance -> a shared component (behavior difficulty).")
    print("  Overlap is well BELOW complete (mean Jaccard ~0.29) -> most jailbroken")
    print("  behaviors are opened by one method and not the other. Equivalent")
    print("  magnitude, substantially different content.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {os.path.relpath(OUT, REPO)}")


if __name__ == "__main__":
    main()
