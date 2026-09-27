#!/usr/bin/env python3
"""GCG optimization-difficulty analysis (zero-GPU, mines existing run artifacts).

Question this answers
---------------------
GCG on JBB-100 shows FFT with substantially higher ASR than LoRA in both tested
families (llama31 -17.0 pp, qwen25 -17.0 pp under HB-Cls). Two readings:

  (a) attack-class boundary -- LoRA and FFT genuinely diverge under white-box
      suffix optimization, bounding the LoRA-approx-FFT equivalence claim;
  (b) confound -- FFT checkpoints are simply softer optimization targets, so GCG
      drives the adversarial loss lower and success follows for reasons that have
      nothing to do with safety behaviour.

Reading (a) may not be written as established until (b) is ruled out. GCG logs the
full 500-step loss trace per behavior, so (b) is directly testable at zero GPU cost.

Tests
-----
1. Paired final-loss comparison (LoRA vs FFT, same behaviors) per family.
   If FFT loss is systematically lower, (b) is live.
2. Point-biserial correlation between final loss and attack success, per cell.
   Establishes whether loss predicts success at all in this regime.
3. Loss-stratified ASR (quartile bins pooled over the family's two cells) plus a
   Mantel-Haenszel common odds ratio. If the FFT>LoRA success gap survives
   conditioning on loss, the gap is not explained by optimization difficulty.

Outputs artifacts/revision/gcg_loss_{summary,paired,strata}.csv and prints a report.
"""

import json
import os
from collections import OrderedDict

import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HB = os.path.join(REPO, "toolkits", "HarmBench")
OUT = os.path.join(REPO, "artifacts", "revision")

# GCG cells: JBB-100, seed 42. The first two families are the published probe
# (artifacts written 2026-07-21); gemma2/qwen3/phi4 were added by tracker item B2
# and completed 2026-08-05.
FAMILIES_2FAM = OrderedDict([
    ("llama31", ["llama31_base_fp16_custom", "llama31_lora_custom", "llama31_fft_custom"]),
    ("qwen25_14b", ["qwen25_14b_base_fp16_custom", "qwen25_14b_lora_custom", "qwen25_14b_fft_custom"]),
])

FAMILIES_5FAM = OrderedDict(list(FAMILIES_2FAM.items()) + [
    ("gemma2", ["gemma2_base_fp16_custom", "gemma2_lora_custom", "gemma2_fft_custom"]),
    ("qwen3", ["qwen3_base_fp16_custom", "qwen3_lora_custom", "qwen3_fft_custom"]),
    ("phi4", ["phi4_base_fp16_custom", "phi4_lora_custom", "phi4_fft_custom"]),
])

# Default to the 5-family run under its own filename suffix so the published
# 2-family artifacts are never clobbered. `--scope 2fam` reproduces the original.
SCOPES = {"5fam": (FAMILIES_5FAM, "_5fam"), "2fam": (FAMILIES_2FAM, "")}
FAMILIES = FAMILIES_5FAM
SUFFIX = "_5fam"


def load_cell(cell):
    """Return {behavior_id: (final_loss, success)} for one GCG cell."""
    logs = json.load(open(os.path.join(HB, "results_jbb/GCG", cell, "test_cases/logs.json")))
    res = json.load(open(os.path.join(HB, "results_jbb/GCG", cell, "results", cell + ".json")))
    out = {}
    for bid, entries in logs.items():
        if bid not in res:
            continue
        # one test case per behavior in this sweep; take the first of each
        loss = entries[0]["final_loss"]
        label = res[bid][0]["label"]
        out[bid] = (float(loss), int(label))
    return out


def mantel_haenszel(tables):
    """Common OR + chi-square across 2x2 strata. tables: list of (a,b,c,d)."""
    num = den = 0.0
    e_sum = v_sum = a_sum = 0.0
    for a, b, c, d in tables:
        n = a + b + c + d
        if n == 0:
            continue
        num += a * d / n
        den += b * c / n
        a_sum += a
        e_sum += (a + b) * (a + c) / n
        if n > 1:
            v_sum += ((a + b) * (c + d) * (a + c) * (b + d)) / (n * n * (n - 1))
    or_mh = num / den if den > 0 else float("nan")
    if v_sum > 0:
        chi2 = (abs(a_sum - e_sum) - 0.5) ** 2 / v_sum
        p = float(stats.chi2.sf(chi2, 1))
    else:
        chi2, p = float("nan"), float("nan")
    return or_mh, chi2, p


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {}
    summary_rows, paired_rows, strata_rows = [], [], []

    print("=" * 78)
    print("GCG OPTIMIZATION-DIFFICULTY ANALYSIS  (JBB-100, seed 42, HB-Cls labels)")
    print("=" * 78)

    # ---- per-cell descriptives + loss/success correlation -------------------
    print("\n[1] Per-cell final GCG loss and attack success\n")
    hdr = f"{'cell':<32}{'n':>4}{'mean':>9}{'median':>9}{'sd':>8}{'ASR':>8}{'r_pb':>8}{'p':>9}"
    print(hdr)
    print("-" * len(hdr))
    for fam, cells in FAMILIES.items():
        for cell in cells:
            d = load_cell(cell)
            data[cell] = d
            loss = np.array([v[0] for v in d.values()])
            succ = np.array([v[1] for v in d.values()])
            asr = 100.0 * succ.mean()
            if 0 < succ.sum() < len(succ):
                r, p = stats.pointbiserialr(succ, loss)
            else:
                r, p = float("nan"), float("nan")
            print(f"{cell:<32}{len(d):>4}{loss.mean():>9.3f}{np.median(loss):>9.3f}"
                  f"{loss.std(ddof=1):>8.3f}{asr:>7.1f}%{r:>8.3f}{p:>9.4f}")
            summary_rows.append(dict(family=fam, cell=cell, n=len(d),
                                     mean_final_loss=loss.mean(),
                                     median_final_loss=float(np.median(loss)),
                                     sd_final_loss=loss.std(ddof=1),
                                     asr_pct=asr, r_pointbiserial=r, r_p=p))

    # ---- TEST 1: is FFT a softer target? ------------------------------------
    print("\n[2] TEST 1 -- paired final-loss comparison on identical behaviors")
    print("    H_confound: FFT loss < LoRA loss (FFT is the easier target)\n")
    for fam, cells in FAMILIES.items():
        base, lora, fft = cells
        for a_name, b_name in [(lora, fft), (base, fft), (base, lora)]:
            shared = sorted(set(data[a_name]) & set(data[b_name]))
            a = np.array([data[a_name][k][0] for k in shared])
            b = np.array([data[b_name][k][0] for k in shared])
            diff = a - b  # >0 means b (second cell) optimized to lower loss
            w = stats.wilcoxon(a, b)
            t = stats.ttest_rel(a, b)
            # Hedges' g for paired data
            dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else float("nan")
            lab = f"{a_name.split('_custom')[0]} vs {b_name.split('_custom')[0]}"
            print(f"  {fam:<11}{lab:<44} n={len(shared):>3}  "
                  f"mean_diff={diff.mean():+.4f}  dz={dz:+.3f}  "
                  f"Wilcoxon p={w.pvalue:.4g}  t p={t.pvalue:.4g}")
            paired_rows.append(dict(family=fam, contrast=lab, n=len(shared),
                                    mean_loss_a=a.mean(), mean_loss_b=b.mean(),
                                    mean_diff=diff.mean(), dz=dz,
                                    wilcoxon_p=w.pvalue, ttest_p=t.pvalue))

    # ---- TEST 2: does the ASR gap survive conditioning on loss? -------------
    print("\n[3] TEST 2 -- loss-stratified LoRA vs FFT success (quartile bins)")
    print("    If the gap survives conditioning on loss, it is not explained by")
    print("    optimization difficulty.\n")
    for fam, cells in FAMILIES.items():
        _, lora, fft = cells
        shared = sorted(set(data[lora]) & set(data[fft]))
        pooled = np.array([data[lora][k][0] for k in shared] +
                          [data[fft][k][0] for k in shared])
        edges = np.quantile(pooled, [0.25, 0.5, 0.75])
        tables = []
        print(f"  {fam}")
        print(f"    {'loss quartile':<22}{'LoRA succ/n':>14}{'FFT succ/n':>14}{'gap pp':>10}")
        for q in range(4):
            lo = -np.inf if q == 0 else edges[q - 1]
            hi = np.inf if q == 3 else edges[q]
            l_s = l_n = f_s = f_n = 0
            for k in shared:
                for cell in (lora, fft):
                    loss, succ = data[cell][k]
                    if lo <= loss < hi:
                        if cell == lora:
                            l_n += 1
                            l_s += succ
                        else:
                            f_n += 1
                            f_s += succ
            gap = (100.0 * f_s / f_n if f_n else float("nan")) - \
                  (100.0 * l_s / l_n if l_n else float("nan"))
            rng = f"[{lo:.3f}, {hi:.3f})".replace("-inf", "-inf").replace("inf", "inf")
            print(f"    {rng:<22}{f'{l_s}/{l_n}':>14}{f'{f_s}/{f_n}':>14}{gap:>+10.1f}")
            # 2x2: rows = FFT/LoRA, cols = success/failure
            tables.append((f_s, f_n - f_s, l_s, l_n - l_s))
            strata_rows.append(dict(family=fam, quartile=q + 1, lo=lo, hi=hi,
                                    lora_succ=l_s, lora_n=l_n, fft_succ=f_s,
                                    fft_n=f_n, gap_pp=gap))
        or_mh, chi2, p = mantel_haenszel(tables)
        # Crude (unadjusted) OR on the same behaviors, for the shrinkage comparison.
        ls = sum(data[lora][k][1] for k in shared)
        fs = sum(data[fft][k][1] for k in shared)
        n = len(shared)
        or_crude = ((fs / (n - fs)) / (ls / (n - ls))) if 0 < ls < n and 0 < fs < n else float("nan")
        shrink = 100.0 * (1 - (or_mh - 1) / (or_crude - 1)) if or_crude > 1 else float("nan")
        # Paired McNemar (exact) -- retains the pairing the stratified test discards.
        b = sum(1 for k in shared if data[fft][k][1] and not data[lora][k][1])
        c = sum(1 for k in shared if data[lora][k][1] and not data[fft][k][1])
        mc_p = float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c else float("nan")
        print(f"    crude OR (FFT vs LoRA)          = {or_crude:.3f}"
              f"   [McNemar b={b} c={c} p={mc_p:.4g}]")
        print(f"    loss-adjusted MH common OR      = {or_mh:.3f}"
              f"   chi2={chi2:.3f}  p={p:.4g}")
        print(f"    -> loss explains {shrink:.0f}% of the excess odds\n")
        strata_rows.append(dict(family=fam, quartile="MH", lo=None, hi=None,
                                lora_succ=ls, lora_n=n, fft_succ=fs,
                                fft_n=n, gap_pp=100.0 * (fs - ls) / n,
                                or_crude=or_crude, or_mh=or_mh, mh_chi2=chi2,
                                mh_p=p, pct_excess_odds_explained=shrink,
                                mcnemar_b=b, mcnemar_c=c, mcnemar_p=mc_p))

    # ---- write ---------------------------------------------------------------
    import csv
    for name, rows in [("gcg_loss_summary", summary_rows),
                       ("gcg_loss_paired", paired_rows),
                       ("gcg_loss_strata", strata_rows)]:
        path = os.path.join(OUT, name + SUFFIX + ".csv")
        keys = sorted({k for r in rows for k in r})
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {os.path.relpath(path, REPO)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=sorted(SCOPES), default="5fam")
    a = ap.parse_args()
    FAMILIES, SUFFIX = SCOPES[a.scope]
    main()
