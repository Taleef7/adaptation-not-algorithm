#!/usr/bin/env python3
"""
DeepInception sampling variability from the multi-rollout ASRs.
Reads the CSV written by score_rollouts.py (family,method,benchmark,seed,n,jb,asr) and reports, per
(family, benchmark) cell: LoRA and FFT mean +- sd across the rollout seeds, the LoRA-FFT gap, the
standard error of the difference, and a Welch t-test of the gap against rollout noise.

Usage: python analyze_rollout_variance.py [--csv artifacts/rollout_di_asr.csv] [--out artifacts/rollout_di_variance.csv]
"""
import argparse, csv, math, os
from collections import defaultdict


def mean_sd(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in xs) / (n - 1)  # sample sd
    return m, math.sqrt(var)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="artifacts/rollout_di_asr.csv")
    ap.add_argument("--out", default="artifacts/rollout_di_variance.csv")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    # bucket asr lists by (family, benchmark, method)
    buck = defaultdict(list)
    for r in rows:
        buck[(r["family"], r["benchmark"], r["method"])].append(float(r["asr"]))

    try:
        from scipy import stats as _st
    except Exception:
        _st = None

    cells = sorted({(f, b) for (f, b, _m) in buck})
    out_rows = []
    hdr = f"{'family':10} {'bench':9} {'LoRA mean±sd':>16} {'FFT mean±sd':>16} {'gap':>7} {'SE_diff':>8} {'Welch t':>8} {'p':>7} {'real?':>6}"
    print(hdr)
    for f, b in cells:
        lo = buck.get((f, b, "lora"), [])
        ff = buck.get((f, b, "fft"), [])
        lm, ls = mean_sd(lo)
        fm, fs = mean_sd(ff)
        gap = lm - fm
        # within-cell sampling noise -> standard error of the mean difference across the 5v5 rollouts
        se = math.sqrt((ls ** 2) / max(len(lo), 1) + (fs ** 2) / max(len(ff), 1)) if len(lo) and len(ff) else float("nan")
        if _st is not None and len(lo) > 1 and len(ff) > 1:
            t, p = _st.ttest_ind(lo, ff, equal_var=False)  # Welch: is the per-cell gap real beyond rollout noise?
        else:
            t = gap / se if se and not math.isnan(se) else float("nan")
            p = float("nan")
        real = bool(p < 0.05) if not math.isnan(p) else None  # gap exceeds sampling noise (py bool, not np.bool_)
        print(f"{f:10} {b:9} {lm:7.2f}±{ls:5.2f}   {fm:7.2f}±{fs:5.2f}   {gap:+6.2f} {se:8.2f} {t:8.2f} {p:7.3f} {str(real):>6}")
        out_rows.append({"family": f, "benchmark": b,
                         "lora_mean": round(lm, 2), "lora_sd": round(ls, 2),
                         "fft_mean": round(fm, 2), "fft_sd": round(fs, 2),
                         "gap_lora_minus_fft": round(gap, 2),
                         "se_diff": round(se, 2),
                         "welch_t": round(float(t), 2) if not math.isnan(float(t)) else "",
                         "welch_p": round(float(p), 4) if not math.isnan(float(p)) else "",
                         "gap_real_beyond_noise": real,
                         "gap_direction": ("LoRA>FFT" if gap > 0 else "FFT>LoRA")})

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=["family", "benchmark", "lora_mean", "lora_sd",
                                           "fft_mean", "fft_sd", "gap_lora_minus_fft", "se_diff",
                                           "welch_t", "welch_p", "gap_real_beyond_noise", "gap_direction"])
        w.writeheader(); w.writerows(out_rows)

    # summary: within-cell DI sampling noise (per-rollout sd) and per-cell gaps
    all_sd = [r["lora_sd"] for r in out_rows] + [r["fft_sd"] for r in out_rows]
    all_sd = [s for s in all_sd if not math.isnan(s)]
    n_real = sum(1 for r in out_rows if r["gap_real_beyond_noise"] is True)
    dirs = [r["gap_direction"] for r in out_rows if r["gap_real_beyond_noise"] is True]
    mean_gap = sum(r["gap_lora_minus_fft"] for r in out_rows) / len(out_rows)
    print(f"\nwithin-cell DI sampling sd: mean {sum(all_sd)/len(all_sd):.2f} pp, max {max(all_sd):.2f} pp")
    print(f"per-cell LoRA-FFT gaps REAL beyond rollout noise (Welch p<0.05): {n_real}/{len(out_rows)} cells")
    print(f"  directions of the gaps with p<0.05: {dirs}")
    print(f"mean of per-cell LoRA-FFT DI gaps: {mean_gap:+.2f} pp")
    print(f"==> wrote {a.out}")


if __name__ == "__main__":
    main()
