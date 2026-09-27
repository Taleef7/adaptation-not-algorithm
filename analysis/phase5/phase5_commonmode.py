#!/usr/bin/env python3
"""
Tests the "common-mode cancellation" reading of the seed-42 re-training.
Per family, are the 3 methods' old->new HB-Cls deltas correlated (common-mode -> cancels in
contrasts) or independent/anti-correlated (-> contrasts do NOT cancel)? Reads the old-vs-new
delta table (artifacts/revision_stats/phase5_old_vs_new_hbcls.csv, 3 families).

Result: common-mode is NOT universal. It holds for gemma2 (all r>0.85) and for
the LoRA-FFT pair (partial), explaining LoRA~=FFT seed-stability. QLoRA deltas ANTI-correlate with
LoRA/FFT on llama31 (r=-0.85) and qwen3 (r=-0.95); QLoRA contrasts amplify, not cancel. QLoRA contrast
stability survives only in the 12-condition MEAN (symmetric swings wash out) while variance persists ->
QLoRA fails TOST in BOTH seed regimes. So: LoRA-FFT = clean cancellation; QLoRA = idiosyncratic.
"""
from pathlib import Path
import pandas as pd, numpy as np

OUT = Path(__file__).resolve().parents[2] / "artifacts" / "revision_stats"
j = pd.read_csv(OUT / "phase5_old_vs_new_hbcls.csv")
FAMS = ["llama31", "gemma2", "qwen3"]
w = j.pivot_table(index=["model_family", "attack", "dataset"], columns="config", values="delta").reset_index()

w["mean3"] = w[["lora", "qlora", "fft"]].mean(axis=1)
w["sd3"] = w[["lora", "qlora", "fft"]].std(axis=1, ddof=1)

rows = []
for fam in FAMS:
    s = w[w.model_family == fam]
    c = s[["lora", "qlora", "fft"]].corr()
    rows.append(dict(family=fam, r_lora_fft=c.loc["lora", "fft"],
                     r_lora_qlora=c.loc["lora", "qlora"], r_qlora_fft=c.loc["qlora", "fft"]))
corr = pd.DataFrame(rows)

contrast = []
for A, B in [("lora", "fft"), ("qlora", "fft"), ("lora", "qlora")]:
    contrast.append(dict(pair=f"{A}-{B}", mean_abs_contrast_delta=(w[A] - w[B]).abs().mean()))
contrast = pd.DataFrame(contrast)

w.round(2).to_csv(OUT / "phase5_commonmode_deltas.csv", index=False)
corr.round(3).to_csv(OUT / "phase5_commonmode_corr.csv", index=False)

if __name__ == "__main__":
    print("=== Per-condition method deltas ===")
    print(w[["model_family", "attack", "dataset", "lora", "qlora", "fft", "sd3"]].round(1).to_string(index=False))
    print(f"\nmean |per-cell delta| = {j['delta'].abs().mean():.2f} pp")
    print(f"mean within-condition SD(3 methods) = {w['sd3'].mean():.2f} pp (NOT small -> not universal common-mode)")
    print("\n=== Per-family correlations (common-mode => high +) ===")
    print(corr.round(2).to_string(index=False))
    print("\n=== Contrast |delta| (LoRA-FFT cancels; QLoRA contrasts amplify) ===")
    print(contrast.round(2).to_string(index=False))
    print(f"\nWrote {OUT}/phase5_commonmode_deltas.csv, phase5_commonmode_corr.csv")
