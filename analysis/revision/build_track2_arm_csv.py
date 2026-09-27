#!/usr/bin/env python3
"""
Build a Track-2 arm CSV (per-cell LoRA vs FFT ASR) from HB-Cls scored labels.

Scored JSONL in, arm CSV out, with a hard row-count guard so a partial slice can
never silently become an ASR.

Scored-label layout (written by evaluation/seed42_grid/recipe_robustness/score_arm.sh):
  results_track2/scored/{family}_{method}_{arm}_{bench}_hbcls.jsonl           DeepInception
  results_track2/scored/{family}_{method}_{arm}_{bench}_artprompt_hbcls.jsonl ArtPrompt
Each line: {"behavior_goal": ..., "jailbroken": 0|1, "response_len": int}
ASR = mean(jailbroken) * 100, denominator = all rows (matching the phase-5 convention
where an ERROR response scores 0 rather than being dropped).

Verify against a known-good arm before trusting a new one:
    python analysis/revision/build_track2_arm_csv.py --arm dolly3ep \
        --families llama31 gemma2 --check artifacts/track2_dolly3ep_2fam.csv

Then build the extended arm:
    python analysis/revision/build_track2_arm_csv.py --arm dolly3ep \
        --families llama31 gemma2 qwen3 --out artifacts/track2_dolly3ep_3fam.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SCORED = REPO / "toolkits" / "HarmBench" / "results_track2" / "scored"

ATTACKS = [("DeepInception", "hbcls"), ("ArtPrompt", "artprompt_hbcls")]
BENCHES = [("jbb", 100), ("harmbench", 400)]


def asr(family: str, method: str, arm: str, suffix: str, bench: str, expected: int) -> float:
    path = SCORED / f"{family}_{method}_{arm}_{bench}_{suffix}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing scored slice: {path}")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if len(rows) != expected:
        raise ValueError(f"{path.name}: n={len(rows)}, expected {expected} (refusing partial slice)")
    return round(sum(r["jailbroken"] for r in rows) / len(rows) * 100, 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="e.g. dolly3ep")
    ap.add_argument("--families", nargs="+", required=True)
    ap.add_argument("--out")
    ap.add_argument("--check", help="existing CSV this build must reproduce exactly")
    a = ap.parse_args()

    rows = []
    for family in a.families:
        for attack, suffix in ATTACKS:
            for bench, expected in BENCHES:
                lora = asr(family, "lora", a.arm, suffix, bench, expected)
                fft = asr(family, "fft", a.arm, suffix, bench, expected)
                rows.append({
                    "family": family, "attack": attack, "benchmark": bench,
                    "lora_asr": lora, "fft_asr": fft,
                    "delta_lora_minus_fft": round(lora - fft, 4),
                })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"\nn = {len(df)} paired cells "
          f"({len(a.families)} families x {len(ATTACKS)} attacks x {len(BENCHES)} benchmarks)")
    print(f"mean delta (LoRA - FFT) = {df.delta_lora_minus_fft.mean():+.3f} pp, "
          f"sd = {df.delta_lora_minus_fft.std(ddof=1):.3f}")

    if a.check:
        ref = pd.read_csv(REPO / a.check if not Path(a.check).is_absolute() else a.check)
        key = ["family", "attack", "benchmark"]
        merged = ref.merge(df, on=key, suffixes=("_ref", "_new"))
        if len(merged) != len(ref):
            sys.exit(f"CHECK FAILED: matched {len(merged)} of {len(ref)} reference rows")
        bad = merged[(merged.lora_asr_ref - merged.lora_asr_new).abs().gt(1e-6)
                     | (merged.fft_asr_ref - merged.fft_asr_new).abs().gt(1e-6)]
        if not bad.empty:
            print(bad.to_string(index=False))
            sys.exit("CHECK FAILED: rebuilt ASRs differ from the reference CSV")
        print(f"\nCHECK PASSED: reproduces all {len(ref)} rows of {a.check} exactly")

    if a.out:
        out = REPO / a.out if not Path(a.out).is_absolute() else Path(a.out)
        if out.exists():
            sys.exit(f"REFUSING to overwrite existing {out} (original results are never overwritten)")
        df.to_csv(out, index=False)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
