#!/usr/bin/env python3
"""Build the OR-Bench per-config table used in the paper.

The five ``base_4bit`` OR-Bench cells are read from the ``regen4bit`` namespace,
which holds the NF4-quantized generations (vLLM path); every other row is taken
unchanged from artifacts/or_bench_vs_legacy_orr.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "artifacts" / "or_bench_vs_legacy_orr.csv"
OUT = ROOT / "artifacts" / "revision" / "or_bench_vs_legacy_orr_genuine4bit.csv"
REGEN = ROOT / "toolkits" / "HarmBench" / "results" / "ORBench"

# model_name stem per family, as used in the results filenames
CELLS = {
    "gemma2": "gemma2_base_4bit_custom",
    "llama31": "llama31_base_4bit_custom",
    "phi4": "phi4_base_4bit_custom",
    "qwen25": "qwen25_14b_base_4bit_custom",
    "qwen3": "qwen3_base_4bit_custom",
}

SUBSETS = {"Hard-1K": "hard_1k", "Toxic": "toxic"}
FIELDS = [
    "n",
    "direct_answer_count",
    "direct_refusal_count",
    "indirect_refusal_count",
    "rejection_count",
    "direct_answer_rate",
    "direct_refusal_rate",
    "indirect_refusal_rate",
    "rejection_rate",
]


def load(subset: str, stem: str) -> dict:
    path = REGEN / subset / "DirectRequest" / "regen4bit" / "results" / f"{stem}.json"
    with path.open() as fh:
        return json.load(fh)


def main() -> None:
    df = pd.read_csv(SRC)
    for family, stem in CELLS.items():
        mask = df["model_family"].eq(family) & df["config"].eq("base_4bit")
        if mask.sum() != 1:
            raise SystemExit(f"expected exactly one base_4bit row for {family}")
        for subset, prefix in SUBSETS.items():
            res = load(subset, stem)
            for field in FIELDS:
                df.loc[mask, f"{prefix}_{field}"] = res[field]
            df.loc[mask, f"{prefix}_rejection_rate_pct"] = round(res["rejection_rate"] * 100, 2)
            for part in ("direct_answer", "direct_refusal", "indirect_refusal"):
                df.loc[mask, f"{prefix}_{part}_rate_pct"] = round(res[f"{part}_rate"] * 100, 2)
            df.loc[mask, f"{prefix}_acceptance_count"] = res["direct_answer_count"]
            df.loc[mask, f"{prefix}_acceptance_rate"] = res["direct_answer_rate"]
            df.loc[mask, f"{prefix}_acceptance_rate_pct"] = round(res["direct_answer_rate"] * 100, 2)
            df.loc[mask, f"{prefix}_path"] = str(
                REGEN / subset / "DirectRequest" / "regen4bit" / "results" / f"{stem}.json"
            )
        gap = float(df.loc[mask, "toxic_rejection_rate_pct"].iloc[0]) - float(
            df.loc[mask, "hard_1k_rejection_rate_pct"].iloc[0]
        )
        df.loc[mask, "safety_utility_gap_pct"] = round(gap, 2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"wrote {OUT.relative_to(ROOT)}  ({len(df)} rows)")
    cols = ["model_family", "config", "hard_1k_rejection_rate_pct",
            "toxic_rejection_rate_pct", "safety_utility_gap_pct"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
