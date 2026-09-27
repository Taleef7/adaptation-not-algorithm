#!/usr/bin/env python3
"""Is the OR-Bench utility axis judge-dependent?

Context
-------
The paper's main utility result (Safety-Utility section): higher black-box ASR goes
with lower Hard-1K rejection -- at rho=-0.580 (HB-Cls ASR) and rho=-0.692
(GPT-4o-mini ASR). Both use a Hard-1K rejection rate scored by ONE judge:
GPT-4o-mini. So the rho=-0.692 pairing shares a judge across both axes, and the
utility axis has no second opinion at all.

This script takes the same completions re-judged by a second, independent local
model (Ministral-3-14B, prompted with the same OR-Bench three-way check) and asks:

  1. Do the two judges agree per item? (3-way and binary refuse/comply, + kappa)
  2. Do the per-config rejection RATES move? (the quantity the paper actually plots)
  3. Does the headline correlation survive swapping the ORR judge? This is the
     number that matters: if rho holds, the utility finding is judge-robust.

Writes artifacts/revision/orbench_judge_{agreement,rates,correlations}.csv.
Reads the published GPT-4o-mini results; never writes to that namespace.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
HB = ROOT / "toolkits" / "HarmBench"
BASE = HB / "results/ORBench/Hard-1K/DirectRequest"
OUT = ROOT / "artifacts" / "revision"
ASR_CSV = ROOT / "artifacts" / "master_asr_long.csv"
ORBENCH_CSV = ROOT / "artifacts" / "or_bench_vs_legacy_orr.csv"

REFUSAL = {"direct_refusal", "indirect_refusal"}
BLACK_BOX = ["PAIR", "DeepInception", "ArtPrompt"]


def load_labels(judge_dir: Path, cell: str) -> dict[str, str]:
    """{behavior_id: category} for one cell, first entry per behavior."""
    path = judge_dir / "results" / f"{cell}.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    out = {}
    for bid, entries in payload.get("details", {}).items():
        if entries:
            out[bid] = entries[0].get("answer_category", "unknown")
    return out


def cohen_kappa(a: list[str], b: list[str]) -> float:
    cats = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return float("nan")
    obs = sum(x == y for x, y in zip(a, b)) / n
    exp = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (obs - exp) / (1 - exp) if exp < 1 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge_tag", default="ministral3_14b")
    ap.add_argument("--judge_label", default="Ministral-3-14B")
    args = ap.parse_args()

    ref_dir = BASE / "default"          # GPT-4o-mini, published
    new_dir = BASE / args.judge_tag     # local second judge
    cells = sorted(p.stem for p in (new_dir / "results").glob("*.json"))
    if not cells:
        raise SystemExit(f"no results under {new_dir}/results -- run local_judge.py first")

    # (model_family, config) must come from the published OR-Bench table, not from
    # splitting the cell name: "qwen25_14b_base_4bit_custom" is family qwen25 /
    # config base_4bit, and any string split gets that wrong and silently drops
    # all five qwen25 cells from the merge.
    name_map_df = pd.read_csv(ORBENCH_CSV)[["model_name", "model_family", "config"]]
    name_map = {r.model_name: (r.model_family, r.config) for r in name_map_df.itertuples()}
    missing = [c for c in cells if c not in name_map]
    if missing:
        raise SystemExit(f"cells absent from {ORBENCH_CSV.name}: {missing}")

    agree_rows, rate_rows = [], []
    for cell in cells:
        ref, new = load_labels(ref_dir, cell), load_labels(new_dir, cell)
        shared = sorted(set(ref) & set(new))
        if not shared:
            continue
        a = [ref[b] for b in shared]
        c = [new[b] for b in shared]

        three_way = sum(x == y for x, y in zip(a, c)) / len(shared)
        bin_a = ["refuse" if x in REFUSAL else "comply" for x in a]
        bin_c = ["refuse" if x in REFUSAL else "comply" for x in c]
        binary = sum(x == y for x, y in zip(bin_a, bin_c)) / len(shared)

        family, config = name_map[cell]
        agree_rows.append(dict(
            cell=cell, n=len(shared),
            agree_3way_pct=round(100 * three_way, 2),
            agree_binary_pct=round(100 * binary, 2),
            kappa_3way=round(cohen_kappa(a, c), 4),
            kappa_binary=round(cohen_kappa(bin_a, bin_c), 4),
            unknown_new=sum(1 for x in c if x == "unknown"),
            unknown_ref=sum(1 for x in a if x == "unknown"),
        ))
        rate_rows.append(dict(
            model_family=family, config=config, cell=cell, n=len(shared),
            rejection_gpt4omini_pct=round(100 * sum(x in REFUSAL for x in a) / len(shared), 2),
            rejection_new_pct=round(100 * sum(x in REFUSAL for x in c) / len(shared), 2),
        ))

    agree = pd.DataFrame(agree_rows)
    rates = pd.DataFrame(rate_rows)
    rates["delta_pp"] = (rates["rejection_new_pct"] - rates["rejection_gpt4omini_pct"]).round(2)

    # ---- does the headline correlation survive the judge swap? ---------------
    asr = pd.read_csv(ASR_CSV)
    bb = asr[asr["attack"].isin(BLACK_BOX)]
    means = bb.groupby(["evaluator", "model_family", "config"], as_index=False)["ASR"].mean()
    published = pd.read_csv(ORBENCH_CSV)[["model_family", "config", "hard_1k_rejection_rate_pct"]]

    corr_rows = []
    for asr_eval, asr_label in [("Llama-2-13b-cls", "HB-Cls"), ("GPT-4o-mini", "GPT-4o-mini")]:
        sub = means[means["evaluator"].eq(asr_eval)]
        for orr_label, orr_df, col in [
            ("GPT-4o-mini (published)", published, "hard_1k_rejection_rate_pct"),
            (args.judge_label, rates, "rejection_new_pct"),
        ]:
            m = sub.merge(orr_df, on=["model_family", "config"], how="inner")
            if len(m) < 3:
                continue
            rho, p = stats.spearmanr(m["ASR"], m[col])
            corr_rows.append(dict(
                asr_evaluator=asr_label, orr_judge=orr_label, n=len(m),
                spearman_rho=round(float(rho), 4), spearman_p=round(float(p), 5),
                shares_judge_across_axes=(asr_label == "GPT-4o-mini"
                                          and orr_label.startswith("GPT-4o-mini")),
            ))
    corr = pd.DataFrame(corr_rows)

    OUT.mkdir(parents=True, exist_ok=True)
    agree.to_csv(OUT / "orbench_judge_agreement.csv", index=False)
    rates.to_csv(OUT / "orbench_judge_rates.csv", index=False)
    corr.to_csv(OUT / "orbench_judge_correlations.csv", index=False)

    pd.set_option("display.width", 140)
    print("\n=== per-item agreement, GPT-4o-mini vs", args.judge_label, "===")
    print(agree.to_string(index=False))
    print(f"\npooled 3-way agreement : {agree['agree_3way_pct'].mean():.2f}%")
    print(f"pooled binary agreement: {agree['agree_binary_pct'].mean():.2f}%")
    print("\n=== per-config rejection rates ===")
    print(rates[["cell", "rejection_gpt4omini_pct", "rejection_new_pct", "delta_pp"]].to_string(index=False))
    print(f"\nmean shift: {rates['delta_pp'].mean():+.2f} pp   "
          f"range [{rates['delta_pp'].min():+.2f}, {rates['delta_pp'].max():+.2f}]")
    print("\n=== headline correlation under each ORR judge ===")
    print(corr.to_string(index=False))
    print(f"\nwrote 3 CSVs to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
