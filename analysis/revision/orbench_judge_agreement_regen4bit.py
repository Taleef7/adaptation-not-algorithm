#!/usr/bin/env python3
"""Judge-agreement analysis with the NF4 base_4bit cells (regen4bit namespace).

The base_4bit OR-Bench cells used in the paper are the NF4-quantized generations
in the regen4bit namespace, judged by both judges:

  * GPT-4o-mini  -> results/ORBench/Hard-1K/DirectRequest/regen4bit/results/
  * Ministral    -> results/ORBench/Hard-1K/DirectRequest/ministral3_14b_regen4bit/results/

This script repeats analysis/revision/orbench_judge_agreement.py exactly, except
the five base_4bit cells read from those regen namespaces (both judges), and the
published-GPT rejection rate used in the correlation arm is likewise replaced by
the regen GPT metric for those five rows. The other 20 cells are byte-identical
inputs to the original run.

Writes artifacts/revision/orbench_judge_{agreement,rates,correlations}_regen4bit.csv
(the non-suffixed files are the default-namespace variant).
"""
from __future__ import annotations

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

JUDGE_LABEL = "Ministral-3-14B"
REFUSAL = {"direct_refusal", "indirect_refusal"}
BLACK_BOX = ["PAIR", "DeepInception", "ArtPrompt"]

REGEN_CELLS = {
    "gemma2_base_4bit_custom",
    "llama31_base_4bit_custom",
    "phi4_base_4bit_custom",
    "qwen25_14b_base_4bit_custom",
    "qwen3_base_4bit_custom",
}


def dirs_for(cell: str) -> tuple[Path, Path]:
    """(GPT-4o-mini dir, Ministral dir) for a cell, regen-aware."""
    if cell in REGEN_CELLS:
        return BASE / "regen4bit", BASE / "ministral3_14b_regen4bit"
    return BASE / "default", BASE / "ministral3_14b"


def load_labels(judge_dir: Path, cell: str) -> dict[str, str]:
    path = judge_dir / "results" / f"{cell}.json"
    if not path.exists():
        raise SystemExit(f"missing judge results: {path}")
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
    cells = sorted(p.stem for p in (BASE / "ministral3_14b" / "results").glob("*.json"))
    missing_regen = REGEN_CELLS - set(cells)
    if missing_regen:
        raise SystemExit(f"regen cells absent from full grid: {missing_regen}")

    name_map_df = pd.read_csv(ORBENCH_CSV)[["model_name", "model_family", "config"]]
    name_map = {r.model_name: (r.model_family, r.config) for r in name_map_df.itertuples()}
    missing = [c for c in cells if c not in name_map]
    if missing:
        raise SystemExit(f"cells absent from {ORBENCH_CSV.name}: {missing}")

    agree_rows, rate_rows = [], []
    for cell in cells:
        ref_dir, new_dir = dirs_for(cell)
        ref, new = load_labels(ref_dir, cell), load_labels(new_dir, cell)
        shared = sorted(set(ref) & set(new))
        if not shared:
            raise SystemExit(f"no shared behaviors for {cell}")
        a = [ref[b] for b in shared]
        c = [new[b] for b in shared]

        three_way = sum(x == y for x, y in zip(a, c)) / len(shared)
        bin_a = ["refuse" if x in REFUSAL else "comply" for x in a]
        bin_c = ["refuse" if x in REFUSAL else "comply" for x in c]
        binary = sum(x == y for x, y in zip(bin_a, bin_c)) / len(shared)

        family, config = name_map[cell]
        agree_rows.append(dict(
            cell=cell, n=len(shared),
            genuine_4bit=cell in REGEN_CELLS,
            agree_3way_pct=round(100 * three_way, 2),
            agree_binary_pct=round(100 * binary, 2),
            kappa_3way=round(cohen_kappa(a, c), 4),
            kappa_binary=round(cohen_kappa(bin_a, bin_c), 4),
            unknown_new=sum(1 for x in c if x == "unknown"),
            unknown_ref=sum(1 for x in a if x == "unknown"),
        ))
        rate_rows.append(dict(
            model_family=family, config=config, cell=cell, n=len(shared),
            genuine_4bit=cell in REGEN_CELLS,
            rejection_gpt4omini_pct=round(100 * sum(x in REFUSAL for x in a) / len(shared), 2),
            rejection_new_pct=round(100 * sum(x in REFUSAL for x in c) / len(shared), 2),
        ))

    agree = pd.DataFrame(agree_rows)
    rates = pd.DataFrame(rate_rows)
    rates["delta_pp"] = (rates["rejection_new_pct"] - rates["rejection_gpt4omini_pct"]).round(2)

    # ---- headline correlation with genuine-4bit ORR points -------------------
    asr = pd.read_csv(ASR_CSV)
    bb = asr[asr["attack"].isin(BLACK_BOX)]
    means = bb.groupby(["evaluator", "model_family", "config"], as_index=False)["ASR"].mean()

    # GPT arm: published table, but the five base_4bit rows take the regen GPT
    # pipeline metric (matches how the published rates were computed: unknowns
    # excluded from the denominator).
    published = pd.read_csv(ORBENCH_CSV)[
        ["model_name", "model_family", "config", "hard_1k_rejection_rate_pct"]
    ].copy()
    for cell in sorted(REGEN_CELLS):
        payload = json.loads((BASE / "regen4bit" / "results" / f"{cell}.json").read_text())
        published.loc[published["model_name"].eq(cell), "hard_1k_rejection_rate_pct"] = \
            round(100 * payload["rejection_rate"], 2)
    published = published.drop(columns=["model_name"])

    corr_rows = []
    for asr_eval, asr_label in [("Llama-2-13b-cls", "HB-Cls"), ("GPT-4o-mini", "GPT-4o-mini")]:
        sub = means[means["evaluator"].eq(asr_eval)]
        for orr_label, orr_df, col in [
            ("GPT-4o-mini (regen 4bit)", published, "hard_1k_rejection_rate_pct"),
            (JUDGE_LABEL + " (regen 4bit)", rates, "rejection_new_pct"),
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
    agree.to_csv(OUT / "orbench_judge_agreement_regen4bit.csv", index=False)
    rates.to_csv(OUT / "orbench_judge_rates_regen4bit.csv", index=False)
    corr.to_csv(OUT / "orbench_judge_correlations_regen4bit.csv", index=False)

    pd.set_option("display.width", 150)
    print("\n=== per-item agreement, GPT-4o-mini vs", JUDGE_LABEL, "(regen 4bit) ===")
    print(agree.to_string(index=False))
    print(f"\npooled 3-way agreement : {agree['agree_3way_pct'].mean():.2f}%")
    print(f"pooled binary agreement: {agree['agree_binary_pct'].mean():.2f}%")
    print("\n=== per-config rejection rates ===")
    print(rates[["cell", "genuine_4bit", "rejection_gpt4omini_pct",
                 "rejection_new_pct", "delta_pp"]].to_string(index=False))
    print(f"\nmean shift: {rates['delta_pp'].mean():+.2f} pp   "
          f"range [{rates['delta_pp'].min():+.2f}, {rates['delta_pp'].max():+.2f}]")
    print("\n=== headline correlation under each ORR judge (genuine 4bit) ===")
    print(corr.to_string(index=False))

    old_corr_path = OUT / "orbench_judge_correlations.csv"
    if old_corr_path.exists():
        old = pd.read_csv(old_corr_path)
        print("\n=== rho: fake-4bit (old) vs genuine-4bit (new) ===")
        for (o, n_) in zip(old.itertuples(), corr.itertuples()):
            print(f"  {o.asr_evaluator:12s} x {o.orr_judge:28s} "
                  f"old={o.spearman_rho:+.4f}  new={n_.spearman_rho:+.4f}  "
                  f"d={n_.spearman_rho - o.spearman_rho:+.4f}")
    print(f"\nwrote 3 CSVs (suffix _regen4bit) to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
