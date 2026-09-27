#!/usr/bin/env python3
"""Behavior-difficulty stratification of Delta ASR (FFT - Base FP16, GPT-4o-mini).

Regenerates the appendix stratification table from the canonical per-behavior
GPT-4o-mini judgements on HarmBench-400.

Several cells carry more than one GPT-4o-mini scoring under `results_api/`, with
no consistent filename convention across attacks. We therefore select, per cell,
the scoring whose attack_success_rate matches `artifacts/master_asr_long.csv` --
the table every other number in the paper is built from. The run asserts that all
30 cells match, so the selection is self-validating rather than assumed.
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "toolkits" / "HarmBench" / "results"
MASTER = ROOT / "artifacts" / "master_asr_long.csv"
OUT = ROOT / "artifacts" / "revision" / "difficulty_stratification.csv"

FAMILIES = {"gemma2": "gemma2", "llama31": "llama31", "qwen3": "qwen3",
            "phi4": "phi4", "qwen25": "qwen25_14b"}
# DeepInception's HarmBench run lives in its own results directory.
ATTACK_DIRS = {"PAIR": "PAIR", "ArtPrompt": "ArtPrompt",
               "DeepInception": "DeepInception_HarmBench"}
BINS = [(0.0, 0.2, "Easy (0-20%)"), (0.2, 0.5, "Medium (20-50%)"),
        (0.5, 0.8, "Hard (50-80%)"), (0.8, 1.01, "Very Hard (80-100%)")]


def pick_scoring(attack: str, stem: str, config: str, target_asr: float):
    """Return the per-behavior details dict whose ASR matches the canonical table."""
    candidates = []
    for d in glob.glob(f"{RESULTS}/{ATTACK_DIRS[attack]}/{stem}_{config}_custom*"):
        suffix = os.path.basename(d).replace(f"{stem}_{config}_custom", "")
        if suffix not in ("", "_pair_experiment"):
            continue  # skip seed / ablation variants
        candidates += glob.glob(f"{d}/results_api/*.json")
    for path in candidates:
        try:
            blob = json.load(open(path))
        except Exception:
            continue
        details, asr = blob.get("details"), blob.get("attack_success_rate")
        if isinstance(details, dict) and asr is not None and abs(asr * 100 - target_asr) < 1e-6:
            return details
    return None


LABELS = ROOT / "data" / "evaluator_labels"


def release_panel():
    """Release mode: GPT-4o-mini HarmBench verdicts from data/evaluator_labels/."""
    lab = pd.read_csv(LABELS / "original_grid.csv.gz")
    lab = lab[(lab.evaluator == "GPT-4o-mini") & (lab.benchmark == "HarmBench")
              & lab.attack.isin(list(ATTACK_DIRS)) & lab.config.isin(["base_fp16", "fft"])]
    acc = {"base_fp16": defaultdict(lambda: [0, 0]), "fft": defaultdict(lambda: [0, 0])}
    for (attack, family, config), g in lab.groupby(["attack", "family", "config"]):
        if len(g) != 400:
            raise SystemExit(f"incomplete cell {attack}/{family}/{config}")
        for b, y in zip(g.behavior_id, g.label):
            acc[config][b][0] += int(y)
            acc[config][b][1] += 1
    return acc


def main() -> None:
    if not RESULTS.is_dir():
        acc = release_panel()
        return summarize(acc)
    master = pd.read_csv(MASTER)
    master = master[(master.evaluator == "GPT-4o-mini") & (master.dataset == "HarmBench")]

    acc = {"base_fp16": defaultdict(lambda: [0, 0]), "fft": defaultdict(lambda: [0, 0])}
    for attack in ATTACK_DIRS:
        for family, stem in FAMILIES.items():
            for config in ("base_fp16", "fft"):
                row = master[(master.attack == attack) & (master.model_family == family)
                             & (master.config == config)]
                if row.empty:
                    raise SystemExit(f"no canonical ASR for {attack}/{family}/{config}")
                details = pick_scoring(attack, stem, config, float(row.ASR.iloc[0]))
                if details is None:
                    raise SystemExit(f"no scoring matches canonical ASR for {attack}/{family}/{config}")
                for behavior, entries in details.items():
                    if isinstance(entries, list) and entries and "judge_label" in entries[0]:
                        acc[config][behavior][0] += int(entries[0]["judge_label"])
                        acc[config][behavior][1] += 1

    summarize(acc)


def summarize(acc) -> None:
    base, fft = acc["base_fp16"], acc["fft"]
    counts = {n for b in base for n in (base[b][1],)} | {n for b in fft for n in (fft[b][1],)}
    if counts != {len(ATTACK_DIRS) * len(FAMILIES)}:
        raise SystemExit(f"unbalanced panel: observations per behavior = {sorted(counts)}")

    rows = [{"behavior_id": b,
             "base_asr": base[b][0] / base[b][1],
             "fft_asr": fft[b][0] / fft[b][1]}
            for b in sorted(set(base) & set(fft))]
    df = pd.DataFrame(rows)
    df["delta_asr"] = df.fft_asr - df.base_asr
    df["bin"] = pd.cut(df.base_asr, [b[0] for b in BINS] + [BINS[-1][1]],
                       labels=[b[2] for b in BINS], right=False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"behaviors={len(df)}, observations per behavior={counts.pop()}")
    print(f"{'Difficulty (Base ASR)':22}{'N':>5}{'Base ASR':>11}{'FFT ASR':>10}{'dASR':>9}")
    for _, _, label in BINS:
        s = df[df.bin == label]
        if len(s):
            print(f"{label:22}{len(s):5d}{s.base_asr.mean()*100:10.1f}%"
                  f"{s.fft_asr.mean()*100:9.1f}%{s.delta_asr.mean()*100:+8.1f}")
    # Spearman on the exact integer success counts, not the float rates. Each behavior
    # has 15 observations, so base and delta take only 16 and 15 distinct values; float
    # division and subtraction split mathematically-equal values by a few ulp, which
    # silently corrupts the rank ties (delta_asr shows 28 float values for 15 real ones)
    # and moves rho by ~0.02 depending on whether the frame round-tripped through CSV.
    n_obs = len(ATTACK_DIRS) * len(FAMILIES)
    base_k = (df.base_asr * n_obs).round().astype(int)
    delta_k = (df.fft_asr * n_obs).round().astype(int) - base_k
    rho, p = stats.spearmanr(base_k, delta_k)
    print(f"\nSpearman rho = {rho:.3f}, p = {p:.3g}")

    # Per-category breakdown (same panel), for the semantic-category table.
    if RESULTS.is_dir():
        behaviors = pd.read_csv(
            ROOT / "toolkits" / "HarmBench" / "data" / "behavior_datasets"
            / "harmbench_behaviors_text_all.csv"
        )
        catmap = dict(zip(behaviors.BehaviorID, behaviors.SemanticCategory))
    else:
        ids = pd.read_csv(LABELS / "behavior_ids.csv")
        catmap = dict(zip(ids.behavior_id, ids.category))
    df["category"] = df.behavior_id.map(catmap)
    if df.category.isna().any():
        raise SystemExit("unmapped behaviors in category join")
    cats = df.groupby("category").agg(n=("base_asr", "size"), base=("base_asr", "mean"),
                                      fft=("fft_asr", "mean"))
    print(f"\n{'category':32}{'n':>5}{'Base ASR':>10}{'FFT ASR':>10}{'dASR':>9}")
    for name, r in cats.iterrows():
        print(f"  {name:30}{int(r.n):5d}{r.base*100:9.1f}%{r.fft*100:9.1f}%"
              f"{(r.fft - r.base)*100:+8.1f}")

    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
