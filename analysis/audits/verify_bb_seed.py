#!/usr/bin/env python3
"""
Traces the black-box seed-robustness numbers to the underlying run files.
Recomputes HB-Cls ASR directly from each run's `is_jailbroken` field.
Seeds: 3407 (primary/original), 0, 123. (There is NO seed-42 run.)
Writes artifacts/revision_stats/bb_seed_proof.csv and prints paths.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "artifacts" / "revision_stats"
OUT.mkdir(parents=True, exist_ok=True)

# (attack, dataset, {seed_label: result-json relative path})
CELLS = {
    ("PAIR", "JBB"): {
        "3407": "toolkits/HarmBench/results_jbb/PAIR/llama31_lora_custom_pair_experiment/results/llama31_lora_custom.json",
        "0":    "toolkits/HarmBench/results_jbb/PAIR/llama31_lora_seed0_custom_pair_experiment/results/llama31_lora_seed0_custom.json",
        "123":  "toolkits/HarmBench/results_jbb/PAIR/llama31_lora_seed123_custom_pair_experiment/results/llama31_lora_seed123_custom.json",
    },
    ("DeepInception", "HarmBench"): {
        "3407": "toolkits/HarmBench/results/DeepInception/llama31_lora_custom/results/llama31_lora_custom.json",
        "0":    "toolkits/HarmBench/results/DeepInception/llama31_lora_seed0_custom/results/llama31_lora_seed0_custom.json",
        "123":  "toolkits/HarmBench/results/DeepInception/llama31_lora_seed123_custom/results/llama31_lora_seed123_custom.json",
    },
    ("ArtPrompt", "HarmBench"): {
        "3407": "toolkits/HarmBench/results/ArtPrompt/llama31_lora_custom/results/llama31_lora_custom.json",
        "0":    "toolkits/HarmBench/results/ArtPrompt/llama31_lora_seed0_custom/results/llama31_lora_seed0_custom.json",
        "123":  "toolkits/HarmBench/results/ArtPrompt/llama31_lora_seed123_custom/results/llama31_lora_seed123_custom.json",
    },
    ("AutoDAN", "HarmBench"): {   # white-box, for contrast
        "3407": "toolkits/HarmBench/results/AutoDAN/llama31_lora_custom/results/llama31_lora_custom.json",
        "0":    "toolkits/HarmBench/results/AutoDAN/llama31_lora_seed0_custom/results/llama31_lora_seed0_custom.json",
        "123":  "toolkits/HarmBench/results/AutoDAN/llama31_lora_seed123_custom/results/llama31_lora_seed123_custom.json",
    },
}
PARADIGM = {"PAIR": "black-box", "DeepInception": "black-box",
            "ArtPrompt": "black-box", "AutoDAN": "white-box"}


def asr_from_file(path):
    d = json.load(open(REPO / path))
    flags = []
    for v in d.values():
        for entry in (v if isinstance(v, list) else [v]):
            if "is_jailbroken" in entry:           # DeepInception-style
                flags.append(1 if entry["is_jailbroken"] else 0)
            elif "label" in entry:                  # PAIR / ArtPrompt-style
                flags.append(int(entry["label"]))
    return 100.0 * np.mean(flags), len(flags)


rows = []
for (attack, dataset), seeds in CELLS.items():
    for seed, path in seeds.items():
        full = REPO / path
        if not full.exists():
            rows.append(dict(attack=attack, dataset=dataset, seed=seed,
                             asr_pct=np.nan, n=0, status="MISSING", path=path))
            continue
        asr, n = asr_from_file(path)
        rows.append(dict(paradigm=PARADIGM[attack], attack=attack, dataset=dataset, seed=seed,
                         asr_pct=round(asr, 2), n=n, status="ok", path=path))

df = pd.DataFrame(rows)
df.to_csv(OUT / "bb_seed_proof.csv", index=False)
print("=== Black-box ASR (HB-Cls) recomputed from is_jailbroken, per seed ===")
for (attack, dataset), g in df.groupby(["attack", "dataset"]):
    vals = g[g.status == "ok"]["asr_pct"].to_numpy()
    sd = np.std(vals, ddof=1) if len(vals) > 1 else float("nan")
    print(f"\n{attack}/{dataset}:  seeds {{3407,0,123}}  ASR={list(vals)}  "
          f"mean={np.mean(vals):.2f}  SD={sd:.2f}  range={max(vals)-min(vals):.2f}")
    for _, r in g.iterrows():
        print(f"    seed {r.seed:>4}: ASR={r.asr_pct}  n={r.n}  [{r.status}]  {r.path}")
print(f"\nWrote {OUT/'bb_seed_proof.csv'}")
