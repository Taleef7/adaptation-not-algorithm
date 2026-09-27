#!/usr/bin/env python
"""Seed-42 PAIR step 1.5: merge per-behavior test cases for all 18 experiments
(9 cells x {JBB-100, HarmBench-400}) in a single process, importing the HarmBench baselines once.
Rewrites <save_dir>/test_cases.json from test_cases_individual_behaviors/; safe to re-run."""
import os, sys, json, time
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
HB = f"{PROJECT_ROOT}/toolkits/HarmBench"
os.chdir(HB)
sys.path.insert(0, HB)
t0 = time.time()
from baselines import get_method_class
print(f"[import] baselines loaded in {time.time()-t0:.1f}s", flush=True)
merge = get_method_class("PAIR").merge_test_cases

BASES = ["llama31_lora","llama31_qlora","llama31_fft","gemma2_lora","gemma2_qlora",
         "gemma2_fft","qwen3_lora","qwen3_qlora","qwen3_fft"]

def do(save_dir, expected, label):
    ind = os.path.join(save_dir, "test_cases_individual_behaviors")
    n_ind = len(os.listdir(ind)) if os.path.isdir(ind) else 0
    if n_ind != expected:
        print(f"SKIP_INCOMPLETE {label} ({n_ind}/{expected})", flush=True); return
    t = time.time()
    merge(save_dir)
    tc = os.path.join(save_dir, "test_cases.json")
    n = len(json.load(open(tc)))
    print(f"MERGED {label} -> {n} behaviors  ({time.time()-t:.1f}s)", flush=True)

print("PAIR_SEED42_MERGE_START", flush=True)
for b in BASES:
    do(f"results_phase5/PAIR_jbb/{b}_seed42_pair_experiment/test_cases", 100, f"JBB:{b}")
for b in BASES:
    do(f"results_phase5/PAIR_hb/test_{b}_seed42_custom/test_cases", 400, f"HB:{b}")
print("PAIR_SEED42_MERGE_DONE", flush=True)
