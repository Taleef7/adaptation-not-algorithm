#!/usr/bin/env python3
"""Do successful attacks transfer across adaptation methods?

The same-behavior analysis compares LoRA against FFT. This extends it to all
three training methods and asks the transfer question directly: given that a
behavior is jailbroken under one adaptation method, how often is the same
behavior jailbroken under another?

Two summaries are reported.

  1. Directed transfer rate  P(jailbroken under B | jailbroken under A) for every
     ordered pair. Asymmetry matters: if method B has a larger success set, the
     rate A->B will exceed B->A for purely arithmetic reasons, so both directions
     are shown alongside the marginal rates.

  2. Three-way partition of the union. Of all behaviors jailbroken by at least one
     method, what fraction falls to all three, to exactly two, to exactly one? A
     shared-vulnerability account predicts mass concentrated on "all three"; a
     method-specific account predicts mass on "exactly one."

Data: per-behavior HB-Cls labels for the seed-42 uniform grid,
      toolkits/HarmBench/results_phase5/scored/*_hbcls.jsonl (controlled access), or the
      shipped data/evaluator_labels/seed42_grid.csv.gz when those files are absent
      (5 families x {DeepInception, ArtPrompt} x {JBB-100, HarmBench-400}).
Output: artifacts/revision/attack_transfer.csv
"""

import csv
import itertools
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCORED = os.path.join(REPO, "toolkits/HarmBench/results_phase5/scored")
OUT = os.path.join(REPO, "artifacts", "revision", "attack_transfer.csv")
LABELS = os.path.join(REPO, "data", "evaluator_labels", "seed42_grid.csv.gz")
_LAB = None
_TIE = None


_ORDER = None


def _load_from_labels(family, method, bench, atk_token):
    """Release mode: read the shipped behavior-level HB-Cls labels.

    The raw scored files are keyed by behavior text, and 12 HarmBench contextual
    behaviors fall into 5 groups that share their text (they differ only in context), so
    the raw path keeps 393 distinct HarmBench keys (the label of the group's last line in
    the scored file). `same_text_group` in behavior_ids.csv encodes those groups, and
    seed42_text_group_labels.csv records the label the raw path used wherever the group's
    members disagree, which reproduces the raw-path result exactly.
    """
    global _LAB, _ORDER, _TIE
    if _LAB is None:
        import csv, gzip
        _LAB = defaultdict(dict)
        with gzip.open(LABELS, "rt") as fh:
            for r in csv.DictReader(fh):
                if r["evaluator"] == "HB-Cls":
                    _LAB[(r["family"], r["config"], r["benchmark"], r["attack"])][r["behavior_id"]] = int(r["label"])
        with open(os.path.join(os.path.dirname(LABELS), "behavior_ids.csv")) as fh:
            _ORDER = [(r["behavior_id"], r["same_text_group"] or r["behavior_id"]) for r in csv.DictReader(fh)]
        with open(os.path.join(os.path.dirname(LABELS), "seed42_text_group_labels.csv")) as fh:
            _TIE = {(r["family"], r["config"], r["benchmark"], r["attack"], r["same_text_group"]): int(r["label"])
                    for r in csv.DictReader(fh) if r["evaluator"] == "HB-Cls"}
    atk = "ArtPrompt" if atk_token else "DeepInception"
    cell = _LAB.get((family, method, {"jbb": "JBB", "harmbench": "HarmBench"}[bench], atk))
    if not cell:
        return None
    b = {"jbb": "JBB", "harmbench": "HarmBench"}[bench]
    out = {}
    for bid, key in _ORDER:
        if bid in cell:
            out[key] = _TIE.get((family, method, b, atk, key), cell[bid])
    return out

FAMILIES = ["llama31", "gemma2", "qwen3", "phi4", "qwen25"]
METHODS = ["lora", "qlora", "fft"]
BENCHES = ["jbb", "harmbench"]
# DeepInception has no attack token in the filename; ArtPrompt does.
ATTACKS = {"DeepInception": "", "ArtPrompt": "_artprompt"}


def load(family, method, bench, atk_token):
    if not os.path.isdir(SCORED):
        return _load_from_labels(family, method, bench, atk_token)
    path = os.path.join(SCORED, f"{family}_{method}_seed42_{bench}{atk_token}_hbcls.jsonl")
    if not os.path.exists(path):
        return None
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["behavior_goal"]] = int(r["jailbroken"])
    return out


def main():
    pair_stats = defaultdict(list)   # (A,B) -> list of directed transfer rates
    pair_tables = defaultdict(list)  # (A,B) -> 2x2 tables for pooling
    partition = defaultdict(int)
    rows = []
    skipped = []

    for family, bench, (atk, tok) in itertools.product(FAMILIES, BENCHES, ATTACKS.items()):
        cells = {m: load(family, m, bench, tok) for m in METHODS}
        missing = [m for m, v in cells.items() if v is None]
        if missing:
            skipped.append(f"{family}/{atk}/{bench}: missing {','.join(missing)}")
            continue
        # restrict to behaviors present in all three
        keys = sorted(set.intersection(*(set(v) for v in cells.values())))
        if not keys:
            skipped.append(f"{family}/{atk}/{bench}: no shared behaviors")
            continue
        # boolean, not int: `~` on an int array is bitwise complement, not logical NOT
        vec = {m: np.array([cells[m][k] for k in keys], dtype=bool) for m in METHODS}

        # three-way partition over the union
        stack = np.vstack([vec[m].astype(int) for m in METHODS])
        hits = stack.sum(axis=0)
        for h in hits:
            if h > 0:
                partition[int(h)] += 1

        for A, B in itertools.permutations(METHODS, 2):
            nA = int(vec[A].sum())
            if nA == 0:
                continue
            both = int((vec[A] & vec[B]).sum())
            rate = both / nA
            pair_stats[(A, B)].append(rate)
            rows.append(dict(family=family, attack=atk, bench=bench, src=A, dst=B,
                             n_src=nA, n_dst=int(vec[B].sum()), both=both,
                             transfer_rate=rate, base_rate_dst=vec[B].mean()))
        for A, B in itertools.combinations(METHODS, 2):
            a = int((vec[A] & vec[B]).sum())
            b = int((vec[A] & ~vec[B]).sum())
            c = int((~vec[A] & vec[B]).sum())
            d = int((~vec[A] & ~vec[B]).sum())
            pair_tables[(A, B)].append((a, b, c, d))

    print("=" * 78)
    print("ATTACK TRANSFER ACROSS ADAPTATION METHODS (seed-42 grid, HB-Cls)")
    print("=" * 78)
    if skipped:
        print(f"\nSKIPPED {len(skipped)} cell(s) with incomplete scoring:")
        for s in skipped:
            print("   ", s)

    print("\n[1] Directed transfer  P(jailbroken under dst | jailbroken under src)\n")
    print(f"{'src -> dst':<20}{'cells':>6}{'mean':>8}{'median':>8}{'dst base rate':>15}{'lift':>7}")
    for (A, B), vals in sorted(pair_stats.items()):
        v = np.array(vals)
        base = np.array([r["base_rate_dst"] for r in rows if r["src"] == A and r["dst"] == B])
        print(f"{A + ' -> ' + B:<20}{len(v):>6}{v.mean():>8.3f}{np.median(v):>8.3f}"
              f"{base.mean():>15.3f}{v.mean() / base.mean():>7.2f}x")

    print("\n[2] Pooled pairwise association (Mantel-Haenszel)\n")
    for (A, B), tabs in sorted(pair_tables.items()):
        num = sum(a * d / (a + b + c + d) for a, b, c, d in tabs)
        den = sum(b * c / (a + b + c + d) for a, b, c, d in tabs)
        or_mh = num / den if den else float("inf")
        jac = np.mean([a / (a + b + c) if (a + b + c) else np.nan for a, b, c, d in tabs])
        print(f"  {A:<6} vs {B:<6}  MH OR = {or_mh:6.2f}   mean Jaccard = {jac:.3f}")

    print("\n[3] Partition of the union of jailbroken behaviors\n")
    tot = sum(partition.values())
    for k in sorted(partition):
        lab = {1: "exactly one method", 2: "exactly two methods", 3: "all three methods"}[k]
        print(f"  {lab:<24}{partition[k]:>6}  ({100 * partition[k] / tot:.1f}%)")
    print(f"  {'union total':<24}{tot:>6}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {os.path.relpath(OUT, REPO)}")


if __name__ == "__main__":
    main()
