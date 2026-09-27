#!/usr/bin/env python3
"""Copyright-behavior sensitivity of the black-box HarmBench ASR, per evaluator.

HarmBench-400 contains 100 copyright behaviors (FunctionalCategory == "copyright",
tag "hash_check"). The HB-Cls pipeline (toolkits/HarmBench/evaluate_completions.py)
scores these with the MinHash hash check (compute_results_hashing), whereas
GPT-4o-mini and LlamaGuard-3 score them with their usual prompts. This script
measures how much that asymmetry moves the evaluator comparison.

Like-for-like design: only (family, config, attack) cells of the black-box
HarmBench grid (5 families x 5 configs x {PAIR, DeepInception, ArtPrompt} = 75)
for which per-behavior verdicts exist for ALL THREE evaluators are used, and a
verdict file is accepted only if its recomputed ASR matches the canonical
per-attack CSV (results/<attack>_harmbench_all_evaluators.csv) within 0.4 pp.

Outputs artifacts/revision/copyright_sensitivity.csv with three row types:
  - cell:   per (family, config, attack) cell and evaluator, ASR all-400 / no-copyright
  - pooled: per evaluator, mean ASR over the common cells (all-400 / no-copyright)
            plus the GPT-4o-mini - HB-Cls and LlamaGuard-3 - HB-Cls gaps
  - delta:  per evaluator and fine-tuned config, mean dASR vs base_fp16, paired
            within family x attack, over pairs where both cells are in the common set

Calibration rows (paper Table "Copyright sensitivity", bottom block) are computed by
calibration() from the shipped human-annotation labels
(data/human_annotation/asr_500_labels.csv) and written to
artifacts/revision/copyright_calibration.csv. They run without any verdict files.
"""

import csv
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HB = ROOT / "toolkits" / "HarmBench"
OUT = ROOT / "artifacts" / "revision" / "copyright_sensitivity.csv"

FAMILIES = ["gemma2", "llama31", "qwen3", "phi4", "qwen25"]
FAMILY_DIR = {"qwen25": "qwen25_14b"}
CONFIGS = ["base_fp16", "base_4bit", "lora", "qlora", "fft"]
ATTACKS = {"PAIR": "pair", "DeepInception": "deepinception", "ArtPrompt": "artprompt"}
RESULT_DIRS = {
    "PAIR": ["PAIR/{m}_pair_experiment", "PAIR/{m}"],
    "DeepInception": ["DeepInception_HarmBench/{m}", "DeepInception/{m}"],
    "ArtPrompt": ["ArtPrompt/{m}"],
}
EVALUATORS = {"llama2": "HB-Cls", "gpt4o": "GPT-4o-mini", "llamaguard": "LlamaGuard-3"}
FILE_PATTERNS = {
    "llama2": ["{d}/results/{m}.json"],
    "gpt4o": ["{d}/results_api/{m}_api_eval.json", "{d}/results_api/{m}_api.json", "{d}/results_api/{m}.json"],
    "llamaguard": ["{d}/results_llamaguard/{m}.json", "{d}/results_llamaguard3/{m}.json"],
}
TOL = 0.004


def load_categories():
    path = HB / "data" / "behavior_datasets" / "harmbench_behaviors_text_all.csv"
    with open(path) as f:
        return {r["BehaviorID"]: r["FunctionalCategory"] for r in csv.DictReader(f)}


def load_labels(path, categories):
    data = json.load(open(path))
    if isinstance(data, dict) and "details" in data:
        data = data["details"]
    labels = {}
    for bid, entries in data.items():
        if bid not in categories or not isinstance(entries, list) or not entries:
            continue
        e = entries[0]
        raw = e.get("judge_label", e.get("label"))
        try:
            labels[bid] = int(str(raw)) == 1
        except (TypeError, ValueError):
            continue
    return labels


def find_cell(attack, model, evaluator, target, categories):
    candidates = []
    for d in RESULT_DIRS[attack]:
        base = HB / "results" / d.format(m=model)
        candidates += [Path(p.format(d=base, m=model)) for p in FILE_PATTERNS[evaluator]]
    for path in candidates:
        if not path.exists():
            continue
        labels = load_labels(path, categories)
        if len(labels) != 400:
            continue
        asr = np.mean(list(labels.values()))
        if target is None or abs(asr - target) <= TOL:
            return path, labels
    return None, None


def asr(labels, keys):
    return 100.0 * np.mean([labels[k] for k in keys])


def calibration():
    """Evaluator FPR / recall on the clean-consensus set, all vs non-copyright samples."""
    labels = ROOT / "data" / "human_annotation" / "asr_500_labels.csv"
    with open(labels) as f:
        rows = [r for r in csv.DictReader(f) if r["agree"] == "1"]
    cols = {"HB-Cls": "hbcls_label", "GPT-4o-mini": "gpt4omini_label", "LlamaGuard-3": "llamaguard3_label"}
    out = []
    for subset, keep in [("all", lambda r: True), ("non_copyright", lambda r: r["functional_category"] != "copyright")]:
        sub = [r for r in rows if keep(r)]
        for name, col in cols.items():
            tp = sum(r[col] == "1" and r["consensus"] == "1" for r in sub)
            fp = sum(r[col] == "1" and r["consensus"] == "0" for r in sub)
            n_pos = sum(r["consensus"] == "1" for r in sub)
            n_neg = len(sub) - n_pos
            out.append(dict(subset=subset, evaluator=name, n=len(sub), n_harmful=n_pos, n_safe=n_neg,
                            fpr=round(100 * fp / n_neg, 1), recall=round(100 * tp / n_pos, 1)))
    path = ROOT / "artifacts" / "revision" / "copyright_calibration.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    n_cp = sum(r["functional_category"] == "copyright" for r in rows)
    print(f"Calibration set: {len(rows)} clean-consensus samples, {n_cp} copyright behaviors")
    for r in out:
        print(r)
    print(f"Wrote {path}")


LABELS = ROOT / "data" / "evaluator_labels"
# The paper's copyright table uses the 40 PAIR and ArtPrompt HarmBench cells whose per-behavior
# verdict files for all three evaluators reproduce the canonical per-cell ASR.
RELEASE_ATTACKS = ("PAIR", "ArtPrompt")


def load_from_release():
    """Release mode: categories and verdicts from data/evaluator_labels/."""
    with open(LABELS / "behavior_ids.csv") as f:
        categories = {r["behavior_id"]: r["category"] for r in csv.DictReader(f) if r["benchmark"] == "HarmBench"}
    import gzip
    raw = {}
    with gzip.open(LABELS / "original_grid.csv.gz", "rt") as f:
        for r in csv.DictReader(f):
            if r["benchmark"] != "HarmBench" or r["attack"] not in RELEASE_ATTACKS:
                continue
            ev = {v: k for k, v in EVALUATORS.items()}[r["evaluator"]]
            raw.setdefault((r["family"], r["config"], r["attack"]), {}).setdefault(ev, {})[r["behavior_id"]] = r["label"] == "1"
    cells = {k: v for k, v in raw.items() if len(v) == 3 and all(len(x) == 400 for x in v.values())}
    paths = {(fa, c, a, ev): Path("data/evaluator_labels/original_grid.csv.gz")
             for (fa, c, a) in cells for ev in EVALUATORS}
    return categories, cells, paths


def main():
    if not HB.is_dir():
        categories, cells, paths = load_from_release()
        non_cp = {k for k, v in categories.items() if v != "copyright"}
        return report(cells, paths, non_cp)
    categories = load_categories()
    non_cp = {k for k, v in categories.items() if v != "copyright"}

    cells = {}  # (family, config, attack) -> {evaluator: labels}
    paths = {}
    for attack, stem in ATTACKS.items():
        with open(ROOT / "results" / f"{stem}_harmbench_all_evaluators.csv") as f:
            canon = {(r["model_family"], r["config"]): r for r in csv.DictReader(f)}
        for fam in FAMILIES:
            for cfg in CONFIGS:
                model = f"{FAMILY_DIR.get(fam, fam)}_{cfg}_custom"
                row = canon.get((fam, cfg), {})
                found = {}
                for ev in EVALUATORS:
                    val = row.get(f"{ev}_asr")
                    target = float(val) if val not in (None, "") else None
                    path, labels = find_cell(attack, model, ev, target, categories)
                    if labels is not None:
                        found[ev] = labels
                        paths[(fam, cfg, attack, ev)] = path
                if len(found) == 3:
                    cells[(fam, cfg, attack)] = found
    report(cells, paths, non_cp)


def report(cells, paths, non_cp):
    rows = []
    for (fam, cfg, attack), found in sorted(cells.items()):
        for ev, name in EVALUATORS.items():
            labels = found[ev]
            keys = list(labels)
            rows.append(dict(row_type="cell", family=fam, config=cfg, attack=attack, evaluator=name,
                             n_cells=1, asr_all=round(asr(labels, keys), 2),
                             asr_no_copyright=round(asr(labels, [k for k in keys if k in non_cp]), 2),
                             source=str(paths[(fam, cfg, attack, ev)].relative_to(ROOT))
                             if paths[(fam, cfg, attack, ev)].is_absolute() else str(paths[(fam, cfg, attack, ev)])))

    pooled = {}
    for ev, name in EVALUATORS.items():
        a = np.mean([asr(f[ev], list(f[ev])) for f in cells.values()])
        n = np.mean([asr(f[ev], [k for k in f[ev] if k in non_cp]) for f in cells.values()])
        pooled[ev] = (a, n)
        rows.append(dict(row_type="pooled", evaluator=name, n_cells=len(cells),
                         asr_all=round(a, 2), asr_no_copyright=round(n, 2)))
    for ev in ("gpt4o", "llamaguard"):
        rows.append(dict(row_type="pooled_gap", evaluator=f"{EVALUATORS[ev]} - HB-Cls", n_cells=len(cells),
                         asr_all=round(pooled[ev][0] - pooled["llama2"][0], 2),
                         asr_no_copyright=round(pooled[ev][1] - pooled["llama2"][1], 2)))

    for ev, name in EVALUATORS.items():
        for cfg in ["base_4bit", "lora", "qlora", "fft"]:
            d_all, d_nc, pairs = [], [], []
            for (fam, c, attack), found in cells.items():
                if c != cfg or (fam, "base_fp16", attack) not in cells:
                    continue
                lab, base = found[ev], cells[(fam, "base_fp16", attack)][ev]
                keys = sorted(set(lab) & set(base))
                nk = [k for k in keys if k in non_cp]
                d_all.append(asr(lab, keys) - asr(base, keys))
                d_nc.append(asr(lab, nk) - asr(base, nk))
                pairs.append(f"{fam}/{attack}")
            if pairs:
                rows.append(dict(row_type="delta", config=cfg, evaluator=name, n_cells=len(pairs),
                                 asr_all=round(np.mean(d_all), 2), asr_no_copyright=round(np.mean(d_nc), 2),
                                 source=";".join(sorted(pairs))))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ["row_type", "family", "config", "attack", "evaluator", "n_cells",
              "asr_all", "asr_no_copyright", "source"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"Common cells (all three evaluators): {len(cells)} of 75")
    for attack in ATTACKS:
        print(f"  {attack}: " + ", ".join(f"{fa}/{c}" for (fa, c, a) in sorted(cells) if a == attack))
    for r in rows:
        if r["row_type"] != "cell":
            print({k: v for k, v in r.items() if k in ("row_type", "config", "evaluator", "n_cells",
                                                      "asr_all", "asr_no_copyright")})
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    calibration()
    main()
