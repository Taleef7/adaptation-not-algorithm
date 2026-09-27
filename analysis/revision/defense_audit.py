"""
Protocol-as-defense-audit: re-measure SafeLoRA's claimed jailbreak "protection"
under 3 evaluators x 3 seeds, against MATCHED standard-LoRA baselines (not the
unprotected base). Mirrors the "AutoDAN LoRA-protection is seed-noise" finding,
applied to a published defense.

Cell: Llama-3.1-8B-Instruct x AutoDAN x HarmBench(400). All data on disk; no new compute.

ASR per evaluator:
  HB-Cls       : frac of 400 behaviors with >=1 completion labeled 1
                 (results/<name>.json, per-behavior 'label')
  GPT-4o-mini  : results_api/<name>_api_eval.json['attack_success_rate']
  LlamaGuard-3 : results_llamaguard/<name>.json['attack_success_rate']

Output: artifacts/revision/defense_audit.csv
"""
from __future__ import annotations
import csv
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "toolkits" / "HarmBench" / "results" / "AutoDAN"
SEEDS = ["main", "seed0", "seed123"]
EVALS = ["HB-Cls", "GPT-4o-mini", "LlamaGuard-3"]


def cell_name(method: str, seed: str) -> str:
    base = {"safelora": "llama31_safe_lora_faithful", "lora": "llama31_lora"}[method]
    suff = "" if seed == "main" else f"_{seed}"
    return f"{base}{suff}_custom"


def asr_hbcls(name: str) -> float:
    d = json.load(open(ROOT / name / "results" / f"{name}.json"))
    hit = sum(1 for _b, cs in d.items() if any(int(c.get("label", 0)) == 1 for c in cs))
    return 100.0 * hit / len(d)


def asr_api(name: str) -> float:
    d = json.load(open(ROOT / name / "results_api" / f"{name}_api_eval.json"))
    return 100.0 * d["attack_success_rate"]


def asr_lg(name: str) -> float:
    d = json.load(open(ROOT / name / "results_llamaguard" / f"{name}.json"))
    return 100.0 * d["attack_success_rate"]


def asr(name: str, ev: str) -> float:
    return {"HB-Cls": asr_hbcls, "GPT-4o-mini": asr_api, "LlamaGuard-3": asr_lg}[ev](name)


def main() -> None:
    cells = {(m, s): {ev: asr(cell_name(m, s), ev) for ev in EVALS}
             for m in ("lora", "safelora") for s in SEEDS}
    base = {ev: asr("llama31_base_fp16_custom", ev) for ev in EVALS}

    print("== Per-cell ASR (Llama-3.1 x AutoDAN x HarmBench-400) ==")
    print(f"{'cell':28s}" + "".join(f"{e:>13s}" for e in EVALS))
    print(f"{'base_fp16 (unprotected)':28s}" + "".join(f"{base[e]:13.1f}" for e in EVALS))
    for m in ("lora", "safelora"):
        for s in SEEDS:
            c = cells[(m, s)]
            print(f"{m + '/' + s:28s}" + "".join(f"{c[e]:13.1f}" for e in EVALS))

    print("\n== Matched-seed protection (LoRA - SafeLoRA); +ve = SafeLoRA helps ==")
    print(f"{'seed':10s}" + "".join(f"{e:>13s}" for e in EVALS))
    deltas = {ev: [] for ev in EVALS}
    for s in SEEDS:
        line = f"{s:10s}"
        for ev in EVALS:
            d = cells[("lora", s)][ev] - cells[("safelora", s)][ev]
            deltas[ev].append(d)
            line += f"{d:13.1f}"
        print(line)
    print(f"{'mean':10s}" + "".join(f"{statistics.mean(deltas[e]):13.1f}" for e in EVALS))
    print(f"{'sd':10s}" + "".join(f"{statistics.stdev(deltas[e]):13.1f}" for e in EVALS))

    print("\n== 'Naive' headline vs honest matched read (HB-Cls) ==")
    b, sm, lm = base["HB-Cls"], cells[("safelora", "main")]["HB-Cls"], cells[("lora", "main")]["HB-Cls"]
    print(f"  vs unprotected base : {b:.1f}% -> {sm:.1f}%  = {b - sm:+.1f}pp 'protection'")
    print(f"  vs MATCHED LoRA-main: {lm:.1f}% -> {sm:.1f}%  = {lm - sm:+.1f}pp")

    out = REPO / "artifacts" / "revision" / "defense_audit.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "seed"] + EVALS)
        w.writerow(["base_fp16", "-"] + [f"{base[e]:.2f}" for e in EVALS])
        for m in ("lora", "safelora"):
            for s in SEEDS:
                w.writerow([m, s] + [f"{cells[(m, s)][e]:.2f}" for e in EVALS])
        w.writerow([])
        w.writerow(["protection_LoRA_minus_SafeLoRA", "seed"] + EVALS)
        for i, s in enumerate(SEEDS):
            w.writerow(["delta", s] + [f"{deltas[e][i]:.2f}" for e in EVALS])
        w.writerow(["delta", "mean"] + [f"{statistics.mean(deltas[e]):.2f}" for e in EVALS])
        w.writerow(["delta", "sd"] + [f"{statistics.stdev(deltas[e]):.2f}" for e in EVALS])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
