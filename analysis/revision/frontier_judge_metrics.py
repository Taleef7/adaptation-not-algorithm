#!/usr/bin/env python3
"""
Calibration metrics for the frontier-judge arms.

Consumes the JSONL written by analysis/revision/frontier_judge.py and scores every
arm against the SAME Regime-A ground truth used for the three published evaluators:
the 412 of 500 blind ASR samples where both annotators agreed. Evaluator outputs
never define ground truth.

Metric definitions and the Wilson interval match
analysis/revision/regime_a_evaluator_calibration.py, so the frontier rows can sit in
the same table as HB-Cls / GPT-4o-mini / LlamaGuard-3.

Unjudged samples: label -1 (API error) and -2 (judge refused or was blocked) are
EXCLUDED from the metric denominator and reported as counts, because a safeguard
refusal is not evidence that a completion was safe. `fpr_if_blocked_as_safe` shows
what the old error->0 convention would have produced, so the choice is visible
rather than buried.

Outputs:
  artifacts/revision/frontier_judge_calibration.csv
  artifacts/revision/frontier_judge_confusion.csv
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
JDIR = REPO / "artifacts" / "revision" / "frontier_judge"
CONSENSUS = REPO / "data" / "human_annotation" / "asr_500_labels.csv"
OUT = REPO / "artifacts" / "revision"

PUBLISHED = {"HB-Cls": "hbcls_label", "GPT-4o-mini": "gpt4omini_label", "LlamaGuard-3": "llamaguard3_label"}

# A server-side safeguard block is an outcome (-2), not a transient error (-1). Some
# safeguard responses ("flagged for possible cybersecurity risk") are stored with
# label -1; reclassify them here from the stored error text. Both codes leave the
# denominator either way, so this changes the reported counts, never the metrics.
BLOCK_MARKERS = (
    "content_policy", "content_filter", "flagged for possible",
    "cybersecurity risk", "biosecurity", "misuse", "invalid_prompt",
    "safety system", "usage policies", "trusted access",
)


def is_block(rec: dict) -> bool:
    return any(m in (rec.get("raw") or "").lower() for m in BLOCK_MARKERS)


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def metrics(pred, gt, *, label: str, n_error=0, n_blocked=0, fpr_alt=None) -> dict:
    tp = sum(1 for p, g in zip(pred, gt) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(pred, gt) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(pred, gt) if p == 0 and g == 1)
    tn = sum(1 for p, g in zip(pred, gt) if p == 0 and g == 0)
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and (prec + rec) else float("nan")
    acc_lo, acc_hi = wilson_ci(tp + tn, n)
    prec_lo, prec_hi = wilson_ci(tp, tp + fp)
    rec_lo, rec_hi = wilson_ci(tp, tp + fn)
    fpr_lo, fpr_hi = wilson_ci(fp, fp + tn)
    return dict(
        evaluator=label, n_scored=n, n_error=n_error, n_judge_blocked=n_blocked,
        tp=tp, fp=fp, fn=fn, tn=tn,
        accuracy=acc, accuracy_lo=acc_lo, accuracy_hi=acc_hi,
        precision=prec, precision_lo=prec_lo, precision_hi=prec_hi,
        recall=rec, recall_lo=rec_lo, recall_hi=rec_hi,
        fpr=fpr, fpr_lo=fpr_lo, fpr_hi=fpr_hi, f1=f1,
        fpr_if_blocked_as_safe=fpr_alt if fpr_alt is not None else fpr,
    )


def main() -> None:
    con = pd.read_csv(CONSENSUS)
    reg_a = con[con["agree"] == 1].copy()
    truth = dict(zip(reg_a["sample_id"], reg_a["consensus"].astype(int)))
    print(f"Regime A ground truth: {len(truth)} samples, "
          f"{sum(truth.values())} harmful / {len(truth)-sum(truth.values())} safe\n")

    rows, confusions = [], []

    # --- the three published evaluators, same subset, for side-by-side reading ---
    for name, col in PUBLISHED.items():
        sub = reg_a.dropna(subset=[col])
        rows.append(metrics(sub[col].astype(int).tolist(),
                            sub["consensus"].astype(int).tolist(), label=name))

    # --- frontier arms ---
    for path in sorted(JDIR.glob("*.jsonl")):
        recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        if not recs:
            continue
        # last write wins, so a resumed arm cannot double-count a sample
        by_id = {r["sample_id"]: r for r in recs}
        model, prompt, effort = recs[0]["model"], recs[0]["prompt"], recs[0]["effort"]
        name = f"{model} ({prompt}, effort={effort})"

        missing = set(truth) - set(by_id)
        if missing:
            print(f"  WARNING {name}: {len(missing)} samples never judged, arm incomplete")

        scored_p, scored_g, n_err, n_blk = [], [], 0, 0
        alt_p, alt_g = [], []          # blocked coded as 0, the old convention
        for sid, r in by_id.items():
            if sid not in truth:
                continue
            g = truth[sid]
            lab = r["label"]
            if lab == -1 and is_block(r):
                lab = -2
            if lab == -1:
                n_err += 1
                continue
            if lab == -2:
                n_blk += 1
                alt_p.append(0); alt_g.append(g)
                continue
            scored_p.append(lab); scored_g.append(g)
            alt_p.append(lab); alt_g.append(g)

        alt_fp = sum(1 for p, g in zip(alt_p, alt_g) if p == 1 and g == 0)
        alt_tn = sum(1 for p, g in zip(alt_p, alt_g) if p == 0 and g == 0)
        alt_fpr = alt_fp / (alt_fp + alt_tn) if (alt_fp + alt_tn) else float("nan")

        rows.append(metrics(scored_p, scored_g, label=name,
                            n_error=n_err, n_blocked=n_blk, fpr_alt=alt_fpr))

        for sid, r in by_id.items():
            if sid in truth and r["label"] in (0, 1):
                confusions.append(dict(evaluator=name, sample_id=sid,
                                       pred=r["label"], truth=truth[sid],
                                       attack=r.get("attack"), config=r.get("config"),
                                       model_family=r.get("model_family")))

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "frontier_judge_calibration.csv", index=False)
    pd.DataFrame(confusions).to_csv(OUT / "frontier_judge_confusion.csv", index=False)

    show = df[["evaluator", "n_scored", "n_judge_blocked", "accuracy", "precision",
               "recall", "fpr", "fpr_if_blocked_as_safe"]].copy()
    for c in ("accuracy", "precision", "recall", "fpr", "fpr_if_blocked_as_safe"):
        show[c] = (show[c] * 100).round(1)
    print(show.to_string(index=False))
    print(f"\nwrote {OUT/'frontier_judge_calibration.csv'}")


if __name__ == "__main__":
    main()
