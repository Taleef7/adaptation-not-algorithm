"""
Evaluator calibration metrics under Regime A.

Regime A: ground truth = "the value both annotators agreed on" for the subset
where annotators agreed. Disagreements are dropped from evaluator-calibration
metrics. Automated evaluator outputs NEVER define ground truth.

Inputs:
  data/human_annotation/asr_500_labels.csv  (annotator_A, annotator_B, evaluator predictions)
  data/human_annotation/orr_250_labels.csv

Outputs:
  artifacts/revision/regime_a_calibration.csv       (per-evaluator metrics + CIs)
  artifacts/revision/regime_a_confusion_{evaluator}_{task}.csv
  artifacts/revision/regime_a_dropped_samples.csv   (audit trail)

Notes:
  - annotator_B's label for ASR sample asr_234 is 4 in the raw file. This is a
    data-entry error and should be 1. The script applies this correction
    explicitly and records it in regime_a_dropped_samples.csv (correction, not
    drop). The sample is a disagreement either way (annotator_A = 0).
  - Wilson 95% confidence intervals are reported for accuracy/precision/recall.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ANN = REPO / "data" / "human_annotation"
ART = REPO / "artifacts"
OUT = ART / "revision"
OUT.mkdir(parents=True, exist_ok=True)

EVALUATORS = {
    "HB-Cls": "hbcls_label",
    "GPT-4o-mini": "gpt4omini_label",
    "LlamaGuard-3": "llamaguard3_label",
}


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_task(task: str) -> tuple[pd.DataFrame, list[dict]]:
    """Return a frame with annotator_A_label, annotator_B_label, evaluator preds.

    `corrections` records every label adjustment applied to raw data.
    """
    if task == "asr":
        lab = pd.read_csv(ANN / "asr_500_labels.csv")
    elif task == "orr":
        lab = pd.read_csv(ANN / "orr_250_labels.csv")
    else:
        raise ValueError(task)

    corrections: list[dict] = []
    if task == "asr":
        bad_mask = lab["annotator_B"] == 4
        for sid in lab.loc[bad_mask, "sample_id"].tolist():
            corrections.append({
                "task": task,
                "sample_id": sid,
                "annotator": "annotator_B",
                "raw_label": 4,
                "corrected_label": 1,
                "reason": "data-entry error (asr_234)",
            })
        lab.loc[bad_mask, "annotator_B"] = 1

    for col in ("annotator_A", "annotator_B"):
        invalid = lab.loc[~lab[col].isin([0, 1])]
        assert invalid.empty, f"unexpected {col} labels: {invalid[col].unique()}"

    merged = lab.rename(columns={"annotator_A": "annotator_A_label",
                                 "annotator_B": "annotator_B_label"})[
        ["sample_id", "annotator_B_label", "annotator_A_label",
         "gpt4omini_label", "hbcls_label", "llamaguard3_label",
         "attack", "benchmark", "model_family", "config"]].copy()
    merged["agree"] = (merged["annotator_A_label"] == merged["annotator_B_label"]).astype(int)
    return merged, corrections


def compute_metrics(df: pd.DataFrame, eval_col: str) -> dict:
    pred = df[eval_col].astype(int).to_numpy()
    gt = df["ground_truth"].astype(int).to_numpy()
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    n = tp + fp + fn + tn
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
    acc_lo, acc_hi = wilson_ci(tp + tn, n)
    prec_lo, prec_hi = wilson_ci(tp, tp + fp) if (tp + fp) > 0 else (float("nan"), float("nan"))
    rec_lo, rec_hi = wilson_ci(tp, tp + fn) if (tp + fn) > 0 else (float("nan"), float("nan"))
    return {
        "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": acc, "precision": prec, "recall": rec, "fpr": fpr, "fnr": fnr,
        "accuracy_ci_low": acc_lo, "accuracy_ci_high": acc_hi,
        "precision_ci_low": prec_lo, "precision_ci_high": prec_hi,
        "recall_ci_low": rec_lo, "recall_ci_high": rec_hi,
    }


def run_task(task: str, results: list[dict], correction_rows: list[dict]) -> None:
    df, corrections = load_task(task)
    correction_rows.extend(corrections)

    n_total = len(df)
    df_agree = df[df["agree"] == 1].copy()
    assert (df_agree["annotator_A_label"] == df_agree["annotator_B_label"]).all(), (
        "Regime A invariant violated: annotator labels differ on agreement subset"
    )
    df_agree["ground_truth"] = df_agree["annotator_A_label"]
    n_agree = len(df_agree)
    n_positive = int(df_agree["ground_truth"].sum())

    print(f"[{task}] N_total={n_total}, N_agree={n_agree} ({n_agree/n_total*100:.1f}%), "
          f"positive_rate={n_positive/n_agree*100:.1f}%")

    for ev_label, ev_col in EVALUATORS.items():
        m = compute_metrics(df_agree, ev_col)
        row = {"task": task, "evaluator": ev_label, "n_total": n_total,
               "n_agree": n_agree, "n_positive": n_positive, **m}
        results.append(row)
        cm = pd.DataFrame(
            [[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
            index=["actual_negative", "actual_positive"],
            columns=["pred_negative", "pred_positive"],
        )
        slug = ev_label.lower().replace("-", "").replace(" ", "")
        cm.to_csv(OUT / f"regime_a_confusion_{slug}_{task}.csv")


def main() -> None:
    results: list[dict] = []
    corrections: list[dict] = []
    for task in ("asr", "orr"):
        run_task(task, results, corrections)

    cols = [
        "task", "evaluator", "n_total", "n_agree", "n_positive",
        "tp", "fp", "fn", "tn",
        "accuracy", "accuracy_ci_low", "accuracy_ci_high",
        "precision", "precision_ci_low", "precision_ci_high",
        "recall", "recall_ci_low", "recall_ci_high",
        "fpr", "fnr",
    ]
    out_df = pd.DataFrame(results)[cols]
    out_path = OUT / "regime_a_calibration.csv"
    out_df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\nwrote {out_path}")

    corr_path = OUT / "regime_a_dropped_samples.csv"
    if corrections:
        pd.DataFrame(corrections).to_csv(corr_path, index=False)
    else:
        pd.DataFrame(columns=["task", "sample_id", "annotator", "raw_label",
                              "corrected_label", "reason"]).to_csv(corr_path, index=False)
    print(f"wrote {corr_path}")

    print("\n=== Summary table ===")
    show = out_df.copy()
    for c in ["accuracy", "precision", "recall", "fpr", "fnr"]:
        show[c] = (show[c] * 100).round(1)
    print(show[["task", "evaluator", "n_agree", "n_positive",
                "accuracy", "precision", "recall", "fpr", "fnr"]].to_string(index=False))


if __name__ == "__main__":
    main()
