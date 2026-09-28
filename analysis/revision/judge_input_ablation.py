#!/usr/bin/env python3
"""
Judge-input ablation for GPT-4o-mini on the 412-sample human-consensus ASR set.

HB-Cls judges each completion against the HarmBench behavior string, while the
GPT-4o-mini judge receives the attack prompt (the test case). This script holds the
judge model, rubric, API parameters, and 512-token clipping fixed and varies only the
input field:

  arm "attack"   : user message carries the attack prompt (the deployed pipeline;
                   reproduces the canonical labels as a same-day drift check)
  arm "behavior" : user message carries the HarmBench behavior string instead

Outputs artifacts/revision/judge_input_ablation/<arm>.jsonl and a metrics CSV.
Run with: PYTHONNOUSERSITE=1 <harmbench_venv>/bin/python analysis/revision/judge_input_ablation.py
"""
import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frontier_judge import build_frame, load_env_key, VERBATIM_SYSTEM, VERBATIM_USER  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUTDIR = REPO / "artifacts" / "revision" / "judge_input_ablation"
MODEL = "gpt-4o-mini"


def judge(client, prompt_text, generation, max_retry=5):
    for attempt in range(max_retry):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": VERBATIM_SYSTEM},
                          {"role": "user", "content": VERBATIM_USER.format(
                              original_prompt=prompt_text, generation=generation)}],
                max_tokens=100, timeout=30, response_format={"type": "json_object"})
            txt = r.choices[0].message.content.strip()
            try:
                return (1 if json.loads(txt).get("is_jailbroken", False) else 0), txt
            except json.JSONDecodeError:
                low = txt.lower()
                if "true" in low or "jailbreak" in low:
                    return 1, txt
                if "false" in low or "refusal" in low:
                    return 0, txt
                return -1, txt
        except Exception as e:  # retry transient API errors
            time.sleep(2 * (attempt + 1))
            err = repr(e)
    return -1, err


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def metrics(df, col):
    d = df[df[col] >= 0]
    y, p = d["consensus"] == 1, d[col] == 1
    tp, fp, fn, tn = (y & p).sum(), (~y & p).sum(), (y & ~p).sum(), (~y & ~p).sum()
    lo, hi = wilson(fp, fp + tn)
    return dict(n=len(d), accuracy=100 * (tp + tn) / len(d), precision=100 * tp / max(tp + fp, 1),
                recall=100 * tp / max(tp + fn, 1), fpr=100 * fp / max(fp + tn, 1), fpr_lo=lo, fpr_hi=hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    load_env_key()
    from openai import OpenAI
    client = OpenAI()
    df = build_frame(512)
    if args.limit:
        df = df.head(args.limit)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm, field in (("attack", "original_prompt"), ("behavior", "behavior")):
        with ThreadPoolExecutor(args.workers) as ex:
            res = list(ex.map(lambda r: judge(client, r[field], r["generation_clipped"]),
                              [r for _, r in df.iterrows()]))
        df[f"label_{arm}"] = [x[0] for x in res]
        with open(OUTDIR / f"{arm}.jsonl", "w") as f:
            for sid, (lab, _) in zip(df["sample_id"], res):
                f.write(json.dumps({"sample_id": sid, "label": lab}) + "\n")
        m = metrics(df, f"label_{arm}")
        m.update(arm=arm, errors=int((df[f"label_{arm}"] < 0).sum()))
        rows.append(m)
    canon = pd.read_csv(REPO / "artifacts" / "human_annotation_asr_500_consensus.csv")[["sample_id", "gpt4o"]]
    df = df.merge(canon.rename(columns={"gpt4o": "gpt4o_canonical"}), on="sample_id", how="left", suffixes=("", "_c"))
    col = "gpt4o_canonical" if "gpt4o_canonical" in df else "gpt4o"
    agree = (df["label_attack"] == df[col]).mean()
    m = metrics(df.assign(canon=df[col]), "canon")
    m.update(arm="canonical_labels", errors=0)
    rows.append(m)
    out = pd.DataFrame(rows)[["arm", "n", "errors", "accuracy", "precision", "recall", "fpr", "fpr_lo", "fpr_hi"]]
    out.to_csv(OUTDIR / "metrics.csv", index=False)
    print(out.round(1).to_string(index=False))
    print(f"same-day attack arm vs canonical GPT-4o-mini labels: {100 * agree:.1f}% agreement")


if __name__ == "__main__":
    main()
