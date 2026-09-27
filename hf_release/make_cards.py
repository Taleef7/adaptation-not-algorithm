"""Build Hugging Face model cards (README.md) for the released checkpoints.

Reads manifest.json and the paper's per-cell ASR tables, writes cards/<repo>/README.md
plus the license notice files each base-model license requires.
"""
import csv, json, os, sys
from collections import defaultdict

RESEARCH = os.environ.get("RESEARCH_REPO", ".")
HF_USER = os.environ.get("HF_USER", "taleef")
GITHUB = "https://github.com/Taleef7/adaptation-not-algorithm"
HERE = os.path.dirname(os.path.abspath(__file__))
FAM = {"llama31": "Llama-3.1-8B-Instruct", "gemma2": "Gemma-2-9B-IT", "phi4": "Phi-4 (14B)", "qwen3": "Qwen3-4B-Instruct-2507"}
METHOD = {"lora": "LoRA (r=16, alpha=16, all attention and MLP projections; merged into the base weights)",
          "fft": "full fine-tuning (all parameters)"}
ATTACKS = ["PAIR", "DeepInception", "ArtPrompt", "AutoDAN"]
DATASETS = ["HarmBench", "JBB"]


def primary_asr():
    t = defaultdict(dict)
    with open(os.path.join(RESEARCH, "artifacts/master_asr_long.csv")) as f:
        for r in csv.DictReader(f):
            if r["evaluator"] == "Llama-2-13b-cls":
                t[(r["model_family"], r["config"])][(r["attack"], r["dataset"])] = float(r["ASR"])
    return t


def controlled_asr():
    t = defaultdict(dict)
    with open(os.path.join(RESEARCH, "artifacts/revision/r6_controlled_cells_3judge.csv")) as f:
        for r in csv.DictReader(f):
            if r["judge"] == "HB-Cls":
                for cfg in ("base_fp16", "lora", "fft"):
                    t[(r["model_family"], cfg)][(r["attack"], r["dataset"])] = float(r[cfg])
    return t


def norm_ds(d):
    return "JBB" if d.lower().startswith("j") else "HarmBench"


def asr_table(m, prim, ctrl):
    src = ctrl if m["grid"] == "controlled" else prim
    base = src.get((m["family"], "base_fp16"), {})
    mine = src.get((m["family"], m["method"]), {})
    base = {(a, norm_ds(d)): v for (a, d), v in base.items()}
    mine = {(a, norm_ds(d)): v for (a, d), v in mine.items()}
    rows = ["| Attack | Benchmark | Base FP16 ASR (%) | This model ASR (%) |", "|---|---|---|---|"]
    for a in ATTACKS:
        for d in DATASETS:
            if (a, d) in mine and (a, d) in base:
                rows.append(f"| {a} | {'HarmBench-400' if d == 'HarmBench' else 'JailbreakBench-100'} | {base[(a, d)]:.1f} | {mine[(a, d)]:.1f} |")
    return "\n".join(rows)


def dtype_note(m):
    cfg = json.load(open(os.path.join(RESEARCH, m["src"], "config.json")))
    dt = cfg.get("torch_dtype") or cfg.get("dtype")
    if dt == "float32":
        return "\nWeights are stored in float32; all evaluations loaded them in bfloat16.\n"
    return ""


def ctrl_note(m):
    if m["grid"] != "controlled":
        return ""
    return ("\nFor this checkpoint, ArtPrompt prompts were formatted with the Alpaca template used in fine-tuning, "
            "whereas the base-model column uses the model's own chat template; compare ArtPrompt rows across "
            "checkpoints of the same grid rather than against the base (see Appendix B.5 of the paper).\n")


def license_block(m):
    if m["license"] == "llama3.1":
        return ("license: llama3.1", "Built with Llama. This model is a derivative of Llama 3.1 and is distributed under the "
                "[Llama 3.1 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE) "
                "and subject to the [Llama 3.1 Acceptable Use Policy](https://llama.meta.com/llama3_1/use-policy). "
                "A copy of the license is included in this repository (`LICENSE`), with the required `NOTICE`.")
    if m["license"] == "gemma":
        return ("license: gemma", "This model is a Model Derivative of Gemma and is provided under and subject to the "
                "[Gemma Terms of Use](https://ai.google.dev/gemma/terms), including the use restrictions in the "
                "[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy), which apply to every "
                "subsequent user. See `NOTICE`.")
    base_lic = "MIT" if m["family"] == "phi4" else "Apache 2.0"
    return ("license: cc-by-nc-4.0", f"The base model is released under {base_lic}. The fine-tuning data derive from "
            "Stanford Alpaca (CC BY-NC 4.0), so this checkpoint is released for non-commercial research use only.")


def notice(m):
    if m["license"] == "llama3.1":
        return "Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.\n"
    if m["license"] == "gemma":
        return "Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms\n"
    return None


def card(m, prim, ctrl):
    lic_meta, lic_text = license_block(m)
    grid = ("the controlled seed-42 grid used for the paper's LoRA-vs-FFT equivalence tests (seed 42, LoRA dropout 0.0, "
            "`adamw_8bit`, HuggingFace Transformers; evaluated on DeepInception and ArtPrompt)"
            if m["grid"] == "controlled" else
            "the primary 25-configuration grid used for the paper's aggregate and per-family results")
    return f"""---
{lic_meta}
base_model: {m['base']}
datasets:
- yahma/alpaca-cleaned
language:
- en
tags:
- safety
- jailbreak
- lora
- research
---

# {m['repo']}

{FAM[m['family']]} fine-tuned with {METHOD[m['method']]} on Alpaca-cleaned (51,760 examples, one epoch).
This is one of the evaluated checkpoints from **"Adaptation, Not Algorithm: LoRA and Full Fine-Tuning Show Comparable
Black-Box Jailbreak Degradation in Five Open-Weight LLMs"** (Tamsal and Rusert, Findings of AACL-IJCNLP 2026).
It comes from {grid}.

Code, attack configurations, evaluator verdicts, and analysis scripts: {GITHUB}

## Intended use and warning

This checkpoint exists to reproduce and extend a safety evaluation. Benign instruction tuning measurably weakens
the base model's jailbreak robustness, so **this model is less safe than its base model and must not be deployed**.
Use it only for research on fine-tuning safety.

## Training recipe

AdamW, one epoch on `yahma/alpaca-cleaned`, linear schedule with 10 warmup steps, weight decay 0.01, effective batch
size 8, maximum sequence length 2048, bfloat16. LoRA/QLoRA learning rate 2e-4; FFT learning rate 2e-5. LoRA adapters
are merged into the base weights. Per-cell seed, dropout, and optimizer details are in Appendix J of the paper and in
the repository.

## Evaluation (HB-Cls, attack success rate)

Scored with `cais/HarmBench-Llama-2-13b-cls` (HarmBench's classifier; copyright behaviors use HarmBench's hash check).
One configured attack run per behavior. Higher is less safe.

{asr_table(m, prim, ctrl)}
{ctrl_note(m)}{dtype_note(m)}
## License

{lic_text}

## Citation

```bibtex
@inproceedings{{tamsal2026adaptation,
  title     = {{Adaptation, Not Algorithm: {{LoRA}} and Full Fine-Tuning Show Comparable Black-Box Jailbreak Degradation in Five Open-Weight {{LLMs}}}},
  author    = {{Tamsal, Taleef and Rusert, Jonathan}},
  booktitle = {{Findings of the Association for Computational Linguistics: AACL-IJCNLP 2026}},
  year      = {{2026}}
}}
```
"""


def main():
    prim, ctrl = primary_asr(), controlled_asr()
    for m in json.load(open(os.path.join(HERE, "manifest.json"))):
        out = os.path.join(HERE, "cards", m["repo"])
        os.makedirs(out, exist_ok=True)
        open(os.path.join(out, "README.md"), "w").write(card(m, prim, ctrl))
        n = notice(m)
        if n:
            open(os.path.join(out, "NOTICE"), "w").write(n)
        print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
