---
license: gemma
base_model: google/gemma-2-9b-it
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

# gemma-2-9b-it-Alpaca-LoRA

Gemma-2-9B-IT fine-tuned with LoRA (r=16, alpha=16, all attention and MLP projections; merged into the base weights) on Alpaca-cleaned (51,760 examples, one epoch).
This is one of the evaluated checkpoints from **"Adaptation, Not Algorithm: LoRA and Full Fine-Tuning Show Comparable
Black-Box Jailbreak Degradation in Five Open-Weight LLMs"** (Tamsal and Rusert, Findings of AACL-IJCNLP 2026).
It comes from the primary 25-configuration grid used for the paper's aggregate and per-family results.

Code, attack configurations, evaluator verdicts, and analysis scripts: https://github.com/Taleef7/adaptation-not-algorithm

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

| Attack | Benchmark | Base FP16 ASR (%) | This model ASR (%) |
|---|---|---|---|
| PAIR | HarmBench-400 | 15.2 | 22.0 |
| PAIR | JailbreakBench-100 | 12.0 | 12.0 |
| DeepInception | HarmBench-400 | 11.0 | 17.0 |
| DeepInception | JailbreakBench-100 | 20.0 | 28.0 |
| ArtPrompt | HarmBench-400 | 13.2 | 16.8 |
| ArtPrompt | JailbreakBench-100 | 9.0 | 15.0 |
| AutoDAN | HarmBench-400 | 65.5 | 68.2 |
| AutoDAN | JailbreakBench-100 | 88.0 | 91.0 |

## License

This model is a Model Derivative of Gemma and is provided under and subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms), including the use restrictions in the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy), which apply to every subsequent user. See `NOTICE`.

## Citation

```bibtex
@inproceedings{tamsal2026adaptation,
  title     = {Adaptation, Not Algorithm: {LoRA} and Full Fine-Tuning Show Comparable Black-Box Jailbreak Degradation in Five Open-Weight {LLMs}},
  author    = {Tamsal, Taleef and Rusert, Jonathan},
  booktitle = {Findings of the Association for Computational Linguistics: AACL-IJCNLP 2026},
  year      = {2026}
}
```
