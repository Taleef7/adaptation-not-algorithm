---
license: cc-by-nc-4.0
base_model: Qwen/Qwen3-4B-Instruct-2507
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

# Qwen3-4B-Instruct-2507-Alpaca-LoRA

Qwen3-4B-Instruct-2507 fine-tuned with LoRA (r=16, alpha=16, all attention and MLP projections; merged into the base weights) on Alpaca-cleaned (51,760 examples, one epoch).
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
| PAIR | HarmBench-400 | 10.0 | 19.5 |
| PAIR | JailbreakBench-100 | 3.0 | 5.0 |
| DeepInception | HarmBench-400 | 2.5 | 7.8 |
| DeepInception | JailbreakBench-100 | 3.0 | 13.0 |
| ArtPrompt | HarmBench-400 | 1.2 | 6.8 |
| ArtPrompt | JailbreakBench-100 | 0.0 | 1.0 |
| AutoDAN | HarmBench-400 | 0.5 | 6.2 |
| AutoDAN | JailbreakBench-100 | 0.0 | 3.0 |

## License

The base model is released under Apache 2.0. The fine-tuning data derive from Stanford Alpaca (CC BY-NC 4.0), so this checkpoint is released for non-commercial research use only.

## Citation

```bibtex
@inproceedings{tamsal2026adaptation,
  title     = {Adaptation, Not Algorithm: {LoRA} and Full Fine-Tuning Show Comparable Black-Box Jailbreak Degradation in Five Open-Weight {LLMs}},
  author    = {Tamsal, Taleef and Rusert, Jonathan},
  booktitle = {Findings of the Association for Computational Linguistics: AACL-IJCNLP 2026},
  year      = {2026}
}
```
