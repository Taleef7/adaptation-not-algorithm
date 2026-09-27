---
license: llama3.1
base_model: meta-llama/Llama-3.1-8B-Instruct
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

# Llama-3.1-8B-Instruct-Alpaca-FFT-seed42

Llama-3.1-8B-Instruct fine-tuned with full fine-tuning (all parameters) on Alpaca-cleaned (51,760 examples, one epoch).
This is one of the evaluated checkpoints from **"Adaptation, Not Algorithm: LoRA and Full Fine-Tuning Show Comparable
Black-Box Jailbreak Degradation in Five Open-Weight LLMs"** (Tamsal and Rusert, Findings of AACL-IJCNLP 2026).
It comes from the controlled seed-42 grid used for the paper's LoRA-vs-FFT equivalence tests (seed 42, LoRA dropout 0.0, `adamw_8bit`, HuggingFace Transformers; evaluated on DeepInception and ArtPrompt).

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
| DeepInception | HarmBench-400 | 16.8 | 21.2 |
| DeepInception | JailbreakBench-100 | 33.0 | 32.0 |
| ArtPrompt | HarmBench-400 | 13.0 | 16.5 |
| ArtPrompt | JailbreakBench-100 | 0.0 | 10.0 |

For this checkpoint, ArtPrompt prompts were formatted with the Alpaca template used in fine-tuning, whereas the base-model column uses the model's own chat template; compare ArtPrompt rows across checkpoints of the same grid rather than against the base (see Appendix B.5 of the paper).

## License

Built with Llama. This model is a derivative of Llama 3.1 and is distributed under the [Llama 3.1 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE) and subject to the [Llama 3.1 Acceptable Use Policy](https://llama.meta.com/llama3_1/use-policy). A copy of the license is included in this repository (`LICENSE`), with the required `NOTICE`.

## Citation

```bibtex
@inproceedings{tamsal2026adaptation,
  title     = {Adaptation, Not Algorithm: {LoRA} and Full Fine-Tuning Show Comparable Black-Box Jailbreak Degradation in Five Open-Weight {LLMs}},
  author    = {Tamsal, Taleef and Rusert, Jonathan},
  booktitle = {Findings of the Association for Computational Linguistics: AACL-IJCNLP 2026},
  year      = {2026}
}
```
