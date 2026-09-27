# Seed-42 controlled grid: training recipe

All 15 {LoRA, QLoRA, FFT} x 5-family cells were retrained under one recipe. The paper's
equivalence tests (TOST) and the MixAT comparison use this grid.

## Framework

HuggingFace `transformers` + `trl` (`SFTTrainer`) + `peft`, with no Unsloth, in every cell.
Seed-42 runs used torch 2.8.0 (cu128), transformers 4.56.2, trl 0.23.0, peft 0.17.1,
datasets 3.6.0 and bitsandbytes 0.47.0.

## Hyperparameters

| Field | LoRA / QLoRA | FFT (<=9B: llama31, gemma2, qwen3) | FFT (14B: phi4, qwen25) |
|---|---|---|---|
| Seed / data_seed | 42 / 42 | 42 / 42 | 42 / 42 |
| Optimizer | adamw_8bit | adamw_8bit | paged_adamw_8bit |
| Learning rate | 2e-4 | 2e-5 | 2e-5 |
| Effective batch | 8 | 8 | 8 |
| Per-device batch x grad-accum | 2 x 4 (llama31, gemma2, qwen3); 1 x 8 (phi4, qwen25) | 2 x 4 (llama31, gemma2); 1 x 8 (qwen3) | 1 x 8 |
| Epochs | 1 | 1 | 1 |
| LR scheduler / warmup / weight decay | linear / 10 steps / 0.01 | linear / 10 steps / 0.01 | linear / 10 steps / 0.01 |
| LoRA r / alpha / dropout / bias | 16 / 16 / 0.0 / none | n/a | n/a |
| LoRA targets | q,k,v,o_proj + gate,up,down_proj (Phi-4: q,k,v,o_proj + gate_up_proj, down_proj) | n/a | n/a |
| QLoRA quantization | NF4, bf16 compute, double quantization | n/a | n/a |
| Precision | bf16 | bf16 | bf16 |
| Max sequence length | 2048 | 2048 | 2048 |
| Gradient checkpointing | yes | yes | yes |
| GPUs | 1 | 1 | 1 (A100-80GB) |

The paged variant on the two 14B FFT cells keeps the 8-bit AdamW optimizer state in
CPU-pageable memory so the full fine-tune fits on a single 80GB GPU. The update rule is the
same 8-bit AdamW as in every other cell. Gradient checkpointing only recomputes activations and
does not change the update.

## Data

One epoch on Alpaca-cleaned (`yahma/alpaca-cleaned`, 51,760 examples; 6,470 optimizer steps at
effective batch 8), loaded with `datasets.load_from_disk` from `$PROJECT_ROOT/data/alpaca_cleaned_dataset`.
Each example is rendered with the Alpaca prompt template, followed by the tokenizer EOS token,
tokenized per example (no packing) and truncated at 2048 tokens.

## Files

| File | Role |
|---|---|
| `train_canonical.py` | LoRA / QLoRA training (`--family {llama31,gemma2,qwen3,phi4,qwen25,llama31_mixat} --method {lora,qlora}`) |
| `merge_adapter.py` | Merge a trained adapter into its bf16 base (`merge_and_unload`) for inference |
| `train_fft_canonical.py` | FFT for llama31, gemma2, qwen3 (and the MixAT llama31 arm) |
| `train_fft_14b_canonical.py` | FFT for phi4 and qwen25 (paged_adamw_8bit, single GPU) |
| `train_fft_14b_phi4.slurm`, `train_fft_14b_qwen25.slurm` | SLURM jobs for the two 14B FFT cells |
| `submit_train.sh` | Submits one PEFT (train + merge) or <=9B FFT job |
| `validate_recipe.py` | Checks a trained cell's `training_args.bin` / `adapter_config.json` against the recipe |
| `provenance.csv` | Per-cell recipe check: seed, LoRA dropout, optimizer, producing script |
| `mixat/merge_mixat_base.py`, `mixat/merge_mixat.slurm` | Merge the released MixAT adapter (`INSAIT-Institute/Llama3.1-8B-MixAT`, LoRA r=64, alpha=16) into Llama-3.1-8B-Instruct to create the defended base |

In `provenance.csv`, seed and optimizer were read back from each cell's saved
`training_args.bin`. LoRA dropout 0.0 is asserted by `train_canonical.py` before training starts.

## MixAT arm (Llama-3.1)

1. `mixat/merge_mixat.slurm` (FAMILY=llama31) writes `$PROJECT_ROOT/models/mixat/llama31_mixat_base`.
2. `./submit_train.sh llama31_mixat lora` and `./submit_train.sh llama31_mixat fft` train on the
   defended base with exactly the recipe above. Only the starting weights differ from the
   undefended llama31 cells.

The scripts also contain a `qwen25_mixat` option. The paper reports only the Llama-3.1 MixAT arm.

## Output layout

- PEFT adapters: `$PROJECT_ROOT/models/phase5/{family}_{method}_seed42`
- Merged / full models: `$PROJECT_ROOT/models/phase5/{family}_{method}_seed42_merged`
- Trainer checkpoints: `$PROJECT_ROOT/outputs/finetune/phase5/{family}_{method}_seed42`

(`phase5` is the directory name the seed-42 grid scripts use for their outputs.)
