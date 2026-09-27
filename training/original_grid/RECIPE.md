# Original grid: per-cell training recipe

The original grid is the 15 trained cells (5 families x {LoRA, QLoRA, FFT}) behind the
aggregate dASR estimates and per-family profiles. The cells were not trained under one uniform
recipe: seed, LoRA dropout, optimizer variant and framework vary between them. The seed-42
controlled grid (`../seed42_grid/`) retrains all 15 cells under a single recipe for the
equivalence tests.

Sources: `config_audit.csv` (values read from each cell's saved `training_args.bin` and
`adapter_config.json` by `config_audit.py`), plus the training scripts in this folder where a
value is not stored in those files. `NO_ADAPTER_CFG` in the CSV means the retained checkpoint
directory had no `adapter_config.json`. For those cells the dropout below comes from the
training script.

Shared by all 15 cells: one epoch on Alpaca-cleaned (`yahma/alpaca-cleaned`, 51,760 examples),
effective batch size 8, max sequence length 2048, bf16. LoRA/QLoRA use r=16, alpha=16,
bias="none", and target all attention and MLP projections (`q,k,v,o_proj` + `gate,up,down_proj`;
Phi-4 uses the fused `gate_up_proj`). QLoRA uses a 4-bit NF4 base with bf16 compute and double
quantization.

## Per-cell recipe

| Family | Method | Framework | Seed | LoRA dropout | Optimizer | LR | Per-device batch x grad-accum (GPUs) | Scheduler / warmup / weight decay |
|---|---|---|---|---|---|---|---|---|
| gemma2 | LoRA | HF transformers + PEFT + TRL | 42 | 0.05 | adamw_torch | 2e-4 | 2 x 4 (1) | linear / 10 / 0.01 |
| gemma2 | QLoRA | HF | 42 | 0.05 | paged_adamw_8bit | 2e-4 | 2 x 4 (1) | linear / 10 / 0.01 |
| gemma2 | FFT | Unsloth | 42 | n/a | adamw_8bit | 2e-5 | 2 x 4 (1) | linear / 10 / 0.01 |
| llama31 | LoRA | Unsloth | 3407 | 0.0 (script) | adamw_8bit | 2e-4 | 2 x 4 (1) | linear / 10 / 0.01 |
| llama31 | QLoRA | HF | 42 | 0.0 | paged_adamw_8bit | 2e-4 | 2 x 4 (1) | linear / 10 / 0.01 |
| llama31 | FFT | HF | 42 | n/a | adamw_torch | 2e-5 | 2 x 4 (1) | linear / 10 / 0.01 |
| phi4 | LoRA | HF | 3407 | 0.05 (script) | adamw_8bit | 2e-4 | 1 x 8 (1) | linear / 10 / 0.01 |
| phi4 | QLoRA | HF | 3407 | 0.0 | adamw_8bit | 2e-4 | 1 x 8 (1) | linear / 10 / 0.01 |
| phi4 | FFT | HF Trainer + FSDP | 42 (Trainer default) | n/a | adamw_torch | 2e-5 | 2 x 1 (4) | cosine / 100 / 0.0 |
| qwen25 | LoRA | HF | 3407 | 0.05 (script) | adamw_8bit | 2e-4 | 1 x 8 (1) | linear / 10 / 0.01 |
| qwen25 | QLoRA | HF | 3407 | 0.0 | adamw_8bit | 2e-4 | 1 x 8 (1) | linear / 10 / 0.01 |
| qwen25 | FFT | HF Trainer + FSDP | 42 (Trainer default) | n/a | adamw_torch | 2e-5 | 2 x 1 (4) | cosine / 100 / 0.0 |
| qwen3 | LoRA | HF | 3407 | 0.05 (script) | adamw_8bit | 2e-4 | 2 x 4 (1) | linear / 10 / 0.01 |
| qwen3 | QLoRA | HF | 42 | 0.0 | paged_adamw_8bit | 2e-4 | 2 x 4 (1) | linear / 10 / 0.01 |
| qwen3 | FFT | HF | 42 | n/a | adamw_8bit | 2e-5 | 1 x 8 (1) | linear / 10 / 0.01 |

Other per-cell differences visible in the scripts:

- Prompt format. Every cell except the two 14B FFT cells formats examples with the Alpaca
  template plus the tokenizer EOS token. The phi4 and qwen25 FFT scripts use plain
  `instruction\ninput\noutput` text, pad to 2048 tokens, and train with the HF `Trainer` and
  `DataCollatorForLanguageModeling`.
- Packing. llama31 LoRA (Unsloth) uses `packing=True` and loads `yahma/alpaca-cleaned` from the
  Hub. All other cells tokenize per example without packing, from a local copy of the dataset
  saved with `datasets.save_to_disk` at `data/alpaca_cleaned_dataset`.
- Gradient clipping. qwen3 FFT sets `max_grad_norm=0.3`. All other cells use the HF default (1.0).
- Gradient checkpointing (recomputes activations only) is enabled in gemma2 FFT, llama31 LoRA
  (Unsloth), phi4 LoRA/QLoRA/FFT, qwen25 LoRA/QLoRA/FFT, and qwen3 FFT.

## Cell -> script -> SLURM job

| Cell | Training script | SLURM job | Merge for inference |
|---|---|---|---|
| gemma2 LoRA | `gemma2/finetune_lora_gemma2-9b.py` | `gemma2/slurm/run_lora_gemma2-9b.slurm` | `merge/merge_peft_models.py` |
| gemma2 QLoRA | `gemma2/finetune_qlora_gemma2-9b.py` | `gemma2/slurm/run_qlora_gemma2-9b.slurm` | `merge/merge_peft_models.py` |
| gemma2 FFT | `gemma2/finetune_fft_unsloth_gemma2-9b.py` | `gemma2/slurm/run_fft_gemma2-9b.slurm` | n/a |
| llama31 LoRA | `llama31/finetune_lora_fp16_llama3.1.py` | `llama31/slurm/run_lora_fp16.slurm` | `merge/merge_peft_models.py` |
| llama31 QLoRA | `llama31/finetune_qlora_llama3.1.py` | `llama31/slurm/run_qlora_3_1.slurm` | `merge/merge_peft_models.py` |
| llama31 FFT | `llama31/finetune_fft_hf_llama3.1.py` | `llama31/slurm/run_fft_hf.slurm` | n/a |
| phi4 LoRA | `phi4/finetune_lora_phi-4.py` | `phi4/slurm/run_lora_phi-4.slurm` | `merge/merge_phi4_adapters.py` (`merge/slurm/merge_phi4_adapters.slurm`) |
| phi4 QLoRA | `phi4/finetune_qlora_phi-4.py` | `phi4/slurm/run_qlora_phi-4.slurm` | `merge/merge_phi4_adapters.py` |
| phi4 FFT | `phi4/finetune_fft_4gpu_fsdp_phi-4.py` | `phi4/slurm/run_fft_4gpu_fsdp_phi-4.slurm` | n/a |
| qwen25 LoRA | `qwen25/finetune_lora_qwen2.5-14b.py` | `qwen25/slurm/run_lora_qwen2.5-14b.slurm` | `qwen25/merge_adapters.py` (`qwen25/slurm/merge_adapters.slurm`) |
| qwen25 QLoRA | `qwen25/finetune_qlora_qwen2.5-14b.py` | `qwen25/slurm/run_qlora_qwen2.5-14b.slurm` | `qwen25/merge_adapters.py` |
| qwen25 FFT | `qwen25/finetune_fft_4gpu_fsdp_qwen2.5-14b.py` | `qwen25/slurm/run_fft_4gpu_fsdp_qwen2.5-14b.slurm` | n/a |
| qwen3 LoRA | `qwen3/finetune_lora_qwen3-4b.py` | `qwen3/slurm/run_lora_qwen3-4b.slurm` | `merge/merge_adapters_for_harmbench.py --model qwen3` (`merge/slurm/merge_adapters_for_harmbench.slurm`) |
| qwen3 QLoRA | `qwen3/finetune_qlora_qwen3-4b.py` | `qwen3/slurm/run_qlora_qwen3-4b.slurm` | `merge/merge_adapters_for_harmbench.py --model qwen3` |
| qwen3 FFT | `qwen3/finetune_fft_qwen3-4b.py` | `qwen3/slurm/run_fft_qwen3-4b.slurm` | n/a |

Base checkpoints are fetched with each family's `download_base_model.py` (gemma-2-9b-it,
Llama-3.1-8B-Instruct, phi-4, Qwen2.5-14B-Instruct, Qwen3-4B-Instruct-2507).

## Ablations (Llama-3.1 only)

All ablations use the llama31 LoRA recipe above unless noted.

| Ablation | Scripts | SLURM job |
|---|---|---|
| Training seeds 0 and 123, LoRA | `ablations/multi_seed/finetune_lora_fp16_llama3.1_seed{0,123}.py` | `ablations/multi_seed/slurm/run_lora_seed{0,123}.slurm` |
| Training seeds 0 and 123, FFT | `ablations/multi_seed/finetune_fft_hf_llama3.1_seed{0,123}.py` | `ablations/multi_seed/slurm/run_fft_seed{0,123}.slurm` |
| Seed-cell adapter merge | `ablations/multi_seed/multi_seed_pipeline.py --step merge` | run after the seed jobs |
| LoRA rank r=4/8/32/64 (alpha = r) | `ablations/rank/finetune_lora_r{4,8,32,64}.py` | `ablations/rank/slurm/run_rank_ablation.slurm` |
| Attention-only / MLP-only LoRA | `ablations/layer/finetune_lora_{attn_only,mlp_only}.py` | `ablations/layer/slurm/run_layer_specific.slurm` |
| Rank / layer adapter merge | inline PEFT `merge_and_unload` | `ablations/merge_rank_layer.slurm` |
| SafeLoRA (faithful), applied to the main, seed-0 and seed-123 LoRA adapters | `ablations/safelora/safe_lora_faithful.py` | `ablations/safelora/slurm/submit_safe_lora_faithful_broader.sh` -> `safe_lora_faithful_pipeline.slurm` |

Ablation recipe notes:

- Seed-0/123 LoRA uses HF PEFT + TRL (`SFTConfig`) instead of Unsloth. It keeps the same data
  source (`yahma/alpaca-cleaned`), template, `packing=True`, r=16, alpha=16, dropout 0.0,
  lr 2e-4 and batch 2 x 4, with `paged_adamw_8bit` in place of `adamw_8bit`.
- Seed-0/123 FFT is the llama31 FFT script with only the seed changed. The seed-0 run also
  enables gradient checkpointing.
- The rank and layer ablations are Unsloth runs with seed 3407, like the llama31 LoRA cell. The
  rank scripts set alpha equal to r. The layer scripts keep r=16, alpha=16 and restrict
  `target_modules` to `q,k,v,o_proj` or `gate,up,down_proj`.
- SafeLoRA derives the alignment direction from `meta-llama/Llama-3.1-8B` (unaligned) and
  Llama-3.1-8B-Instruct (aligned) and projects the top 10 of the 224 candidate LoRA matrices
  (`--select_layers_type number --num_proj_layers 10`). The pipeline job also runs the AutoDAN
  generation and scoring steps through the HarmBench evaluation toolkit.

## Running the audit

```bash
export PROJECT_ROOT=/path/to/repo   # directory containing models/ and outputs/
python training/original_grid/config_audit.py
```
