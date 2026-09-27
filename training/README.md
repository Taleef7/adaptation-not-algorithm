# Training

Fine-tuning scripts and SLURM jobs for every trained cell reported in the paper. The folder has
three parts:

| Folder | Cells | Used in the paper for |
|---|---|---|
| `original_grid/` | 15 cells (5 families x {LoRA, QLoRA, FFT}), plus Llama-3.1 ablations: training seeds 0/123 (LoRA and FFT), LoRA rank r=4/8/32/64, attention-only / MLP-only LoRA, faithful SafeLoRA | Aggregate dASR estimates and per-family profiles (Claims 1-3), white-box AutoDAN multi-seed, rank and layer ablations, SafeLoRA baseline |
| `seed42_grid/` | The same 15 cells retrained under one uniform recipe (seed 42, LoRA dropout 0.0, 8-bit AdamW, HF throughout), plus the MixAT-defended Llama-3.1 LoRA and FFT cells | LoRA-vs-FFT equivalence tests (TOST), method-transfer analysis, MixAT baseline |
| `track2/` | Llama-3.1, Gemma-2, Qwen-3 x {LoRA, FFT} under four recipe perturbations: Dolly/1 epoch, Dolly step-matched (6,470 steps), Alpaca/3 epochs, Dolly/3 epochs | Recipe-robustness appendix ("A Second Dataset and Training Length") |

Per-cell recipes are in `original_grid/RECIPE.md` and `seed42_grid/RECIPE.md`. The original grid
is not recipe-uniform: seed, LoRA dropout, optimizer variant and framework differ across cells,
as the tables there show. The track-2 arms use the seed-42 recipe with only the dataset and/or
training length changed.

Families: gemma2 (`google/gemma-2-9b-it`), llama31 (`meta-llama/Llama-3.1-8B-Instruct`),
phi4 (`microsoft/phi-4`), qwen25 (`Qwen/Qwen2.5-14B-Instruct`), qwen3 (`Qwen/Qwen3-4B-Instruct-2507`).

## Training data

- Alpaca-cleaned: `yahma/alpaca-cleaned` (51,760 examples). Most scripts read a local copy:
  ```python
  from datasets import load_dataset
  load_dataset("yahma/alpaca-cleaned", split="train").save_to_disk("data/alpaca_cleaned_dataset")
  ```
  The Unsloth-based Llama-3.1 LoRA cell and the Llama-3.1 LoRA ablations load it directly from the Hub.
- Dolly: `databricks/databricks-dolly-15k` (15,011 examples). `track2/prep_dolly.py` converts it
  to the Alpaca schema (`context` -> `input`, `response` -> `output`) and saves
  `data/dolly15k_dataset`.

## Running

All paths are relative to the repository root. Set these before submitting:

```bash
export PROJECT_ROOT=/path/to/adaptation-not-algorithm   # holds models/, data/, outputs/, logs/
export VENV_DIR=/path/to/python/venv                     # seed-42 and track-2 submit scripts
export HF_HOME=/path/to/hf_cache                         # optional
export HF_TOKEN=...                                      # needed for gated bases (Llama, Gemma)
cd "$PROJECT_ROOT"
```

Submit the original-grid SLURM files from the repository root, e.g.
`sbatch training/original_grid/gemma2/slurm/run_lora_gemma2-9b.slurm`. Replace `YOUR_ACCOUNT`
and adjust the partition, QOS and module/conda lines for your cluster. The original-grid jobs
activate the conda environments they were run with (`thesis_env`, plus `gemma_fft_env` for the
Unsloth Gemma-2 FFT cell). The LoRA and QLoRA cells write adapters under `models/<family>/`,
which the scripts in `original_grid/merge/` merge into their bf16 bases for inference.

Seed-42 grid:

```bash
bash training/seed42_grid/submit_train.sh llama31 lora      # train + merge
bash training/seed42_grid/submit_train.sh gemma2 fft        # <=9B FFT
sbatch training/seed42_grid/train_fft_14b_phi4.slurm        # 14B FFT
```

Track 2:

```bash
python training/track2/prep_dolly.py
bash training/track2/submit_arm.sh llama31 lora dolly1ep    # also: dollystep, alpaca3ep
bash training/track2/submit_track2.sh llama31 lora          # Dolly / 3 epochs
```

Attack generation and scoring for the trained cells are in `../evaluation/`.
