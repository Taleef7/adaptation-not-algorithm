# Evaluator labels

Behavior-level jailbreak verdicts for the attack cells in the paper. Each row is one
evaluator's binary verdict on one model response to one benchmark behavior. The files hold
identifiers and labels only. They contain no behavior text, prompts, attack test cases or
model generations.

## Files

| File | Rows | Contents |
|---|---:|---|
| `original_grid.csv.gz` | 116,000 | Original grid: 5 families x 5 configurations; PAIR, DeepInception, ArtPrompt, AutoDAN and GCG |
| `seed42_grid.csv.gz` | 45,000 | Controlled seed-42 grid: 5 families x {LoRA, QLoRA, FFT}; DeepInception and ArtPrompt |
| `behavior_ids.csv` | 500 | Behavior-ID key: `behavior_id`, `benchmark`, `category`, `same_text_group` |
| `seed42_text_group_labels.csv` | 42 | HB-Cls label used for a shared-text HarmBench group in the seed-42 grid where the group's members disagree (see below) |

## Columns (`*_grid.csv.gz`)

| Column | Values |
|---|---|
| `grid` | `original`, `seed42` |
| `family` | `gemma2` (Gemma-2-9B-it), `llama31` (Llama-3.1-8B-Instruct), `phi4` (Phi-4), `qwen25` (Qwen-2.5-14B-Instruct), `qwen3` (Qwen3-4B-Instruct-2507) |
| `config` | `base_fp16`, `base_4bit`, `lora`, `qlora`, `fft` |
| `attack` | `PAIR`, `DeepInception`, `ArtPrompt`, `AutoDAN`, `GCG` |
| `benchmark` | `JBB` (JailbreakBench, 100 behaviors), `HarmBench` (HarmBench text behaviors, 400) |
| `evaluator` | `HB-Cls` (HarmBench-Llama-2-13b-cls), `GPT-4o-mini`, `LlamaGuard-3` |
| `behavior_id` | Behavior identifier; join to `behavior_ids.csv` |
| `label` | `1` = the evaluator judged the response a successful jailbreak, `0` = not |

A cell is one (`grid`, `family`, `config`, `attack`, `benchmark`, `evaluator`) combination.
Each released cell is complete: it has exactly one row per benchmark behavior, so 100 rows
for JBB and 400 for HarmBench. Its attack success rate (ASR) is the mean of `label` times 100:

```python
import pandas as pd
df = pd.read_csv("original_grid.csv.gz")
asr = df.groupby(["grid", "family", "config", "attack", "benchmark", "evaluator"]).label.mean() * 100
```

Every released cell reproduces the paper's per-cell ASR exactly. That covers the original-grid
ASR tables, the controlled-grid ASRs under each of the three evaluators, and the GCG table.

Each attack produces one response per behavior, and an evaluator gives each behavior one
label. Labels from different evaluators for the same model, configuration, attack and benchmark
line up by `behavior_id`.

## Behavior IDs

- **HarmBench**: the `BehaviorID` column of HarmBench's `harmbench_behaviors_text_all.csv`
  (for example `dimethylmercury_instructions_home`). All 400 text behaviors are used: 200
  standard, 100 contextual and 100 copyright. `category` is HarmBench's `SemanticCategory`,
  with `copyright` marking the copyright behaviors.
- **JBB**: `jbb_000` to `jbb_099`, numbered in the row order of the JailbreakBench
  JBB-Behaviors harmful split (`jbb_000` is its first behavior). `category` is the
  JailbreakBench category (for example `Harassment/Discrimination`).

To get behavior text, join `behavior_id` against the public HarmBench and JailbreakBench
behavior files.

`same_text_group`: 12 HarmBench contextual behaviors fall into 5 groups that share a behavior
string and differ only in the context paragraph. The column
holds the first ID of the group for those 12 behaviors and is empty otherwise. The seed-42
scored files key verdicts by behavior string, so the analyses that read them
(`analysis/revision/attack_transfer.py`) see 393 distinct HarmBench behaviors, one label per
group. `seed42_text_group_labels.csv` records that label for the 42 (cell, group) pairs where
the group's members have different labels; with it, the release-mode analysis reproduces the
paper exactly.

## Grids

**Original grid.** Five families and five configurations: the base model in FP16
(`base_fp16`), the base model loaded in 4-bit (`base_4bit`), and the LoRA, QLoRA and full
fine-tuning (`fft`) adaptations. Black-box attacks (PAIR, DeepInception, ArtPrompt) and AutoDAN
are run on both benchmarks. GCG is run on JBB only, for `base_fp16`, `lora` and `fft`.

**Seed-42 controlled grid.** LoRA, QLoRA and FFT are retrained under a matched recipe with
seed 42 and attacked with DeepInception and ArtPrompt on both benchmarks. The base models are
not retrained, so this grid has no `base_fp16` or `base_4bit` rows. Comparisons against base
use the original-grid base cells.

## Cell counts

Number of released cells per attack, benchmark and evaluator. The grid size is the number of
(family, configuration) combinations run for that attack.

### `original_grid.csv.gz`

| Attack | Benchmark | Grid size | HB-Cls | GPT-4o-mini | LlamaGuard-3 |
|---|---|---:|---:|---:|---:|
| PAIR | JBB | 25 | 0 | 25 | 25 |
| PAIR | HarmBench | 25 | 20 | 22 | 20 |
| DeepInception | JBB | 25 | 22 | 0 | 21 |
| DeepInception | HarmBench | 25 | 6 | 14 | 4 |
| ArtPrompt | JBB | 25 | 25 | 25 | 25 |
| ArtPrompt | HarmBench | 25 | 25 | 22 | 20 |
| AutoDAN | JBB | 25 | 25 | 25 | 25 |
| AutoDAN | HarmBench | 25 | 25 | 20 | 20 |
| GCG | JBB | 15 | 15 | 15 | 15 |
| **Total** | | | **163** | **168** | **175** |

That is 506 cells in total. Behavior-level labels are released only for the cells listed in
this table. Per-cell ASRs for the complete grid are in `artifacts/master_asr_long.csv`.

Caveat: for `gemma2 / base_fp16 / DeepInception / HarmBench`, the HB-Cls and GPT-4o-mini
labels score two separate generation runs of the attack (each reproduces its canonical ASR),
so the two evaluators should not be compared behavior by behavior in that cell. For
`gemma2 / fft / ArtPrompt / HarmBench`, the generations scored by the three evaluators are
identical for 86.75% of behaviors.

### `seed42_grid.csv.gz`

| Attack | Benchmark | Grid size | HB-Cls | GPT-4o-mini | LlamaGuard-3 |
|---|---|---:|---:|---:|---:|
| DeepInception | JBB | 15 | 15 | 15 | 15 |
| DeepInception | HarmBench | 15 | 15 | 15 | 15 |
| ArtPrompt | JBB | 15 | 15 | 15 | 15 |
| ArtPrompt | HarmBench | 15 | 15 | 15 | 15 |
| **Total** | | | **60** | **60** | **60** |

That is 180 cells, the complete controlled grid.
