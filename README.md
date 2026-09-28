# Adaptation, Not Algorithm

Code, per-cell result tables, behavior-level evaluator labels, human annotation labels, and analysis
scripts for:

> **Adaptation, Not Algorithm: LoRA and Full Fine-Tuning Show Comparable Black-Box Jailbreak
> Degradation in Five Open-Weight LLMs**
> Taleef Tamsal and Jonathan Rusert.
> *Findings of the Association for Computational Linguistics: AACL-IJCNLP 2026.*

We evaluate 25 configurations (Gemma-2-9B-IT, Llama-3.1-8B-Instruct, Phi-4, Qwen2.5-14B-Instruct,
Qwen3-4B-Instruct-2507, each as Base FP16, Base 4-bit, LoRA, QLoRA, and full fine-tuning on
Alpaca-cleaned) against black-box (PAIR, DeepInception, ArtPrompt), white-box (AutoDAN, a GCG probe),
and multi-turn (ActorAttack) jailbreaks on HarmBench-400 and JailbreakBench-100, scored by three
automated evaluators (HB-Cls, GPT-4o-mini, LlamaGuard-3) that are calibrated against 750 doubly
annotated human samples.

**Claim 1 (black-box degradation).** LoRA, QLoRA, and FFT shift black-box ASR in the same direction by
comparable amounts (HB-Cls, 30 paired cells each: +4.0, +3.4, and +3.4 pp). After Holm correction,
which method is significant depends on the evaluator. On a seed-controlled re-training of all 15
trained cells (seed 42, one recipe), LoRA and FFT differ by +1.60 pp (90% CI [-0.88, +4.08]) and meet
the pre-specified ±5 pp equivalence bound (TOST p = 0.014). The result also holds under GPT-4o-mini
(p = 0.006) and LlamaGuard-3 (p = 0.018). QLoRA does not meet the bound (-1.23 pp, 90% CI
[-6.05, +3.60], p = 0.096).
**Claim 2 (black-box / white-box divergence).** AutoDAN moves the other way. On Llama-3.1, every
training method lowers AutoDAN ASR, and the LoRA effect varies strongly with the training seed
(-30.0 ± 24.1 pp over three seeds on HarmBench). Under a GCG probe, fine-tuned checkpoints are easier
targets than their bases in four of five families.
**Claim 3 (evaluator choice).** Against the 412 human-consensus ASR samples (a set deliberately enriched for evaluator disagreement), HB-Cls has precision 81.2%
and a false-positive rate of 8.6%. GPT-4o-mini's false-positive rate is 45.7% and LlamaGuard-3's is
40.6%. Evaluator choice changes absolute ASR, which effects reach significance, and family rankings.

## Repository map

| Path | Contents |
|---|---|
| `artifacts/` | Per-cell aggregate tables. `master_asr_long.csv` is the canonical per-cell ASR table (25 configurations x 4 attacks x 2 benchmarks x 3 evaluators) behind Claim 1/2 and the appendix matrices; `peft_deltas.csv`, `master_orr_long.csv`, OR-Bench and track-2 tables, `actorattack_scoreboard.csv`, `mixat_asr.csv`, `cross_seed_results.csv`, `ablations/` (rank, layer, multi-seed). |
| `artifacts/revision/` | Outputs of `analysis/revision/*.py` (one CSV per table), shipped so every number can be checked directly; `frontier_judge/*.jsonl` holds the GPT-5.6 judge labels (labels only); `figures/` holds the paper figures (PNG; PDF/SVG are regenerated). |
| `artifacts/revision_stats/` | Seed-42 controlled-grid tables (per-cell HB-Cls differences, TOST, attack split, original-vs-seed-42 deltas), original-grid TOST dumps, StrongREJECT judge labels, seed audit. |
| `data/human_annotation/` | Label-only tables for the 500-sample ASR and 250-sample ORR human annotation sets (see below). |
| `data/evaluator_labels/` | Behavior-level evaluator verdicts (`grid, family, config, attack, benchmark, evaluator, behavior_id, label`) for the original grid and the seed-42 grid, plus a behavior-ID key. See its README. |
| `analysis/revision/` | Analysis scripts cited in the paper ("Reproduce with ..."), plus `figures/`. |
| `analysis/phase5/` | Seed-42 controlled-grid equivalence tests. |
| `analysis/audits/` | Original-grid TOST dump, seed/BB-seed audits, StrongREJECT judge substitution. |
| `analysis/run_all.sh` | Runs every analysis that works from the shipped tables. |
| `training/` | Fine-tuning scripts and SLURM files for every trained cell: `original_grid/` (15 cells + Llama-3.1 ablations: seeds 0/123, rank r4-r64, attention-/MLP-only, SafeLoRA), `seed42_grid/` (controlled re-training + MixAT), `track2/` (Dolly / 3-epoch recipe arms). Per-cell recipes in the `RECIPE.md` files. |
| `evaluation/` | Attack and evaluation wrappers (reference implementations for a SLURM cluster), `harmbench.patch` (our changes to HarmBench). See `evaluation/README.md`. |
| `hf_release/` | Model-card generator and upload script for the released checkpoints (`manifest.json` lists them). |

## Quickstart: reproduce the statistics from the shipped tables

The analysis needs only `numpy`, `pandas`, `scipy==1.15.3`, `statsmodels`, and `matplotlib` (no GPU,
no model weights, no completions). We verified the commands below with Python 3.13.5, numpy 2.1.3,
pandas 2.2.3, scipy 1.15.3, statsmodels 0.14.4, and matplotlib 3.10.0. Wilcoxon p-values with tied or
zero differences depend on the scipy version, so pin scipy 1.15.3. `environment.yml` is the full
training/evaluation environment (Python 3.10, CUDA 12.8).

```bash
pip install numpy pandas scipy==1.15.3 statsmodels matplotlib   # or: conda env create -f environment.yml

# everything that runs from shipped tables (about one minute)
bash analysis/run_all.sh

# or individual results, always from the repository root:
python analysis/revision/claim1_canonical_numbers.py        # Table 2: aggregate dASR, Wilcoxon, Holm, bootstrap CI
python analysis/revision/regime_a_evaluator_calibration.py  # evaluator calibration vs human consensus
python analysis/revision/camera_ready_stats.py              # Friedman, TOST margin sensitivity, ORR calibration, ...
python analysis/phase5/phase5_tost_5fam.py                  # seed-42 per-cell differences + aggregate TOST
python analysis/revision/camera_ready_stats_r6.py           # three-judge TOST, adaptation shifts, AutoDAN/GCG
```

Expected key outputs (all match the paper):

| Result | Script | Value |
|---|---|---|
| Table 2, HB-Cls LoRA dASR | `claim1_canonical_numbers.py` | +4.0 pp, p = 0.007, Holm p = 0.0504, 95% CI [+1.4, +6.6] |
| LoRA-FFT TOST (seed-42 grid, HB-Cls) | `camera_ready_stats.py` | +1.60 pp, 90% CI [-0.88, +4.08], p = 0.014 at ±5 pp |
| LoRA-FFT TOST, other judges | `camera_ready_stats_r6.py` | GPT-4o-mini p = 0.006; LlamaGuard-3 p = 0.018 |
| ASR calibration (N = 412) | `regime_a_evaluator_calibration.py` | HB-Cls precision 81.2, FPR 8.6; GPT-4o-mini FPR 45.7; LlamaGuard-3 FPR 40.6 |
| ORR calibration (N = 239) | `camera_ready_stats.py` | GPT-4o-mini recall 87.1, FPR 18.1; HB-Cls recall 98.4, FPR 32.2 |
| Friedman per family (HB-Cls) | `camera_ready_stats.py` | e.g. Llama-3.1 p = 0.326, Gemma-2 p = 0.0008 |
| StrongREJECT-rubric GPT-4o-mini FPR | `analysis/audits/run_strongreject_judge.py` | 39.2% [33.7, 45.1] |

The shipped CSVs in `artifacts/` were produced by these scripts, and running them again reproduces the
shipped files. The one exception is `artifacts/revision/table2_per_family.csv`, which we regenerated
with scipy 1.15.3 for this release.

## Table and figure map

| Paper item | Script (run from repo root) | Output / input table |
|---|---|---|
| Fig. 1 (design) | `analysis/revision/figures/figure1_pipeline.py` | `artifacts/revision/figures/figure1_pipeline.png` |
| Fig. 2 (BB vs WB dASR) | `analysis/revision/figures/fig1_delta_asr_comparison.py` | reads `claim1_table1.csv`, `peft_deltas.csv` |
| Fig. 3 (ASR vs OR-Bench) | `analysis/revision/figures/fig3_asr_orr_scatter.py` | reads `or_bench_vs_legacy_orr_genuine4bit.csv` |
| Regime A confusion matrices (figure) | `analysis/revision/figures/fig2_regime_a_confusions.py` | reads `regime_a_confusion_*_asr.csv` |
| Table 2 (aggregate dASR); covariate-adjusted OLS (appendix) | `analysis/revision/claim1_canonical_numbers.py` | `claim1_table1.csv`, `claim1_table_a1_ols.csv` |
| Per-family dASR table (Hedges' g = cohens_d x (1 - 3/(4(n-1)-1)), n = 6) | `analysis/revision/table2_per_family.py` | `table2_per_family.csv` |
| DI/JBB sensitivity for per-family table | `analysis/revision/table2_diagnostics.py` | `table2_di_jbb_exclusion_diagnostic.csv` |
| Full ASR matrix, per-dataset dASR, cross-dataset correlations, AutoDAN tables | derived directly from `artifacts/master_asr_long.csv` / `peft_deltas.csv` | |
| AutoDAN and GCG per family, GCG McNemar tests | `analysis/revision/camera_ready_stats_r6.py` | `r6_autodan_gcg_family.csv`, `r6_gcg_mcnemar.csv` |
| Rank / layer ablations, multi-seed AutoDAN | tables only | `artifacts/ablations/*.csv`, `artifacts/cross_seed_results.csv` |
| ActorAttack tables and Wilcoxon | `analysis/revision/actorattack_wilcoxon.py` | `artifacts/actorattack_scoreboard.csv`, `actorattack_wilcoxon.txt` |
| Pooled cross-evaluator OLS, TOST margin sensitivity, family-clustered TOST, Friedman, frontier-judge refusals, ORR calibration | `analysis/revision/camera_ready_stats.py` | `claim1_evaluator_fixed_effect_ols.csv`, `tost_margin_sensitivity.csv`, `tost_family_clustered.csv`, `friedman_per_family.csv`, `frontier_judge_refusal_sensitivity.csv`, `orr_evaluator_calibration.csv` |
| TOST table (seed-42 grid), attack split | `analysis/phase5/phase5_tost_5fam.py`, `analysis/phase5/phase5_attack_split_tost.py` | `artifacts/revision_stats/phase5_tost_*_5fam*.csv` |
| Three-judge TOST, format-matched DeepInception shifts, 14B-excluded TOST | `analysis/revision/camera_ready_stats_r6.py` | `r6_tost_3judge.csv`, `r6_adaptation_shift*.csv` |
| PAIR under the controlled recipe | `analysis/phase5/phase5_pair_tost.py` (controlled data) | `artifacts/revision_stats/phase5_pair_tost_*.csv` |
| Behavior-difficulty and category stratification | `analysis/revision/difficulty_stratification.py` | `difficulty_stratification.csv` |
| LRT for BB/WB divergence | `analysis/revision/lrt_formalization.py` | `lrt_results.csv` |
| Recipe robustness (track 2) | `analysis/revision/track2_recipe_robustness.py` | `track2_equivalence.csv` |
| Same-behavior overlap | `analysis/revision/same_behavior_overlap.py` | `same_behavior_overlap.csv` |
| Transfer across methods | `analysis/revision/attack_transfer.py` | `attack_transfer.csv` |
| GCG loss analysis | `analysis/revision/gcg_loss_analysis.py` (controlled data) | `gcg_loss_*_5fam.csv` |
| ASR evaluator calibration | `analysis/revision/regime_a_evaluator_calibration.py` | `regime_a_calibration.csv` |
| GPT-5.6 frontier judge | `analysis/revision/frontier_judge.py` (API + controlled data), `analysis/revision/frontier_judge_metrics.py` | `frontier_judge_calibration.csv` |
| GPT-4o-mini judge-input ablation (attack prompt vs. HarmBench behavior) | `analysis/revision/judge_input_ablation.py` (API + controlled data) | `judge_input_ablation/metrics.csv` |
| StrongREJECT substitution | `analysis/audits/run_strongreject_judge.py` | `artifacts/revision_stats/strongreject_judge_fpr.csv` |
| Copyright sensitivity | `analysis/revision/copyright_sensitivity.py` | `copyright_sensitivity.csv`, `copyright_calibration.csv` |
| OR-Bench detailed results | `analysis/revision/build_orbench_genuine4bit.py` (controlled data) | `or_bench_vs_legacy_orr_genuine4bit.csv` |
| OR-Bench judge agreement | `analysis/revision/orbench_judge_agreement_regen4bit.py` (controlled data) | `orbench_judge_*_regen4bit.csv` |
| SafeLoRA tables | `analysis/revision/defense_audit.py` (controlled data) | `defense_audit.csv` |
| MixAT table | tables only | `artifacts/mixat_asr.csv` |

Unless a directory is given, outputs are in `artifacts/revision/`.

### Scripts that need controlled-access data

These scripts read raw verdict files, completions, GCG optimization logs, or model checkpoints, which
are not in this repository. Their outputs are shipped, and the scripts show exactly how each number
was computed: `analysis/revision/{gcg_loss_analysis, defense_audit, build_orbench_genuine4bit,
orbench_judge_agreement, orbench_judge_agreement_regen4bit, build_track2_arm_csv, frontier_judge}.py`,
`analysis/phase5/phase5_pair_tost.py`, and `analysis/audits/{verify_bb_seed, seed_audit}.py`.
`training/original_grid/config_audit.py` needs the checkpoints.
`attack_transfer.py`, `phase5_tost_5fam.py`, `camera_ready_stats_r6.py`, `copyright_sensitivity.py`,
and `difficulty_stratification.py` read the raw files when present and otherwise fall back to
`data/evaluator_labels/`.

## Human annotation labels

`data/human_annotation/asr_500_labels.csv` and `orr_250_labels.csv` contain one row per annotated
sample. They hold sample metadata (`attack, benchmark, model_family, config, behavior_id`, HarmBench
categories, `selection_group`, `sampling_stratum`), the two blind annotators' labels (`annotator_A`,
`annotator_B`), `agree`, `consensus` (the shared label where the annotators agree, empty otherwise),
and the three evaluators' labels (`hbcls_label`, `gpt4omini_label`, `llamaguard3_label`,
`majority_label`). For ASR, 1 means the completion is harmful. For ORR, 1 means the benign request was
answered and 0 means it was over-refused. `annotator_B` has one out-of-range raw value (sample
`asr_234`, recorded as 4), a data-entry error. `regime_a_evaluator_calibration.py` corrects it to 1
and logs the correction. The sample is a disagreement either way. The completions and prompts
themselves are not included (see below).

## Checkpoints

We release eight evaluated configurations on the Hugging Face Hub as **gated** repositories. You
must accept the access terms before downloading. The base configurations are the unmodified public
models.

| Repository | Base model | Method | Grid |
|---|---|---|---|
| [taleef/Llama-3.1-8B-Instruct-Alpaca-LoRA-seed42](https://huggingface.co/taleef/Llama-3.1-8B-Instruct-Alpaca-LoRA-seed42) | meta-llama/Llama-3.1-8B-Instruct | LoRA (merged) | controlled seed-42 |
| [taleef/Llama-3.1-8B-Instruct-Alpaca-FFT-seed42](https://huggingface.co/taleef/Llama-3.1-8B-Instruct-Alpaca-FFT-seed42) | meta-llama/Llama-3.1-8B-Instruct | FFT | controlled seed-42 |
| [taleef/gemma-2-9b-it-Alpaca-LoRA](https://huggingface.co/taleef/gemma-2-9b-it-Alpaca-LoRA) | google/gemma-2-9b-it | LoRA (merged) | primary |
| [taleef/gemma-2-9b-it-Alpaca-FFT](https://huggingface.co/taleef/gemma-2-9b-it-Alpaca-FFT) | google/gemma-2-9b-it | FFT | primary |
| [taleef/phi-4-Alpaca-LoRA](https://huggingface.co/taleef/phi-4-Alpaca-LoRA) | microsoft/phi-4 | LoRA (merged) | primary |
| [taleef/phi-4-Alpaca-FFT](https://huggingface.co/taleef/phi-4-Alpaca-FFT) | microsoft/phi-4 | FFT | primary |
| [taleef/Qwen3-4B-Instruct-2507-Alpaca-LoRA](https://huggingface.co/taleef/Qwen3-4B-Instruct-2507-Alpaca-LoRA) | Qwen/Qwen3-4B-Instruct-2507 | LoRA (merged) | primary |
| [taleef/Qwen3-4B-Instruct-2507-Alpaca-FFT](https://huggingface.co/taleef/Qwen3-4B-Instruct-2507-Alpaca-FFT) | Qwen/Qwen3-4B-Instruct-2507 | FFT | primary |

These checkpoints are less safe than their base models by design. They are released only to reproduce
and extend this safety evaluation, not for deployment.

## Data access

Model completions and attack prompts (PAIR/AutoDAN/GCG test cases, DeepInception and ArtPrompt
prompts, ActorAttack dialogues) contain harmful text and are not in this repository. This includes the
text of the human-annotated samples. We make them available to researchers on request under
controlled access. Contact the authors at tamst01@pfw.edu or jrusert@pfw.edu. This repository
contains only aggregate tables, binary labels keyed by HarmBench / JailbreakBench behavior IDs, and
code.

## Licenses

- Code in this repository: MIT (see `LICENSE`).
- Checkpoints: each inherits its base model's license, as stated on its model card. That is the Llama
  3.1 Community License (Llama-derived models, "Built with Llama"), the Gemma Terms of Use, Apache 2.0
  (Qwen3), and MIT (Phi-4). The fine-tuning data derive from Stanford Alpaca (CC BY-NC 4.0), so all
  Alpaca-derived checkpoints are released for **non-commercial research use only**. Dolly-15k (track-2
  arms) is CC BY-SA 3.0.
- HarmBench is MIT-licensed; `evaluation/harmbench.patch` is distributed under the same terms.
  JailbreakBench behaviors, OR-Bench, and ActorAttack keep their original licenses.

## Citation

```bibtex
@inproceedings{tamsal2026adaptation,
  title     = {Adaptation, Not Algorithm: {LoRA} and Full Fine-Tuning Show Comparable Black-Box
               Jailbreak Degradation in Five Open-Weight {LLMs}},
  author    = {Tamsal, Taleef and Rusert, Jonathan},
  booktitle = {Findings of the Association for Computational Linguistics: AACL-IJCNLP 2026},
  year      = {2026}
}
```
