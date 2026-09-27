# Attack and evaluation code

This directory contains the attack drivers, evaluator wrappers and SLURM submission scripts used to
produce the attack success rates (ASR) and over-refusal numbers in the paper, plus
`harmbench.patch`, our changes to HarmBench.

These are reference implementations. They were written for the Purdue RCAC Gilbreth cluster (SLURM
partitions such as `a30`, `a100-40gb`, `a100-80gb`, `standby` QOS, `module load anaconda`,
a conda environment called `harmbench_env`) and for the directory layout of our research
repository. They are not a turnkey pipeline, and you will need to adapt partitions, environments and
paths before running them on another cluster.

No data are included here: no behavior datasets, attack prompts, test cases, model completions or
evaluator outputs. Evaluator prompt templates (HarmBench classifier prompt, GPT-4o-mini judge
rubric, OR-Bench response-check prompt) and attack templates (DeepInception prompt, ArtPrompt code)
are part of the code.

## Conventions

* `${PROJECT_ROOT}`: root of the research repository. The scripts expect this layout:
  * `${PROJECT_ROOT}/toolkits/HarmBench`: HarmBench at upstream commit `8e1604d` with `harmbench.patch` applied
  * `${PROJECT_ROOT}/toolkits/ActorAttack`: the ActorAttack checkout described below
  * `${PROJECT_ROOT}/evaluation`: this directory
  * `${PROJECT_ROOT}/models/...`: model checkpoints (see `training/` and the model cards)
* `${VENV_DIR}`: Python environment with the HarmBench requirements, vLLM and transformers.
  `${HF_HOME}` is the Hugging Face cache. `slurm/env.sh` sets the common variables for a job.
* `YOUR_ACCOUNT`: replace with your SLURM account.
* Python files read `PROJECT_ROOT` with `os.environ.get("PROJECT_ROOT", ".")`. Paths inside YAML
  files and some Python defaults are written literally as `${PROJECT_ROOT}/...` and are not expanded,
  so replace them with real paths before use.
* API keys are read from the environment (`OPENAI_API_KEY`; `HF_TOKEN` for gated models). Some
  scripts also load a `.env` file from the repository root if one exists. No keys are included.
* Model configuration names follow HarmBench's `models.yaml`: `<family>_<config>_custom`, with
  families `gemma2`, `llama31`, `qwen3`, `phi4`, `qwen25_14b` and configurations `base_fp16`,
  `base_4bit`, `lora`, `qlora`, `fft`. Seed-42 controlled-grid checkpoints are
  `models/phase5/<family>_<method>_seed42_merged` and the corresponding HarmBench entries are
  `<family>_<method>_seed42_custom`.

## HarmBench modifications (`harmbench.patch`)

`harmbench.patch` is `git diff 8e1604d HEAD` of our HarmBench fork, restricted to code,
configuration and job scripts. It touches 49 files (listed at the end of this section). Our model
entries for HarmBench's `configs/model_configs/models.yaml` are shipped separately in
`harmbench_models_custom.yaml` (72 `*_custom` entries; the upstream entries are unchanged).

Apply it to upstream HarmBench:

```bash
git clone https://github.com/centerforaisafety/HarmBench
cd HarmBench
git checkout 8e1604d
git apply /path/to/harmbench.patch
cat /path/to/harmbench_models_custom.yaml >> configs/model_configs/models.yaml
```

The patch applies cleanly to `8e1604d` (`git apply` prints trailing-whitespace warnings, which can
be ignored). Notes:

* Absolute paths in the patch and in `harmbench_models_custom.yaml` were replaced with
  `${PROJECT_ROOT}` (model paths, a few Python defaults and shell scripts), cluster account
  names with `YOUR_ACCOUNT`, and some comments were reworded. These edits are confined to added
  lines, so the patch still applies, but the placeholders must be replaced before the configs or
  scripts can be used.
* Data files are not in the patch. The runs used, under `data/`:
  `behavior_datasets/jbb_behaviors.csv` (JailbreakBench JBB-Behaviors in HarmBench CSV format,
  see `original_grid/jbb/convert_jbb_to_harmbench_format.py`),
  `optimizer_targets/jbb_targets_text.json` (JBB target strings, used by AutoDAN and GCG on JBB),
  `behavior_datasets/data/benign-behaviors.csv` (the 100 JBB benign behaviors) and
  `behavior_datasets/or_bench/or_bench_{hard_1k,toxic}.csv` (see `or_bench/download_or_bench.py`).
  HarmBench-400 is upstream `data/behavior_datasets/harmbench_behaviors_text_all.csv`.
* `evaluate_or_bench.py` in the patch imports `scripts.or_bench.evaluation` (the location of that
  module in our research repository). In this release the module is `evaluation/or_bench/evaluation.py`,
  and the copy in `harmbench/evaluate_or_bench.py` imports it from there.
* Per-family job scripts for individual attack steps and one-off submission helpers from the fork are
  not part of the patch; the generic step scripts (`scripts/generate_test_cases.sh`,
  `scripts/generate_completions.sh`, `scripts/evaluate_completions.sh`, `scripts/run_pipeline.py`) are.

What the patch changes, in brief:

* `baselines/model_utils.py`: conversation templates for Phi-4, Gemma-2 and Llama-3.1, a
  `chat_template` override, tokenizer pad-token handling.
* `baselines/autodan/*`: Mistral-7B-Instruct-v0.3 mutation model, numerical guards on the mutation
  model's sampling, memory handling for the population loss.
* `baselines/artprompt/artprompt.py`: gpt-4o-mini masking model, API key from the environment,
  fallback when a suggested mask word is not in the behavior.
* `baselines/pair/language_models.py`: pad-token handling for local HF models.
* `configs/method_configs/*`: PAIR, AutoDAN, ArtPrompt and GCG experiments for the 25 configurations
  (HarmBench-400) and JBB-100 variants (`PAIR_config_jbb.yaml`, `AutoDAN_config_jbb.yaml`).
* `harmbench_models_custom.yaml`: the 25 original-grid configurations, the seed-42
  controlled-grid checkpoints, and the LoRA rank / layer ablation and multi-seed variants.
* `generate_completions.py`, `generate_test_cases.py`, `evaluate_completions.py`, `eval_utils.py`:
  per-model vLLM/HF selection (`use_vllm` in `models.yaml`), direct-request prompts read from a
  behaviors CSV when no test cases are given (OR-Bench, benign set), an HF-transformers path for the
  HB-Cls classifier, and support for benign-behavior CSV columns.
* New evaluators: `evaluate_attacks_api.py` (GPT-4o-mini), `evaluate_completions_llamaguard.py`
  (LlamaGuard-3), `evaluate_completions_fixed.py`, `evaluate_deepinception*.py`,
  `evaluate_refusals_api.py`, `evaluate_benign_llama2.py`, `evaluate_benign_llamaguard.py`,
  `calculate_orr.py`, `evaluate_or_bench.py`. Readable copies are in `harmbench/`.

Files touched: `api_models.py`, `baselines/artprompt/artprompt.py`, `baselines/autodan/AutoDAN.py`,
`baselines/autodan/mutate_models.py`, `baselines/model_utils.py`, `baselines/pair/language_models.py`,
`calculate_orr.py`, `configs/method_configs/{ArtPrompt_config,AutoDAN_config,AutoDAN_config_jbb,GCG_config,PAIR_config,PAIR_config_jbb}.yaml`,
`configs/pipeline_configs/run_pipeline.yaml`, `eval_utils.py`,
`evaluate_attacks_api.py`, `evaluate_benign_llama2.py`, `evaluate_benign_llamaguard.py`,
`evaluate_completions.py`, `evaluate_completions_fixed.py`, `evaluate_completions_llamaguard.py`,
`evaluate_deepinception.py`, `evaluate_deepinception_harmbench.py`,
`evaluate_deepinception_harmbench_api.py`, `evaluate_or_bench.py`, `evaluate_refusals_api.py`,
`generate_completions.py`, `generate_test_cases.py`, and in `scripts/`: `autodan_jbb_job_template.sh`,
`download_llama_guard.py`, `evaluate_attacks_api.sh`, `evaluate_completions.sh`,
`evaluate_refusals_api.sh`, `generate_completions.sh`, `generate_test_cases.sh`,
`merge_completions_custom.py`, `merge_test_cases_env.sh`, `run_benign_evaluation.sh`,
`run_deepinception_harmbench_eval_api.sh`, `run_or_bench_evaluation.sh`, `run_pipeline.py`,
`submit_benign_llamaguard.sh`, `submit_deepinception_eval.sh`, `submit_llamaguard_deepinception_hb.sh`,
`submit_llamaguard_harmbench_all.sh`, `submit_llamaguard_jbb_all.sh`,
`submit_llamaguard_jbb_deepinception.sh`, `submit_llamaguard_jbb_gcg.sh`, `submit_sequential_all_80.sh`.

## ActorAttack

ActorAttack was run from a fork of the upstream repository `AI45Lab/ActorAttack`:

* upstream base commit: `dbd233a19b234d8dd6eee57d5900b1aa21d6b1db`
* fork commit used for all runs: `1472602f5fba1199c0af90cf61ca449c59527d08`
  (one commit on top of the base; changes `inattack.py`, `preattack.py`, `utils.py` and `.gitignore`
  for local Hugging Face inference of target and attacker models)

The checkout had no uncommitted code changes, so no separate ActorAttack patch is needed. The
wrappers in `actorattack/` replace ActorAttack's `utils` module with `utils_proxy.py` (local HF
clients) and call the toolkit's pre-attack and in-attack stages.

## Evaluators

* **HB-Cls**: `cais/HarmBench-Llama-2-13b-cls` with HarmBench's `LLAMA2_CLS_PROMPT`, one greedy
  yes/no token. Original grid: HarmBench `evaluate_completions.py` (as patched; copyright behaviors
  use HarmBench's hash check) and `harmbench/evaluate_completions_fixed.py` /
  `harmbench/evaluate_deepinception*.py`. Seed-42 grid: `seed42_grid/hb_cls_score.py` (vLLM).
* **GPT-4o-mini**: LLM-as-judge rubric defined as `JUDGE_SYSTEM_PROMPT` /
  `JUDGE_USER_PROMPT_TEMPLATE` in `harmbench/evaluate_attacks_api.py` (JSON output with
  `is_jailbroken` and `refusal_type`); the judge sees the attack prompt and the response clipped to
  512 gpt2 tokens. `seed42_grid/secondary_score.py` uses the same rubric and clipping.
  DeepInception JBB cells: `deepinception/jbb/judge_deepinception_gpt4omini.py`; DeepInception
  HarmBench-400 cells: `harmbench/evaluate_deepinception_harmbench_api.py`.
* **LlamaGuard-3-8B**: `harmbench/evaluate_completions_llamaguard.py` (conversation
  [user: prompt, assistant: response] through the model's chat template, greedy, "unsafe" = success,
  response clipped to 512 tokens); `seed42_grid/secondary_score.py --evaluator llamaguard` for the
  seed-42 grid.
* **OR-Bench**: GPT-4o-mini with the OR-Bench three-way response-check prompt
  (`OR_BENCH_RESPONSE_CHECKING_PROMPT` in `or_bench/evaluation.py`; labels `direct_answer`,
  `direct_refusal`, `indirect_refusal`), called by `harmbench/evaluate_or_bench.py`.
* **Legacy benign ORR set** (100 JBB benign behaviors): GPT-4o-mini refusal prompt
  (`harmbench/evaluate_refusals_api.py`), HB-Cls (`harmbench/evaluate_benign_llama2.py`),
  LlamaGuard-3 (`harmbench/evaluate_benign_llamaguard.py`), aggregated by `harmbench/calculate_orr.py`.
* **StrongREJECT judge substitution** and the **GPT-5.6 frontier judge** on the human-annotated set
  are in `analysis/audits/run_strongreject_judge.py` and `analysis/revision/frontier_judge.py`.

## Which scripts produce which results

### Original grid (25 configurations), PAIR / ArtPrompt / AutoDAN

* HarmBench-400: HarmBench's three-step pipeline (`scripts/run_pipeline.py` with
  `configs/pipeline_configs/run_pipeline.yaml`, or `scripts/generate_test_cases.sh` ->
  `merge_test_cases` -> `scripts/generate_completions.sh` -> `scripts/evaluate_completions.sh`), with
  the method configs `PAIR_config.yaml`, `ArtPrompt_config.yaml`, `AutoDAN_config.yaml` from the patch.
  AutoDAN test cases are optimized separately against each configuration. Completions are greedy,
  512 new tokens, generated with `--generate_with_vllm`.
* JBB-100: `original_grid/jbb/`. `run_harmbench_attacks_jbb.sh` drives the same HarmBench steps on
  `jbb_behaviors.csv`; `pair/submit_pair_step*.sh`, `autodan/submit_autodan_step*_direct.sh`,
  `autodan/run_autodan_jbb.sh` (uses `AutoDAN_config_jbb.yaml`) and `artprompt/run_artprompt_jbb.sh`
  submit the individual steps; `*/evaluate_*_jbb_api.sh` run the GPT-4o-mini judge.
* Secondary evaluators for these cells: GPT-4o-mini via `scripts/submit_sequential_all_80.sh` /
  `scripts/evaluate_attacks_api.sh` (patch); LlamaGuard-3 via `scripts/submit_llamaguard_harmbench_all.sh`
  and `scripts/submit_llamaguard_jbb_all.sh` (patch).
* The Qwen-2.5 cells were submitted with family-specific copies of the same step commands.

### Original grid, DeepInception

* JBB-100 generation: `deepinception/jbb/<family>/deep_inception_*.py` (one script per
  configuration; chat template, temperature 0.6, top-p 0.9, 1024 new tokens), submitted with
  `deepinception/jbb/run_deepinception_jbb.slurm`. Outputs are JSONL files under
  `outputs/deepinception/<family>/`.
* JBB-100 scoring: HB-Cls with `harmbench/evaluate_deepinception.py` (after
  `deepinception/jbb/convert_deepinception_to_harmbench_format.py`), GPT-4o-mini with
  `deepinception/jbb/judge_deepinception_gpt4omini.py`, LlamaGuard-3 with
  `scripts/submit_llamaguard_jbb_deepinception.sh` (patch) after `convert_for_llamaguard.py` /
  `convert_jsonl_to_json.py`.
* HarmBench-400 generation: `deepinception/harmbench400/deepinception_harmbench.py`, submitted
  per configuration by `submit_deepinception_harmbench.sh` (same generation settings).
* HarmBench-400 scoring: `deepinception/harmbench400/evaluate_deepinception_harmbench.sh` (HB-Cls),
  `scripts/run_deepinception_harmbench_eval_api.sh` (GPT-4o-mini, patch),
  `scripts/submit_llamaguard_deepinception_hb.sh` (LlamaGuard-3, patch).

### Seed-42 controlled grid (equivalence tests)

All in `seed42_grid/`. The 15 re-trained {LoRA, QLoRA, FFT} cells are attacked with DeepInception
and ArtPrompt on JBB-100 and HarmBench-400 (60 cells) and scored by all three evaluators.

* Generation: `submit_seed42_inference.sh` runs `deepinception_phase5.py` (same settings as the
  original DeepInception scripts) and `artprompt_phase5.py` (original-grid ArtPrompt test cases of the
  same family and method, greedy, 512 new tokens, `--template alpaca`, the format used for all
  controlled-grid ArtPrompt cells).
* HB-Cls: `score_seed42_hbcls.sh` (run via `score_seed42_hbcls.slurm`), which calls `hb_cls_score.py`.
* GPT-4o-mini and LlamaGuard-3: `build_secondary_manifests.sh`, then
  `sbatch --export=ALL,EVAL=gpt4omini|llamaguard,MANIFEST=... secondary_eval.slurm` (runs
  `secondary_score.py`).
* PAIR subset (Llama-3.1, Gemma-2, Qwen-3): `pair/submit_pair_seed42_step1_{jbb,hb}.sh` ->
  `pair/merge_pair_seed42.py` -> `pair/submit_pair_seed42_step2.sh` -> `pair/submit_pair_seed42_step3.sh`
  (HarmBench PAIR with `PAIR_config_jbb.yaml` / `PAIR_config.yaml`, HB-Cls scoring).
* DeepInception sampling variability: `rollout/submit_di_rollout.sh` (5 seeded rollouts per cell,
  `deepinception_rollout.py`), `rollout/submit_score_rollouts.sh` (`score_rollouts.py`),
  `rollout/analyze_rollout_variance.py`.
* Recipe robustness (second dataset / training length): `recipe_robustness/submit_arm_infer.sh` and
  `recipe_robustness/score_arm.sh`, using the same drivers. The ArtPrompt format is set with
  `AP_TEMPLATE` (Alpaca by default; the Qwen-3 Dolly 3-epoch cells were generated with the
  tokenizer chat template).
* MixAT defended-base comparison (Llama-3.1, JBB-100): `mixat/submit_mixat_inference.sh`
  (DeepInception and ArtPrompt with the native Llama-3.1 chat template, `--template tokenizer`),
  `mixat/submit_mixat_hbcls.sh`, `mixat/submit_mixat_llamaguard.sh`, `mixat/run_mixat_gpt4omini.sh`.

### GCG white-box probe (JBB-100, five families x {base_fp16, lora, fft})

* `gcg/submit_gcg_jbb.sh <cell> <partitions>`: one SLURM array task per behavior
  (`scripts/generate_test_cases.sh GCG ...`; 500 steps, search width 512, 20-token suffix, from
  `GCG_config.yaml`).
* `gcg/submit_gcg_jbb_score.sh <cell> <partitions>`: merge test cases, greedy completions
  (512 new tokens), HB-Cls scoring.
* `gcg/submit_llamaguard_jbb_gcg.sh` (LlamaGuard-3) and `gcg/run_gcg_jbb_api_eval.sh` (GPT-4o-mini).

### ActorAttack (HarmBench behaviors, 25 configurations)

* `actorattack/slurm/submit_actorattack_sweep.sh` submits one shared pre-attack job
  (`actorattack_preattack.slurm`: actors and query chains generated once by
  Mistral-7B-Instruct-v0.3, up to 2 actors per behavior) and then, per target model,
  `submit_actorattack_model.sh`, which runs the in-attack stage in behavior chunks
  (`actorattack_sweep.slurm`; live dialogues, query rewriting and early stopping with the attacker
  model, greedy target decoding, 512 new tokens).
* A dependent job converts the dialogues to HarmBench completions
  (`actorattack/convert_actorattack_to_harmbench.py`, final target response per behavior) and
  `submit_actorattack_evals.sh` submits HB-Cls, LlamaGuard-3 and GPT-4o-mini scoring.
* Python entry point: `python -m evaluation.actorattack.run_actorattack` (run from `${PROJECT_ROOT}`).

### OR-Bench (Hard-1K and Toxic, 25 configurations)

* `or_bench/download_or_bench.py` writes the two subsets as HarmBench-style CSVs.
* `or_bench/submit_or_bench_sweep.sh`: per family, a generation array job
  (`or_bench_family_generate.slurm`, HarmBench `generate_completions.py`, greedy, 512 new tokens)
  followed by a GPT-4o-mini judging array job (`or_bench_family_evaluate.slurm`,
  `evaluate_or_bench.py`).
* Base 4-bit cells: `or_bench/submit_or_bench_4bit.sh` (`or_bench_4bit_generate.slurm`,
  `or_bench_4bit_evaluate.slurm`) generates them through the vLLM path, which applies the
  bitsandbytes 4-bit setting, and writes them to the `regen4bit` results namespace. It uses a copy
  of `models.yaml` (`models_4bit_regen.yaml`) without `use_vllm: false` on the `phi4_base_4bit_custom`
  and `qwen25_14b_base_4bit_custom` entries.

### Legacy benign ORR set (100 JBB benign behaviors)

* `benign_orr/run_benign_evaluation.sh`: greedy completions (512 new tokens) and GPT-4o-mini refusal
  judging (`evaluate_refusals_api.sh` -> `harmbench/evaluate_refusals_api.py`).
* `benign_orr/evaluate_orr_llama2.slurm`: HB-Cls (`harmbench/evaluate_benign_llama2.py`).
* `benign_orr/submit_benign_llamaguard.sh`: LlamaGuard-3 (`harmbench/evaluate_benign_llamaguard.py`).

## Not included here

* Fine-tuning, adapter merging and MixAT merging: see `training/`.
* Statistics, tables and figures: see `analysis/`.
* Evaluation jobs for the AutoDAN supplementary analyses (multi-seed Llama-3.1, LoRA rank and layer
  ablations, Safe LoRA, temperature 0.7 decoding check). These reuse the HarmBench
  `generate_completions.py` / `evaluate_completions.py`, `evaluate_attacks_api.py` and
  `evaluate_completions_llamaguard.py` calls shown above with the corresponding `*_custom` model
  entries in `models.yaml` and the existing AutoDAN test cases.
