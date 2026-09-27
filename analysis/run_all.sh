#!/usr/bin/env bash
# Recompute every statistic that can be derived from the shipped tables.
# Run from the repository root:  bash analysis/run_all.sh
# Outputs are written to artifacts/revision/, artifacts/revision_stats/ and
# artifacts/revision/figures/ (overwriting the shipped copies with identical values).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

run() { echo; echo "################ $1"; "$PY" "$1"; }

# Claim 3: human-calibrated evaluator metrics
run analysis/revision/regime_a_evaluator_calibration.py
run analysis/revision/frontier_judge_metrics.py
run analysis/audits/run_strongreject_judge.py

# Claim 1: aggregate Delta-ASR, per-family profiles, robustness checks
run analysis/revision/claim1_canonical_numbers.py
run analysis/revision/table2_per_family.py
run analysis/revision/table2_diagnostics.py
run analysis/audits/tost_dump.py
run analysis/audits/revision_stats.py

# Seed-42 controlled grid: equivalence tests
run analysis/phase5/phase5_tost_5fam.py
run analysis/phase5/phase5_attack_split_tost.py
run analysis/phase5/phase5_commonmode.py
run analysis/revision/camera_ready_stats.py
run analysis/revision/camera_ready_stats_r6.py
run analysis/revision/track2_recipe_robustness.py
run analysis/revision/same_behavior_overlap.py
run analysis/revision/attack_transfer.py

# Behavior-level and appendix analyses
run analysis/revision/copyright_sensitivity.py
run analysis/revision/difficulty_stratification.py
run analysis/revision/lrt_formalization.py
run analysis/revision/actorattack_wilcoxon.py

# Figures
run analysis/revision/figures/figure1_pipeline.py
run analysis/revision/figures/fig1_delta_asr_comparison.py
run analysis/revision/figures/fig2_regime_a_confusions.py
run analysis/revision/figures/fig3_asr_orr_scatter.py
