#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ai_env/bin/python}"
ENTRYPOINT="$PROJECT_ROOT/simple_lab_test/search/tpp_experiment.py"
ARTIFACT_ROOT="$PROJECT_ROOT/search_artifacts"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"
mkdir -p "$MPLCONFIGDIR"

run_model_test() {
  local experiment_name="$1"
  shift
  local output_dir="$ARTIFACT_ROOT/$experiment_name"
  mkdir -p "$output_dir/logs"
  "$PYTHON_BIN" "$ENTRYPOINT" model-test \
    --output-dir "$output_dir" \
    "$@" \
    2>&1 | tee -a "$output_dir/logs/run.log"
}

run_long_epoch() {
  local experiment_name="$1"
  shift
  local base_dir="$ARTIFACT_ROOT/$experiment_name"
  mkdir -p "$base_dir/logs"
  "$PYTHON_BIN" "$ENTRYPOINT" long-epoch \
    --base-dir "$base_dir" \
    "$@" \
    2>&1 | tee -a "$base_dir/logs/run.log"
}

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] V5a CUDA model-test"
run_model_test model_enhancement_v5a_model_test_0712 \
  --models titantpp \
  --titan-candidates small_lmm \
  --device cuda \
  --seq-len 16 \
  --num-marks 12 \
  --rmtpp-hidden-dim 64 \
  --value-head-mode shared \
  --qty-mark-gradient-mode coupled \
  --value-encoder-gradient-mode coupled \
  --marker-loss-mode ce_rps \
  --lambda-ordinal 0.10 \
  --left-pad \
  --stop-on-error

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] V5a Instacart top-20 e1 smoke"
run_long_epoch model_enhancement_v5a_insta_smoke_e1_0712 \
  --datasets insta_market_basket \
  --models titantpp \
  --titan-candidates small_lmm \
  --epochs 1 \
  --seeds 42 \
  --lr 1e-3 \
  --batch-size 16 \
  --lookback-weeks 10 \
  --max-seq-len 16 \
  --insta-max-series 20 \
  --split-mode fixed \
  --value-head-activation identity \
  --value-head-mode shared \
  --qty-mark-gradient-mode coupled \
  --value-encoder-gradient-mode coupled \
  --value-input-mode residual \
  --train-loss-scope target_only \
  --loss-mode hybrid \
  --marker-loss-mode ce_rps \
  --lambda-ordinal 0.10 \
  --eval-selections best_val_nll,best_score,final \
  --device cuda \
  --force-rerun \
  --stop-on-error

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] V5a integration smoke sequence completed"
