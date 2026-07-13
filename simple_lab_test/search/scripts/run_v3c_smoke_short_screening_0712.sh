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

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] V3c CUDA model-test"
run_model_test model_enhancement_v3c_model_test_0712 \
  --models titantpp \
  --titan-candidates small_lmm \
  --device cuda \
  --seq-len 16 \
  --num-marks 12 \
  --rmtpp-hidden-dim 64 \
  --value-head-mode mark_conditioned_experts \
  --qty-mark-gradient-mode detached \
  --value-encoder-gradient-mode detached \
  --left-pad \
  --stop-on-error

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] V3c Instacart e1 smoke"
run_long_epoch model_enhancement_v3c_insta_smoke_e1_0712 \
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
  --value-head-mode mark_conditioned_experts \
  --qty-mark-gradient-mode detached \
  --value-encoder-gradient-mode detached \
  --value-input-mode residual \
  --train-loss-scope target_only \
  --loss-mode hybrid \
  --eval-selections best_val_nll,best_score,final \
  --device cuda \
  --force-rerun \
  --stop-on-error

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] V3c Intermittent seed-42 e50"
run_long_epoch model_enhancement_v3c_inter_short_e50_0712 \
  --datasets intermittent \
  --models titantpp \
  --titan-candidates small_lmm \
  --epochs 50 \
  --seeds 42 \
  --lr 1e-3 \
  --batch-size 128 \
  --lookback-weeks 52 \
  --max-seq-len 16 \
  --split-mode fixed \
  --value-head-activation identity \
  --value-head-mode mark_conditioned_experts \
  --qty-mark-gradient-mode detached \
  --value-encoder-gradient-mode detached \
  --value-input-mode residual \
  --train-loss-scope target_only \
  --loss-mode hybrid \
  --eval-selections best_val_nll,best_score,final \
  --device cuda \
  --force-rerun \
  --stop-on-error

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] V3c screening sequence completed"
