#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ENTRYPOINT="$PROJECT_ROOT/simple_lab_test/search/tpp_experiment.py"
ARTIFACT_ROOT="$PROJECT_ROOT/search_artifacts"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"
mkdir -p "$MPLCONFIGDIR"

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

run_long_epoch model_enhancement_v3b_insta_smoke_e1_0710 \
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
  --value-input-mode residual \
  --train-loss-scope target_only \
  --loss-mode hybrid \
  --eval-selections best_val_nll,best_score,final \
  --device cuda \
  --force-rerun \
  --stop-on-error

run_long_epoch model_enhancement_v3b_inter_short_e50_0710 \
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
  --value-input-mode residual \
  --train-loss-scope target_only \
  --loss-mode hybrid \
  --eval-selections best_val_nll,best_score,final \
  --device cuda \
  --force-rerun \
  --stop-on-error

run_long_epoch model_enhancement_v3b_taxi_short_e50_0710 \
  --datasets yellow_trip_hourly \
  --models titantpp \
  --titan-candidates mid_lmm \
  --epochs 50 \
  --seeds 42 \
  --lr 1e-3 \
  --batch-size 128 \
  --lookback-weeks 168 \
  --max-seq-len 256 \
  --split-mode fixed \
  --value-head-activation identity \
  --value-head-mode mark_conditioned_experts \
  --qty-mark-gradient-mode detached \
  --value-input-mode residual \
  --train-loss-scope target_only \
  --loss-mode hybrid \
  --eval-selections best_val_nll,best_score,final \
  --device cuda \
  --force-rerun \
  --stop-on-error
