#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ENTRYPOINT="$PROJECT_ROOT/simple_lab_test/search/tpp_experiment.py"
BASE_DIR="$PROJECT_ROOT/search_artifacts/model_enhancement_v2_taxi_multiseed_e50_0710"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"
mkdir -p "$MPLCONFIGDIR" "$BASE_DIR/logs"

"$PYTHON_BIN" "$ENTRYPOINT" long-epoch \
  --base-dir "$BASE_DIR" \
  --datasets yellow_trip_hourly \
  --models titantpp \
  --titan-candidates mid_lmm \
  --epochs 50 \
  --seeds 42,52,62 \
  --lr 1e-3 \
  --batch-size 128 \
  --lookback-weeks 168 \
  --max-seq-len 256 \
  --split-mode fixed \
  --value-head-activation identity \
  --value-head-mode shared \
  --qty-mark-gradient-mode coupled \
  --value-input-mode residual \
  --train-loss-scope target_only \
  --loss-mode hybrid \
  --eval-selections best_val_nll,best_score,final \
  --device cuda \
  --force-rerun \
  --stop-on-error \
  2>&1 | tee -a "$BASE_DIR/logs/run.log"
