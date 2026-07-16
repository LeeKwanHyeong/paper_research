#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ai_env/bin/python}"
TMUX_SESSION="${TMUX_SESSION:-titantpp_v4_taxi_time_audit_0716}"
OUTPUT_DIR="${PROJECT_ROOT}/search_artifacts/model_enhancement_titantpp_v4_taxi_train_time_audit_0716"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" simple_lab_test/search/analyze_taxi_mark_time_audit.py \
  --dataset sample_data/new_york_taxi/yellow_trip_hourly_train.parquet \
  --output-dir "${OUTPUT_DIR}" \
  --lookback-weeks 168 \
  --max-seq-len 256 \
  --execution-server 5090 \
  --tmux-session "${TMUX_SESSION}"
