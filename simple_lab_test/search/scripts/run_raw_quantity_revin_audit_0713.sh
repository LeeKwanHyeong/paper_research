#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ai_env/bin/python}"
TMUX_SESSION="${TMUX_SESSION:-titantpp_raw_revin_audit_0713}"
OUTPUT_DIR="${PROJECT_ROOT}/search_artifacts/model_enhancement_raw_quantity_revin_audit_0713"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" simple_lab_test/search/analyze_raw_quantity_revin_audit.py \
  --dataset sample_data/head_office/marked_target_with_split.parquet \
  --output-dir "${OUTPUT_DIR}" \
  --lookback-weeks 52 \
  --max-seq-len 16 \
  --execution-server 5090 \
  --tmux-session "${TMUX_SESSION}"
