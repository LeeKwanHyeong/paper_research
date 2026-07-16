#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ai_env/bin/python}"
TMUX_SESSION="${TMUX_SESSION:-titantpp_v6_taxi_memory_audit_0717}"
EXECUTION_SERVER="${EXECUTION_SERVER:-5090}"
OUTPUT_DIR="${PROJECT_ROOT}/search_artifacts/model_enhancement_titantpp_v6_taxi_train_memory_audit_0717"
DATASET="${PROJECT_ROOT}/sample_data/new_york_taxi/yellow_trip_hourly_train.parquet"

: "${SOURCE_REVISION:?SOURCE_REVISION must be the checksum-synced full commit SHA}"

if [[ "${EXECUTION_SERVER}" != "5090" ]]; then
  echo "This audit is locked to execution server 5090; received ${EXECUTION_SERVER}." >&2
  exit 2
fi

if [[ ! "${SOURCE_REVISION}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "SOURCE_REVISION must be a 40-character lowercase Git SHA." >&2
  exit 2
fi

if [[ ! -f "${DATASET}" ]]; then
  echo "Taxi train-only parquet is missing: ${DATASET}" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"

export PYTHONHASHSEED=42
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"${PYTHON_BIN}" -c "import matplotlib, numpy, polars, sklearn"

"${PYTHON_BIN}" simple_lab_test/search/analyze_taxi_pre_window_memory_audit.py \
  --dataset "${DATASET}" \
  --output-dir "${OUTPUT_DIR}" \
  --lookback-weeks 168 \
  --max-seq-len 256 \
  --execution-server "${EXECUTION_SERVER}" \
  --tmux-session "${TMUX_SESSION}" \
  --source-revision "${SOURCE_REVISION}"
