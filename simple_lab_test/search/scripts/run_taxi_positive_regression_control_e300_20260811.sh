#!/usr/bin/env bash
set -euo pipefail

: "${SOURCE_REVISION:?SOURCE_REVISION must be the frozen 40-character source revision}"
: "${VARIANT:?VARIANT must be minmax_sigmoid or log_regression}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT must identify the variant artifact directory}"

if [[ ! "${SOURCE_REVISION}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "[preflight_error] SOURCE_REVISION must be a lowercase 40-character Git SHA." >&2
  exit 2
fi
if [[ "${VARIANT}" != "minmax_sigmoid" && "${VARIANT}" != "log_regression" ]]; then
  echo "[preflight_error] unsupported VARIANT=${VARIANT}" >&2
  exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ai_env/bin/python}"
DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/search_artifacts/final_fair_matched_rmtpp_thp_e300_20260805/yellow_trip_hourly/cache/yellow_trip_hourly/fixed_split/marked_fixed_base_10p0.parquet}"
PROPOSAL_ROOT="${PROPOSAL_ROOT:-${PROJECT_ROOT}/search_artifacts/final_fair_matched_rmtpp_thp_e300_20260805}"
EXPECTED_DATA_SHA="b47e98e9fdb75d4274a18e3f8a5d8f463418a1d56a6db4db7d9b834c9d89ca46"
RUNNER="${PROJECT_ROOT}/paper/scripts/run_taxi_quantity_interface_ablation.py"

mkdir -p "${OUTPUT_ROOT}/logs"

observed_data_sha="$(sha256sum "${DATA_PATH}")"
observed_data_sha="${observed_data_sha%% *}"
if [[ "${observed_data_sha}" != "${EXPECTED_DATA_SHA}" ]]; then
  echo "[preflight_error] fixed split SHA mismatch: ${observed_data_sha}" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1 MPLBACKEND=Agg CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "[launch] variant=${VARIANT} source_revision=${SOURCE_REVISION} data_sha=${observed_data_sha}"
echo "[launch] output_root=${OUTPUT_ROOT}"

exec "${PYTHON_BIN}" "${RUNNER}" \
  --data "${DATA_PATH}" \
  --proposal-root "${PROPOSAL_ROOT}" \
  --output-dir "${OUTPUT_ROOT}" \
  --source-revision "${SOURCE_REVISION}" \
  --device cuda \
  --epochs 300 \
  --batch-size 128 \
  --lr 0.001 \
  --lookback-weeks 168 \
  --max-seq-len 256 \
  --hidden-dim 128 \
  --lambda-raw 1.0 \
  --grad-clip 1.0 \
  --early-stopping-patience 60 \
  --min-epochs 50 \
  --variants "${VARIANT}" \
  --seeds 42,52,62 \
  --skip-proposal
