#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/sample_data/intermittent_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/search_artifacts/count_aware_lognormal_k1_screening_e300_20260815}"
SOURCE_REVISION="${SOURCE_REVISION:?SOURCE_REVISION must be the frozen 40-character Git revision}"
EXECUTION_ROLE="${EXECUTION_ROLE:-primary_5080}"
MODELS="${MODELS:-thp,titantpp}"
VARIANTS="${VARIANTS:-log_mse,lognormal_k1}"
SEEDS="${SEEDS:-42}"
DRY_RUN="${DRY_RUN:-0}"

DATA="${DATA_ROOT}/intermittent_frozen_5000_with_split.parquet"
SPLIT_MANIFEST="${DATA_ROOT}/intermittent_frozen_5000_split_manifest.json"

[[ -x "${PYTHON_BIN}" ]]
[[ -f "${DATA}" ]]
[[ -f "${SPLIT_MANIFEST}" ]]
[[ "${SOURCE_REVISION}" =~ ^[0-9a-f]{40}$ ]]

CMD=(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/paper/scripts/run_count_aware_tpp_backbone_control.py"
  --data "${DATA}"
  --split-manifest "${SPLIT_MANIFEST}"
  --output-dir "${OUTPUT_ROOT}"
  --source-revision "${SOURCE_REVISION}"
  --execution-role "${EXECUTION_ROLE}"
  --device cuda
  --epochs 300
  --batch-size 128
  --lr 1e-3
  --lookback-weeks 520
  --max-seq-len 256
  --hidden-dim 64
  --lambda-log-qty 1.0
  --quantity-variants "${VARIANTS}"
  --quantity-sigma-floor 1e-3
  --lambda-location-huber 1.0
  --location-huber-delta 0.25
  --grad-clip 1.0
  --min-epochs 40
  --early-stopping-patience 40
  --backbones "${MODELS}"
  --seeds "${SEEDS}"
  --allow-partial-contract
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[dry_run]'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}/logs"
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_lognormal_k1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_lognormal_k1}"

exec > >(tee -a "${OUTPUT_ROOT}/logs/launcher.log") 2>&1
cd "${PROJECT_ROOT}"
"${CMD[@]}"
"${PYTHON_BIN}" "${PROJECT_ROOT}/paper/scripts/compare_count_aware_lognormal_k1_screening.py" \
  --artifact-dir "${OUTPUT_ROOT}"
