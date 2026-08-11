#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:?PYTHON_BIN must point to the ai_env Python executable}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/sample_data/intermittent_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/search_artifacts/count_aware_tpp_backbone_control_e300_20260812}"
MODELS="${MODELS:?MODELS must be a comma-separated backbone list}"
SEEDS="${SEEDS:-42,52,62}"
EXECUTION_ROLE="${EXECUTION_ROLE:?EXECUTION_ROLE is required}"
SOURCE_REVISION="${SOURCE_REVISION:?SOURCE_REVISION must be the frozen 40-character Git revision}"
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MIN_EPOCHS="${MIN_EPOCHS:-40}"
PATIENCE="${PATIENCE:-40}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-}"
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
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr 1e-3
  --lookback-weeks 520
  --max-seq-len 256
  --hidden-dim 64
  --lambda-log-qty 1.0
  --grad-clip 1.0
  --min-epochs "${MIN_EPOCHS}"
  --early-stopping-patience "${PATIENCE}"
  --backbones "${MODELS}"
  --seeds "${SEEDS}"
  --allow-partial-contract
)

if [[ -n "${MAX_TRAIN_BATCHES}" ]]; then
  CMD+=(--max-train-batches "${MAX_TRAIN_BATCHES}")
fi
if [[ -n "${MAX_VAL_BATCHES}" ]]; then
  CMD+=(--max-val-batches "${MAX_VAL_BATCHES}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[dry_run]'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}/logs"
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_count_aware_tpp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_count_aware_tpp}"

exec > >(tee -a "${OUTPUT_ROOT}/logs/launcher.log") 2>&1
cd "${PROJECT_ROOT}"
"${CMD[@]}"
