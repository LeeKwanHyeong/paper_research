#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/sample_data/intermittent_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/search_artifacts/count_aware_tail_auxiliary_screening_e300_20260816}"
SOURCE_REVISION="${SOURCE_REVISION:?SOURCE_REVISION must be the frozen 40-character Git revision}"
LAMBDA_TAIL="${LAMBDA_TAIL:?LAMBDA_TAIL must be frozen by train-only calibration}"
EXECUTION_ROLE="${EXECUTION_ROLE:-primary_5080}"

DATA="${DATA_ROOT}/intermittent_frozen_5000_with_split.parquet"
SPLIT_MANIFEST="${DATA_ROOT}/intermittent_frozen_5000_split_manifest.json"
SOURCE_FILES=(
  "${PROJECT_ROOT}/paper/contracts/count_aware_tail_auxiliary_v1.json"
  "${PROJECT_ROOT}/paper/contracts/count_aware_tail_auxiliary_v1.md"
  "${PROJECT_ROOT}/paper/scripts/compare_count_aware_tail_auxiliary_screening.py"
  "${PROJECT_ROOT}/paper/scripts/run_count_aware_tpp_backbone_control.py"
  "${PROJECT_ROOT}/simple_lab_test/search/notion_writer_prompts/titantpp_count_aware_tail_auxiliary_start_0816.md"
  "${PROJECT_ROOT}/simple_lab_test/search/scripts/run_count_aware_tail_screening_e300_20260816.sh"
)

[[ -x "${PYTHON_BIN}" ]]
[[ -f "${DATA}" ]]
[[ -f "${SPLIT_MANIFEST}" ]]
[[ "${SOURCE_REVISION}" =~ ^[0-9a-f]{40}$ ]]
for source_file in "${SOURCE_FILES[@]}"; do
  [[ -f "${source_file}" ]]
done

mkdir -p "${OUTPUT_ROOT}/logs"
{
  printf 'source_revision=%s\n' "${SOURCE_REVISION}"
  printf 'lambda_tail=%s\n' "${LAMBDA_TAIL}"
  printf 'generated_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sha256sum "${SOURCE_FILES[@]}"
} > "${OUTPUT_ROOT}/source_manifest.txt"

export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_count_tail_e300}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_count_tail_e300}"

exec > >(tee -a "${OUTPUT_ROOT}/logs/launcher.log") 2>&1
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" "${PROJECT_ROOT}/paper/scripts/run_count_aware_tpp_backbone_control.py" \
  --data "${DATA}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --output-dir "${OUTPUT_ROOT}" \
  --source-revision "${SOURCE_REVISION}" \
  --execution-role "${EXECUTION_ROLE}" \
  --dataset-contract intermittent_frozen_5000 \
  --device cuda \
  --epochs 300 \
  --batch-size 128 \
  --lr 1e-3 \
  --lookback-weeks 520 \
  --max-seq-len 256 \
  --hidden-dim 64 \
  --lambda-log-qty 1 \
  --lambda-tail "${LAMBDA_TAIL}" \
  --tail-threshold 46 \
  --tail-normalization-scale 46 \
  --tail-clip-cap 187 \
  --tail-huber-delta 1 \
  --quantity-variants log_mse,tail_shared,tail_head_only \
  --grad-clip 1 \
  --min-epochs 40 \
  --early-stopping-patience 40 \
  --backbones titantpp \
  --seeds 42 \
  --allow-partial-contract

"${PYTHON_BIN}" "${PROJECT_ROOT}/paper/scripts/compare_count_aware_tail_auxiliary_screening.py" \
  --artifact-dir "${OUTPUT_ROOT}"
