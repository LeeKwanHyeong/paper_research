#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/search_artifacts/online_retail_time_scale_gradient_audit_20260824}"
SOURCE_REVISION="${SOURCE_REVISION:?SOURCE_REVISION must be the frozen 40-character Git revision}"

SOURCE_FILES=(
  "${PROJECT_ROOT}/paper/contracts/online_retail_time_scale_gradient_audit_v1.json"
  "${PROJECT_ROOT}/paper/contracts/online_retail_time_scale_gradient_audit_v1.md"
  "${PROJECT_ROOT}/paper/scripts/audit_online_retail_time_scale_gradients.py"
  "${PROJECT_ROOT}/simple_lab_test/search/scripts/run_online_retail_time_scale_gradient_audit_20260824.sh"
  "${PROJECT_ROOT}/simple_lab_test/search/tests/test_online_retail_time_scale_gradient_audit.py"
)

[[ -x "${PYTHON_BIN}" ]]
[[ "${SOURCE_REVISION}" =~ ^[0-9a-f]{40}$ ]]
for source_file in "${SOURCE_FILES[@]}"; do
  [[ -f "${source_file}" ]]
done

mkdir -p "${OUTPUT_ROOT}/logs"
{
  printf 'source_revision=%s\n' "${SOURCE_REVISION}"
  printf 'execution_server=5080\n'
  printf 'generated_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sha256sum "${SOURCE_FILES[@]}"
} > "${OUTPUT_ROOT}/source_manifest.txt"

export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_online_retail_time_audit}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_online_retail_time_audit}"

exec > >(tee -a "${OUTPUT_ROOT}/logs/run.log") 2>&1
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m pytest \
  simple_lab_test/search/tests/test_online_retail_time_scale_gradient_audit.py \
  -q

"${PYTHON_BIN}" paper/scripts/audit_online_retail_time_scale_gradients.py \
  --data "${PROJECT_ROOT}/benchmark_data/data/main/online_retail_ii/online_retail_ii_with_split.parquet" \
  --split-manifest "${PROJECT_ROOT}/benchmark_data/data/main/online_retail_ii/online_retail_ii_split_manifest.json" \
  --output-dir "${OUTPUT_ROOT}" \
  --source-revision "${SOURCE_REVISION}" \
  --execution-role "primary_5080_online_retail_train_only_time_scale_gradient_audit" \
  --device cuda \
  --epochs 3 \
  --batch-size 128 \
  --lr 0.001 \
  --lookback-hours 8760 \
  --max-seq-len 256 \
  --hidden-dim 64 \
  --grad-clip 1.0 \
  --max-train-batches 16 \
  --force-rerun
