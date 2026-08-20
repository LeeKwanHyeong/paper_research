#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/search_artifacts/count_aware_time_head_v2_train_stability_20260820}"
SOURCE_REVISION="${SOURCE_REVISION:?SOURCE_REVISION must be the frozen 40-character Git revision}"
EXECUTION_ROLE="${EXECUTION_ROLE:-primary_5080_train_only_stability_selection}"

DATA="${PROJECT_ROOT}/sample_data/intermittent_v2/intermittent_frozen_5000_with_split.parquet"
SPLIT_MANIFEST="${PROJECT_ROOT}/sample_data/intermittent_v2/intermittent_frozen_5000_split_manifest.json"
SOURCE_FILES=(
  "${PROJECT_ROOT}/models/TPPs/CountAwareFactory.py"
  "${PROJECT_ROOT}/models/TPPs/CountAwareTPP.py"
  "${PROJECT_ROOT}/paper/scripts/count_aware_tpp_backbone/core.py"
  "${PROJECT_ROOT}/paper/scripts/count_aware_tpp_backbone/training.py"
  "${PROJECT_ROOT}/paper/scripts/run_count_aware_time_head_stability.py"
  "${PROJECT_ROOT}/paper/scripts/run_count_aware_tpp_backbone_control.py"
  "${PROJECT_ROOT}/simple_lab_test/search/scripts/run_count_aware_time_head_v2_stability_20260820.sh"
  "${PROJECT_ROOT}/simple_lab_test/search/tests/test_count_aware_stable_time_head_v2.py"
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
  printf 'backbone=titantpp_persistent_only\n'
  printf 'variants=H0,H1,H2_only_if_H1_fails\n'
  printf 'evaluation_scope=train_only\n'
  printf 'epochs=3\n'
  printf 'batch_size=128\n'
  printf 'generated_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sha256sum "${SOURCE_FILES[@]}"
} > "${OUTPUT_ROOT}/source_manifest.txt"

export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_time_head_v2_stability}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_time_head_v2_stability}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_time_head_v2_stability}"

exec > >(tee -a "${OUTPUT_ROOT}/logs/launcher.log") 2>&1
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/paper/scripts/run_count_aware_time_head_stability.py" \
  --data "${DATA}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --output-dir "${OUTPUT_ROOT}" \
  --source-revision "${SOURCE_REVISION}" \
  --execution-role "${EXECUTION_ROLE}" \
  --device cuda \
  --epochs 3 \
  --batch-size 128 \
  --lr 1e-3 \
  --lookback-weeks 520 \
  --max-seq-len 256 \
  --hidden-dim 64 \
  --grad-clip 1
