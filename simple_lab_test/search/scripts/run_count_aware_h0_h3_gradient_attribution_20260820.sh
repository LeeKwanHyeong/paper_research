#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/search_artifacts/count_aware_h0_h3_gradient_attribution_20260820}"
SOURCE_REVISION="${SOURCE_REVISION:?SOURCE_REVISION must be the frozen 40-character Git revision}"
EXECUTION_ROLE="${EXECUTION_ROLE:-primary_5080_train_only_h0_h3_gradient_attribution}"

DATA="${PROJECT_ROOT}/sample_data/intermittent_v2/intermittent_frozen_5000_with_split.parquet"
SPLIT_MANIFEST="${PROJECT_ROOT}/sample_data/intermittent_v2/intermittent_frozen_5000_split_manifest.json"
INTEGRATION_ARTIFACT="${PROJECT_ROOT}/search_artifacts/count_aware_final_time_t1_integration_e300_20260820"
SOURCE_FILES=(
  "${PROJECT_ROOT}/models/TPPs/CountAwareFactory.py"
  "${PROJECT_ROOT}/models/TPPs/CountAwareTPP.py"
  "${PROJECT_ROOT}/paper/scripts/count_aware_tpp_backbone/core.py"
  "${PROJECT_ROOT}/paper/scripts/audit_count_aware_h0_h3_gradient_attribution.py"
  "${PROJECT_ROOT}/simple_lab_test/search/scripts/run_count_aware_h0_h3_gradient_attribution_20260820.sh"
  "${PROJECT_ROOT}/simple_lab_test/search/tests/test_count_aware_h0_h3_gradient_attribution.py"
)

[[ -x "${PYTHON_BIN}" ]]
[[ -f "${DATA}" ]]
[[ -f "${SPLIT_MANIFEST}" ]]
[[ -d "${INTEGRATION_ARTIFACT}" ]]
[[ "${SOURCE_REVISION}" =~ ^[0-9a-f]{40}$ ]]
for source_file in "${SOURCE_FILES[@]}"; do
  [[ -f "${source_file}" ]]
done

mkdir -p "${OUTPUT_ROOT}/logs"
{
  printf 'source_revision=%s\n' "${SOURCE_REVISION}"
  printf 'source_integration_artifact=%s\n' "${INTEGRATION_ARTIFACT}"
  printf 'variants=H0_scaled_exact,H3_lognormal_duration\n'
  printf 'checkpoint_stages=initial,best,final\n'
  printf 'evaluation_scope=train_only\n'
  printf 'batch_size=128\n'
  printf 'audit_batches=32\n'
  printf 'generated_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sha256sum "${SOURCE_FILES[@]}"
} > "${OUTPUT_ROOT}/source_manifest.txt"

export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_h0_h3_gradient_attribution}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_h0_h3_gradient_attribution}"

exec > >(tee -a "${OUTPUT_ROOT}/logs/launcher.log") 2>&1
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/paper/scripts/audit_count_aware_h0_h3_gradient_attribution.py" \
  --data "${DATA}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --integration-artifact "${INTEGRATION_ARTIFACT}" \
  --output-dir "${OUTPUT_ROOT}" \
  --source-revision "${SOURCE_REVISION}" \
  --execution-role "${EXECUTION_ROLE}" \
  --device cuda \
  --batch-size 128 \
  --lookback-weeks 520 \
  --max-seq-len 256 \
  --audit-batches 32
