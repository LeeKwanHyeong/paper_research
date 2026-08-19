#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/sample_data/intermittent_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/search_artifacts/titantpp_persistent_dual_scaled_time_screening_e300_20260819}"
SOURCE_REVISION="${SOURCE_REVISION:?SOURCE_REVISION must be the frozen 40-character Git revision}"
EXECUTION_ROLE="${EXECUTION_ROLE:-primary_5080_seed42_validation_screening}"
DRY_RUN="${DRY_RUN:-0}"
BACKBONES="titantpp_persistent_only,titantpp,titantpp_persistent_surprise_memory,titantpp_dual_memory_shared,titantpp_dual_memory_adapter_only"

DATA="${DATA_ROOT}/intermittent_frozen_5000_with_split.parquet"
SPLIT_MANIFEST="${DATA_ROOT}/intermittent_frozen_5000_split_manifest.json"
SOURCE_FILES=(
  "${PROJECT_ROOT}/models/TPPs/CountAwareFactory.py"
  "${PROJECT_ROOT}/models/TPPs/CountAwareTPP.py"
  "${PROJECT_ROOT}/models/Titan/common/memory.py"
  "${PROJECT_ROOT}/paper/scripts/compare_titantpp_persistent_dual_screening.py"
  "${PROJECT_ROOT}/paper/scripts/count_aware_tpp_backbone/constants.py"
  "${PROJECT_ROOT}/paper/scripts/count_aware_tpp_backbone/core.py"
  "${PROJECT_ROOT}/paper/scripts/count_aware_tpp_backbone/reporting.py"
  "${PROJECT_ROOT}/paper/scripts/count_aware_tpp_backbone/training.py"
  "${PROJECT_ROOT}/paper/scripts/run_count_aware_tpp_backbone_control.py"
  "${PROJECT_ROOT}/simple_lab_test/search/scripts/run_titantpp_persistent_dual_scaled_time_e300_20260819.sh"
  "${PROJECT_ROOT}/simple_lab_test/search/tests/test_count_aware_scaled_exact_time_head.py"
  "${PROJECT_ROOT}/simple_lab_test/search/tests/test_count_aware_titan_memory_backbones.py"
  "${PROJECT_ROOT}/simple_lab_test/search/tests/test_count_aware_titan_persistent_dual_memory.py"
  "${PROJECT_ROOT}/simple_lab_test/search/tests/test_titantpp_persistent_dual_comparator.py"
)

[[ -x "${PYTHON_BIN}" ]]
[[ -f "${DATA}" ]]
[[ -f "${SPLIT_MANIFEST}" ]]
[[ "${SOURCE_REVISION}" =~ ^[0-9a-f]{40}$ ]]
for source_file in "${SOURCE_FILES[@]}"; do
  [[ -f "${source_file}" ]]
done

TRAIN_CMD=(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/paper/scripts/run_count_aware_tpp_backbone_control.py"
  --data "${DATA}"
  --split-manifest "${SPLIT_MANIFEST}"
  --output-dir "${OUTPUT_ROOT}"
  --source-revision "${SOURCE_REVISION}"
  --execution-role "${EXECUTION_ROLE}"
  --dataset-contract intermittent_frozen_5000
  --device cuda
  --epochs 300
  --batch-size 128
  --lr 1e-3
  --lookback-weeks 520
  --max-seq-len 256
  --hidden-dim 64
  --lambda-log-qty 1
  --quantity-variants log_mse
  --time-head-mode scaled_exact_rmtpp
  --time-scale 3
  --time-w-max 3.3333333333333335
  --time-intercept-limit 30
  --grad-clip 1
  --min-epochs 40
  --early-stopping-patience 40
  --backbones "${BACKBONES}"
  --seeds 42
  --allow-partial-contract
)
COMPARE_CMD=(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/paper/scripts/compare_titantpp_persistent_dual_screening.py"
  --artifact-dir "${OUTPUT_ROOT}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[train]'
  printf ' %q' "${TRAIN_CMD[@]}"
  printf '\n[compare]'
  printf ' %q' "${COMPARE_CMD[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}/logs"
{
  printf 'source_revision=%s\n' "${SOURCE_REVISION}"
  printf 'backbones=%s\n' "${BACKBONES}"
  printf 'quantity_variant=log_mse\n'
  printf 'time_head=scaled_exact_rmtpp\n'
  printf 'time_scale=3\n'
  printf 'time_w_max=3.3333333333333335\n'
  printf 'generated_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sha256sum "${SOURCE_FILES[@]}"
} > "${OUTPUT_ROOT}/source_manifest.txt"

export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_persistent_dual_e300}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_persistent_dual_e300}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_persistent_dual_e300}"

exec > >(tee -a "${OUTPUT_ROOT}/logs/launcher.log") 2>&1
cd "${PROJECT_ROOT}"
"${TRAIN_CMD[@]}"
"${COMPARE_CMD[@]}"
printf '{"status":"complete","source_revision":"%s","held_out_test_evaluated":false}\n' \
  "${SOURCE_REVISION}" > "${OUTPUT_ROOT}/status.json"
