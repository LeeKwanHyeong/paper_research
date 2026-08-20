#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/sample_data/intermittent_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/search_artifacts/count_aware_external_t0_nhp_sahp_e300_20260820}"
SOURCE_REVISION="${SOURCE_REVISION:?SOURCE_REVISION must be the frozen 40-character Git revision}"
EXECUTION_ROLE="${EXECUTION_ROLE:-primary_5080_external_t0}"
DRY_RUN="${DRY_RUN:-0}"

DATA="${DATA_ROOT}/intermittent_frozen_5000_with_split.parquet"
SPLIT_MANIFEST="${DATA_ROOT}/intermittent_frozen_5000_split_manifest.json"
SOURCE_FILES=(
  "${PROJECT_ROOT}/paper/contracts/count_aware_model_baseline_v1.json"
  "${PROJECT_ROOT}/paper/scripts/run_count_aware_tpp_backbone_control.py"
  "${PROJECT_ROOT}/paper/scripts/build_count_aware_external_t0_results.py"
  "${PROJECT_ROOT}/simple_lab_test/search/scripts/run_count_aware_external_t0_nhp_sahp_e300_20260820.sh"
)

[[ -x "${PYTHON_BIN}" ]]
[[ -f "${DATA}" ]]
[[ -f "${SPLIT_MANIFEST}" ]]
[[ "${SOURCE_REVISION}" =~ ^[0-9a-f]{40}$ ]]
for source_file in "${SOURCE_FILES[@]}"; do
  [[ -f "${source_file}" ]]
done

CMD=(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/paper/scripts/run_count_aware_tpp_backbone_control.py"
  --data "${DATA}"
  --split-manifest "${SPLIT_MANIFEST}"
  --output-dir "${OUTPUT_ROOT}"
  --source-revision "${SOURCE_REVISION}"
  --execution-role "${EXECUTION_ROLE}"
  --dataset-contract intermittent_frozen_5000
  --model-role t0_common_control
  --device cuda
  --epochs 300
  --batch-size 128
  --lr 1e-3
  --lookback-weeks 520
  --max-seq-len 256
  --hidden-dim 64
  --lambda-log-qty 1
  --lambda-tail 0
  --quantity-variants log_mse
  --time-head-mode legacy_clamped_rmtpp
  --grad-clip 1
  --min-epochs 40
  --early-stopping-patience 40
  --backbones nhp,sahp
  --seeds 42,52,62
  --allow-partial-contract
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[dry_run]'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}/logs"
{
  printf 'source_revision=%s\n' "${SOURCE_REVISION}"
  printf 'generated_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sha256sum "${SOURCE_FILES[@]}"
} > "${OUTPUT_ROOT}/source_manifest.txt"

export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_external_t0}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_external_t0}"

exec > >(tee -a "${OUTPUT_ROOT}/logs/launcher.log") 2>&1
cd "${PROJECT_ROOT}"
"${CMD[@]}"
