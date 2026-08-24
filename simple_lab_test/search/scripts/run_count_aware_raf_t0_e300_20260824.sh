#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/search_artifacts/count_aware_raf_t0_e300_20260824}"
SOURCE_REVISION="${SOURCE_REVISION:?SOURCE_REVISION must be the frozen 40-character Git revision}"
DRY_RUN="${DRY_RUN:-0}"
T0_BACKBONES="rmtpp,thp,nhp,sahp,titantpp"

DATA_PATH="${PROJECT_ROOT}/benchmark_data/data/candidates/raf_spare_parts/raf_spare_parts_with_split.parquet"
SPLIT_MANIFEST="${PROJECT_ROOT}/benchmark_data/data/candidates/raf_spare_parts/raf_spare_parts_split_manifest.json"

SOURCE_FILES=(
  "${PROJECT_ROOT}/paper/contracts/count_aware_model_baseline_v2.json"
  "${PROJECT_ROOT}/paper/contracts/count_aware_model_baseline_v2.md"
  "${PROJECT_ROOT}/paper/scripts/count_aware_tpp_backbone/datasets.py"
  "${PROJECT_ROOT}/paper/scripts/run_count_aware_tpp_backbone_control.py"
  "${PROJECT_ROOT}/simple_lab_test/search/scripts/run_count_aware_raf_t0_e300_20260824.sh"
)

[[ -x "${PYTHON_BIN}" ]]
[[ "${SOURCE_REVISION}" =~ ^[0-9a-f]{40}$ ]]
[[ -f "${DATA_PATH}" ]]
[[ -f "${SPLIT_MANIFEST}" ]]
for source_file in "${SOURCE_FILES[@]}"; do
  [[ -f "${source_file}" ]]
done

COMMAND=(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/paper/scripts/run_count_aware_tpp_backbone_control.py"
  --data "${DATA_PATH}"
  --split-manifest "${SPLIT_MANIFEST}"
  --output-dir "${OUTPUT_ROOT}/raf_spare_parts/t0_common_control"
  --source-revision "${SOURCE_REVISION}"
  --execution-role primary_5080_raf_spare_parts_t0_common_control_e300
  --dataset-contract raf_spare_parts
  --model-role t0_common_control
  --device cuda
  --epochs 300
  --min-epochs 40
  --early-stopping-patience 40
  --batch-size 128
  --lr 0.001
  --lookback-weeks 84
  --max-seq-len 84
  --hidden-dim 64
  --lambda-log-qty 1
  --lambda-tail 0
  --tail-threshold 60
  --tail-normalization-scale 60
  --tail-clip-cap 200
  --tail-huber-delta 1
  --quantity-variants log_mse
  --backbones "${T0_BACKBONES}"
  --seeds 42,52,62
  --time-head-mode legacy_clamped_rmtpp
  --grad-clip 1
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[dry_run]'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}/logs"
{
  printf 'source_revision=%s\n' "${SOURCE_REVISION}"
  printf 'execution_server=5080\n'
  printf 'generated_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sha256sum "${SOURCE_FILES[@]}"
} > "${OUTPUT_ROOT}/source_manifest.txt"

export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_raf_t0_e300}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_raf_t0_e300}"

cd "${PROJECT_ROOT}"
"${COMMAND[@]}" 2>&1 | tee -a "${OUTPUT_ROOT}/logs/launcher.log"

printf '{"status":"complete","source_revision":"%s","execution_server":"5080","held_out_test_evaluated":false}\n' \
  "${SOURCE_REVISION}" > "${OUTPUT_ROOT}/status.json"
