#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/search_artifacts/count_aware_b012_seed42_screening_e300_20260827}"
SOURCE_REVISION="${SOURCE_REVISION:?SOURCE_REVISION must be the frozen 40-character Git revision}"
BACKBONES="titantpp,titantpp_titans_mac,titantpp_tpp_gated_memory"
MODEL_ROLE="titan_b012_screening"

SOURCE_FILES=(
  "${PROJECT_ROOT}/paper/contracts/count_aware_titans_backbone_reproduction_v1.json"
  "${PROJECT_ROOT}/paper/contracts/count_aware_titans_backbone_reproduction_v1.md"
  "${PROJECT_ROOT}/paper/contracts/count_aware_titan_b012_screening_v1.json"
  "${PROJECT_ROOT}/paper/contracts/count_aware_titan_b012_screening_v1.md"
  "${PROJECT_ROOT}/models/TPPs/CountAwareFactory.py"
  "${PROJECT_ROOT}/models/TPPs/CountAwareTPP.py"
  "${PROJECT_ROOT}/models/Titan/__init__.py"
  "${PROJECT_ROOT}/models/Titan/backbone.py"
  "${PROJECT_ROOT}/models/Titan/common/__init__.py"
  "${PROJECT_ROOT}/models/Titan/common/titans_mac.py"
  "${PROJECT_ROOT}/models/Titan/common/tpp_gated_memory.py"
  "${PROJECT_ROOT}/paper/scripts/compare_count_aware_b012_seed42_screening.py"
  "${PROJECT_ROOT}/paper/scripts/count_aware_tpp_backbone/constants.py"
  "${PROJECT_ROOT}/paper/scripts/count_aware_tpp_backbone/core.py"
  "${PROJECT_ROOT}/paper/scripts/count_aware_tpp_backbone/datasets.py"
  "${PROJECT_ROOT}/paper/scripts/count_aware_tpp_backbone/reporting.py"
  "${PROJECT_ROOT}/paper/scripts/count_aware_tpp_backbone/training.py"
  "${PROJECT_ROOT}/paper/scripts/run_count_aware_tpp_backbone_control.py"
  "${PROJECT_ROOT}/simple_lab_test/search/scripts/run_count_aware_b012_seed42_screening_e300_20260827.sh"
  "${PROJECT_ROOT}/simple_lab_test/search/tests/test_count_aware_b012_screening_contract.py"
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
  printf 'contract_id=count_aware_titan_b012_screening_v1\n'
  printf 'generated_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sha256sum "${SOURCE_FILES[@]}"
} > "${OUTPUT_ROOT}/source_manifest.txt"

printf '{"status":"running","source_revision":"%s","execution_server":"5080","held_out_test_evaluated":false}\n' \
  "${SOURCE_REVISION}" > "${OUTPUT_ROOT}/screening_status.json"
trap 'printf "{\"status\":\"failed\",\"source_revision\":\"%s\",\"execution_server\":\"5080\",\"held_out_test_evaluated\":false}\n" "${SOURCE_REVISION}" > "${OUTPUT_ROOT}/screening_status.json"' ERR

exec > >(tee -a "${OUTPUT_ROOT}/logs/run.log") 2>&1
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_b012_seed42_screening}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_b012_seed42_screening}"
cd "${PROJECT_ROOT}"

run_dataset() {
  local dataset_id="$1"
  local data_path="$2"
  local manifest_path="$3"
  local lookback="$4"
  local max_seq_len="$5"

  [[ -f "${data_path}" ]]
  [[ -f "${manifest_path}" ]]
  "${PYTHON_BIN}" paper/scripts/run_count_aware_tpp_backbone_control.py \
    --data "${data_path}" \
    --split-manifest "${manifest_path}" \
    --output-dir "${OUTPUT_ROOT}/${dataset_id}/${MODEL_ROLE}" \
    --source-revision "${SOURCE_REVISION}" \
    --execution-role "primary_5080_${dataset_id}_b012_seed42_e300" \
    --dataset-contract "${dataset_id}" \
    --model-role "${MODEL_ROLE}" \
    --device cuda \
    --epochs 300 \
    --min-epochs 40 \
    --early-stopping-patience 40 \
    --batch-size 128 \
    --lr 0.001 \
    --lookback-weeks "${lookback}" \
    --max-seq-len "${max_seq_len}" \
    --hidden-dim 64 \
    --lambda-log-qty 1 \
    --lambda-tail 0 \
    --quantity-variants log_mse \
    --backbones "${BACKBONES}" \
    --seeds 42 \
    --time-head-mode legacy_clamped_rmtpp \
    --grad-clip 1 \
    --allow-partial-contract \
    --force-rerun
}

run_dataset \
  intermittent_frozen_5000 \
  "${PROJECT_ROOT}/benchmark_data/data/main/intermittent_v2/intermittent_frozen_5000_with_split.parquet" \
  "${PROJECT_ROOT}/benchmark_data/data/main/intermittent_v2/intermittent_frozen_5000_split_manifest.json" \
  520 256

run_dataset \
  yellow_trip_hourly \
  "${PROJECT_ROOT}/sample_data/new_york_taxi/yellow_trip_hourly_with_split.parquet" \
  "${PROJECT_ROOT}/sample_data/new_york_taxi/yellow_trip_hourly_split_manifest.json" \
  168 256

run_dataset \
  raf_spare_parts \
  "${PROJECT_ROOT}/benchmark_data/data/candidates/raf_spare_parts/raf_spare_parts_with_split.parquet" \
  "${PROJECT_ROOT}/benchmark_data/data/candidates/raf_spare_parts/raf_spare_parts_split_manifest.json" \
  84 84

"${PYTHON_BIN}" paper/scripts/compare_count_aware_b012_seed42_screening.py \
  --artifact-root "${OUTPUT_ROOT}" \
  --source-revision "${SOURCE_REVISION}" \
  --output-dir "${OUTPUT_ROOT}/comparison"

trap - ERR
printf '{"status":"complete","source_revision":"%s","execution_server":"5080","held_out_test_evaluated":false}\n' \
  "${SOURCE_REVISION}" > "${OUTPUT_ROOT}/screening_status.json"
