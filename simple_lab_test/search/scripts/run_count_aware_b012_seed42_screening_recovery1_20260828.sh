#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python}"
SOURCE_ARTIFACT="${SOURCE_ARTIFACT:-${PROJECT_ROOT}/search_artifacts/count_aware_b012_seed42_screening_e300_20260827_final1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/search_artifacts/count_aware_b012_seed42_screening_e300_20260828_recovery1}"
SOURCE_REVISION="${SOURCE_REVISION:?SOURCE_REVISION must be the frozen training Git revision}"
RECOVERY_REVISION="${RECOVERY_REVISION:?RECOVERY_REVISION must be the recovery orchestration Git revision}"
MINIMUM_FREE_MIB="${MINIMUM_FREE_MIB:-15000}"
MAXIMUM_USED_MIB="${MAXIMUM_USED_MIB:-512}"
PREFLIGHT_ATTEMPTS="${PREFLIGHT_ATTEMPTS:-12}"
PREFLIGHT_INTERVAL_SECONDS="${PREFLIGHT_INTERVAL_SECONDS:-5}"
VERIFY_ONLY="${VERIFY_ONLY:-0}"

MODEL_ROLE="titan_b012_screening"
RECOVERY_CONTRACT="${PROJECT_ROOT}/paper/contracts/count_aware_titan_b012_screening_recovery1_v1.json"
RECOVERY_TOOL="${PROJECT_ROOT}/paper/scripts/recover_count_aware_b012_seed42_screening.py"
RUNNER="${PROJECT_ROOT}/paper/scripts/run_count_aware_tpp_backbone_control.py"
COMPARATOR="${PROJECT_ROOT}/paper/scripts/compare_count_aware_b012_seed42_screening.py"

TRAINING_RELATIVE_FILES=(
  paper/contracts/count_aware_titans_backbone_reproduction_v1.json
  paper/contracts/count_aware_titans_backbone_reproduction_v1.md
  paper/contracts/count_aware_titan_b012_screening_v1.json
  paper/contracts/count_aware_titan_b012_screening_v1.md
  models/TPPs/CountAwareFactory.py
  models/TPPs/CountAwareTPP.py
  models/Titan/__init__.py
  models/Titan/backbone.py
  models/Titan/common/__init__.py
  models/Titan/common/titans_mac.py
  models/Titan/common/tpp_gated_memory.py
  paper/scripts/count_aware_tpp_backbone/constants.py
  paper/scripts/count_aware_tpp_backbone/core.py
  paper/scripts/count_aware_tpp_backbone/datasets.py
  paper/scripts/count_aware_tpp_backbone/reporting.py
  paper/scripts/count_aware_tpp_backbone/training.py
  paper/scripts/run_count_aware_tpp_backbone_control.py
)
RECOVERY_RELATIVE_FILES=(
  paper/contracts/count_aware_titan_b012_screening_recovery1_v1.json
  paper/contracts/count_aware_titan_b012_screening_recovery1_v1.md
  paper/scripts/compare_count_aware_b012_seed42_screening.py
  paper/scripts/recover_count_aware_b012_seed42_screening.py
  simple_lab_test/search/scripts/run_count_aware_b012_seed42_screening_recovery1_20260828.sh
  simple_lab_test/search/tests/test_count_aware_b012_screening_recovery.py
)

CURRENT_DATASET=""
CURRENT_BACKBONE=""
RECOVERY_PREPARED=0
FINALIZED=0

write_recovery_status() {
  local state="$1"
  local message="$2"
  local exit_code="${3:-}"
  local args=(
    status
    --output-root "${OUTPUT_ROOT}"
    --state "${state}"
    --source-revision "${SOURCE_REVISION}"
    --recovery-revision "${RECOVERY_REVISION}"
    --message "${message}"
  )
  if [[ -n "${CURRENT_DATASET}" ]]; then
    args+=(--current-dataset "${CURRENT_DATASET}")
  fi
  if [[ -n "${CURRENT_BACKBONE}" ]]; then
    args+=(--current-backbone "${CURRENT_BACKBONE}")
  fi
  if [[ -n "${exit_code}" ]]; then
    args+=(--exit-code "${exit_code}")
  fi
  "${PYTHON_BIN}" "${RECOVERY_TOOL}" "${args[@]}"
}

on_exit() {
  local exit_code=$?
  trap - EXIT
  if [[ "${FINALIZED}" != "1" && "${RECOVERY_PREPARED}" == "1" ]]; then
    if [[ "${exit_code}" == "0" ]]; then
      exit_code=1
    fi
    set +e
    write_recovery_status \
      failed \
      "Recovery stopped before completion at ${CURRENT_DATASET:-setup}/${CURRENT_BACKBONE:-setup}." \
      "${exit_code}"
  fi
  exit "${exit_code}"
}
trap on_exit EXIT

verify_snapshot_training_files() {
  local source_manifest="${SOURCE_ARTIFACT}/source_manifest.txt"
  local relative_path
  local absolute_path
  local expected_sha
  local observed_sha

  [[ -f "${source_manifest}" ]]
  grep -qx "source_revision=${SOURCE_REVISION}" "${source_manifest}"
  for relative_path in "${TRAINING_RELATIVE_FILES[@]}"; do
    absolute_path="${PROJECT_ROOT}/${relative_path}"
    expected_sha="$(awk -v path="${absolute_path}" '$2 == path {print $1}' \
      "${source_manifest}")"
    [[ "${expected_sha}" =~ ^[0-9a-f]{64}$ ]]
    observed_sha="$(sha256sum "${absolute_path}")"
    observed_sha="${observed_sha%% *}"
    [[ "${observed_sha}" == "${expected_sha}" ]]
  done
}

export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_b012_recovery1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_b012_recovery1}"

[[ -x "${PYTHON_BIN}" ]]
[[ "${SOURCE_REVISION}" =~ ^[0-9a-f]{40}$ ]]
[[ "${RECOVERY_REVISION}" =~ ^[0-9a-f]{40}$ ]]
[[ "${VERIFY_ONLY}" =~ ^[01]$ ]]
[[ -d "${SOURCE_ARTIFACT}" ]]
[[ "${SOURCE_ARTIFACT}" != "${OUTPUT_ROOT}" ]]
for relative_path in "${TRAINING_RELATIVE_FILES[@]}" "${RECOVERY_RELATIVE_FILES[@]}"; do
  [[ -f "${PROJECT_ROOT}/${relative_path}" ]]
done

if git -C "${PROJECT_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  observed_revision="$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
  [[ "${observed_revision}" == "${RECOVERY_REVISION}" ]]
  git -C "${PROJECT_ROOT}" cat-file -e "${SOURCE_REVISION}^{commit}"
  git -C "${PROJECT_ROOT}" diff --quiet \
    "${SOURCE_REVISION}" -- "${TRAINING_RELATIVE_FILES[@]}"
  git -C "${PROJECT_ROOT}" diff --quiet \
    "${RECOVERY_REVISION}" -- "${RECOVERY_RELATIVE_FILES[@]}"
else
  verify_snapshot_training_files
fi
if [[ "${VERIFY_ONLY}" == "1" ]]; then
  printf 'Recovery source verification passed: training=%s recovery=%s\n' \
    "${SOURCE_REVISION}" "${RECOVERY_REVISION}"
  FINALIZED=1
  exit 0
fi

"${PYTHON_BIN}" "${RECOVERY_TOOL}" prepare \
  --source-artifact "${SOURCE_ARTIFACT}" \
  --output-root "${OUTPUT_ROOT}" \
  --source-revision "${SOURCE_REVISION}" \
  --recovery-revision "${RECOVERY_REVISION}" \
  --contract "${RECOVERY_CONTRACT}"
RECOVERY_PREPARED=1

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/preflight"
SOURCE_FILES=()
for relative_path in "${TRAINING_RELATIVE_FILES[@]}" "${RECOVERY_RELATIVE_FILES[@]}"; do
  SOURCE_FILES+=("${PROJECT_ROOT}/${relative_path}")
done
{
  printf 'training_source_revision=%s\n' "${SOURCE_REVISION}"
  printf 'recovery_orchestration_revision=%s\n' "${RECOVERY_REVISION}"
  printf 'execution_server=5080\n'
  printf 'contract_id=count_aware_titan_b012_screening_recovery1_v1\n'
  printf 'source_artifact=%s\n' "${SOURCE_ARTIFACT}"
  printf 'output_artifact=%s\n' "${OUTPUT_ROOT}"
  printf 'generated_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sha256sum "${SOURCE_FILES[@]}"
} > "${OUTPUT_ROOT}/source_manifest.txt"

exec > >(tee -a "${OUTPUT_ROOT}/logs/run.log") 2>&1
cd "${PROJECT_ROOT}"

run_isolated_backbone() {
  local ordinal="$1"
  local dataset_id="$2"
  local backbone="$3"
  local data_path="$4"
  local manifest_path="$5"
  local lookback="$6"
  local max_seq_len="$7"
  local shard_output="${OUTPUT_ROOT}/shards/${dataset_id}/${backbone}/${MODEL_ROLE}"
  local inspection_path="${OUTPUT_ROOT}/preflight/${ordinal}_${dataset_id}_${backbone}_shard.json"
  local gpu_path="${OUTPUT_ROOT}/preflight/${ordinal}_${dataset_id}_${backbone}_gpu.json"
  local action

  CURRENT_DATASET="${dataset_id}"
  CURRENT_BACKBONE="${backbone}"
  [[ -f "${data_path}" ]]
  [[ -f "${manifest_path}" ]]
  action="$("${PYTHON_BIN}" "${RECOVERY_TOOL}" inspect-shard \
    --output-root "${OUTPUT_ROOT}" \
    --dataset "${dataset_id}" \
    --backbone "${backbone}" \
    --source-revision "${SOURCE_REVISION}" \
    --output "${inspection_path}" \
    --action-only)"
  if [[ "${action}" == "reuse_completed" ]]; then
    printf '[reuse] %s/%s already has a validated completed shard\n' \
      "${dataset_id}" "${backbone}"
    return
  fi
  [[ \
    "${action}" == "execute_fresh" \
    || "${action}" == "resume_partial" \
    || "${action}" == "finalize_completed" \
  ]]

  write_recovery_status \
    running \
    "Starting isolated run ${ordinal}/8 (${action})."
  "${PYTHON_BIN}" "${RECOVERY_TOOL}" preflight-gpu \
    --output "${gpu_path}" \
    --dataset "${dataset_id}" \
    --backbone "${backbone}" \
    --minimum-free-mib "${MINIMUM_FREE_MIB}" \
    --maximum-used-mib "${MAXIMUM_USED_MIB}" \
    --attempts "${PREFLIGHT_ATTEMPTS}" \
    --interval-seconds "${PREFLIGHT_INTERVAL_SECONDS}"

  "${PYTHON_BIN}" "${RUNNER}" \
    --data "${data_path}" \
    --split-manifest "${manifest_path}" \
    --output-dir "${shard_output}" \
    --source-revision "${SOURCE_REVISION}" \
    --execution-role "recovery_5080_${dataset_id}_${backbone}_seed42_e300" \
    --dataset-contract "${dataset_id}" \
    --model-role experimental \
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
    --backbones "${backbone}" \
    --seeds 42 \
    --time-head-mode legacy_clamped_rmtpp \
    --grad-clip 1 \
    --allow-partial-contract

  action="$("${PYTHON_BIN}" "${RECOVERY_TOOL}" inspect-shard \
    --output-root "${OUTPUT_ROOT}" \
    --dataset "${dataset_id}" \
    --backbone "${backbone}" \
    --source-revision "${SOURCE_REVISION}" \
    --output "${inspection_path}" \
    --action-only)"
  [[ "${action}" == "reuse_completed" ]]
}

run_isolated_backbone \
  1 intermittent_frozen_5000 titantpp_titans_mac \
  "${PROJECT_ROOT}/benchmark_data/data/main/intermittent_v2/intermittent_frozen_5000_with_split.parquet" \
  "${PROJECT_ROOT}/benchmark_data/data/main/intermittent_v2/intermittent_frozen_5000_split_manifest.json" \
  520 256
run_isolated_backbone \
  2 intermittent_frozen_5000 titantpp_tpp_gated_memory \
  "${PROJECT_ROOT}/benchmark_data/data/main/intermittent_v2/intermittent_frozen_5000_with_split.parquet" \
  "${PROJECT_ROOT}/benchmark_data/data/main/intermittent_v2/intermittent_frozen_5000_split_manifest.json" \
  520 256
run_isolated_backbone \
  3 yellow_trip_hourly titantpp \
  "${PROJECT_ROOT}/sample_data/new_york_taxi/yellow_trip_hourly_with_split.parquet" \
  "${PROJECT_ROOT}/sample_data/new_york_taxi/yellow_trip_hourly_split_manifest.json" \
  168 256
run_isolated_backbone \
  4 yellow_trip_hourly titantpp_titans_mac \
  "${PROJECT_ROOT}/sample_data/new_york_taxi/yellow_trip_hourly_with_split.parquet" \
  "${PROJECT_ROOT}/sample_data/new_york_taxi/yellow_trip_hourly_split_manifest.json" \
  168 256
run_isolated_backbone \
  5 yellow_trip_hourly titantpp_tpp_gated_memory \
  "${PROJECT_ROOT}/sample_data/new_york_taxi/yellow_trip_hourly_with_split.parquet" \
  "${PROJECT_ROOT}/sample_data/new_york_taxi/yellow_trip_hourly_split_manifest.json" \
  168 256
run_isolated_backbone \
  6 raf_spare_parts titantpp \
  "${PROJECT_ROOT}/benchmark_data/data/candidates/raf_spare_parts/raf_spare_parts_with_split.parquet" \
  "${PROJECT_ROOT}/benchmark_data/data/candidates/raf_spare_parts/raf_spare_parts_split_manifest.json" \
  84 84
run_isolated_backbone \
  7 raf_spare_parts titantpp_titans_mac \
  "${PROJECT_ROOT}/benchmark_data/data/candidates/raf_spare_parts/raf_spare_parts_with_split.parquet" \
  "${PROJECT_ROOT}/benchmark_data/data/candidates/raf_spare_parts/raf_spare_parts_split_manifest.json" \
  84 84
run_isolated_backbone \
  8 raf_spare_parts titantpp_tpp_gated_memory \
  "${PROJECT_ROOT}/benchmark_data/data/candidates/raf_spare_parts/raf_spare_parts_with_split.parquet" \
  "${PROJECT_ROOT}/benchmark_data/data/candidates/raf_spare_parts/raf_spare_parts_split_manifest.json" \
  84 84

CURRENT_DATASET="merge"
CURRENT_BACKBONE="all"
"${PYTHON_BIN}" "${RECOVERY_TOOL}" merge \
  --output-root "${OUTPUT_ROOT}" \
  --source-revision "${SOURCE_REVISION}" \
  --recovery-revision "${RECOVERY_REVISION}"
"${PYTHON_BIN}" "${COMPARATOR}" \
  --artifact-root "${OUTPUT_ROOT}" \
  --source-revision "${SOURCE_REVISION}" \
  --output-dir "${OUTPUT_ROOT}/comparison"

write_recovery_status \
  complete \
  "All nine B0/B1/B2 validation runs passed recovery validation and comparison."
FINALIZED=1
