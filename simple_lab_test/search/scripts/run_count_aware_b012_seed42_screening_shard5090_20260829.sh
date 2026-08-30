#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
PYTHON_BIN="${PYTHON_BIN:?PYTHON_BIN is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/search_artifacts/count_aware_b012_seed42_screening_e300_20260829_shard5090}"
SOURCE_REVISION="${SOURCE_REVISION:?SOURCE_REVISION must be the frozen training Git revision}"
SHARD_REVISION="${SHARD_REVISION:?SHARD_REVISION must be the shard orchestration Git revision}"
DEPLOYMENT_METADATA="${DEPLOYMENT_METADATA:?DEPLOYMENT_METADATA is required}"
DEPLOYMENT_SHA256="${DEPLOYMENT_SHA256:?DEPLOYMENT_SHA256 is required}"
MINIMUM_FREE_MIB="${MINIMUM_FREE_MIB:-30000}"
MAXIMUM_USED_MIB="${MAXIMUM_USED_MIB:-512}"
PREFLIGHT_ATTEMPTS="${PREFLIGHT_ATTEMPTS:-12}"
PREFLIGHT_INTERVAL_SECONDS="${PREFLIGHT_INTERVAL_SECONDS:-5}"
VERIFY_ONLY="${VERIFY_ONLY:-0}"
DYNAMO_RECOMPILE_LIMIT="${DYNAMO_RECOMPILE_LIMIT:-64}"
DYNAMO_ACCUMULATED_RECOMPILE_LIMIT="${DYNAMO_ACCUMULATED_RECOMPILE_LIMIT:-512}"

MODEL_ROLE="titan_b012_screening"
EXECUTION_SERVER="5090"
SHARD_CONTRACT="${PROJECT_ROOT}/paper/contracts/count_aware_titan_b012_screening_shard5090_v1.json"
RECOVERY_TOOL="${PROJECT_ROOT}/paper/scripts/recover_count_aware_b012_seed42_screening.py"
RUNNER="${PROJECT_ROOT}/paper/scripts/run_count_aware_tpp_backbone_control.py"
DYNAMO_RUNNER="${PROJECT_ROOT}/paper/scripts/run_with_b012_dynamo_policy.py"
CURRENT_DATASET=""
CURRENT_BACKBONE=""
SHARD_PREPARED=0
FINALIZED=0

write_shard_status() {
  local state="$1"
  local message="$2"
  local exit_code="${3:-}"
  local args=(
    status
    --output-root "${OUTPUT_ROOT}"
    --state "${state}"
    --source-revision "${SOURCE_REVISION}"
    --recovery-revision "${SHARD_REVISION}"
    --execution-server "${EXECUTION_SERVER}"
    --revision-field shard_orchestration_revision
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
  if [[ "${FINALIZED}" != "1" && "${SHARD_PREPARED}" == "1" ]]; then
    if [[ "${exit_code}" == "0" ]]; then
      exit_code=1
    fi
    set +e
    write_shard_status \
      failed \
      "5090 shard stopped at ${CURRENT_DATASET:-setup}/${CURRENT_BACKBONE:-setup}." \
      "${exit_code}"
  fi
  exit "${exit_code}"
}
trap on_exit EXIT

verify_deployment() {
  "${PYTHON_BIN}" - "${DEPLOYMENT_METADATA}" \
    "${SOURCE_REVISION}" "${SHARD_REVISION}" <<'PY'
import json
import sys
from pathlib import Path

metadata = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "training_source_revision": sys.argv[2],
    "shard_orchestration_revision": sys.argv[3],
    "execution_server": "5090",
}
mismatches = {
    key: {"expected": value, "observed": metadata.get(key)}
    for key, value in expected.items()
    if metadata.get(key) != value
}
if mismatches:
    raise SystemExit(f"Deployment metadata mismatch: {mismatches}")
PY
  (
    cd "${PROJECT_ROOT}"
    sha256sum --check --strict "${DEPLOYMENT_SHA256}"
  )
}

export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_b012_shard5090}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_b012_shard5090}"

[[ -x "${PYTHON_BIN}" ]]
[[ "${SOURCE_REVISION}" =~ ^[0-9a-f]{40}$ ]]
[[ "${SHARD_REVISION}" =~ ^[0-9a-f]{40}$ ]]
[[ "${VERIFY_ONLY}" =~ ^[01]$ ]]
[[ "${DYNAMO_RECOMPILE_LIMIT}" =~ ^[1-9][0-9]*$ ]]
[[ "${DYNAMO_ACCUMULATED_RECOMPILE_LIMIT}" =~ ^[1-9][0-9]*$ ]]
(( DYNAMO_ACCUMULATED_RECOMPILE_LIMIT >= DYNAMO_RECOMPILE_LIMIT ))
[[ -f "${SHARD_CONTRACT}" ]]
[[ -f "${RECOVERY_TOOL}" ]]
[[ -f "${RUNNER}" ]]
[[ -f "${DYNAMO_RUNNER}" ]]
[[ -f "${DEPLOYMENT_METADATA}" ]]
[[ -f "${DEPLOYMENT_SHA256}" ]]
verify_deployment

if [[ "${VERIFY_ONLY}" == "1" ]]; then
  printf '5090 shard deployment verification passed: training=%s shard=%s\n' \
    "${SOURCE_REVISION}" "${SHARD_REVISION}"
  FINALIZED=1
  exit 0
fi

"${PYTHON_BIN}" "${RECOVERY_TOOL}" prepare-shard-5090 \
  --output-root "${OUTPUT_ROOT}" \
  --source-revision "${SOURCE_REVISION}" \
  --recovery-revision "${SHARD_REVISION}" \
  --contract "${SHARD_CONTRACT}" \
  --execution-server "${EXECUTION_SERVER}"
SHARD_PREPARED=1

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/preflight" \
  "${OUTPUT_ROOT}/provenance"
cp "${DEPLOYMENT_METADATA}" "${OUTPUT_ROOT}/provenance/deployment_metadata.json"
cp "${DEPLOYMENT_SHA256}" "${OUTPUT_ROOT}/provenance/deployment_files.sha256"
{
  printf 'training_source_revision=%s\n' "${SOURCE_REVISION}"
  printf 'shard_orchestration_revision=%s\n' "${SHARD_REVISION}"
  printf 'execution_server=%s\n' "${EXECUTION_SERVER}"
  printf 'contract_id=count_aware_titan_b012_screening_shard5090_v1\n'
  printf 'dynamo_recompile_limit=%s\n' "${DYNAMO_RECOMPILE_LIMIT}"
  printf 'dynamo_accumulated_recompile_limit=%s\n' \
    "${DYNAMO_ACCUMULATED_RECOMPILE_LIMIT}"
  printf 'output_artifact=%s\n' "${OUTPUT_ROOT}"
  printf 'generated_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sha256sum "${SHARD_CONTRACT}" "${RECOVERY_TOOL}" "${RUNNER}" \
    "${DYNAMO_RUNNER}"
} > "${OUTPUT_ROOT}/source_manifest.txt"

exec > >(tee -a "${OUTPUT_ROOT}/logs/run.log") 2>&1
cd "${PROJECT_ROOT}"

run_isolated_backbone() {
  local canonical_ordinal="$1"
  local dataset_id="$2"
  local backbone="$3"
  local data_path="$4"
  local manifest_path="$5"
  local lookback="$6"
  local max_seq_len="$7"
  local shard_output="${OUTPUT_ROOT}/shards/${dataset_id}/${backbone}/${MODEL_ROLE}"
  local inspection_path="${OUTPUT_ROOT}/preflight/${canonical_ordinal}_${dataset_id}_${backbone}_shard.json"
  local gpu_path="${OUTPUT_ROOT}/preflight/${canonical_ordinal}_${dataset_id}_${backbone}_gpu.json"
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
    printf '[reuse] canonical %s %s/%s\n' \
      "${canonical_ordinal}" "${dataset_id}" "${backbone}"
    return
  fi
  [[ \
    "${action}" == "execute_fresh" \
    || "${action}" == "resume_partial" \
    || "${action}" == "finalize_completed" \
  ]]

  write_shard_status \
    running \
    "Starting canonical run ${canonical_ordinal}/9 (${action})."
  "${PYTHON_BIN}" "${RECOVERY_TOOL}" preflight-gpu \
    --output "${gpu_path}" \
    --dataset "${dataset_id}" \
    --backbone "${backbone}" \
    --minimum-free-mib "${MINIMUM_FREE_MIB}" \
    --maximum-used-mib "${MAXIMUM_USED_MIB}" \
    --attempts "${PREFLIGHT_ATTEMPTS}" \
    --interval-seconds "${PREFLIGHT_INTERVAL_SECONDS}" \
    --forbidden-graphics-process gnome-shell \
    --forbidden-graphics-process Xwayland \
    --forbidden-graphics-process Xorg

  "${PYTHON_BIN}" "${DYNAMO_RUNNER}" \
    --recompile-limit "${DYNAMO_RECOMPILE_LIMIT}" \
    --accumulated-recompile-limit "${DYNAMO_ACCUMULATED_RECOMPILE_LIMIT}" \
    "${RUNNER}" \
    --data "${data_path}" \
    --split-manifest "${manifest_path}" \
    --output-dir "${shard_output}" \
    --source-revision "${SOURCE_REVISION}" \
    --execution-role "shard_5090_${dataset_id}_${backbone}_seed42_e300" \
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
  4 yellow_trip_hourly titantpp \
  "${PROJECT_ROOT}/sample_data/new_york_taxi/yellow_trip_hourly_with_split.parquet" \
  "${PROJECT_ROOT}/sample_data/new_york_taxi/yellow_trip_hourly_split_manifest.json" \
  168 256
run_isolated_backbone \
  5 yellow_trip_hourly titantpp_titans_mac \
  "${PROJECT_ROOT}/sample_data/new_york_taxi/yellow_trip_hourly_with_split.parquet" \
  "${PROJECT_ROOT}/sample_data/new_york_taxi/yellow_trip_hourly_split_manifest.json" \
  168 256
run_isolated_backbone \
  6 yellow_trip_hourly titantpp_tpp_gated_memory \
  "${PROJECT_ROOT}/sample_data/new_york_taxi/yellow_trip_hourly_with_split.parquet" \
  "${PROJECT_ROOT}/sample_data/new_york_taxi/yellow_trip_hourly_split_manifest.json" \
  168 256
run_isolated_backbone \
  7 raf_spare_parts titantpp \
  "${PROJECT_ROOT}/benchmark_data/data/candidates/raf_spare_parts/raf_spare_parts_with_split.parquet" \
  "${PROJECT_ROOT}/benchmark_data/data/candidates/raf_spare_parts/raf_spare_parts_split_manifest.json" \
  84 84
run_isolated_backbone \
  8 raf_spare_parts titantpp_titans_mac \
  "${PROJECT_ROOT}/benchmark_data/data/candidates/raf_spare_parts/raf_spare_parts_with_split.parquet" \
  "${PROJECT_ROOT}/benchmark_data/data/candidates/raf_spare_parts/raf_spare_parts_split_manifest.json" \
  84 84
run_isolated_backbone \
  9 raf_spare_parts titantpp_tpp_gated_memory \
  "${PROJECT_ROOT}/benchmark_data/data/candidates/raf_spare_parts/raf_spare_parts_with_split.parquet" \
  "${PROJECT_ROOT}/benchmark_data/data/candidates/raf_spare_parts/raf_spare_parts_split_manifest.json" \
  84 84

CURRENT_DATASET="finalize"
CURRENT_BACKBONE="all"
"${PYTHON_BIN}" "${RECOVERY_TOOL}" finalize-shard-5090 \
  --output-root "${OUTPUT_ROOT}" \
  --source-revision "${SOURCE_REVISION}" \
  --recovery-revision "${SHARD_REVISION}" \
  --contract "${SHARD_CONTRACT}" \
  --execution-server "${EXECUTION_SERVER}"
FINALIZED=1
