#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research_titantpp_mac_08e5988}"
PYTHON_BIN="${PYTHON_BIN:-/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/search_artifacts/count_aware_titantpp_mac_three_seed_validation_20260830_5090}"
ORCHESTRATION_REVISION="${ORCHESTRATION_REVISION:?ORCHESTRATION_REVISION is required}"
VERIFY_ONLY="${VERIFY_ONLY:-0}"
MINIMUM_FREE_MIB="${MINIMUM_FREE_MIB:-12000}"

SOURCE_REVISION="08e59880cd61cbd27cec40aa04636452b87bebfc"
CONTRACT="${PROJECT_ROOT}/paper/contracts/count_aware_titantpp_mac_three_seed_validation_v1.json"
AMENDMENT="${PROJECT_ROOT}/paper/contracts/count_aware_titantpp_mac_primary_v1_amendment_1.json"
VALIDATOR="${PROJECT_ROOT}/paper/scripts/validate_count_aware_titantpp_mac_three_seed_validation.py"
POLICY_RUNNER="${PROJECT_ROOT}/paper/scripts/run_with_titantpp_mac_dynamo_policy.py"
TRAINING_RUNNER="${PROJECT_ROOT}/paper/scripts/run_count_aware_tpp_backbone_control.py"
LAUNCHER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
BACKBONE="titantpp_titans_mac"

CURRENT_DATASET=""
CURRENT_SEED=""
COMPLETED_RUNS=0
FINALIZED=0

write_status() {
  local state="$1"
  local message="$2"
  local args=(
    status
    --output-root "${OUTPUT_ROOT}"
    --state "${state}"
    --orchestration-revision "${ORCHESTRATION_REVISION}"
    --completed-run-count "${COMPLETED_RUNS}"
    --message "${message}"
  )
  if [[ -n "${CURRENT_DATASET}" ]]; then
    args+=(--current-dataset "${CURRENT_DATASET}")
  fi
  if [[ -n "${CURRENT_SEED}" ]]; then
    args+=(--current-seed "${CURRENT_SEED}")
  fi
  "${PYTHON_BIN}" "${VALIDATOR}" "${args[@]}"
}

on_exit() {
  local exit_code=$?
  trap - EXIT
  if [[ "${FINALIZED}" != "1" ]]; then
    set +e
    write_status failed \
      "TitanTPP-MAC validation stopped at ${CURRENT_DATASET:-setup}/seed_${CURRENT_SEED:-none}."
  fi
  exit "${exit_code}"
}
trap on_exit EXIT

[[ -x "${PYTHON_BIN}" ]]
[[ "${ORCHESTRATION_REVISION}" =~ ^[0-9a-f]{40}$ ]]
[[ "${VERIFY_ONLY}" =~ ^[01]$ ]]
for path in \
  "${CONTRACT}" \
  "${AMENDMENT}" \
  "${VALIDATOR}" \
  "${POLICY_RUNNER}" \
  "${TRAINING_RUNNER}" \
  "${LAUNCHER}"; do
  [[ -f "${path}" ]]
done

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/preflight"
"${PYTHON_BIN}" "${VALIDATOR}" verify-source \
  --project-root "${PROJECT_ROOT}" \
  --contract "${CONTRACT}" \
  --output "${OUTPUT_ROOT}/preflight/frozen_source_validation.json"

if [[ "${VERIFY_ONLY}" == "1" ]]; then
  printf 'TitanTPP-MAC source verification passed.\n'
  FINALIZED=1
  exit 0
fi

{
  printf 'model_name=TitanTPP-MAC\n'
  printf 'training_source_revision=%s\n' "${SOURCE_REVISION}"
  printf 'orchestration_revision=%s\n' "${ORCHESTRATION_REVISION}"
  printf 'execution_server=5090\n'
  printf 'contract_id=count_aware_titantpp_mac_three_seed_validation_v1\n'
  printf 'generated_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sha256sum "${CONTRACT}" "${AMENDMENT}" "${VALIDATOR}" \
    "${POLICY_RUNNER}" "${TRAINING_RUNNER}" "${LAUNCHER}"
} > "${OUTPUT_ROOT}/source_manifest.txt"

exec > >(tee -a "${OUTPUT_ROOT}/logs/launcher.log") 2>&1
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_titantpp_mac_three_seed}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_titantpp_mac_three_seed}"
cd "${PROJECT_ROOT}"

gpu_preflight() {
  local free_mib
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  [[ "${free_mib}" =~ ^[0-9]+$ ]]
  if (( free_mib < MINIMUM_FREE_MIB )); then
    printf 'Insufficient free VRAM: required=%s MiB observed=%s MiB\n' \
      "${MINIMUM_FREE_MIB}" "${free_mib}" >&2
    return 1
  fi
  if pgrep -af 'gnome-shell|Xwayland' | grep -v grep >/dev/null; then
    printf 'Desktop GPU process detected before training.\n' >&2
    return 1
  fi
}

run_one() {
  local ordinal="$1"
  local dataset="$2"
  local seed="$3"
  local data_path="$4"
  local split_manifest="$5"
  local lookback="$6"
  local max_seq_len="$7"
  local run_root="${OUTPUT_ROOT}/shards/${dataset}/seed_${seed}"

  CURRENT_DATASET="${dataset}"
  CURRENT_SEED="${seed}"
  [[ -f "${data_path}" ]]
  [[ -f "${split_manifest}" ]]

  if "${PYTHON_BIN}" "${VALIDATOR}" validate-run \
    --run-root "${run_root}" \
    --dataset "${dataset}" \
    --seed "${seed}" \
    --contract "${CONTRACT}" >/dev/null 2>&1; then
    COMPLETED_RUNS=$((COMPLETED_RUNS + 1))
    write_status running \
      "Reused validated TitanTPP-MAC run ${ordinal}/9."
    return
  fi

  gpu_preflight
  write_status running \
    "Starting isolated TitanTPP-MAC run ${ordinal}/9."
  "${PYTHON_BIN}" "${POLICY_RUNNER}" \
    --recompile-limit 64 \
    --accumulated-recompile-limit 512 \
    "${TRAINING_RUNNER}" \
    --data "${data_path}" \
    --split-manifest "${split_manifest}" \
    --output-dir "${run_root}" \
    --source-revision "${SOURCE_REVISION}" \
    --execution-role "primary_5090_titantpp_mac_${dataset}_seed${seed}_e300" \
    --dataset-contract "${dataset}" \
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
    --backbones "${BACKBONE}" \
    --seeds "${seed}" \
    --time-head-mode legacy_clamped_rmtpp \
    --grad-clip 1 \
    --allow-partial-contract

  "${PYTHON_BIN}" "${VALIDATOR}" validate-run \
    --run-root "${run_root}" \
    --dataset "${dataset}" \
    --seed "${seed}" \
    --contract "${CONTRACT}"
  COMPLETED_RUNS=$((COMPLETED_RUNS + 1))
  write_status running \
    "Validated TitanTPP-MAC run ${ordinal}/9."
}

INTERMITTENT_ROOT="${PROJECT_ROOT}/sample_data/intermittent_v2"
TAXI_ROOT="${PROJECT_ROOT}/sample_data/new_york_taxi"
RAF_ROOT="${PROJECT_ROOT}/benchmark_data/data/candidates/raf_spare_parts"
INSTACART_ROOT="${PROJECT_ROOT}/sample_data/insta_market_basket"

write_status running "TitanTPP-MAC nine-run validation launcher started."
run_one 1 raf_spare_parts 52 \
  "${RAF_ROOT}/raf_spare_parts_with_split.parquet" \
  "${RAF_ROOT}/raf_spare_parts_split_manifest.json" 84 84
run_one 2 raf_spare_parts 62 \
  "${RAF_ROOT}/raf_spare_parts_with_split.parquet" \
  "${RAF_ROOT}/raf_spare_parts_split_manifest.json" 84 84
run_one 3 yellow_trip_hourly 52 \
  "${TAXI_ROOT}/yellow_trip_hourly_with_split.parquet" \
  "${TAXI_ROOT}/yellow_trip_hourly_split_manifest.json" 168 256
run_one 4 yellow_trip_hourly 62 \
  "${TAXI_ROOT}/yellow_trip_hourly_with_split.parquet" \
  "${TAXI_ROOT}/yellow_trip_hourly_split_manifest.json" 168 256
run_one 5 insta_market_basket 42 \
  "${INSTACART_ROOT}/instacart_marked_target_with_split.parquet" \
  "${INSTACART_ROOT}/instacart_marked_target_split_manifest.json" 52 64
run_one 6 insta_market_basket 52 \
  "${INSTACART_ROOT}/instacart_marked_target_with_split.parquet" \
  "${INSTACART_ROOT}/instacart_marked_target_split_manifest.json" 52 64
run_one 7 insta_market_basket 62 \
  "${INSTACART_ROOT}/instacart_marked_target_with_split.parquet" \
  "${INSTACART_ROOT}/instacart_marked_target_split_manifest.json" 52 64
run_one 8 intermittent_frozen_5000 52 \
  "${INTERMITTENT_ROOT}/intermittent_frozen_5000_with_split.parquet" \
  "${INTERMITTENT_ROOT}/intermittent_frozen_5000_split_manifest.json" 520 256
run_one 9 intermittent_frozen_5000 62 \
  "${INTERMITTENT_ROOT}/intermittent_frozen_5000_with_split.parquet" \
  "${INTERMITTENT_ROOT}/intermittent_frozen_5000_split_manifest.json" 520 256

CURRENT_DATASET="finalize"
CURRENT_SEED=""
"${PYTHON_BIN}" "${VALIDATOR}" finalize \
  --output-root "${OUTPUT_ROOT}" \
  --contract "${CONTRACT}"
write_status complete "All nine TitanTPP-MAC validation runs passed validation."
FINALIZED=1
