#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/search_artifacts/count_aware_taxi_instacart_t0_t1_e300_20260824}"
TAXI_DATA_ROOT="${TAXI_DATA_ROOT:-${PROJECT_ROOT}/sample_data/new_york_taxi}"
INSTACART_DATA_ROOT="${INSTACART_DATA_ROOT:-${PROJECT_ROOT}/sample_data/insta_market_basket}"
SOURCE_REVISION="${SOURCE_REVISION:?SOURCE_REVISION must be the frozen 40-character Git revision}"
DATASET_FILTER="${DATASET_FILTER:-all}"
ROLE_FILTER="${ROLE_FILTER:-all}"
DRY_RUN="${DRY_RUN:-0}"
LAMBDA_TAIL="0.09111380335463036"
T0_BACKBONES="rmtpp,thp,nhp,sahp,titantpp"

SOURCE_FILES=(
  "${PROJECT_ROOT}/paper/contracts/count_aware_taxi_instacart_matched_validation_v1.json"
  "${PROJECT_ROOT}/paper/contracts/count_aware_taxi_instacart_matched_validation_v1.md"
  "${PROJECT_ROOT}/paper/scripts/count_aware_tpp_backbone/datasets.py"
  "${PROJECT_ROOT}/paper/scripts/run_count_aware_tpp_backbone_control.py"
  "${PROJECT_ROOT}/simple_lab_test/search/scripts/run_count_aware_taxi_instacart_e300_20260824.sh"
)

[[ -x "${PYTHON_BIN}" ]]
[[ "${SOURCE_REVISION}" =~ ^[0-9a-f]{40}$ ]]
[[ "${DATASET_FILTER}" =~ ^(all|yellow_trip_hourly|insta_market_basket)$ ]]
[[ "${ROLE_FILTER}" =~ ^(all|t0_common_control|t1_incumbent)$ ]]
for source_file in "${SOURCE_FILES[@]}"; do
  [[ -f "${source_file}" ]]
done

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry_run]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_role() {
  local dataset_id="$1" data_path="$2" manifest_path="$3"
  local lookback="$4" max_seq_len="$5" tail_threshold="$6" tail_cap="$7"
  local model_role="$8" quantity_variant="$9" backbones="${10}" lambda_tail="${11}"
  local partial_flag=""
  if [[ "${model_role}" == "t1_incumbent" ]]; then
    partial_flag="--allow-partial-contract"
  fi

  run_cmd "${PYTHON_BIN}" "${PROJECT_ROOT}/paper/scripts/run_count_aware_tpp_backbone_control.py" \
    --data "${data_path}" \
    --split-manifest "${manifest_path}" \
    --output-dir "${OUTPUT_ROOT}/${dataset_id}/${model_role}" \
    --source-revision "${SOURCE_REVISION}" \
    --execution-role "primary_5080_${dataset_id}_${model_role}_e300" \
    --dataset-contract "${dataset_id}" \
    --model-role "${model_role}" \
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
    --lambda-tail "${lambda_tail}" \
    --tail-threshold "${tail_threshold}" \
    --tail-normalization-scale "${tail_threshold}" \
    --tail-clip-cap "${tail_cap}" \
    --tail-huber-delta 1 \
    --quantity-variants "${quantity_variant}" \
    --backbones "${backbones}" \
    --seeds 42,52,62 \
    --time-head-mode legacy_clamped_rmtpp \
    --grad-clip 1 \
    ${partial_flag:+${partial_flag}}
}

run_dataset() {
  local dataset_id="$1" data_path="$2" manifest_path="$3"
  local lookback="$4" max_seq_len="$5" tail_threshold="$6" tail_cap="$7"
  [[ "${DATASET_FILTER}" == "all" || "${DATASET_FILTER}" == "${dataset_id}" ]] || return 0
  [[ -f "${data_path}" ]]
  [[ -f "${manifest_path}" ]]

  if [[ "${ROLE_FILTER}" == "all" || "${ROLE_FILTER}" == "t0_common_control" ]]; then
    run_role "${dataset_id}" "${data_path}" "${manifest_path}" \
      "${lookback}" "${max_seq_len}" "${tail_threshold}" "${tail_cap}" \
      t0_common_control log_mse "${T0_BACKBONES}" 0
  fi
  if [[ "${ROLE_FILTER}" == "all" || "${ROLE_FILTER}" == "t1_incumbent" ]]; then
    run_role "${dataset_id}" "${data_path}" "${manifest_path}" \
      "${lookback}" "${max_seq_len}" "${tail_threshold}" "${tail_cap}" \
      t1_incumbent tail_shared titantpp "${LAMBDA_TAIL}"
  fi
}

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${OUTPUT_ROOT}/logs"
  {
    printf 'source_revision=%s\n' "${SOURCE_REVISION}"
    printf 'execution_server=5080\n'
    printf 'generated_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    sha256sum "${SOURCE_FILES[@]}"
  } > "${OUTPUT_ROOT}/source_manifest.txt"
  exec > >(tee -a "${OUTPUT_ROOT}/logs/launcher.log") 2>&1
fi

export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_taxi_instacart_e300}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_taxi_instacart_e300}"
cd "${PROJECT_ROOT}"

run_dataset \
  yellow_trip_hourly \
  "${TAXI_DATA_ROOT}/yellow_trip_hourly_with_split.parquet" \
  "${TAXI_DATA_ROOT}/yellow_trip_hourly_split_manifest.json" \
  168 256 1562 3449

run_dataset \
  insta_market_basket \
  "${INSTACART_DATA_ROOT}/instacart_marked_target_with_split.parquet" \
  "${INSTACART_DATA_ROOT}/instacart_marked_target_split_manifest.json" \
  52 64 25 35

if [[ "${DRY_RUN}" != "1" ]]; then
  printf '{"status":"complete","source_revision":"%s","execution_server":"5080","held_out_test_evaluated":false}\n' \
    "${SOURCE_REVISION}" > "${OUTPUT_ROOT}/status.json"
fi
