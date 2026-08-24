#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/search_artifacts/count_aware_taxi_instacart_t0_t1_cuda_smoke_20260824}"
SOURCE_REVISION="${SOURCE_REVISION:?SOURCE_REVISION must be the frozen 40-character Git revision}"
DRY_RUN="${DRY_RUN:-0}"
LAMBDA_TAIL="0.09111380335463036"
T0_BACKBONES="rmtpp,thp,nhp,sahp,titantpp"

SOURCE_FILES=(
  "${PROJECT_ROOT}/paper/contracts/count_aware_taxi_instacart_matched_validation_v1.json"
  "${PROJECT_ROOT}/paper/contracts/count_aware_taxi_instacart_matched_validation_v1.md"
  "${PROJECT_ROOT}/paper/scripts/count_aware_tpp_backbone/datasets.py"
  "${PROJECT_ROOT}/paper/scripts/run_count_aware_cuda_model_test.py"
  "${PROJECT_ROOT}/paper/scripts/run_count_aware_tpp_backbone_control.py"
  "${PROJECT_ROOT}/paper/scripts/validate_count_aware_taxi_instacart_smoke.py"
  "${PROJECT_ROOT}/simple_lab_test/search/scripts/run_count_aware_taxi_instacart_cuda_smoke_20260824.sh"
  "${PROJECT_ROOT}/simple_lab_test/search/tests/test_count_aware_dataset_contracts.py"
)

[[ -x "${PYTHON_BIN}" ]]
[[ "${SOURCE_REVISION}" =~ ^[0-9a-f]{40}$ ]]
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

  run_cmd "${PYTHON_BIN}" "${PROJECT_ROOT}/paper/scripts/run_count_aware_tpp_backbone_control.py" \
    --data "${data_path}" \
    --split-manifest "${manifest_path}" \
    --output-dir "${OUTPUT_ROOT}/${dataset_id}/${model_role}" \
    --source-revision "${SOURCE_REVISION}" \
    --execution-role "primary_5080_${dataset_id}_${model_role}_e1_smoke" \
    --dataset-contract "${dataset_id}" \
    --model-role "${model_role}" \
    --device cuda \
    --epochs 1 \
    --min-epochs 1 \
    --early-stopping-patience 1 \
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
    --seeds 42 \
    --time-head-mode legacy_clamped_rmtpp \
    --grad-clip 1 \
    --max-series 20 \
    --max-train-batches 2 \
    --max-val-batches 2 \
    --allow-partial-contract \
    --force-rerun
}

run_dataset() {
  local dataset_id="$1" data_path="$2" manifest_path="$3"
  local lookback="$4" max_seq_len="$5" tail_threshold="$6" tail_cap="$7"
  [[ -f "${data_path}" ]]
  [[ -f "${manifest_path}" ]]
  run_role "${dataset_id}" "${data_path}" "${manifest_path}" \
    "${lookback}" "${max_seq_len}" "${tail_threshold}" "${tail_cap}" \
    t0_common_control log_mse "${T0_BACKBONES}" 0
  run_role "${dataset_id}" "${data_path}" "${manifest_path}" \
    "${lookback}" "${max_seq_len}" "${tail_threshold}" "${tail_cap}" \
    t1_incumbent tail_shared titantpp "${LAMBDA_TAIL}"
}

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${OUTPUT_ROOT}/logs"
  {
    printf 'source_revision=%s\n' "${SOURCE_REVISION}"
    printf 'execution_server=5080\n'
    printf 'generated_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    sha256sum "${SOURCE_FILES[@]}"
  } > "${OUTPUT_ROOT}/source_manifest.txt"
  exec > >(tee -a "${OUTPUT_ROOT}/logs/run.log") 2>&1
fi

export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_taxi_instacart_smoke}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_taxi_instacart_smoke}"
cd "${PROJECT_ROOT}"

run_cmd "${PYTHON_BIN}" -m pytest \
  simple_lab_test/search/tests/test_count_aware_dataset_contracts.py \
  simple_lab_test/search/tests/test_count_aware_model_role_contract.py \
  simple_lab_test/search/tests/test_count_aware_tail_auxiliary_contract.py \
  -q
run_cmd "${PYTHON_BIN}" paper/scripts/run_count_aware_cuda_model_test.py \
  --device cuda \
  --output "${OUTPUT_ROOT}/cuda_model_test.json"

run_dataset \
  yellow_trip_hourly \
  "${PROJECT_ROOT}/benchmark_data/data/auxiliary/yellow_trip_hourly/yellow_trip_hourly_with_split.parquet" \
  "${PROJECT_ROOT}/benchmark_data/data/auxiliary/yellow_trip_hourly/yellow_trip_hourly_split_manifest.json" \
  168 256 1562 3449
run_dataset \
  insta_market_basket \
  "${PROJECT_ROOT}/benchmark_data/data/auxiliary/instacart/instacart_marked_target_with_split.parquet" \
  "${PROJECT_ROOT}/benchmark_data/data/auxiliary/instacart/instacart_marked_target_split_manifest.json" \
  52 64 25 35

if [[ "${DRY_RUN}" != "1" ]]; then
  printf '{"status":"complete","source_revision":"%s","execution_server":"5080","held_out_test_evaluated":false}\n' \
    "${SOURCE_REVISION}" > "${OUTPUT_ROOT}/status.json"
fi
