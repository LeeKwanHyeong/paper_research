#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/search_artifacts/count_aware_tail_cuda_smoke_20260816}"
SOURCE_REVISION="${SOURCE_REVISION:?SOURCE_REVISION must be the frozen 40-character Git revision}"
LAMBDA_TAIL="${LAMBDA_TAIL:?LAMBDA_TAIL must be frozen by train-only calibration}"
VARIANTS="log_mse,tail_shared,tail_head_only"

INTER_DATA="${PROJECT_ROOT}/sample_data/intermittent_v2/intermittent_frozen_5000_with_split.parquet"
INTER_MANIFEST="${PROJECT_ROOT}/sample_data/intermittent_v2/intermittent_frozen_5000_split_manifest.json"
INSTA_DATA="${PROJECT_ROOT}/sample_data/insta_market_basket/instacart_marked_target_with_split.parquet"
INSTA_MANIFEST="${PROJECT_ROOT}/sample_data/insta_market_basket/instacart_marked_target_split_manifest.json"
SOURCE_FILES=(
  "${PROJECT_ROOT}/paper/contracts/count_aware_tail_auxiliary_v1.json"
  "${PROJECT_ROOT}/paper/scripts/run_count_aware_tpp_backbone_control.py"
  "${PROJECT_ROOT}/simple_lab_test/search/scripts/run_count_aware_tail_cuda_smoke_20260816.sh"
  "${PROJECT_ROOT}/simple_lab_test/search/tests/test_count_aware_tail_auxiliary_contract.py"
  "${PROJECT_ROOT}/simple_lab_test/search/tests/test_count_aware_tail_lambda_calibration.py"
)

[[ -x "${PYTHON_BIN}" ]]
[[ "${SOURCE_REVISION}" =~ ^[0-9a-f]{40}$ ]]
for input_file in "${INTER_DATA}" "${INTER_MANIFEST}" "${INSTA_DATA}" "${INSTA_MANIFEST}" "${SOURCE_FILES[@]}"; do
  [[ -f "${input_file}" ]]
done

mkdir -p "${OUTPUT_ROOT}/logs"
{
  printf 'source_revision=%s\n' "${SOURCE_REVISION}"
  printf 'lambda_tail=%s\n' "${LAMBDA_TAIL}"
  printf 'generated_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sha256sum "${SOURCE_FILES[@]}"
} > "${OUTPUT_ROOT}/source_manifest.txt"

export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_count_tail_smoke}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_count_tail_smoke}"

exec > >(tee -a "${OUTPUT_ROOT}/logs/smoke.log") 2>&1
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m pytest \
  simple_lab_test/search/tests/test_count_aware_tpp_contract.py \
  simple_lab_test/search/tests/test_count_aware_tail_auxiliary_contract.py \
  simple_lab_test/search/tests/test_count_aware_tail_lambda_calibration.py \
  simple_lab_test/search/tests/test_count_aware_tail_auxiliary_comparator.py \
  -q

"${PYTHON_BIN}" "${PROJECT_ROOT}/paper/scripts/run_count_aware_tpp_backbone_control.py" \
  --data "${INTER_DATA}" \
  --split-manifest "${INTER_MANIFEST}" \
  --output-dir "${OUTPUT_ROOT}/intermittent_cuda_model_test" \
  --source-revision "${SOURCE_REVISION}" \
  --execution-role primary_5080_cuda_model_test \
  --dataset-contract intermittent_frozen_5000 \
  --device cuda \
  --epochs 1 \
  --batch-size 8 \
  --lookback-weeks 520 \
  --max-seq-len 256 \
  --hidden-dim 64 \
  --lambda-log-qty 1 \
  --lambda-tail "${LAMBDA_TAIL}" \
  --quantity-variants "${VARIANTS}" \
  --backbones titantpp \
  --seeds 42 \
  --max-train-batches 2 \
  --max-val-batches 2 \
  --allow-partial-contract \
  --force-rerun

"${PYTHON_BIN}" "${PROJECT_ROOT}/paper/scripts/run_count_aware_tpp_backbone_control.py" \
  --data "${INSTA_DATA}" \
  --split-manifest "${INSTA_MANIFEST}" \
  --output-dir "${OUTPUT_ROOT}/instacart_top20_e1" \
  --source-revision "${SOURCE_REVISION}" \
  --execution-role primary_5080_instacart_smoke \
  --dataset-contract insta_market_basket \
  --max-series 20 \
  --device cuda \
  --epochs 1 \
  --batch-size 16 \
  --lookback-weeks 520 \
  --max-seq-len 64 \
  --hidden-dim 64 \
  --lambda-log-qty 1 \
  --lambda-tail "${LAMBDA_TAIL}" \
  --quantity-variants "${VARIANTS}" \
  --backbones titantpp \
  --seeds 42 \
  --allow-partial-contract \
  --force-rerun

printf '{"status":"complete","source_revision":"%s","lambda_tail":%s}\n' \
  "${SOURCE_REVISION}" "${LAMBDA_TAIL}" > "${OUTPUT_ROOT}/status.json"
