#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ai_env/bin/python}"
ENTRYPOINT="${PROJECT_ROOT}/simple_lab_test/search/tpp_experiment.py"
VALIDATOR="${PROJECT_ROOT}/simple_lab_test/search/validate_titantpp_v4_cuda_insta_smoke.py"
EXPERIMENT_ID="model_enhancement_titantpp_v4_cuda_insta_smoke_0716"
OUTPUT_ROOT="${PROJECT_ROOT}/search_artifacts/${EXPERIMENT_ID}"
LOG_DIR="${OUTPUT_ROOT}/logs"
LOG_PATH="${LOG_DIR}/run.log"
STATUS_PATH="${OUTPUT_ROOT}/variant_status.tsv"
CUDA13_LIB="/opt/miniconda3/envs/ai_env/lib/python3.12/site-packages/nvidia/cu13/lib"

export LD_LIBRARY_PATH="${CUDA13_LIB}:/opt/miniconda3/envs/ai_env/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="${OUTPUT_ROOT}/cache-runtime"
export MPLCONFIGDIR="${OUTPUT_ROOT}/matplotlib-cache"

mkdir -p "${LOG_DIR}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"
cd "${PROJECT_ROOT}"

if [[ -z "${SOURCE_REVISION:-}" ]]; then
  echo "[preflight_error] SOURCE_REVISION is required"
  exit 2
fi

cat > "${OUTPUT_ROOT}/experiment_manifest.json" <<JSON
{
  "experiment_id": "${EXPERIMENT_ID}",
  "started_at": "$(date '+%Y-%m-%d %H:%M:%S %Z')",
  "server": "$(hostname)",
  "device": "cuda",
  "source_revision": "${SOURCE_REVISION}",
  "stages": ["V4a/V4b CUDA model-test", "V4a/V4b Instacart top-20 e1 smoke"],
  "dataset": "insta_market_basket",
  "insta_max_series": 20,
  "expected_loader_samples": {"train": 1380, "validation": 300, "test": 300},
  "split_mode": "fixed",
  "evaluation_scope": "validation_only",
  "epochs": 1,
  "seed": 42,
  "candidate": "mid_lmm",
  "variants": {
    "v4a_shared_value_mark_time": {"value_head_mode": "shared", "qty_mark_gradient_mode": "coupled", "time_head_mode": "mark_conditioned"},
    "v4b_mark_value_mark_time": {"value_head_mode": "mark_conditioned_experts", "qty_mark_gradient_mode": "detached", "time_head_mode": "mark_conditioned"}
  },
  "acceptance_scope": "CUDA, forward/loss, one-epoch train/validation, artifact identity, and held-out lock only",
  "performance_ranking_allowed": false
}
JSON
printf 'stage_variant\texit_code\n' > "${STATUS_PATH}"

run_model_test() {
  local variant="$1"
  local value_mode="$2"
  local mark_gradient="$3"
  local output_dir="${OUTPUT_ROOT}/cuda_model_test/${variant}"

  echo "[model_test_start] ${variant} $(date '+%Y-%m-%d %H:%M:%S %Z')"
  "${PYTHON_BIN}" "${ENTRYPOINT}" model-test \
    --output-dir "${output_dir}" \
    --models titantpp \
    --titan-candidates mid_lmm \
    --device cuda \
    --seed 42 \
    --batch-size 4 \
    --seq-len 256 \
    --num-marks 5 \
    --scale-base 10 \
    --value-head-mode "${value_mode}" \
    --time-head-mode mark_conditioned \
    --qty-mark-gradient-mode "${mark_gradient}" \
    --value-encoder-gradient-mode coupled \
    --left-pad \
    --stop-on-error
  local status=$?
  printf 'model_test:%s\t%s\n' "${variant}" "${status}" >> "${STATUS_PATH}"
  echo "[model_test_end] ${variant} exit_code=${status} $(date '+%Y-%m-%d %H:%M:%S %Z')"
  return "${status}"
}

run_insta_smoke() {
  local variant="$1"
  local value_mode="$2"
  local mark_gradient="$3"
  local variant_dir="${OUTPUT_ROOT}/insta_smoke/${variant}"

  mkdir -p "${variant_dir}"
  echo "[insta_smoke_start] ${variant} $(date '+%Y-%m-%d %H:%M:%S %Z')"
  "${PYTHON_BIN}" "${ENTRYPOINT}" long-epoch \
    --base-dir "${variant_dir}" \
    --datasets insta_market_basket \
    --models titantpp \
    --titan-profile dataset_best \
    --titan-candidates mid_lmm \
    --epochs 1 \
    --seeds 42 \
    --lr 1e-3 \
    --lambda-dt 1.0 \
    --batch-size 16 \
    --lookback-weeks 10 \
    --max-seq-len 64 \
    --insta-max-series 20 \
    --split-mode fixed \
    --evaluation-scope validation_only \
    --reproducibility-mode strict \
    --value-head-activation identity \
    --value-head-mode "${value_mode}" \
    --time-head-mode mark_conditioned \
    --qty-mark-gradient-mode "${mark_gradient}" \
    --value-encoder-gradient-mode coupled \
    --value-input-mode residual \
    --train-loss-scope target_only \
    --loss-mode hybrid \
    --marker-loss-mode ce \
    --lambda-ordinal 0 \
    --qty-decoder-mode mark_residual \
    --test-time-memory none \
    --analysis-scale-base 10 \
    --analysis-tail-order 4 \
    --eval-selections best_val_nll \
    --device cuda \
    --force-rerun \
    --stop-on-error \
    2>&1 | tee "${variant_dir}/run.log"
  local status=${PIPESTATUS[0]}
  printf 'insta_smoke:%s\t%s\n' "${variant}" "${status}" >> "${STATUS_PATH}"
  echo "[insta_smoke_end] ${variant} exit_code=${status} $(date '+%Y-%m-%d %H:%M:%S %Z')"
  return "${status}"
}

{
  echo "[start] $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "[server] $(hostname)"
  echo "[source_revision] ${SOURCE_REVISION}"
  echo "[python] ${PYTHON_BIN}"
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader
  "${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(); print(f"[torch] {torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")' || exit 2

  overall_status=0
  run_model_test v4a_shared_value_mark_time shared coupled || overall_status=1
  run_model_test v4b_mark_value_mark_time mark_conditioned_experts detached || overall_status=1
  if [[ "${overall_status}" -eq 0 ]]; then
    run_insta_smoke v4a_shared_value_mark_time shared coupled || overall_status=1
    run_insta_smoke v4b_mark_value_mark_time mark_conditioned_experts detached || overall_status=1
  fi
  if [[ "${overall_status}" -eq 0 ]]; then
    "${PYTHON_BIN}" "${VALIDATOR}" --artifact-root "${OUTPUT_ROOT}" || overall_status=1
  fi

  echo "[end] $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "[exit_code] ${overall_status}"
  exit "${overall_status}"
} 2>&1 | tee "${LOG_PATH}"

status=${PIPESTATUS[0]}
if [[ "${status}" -eq 0 ]]; then
  touch "${OUTPUT_ROOT}/INTEGRATION_SUCCESS"
else
  touch "${OUTPUT_ROOT}/INTEGRATION_FAILED"
fi
exit "${status}"
