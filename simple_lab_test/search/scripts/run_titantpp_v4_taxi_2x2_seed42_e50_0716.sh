#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ai_env/bin/python}"
ENTRYPOINT="${PROJECT_ROOT}/simple_lab_test/search/tpp_experiment.py"
COMPARATOR="${PROJECT_ROOT}/simple_lab_test/search/compare_titantpp_v4_taxi_validation.py"
INTEGRATION_ROOT="${PROJECT_ROOT}/search_artifacts/model_enhancement_titantpp_v4_cuda_insta_smoke_0716"
EXPERIMENT_ID="model_enhancement_titantpp_v4_taxi_2x2_seed42_e50_0716"
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
"${PYTHON_BIN}" -c "import json,pathlib; p=pathlib.Path('${INTEGRATION_ROOT}/integration_summary.json'); d=json.loads(p.read_text()); assert d['status']=='PASS' and d['held_out_test_evaluated'] is False" || {
  echo "[preflight_error] V4 CUDA/Instacart integration gate has not passed"
  exit 2
}

cat > "${OUTPUT_ROOT}/experiment_manifest.json" <<JSON
{
  "experiment_id": "${EXPERIMENT_ID}",
  "started_at": "$(date '+%Y-%m-%d %H:%M:%S %Z')",
  "server": "$(hostname)",
  "device": "cuda",
  "source_revision": "${SOURCE_REVISION}",
  "dataset": "yellow_trip_hourly",
  "split_mode": "fixed",
  "evaluation_scope": "validation_only",
  "expected_loader_samples": {"train": 38393, "validation": 8268, "test": 8327},
  "epochs": 50,
  "seed": 42,
  "learning_rate": 0.001,
  "batch_size": 128,
  "lookback_weeks": 168,
  "max_seq_len": 256,
  "candidate": "mid_lmm",
  "variants": {
    "v2_shared_value_shared_time": {"value_head_mode": "shared", "qty_mark_gradient_mode": "coupled", "time_head_mode": "shared"},
    "v3b_mark_value_shared_time": {"value_head_mode": "mark_conditioned_experts", "qty_mark_gradient_mode": "detached", "time_head_mode": "shared"},
    "v4a_shared_value_mark_time": {"value_head_mode": "shared", "qty_mark_gradient_mode": "coupled", "time_head_mode": "mark_conditioned"},
    "v4b_mark_value_mark_time": {"value_head_mode": "mark_conditioned_experts", "qty_mark_gradient_mode": "detached", "time_head_mode": "mark_conditioned"}
  },
  "decision_checkpoint": "best_val_nll",
  "held_out_policy": "test metric evaluation disabled until a validation pair passes",
  "acceptance_contract": "${OUTPUT_ROOT}/acceptance_contract.json"
}
JSON

cat > "${OUTPUT_ROOT}/acceptance_contract.json" <<'JSON'
{
  "decision_split": "validation",
  "decision_checkpoint": "best_val_nll",
  "pairs": {"V4a": "V2", "V4b": "V3b"},
  "thresholds": {
    "time_nll_improvement_pct_min": 0.5,
    "total_nll_regression_pct_max": 0.5,
    "dt_mae_regression_pct_max": 1.0,
    "marker_nll_regression_pct_max": 2.0,
    "mark_accuracy_delta_pp_min": -0.25,
    "qty_mae_regression_pct_max": 5.0
  },
  "selection": "V4b is the Taxi promotion candidate; V4a is an attribution control",
  "held_out_lock": "No test metric or multi-seed execution before a validation pair passes"
}
JSON
printf 'variant\texit_code\n' > "${STATUS_PATH}"

run_variant() {
  local variant="$1"
  local value_mode="$2"
  local mark_gradient="$3"
  local time_mode="$4"
  local variant_dir="${OUTPUT_ROOT}/variants/${variant}"

  mkdir -p "${variant_dir}"
  echo "[variant_start] ${variant} value=${value_mode} qty_mark_grad=${mark_gradient} time=${time_mode} $(date '+%Y-%m-%d %H:%M:%S %Z')"
  "${PYTHON_BIN}" "${ENTRYPOINT}" long-epoch \
    --base-dir "${variant_dir}" \
    --datasets yellow_trip_hourly \
    --models titantpp \
    --titan-profile dataset_best \
    --titan-candidates mid_lmm \
    --epochs 50 \
    --seeds 42 \
    --lr 1e-3 \
    --lambda-dt 1.0 \
    --batch-size 128 \
    --lookback-weeks 168 \
    --max-seq-len 256 \
    --split-mode fixed \
    --evaluation-scope validation_only \
    --reproducibility-mode strict \
    --value-head-activation identity \
    --value-head-mode "${value_mode}" \
    --time-head-mode "${time_mode}" \
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
  printf '%s\t%s\n' "${variant}" "${status}" >> "${STATUS_PATH}"
  echo "[variant_end] ${variant} exit_code=${status} $(date '+%Y-%m-%d %H:%M:%S %Z')"
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
  run_variant v2_shared_value_shared_time shared coupled shared || overall_status=1
  run_variant v3b_mark_value_shared_time mark_conditioned_experts detached shared || overall_status=1
  run_variant v4a_shared_value_mark_time shared coupled mark_conditioned || overall_status=1
  run_variant v4b_mark_value_mark_time mark_conditioned_experts detached mark_conditioned || overall_status=1

  if [[ "${overall_status}" -eq 0 ]]; then
    "${PYTHON_BIN}" "${COMPARATOR}" \
      --artifact-root "${OUTPUT_ROOT}" \
      --source-revision "${SOURCE_REVISION}" || overall_status=1
  fi
  echo "[end] $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "[exit_code] ${overall_status}"
  exit "${overall_status}"
} 2>&1 | tee "${LOG_PATH}"

status=${PIPESTATUS[0]}
if [[ "${status}" -eq 0 ]]; then
  touch "${OUTPUT_ROOT}/SCREENING_COMPLETE"
else
  touch "${OUTPUT_ROOT}/SCREENING_FAILED"
fi
exit "${status}"
