#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ai_env/bin/python}"
EXPERIMENT_ID="model_enhancement_titantpp_q3_cuda_model_test_0713"
OUTPUT_DIR="${PROJECT_ROOT}/search_artifacts/${EXPERIMENT_ID}"
LOG_DIR="${OUTPUT_DIR}/logs"
LOG_PATH="${LOG_DIR}/run.log"
STATUS_PATH="${OUTPUT_DIR}/variant_status.tsv"
CUDA13_LIB="/opt/miniconda3/envs/ai_env/lib/python3.12/site-packages/nvidia/cu13/lib"

export LD_LIBRARY_PATH="${CUDA13_LIB}:/opt/miniconda3/envs/ai_env/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

started_at="$(date '+%Y-%m-%d %H:%M:%S %Z')"
code_revision="$(git rev-parse HEAD 2>/dev/null || printf 'unknown')"
printf '{\n  "experiment_id": "%s",\n  "started_at": "%s",\n  "server": "%s",\n  "code_revision": "%s",\n  "device": "cuda",\n  "seed": 42,\n  "batch_size": 4,\n  "seq_len": 16,\n  "num_marks": 12,\n  "model": "titantpp",\n  "candidate": "small_lmm",\n  "qty_decoder_mode": "direct_raw_qty",\n  "magnitude_norm_mode": "causal_shrinkage_revin",\n  "magnitude_shrinkage_k": 8,\n  "lambda_magnitude": 1.0,\n  "lambda_qty": 0.25,\n  "lambda_log_qty": 0.25,\n  "log_qty_huber_delta": 1.0,\n  "log_qty_floor": 1.0,\n  "expected_parameter_count": 78111,\n  "variants": ["q2_coupled_none", "q3a_detached_none", "q3b_coupled_log_huber", "q3c_detached_log_huber"],\n  "acceptance_scope": "CUDA runtime, forward/loss equivalence, and artifact identity only"\n}\n' \
  "${EXPERIMENT_ID}" \
  "${started_at}" \
  "$(hostname)" \
  "${code_revision}" \
  > "${OUTPUT_DIR}/experiment_manifest.json"
printf 'variant\tgradient_mode\taux_loss_mode\texit_code\n' > "${STATUS_PATH}"

run_variant() {
  local variant="$1"
  local gradient_mode="$2"
  local aux_loss_mode="$3"

  echo "[variant_start] ${variant} gradient=${gradient_mode} aux=${aux_loss_mode} $(date '+%Y-%m-%d %H:%M:%S %Z')"
  "${PYTHON_BIN}" simple_lab_test/search/tpp_experiment.py model-test \
    --output-dir "${OUTPUT_DIR}/${variant}" \
    --models titantpp \
    --titan-candidates small_lmm \
    --device cuda \
    --seed 42 \
    --batch-size 4 \
    --seq-len 16 \
    --num-marks 12 \
    --rmtpp-hidden-dim 64 \
    --scale-base 2 \
    --qty-decoder-mode direct_raw_qty \
    --magnitude-norm-mode causal_shrinkage_revin \
    --magnitude-input-emb-dim 8 \
    --lambda-magnitude 1.0 \
    --magnitude-encoder-gradient-mode "${gradient_mode}" \
    --magnitude-aux-loss-mode "${aux_loss_mode}" \
    --lambda-log-qty 0.25 \
    --log-qty-huber-delta 1.0 \
    --log-qty-floor 1.0 \
    --magnitude-sigma-floor 0.0550124034288891 \
    --magnitude-revin-eps 1e-5 \
    --magnitude-shrinkage-k 8 \
    --magnitude-center-mode mean \
    --no-magnitude-revin-affine \
    --magnitude-stat-context-mode none \
    --left-pad \
    --stop-on-error
  local status=$?
  printf '%s\t%s\t%s\t%s\n' \
    "${variant}" "${gradient_mode}" "${aux_loss_mode}" "${status}" \
    >> "${STATUS_PATH}"
  echo "[variant_end] ${variant} exit_code=${status} $(date '+%Y-%m-%d %H:%M:%S %Z')"
  return "${status}"
}

{
  echo "[start] ${started_at}"
  echo "[server] $(hostname)"
  echo "[code_revision] ${code_revision}"
  echo "[python] ${PYTHON_BIN}"
  echo "[cuda_lib] ${CUDA13_LIB}"
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader

  if ! "${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(); print(f"[torch] {torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")'; then
    echo "[preflight_failed] PyTorch CUDA is unavailable."
    exit 1
  fi

  overall_status=0
  run_variant q2_coupled_none coupled none || overall_status=1
  run_variant q3a_detached_none detached none || overall_status=1
  run_variant q3b_coupled_log_huber coupled log_huber || overall_status=1
  run_variant q3c_detached_log_huber detached log_huber || overall_status=1

  echo "[end] $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "[exit_code] ${overall_status}"
  exit "${overall_status}"
} 2>&1 | tee "${LOG_PATH}"

status=${PIPESTATUS[0]}
if [[ "${status}" -eq 0 ]]; then
  touch "${OUTPUT_DIR}/MODEL_TEST_SUCCESS"
else
  touch "${OUTPUT_DIR}/MODEL_TEST_FAILED"
fi
exit "${status}"
