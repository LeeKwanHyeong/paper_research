#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ai_env/bin/python}"
OUTPUT_DIR="${PROJECT_ROOT}/search_artifacts/model_enhancement_m0_cuda_model_test_0713"
LOG_DIR="${OUTPUT_DIR}/logs"
LOG_PATH="${LOG_DIR}/run.log"
CUDA13_LIB="/opt/miniconda3/envs/ai_env/lib/python3.12/site-packages/nvidia/cu13/lib"

export LD_LIBRARY_PATH="${CUDA13_LIB}:/opt/miniconda3/envs/ai_env/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

{
  echo "[start] $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "[server] $(hostname)"
  echo "[python] ${PYTHON_BIN}"
  echo "[cuda_lib] ${CUDA13_LIB}"
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

  "${PYTHON_BIN}" simple_lab_test/search/tpp_experiment.py model-test \
    --output-dir "${OUTPUT_DIR}" \
    --models titantpp \
    --titan-candidates small_lmm \
    --device cuda \
    --batch-size 4 \
    --seq-len 16 \
    --num-marks 12 \
    --rmtpp-hidden-dim 64 \
    --scale-base 2 \
    --qty-decoder-mode direct_log_qty \
    --magnitude-norm-mode global \
    --magnitude-input-emb-dim 8 \
    --lambda-magnitude 1.0 \
    --magnitude-sigma-floor 0.0014535461338152059 \
    --magnitude-exp-clamp-min -2 \
    --magnitude-exp-clamp-max 15 \
    --left-pad \
    --stop-on-error
  status=$?
  echo "[end] $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "[exit_code] ${status}"
  exit "${status}"
} 2>&1 | tee "${LOG_PATH}"

status=${PIPESTATUS[0]}
if [[ "${status}" -eq 0 ]]; then
  touch "${OUTPUT_DIR}/MODEL_TEST_SUCCESS"
else
  touch "${OUTPUT_DIR}/MODEL_TEST_FAILED"
fi
exit "${status}"
