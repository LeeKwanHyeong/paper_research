#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ai_env/bin/python}"
ENTRYPOINT="${PROJECT_ROOT}/simple_lab_test/search/tpp_experiment.py"
REFERENCE_ENTRYPOINT="${PROJECT_ROOT}/simple_lab_test/search/reevaluate_titantpp_validation.py"
ARTIFACT_ROOT="${PROJECT_ROOT}/search_artifacts"
OUTPUT_DIR="${ARTIFACT_ROOT}/model_enhancement_m0_inter_short_e50_0713"
REFERENCE_DIR="${ARTIFACT_ROOT}/model_enhancement_v2_inter_validation_reference_m0_0713"
LOG_PATH="${OUTPUT_DIR}/logs/run.log"
REFERENCE_LOG_PATH="${REFERENCE_DIR}/logs/run.log"
CUDA13_LIB="/opt/miniconda3/envs/ai_env/lib/python3.12/site-packages/nvidia/cu13/lib"

V2_ROOT="${ARTIFACT_ROOT}/model_enhancement_v2_inter_short_e50_0710"
V2_RUN_REL="runs/intermittent/titantpp/lossmode_hybrid/split_fixed/value_identity/valueinput_residual/valueemb_8/trainscope_target_only/profile_dataset_best/base_2p0/small_lmm/epochs_50/seed_42"
V2_CHECKPOINT="${V2_ROOT}/${V2_RUN_REL}/checkpoints/best_val_nll_model.pt"
V2_MARKED="${V2_ROOT}/cache/intermittent/fixed_split/marked_fixed_base_2p0.parquet"

export LD_LIBRARY_PATH="${CUDA13_LIB}:/opt/miniconda3/envs/ai_env/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="${OUTPUT_DIR}/cache-runtime"
export MPLCONFIGDIR="${OUTPUT_DIR}/matplotlib-cache"

mkdir -p \
  "${OUTPUT_DIR}/logs" \
  "${REFERENCE_DIR}/logs" \
  "${XDG_CACHE_HOME}" \
  "${MPLCONFIGDIR}"
cd "${PROJECT_ROOT}"

run_validation_reference() {
  "${PYTHON_BIN}" "${REFERENCE_ENTRYPOINT}" \
    --checkpoint "${V2_CHECKPOINT}" \
    --marked-parquet "${V2_MARKED}" \
    --output-dir "${REFERENCE_DIR}" \
    --device cuda \
    2>&1 | tee "${REFERENCE_LOG_PATH}"
  return "${PIPESTATUS[0]}"
}

{
  echo "[start] $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "[server] $(hostname)"
  echo "[python] ${PYTHON_BIN}"
  echo "[cuda_lib] ${CUDA13_LIB}"
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

  echo "[stage] freeze V2 validation-only reference with log2 quantity metrics"
  run_validation_reference || exit $?

  echo "[stage] train Intermittent M0 seed-42 e50"
  "${PYTHON_BIN}" "${ENTRYPOINT}" long-epoch \
    --base-dir "${OUTPUT_DIR}" \
    --datasets intermittent \
    --models titantpp \
    --titan-candidates small_lmm \
    --epochs 50 \
    --seeds 42 \
    --lr 1e-3 \
    --batch-size 128 \
    --lookback-weeks 52 \
    --max-seq-len 16 \
    --split-mode fixed \
    --value-head-activation identity \
    --value-head-mode shared \
    --qty-mark-gradient-mode coupled \
    --value-encoder-gradient-mode coupled \
    --value-input-mode none \
    --train-loss-scope target_only \
    --loss-mode hybrid \
    --marker-loss-mode ce \
    --lambda-ordinal 0 \
    --qty-decoder-mode direct_log_qty \
    --magnitude-norm-mode global \
    --magnitude-input-emb-dim 8 \
    --lambda-magnitude 1.0 \
    --magnitude-sigma-floor 0.0014535461338152059 \
    --magnitude-exp-clamp-min -2 \
    --magnitude-exp-clamp-max 15 \
    --eval-selections best_val_nll,best_score,final \
    --device cuda \
    --force-rerun \
    --stop-on-error
  status=$?
  echo "[end] $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "[exit_code] ${status}"
  exit "${status}"
} 2>&1 | tee "${LOG_PATH}"

status=${PIPESTATUS[0]}
if [[ "${status}" -eq 0 ]]; then
  touch "${OUTPUT_DIR}/SCREENING_SUCCESS"
else
  touch "${OUTPUT_DIR}/SCREENING_FAILED"
fi
exit "${status}"
