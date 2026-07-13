#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ai_env/bin/python}"
ENTRYPOINT="${PROJECT_ROOT}/simple_lab_test/search/tpp_experiment.py"
REFERENCE_ENTRYPOINT="${PROJECT_ROOT}/simple_lab_test/search/reevaluate_titantpp_validation.py"
ARTIFACT_ROOT="${PROJECT_ROOT}/search_artifacts"
OUTPUT_ROOT="${ARTIFACT_ROOT}/model_enhancement_direct_raw_qty_q012_inter_seed42_e50_0713"
REFERENCE_DIR="${ARTIFACT_ROOT}/model_enhancement_v2_inter_validation_reference_raw_q012_0713"
LOG_DIR="${OUTPUT_ROOT}/logs"
LOG_PATH="${LOG_DIR}/run.log"
STATUS_PATH="${OUTPUT_ROOT}/variant_status.tsv"
CUDA13_LIB="/opt/miniconda3/envs/ai_env/lib/python3.12/site-packages/nvidia/cu13/lib"

V2_ROOT="${ARTIFACT_ROOT}/model_enhancement_v2_inter_short_e50_0710"
V2_RUN_REL="runs/intermittent/titantpp/lossmode_hybrid/split_fixed/value_identity/valueinput_residual/valueemb_8/trainscope_target_only/profile_dataset_best/base_2p0/small_lmm/epochs_50/seed_42"
V2_CHECKPOINT="${V2_ROOT}/${V2_RUN_REL}/checkpoints/best_val_nll_model.pt"
V2_MARKED="${V2_ROOT}/cache/intermittent/fixed_split/marked_fixed_base_2p0.parquet"

export LD_LIBRARY_PATH="${CUDA13_LIB}:/opt/miniconda3/envs/ai_env/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="${OUTPUT_ROOT}/cache-runtime"
export MPLCONFIGDIR="${OUTPUT_ROOT}/matplotlib-cache"

mkdir -p \
  "${LOG_DIR}" \
  "${REFERENCE_DIR}/logs" \
  "${XDG_CACHE_HOME}" \
  "${MPLCONFIGDIR}"
cd "${PROJECT_ROOT}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[preflight_error] missing_file=${path}"
    return 1
  fi
}

run_validation_reference() {
  echo "[reference_start] $(date '+%Y-%m-%d %H:%M:%S %Z')"
  "${PYTHON_BIN}" "${REFERENCE_ENTRYPOINT}" \
    --checkpoint "${V2_CHECKPOINT}" \
    --marked-parquet "${V2_MARKED}" \
    --output-dir "${REFERENCE_DIR}" \
    --device cuda \
    --analysis-scale-base 10 \
    --analysis-tail-order 4 \
    2>&1 | tee "${REFERENCE_DIR}/logs/run.log"
  local status=${PIPESTATUS[0]}
  echo "[reference_end] exit_code=${status} $(date '+%Y-%m-%d %H:%M:%S %Z')"
  if [[ "${status}" -eq 0 ]]; then
    touch "${REFERENCE_DIR}/REFERENCE_SUCCESS"
  else
    touch "${REFERENCE_DIR}/REFERENCE_FAILED"
  fi
  return "${status}"
}

run_variant() {
  local variant="$1"
  local norm_mode="$2"
  local variant_dir="${OUTPUT_ROOT}/${variant}"

  mkdir -p "${variant_dir}/logs"
  echo "[variant_start] ${variant} norm=${norm_mode} $(date '+%Y-%m-%d %H:%M:%S %Z')"
  "${PYTHON_BIN}" "${ENTRYPOINT}" long-epoch \
    --base-dir "${variant_dir}" \
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
    --qty-decoder-mode direct_raw_qty \
    --magnitude-norm-mode "${norm_mode}" \
    --magnitude-input-emb-dim 8 \
    --lambda-magnitude 1.0 \
    --magnitude-sigma-floor 0.0550124034288891 \
    --magnitude-revin-eps 1e-5 \
    --magnitude-shrinkage-k 8 \
    --magnitude-center-mode mean \
    --no-magnitude-revin-affine \
    --magnitude-stat-context-mode none \
    --eval-selections best_val_nll,best_score,final \
    --device cuda \
    --force-rerun \
    --stop-on-error \
    2>&1 | tee "${variant_dir}/logs/run.log"
  local status=${PIPESTATUS[0]}
  printf '%s\t%s\t%s\n' "${variant}" "${norm_mode}" "${status}" >> "${STATUS_PATH}"
  echo "[variant_end] ${variant} exit_code=${status} $(date '+%Y-%m-%d %H:%M:%S %Z')"
  return "${status}"
}

printf '{\n  "experiment_id": "model_enhancement_direct_raw_qty_q012_inter_seed42_e50_0713",\n  "started_at": "%s",\n  "server": "%s",\n  "device": "cuda",\n  "dataset": "intermittent",\n  "split_mode": "fixed",\n  "epochs": 50,\n  "seed": 42,\n  "batch_size": 128,\n  "lookback_weeks": 52,\n  "max_seq_len": 16,\n  "model": "titantpp",\n  "candidate": "small_lmm",\n  "qty_decoder_mode": "direct_raw_qty",\n  "variants": ["q0_global", "q1_causal_revin", "q2_causal_shrinkage_revin"],\n  "v2_reference_dir": "%s",\n  "held_out_policy": "validation decision before reading test or merged artifacts",\n  "acceptance_scope": "seed-42 validation-only candidate and RevIN-benefit gate"\n}\n' \
  "$(date '+%Y-%m-%d %H:%M:%S %Z')" \
  "$(hostname)" \
  "${REFERENCE_DIR}" \
  > "${OUTPUT_ROOT}/experiment_manifest.json"
printf 'variant\tnorm_mode\texit_code\n' > "${STATUS_PATH}"

{
  echo "[start] $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "[server] $(hostname)"
  echo "[python] ${PYTHON_BIN}"
  echo "[cuda_lib] ${CUDA13_LIB}"
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

  require_file "${ENTRYPOINT}" || exit 2
  require_file "${REFERENCE_ENTRYPOINT}" || exit 2
  require_file "${V2_CHECKPOINT}" || exit 2
  require_file "${V2_MARKED}" || exit 2

  echo "[stage] freeze V2 validation-only reference"
  run_validation_reference || exit $?

  echo "[stage] train matched Q0/Q1/Q2"
  overall_status=0
  run_variant q0_global global || overall_status=1
  run_variant q1_causal_revin causal_revin || overall_status=1
  run_variant q2_causal_shrinkage_revin causal_shrinkage_revin || overall_status=1

  echo "[end] $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "[exit_code] ${overall_status}"
  exit "${overall_status}"
} 2>&1 | tee "${LOG_PATH}"

status=${PIPESTATUS[0]}
if [[ "${status}" -eq 0 ]]; then
  touch "${OUTPUT_ROOT}/SCREENING_SUCCESS"
else
  touch "${OUTPUT_ROOT}/SCREENING_FAILED"
fi
exit "${status}"
