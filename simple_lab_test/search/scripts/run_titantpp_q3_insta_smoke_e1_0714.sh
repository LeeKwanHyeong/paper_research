#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ai_env/bin/python}"
OUTPUT_ROOT="${PROJECT_ROOT}/search_artifacts/model_enhancement_titantpp_q3_insta_smoke_e1_0714"
LOG_DIR="${OUTPUT_ROOT}/logs"
LOG_PATH="${LOG_DIR}/run.log"
STATUS_PATH="${OUTPUT_ROOT}/variant_status.tsv"
CUDA13_LIB="/opt/miniconda3/envs/ai_env/lib/python3.12/site-packages/nvidia/cu13/lib"

export LD_LIBRARY_PATH="${CUDA13_LIB}:/opt/miniconda3/envs/ai_env/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="${OUTPUT_ROOT}/cache"
export MPLCONFIGDIR="${OUTPUT_ROOT}/matplotlib-cache"

mkdir -p "${LOG_DIR}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"
rm -f "${OUTPUT_ROOT}/SMOKE_SUCCESS" "${OUTPUT_ROOT}/SMOKE_FAILED"
cd "${PROJECT_ROOT}"

cat > "${OUTPUT_ROOT}/experiment_manifest.json" <<JSON
{
  "experiment_id": "model_enhancement_titantpp_q3_insta_smoke_e1_0714",
  "q3_implementation_revision": "14c2978",
  "cuda_gate_result_revision": "696f907",
  "source_sync_manifest_required": true,
  "started_at": "$(date '+%Y-%m-%d %H:%M:%S %Z')",
  "server": "$(hostname)",
  "device": "cuda",
  "dataset": "insta_market_basket",
  "insta_max_series": 20,
  "expected_sample_counts": {"train": 1380, "validation": 300, "test": 300},
  "split_mode": "fixed",
  "epochs": 1,
  "seed": 42,
  "learning_rate": 0.001,
  "batch_size": 16,
  "lookback_weeks": 10,
  "max_seq_len": 16,
  "model": "titantpp",
  "candidate": "small_lmm",
  "expected_parameter_count": 78111,
  "qty_decoder_mode": "direct_raw_qty",
  "magnitude_norm_mode": "causal_shrinkage_revin",
  "magnitude_shrinkage_k": 8,
  "magnitude_sigma_floor": 0.0550124034288891,
  "lambda_magnitude": 1.0,
  "lambda_qty": 0.25,
  "lambda_log_qty": 0.25,
  "log_qty_huber_delta": 1.0,
  "log_qty_floor": 1.0,
  "variants": {
    "q2_coupled_none": {"magnitude_encoder_gradient_mode": "coupled", "magnitude_aux_loss_mode": "none"},
    "q3a_detached_none": {"magnitude_encoder_gradient_mode": "detached", "magnitude_aux_loss_mode": "none"},
    "q3b_coupled_log_huber": {"magnitude_encoder_gradient_mode": "coupled", "magnitude_aux_loss_mode": "log_huber"},
    "q3c_detached_log_huber": {"magnitude_encoder_gradient_mode": "detached", "magnitude_aux_loss_mode": "log_huber"}
  },
  "acceptance_scope": "actual-data backward, checkpoint, cache identity, and artifact integration only",
  "performance_ranking_allowed": false
}
JSON
printf 'variant\tgradient_mode\taux_loss_mode\texit_code\n' > "${STATUS_PATH}"

run_variant() {
  local variant="$1"
  local gradient_mode="$2"
  local aux_loss_mode="$3"
  local variant_dir="${OUTPUT_ROOT}/${variant}"

  mkdir -p "${variant_dir}/logs"
  echo "[variant_start] ${variant} gradient=${gradient_mode} aux=${aux_loss_mode} $(date '+%Y-%m-%d %H:%M:%S %Z')"
  "${PYTHON_BIN}" simple_lab_test/search/tpp_experiment.py long-epoch \
    --base-dir "${variant_dir}" \
    --datasets insta_market_basket \
    --models titantpp \
    --titan-candidates small_lmm \
    --epochs 1 \
    --seeds 42 \
    --lr 1e-3 \
    --batch-size 16 \
    --lookback-weeks 10 \
    --max-seq-len 16 \
    --insta-max-series 20 \
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
    --eval-selections best_val_nll,best_score,final \
    --device cuda \
    --force-rerun \
    --stop-on-error \
    2>&1 | tee "${variant_dir}/logs/run.log"
  local status=${PIPESTATUS[0]}
  printf '%s\t%s\t%s\t%s\n' \
    "${variant}" "${gradient_mode}" "${aux_loss_mode}" "${status}" \
    >> "${STATUS_PATH}"
  echo "[variant_end] ${variant} exit_code=${status} $(date '+%Y-%m-%d %H:%M:%S %Z')"
  return "${status}"
}

{
  echo "[start] $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "[server] $(hostname)"
  echo "[python] ${PYTHON_BIN}"
  echo "[cuda_lib] ${CUDA13_LIB}"
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

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
  touch "${OUTPUT_ROOT}/SMOKE_SUCCESS"
else
  touch "${OUTPUT_ROOT}/SMOKE_FAILED"
fi
exit "${status}"
