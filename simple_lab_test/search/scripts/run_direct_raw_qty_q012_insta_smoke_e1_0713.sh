#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ai_env/bin/python}"
OUTPUT_ROOT="${PROJECT_ROOT}/search_artifacts/model_enhancement_direct_raw_qty_q012_insta_smoke_e1_0713"
LOG_DIR="${OUTPUT_ROOT}/logs"
LOG_PATH="${LOG_DIR}/run.log"
STATUS_PATH="${OUTPUT_ROOT}/variant_status.tsv"
CUDA13_LIB="/opt/miniconda3/envs/ai_env/lib/python3.12/site-packages/nvidia/cu13/lib"

export LD_LIBRARY_PATH="${CUDA13_LIB}:/opt/miniconda3/envs/ai_env/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="${OUTPUT_ROOT}/cache"
export MPLCONFIGDIR="${OUTPUT_ROOT}/matplotlib-cache"

mkdir -p "${LOG_DIR}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"
cd "${PROJECT_ROOT}"

printf '{\n  "experiment_id": "model_enhancement_direct_raw_qty_q012_insta_smoke_e1_0713",\n  "started_at": "%s",\n  "server": "%s",\n  "device": "cuda",\n  "dataset": "insta_market_basket",\n  "insta_max_series": 20,\n  "split_mode": "fixed",\n  "epochs": 1,\n  "seed": 42,\n  "batch_size": 16,\n  "lookback_weeks": 10,\n  "max_seq_len": 16,\n  "model": "titantpp",\n  "candidate": "small_lmm",\n  "qty_decoder_mode": "direct_raw_qty",\n  "variants": ["q0_global", "q1_causal_revin", "q2_causal_shrinkage_revin"],\n  "acceptance_scope": "actual-data integration and artifact smoke only"\n}\n' \
  "$(date '+%Y-%m-%d %H:%M:%S %Z')" \
  "$(hostname)" \
  > "${OUTPUT_ROOT}/experiment_manifest.json"
printf 'variant\tnorm_mode\texit_code\n' > "${STATUS_PATH}"

run_variant() {
  local variant="$1"
  local norm_mode="$2"
  local variant_dir="${OUTPUT_ROOT}/${variant}"

  mkdir -p "${variant_dir}/logs"
  echo "[variant_start] ${variant} norm=${norm_mode} $(date '+%Y-%m-%d %H:%M:%S %Z')"
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

{
  echo "[start] $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "[server] $(hostname)"
  echo "[python] ${PYTHON_BIN}"
  echo "[cuda_lib] ${CUDA13_LIB}"
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

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
  touch "${OUTPUT_ROOT}/SMOKE_SUCCESS"
else
  touch "${OUTPUT_ROOT}/SMOKE_FAILED"
fi
exit "${status}"
