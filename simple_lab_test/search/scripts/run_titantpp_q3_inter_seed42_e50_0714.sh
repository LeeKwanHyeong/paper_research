#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ai_env/bin/python}"
ENTRYPOINT="${PROJECT_ROOT}/simple_lab_test/search/tpp_experiment.py"
REFERENCE_ENTRYPOINT="${PROJECT_ROOT}/simple_lab_test/search/reevaluate_titantpp_validation.py"
ARTIFACT_ROOT="${PROJECT_ROOT}/search_artifacts"
EXPERIMENT_ID="model_enhancement_titantpp_q3_inter_seed42_e50_0714"
OUTPUT_ROOT="${ARTIFACT_ROOT}/${EXPERIMENT_ID}"
REFERENCE_DIR="${ARTIFACT_ROOT}/model_enhancement_v2_inter_validation_reference_q3_0714"
LOG_DIR="${OUTPUT_ROOT}/logs"
LOG_PATH="${LOG_DIR}/run.log"
STATUS_PATH="${OUTPUT_ROOT}/variant_status.tsv"
CUDA13_LIB="/opt/miniconda3/envs/ai_env/lib/python3.12/site-packages/nvidia/cu13/lib"

V2_ROOT="${ARTIFACT_ROOT}/model_enhancement_v2_inter_short_e50_0710"
V2_RUN_REL="runs/intermittent/titantpp/lossmode_hybrid/split_fixed/value_identity/valueinput_residual/valueemb_8/trainscope_target_only/profile_dataset_best/base_2p0/small_lmm/epochs_50/seed_42"
V2_CHECKPOINT="${V2_ROOT}/${V2_RUN_REL}/checkpoints/best_val_nll_model.pt"
V2_MARKED="${V2_ROOT}/cache/intermittent/fixed_split/marked_fixed_base_2p0.parquet"
V2_CHECKPOINT_SHA256="1a901eb2ac912537e25b6c798978870a6f650857b41642f2a0b773030cc103c0"
V2_MARKED_SHA256="dab4d8a7217f9c14d1c2336f649aef9ddaf2ba440d074e446d8fd5cc41506a30"

FROZEN_Q2_ROOT="${ARTIFACT_ROOT}/model_enhancement_direct_raw_qty_q012_inter_seed42_e50_0713"
FROZEN_Q2_SUMMARY="${FROZEN_Q2_ROOT}/q2_causal_shrinkage_revin/leaderboard/summary.csv"
FROZEN_Q2_SUCCESS="${FROZEN_Q2_ROOT}/SCREENING_SUCCESS"
FROZEN_Q2_SUMMARY_SHA256="256fbd69a4a63bbb1b6e2cb97f3a223067990b2a639a8bc4a1428e61ae8066f2"

INTER_DATA_ROOT="${PROJECT_ROOT}/sample_data/head_office"
INTER_SPLIT_WITH="${INTER_DATA_ROOT}/marked_target_with_split.parquet"
INTER_SPLIT_TRAIN="${INTER_DATA_ROOT}/marked_target_train.parquet"
INTER_SPLIT_VALIDATION="${INTER_DATA_ROOT}/marked_target_validation.parquet"
INTER_SPLIT_TEST="${INTER_DATA_ROOT}/marked_target_test.parquet"
INTER_SPLIT_MANIFEST="${INTER_DATA_ROOT}/marked_target_split_manifest.json"
INTER_SPLIT_WITH_SHA256="dab4d8a7217f9c14d1c2336f649aef9ddaf2ba440d074e446d8fd5cc41506a30"
INTER_SPLIT_TRAIN_SHA256="3d66e0dc2ef671f652427b5f4756604b29efd10301ec42f9f5b9a7631eb8c242"
INTER_SPLIT_VALIDATION_SHA256="10c4811d02db5e4bff50af230e068754901bb0cf106f7ca29a5f8b694294ac72"
INTER_SPLIT_TEST_SHA256="191d675819db63647f34446bb9fae79f0822d2c09ef05387aeae3877b6fe8263"
INTER_SPLIT_MANIFEST_SHA256="49752a1bd4ccaf1c2b8e37321e3657cb098d303c51a093eecf1c91ea3ef9bdfe"

export LD_LIBRARY_PATH="${CUDA13_LIB}:/opt/miniconda3/envs/ai_env/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="${OUTPUT_ROOT}/cache-runtime"
export MPLCONFIGDIR="${OUTPUT_ROOT}/matplotlib-cache"

mkdir -p \
  "${LOG_DIR}" \
  "${REFERENCE_DIR}/logs" \
  "${XDG_CACHE_HOME}" \
  "${MPLCONFIGDIR}"
rm -f \
  "${OUTPUT_ROOT}/SCREENING_SUCCESS" \
  "${OUTPUT_ROOT}/SCREENING_FAILED" \
  "${REFERENCE_DIR}/REFERENCE_SUCCESS" \
  "${REFERENCE_DIR}/REFERENCE_FAILED"
cd "${PROJECT_ROOT}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[preflight_error] missing_file=${path}"
    return 1
  fi
}

require_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "[preflight_error] sha256_mismatch path=${path} expected=${expected} actual=${actual}"
    return 1
  fi
  echo "[preflight_sha256] path=${path} sha256=${actual}"
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
  local gradient_mode="$2"
  local aux_loss_mode="$3"
  local variant_dir="${OUTPUT_ROOT}/${variant}"

  mkdir -p "${variant_dir}/logs"
  echo "[variant_start] ${variant} gradient=${gradient_mode} aux=${aux_loss_mode} $(date '+%Y-%m-%d %H:%M:%S %Z')"
  "${PYTHON_BIN}" "${ENTRYPOINT}" long-epoch \
    --base-dir "${variant_dir}" \
    --datasets intermittent \
    --models titantpp \
    --titan-profile dataset_best \
    --titan-candidates small_lmm \
    --epochs 50 \
    --seeds 42 \
    --lr 1e-3 \
    --lambda-dt 1.0 \
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
    --magnitude-exp-clamp-min -2 \
    --magnitude-exp-clamp-max 15 \
    --test-time-memory none \
    --analysis-scale-base 10 \
    --analysis-tail-order 4 \
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

cat > "${OUTPUT_ROOT}/experiment_manifest.json" <<JSON
{
  "experiment_id": "${EXPERIMENT_ID}",
  "started_at": "$(date '+%Y-%m-%d %H:%M:%S %Z')",
  "server": "$(hostname)",
  "device": "cuda",
  "dataset": "intermittent",
  "split_mode": "fixed",
  "expected_sample_counts": {"train": 136256, "validation": 41901, "test": 41344},
  "expected_train_event_count": 159643,
  "expected_num_marks": 12,
  "expected_parameter_count": 78111,
  "epochs": 50,
  "seed": 42,
  "learning_rate": 0.001,
  "batch_size": 128,
  "lookback_weeks": 52,
  "max_seq_len": 16,
  "model": "titantpp",
  "candidate": "small_lmm",
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
  "v2_reference_dir": "${REFERENCE_DIR}",
  "v2_checkpoint": "${V2_CHECKPOINT}",
  "v2_checkpoint_sha256": "${V2_CHECKPOINT_SHA256}",
  "v2_marked_parquet": "${V2_MARKED}",
  "v2_marked_parquet_sha256": "${V2_MARKED_SHA256}",
  "frozen_q2_summary": "${FROZEN_Q2_SUMMARY}",
  "frozen_q2_summary_sha256": "${FROZEN_Q2_SUMMARY_SHA256}",
  "fixed_split_sources": {
    "with_split": {"path": "${INTER_SPLIT_WITH}", "sha256": "${INTER_SPLIT_WITH_SHA256}"},
    "train": {"path": "${INTER_SPLIT_TRAIN}", "sha256": "${INTER_SPLIT_TRAIN_SHA256}"},
    "validation": {"path": "${INTER_SPLIT_VALIDATION}", "sha256": "${INTER_SPLIT_VALIDATION_SHA256}"},
    "test": {"path": "${INTER_SPLIT_TEST}", "sha256": "${INTER_SPLIT_TEST_SHA256}"},
    "manifest": {"path": "${INTER_SPLIT_MANIFEST}", "sha256": "${INTER_SPLIT_MANIFEST_SHA256}"}
  },
  "acceptance_contract": "${OUTPUT_ROOT}/acceptance_contract.json",
  "held_out_policy": "do not read test or merged artifacts before the seed-42 validation decision is recorded",
  "performance_ranking_scope": "validation-only candidate and factorial mechanism gate",
  "source_sync_manifest_required": true
}
JSON

cat > "${OUTPUT_ROOT}/acceptance_contract.json" <<'JSON'
{
  "schema_version": 1,
  "decision_split": "validation",
  "decision_checkpoint": "best_val_nll",
  "threshold_precision_policy": "compare against the unrounded numeric values in this file",
  "frozen_v2": {
    "checkpoint_epoch": 19,
    "validation_samples": 41901,
    "total_nll": 5.666519984041392,
    "marker_nll": 0.9912735789787377,
    "time_nll": 4.675246420357495,
    "raw_qty_mae": 3.060181811903915,
    "history_le_4_raw_qty_mae": 2.2961239370036615,
    "log2_qty_mae": 0.5887420091483135,
    "mark_accuracy": 0.5724923032863177,
    "dt_mae": 42.06458129868246,
    "true_mark_0_share": 0.4118040142240042,
    "pred_mark_0_share": 0.45029951552468916,
    "mark_1_recall": 0.4961617829137433,
    "scale_1_9_share": 0.8866614161953176,
    "scale_1_9_raw_qty_mae": 0.9797524960260777,
    "scale_10_99_share": 0.10722894441660104,
    "scale_10_99_raw_qty_mae": 9.318595246379317
  },
  "frozen_q2": {
    "checkpoint_epoch": 46,
    "validation_samples": 41901,
    "total_nll": 5.625528356647216,
    "marker_nll": 0.9914524970295088,
    "time_nll": 4.634075864078703,
    "raw_qty_mae": 2.6064578366078743,
    "history_le_4_raw_qty_mae": 1.9554197624079517,
    "log2_qty_mae": 0.6317777496878308,
    "mark_accuracy": 0.5399632467005561,
    "dt_mae": 41.6666394855489,
    "pred_mark_0_share": 0.6123481539820052,
    "mark_1_recall": 0.17730086669418077
  },
  "fresh_q2_reproduction_gate": {
    "relative_difference_max": {
      "total_nll": 0.01,
      "raw_qty_mae": 0.01,
      "log2_qty_mae": 0.01
    },
    "mark_accuracy_absolute_difference_max": 0.0025,
    "exact_match_required": [
      "train_validation_test_sample_counts",
      "train_event_count",
      "train_raw_moments",
      "model_and_optimizer_config",
      "checkpoint_policy"
    ],
    "failure_action": "stop Q3 attribution and investigate code or data drift"
  },
  "candidate_full_gate": {
    "raw_qty_mae_max": 2.7367807284382684,
    "history_le_4_raw_qty_mae_max": 2.0531907505283495,
    "log2_qty_mae_max": 0.6005168493312798,
    "scale_1_9_raw_qty_mae_max": 0.9993475459465992,
    "scale_10_99_raw_qty_mae_max": 9.784525008698283,
    "marker_nll_max": 1.0011863147685252,
    "total_nll_max": 5.694852583961598,
    "time_nll_max": 4.6986226524592825,
    "mark_accuracy_min": 0.5699923032863178,
    "dt_mae_max": 42.905872924656116,
    "pred_mark_0_absolute_share_error_max": 0.05849550130068497,
    "mark_1_recall_min": 0.4461617829137433,
    "preclamp_negative_share_max": 0.01,
    "normalized_target_nonfinite_count_max": 0,
    "all_applicable_metrics_finite": true
  },
  "mechanism_diagnostics": {
    "q3a_mark_accuracy_recovery_ratio_min": 0.50,
    "q3a_raw_and_short_mae_regression_vs_fresh_q2_max": 0.05,
    "q3b_log2_mae_improvement_vs_fresh_q2_min": 0.05,
    "q3b_raw_and_short_mae_regression_vs_fresh_q2_max": 0.05,
    "factorial_interaction_formula": "(Q3c - Q3a) - (Q3b - Q2)"
  },
  "selection_rule": [
    "complete all four variants before selection",
    "discard every candidate that fails any full-gate condition",
    "prefer a passing single intervention over Q3c",
    "if Q3a and Q3b both pass, prefer Q3a",
    "select Q3c only when the combination is required to pass",
    "if no Q3 candidate passes, retain V2"
  ],
  "held_out_lock": {
    "locked_until": "a frozen seed-42 candidate passes and then passes strict matched multi-seed",
    "do_not_read_before_decision": [
      "leaderboard/runs.csv",
      "leaderboard/test_*",
      "run-local test_*",
      "paper_outputs/report.md",
      "test plots"
    ],
    "allowed_before_decision": [
      "experiment_manifest.json",
      "acceptance_contract.json",
      "logs/run.log",
      "variant_status.tsv",
      "leaderboard/summary.csv",
      "leaderboard/histories.csv",
      "leaderboard/scale_wise_summary.csv",
      "validation confusion and class metrics",
      "validation plots"
    ]
  }
}
JSON

printf 'variant\tgradient_mode\taux_loss_mode\texit_code\n' > "${STATUS_PATH}"

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
  require_file "${FROZEN_Q2_SUCCESS}" || exit 2
  require_file "${FROZEN_Q2_SUMMARY}" || exit 2
  require_file "${INTER_SPLIT_WITH}" || exit 2
  require_file "${INTER_SPLIT_TRAIN}" || exit 2
  require_file "${INTER_SPLIT_VALIDATION}" || exit 2
  require_file "${INTER_SPLIT_TEST}" || exit 2
  require_file "${INTER_SPLIT_MANIFEST}" || exit 2
  require_sha256 "${V2_CHECKPOINT}" "${V2_CHECKPOINT_SHA256}" || exit 2
  require_sha256 "${V2_MARKED}" "${V2_MARKED_SHA256}" || exit 2
  require_sha256 "${FROZEN_Q2_SUMMARY}" "${FROZEN_Q2_SUMMARY_SHA256}" || exit 2
  require_sha256 "${INTER_SPLIT_WITH}" "${INTER_SPLIT_WITH_SHA256}" || exit 2
  require_sha256 "${INTER_SPLIT_TRAIN}" "${INTER_SPLIT_TRAIN_SHA256}" || exit 2
  require_sha256 "${INTER_SPLIT_VALIDATION}" "${INTER_SPLIT_VALIDATION_SHA256}" || exit 2
  require_sha256 "${INTER_SPLIT_TEST}" "${INTER_SPLIT_TEST_SHA256}" || exit 2
  require_sha256 "${INTER_SPLIT_MANIFEST}" "${INTER_SPLIT_MANIFEST_SHA256}" || exit 2

  echo "[stage] freeze V2 validation-only reference"
  run_validation_reference || exit $?

  echo "[stage] train matched Q2/Q3a/Q3b/Q3c"
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
  touch "${OUTPUT_ROOT}/SCREENING_SUCCESS"
else
  touch "${OUTPUT_ROOT}/SCREENING_FAILED"
fi
exit "${status}"
