#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ai_env/bin/python}"
ENTRYPOINT="${PROJECT_ROOT}/simple_lab_test/search/tpp_experiment.py"
COMPARATOR="${PROJECT_ROOT}/simple_lab_test/search/compare_titantpp_strict_repro.py"
ARTIFACT_ROOT="${PROJECT_ROOT}/search_artifacts"
EXPERIMENT_ID="${EXPERIMENT_ID:-model_enhancement_titantpp_q2_strict_repro_e3_0715}"
OUTPUT_ROOT="${ARTIFACT_ROOT}/${EXPERIMENT_ID}"
RUN_A_ROOT="${OUTPUT_ROOT}/run_a"
RUN_B_ROOT="${OUTPUT_ROOT}/run_b"
LOG_DIR="${OUTPUT_ROOT}/logs"
LOG_PATH="${LOG_DIR}/run.log"
COMPARATOR_LOG="${LOG_DIR}/exact_comparator.log"
STATUS_PATH="${OUTPUT_ROOT}/process_status.tsv"
REPORT_PATH="${OUTPUT_ROOT}/exact_reproduction_report.json"
ACCEPTANCE_PATH="${OUTPUT_ROOT}/acceptance_contract.json"
SOURCE_SYNC_MANIFEST="${SOURCE_SYNC_MANIFEST:-${OUTPUT_ROOT}/source_sync_manifest.json}"
SOURCE_REVISION="${SOURCE_REVISION:-}"
CUDA13_LIB="/opt/miniconda3/envs/ai_env/lib/python3.12/site-packages/nvidia/cu13/lib"

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

mkdir -p "${LOG_DIR}"
rm -f "${OUTPUT_ROOT}/REPRO_SUCCESS" "${OUTPUT_ROOT}/REPRO_FAILED" "${REPORT_PATH}"
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

validate_source_revision() {
  if [[ ! "${SOURCE_REVISION}" =~ ^([0-9a-fA-F]{40}|[0-9a-fA-F]{64})$ ]]; then
    echo "[preflight_error] SOURCE_REVISION must be a full 40- or 64-character hexadecimal revision"
    return 1
  fi
}

validate_source_sync_manifest() {
  "${PYTHON_BIN}" -c '
import json
import sys

path, expected = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as file_obj:
    manifest = json.load(file_obj)
actual = manifest.get("synced_revision", manifest.get("source_revision"))
if actual != expected:
    raise SystemExit(f"source revision mismatch: expected={expected} actual={actual}")
print(f"[preflight_source_revision] {actual}")
' "${SOURCE_SYNC_MANIFEST}" "${SOURCE_REVISION}"
}

run_strict_cuda_preflight() {
  env \
    PYTHONHASHSEED=42 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    SOURCE_REVISION="${SOURCE_REVISION}" \
    XDG_CACHE_HOME="${OUTPUT_ROOT}/preflight-cache" \
    MPLCONFIGDIR="${OUTPUT_ROOT}/preflight-matplotlib" \
    "${PYTHON_BIN}" -c '
import json
import torch
from simple_lab_test.search.common.runner import configure_reproducibility

runtime = configure_reproducibility("strict")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
probe = torch.ones(1, device="cuda")
if not bool(torch.isfinite(probe).all().item()):
    raise SystemExit("CUDA allocation produced a non-finite tensor")
print(json.dumps({
    "device": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "deterministic": runtime["torch_deterministic_algorithms"],
    "warn_only": runtime["torch_deterministic_warn_only"],
}, sort_keys=True))
'
}

run_probe_process() {
  local process_name="$1"
  local process_root="$2"

  mkdir -p "${process_root}/logs"
  echo "[process_start] ${process_name} $(date '+%Y-%m-%d %H:%M:%S %Z')"
  env \
    PYTHONHASHSEED=42 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    SOURCE_REVISION="${SOURCE_REVISION}" \
    XDG_CACHE_HOME="${process_root}/runtime-cache" \
    MPLCONFIGDIR="${process_root}/matplotlib-cache" \
    "${PYTHON_BIN}" "${ENTRYPOINT}" long-epoch \
    --base-dir "${process_root}" \
    --datasets intermittent \
    --models titantpp \
    --titan-profile dataset_best \
    --titan-candidates small_lmm \
    --epochs 3 \
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
    --magnitude-encoder-gradient-mode coupled \
    --magnitude-aux-loss-mode none \
    --lambda-log-qty 0.25 \
    --log-qty-huber-delta 1.0 \
    --log-qty-floor 1.0 \
    --magnitude-sigma-floor 0.0550124034288891 \
    --magnitude-revin-eps 1e-5 \
    --magnitude-shrinkage-k 8 \
    --magnitude-center-mode mean \
    --no-magnitude-revin-affine \
    --magnitude-stat-context-mode none \
    --test-time-memory none \
    --analysis-scale-base 10 \
    --analysis-tail-order 4 \
    --eval-selections best_val_nll,best_score,final \
    --reproducibility-mode strict \
    --device cuda \
    --force-rerun \
    --stop-on-error \
    2>&1 | tee "${process_root}/logs/run.log"
  local status=${PIPESTATUS[0]}
  printf '%s\t%s\n' "${process_name}" "${status}" >> "${STATUS_PATH}"
  echo "[process_end] ${process_name} exit_code=${status} $(date '+%Y-%m-%d %H:%M:%S %Z')"
  return "${status}"
}

cat > "${OUTPUT_ROOT}/experiment_manifest.json" <<JSON
{
  "experiment_id": "${EXPERIMENT_ID}",
  "started_at": "$(date '+%Y-%m-%d %H:%M:%S %Z')",
  "server": "$(hostname)",
  "device": "cuda",
  "source_revision": "${SOURCE_REVISION}",
  "source_sync_manifest": "${SOURCE_SYNC_MANIFEST}",
  "dataset": "intermittent",
  "split_mode": "fixed",
  "model": "titantpp",
  "candidate": "small_lmm",
  "variant": "Q2 causal-shrinkage raw-quantity control",
  "epochs": 3,
  "seed": 42,
  "learning_rate": 0.001,
  "batch_size": 128,
  "lookback_weeks": 52,
  "max_seq_len": 16,
  "reproducibility_mode": "strict",
  "python_hash_seed": "42",
  "cublas_workspace_config": ":4096:8",
  "processes": {
    "run_a": {"base_dir": "${RUN_A_ROOT}", "process_contract": "fresh independent Python process"},
    "run_b": {"base_dir": "${RUN_B_ROOT}", "process_contract": "fresh independent Python process"}
  },
  "expected_sample_counts": {"train": 136256, "validation": 41901, "test": 41344},
  "fixed_split_sha256": {
    "with_split": "${INTER_SPLIT_WITH_SHA256}",
    "train": "${INTER_SPLIT_TRAIN_SHA256}",
    "validation": "${INTER_SPLIT_VALIDATION_SHA256}",
    "test": "${INTER_SPLIT_TEST_SHA256}",
    "manifest": "${INTER_SPLIT_MANIFEST_SHA256}"
  },
  "acceptance_contract": "${ACCEPTANCE_PATH}",
  "exact_report": "${REPORT_PATH}",
  "held_out_policy": "test files may be hashed for split identity and generated by the shared runner, but the comparator does not read test metrics or plots",
  "performance_ranking_allowed": false
}
JSON

cat > "${ACCEPTANCE_PATH}" <<'JSON'
{
  "schema_version": 1,
  "decision": "strict exact-reproduction infrastructure gate",
  "required_exact_matches": [
    "history.json bytes and SHA256",
    "best_score, best_val_nll, and final selected epochs",
    "best_score, best_val_nll, and final canonical tensor-state SHA256"
  ],
  "required_identity_matches": [
    "full source revision",
    "PyTorch, CUDA, cuDNN, and GPU runtime",
    "strict deterministic flags and process environment",
    "fixed-split dataset SHA256",
    "run, training, RMTPP, and Titan encoder configs",
    "train, validation, and test loader sample counts"
  ],
  "pass_action": "reproducibility infrastructure is accepted; Q3 remains closed unless explicitly reopened",
  "fail_action": "inspect the first differing batch or state and do not run e50",
  "held_out_lock": "the comparator must not read test metrics, test scale files, reports, or test plots"
}
JSON

printf 'stage\texit_code\n' > "${STATUS_PATH}"

{
  echo "[start] $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "[server] $(hostname)"
  echo "[python] ${PYTHON_BIN}"
  echo "[source_revision] ${SOURCE_REVISION}"
  echo "[cuda_lib] ${CUDA13_LIB}"
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

  require_file "${ENTRYPOINT}" || exit 2
  require_file "${COMPARATOR}" || exit 2
  require_file "${SOURCE_SYNC_MANIFEST}" || exit 2
  require_file "${INTER_SPLIT_WITH}" || exit 2
  require_file "${INTER_SPLIT_TRAIN}" || exit 2
  require_file "${INTER_SPLIT_VALIDATION}" || exit 2
  require_file "${INTER_SPLIT_TEST}" || exit 2
  require_file "${INTER_SPLIT_MANIFEST}" || exit 2
  validate_source_revision || exit 2
  validate_source_sync_manifest || exit 2
  require_sha256 "${INTER_SPLIT_WITH}" "${INTER_SPLIT_WITH_SHA256}" || exit 2
  require_sha256 "${INTER_SPLIT_TRAIN}" "${INTER_SPLIT_TRAIN_SHA256}" || exit 2
  require_sha256 "${INTER_SPLIT_VALIDATION}" "${INTER_SPLIT_VALIDATION_SHA256}" || exit 2
  require_sha256 "${INTER_SPLIT_TEST}" "${INTER_SPLIT_TEST_SHA256}" || exit 2
  require_sha256 "${INTER_SPLIT_MANIFEST}" "${INTER_SPLIT_MANIFEST_SHA256}" || exit 2
  run_strict_cuda_preflight || exit 2

  process_status=0
  run_probe_process run_a "${RUN_A_ROOT}" || process_status=1
  run_probe_process run_b "${RUN_B_ROOT}" || process_status=1

  if [[ "${process_status}" -ne 0 ]]; then
    printf 'exact_comparator\t3\n' >> "${STATUS_PATH}"
    echo "[compare_skipped] one or more training processes failed"
    exit 3
  fi

  env \
    XDG_CACHE_HOME="${OUTPUT_ROOT}/comparator-cache" \
    MPLCONFIGDIR="${OUTPUT_ROOT}/comparator-matplotlib" \
    "${PYTHON_BIN}" -m simple_lab_test.search.compare_titantpp_strict_repro \
    --run-a "${RUN_A_ROOT}" \
    --run-b "${RUN_B_ROOT}" \
    --output "${REPORT_PATH}" \
    --expected-epochs 3 \
    --expected-seed 42 \
    --expected-train-samples 136256 \
    --expected-validation-samples 41901 \
    --expected-test-samples 41344 \
    2>&1 | tee "${COMPARATOR_LOG}"
  compare_status=${PIPESTATUS[0]}
  printf 'exact_comparator\t%s\n' "${compare_status}" >> "${STATUS_PATH}"
  echo "[end] $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "[exit_code] ${compare_status}"
  exit "${compare_status}"
} 2>&1 | tee "${LOG_PATH}"

status=${PIPESTATUS[0]}
if [[ "${status}" -eq 0 ]]; then
  touch "${OUTPUT_ROOT}/REPRO_SUCCESS"
else
  touch "${OUTPUT_ROOT}/REPRO_FAILED"
fi
exit "${status}"
