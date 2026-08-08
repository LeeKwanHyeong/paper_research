#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/search_artifacts/final_fair_titantpp_e300_20260808}"
EXECUTION_SERVER="${EXECUTION_SERVER:-5080}"
TMUX_SESSION="${TMUX_SESSION:-titan_final_fair_titantpp_e300}"
DRY_RUN="${DRY_RUN:-0}"

: "${SOURCE_REVISION:?SOURCE_REVISION must be the frozen 40-character source revision}"

if [[ "${EXECUTION_SERVER}" != "5080" ]]; then
  echo "[preflight_error] EXECUTION_SERVER must be 5080; received ${EXECUTION_SERVER}." >&2
  exit 2
fi
if [[ ! "${SOURCE_REVISION}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "[preflight_error] SOURCE_REVISION must be a lowercase 40-character Git SHA." >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[preflight_error] Python is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SOURCE_REVISION
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_paper_research_titantpp_e300}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache_paper_research_titantpp_e300}"
export PROJECT_ROOT PYTHON_BIN OUTPUT_ROOT EXECUTION_SERVER TMUX_SESSION

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${OUTPUT_ROOT}/logs"
  exec > >(tee -a "${OUTPUT_ROOT}/logs/launcher.log") 2>&1
fi

COMMON_ARGS=(
  long-epoch
  --models titantpp
  --epochs 300
  --seeds 42,52,62
  --lr 1e-3
  --lambda-dt 1.0
  --batch-size 128
  --split-mode fixed
  --evaluation-scope validation_only
  --reproducibility-mode strict
  --device cuda
  --titan-profile dataset_best
  --rmtpp-rnn-type gru
  --rmtpp-mark-emb-dim 32
  --loss-mode hybrid
  --value-input-mode residual
  --value-input-emb-dim 8
  --value-head-activation identity
  --time-head-mode shared
  --value-encoder-gradient-mode coupled
  --marker-loss-mode ce
  --lambda-ordinal 0
  --qty-decoder-mode mark_residual
  --train-loss-scope target_only
  --test-time-memory none
  --eval-selections best_val_nll
  --stop-on-error
)

run_dataset() {
  local dataset="$1"
  local paper_label="$2"
  local candidate="$3"
  local hidden_dim="$4"
  local lookback="$5"
  local max_seq_len="$6"
  local scale_base="$7"
  local value_head_mode="$8"
  local qty_mark_gradient_mode="$9"
  local dataset_output="${OUTPUT_ROOT}/${dataset}"
  local cmd=(
    "${PYTHON_BIN}"
    "${PROJECT_ROOT}/simple_lab_test/search/tpp_experiment.py"
    "${COMMON_ARGS[@]}"
    --base-dir "${dataset_output}"
    --datasets "${dataset}"
    --titan-candidates "${candidate}"
    --rmtpp-hidden-dim "${hidden_dim}"
    --lookback-weeks "${lookback}"
    --max-seq-len "${max_seq_len}"
    --analysis-scale-base "${scale_base}"
    --analysis-tail-order 4
    --value-head-mode "${value_head_mode}"
    --qty-mark-gradient-mode "${qty_mark_gradient_mode}"
  )

  printf '[queue] dataset=%s model=%s candidate=%s seeds=42,52,62 runs=3 output=%s\n' \
    "${dataset}" "${paper_label}" "${candidate}" "${dataset_output}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry_run]'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  "${cmd[@]}"
}

if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON_BIN}" -c \
    'import hashlib,json,os,platform,sys
from datetime import datetime
from pathlib import Path
root=Path(os.environ["OUTPUT_ROOT"])
project=Path(os.environ["PROJECT_ROOT"])
files=[
 "simple_lab_test/search/tpp_experiment.py",
 "simple_lab_test/search/common/benchmark_utils.py",
 "simple_lab_test/search/common/configs.py",
 "simple_lab_test/search/common/experiment_utils.py",
 "simple_lab_test/search/common/models.py",
 "simple_lab_test/search/common/runner.py",
 "models/RMTPPs/RMTPP.py",
 "models/RMTPPs/TitanTPP.py",
 "models/RMTPPs/TransformerHawkesTPP.py",
 "models/RMTPPs/config.py",
 "models/RMTPPs/value_conditioning.py",
 "utils/training.py",
]
payload={
 "schema_version":1,
 "status":"RUNNING",
 "started_at":datetime.now().astimezone().isoformat(),
 "execution_server":os.environ["EXECUTION_SERVER"],
 "hostname":platform.node(),
 "tmux_session":os.environ["TMUX_SESSION"],
 "source_revision":os.environ["SOURCE_REVISION"],
 "python":sys.executable,
 "expected_run_count":9,
 "evaluation_scope":"validation_only",
 "checkpoint_rule":"best_val_nll",
 "held_out_test_evaluated":False,
 "queue":[
  {"dataset":"yellow_trip_hourly","paper_label":"TitanTPP V3b","candidate":"mid_lmm","seeds":[42,52,62],"runs":3},
  {"dataset":"intermittent","paper_label":"TitanTPP V2","candidate":"small_lmm","seeds":[42,52,62],"runs":3},
  {"dataset":"insta_market_basket","paper_label":"TitanTPP V2","candidate":"small_lmm","seeds":[42,52,62],"runs":3},
 ],
 "source_sha256":{name:hashlib.sha256((project/name).read_bytes()).hexdigest() for name in files},
}
(root/"launch_contract.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")'
fi

echo "[launch] source_revision=${SOURCE_REVISION} server=${EXECUTION_SERVER} tmux=${TMUX_SESSION}"
echo "[launch] expected_runs=9 evaluation_scope=validation_only checkpoint=best_val_nll"

run_dataset yellow_trip_hourly "TitanTPP V3b" mid_lmm 128 168 256 10 mark_conditioned_experts detached
run_dataset intermittent "TitanTPP V2" small_lmm 64 52 16 2 shared coupled
run_dataset insta_market_basket "TitanTPP V2" small_lmm 64 52 64 2 shared coupled

if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON_BIN}" -c \
    'import json,os
from datetime import datetime
from pathlib import Path
p=Path(os.environ["OUTPUT_ROOT"])/"launch_contract.json"
payload=json.loads(p.read_text(encoding="utf-8"))
payload["status"]="COMPLETE"
payload["completed_at"]=datetime.now().astimezone().isoformat()
p.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")'
fi

echo "[complete] final_fair_titantpp_e300 expected_runs=9"
