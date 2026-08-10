#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/sample_data/intermittent_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/search_artifacts/intermittent_frozen_5000_e300_20260810}"
EXECUTION_SERVER="${EXECUTION_SERVER:-5080}"
TMUX_SESSION="${TMUX_SESSION:-intermittent_frozen_5000_e300}"
MODELS="${MODELS:-rmtpp,thp,titantpp}"
EXPECTED_RUN_COUNT="${EXPECTED_RUN_COUNT:-9}"
EXECUTION_ROLE="${EXECUTION_ROLE:-all_models}"
DRY_RUN="${DRY_RUN:-0}"
: "${SOURCE_REVISION:?SOURCE_REVISION must be the frozen 40-character source revision}"

PREFIX="intermittent_frozen_5000"
export PROJECT_ROOT PYTHON_BIN DATA_ROOT OUTPUT_ROOT EXECUTION_SERVER TMUX_SESSION PREFIX
export MODELS EXPECTED_RUN_COUNT EXECUTION_ROLE
WITH_SPLIT="${DATA_ROOT}/${PREFIX}_with_split.parquet"
TRAIN="${DATA_ROOT}/${PREFIX}_train.parquet"
VALIDATION="${DATA_ROOT}/${PREFIX}_validation.parquet"
TEST="${DATA_ROOT}/${PREFIX}_test.parquet"
SPLIT_MANIFEST="${DATA_ROOT}/${PREFIX}_split_manifest.json"
SAMPLING_MANIFEST="${DATA_ROOT}/${PREFIX}_sampling_manifest.json"

[[ "${EXECUTION_SERVER}" == "5080" || "${EXECUTION_SERVER}" == "5090" ]]
[[ "${SOURCE_REVISION}" =~ ^[0-9a-f]{40}$ ]]
[[ "${EXPECTED_RUN_COUNT}" =~ ^[1-9][0-9]*$ ]]
[[ -x "${PYTHON_BIN}" ]]
for path in "${WITH_SPLIT}" "${TRAIN}" "${VALIDATION}" "${TEST}" "${SPLIT_MANIFEST}" "${SAMPLING_MANIFEST}"; do
  [[ -f "${path}" ]] || { echo "[preflight_error] missing ${path}" >&2; exit 2; }
done

cd "${PROJECT_ROOT}"
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg SOURCE_REVISION
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_inter_frozen_5000}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_inter_frozen_5000}"

CMD=(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/simple_lab_test/search/tpp_experiment.py" long-epoch
  --base-dir "${OUTPUT_ROOT}/intermittent"
  --datasets intermittent
  --models "${MODELS}"
  --titan-candidates small_lmm
  --thp-candidates small
  --epochs 300
  --seeds 42,52,62
  --lr 1e-3
  --lambda-dt 1.0
  --batch-size 128
  --lookback-weeks 520
  --max-seq-len 96
  --intermittent-runtime-profile long
  --intermittent-split-with-path "${WITH_SPLIT}"
  --intermittent-split-train-path "${TRAIN}"
  --intermittent-split-validation-path "${VALIDATION}"
  --intermittent-split-test-path "${TEST}"
  --intermittent-split-manifest-path "${SPLIT_MANIFEST}"
  --split-mode fixed
  --evaluation-scope validation_only
  --reproducibility-mode strict
  --device cuda
  --rmtpp-rnn-type gru
  --rmtpp-hidden-dim 64
  --rmtpp-mark-emb-dim 32
  --loss-mode hybrid
  --value-input-mode residual
  --value-input-emb-dim 8
  --value-head-activation identity
  --value-head-mode shared
  --time-head-mode shared
  --qty-mark-gradient-mode coupled
  --value-encoder-gradient-mode coupled
  --marker-loss-mode ce
  --lambda-ordinal 0
  --qty-decoder-mode mark_residual
  --train-loss-scope target_only
  --test-time-memory none
  --analysis-scale-base 2
  --analysis-tail-order 7
  --eval-selections best_val_nll
  --early-stopping-patience 40
  --min-epochs 40
  --stop-on-error
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[dry_run]'; printf ' %q' "${CMD[@]}"; printf '\n'; exit 0
fi

mkdir -p "${OUTPUT_ROOT}/logs"
exec > >(tee -a "${OUTPUT_ROOT}/logs/launcher.log") 2>&1

"${PYTHON_BIN}" -c 'import hashlib,json,os,platform
from datetime import datetime
from pathlib import Path
root=Path(os.environ["OUTPUT_ROOT"]); data=Path(os.environ["DATA_ROOT"]); prefix=os.environ["PREFIX"]
files={name:data/f"{prefix}_{name}" for name in ["with_split.parquet","train.parquet","validation.parquet","test.parquet","split_manifest.json","sampling_manifest.json"]}
payload={"schema_version":1,"status":"RUNNING","started_at":datetime.now().astimezone().isoformat(),"execution_server":os.environ["EXECUTION_SERVER"],"execution_role":os.environ["EXECUTION_ROLE"],"hostname":platform.node(),"tmux_session":os.environ["TMUX_SESSION"],"source_revision":os.environ["SOURCE_REVISION"],"expected_run_count":int(os.environ["EXPECTED_RUN_COUNT"]),"models":[name.strip() for name in os.environ["MODELS"].split(",") if name.strip()],"seeds":[42,52,62],"max_epochs":300,"early_stopping":{"metric":"validation_nll","min_epochs":40,"patience":40},"evaluation_scope":"validation_only","held_out_test_evaluated":False,"dataset_sha256":{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}}
(root/"launch_contract.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")'

"${CMD[@]}"

"${PYTHON_BIN}" -c 'import json,os
from datetime import datetime
from pathlib import Path
p=Path(os.environ["OUTPUT_ROOT"])/"launch_contract.json"; d=json.loads(p.read_text()); d["status"]="COMPLETE"; d["completed_at"]=datetime.now().astimezone().isoformat(); p.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")'
