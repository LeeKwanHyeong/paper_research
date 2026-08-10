#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/leekwanhyeong/workspace/paper_research}"
PYTHON_BIN="${PYTHON_BIN:-/opt/miniconda3/envs/ai_env/bin/python}"
TMUX_SESSION="${TMUX_SESSION:-inter_mark_cap5_titantpp_e300}"
EXECUTION_SERVER="${EXECUTION_SERVER:-5090}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/search_artifacts/inter_mark_cap5_titantpp_e300_20260810}"
SPLIT_ROOT="${SPLIT_ROOT:-${OUTPUT_ROOT}/fixed_split_cap5}"
DRY_RUN="${DRY_RUN:-0}"

: "${SOURCE_REVISION:?SOURCE_REVISION must be the checksum-verified 40-character Git SHA}"

if [[ "${EXECUTION_SERVER}" != "5090" ]]; then
  echo "[preflight_error] EXECUTION_SERVER must be 5090; received ${EXECUTION_SERVER}." >&2
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
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig_inter_mark_cap5}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache_inter_mark_cap5}"
export PROJECT_ROOT PYTHON_BIN OUTPUT_ROOT SPLIT_ROOT EXECUTION_SERVER TMUX_SESSION SOURCE_REVISION

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${OUTPUT_ROOT}/logs"
  exec > >(tee -a "${OUTPUT_ROOT}/logs/launcher.log") 2>&1
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON_BIN}" "${PROJECT_ROOT}/simple_lab_test/search/scripts/build_inter_mark_cap_split.py" \
    --source-dir "${PROJECT_ROOT}/sample_data/head_office" \
    --output-dir "${SPLIT_ROOT}" \
    --prefix marked_target_cap5 \
    --max-order 5 \
    --scale-base 2.0

  "${PYTHON_BIN}" -c \
    'import hashlib,json,os,platform,sys
from datetime import datetime
from pathlib import Path
root=Path(os.environ["OUTPUT_ROOT"])
project=Path(os.environ["PROJECT_ROOT"])
files=[
 "simple_lab_test/search/tpp_experiment.py",
 "simple_lab_test/search/scripts/build_inter_mark_cap_split.py",
 "simple_lab_test/search/common/benchmark_utils.py",
 "simple_lab_test/search/common/configs.py",
 "simple_lab_test/search/common/experiment_utils.py",
 "simple_lab_test/search/common/models.py",
 "simple_lab_test/search/common/runner.py",
 "models/RMTPPs/TitanTPP.py",
 "models/RMTPPs/config.py",
 "models/RMTPPs/value_conditioning.py",
 "utils/training.py",
]
payload={
 "schema_version":1,
 "status":"RUNNING",
 "started_at":datetime.now().astimezone().isoformat(),
 "purpose":"Intermittent capped-tail mark-design screening",
 "execution_server":os.environ["EXECUTION_SERVER"],
 "hostname":platform.node(),
 "tmux_session":os.environ["TMUX_SESSION"],
 "source_revision":os.environ["SOURCE_REVISION"],
 "python":sys.executable,
 "expected_run_count":3,
 "evaluation_scope":"validation_only",
 "checkpoint_rule":"best_val_nll",
 "held_out_test_evaluated":False,
 "dataset":"intermittent",
 "mark_design":{"scale_base":2.0,"max_order":5,"num_real_marks":6,"tail_definition":"demand_qty >= 32"},
 "queue":[{"dataset":"intermittent","paper_label":"TitanTPP V2 cap5","candidate":"small_lmm","seeds":[42,52,62],"runs":3}],
 "source_sha256":{name:hashlib.sha256((project/name).read_bytes()).hexdigest() for name in files},
}
(root/"launch_contract.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")'
fi

cmd=(
  "${PYTHON_BIN}"
  "${PROJECT_ROOT}/simple_lab_test/search/tpp_experiment.py"
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
  --titan-candidates small_lmm
  --lookback-weeks 52
  --max-seq-len 16
  --analysis-scale-base 2
  --analysis-tail-order 5
  --value-head-mode shared
  --qty-mark-gradient-mode coupled
  --value-head-activation identity
  --loss-mode hybrid
  --value-input-mode residual
  --value-input-emb-dim 8
  --qty-decoder-mode mark_residual
  --train-loss-scope target_only
  --test-time-memory none
  --eval-selections best_val_nll
  --base-dir "${OUTPUT_ROOT}/intermittent_cap5"
  --datasets intermittent
  --intermittent-split-with-path "${SPLIT_ROOT}/marked_target_cap5_with_split.parquet"
  --intermittent-split-train-path "${SPLIT_ROOT}/marked_target_cap5_train.parquet"
  --intermittent-split-validation-path "${SPLIT_ROOT}/marked_target_cap5_validation.parquet"
  --intermittent-split-test-path "${SPLIT_ROOT}/marked_target_cap5_test.parquet"
  --intermittent-split-manifest-path "${SPLIT_ROOT}/marked_target_cap5_split_manifest.json"
  --stop-on-error
)

echo "[launch] source_revision=${SOURCE_REVISION} server=${EXECUTION_SERVER} tmux=${TMUX_SESSION}"
echo "[launch] expected_runs=3 dataset=intermittent model=TitanTPP mark_design=cap5 evaluation_scope=validation_only"
if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[dry_run]'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}"

"${PYTHON_BIN}" -c \
  'import json,os
from datetime import datetime
from pathlib import Path
p=Path(os.environ["OUTPUT_ROOT"])/"launch_contract.json"
payload=json.loads(p.read_text(encoding="utf-8"))
payload["status"]="COMPLETE"
payload["completed_at"]=datetime.now().astimezone().isoformat()
p.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")'

echo "[complete] inter_mark_cap5_titantpp_e300 expected_runs=3"
