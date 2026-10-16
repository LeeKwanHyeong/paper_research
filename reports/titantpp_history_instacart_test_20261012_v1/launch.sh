#!/bin/bash
set -u
TASK_ROOT=/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_instacart_test_20261012_v1_5080
TASK_PYTHON=/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python3.12
cd "$TASK_ROOT" || exit 120
export PYTHONDONTWRITEBYTECODE=1
"$TASK_PYTHON" "$TASK_ROOT/control/pipeline.py" --root "$TASK_ROOT" --operation-seal-sha256 "$1" >> "$TASK_ROOT/control/pipeline.log" 2>&1
TASK_EXIT=$?
"$TASK_PYTHON" -c 'import json,sys,time,pathlib; p=pathlib.Path(sys.argv[2])/"control/pipeline_exit.json"; f=p.open("x"); json.dump({"exit_code":int(sys.argv[1]),"actual_child_exit_observed":True,"observed_unix":time.time()},f,indent=2);f.write("\n");f.close()' "$TASK_EXIT" "$TASK_ROOT"
exit "$TASK_EXIT"
