"""Observe initial entry once, wait for supervised exit, then exclusive retrieval."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

root=Path('/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_train_fit_20261012_v1_5080')
seal='157791af8ce197cfc858659e195eda314b2d1de7ca139971d627e51d6b91819b'
contract='76248e2174d7bd651d821f649ad860030da985d59da88a8ce50725b476f55ccf'
def read(path):return json.loads(path.read_bytes()) if path.exists() else None
state=read(root/'control/pipeline_latest.json')
# One bounded initial-entry check; no repeated SSH or full-run polling.
entry_deadline=time.monotonic()+45
while state and state.get('phase')!='train' and state.get('status')=='running' and time.monotonic()<entry_deadline:
 time.sleep(1)
 state=read(root/'control/pipeline_latest.json')
qualification=read(root/'control/native_qualification.json')
processes=[]
for record in ([{'pid':state['pid']}] if state else [])+(state or {}).get('active',[]):
 pid=record['pid'];proc=Path('/proc')/str(pid)
 processes.append({'pid':pid,'exists':proc.exists(),
  'argv':proc.joinpath('cmdline').read_bytes().decode().replace('\0',' ').strip() if proc.exists() else None})
smi=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],capture_output=True,text=True)
print(json.dumps({'observation':'initial_entry','observed_unix':time.time(),'state':state,
 'native_qualification':qualification,'processes':processes,
 'gpu_processes':{'exit_code':smi.returncode,'stdout':smi.stdout.strip(),'stderr':smi.stderr.strip()}},ensure_ascii=False),flush=True)
if not (root/'control/pipeline_exit.json').exists():
 subprocess.run(['/usr/bin/tmux','wait-for','titantpp-history-train-fit-20261012-v1-complete'],check=True)
completion=read(root/'control/pipeline_complete.json');exit_receipt=read(root/'control/pipeline_exit.json')
if completion is None or exit_receipt is None or exit_receipt['exit_code']!=0:
 print(json.dumps({'status':'not_complete','pipeline_exit':exit_receipt,
  'failure':read(root/'control/pipeline_failure.json'),'state':read(root/'control/pipeline_latest.json')},ensure_ascii=False),flush=True)
 raise SystemExit(1)
subprocess.run([sys.executable,str(root/'control/finalize.py'),str(root),seal,contract],check=True)
