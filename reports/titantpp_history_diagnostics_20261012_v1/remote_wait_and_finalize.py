"""Observe initial entry once, wait for supervised exit, then exclusive retrieval."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

root=Path('/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_diagnostics_20261012_v1_5080')
seal='83006f2d337462a6396e50a7f15b791ef67aa6cf1232d5ad249342a24a9c42e7'
contract='38fb8820071fdb9b13f1576528d03da952a5a81a7e51b1a9db5d0cd97553ace4'
def read(path):return json.loads(path.read_bytes()) if path.exists() else None
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
 subprocess.run(['/usr/bin/tmux','wait-for','titantpp-history-diagnostics-20261012-complete'],check=True)
completion=read(root/'control/pipeline_complete.json');exit_receipt=read(root/'control/pipeline_exit.json')
if completion is None or exit_receipt is None or exit_receipt['exit_code']!=0:
 print(json.dumps({'status':'not_complete','pipeline_exit':exit_receipt,
  'failure':read(root/'control/pipeline_failure.json'),'state':read(root/'control/pipeline_latest.json')},ensure_ascii=False),flush=True)
 raise SystemExit(1)
subprocess.run([sys.executable,str(root/'control/finalize.py'),str(root),seal,contract],check=True)
