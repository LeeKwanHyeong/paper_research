"""Observe initial entry once, wait for supervised exit, then exclusive retrieval."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

root=Path('/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_raf_log_raw_20261012_v2_5080')
seal=sys.argv[1]
contract='0c75bdd4d70adf8569ea2602c942e93f6b27b1206a4595711b2673997ee0758a'
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
 subprocess.run(['/usr/bin/tmux','wait-for','titantpp-raf-log-raw-20261012-v2-complete'],check=True)
completion=read(root/'control/pipeline_complete.json');exit_receipt=read(root/'control/pipeline_exit.json')
if completion is None or exit_receipt is None or exit_receipt['exit_code']!=0:
 print(json.dumps({'status':'not_complete','pipeline_exit':exit_receipt,
  'failure':read(root/'control/pipeline_failure.json'),'state':read(root/'control/pipeline_latest.json')},ensure_ascii=False),flush=True)
 raise SystemExit(1)
subprocess.run([sys.executable,str(root/'control/finalize.py'),str(root),seal,contract],check=True)
