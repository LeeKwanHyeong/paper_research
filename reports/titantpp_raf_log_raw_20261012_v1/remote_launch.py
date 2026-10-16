from pathlib import Path
import hashlib
import json
import shlex
import subprocess
import sys
import time

root=Path('/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_raf_log_raw_20261012_v1_5080')
seal=sys.argv[1]
contract='b1d554f5e80e874b8c7a62aec5b3c113d6b74a2a956f9044964f910cdee56f4c'
session='titantpp_raf_log_raw_20261012_v1'
def require(value,message):
 if not value:raise ValueError(message)
def read(path):return json.loads(path.read_bytes())
def write(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
require(hashlib.sha256((root/'operation_integrity.json').read_bytes()).hexdigest()==seal,'Operation seal differs')
c=read(root/'contract.json');d=read(root/'control/deployment_receipt.json')
require(c['contract_sha256']==d['contract_sha256']==contract and d['status']=='verified','Deployment binding differs')
require(not any((root/'control'/name).exists() for name in ['start_permit.json','pipeline_claim.json','pipeline_failure.json','pipeline_exit.json']),'Prior launch state prohibits retry')
require(not Path('/var/run/reboot-required').exists(),'Reboot pending')
compute=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
require(not compute,'GPU busy; no launch')
existing=subprocess.run(['/usr/bin/tmux','has-session','-t',session],capture_output=True,text=True)
require(existing.returncode!=0,'Diagnostic tmux session already exists')
intent={'contract_sha256':contract,'operation_integrity_sha256':seal,'human_request':'RAF18부터 이어서 Intermittent18, 새 학습 없이 Train·Validation만 평가하자',
 'scope':'selected RAF9 x Train/Validation18 inference-only diagnostics','new_fits':0,'interventions':0,
 'session':session,'automatic_retry':False,'created_unix':time.time()}
write(root/'control/start_permit.json',intent)
command='bash '+shlex.quote(str(root/'control/launch.sh'))+' '+shlex.quote(seal)
argv=['/usr/bin/tmux','new-session','-d','-s',session,command]
result=subprocess.run(argv,capture_output=True,text=True)
receipt={**intent,'argv':argv,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
 'actual_launch_command_exit_observed':True,'observed_unix':time.time()}
write(root/'control/launch_receipt.json',receipt)
require(result.returncode==0,'tmux launch failed; preserve intent and no retry')
print(json.dumps(receipt),flush=True)
