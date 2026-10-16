"""One read-only observation of the failed terminal argv verification."""
from pathlib import Path
import json
import subprocess
import time
root=Path('/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_train_fit_20261012_v1_5080')
def read(p):return json.loads(p.read_bytes())
c=read(root/'contract.json')
rows=[]
for j in c['jobs']:
 d=root/'results'/j['id'];o=read(d/'train_metrics.json');e=read(d/'train_worker_exit.json');l=read(d/'train_launch.json')
 rows.append({'id':j['id'],'worker_pid':o['pid'],'launch_pid':l['pid'],'exit_pid':e['pid'],
  'worker_ppid':o['ppid'],'launch_ppid':l['ppid'],'exit_ppid':e['ppid'],
  'worker_argv':o['argv'],'launch_argv':l['argv'],'exit_argv':e['argv'],'exit_code':e['exit_code'],
  'actual_child_exit_observed':e['actual_child_exit_observed'],'timed_out':e['timed_out'],
  'supervisor_aborted':e['supervisor_aborted'],'receipt_status':o['status'],
  'checkpoint_sha256':o['checkpoint_sha256'],'state_sha256':o['state_sha256'],
  'state_after_sha256':o['state_after_sha256'],'population':o['original_population_identity']})
smi=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],capture_output=True,text=True)
print(json.dumps({'observed_unix':time.time(),'mutations':0,'root':str(root),
 'pipeline_complete':read(root/'control/pipeline_complete.json'),'pipeline_exit':read(root/'control/pipeline_exit.json'),
 'last_phase':read(root/'control/pipeline_latest.json'),'retrieval_claim_exists':(root/'retrieval_claim.json').exists(),
 'retrieval_directory_exists':(root/'retrieval').exists(),'pipeline_failure_exists':(root/'control/pipeline_failure.json').exists(),
 'gpu':{'exit_code':smi.returncode,'stdout':smi.stdout.strip(),'stderr':smi.stderr.strip()},'rows':rows}),flush=True)
