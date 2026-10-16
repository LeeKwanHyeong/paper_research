"""Verify H1 actual completion and pinned local/remote originals before H2 starts."""
from pathlib import Path
import json,hashlib

def require(v,msg):
 if not v:raise ValueError(msg)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(p.read_bytes())
def verify(root,c):
 p=c['prerequisite_H1'];remote=Path(p['remote_root'])
 require(str(remote)=='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_raf_log_raw_20261012_v2_5080','Wrong H1 prerequisite root')
 require(p['status']=='complete' and p['actual_pipeline_exit0'] is True and p['actual_all_worker_exit0'] is True,'H1 prerequisite not complete')
 for name,e in p['evidence_files'].items():
  copy=(root/e['relative_path']).resolve()
  require(copy.is_relative_to(root) and sha(copy)==e['sha256'],'H1 copied evidence differs: '+name)
  if name.startswith('control/') or name in ('originals/receipt.json','originals/manifest.json'):
   remote_rel=name.replace('originals/','retrieval/')
   require(sha(remote/remote_rel)==e['sha256'],'H1 remote terminal evidence changed: '+name)
 require(sha(remote/'operation_integrity.json')==p['operation_integrity_sha256'],'H1 operation changed')
 require(sha(remote/'retrieval/originals.tar.gz')==p['archive_sha256'] and sha(remote/'retrieval/manifest.json')==p['manifest_sha256'],'H1 original binary archive changed')
 completed=read(remote/'control/pipeline_complete.json');ex=read(remote/'control/pipeline_exit.json')
 require(completed['contract_sha256']==p['contract_sha256'] and completed['model_split_passes']==18
  and completed['actual_all_worker_exit0'] is True and ex['exit_code']==0 and ex['actual_child_exit_observed'] is True,'H1 actual18 completion changed')
 v=read(root/p['evidence_files']['originals/LOCAL_VERIFICATION.json']['relative_path'])
 require(v['status']=='verified' and v['all_archive_members_and_local_files_sha_verified'] is True
  and v['archive_sha256']==p['archive_sha256'] and v['manifest_sha256']==p['manifest_sha256'],'H1 local binary verification absent')
 qa=(root/p['evidence_files']['FINAL_QA.md']['relative_path']).read_text()
 require([line.strip() for line in qa.splitlines() if line.startswith('## Review Result:')]==['## Review Result: PASS'],'H1 independent QA absent')
 a=read(root/p['evidence_files']['analysis/completion_receipt.json']['relative_path'])
 require(a['status']=='complete' and a['contract_sha256']==p['contract_sha256'] and a['actual_worker_exit0_count']==18
  and a['actual_pipeline_os_exit0_verified'] is True,'H1 completed analysis absent')
 return True
