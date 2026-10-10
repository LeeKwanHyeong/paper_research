"""Preserve completed Instacart originals without changing the sealed campaign.

NVML global idle is explicitly not asserted. This authorized recovery verifies
exact owner exit, absent owner processes/device clients, GPU UUID from procfs,
and all terminal binaries, then makes one immutable archive per host.
"""
from pathlib import Path
import hashlib,json,math,subprocess,sys,tarfile,time
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from paper.scripts.diagnose_titantpp_instacart_recovery import local_verify,sha,canonical,BUNDLE
from paper.scripts.finalize_titantpp_contribution_campaigns import report
from paper.scripts import titantpp_contribution_runtime as op
RECOVERY=PROJECT/'search_artifacts/titantpp_instacart_completion_recovery_20261010_v1'
REPORT=PROJECT/'reports/titantpp_history_recovery_20261010_v1'

def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')

def expected_manifests(c):
 pointer=json.loads((REPORT/'INSTACART_DIAGNOSIS.json').read_text());p=Path(pointer['path']);assert sha(p)==pointer['sha256']
 receipt=json.loads(p.read_text());assert receipt['returncode']==0 and receipt['local_seals_verified']
 diagnosis=json.loads(receipt['stdout'])
 assert not diagnosis['owned_processes'] and diagnosis['files']['supervisor_exit.json']['content']['returncode']==0
 cached=BUNDLE/'hourly_monitor/terminal_cache_5090.json';ptr=json.loads((BUNDLE/'hourly_monitor/terminal_pointer_5090.json').read_text());assert sha(cached)==ptr['sha256']
 cache=json.loads(cached.read_text());assert cache['verified_terminal'] and cache['supervisor_exit']['returncode']==0
 assert cache['contract_sha256']==canonical(c)
 assert not cache['processes'] and not cache['gpu_processes'].strip()
 return {h:{k:v['terminal_manifest_sha256'] for k,v in d['jobs'].items()} for h,d in [('5080',diagnosis),('5090',cache)]}

def code(c,host,expected):
 return 'c='+repr(c)+'\nhost='+repr(host)+'\nexpected='+repr(expected)+'\n'+r'''
from pathlib import Path
import json,hashlib,subprocess,time,tarfile,os
s=c['hosts'][host];b=Path(s['root']);pres=b.parent/('titantpp_instacart_completion_recovery_20261010_v1_'+host)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def command(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=30)
 return dict(argv=args,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
assert read(b/'execution_contract.json')==c
for p,v in {**c['source']['files'],**c['input_files']}.items():assert sha(b/'source'/p)==v,p
for p,v in c['operation']['files'].items():assert sha(b/p)==v,p
assert read(b/'status.json')['status']=='complete' and read(b/'supervisor_exit.json')['returncode']==0
assert not (b/'failure.json').exists()
owner_pids=set()
for p in b.joinpath('claims').glob('*.json'):
 claim=read(p)
 for key in ('supervisor_pid','worker_group_pid','worker_pid'):
  if claim.get(key):owner_pids.add(claim[key])
lease=read(b/'server_lease.json')
owner_pids.update(lease[k] for k in ('supervisor_pid','worker_group_pid'))
# Match argv, not PID alone: historical IDs may be reused by unrelated processes.
owned=[]
for p in Path('/proc').glob('[0-9]*'):
 try:
  args=p.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
  if any(str(b) in x for x in args):owned.append(dict(pid=int(p.name),argv=args))
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert not owned,'Owner process still present'
gpu_info={str(p):p.read_text() for p in Path('/proc/driver/nvidia/gpus').glob('*/information')}
assert any(s['gpu_uuid'] in x for x in gpu_info.values()),'GPU UUID missing from kernel procfs'
fuser=command(['fuser','-v','/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm'])
assert fuser['returncode'] in (0,1),'Device client query failed'
clients={int(x) for x in fuser['stdout'].split() if x.isdigit()}
# The owner process is absent from procfs, and its last known timeout/supervisor
# IDs are absent from the same-user device client listing.
assert not (clients & owner_pids),'Prior owner remains a GPU device client'
terminal={}
for job in c['jobs']:
 if job['host']!=host:continue
 folder=b/'run'/job['id'];m=read(folder/'terminal_manifest.json')
 assert sha(folder/'terminal_manifest.json')==expected[job['id']]
 assert m['job']==job and m['scientific_success'] and not m['held_out_test_evaluated']
 for p,v in m['files'].items():assert sha(folder/p)==v,p
 run=folder/'runs'/job['arm']/'count_only_log_regression'/('seed_'+str(job['seed']))
 e=read(run/'endpoint_replays.json')
 for label in ('selected','last'):
  path=Path(e[label]['checkpoint_path']);assert path.is_relative_to(run)
  rel=str(path.relative_to(folder));assert sha(path)==m['files'][rel]
 terminal[job['id']]=sha(folder/'terminal_manifest.json')
assert len(terminal)==len(expected)
assert not pres.exists(),'Preservation already attempted; no automatic retry'
pres.mkdir()
claim=dict(host=host,created_unix=time.time(),automatic_retry=False)
(pres/'claim.json').write_text(json.dumps(claim))
files={str(p.relative_to(b)):sha(p) for p in sorted(b.rglob('*')) if p.is_file() and not p.is_relative_to(b/'cache')}
manifest=dict(host=host,root=str(b),files=files,terminal_manifests=terminal,
 owner_processes_absent=True,owner_device_clients_absent=True,global_NVML_idle_verified=False,
 owner_ids=sorted(owner_pids),device_clients=fuser,kernel_GPU_information=gpu_info,
 supervisor_exit=read(b/'supervisor_exit.json'),observed_unix=time.time(),held_out_test_evaluated=False)
(pres/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
archive=pres/'originals.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(b/p,arcname=p,recursive=False)
 t.add(pres/'manifest.json',arcname='preservation/manifest.json',recursive=False)
# Verify source stability across the full archive operation.
for p,v in files.items():assert sha(b/p)==v,'Original changed while archiving '+p
receipt=dict(host=host,archive=str(archive),archive_sha256=sha(archive),manifest_sha256=sha(pres/'manifest.json'),files=len(files),completed_unix=time.time())
(pres/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''

def collect(c,h,expected):
 write(RECOVERY/'retrieval'/('claim_'+h+'.json'),dict(host=h,created_unix=time.time(),automatic_retry=False))
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',h,'python3 -'],input=code(c,h,expected),text=True,capture_output=True,timeout=600)
 write(RECOVERY/'retrieval'/('readback_'+h+'.json'),dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,received_unix=time.time()))
 assert r.returncode==0,r.stderr
 receipt=json.loads(r.stdout);archive=RECOVERY/'retrieval'/('archive_'+h+'.tar.gz')
 r=subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15',h+':'+receipt['archive'],str(archive)],capture_output=True,text=True,timeout=1200)
 assert r.returncode==0,r.stderr
 assert sha(archive)==receipt['archive_sha256']
 dest=RECOVERY/'originals'/h;dest.mkdir(parents=True,exist_ok=False)
 with tarfile.open(archive) as t:
  for m in t.getmembers():
   p=Path(m.name);assert m.isfile() and not p.is_absolute() and '..' not in p.parts
  t.extractall(dest,filter='data')
 mp=dest/'preservation/manifest.json';assert sha(mp)==receipt['manifest_sha256']
 for p,v in json.loads(mp.read_text())['files'].items():assert sha(dest/p)==v,p
 write(RECOVERY/'retrieval'/('receipt_'+h+'.json'),receipt)
 return receipt

def selected_records(c,expected):
 records=[]
 for j in c['jobs']:
  folder=RECOVERY/'originals'/j['host']/'run'/j['id'];m=op.read(folder/'terminal_manifest.json')
  assert sha(folder/'terminal_manifest.json')==expected[j['host']][j['id']]
  assert m['job']==j and m['contract_sha256']==canonical(c) and m['scientific_success'] is True
  for name,digest in m['files'].items():assert sha(folder/name)==digest
  run=folder/'runs'/j['arm']/op.VARIANT/('seed_'+str(j['seed']))
  e=op.read(run/'endpoint_replays.json');history=op.read(run/'history.json')['history'];binary={}
  remote_folder=Path(c['hosts'][j['host']]['root'])/'run'/j['id'];remote_run=remote_folder/'runs'/j['arm']/op.VARIANT/('seed_'+str(j['seed']))
  for label in ('selected','last'):
   checkpoint=Path(e[label]['checkpoint_path']);assert checkpoint.is_relative_to(remote_run)
   name=str(checkpoint.relative_to(remote_folder));assert sha(folder/name)==m['files'][name]
   binary[label]={'relative_path':name,'raw_sha256':m['files'][name]}
  finite=[x for x in history if math.isfinite(x['val_qty_rmse'])];best=min(finite,key=lambda x:x['val_qty_rmse'])
  assert best['epoch']==e['best_epoch'] and math.isclose(best['val_qty_rmse'],e['selected']['qty_rmse'],rel_tol=1e-10,abs_tol=1e-8)
  data=next(d for d in c['datasets'] if d['dataset_id']==j['dataset'])
  op.verify_full_validation(e['selected'],data);op.verify_full_validation(e['last'],data)
  initial=op.read(folder/'initialization.json')[j['arm']];assert initial==e['initial_state_sha256']
  records.append({**j,'best_epoch':e['best_epoch'],'last_epoch':e['completed_epochs'],'initial_state_sha256':initial,
   'selected':e['selected'],'last':e['last'],'terminal_manifest_sha256':sha(folder/'terminal_manifest.json'),
   'endpoint_raw_sha256':sha(run/'endpoint_replays.json'),'train_exposure':op.read(run/'exposure.json'),'binary_originals':binary})
 return records

def main():
 c=local_verify();expected=expected_manifests(c)
 write(RECOVERY/'recovery_approval.json',dict(authorization='User 2026-10-10: 문제 확인해서 수정하고 남은 학습 진행하자',
  scope='Completed Instacart originals preservation and Validation aggregation; no training restart, runtime changes or evaluation',parent_contract_sha256=canonical(c),created_unix=time.time()))
 try:
  receipts={h:collect(c,h,expected[h]) for h in ('5080','5090')}
  records=selected_records(c,expected);assert len(records)==9
  (RECOVERY/'analysis').mkdir(exist_ok=False)
  decision=report(RECOVERY,c,records)
  hashes={str(p.relative_to(RECOVERY)):sha(p) for p in RECOVERY.joinpath('analysis').iterdir() if p.is_file()}
  write(RECOVERY/'analysis/completion_receipt.json',dict(status='complete',completed_conditions=9,contract_sha256=canonical(c),
   actual_owner_shutdown_verified=True,originals_rawSHA_verified=True,final_Validation_report_verified=True,
   global_NVML_idle_verified=False,owner_exit_evidence='procfs argv absence, prior owner IDs absent from device clients, supervisor exit0, terminal manifests',
   archive_receipts=receipts,analysis_files=hashes,decision=decision,CPU_binary_reinference_audit=False,Test_evaluation=False,new_independent_evaluation=False,completed_unix=time.time()))
  print(json.dumps(dict(status='complete',completed_conditions=9,bundle=str(RECOVERY))))
 except BaseException as exc:
  write(RECOVERY/'retrieval/failure.json',dict(type=type(exc).__name__,message=str(exc),time=time.time(),automatic_retry=False));raise
if __name__=='__main__':main()
