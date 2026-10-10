"""Single read-only SSH diagnostic of the approved Instacart 5080 owner."""
from pathlib import Path
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone, timedelta

PROJECT = Path(__file__).resolve().parents[2]
BUNDLE = PROJECT / 'search_artifacts/titantpp_contribution_instacart_20261008_v1'
REPORT = PROJECT / 'reports/titantpp_history_recovery_20261010_v1'
CONTRACT = '82cd790c57cea68ea5debb18331a7521ae32fa5753374e592f6b46febf418daf'
SEAL = '03e8faf9a6e059a40be752859ccee5c9abd7a65850ba7969a62102d6e4d68438'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda: f.read(1024*1024), b''): h.update(x)
    return h.hexdigest()

def canonical(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def local_verify():
    c = json.loads((BUNDLE/'execution_contract.json').read_text())
    current = json.loads((BUNDLE/'current.json').read_text())
    assert canonical(c) == current['canonical_sha256'] == CONTRACT
    assert sha(BUNDLE/'control/integrity_receipt.json') == current['control_integrity_raw_sha256'] == SEAL
    seal = json.loads((BUNDLE/'control/integrity_receipt.json').read_text())
    assert seal['contract_sha256'] == CONTRACT
    assert current['hosts'] == {h:s['root'] for h,s in c['hosts'].items()}
    assert json.loads((BUNDLE/'approval.json').read_text())['contract_sha256'] == CONTRACT
    assert canonical(c['source']['files']) == current['source_closure_sha256']
    assert canonical(c['operation']['files']) == current['operation_closure_sha256']
    for p,v in {**c['source']['files'], **c['input_files']}.items(): assert sha(BUNDLE/'source'/p) == v, p
    for p,v in c['operation']['files'].items(): assert sha(BUNDLE/p) == v, p
    for p,v in seal['project_files'].items(): assert sha(PROJECT/p) == v, p
    return c

def remote_code(c):
    return 'c = '+repr(c)+'\n'+r'''
from pathlib import Path
import json,hashlib,subprocess,time,os,math
b=Path(c['hosts']['5080']['root'])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(p.read_text()) if p.exists() else None
def command(args):
 try:
  r=subprocess.run(args,capture_output=True,text=True,timeout=30)
  return dict(argv=args,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
 except Exception as e:return dict(argv=args,error=repr(e))
assert read(b/'execution_contract.json')==c,'Remote contract drift'
for p,v in {**c['source']['files'],**c['input_files']}.items():assert sha(b/'source'/p)==v,p
for p,v in c['operation']['files'].items():assert sha(b/p)==v,p
out=dict(observed_unix=time.time(),root=str(b),source_input_operation_SHA_verified=True,
 gpu_query=command(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader,nounits']),
 gpu_compute=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits']),
 driver_gpu_information={},owned_processes=[],previous_pid_state={},files={},jobs={})
for p in Path('/proc/driver/nvidia/gpus').glob('*/information'):
 text=p.read_text(errors='replace')
 if c['hosts']['5080']['gpu_uuid'] in text:out['driver_gpu_information'][str(p)]=text
for p in Path('/proc').glob('[0-9]*'):
 try:
  argv=p.joinpath('cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
  pid=int(p.name)
  if not any(str(b) in arg for arg in argv) and pid not in (52338,52337,41573):continue
  stat=p.joinpath('stat').read_text();fields=stat[stat.rfind(')')+2:].split()
  item=dict(pid=pid,ppid=int(fields[1]),pgrp=int(fields[2]),state=fields[0],argv=argv)
  if pid in (52338,52337,41573):out['previous_pid_state'][str(pid)]=item
  if any(str(b) in arg for arg in argv):out['owned_processes'].append(item)
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
for pid in (52338,52337,41573):out['previous_pid_state'].setdefault(str(pid),None)
for name in ('status.json','failure.json','supervisor_exit.json','server_lease.json','start_permit.json'):
 p=b/name
 out['files'][name]=dict(content=read(p),sha256=sha(p) if p.exists() else None)
for job in c['jobs']:
 if job['host']!='5080':continue
 folder=b/'run'/job['id'];run=folder/'runs'/job['arm']/'count_only_log_regression'/('seed_'+str(job['seed']))
 rec=dict(job=job,status=read(folder/'status.json'),claim=read(b/'claims'/(job['id']+'.json')),terminal=read(folder/'terminal_manifest.json'),checkpoint_receipt=read(run/'server_checkpoint_receipt.json'))
 if rec['terminal']:
  rec['terminal_files_verified']=all(sha(folder/p)==v for p,v in rec['terminal']['files'].items())
  rec['terminal_manifest_sha256']=sha(folder/'terminal_manifest.json')
 receipt=rec['checkpoint_receipt']
 if receipt:
  rec['checkpoint_files_verified']=all((run/p).exists() and sha(run/p)==v for p,v in receipt['files'].items()) and receipt==read(run/'server_checkpoint_receipt.json')
 history=read(run/'history.json')
 if history:
  rows=history.get('history',[]);valid=[r for r in rows if isinstance(r.get('val_qty_rmse'),(int,float)) and math.isfinite(r['val_qty_rmse'])]
  best=min(valid,key=lambda r:r['val_qty_rmse']) if valid else None
  rec['history_summary']=dict(rows=len(rows),last_epoch=rows[-1].get('epoch') if rows else None,best_epoch=best['epoch'] if best else None)
 timing=read(run/'epoch_timing.json')
 if timing:rec['recent_epoch_timing']=timing.get('epochs',[])[-10:]
 rec['binary_files']={p.name:dict(size=p.stat().st_size,sha256=sha(p),modified_unix=p.stat().st_mtime) for p in run.glob('*.pt')}
 out['jobs'][job['id']]=rec
p=b/'dispatch.log'
if p.exists():out['dispatch_log_tail']=p.read_text(errors='replace')[-12000:]
d=command(['dmesg','--color=never'])
d['stdout']='\n'.join(x for x in d.get('stdout','').splitlines() if any(w in x.lower() for w in ('nvrm','nvidia','xid')))[-6000:]
out['kernel_gpu_messages']=d
out['finished_unix']=time.time()
print(json.dumps(out,ensure_ascii=False))
'''

def main():
    c=local_verify();REPORT.mkdir(parents=True,exist_ok=True)
    code=remote_code(c)
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','5080','python3 -'],input=code,capture_output=True,text=True,timeout=300)
    stamp=datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%SKST')
    receipt={'local_received_unix':time.time(),'command':['ssh','5080','python3 -'],'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'contract_sha256':CONTRACT,'local_seals_verified':True,'remote_mutations':0}
    path=REPORT/('INSTACART_DIAGNOSIS_'+stamp+'.json');path.write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    pointer=REPORT/'INSTACART_DIAGNOSIS.json';pointer.write_text(json.dumps({'path':str(path),'sha256':sha(path)},indent=2)+'\n')
    print(json.dumps({'path':str(path),'sha256':sha(path),'returncode':result.returncode,'stderr':result.stderr},ensure_ascii=False))
    if result.returncode==0:
        out=json.loads(result.stdout)
        print(json.dumps({k:v for k,v in out.items() if k not in ('jobs','dispatch_log_tail')},ensure_ascii=False))
        print(json.dumps({k:{x:y for x,y in v.items() if x in ('history_summary','checkpoint_files_verified','terminal_files_verified','job','checkpoint_receipt')} for k,v in out['jobs'].items()},ensure_ascii=False))
    raise SystemExit(result.returncode)
if __name__=='__main__':main()
