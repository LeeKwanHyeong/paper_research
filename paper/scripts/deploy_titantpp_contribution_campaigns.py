"""Approved staged activation; independent hosts; preserved intents/no retries."""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import time
import sys
_PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(_PROJECT_ROOT))
from paper.scripts import titantpp_contribution_runtime as op
from paper.scripts.prepare_titantpp_contribution_campaigns import PROJECT,REPORT,write

def bundle(stage):return PROJECT/'search_artifacts'/('titantpp_contribution_'+stage+'_20261008_v1')

def local_contract(stage):
    b=bundle(stage);c=op.read(b/'execution_contract.json');cur=op.read(b/'current.json')
    op.validate(c,source_root=b/'source',operation_root=b/'operation',verify_files=False)
    assert cur['canonical_sha256']==op.sha_json(c) and cur['hosts']=={h:s['root'] for h,s in c['hosts'].items()}
    for p,v in {**c['source']['files'],**c['input_files']}.items():assert op.sha_file(b/'source'/p)==v,'Local source/input drift '+p
    for p,v in c['operation']['files'].items():assert op.sha_file(b/p)==v,'Local operation drift '+p
    assert op.read(b/'approval.json')['contract_sha256']==op.sha_json(c)
    seal=b/'control/integrity_receipt.json'
    assert op.sha_file(seal)==cur['control_integrity_raw_sha256'],'Local control receipt drift'
    integrity=op.read(seal);assert integrity['contract_sha256']==op.sha_json(c)
    for p,v in integrity['project_files'].items():assert op.sha_file(PROJECT/p)==v,'Sealed project control drift '+p
    return b,c

def dependency_ready(c):
    previous=c['prior_stage']
    if previous is None:return True
    p=PROJECT/previous['bundle']/'analysis/completion_receipt.json'
    if not p.exists():return False
    r=op.read(p)
    assert r['contract_sha256']==previous['contract_sha256'] and r['completed_conditions']==previous['jobs']
    assert r['actual_owner_shutdown_verified'] is True and r['originals_rawSHA_verified'] is True and r['final_Validation_report_verified'] is True
    for name,digest in r['local_files'].items():assert op.sha_file(PROJECT/previous['bundle']/name)==digest,'Prior originals/report drift'
    return True

def remote(c,host,code,action,timeout=120):
    b=bundle(c['stage']);result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,'python3 -'],
        input=code,text=True,capture_output=True,timeout=timeout)
    receipt={'host':host,'action':action,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
        'local_received_unix':time.time(),'contract_sha256':op.sha_json(c)}
    write(b/'deployment/readbacks'/f'{action}_{host}_{time.time_ns()}.json',receipt)
    if result.returncode:raise RuntimeError(f'{host} {action} failed; preserved readback: '+result.stderr[-2000:])
    payload=json.loads(result.stdout)
    payload['local_received_unix']=receipt['local_received_unix']
    return payload

def prefix(c,host,*,idle=False,deployed=True):
    return '''import json,hashlib,subprocess,time,os,shutil
from pathlib import Path
c=%r
s=c['hosts'][%r];b=Path(s['root'])
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
 return h.hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
g=subprocess.run(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=20).stdout
assert s['gpu_uuid'] in g and s['gpu_name_contains'] in g,'GPU identity drift'
compute=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=20).stdout
if %r:assert not any(s['gpu_uuid'] in line for line in compute.splitlines()),'GPU is occupied; do not interfere'
assert Path(s['python']).is_file() and Path(s['tmux_binary']).is_file(),'Existing Runtime/tmux unavailable'
assert shutil.disk_usage(b.parent).free>c['limits']['min_free_bytes'],'Insufficient disk'
if %r:
 assert json.loads((b/'execution_contract.json').read_text())==c,'Remote contract drift'
 assert not (b/'failure.json').exists() and not (b/'qualification/failure.json').exists(),'Preserved failure; no retry'
 for p,v in {**c['source']['files'],**c['input_files']}.items():assert sha(b/'source'/p)==v,'Source/input drift '+p
 for p,v in c['operation']['files'].items():assert sha(b/p)==v,'Operation drift '+p
''' % (c,host,idle,deployed)

def package(b,c,host):
    s=c['hosts'][host];start=b/f'start_permit_{host}.json'
    if not start.exists():
        t=time.time();write(start,{'host':host,'contract_sha256':op.sha_json(c),'approval_sha256':op.sha_json(op.read(b/'approval.json')),
            'started_at_unix':t,'deadline_unix':t+c['limits']['total_wall_seconds']})
    p=b/'deployment'/('package_'+host);archive=p.with_suffix('.tar.gz')
    if archive.exists():
        m=op.read(p/'package_manifest.json');assert m['contract_sha256']==op.sha_json(c)
        for name,v in m['files'].items():assert op.sha_file(p/name)==v
        return archive,m
    p.mkdir(parents=True,exist_ok=False);shutil.copytree(b/'source',p/'source');shutil.copytree(b/'operation',p/'operation')
    for name in ('execution_contract.json','approval.json'):shutil.copyfile(b/name,p/name)
    shutil.copyfile(start,p/'start_permit.json')
    m={'host':host,'contract_sha256':op.sha_json(c),'files':{str(x.relative_to(p)):op.sha_file(x) for x in sorted(p.rglob('*')) if x.is_file()}}
    write(p/'package_manifest.json',m)
    with tarfile.open(archive,'w:gz') as tf:
        for x in sorted(p.rglob('*')):
            if x.is_file():tf.add(x,arcname=str(x.relative_to(p)),recursive=False)
    write(b/'deployment'/f'package_receipt_{host}.json',{'archive_sha256':op.sha_file(archive),'manifest_sha256':op.sha_file(p/'package_manifest.json')})
    return archive,m

def command(c,host,mode):
    s=c['hosts'][host];args=[s['python'],s['root']+'/operation/campaign.py','--contract',s['root']+'/execution_contract.json','--host',host,'--mode',mode]
    limit=c['limits']['qualification_seconds'] if mode=='qualify' else c['limits']['total_wall_seconds']
    exit_path=s['root']+'/'+('qualification_exit.json' if mode=='qualify' else 'supervisor_exit.json')
    log=s['root']+'/'+mode+'.log'
    exit_code="import json,time; from pathlib import Path; p=Path(%r); p.write_text(json.dumps({'returncode':int(__import__('sys').argv[1]),'completed_unix':time.time()}))" % exit_path
    shell='cd '+shlex.quote(s['source_root'])+' || exit $?; env '+shlex.join([k+'='+v for k,v in s['environment'].items()])
    shell+=' timeout --signal=TERM --kill-after=15s '+str(limit)+' '+shlex.join(args)+' > '+shlex.quote(log)+' 2>&1; rc=$?; '
    shell+='python3 -c '+shlex.quote(exit_code)+' "$rc"; exit "$rc"'
    return [s['tmux_binary'],'new-session','-d','-s',s['tmux']+'_'+mode,'bash -c '+shlex.quote(shell)]

def deploy(c,host):
    b=bundle(c['stage']);archive,m=package(b,c,host)
    pre=remote(c,host,prefix(c,host,idle=True,deployed=False)+"assert not b.exists(),'Owned root already exists'\nprint(json.dumps({'observed_unix':time.time(),'GPU':g,'gpu_processes':compute,'root_absent':True}))",'preflight')
    write(b/'deployment'/f'transfer_intent_{host}.json',{'archive_sha256':op.sha_file(archive),'root':c['hosts'][host]['root'],'contract_sha256':op.sha_json(c)})
    root=c['hosts'][host]['root'];r=subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15',str(archive),host+':'+root+'.tar.gz'],capture_output=True,text=True,timeout=600)
    write(b/'deployment'/f'transfer_result_{host}.json',{'returncode':r.returncode,'stderr':r.stderr,'observed_unix':time.time()})
    assert r.returncode==0,'Transfer failed; no retry'
    code=prefix(c,host,idle=True,deployed=False)+'''import tarfile
assert not b.exists(),'Root exists; no overwrite'
archive=Path(str(b)+'.tar.gz');assert sha(archive)==%r,'Archive drift'
b.mkdir()
with tarfile.open(archive,'r:gz') as t:
 for member in t.getmembers():
  p=Path(member.name);assert member.isfile() and not p.is_absolute() and '..' not in p.parts,'Unsafe package'
 t.extractall(b,filter='data')
manifest=json.loads((b/'package_manifest.json').read_text());assert manifest==%r
for p,v in manifest['files'].items():assert sha(b/p)==v,'Deployed file drift '+p
(b/'cache').mkdir();(b/'logs').mkdir()
out={'status':'deployed_SHA_verified','contract_sha256':canonical(c),'observed_unix':time.time(),'GPU':g,'gpu_processes':compute,'preflight':%r}
with (b/'deployment_receipt.json').open('x') as f:json.dump(out,f)
print(json.dumps(out))
''' % (op.sha_file(archive),m,pre)
    out=remote(c,host,code,'deploy',timeout=240);write(b/'deployment'/f'deployment_receipt_{host}.json',out)

def start_qualification(c,host):
    b=bundle(c['stage']);cmd=command(c,host,'qualify')
    write(b/'deployment'/f'qualification_intent_{host}.json',{'command':cmd,'contract_sha256':op.sha_json(c)})
    code=prefix(c,host,idle=True)+'''assert not (b/'qualification.claim').exists() and not (b/'qualification_start.json').exists(),'Qualification already attempted'
cmd=%r
with (b/'qualification_start.json').open('x') as f:json.dump({'command':cmd,'contract_sha256':canonical(c),'started_unix':time.time()},f)
subprocess.run(cmd,check=True)
print(json.dumps({'status':'qualification_started','host':%r,'command':cmd,'observed_unix':time.time()}))
'''%(cmd,host)
    out=remote(c,host,code,'qualification_start');write(b/'deployment'/f'qualification_start_{host}.json',out)
    return out

def collect_qualification(c,host):
    code=prefix(c,host)+'''def readif(p):return json.loads((b/p).read_text()) if (b/p).exists() else None
q=b/'qualification/receipt.json'
ps=subprocess.run(['ps','-eo','pid=,ppid=,args='],capture_output=True,text=True,check=True).stdout
owned=[x for x in ps.splitlines() if str(b) in x]
print(json.dumps({'host':%r,'observed_unix':time.time(),'receipt_text':q.read_text() if q.exists() else None,'receipt_sha256':sha(q) if q.exists() else None,'exit':readif('qualification_exit.json'),'failure':readif('qualification/failure.json'),'owned_processes':owned,'gpu_processes':compute,'log_tail':(b/'qualify.log').read_text()[-6000:] if (b/'qualify.log').exists() else ''}))
'''%host
    out=remote(c,host,code,'collect_qualification');write(bundle(c['stage'])/'qualification'/host/f'{time.time_ns()}.json',out)
    if out['failure'] or (out['exit'] and out['exit']['returncode']!=0):raise ValueError('Native qualification failed; evidence preserved')
    if not out['receipt_text'] or not out['exit'] or out['owned_processes']:return None
    assert not any(c['hosts'][host]['gpu_uuid'] in x for x in out['gpu_processes'].splitlines()),'GPU owner remained'
    q=json.loads(out['receipt_text']);assert q['status']=='passed' and q['contract_sha256']==op.sha_json(c)
    assert __import__('hashlib').sha256(out['receipt_text'].encode()).hexdigest()==out['receipt_sha256']
    assert abs(out['local_received_unix']-out['observed_unix'])<=10,'Server/local clock skew >10s'
    dest=bundle(c['stage'])/'qualification'/host/'receipt.json'
    if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(out['receipt_text'])
    assert op.sha_file(dest)==out['receipt_sha256'],'Preserved receipt drift'
    return out

def launch(c,host,observed):
    b=bundle(c['stage']);q=json.loads(observed['receipt_text']);start=op.read(b/f'start_permit_{host}.json')
    assert 0<=time.time()-observed['local_received_unix']<600 and observed['exit']['returncode']==0
    assert abs(observed['observed_unix']-observed['local_received_unix'])<=10,'Clock skew exceeds10s'
    permit={'host':host,'contract_sha256':op.sha_json(c),'start_permit_sha256':op.sha_json(start),'qualification_sha256':observed['receipt_sha256'],'qualification':q,
        'dependency_gate_verified':True,'prior_stage':c['prior_stage']}
    write(b/f'training_permit_{host}.json',permit);cmd=command(c,host,'dispatch')
    write(b/'deployment'/f'launch_intent_{host}.json',{'command':cmd,'contract_sha256':op.sha_json(c)})
    code=prefix(c,host,idle=True)+'''assert not (b/'supervisor.claim').exists() and not (b/'training_permit.json').exists(),'Training already attempted'
assert sha(b/'qualification/receipt.json')==%r and json.loads((b/'qualification_exit.json').read_text())['returncode']==0,'Qualification binding/actual exit drift'
permit=%r
assert permit['start_permit_sha256']==canonical(json.loads((b/'start_permit.json').read_text()))
assert time.time()<json.loads((b/'start_permit.json').read_text())['deadline_unix'],'Expired lease'
with (b/'training_permit.json').open('x') as f:json.dump(permit,f)
cmd=%r
with (b/'launch_intent.json').open('x') as f:json.dump({'command':cmd,'contract_sha256':canonical(c)},f)
subprocess.run(cmd,check=True)
print(json.dumps({'status':'training_launched','host':%r,'command':cmd,'observed_unix':time.time()}))
'''%(observed['receipt_sha256'],permit,cmd,host)
    out=remote(c,host,code,'launch');write(b/'deployment'/f'launch_receipt_{host}.json',out);return out

def activate_host(stage,host):
    b,c=local_contract(stage)
    if not dependency_ready(c):return {'host':host,'stage':stage,'status':'dependency_pending','remote_calls':0}
    d=b/'deployment'
    if (d/f'launch_receipt_{host}.json').exists():return {'host':host,'stage':stage,'status':'already_launched','remote_calls':0}
    if not (d/f'deployment_receipt_{host}.json').exists():
        assert not (d/f'transfer_intent_{host}.json').exists(),'Preserved incomplete transfer intent'
        deploy(c,host)
    if not (d/f'qualification_start_{host}.json').exists():
        assert not (d/f'qualification_intent_{host}.json').exists(),'Preserved incomplete qualification intent'
        return start_qualification(c,host)
    assert not (d/f'launch_intent_{host}.json').exists(),'Preserved incomplete launch intent'
    observed=collect_qualification(c,host)
    if observed is None:return {'host':host,'stage':stage,'status':'qualification_pending_no_retry'}
    return launch(c,host,observed)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=tuple(op.STAGES),required=True);p.add_argument('--host',choices=('5080','5090','both'),default='both');a=p.parse_args()
    hosts=('5080','5090') if a.host=='both' else (a.host,)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures={h:pool.submit(activate_host,a.stage,h) for h in hosts}
        result={}
        for h,f in futures.items():
            try:result[h]=f.result()
            except Exception as exc:result[h]={'host':h,'status':'failed_preserved','type':type(exc).__name__,'message':str(exc)}
    print(json.dumps(result,ensure_ascii=False))
    if any(x.get('status')=='failed_preserved' for x in result.values()):raise SystemExit(1)

if __name__=='__main__':main()
