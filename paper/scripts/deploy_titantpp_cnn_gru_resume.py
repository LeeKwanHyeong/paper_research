"""Explicit deployment and qualification of the two approved continuations."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

PROJECT=Path(__file__).resolve().parents[2]
B=PROJECT/'search_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def contract(host):
    current=read(B/'current.json') if (B/'current.json').exists() else {}
    return read(B/current.get('host_contracts',{}).get(host,{}).get('path','execution_contract.json'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def ssh(host,code,timeout=120):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,'python3 -'],
        input=code,text=True,capture_output=True,timeout=timeout)
    if p.returncode:raise RuntimeError('Owned remote operation failed: '+p.stderr[-3000:])
    return json.loads(p.stdout)

def deploy(host):
    c=contract(host);s=c['hosts'][host];root=s['root'];archive=B/'deployment'/('package_'+host+'.tar.gz')
    remote_archive=root+'.tar.gz'
    pre=ssh(host,"import json,subprocess;from pathlib import Path\nassert not Path(%r).exists() and not Path(%r).exists()\np=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True);assert p.returncode==0 and not p.stdout.strip();print(json.dumps({'gpu_idle':True,'root_absent':True}))"%(root,remote_archive))
    write(B/'deployment'/('transfer_intent_'+host+'.json'),{'host':host,'root':root,'archive_sha256':sha(archive),'time':time.time(),'preflight':pre})
    p=subprocess.run(['scp','-o','BatchMode=yes',str(archive),host+':'+remote_archive],capture_output=True,timeout=240)
    assert p.returncode==0,'Owned package transfer failed; preserve staged evidence'
    code='''import json,hashlib,tarfile
from pathlib import Path
b=Path(%r);a=Path(%r);assert hashlib.sha256(a.read_bytes()).hexdigest()==%r
assert not b.exists();b.mkdir()
with tarfile.open(a) as t:
 ms=t.getmembers();assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in ms)
 assert len(set(m.name for m in ms))==len(ms);t.extractall(b,filter='data')
manifest=json.loads((b/'package_manifest.json').read_text());assert all(hashlib.sha256((b/n).read_bytes()).hexdigest()==v for n,v in manifest['files'].items())
receipt={'status':'SHA_verified','files':len(manifest['files']),'archive_sha256':%r,'contract_sha256':%r}
(b/'deployment_receipt.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
'''%(root,remote_archive,sha(archive),sha(archive),canonical(c))
    result=ssh(host,code)
    write(B/'deployment'/('deployment_receipt_'+host+'.json'),{**result,'host':host,'time':time.time()})
    print(json.dumps(result))

def qualify(host):
    c=contract(host);s=c['hosts'][host];job=next(j for j in c['jobs'] if j['host']==host)
    root=s['root'];command=[s['python'],root+'/operation/titantpp_cnn_gru_resume_runtime.py','--contract',root+'/execution_contract.json','--host',host,'--job',job['id'],'--mode','qualify']
    write(B/'deployment'/('qualification_intent_'+host+'.json'),{'host':host,'job':job,'command':command,'time':time.time(),'starts_research_training':False})
    code='''import json,subprocess,time
from pathlib import Path
b=Path(%r);c=json.loads((b/'execution_contract.json').read_text());s=c['hosts'][%r]
assert (b/'deployment_receipt.json').exists() and not (b/'qualification_start.json').exists()
p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True);assert p.returncode==0 and not p.stdout.strip()
session=%r;command=%r
with (b/'qualification_start.json').open('x') as f:json.dump({'session':session,'command':command,'time':time.time(),'starts_research_training':False},f)
subprocess.run([s['tmux_binary'],'new-session','-d','-s',session,'bash','-lc',%r],check=True)
print(json.dumps({'status':'qualification_started','host':%r,'session':session,'job':%r}))
'''%(root,host,s['tmux']+'_qualification',command,
    'cd '+shlex.quote(s['source_root'])+' && exec env '+ ' '.join(shlex.quote(k+'='+v) for k,v in s['environment'].items())+
    ' timeout --signal=TERM --kill-after=15s 5400 '+shlex.join(command)+' > '+shlex.quote(root+'/qualification.log')+' 2>&1',host,job['id'])
    result=ssh(host,code)
    write(B/'deployment'/('qualification_start_'+host+'.json'),{**result,'time':time.time()});print(json.dumps(result))

def collect(host):
    c=contract(host);s=c['hosts'][host];root=s['root'];job=next(j for j in c['jobs'] if j['host']==host)
    code='''import json,subprocess,time
from pathlib import Path
b=Path(%r);q=b/'qualification'/%r/'receipt.json';log=b/'qualification.log'
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True)
ps=subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True)
out={'host':%r,'observed_unix':time.time(),'receipt':json.loads(q.read_text()) if q.exists() else None,'receipt_sha256':__import__('hashlib').sha256(q.read_bytes()).hexdigest() if q.exists() else None,'gpu_processes':gpu.stdout,'owned_processes':[r for r in ps.stdout.splitlines() if str(b) in r], 'log_tail':log.read_text()[-12000:] if log.exists() else ''}
print(json.dumps(out))
'''%(root,job['id'],host)
    result=ssh(host,code,120)
    dest=B/'qualification'/host;dest.mkdir(parents=True,exist_ok=True)
    stamp=str(time.time_ns());write(dest/(stamp+'.json'),result)
    if result['receipt']:
        q=dest/'receipt.json'
        if not q.exists():write(q,result['receipt'])
    print(json.dumps({k:result[k] for k in ('host','observed_unix','receipt_sha256','gpu_processes')}))
    print(result['log_tail'][-2500:] if not result['receipt'] else json.dumps({'status':result['receipt']['status'],'selected':{k:result['receipt']['selected'][k] for k in ('qty_rmse','qty_mae','time_nll')}}))

def launch(host):
    c=contract(host);s=c['hosts'][host];job=next(j for j in c['jobs'] if j['host']==host);root=s['root']
    snapshots=sorted((B/'qualification'/host).glob('[0-9]*.json'));raw=read(snapshots[-1]);q=raw['receipt'];assert q['status']=='passed' and q['contract_sha256']==canonical(c)
    start=read(B/('start_permit_'+host+'.json')) if (B/('start_permit_'+host+'.json')).exists() else read(B/'start_permit.json')
    permit={'contract_sha256':canonical(c),'host':host,'start_permit_sha256':canonical(start),
        'qualification_files':{job['id']:raw['receipt_sha256']},'job_ids':[job['id']],'manual_resume_once':True,'automatic_retry':False}
    write(B/('training_permit_'+host+'.json'),permit)
    command=[s['python'],root+'/operation/titantpp_cnn_gru_resume_supervisor.py','--contract',root+'/execution_contract.json','--host',host]
    shell='cd '+shlex.quote(s['source_root'])+' && exec env '+ ' '.join(shlex.quote(k+'='+v) for k,v in s['environment'].items())+' '+shlex.join(command)+' > '+shlex.quote(root+'/supervisor.log')+' 2>&1'
    code='''import json,subprocess,hashlib,time
from pathlib import Path
b=Path(%r);q=b/'qualification'/%r/'receipt.json';assert hashlib.sha256(q.read_bytes()).hexdigest()==%r
p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True);assert p.returncode==0 and not p.stdout.strip()
assert not (b/'launch_intent.json').exists() and not (b/'supervisor.claim').exists() and not (b/'claims'/(%r+'.json')).exists()
with (b/%r).open('x') as f:json.dump(%r,f)
with (b/'launch_intent.json').open('x') as f:json.dump({'host':%r,'job':%r,'time':time.time(),'automatic_retry':False},f)
subprocess.run([%r,'new-session','-d','-s',%r,'bash','-lc',%r],check=True)
print(json.dumps({'status':'supervisor_launched','host':%r,'session':%r,'job':%r}))
'''%(root,job['id'],raw['receipt_sha256'],job['id'],'training_permit_'+host+'.json',permit,host,job['id'],s['tmux_binary'],s['tmux'],shell,host,s['tmux'],job['id'])
    result=ssh(host,code);write(B/'deployment'/('launch_receipt_'+host+'.json'),{**result,'time':time.time(),'training_entry_confirmed':False});print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--host',choices=('5080','5090'),required=True);p.add_argument('--mode',choices=('deploy','qualify','collect','launch'),required=True)
    a=p.parse_args();globals()[a.mode](a.host)
