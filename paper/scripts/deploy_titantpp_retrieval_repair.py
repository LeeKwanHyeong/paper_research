"""One explicitly approved operational repair; frozen science and old owners remain immutable."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import time

PROJECT = Path(__file__).resolve().parents[2]
PARENT = PROJECT / 'search_artifacts/titantpp_contextual_retrieval_pilot_dual_20261007_v1'
B = PROJECT / 'search_artifacts/titantpp_contextual_retrieval_pilot_repair_20261007_v1'
SCHEMA = B.name
PARENT_SHA = '5372f2ec93c98e6b45ecf37d2b19ae1375a838fdfd9eb8a9c4f3a908456e7953'

def read(p): return json.loads(Path(p).read_text())
def canonical(v): return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''): h.update(chunk)
    return h.hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f: json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def build():
    """Freeze the repair before a network action; existing frozen files are never rewritten."""
    if (B/'execution_contract.json').exists(): return verify()
    p=read(PARENT/'execution_contract.json')
    if canonical(p)!=PARENT_SHA: raise ValueError('Parent contract changed')
    pointer=read(PARENT/'hourly_monitor/latest_5090.json');snapshot=Path(pointer['snapshot']);s=read(snapshot)
    fixed=next(r for r in s['rows'] if r['job']['arm']=='titantpp_retrieval_fixed')
    if fixed['actual_saved_epoch']!=148 or fixed['actual_best_epoch']!=120: raise ValueError('Resume boundary changed')
    if s['processes'] or s['gpu_pids'] or s['supervisor_exit']['returncode']!=1: raise ValueError('Original timeout termination unverified')
    failure=s['failure'];fid=fixed['job']['id']
    if failure['type']!='ValueError' or failure['message']!='Worker exited 124; no automatic retry' or failure['active_job']['id']!=fid:
        raise ValueError('Failure differs from approved timeout repair')
    hosts=deepcopy(p['hosts'])
    for h,spec in hosts.items():
        root='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/'+SCHEMA+'_'+h
        spec.update(root=root,source_root=root+'/source',operation_root=root+'/operation',output_dir=root+'/run',tmux=SCHEMA+'_'+h,old_owner_roots=[p['hosts'][h]['root']])
        spec['environment'].update(PYTHONPATH=root+'/source',MPLCONFIGDIR=root+'/cache/matplotlib',XDG_CACHE_HOME=root+'/cache')
    ops={'operation/repair_runtime.py':'paper/scripts/titantpp_contextual_retrieval_repair_runtime.py',
         'operation/observer.py':'paper/scripts/observe_titantpp_retrieval_repair.py'}
    files={k:sha(PROJECT/v) for k,v in ops.items()}
    c={'schema':SCHEMA,'evaluation_scope':'validation_only','held_out_test_evaluated':False,
       'parent_contract_sha256':PARENT_SHA,'parent_contract':p,'source':p['source'],'parent_operation':p['operation'],
       'hosts':hosts,'limits':{**p['limits'],'per_condition_seconds':21600,'total_wall_seconds':129600,'qualification_seconds':1800},
       'jobs':[j for j in p['jobs'] if j['host']=='5080' or j['arm'] not in ('titantpp_retrieval_mean','titantpp_retrieval_weighted')],
       'reused_jobs':[j for j in p['jobs'] if j['host']=='5090' and j['arm'] in ('titantpp_retrieval_mean','titantpp_retrieval_weighted')],
       'resume':{fid:{'parent_run_dir':p['hosts']['5090']['root']+'/run/'+fid+'/runs/titantpp_retrieval_fixed/count_only_log_regression/seed_42',
                      'last_epoch':148,'best_epoch':120,'files':fixed['checkpoint_receipt']['files']}},
       'qualification_bindings':{h:{'receipt_file':'qualification/receipt.json','receipt_sha256':sha(PARENT/'qualification'/h/'receipt.json')} for h in hosts},
       'parent_terminal_evidence':{'snapshot':'evidence/parent_5090_snapshot.json','snapshot_sha256':sha(snapshot)},
       'operation':{'files':files,'files_sha256':canonical(files)},
       'policy':{'automatic_retry':False,'explicit_manual_repair_once':True,'held_out_test_evaluated':False,
                 'science_changed':False,'shared_Runtime_or_auth_changed':False,'preserve_parent_failure_claim_and_partial_originals':True}}
    B.mkdir(parents=True,exist_ok=True)
    for name,value in {**p['source']['files'],**p['input_files']}.items():
        original=PARENT/'source'/name
        if sha(original)!=value: raise ValueError('Parent scientific source/input changed '+name)
        dest=B/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,dest)
    for name,value in p['operation']['files'].items():
        if sha(PARENT/name)!=value: raise ValueError('Parent operation changed '+name)
        dest=B/'parent_operation'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PARENT/name,dest)
    for dest,original in ops.items():
        target=B/dest;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PROJECT/original,target)
    (B/'evidence').mkdir(exist_ok=True);shutil.copyfile(snapshot,B/'evidence/parent_5090_snapshot.json')
    for h in hosts:
        dest=B/'qualification'/h/'receipt.json';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PARENT/'qualification'/h/'receipt.json',dest)
    write(B/'execution_contract.json',c)
    write(B/'approval.json',{'approved':True,'hosts':['5090','5080'],'contract_sha256':canonical(c),
         'user_instruction':'문제 해결해서 진행하자','authorized_operations':['one manual same-GPU E148 continuation on 5090','unstarted arms fresh fits on both GPUs','clock-skew-safe launch observation','six-hour fit and thirty-six-hour host repair limits'],
         'preserve_originals':True,'automatic_retry':False,'approved_date_KST':'2026-10-07'})
    controlnames={'control/deploy.py':'paper/scripts/deploy_titantpp_retrieval_repair.py',
                  'control/observer.py':'paper/scripts/observe_titantpp_retrieval_repair.py',
                  'control/finalize_once.py':'paper/scripts/finalize_titantpp_retrieval_repair.py',
                  'control/parent_deploy_clockfix.py':'paper/scripts/deploy_titantpp_contextual_retrieval.py'}
    for target,original in controlnames.items():
        dest=B/target;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PROJECT/original,dest)
    integrity={'contract_sha256':canonical(c),'files':{n:sha(B/n) for n in controlnames},
               'project_files':{n:sha(PROJECT/n) for n in controlnames.values()},'automatic_retry':False}
    write(B/'control/integrity_receipt.json',integrity)
    write(B/'current.json',{'schema':SCHEMA,'canonical_sha256':canonical(c),'contract_path':'execution_contract.json',
        'hosts':{h:s['root'] for h,s in hosts.items()},'source_closure_sha256':p['source']['files_sha256'],
        'parent_operation_closure_sha256':p['operation']['files_sha256'],'operation_closure_sha256':canonical(files),
        'control_integrity_receipt_sha256':sha(B/'control/integrity_receipt.json'),'parent_contract_sha256':PARENT_SHA,
        'canonical_configs':5,'execution_conditions':10,'new_fresh_fits':7,'resumed_fits':1,'reused_completed':2,'automatic_retry':False})
    return verify()

def verify():
    c=read(B/'execution_contract.json');cur=read(B/'current.json');a=read(B/'approval.json')
    if cur['canonical_sha256']!=canonical(c) or a['contract_sha256']!=canonical(c) or a['approved'] is not True: raise ValueError('Repair approval/current changed')
    if cur['hosts']!={h:s['root'] for h,s in c['hosts'].items()}: raise ValueError('Repair ownership changed')
    receipt=B/'control/integrity_receipt.json'
    if sha(receipt)!=cur['control_integrity_receipt_sha256']: raise ValueError('Repair control receipt changed')
    for n,d in read(receipt)['files'].items():
        if sha(B/n)!=d: raise ValueError('Repair frozen control changed '+n)
    for n,d in read(receipt)['project_files'].items():
        if sha(PROJECT/n)!=d: raise ValueError('Repair project control changed '+n)
    m=module(B/'operation/repair_runtime.py','repair_preflight');m.validate(c,bundle=B)
    return c

def ssh(host,code,action,timeout=240):
    start=time.time_ns()
    try:
        r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,'python3 -'],input=code,text=True,capture_output=True,timeout=timeout)
    except BaseException as exc:
        write(B/'deployment'/f'{action}_{host}_{start}.json',{'host':host,'type':type(exc).__name__,'message':str(exc),'automatic_retry':False});raise
    received=time.time_ns()
    write(B/'deployment'/f'{action}_{host}_{start}.json',{'host':host,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'local_received_unix_ns':received,'automatic_retry':False})
    if r.returncode: raise RuntimeError('Repair action failed; evidence preserved: '+r.stderr[-2500:])
    out=json.loads(r.stdout.strip().splitlines()[-1]);out['local_received_unix_ns']=received
    if abs(out['observed_unix']-received/1e9)>10: raise ValueError('Server/local clock skew exceeds ten seconds')
    return out

def remote_prefix(c,host):
    return '''import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
c=%r;host=%r;s=c['hosts'][host];b=Path(s['root']);old=Path(c['parent_contract']['hosts'][host]['root'])
def read(p):return json.loads(Path(p).read_text())
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(1024**2),b''):h.update(v)
 return h.hexdigest()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True).strip().splitlines()
assert len(gpu)==1 and gpu[0].split(',')[0].strip()==s['gpu_uuid'] and host in gpu[0],'GPU UUID/name mismatch'
compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True).strip();assert not compute,'GPU occupied; no process will be stopped'
ps=subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True);assert not any(str(p) in line for p in (old,b) for line in ps.splitlines()),'Owned process still present'
assert read(old/'execution_contract.json')==c['parent_contract'],'Parent deployed contract changed'
for n,d in c['parent_contract']['source']['files'].items():assert sha(old/'source'/n)==d,'Parent source changed '+n
for n,d in c['parent_contract']['input_files'].items():assert sha(old/'source'/n)==d,'Parent input changed '+n
for n,d in c['parent_contract']['operation']['files'].items():assert sha(old/n)==d,'Parent operation changed '+n
assert sha(old/'qualification/receipt.json')==c['qualification_bindings'][host]['receipt_sha256'],'Native qualification changed'
if host=='5090':
 failure=read(old/'failure.json');assert failure['type']=='ValueError' and failure['message']=='Worker exited 124; no automatic retry' and failure['active_job']['id']==%r,'Foreign parent failure'
 assert read(old/'supervisor_process_exit.json')['returncode']==1,'Original supervisor timeout evidence changed'
 for j,binding in c['resume'].items():
  for n,d in binding['files'].items():assert sha(Path(binding['parent_run_dir'])/n)==d,'Resume original changed '+n
else:
 assert not (old/'failure.json').exists() and read(old/'qualification_shell_exit.json')['returncode']==0,'Original 5080 qualification termination changed'
assert Path(s['python']).is_file() and Path(s['tmux_binary']).is_file() and shutil.which('timeout'),'Native runtime tools missing'
env={**os.environ,**s['environment']}
probe="import sys,torch,numpy,polars,json;print(json.dumps(dict(python=sys.executable,torch=str(torch.__version__),numpy=numpy.__version__,polars=polars.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())))"
runtime=json.loads(subprocess.check_output([s['python'],'-c',probe],env=env,text=True));assert Path(runtime['python']).resolve()==Path(s['python']).resolve(),'Foreign native Python'
assert all(runtime.get(k)==v for k,v in s['runtime_expected'].items()),'Native runtime changed'
''' % (c,host,'5090__yellow_trip_hourly__42__titantpp_retrieval_fixed')

def package(c,host):
    now=time.time();start={'host':host,'contract_sha256':canonical(c),'approval_sha256':canonical(read(B/'approval.json')),'started_at_unix':now,'deadline_unix':now+c['limits']['total_wall_seconds'],'automatic_retry':False}
    write(B/f'start_permit_{host}.json',start)
    permit={'host':host,'contract_sha256':canonical(c),'start_permit_sha256':canonical(start),
            'qualification_sha256':c['qualification_bindings'][host]['receipt_sha256'],'qualification_parent_contract_sha256':PARENT_SHA,'automatic_retry':False}
    write(B/f'training_permit_{host}.json',permit)
    folder=B/'deployment'/f'package_{host}';folder.mkdir(parents=True,exist_ok=False)
    for name in ['execution_contract.json','approval.json','current.json']:
        shutil.copyfile(B/name,folder/name)
    shutil.copyfile(B/f'start_permit_{host}.json',folder/'start_permit.json');shutil.copyfile(B/f'training_permit_{host}.json',folder/'training_permit.json')
    for name in ('source','parent_operation','operation'):
        shutil.copytree(B/name,folder/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (folder/'qualification').mkdir();shutil.copyfile(B/'qualification'/host/'receipt.json',folder/'qualification/receipt.json')
    files={str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}
    write(folder/'package_manifest.json',{'host':host,'contract_sha256':canonical(c),'files':files})
    archive=B/'deployment'/f'package_{host}.tar.gz'
    with tarfile.open(archive,'x:gz') as t:
        for p in sorted(folder.rglob('*')):
            if p.is_file():t.add(p,arcname=str(p.relative_to(folder)),recursive=False)
    return archive,sha(folder/'package_manifest.json')

def activate(host):
    c=verify();launch=B/'deployment'/f'launch_receipt_{host}.json'
    if launch.exists(): print(json.dumps({'status':'already_launched','host':host,'remote_calls':0}));return
    if (B/'deployment'/f'transfer_intent_{host}.json').exists(): raise ValueError('Preserved repair intent; no automatic retry')
    pre=ssh(host,remote_prefix(c,host)+"assert not b.exists() and not Path(str(b)+'.tar.gz').exists(),'Repair root already exists'\nprint(json.dumps({'host':host,'status':'parent_idle_source_originals_verified','runtime':runtime,'observed_unix':time.time(),'gpu_uuid':s['gpu_uuid']}))\n",'preflight')
    write(B/'deployment'/f'preflight_{host}.json',pre)
    archive,manifestsha=package(c,host);write(B/'deployment'/f'transfer_intent_{host}.json',{'host':host,'archive_sha256':sha(archive),'contract_sha256':canonical(c),'automatic_retry':False})
    r=subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15',str(archive),host+':'+c['hosts'][host]['root']+'.tar.gz'],text=True,capture_output=True,timeout=300)
    write(B/'deployment'/f'transfer_result_{host}.json',{'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'automatic_retry':False})
    if r.returncode: raise RuntimeError('Repair package transfer failed; no retry')
    s=c['hosts'][host];cmd=[s['python'],s['root']+'/operation/repair_runtime.py','--bundle',s['root'],'--host',host,'--mode','dispatch']
    exit_code="import json,sys,time;from pathlib import Path;f=Path(sys.argv[1]).open('x');json.dump({'returncode':int(sys.argv[2]),'finished_unix':time.time(),'automatic_retry':False},f);f.close()"
    trap='rc=$?; trap - EXIT; '+shlex.join([s['python'],'-c',exit_code,s['root']+'/supervisor_process_exit.json'])+' "$rc"; exit "$rc"'
    shell='trap '+shlex.quote(trap)+' EXIT; cd '+shlex.quote(s['source_root'])+' || exit $?; env '+' '.join(shlex.quote(k+'='+v) for k,v in sorted(s['environment'].items()))+' '+shlex.join(cmd)+' > '+shlex.quote(s['root']+'/supervisor.log')+' 2>&1'
    code=remote_prefix(c,host)+'''import tarfile
a=Path(str(b)+'.tar.gz');assert sha(a)==%r,'Transferred package SHA changed';assert not b.exists(),'Fresh repair root required'
with tarfile.open(a,'r:gz') as t:
 members=t.getmembers();names=[m.name for m in members]
 assert len(names)==len(set(names)) and all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members),'Unsafe archive'
 raw=t.extractfile('package_manifest.json').read();assert hashlib.sha256(raw).hexdigest()==%r,'Package manifest changed'
 manifest=json.loads(raw);assert set(names)==set(manifest['files'])|{'package_manifest.json'},'Package coverage changed'
 b.mkdir(mode=0o700)
 for m in members:
  target=b/m.name;target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as out:shutil.copyfileobj(t.extractfile(m),out)
for n,d in manifest['files'].items():assert sha(b/n)==d,'Extracted package changed '+n
assert read(b/'execution_contract.json')==c,'Deployed repair contract changed'
session=s['tmux'];assert subprocess.run([s['tmux_binary'],'has-session','-t',session],capture_output=True).returncode!=0,'Repair tmux exists'
with (b/'deployment_receipt.json').open('x') as f:json.dump({'contract_sha256':canonical(c),'host':host,'archive_sha256':%r,'package_manifest_sha256':%r,'runtime':runtime,'parent_idle':True,'observed_unix':time.time()},f)
with (b/'launch_intent.json').open('x') as f:json.dump({'contract_sha256':canonical(c),'host':host,'command':%r,'automatic_retry':False,'observed_unix':time.time()},f)
subprocess.run([s['tmux_binary'],'new-session','-d','-s',session,'bash','-lc',%r],check=True)
print(json.dumps({'host':host,'root':str(b),'status':'repair_dispatcher_launched','contract_sha256':canonical(c),'command':%r,'session':session,'gpu_uuid':s['gpu_uuid'],'observed_unix':time.time(),'training_entry_confirmed':False,'automatic_retry':False}))
'''%(sha(archive),manifestsha,sha(archive),manifestsha,cmd,shell,cmd)
    out=ssh(host,code,'extract_launch',300);write(launch,out);print(json.dumps(out))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--mode',choices=['build','verify','activate'],required=True);ap.add_argument('--host',choices=['5080','5090']);args=ap.parse_args()
    if args.mode=='activate':
        if not args.host:ap.error('--host required')
        activate(args.host)
    else:print(json.dumps({'status':'verified','contract_sha256':canonical(build() if args.mode=='build' else verify())}))
