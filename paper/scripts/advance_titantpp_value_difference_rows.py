"""One isolated current-candidate row diagnostic; no historical-model inference or retry."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import shlex
import subprocess
import tarfile
import time
import traceback
from paper.scripts import prepare_titantpp_value_difference_rows as prep
from paper.scripts import titantpp_value_difference_rows_runtime as rt
from paper.scripts.titantpp_inter62_transport_evidence import recorded_run

PROJECT=prep.PROJECT
BUNDLE=prep.BUNDLE
PREFIXES=('runtime/','logs/','results/','rows/','permits/')
ROOT_FILES=('qualification_gate.json','validation_gate.json')


def require(ok,message):
    if not ok:raise ValueError(message)


sha,read,write,safe=prep.sha,prep.read,prep.write,prep.safe


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def verify_integrity(expected):
    require(sha(BUNDLE/'bootstrap_integrity.json')==expected,'Bootstrap SHA differs; remote0')
    b=read(BUNDLE/'bootstrap_integrity.json')
    require(b['schema']=='titantpp_value_difference_rows_bootstrap_v1' and b['spec_sha256']==prep.SPEC_SHA,'Bootstrap scope differs')
    for rel,digest in b['files'].items():require(sha(safe(BUNDLE,rel))==digest,'New bundle file changed; remote0: '+rel)
    for name,digest in b['external_dependencies'].items():
        p=Path(name);require(p.is_absolute() and p.resolve().is_relative_to(PROJECT),'Dependency outside project')
        require(sha(p)==digest,'Existing source/reference changed; remote0: '+name)
    require(b['contract_raw_sha256']==b['files']['contract.json'],'Contract raw binding differs')
    c=rt.verify_contract(BUNDLE,b['contract_raw_sha256'])
    require(c['contract_sha256']==b['contract_sha256'],'Contract binding differs')
    expected_packet={k:v for k,v in b['files'].items() if k!='baseline_local_references.json'}
    require(b['native_packet_files']==expected_packet,'Native packet scope differs')
    require('baseline_local_references.json' not in expected_packet,'Historical reference cannot deploy')
    return b,c


def make_packet(b):
    out=io.BytesIO()
    with tarfile.open(fileobj=out,mode='w:gz') as tar:
        for rel,digest in sorted(b['native_packet_files'].items()):
            p=safe(BUNDLE,rel);require(sha(p)==digest,'Packet source changed')
            tar.add(p,arcname=rel,recursive=False)
        raw=json.dumps({'schema':'current_candidate_rows_packet_v1','files':b['native_packet_files']},sort_keys=True).encode()
        info=tarfile.TarInfo('PACKET_MANIFEST.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
    return out.getvalue()


# This installer receives one tar packet on stdin, owns only the new exact root,
# then waits for its one supervisor and returns one immutable artifact archive.
INSTALLER=r'''
import hashlib,io,json,os,signal,subprocess,sys,tarfile,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
root=Path(ROOT)
def need(v,m):
 if not v:raise ValueError(m)
def h(p):
 d=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):d.update(b)
 return d.hexdigest()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def safe(n):
 p=Path(n);need(not p.is_absolute() and p.parts and '..' not in p.parts,'Unsafe packet name')
 q=root/p;need(q.resolve().is_relative_to(root.resolve()),'Escaping packet member');return q
need(str(root)==APPROVED_ROOT,'Root differs')
need(time.time()<DEADLINE,'Inherited deadline expired')
raw=sys.stdin.buffer.read();need(hashlib.sha256(raw).hexdigest()==PACKET_SHA,'Packet SHA mismatch')
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:
 members=tar.getmembers();names=[m.name for m in members]
 need(len(names)==len(set(names)) and all(m.isfile() for m in members),'Duplicate or non-file packet member')
 need('PACKET_MANIFEST.json' in names,'Packet manifest absent')
 manifest=json.load(tar.extractfile('PACKET_MANIFEST.json'))
 need(manifest['schema']=='current_candidate_rows_packet_v1' and manifest['files']==EXPECTED_FILES,'Packet manifest differs')
 need(set(names)==set(EXPECTED_FILES)|{'PACKET_MANIFEST.json'},'Unexpected packet member')
 for m in members:
  safe(m.name);need(m.size<=1000000000,'Oversized packet member')
  if m.name!='PACKET_MANIFEST.json':need(hashlib.sha256(tar.extractfile(m).read()).hexdigest()==EXPECTED_FILES[m.name],'Packet member SHA mismatch')
 need(not root.exists() and not root.is_symlink(),'Existing native root; retry forbidden')
 root.mkdir(parents=True,exist_ok=False)
 for m in members:
  p=safe(m.name);p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(tar.extractfile(m).read())
need(h(root/'contract.json')==CONTRACT_SHA,'Contract raw differs')
(root/'runtime').mkdir(exist_ok=False)
write(root/'runtime/installation.json',{'status':'complete','contract_raw_sha256':CONTRACT_SHA,'packet_sha256':PACKET_SHA,'files':EXPECTED_FILES,'original_bundle_writes':0,'installed_unix':time.time()})
p=None;exitcode=None;error=None;handlers={}
def interrupted(signum,frame):raise RuntimeError('Diagnostic installer interrupted: '+str(signum))
argv=['/usr/bin/python3',str(root/'operation/runtime.py'),'--root',str(root),'--contract-sha',CONTRACT_SHA,'--mode','supervise']
try:
 for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):handlers[sig]=signal.signal(sig,interrupted)
 with (root/'runtime/supervisor.stdout').open('xb') as out,(root/'runtime/supervisor.stderr').open('xb') as err:
  p=subprocess.Popen(argv,cwd=root,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
  write(root/'runtime/supervisor_launch.json',{'pid':p.pid,'argv':argv,'started_unix':time.time(),'direct_child':True})
  exitcode=p.wait(timeout=max(1,min(PIPELINE_TIMEOUT+30,DEADLINE-time.time())))
except BaseException as exc:
 error=type(exc).__name__+': '+str(exc)
 if p is not None and p.poll() is None:
  need(os.getpgid(p.pid)==p.pid and os.getsid(p.pid)==p.pid,'Owned supervisor group identity changed')
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=20)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
 if p is not None:exitcode=p.wait()
 write(root/'runtime/installer_failure.json',{'error':error,'traceback':traceback.format_exc(),'automatic_retry':False})
finally:
 for sig,handler in handlers.items():signal.signal(sig,handler)
write(root/'runtime/supervisor_child_exit.json',{'actual_child_exit_observed':p is not None,'exit_code':exitcode,'pid':p.pid if p else None,'argv':argv,'completed_unix':time.time(),'error':error,'supervisor_stdout_sha256':h(root/'runtime/supervisor.stdout'),'supervisor_stderr_sha256':h(root/'runtime/supervisor.stderr')})
# Failure evidence can be recovered, but can never be accepted as completion.
files={}
for pth in root.rglob('*'):
 if pth.is_file():
  rel=pth.relative_to(root).as_posix()
  if rel.startswith(('runtime/','logs/','results/','rows/','permits/')) or rel in ('qualification_gate.json','validation_gate.json'):
   need(not pth.is_symlink(),'Artifact symlink forbidden');files[rel]={'sha256':h(pth),'bytes':pth.stat().st_size}
manifest={'schema':'current_candidate_rows_archive_v1','contract_raw_sha256':CONTRACT_SHA,'files':files,'supervisor_actual_exit':exitcode,'error':error,'archived_unix':time.time()}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for rel in sorted(files):tar.add(root/rel,arcname=rel,recursive=False)
 payload=json.dumps(manifest,sort_keys=True,indent=2).encode();info=tarfile.TarInfo('ARCHIVE_MANIFEST.json');info.size=len(payload);tar.addfile(info,io.BytesIO(payload))
sys.stdout.buffer.flush()
sys.exit(0 if exitcode==0 and error is None else 1)
'''


def installer_code(b,c,packet_sha):
    values={'ROOT':c['remote_root'],'APPROVED_ROOT':prep.REMOTE,'DEADLINE':prep.DEADLINE,
            'CONTRACT_SHA':b['contract_raw_sha256'],'PACKET_SHA':packet_sha,
            'EXPECTED_FILES':b['native_packet_files'],'PIPELINE_TIMEOUT':c['pipeline_timeout_seconds']}
    return '\n'.join(k+'='+repr(v) for k,v in values.items())+'\n'+INSTALLER


def extract_archive(path,destination,raw_contract_sha):
    """Validate every byte/member before making the exclusive originals directory."""
    require(not destination.exists(),'Existing retrieval originals; retry prohibited')
    with tarfile.open(path,'r:gz') as tar:
        members=tar.getmembers();names=[m.name for m in members]
        require(len(names)==len(set(names)) and all(m.isfile() for m in members),'Duplicate/nonregular artifact')
        require('ARCHIVE_MANIFEST.json' in names,'Archive manifest absent')
        manifest=json.load(tar.extractfile('ARCHIVE_MANIFEST.json'))
        require(manifest.get('schema')=='current_candidate_rows_archive_v1'
                and manifest['contract_raw_sha256']==raw_contract_sha,'Archive contract differs')
        require(set(names)==set(manifest['files'])|{'ARCHIVE_MANIFEST.json'},'Archive members differ')
        for member in members:
            safe(destination,member.name)
            require(member.size<=1000000000,'Oversized archive member')
            if member.name=='ARCHIVE_MANIFEST.json':continue
            require(member.name.startswith(PREFIXES) or member.name in ROOT_FILES,'Foreign artifact scope')
            item=manifest['files'][member.name];data=tar.extractfile(member).read()
            require(len(data)==member.size==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],'Archive file mismatch')
        destination.mkdir(parents=True,exist_ok=False)
        for member in members:
            p=safe(destination,member.name);p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('xb') as f:f.write(tar.extractfile(member).read())
    return manifest


def check_receipts(originals,c,rawsha):
    worker=module(BUNDLE/'operation/evaluate.py','current_rows_collector_guard')
    helper=module(BUNDLE/'operation/helper.py','current_rows_frozen_runtime_guard')
    outcomes=[];data_receipts={}
    for stage in rt.STAGES:
        for job in c['jobs']:
            entry=rt.validate_result(originals,c,job,stage)
            r=read(originals/entry['path'])
            require(r.get('contract_raw_sha256')==rawsha and r.get('source_closure')==c['source']['closure']
                    and r.get('source_revision')==c['source']['revision'],'Worker lineage differs')
            helper.check_runtime(r['runtime'],c['runtimes']['5090'],'cuda',current_process=False)
            helper.check_runtime(r['runtime_after'],c['runtimes']['5090'],'cuda',current_process=False)
            require(r['runtime']==r['runtime_after'],'Worker Runtime drift')
            exitpath=originals/f"runtime/worker_exits/{stage}_{job['id']}.json";x=read(exitpath)
            require(x.get('actual_child_exit_observed') is True and x.get('exit_code')==0
                    and x.get('status')=='complete' and x.get('job_id')==job['id'] and x.get('stage')==stage
                    and x.get('contract_sha256')==c['contract_sha256'] and x.get('receipt')==entry
                    and x.get('pid')==r.get('pid') and x.get('argv',[None])[1:]==r.get('argv'),'Worker actual exit/receipt binding differs')
            require(x['launch_identity']['state']=='present' and x['launch_identity']['pid']==x['pid']
                    and x['launch_identity']['argv']==x['argv'] and x['start_ticks']==x['launch_identity']['start_ticks'],'Worker launch identity differs')
            for stream in ('stdout','stderr'):
                require(sha(originals/f"logs/{stage}_{job['id']}.{stream}")==x[stream+'_sha256'],'Worker stream SHA differs')
            outcomes.append(x)
            if stage!='qualification':
                worker.validate_six_metrics(r['metrics'],job['validation_reference'] if stage=='validation' else r['metrics'])
                data_receipts[job['id']+'/'+stage]=entry
    for stage in ('qualification','validation'):
        gate=read(originals/(stage+'_gate.json'))
        worker.verify_gate(originals,c,stage,sha(originals/(stage+'_gate.json')))
        require(len(gate['receipts'])==len(gate['worker_exits'])==3,'Incomplete prior stage gate')
    pipeline=read(originals/'runtime/pipeline_complete.json');direct=read(originals/'runtime/pipeline_actual_exit.json')
    require(pipeline['status']=='complete' and pipeline['qualification_count']==3 and pipeline['data_worker_count']==6
            and {(x['stage'],x['job_id']):x for x in pipeline['workers']}=={(x['stage'],x['job_id']):x for x in outcomes} and len(pipeline['workers'])==9 and pipeline['contract_sha256']==c['contract_sha256'],'Pipeline stage proof differs')
    require(direct.get('actual_child_exit_observed') is True and direct.get('exit_code')==0
            and direct['contract_sha256']==c['contract_sha256'],'Pipeline direct exit missing')
    for stream in ('stdout','stderr'):require(sha(originals/('runtime/pipeline.'+stream))==direct[stream+'_sha256'],'Pipeline log changed')
    supervisor=read(originals/'runtime/supervisor_child_exit.json')
    require(supervisor.get('actual_child_exit_observed') is True and supervisor.get('exit_code')==0,'Supervisor child actual exit missing')
    terminal=read(originals/'runtime/terminal.json')
    require(terminal.get('status')=='complete' and terminal.get('all_owned_absent') is True
            and terminal['contract_sha256']==c['contract_sha256'] and terminal['pipeline_actual_exit']==0
            and not terminal['root_owners']['owners'] and not terminal['root_owners']['query_errors']
            and terminal['gpu']['gpu_uuid']==c['runtimes']['5090']['gpu_uuid'] and not terminal['gpu']['compute_pids'], 'Terminal owner/GPU proof missing')
    require(len(terminal['owner_probes'])==10 and {x['pid'] for x in terminal['owner_probes']}=={x['pid'] for x in outcomes}|{direct['pid']}
            and all(x['state']=='absent' and x.get('explicit_absence') is True for x in terminal['owner_probes']), 'Explicit owned PID absence missing')
    require(not (originals/'runtime/failure.json').exists() and not (originals/'runtime/installer_failure.json').exists(),'Failure evidence cannot be completed')
    return data_receipts,terminal


def complete_receipt(b,c,expected,archive,manifest):
    originals=BUNDLE/'retrieval/originals'
    require(manifest['supervisor_actual_exit']==0 and manifest.get('error') is None,'Archived supervisor failed')
    data_receipts,terminal=check_receipts(originals,c,b['contract_raw_sha256'])
    verify_integrity(expected)
    transport=read(BUNDLE/'control/owned_attempt_exit.json')
    require(transport['actual_transport_exit_observed'] is True and transport['actual_transport_exit']==0 and not transport['timeout'], 'Transport exit unconfirmed')
    files={str(p.relative_to(PROJECT)):sha(p) for p in sorted((BUNDLE/'retrieval').rglob('*')) if p.is_file()}
    files.update({str(p.relative_to(PROJECT)):sha(p) for p in sorted((BUNDLE/'control').rglob('*')) if p.is_file()})
    files[str((BUNDLE/'bootstrap_integrity.json').relative_to(PROJECT))]=expected
    receipt={'schema':'current_candidate_rows_final_v1','status':'complete','candidate_count':3,'seed_count':3,'data_worker_count':6,
        'qualification_count':3,'actual_native_worker_exit0_count':9,'actual_data_worker_exit0_count':6,'actual_pipeline_exit0_count':1,
        'actual_supervisor_child_exit0_count':1,'validation_replay_count':3,'stages':list(rt.STAGES),
        'source_closure':c['source']['closure'],'training_source_revision':c['source']['revision'],'operation_code_commit':c['operation_code_commit'],
        'selected_models':{j['id']:{'seed':j['seed'],'epoch':j['selected_epoch'],'checkpoint_sha256':j['checkpoint']['sha256'],
              'tensor_sha256':j['checkpoint_state_sha256'],'metadata_sha256':j['checkpoint_metadata_sha256'],'history_sha256':j['selection_history_sha256']} for j in c['jobs']},
        'input_sha256':c['datasets']['raf_spare_parts']['sha256'],'target_populations':c['datasets']['raf_spare_parts']['populations'],
        'contract_raw_sha256':b['contract_raw_sha256'],'contract_sha256':c['contract_sha256'],'bootstrap_integrity_sha256':expected,
        'archive_sha256':sha(archive),'manifest_sha256':sha(originals/'ARCHIVE_MANIFEST.json'),'originals_path':str(originals.relative_to(PROJECT)),
        'data_receipts':data_receipts,'native_runtime':c['runtimes']['5090'],'terminal_observed_unix':terminal['observed_unix'],
        'all_owned_absent':True,'new_training':False,'historical_CNN_inference':False,'held_out_test_evaluated':False,'automatic_retry':False,
        'supervisor_own_os_exit_directly_observed':False,'installer_observed_supervisor_child_exit':True,'files':files,'completed_unix':time.time()}
    write(BUNDLE/'retrieval/final_receipt.json',receipt);return receipt


def verify_complete(expected):
    b,c=verify_integrity(expected)
    path=BUNDLE/'retrieval/final_receipt.json';r=read(path)
    require(r.get('status')=='complete' and r.get('candidate_count')==3 and r.get('data_worker_count')==6
            and r.get('qualification_count')==3 and r.get('actual_native_worker_exit0_count')==9
            and r.get('actual_data_worker_exit0_count')==6 and r.get('actual_pipeline_exit0_count')==1
            and r.get('validation_replay_count')==3 and r.get('historical_CNN_inference') is False
            and r.get('held_out_test_evaluated') is False and r.get('new_training') is False
            and r.get('bootstrap_integrity_sha256')==expected and r.get('contract_sha256')==c['contract_sha256']
            and r.get('contract_raw_sha256')==b['contract_raw_sha256'],'Completion receipt scope differs')
    originals=BUNDLE/'retrieval/originals';archive=BUNDLE/'retrieval/originals.tar.gz'
    require(r['originals_path']==str(originals.relative_to(PROJECT)) and r['archive_sha256']==sha(archive)
            and r['manifest_sha256']==sha(originals/'ARCHIVE_MANIFEST.json'),'Completion archive binding differs')
    manifest=read(originals/'ARCHIVE_MANIFEST.json')
    require(manifest['contract_raw_sha256']==b['contract_raw_sha256'] and manifest['supervisor_actual_exit']==0
            and manifest.get('error') is None,'Completed manifest identity differs')
    # Re-read archived originals too; a separately modified manifest cannot bless changed files.
    with tarfile.open(archive,'r:gz') as tar:
        names=tar.getnames();require(len(names)==len(set(names)) and set(names)==set(manifest['files'])|{'ARCHIVE_MANIFEST.json'},'Archive coverage differs')
        require(tar.extractfile('ARCHIVE_MANIFEST.json').read()==(originals/'ARCHIVE_MANIFEST.json').read_bytes(),'Stored manifest differs from archive')
        for rel,item in manifest['files'].items():
            actual=safe(originals,rel);raw=tar.extractfile(rel).read()
            require(sha(actual)==hashlib.sha256(raw).hexdigest()==item['sha256'] and len(raw)==actual.stat().st_size==item['bytes'],'Completed original changed')
    expected_files={str(x.relative_to(PROJECT)) for base in (BUNDLE/'retrieval',BUNDLE/'control') for x in base.rglob('*') if x.is_file() and x!=path}
    expected_files.add(str((BUNDLE/'bootstrap_integrity.json').relative_to(PROJECT)))
    require(set(r['files'])==expected_files and len(expected_files)>20,'Completion file coverage differs')
    for name,digest in r['files'].items():require(sha(safe(PROJECT,name))==digest,'Completed artifact changed')
    receipts,terminal=check_receipts(originals,c,b['contract_raw_sha256'])
    require(receipts==r['data_receipts'] and r['source_closure']==c['source']['closure']
            and r['training_source_revision']==c['source']['revision'] and r['input_sha256']==c['datasets']['raf_spare_parts']['sha256']
            and r['target_populations']==c['datasets']['raf_spare_parts']['populations'],'Completed scientific linkage differs')
    return r


def advance(expected):
    b,c=verify_integrity(expected)
    if (BUNDLE/'retrieval/final_receipt.json').exists():return verify_complete(expected)
    require(not (BUNDLE/'ACTION_REQUIRED.json').exists() and not (BUNDLE/'control/attempt_intent.json').exists()
            and not (BUNDLE/'retrieval').exists(),'Previous attempt/incomplete retrieval; automatic retry forbidden')
    require(time.time()<prep.DEADLINE,'Inherited absolute deadline expired')
    packet=make_packet(b);packet_sha=hashlib.sha256(packet).hexdigest()
    write(BUNDLE/'control/attempt_intent.json',{'contract_sha256':c['contract_sha256'],'bootstrap_integrity_sha256':expected,
          'packet_sha256':packet_sha,'remote_root':c['remote_root'],'host':'5090','started_unix':time.time(),'automatic_retry':False,
          'new_training':False,'historical_CNN_inference':False,'held_out_test_evaluated':False})
    code=installer_code(b,c,packet_sha)
    argv=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','5090','/usr/bin/python3 -c '+shlex.quote(code)]
    retrieval=BUNDLE/'retrieval';retrieval.mkdir(exist_ok=False);archive=retrieval/'originals.tar.gz'
    try:
        with archive.open('xb') as output:
            result=recorded_run(BUNDLE,'owned_attempt',argv,input=packet,stdout=output,
                                timeout=max(1,min(c['pipeline_timeout_seconds']+180,prep.DEADLINE-time.time()+120)))
        manifest=extract_archive(archive,retrieval/'originals',b['contract_raw_sha256'])
        write(retrieval/'archive_receipt.json',{'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
              'manifest_sha256':sha(retrieval/'originals/ARCHIVE_MANIFEST.json'),'file_count':len(manifest['files']),
              'actual_transport_exit':result.returncode,'held_out_test_evaluated':False})
        require(result.returncode==0,'Native diagnostic failed; preserved output; no automatic retry')
        return complete_receipt(b,c,expected,archive,manifest)
    except BaseException as error:
        write(BUNDLE/'ACTION_REQUIRED.json',{'status':'failed_or_unconfirmed','error':type(error).__name__,'message':str(error),
            'traceback':traceback.format_exc(),'automatic_retry':False,'existing_training_modified':False,
            'observed_unix':time.time(),'transport_receipt':'control/owned_attempt_exit.json'})
        raise


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','advance'],required=True);p.add_argument('--integrity-sha')
    a=p.parse_args(argv)
    if a.mode=='prepare':result=prep.prepare()
    else:
        require(a.integrity_sha is not None,'Explicit integrity pin required');result=advance(a.integrity_sha)
    print(json.dumps({'status':result['status'],'bundle':str(BUNDLE),'bootstrap_integrity_sha256':result.get('bootstrap_integrity_sha256'),
                      'candidate_count':result.get('candidate_count'),'data_worker_count':result.get('data_worker_count')},ensure_ascii=False))


if __name__=='__main__':main()
