"""One approved exclusive RAF6 post-attention FFN CNN deployment; native checks gate all fresh fits."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

ROOT=Path(__file__).resolve().parents[2]
BUNDLE=ROOT/'search_artifacts/titantpp_backbone_post_attention_ffn_raf6_20261016_v1'
REPORT=ROOT/'reports/titantpp_backbone_post_attention_ffn_raf6_20261016_v1'

REMOTE_CODE=r'''
import os,sys,json,hashlib,tarfile,io,subprocess,shlex,time
from pathlib import Path
root=Path(ROOT);blob=sys.stdin.buffer.read()
assert hashlib.sha256(blob).hexdigest()==ARCHIVE_SHA
assert not root.exists(),'Existing RAF root: no retry or overwrite'
out=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
assert not any('GPU-c9adc246-ef66-7906-bc07-6e152fa9952f' in row for row in out.splitlines()),'Pinned GPU busy'
root.mkdir(parents=False,exist_ok=False)
names=set()
with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as t:
 for m in t.getmembers():
  p=Path(m.name);assert m.isfile() and not p.is_absolute() and '..' not in p.parts and m.name not in names
  names.add(m.name);target=root/p;target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as f:f.write(t.extractfile(m).read())
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert h(root/'deployment_manifest.json')==MANIFEST_SHA
manifest=json.loads((root/'deployment_manifest.json').read_text())
assert names==set(manifest['files'])|{'deployment_manifest.json'}
for n,d in manifest['files'].items():assert h(root/n)==d,n
c=json.loads((root/'execution_contract.json').read_text());host=c['hosts']['5090']
assert host['root']==str(root)
assert c['limits']['inherited_absolute_host_deadline_unix']==1792679647.915156 and time.time()<c['limits']['inherited_absolute_host_deadline_unix'],'Inherited host deadline expired'
(root/'cache/matplotlib').mkdir(parents=True,exist_ok=True)
command=[host['python'],str(root/'operation/runtime.py'),'--contract',str(root/'execution_contract.json'),'--host','5090','--mode','pipeline']
wrapper='\n'.join([
 'import subprocess,json,time,sys,os,hashlib',
 'from pathlib import Path',
 'root=Path('+repr(str(root))+'); cmd='+repr(command)+'; digest='+repr(manifest['contract_sha256']),
 'ticks=lambda pid:int(Path("/proc",str(pid),"stat").read_text().rsplit(")",1)[1].split()[19])',
 'log=(root/"pipeline.log").open("xb")',
 'p=subprocess.Popen(cmd,cwd=root/"source",stdout=log,stderr=subprocess.STDOUT)',
 'launch={"pid":p.pid,"start_ticks":ticks(p.pid),"observer_pid":os.getpid(),"observer_start_ticks":ticks(os.getpid()),"argv":cmd,"contract_sha256":digest,"started_unix":time.time()}',
 '(root/"pipeline_launch.json").open("x").write(json.dumps(launch,sort_keys=True,indent=2)+chr(10))',
 'rc=p.wait(); log.close()',
 'receipt={"returncode":rc,"actual_child_exit_observed":True,"pid":p.pid,"observer_pid":os.getpid(),"argv":cmd,"observed_unix":time.time(),"contract_sha256":digest,"launch_sha256":hashlib.sha256((root/"pipeline_launch.json").read_bytes()).hexdigest()}',
 '(root/"pipeline_exit.json").open("x").write(json.dumps(receipt,sort_keys=True,indent=2)+chr(10))',
 'sys.exit(rc)'])
compile(wrapper,'owned_pipeline_wrapper','exec')
launch=shlex.join(['env',*[k+'='+v for k,v in host['environment'].items()],'/usr/bin/python3','-c',wrapper])
with (root/'deployment_receipt.json').open('x') as f:
 json.dump({'archive_sha256':ARCHIVE_SHA,'manifest_sha256':MANIFEST_SHA,'source_commit':COMMIT,'installed_unix':time.time(),'file_count':len(manifest['files']),'launch_command':launch},f,indent=2)
subprocess.run([host['tmux_binary'],'new-session','-d','-s',host['tmux'],launch],check=True)
print(json.dumps({'status':'qualification_pipeline_started','root':str(root),'session':host['tmux'],'installed_files':len(manifest['files']),'started_unix':time.time()}))
'''

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n')

def deploy(integrity_sha):
    integrity=BUNDLE/'followup_integrity.json'
    assert sha(integrity)==integrity_sha,'Local integrity mismatch'
    for n,d in read(integrity)['files'].items():assert sha(ROOT/n)==d,'Local SHA mismatch '+n
    seal=read(integrity); assert read(ROOT/seal['local_tests_receipt'])['actual_exit']==0
    assert not (REPORT/'deployment_intent.json').exists(),'Existing attempt; no automatic retry'
    manifest=read(BUNDLE/'deployment_manifest.json');c=read(BUNDLE/'execution_contract.json')
    assert c['limits']['inherited_absolute_host_deadline_unix']==1792679647.915156 and time.time()<c['limits']['inherited_absolute_host_deadline_unix'],'Inherited host deadline expired'
    approval=read(BUNDLE/'approval.json'); assert approval['approved'] and approval['fresh_fit_count']==6 and approval['maximum_concurrent_workers']==3 and approval['fixed_checkpoint_test_authorized']
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as tar:
        for n in sorted(set(manifest['files'])|{'deployment_manifest.json'}):tar.add(BUNDLE/n,arcname=n,recursive=False)
    raw=buffer.getvalue();archive_sha=hashlib.sha256(raw).hexdigest()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    remote='ROOT='+repr(c['hosts']['5090']['root'])+'\nARCHIVE_SHA='+repr(archive_sha)+'\nMANIFEST_SHA='+repr(sha(BUNDLE/'deployment_manifest.json'))+'\nCOMMIT='+repr(commit)+'\n'+REMOTE_CODE
    compile(remote,'remote_deployment','exec')
    with (REPORT/'deployment_command.py.txt').open('x') as f:f.write(remote)
    with (REPORT/'deployment.tar.gz').open('xb') as f:f.write(raw)
    write(REPORT/'deployment_intent.json',{'approved':True,'host':'5090','root':c['hosts']['5090']['root'],'commit':commit,'archive_sha256':archive_sha,'created_unix':time.time(),'automatic_retry':False})
    try:
        p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','5090',shlex.join(['/usr/bin/python3','-c',remote])],input=raw,capture_output=True,timeout=180)
        stdout,stderr,rc=p.stdout,p.stderr,p.returncode
    except subprocess.TimeoutExpired as exc:
        stdout,stderr,rc=exc.stdout or b'',exc.stderr or b'',None
    for name,data in [('deployment.stdout',stdout),('deployment.stderr',stderr)]:
        with (REPORT/name).open('xb') as f:f.write(data)
    write(REPORT/'deployment_transport.json',{'actual_transport_exit':rc,'timeout':rc is None,'stdout_sha256':sha(REPORT/'deployment.stdout'),'stderr_sha256':sha(REPORT/'deployment.stderr'),'stdout_bytes':len(stdout),'stderr_bytes':len(stderr),'truncated':False})
    print(stdout.decode(errors='replace'));print(stderr.decode(errors='replace'))
    if rc!=0:raise RuntimeError('Deployment outcome requires review; no automatic retry')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--integrity-sha',required=True);a=p.parse_args();deploy(a.integrity_sha)
