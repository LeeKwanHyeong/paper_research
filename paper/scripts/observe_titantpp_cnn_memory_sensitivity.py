"""Read-only observation and single terminal collection of the new diagnostic."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time
from paper.scripts.prepare_titantpp_cnn_memory_sensitivity import PROJECT, sha

OBSERVE = r'''
import pathlib,json,sys,time,subprocess,hashlib
r=pathlib.Path(sys.argv[1]);h=sys.argv[2];expected=sys.argv[3]
assert hashlib.sha256((r/'contract.json').read_bytes()).hexdigest()==expected
c=json.loads((r/'contract.json').read_text());assert c['remote_roots'][h]==str(r)
o=dict(host=h,observed_unix=time.time(),contract_sha256=expected,files={},workers=[])
for rel in ['launch/dispatcher_exit.json','runtime/'+h+'/receipt.json']:
 p=r/rel
 if p.exists():
  b=p.read_bytes();o['files'][rel]=dict(sha256=hashlib.sha256(b).hexdigest(),value=json.loads(b))
e=r/'runtime'/h/'events.jsonl'
rows=[json.loads(x) for x in e.read_text().splitlines()] if e.exists() else []
launched={x['pid']:x for x in rows if x['event']=='launch'};finished={x['pid'] for x in rows if x['event']=='finish'}
o['finished']=[dict(job=x['job_id'],split=x['split'],status=x['status'],exit=x['actual_worker_exit_code']) for x in rows if x['event']=='finish']
for pid,x in launched.items():
 proc=pathlib.Path('/proc')/str(pid)
 cmd=(proc/'cmdline').read_bytes().replace(bytes([0]),b' ').decode() if proc.exists() else None
 if pid in finished:
  if cmd and str(r) in cmd:o.setdefault('unexpected_live_finished',[]).append(pid)
  continue
 w=dict(pid=pid,job=x['job_id'],split=x['split'],cmdline=cmd)
 p=r/'runtime'/h/'workers'/x['job_id']/(x['split']+'.stdout')
 lines=p.read_text().splitlines() if p.exists() else []
 w['progress']=lines[-1] if lines else None;o['workers'].append(w)
for j in (r/'runtime'/h/'workers').glob('*/*.stderr'):
 b=j.read_bytes()
 if b'Traceback' in b:o.setdefault('errors',[]).append(dict(path=str(j.relative_to(r)),sha256=hashlib.sha256(b).hexdigest(),text=b.decode()[-4000:]))
o['gpu_compute']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader']).decode()
print(json.dumps(o))
'''

COLLECT = r'''
import pathlib,json,sys,hashlib,tarfile
r=pathlib.Path(sys.argv[1]);host=sys.argv[2];expected=sys.argv[3]
assert hashlib.sha256((r/'contract.json').read_bytes()).hexdigest()==expected
c=json.loads((r/'contract.json').read_text());assert c['remote_roots'][host]==str(r)
receipt=json.loads((r/'runtime'/host/'receipt.json').read_text())
assert receipt['status']=='complete' and receipt['actual_worker_exit0_count']==receipt['expected_worker_count']
assert json.loads((r/'launch/dispatcher_exit.json').read_text())['actual_dispatcher_os_exit']==0
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for directory,arc in [(r/'results','results'),(r/'runtime'/host,'runtime/'+host),(r/'launch','launch/'+host)]:
  for p in sorted(directory.rglob('*')):
   if p.is_file() and 'cache' not in p.relative_to(directory).parts:
    assert not p.is_symlink();tar.add(p,arcname=arc+'/'+p.relative_to(directory).as_posix(),recursive=False)
'''


def observe(bundle, digest):
    assert sha(bundle/'contract.json') == digest
    c=json.loads((bundle/'contract.json').read_text())
    folder=PROJECT/'reports'/bundle.name/'observations'/str(time.time_ns())
    folder.mkdir(parents=True,exist_ok=False)
    def one(h):
        cmd=['/usr/bin/python3','-c',OBSERVE,c['remote_roots'][h],h,digest]
        r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',h,' '.join(map(shlex.quote,cmd))],capture_output=True,timeout=45)
        for n,b in [('stdout',r.stdout),('stderr',r.stderr)]:
            with (folder/(h+'.'+n)).open('xb') as f:f.write(b)
        assert r.returncode==0,r.stderr.decode()
        value=json.loads(r.stdout)
        with (folder/(h+'.json')).open('x') as f:json.dump(value,f,indent=2)
        print(json.dumps(dict(host=h,observed_unix=value['observed_unix'],finished=value['finished'],
            workers=[{k:v for k,v in w.items() if k!='cmdline'} for w in value['workers']],gpu=value['gpu_compute'],
            errors=value.get('errors',[]),dispatcher_exit=value['files'].get('launch/dispatcher_exit.json',{}).get('value')),ensure_ascii=False),flush=True)
        return value
    with ThreadPoolExecutor(2) as pool:
        result=dict(zip(['5080','5090'],pool.map(one,['5080','5090'])))
    return result


def collect(bundle, digest, observed):
    c=json.loads((bundle/'contract.json').read_text())
    for h,o in observed.items():
        assert not o['workers'] and not o.get('errors') and not o.get('unexpected_live_finished')
        assert o['files']['runtime/'+h+'/receipt.json']['value']['status']=='complete'
        assert o['files']['launch/dispatcher_exit.json']['value']['actual_dispatcher_os_exit']==0
    root=bundle/'retrieval';root.mkdir(exist_ok=False)
    manifest={}
    for h in ('5080','5090'):
        archive=root/(h+'.tar.gz');cmd=['/usr/bin/python3','-c',COLLECT,c['remote_roots'][h],h,digest]
        with archive.open('xb') as stream:
            r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',h,' '.join(map(shlex.quote,cmd))],stdout=stream,stderr=subprocess.PIPE,timeout=180)
        with (root/(h+'.stderr')).open('xb') as f:f.write(r.stderr)
        assert r.returncode==0,r.stderr.decode()
        manifest[archive.name]=sha(archive)
        with tarfile.open(archive,'r:gz') as tar:
            for m in tar:
                path=Path(m.name)
                assert m.isfile() and not path.is_absolute() and '..' not in path.parts
                assert path.parts[0] in ('results','runtime','launch')
                target=root/path;target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as out:out.write(tar.extractfile(m).read())
                manifest[m.name]=sha(target)
    with (root/'manifest.json').open('x') as f:json.dump(dict(contract_sha256=digest,files=manifest),f,indent=2)
    print(json.dumps(dict(retrieval=str(root),manifest_sha256=sha(root/'manifest.json'),file_count=len(manifest))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True,type=Path);p.add_argument('--contract-sha',required=True)
    p.add_argument('--collect-if-complete',action='store_true');a=p.parse_args();b=a.bundle.resolve()
    obs=observe(b,a.contract_sha)
    if a.collect_if_complete and all(o['files'].get('runtime/'+h+'/receipt.json',{}).get('value',{}).get('status')=='complete' for h,o in obs.items()):
        collect(b,a.contract_sha,obs)
