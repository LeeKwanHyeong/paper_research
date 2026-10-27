"""Explicit transport-only recovery: framed deployment, unchanged frozen experiment."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ART = ROOT / 'search_artifacts/phen_taxi_internal_holdout_20261027_v1'
PAYLOAD = ART / 'payload'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def safe_relative(name: str) -> bool:
    p = PurePosixPath(name)
    return bool(p.parts) and not p.is_absolute() and '..' not in p.parts


DISPATCHER = r'''import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
c=json.loads((root/'EXECUTION.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
with (root/'DISPATCHER.claim').open('x') as f:f.write(str(os.getpid())+'\n')
start=time.time()
with (root/'supervisor.stdout.log').open('xb') as out,(root/'supervisor.stderr.log').open('xb') as err:
 p=subprocess.Popen([c['python'],'-u',str(root/'supervise.py')],cwd=root,stdin=subprocess.DEVNULL,stdout=out,stderr=err)
 code=p.wait()
record={'actual_supervisor_exit_code':code,'supervisor_pid':p.pid,'observed_by_dispatcher_pid':os.getpid(),'elapsed_seconds':time.time()-start,'streams':{n:{'bytes':(root/n).stat().st_size,'sha256':sha(root/n)} for n in ['supervisor.stdout.log','supervisor.stderr.log']}}
with (root/'supervisor.actualexit.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
raise SystemExit(code if code>=0 else 1)
'''


def deploy_script(config: dict, package: dict) -> str:
    return r'''
import hashlib,io,json,os,shlex,subprocess,sys,tarfile,time
from pathlib import Path,PurePosixPath
root=Path(__ROOT__)
expected=__MEMBERS__
content=sys.stdin.buffer.read()
assert hashlib.sha256(content).hexdigest()==__ARCHIVE_SHA__
assert root.parent.is_dir(), 'Experiment artifact parent must already exist'
assert not root.exists(), 'Existing remote attempt; no overwrite/resume'
with tarfile.open(fileobj=io.BytesIO(content),mode='r:gz') as tar:
 members=tar.getmembers()
 assert len(members)==len(expected) and {m.name for m in members}==set(expected)
 for m in members:
  p=PurePosixPath(m.name)
  assert m.isfile() and not p.is_absolute() and '..' not in p.parts
  raw=tar.extractfile(m).read()
  assert len(raw)==expected[m.name]['bytes'] and hashlib.sha256(raw).hexdigest()==expected[m.name]['sha256']
 root.mkdir(exist_ok=False)
 for m in members:
  p=root/m.name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(tar.extractfile(m).read())
c=json.loads((root/'EXECUTION.json').read_text())
assert str(root)==c['remote_root'] and c['host']=='5090' and c['test_access'] is False
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==c['boot_id']
tmux=next((str(p) for p in [Path('/opt/miniconda3/envs/ai_env/bin/tmux'),Path('/usr/bin/tmux')] if p.is_file() and os.access(p,os.X_OK)),None)
assert tmux, 'No approved tmux executable'
session=c.get('tmux_session','taxi_internal_holdout_seed42_20261027_v1')
assert session and all(x.isalnum() or x=='_' for x in session)
has=subprocess.run([tmux,'has-session','-t',session],capture_output=True,text=True)
assert has.returncode!=0, 'Existing tmux session; no duplicate start'
env=os.environ.copy();launch_env=dict(c['environment']);env.update(launch_env)
for k,suffix in [('MPLCONFIGDIR','matplotlib'),('XDG_CACHE_HOME','xdg'),('TORCHINDUCTOR_CACHE_DIR','torchinductor')]:
 p=root/'cache'/suffix;p.mkdir(parents=True,exist_ok=True);env[k]=str(p);launch_env[k]=str(p)
env['PYTHONDONTWRITEBYTECODE']='1';launch_env['PYTHONDONTWRITEBYTECODE']='1'
dispatcher=__DISPATCHER__
with (root/'dispatcher.py').open('x') as f:f.write(dispatcher)
# The shell observes the dispatcher process's real exit; the dispatcher observes
# the supervisor's exit. Neither is claimed to observe its own OS exit.
lines=['#!/bin/sh','set -C','cd '+shlex.quote(str(root))+' || exit 90',shlex.join(['/usr/bin/env',*[k+'='+v for k,v in sorted(launch_env.items())],c['python'],'-u',str(root/'dispatcher.py')]),'code=$?','printf \'%s\\n\' "$code" > '+shlex.quote(str(root/'dispatcher.actualexit')),'exit "$code"']
wrapper='\n'.join(lines)+'\n'
with (root/'launch_tmux.sh').open('x') as f:f.write(wrapper)
aux={n:{'bytes':(root/n).stat().st_size,'sha256':hashlib.sha256((root/n).read_bytes()).hexdigest()} for n in ['dispatcher.py','launch_tmux.sh']}
with (root/'LAUNCH_AUXILIARY.json').open('x') as f:json.dump(aux,f,indent=2)
launched=subprocess.run([tmux,'new-session','-d','-s',session,shlex.join(['/bin/sh',str(root/'launch_tmux.sh')])],cwd=root,env=env,capture_output=True,text=True)
receipt={'actual_tmux_launch_exit_code':launched.returncode,'stdout':launched.stdout,'stderr':launched.stderr,'remote_root':str(root),'boot_id':c['boot_id'],'tmux':tmux,'tmux_session':session,'archive_sha256':__ARCHIVE_SHA__,'auxiliary':aux,'launched_unix':time.time(),'test_access':False}
with (root/'LAUNCHED.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
raise SystemExit(launched.returncode)
'''.replace('__ROOT__', repr(config['remote_root'])).replace('__MEMBERS__', repr(package['members'])).replace('__ARCHIVE_SHA__', repr(package['archive_sha256'])).replace('__DISPATCHER__', repr(DISPATCHER))


def status_script(config: dict) -> str:
    return r'''
import json,os
from pathlib import Path,PurePosixPath
root=Path(__ROOT__)
c=json.loads((root/'EXECUTION.json').read_text())
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert boot==c['boot_id'], 'Host boot changed'
def under(n):
 p=PurePosixPath(n);assert not p.is_absolute() and '..' not in p.parts
 q=root/n;assert q.resolve().is_relative_to(root.resolve());return q
def tail(p):
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-4096));return f.read().decode(errors='replace')
r={'remote_root':str(root),'boot_id':boot,'read_only':True}
names=['LAUNCHED.json','SUPERVISOR_STARTED.json','fit.STARTED.json','fit.ACTUAL_EXIT.json','PROCESS_EXIT.json','SUPERVISOR_FAILURE.json','supervisor.actualexit.json','NATIVE_SMOKE.json',c.get('complete_relative_path','run/COMPLETE.json')]
names.extend(f'prelaunch_{i:02d}.ACTUAL_EXIT.json' for i in range(1,len(c.get('prelaunch_commands',[]))+1))
for n in names:
 p=under(n)
 if p.is_file():r[n]=json.loads(p.read_text())
p=root/'dispatcher.actualexit'
if p.is_file():r[p.name]=int(p.read_text().strip())
for n in ['fit.stdout.log','fit.stderr.log','supervisor.stderr.log']:
 p=root/n
 if p.is_file():r[n+'_tail']=tail(p)
for key,sub in [('SUPERVISOR_STARTED.json','identity'),('fit.STARTED.json','child')]:
 identity=r.get(key,{}).get(sub)
 if identity:
  p=Path('/proc')/str(identity['pid']);same=False
  if p.exists():
   s=(p/'stat').read_text();v=s[s.rfind(')')+2:].split();same=int(v[19])==identity['start_ticks']
  r[key+'_same_process_present']=same
out=under(c.get('output_relative_path','run'))
histories=list(out.rglob('history.json')) if out.is_dir() else []
assert len(histories)<=8, 'Unexpected history inventory'
for p in histories:
 rows=json.loads(p.read_text())
 if isinstance(rows,dict):rows=rows.get('history')
 if isinstance(rows,list):r[str(p.relative_to(root))]={'epochs_recorded':len(rows),'last':rows[-1] if rows else None}
print(json.dumps(r))
'''.replace('__ROOT__', repr(config['remote_root']))


def collect_script(config: dict) -> str:
    return r'''
import hashlib,io,json,sys,tarfile
from pathlib import Path,PurePosixPath
root=Path(__ROOT__)
c=json.loads((root/'EXECUTION.json').read_text())
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==c['boot_id']
def under(n):
 p=PurePosixPath(n);assert not p.is_absolute() and '..' not in p.parts
 q=root/n;assert q.resolve().is_relative_to(root.resolve());return q
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
e=json.loads((root/'PROCESS_EXIT.json').read_text())
assert e['actual_child_exit_code']==0 and not e['hard_timeout'] and e['input_unchanged']
assert json.loads((root/'supervisor.actualexit.json').read_text())['actual_supervisor_exit_code']==0
assert int((root/'dispatcher.actualexit').read_text())==0
assert not (root/'SUPERVISOR_FAILURE.json').exists()
for n,s in c['input_files'].items():assert sha(under(n))==s,n
complete=under(c.get('complete_relative_path','run/COMPLETE.json'))
assert complete.is_file() and complete.stat().st_size>0
completion=json.loads(complete.read_text());assert isinstance(completion,dict) and completion
assert completion.get('status') not in ['FAIL','FAILED','ERROR']
out=under(c.get('output_relative_path','run'));assert out.is_dir()
paths=set(p for p in out.rglob('*') if p.is_file())
paths.add(complete)
for p in root.iterdir():
 if p.is_file() and (p.suffix in ['.json','.log','.actualexit'] or p.name.endswith('.claim')):paths.add(p)
assert paths
for p in paths:assert not p.is_symlink() and p.resolve().is_relative_to(root.resolve())
manifest={str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(paths)}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for p in sorted(paths):tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
 raw=json.dumps(manifest,sort_keys=True).encode();m=tarfile.TarInfo('TRANSFER_MANIFEST.json');m.size=len(raw);tar.addfile(m,io.BytesIO(raw))
'''.replace('__ROOT__', repr(config['remote_root']))


# Only deployment transport framing changes; the frozen remote script and archive do not.
BOOTSTRAP = r'''import sys
stream=sys.stdin.buffer
def read_exact(size):
 parts=[]
 while size:
  part=stream.read(size)
  if not part:raise ValueError('Truncated deployment script frame')
  parts.append(part);size-=len(part)
 return b''.join(parts)
size=int.from_bytes(read_exact(8),'big')
if not 0<size<=8388608:raise ValueError('Deployment script length outside bound')
script=read_exact(size)
exec(compile(script,'<audited_deploy_script>','exec'),{'__name__':'__main__'})
'''


def deployment_frame(remote: str, archive: bytes) -> bytes:
    script = remote.encode('utf-8')
    if not 0 < len(script) <= 8388608:
        raise ValueError('Deployment script length outside bound')
    return len(script).to_bytes(8, 'big') + script + archive


def recovery_bound_paths() -> set[Path]:
    return {HERE / name for name in (
        'remote_control.py', 'supervise.py', 'SCIENCE_SPEC.json', 'AUTHORIZATION.json',
        'LOCAL_QA.json', 'PREFLIGHT_QA.json', 'PAYLOAD.json',
        'transport/deploy.claim', 'transport/deploy.json', 'transport/deploy.stdout',
        'transport/deploy.stderr', 'transport/TRANSPORT_FAILURE.json',
    )} | {ART / 'payload.tar.gz', PAYLOAD / 'EXECUTION.json'}


def verify_recovery_qa() -> dict:
    qa = json.loads((HERE / 'TRANSPORT_RECOVERY_QA.json').read_text())
    if qa.get('status') != 'PASS' or qa.get('parent_reviewed') is not True:
        raise ValueError('Transport recovery awaits explicit parent QA review')
    if qa.get('schema') != 'taxi_internal_holdout_transport_recovery_v2':
        raise ValueError('Unexpected recovery QA schema')
    if qa.get('controller_sha256') != sha(Path(__file__)):
        raise ValueError('Recovery controller differs from reviewed source')
    bound = qa.get('preserved_files', {})
    if set(bound) != {str(p) for p in recovery_bound_paths()}:
        raise ValueError('Incomplete or unexpected recovery evidence inventory')
    for name, expected in bound.items():
        path = Path(name)
        if path.is_symlink() or path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
            raise ValueError('Preserved recovery evidence changed: ' + name)
    prior = json.loads((HERE / 'transport/deploy.json').read_text())
    failure = json.loads((HERE / 'transport/TRANSPORT_FAILURE.json').read_text())
    stderr = (HERE / 'transport/deploy.stderr').read_bytes()
    stdout = (HERE / 'transport/deploy.stdout').read_bytes()
    if not (prior == failure and prior.get('mode') == 'deploy'
            and prior.get('actual_ssh_exit_code') == 1 and prior.get('timed_out') is False
            and stdout == b'' and stderr == b'/bin/bash: Argument list too long\n'):
        raise ValueError('Recovery is limited to the preserved argv-length deployment failure')
    for label, raw in [('stdout', stdout), ('stderr', stderr)]:
        if prior[label + '_bytes'] != len(raw) or prior[label + '_sha256'] != hashlib.sha256(raw).hexdigest():
            raise ValueError('Prior transport stream does not match its receipt')
    return qa


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['deploy', 'status', 'collect'])
    args = parser.parse_args()
    config = json.loads((PAYLOAD/'EXECUTION.json').read_text())
    if config['host'] != '5090' or config.get('test_access') is not False:
        raise ValueError('Only the authorized 5090 task with Test locked is supported')
    verify_recovery_qa()
    logs = HERE/'transport_v2'
    logs.mkdir(exist_ok=True)
    if (logs/'TRANSPORT_FAILURE.json').exists():
        raise ValueError('Previous transport failure preserved; no automatic retry')
    if args.mode == 'status':
        n = len(list(logs.glob('status_*.json'))) + 1
        label = f'status_{n:03d}'
    else:
        label = args.mode
    if any((logs/(label+suffix)).exists() for suffix in ['.claim','.json','.stdout','.stderr']):
        raise ValueError('Existing transport attempt; no implicit retry')
    payload = None
    if args.mode == 'deploy':
        qa = json.loads((HERE/'PREFLIGHT_QA.json').read_text())
        package = json.loads((HERE/'PAYLOAD.json').read_text())
        if qa['status'] != 'PASS' or qa['archive_sha256'] != package['archive_sha256'] or qa['execution_sha256'] != sha(PAYLOAD/'EXECUTION.json'):
            raise ValueError('Preflight QA does not bind this payload')
        if not qa.get('reviewed_sources') or not package.get('members'):
            raise ValueError('Empty QA or package inventory')
        for name, expected in qa['reviewed_sources'].items():
            if sha(Path(name)) != expected: raise ValueError('Reviewed source changed: '+name)
        for name, entry in package['members'].items():
            if not safe_relative(name): raise ValueError('Unsafe package path')
            p=PAYLOAD/name
            if p.is_symlink() or p.stat().st_size != entry['bytes'] or sha(p) != entry['sha256']:
                raise ValueError('Payload mismatch: '+name)
        payload = (ART/'payload.tar.gz').read_bytes()
        if hashlib.sha256(payload).hexdigest() != package['archive_sha256']:
            raise ValueError('Archive mismatch')
        remote = deploy_script(config, package)
        payload = deployment_frame(remote, payload)
        remote = BOOTSTRAP
    elif args.mode == 'status':
        remote = status_script(config)
    else:
        remote = collect_script(config)
    command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',config['host'],
             shlex.join([config['python'],'-c',remote])]
    write_new(logs/(label+'.claim'), {'mode':args.mode,'remote_root':config['remote_root']})
    started=datetime.now(timezone.utc).isoformat()
    timed_out=False
    try:
        run=subprocess.run(command,input=payload,capture_output=True,timeout={'deploy':120,'status':45,'collect':900}[args.mode])
        stdout,stderr,code=run.stdout,run.stderr,run.returncode
    except subprocess.TimeoutExpired as exc:
        stdout,stderr,code=exc.stdout or b'',exc.stderr or b'',124
        timed_out=True
    for suffix,raw in [('stdout',stdout),('stderr',stderr)]:
        with (logs/f'{label}.{suffix}').open('xb') as f:f.write(raw)
    receipt={'mode':args.mode,'controller_version':2,
             'recovery_qa_sha256':sha(HERE/'TRANSPORT_RECOVERY_QA.json'),
             'preserved_failure_receipt_sha256':sha(HERE/'transport/deploy.json'),
             'stdin_bytes':len(payload) if payload is not None else 0,
             'stdin_sha256':hashlib.sha256(payload or b'').hexdigest(),
             'actual_ssh_exit_code':None if timed_out else code,
             'timed_out':timed_out,'local_wrapper_exit_code':code,'started_utc':started,
             'finished_utc':datetime.now(timezone.utc).isoformat(),
             'stdout_bytes':len(stdout),'stderr_bytes':len(stderr),
             'stdout_sha256':hashlib.sha256(stdout).hexdigest(),'stderr_sha256':hashlib.sha256(stderr).hexdigest()}
    write_new(logs/(label+'.json'),receipt)
    if code:
        write_new(logs/'TRANSPORT_FAILURE.json',receipt)
        print(json.dumps(receipt));print(stderr.decode(errors='replace')[-3000:])
        return code
    if args.mode != 'collect':
        info=json.loads(stdout)
        if info['boot_id'] != config['boot_id']:raise ValueError('Host boot identity changed')
        print(json.dumps(info,ensure_ascii=False))
    else:
        destination=ART/'remote_run'
        if destination.exists():raise ValueError('Existing collection; no overwrite')
        with tarfile.open(fileobj=io.BytesIO(stdout),mode='r:gz') as tar:
            members=tar.getmembers()
            if not all(m.isfile() and safe_relative(m.name) for m in members):raise ValueError('Unsafe archive')
            if len({m.name for m in members}) != len(members):raise ValueError('Duplicate archive member')
            manifest=json.load(tar.extractfile('TRANSFER_MANIFEST.json'))
            if not manifest or {m.name for m in members} != set(manifest)|{'TRANSFER_MANIFEST.json'}:raise ValueError('Incomplete manifest')
            for m in members:
                raw=tar.extractfile(m).read()
                if m.name in manifest and (len(raw) != manifest[m.name]['bytes'] or hashlib.sha256(raw).hexdigest() != manifest[m.name]['sha256']):raise ValueError('Member checksum mismatch: '+m.name)
            destination.mkdir(exist_ok=False)
            for m in members:
                p=destination/m.name;p.parent.mkdir(parents=True,exist_ok=True)
                with p.open('xb') as f:f.write(tar.extractfile(m).read())
        write_new(HERE/'COLLECTION_COMPLETE.json',{'files':len(manifest),'archive_sha256':receipt['stdout_sha256'],'destination':str(destination),'test_access':False})
        print('Collected and SHA-verified',len(manifest),'files to',destination)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
