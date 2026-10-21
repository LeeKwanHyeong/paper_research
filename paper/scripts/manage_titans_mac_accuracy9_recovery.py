"""Prepare and deploy one approved immutable MAC recovery; short completion reads."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

from paper.scripts import titans_mac_accuracy9_recovery_runtime as r

ROOT = Path(__file__).resolve().parents[2]
REPORT_BASE = ROOT / 'reports/titans_mac_accuracy9_recovery_20261021_v1'
REPORT = REPORT_BASE / 'execution_v3'
BUNDLE = ROOT / 'search_artifacts/titans_mac_accuracy9_recovery_20261021_v3'
ORIGINAL = ROOT / 'search_artifacts/titans_mac_accuracy9_20261020_v1/execution_contract.json'
REMOTE = '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titans_mac_accuracy9_recovery_20261021_v3_5090'
SNAPSHOT = REPORT_BASE / 'preflight/ORIGINAL_SNAPSHOT_v3.stdout.json'
OPERATION_FILES = ('paper/scripts/run_titans_mac_accuracy9_recovery.py',
                   'paper/scripts/titans_mac_accuracy9_recovery_runtime.py',
                   'paper/scripts/titantpp_late_path18_5090_recovery_runtime.py')
require, read, sha, canonical, write = r.require, r.read, r.sha_file, r.sha_json, r.write


def prepare():
    require(not BUNDLE.exists(), 'Existing recovery bundle; preserve all attempts')
    original, snapshot = read(ORIGINAL), read(SNAPSHOT)
    require(canonical(original) == r.ORIGINAL_CANONICAL and snapshot['root'] == original['hosts']['5090']['root'], 'Original campaign changed')
    require(snapshot['gpu_processes']['actual_os_returncode'] == 0 and not snapshot['gpu_processes']['stdout'].strip(), 'Snapshot GPU not idle')
    h = original['hosts']['5090']; jobs = []
    for row in snapshot['jobs']:
        j = {**row['job'], 'recovery_mode': row['mode'], 'original_started_at': row['launch']['started_at']}
        folder = Path(h['root']) / 'run' / j['id']
        if row['mode'] == 'resume':
            j.update(resume_epoch=row['checkpoint']['epoch'], origin_resume_checkpoint={
                'path': row['checkpoint']['path'], 'sha256': row['checkpoint']['sha256']})
        else:
            for key, name in (('origin_train_receipt', 'train_receipt.json'), ('origin_train_exit', 'train_exit.json')):
                path = folder / name
                j[key] = {'path': str(path), 'sha256': snapshot['files'][str(path.relative_to(h['root']))]}
        jobs.append(j)
    operations = {name: sha(ROOT / name) for name in OPERATION_FILES}
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'diff', '--name-only', 'HEAD', '--', *OPERATION_FILES], cwd=ROOT, text=True)
    tracked = subprocess.check_output(['git', 'ls-files', '--', *OPERATION_FILES], cwd=ROOT, text=True).splitlines()
    require(not dirty.strip() and set(tracked) == set(OPERATION_FILES), 'Commit operation files before preparing')
    c = {'schema': r.SCHEMA, 'host': '5090', 'root': REMOTE, 'source_root': REMOTE + '/source',
        **{k: deepcopy(h[k]) for k in ('python', 'gpu_uuid', 'runtime_expected', 'absolute_host_deadline_unix')},
        'max_workers': 3, 'limits': original['limits'], 'boot_id': snapshot['boot_id'],
        'scientific_contract': {'path': h['root'] + '/execution_contract.json', 'raw_sha256': sha(ORIGINAL),
                                'canonical_sha256': canonical(original)},
        'source': {**deepcopy(original['source']), 'source_revision': original['source']['base_git_revision']},
        'input_files': original['input_files'], 'operation_files': operations,
        'required_source_directories': ['models', 'utils', 'sample_data'],
        'operation_files_sha256': canonical(operations), 'operation_git_revision': revision,
        'immutable_originals': {'root': h['root'], 'files': snapshot['files'], 'files_sha256': canonical(snapshot['files'])},
        'jobs': jobs, 'training_jobs': [j['id'] for j in jobs if j['recovery_mode'] == 'resume'],
        'reference_jobs': [j['id'] for j in jobs if j['recovery_mode'] == 'reference'],
        'validation_jobs': [j['id'] for j in jobs], 'test_jobs': [j['id'] for j in jobs],
        'authorization': {'resume_three_only': True, 'originals_immutable': True,
            'user_instruction': '그럼 다음에 또 끊겼을 때 원인을 잡을 수 있게 5090에 네트워크 상태랑 로그 수집을 걸어두자. 기존 실험이랑 checkpoint는 그대로 두고. 그리고 지금 실험은 재개할 수 있도록하자',
            'validation_test_collection_previously_authorized': True},
        'environment': {**h['environment'], 'PYTHONPATH': REMOTE + '/source', 'MPLCONFIGDIR': REMOTE + '/cache/matplotlib',
            'XDG_CACHE_HOME': REMOTE + '/cache', 'TORCHINDUCTOR_CACHE_DIR': REMOTE + '/cache/torchinductor'},
        'snapshot_raw_sha256': sha(SNAPSHOT), 'tmux_binary': h['tmux_binary'],
        'tmux_session': 'titans_mac9_resume_20261021_v3'}
    # Validation uses the frozen original contract on5090; here its local byte
    # equivalent is substituted only for the pure scope check.
    remote_binding = c['scientific_contract']['path']
    c['scientific_contract']['path'] = str(ORIGINAL)
    r.validate_contract(c)
    c['scientific_contract']['path'] = remote_binding
    BUNDLE.mkdir(parents=True, exist_ok=False)
    write(BUNDLE / 'recovery_contract.json', c, exclusive=True)
    write(REPORT / 'PREPARED.json', {'status': 'prepared', 'contract_sha256': canonical(c),
        'contract_raw_sha256': sha(BUNDLE / 'recovery_contract.json'), 'operation_commit': revision,
        'original_file_count': len(snapshot['files']), 'saved_epochs': {str(j['seed']): j['resume_epoch'] for j in jobs if j['recovery_mode'] == 'resume'}}, exclusive=True)
    with tarfile.open(BUNDLE / 'deployment.tar.gz', 'w:gz') as t:
        t.add(BUNDLE / 'recovery_contract.json', arcname='recovery_contract.json')
        for name in operations:
            t.add(ROOT / name, arcname='source/' + name)
    return read(REPORT / 'PREPARED.json')


def verify_local(expected):
    c = read(BUNDLE / 'recovery_contract.json'); require(canonical(c) == expected, 'Pinned recovery contract changed')
    require(sha(ORIGINAL) == c['scientific_contract']['raw_sha256'] and sha(SNAPSHOT) == c['snapshot_raw_sha256'], 'Origin binding changed')
    for name, digest in c['operation_files'].items():
        require(sha(ROOT / name) == digest, 'Recovery operation changed after seal')
    return c


def transport(label, command, *, data=None, timeout=45):
    stamp = time.strftime('%Y%m%dT%H%M%S', time.gmtime()) + '_' + str(time.time_ns())
    folder = REPORT / 'transport' / (stamp + '_' + label); folder.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    with (folder / 'stdout').open('xb') as out, (folder / 'stderr').open('xb') as err:
        process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=10',
            '-o', 'ServerAliveCountMax=2', '5090', command], stdin=subprocess.PIPE if data is not None else subprocess.DEVNULL, stdout=out, stderr=err)
        timed_out = False
        try:
            process.communicate(input=data, timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True; process.kill(); process.communicate()
    value = {'actual_os_returncode': process.returncode, 'timeout': timed_out, 'elapsed_seconds': time.monotonic() - began,
        'stdout_sha256': sha(folder / 'stdout'), 'stderr_sha256': sha(folder / 'stderr'), 'command': command}
    write(folder / 'TRANSPORT.json', value, exclusive=True)
    require(process.returncode == 0 and not timed_out, 'Transport failure; raw output preserved at ' + str(folder))
    return folder


def deploy(expected):
    c = verify_local(expected)
    write(REPORT / 'DEPLOYMENT_ATTEMPT.json', {'at': r.now(), 'contract_sha256': expected, 'automatic_retry': False}, exclusive=True)
    # The remote installer copies frozen source/input bytes into a new root; it
    # never writes any old artifact. All originals are checked before mkdir.
    program = '''import hashlib,io,json,pathlib,shutil,sys,tarfile
from pathlib import Path
b=sys.stdin.buffer.read();assert hashlib.sha256(b).hexdigest()==__ARCHIVE__
t=tarfile.open(fileobj=io.BytesIO(b),mode='r:gz');members=t.getmembers()
assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
c=json.load(t.extractfile('recovery_contract.json'));root=Path(c['root']);old=Path(c['immutable_originals']['root'])
assert not root.exists(),'Existing attempt; no retry'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==c['boot_id']
for n,s in c['immutable_originals']['files'].items():assert sha(old/n)==s,n
root.mkdir();t.extractall(root,filter='data')
source=root/'source'
for name in c['required_source_directories']:(source/name).mkdir(parents=True,exist_ok=True)
for n,s in {**c['source']['files'],**c['input_files']}.items():
 p=source/n;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists()
 shutil.copyfile(old/'source'/n,p);assert sha(p)==s,n
print(json.dumps({'status':'installed','root':str(root),'source_files':len(c['source']['files']),'original_files_preserved':len(c['immutable_originals']['files'])}))
'''.replace('__ARCHIVE__', repr(sha(BUNDLE / 'deployment.tar.gz')))
    folder = transport('install', shlex.join(['/usr/bin/python3', '-c', program]), data=(BUNDLE / 'deployment.tar.gz').read_bytes(), timeout=300)
    check = shlex.join(['env', *[k + '=' + v for k,v in c['environment'].items()], c['python'], '-m', r.MODULE,
        '--contract', c['root'] + '/recovery_contract.json', '--contract-sha', expected, '--mode', 'check'])
    transport('installed_source_check', 'cd ' + shlex.quote(c['source_root']) + ' && ' + check, timeout=120)
    command = shlex.join(['env', *[k + '=' + v for k,v in c['environment'].items()], c['python'], '-u', '-m', r.MODULE,
        '--contract', c['root'] + '/recovery_contract.json', '--contract-sha', expected, '--mode', 'supervise'])
    shell = 'cd ' + shlex.quote(c['source_root']) + ' && ' + command + ' > ' + shlex.quote(c['root'] + '/supervisor.stdout.log') + ' 2> ' + shlex.quote(c['root'] + '/supervisor.stderr.log')
    launched = transport('launch', shlex.join([c['tmux_binary'], 'new-session', '-d', '-s', c['tmux_session'], shell]))
    result = {'status': 'launch_requested_not_epoch_verified', 'at': r.now(), 'contract_sha256': expected,
        'install_transport': str(folder), 'launch_transport': str(launched), 'tmux': c['tmux_session']}
    write(REPORT / 'LAUNCHED.json', result, exclusive=True)
    return result


def status(expected):
    c = verify_local(expected)
    program = '''import json,pathlib,sys
from paper.scripts import titans_mac_accuracy9_recovery_runtime as r
p=pathlib.Path(sys.argv[1]);c=r.read(p/'recovery_contract.json');r.require(r.sha_json(c)==sys.argv[2],'Contract changed')
result={'at':r.now(),'boot_id':r.boot_id(),'records':{},'jobs':[]}
for name in ('start.json','status.json','dispatch_failure.json','final_receipt.json','dispatcher_exit.json'):
 if (p/name).exists():result['records'][name]=r.read(p/name)
for j in c['jobs']:
 f=p/'run'/j['id'];row={'job_id':j['id'],'mode':j['recovery_mode']}
 for h in f.glob('runs/*/*/seed_*/history.json'):
  v=r.read(h);history=v.get('history',[]);row['last_epoch']=history[-1]['epoch'] if history else None;row['recent_epochs']=history[-10:]
 for name in ('train_launch.json','train_exit.json','train_recovery_receipt.json','validation_exit.json','test_exit.json'):
  if (f/name).exists():row[name]=r.read(f/name)
 result['jobs'].append(row)
print(json.dumps(result,ensure_ascii=False,allow_nan=False))
'''
    command = 'cd ' + shlex.quote(c['source_root']) + ' && ' + shlex.join(['env', 'PYTHONDONTWRITEBYTECODE=1', c['python'], '-c', program, c['root'], expected])
    return transport('status', command)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', required=True, choices=('prepare', 'deploy', 'status'))
    p.add_argument('--contract-sha'); a = p.parse_args(argv)
    if a.mode == 'prepare':
        result = prepare()
    else:
        require(a.contract_sha, 'Pinned recovery SHA required')
        result = deploy(a.contract_sha) if a.mode == 'deploy' else {'transport': str(status(a.contract_sha))}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
