"""Deploy an evaluation-only bridge without interrupting the running MAC fits."""
from __future__ import annotations

from contextlib import contextmanager
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile

from paper.scripts import manage_titans_mac_accuracy9_recovery as recovery

ROOT, ORIGINAL, SNAPSHOT = recovery.ROOT, recovery.ORIGINAL, recovery.SNAPSHOT
REPORT_BASE = recovery.REPORT_BASE
REPORT = REPORT_BASE / 'execution_v4'
BUNDLE = ROOT / 'search_artifacts/titans_mac_accuracy9_recovery_20261021_v4'
REMOTE = '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titans_mac_accuracy9_recovery_20261021_v4_5090'
UPSTREAM_BUNDLE = ROOT / 'search_artifacts/titans_mac_accuracy9_recovery_20261021_v3'
LIVE_PROOF = REPORT_BASE / 'execution_v3/INITIAL_LIVE_OBSERVATION.json'
BRIDGE = 'paper/scripts/titans_mac_accuracy9_completion_bridge.py'
MODULE = 'paper.scripts.titans_mac_accuracy9_completion_bridge'
read, sha, canonical, require, write = recovery.read, recovery.sha, recovery.canonical, recovery.require, recovery.write
OPERATION_FILES = recovery.OPERATION_FILES


@contextmanager
def context(*, bundle=BUNDLE, report=REPORT):
    values = {'BUNDLE': bundle, 'REPORT': report, 'REMOTE': REMOTE}
    previous = {k: getattr(recovery, k) for k in values}
    for k,v in values.items(): setattr(recovery, k, v)
    try:
        yield
    finally:
        for k,v in previous.items(): setattr(recovery, k, v)


def prepare():
    require(not BUNDLE.exists(), 'Existing completion bridge; no retry')
    stage = REPORT / 'preparation_versions/base'
    with context(bundle=stage / 'bundle', report=stage / 'report'):
        recovery.prepare()
    c = read(stage / 'bundle/recovery_contract.json')
    upstream = read(UPSTREAM_BUNDLE / 'recovery_contract.json')
    live = read(LIVE_PROOF)
    require(all(j['exact_command_cwd_pgid_match'] for j in live['jobs']) and len(live['jobs']) == 3,
            'All three actual upstream identities must be recorded')
    c.update(continuation_from={'root': upstream['root'], 'contract_sha256': canonical(upstream),
        'contract_raw_sha256': sha(UPSTREAM_BUNDLE / 'recovery_contract.json'),
        'initial_live_observation': {'path': REMOTE + '/proofs/INITIAL_LIVE_OBSERVATION.json', 'sha256': sha(LIVE_PROOF)}},
        bridge_files={BRIDGE: sha(ROOT / BRIDGE)}, continuation_only=True,
        fresh_training_authorized=False, tmux_session='titans_mac9_validation_bridge_20261021_v4')
    c['bridge_files_sha256'] = canonical(c['bridge_files'])
    BUNDLE.mkdir(parents=True, exist_ok=False)
    (BUNDLE / 'proofs').mkdir()
    shutil.copyfile(LIVE_PROOF, BUNDLE / 'proofs/INITIAL_LIVE_OBSERVATION.json')
    write(BUNDLE / 'recovery_contract.json', c, exclusive=True)
    with tarfile.open(BUNDLE / 'deployment.tar.gz', 'w:gz') as t:
        t.add(BUNDLE / 'recovery_contract.json', arcname='recovery_contract.json')
        t.add(BUNDLE / 'proofs/INITIAL_LIVE_OBSERVATION.json', arcname='proofs/INITIAL_LIVE_OBSERVATION.json')
        for name in {**c['operation_files'], **c['bridge_files']}:
            t.add(ROOT / name, arcname='source/' + name)
    result = {'status': 'prepared_evaluation_only', 'contract_sha256': canonical(c),
        'contract_raw_sha256': sha(BUNDLE / 'recovery_contract.json'), 'upstream_contract_sha256': canonical(upstream),
        'fresh_fit_count': 0, 'already_running_fits_untouched': 3, 'validation_count': 9, 'test_count': 9,
        'operation_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    write(REPORT / 'PREPARED.json', result, exclusive=True)
    return result


def verify_local(expected):
    with context(): c = recovery.verify_local(expected)
    require(c['continuation_only'] is True and c['fresh_training_authorized'] is False
            and canonical(c['bridge_files']) == c['bridge_files_sha256'], 'Evaluation-only scope changed')
    for name, digest in c['bridge_files'].items(): require(sha(ROOT / name) == digest, 'Bridge source changed')
    require(sha(BUNDLE / 'proofs/INITIAL_LIVE_OBSERVATION.json') == c['continuation_from']['initial_live_observation']['sha256'], 'Live identity proof changed')
    return c


def transport(label, command, *, data=None, timeout=45):
    with context(): return recovery.transport(label, command, data=data, timeout=timeout)


def deploy(expected):
    c = verify_local(expected)
    write(REPORT / 'DEPLOYMENT_ATTEMPT.json', {'at': recovery.r.now(), 'contract_sha256': expected,
        'fresh_training_count': 0, 'upstream_fits_modified': False}, exclusive=True)
    program = '''import hashlib,io,json,pathlib,shutil,sys,tarfile
from pathlib import Path
b=sys.stdin.buffer.read();assert hashlib.sha256(b).hexdigest()==__ARCHIVE__
t=tarfile.open(fileobj=io.BytesIO(b),mode='r:gz');members=t.getmembers()
assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
c=json.load(t.extractfile('recovery_contract.json'));root=Path(c['root']);old=Path(c['immutable_originals']['root'])
assert not root.exists(),'Existing completion bridge'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==c['boot_id']
for n,s in c['immutable_originals']['files'].items():assert sha(old/n)==s,n
root.mkdir();t.extractall(root,filter='data');source=root/'source'
for name in c['required_source_directories']:(source/name).mkdir(parents=True,exist_ok=True)
for n,s in {**c['source']['files'],**c['input_files']}.items():
 p=source/n;p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists()
 shutil.copyfile(old/'source'/n,p);assert sha(p)==s,n
print(json.dumps({'status':'installed_evaluation_bridge','training_started':False,'root':str(root)}))
'''.replace('__ARCHIVE__', repr(sha(BUNDLE / 'deployment.tar.gz')))
    installed = transport('install', shlex.join(['/usr/bin/python3', '-c', program]),
        data=(BUNDLE / 'deployment.tar.gz').read_bytes(), timeout=300)
    def invocation(mode):
        return 'cd ' + shlex.quote(c['source_root']) + ' && ' + shlex.join(['env',
            *[k + '=' + v for k,v in c['environment'].items()], c['python'], '-u', '-m', MODULE,
            '--contract', c['root'] + '/recovery_contract.json', '--contract-sha', expected, '--mode', mode])
    checked = transport('bridge_preflight', invocation('preflight'), timeout=120)
    command = invocation('supervise') + ' > ' + shlex.quote(c['root'] + '/supervisor.stdout.log') + ' 2> ' + shlex.quote(c['root'] + '/supervisor.stderr.log')
    launched = transport('launch', shlex.join([c['tmux_binary'], 'new-session', '-d', '-s', c['tmux_session'], command]))
    result = {'status': 'bridge_launch_requested', 'at': recovery.r.now(), 'contract_sha256': expected,
        'upstream_training_unchanged': True, 'training_started_by_bridge': 0,
        'install_transport': str(installed), 'preflight_transport': str(checked), 'launch_transport': str(launched)}
    write(REPORT / 'LAUNCHED.json', result, exclusive=True)
    return result


def status(expected):
    verify_local(expected)
    with context(): return recovery.status(expected)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('prepare','deploy','status'), required=True)
    p.add_argument('--contract-sha'); a = p.parse_args(argv)
    if a.mode == 'prepare': result = prepare()
    else:
        require(a.contract_sha, 'Pinned bridge SHA required')
        result = deploy(a.contract_sha) if a.mode == 'deploy' else {'transport': str(status(a.contract_sha))}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__': main()
