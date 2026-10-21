"""Prepare a separate Test-only controller after the preserved MAC terminal race."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import time

from paper.scripts import titans_mac_accuracy9_terminal_runtime as native

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/titans_mac_accuracy9_recovery_20261021_v1'
REPORT = BASE / 'execution_v5'
BUNDLE = ROOT / 'search_artifacts/titans_mac_accuracy9_recovery_20261021_v5'
UPSTREAM_BUNDLE = ROOT / 'search_artifacts/titans_mac_accuracy9_recovery_20261021_v4'
ORIGINAL_BUNDLE = ROOT / 'search_artifacts/titans_mac_accuracy9_20261020_v1'
REVIEW = BASE / 'execution_v4/terminal_failure_review_v1'
DIAGNOSIS = REVIEW / 'diagnosis/native/titans_mac_accuracy9_recovery_20261021_v4_5090'
UPSTREAM_BINDING = BASE / 'execution_v4/CONTROL_BINDINGS.json'
UPSTREAM_BINDING_SHA = 'f255070c3d55089d6209a690d0a1c1791afd4ec33ed9760f831cb2d47ed7db01'
REMOTE = '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titans_mac_accuracy9_recovery_20261021_v5_5090'
SOURCE = ('paper/scripts/manage_titans_mac_accuracy9_terminal_finish.py',
          'paper/scripts/collect_titans_mac_accuracy9_terminal_finish.py',
          'paper/scripts/titans_mac_accuracy9_terminal_runtime.py',
          'paper/scripts/titans_mac_accuracy9_terminal_identity.py')
TESTS = ('paper/tests/test_manage_titans_mac_accuracy9_terminal_finish.py',
         'paper/tests/test_collect_titans_mac_accuracy9_terminal_finish.py',
         'paper/tests/test_titans_mac_accuracy9_terminal_runtime.py',
         'paper/tests/test_titans_mac_accuracy9_terminal_identity.py')
EVIDENCE = ('dispatcher_launch.json', 'dispatcher_exit.json', 'dispatch_failure.json',
            'validation_stage_receipt.json', 'final_receipt.json', 'supervisor_worker_cleanup.json')
sha, canonical, read, require, write = native.sha_file, native.sha_json, native.read, native.require, native.write


def original_local_checks():
    require(sha(UPSTREAM_BINDING) == UPSTREAM_BINDING_SHA, 'Preserved v4 binding changed')
    binding = read(UPSTREAM_BINDING)
    for name, digest in binding['source_files'].items():
        require(sha(ROOT / name) == digest, 'Preserved v4 operating source changed: ' + name)
    c = read(UPSTREAM_BUNDLE / 'recovery_contract.json')
    require(canonical(c) == native.UPSTREAM_SHA == binding['contract_sha256']
            and sha(UPSTREAM_BUNDLE / 'recovery_contract.json') == binding['contract_raw_sha256'],
            'Preserved scientific recovery contract changed')
    original = read(ORIGINAL_BUNDLE / 'execution_contract.json')
    require(sha(ORIGINAL_BUNDLE / 'execution_contract.json') == c['scientific_contract']['raw_sha256']
            and canonical(original) == c['scientific_contract']['canonical_sha256']
            and original['source']['files'] == c['source']['files']
            and original['input_files'] == c['input_files']
            and len(c['source']['files']) == 989 and len(c['input_files']) == 9,
            'Original source/input contract changed')
    require(canonical(c['source']['files']) == c['source']['files_sha256'], 'Frozen source inventory changed')
    for name, digest in c['source']['files'].items():
        require(sha(ROOT / name) == digest, 'Frozen scientific source changed: ' + name)
    for name, digest in c['input_files'].items():
        require(sha(ORIGINAL_BUNDLE / 'input_snapshot' / name) == digest, 'Frozen input changed: ' + name)
    for key, filename in (('science_spec_sha256', 'SCIENCE_SPEC.json'),
                          ('frozen_conditions_sha256', 'FROZEN_CONDITIONS.json')):
        require(sha(ROOT / 'reports/titans_mac_accuracy9_20261020_v1' / filename) == original[key],
                'Frozen scientific conditions changed')
    return c


def reviewed_sources():
    qa = read(REPORT / 'INDEPENDENT_QA.json')
    require(qa.get('status') == 'PASS', 'Independent terminal repair review must pass')
    files = qa['files_reviewed_sha256']
    require(set(SOURCE + TESTS) <= set(files), 'Independent review omitted terminal files')
    for name, digest in files.items():
        require(sha(ROOT / name) == digest, 'Reviewed terminal file changed: ' + name)
    tracked = subprocess.check_output(['git', 'ls-files', '--', *SOURCE, *TESTS], cwd=ROOT, text=True).splitlines()
    dirty = subprocess.check_output(['git', 'diff', 'HEAD', '--name-only', '--', *SOURCE, *TESTS], cwd=ROOT, text=True)
    require(set(tracked) == set(SOURCE + TESTS) and not dirty.strip(), 'Commit reviewed operating files before preparation')
    return files


def prepare():
    require(not BUNDLE.exists() and not (REPORT / 'CONTROL_BINDINGS.json').exists(), 'Existing finish seal; preserve attempt')
    c = original_local_checks()
    files = reviewed_sources()
    auth = read(REVIEW / 'NEW_AUTHORIZATION.json')
    require(auth['user_instruction'] == '오류 수정해서 진행하기로 한거 전부 진행하자'
            and auth['new_scientific_training_authorized'] is False, 'Missing current repair authorization')
    evidence = {name: sha(DIAGNOSIS / name) for name in EVIDENCE}
    f = {'schema': native.SCHEMA, 'root': REMOTE, 'upstream_root': c['root'],
         'upstream_contract_path': c['root'] + '/recovery_contract.json',
         'upstream_contract_sha256': canonical(c),
         'upstream_contract_raw_sha256': sha(UPSTREAM_BUNDLE / 'recovery_contract.json'),
         'upstream_evidence_files': evidence,
         **{k: deepcopy(c[k]) for k in ('python', 'boot_id', 'gpu_uuid', 'limits', 'absolute_host_deadline_unix')},
         'queue_started_at': read(DIAGNOSIS / 'dispatcher_exit.json')['started_at'],
         'operation_files': {'source/' + Path(name).name: sha(ROOT / name) for name in SOURCE[-2:]},
         'local_control_files': files, 'independent_qa_sha256': sha(REPORT / 'INDEPENDENT_QA.json'),
         'authorization': {'instruction': auth['user_instruction'], 'path': REMOTE + '/AUTHORIZATION.json',
                           'sha256': sha(REVIEW / 'NEW_AUTHORIZATION.json')},
         'training_worker_count': 0, 'validation_worker_count': 0, 'test_worker_count': 9,
         'tmux_binary': c['tmux_binary'], 'tmux_session': 'titans_mac9_terminal_20261021_v5',
         'operation_git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    BUNDLE.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(REVIEW / 'NEW_AUTHORIZATION.json', BUNDLE / 'AUTHORIZATION.json')
    write(BUNDLE / 'finish_contract.json', f)
    with (BUNDLE / 'deployment.tar.gz').open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as archive:
        for name in ('finish_contract.json', 'AUTHORIZATION.json'):
            archive.add(BUNDLE / name, arcname=name, recursive=False)
        for name in SOURCE[-2:]:
            archive.add(ROOT / name, arcname='source/' + Path(name).name, recursive=False)
    value = {'schema': native.SCHEMA, 'status': 'sealed', 'created_at': native.now(),
             'contract_sha256': canonical(f), 'contract_raw_sha256': sha(BUNDLE / 'finish_contract.json'),
             'deployment_archive_sha256': sha(BUNDLE / 'deployment.tar.gz'),
             'authorization_sha256': sha(BUNDLE / 'AUTHORIZATION.json'),
             'upstream_binding_sha256': UPSTREAM_BINDING_SHA, 'source_files': files,
             'independent_qa_sha256': sha(REPORT / 'INDEPENDENT_QA.json'),
             'operation_git_revision': f['operation_git_revision'], 'new_training': 0,
             'new_validation': 0, 'new_fixed_test': 9, 'automatic_scientific_retry': False}
    write(REPORT / 'CONTROL_BINDINGS.json', value)
    write(REPORT / 'PREPARED.json', {'status': 'prepared_test_only', 'contract_sha256': canonical(f),
                                   'binding_sha256': sha(REPORT / 'CONTROL_BINDINGS.json')})
    return read(REPORT / 'PREPARED.json')


def verify_local(expected):
    c = original_local_checks()
    f, b = read(BUNDLE / 'finish_contract.json'), read(REPORT / 'CONTROL_BINDINGS.json')
    require(canonical(f) == expected == b['contract_sha256']
            and sha(BUNDLE / 'finish_contract.json') == b['contract_raw_sha256']
            and f['upstream_contract_sha256'] == canonical(c)
            and f['upstream_root'] == c['root'] and f['root'] == REMOTE,
            'Sealed terminal scope changed')
    require(sha(BUNDLE / 'deployment.tar.gz') == b['deployment_archive_sha256']
            and sha(BUNDLE / 'AUTHORIZATION.json') == b['authorization_sha256'] == f['authorization']['sha256']
            and sha(REPORT / 'INDEPENDENT_QA.json') == b['independent_qa_sha256'], 'Terminal seal changed')
    require(b['source_files'], 'Empty terminal source inventory')
    require(b['source_files'] == f['local_control_files']
            and set(SOURCE + TESTS) <= set(b['source_files'])
            and b['independent_qa_sha256'] == f['independent_qa_sha256'], 'Local control closure changed')
    for name, digest in b['source_files'].items():
        require(sha(ROOT / name) == digest, 'Terminal source changed: ' + name)
    for name, digest in f['operation_files'].items():
        require(sha(ROOT / 'paper/scripts' / Path(name).name) == digest, 'Native finish operation changed')
    for name, digest in f['upstream_evidence_files'].items():
        require(sha(DIAGNOSIS / name) == digest, 'Preserved raw failure evidence changed')
    return f, REPORT, BUNDLE


def transport(label, command, *, data=None, timeout=45):
    folder = REPORT / 'transport' / (time.strftime('%Y%m%dT%H%M%S', time.gmtime()) + '_' + str(time.time_ns()) + '_' + label)
    folder.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    with (folder / 'stdout').open('xb') as out, (folder / 'stderr').open('xb') as err:
        process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
            '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2', '5090', command],
            stdin=subprocess.PIPE if data is not None else subprocess.DEVNULL, stdout=out, stderr=err)
        timed_out = False
        try:
            process.communicate(input=data, timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.communicate()
    result = {'actual_os_returncode': process.returncode, 'timeout': timed_out,
              'elapsed_seconds': time.monotonic() - began, 'stdout_sha256': sha(folder / 'stdout'),
              'stderr_sha256': sha(folder / 'stderr'), 'stdout_bytes': (folder / 'stdout').stat().st_size,
              'stderr_bytes': (folder / 'stderr').stat().st_size, 'command': command}
    write(folder / 'TRANSPORT.json', result)
    require(process.returncode == 0 and not timed_out, 'Transport failure; preserved at ' + str(folder))
    return folder


def invocation(f, mode):
    c = read(UPSTREAM_BUNDLE / 'recovery_contract.json')
    environment = {**c['environment'], 'PYTHONDONTWRITEBYTECODE': '1',
        'MPLCONFIGDIR': f['root'] + '/cache/matplotlib', 'XDG_CACHE_HOME': f['root'] + '/cache',
        'TORCHINDUCTOR_CACHE_DIR': f['root'] + '/cache/torchinductor'}
    return 'cd ' + shlex.quote(c['source_root']) + ' && ' + shlex.join(['env',
        *[k + '=' + v for k, v in environment.items()], f['python'], '-u', f['root'] + '/source/titans_mac_accuracy9_terminal_runtime.py',
        '--contract', f['root'] + '/finish_contract.json', '--contract-sha', canonical(f), '--mode', mode])


def deploy(expected):
    f, _, _ = verify_local(expected)
    write(REPORT / 'DEPLOYMENT_ATTEMPT.json', {'at': native.now(), 'contract_sha256': expected,
                                             'new_training': 0, 'new_validation': 0, 'automatic_retry': False})
    program = '''import hashlib,io,json,sys,tarfile
from pathlib import Path
b=sys.stdin.buffer.read();assert hashlib.sha256(b).hexdigest()==__ARCHIVE__
t=tarfile.open(fileobj=io.BytesIO(b),mode='r:gz');members=t.getmembers()
assert len(members)==4 and all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
f=json.load(t.extractfile('finish_contract.json'))
assert hashlib.sha256(json.dumps(f,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()==__EXPECTED__
root=Path(f['root']);assert not root.exists(),'Existing finish attempt'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as stream:
  for block in iter(lambda:stream.read(4194304),b''):h.update(block)
 return h.hexdigest()
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==f['boot_id']
old=Path(f['upstream_root']);assert sha(f['upstream_contract_path'])==f['upstream_contract_raw_sha256']
for name,digest in f['upstream_evidence_files'].items():assert sha(old/name)==digest,name
root.mkdir();t.extractall(root,filter='data')
for name,digest in f['operation_files'].items():assert sha(root/name)==digest,name
assert sha(f['authorization']['path'])==f['authorization']['sha256']
print(json.dumps({'status':'installed_test_only','root':str(root),'new_training':0,'new_validation':0}))
'''.replace('__ARCHIVE__', repr(sha(BUNDLE / 'deployment.tar.gz'))).replace('__EXPECTED__', repr(expected))
    installed = transport('install', shlex.join(['/usr/bin/python3', '-c', program]),
                          data=(BUNDLE / 'deployment.tar.gz').read_bytes(), timeout=120)
    checked = transport('preflight', invocation(f, 'preflight'), timeout=120)
    shell = invocation(f, 'supervise') + ' > ' + shlex.quote(f['root'] + '/supervisor.stdout.log') + ' 2> ' + shlex.quote(f['root'] + '/supervisor.stderr.log')
    launched = transport('launch', shlex.join([f['tmux_binary'], 'new-session', '-d', '-s', f['tmux_session'], shell]))
    result = {'status': 'test_only_launch_requested', 'at': native.now(), 'contract_sha256': expected,
              'new_training': 0, 'new_validation': 0, 'test_count': 9,
              'install_transport': str(installed), 'preflight_transport': str(checked), 'launch_transport': str(launched)}
    write(REPORT / 'LAUNCHED.json', result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('prepare', 'check-local', 'deploy'), required=True)
    p.add_argument('--contract-sha')
    a = p.parse_args(argv)
    if a.mode == 'prepare':
        result = prepare()
    else:
        require(a.contract_sha, 'Pinned finish contract SHA required')
        result = deploy(a.contract_sha) if a.mode == 'deploy' else {'status': 'passed_local',
            'contract_sha256': canonical(verify_local(a.contract_sha)[0]), 'remote_calls': 0}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
