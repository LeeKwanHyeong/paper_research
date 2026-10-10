"""One-attempt RTX deployment, bound to the approved unstarted history12.

Existing servers, runtimes, authentication and A100 originals are untouched.
Native qualification and concurrent admission belong to the sealed pipeline.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

PROJECT = Path(__file__).resolve().parents[2]
B = PROJECT / 'search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1'
C_SHA = '518467643d248ba7384482cdb9c8806877392be0eaa81b961992cfa1a0381d6b'
PARENT_SHA = 'da077e96e61255286c5cb35b54dc09ec8144e2d2f410e6eb46fea4681ee17899'
RESERVATION = PROJECT / 'reports/titantpp_history_rtx_handoff_20261010_v1/a100_reservation_stdout.json'
PARENT = PROJECT / 'search_artifacts/titantpp_contribution_history_recovery_20261010_v1'
POD = 'z278euk8d74nqt'
GPU = 'GPU-17e020b0-7bbe-d835-af19-881e511f9a2a'
REMOTE_PARENT = '/workspace/titantpp_contribution_history_recovery_20261010_v1_a100'


def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''): h.update(block)
    return h.hexdigest()
def canonical(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()
def require(ok, message):
    if not ok: raise ValueError(message)
def write(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f: json.dump(v, f, ensure_ascii=False, indent=2, allow_nan=False)


def validate_local(host):
    c = read(B / 'execution_contract.json')
    require(canonical(c) == C_SHA and host in ('5080', '5090'), 'Foreign contract/host')
    seal = read(B / 'control/local_tools_integrity.json')
    require(seal['contract_sha256'] == C_SHA, 'Local control contract binding')
    for name, digest in seal['files'].items():
        p = (B / name).resolve()
        require(p.is_relative_to(B.resolve()) and sha(p) == digest, 'Local seal drift: ' + name)
    for name, digest in seal.get('project_files', {}).items():
        path = (PROJECT / name).resolve()
        require(path.is_relative_to(PROJECT.resolve()) and sha(path) == digest,
            'Project tool drift: ' + name)
    for name, digest in {**c['source']['files'], **c['input_files']}.items():
        require(sha(B / 'source' / name) == digest, 'Frozen source/input drift: ' + name)
    for name, digest in c['operation']['files'].items():
        require(sha(B / name) == digest, 'Operation drift: ' + name)
    return c


def bind_release(proof_path):
    c = read(B / 'execution_contract.json'); proof = read(proof_path)
    require(canonical(c) == C_SHA, 'Contract drift')
    reservation = read(RESERVATION); r = reservation['payload']
    parent = read(PARENT / 'execution_contract.json')
    require(canonical(parent) == PARENT_SHA and r['contract_sha256'] == PARENT_SHA
        and r['root'] == REMOTE_PARENT and r['pod_id'] == POD and r['gpu_uuid'] == GPU
        and reservation['status'] == 'reserved_only_not_released', 'Reservation identity drift')
    require(r['transfer_jobs'] == parent['jobs'][15:] and proof['transfer_jobs'] == r['transfer_jobs'],
        'Transferred scientific job drift')
    require(proof['barrier_raw_sha256'] == reservation['barrier_raw_sha256'], 'Reservation raw binding drift')
    require(len(proof['terminal_manifest_sha256']) == 15
        and set(proof['terminal_manifest_sha256']) == {j['id'] for j in parent['jobs'][:15]},
        'Exact fifteen terminal originals missing')
    hashes = list(proof['terminal_manifest_sha256'].values()) + [proof[k] for k in (
        'barrier_raw_sha256', 'supervisor_exit_raw_sha256', 'failure_raw_sha256', 'pipeline_failure_raw_sha256')]
    require(all(isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v) for v in hashes),
        'Original SHA binding missing')
    require(proof['status'] == 'released_after_planned_operational_handoff'
        and proof['completed_count'] == 15 and proof['scientific_failures'] == 0
        and proof['operational_exit_preserved'] == 1, 'A100 queue release incomplete')
    expected = [j['id'] for j in c['jobs']]
    require([j['id'] for j in proof['transfer_jobs']] == expected, 'Transferred identity drift')
    idle = proof['actual_idle_observation']
    require(idle['gpu'].split(',')[0].strip() == GPU, 'Foreign A100 GPU proof')
    require(not idle['gpu_processes'] and not idle['processes'], 'A100 owner remains')
    source = B / 'handoff/ownership_release_source.json'
    source.parent.mkdir(parents=True, exist_ok=True)
    with source.open('xb') as f: f.write(Path(proof_path).read_bytes())
    release = {'contract_sha256': C_SHA, 'parent_contract_sha256': PARENT_SHA,
        'execution_ids': expected, 'A100_queue_exclusion_verified': True,
        'unstarted_verified': True, 'released_to_rtx': True,
        'source_receipt_raw_sha256': sha(source), 'reservation_raw_sha256': sha(RESERVATION),
        'barrier_raw_sha256': reservation['barrier_raw_sha256'],
        'A100_pod_id': POD, 'A100_gpu_uuid': GPU, 'released_unix': time.time()}
    write(B / 'ownership_release.json', release)
    return release


def remote(host, code, timeout=180):
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
        host, 'python3 -'], input=code.encode(), capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError('Native remote operation failed (no retry): ' +
            result.stderr.decode(errors='replace')[-2000:])
    return result.stdout


def copy_to(host, path, destination):
    result = subprocess.run(['scp', '-o', 'BatchMode=yes', str(path),
        host + ':' + destination], capture_output=True, timeout=180)
    require(result.returncode == 0, 'SCP failed; preserve intent, no automatic retry')


def deploy(host):
    c = validate_local(host); spec = c['hosts'][host]; root = spec['root']
    require((B / 'ownership_release.json').exists(), 'A100 release missing')
    release = read(B / 'ownership_release.json')
    require(release['contract_sha256'] == C_SHA and release['released_to_rtx'] is True,
        'Foreign A100 release')
    proof = B / 'handoff/ownership_release_source.json'
    require(sha(proof) == release['source_receipt_raw_sha256'], 'Release proof drift')
    health = read(B / 'deployment' / ('health_' + host + '.json'))
    require(health['host'] == host and health['gpu']['rc'] == 0
        and health['gpu_pids']['rc'] == 0 and not health['gpu_pids']['stdout'].strip()
        and not health['reboot_required'], 'Native GPU health/idle not verified')
    runtime = json.loads(health['runtime']['stdout'])
    require(all(runtime[k] == v for k, v in spec['runtime_expected'].items()), 'Runtime health drift')
    require(spec['gpu_uuid'] in health['gpu']['stdout'], 'GPU health UUID drift')
    package = B / 'deployment' / ('package_' + host + '.tar.gz')
    manifest = read(B / 'deployment' / ('package_manifest_' + host + '.json'))
    require(sha(package) == manifest['archive_raw_sha256'], 'Package drift')
    start = time.time()
    permit = {'host': host, 'contract_sha256': C_SHA,
        'approval_sha256': canonical(read(B / 'approval.json')),
        'started_at_unix': start, 'deadline_unix': start + c['limits']['total_wall_seconds'],
        'per_condition_seconds': c['limits']['per_condition_seconds'],
        'ownership_release_raw_sha256': sha(B / 'ownership_release.json'),
        'automatic_retry': False}
    permit_path = B / ('start_permit_' + host + '.json'); write(permit_path, permit)
    intent = B / 'deployment' / ('deploy_intent_' + host + '.json')
    write(intent, {'host': host, 'root': root, 'contract_sha256': C_SHA,
        'package_sha256': sha(package), 'created_unix': start, 'automatic_retry': False})
    # An existing root is preserved. A failed intent is never replayed.
    remote(host, 'from pathlib import Path\np=Path(' + repr(root) +
        ')\np.mkdir(parents=False,exist_ok=False)\n(p/"deployment").mkdir()\n(p/"handoff").mkdir()\n')
    targets = [(package, root + '/deployment/package.tar.gz'),
        (B / 'ownership_release.json', root + '/ownership_release.json'),
        (proof, root + '/handoff/ownership_release_source.json'),
        (permit_path, root + '/start_permit.json')]
    for source, destination in targets: copy_to(host, source, destination)
    code = '''import hashlib,json,tarfile,time
from pathlib import Path
root=Path(ROOT)
p=root/'deployment/package.tar.gz'
assert hashlib.sha256(p.read_bytes()).hexdigest()==ARCHIVE
with tarfile.open(p,'r:gz') as t:
    members=t.getmembers()
    assert {m.name for m in members}==set(FILES)
    for m in members:
        assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
        q=root/m.name;q.parent.mkdir(parents=True,exist_ok=True)
        data=t.extractfile(m).read()
        assert hashlib.sha256(data).hexdigest()==FILES[m.name]
        with q.open('xb') as f:f.write(data)
for name in ('cache','logs'):(root/name).mkdir(exist_ok=True)
receipt={'host':HOST,'contract_sha256':CONTRACT,'archive_sha256':ARCHIVE,'file_count':len(FILES),'deployed_unix':time.time(),'new_root_only':True}
with (root/'deployment/deploy_receipt.json').open('x') as f:json.dump(receipt,f)
print(json.dumps(receipt))
'''
    code = 'ROOT=' + repr(root) + '\nARCHIVE=' + repr(sha(package)) + '\nFILES=' + repr(manifest['files']) + '\nHOST=' + repr(host) + '\nCONTRACT=' + repr(C_SHA) + '\n' + code
    result = json.loads(remote(host, code))
    write(B / 'deployment' / ('deploy_receipt_' + host + '.json'), result)
    return result


def launch(host):
    c = validate_local(host); spec = c['hosts'][host]; root = spec['root']
    require(read(B / 'deployment' / ('deploy_receipt_' + host + '.json'))['contract_sha256'] == C_SHA,
        'Successful deployment missing')
    control_sha = sha(B / 'control/hosts' / host / 'integrity_receipt.json')
    write(B / 'deployment' / ('launch_intent_' + host + '.json'),
        {'host': host, 'root': root, 'contract_sha256': C_SHA,
         'integrity_raw_sha256': control_sha, 'created_unix': time.time(), 'automatic_retry': False})
    argv = [spec['python'], root + '/control/pipeline.py', '--root', root,
        '--contract-sha256', C_SHA, '--integrity-sha256', control_sha]
    command = ('cd ' + shlex.quote(spec['source_root']) + ' && exec env ' + ' '.join(
        shlex.quote(k + '=' + v) for k, v in spec['environment'].items()) + ' ' + shlex.join(argv) +
        ' > ' + shlex.quote(root + '/logs/pipeline.log') + ' 2>&1')
    code = '''import json,subprocess,time
from pathlib import Path
root=Path(ROOT)
assert not (root/'pipeline.claim').exists()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
assert gpu==UUID
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert subprocess.run([TMUX,'has-session','-t',SESSION],capture_output=True).returncode!=0
subprocess.run([TMUX,'new-session','-d','-s',SESSION,COMMAND],check=True)
r={'host':HOST,'root':ROOT,'contract_sha256':CONTRACT,'gpu_uuid':UUID,'actual_launch':True,'tmux':SESSION,'launched_unix':time.time(),'native_pipeline':'qualification_then_concurrency_then_dispatch','qualification_is_fit':False,'automatic_retry':False}
with (root/'deployment/launch_receipt.json').open('x') as f:json.dump(r,f)
print(json.dumps(r))
'''
    prefix = {'ROOT': root, 'UUID': spec['gpu_uuid'], 'TMUX': spec['tmux_binary'],
        'SESSION': spec['tmux'], 'COMMAND': command, 'HOST': host, 'CONTRACT': C_SHA}
    result = json.loads(remote(host, '\n'.join(k + '=' + repr(v) for k, v in prefix.items()) + '\n' + code))
    write(B / 'deployment' / ('launch_receipt_' + host + '.json'), result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['bind-release', 'deploy', 'launch'])
    p.add_argument('--host', choices=['5080', '5090']); p.add_argument('--release-receipt', type=Path)
    a = p.parse_args()
    result = bind_release(a.release_receipt) if a.mode == 'bind-release' else (
        deploy(a.host) if a.mode == 'deploy' else launch(a.host))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()
