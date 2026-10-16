"""Manual, campaign-bound recovery. Never invokes training or removes originals."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

PROJECT = Path(__file__).resolve().parents[2]
PARENT = PROJECT / 'search_artifacts/titantpp_cnn_gru_a100_seed42_20261004_v1'
BUNDLE = PROJECT / 'search_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1'
RECOVERY = BUNDLE / 'recovery'
spec = importlib.util.spec_from_file_location('owned_recovery_provider', PARENT / 'control/provider.py')
provider = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provider)


def write(name, value):
    path = RECOVERY / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def owned(a):
    return provider.owned(a)


def guarded_stop(a, reason):
    owned(a)
    provider.api('pod', 'stop', a['pod_id'])
    p = owned(a)
    write('stop_receipt.json', {'pod_id': a['pod_id'], 'reason': reason,
          'observed_unix': time.time(), 'desiredStatus': p.get('desiredStatus'),
          'runtimeStatus': p.get('runtimeStatus'), 'delete_called': False,
          'original_volume_integrity_verified': False})


def guard():
    a = provider.read(PARENT / 'control/allocation.json')
    intent = provider.read(RECOVERY / 'intent.json')
    write('guard_started.json', {'pid': os.getpid(), 'deadline_unix': intent['deadline_unix'],
          'starts_training': False, 'mac_sleep_may_delay_local_guard': True})
    while time.time() < intent['deadline_unix']:
        if (RECOVERY / 'stop_receipt.json').exists():
            return
        time.sleep(min(30, max(0, intent['deadline_unix'] - time.time())))
    try:
        guarded_stop(a, 'Bounded recovery deadline; retain originals')
    except Exception as error:
        write('guard_error.json', {'type': type(error).__name__, 'time': time.time(),
              'recovery_required': True})


def ssh(a, command, timeout=90):
    cfg = a['ssh']
    result = subprocess.run(['ssh', '-i', cfg['key'], '-p', str(cfg['port']),
        '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
        '-o', 'StrictHostKeyChecking=accept-new',
        '-o', 'UserKnownHostsFile=' + str(RECOVERY / 'known_hosts'),
        'root@' + cfg['ip'], command], capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError('Recovery SSH failed; raw output suppressed')
    return result.stdout


def main():
    a = provider.read(PARENT / 'control/allocation.json')
    approval = provider.read(BUNDLE / 'approval.json')
    assert approval['approved'] is True and approval['owned_pod'] == a['pod_id']
    assert not (RECOVERY / 'start_intent.json').exists(), 'Manual recovery start already attempted; inspect saved evidence'
    before = owned(a)
    assert before['desiredStatus'] == 'EXITED' and before['runtimeStatus'] == 'stopped'
    assert not before.get('dockerArgs'), 'Recovery must not autostart campaign bootstrap'
    now = time.time()
    deadline = min(a['deadline_unix'], now + approval['provider_recovery_window_seconds'])
    if RECOVERY.name == 'ssh_info_attempt2':
        first = provider.read(BUNDLE / 'recovery/error.json')
        stopped = provider.read(BUNDLE / 'recovery/stop_receipt.json')
        assert first['message'] == 'Recovery container unavailable within bounded readiness window'
        assert stopped['desiredStatus'] == 'EXITED' and stopped['runtimeStatus'] == 'stopped'
        deadline = min(deadline, provider.read(BUNDLE / 'recovery/intent.json')['deadline_unix'])
    assert deadline > now + 120, 'Insufficient approved recovery time'
    write('intent.json', {'pod_id': a['pod_id'], 'created_unix': now, 'deadline_unix': deadline,
          'gpu_rate_usd_hour': a['gpu_rate_usd_hour'], 'max_recovery_running_seconds': deadline - now,
          'starts_training': False, 'original_parent_deadline_unmodified': True})
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--guard'] +
            (['--ssh-info-attempt2'] if RECOVERY.name == 'ssh_info_attempt2' else []),
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
    write('start_intent.json', {'pod_id': a['pod_id'], 'guard_pid': proc.pid, 'time': time.time()})
    started = False
    try:
        provider.api('pod', 'start', a['pod_id'])
        started = True
        write('start_receipt.json', {'pod_id': a['pod_id'], 'time': time.time(),
              'starts_training': False, 'guardian_pid': proc.pid})
        print(json.dumps({'phase': 'recovery_started', 'pod_id': a['pod_id']}), flush=True)
        connection = None
        for delay in (8, 15, 25, 30, 30):
            time.sleep(delay)
            current = owned(a)
            write('provider_readback.json', {k: current.get(k) for k in
                  ('id', 'name', 'desiredStatus', 'runtimeStatus', 'publicIp', 'portMappings')})
            if current.get('runtimeStatus') == 'running':
                info = provider.api('ssh', 'info', a['pod_id'])
                if info.get('ip') and info.get('port'):
                    assert info['ssh_key']['path'] == a['ssh']['key'], 'SSH key changed'
                    connection = info
                    break
        assert connection is not None, 'Recovery container unavailable within bounded readiness window'
        a = dict(a, ssh=dict(a['ssh'], ip=connection['ip'], port=connection['port']))
        write('connection_receipt.json', {'pod_id': a['pod_id'], 'ssh': a['ssh'], 'time': time.time()})
        code = (PARENT / 'control/terminal_observer.py').read_text()
        snapshot = json.loads(ssh(a, 'python3 -c ' + shlex.quote(code) + ' --root ' + shlex.quote(a['remote_root'])))
        assert snapshot['root'] == a['remote_root']
        assert snapshot['compute']['returncode'] == 0 and not snapshot['compute']['stdout'].strip(), 'GPU process active during recovery'
        workers = [line for line in snapshot.get('owned_processes', []) if any(token in line for token in
                   ('--mode fit', '/worker/dispatch.py', '/worker/qualify.py', '/control/training_pipeline.py'))]
        assert not workers, 'Owned training process active during recovery'
        write('recovered_snapshot.json', snapshot)
        print(json.dumps({'phase': 'originals_accessible', 'runs': len(snapshot['runs'])}), flush=True)
        # Archive all original payloads after confirming the stopped campaign did not restart.
        code = '''import json,hashlib,tarfile
from pathlib import Path
b=Path(%r)
package=json.loads((b/'package_manifest.json').read_text())
files={}
for folder in ('run','source','data','baselines','references','worker','qualification','claims','leases','progress','logs','control'):
 for p in sorted((b/folder).rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts:
   assert not p.is_symlink()
   if folder=='run':assert not p.name.startswith(('test_','held_out_'))
   files[str(p.relative_to(b))]=hashlib.sha256(p.read_bytes()).hexdigest()
for p in sorted(b.iterdir()):
 if p.is_file() and p.name not in ('retrieval_manifest.json','retrieval.tar.gz','manual_recovery_manifest.json','manual_recovery.tar.gz'):
  assert not p.is_symlink()
  files[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
assert set(package['files']).issubset(files)
(b/'manual_recovery_manifest.json').write_text(json.dumps({'files':files,'complete_originals':True}))
with tarfile.open(b/'manual_recovery.tar.gz','w:gz') as t:
 for name in [*files,'manual_recovery_manifest.json']:t.add(b/name,arcname=name,recursive=False)
print(json.dumps({'archive_sha256':hashlib.sha256((b/'manual_recovery.tar.gz').read_bytes()).hexdigest(),'files':len(files)}))
''' % a['remote_root']
        remote = json.loads(ssh(a, 'python3 -c ' + shlex.quote(code), timeout=240))
        target = RECOVERY
        part = target / 'original.tar.gz.part'
        cfg = a['ssh']
        result = subprocess.run(['scp', '-i', cfg['key'], '-P', str(cfg['port']), '-o', 'BatchMode=yes',
            '-o', 'StrictHostKeyChecking=accept-new', '-o', 'UserKnownHostsFile=' + str(target / 'known_hosts'),
            'root@' + cfg['ip'] + ':' + a['remote_root'] + '/manual_recovery.tar.gz', str(part)],
            capture_output=True, timeout=360)
        assert result.returncode == 0, 'Recovery transfer failed; raw output suppressed'
        assert provider.sha(part) == remote['archive_sha256'], 'Recovery archive SHA mismatch'
        archive = target / 'original.tar.gz'
        os.replace(part, archive)
        original = target / 'original'
        assert not original.exists(), 'Never overwrite recovered originals'
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
            assert len({m.name for m in members}) == len(members)
            tar.extractall(original, filter='data')
        manifest = provider.read(original / 'manual_recovery_manifest.json')
        for name, expected in manifest['files'].items():
            assert provider.sha(original / name) == expected, 'Original SHA mismatch: ' + name
        write('retrieval_receipt.json', {'status': 'sha_verified', 'pod_id': a['pod_id'],
              'remote_root': a['remote_root'], 'complete_originals': True, 'files': len(manifest['files']),
              'archive_sha256': remote['archive_sha256'], 'manifest_files_sha256': provider.canonical(manifest['files']),
              'observed_unix': time.time(), 'binary_cpu_inference_audit': 'not_performed'})
        print(json.dumps({'phase': 'originals_sha_verified', 'files': len(manifest['files'])}), flush=True)
    except Exception as error:
        write('error.json', {'type': type(error).__name__, 'message': str(error), 'time': time.time(),
              'automatic_retry': False, 'originals_must_be_preserved': True})
        raise
    finally:
        if started:
            guarded_stop(a, 'Manual recovery finished; preserve originals pending final verification')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--guard', action='store_true')
    parser.add_argument('--ssh-info-attempt2', action='store_true', help='One manual repair after verified failed readiness and stop; preserve first evidence and original deadline')
    args = parser.parse_args()
    if args.ssh_info_attempt2:
        RECOVERY = BUNDLE / 'recovery/ssh_info_attempt2'
    guard() if args.guard else main()
