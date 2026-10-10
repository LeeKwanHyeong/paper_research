"""One-shot, fail-closed queue handoff without changing a running dispatcher.

The exclusive next-job claim is explicitly an administrative reservation, never a
scientific worker claim. Its expected FileExistsError is preserved, not relabeled
as parent completion. No process signals, remote calls, training or retries.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

ROOT = '/workspace/titantpp_contribution_history_recovery_20261010_v1_a100'
CONTRACT = 'da077e96e61255286c5cb35b54dc09ec8144e2d2f410e6eb46fea4681ee17899'
CONTROL = 'e1ab1ba7e3765fed14079430372e40a8f938bb547e4e191566d28ed4377e8d58'
POD = 'z278euk8d74nqt'
GPU = 'GPU-17e020b0-7bbe-d835-af19-881e511f9a2a'
PYTHON = '/opt/titantpp-history27-venv/bin/python'

def require(ok, message):
    if not ok:
        raise ValueError(message)

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def safe(root, name):
    path = root / name
    require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), 'Unsafe original path')
    return path

def verify_seals(root):
    require(str(root) == ROOT and not root.is_symlink(), 'Foreign root')
    c = read(root / 'execution_contract.json')
    require(canonical(c) == CONTRACT and c['recovery']['destination_pod_id'] == POD, 'Contract/Pod mismatch')
    spec = c['hosts']['a100']
    require(spec['root'] == ROOT and spec['gpu_uuid'] == GPU and spec['python'] == PYTHON, 'Host identity mismatch')
    require(sha(root / 'control/integrity_receipt.json') == CONTROL, 'Control seal mismatch')
    for base, files in [(root / 'source', {**c['source']['files'], **c['input_files']}), (root, c['operation']['files']), (root, read(root / 'control/integrity_receipt.json')['files'])]:
        for name, digest in files.items():
            require(sha(safe(base, name)) == digest, 'Sealed file mismatch: ' + name)
    require(read(root / 'start_permit.json')['contract_sha256'] == CONTRACT, 'Start permit mismatch')
    permit = read(root / 'training_permit.json')
    require(permit['contract_sha256'] == CONTRACT, 'Training permit mismatch')
    qualification = read(root / 'qualification/receipt.json')
    require(permit['qualification'] == qualification and sha(root / 'qualification/receipt.json') == permit['qualification_sha256'], 'Qualification binding changed')
    require(qualification['status'] == 'passed' and qualification['runtime']['gpu']['uuid'] == GPU, 'Native qualification identity mismatch')
    for key, value in spec['runtime_expected'].items():
        require(qualification['runtime'][key] == value, 'Qualified runtime changed')
    for name, digest in c['recovery']['parent_files'].items():
        require(sha(safe(root / 'recovery_parent', name)) == digest, 'Preserved parent changed')
    for name, key in [('parent_archive.tar.gz', 'parent_archive_sha256'), ('parent_manifest.json', 'parent_manifest_sha256')]:
        require(sha(root / 'recovery_parent' / name) == c['recovery'][key], 'Parent immutable source changed')
    return c

def groups(c):
    current = c['jobs'][12:15]
    waiting = c['jobs'][15:]
    require(len(current) == 3 and all((j['dataset'], j['seed']) == ('raf_spare_parts', 52) for j in current), 'Current batch changed')
    require(len(waiting) == 12 and waiting[0]['id'] == 'raf_spare_parts__62__titantpp_cnn_gru54', 'Handoff queue changed')
    require(len({j['id'] for j in current + waiting}) == 15, 'Duplicate ownership')
    return current, waiting

def never_started(root, waiting, barrier=None):
    for job in waiting:
        name = job['id']
        require('/' not in name and '..' not in name, 'Unsafe job')
        paths = [root / 'run' / name, root / 'logs' / (name + '.log'), root / 'leases' / (name + '.json'), root / 'worker_exits' / (name + '.json'), root / 'progress' / (name + '.json')]
        claim = root / 'claims' / (name + '.json')
        if claim != barrier:
            paths.append(claim)
        require(all(not p.exists() and not p.is_symlink() for p in paths), 'Transfer job already attempted: ' + name)

def inventory(root):
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name', '--format=csv,noheader'], text=True).strip()
    require(len(gpu.splitlines()) == 1 and gpu.split(',')[0].strip() == GPU, 'Physical GPU mismatch')
    gpu_processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True).strip()
    processes = []
    for line in subprocess.check_output(['ps', '-eo', 'pid=,ppid=,args='], text=True).splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3 and str(root) in parts[2] and int(parts[0]) != os.getpid():
            processes.append({'pid': int(parts[0]), 'ppid': int(parts[1]), 'argv': parts[2]})
    return {'observed_unix': time.time(), 'gpu': gpu, 'gpu_processes': gpu_processes, 'processes': processes}

def atomic_exclusive(path, payload):
    """Publish complete bytes atomically; a dispatcher winning the race is kept."""
    data = (json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    fd, temp = tempfile.mkstemp(prefix='.handoff-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)  # EEXIST is intentional; never overwrite a real claim.
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        os.unlink(temp)
    return hashlib.sha256(data).hexdigest()

def reserve_boundary(root, approval, *, probe=inventory, verifier=verify_seals):
    c = verifier(root)
    current, waiting = groups(c)
    require(isinstance(approval, str) and approval.strip(), 'Explicit user authorization required')
    observed = probe(root)
    status = read(root / 'status.json')
    require(status.get('status') == 'running' and -10 <= time.time() - status['updated_unix'] < 90, 'Fresh running supervisor required')
    active = status.get('active_jobs', {})
    current_ids = {j['id'] for j in current}
    require(bool(active) and set(active) <= current_ids, 'Current batch changed; do not reserve')
    require(any(p['pid'] == status['supervisor_pid'] and p['argv'].split() == [PYTHON, str(root / 'operation/campaign.py'), '--contract', str(root / 'execution_contract.json'), '--host', 'a100', '--mode', 'dispatch'] for p in observed['processes']), 'Exact live dispatcher missing')
    require(all(not (root / name).exists() for name in ('failure.json', 'pipeline_failure.json', 'pipeline_complete.json', 'supervisor_exit.json')), 'Owner already terminal')
    never_started(root, waiting)
    barrier = root / 'claims' / (waiting[0]['id'] + '.json')
    payload = {'schema': 'titantpp_planned_rtx_handoff_reservation_v1', 'kind': 'administrative_reservation_not_scientific_claim', 'root': str(root), 'pod_id': POD, 'gpu_uuid': GPU, 'contract_sha256': CONTRACT, 'python': PYTHON, 'runtime_expected': c['hosts']['a100']['runtime_expected'], 'user_approval': approval, 'created_unix': time.time(), 'current_batch': current, 'transfer_jobs': waiting, 'boundary_job': waiting[0], 'observation': observed, 'supervisor_status': status, 'expected_native_exit': 'FileExistsError at this exclusive claim after current batch terminal audit', 'scientific_worker_created': False, 'process_signals_sent': 0, 'parent_completion_modified': False}
    digest = atomic_exclusive(barrier, payload)
    never_started(root, waiting, barrier)
    return {'status': 'reserved_only_not_released', 'barrier_path': str(barrier), 'barrier_raw_sha256': digest, 'transferred_count': len(waiting), 'current_batch_count': len(current), 'payload': payload}

def confirm_release(root, barrier_sha256, *, probe=inventory, verifier=verify_seals):
    c = verifier(root)
    current, waiting = groups(c)
    barrier = root / 'claims' / (waiting[0]['id'] + '.json')
    require(sha(barrier) == barrier_sha256, 'Reservation changed')
    reservation = read(barrier)
    require(reservation['schema'] == 'titantpp_planned_rtx_handoff_reservation_v1' and reservation['contract_sha256'] == CONTRACT and reservation['transfer_jobs'] == waiting and reservation['current_batch'] == current and reservation['root'] == str(root) and reservation['pod_id'] == POD and reservation['gpu_uuid'] == GPU, 'Foreign reservation')
    never_started(root, waiting, barrier)
    observed = probe(root)
    require(not observed['processes'] and not observed['gpu_processes'], 'Actual owner/GPU absence required')
    exit_receipt = read(root / 'supervisor_exit.json')
    require(exit_receipt['actual_child_exit_observed'] is True and exit_receipt['returncode'] == 1 and exit_receipt['contract_sha256'] == CONTRACT and exit_receipt['supervisor_exception'] is None, 'Expected actual supervisor exit1 missing')
    failure = read(root / 'failure.json')
    require(failure['type'] == 'FileExistsError' and str(barrier) in failure['message'] and failure['active_jobs'] == [] and failure['owned_children_stopped'] == [] and failure['active_batch'] == waiting[:3], 'Unexpected failure; keep release blocked')
    pipeline = read(root / 'pipeline_failure.json')
    require(pipeline['phase'] == 'dispatch' and pipeline['contract_sha256'] == CONTRACT, 'Pipeline terminal identity mismatch')
    terminals = {}
    for job in c['jobs'][:15]:
        reused = job['id'] in c['recovery']['reused_jobs']
        folder = root / ('recovery_parent/run' if reused else 'run') / job['id']
        manifest = read(folder / 'terminal_manifest.json')
        expected = c['recovery']['parent_contract_sha256'] if reused else CONTRACT
        require(manifest['job'] == job and manifest['contract_sha256'] == expected and manifest['scientific_success'] is True and manifest['status'] == 'complete' and manifest['held_out_test_evaluated'] is False, 'Successful terminal missing')
        for name, digest in manifest['files'].items():
            require(sha(safe(folder, name)) == digest, 'Terminal original mismatch')
        terminals[job['id']] = sha(folder / 'terminal_manifest.json')
    return {'status': 'released_after_planned_operational_handoff', 'barrier_raw_sha256': barrier_sha256, 'transfer_jobs': waiting, 'completed_count': 15, 'scientific_failures': 0, 'operational_exit_preserved': 1, 'terminal_manifest_sha256': terminals, 'actual_idle_observation': observed, 'supervisor_exit_raw_sha256': sha(root / 'supervisor_exit.json'), 'failure_raw_sha256': sha(root / 'failure.json'), 'pipeline_failure_raw_sha256': sha(root / 'pipeline_failure.json'), 'parent_scientific_completion': False}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['reserve', 'confirm'])
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--approval')
    p.add_argument('--barrier-sha256')
    args = p.parse_args()
    result = reserve_boundary(args.root, args.approval) if args.mode == 'reserve' else confirm_release(args.root, args.barrier_sha256)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))

if __name__ == '__main__':
    main()
