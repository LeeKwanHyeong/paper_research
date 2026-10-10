"""Read-only observation of the two approved RTX handoff owners, once per host.

A100 and Instacart are local evidence only. This tool never launches, retries,
resumes, evaluates a model, contacts a Pod API, or opens held-out predictions.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import inspect
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
BUNDLE = PROJECT / 'search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1'
A100 = PROJECT / 'search_artifacts/titantpp_contribution_history_recovery_20261010_v1'
INSTACART = PROJECT / 'search_artifacts/titantpp_instacart_completion_recovery_20261010_v1'
CONTRACT = '518467643d248ba7384482cdb9c8806877392be0eaa81b961992cfa1a0381d6b'
SOURCE = 'c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a'
OPERATION = '8a7ff350488fe8b03289ea6e6683a41ace4c95c2bdf0d24f2c06f3d78f5c83da'
INTEGRITY = '1fce3395f77f05af3b79fe54e39fddf593d555d18145bea5c237c27deae5caa9'
RELEASE_SOURCE = 'e7d6fbaf59b1caa7763d2662ac1ed29d65f003f6846f8bc3aaff96fa6d3a8ea1'
INSTACART_RECEIPT = '459ae9f15f6f4247584a8d47bd8e3901ab6e7ba91d6a98df3f64b09dde95cea6'


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
    require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), 'Unsafe artifact path')
    return path


def write(path, value, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as stream:
            stream.write(data)
    else:
        temp = path.with_name(path.name + '.' + str(time.time_ns()) + '.tmp')
        temp.write_text(data)
        temp.replace(path)


def kst(value):
    return datetime.fromtimestamp(value, timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST')


def verify_local(bundle=BUNDLE):
    c = read(bundle / 'execution_contract.json')
    cur = read(bundle / 'current.json')
    require(canonical(c) == CONTRACT and cur['canonical_sha256'] == CONTRACT, 'Contract pointer mismatch; remote0')
    require(cur['hosts'] == {h: s['root'] for h, s in c['hosts'].items()}, 'Root pointer mismatch; remote0')
    require(canonical(c['source']['files']) == SOURCE and canonical(c['operation']['files']) == OPERATION, 'Frozen closure mismatch; remote0')
    require(sha(bundle / 'control/integrity_receipt.json') == INTEGRITY, 'Integrity seal mismatch; remote0')
    integrity = read(bundle / 'control/integrity_receipt.json')
    maps = [(bundle / 'source', {**c['source']['files'], **c['input_files']}), (bundle, c['operation']['files']), (PROJECT, integrity['project_files'])]
    local_tools = bundle / 'control/local_tools_integrity.json'
    if local_tools.exists():
        seal = read(local_tools)
        require(seal['contract_sha256'] == CONTRACT, 'Observer tool seal identity mismatch')
        maps += [(bundle, seal['files']), (PROJECT, seal.get('project_files', {}))]
    for base, files in maps:
        for name, digest in files.items():
            require(sha(safe(base, name)) == digest, 'Local raw SHA mismatch: ' + name)
    require(read(bundle / 'approval.json')['contract_sha256'] == CONTRACT, 'Approval binding mismatch')
    release = read(bundle / 'ownership_release.json')
    proof_path = bundle / 'handoff/ownership_release_source.json'
    require(sha(proof_path) == release['source_receipt_raw_sha256'] == RELEASE_SOURCE, 'A100 release source SHA mismatch; remote0')
    proof = read(proof_path)
    require(release['contract_sha256'] == CONTRACT and release['released_to_rtx'] is True and release['unstarted_verified'] is True and set(release['execution_ids']) == {j['id'] for j in c['jobs']}, 'Ownership release mismatch; remote0')
    require(proof['status'] == 'released_after_planned_operational_handoff' and proof['completed_count'] == 15 and proof['scientific_failures'] == 0 and not proof['actual_idle_observation']['processes'] and not proof['actual_idle_observation']['gpu_processes'], 'Parent release is incomplete')
    return c


def remote_snapshot(c, host, expected_controls):
    """Executed remotely as stdlib source; all operations are reads."""
    import os
    root = Path(c['hosts'][host]['root'])
    spec = c['hosts'][host]
    require(read(root / 'execution_contract.json') == c, 'Remote contract mismatch')
    for base, files in [(root / 'source', {**c['source']['files'], **c['input_files']}), (root, c['operation']['files']), (root, expected_controls)]:
        for name, digest in files.items():
            require(sha(safe(base, name)) == digest, 'Remote source/operation/control SHA mismatch: ' + name)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name', '--format=csv,noheader'], text=True, timeout=20).strip()
    require(any(line.split(',')[0].strip() == spec['gpu_uuid'] and spec['gpu_name_contains'] in line for line in gpu.splitlines()), 'GPU UUID mismatch')
    gpu_processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True, timeout=20).strip()
    processes = []
    for line in subprocess.check_output(['ps', '-eo', 'pid=,ppid=,args='], text=True, timeout=20).splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) == 3 and str(root) in fields[2] and int(fields[0]) != os.getpid():
            processes.append({'pid': int(fields[0]), 'ppid': int(fields[1]), 'argv': shlex.split(fields[2])})
    def optional(path):
        return read(path) if path.exists() else None
    out = {'host': host, 'root': str(root), 'contract_sha256': canonical(c), 'observed_unix': time.time(), 'gpu': gpu, 'gpu_processes': gpu_processes, 'processes': processes, 'jobs': {}, 'rawSHA_verified': True, 'raw_files_sha256': {}}
    for name in ['status.json', 'failure.json', 'supervisor_exit.json', 'supervisor.claim', 'start_permit.json', 'training_permit.json', 'pipeline_heartbeat.json', 'pipeline_failure.json', 'pipeline_complete.json', 'qualification/receipt.json', 'qualification_exit.json', 'concurrency_exit.json', 'concurrency_admission/receipt.json']:
        path = root / name
        if path.exists():
            raw = path.read_bytes()
            out[name] = json.loads(raw)
            out['raw_files_sha256'][name] = hashlib.sha256(raw).hexdigest()
        else:
            out[name] = None
    start = out['start_permit.json']
    require(start and start['contract_sha256'] == canonical(c) and start['host'] == host, 'Start permit identity mismatch')
    if out['training_permit.json']:
        permit = out['training_permit.json']
        require(permit['host'] == host and permit['start_permit_sha256'] == canonical(start), 'Start/training permit bridge mismatch')
        qualification = out['qualification/receipt.json']
        require(permit['contract_sha256'] == canonical(c) and permit['qualification'] == qualification and permit['qualification_sha256'] == sha(root / 'qualification/receipt.json'), 'Qualification bridge mismatch')
        require(qualification['status'] == 'passed' and qualification['runtime']['gpu']['uuid'] == spec['gpu_uuid'], 'Native qualification identity mismatch')
        for key, value in spec['runtime_expected'].items():
            require(qualification['runtime'][key] == value, 'Native runtime mismatch: ' + key)
        qexit = out['qualification_exit.json']
        require(qexit and qexit['actual_child_exit_observed'] is True and qexit['returncode'] == 0 and canonical(qexit) == permit['qualification_exit_sha256'], 'Qualification actual exit mismatch')
        if 'concurrency_admission' in permit:
            admission = out['concurrency_admission/receipt.json']
            cexit = out['concurrency_exit.json']
            require(permit['concurrency_admission'] == admission and sha(root / 'concurrency_admission/receipt.json') == permit['concurrency_admission_sha256'], 'Admission bridge mismatch')
            require(admission['status'] == 'passed' and admission['workers'] == 3 and admission['simultaneous_ready_verified'] is True and cexit['actual_child_exit_observed'] is True and cexit['returncode'] == 0, 'Admission actual exit mismatch')
            for name, digest in admission['files'].items():
                require(sha(safe(root / 'concurrency_admission', name)) == digest, 'Admission evidence mismatch')
    if (out['status.json'] or {}).get('active_jobs'):
        require(out['training_permit.json'] and 'concurrency_admission' in out['training_permit.json'], 'Fit lacks admission-backed training permit')
    for job in c['jobs']:
        if job['host'] != host:
            continue
        folder = root / 'run' / job['id']
        run = folder / 'runs' / job['arm'] / 'count_only_log_regression' / ('seed_' + str(job['seed']))
        record = {'job': job, 'status': optional(folder / 'status.json'), 'claim': optional(root / 'claims' / (job['id'] + '.json')), 'lease': optional(root / 'leases' / (job['id'] + '.json')), 'worker_exit': optional(root / 'worker_exits' / (job['id'] + '.json')), 'terminal': optional(folder / 'terminal_manifest.json'), 'checkpoint_receipt': None}
        if record['worker_exit']:
            require(record['worker_exit']['job'] == job and record['worker_exit']['contract_sha256'] == canonical(c), 'Worker exit identity mismatch')
        if record['terminal']:
            terminal = record['terminal']
            require(terminal['job'] == job and terminal['contract_sha256'] == canonical(c) and terminal['scientific_success'] is True and terminal['status'] == 'complete' and terminal['held_out_test_evaluated'] is False, 'Terminal identity mismatch')
            for name, digest in terminal['files'].items():
                require(sha(safe(folder, name)) == digest, 'Terminal raw SHA mismatch')
            record.update(terminal_manifest_sha256=sha(folder / 'terminal_manifest.json'), endpoint=optional(run / 'endpoint_replays.json'), history=optional(run / 'history.json'), timing=optional(run / 'epoch_timing.json'))
        else:
            receipt_path = run / 'server_checkpoint_receipt.json'
            if receipt_path.exists():
                raw = receipt_path.read_bytes()
                receipt = json.loads(raw)
                require(receipt['contract_sha256'] == canonical(c), 'Checkpoint contract mismatch')
                snapshots = {name: (run / name).read_bytes() for name in ('history.json', 'epoch_timing.json') if (run / name).exists()}
                stable = all((hashlib.sha256(snapshots[name]).hexdigest() if name in snapshots else sha(safe(run, name)) if (run / name).exists() else None) == digest for name, digest in receipt['files'].items())
                stable = stable and raw == receipt_path.read_bytes()
                record.update(checkpoint_receipt=receipt, checkpoint_stable=stable, checkpoint_receipt_raw_sha256=hashlib.sha256(raw).hexdigest())
                if stable:
                    record.update(history=json.loads(snapshots['history.json']), timing=json.loads(snapshots['epoch_timing.json']))
        out['jobs'][job['id']] = record
    return out


def own_eta(timing, epoch, best):
    rows = [r for r in timing.get('epochs', []) if type(r.get('epoch')) is int and 0 < r['epoch'] <= epoch and type(r.get('elapsed_seconds')) in (int, float) and math.isfinite(r['elapsed_seconds']) and r['elapsed_seconds'] > 0]
    require(len({r['epoch'] for r in rows}) == len(rows), 'Duplicated completed epoch timing')
    rows = sorted(rows, key=lambda r: r['epoch'])[-10:]
    if not rows:
        return {'status': 'unconfirmed_no_own_completed_epoch_timing'}
    median = statistics.median(r['elapsed_seconds'] for r in rows)
    end = min(300, max(40, best + 40))
    return {'median_epoch_seconds': median, 'sample_count': len(rows), 'unchanged_best_end_epoch': end, 'unchanged_best_seconds': max(0, end - epoch) * median, 'max300_seconds': max(0, 300 - epoch) * median, 'whole_queue_eta': 'unconfirmed', 'scope': 'this fit and physical GPU only; endpoint/retrieval/CPU audit excluded'}


def process_match(c, host, job, row, snapshot):
    spec = c['hosts'][host]
    status = snapshot.get('status.json') or {}
    active = (status.get('active_jobs') or {}).get(job['id'], {})
    claim, lease = row.get('claim'), row.get('lease')
    if not claim or not lease or not active:
        return None
    deadline = claim.get('deadline_unix')
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        return None
    root, python = spec['root'], spec['python']
    sup_argv = [python, root + '/operation/campaign.py', '--contract', root + '/execution_contract.json', '--host', host, '--mode', 'dispatch']
    fit_argv = [python, root + '/operation/campaign.py', '--contract', root + '/execution_contract.json', '--host', host, '--mode', 'fit', '--job', job['id'], '--deadline', str(deadline)]
    guard = active.get('worker_guard_argv')
    if not isinstance(guard, list) or len(guard) != len(fit_argv) + 4 or guard[:3] != ['timeout', '--signal=TERM', '--kill-after=15s'] or guard[4:] != fit_argv:
        return None
    try:
        if not 0 < int(guard[3]) <= c['limits']['per_condition_seconds']:
            return None
    except (ValueError, TypeError):
        return None
    processes = {p['pid']: p for p in snapshot['processes']}
    supervisor = processes.get(status.get('supervisor_pid'))
    timeout = processes.get(active.get('worker_group_pid'))
    workers = [p for p in snapshot['processes'] if timeout and p['ppid'] == timeout['pid'] and p['argv'] == fit_argv]
    if not supervisor or not timeout or len(workers) != 1:
        return None
    worker = workers[0]
    valid = (claim.get('contract_sha256') == CONTRACT and claim.get('job') == job and supervisor['argv'] == sup_argv and status.get('supervisor_argv') == sup_argv and timeout['argv'] == guard and timeout['ppid'] == supervisor['pid'] and lease.get('contract_sha256') == CONTRACT and lease.get('job') == job['id'] and lease.get('supervisor_pid') == supervisor['pid'] and lease.get('worker_group_pid') == timeout['pid'] and lease.get('issued_unix', math.inf) <= snapshot['observed_unix'] < lease.get('expires_unix', 0) <= deadline and active.get('condition_deadline_unix') == deadline)
    if not valid:
        return None
    return {'worker_pid': worker['pid'], 'worker_ppid': worker['ppid'], 'timeout_pid': timeout['pid'], 'supervisor_pid': supervisor['pid'], 'argv_verified': True}


def classify(c, host, snapshot):
    require(snapshot['contract_sha256'] == CONTRACT and snapshot['root'] == c['hosts'][host]['root'] and snapshot['rawSHA_verified'] is True, 'Snapshot identity mismatch')
    counts = dict.fromkeys(['complete', 'failed', 'running', 'waiting', 'unconfirmed'], 0)
    jobs = [j for j in c['jobs'] if j['host'] == host]
    require(set(snapshot['jobs']) == {j['id'] for j in jobs}, 'Snapshot job ownership changed')
    matches = {j['id']: process_match(c, host, j, snapshot['jobs'][j['id']], snapshot) for j in jobs}
    gpu_pids = set()
    for line in snapshot['gpu_processes'].splitlines():
        uuid, pid = (x.strip() for x in line.split(',')[:2])
        require(uuid == c['hosts'][host]['gpu_uuid'], 'Foreign GPU UUID in compute inventory')
        gpu_pids.add(int(pid))
    valid_gpu = gpu_pids <= {m['worker_pid'] for m in matches.values() if m}
    rows = []
    for job in jobs:
        row = snapshot['jobs'][job['id']]
        result = {'job': job, 'state': 'waiting', 'stored_epoch': None, 'best_epoch': None}
        if row.get('terminal'):
            result['state'] = 'complete'
            endpoint = row.get('endpoint') or {}
            result.update(stored_epoch=endpoint.get('completed_epochs'), best_epoch=endpoint.get('best_epoch'), terminal_manifest_sha256=row.get('terminal_manifest_sha256'))
        elif row.get('worker_exit') and row['worker_exit'].get('actual_child_exit_observed') is True and row['worker_exit'].get('returncode') not in (None, 0):
            result['state'] = 'failed'
        elif job['id'] in ((snapshot.get('status.json') or {}).get('active_jobs') or {}):
            result['state'] = 'running' if matches[job['id']] and valid_gpu else 'unconfirmed'
            result.update(matches[job['id']] or {})
            result['GPU_PID_present'] = bool(matches[job['id']] and matches[job['id']]['worker_pid'] in gpu_pids)
            receipt = row.get('checkpoint_receipt')
            result['first_checkpoint_not_saved'] = receipt is None
            result['checkpoint_stable'] = row.get('checkpoint_stable')
            if receipt and row.get('checkpoint_stable'):
                history = (row.get('history') or {}).get('history', [])
                epoch = receipt['epoch']
                require(len(history) == epoch, 'History/receipt epoch mismatch')
                finite = [(r['val_qty_rmse'], i + 1) for i, r in enumerate(history) if type(r.get('val_qty_rmse')) in (int, float) and math.isfinite(r['val_qty_rmse'])]
                require(bool(finite), 'No finite selector epoch')
                best = min(finite)[1]
                result.update(stored_epoch=epoch, best_epoch=best, ETA=own_eta(row.get('timing') or {}, epoch, best))
        elif row.get('claim') or row.get('status'):
            result['state'] = 'unconfirmed'
        counts[result['state']] += 1
        rows.append(result)
    exit_receipt = snapshot.get('supervisor_exit.json') or {}
    terminal = False
    terminal_pending_reason = None
    if counts['complete'] == len(jobs):
        from paper.scripts.finalize_titantpp_history_rtx_handoff import verify_terminal_snapshot
        try:
            raw_bindings = snapshot.get('raw_files_sha256') or {}
            for name in ('supervisor_exit.json', 'pipeline_complete.json', 'pipeline_heartbeat.json'):
                digest = raw_bindings.get(name)
                require(isinstance(digest, str) and len(digest) == 64 and all(ch in '0123456789abcdef' for ch in digest), 'Missing terminal raw binding: ' + name)
            complete = snapshot.get('pipeline_complete.json') or {}
            pulse = snapshot.get('pipeline_heartbeat.json') or {}
            require(type(complete.get('completed_unix')) in (int, float) and type(pulse.get('updated_unix')) in (int, float) and complete['completed_unix'] <= pulse['updated_unix'] <= snapshot['observed_unix'], 'Stale pipeline final heartbeat')
            verify_terminal_snapshot(c, host, snapshot)
            terminal = True
        except (ValueError, KeyError, TypeError) as exc:
            terminal_pending_reason = str(exc)
    return {'host': host, 'root': c['hosts'][host]['root'], 'pipeline_failure': snapshot.get('pipeline_failure.json'), 'failure': snapshot.get('failure.json'), 'contract_sha256': CONTRACT, 'observed_unix': snapshot['observed_unix'], 'observed_kst': kst(snapshot['observed_unix']), 'counts': counts, 'rows': rows, 'verified_terminal': terminal, 'terminal_pending_reason': terminal_pending_reason, 'phase': (snapshot.get('pipeline_heartbeat.json') or {}).get('phase'), 'qualification_is_fit': False}


def ssh_snapshot(c, host, controls):
    names = [require, read, sha, canonical, safe, remote_snapshot]
    code = 'import hashlib,json,math,subprocess,shlex,time\nfrom pathlib import Path\n' + '\n'.join(inspect.getsource(f) for f in names)
    code += '\nprint(json.dumps(remote_snapshot(' + repr(c) + ',' + repr(host) + ',' + repr(controls) + '),allow_nan=False))\n'
    run = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host, 'python3 -'], input=code, text=True, capture_output=True, timeout=240)
    require(run.returncode == 0, 'Read-only SSH failed, not training failure: ' + run.stderr[-1200:])
    return json.loads(run.stdout)


def observe_host(host, *, bundle=BUNDLE, fetch=ssh_snapshot, verifier=verify_local):
    c = verifier(bundle)
    require(host in ('5080', '5090'), 'Unknown host')
    monitor = bundle / 'hourly_monitor'
    cache = monitor / ('terminal_cache_' + host + '.json')
    if cache.exists():
        pointer = read(monitor / ('terminal_pointer_' + host + '.json'))
        require(sha(cache) == pointer['sha256'], 'Terminal cache SHA mismatch')
        out = read(cache)
        require(out['verified_terminal'] is True and out['contract_sha256'] == CONTRACT and out['host'] == host and out['root'] == c['hosts'][host]['root'], 'Terminal cache identity mismatch')
        snapshot_path = Path(out['snapshot_path'])
        require(snapshot_path.resolve().is_relative_to(monitor.resolve()) and sha(snapshot_path) == out['snapshot_raw_sha256'], 'Terminal raw snapshot SHA mismatch')
        reclassified = classify(c, host, read(snapshot_path))
        if reclassified['verified_terminal']:
            require(reclassified['counts'] == out['counts'] and reclassified['rows'] == out['rows'], 'Terminal cached classification drift')
            return {**out, 'observation_source': 'verified_terminal_cache', 'remote_calls': 0}
        # An old insufficient terminal observation must not freeze SSH forever.
        # Keep both original cache and pointer bytes before allowing one refresh.
        for old_path in (cache, monitor / ('terminal_pointer_' + host + '.json')):
            archived = old_path.with_name(old_path.stem + '_superseded_' + sha(old_path)[:16] + '.json')
            if archived.exists():
                require(archived.read_bytes() == old_path.read_bytes(), 'Superseded cache evidence changed')
            else:
                with archived.open('xb') as stream:
                    stream.write(old_path.read_bytes())
    launch = bundle / 'deployment' / ('launch_receipt_' + host + '.json')
    if not launch.exists():
        rows = [{'job': j, 'state': 'waiting', 'stored_epoch': None, 'best_epoch': None} for j in c['jobs'] if j['host'] == host]
        return {'host': host, 'contract_sha256': CONTRACT, 'status': 'not_launched', 'remote_calls': 0, 'observed_kst': None, 'counts': {'complete': 0, 'failed': 0, 'running': 0, 'waiting': len(rows), 'unconfirmed': 0}, 'rows': rows}
    receipt = read(launch)
    require(receipt['contract_sha256'] == CONTRACT and receipt['host'] == host and receipt['root'] == c['hosts'][host]['root'] and receipt['gpu_uuid'] == c['hosts'][host]['gpu_uuid'] and receipt['actual_launch'] is True, 'Launch receipt binding mismatch; remote0')
    control_root = bundle / 'control/hosts' / host
    seal = read(control_root / 'integrity_receipt.json')
    require(seal['contract_sha256'] == CONTRACT, 'Host control seal binding mismatch')
    controls = dict(seal['files'])
    controls['control/integrity_receipt.json'] = sha(control_root / 'integrity_receipt.json')
    for name, digest in seal['files'].items():
        local = control_root / Path(name).name if name.startswith('control/') else bundle / name
        require(sha(local) == digest, 'Host controller raw SHA mismatch')
    snapshot = None
    try:
        snapshot = fetch(c, host, controls)
        out = classify(c, host, snapshot)
    except Exception as exc:
        error = {'host': host, 'contract_sha256': CONTRACT, 'status': 'observation_failed_not_training_failure', 'remote_calls': 1, 'local_received_unix': time.time(), 'message': str(exc), 'counts': {'complete': 0, 'failed': 0, 'running': 0, 'waiting': 0, 'unconfirmed': sum(j['host'] == host for j in c['jobs'])}, 'rows': [{'job': j, 'state': 'unconfirmed'} for j in c['jobs'] if j['host'] == host]}
        error_path = monitor / ('observation_error_' + host + '_' + str(time.time_ns()) + '.json')
        write(error_path, {'error': error, 'unvalidated_snapshot': snapshot}, True)
        error['error_evidence_path'] = str(error_path)
        error['error_evidence_raw_sha256'] = sha(error_path)
        return error
    raw_path = monitor / ('snapshot_' + host + '_' + str(time.time_ns()) + '.json')
    write(raw_path, snapshot, True)
    out.update(snapshot_path=str(raw_path), snapshot_raw_sha256=sha(raw_path), remote_calls=1, observation_source='live_read_only_SSH')
    path = monitor / ('observation_' + host + '_' + str(time.time_ns()) + '.json')
    write(path, out, True)
    write(monitor / ('latest_' + host + '.json'), {'path': str(path), 'sha256': sha(path), 'observed_kst': out['observed_kst']})
    if out['verified_terminal']:
        write(cache, out, not cache.exists())
        terminal_pointer = monitor / ('terminal_pointer_' + host + '.json')
        write(terminal_pointer, {'sha256': sha(cache)}, not terminal_pointer.exists())
    return out


def unique_counts(a100_rows, rtx_rows):
    require(len(a100_rows) == 15 and len(rtx_rows) == 12, 'Expected15A100+12RTX evidence rows')
    rows = a100_rows + rtx_rows
    ids = [row['job']['id'] for row in rows]
    require(len(set(ids)) == len(ids) == 27, 'Duplicate execution ID; do not double-count transferred jobs')
    counts = dict.fromkeys(['complete', 'failed', 'running', 'waiting', 'unconfirmed'], 0)
    for row in rows:
        counts[row['state']] += 1
    return counts


def local_a100_rows():
    pointer = read(A100 / 'hourly_monitor/latest_a100.json')
    snapshot = Path(pointer['path'])
    require(snapshot.resolve().is_relative_to(A100.resolve()) and sha(snapshot) == pointer['raw_sha256'], 'A100 cache raw SHA mismatch')
    value = read(snapshot)
    c = read(A100 / 'execution_contract.json')
    require(value['contract_sha256'] == canonical(c) == 'da077e96e61255286c5cb35b54dc09ec8144e2d2f410e6eb46fea4681ee17899', 'A100 cache contract mismatch')
    retained = {j['id'] for j in c['jobs'][:15]}
    rows = []
    for row in value['jobs']:
        if row['job']['id'] not in retained:
            continue
        state = row['state']
        state = 'running' if state in ('training', 'running_before_first_save') else state
        rows.append({'job': row['job'], 'state': state if state in ('complete', 'running', 'waiting') else 'unconfirmed', 'stored_epoch': (row.get('endpoints') or {}).get('completed_epochs', row.get('saved_epoch')), 'best_epoch': (row.get('endpoints') or {}).get('best_epoch', row.get('best_epoch'))})
    proof_path = BUNDLE / 'handoff/ownership_release_source.json'
    require(sha(proof_path) == RELEASE_SOURCE, 'Planned handoff release changed')
    proof = read(proof_path)
    require(set(proof['terminal_manifest_sha256']) == retained and proof['completed_count'] == 15 and proof['scientific_failures'] == 0, 'Retained A100 terminal ownership mismatch')
    from paper.scripts.finalize_titantpp_history_rtx_handoff import parent_records
    rtx_contract = read(BUNDLE / 'execution_contract.json')
    require(canonical(rtx_contract) == CONTRACT, 'RTX contract mismatch before parent audit')
    records = parent_records(rtx_contract, a100=A100, bundle=BUNDLE)
    rows = parent_epoch_rows(rows, records, retained)
    return rows, {'cached_epoch_observed_kst': kst(value['observed_unix']), 'actual_terminal_observed_kst': kst(proof['actual_idle_observation']['observed_unix']), 'remote_calls': 0, 'planned_handoff_is_not_numerical_failure': True, 'source_receipt_raw_sha256': RELEASE_SOURCE, 'all15_archived_endpoints_verified': records is not None}


def parent_epoch_rows(cached_rows, records, retained):
    if records is not None:
        require(len(records) == 15 and {r['id'] for r in records} == retained, 'Verified parent endpoint ownership mismatch')
        by_id = {r['id']: r for r in records}
        rows = []
        for row in cached_rows:
            record = by_id[row['job']['id']]
            rows.append({**row, 'state': 'complete', 'stored_epoch': record['last_epoch'], 'best_epoch': record['best_epoch'], 'terminal_manifest_sha256': record['terminal_manifest_sha256'], 'endpoint_raw_sha256': record['endpoint_raw_sha256'], 'original_provenance': record['provenance'], 'epoch_evidence': 'immutable archived endpoint; source observation timestamp preserved'})
        return rows
    return [{**row, **({'state': 'complete', 'stored_epoch': None, 'best_epoch': None, 'epoch_evidence': 'terminal validated at release; earlier checkpoint is not final epoch'} if row['state'] != 'complete' else {})} for row in cached_rows]


def verify_instacart():
    path = INSTACART / 'analysis/completion_receipt.json'
    require(sha(path) == INSTACART_RECEIPT, 'Instacart completion receipt changed')
    receipt = read(path)
    for name, digest in receipt['analysis_files'].items():
        require(sha(safe(INSTACART, name)) == digest, 'Instacart analysis raw SHA mismatch')
    checked = 0
    for host, archive in receipt['archive_receipts'].items():
        require(sha(INSTACART / 'retrieval' / ('archive_' + host + '.tar.gz')) == archive['archive_sha256'], 'Instacart archive changed')
        base = INSTACART / 'originals' / host
        manifest = base / 'preservation/manifest.json'
        require(sha(manifest) == archive['manifest_sha256'], 'Instacart manifest changed')
        files = read(manifest)['files']
        require(len(files) == archive['files'], 'Instacart original count changed')
        for name, digest in files.items():
            require(sha(safe(base, name)) == digest, 'Instacart original SHA mismatch')
            checked += 1
    require(checked == 614, 'Instacart originals incomplete')
    return {'status': 'complete', 'completed': 9, 'local_originals_SHA_verified': checked, 'remote_calls': 0, 'completion_receipt_raw_sha256': INSTACART_RECEIPT}


def main():
    p = argparse.ArgumentParser()
    scope = p.add_mutually_exclusive_group(required=True)
    scope.add_argument('--host', choices=['5080', '5090'])
    scope.add_argument('--all', action='store_true')
    args = p.parse_args()
    hosts = ('5080', '5090') if args.all else (args.host,)
    instacart = verify_instacart()
    rows, info = local_a100_rows()
    results = {host: observe_host(host) for host in hosts}
    result = {'hosts': results, 'Instacart': instacart, 'A100_remote_calls': 0, 'Pod_API_calls': 0}
    if args.all:
        result.update(A100=info, counts=unique_counts(rows, [row for out in results.values() for row in out['rows']]))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))

if __name__ == '__main__':
    main()
