"""Recover terminal RTX originals once and combine the approved 27 history fits.

This is an originals preservation and Validation-only reporting tool. It never
launches or retries a fit, contacts a Pod API, or opens held-out predictions.
Live owners are local-pending with zero remote calls. Failed claims are retained.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import subprocess
import sys
import tarfile
import time

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from paper.scripts import observe_titantpp_history_rtx_handoff as observer
from paper.scripts import deploy_titantpp_history_rtx_handoff as deployment

BUNDLE = observer.BUNDLE
A100 = observer.A100
CONTRACT = observer.CONTRACT
LOCAL_TOOLS = '9349554f61f5861324dd32f1d1ea6a447eb7f0fc5ea3c26e3dc02eaebc8f1c51'
A100_CONTRACT = 'da077e96e61255286c5cb35b54dc09ec8144e2d2f410e6eb46fea4681ee17899'
ARMS = ('titantpp_cnn_gru54', 'titantpp_cnn_gru54_correction_disabled', 'titantpp_cnn_gru54_current_only')
SEEDS = (42, 52, 62)
VARIANT = 'count_only_log_regression'
METRICS = ('qty_rmse', 'qty_mae', 'time_nll')


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
    p = PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and str(p) == name and name not in ('', '.') and '\\' not in name, 'Unsafe original path: ' + name)
    target = Path(root).joinpath(*p.parts)
    require(not target.is_symlink() and target.resolve().is_relative_to(Path(root).resolve()), 'Foreign/symlink original: ' + name)
    return target


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def no_held_out(name):
    parts = PurePosixPath(name).parts
    # Frozen source can contain test function names; experiment artifacts cannot.
    if parts and parts[0] in ('source', 'operation', 'control'):
        return
    require(not any(p.lower() == 'test' or p.lower().startswith(('test_', 'heldout', 'held_out')) for p in parts), 'Held-out artifact excluded: ' + name)


def validate_local(bundle=BUNDLE):
    require(sha(bundle / 'control/local_tools_integrity.json') == LOCAL_TOOLS, 'Local tools seal drift; remote0')
    c = observer.verify_local(bundle)
    require(c['training']['tie'] == 'strict_earliest_finite_minimum' and c['training']['monitor'] == 'validation_raw_quantity_rmse', 'Selector changed')
    require(len(c['jobs']) == 12 and set(c['arms']) == set(ARMS), 'RTX scientific ownership changed')
    return c


def verify_terminal_snapshot(c, host, snapshot):
    """Extra finalization gates beyond the observer's scientific terminal flag."""
    require(snapshot['contract_sha256'] == canonical(c) and snapshot['host'] == host and snapshot['root'] == c['hosts'][host]['root'] and snapshot['rawSHA_verified'] is True, 'Snapshot identity drift')
    require(not snapshot['processes'] and not snapshot['gpu_processes'].strip(), 'Owned/GPU process still present')
    jobs = [j for j in c['jobs'] if j['host'] == host]
    require(len(jobs) == (3 if host == '5080' else 9) and set(snapshot['jobs']) == {j['id'] for j in jobs}, 'Host terminal identity set drift')
    require(snapshot.get('failure.json') is None and snapshot.get('pipeline_failure.json') is None, 'Preserved host failure requires user action')
    sexit = snapshot.get('supervisor_exit.json') or {}
    require(sexit.get('actual_child_exit_observed') is True and sexit.get('returncode') == 0 and sexit.get('contract_sha256') == canonical(c) and sexit.get('phase') == 'dispatch', 'Actual dispatch exit0 missing')
    status = snapshot.get('status.json') or {}
    complete = snapshot.get('pipeline_complete.json') or {}
    pulse = snapshot.get('pipeline_heartbeat.json') or {}
    for name in ('supervisor_exit.json', 'pipeline_complete.json', 'pipeline_heartbeat.json'):
        digest = (snapshot.get('raw_files_sha256') or {}).get(name)
        require(isinstance(digest, str) and len(digest) == 64 and all(ch in '0123456789abcdef' for ch in digest), 'Missing terminal raw binding: ' + name)
    require(status.get('status') == 'complete' and status.get('completed_conditions') == len(jobs), 'Host scientific completion absent')
    require(complete.get('status') == 'complete' and complete.get('completed_conditions') == len(jobs) and complete.get('contract_sha256') == canonical(c) and complete.get('held_out_test_evaluated') is False, 'Pipeline completion binding missing')
    require(complete.get('supervisor_exit_sha256') == canonical_raw_json_match(snapshot, 'supervisor_exit.json'), 'Pipeline dispatch receipt raw binding mismatch')
    idle = complete.get('actual_exit_and_idle_observation') or {}
    require(not idle.get('owned_processes', [None]) and not idle.get('gpu_pids', [None]), 'Pipeline completion lacks actual owner absence')
    require(pulse.get('phase') == 'complete' and pulse.get('contract_sha256') == canonical(c), 'Pipeline final heartbeat missing')
    require(type(complete.get('completed_unix')) in (int, float) and type(pulse.get('updated_unix')) in (int, float) and complete['completed_unix'] <= pulse['updated_unix'] <= snapshot['observed_unix'], 'Stale pipeline final heartbeat')
    for job in jobs:
        row = snapshot['jobs'][job['id']]
        terminal = row.get('terminal') or {}
        require(terminal.get('job') == job and terminal.get('contract_sha256') == canonical(c) and terminal.get('scientific_success') is True and terminal.get('status') == 'complete' and terminal.get('held_out_test_evaluated') is False, 'Invalid scientific terminal')
        worker = row.get('worker_exit') or {}
        require(worker.get('job') == job and worker.get('contract_sha256') == canonical(c) and worker.get('actual_child_exit_observed') is True and worker.get('returncode') == 0, 'Actual worker exit0 missing')
    return {j['id']: snapshot['jobs'][j['id']]['terminal_manifest_sha256'] for j in jobs}


def canonical_raw_json_match(snapshot, name):
    # The observer carries parsed JSON. Raw receipt hashes must be carried too;
    # where an older snapshot lacks them, this gate remains pending, never guessed.
    return (snapshot.get('raw_files_sha256') or {}).get(name)


def terminal_cache(c, host, bundle=BUNDLE):
    monitor = bundle / 'hourly_monitor'
    path = monitor / ('terminal_cache_' + host + '.json')
    if not path.exists():
        return None
    require(sha(path) == read(monitor / ('terminal_pointer_' + host + '.json'))['sha256'], 'Terminal cache raw SHA drift')
    cache = read(path)
    require(cache.get('verified_terminal') is True and cache['host'] == host and cache['contract_sha256'] == canonical(c), 'Terminal cache identity drift')
    snapshot_path = Path(cache['snapshot_path'])
    require(snapshot_path.resolve().is_relative_to(monitor.resolve()) and sha(snapshot_path) == cache['snapshot_raw_sha256'], 'Raw snapshot SHA drift')
    snapshot = read(snapshot_path)
    # Missing terminal receipts mean wait; inconsistent receipts mean fail closed.
    if not snapshot.get('pipeline_complete.json') or not (snapshot.get('raw_files_sha256') or {}).get('supervisor_exit.json'):
        return None
    classified = observer.classify(c, host, snapshot)
    require(classified['verified_terminal'] is True and classified['counts'] == cache['counts'] and classified['rows'] == cache['rows'], 'Cached terminal classification drift')
    terminals = verify_terminal_snapshot(c, host, snapshot)
    launch_path = bundle / 'deployment' / ('launch_receipt_' + host + '.json')
    launch = read(launch_path)
    require(launch['actual_launch'] is True and launch['contract_sha256'] == canonical(c) and launch['root'] == c['hosts'][host]['root'] and launch['gpu_uuid'] == c['hosts'][host]['gpu_uuid'], 'Launch binding drift')
    return {'cache_raw_sha256': sha(path), 'snapshot_raw_sha256': sha(snapshot_path), 'snapshot_path': str(snapshot_path), 'observed_unix': cache['observed_unix'], 'observed_kst': cache['observed_kst'], 'terminals': terminals, 'pipeline_exit_code_directly_observed': False}


def verify_root_bindings(root, request):
    for name, digest in request['bindings'].items():
        require(sha(safe(root, name)) == digest, 'Authorization receipt raw SHA drift: ' + name)
    # Launch returns an object and the two writers serialize it differently.
    # Compare the entire object; the native raw bytes remain in the archive.
    for name, expected in request['object_bindings'].items():
        require(canonical(read(safe(root, name))) == canonical(expected), 'Launch receipt object binding drift: ' + name)


def collect_remote(c, host, request):
    """Executed only by the root's approved finalizer, under the new owned root."""
    import os
    root = Path(c['hosts'][host]['root'])
    snapshot = remote_snapshot(c, host, request['controls'])
    snapshot['raw_files_sha256'] = {n: sha(root / n) for n in ('supervisor_exit.json', 'pipeline_complete.json', 'pipeline_heartbeat.json')}
    terminals = verify_terminal_snapshot(c, host, snapshot)
    require(terminals == request['terminals'], 'Observed terminal changed before originals recovery')
    verify_root_bindings(root, request)
    pipeline_claim = read(root / 'pipeline.claim')
    argv = [c['hosts'][host]['python'], str(root / 'control/pipeline.py'), '--root', str(root), '--contract-sha256', canonical(c), '--integrity-sha256', request['controls']['control/integrity_receipt.json']]
    require(pipeline_claim['argv'] == argv and pipeline_claim['contract_sha256'] == canonical(c), 'Original pipeline argv/contract changed')
    permit = snapshot['training_permit.json']
    require(permit and 'concurrency_admission' in permit, 'Native three-worker training authorization missing')
    preservation = root / 'preservation'
    preservation.mkdir(exist_ok=False)
    write_new(preservation / 'claim.json', {'contract_sha256': canonical(c), 'host': host, 'pid': os.getpid(), 'created_unix': time.time(), 'automatic_retry': False})
    try:
        files = {}
        for directory in ('source', 'operation', 'control', 'run', 'logs', 'claims', 'leases', 'worker_exits', 'qualification', 'concurrency_admission', 'deployment', 'handoff'):
            base = root / directory
            if not base.exists():
                continue
            for path in sorted(base.rglob('*')):
                if any(p in ('__pycache__', 'cache', '.git', '.venv', 'runtime_env') for p in path.relative_to(base).parts):
                    continue
                require(not path.is_symlink(), 'Symlink in originals')
                if not path.is_file():
                    continue
                name = path.relative_to(root).as_posix()
                no_held_out(name)
                files[name] = {'sha256': sha(path), 'size': path.stat().st_size}
        for name in ('execution_contract.json', 'current.json', 'approval.json', 'ownership_release.json', 'start_permit.json', 'training_permit.json', 'status.json', 'qualification_exit.json', 'concurrency_exit.json', 'supervisor_exit.json', 'pipeline_complete.json', 'pipeline_heartbeat.json', 'pipeline.claim', 'supervisor.claim', 'server_lease.json'):
            path = safe(root, name)
            if path.exists():
                files[name] = {'sha256': sha(path), 'size': path.stat().st_size}
        manifest = {'schema': 'titantpp_rtx_history_originals_v1', 'contract_sha256': canonical(c), 'host': host, 'root': str(root), 'gpu_uuid': c['hosts'][host]['gpu_uuid'], 'files': files, 'terminal_manifest_sha256': terminals, 'scientific_success': True, 'selected_last_binary_included': True, 'held_out_test_evaluated': False, 'actual_idle_observation': {'owned_processes': snapshot['processes'], 'gpu_pids': []}, 'dispatch_actual_exit_code': 0, 'pipeline_exit_code_directly_observed': False, 'original_observed_unix': request['observed_unix'], 'original_snapshot_raw_sha256': request['snapshot_raw_sha256']}
        write_new(preservation / 'manifest.json', manifest)
        archive = preservation / 'originals.tar.gz'
        with archive.open('xb') as raw:
            with tarfile.open(fileobj=raw, mode='w:gz', compresslevel=1) as tar:
                tar.add(preservation / 'manifest.json', arcname='manifest.json', recursive=False)
                for name in sorted(files):
                    tar.add(root / name, arcname='originals/' + name, recursive=False)
        for name, entry in files.items():
            require(sha(root / name) == entry['sha256'] and (root / name).stat().st_size == entry['size'], 'Original changed while archiving: ' + name)
        after = remote_snapshot(c, host, request['controls'])
        require(not after['processes'] and not after['gpu_processes'].strip(), 'Owner appeared during originals recovery')
        receipt = {'contract_sha256': canonical(c), 'host': host, 'root': str(root), 'archive_path': str(archive), 'archive_sha256': sha(archive), 'archive_size': archive.stat().st_size, 'manifest_sha256': sha(preservation / 'manifest.json'), 'files': len(files), 'terminal_manifest_sha256': terminals, 'actual_owner_absence_verified': True, 'dispatch_actual_exit_code': 0, 'pipeline_exit_code_directly_observed': False, 'held_out_test_evaluated': False, 'automatic_retry': False}
        write_new(preservation / 'archive_receipt.json', receipt)
        return receipt
    except BaseException as exc:
        write_new(preservation / 'failure.json', {'type': type(exc).__name__, 'message': str(exc), 'automatic_retry': False, 'created_unix': time.time()})
        raise


def export_code(c, host, evidence, bundle=BUNDLE):
    control = bundle / 'control/hosts' / host
    controls = dict(read(control / 'integrity_receipt.json')['files'])
    controls['control/integrity_receipt.json'] = sha(control / 'integrity_receipt.json')
    bindings = {'execution_contract.json': sha(bundle / 'execution_contract.json'), 'approval.json': sha(bundle / 'approval.json'), 'ownership_release.json': sha(bundle / 'ownership_release.json'), 'handoff/ownership_release_source.json': sha(bundle / 'handoff/ownership_release_source.json'), 'start_permit.json': sha(bundle / ('start_permit_' + host + '.json'))}
    object_bindings = {'deployment/launch_receipt.json': read(bundle / 'deployment' / ('launch_receipt_' + host + '.json'))}
    request = {**evidence, 'controls': controls, 'bindings': bindings, 'object_bindings': object_bindings}
    functions = [require, read, sha, canonical, safe, write_new, no_held_out, canonical_raw_json_match, verify_terminal_snapshot, verify_root_bindings, observer.remote_snapshot, collect_remote]
    code = 'import hashlib,json,subprocess,shlex,tarfile,time\nfrom pathlib import Path,PurePosixPath\n'
    code += '\n'.join(inspect.getsource(f) for f in functions)
    return code + '\nprint(json.dumps(collect_remote(' + repr(c) + ',' + repr(host) + ',' + repr(request) + '),allow_nan=False))\n'


def verify_archive(archive, destination, receipt, c, host):
    """Validate all regular members and hashes before any immutable extraction."""
    archive, destination = Path(archive), Path(destination)
    require(sha(archive) == receipt['archive_sha256'] and archive.stat().st_size == receipt['archive_size'], 'Archive raw SHA/size mismatch')
    require(receipt['contract_sha256'] == canonical(c) and receipt['host'] == host and receipt['root'] == c['hosts'][host]['root'] and receipt['held_out_test_evaluated'] is False, 'Archive receipt identity/split drift')
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        require(len(names) == len(set(names)), 'Duplicated archive member')
        for m in members:
            safe(destination, m.name)
            require(m.isfile(), 'Nonregular archive member')
        require('manifest.json' in names, 'Missing archive manifest')
        raw = tar.extractfile('manifest.json').read()
        require(hashlib.sha256(raw).hexdigest() == receipt['manifest_sha256'], 'Manifest raw SHA drift')
        manifest = json.loads(raw)
        require(manifest['schema'] == 'titantpp_rtx_history_originals_v1' and manifest['contract_sha256'] == canonical(c) and manifest['host'] == host and manifest['root'] == c['hosts'][host]['root'] and manifest['gpu_uuid'] == c['hosts'][host]['gpu_uuid'], 'Manifest identity drift')
        require(manifest['scientific_success'] is True and manifest['held_out_test_evaluated'] is False and not manifest['actual_idle_observation']['owned_processes'] and not manifest['actual_idle_observation']['gpu_pids'], 'Manifest completion/split drift')
        require(set(names) == {'manifest.json', *('originals/' + n for n in manifest['files'])} and len(manifest['files']) == receipt['files'], 'Unexpected/missing archive member')
        expected = {j['id'] for j in c['jobs'] if j['host'] == host}
        require(set(manifest['terminal_manifest_sha256']) == expected and manifest['terminal_manifest_sha256'] == receipt['terminal_manifest_sha256'], 'Archived scientific ownership drift')
        for name, entry in manifest['files'].items():
            safe(destination, name)
            no_held_out(name)
            member = tar.getmember('originals/' + name)
            digest = hashlib.sha256()
            stream = tar.extractfile(member)
            for chunk in iter(lambda: stream.read(1024**2), b''):
                digest.update(chunk)
            require(member.size == entry['size'] and digest.hexdigest() == entry['sha256'], 'Archived original raw SHA/size drift: ' + name)
        for name, digest in {**c['source']['files'], **c['input_files']}.items():
            require(manifest['files'].get('source/' + name, {}).get('sha256') == digest, 'Archived source/input changed')
        for name, digest in c['operation']['files'].items():
            require(manifest['files'].get(name, {}).get('sha256') == digest, 'Archived operation changed')
        require(canonical(json.loads(tar.extractfile('originals/execution_contract.json').read())) == canonical(c), 'Archived contract changed')
        for job in (j for j in c['jobs'] if j['host'] == host):
            prefix = 'run/' + job['id'] + '/'
            terminal_name = prefix + 'terminal_manifest.json'
            require(manifest['files'].get(terminal_name, {}).get('sha256') == manifest['terminal_manifest_sha256'][job['id']], 'Terminal observation/archive raw binding changed')
            terminal = json.loads(tar.extractfile('originals/' + terminal_name).read())
            require(terminal['job'] == job and terminal['contract_sha256'] == canonical(c) and terminal['scientific_success'] is True and terminal['status'] == 'complete' and terminal['held_out_test_evaluated'] is False, 'Archived terminal identity changed')
            for name, digest in terminal['files'].items():
                require(manifest['files'].get(prefix + name, {}).get('sha256') == digest, 'Terminal scientific file missing from archive')
        require(not destination.exists(), 'Immutable originals already exist')
        destination.mkdir(parents=True)
        for member in members:
            target = safe(destination, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as output:
                stream = tar.extractfile(member)
                for chunk in iter(lambda: stream.read(1024**2), b''):
                    output.write(chunk)
    verify_original_files(destination / 'originals', manifest)
    return manifest


def verify_original_files(root, manifest):
    for name, entry in manifest['files'].items():
        path = safe(root, name)
        require(sha(path) == entry['sha256'] and path.stat().st_size == entry['size'], 'Local original raw SHA/size changed: ' + name)


def copy_from(host, remote_path, local_path):
    require(host in ('5080', '5090') and remote_path.startswith('/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/'), 'Foreign transfer owner/root')
    local_path = Path(local_path)
    require(not local_path.exists(), 'Preserved archive already exists; no retry')
    run = subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host + ':' + remote_path, str(local_path)], capture_output=True, text=True, timeout=1800)
    require(run.returncode == 0, 'Originals transfer failed; no automatic retry: ' + run.stderr[-1200:])


def recover_host(c, host, bundle=BUNDLE, remote=None, transfer=None):
    base = bundle / 'originals' / host
    receipt_path = base / 'receipt.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        require(receipt['status'] == 'verified' and receipt['contract_sha256'] == canonical(c) and receipt['host'] == host and receipt['root'] == c['hosts'][host]['root'] and receipt['held_out_test_evaluated'] is False and receipt['actual_owner_absence_verified'] is True, 'Retained archive receipt identity drift')
        require(sha(base / 'originals.tar.gz') == receipt['archive_sha256'] and sha(base / 'verified/manifest.json') == receipt['manifest_sha256'], 'Retained originals archive/manifest drift')
        verify_original_files(base / 'verified/originals', read(base / 'verified/manifest.json'))
        return {'host': host, 'status': 'already_verified_local_originals', 'remote_calls': 0, 'receipt_raw_sha256': sha(receipt_path)}
    if (base / 'claim.json').exists() or (base / 'failure.json').exists():
        return {'host': host, 'status': 'claimed_or_failed_preserved', 'remote_calls': 0, 'user_action_required': True}
    evidence = terminal_cache(c, host, bundle)
    if evidence is None:
        return {'host': host, 'status': 'pending', 'remote_calls': 0, 'reason': 'Scientific terminal, actual dispatch exit and owner absence not all verified'}
    write_new(base / 'claim.json', {'host': host, 'contract_sha256': canonical(c), 'created_unix': time.time(), 'automatic_retry': False, 'terminal_evidence': evidence})
    remote_calls = 0
    try:
        code = export_code(c, host, evidence, bundle)
        remote_calls += 1
        receipt = json.loads((remote or deployment.remote)(host, code, timeout=1200))
        require(receipt['archive_path'] == c['hosts'][host]['root'] + '/preservation/originals.tar.gz', 'Foreign remote archive path')
        write_new(base / 'remote_archive_receipt.json', receipt)
        remote_calls += 1
        (transfer or copy_from)(host, receipt['archive_path'], base / 'originals.tar.gz')
        verify_archive(base / 'originals.tar.gz', base / 'verified', receipt, c, host)
        write_new(receipt_path, {**receipt, 'status': 'verified', 'local_verified_unix': time.time(), 'snapshot_binding': evidence, 'evaluation_performed': False, 'CPU_binary_reinference_audit': False})
        return {'host': host, 'status': 'verified_originals_recovered', 'remote_calls': 2, 'receipt_raw_sha256': sha(receipt_path)}
    except BaseException as exc:
        write_new(base / 'failure.json', {'type': type(exc).__name__, 'message': str(exc), 'created_unix': time.time(), 'remote_calls': remote_calls, 'automatic_retry': False})
        raise


def selected_record(folder, job, c, expected_contract, remote_folder, terminal_sha, provenance):
    manifest_path = folder / 'terminal_manifest.json'
    require(sha(manifest_path) == terminal_sha, 'Original terminal differs from observed terminal')
    terminal = read(manifest_path)
    require(terminal['job'] == job and terminal['contract_sha256'] == expected_contract and terminal['scientific_success'] is True and terminal['status'] == 'complete' and terminal['held_out_test_evaluated'] is False, 'Foreign scientific original')
    for name, digest in terminal['files'].items():
        no_held_out('run/' + name)
        require(sha(safe(folder, name)) == digest, 'Scientific file raw SHA drift: ' + name)
    run = folder / 'runs' / job['arm'] / VARIANT / ('seed_' + str(job['seed']))
    endpoints = read(run / 'endpoint_replays.json')
    history = read(run / 'history.json')['history']
    n = endpoints['completed_epochs']
    require(endpoints['job'] == job and endpoints['evaluation_scope'] == 'validation_only' and endpoints['held_out_test_evaluated'] is False, 'Endpoint job/split drift')
    require(len(history) == n and list(r['epoch'] for r in history) == list(range(1, n + 1)) and c['training']['minimum_epochs'] <= n <= c['training']['maximum_epochs'], 'Completed history epoch drift')
    finite = [r for r in history if type(r.get('val_qty_rmse')) in (int, float) and math.isfinite(r['val_qty_rmse'])]
    require(bool(finite), 'No finite quantity selector')
    best = min(finite, key=lambda r: r['val_qty_rmse'])
    require(best['epoch'] == endpoints['best_epoch'], 'Earliest strict finite minimum epoch changed')
    data = next(d for d in c['datasets'] if d['dataset_id'] == job['dataset'])
    binary = {}
    for label, epoch in (('selected', best['epoch']), ('last', n)):
        endpoint = endpoints[label]
        require(endpoint['evaluation_scope'] == 'validation_only' and endpoint['held_out_test_evaluated'] is False and endpoint['count'] == data['inherited_data_identity']['populations']['validation']['target_count'] and endpoint['quantity_boundaries'] == data['quantity_boundaries_all_train_rows'], 'Full Validation population/tail/split changed')
        for cell in (endpoint, endpoint['tail']):
            require(cell['count'] > 0 and all(type(cell[k]) in (int, float) and math.isfinite(cell[k]) for k in METRICS), 'Nonfinite same-epoch endpoint metric')
        for k in METRICS:
            require(math.isclose(endpoint[k], history[epoch - 1]['val_' + k], rel_tol=1e-10, abs_tol=1e-8), 'Same-epoch Validation metric mismatch')
        checkpoint = Path(endpoint['checkpoint_path'])
        remote_run = Path(remote_folder) / 'runs' / job['arm'] / VARIANT / ('seed_' + str(job['seed']))
        require(checkpoint.is_relative_to(remote_run), 'Foreign selected/last checkpoint path')
        name = checkpoint.relative_to(remote_folder).as_posix()
        require(name in terminal['files'] and sha(safe(folder, name)) == terminal['files'][name], 'Selected/last binary missing or changed')
        binary[label] = {'relative_path': name, 'raw_sha256': terminal['files'][name]}
    initial = read(folder / 'initialization.json')[job['arm']]
    require(initial == endpoints['initial_state_sha256'], 'Initial tensor binding drift')
    exposure = read(run / 'exposure.json')
    require(len(exposure['train']) == n and len(exposure['validation']) >= n, 'Train/Validation exposure missing')
    timing = read(run / 'epoch_timing.json')
    require([r['epoch'] for r in timing['epochs']] == list(range(1, n + 1)) and all(math.isfinite(r['elapsed_seconds']) and r['elapsed_seconds'] > 0 for r in timing['epochs']), 'Own completed-epoch timing missing')
    return {**job, 'best_epoch': best['epoch'], 'last_epoch': n, 'selected': endpoints['selected'], 'last': endpoints['last'], 'initial_state_sha256': initial, 'parameter_count': endpoints['parameter_count'], 'terminal_manifest_sha256': terminal_sha, 'endpoint_raw_sha256': sha(run / 'endpoint_replays.json'), 'binary_originals': binary, 'train_exposure': exposure, 'epoch_timing': timing, 'provenance': provenance}


def parent_records(c, a100=A100, bundle=BUNDLE):
    base = a100 / 'originals/partial'
    if not (base / 'receipt.json').exists():
        return None
    parent = read(a100 / 'execution_contract.json')
    require(canonical(parent) == A100_CONTRACT == c['handoff']['parent_contract_sha256'], 'Parent science contract drift')
    require(parent['training'] == c['training'] and parent['source']['files'] == c['source']['files'] and parent['input_files'] == c['input_files'] and parent['datasets'] == c['datasets'], 'Handoff shared science/data changed')
    proof = read(bundle / 'handoff/ownership_release_source.json')
    receipt = read(base / 'receipt.json')
    require(receipt['status'] == 'verified' and receipt['mode'] == 'partial' and receipt['scientific_success'] is False and receipt['contract_sha256'] == A100_CONTRACT, 'Parent partial receipt mislabeled')
    require(sha(base / 'originals.tar.gz') == receipt['archive_sha256'] and sha(base / 'verified/manifest.json') == receipt['manifest_sha256'], 'Parent originals archive/manifest drift')
    manifest = read(base / 'verified/manifest.json')
    require(manifest['schema'] == 'titantpp_a100_originals_v1' and manifest['partial_is_not_success'] is True and manifest['verified_scientific_terminal_count'] == 15 and manifest['terminal_manifest_sha256'] == proof['terminal_manifest_sha256'], 'Parent15 scientific terminal identity drift')
    require(not manifest['actual_idle_observation']['owned_processes'] and not manifest['actual_idle_observation']['gpu_pids'], 'Parent owner absence missing')
    root = base / 'verified/originals'
    verify_original_files(root, manifest)
    require(canonical(read(root / 'execution_contract.json')) == A100_CONTRACT, 'Archived A100 scientific contract changed')
    for name, digest in {**parent['source']['files'], **parent['input_files']}.items():
        require(manifest['files'].get('source/' + name, {}).get('sha256') == digest, 'Parent frozen source/input binding drift')
    for name, digest in parent['operation']['files'].items():
        require(manifest['files'].get(name, {}).get('sha256') == digest, 'Parent operation binding drift')
    for name, key in (('supervisor_exit.json', 'supervisor_exit_raw_sha256'), ('failure.json', 'failure_raw_sha256'), ('pipeline_failure.json', 'pipeline_failure_raw_sha256')):
        require(sha(root / name) == proof[key], 'Preserved A100 planned handoff exit/failure drift')
    require(read(root / 'supervisor_exit.json')['returncode'] == 1, 'Parent planned operational exit erased')
    recovery = parent['recovery']
    require(sha(root / 'recovery_parent/parent_archive.tar.gz') == recovery['parent_archive_sha256'] and sha(root / 'recovery_parent/parent_manifest.json') == recovery['parent_manifest_sha256'], 'Parent10 archive provenance drift')
    for name, digest in recovery['parent_files'].items():
        require(sha(safe(root / 'recovery_parent', name)) == digest, 'Parent original scientific file drift')
    old_manifest = read(root / 'recovery_parent/parent_manifest.json')
    require(old_manifest['contract_sha256'] == recovery['parent_contract_sha256'] and old_manifest['root'] == '/workspace/titantpp_contribution_history_a100_parallel3_20261008_v1_a100', 'Original A100 contract/root drift')
    retained = {j['id']: j for j in parent['jobs'] if j['id'] in proof['terminal_manifest_sha256']}
    require(len(retained) == 15 and set(retained).isdisjoint(j['id'] for j in c['jobs']) and len(set(retained) | {j['id'] for j in c['jobs']}) == 27, 'Original27 execution IDs duplicated or missing')
    records = []
    for job in retained.values():
        reused = job['id'] in recovery['reused_jobs']
        prefix = 'recovery_parent/run/' if reused else 'run/'
        remote_root = old_manifest['root'] if reused else parent['hosts']['a100']['root']
        origin = {'owner': 'a100_original' if reused else 'a100_recovery', 'contract_sha256': recovery['parent_contract_sha256'] if reused else A100_CONTRACT, 'gpu_uuid': recovery['source_gpu_uuid'] if reused else parent['hosts']['a100']['gpu_uuid'], 'archive_sha256': receipt['archive_sha256'], 'manifest_sha256': receipt['manifest_sha256'], 'RAF42_physical_GPU_migration': job['dataset'] == 'raf_spare_parts' and job['seed'] == 42, 'timing_not_mixed_between_physical_GPUs': True}
        record = selected_record(root / prefix / job['id'], job, parent, origin['contract_sha256'], remote_root + '/run/' + job['id'], proof['terminal_manifest_sha256'][job['id']], origin)
        if job['id'] in recovery['resumed_jobs']:
            resume = recovery['resumed_jobs'][job['id']]
            epoch = resume['epoch']
            old_run = root / 'recovery_parent/run' / job['id'] / 'runs' / job['arm'] / VARIANT / ('seed_' + str(job['seed']))
            require(sha(old_run / 'server_checkpoint_receipt.json') == resume['receipt_sha256'] and sha(old_run / 'last_epoch_state.pt') == resume['last_checkpoint_sha256'], 'Frozen resume checkpoint/receipt raw binding changed')
            bridge = read(root / 'run' / job['id'] / 'checkpointreceipt_parent.json')
            require(bridge['parent'] == resume and bridge['parent_contract_sha256'] == recovery['parent_contract_sha256'] and bridge['recovery_contract_sha256'] == A100_CONTRACT and bridge['source_gpu_uuid'] == recovery['source_gpu_uuid'] and bridge['destination_gpu_uuid'] == parent['hosts']['a100']['gpu_uuid'] and bridge['physical_gpu_migration'] is True, 'Resume parent checkpoint receipt bridge changed')
            require(read(old_run / 'history.json')['history'] == read(root / 'run' / job['id'] / 'runs' / job['arm'] / VARIANT / ('seed_' + str(job['seed'])) / 'history.json')['history'][:epoch], 'Resume history prefix continuity lost')
            require(read(old_run / 'epoch_timing.json')['epochs'] == record['epoch_timing']['epochs'][:epoch], 'Resume timing prefix continuity lost')
            record['provenance']['resume_from_epoch'] = epoch
            record['provenance']['timing_segments'] = [{'gpu_uuid': recovery['source_gpu_uuid'], 'epochs': [1, epoch]}, {'gpu_uuid': parent['hosts']['a100']['gpu_uuid'], 'epochs': [epoch + 1, record['last_epoch']]}]
        records.append(record)
    return records


def rtx_records(c, bundle=BUNDLE):
    records = []
    for host in ('5080', '5090'):
        base = bundle / 'originals' / host
        if not (base / 'receipt.json').exists():
            return None
        receipt = read(base / 'receipt.json')
        manifest = read(base / 'verified/manifest.json')
        verify_original_files(base / 'verified/originals', manifest)
        for job in (j for j in c['jobs'] if j['host'] == host):
            root = base / 'verified/originals'
            provenance = {'owner': host, 'contract_sha256': canonical(c), 'gpu_uuid': c['hosts'][host]['gpu_uuid'], 'archive_sha256': receipt['archive_sha256'], 'manifest_sha256': receipt['manifest_sha256'], 'RAF42_physical_GPU_migration': False, 'original_observed_kst': receipt['snapshot_binding']['observed_kst'], 'timing_not_mixed_between_physical_GPUs': True}
            records.append(selected_record(root / 'run' / job['id'], job, c, canonical(c), c['hosts'][host]['root'] + '/run/' + job['id'], manifest['terminal_manifest_sha256'][job['id']], provenance))
    return records


def archived_json(archive, name, manifest):
    """Read one allowed metadata file from the already SHA-verified parent tar."""
    no_held_out(name)
    require(name in manifest['files'], 'Archived native metadata missing')
    with tarfile.open(archive, 'r:gz') as tar:
        member = tar.getmember('originals/' + name)
        require(member.isfile(), 'Nonregular archived metadata')
        raw = tar.extractfile(member).read()
    require(digest_bytes(raw) == manifest['files'][name]['sha256'], 'Archived native metadata raw SHA changed')
    return json.loads(raw)


def digest_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def native_costs(c, bundle=BUNDLE, a100=A100):
    parent_root = a100 / 'originals/partial/verified/originals'
    parent = read(a100 / 'execution_contract.json')
    parent_manifest = read(parent_root / 'recovery_parent/parent_manifest.json')
    old_q = archived_json(parent_root / 'recovery_parent/parent_archive.tar.gz', 'qualification/receipt.json', parent_manifest)
    qualifications = [('a100_original', parent['recovery']['source_gpu_uuid'], old_q), ('a100_recovery', parent['hosts']['a100']['gpu_uuid'], read(parent_root / 'qualification/receipt.json'))]
    for h in ('5080', '5090'):
        qualifications.append((h, c['hosts'][h]['gpu_uuid'], read(bundle / 'originals' / h / 'verified/originals/qualification/receipt.json')))
    output = []
    for owner, uuid, qualification in qualifications:
        require(qualification['status'] == 'passed' and qualification['runtime']['gpu']['uuid'] == uuid and qualification['held_out_test_evaluated'] is False, 'Native qualification identity/split changed')
        for dataset, lengths in qualification['measurements'].items():
            for length, arms in lengths.items():
                reference = arms[ARMS[0]]
                for arm, values in arms.items():
                    keys = ('inference_median_seconds', 'training_step_median_seconds', 'peak_allocated_bytes')
                    require(all(type(values[k]) in (int, float) and math.isfinite(values[k]) and values[k] > 0 and reference[k] > 0 for k in keys), 'Invalid native cost measurements')
                    output.append({'owner': owner, 'gpu_uuid': uuid, 'dataset': dataset, 'length': int(length), 'arm': arm, 'measurements': {k: values[k] for k in keys}, 'ratios_vs_same_GPU_full': {k: values[k] / reference[k] for k in keys}, 'scope': 'original isolated native qualification, not concurrent fit throughput', 'Q1_cost_success_gate': False})
    return output


def summarize(records):
    require(len(records) == 27 and len({r['id'] for r in records}) == 27, 'Exactly27 distinct original execution IDs required')
    cells = {(r['dataset'], r['seed'], r['arm']): r for r in records}
    require(len(cells) == 27 and set(r['arm'] for r in records) == set(ARMS) and set(r['seed'] for r in records) == set(SEEDS) and len({r['dataset'] for r in records}) == 3, 'Scientific matrix incomplete')
    groups = []
    details, supported = [], []
    for dataset in sorted({r['dataset'] for r in records}):
        wins = []
        for seed in SEEDS:
            paired = [cells[(dataset, seed, arm)] for arm in ARMS]
            # Paired arms share dataset/seed exposures; migrated RAF42 is marked.
            for left, right in zip(paired, paired[1:]):
                a, b = left['train_exposure'], right['train_exposure']
                n = min(len(a['train']), len(b['train']))
                require(n > 0 and a['train'][:n] == b['train'][:n], 'Paired Train order/exposure changed')
                require(len({r['batch_order_sha256'] for rows in (a['validation'], b['validation']) for r in rows}) == 1, 'Paired full Validation order changed')
            full = paired[0]
            comparisons = {}
            for control in paired[1:]:
                comparisons[control['arm']] = {'overall_RMSE_strict_improvement': full['selected']['qty_rmse'] < control['selected']['qty_rmse'], **{prefix + '_' + k + '_delta': (full['selected'] if prefix == 'overall' else full['selected']['tail'])[k] - (control['selected'] if prefix == 'overall' else control['selected']['tail'])[k] for prefix in ('overall', 'tail') for k in METRICS}}
            wins.append(all(x['overall_RMSE_strict_improvement'] for x in comparisons.values()))
            details.append({'dataset': dataset, 'seed': seed, 'comparisons': comparisons, 'physical_GPU_migration': any(r['provenance'].get('RAF42_physical_GPU_migration') for r in paired)})
        if all(wins):
            supported.append(dataset)
        for arm in ARMS:
            group = [cells[(dataset, seed, arm)] for seed in SEEDS]
            metrics = {}
            for prefix in ('overall', 'tail'):
                for k in METRICS:
                    values = [(r['selected'] if prefix == 'overall' else r['selected']['tail'])[k] for r in group]
                    metrics[prefix + '_' + k] = {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values), 'seeds': list(SEEDS)}
            groups.append({'dataset': dataset, 'arm': arm, 'metrics': metrics})
    return {'seed_summary': groups, 'decision': {'Q1_development_support': len(supported) >= 2, 'same_datasets_all3seed_strict_RMSE_wins': supported, 'paired_results': details, 'independent_confirmation': False, 'statistical_significance_established': False, 'RAF42_physical_GPU_migration': True}}


def report_text(records, summary):
    lines = ['# CNN+GRU54 이력 보정 27조건 최종 Validation 비교', '', '각 실행의 최초 strict finite 최소 전체 raw 수량 RMSE epoch에서 전체·큰 수량 RMSE/MAE/TimeNLL을 함께 보고한다. 부모 A100 완료15와 RTX 완료12의 원래 실행 ID를 각각 한 번 집계했다.', '', '|데이터|방식|seed|소유 GPU|선택/최종 epoch|전체 RMSE|전체 MAE|전체 TimeNLL|큰 수량 RMSE|큰 수량 MAE|큰 수량 TimeNLL|', '|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted(records, key=lambda x: (x['dataset'], x['arm'], x['seed'])):
        values = [r['selected'][k] for k in METRICS] + [r['selected']['tail'][k] for k in METRICS]
        lines.append('|' + '|'.join([r['dataset'], r['arm'], str(r['seed']), r['provenance']['owner'], f"{r['best_epoch']}/{r['last_epoch']}", *(f'{v:.7g}' for v in values)]) + '|')
    lines += ['', '3seed 평균 ± 표본 표준편차:', '', '|데이터|방식|전체 RMSE|전체 MAE|전체 TimeNLL|큰 수량 RMSE|큰 수량 MAE|큰 수량 TimeNLL|', '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in summary['seed_summary']:
        lines.append('|' + '|'.join([row['dataset'], row['arm'], *(f"{v['mean']:.7g} ± {v['sample_sd']:.7g}" for v in row['metrics'].values())]) + '|')
    decision = summary['decision']
    lines += ['', f"고정 Q1 개발 기준 충족: **{decision['Q1_development_support']}**. full이 두 대조군을 같은 데이터의 모든3seed에서 전체 RMSE로 strict 개선한 데이터: {decision['same_datasets_all3seed_strict_RMSE_wins']}.", '', '수량 변화와 같은 선택 epoch의 시간 손해:', '', '|데이터|seed|대조|전체 RMSE Δ|큰 수량 RMSE Δ|전체 MAE Δ|큰 수량 MAE Δ|전체 TimeNLL Δ|큰 수량 TimeNLL Δ|', '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in decision['paired_results']:
        for arm, values in row['comparisons'].items():
            keys = ('overall_qty_rmse_delta', 'tail_qty_rmse_delta', 'overall_qty_mae_delta', 'tail_qty_mae_delta', 'overall_time_nll_delta', 'tail_time_nll_delta')
            lines.append('|' + '|'.join([row['dataset'], str(row['seed']), arm, *(f'{values[k]:+.7g}' for k in keys)]) + '|')
    lines += ['', '동일 물리 GPU의 full 대비 native qualification 계산비용 (Q1 성공 판정에 사후 비용 허용치를 추가하지 않음):', '', '|소유 GPU|데이터|길이|방식|추론 비율|학습 step 비율|peak 메모리 비율|', '|---|---|---:|---|---:|---:|---:|']
    for cost in summary.get('native_cost', []):
        ratios = cost['ratios_vs_same_GPU_full']
        lines.append('|' + '|'.join([cost['owner'], cost['dataset'], str(cost['length']), cost['arm'], *(f'{ratios[k]:.5g}' for k in ('inference_median_seconds', 'training_step_median_seconds', 'peak_allocated_bytes'))]) + '|')
    lines += ['', 'Δ는 full − 대조군이며 양수는 손해다. 불리한 조건도 모두 보존했다.', '', '실행 및 해석 범위:', '', '- 실제 dispatch supervisor와 모든 scientific worker의 exit0, pipeline completion 및 owner/GPU PID 부재를 검증했다. pipeline OS 자체의 exit 코드는 별도 관측하지 않았다 (`pipeline_exit_code_directly_observed=false`).', '- 부모 A100은 계획된 인계에서 운영 exit1로 종료했다. 완료15의 과학 원본만 재사용했고, 부모 failure와 partial 원본을 보존했다. 부모 전체 완료로 변경하지 않았다.', '- RAF seed42의 full/current_only는 물리 A100 GPU를 이관해 재개했다. RAF seed62는5080, Intermittent의 세 seed는5090이다. GPU/Runtime 차이는 JSON provenance에 기록했다.', '- 물리 GPU별 epoch timing 원본을 보존하고 합산하지 않았다. 동시3개 실행 시간을 구조 효율로 해석하지 않는다. 현재 상태 대조군의 hidden에는 prefix 이력이 포함되며, 활성 파라미터 수를 맞추어도 activation/depth/cost가 같다거나 이력이 완전히 제거됐다는 뜻은 아니다.', '- 기존 Test 관찰에서 생긴 가설의 Validation 개발 증거다. 참신성·통계적 유의성·외부 모델 우월성·새 독립 평가의 증명이 아니다.', '- CPU binary 재추론 감사, Test, 새 독립 평가는 미완료다. 고정 검색 후속 학습은 별도 자원·공통 mean 계약 전 대기다.', '']
    return '\n'.join(lines)


def verify_completion(receipt, bundle=BUNDLE):
    require(receipt['contract_sha256'] == CONTRACT and receipt['distinct_execution_ids'] == 27 and receipt['held_out_test_evaluated'] is False, 'Completion receipt identity drift')
    for name, digest in receipt['local_files'].items():
        require(sha(safe(bundle, name)) == digest, 'Finalized local raw SHA drift: ' + name)
    for name, digest in receipt['external_local_files'].items():
        path = Path(name)
        require(path.resolve().is_relative_to(A100.resolve()) and sha(path) == digest, 'Parent immutable local evidence drift')


def finalize(host=None, *, bundle=BUNDLE, verifier=validate_local, recover=recover_host):
    require(host in (None, '5080', '5090'), 'Unknown host')
    c = verifier(bundle)
    complete = bundle / 'analysis/completion_receipt.json'
    if complete.exists():
        receipt = read(complete)
        verify_completion(receipt, bundle)
        for h in ('5080', '5090'):
            recover(c, h, bundle)
        parent_records(c, bundle=bundle)
        return {'status': 'already_finalized_local_rawSHA_verified', 'remote_calls': 0, 'decision': receipt['decision']}
    results = []
    for h in (host,) if host else ('5080', '5090'):
        try:
            results.append(recover(c, h, bundle))
        except Exception as exc:
            failure = bundle / 'originals' / h / 'failure.json'
            calls = read(failure).get('remote_calls', 0) if failure.exists() else 0
            results.append({'host': h, 'status': 'integrity_or_retrieval_failure', 'message': str(exc), 'remote_calls': calls, 'user_action_required': True})
    remote_calls = sum(r.get('remote_calls', 0) for r in results)
    if host:
        return {'status': 'host_originals_check', 'hosts': results, 'remote_calls': remote_calls}
    if not all(r['status'] in ('already_verified_local_originals', 'verified_originals_recovered') for r in results):
        return {'status': 'pending', 'hosts': results, 'remote_calls': remote_calls, 'user_action_required': any(r.get('user_action_required') for r in results)}
    parent = parent_records(c, bundle=bundle)
    if parent is None:
        return {'status': 'pending_parent_immutable_originals', 'hosts': results, 'remote_calls': remote_calls}
    claim = bundle / 'analysis/finalize.claim'
    if claim.exists():
        return {'status': 'analysis_claimed_or_failed_preserved', 'hosts': results, 'remote_calls': remote_calls, 'user_action_required': True}
    write_new(claim, {'contract_sha256': CONTRACT, 'created_unix': time.time(), 'automatic_retry': False})
    try:
        records = parent + rtx_records(c, bundle)
        summary = summarize(records)
        summary['native_cost'] = native_costs(c, bundle)
        analysis = bundle / 'analysis'
        write_new(analysis / 'selected_conditions.json', {'evaluation_scope': 'validation_only', 'held_out_test_evaluated': False, 'conditions': records, **summary})
        report = analysis / 'VALIDATION_HISTORY_REPORT.md'
        with report.open('x') as stream:
            stream.write(report_text(records, summary))
        local = {p.relative_to(bundle).as_posix(): sha(p) for p in (analysis / 'selected_conditions.json', report, bundle / 'ownership_release.json', bundle / 'handoff/ownership_release_source.json')}
        for h in ('5080', '5090'):
            base = bundle / 'originals' / h
            for path in (base / 'receipt.json', base / 'originals.tar.gz', base / 'verified/manifest.json'):
                local[path.relative_to(bundle).as_posix()] = sha(path)
        external = {str(p): sha(p) for p in (A100 / 'originals/partial/receipt.json', A100 / 'originals/partial/originals.tar.gz', A100 / 'originals/partial/verified/manifest.json')}
        receipt = {'status': 'complete', 'contract_sha256': CONTRACT, 'distinct_execution_ids': 27, 'parent_scientific_terminals': 15, 'RTX_scientific_terminals': 12, 'decision': summary['decision'], 'local_files': local, 'external_local_files': external, 'completed_unix': time.time(), 'completed_kst': datetime.now(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST'), 'actual_owner_absence_verified': True, 'dispatch_actual_exit_code': 0, 'pipeline_exit_code_directly_observed': False, 'held_out_test_evaluated': False, 'CPU_binary_reinference_audit': False, 'new_independent_evaluation': False, 'parent_partial_preserved': True, 'automatic_retry': False}
        write_new(complete, receipt)
        return {'status': 'complete', 'hosts': results, 'remote_calls': remote_calls, 'decision': summary['decision'], 'report': str(report)}
    except BaseException as exc:
        write_new(bundle / 'analysis/failure.json', {'type': type(exc).__name__, 'message': str(exc), 'automatic_retry': False, 'created_unix': time.time()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--all', action='store_true')
    mode.add_argument('--host', choices=('5080', '5090'))
    args = parser.parse_args()
    print(json.dumps(finalize(args.host), ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
