"""One read-only observation; exact evidence and process ownership, no retries."""
from pathlib import Path
import datetime
import hashlib
import inspect
import json
import math
import shlex
import subprocess

BUNDLE = Path(__file__).resolve().parents[1]


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def safe_file(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe manifest path')
    path = root / relative
    require(not path.is_symlink() and path.resolve().is_relative_to(root), 'Manifest symlink/path escape')
    require(path.is_file(), 'Missing manifest file: ' + str(relative))
    return path


def verify_files(root, files, required=()):
    require(isinstance(files, dict) and set(required) <= set(files), 'Incomplete manifest')
    for relative, expected in files.items():
        require(valid_sha(expected) and sha(safe_file(root, relative)) == expected, 'File SHA mismatch: ' + relative)


def verify_evidence(c, job, documents):
    """Validate the JSON proof without loading checkpoint tensors or data."""
    r, t, s, h, e, inp = (documents[k] for k in ('result', 'terminal', 'startup', 'history', 'endpoint', 'input'))
    expected = digest(c)
    require(r['schema'] == 'titantpp_time_head_bias_refit_v1', 'Wrong result schema')
    require(r['status'] == t['status'] == 'complete' and t['scientific_success'] is True, 'Incomplete worker terminal')
    require(r['job'] == job['id'] and t['job'] == job and s['job'] == job['id'], 'Wrong job identity')
    require(all(doc['contract_sha256'] == expected for doc in (r, t, s)), 'Wrong worker contract')
    require(all(doc['held_out_test_evaluated'] is False for doc in (r, t, s, h, e)), 'Wrong evaluation scope')
    require(all(doc['evaluation_scope'] == 'validation_only' for doc in (r, t, s, h, e)), 'Wrong evaluation split')
    require(r['original_checkpoint'] == s['original_checkpoint'] == job['checkpoint'], 'Original checkpoint binding changed')
    require(t['original_checkpoint_sha_preserved'] is True, 'Original checkpoint preservation absent')
    require(r['source_files_sha256'] == c['source']['files_sha256'], 'Source closure changed')
    require(r['trainable_tensors'] == ['b_t', 'w_raw'] and all(doc['trainable_scalar_count'] == 2 for doc in (r, t, e)), 'Trainable scope changed')
    frozen = r['frozen_state_sha256_before']
    require(valid_sha(frozen) and all(v == frozen for v in (r['frozen_state_sha256_after'], r['frozen_state_sha256'], t['frozen_state_sha256'], s['frozen_state_sha256'], e['frozen_state_sha256'])), 'Frozen state changed')
    quantity = r['quantity_prediction_sha256_before']
    require(set(quantity) == {'train', 'validation'} and all(valid_sha(v) for v in quantity.values()), 'Missing quantity hashes')
    require(quantity == r['quantity_prediction_sha256_after'] == s['quantity_prediction_sha256'], 'Quantity prediction SHA changed')
    require(r['quantity_output_unchanged'] is True and r['quantity_predictions_exactly_equal'] is True and e['quantity_predictions_exactly_equal'] is True, 'Quantity preservation absent')
    for actual, original in ((r['E0'], job['baseline_metrics']), (r['E0']['tail'], job['baseline_metrics']['tail'])):
        require(actual['count'] == original['count'], 'Original E0 population mismatch')
        for name in ('qty_rmse', 'qty_mae', 'time_nll'):
            require(math.isclose(actual[name], original[name], rel_tol=1e-5, abs_tol=1e-5), 'Original E0 metric mismatch')
    require(s['status'] == 'passed' and s['input_receipt_sha256'] == t['files']['input_receipt.json'], 'Startup receipt binding changed')
    data = next(d for d in c['datasets'] if d['dataset_id'] == job['dataset'])
    require(inp['held_out_materialized'] is False and inp['populations'] == data['inherited_data_identity']['populations'], 'Input population changed')
    require(s['runtime'] == inp['refit_runtime'] and s['runtime']['gpu']['uuid'] == c['runtime']['gpu_uuid'], 'Runtime GPU identity changed')
    for key, value in c['runtime']['runtime_expected'].items():
        require(s['runtime'].get(key) == value, 'Runtime version changed: ' + key)
    rows = h['history']
    require(rows and [row['epoch'] for row in rows] == list(range(len(rows))), 'Missing/reordered E0 history')
    require(1 <= rows[-1]['epoch'] <= c['fitting']['maximum_epochs'], 'Wrong completed epoch count')
    require(all(math.isfinite(row['validation']['time_nll']) for row in rows), 'Nonfinite Validation NLL')
    selected = min(rows, key=lambda row: row['validation']['time_nll'])
    require(r['best_epoch'] == selected['epoch'] and r['completed_epochs'] == rows[-1]['epoch'], 'Earliest minimum selector changed')
    for doc in (r, e):
        require(doc['E0'] == rows[0]['validation'] == s['E0'] and doc['selected'] == selected['validation'] and doc['last'] == rows[-1]['validation'], 'Endpoint/history mismatch')
        require(doc['best_epoch'] == selected['epoch'] and doc['completed_epochs'] == rows[-1]['epoch'], 'Endpoint epoch mismatch')
        require(doc['selected_is_identity'] == (selected['epoch'] == 0), 'Identity fallback mismatch')
    require(r['selected']['time_nll'] <= r['E0']['time_nll'], 'Identity fallback lost')
    require(rows[-1]['epoch'] == c['fitting']['maximum_epochs'] or rows[-1]['epoch'] - selected['epoch'] == c['fitting']['patience'], 'Wrong early-stop endpoint')
    for row in rows:
        for key in ('qty_rmse', 'qty_mae'):
            require(row['validation'][key] == r['E0'][key], 'Quantity metric changed')
            require(row['validation']['tail'][key] == r['E0']['tail'][key], 'Tail quantity metric changed')
    return r


def verify_job(c, job, directory, *, verify_original=True):
    directory = Path(directory)
    require(not (directory/'failure.json').exists(), 'Worker failure exists')
    names = {'result':'result.json', 'terminal':'terminal_manifest.json', 'startup':'startup_gate.json',
        'history':'history.json', 'endpoint':'endpoint_replays.json', 'input':'input_receipt.json'}
    docs = {key: read(safe_file(directory, name)) for key, name in names.items()}
    required = ('result.json', 'startup_gate.json', 'history.json', 'endpoint_replays.json',
        'input_receipt.json', 'selected_refit.pt', 'last_refit.pt', 'training_permit.json', 'worker_process.json')
    verify_files(directory, docs['terminal']['files'], required)
    supervisor = read(safe_file(directory, 'supervisor_terminal_receipt.json'))
    require(supervisor['status'] == 'complete' and supervisor['scientific_success'] is True
        and supervisor['job'] == job['id'] and supervisor['contract_sha256'] == digest(c)
        and supervisor['held_out_test_evaluated'] is False, 'Wrong supervisor terminal')
    verify_files(directory, supervisor['files'], required + ('terminal_manifest.json',))
    verify_evidence(c, job, docs)
    if verify_original:
        require(sha(job['checkpoint']['path']) == job['checkpoint']['sha256'], 'Original checkpoint changed')
    return {'result': docs['result'], 'proof': docs, 'supervisor_terminal': supervisor,
        'worker_process': read(directory/'worker_process.json'), 'file_SHA_verified': True,
        'selected_checkpoint_sha256': docs['terminal']['files']['selected_refit.pt'],
        'last_checkpoint_sha256': docs['terminal']['files']['last_refit.pt'],
        'worker_terminal_sha256': sha(directory/'terminal_manifest.json')}


def parse_processes(lines):
    result = {}
    for line in lines:
        parts = line.strip().split(None, 2)
        if len(parts) != 3 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        result[int(parts[0])] = {'pid':int(parts[0]), 'ppid':int(parts[1]), 'argv':shlex.split(parts[2])}
    return result


def flag(argv, name):
    return argv[argv.index(name)+1] if name in argv and argv.index(name)+1 < len(argv) else None


def process_identity(c, root, processes, gpu_pids, supervisor, rows):
    """Exact argv and parent linkage; an unbound process never proves progress."""
    root = Path(root)
    controller = str(root/'operation/controller.py')
    worker = str(root/'operation/titantpp_time_head_refit_runtime.py')
    owned = [p for p in processes.values() if controller in p['argv'] or worker in p['argv']]
    issues = []
    owner_pid = supervisor.get('pid') if supervisor else None
    controllers = [p for p in owned if controller in p['argv']]
    workers = [p for p in owned if worker in p['argv']]
    for p in controllers:
        a = p['argv']
        if not (p['pid'] == owner_pid and flag(a,'--contract') == str(root/'execution_contract.json') and flag(a,'--mode') == 'dispatch'):
            issues.append('Controller PID/argv mismatch')
    if len(controllers) > 1 or len(workers) > 1:
        issues.append('Multiple owned processes')
    for p in workers:
        a = p['argv']; job_id = flag(a, '--job')
        candidates = [row for row in rows if row['job']['id'] == job_id]
        wp = candidates[0].get('worker_process', {}) if len(candidates) == 1 else {}
        if not (len(candidates) == 1 and p['pid'] == wp.get('pid') and p['ppid'] == wp.get('ppid') == owner_pid
            and flag(a,'--owner-pid') == str(owner_pid) and flag(a,'--contract') == str(root/'execution_contract.json')
            and flag(a,'--training-permit') == str(Path(candidates[0]['job']['output_dir'])/'training_permit.json')
            and a == wp.get('command') and len(controllers) == 1):
            issues.append('Worker PID/argv/parent mismatch')
    running = [row for row in rows if row['state'] == 'running']
    for row in running:
        matches = [p for p in workers if flag(p['argv'], '--job') == row['job']['id']]
        if len(matches) != 1:
            issues.append('Running row without matching worker')
        elif matches[0]['pid'] not in gpu_pids:
            issues.append('Running worker not present in GPU process list')
    completed_pids = {row.get('worker_process', {}).get('pid') for row in rows if row['state'] == 'complete'}
    if any(pid in processes or pid in gpu_pids for pid in completed_pids if pid is not None):
        issues.append('Completed worker PID is still present (or reused)')
    owned_ids = {p['pid'] for p in owned}
    return owned, issues, sorted(owned_ids & gpu_pids)


def progress_history(c, job, document):
    require(document['evaluation_scope'] == 'validation_only' and document['held_out_test_evaluated'] is False, 'Progress history scope changed')
    rows = document['history']
    require(rows and [r['epoch'] for r in rows] == list(range(len(rows))), 'Progress history E0/sequence changed')
    require(all(math.isfinite(r['validation']['time_nll']) for r in rows), 'Progress NLL nonfinite')
    best = min(rows, key=lambda r: r['validation']['time_nll'])
    durations = [r['epoch_seconds'] for r in rows[-10:] if r['epoch'] > 0 and 'epoch_seconds' in r]
    require(all(math.isfinite(v) and v > 0 for v in durations), 'Invalid epoch duration')
    median = None
    if durations:
        a = sorted(durations); n = len(a); median = a[n//2] if n % 2 else (a[n//2-1]+a[n//2])/2
    remaining = min(c['fitting']['maximum_epochs'], best['epoch']+c['fitting']['patience'])-rows[-1]['epoch']
    return {'stored_epoch':rows[-1]['epoch'], 'selected_epoch':best['epoch'], 'selected_validation':best['validation'],
        'latest_validation':rows[-1]['validation'], 'recent_epoch_seconds_count':len(durations), 'recent_epoch_seconds_median':median,
        'seconds_to_current_patience_boundary':None if median is None else max(0,remaining)*median,
        'seconds_to_max_epochs':None if median is None else max(0,c['fitting']['maximum_epochs']-rows[-1]['epoch'])*median,
        'ETA_assumption':'Observed own-fit epoch median; further Validation improvements can extend patience',
        'last_epoch_completed_utc':rows[-1].get('completed_utc')}


def collect(c, root, expected):
    root = Path(root)
    require(digest(c) == expected and Path(c['root']) == root, 'Wrong contract/root')
    verify_files(root, c['operation']['files'])
    verify_files(c['source']['root'], c['source']['files'])
    require(digest(c['source']['files']) == c['source']['files_sha256'], 'Source closure mismatch')
    require(digest(c['operation']['files']) == c['operation']['files_sha256'], 'Operation closure mismatch')
    raw_ps = subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True).splitlines()
    processes = parse_processes(raw_ps)
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True).strip()
    require(any(line.split(',')[0].strip() == c['runtime']['gpu_uuid'] and '5080' in line for line in gpu.splitlines()), 'Wrong GPU')
    raw_gpu = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    gpu_pids = {int(line.split(',')[0].strip()) for line in raw_gpu.splitlines() if line.split(',')[0].strip().isdigit()}
    rows = []
    for job in c['jobs']:
        d = Path(job['output_dir'])
        require(d.resolve() == (root/'run'/job['id']).resolve(), 'Job output path changed')
        row = {'job':job, 'state':'waiting'}
        if (d/'worker_process.json').exists(): row['worker_process'] = read(d/'worker_process.json')
        if (d/'failure.json').exists(): row.update(state='failed', failure=read(d/'failure.json'))
        elif (d/'supervisor_terminal_receipt.json').exists(): row.update(state='complete', **verify_job(c,job,d))
        elif d.exists():
            row.update(state='running', status=read(d/'status.json') if (d/'status.json').exists() else None)
            if (d/'history.json').exists(): row['progress'] = progress_history(c,job,read(d/'history.json'))
        rows.append(row)
    status = read(root/'status.json') if (root/'status.json').exists() else None
    failure = read(root/'failure.json') if (root/'failure.json').exists() else None
    supervisor = read(root/'supervisor.json') if (root/'supervisor.json').exists() else None
    exit_receipt = read(root/'supervisor_process_exit.json') if (root/'supervisor_process_exit.json').exists() else None
    owned, issues, owned_gpu = process_identity(c, root, processes, gpu_pids, supervisor, rows)
    if supervisor:
        require(supervisor['contract_sha256'] == expected, 'Supervisor contract changed')
    for row in rows:
        if row['state'] == 'running' and issues: row.update(state='uncertain', reason='; '.join(issues))
    complete_ids = [row['job']['id'] for row in rows if row['state'] == 'complete']
    terminal = (len(complete_ids) == 6 and not owned and not issues and not failure and
        status is not None and status['status'] == 'complete' and status['complete'] == complete_ids and
        exit_receipt is not None and exit_receipt['returncode'] == 0 and exit_receipt['contract_sha256'] == expected)
    return {'contract_sha256':expected, 'actual_observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'host':'5080', 'rows':rows, 'counts':{s:sum(row['state']==s for row in rows) for s in ('complete','failed','running','waiting','uncertain')},
        'owned_processes':owned, 'owned_gpu_pids':owned_gpu, 'process_identity_issues':issues,
        'gpu':gpu, 'gpu_processes':raw_gpu, 'server_status':status, 'failure':failure, 'supervisor':supervisor,
        'supervisor_exit':exit_receipt, 'server_terminal_verified':bool(terminal), 'held_out_test_evaluated':False}


def validate_snapshot(c, snapshot, *, terminal=False):
    require(snapshot['contract_sha256'] == digest(c) and snapshot['host'] == '5080' and snapshot['held_out_test_evaluated'] is False, 'Snapshot identity mismatch')
    require([r['job'] for r in snapshot['rows']] == c['jobs'], 'Snapshot jobs changed')
    require(snapshot['counts'] == {s:sum(r['state']==s for r in snapshot['rows']) for s in ('complete','failed','running','waiting','uncertain')}, 'Snapshot count mismatch')
    for row in snapshot['rows']:
        if row['state'] == 'complete':
            verify_evidence(c, row['job'], row['proof'])
            require(row['result'] == row['proof']['result'] and row['file_SHA_verified'] is True, 'Snapshot proof mismatch')
            t = row['supervisor_terminal']
            require(t['job'] == row['job']['id'] and t['contract_sha256'] == digest(c) and t['scientific_success'] is True and t['status'] == 'complete', 'Snapshot supervisor mismatch')
            require(t['files']['terminal_manifest.json'] == row['worker_terminal_sha256'], 'Terminal SHA binding mismatch')
            for label in ('selected','last'):
                require(row[label+'_checkpoint_sha256'] == row['proof']['terminal']['files'][label+'_refit.pt'] == t['files'][label+'_refit.pt'], 'Snapshot checkpoint SHA mismatch')
    if terminal or snapshot['server_terminal_verified']:
        require(snapshot['server_terminal_verified'] is True and snapshot['counts']['complete'] == 6 and not snapshot['owned_processes'] and not snapshot['owned_gpu_pids'] and not snapshot['process_identity_issues'] and not snapshot['failure'], 'Unverified terminal cache')
        require(snapshot['server_status']['status'] == 'complete' and snapshot['server_status']['complete'] == [j['id'] for j in c['jobs']], 'Supervisor completion mismatch')
        require(snapshot['supervisor_exit']['returncode'] == 0 and snapshot['supervisor_exit']['contract_sha256'] == digest(c), 'Supervisor exit mismatch')
    return snapshot


def core_code():
    names = (require,read,digest,sha,valid_sha,safe_file,verify_files,verify_evidence,verify_job,
        parse_processes,flag,process_identity,progress_history,collect,validate_snapshot)
    return 'from pathlib import Path\nimport datetime,hashlib,json,math,shlex,subprocess\n' + '\n\n'.join(inspect.getsource(f) for f in names)


def observation_failure(exc):
    return {'status':'observation_failed', 'error_type':type(exc).__name__, 'error':str(exc),
        'training_failure':False, 'automatic_retry':False}


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    c = read(BUNDLE/'execution_contract.json'); pointer = read(BUNDLE/'current.json')
    require(pointer['canonical_sha256'] == digest(c) and pointer['root'] == c['root'], 'Current pointer mismatch')
    out = BUNDLE/'hourly_monitor'; out.mkdir(exist_ok=True)
    cache = out/'terminal_cache.json'
    if cache.exists():
        snapshot = validate_snapshot(c, read(cache), terminal=True)
        print(json.dumps({**snapshot,'terminal_cache_reused':True},ensure_ascii=False)); return
    program = core_code() + '\nc=read('+repr(c['root']+'/execution_contract.json')+')\nprint(json.dumps(collect(c,'+repr(c['root'])+','+repr(digest(c))+')))\n'
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    try:
        result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','5080','python3 -'],
            input=program,text=True,capture_output=True,check=True,timeout=120)
        snapshot = validate_snapshot(c, json.loads(result.stdout))
        path = out/(stamp+'.json'); write(path,snapshot)
        write(out/'latest.json',{'snapshot':str(path),'actual_observed_utc':snapshot['actual_observed_utc']})
        if snapshot['server_terminal_verified']: write(cache,snapshot)
        print(json.dumps(snapshot,ensure_ascii=False))
    except Exception as exc:
        write(out/(stamp+'_observation_failure.json'),observation_failure(exc)); raise


if __name__ == '__main__': main()
