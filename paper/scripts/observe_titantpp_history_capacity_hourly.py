"""One read-only SSH observation of the frozen width8/12 campaign (no launch/retry)."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / 'search_artifacts/titantpp_history_width8_12_dual_20261004_v1'
CONTRACT = '8434041a6a301995a8d22baa77d5ca0da4eb2cd942d091a8aa8cef379406c48d'
SOURCE = '41a4a4ff79e5e17d907b9ee8966723004516ee2e42608d31c7f1a871f0acd3e1'
KST = timezone(timedelta(hours=9))
METRICS = ('qty_rmse', 'qty_mae', 'time_nll')
canonical = lambda v: hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(v):
    return type(v) in (int, float) and math.isfinite(v)


def close(a, b):
    return finite(a) and finite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(tmp, path)


def load_contract(bundle=B):
    """Observation intentionally remains available after the execution deadline."""
    c, pointer = read(bundle / 'execution_contract.json'), read(bundle / 'current.json')
    pointed = Path(pointer['bundle'])
    if not pointed.is_absolute():
        pointed = ROOT / pointed
    require(pointed.resolve() == bundle.resolve(), 'Current bundle changed')
    require(canonical(c) == pointer['contract_sha256'] == CONTRACT, 'Frozen contract changed')
    require(canonical(c['source']['files']) == c['source']['files_sha256'] == pointer['source_closure_sha256'] == SOURCE, 'Frozen source closure changed')
    require(len(c['source']['files']) == 114, 'Unexpected source count')
    require(pointer['hosts'] == {h: v['root'] for h, v in c['hosts'].items()}, 'Host roots changed')
    require(c['evaluation_scope'] == 'validation_only' and c['held_out_test_evaluated'] is False, 'Held-out scope forbidden')
    require({h: sum(j['host'] == h for j in c['jobs']) for h in c['hosts']} == {'5080': 12, '5090': 6}, 'Condition coverage changed')
    for name, digest in c['source']['files'].items():
        path = bundle / 'frozen_source' / name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and not path.is_symlink(), 'Unsafe frozen source')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, 'Frozen source changed: ' + name)
    return c


REMOTE = r'''
import hashlib,json,subprocess,time
from pathlib import Path
root=Path(ROOT_PATH)
canon=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
c=json.loads((root/'execution_contract.json').read_text())
assert canon(c)==EXPECTED_CONTRACT and c['evaluation_scope']=='validation_only' and c['held_out_test_evaluated'] is False
assert canon(c['source']['files'])==c['source']['files_sha256']==EXPECTED_SOURCE
for name,digest in c['source']['files'].items():
 p=root/'source'/name
 assert not p.is_symlink() and not Path(name).is_absolute() and '..' not in Path(name).parts
 assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
out={'host':HOST_NAME,'root':str(root),'observed_unix':time.time(),'contract_sha256':canon(c),'source_closure_sha256':c['source']['files_sha256'],'files':{},'file_sha256':{},'runs':[]}
def read_record(p):
 if not p.exists():return None
 assert not p.is_symlink() and p.resolve().is_relative_to(root.resolve())
 raw=p.read_bytes();assert len(raw)<20*1024*1024
 out['file_sha256'][str(p.relative_to(root))]=hashlib.sha256(raw).hexdigest()
 return json.loads(raw)
for name in ('status.json','failure.json','progress.json','server_lease.json','supervisor.claim'):
 out['files'][name]=read_record(root/name)
for job in c['jobs']:
 if job['host']!=HOST_NAME:continue
 folder=root/'run'/job['id'];leaf=folder/'runs'/job['arm']/c['quantity_variants'][job['arm']]/('seed_'+str(job['seed']))
 row={'job':job,'status':read_record(folder/'status.json'),'manifest':read_record(folder/'terminal_manifest.json')}
 for key,name in (('history','history.json'),('timing','epoch_timing.json'),('checkpoint','server_checkpoint_receipt.json'),('endpoints','endpoint_replays.json'),('diagnostic','selected_train_validation_diagnostic.json')):
  row[key]=read_record(leaf/name)
 row['binary_sizes']={name:(leaf/name).stat().st_size if (leaf/name).is_file() and not (leaf/name).is_symlink() else 0 for name in ('last_epoch_state.pt','best_val_qty_rmse_model.pt')}
 out['runs'].append(row)
for key,command in (('gpu',['nvidia-smi','--query-gpu=uuid,name,utilization.gpu,memory.used,memory.total','--format=csv,noheader']),('compute',['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory','--format=csv,noheader'])):
 p=subprocess.run(command,capture_output=True,text=True,timeout=15)
 out[key]={'returncode':p.returncode,'stdout':p.stdout}
p=subprocess.run(['ps','-eo','pid,ppid,pgid,args'],capture_output=True,text=True,timeout=15)
out['ps_returncode']=p.returncode
out['processes']=[line for line in p.stdout.splitlines() if str(root) in line and 'python3 -' not in line]
print(json.dumps(out,allow_nan=False))
'''


def remote_script(c, host):
    return 'ROOT_PATH=' + repr(c['hosts'][host]['root']) + '\nHOST_NAME=' + repr(host) + '\nEXPECTED_CONTRACT=' + repr(CONTRACT) + '\nEXPECTED_SOURCE=' + repr(SOURCE) + '\n' + REMOTE


def snapshot_once(c, host, runner=subprocess.run):
    result = runner(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host, 'python3 -'],
                    input=remote_script(c, host), text=True, capture_output=True, timeout=80)
    require(result.returncode == 0, 'SSH observation failed; raw output suppressed')
    return json.loads(result.stdout)


def process_evidence(snapshot, c, host):
    rows, fits, supervisors = {}, [], []
    for line in snapshot.get('processes', []):
        parts = line.split(None, 3)
        if len(parts) != 4:
            continue
        try:
            pid, ppid, pgid = map(int, parts[:3]); words = shlex.split(parts[3])
        except ValueError:
            continue
        rows[pid] = {'pid': pid, 'ppid': ppid, 'pgid': pgid, 'words': words}
        if not words or words[0] != c['hosts'][host]['python']:
            continue
        try:
            script = 'paper/scripts/run_titantpp_history_capacity_campaign.py'
            require(script in words or c['hosts'][host]['source_root'] + '/' + script in words, 'Foreign process entrypoint')
            require(words[words.index('--contract') + 1] == c['hosts'][host]['root'] + '/execution_contract.json', 'Foreign process contract')
            require(words[words.index('--host') + 1] == host, 'Foreign process host')
            mode = words[words.index('--mode') + 1]
            if mode == 'fit':
                fits.append({'pid': pid, 'job_id': words[words.index('--job') + 1], 'pgid': pgid})
            elif mode == 'dispatch':
                supervisors.append(pid)
        except (ValueError, IndexError):
            continue
    compute = snapshot.get('compute', {})
    gpu_pids = set()
    if compute.get('returncode') == 0:
        for line in compute.get('stdout', '').splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 2 and parts[1] == c['hosts'][host]['gpu_uuid']:
                try: gpu_pids.add(int(parts[0]))
                except ValueError: pass
    observed_uuids = {line.split(',')[0].strip() for line in snapshot.get('gpu', {}).get('stdout', '').splitlines()}
    available = snapshot.get('ps_returncode') == 0 and snapshot.get('gpu', {}).get('returncode') == compute.get('returncode') == 0 and observed_uuids == {c['hosts'][host]['gpu_uuid']}
    return rows, fits, supervisors, gpu_pids, available


def descendant(pid, group, rows):
    seen = set()
    while pid in rows and pid not in seen:
        if pid == group or rows[pid]['pgid'] == group:
            return True
        seen.add(pid); pid = rows[pid]['ppid']
    return False


def audit_endpoint(v, data):
    require(v.get('evaluation_scope') == 'validation_only' and v.get('held_out_test_evaluated') is False, 'Replay scope changed')
    require(v.get('time_metric') == 'recorded_positive_integer_time_nll', 'Time metric changed')
    require(v['count'] == data['inherited_data_identity']['populations']['validation']['target_count'], 'Replay population count changed')
    require(v['quantity_boundaries'] == data['quantity_boundaries_all_train_rows'], 'Replay strata changed')
    require(isinstance(v.get('state_sha256'), str) and len(v['state_sha256']) == 64, 'Replay state SHA missing')
    def cell(value):
        n = value.get('count')
        require(type(n) is int and n >= 0 and finite(value.get('qty_sse')) and value['qty_sse'] >= 0, 'Replay cell count/SSE invalid')
        if not n:
            require(value['qty_sse'] == 0 and all(value.get(k) is None for k in METRICS), 'Empty cell metrics changed')
        else:
            require(all(finite(value.get(k)) for k in METRICS) and value['qty_rmse'] >= 0 and value['qty_mae'] >= 0 and close(value['qty_rmse'] ** 2 * n, value['qty_sse']), 'Replay RMSE/SSE invalid')
    for value in (v, v['tail'], v['body']): cell(value)
    for kind, size in (('quantity', 5), ('history', 3)):
        cells = v[kind + '_cells']
        require(len(cells) == size and [x['bin'] for x in cells] == list(range(size)), 'Replay bins changed')
        for value in cells: cell(value)
        require(sum(x['count'] for x in cells) == v['count'] and close(sum(x['qty_sse'] for x in cells), v['qty_sse']), 'Replay disjoint accounting changed')
        for key in ('qty_mae', 'time_nll'):
            require(close(sum(x[key] * x['count'] for x in cells if x['count']), v[key] * v['count']), 'Replay disjoint metric changed')
    cells = v['quantity_cells']
    require(v['tail'] == {k: val for k, val in cells[4].items() if k != 'bin'}, 'Replay tail changed')
    require(v['body']['count'] == sum(x['count'] for x in cells[:3]) and close(v['body']['qty_sse'], sum(x['qty_sse'] for x in cells[:3])), 'Replay body changed')


def terminal_proof(raw, snapshot, c, job, completed):
    folder = 'run/' + job['id']
    prefix = 'runs/' + job['arm'] + '/' + c['quantity_variants'][job['arm']] + '/seed_' + str(job['seed']) + '/'
    m = raw.get('manifest') or {}
    require(m.get('status') == 'complete' and m.get('scientific_success') is True and m.get('job') == job and m.get('contract_sha256') == CONTRACT, 'Terminal identity/success absent')
    require(completed.get(job['id'], {}).get('terminal_manifest_sha256') == snapshot['file_sha256'].get(folder + '/terminal_manifest.json'), 'Supervisor did not admit current terminal manifest')
    for name in m['files']:
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts and not any(any(token in part.lower() for token in ('test', 'heldout', 'held_out')) for part in path.parts), 'Unsafe or held-out manifest reference')
    needed = ('history.json', 'epoch_timing.json', 'server_checkpoint_receipt.json', 'endpoint_replays.json', 'selected_train_validation_diagnostic.json')
    require(all(m['files'].get(prefix + n) == snapshot['file_sha256'].get(folder + '/' + prefix + n) and m['files'].get(prefix + n) for n in needed), 'Terminal small-record SHA mismatch')
    require(all(prefix + n in m['files'] and raw['binary_sizes'].get(n, 0) > 0 for n in ('last_epoch_state.pt', 'best_val_qty_rmse_model.pt')), 'Original checkpoint presence unconfirmed')
    history = raw['history']['history']
    require(c['training']['minimum_epochs'] <= len(history) <= c['training']['maximum_epochs'] and [x['epoch'] for x in history] == list(range(1, len(history) + 1)), 'Terminal epochs changed')
    require(all(finite(x.get('val_qty_rmse')) for x in history), 'Nonfinite selection history')
    best = min(history, key=lambda x: (x['val_qty_rmse'], x['epoch']))['epoch']
    endpoints = raw['endpoints']
    require(endpoints.get('status') == 'complete' and endpoints.get('job') == job and endpoints.get('evaluation_scope') == 'validation_only' and endpoints.get('held_out_test_evaluated') is False, 'Endpoint identity/scope changed')
    require(endpoints['best_epoch'] == best and endpoints['completed_epochs'] == len(history), 'Replay selection changed')
    data = next(d for d in c['datasets'] if d['dataset_id'] == job['dataset'])
    for label, epoch, name in (('selected', best, 'best_val_qty_rmse_model.pt'), ('last', len(history), 'last_epoch_state.pt')):
        endpoint = endpoints[label]; audit_endpoint(endpoint, data)
        require(endpoint['checkpoint_path'] == c['hosts'][job['host']]['root'] + '/' + folder + '/' + prefix + name, 'Replay checkpoint path changed')
        require(all(close(endpoint[k], history[epoch - 1]['val_' + k]) for k in METRICS), 'Replay differs from same-epoch history')
    d = raw['diagnostic']; population = d['validation']; provenance = d['provenance']
    require((d['dataset'], d['seed'], d['model'], d['epoch']) == (job['dataset'], job['seed'], job['arm'], best), 'Diagnostic identity changed')
    require(provenance['held_out_test_evaluated'] is False and provenance['selection_unchanged'] is True and provenance['new_training'] is False and provenance['state_sha256'] == endpoints['selected']['state_sha256'], 'Diagnostic provenance changed')
    require(population['split'] == 'validation' and population['full_population'] is True and population['population'] == data['inherited_data_identity']['populations']['validation'] and all(close(population[k], endpoints['selected'][k]) for k in METRICS), 'Full Validation population differs')
    return best, {k: endpoints['selected'][k] for k in METRICS}


def analyse(snapshot, c, host, reused=False):
    require(snapshot['host'] == host and snapshot['root'] == c['hosts'][host]['root'] and snapshot['contract_sha256'] == CONTRACT and snapshot['source_closure_sha256'] == SOURCE, 'Snapshot identity changed')
    jobs = [j for j in c['jobs'] if j['host'] == host]
    by_id = {r['job']['id']: r for r in snapshot['runs']}
    require(len(by_id) == len(snapshot['runs']) and set(by_id) == {j['id'] for j in jobs}, 'Snapshot condition coverage changed')
    processes, fits, supervisors, gpu_pids, gpu_available = process_evidence(snapshot, c, host)
    status = snapshot['files'].get('status.json') or {}
    completed = status.get('completed', {})
    failure = snapshot['files'].get('failure.json') or {}
    active = status.get('active_job') or {}
    rows = []
    for job in jobs:
        raw = by_id[job['id']]; require(raw['job'] == job, 'Foreign run identity')
        history = (raw.get('history') or {}).get('history', [])
        timing = (raw.get('timing') or {}).get('epochs', [])
        checkpoint = raw.get('checkpoint') or {}
        saved = checkpoint.get('epoch'); eligible = [r for r in history if finite(r.get('val_qty_rmse'))]
        best = min(eligible, key=lambda r: (r['val_qty_rmse'], r['epoch']))['epoch'] if eligible else None
        owned = [f for f in fits if f['job_id'] == job['id']]
        matches = active == job and status.get('supervisor_pid') in supervisors and all(descendant(f['pid'], status.get('worker_group_pid'), processes) for f in owned)
        confirmed_gpu = [f['pid'] for f in owned if f['pid'] in gpu_pids]
        prefix = 'run/' + job['id'] + '/runs/' + job['arm'] + '/' + c['quantity_variants'][job['arm']] + '/seed_' + str(job['seed']) + '/'
        consistent = (type(saved) is int and saved > 0 and raw.get('binary_sizes', {}).get('last_epoch_state.pt', 0) > 0 and checkpoint.get('contract_sha256') == CONTRACT and [r.get('epoch') for r in history] == [r.get('epoch') for r in timing] == list(range(1, saved + 1)) and all(checkpoint.get('files', {}).get(n) == snapshot['file_sha256'].get(prefix + n) and checkpoint.get('files', {}).get(n) for n in ('history.json', 'epoch_timing.json')))
        reason = None; verified = False; metrics = None
        if raw.get('manifest') or job['id'] in completed:
            try:
                best, metrics = terminal_proof(raw, snapshot, c, job, completed); verified = True; state = 'completed'
            except (ValueError, KeyError, TypeError) as error:
                state = 'unknown'; reason = str(error)
        elif failure.get('status') == 'failed' and failure.get('active_job') == job and (history or raw.get('status')):
            state = 'failed'
        elif failure.get('status') == 'failed' and failure.get('active_job') == job:
            state = 'unknown'; reason = 'Condition failed before a fit start was confirmed; not counted as a failed fit'
        elif owned and confirmed_gpu and matches and consistent and gpu_available:
            state = 'training'
        elif owned and matches and not saved and gpu_available:
            state = 'preparing'
        elif not history and not owned and not raw.get('status'):
            state = 'unstarted_terminal' if status.get('status') == 'failed' else 'pending'
        else:
            state = 'unknown'; reason = 'PID/GPU/supervisor/checkpoint evidence does not agree'
        recent = [r['elapsed_seconds'] for r in timing if finite(r.get('elapsed_seconds')) and r['elapsed_seconds'] > 0][-10:]
        median = statistics.median(recent) if recent else None
        eta = None
        if state == 'training' and median and best:
            stop = min(c['training']['maximum_epochs'], max(c['training']['minimum_epochs'], best + c['training']['patience']))
            eta = {'target_epoch': stop, 'remaining_seconds': max(0, stop - saved) * median,
                   'conditional_finish_unix': snapshot['observed_unix'] + max(0, stop - saved) * median,
                   'assumption': 'No further best RMSE improvement; same GPU/speed; excludes replay/retrieval/CPU audit'}
        rows.append({'job': job, 'state': state, 'saved_epoch': saved, 'best_epoch': best, 'checkpoint_consistent': bool(consistent), 'terminal_verified': verified, 'validation_selected': metrics, 'worker_pids': [f['pid'] for f in owned], 'confirmed_gpu_pids': confirmed_gpu, 'recent10_epoch_seconds_median': median, 'eta': eta, 'issue': reason})
    counts = Counter(r['state'] for r in rows)
    declared = status.get('supervisor_pid') in supervisors and (not fits or all(f['job_id'] == active.get('id') and descendant(f['pid'], status.get('worker_group_pid'), processes) for f in fits))
    all_terminal = counts['completed'] == len(jobs) and status.get('status') == 'complete' and gpu_available and not fits and not supervisors and not gpu_pids
    return {'host': host, 'contract_sha256': CONTRACT, 'source_closure_sha256': SOURCE, 'observed_unix': snapshot['observed_unix'], 'observed_kst': datetime.fromtimestamp(snapshot['observed_unix'], KST).strftime('%Y-%m-%d %H:%M:%S KST'), 'reused_terminal_evidence': reused, 'evaluation_scope': 'validation_only', 'held_out_test_evaluated': False, 'status_counts': {'completed': counts['completed'], 'failed': counts['failed'], 'in_progress': counts['training'] + counts['preparing'], 'pending': counts['pending'], 'unknown': counts['unknown']}, 'state_counts': dict(counts), 'rows': rows, 'server_status': status.get('status'), 'supervisor_pids': supervisors, 'actual_owned_worker_pids': [f['pid'] for f in fits], 'actual_owned_gpu_pids': sorted(gpu_pids & {f['pid'] for f in fits}), 'GPU_UUID_verified': gpu_available, 'declared_workers_match_processes': bool(declared), 'server_terminal_verified': all_terminal, 'queue_eta': None, 'queue_eta_reason': 'All conditions terminal verified' if all_terminal else '미확정: 미시작 조건의 실측과 대기시간 또는 실행 증거 부족', 'binary_checkpoint_hashed': False, 'original_checkpoint_retrieval_and_CPU_audit': 'separate_not_performed'}


def report(a):
    lines = [f"# {a['host']} 폭8·12 시간별 관측", '', f"실제 관측: {a.get('observed_kst') or '실패'}. 보존 종료 증거 재사용: {a.get('reused_terminal_evidence', False)}.", f"상태: {a['status_counts']}", '', '| 데이터 / seed | 후보 | 상태 | 저장 / best epoch | 조건부 종료 KST |', '|---|---|---|---:|---|']
    for row in a.get('rows', []):
        j = row['job']; eta = row.get('eta')
        finish = datetime.fromtimestamp(eta['conditional_finish_unix'], KST).strftime('%Y-%m-%d %H:%M:%S KST') if eta else '미확정'
        lines.append(f"| {j['dataset']} / {j['seed']} | {j['arm']} | {row['state']} | {row.get('saved_epoch')} / {row.get('best_epoch')} | {finish} |")
    lines += [''] + [row['job']['id'] + ': ' + row['issue'] for row in a.get('rows', []) if row.get('issue')]
    lines += ['', '전체 큐 ETA: ' + a.get('queue_eta_reason', '미확정'), 'ETA는 best 추가 개선이 없다는 조건의 최근10epoch 중앙값이며 재추론·회수·CPU 감사는 포함하지 않습니다. 원본 checkpoint 회수와 CPU 감사는 별도입니다.']
    return '\n'.join(lines) + '\n'


def run_once(host, bundle=B, runner=subprocess.run):
    requested = time.time(); stamp = datetime.fromtimestamp(requested, timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    dest = bundle / 'hourly_monitor' / (stamp + '_' + host)
    snapshot_path = dest / 'host_snapshot.json'; reused = False; fresh = False; calls = 0; error_type = None
    try:
        c = load_contract(bundle)
        latest = bundle / 'hourly_monitor' / ('latest_' + host + '.json')
        prior = read(latest) if latest.exists() else {}
        if prior.get('server_terminal_verified') is True:
            prior_snapshot = Path(prior['snapshot']).resolve()
            require(prior_snapshot.is_relative_to((bundle / 'hourly_monitor').resolve()), 'Unsafe saved snapshot path')
            raw = prior_snapshot.read_bytes(); require(hashlib.sha256(raw).hexdigest() == prior['snapshot_sha256'], 'Saved snapshot SHA changed')
            snapshot = json.loads(raw); a = analyse(snapshot, c, host, reused=True)
            require(a['server_terminal_verified'], 'Saved terminal evidence no longer verifies'); reused = True
        else:
            calls = 1; snapshot = snapshot_once(c, host, runner)
            require(requested - 120 <= snapshot['observed_unix'] <= time.time() + 120, 'Stale observation returned')
            a = analyse(snapshot, c, host); fresh = True
        write(snapshot_path, snapshot)
    except Exception as error:
        error_type = type(error).__name__
        a = {'host': host, 'status': 'observation_failed_not_training_failure', 'observed_unix': None, 'observed_kst': None, 'reused_terminal_evidence': False, 'status_counts': {'completed': 0, 'failed': 0, 'in_progress': 0, 'pending': 0, 'unknown': 12 if host == '5080' else 6}, 'server_terminal_verified': False, 'queue_eta_reason': '이번 조회 실패; 과거 상태를 새 관측으로 표시하지 않음', 'error_type': error_type}
    write(dest / 'analysis.json', a); (dest / 'report.md').write_text(report(a))
    receipt = {'host': host, 'requested_unix': requested, 'observed_unix': a['observed_unix'], 'fresh_observation': fresh, 'reused_terminal_evidence': reused, 'SSH_invocations': calls, 'status': a.get('status', 'observed'), 'error_type': error_type, 'status_counts': a['status_counts'], 'server_terminal_verified': a['server_terminal_verified'], 'snapshot': str(snapshot_path) if snapshot_path.exists() else None, 'snapshot_sha256': hashlib.sha256(snapshot_path.read_bytes()).hexdigest() if snapshot_path.exists() else None, 'analysis': str(dest / 'analysis.json'), 'report': str(dest / 'report.md')}
    write(bundle / 'hourly_monitor' / ('latest_' + host + '.json'), receipt)
    print(json.dumps(receipt, ensure_ascii=False)); return 0 if error_type is None else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--host', choices=('5080', '5090'), required=True)
    raise SystemExit(run_once(parser.parse_args().host))
