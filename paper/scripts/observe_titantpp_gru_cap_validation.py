"""Bounded, read-only observations of the nine approved GRU-cap Validation fits.

No training, qualification, retry, result retrieval or Test evaluation occurs here.
Terminal caches retain their original observation time and prove only that root's
past terminal state. A later GPU user does not invalidate that historical fact.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import FunctionType
import uuid
from zoneinfo import ZoneInfo

from paper.scripts import launch_titantpp_cnn_gru_components_v5 as original
from paper.scripts import titantpp_components_status_transport_v2 as transport
from paper.scripts import titantpp_components_terminal_verifier as legacy
from paper.scripts.titantpp_component_hourly_eta import estimate_fit_eta

PROJECT = original.PROJECT
BUNDLE = PROJECT / 'search_artifacts/titantpp_gru_residual_cap_validation_20261014_v1/execution'
OUTPUT = PROJECT / 'reports/titantpp_gru_residual_cap_validation_20261014_v1/hourly_monitor'
HOSTS = ('5080', '5090')
ARM = 'titantpp_cnn_gru54_residual_cap025'
TERMINAL = 'training_complete_remote_verified'
require, canonical, sha, read = original.require, original.canonical, original.sha, original.read
write = original.write_exclusive


def kst(value):
    return datetime.fromtimestamp(value, ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S KST')


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def registry(c):
    expected = {(dataset, seed, ARM, '5090' if dataset == 'intermittent_frozen_5000' and seed != 62 else '5080')
        for dataset in ('yellow_trip_hourly', 'raf_spare_parts', 'intermittent_frozen_5000')
        for seed in (42, 52, 62)}
    jobs = c['jobs']
    require(len(jobs) == len({job['id'] for job in jobs}) == 9
        and {(j['dataset'], j['seed'], j['arm'], j['host']) for j in jobs} == expected,
        'Exactly nine fixed candidate IDs and physical owners are required; SSH0')
    require(all(j['id'] == f"{j['dataset']}__{j['seed']}__{ARM}" for j in jobs), 'Candidate ID drift; SSH0')
    require(c['evaluation_scope'] == 'validation_only' and c['held_out_test_evaluated'] is False,
        'Validation boundary changed; SSH0')
    return {host: [j for j in jobs if j['host'] == host] for host in HOSTS}


def require_absent(snapshot, pids, message):
    """An empty process map is not evidence that a recorded PID was probed."""
    for pid in pids:
        require(type(pid) is int and pid > 0, 'Malformed owner PID')
        observation = snapshot.get('process_observations', {}).get(str(pid), {})
        require(observation.get('pid') == pid and observation.get('state') == 'absent'
            and observation.get('absence_observed') is True, message + ': ' + str(pid))
        require(pid in snapshot.get('observed_owner_pids', []), 'Owner PID probe coverage incomplete')
    require(not set(snapshot.get('gpu_pids', [])) & pids, message)
    for pid, process in snapshot.get('processes', {}).items():
        require(not (int(pid) in pids or process.get('pgrp') in pids or process.get('ppid') in pids), message)


def verifier_namespace(launcher):
    # Rebind every function to one isolated globals dict; never mutate old modules.
    namespace = dict(vars(legacy))
    namespace.update(base=launcher, canonical=launcher.canonical, decode_snapshot=launcher.decode_snapshot,
                     _require_absent=require_absent)
    for name, value in vars(legacy).items():
        if isinstance(value, FunctionType) and value.__module__ == legacy.__name__ and name != '_require_absent':
            namespace[name] = FunctionType(value.__code__, namespace, value.__name__, value.__defaults__, value.__closure__)
    return namespace


def owner_absence(snapshot, c, host, files):
    jobs = registry(c)[host]
    expected = {}
    def add(pid, name):
        require(type(pid) is int and pid > 0, 'Malformed owner PID')
        expected.setdefault(pid, set()).add(name)
    for name in ('supervisor.claim', 'phases/train_launch.json', 'phases/train_exit.json'):
        item = json.loads(files[name])
        require(item['contract_sha256'] == canonical(c), 'Owner contract mismatch')
        add(item['pid'], name)
        add(item['ppid'], name)
    for job in jobs:
        name = 'worker_exits/' + job['id'] + '.json'
        item = json.loads(files[name])
        require(item['job'] == job and item['contract_sha256'] == canonical(c), 'Worker owner mismatch')
        require(item.get('pid') != item['worker_group_pid'] and item.get('ppid') == item['worker_group_pid'],
            'Actual scientific worker PID/guard relationship missing')
        for field in ('pid', 'ppid', 'worker_group_pid', 'supervisor_pid'):
            add(item[field], name)
    require(snapshot.get('gpu_inventory_uuids', []).count(c['hosts'][host]['gpu_uuid']) == 1,
        'Fixed GPU inventory identity unresolved')
    require(snapshot['gpu_pids'] == [] and snapshot['processes'] == {}, 'Owned or GPU process remains observed')
    require_absent(snapshot, set(expected), 'Owner PID not explicitly absent')
    for pid, sources in expected.items():
        require(sources <= set(snapshot['process_observations'][str(pid)].get('evidence_sources', [])),
            'Owner absence probe is not bound to every source receipt')
    return {'owner_process_absence_verified': True, 'owner_process_probe_count': len(expected),
        'owner_process_probe_pids': sorted(expected),
        'owner_process_observations_sha256': canonical(snapshot['process_observations'])}


def live_probes(snapshot, c, host, files, report):
    if report['status'] != 'training_running':
        return report
    state = json.loads(files['status.json'])
    active = state.get('active_jobs', {})
    launch = json.loads(files['phases/train_launch.json'])
    require(isinstance(active, dict), 'Malformed active workers')
    pids = {state['supervisor_pid'], launch['ppid'], *snapshot['gpu_pids'],
        *(row['worker_group_pid'] for row in active.values())}
    require(snapshot.get('gpu_inventory_uuids', []).count(c['hosts'][host]['gpu_uuid']) == 1,
        'Live fixed GPU inventory identity unresolved')
    for pid in pids:
        require(type(pid) is int and pid > 0, 'Malformed live PID')
        observation = snapshot.get('process_observations', {}).get(str(pid), {})
        process = snapshot.get('processes', {}).get(str(pid), {})
        actual = observation.get('actual', {})
        require(observation.get('state') == process.get('observation_state') == 'live'
            and actual.get('identity_stable_during_query') is True
            and process.get('identity_stable_during_query') is True
            and all(actual.get(k) == process.get(k) for k in ('pid', 'ppid', 'pgrp', 'argv', 'start_ticks')),
            'Live owner PID is reused, racy or unresolved: ' + str(pid))
    # A timeout guard alone does not establish a live scientific CUDA worker.
    for row in active.values():
        matches = [snapshot['processes'][str(pid)] for pid in snapshot['gpu_pids']
            if snapshot['processes'][str(pid)].get('ppid') == row['worker_group_pid']]
        require(len(matches) == 1, 'Active scientific GPU worker not yet uniquely observed')
    return {**report, 'live_process_probes_verified': True}


def verify_snapshot(snapshot, c, host, launcher):
    registry(c)
    require(finite(snapshot.get('observed_unix')), 'Original observation time missing')
    files = launcher.decode_snapshot(snapshot, c, host)
    report = verifier_namespace(launcher)['training_status'](snapshot, c, host)
    for row in report['jobs']:
        if row['status'] == 'complete':
            name = 'worker_exits/' + row['job']['id'] + '.json'
            worker = json.loads(files[name])
            require(type(worker.get('pid')) is int and worker['pid'] != worker['worker_group_pid']
                and worker.get('ppid') == worker['worker_group_pid'], 'Scientific worker identity missing')
            require_absent(snapshot, {worker['pid'], worker['ppid']}, 'Completed scientific worker not absent')
            require(all(name in snapshot['process_observations'][str(pid)].get('evidence_sources', [])
                for pid in (worker['pid'], worker['ppid'])), 'Completed worker probe source missing')
    if report['status'] == TERMINAL:
        report.update(owner_absence(snapshot, c, host, files))
    else:
        report = live_probes(snapshot, c, host, files, report)
    if report['status'] != 'training_running':
        for row in report['jobs']:
            if row['status'] in ('running', 'running_before_first_checkpoint'):
                row['status'] = 'unconfirmed'
    return report


def qualification_bridge(bundle, c, host, snapshot, launcher):
    """Match local issued permits and original qualification to observed bytes."""
    bundle = Path(bundle)
    files = launcher.decode_snapshot(snapshot, c, host)
    approval = launcher.authorization(bundle, c, host)
    qpath = bundle / 'control/qualification_cache' / host / 'snapshot.json'
    cached = read(qpath)
    binding = launcher.qualification_binding(cached, c, host)
    original_files = launcher.decode_snapshot(cached, c, host)
    evidence = {'qualification/receipt.json': binding['raw_sha256'], **binding['receipt']['qualification_files']}
    for name in ('phases/qualify_launch.json', 'phases/qualify_exit.json', 'phases/qualify_complete.json'):
        evidence[name] = hashlib.sha256(original_files[name]).hexdigest()
    for name, expected in evidence.items():
        require(name in files and files[name] == original_files[name]
            and hashlib.sha256(files[name]).hexdigest() == expected, 'Qualification raw bridge changed')
    start = json.loads(files['start_permit.json'])
    require(json.loads(files['approval.json']) == approval and start['host'] == host
        and start['contract_sha256'] == canonical(c) and start['approval_sha256'] == canonical(approval),
        'Approval/start identity mismatch')
    require(finite(start['started_at_unix']) and finite(start['deadline_unix'])
        and start['deadline_unix'] - start['started_at_unix'] == c['limits']['total_wall_seconds']
        and start['started_at_unix'] <= cached['observed_unix'] <= snapshot['observed_unix'],
        'Original host lease/observation mismatch')
    deployment_path = bundle / 'control/deployment' / (host + '.json')
    deployment = read(deployment_path)
    require(deployment['status'] == 'deployed' and deployment['host'] == host
        and deployment['root'] == c['hosts'][host]['root'] and deployment['contract_sha256'] == canonical(c)
        and deployment['start_permit'] == start and deployment['transport_os_exit_code'] == 0,
        'Deployment/start binding changed')
    permit_path = bundle / 'control/training_permits' / (host + '.json')
    permit = read(permit_path)
    require(permit['contract_sha256'] == canonical(c) and permit['start_permit_sha256'] == canonical(start)
        and permit['qualification_sha256'] == binding['raw_sha256'] and permit['qualification'] == binding['receipt']
        and permit['qualifications'] == {host: binding} and permit['fresh_fits'] is True
        and permit['automatic_retry'] is False and finite(permit['issued_unix'])
        and cached['observed_unix'] <= permit['issued_unix'] < start['deadline_unix'],
        'Issued training permit/qualification bridge mismatch')
    if 'phases/train_complete.json' in files:
        done = json.loads(files['phases/train_complete.json'])
        require(permit['issued_unix'] <= done['completed_unix'] <= start['deadline_unix'], 'Completion exceeded host deadline')
    return {'qualification_raw_sha256': binding['raw_sha256'], 'original_host_start': start,
        'local_proof_files': {str(p): sha(p) for p in (qpath, deployment_path, permit_path)},
        'remote_training_permit_raw_directly_observed': False}


def cache_path(output, host):
    return Path(output) / 'terminal_cache' / (host + '.json')


def load_cache(bundle, output, c, host, launcher):
    path = cache_path(output, host)
    pin = path.with_suffix('.sha256')
    require(path.exists() == pin.exists(), 'Incomplete terminal cache seal; SSH0')
    if not path.exists():
        return None
    require(sha(path) == pin.read_text().strip(), 'Terminal cache rawSHA changed; SSH0')
    item = read(path)
    snapshot_path = Path(item['snapshot_path'])
    require(item['host'] == host and item['contract_sha256'] == canonical(c)
        and item['root'] == c['hosts'][host]['root'] and item['gpu_uuid'] == c['hosts'][host]['gpu_uuid'],
        'Terminal cache identity changed; SSH0')
    require(snapshot_path.is_relative_to(Path(bundle).resolve() / 'control/status' / host)
        and sha(snapshot_path) == item['snapshot_raw_sha256'], 'Terminal cache snapshot changed; SSH0')
    snapshot = read(snapshot_path)
    require(snapshot['observed_unix'] == item['original_observed_unix'], 'Cache observation time changed; SSH0')
    bridge = qualification_bridge(bundle, c, host, snapshot, launcher)
    require(bridge == item['qualification_bridge'], 'Terminal cache qualification changed; SSH0')
    report = verify_snapshot(snapshot, c, host, launcher)
    require(report['status'] == TERMINAL and report['owner_process_absence_verified'] is True,
        'Cached terminal state no longer verifies; SSH0')
    return snapshot_path, snapshot, report


def save_cache(bundle, output, c, host, path, snapshot, report, bridge):
    require(report['status'] == TERMINAL and report.get('owner_process_absence_verified') is True,
        'Only fully verified terminal evidence may be cached')
    target = cache_path(output, host)
    require(not target.exists() and not target.with_suffix('.sha256').exists(), 'Existing terminal cache preserved')
    item = {'schema': 'titantpp_gru_cap_terminal_cache_v1', 'host': host,
        'contract_sha256': canonical(c), 'root': c['hosts'][host]['root'], 'gpu_uuid': c['hosts'][host]['gpu_uuid'],
        'snapshot_path': str(Path(path).resolve()), 'snapshot_raw_sha256': sha(path),
        'original_observed_unix': snapshot['observed_unix'], 'qualification_bridge': bridge,
        'scope': 'past terminal state of this root, not current global GPU idle'}
    write(target, item)
    with target.with_suffix('.sha256').open('x') as stream:
        stream.write(sha(target) + '\n')


def enrich(report, snapshot, c, host, path, calls, origin, launcher):
    files = launcher.decode_snapshot(snapshot, c, host)
    rows = []
    for value in report['jobs']:
        row, job = dict(value), value['job']
        eta = {'status': 'unconfirmed', 'reason': 'No verified live saved epoch for this fit'}
        if row['status'] == 'complete':
            eta = {'status': 'complete', 'conditional_early_stop_seconds': 0, 'conditional_maximum_seconds': 0}
        elif report.get('live_process_probes_verified') is True and row['status'] == 'running' and row.get('saved_epoch'):
            run = f"run/{job['id']}/runs/{job['arm']}/count_only_log_regression/seed_{job['seed']}"
            eta = estimate_fit_eta(json.loads(files[run + '/history.json']), json.loads(files[run + '/epoch_timing.json']), row['saved_epoch'])
            if eta['status'] == 'estimated':
                require(eta['best_epoch'] == row['best_epoch'], 'ETA/history selector mismatch')
                claim = json.loads(files['claims/' + job['id'] + '.json'])
                remaining = claim['deadline_unix'] - snapshot['observed_unix']
                eta.update(remaining_condition_seconds=remaining,
                    maximum_scenario_exceeds_deadline=eta['conditional_maximum_seconds'] > remaining,
                    conditional_early_stop_KST=kst(snapshot['observed_unix'] + eta['conditional_early_stop_seconds']),
                    conditional_maximum_KST=kst(snapshot['observed_unix'] + eta['conditional_maximum_seconds']))
        row['eta'] = eta
        rows.append(row)
    return {**report, 'host': host, 'jobs': rows, 'remote_calls': calls,
        'snapshot_path': str(path), 'snapshot_raw_sha256': sha(path), 'observed_unix': snapshot['observed_unix'],
        'observed_KST': kst(snapshot['observed_unix']), 'observation_source': origin,
        'transport_evidence_path': snapshot.get('transport_evidence_path'),
        'transport_evidence_raw_sha256': snapshot.get('transport_evidence_sha256'),
        'user_action_required': report['status'] not in (TERMINAL, 'training_running'),
        'local_binary_original_retrieval': 'not_performed_by_monitor', 'whole_queue_eta': 'unconfirmed'}


def error_report(c, host, error, calls=0, path=None, snapshot=None):
    result = {'host': host, 'status': 'query_or_evidence_unconfirmed', 'completed': None,
        'total': len(registry(c)[host]), 'jobs': [], 'identity_verified': False,
        'reason': type(error).__name__ + ': ' + str(error), 'training_failure_inferred': False,
        'remote_calls': calls, 'user_action_required': True, 'whole_queue_eta': 'unconfirmed'}
    if path is not None and Path(path).exists():
        result.update(snapshot_path=str(path), snapshot_raw_sha256=sha(path))
    if snapshot is not None and finite(snapshot.get('observed_unix')):
        result.update(observed_unix=snapshot['observed_unix'], observed_KST=kst(snapshot['observed_unix']))
    evidence = getattr(error, 'evidence_receipt_path', None)
    if evidence and Path(evidence).exists():
        result.update(transport_evidence_path=str(evidence), transport_evidence_raw_sha256=sha(evidence))
    return result


def observe_host(bundle, output, c, host, launcher, cached=None, *, acquire=transport.read_status_once):
    if cached:
        path, snapshot, report = cached
        return enrich(report, snapshot, c, host, path, 0, 'verified_terminal_cache_original_observation', launcher)
    if not (Path(bundle) / 'control/intents' / ('train_' + host + '.json')).exists():
        return {'host': host, 'status': 'not_launched_local', 'total': len(registry(c)[host]), 'completed': 0,
            'jobs': [{'job': job, 'status': 'waiting', 'saved_epoch': None, 'best_epoch': None,
                'eta': {'status': 'unconfirmed', 'reason': 'This fit has not been launched'}} for job in registry(c)[host]],
            'remote_calls': 0, 'observation_source': 'local_train_intent_absent', 'user_action_required': False}
    path = snapshot = None
    try:
        path, snapshot = acquire(bundle, launcher, c, host, Path(output) / 'diagnostics')
        bridge = qualification_bridge(bundle, c, host, snapshot, launcher)
        report = verify_snapshot(snapshot, c, host, launcher)
        if report['status'] == TERMINAL:
            save_cache(bundle, output, c, host, path, snapshot, report, bridge)
        return enrich(report, snapshot, c, host, path, 1, 'single_read_only_status_call', launcher)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        return error_report(c, host, error, 1, path, snapshot)


def unique_counts(c, reports):
    assigned = registry(c)
    require(set(reports) == set(HOSTS), 'Host reports missing')
    counts = {'total': 9, 'completed': 0, 'running': 0, 'waiting': 0, 'unconfirmed': 0}
    seen = set()
    for host in HOSTS:
        rows, expected = reports[host].get('jobs', []), {j['id']: j for j in assigned[host]}
        if not rows and reports[host].get('user_action_required') is True:
            counts['unconfirmed'] += len(expected)
            continue
        require(len(rows) == len(expected) and {r['job']['id'] for r in rows} == set(expected), 'Missing or duplicate report jobs')
        for row in rows:
            job = row['job']
            require(job == expected[job['id']] and job['id'] not in seen, 'Duplicate fit ID or wrong physical owner')
            seen.add(job['id'])
            state = {'complete': 'completed', 'running': 'running', 'running_before_first_checkpoint': 'running', 'waiting': 'waiting'}.get(row['status'], 'unconfirmed')
            if state == 'running' and reports[host].get('live_process_probes_verified') is not True:
                state = 'unconfirmed'
            counts[state] += 1
    require(sum(counts[k] for k in ('completed', 'running', 'waiting', 'unconfirmed')) == 9, 'Count closure failed')
    counts['all_training_terminal_verified'] = counts['completed'] == 9 and all(
        reports[h]['status'] == TERMINAL and reports[h].get('owner_process_absence_verified') is True for h in HOSTS)
    return counts


def verify_monitor(output, digest, c, bundle=BUNDLE):
    path = Path(output) / 'monitor_integrity.json'
    require(sha(path) == digest, 'Monitor integrity rawSHA changed; SSH0')
    seal = read(path)
    require(seal['contract_sha256'] == canonical(c) and canonical(seal['files']) == seal['files_sha256'],
        'Monitor contract/code closure changed; SSH0')
    execution_seal = str((Path(bundle).resolve() / 'execution_integrity.json').relative_to(PROJECT.resolve()))
    required = {'paper/scripts/observe_titantpp_gru_cap_validation.py', 'paper/scripts/titantpp_components_status_transport_v2.py',
        'paper/scripts/launch_titantpp_gru_cap_validation.py', 'paper/scripts/launch_titantpp_cnn_gru_components_v5.py', execution_seal,
        'paper/scripts/titantpp_components_terminal_verifier.py', 'paper/scripts/titantpp_component_hourly_eta.py'}
    require(required <= set(seal['files']), 'Monitor dependency seal incomplete; SSH0')
    for name, expected in seal['files'].items():
        require(sha(original.safe(PROJECT, name)) == expected, 'Monitor dependency changed; SSH0')


def duration(seconds):
    if seconds is None:
        return '미확정'
    minutes = round(seconds / 60)
    return f'{minutes // 60}시간 {minutes % 60}분' if minutes >= 60 else f'{minutes}분'


def markdown(result):
    labels = {'yellow_trip_hourly': 'Taxi', 'raf_spare_parts': 'RAF', 'intermittent_frozen_5000': 'Intermittent'}
    states = {'complete': '완료', 'running': '진행', 'running_before_first_checkpoint': '진행·저장 전', 'waiting': '대기'}
    counts = result['counts']
    lines = ['# GRU 보정량 제한 후보 9개 관측', '',
        f"고유 9개: 완료 {counts['completed']} · 진행 {counts['running']} · 대기 {counts['waiting']} · 미확정 {counts['unconfirmed']}", '',
        '| GPU · 원관측 KST | 데이터 · seed · 후보 | 상태 | 저장 / best epoch | best 미갱신 조건 | 300 epoch 조건 |',
        '|---|---|---|---:|---:|---:|']
    notes = []
    for host in HOSTS:
        report = result['hosts'][host]
        for row in report.get('jobs', []):
            job, eta = row['job'], row.get('eta', {})
            saved = '저장 전' if row.get('saved_epoch') is None else str(row['saved_epoch'])
            best = '미확정' if row.get('best_epoch') is None else str(row['best_epoch'])
            maximum = duration(eta.get('conditional_maximum_seconds'))
            if eta.get('maximum_scenario_exceeds_deadline'):
                maximum += ' · 현 기한 내 보장 불가'
            lines.append(f"| {host} · {report.get('observed_KST', '새 원격 관측 없음')} | {labels[job['dataset']]} · {job['seed']} · CAP0.25 | {states.get(row['status'], '미확정')} | {saved} / {best} | {duration(eta.get('conditional_early_stop_seconds'))} | {maximum} |")
            if eta.get('status') == 'estimated' and eta['recent_epoch_count'] < 10:
                notes.append(f"- {host} {job['dataset']} seed{job['seed']}: timing 표본 {eta['recent_epoch_count']}개.")
        if not report.get('jobs') or report.get('user_action_required'):
            notes.append(f"- {host}: {report.get('reason', report['status'])}")
    lines += ['', *notes]
    lines += ['', '- ETA는 해당 fit의 최근 최대 10개 완료 epoch의 유한 양수 시간 중앙값이며 원관측 기준이다. best 미갱신 목표는 min(300,max(40,best+40))이다. 부하·속도·best 갱신에 따라 달라지고 전체 큐 종료시각은 미확정이다.',
        '- 종료 캐시는 해당 root의 과거 종료 증적이며 현재 전역 GPU 유휴 상태를 뜻하지 않는다. pipeline wrapper 자체 OS exit는 직접 관측하지 않았다.',
        '- Intermittent seed42·52는 5090/cuDNN92000, seed62는 5080/cuDNN91900이다. 시간은 host별로 분리한다.',
        '- 원격 학습 종료와 로컬 원본 회수·최종 3seed Validation 집계는 별개다. 이 관측 도구는 학습·재시도·Test·원본 회수를 실행하지 않는다.']
    return '\n'.join(lines) + '\n'


def observe(bundle, output, *, monitor_integrity_sha, acquire=transport.read_status_once):
    from paper.scripts import launch_titantpp_gru_cap_validation as launch
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    # Externally pinned monitor closure binds the otherwise self-contained
    # execution integrity document before any operational bundle is accepted.
    c = read(bundle / 'execution_contract.json')
    verify_monitor(output, monitor_integrity_sha, c, bundle)
    c, _ = launch.check_bundle(bundle)
    registry(c)
    launcher = launch.launcher_for(bundle)
    for host in HOSTS:
        launcher.authorization(bundle, c, host)
    # Validate every existing cache before issuing even the first live-host SSH.
    cached = {host: load_cache(bundle, output, c, host, launcher) for host in HOSTS}
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'observer.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        token = uuid.uuid4().hex
        write(output / 'rounds' / (token + '_intent.json'), {'pid': os.getpid(), 'started_unix': time.time(),
            'hosts': list(HOSTS), 'maximum_live_status_calls_per_host': 1, 'remote_mutations': 0})
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {h: pool.submit(observe_host, bundle, output, c, h, launcher, cached[h], acquire=acquire) for h in HOSTS}
            reports = {h: future.result() for h, future in futures.items()}
        counts = unique_counts(c, reports)
        result = {'schema': 'titantpp_gru_cap_validation_monitor_v1', 'contract_sha256': canonical(c),
            'monitor_integrity_raw_sha256': monitor_integrity_sha, 'report_generated_KST': kst(time.time()),
            'hosts': reports, 'counts': counts, 'all_training_terminal_verified': counts['all_training_terminal_verified'],
            'whole_queue_eta': 'unconfirmed', 'remote_mutations': 0, 'held_out_test_evaluated': False,
            'local_original_retrieval': 'not_performed_by_monitor', 'pipeline_wrapper_OS_exit_directly_observed': False}
        path = output / 'rounds' / (token + '.json')
        write(path, result)
        md = path.with_suffix('.md')
        with md.open('x') as stream:
            stream.write(markdown(result))
        latest = {'round_path': str(path), 'round_raw_sha256': sha(path), 'markdown_path': str(md),
            'markdown_raw_sha256': sha(md), 'contract_sha256': canonical(c)}
        temp = output / (token + '.latest.tmp')
        write(temp, latest)
        temp.replace(output / 'latest.json')
        return {**latest, 'latest_path': str(output / 'latest.json'), 'latest_raw_sha256': sha(output / 'latest.json'),
            'counts': counts, 'all_training_terminal_verified': counts['all_training_terminal_verified']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=BUNDLE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--monitor-integrity-sha', required=True)
    args = parser.parse_args()
    require(args.output.resolve() == OUTPUT, 'Monitor output outside the approved report directory; SSH0')
    print(json.dumps(observe(args.bundle, args.output, monitor_integrity_sha=args.monitor_integrity_sha), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
