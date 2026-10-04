"""Observe original owners and the approved seed62 handoff once per host."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import observe_titantpp_history_capacity_hourly as old
import capacity_handoff_runtime as handoff

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / 'search_artifacts' / handoff.NAME
PARENT = old.B


def target_process_evidence(snapshot, c, host):
    """Reuse strict PID/GPU ancestry checks with the sealed sidecar entrypoint."""
    translated = deepcopy(snapshot)
    import shlex
    lines = []
    for line in snapshot.get('processes', []):
        parts = line.split(None, 3)
        if len(parts) != 4:
            continue
        try:
            words = shlex.split(parts[3])
            if (words[0] != c['hosts'][host]['python'] or
                words[1] != c['hosts'][host]['root'] + '/control/capacity_handoff_runtime.py' or
                words[words.index('--bundle') + 1] != c['hosts'][host]['root']):
                continue
            mode = words[words.index('--mode') + 1]
            command = [words[0], c['hosts'][host]['source_root'] + '/paper/scripts/run_titantpp_history_capacity_campaign.py',
                       '--contract', c['hosts'][host]['root'] + '/execution_contract.json', '--host', host, '--mode', mode]
            if mode == 'fit':
                command += ['--job', words[words.index('--job') + 1]]
            lines.append(' '.join(parts[:3]) + ' ' + shlex.join(command))
        except (ValueError, IndexError):
            continue
    translated['processes'] = lines
    return ORIGINAL_PROCESS(translated, c, host)


ORIGINAL_PROCESS = old.process_evidence


def analyse_target(snapshot, parent, c):
    selected = {j['id'] for j in handoff.target_jobs(parent)}
    view = deepcopy(c)
    view['jobs'] = [j for j in c['jobs'] if j['id'] in selected]
    original_sha = old.CONTRACT
    try:
        old.CONTRACT = handoff.canonical(c)
        old.process_evidence = target_process_evidence
        result = old.analyse(snapshot, view, '5080')
    finally:
        old.CONTRACT = original_sha
        old.process_evidence = ORIGINAL_PROCESS
    result['origin_host'] = '5090'
    result['administrative_transfer'] = True
    return result


def analyse_source(snapshot, parent, c, q):
    selected = {j['id'] for j in handoff.target_jobs(parent)}
    reservation = snapshot['files'].get('handoff_seed62/reservation.json')
    handoff.validate_reservation(parent, c, q, reservation)
    for item in reservation['reserved']:
        name = 'claims/' + item['job_id'] + '.json'
        old.require(snapshot['files'].get(name) == item['claim'] and
                    snapshot['file_sha256'].get(name) == item['claim_sha256'], 'Live source claim differs')
        raw = next(r for r in snapshot['runs'] if r['job']['id'] == item['job_id'])
        old.require(snapshot.get('transferred_run_folder_exists', {}).get(item['job_id']) is False and
                    not any(raw.get(k) for k in ('status', 'manifest', 'history', 'checkpoint', 'timing', 'endpoints', 'diagnostic')), 'Transferred source fit exists')
    view = deepcopy(parent)
    view['jobs'] = [j for j in parent['jobs'] if j['id'] not in selected]
    filtered = deepcopy(snapshot)
    filtered['runs'] = [r for r in snapshot['runs'] if r['job']['id'] not in selected]
    result = old.analyse(filtered, view, '5090')
    failure = snapshot['files'].get('failure.json') or {}
    first = handoff.target_jobs(parent)[0]
    expected_claim = parent['hosts']['5090']['root'] + '/claims/' + first['id'] + '.json'
    boundary = (failure.get('status') == 'failed' and failure.get('type') == 'FileExistsError' and
                failure.get('active_job') == first and expected_claim in failure.get('message', '') and result['status_counts']['completed'] == 4 and
                result['GPU_UUID_verified'] and not result['actual_owned_worker_pids'] and not result['supervisor_pids'])
    for row in result['rows']:
        if row['state'] == 'unstarted_terminal':
            result['status_counts']['unknown'] += 1
            row['issue'] = 'Source stopped before this owned condition started; user action required; no retry'
    if boundary:
        result['server_terminal_verified'] = True
        result['server_status'] = 'approved_handoff_boundary'
        result['queue_eta_reason'] = 'All four owned conditions terminal; original FileExistsError is the reserved seed62 boundary'
    result['transferred_job_ids'] = sorted(selected)
    result['transferred_count'] = 2
    result['source_failure_preserved'] = failure
    result['approved_handoff_boundary_verified'] = bool(boundary)
    return result


def source_script(parent):
    script = old.remote_script(parent, '5090')
    extra = "\nfor name in ('handoff_seed62/reservation.json', " + ', '.join(repr('claims/' + j['id'] + '.json') for j in handoff.target_jobs(parent)) + "):\n out['files'][name]=read_record(root/name)\n"
    extra += "out['transferred_run_folder_exists']={j:(root/'run'/j).exists() for j in " + repr([j['id'] for j in handoff.target_jobs(parent)]) + "}\n"
    return script.replace('print(json.dumps(out,allow_nan=False))', extra + 'print(json.dumps(out,allow_nan=False))')


def snapshot_once(host, parent, c, runner=subprocess.run):
    if host == '5090':
        argv = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host, 'python3 -']
        kwargs = {'input': source_script(parent)}
    else:
        import shlex
        spec = c['hosts']['5080']
        command = shlex.join([spec['python'], spec['root'] + '/control/capacity_handoff_runtime.py', '--bundle', spec['root'], '--mode', 'observe'])
        argv = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host, command]
        kwargs = {}
    output = runner(argv, text=True, capture_output=True, timeout=90, **kwargs)
    old.require(output.returncode == 0, 'SSH observation failed; raw output suppressed')
    return json.loads(output.stdout)


def cached_terminal(path, analyser):
    receipt = old.read(path)
    if not receipt.get('server_terminal_verified'):
        return None
    snapshot = Path(receipt['snapshot']).resolve()
    old.require(snapshot.is_relative_to(path.parent.resolve()), 'Unsafe terminal cache path')
    raw = snapshot.read_bytes()
    old.require(hashlib.sha256(raw).hexdigest() == receipt['snapshot_sha256'], 'Terminal cache SHA changed')
    result = analyser(json.loads(raw))
    old.require(result['server_terminal_verified'], 'Cached terminal no longer verifies')
    result['reused_terminal_evidence'] = True
    return json.loads(raw), result


def run_once(host, runner=subprocess.run):
    requested = time.time()
    stamp = datetime.fromtimestamp(requested, timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = B / 'hourly_monitor' / (stamp + '_' + host)
    latest = B / 'hourly_monitor' / ('latest_' + host + '.json')
    calls = 0
    snapshot = None
    try:
        parent = old.load_contract(PARENT)
        p, c, meta = handoff.load(B)
        old.require(p == parent, 'Parent current differs')
        q = handoff.verify_native(B, parent, c)
        handoff.validate_reservation(parent, c, q, old.read(B / 'source_reservation.json'))
        analyser = (lambda raw: analyse_target(raw, parent, c)) if host == '5080' else (lambda raw: analyse_source(raw, parent, c, q))
        cache = cached_terminal(latest, analyser) if latest.exists() else None
        if cache:
            snapshot, analysis = cache
        else:
            calls = 1
            snapshot = snapshot_once(host, parent, c, runner)
            old.require(requested - 120 <= snapshot['observed_unix'] <= time.time() + 120, 'Stale snapshot returned')
            analysis = analyser(snapshot)
        historical = None
        if host == '5080':
            historical_cache = cached_terminal(PARENT / 'hourly_monitor/latest_5080.json', lambda raw: old.analyse(raw, parent, '5080', reused=True))
            old.require(historical_cache is not None, 'Original Taxi/RAF12 terminal evidence missing')
            historical = historical_cache[1]
        old.write(folder / 'host_snapshot.json', snapshot)
        analysis['historical_original_5080'] = historical
        analysis['owned_conditions'] = 2 if host == '5080' else 4
        analysis['original_campaign_total'] = 18
        analysis['operational_contract_sha256'] = meta['contract_sha256']
    except Exception as error:
        analysis = {'host': host, 'status': 'observation_failed_not_training_failure', 'observed_unix': None,
                    'error_type': type(error).__name__, 'issue': str(error), 'rows': [],
                    'status_counts': {'completed': 0, 'failed': 0, 'in_progress': 0, 'pending': 0, 'unknown': 2 if host == '5080' else 4},
                    'server_terminal_verified': False, 'queue_eta_reason': '이번 조회 실패; 과거 결과를 새 관측으로 표시하지 않음'}
    old.write(folder / 'analysis.json', analysis)
    (folder / 'report.md').write_text(old.report(analysis))
    receipt = {'host': host, 'requested_unix': requested, 'observed_unix': analysis['observed_unix'],
               'SSH_invocations': calls, 'status': analysis.get('status', 'observed'),
               'status_counts': analysis['status_counts'], 'server_terminal_verified': analysis['server_terminal_verified'],
               'reused_terminal_evidence': analysis.get('reused_terminal_evidence', False),
               'snapshot': str(folder / 'host_snapshot.json') if snapshot is not None else None,
               'snapshot_sha256': hashlib.sha256((folder / 'host_snapshot.json').read_bytes()).hexdigest() if snapshot is not None and (folder / 'host_snapshot.json').exists() else None,
               'analysis': str(folder / 'analysis.json'), 'report': str(folder / 'report.md')}
    old.write(latest, receipt)
    print(json.dumps(receipt, ensure_ascii=False))
    return int('error_type' in analysis)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', choices=('5080', '5090'), required=True)
    raise SystemExit(run_once(parser.parse_args().host))
