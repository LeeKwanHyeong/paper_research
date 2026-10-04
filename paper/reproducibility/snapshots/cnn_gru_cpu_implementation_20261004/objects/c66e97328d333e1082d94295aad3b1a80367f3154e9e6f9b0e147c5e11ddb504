"""One explicit recovery of four unfinished fits; completed evidence is immutable."""
from copy import deepcopy
from pathlib import Path
import subprocess
from paper.scripts.observed_slot_parallel_common import require, sha_file, sha_json, source_manifest
from paper.scripts.local_detail_benchmark_contract import read

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'local_detail_instacart_remaining_recovery_v1'
BASE = 'paper/contracts/local_detail_instacart_replication_v1.json'
BASE_SHA = 'f03afd26bf0085d6f9f6b601ae4b2b25072bf6fc67897c2173653536ac209aa7'
LINEAGE = 'paper/contracts/local_detail_instacart_remaining_lineage_v1.json'
RUN_NAME = 'local_detail_instacart_seed62_remaining_5090_20260926_v1'
REMAINING = ('rmtpp', 'thp', 'nhp', 'sahp')
ENTRYPOINTS = ('paper/scripts/run_local_detail_benchmark.py',
              'paper/scripts/run_local_detail_instacart_remaining.py',
              'paper/scripts/local_detail_instacart_remaining_contract.py',
              'paper/scripts/verify_local_detail_benchmark.py')


def source_files():
    parent = read(ROOT / BASE)
    return source_manifest(ROOT, entrypoints=ENTRYPOINTS,
                           extra_files=(*parent['source']['files'], BASE, LINEAGE))


def protocol():
    parent = read(ROOT / BASE)
    require(sha_json(parent) == BASE_SHA, 'Parent comparison contract changed')
    c = {k: deepcopy(v) for k, v in parent.items() if k not in ('source', 'verification', 'cost_estimate')}
    c.update(schema=SCHEMA, total_arms=4, endpoint_replays=8,
             parent_contract_sha256=BASE_SHA, qualification_seed=62,
             maximum_optimizer_steps=4 * 300 * 15557)
    spec = c['hosts']['5090']
    spec['root'] = str(Path(spec['root']).parent / RUN_NAME)
    spec.update(source_root=spec['root']+'/source', output_dir=spec['root']+'/run', tmux=RUN_NAME)
    c['process_environment']['5090']['XDG_CACHE_HOME'] = spec['root']+'/cache'
    c['replication']['jobs'] = [{'host':'5090', 'dataset':'insta_market_basket', 'seed':62}]
    c['replication']['seeds'] = [62]
    c['remaining_recovery'] = read(ROOT / LINEAGE)
    c['execution_arms'] = list(REMAINING)
    c['execution_isolation'] = {'fresh_exec_process_per_arm':True, 'automatic_retry':False,
        'progress_write_interval_seconds':10, 'stack_dump_after_stall_seconds':300,
        'no_progress_timeout_seconds':1800,
        'reason':'Contain long-lived native process state; capture and stop a stalled worker without retry'}
    c['launch']['qualification_scope'] = 'same-host seed62 synthetic 60 updates, then four remaining fits once'
    c['policy']['other_hosts'] = 'preserve all completed 5080 and seed42/52/62 predecessor results'
    return c


def validate_contract(c, *, verify_source=True):
    for key, value in protocol().items():
        require(c.get(key) == value, 'Frozen recovery protocol changed: ' + key)
    require(c['execution_arms'] == list(REMAINING), 'Only four remaining fits are authorized')
    require(c['remaining_recovery']['remaining_arms'] == list(REMAINING), 'Lineage scope changed')
    source = c['source']
    require(len(source['base_git_revision']) == 40 and sha_json(source['files']) == source['files_sha256'],
            'Source closure inconsistent')
    if verify_source:
        require(source_files() == source['files'], 'Frozen recovery source changed')
    require(c['verification']['status'] == 'passed_synthetic_cpu' and c['verification']['failed'] == 0,
            'CPU verification required')


def verify_predecessor(c, host):
    require(host == '5090', 'Recovery host changed')
    lineage = c['remaining_recovery']; root = Path(lineage['old_root'])
    old = read(root/'frozen_execution/execution_contract.json')
    require(sha_json(old) == BASE_SHA and old['source']['files_sha256'] == lineage['old_source_files_sha256'],
            'Predecessor identity changed')
    for name, status in (('run/status.json','stopped'), ('dispatch_status.json','failed')):
        record = read(root/name)
        require(record.get('status') == status and record.get('error') == lineage['old_stop_error'],
                'Predecessor was not stopped under the recovery authorization')
    for name, digest in lineage['completed_files_sha256'].items():
        require(sha_file(root/name) == digest, 'Completed predecessor evidence changed: '+name)
    for arm in REMAINING:
        path = root/'run/insta_market_basket/seed_62/runs'/arm
        require(not any(p.is_file() or p.is_symlink() for p in path.rglob('*')),
                'Unfinished arm has saved progress; fresh fit is not authorized')
    require(subprocess.run([c['hosts'][host]['tmux_binary'], 'has-session', '-t', lineage['old_tmux']],
                          capture_output=True).returncode != 0, 'Predecessor tmux is still active')
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            command = path.read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        require(str(root).encode() not in command, 'Predecessor owned process is still active')
    return root
