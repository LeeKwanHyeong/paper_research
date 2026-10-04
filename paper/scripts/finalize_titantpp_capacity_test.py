"""One bounded hourly step: proven capacity18 -> immutable originals -> Test6.

Pending means zero network/GPU calls. Deployment/launch intents are exclusive;
ambiguous actions and scientific failures never trigger an automatic retry.
"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / 'search_artifacts/titantpp_history_width8_12_dual_20261004_v1'
B = ROOT / 'search_artifacts/titantpp_history_capacity_handoff_20261004_v1'
HERE = ROOT / 'reports/titantpp_width8_12_test_20261004_v1'
AUTHORITY = ROOT / 'reports/titantpp_capacity_test_handoff_20261004_v1/approval.json'
CAMPAIGN = HERE / 'intermittent'
STATE = B / 'test_finalizer'
REMOTE = '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_width8_12_test_20261004_v1_intermittent_5080'
DATA_ROOT = '/home/leekwanhyeong/workspace/paper_research'
SESSION = 'titantpp_width8_12_test_intermittent_5080'
MAX_FILE = 64 * 1024**2
MAX_TOTAL = 1024**3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canon(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def write(path, value, exclusive=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as stream: stream.write(raw)
    else:
        temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
        temporary.write_text(raw); os.replace(temporary, path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def observer():
    sys.path.insert(0, str(ROOT / 'paper/scripts'))
    return module(ROOT / 'paper/scripts/observe_titantpp_capacity_handoff_hourly.py', '_finalizer_observer')


def validate_authority(parent_sha):
    approval = read(AUTHORITY)
    scope = approval['evaluation']
    require(approval['approved'] is True and approval['user_instruction'] and approval['parent_contract_sha256'] == parent_sha,
            'Missing scoped human authority')
    require(scope['deferred_until_training_complete'] == ['intermittent_frozen_5000'] and scope['widths'] == [8, 12]
            and scope['seeds'] == [42, 52, 62] and scope['splits'] == ['validation', 'test']
            and scope['new_training'] is False and scope['raw_predictions_written'] is False,
            'Intermittent Test authority changed')
    return sha(AUTHORITY)


def latest_refs():
    refs = {}
    for role, path in [('target2', B / 'hourly_monitor/latest_5080.json'), ('source4', B / 'hourly_monitor/latest_5090.json'),
                       ('historical12', PARENT / 'hourly_monitor/latest_5080.json')]:
        if not path.exists(): return None
        receipt = read(path)
        if receipt.get('server_terminal_verified') is not True: return None
        refs[role] = {'snapshot': receipt['snapshot'], 'snapshot_sha256': receipt['snapshot_sha256']}
    return refs


def verify_snapshot(ref, base, analyse):
    path = Path(ref['snapshot']).resolve()
    require(path.is_relative_to((base / 'hourly_monitor').resolve()) and not path.is_symlink(), 'Unsafe completion snapshot path')
    require(sha(path) == ref['snapshot_sha256'], 'Completion snapshot digest differs')
    snapshot = read(path); analysis = analyse(snapshot)
    require(analysis['server_terminal_verified'] is True, 'Reanalysed terminal is unknown/incomplete')
    return snapshot, analysis


def gate(refs=None, obs=None):
    refs = latest_refs() if refs is None else refs
    if refs is None: return None
    obs = observer() if obs is None else obs
    parent = obs.old.load_contract(PARENT)
    p, c, meta = obs.handoff.load(B)
    require(p == parent, 'Current parent differs')
    q = obs.handoff.verify_native(B, parent, c)
    obs.handoff.validate_reservation(parent, c, q, read(B / 'source_reservation.json'))
    authority = validate_authority(obs.handoff.PARENT_SHA)
    target = verify_snapshot(refs['target2'], B, lambda raw: obs.analyse_target(raw, parent, c))
    source = verify_snapshot(refs['source4'], B, lambda raw: obs.analyse_source(raw, parent, c, q))
    historical = verify_snapshot(refs['historical12'], PARENT, lambda raw: obs.old.analyse(raw, parent, '5080', reused=True))
    expected_counts = (2, 4, 12)
    rows = []
    for pair, expected in zip((target, source, historical), expected_counts):
        analysis = pair[1]
        require(analysis['status_counts'] == {'completed': expected, 'failed': 0, 'in_progress': 0, 'pending': 0, 'unknown': 0}
                and len(analysis['rows']) == expected and all(row['terminal_verified'] for row in analysis['rows']), 'All18 conditions are not terminal verified')
        rows.extend(analysis['rows'])
    require(source[1]['approved_handoff_boundary_verified'] is True, 'Source queue boundary unverified')
    require({r['job']['id'] for r in rows} == {j['id'] for j in parent['jobs']} and len(rows) == 18, 'Duplicate/missing scientific condition')
    return {'parent': parent, 'contract': c, 'target': target[0], 'source': source[0],
            'receipt': {'status': 'capacity18_terminal_verified', 'conditions': 18, 'intermittent_conditions': 6,
                        'refs': refs, 'authority_sha256': authority, 'parent_contract_sha256': obs.handoff.PARENT_SHA,
                        'operational_contract_sha256': meta['contract_sha256'], 'verified_unix': time.time()}}


def safe_relative(name):
    path = Path(name)
    require(not path.is_absolute() and '..' not in path.parts and bool(path.parts), 'Unsafe original/archive path')
    return path


def verify_original(folder, job, digest, proof):
    folder = Path(folder); terminal = folder / 'terminal_manifest.json'
    require(not terminal.is_symlink() and sha(terminal) == proof['terminal_manifest_sha256'], 'Original manifest raw SHA differs')
    manifest = read(terminal)
    require(manifest['status'] == 'complete' and manifest['scientific_success'] is True and manifest['job'] == job
            and manifest['contract_sha256'] == digest, 'Original manifest identity differs')
    for name, expected in manifest['files'].items():
        relative = safe_relative(name); path = folder / relative
        require(not any(token in part.lower() for part in relative.parts for token in ('test', 'heldout', 'held_out')),
                'Non-validation training original forbidden')
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(folder.resolve())
                and path.stat().st_size < MAX_FILE and sha(path) == expected, 'Original file raw SHA differs: ' + name)
    return manifest


def retrieval_script(remote, jobs, digest, completed):
    return 'REMOTE=' + repr(remote) + '\nJOBS=' + repr(jobs) + '\nDIGEST=' + repr(digest) + '\nCOMPLETED=' + repr(completed) + '\n' + r'''
import hashlib,json,sys,tarfile
from pathlib import Path
root=Path(REMOTE).resolve();total=0
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
canon=lambda c:hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
assert canon(json.loads((root/'execution_contract.json').read_text()))==DIGEST
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for job in JOBS:
  folder=root/'run'/job['id'];path=folder/'terminal_manifest.json'
  assert sha(path)==COMPLETED[job['id']]['terminal_manifest_sha256']
  m=json.loads(path.read_text());assert m['status']=='complete' and m['scientific_success'] is True and m['job']==job and m['contract_sha256']==DIGEST
  for name,expected in [*m['files'].items(),('terminal_manifest.json',COMPLETED[job['id']]['terminal_manifest_sha256'])]:
   rel=Path(name);p=folder/rel
   assert not rel.is_absolute() and '..' not in rel.parts and not any(t in part.lower() for part in rel.parts for t in ('test','heldout','held_out'))
   assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(folder.resolve()) and p.stat().st_size<64*1024**2
   total+=p.stat().st_size;assert total<1024**3
   assert sha(p)==expected
   archive.add(p,arcname=job['id']+'/'+name,recursive=False)
   assert sha(p)==expected
'''


def retrieve_originals(host, remote, jobs, digest, completed, destination, runner=subprocess.run):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    missing = []
    for job in jobs:
        if (destination / job['id']).exists(): verify_original(destination / job['id'], job, digest, completed[job['id']])
        else: missing.append(job)
    if not missing: return {'host': host, 'remote_calls': 0, 'verified_conditions': len(jobs)}
    with tempfile.TemporaryDirectory(prefix='capacity-originals-', dir=STATE) as temporary:
        stage = Path(temporary); archive = stage / 'originals.tar.gz'
        with archive.open('xb') as output:
            result = runner(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host, 'python3 -'],
                            input=retrieval_script(remote, missing, digest, completed).encode(), stdout=output,
                            stderr=subprocess.PIPE, timeout=180)
        require(result.returncode == 0, 'Original retrieval failed; no evaluation launched')
        require(archive.stat().st_size < MAX_TOTAL, 'Original archive too large')
        with tarfile.open(archive) as stream:
            members = stream.getmembers(); total = 0
            for entry in members:
                path = safe_relative(entry.name); total += entry.size
                require(entry.isfile() and not entry.issym() and not entry.islnk() and path.parts[0] in {j['id'] for j in missing}
                        and entry.size < MAX_FILE and total < MAX_TOTAL, 'Unsafe original archive member')
            require(len({entry.name for entry in members}) == len(members), 'Duplicate archive entry')
            stream.extractall(stage / 'unpacked', filter='data')
        for job in missing:
            folder = stage / 'unpacked' / job['id']; manifest = verify_original(folder, job, digest, completed[job['id']])
            require({str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()} == {*manifest['files'], 'terminal_manifest.json'}, 'Unbound original archive file')
            shutil.move(str(folder), destination / job['id'])
    return {'host': host, 'remote_calls': 1, 'verified_conditions': len(jobs)}


def prepare_followup():
    followup = module(HERE / 'followup.py', '_capacity_test_followup')
    return followup.main(SimpleNamespace(original_root=[PARENT / 'retrieved/5090/run', B / 'retrieved/5080/run'],
                                       training_contract=[B / 'execution_contract.json'], remote_root=REMOTE,
                                       dataset_root=DATA_ROOT, output=CAMPAIGN, prepare=True))


def validate_request(request, native=None):
    require(request['root'] == REMOTE and request['conditions'] == 6 and request['dataset_transfer_required'] is False,
            'Evaluation deployment scope changed')
    prefixes = [HERE.resolve(), (PARENT / 'frozen_source').resolve(), (PARENT / 'retrieved/5090/run').resolve(),
                (B / 'source').resolve(), (B / 'retrieved/5080/run').resolve()]
    require(any(name.endswith('/sample_data/.keep') for name in request['files']), 'Frozen source marker missing from deployment')
    for name, digest in request['files'].items():
        path = ROOT / safe_relative(name)
        require(path.is_file() and not path.is_symlink() and any(path.resolve().is_relative_to(prefix) for prefix in prefixes)
                and sha(path) == digest and path.stat().st_size < MAX_FILE, 'Non-whitelisted or changed deployment file')
    require(sum((ROOT / name).stat().st_size for name in request['files']) < MAX_TOTAL, 'Evaluation deployment too large')
    c = read(CAMPAIGN / 'execution_contract.json')
    native = read(B / 'execution_contract.json')['hosts']['5080'] if native is None else native
    expected_env = {k: v for k, v in native['environment'].items() if k != 'SOURCE_REVISION'}
    expected_env['PYTHONDONTWRITEBYTECODE'] = '1'
    require(c['resources']['python'] == native['python'] and c['resources']['gpu_uuid'] == native['gpu_uuid']
            and c['environment'] == expected_env and request['environment'] == expected_env, 'Qualified native Python/GPU/environment differs')
    require(c['resources']['root'] == REMOTE and c['resources']['concurrent_workers'] == 1
            and 0 < c['resources']['campaign_timeout_seconds'] <= 7200 and 0 < c['resources']['condition_timeout_seconds'] <= 900
            and c['resources']['output_limit_gib'] <= 1
            and c['approval']['retraining_authorized'] is False and c['raw_predictions_written'] is False
            and c['validation_gate']['conditions'] == 6 and c['validation_gate']['all_before_test'] is True, 'Evaluation resources/training gate changed')
    expected = [c['resources']['python'], str(Path(REMOTE) / HERE.relative_to(ROOT) / 'pipeline.py'),
                '--root', REMOTE, '--campaign', str(Path(REMOTE) / CAMPAIGN.relative_to(ROOT)), '--device', 'cuda']
    require(request['command'] == expected, 'Unknown evaluation entrypoint')
    return c


def ssh(host, script, runner=subprocess.run, timeout=60, stdin=None):
    kwargs = {'capture_output': True, 'timeout': timeout}
    if stdin is None: kwargs.update(input=script, text=True); command = 'python3 -'
    else: kwargs['stdin'] = stdin; command = shlex.join(['python3', '-c', script])
    outcome = runner(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host, command], **kwargs)
    require(outcome.returncode == 0, 'SSH action failed; preserved intent prohibits automatic retry')
    return json.loads(outcome.stdout)


def deploy_launch(request, runner=subprocess.run, native=None):
    native = read(B / 'execution_contract.json')['hosts']['5080'] if native is None else native
    c = validate_request(request, native); digest = canon(request)
    for phase in ('deployment', 'preflight', 'launch'):
        intent, receipt = STATE / (phase + '_intent.json'), STATE / (phase + '_receipt.json')
        if receipt.exists():
            require(read(receipt)['request_sha256'] == digest, 'Existing action receipt changed')
            continue
        require(not intent.exists(), 'Ambiguous ' + phase + ' intent preserved; automatic retry forbidden')
        write(intent, {'request_sha256': digest, 'requested_unix': time.time()}, exclusive=True)
        if phase == 'deployment':
            archive = STATE / 'deployment.tar.gz'
            manifest = STATE / 'deployment_manifest.json'; write(manifest, request['files'], exclusive=True)
            with tarfile.open(archive, 'x:gz') as stream:
                for name in sorted(request['files']): stream.add(ROOT / name, arcname=name, recursive=False)
                stream.add(manifest, arcname='deployment_manifest.json', recursive=False)
            script = 'REMOTE=' + repr(REMOTE) + '\nFILES=' + repr(request['files']) + '\n' + r'''
import hashlib,json,sys,tarfile
from pathlib import Path
root=Path(REMOTE);root.mkdir(exist_ok=False);seen=set()
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz') as stream:
 for entry in stream:
  p=Path(entry.name)
  assert entry.isfile() and not p.is_absolute() and '..' not in p.parts and entry.name in {*FILES,'deployment_manifest.json'} and entry.name not in seen
  seen.add(entry.name);stream.extract(entry,root,filter='data')
assert seen=={*FILES,'deployment_manifest.json'}
assert json.loads((root/'deployment_manifest.json').read_text())==FILES
for name,digest in FILES.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
print(json.dumps({'status':'deployed','files':len(FILES),'root':str(root)}))
'''
            with archive.open('rb') as stream: result = ssh('5080', script, runner, timeout=180, stdin=stream)
        elif phase == 'preflight':
            command = request['preflight_command']
            expected = [c['resources']['python'], str(Path(REMOTE) / HERE.relative_to(ROOT) / 'preflight.py'),
                        '--root', REMOTE, '--campaign', str(Path(REMOTE) / CAMPAIGN.relative_to(ROOT))]
            require(command == expected, 'Unknown preflight entrypoint')
            result = ssh('5080', 'import subprocess,json\np=subprocess.run(' + repr(command) + ',capture_output=True,text=True,timeout=180)\nassert p.returncode==0,p.stderr\nprint(p.stdout)', runner, timeout=200)
            require(result['status'] == 'complete' and result['inference_calls'] == result['gpu_calls'] == 0, 'Byte/existence preflight failed')
        else:
            env = {**native['environment'], 'PYTHONDONTWRITEBYTECODE': '1'}
            command = 'cd ' + shlex.quote(REMOTE) + ' && exec ' + shlex.join(['timeout', '--signal=TERM', '--kill-after=15s',
                       str(c['resources']['campaign_timeout_seconds']), 'env', *[k + '=' + v for k, v in env.items()], *request['command']])
            command += ' > ' + shlex.quote(str(Path(REMOTE) / CAMPAIGN.relative_to(ROOT) / 'pipeline.log')) + ' 2>&1'
            script = 'SESSION=' + repr(SESSION) + '\nCOMMAND=' + repr(command) + '\nCAMPAIGN=' + repr(str(Path(REMOTE) / CAMPAIGN.relative_to(ROOT))) + '\n' + r'''
import json,subprocess
from pathlib import Path
campaign=Path(CAMPAIGN)
assert not (campaign/'pipeline_status.json').exists() and not (campaign/'campaign.lock').exists()
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=15).strip()
assert subprocess.run(['/usr/bin/tmux','has-session','-t',SESSION],capture_output=True).returncode!=0
subprocess.run(['/usr/bin/tmux','new-session','-d','-s',SESSION,COMMAND],check=True)
print(json.dumps({'status':'launch_requested','session':SESSION,'actual_evaluation_started_confirmed':False}))
'''
            result = ssh('5080', script, runner)
        write(receipt, {'request_sha256': digest, 'result': result, 'received_unix': time.time()}, exclusive=True)
    return {'status': 'running', 'launch_requested': True, 'automatic_retry': False}


def aggregate_names():
    registry = read(CAMPAIGN / 'evaluation_registry.json')
    names = ['pipeline_status.json', 'inference_completion.json', 'validation_gate.json', 'pipeline.log']
    for row in registry['rows']:
        identity = f"{row['dataset']}__{row['model']}__seed{row['seed']}"
        for split in ('validation', 'test'):
            names += [f'runs/{split}/{identity}/receipt.json', f'runs/{split}/{identity}/failure.json', f'logs/{split}__{identity}.log']
    return names


def collect_aggregates(runner=subprocess.run):
    remote = str(Path(REMOTE) / CAMPAIGN.relative_to(ROOT))
    script = 'CAMPAIGN=' + repr(remote) + '\nNAMES=' + repr(aggregate_names()) + '\n' + r'''
import hashlib,json,time,subprocess
from pathlib import Path
root=Path(CAMPAIGN).resolve();result={'observed_unix':time.time(),'files':{}}
for name in NAMES:
 path=root/name
 if not path.exists():continue
 assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root) and path.stat().st_size<2*1024**2
 raw=path.read_bytes();result['files'][name]={'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode()}
result['session_alive']=subprocess.run(['/usr/bin/tmux','has-session','-t','titantpp_width8_12_test_intermittent_5080'],capture_output=True,timeout=15).returncode==0
print(json.dumps(result))
'''
    result = ssh('5080', script, runner)
    allowed = set(aggregate_names())
    for name, proof in result['files'].items():
        require(name in allowed and hashlib.sha256(proof['text'].encode()).hexdigest() == proof['sha256'], 'Aggregate transport digest differs')
        path = CAMPAIGN / safe_relative(name); path.parent.mkdir(parents=True, exist_ok=True)
        immutable = name.endswith('/receipt.json') or name in ('inference_completion.json', 'validation_gate.json')
        if immutable and path.exists(): require(sha(path) == proof['sha256'], 'Completed aggregate changed')
        else:
            temporary = path.with_name(path.name + '.tmp'); temporary.write_text(proof['text']); os.replace(temporary, path)
    status = read(CAMPAIGN / 'pipeline_status.json') if (CAMPAIGN / 'pipeline_status.json').exists() else {'status': 'launch_requested'}
    if status['status'] != 'complete' and status['status'] != 'failed' and not result['session_alive']:
        return {'status': 'failed', 'message': 'Bounded evaluation session exited without a complete receipt; preserve outputs, no retry'}
    return status


def verify_completion():
    c, registry = read(CAMPAIGN / 'execution_contract.json'), read(CAMPAIGN / 'evaluation_registry.json')
    completion = read(CAMPAIGN / 'inference_completion.json'); gate_receipt = read(CAMPAIGN / 'validation_gate.json')
    expected = {(model, seed) for model in ('titantpp_history_mlp_width8', 'titantpp_history_mlp_width12') for seed in (42, 52, 62)}
    require(len(registry['rows']) == 6 and {(r['model'], r['seed']) for r in registry['rows']} == expected
            and all(r['dataset'] == 'intermittent_frozen_5000' for r in registry['rows']), 'Unknown/partial Test condition')
    require(completion['status'] == 'complete' and completion['conditions'] == 6 and completion['full_population_splits'] == 12
            and completion['contract_sha256'] == sha(CAMPAIGN / 'execution_contract.json')
            and completion['new_training'] is False and completion['selection_changed'] is False
            and completion['raw_predictions_written'] is False and len(completion['completed']) == 12, 'Incomplete/foreign Test completion')
    pipeline = module(HERE / 'pipeline.py', '_capacity_finalizer_receipts')
    checked = []
    for split in ('validation', 'test'):
        for row in registry['rows']:
            checked.append(pipeline.check_receipt(CAMPAIGN / 'runs' / split / pipeline.identity(row), row, split, c, CAMPAIGN))
    require(completion['completed'] == checked and gate_receipt['passed'] is True and gate_receipt['conditions'] == 6
            and gate_receipt['contract_sha256'] == completion['contract_sha256'] and gate_receipt['rows'] == checked[:6], 'Validation-before-Test/completion receipt differs')
    return completion


def analyse():
    output = CAMPAIGN / 'results'
    if output.exists():
        receipt = read(output / 'analysis_receipt.json')
        require(receipt['status'] == 'complete' and receipt['new_conditions'] == 6 and receipt['new_inference_calls_here'] == 0,
                'Existing analysis is incomplete')
        for name, digest in receipt['outputs'].items(): require(sha(output / safe_relative(name)) == digest, 'Analysis output SHA differs')
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            module(HERE / 'analyze.py', '_capacity_followup_analysis').main(SimpleNamespace(campaign=[CAMPAIGN], output=output))
    return str(output)


def run_once(runner=subprocess.run):
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'driver.lock').open('a+') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return {'status': 'pending', 'reason': 'Another finalizer owns the lock', 'remote_calls': 0}
        try:
            saved = read(STATE / 'gate_receipt.json') if (STATE / 'gate_receipt.json').exists() else None
            verified = gate(saved['refs'] if saved else None)
            if verified is None: return {'status': 'pending', 'reason': 'Capacity18 terminal gate not confirmed', 'remote_calls': 0, 'new_training': False}
            if saved:
                require(saved['authority_sha256'] == verified['receipt']['authority_sha256'], 'Authority changed after gate')
            else: write(STATE / 'gate_receipt.json', verified['receipt'], exclusive=True)
            parent, c = verified['parent'], verified['contract']
            if (STATE / 'completion.json').exists():
                request = read(CAMPAIGN / 'deployment_request.json')
                require(read(STATE / 'launch_receipt.json')['request_sha256'] == canon(request), 'Completed launch request changed')
                validate_request(request, c['hosts']['5080'])
                verify_completion(); output = analyse()
                result = read(STATE / 'completion.json')
                require(result['status'] == 'completed' and result['report'] == output, 'Saved finalizer completion differs')
                return {**result, 'reused_completed_evidence': True, 'remote_calls': 0}
            if not (STATE / 'launch_receipt.json').exists():
                retrievals = []
                for host, contract, snapshot, destination in [('5090', parent, verified['source'], PARENT / 'retrieved/5090/run'),
                                                             ('5080', c, verified['target'], B / 'retrieved/5080/run')]:
                    completed = (snapshot['files'].get('failure.json') if host == '5090' else snapshot['files']['status.json'])['completed']
                    jobs = [j for j in contract['jobs'] if j['host'] == host and j['dataset'] == 'intermittent_frozen_5000' and j['id'] in completed]
                    require(len(jobs) == (4 if host == '5090' else 2), 'Original retrieval ownership/count differs')
                    retrievals.append(retrieve_originals(host, contract['hosts'][host]['root'], jobs, canon(contract), completed, destination, runner))
                prepared = prepare_followup(); require(prepared['status'] == 'prepared', 'Followup preparation not ready/no retry')
                request = read(CAMPAIGN / 'deployment_request.json')
                result = deploy_launch(request, runner, c['hosts']['5080'])
                result['retrievals'] = retrievals; return result
            request = read(CAMPAIGN / 'deployment_request.json')
            require(read(STATE / 'launch_receipt.json')['request_sha256'] == canon(request), 'Launch deployment request changed')
            validate_request(request, c['hosts']['5080'])
            status = collect_aggregates(runner)
            if status['status'] == 'failed': return {'status': 'failure', 'reason': 'Evaluation failed; preserved originals/receipts; no retry', 'pipeline_status': status}
            if status['status'] != 'complete': return {'status': 'running', 'pipeline_status': status, 'automatic_retry': False}
            verify_completion(); output = analyse()
            result = {'status': 'completed', 'conditions': 6, 'population_splits': 12, 'report': output, 'new_training': False,
                      'selection_changed': False, 'raw_predictions_written': False, 'new_independent_test_claim': False}
            write(STATE / 'completion.json', result); return result
        except Exception as error:
            result = {'status': 'unknown' if not (STATE / 'deployment_intent.json').exists() else 'failure',
                      'error_type': type(error).__name__, 'reason': str(error), 'automatic_retry': False, 'new_training': False}
            write(STATE / 'last_failure.json', result); return result


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(run_once(), ensure_ascii=False, allow_nan=False))
