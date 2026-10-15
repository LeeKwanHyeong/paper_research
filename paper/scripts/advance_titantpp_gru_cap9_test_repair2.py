"""Approved CAP9 frozen evaluation. No training, reselection or automatic retry.

Preparation is local. advance waits with remote0 until the original sealed
collector has verified all nine fits and immutable originals. Native Validation9
must pass before any Test worker. Existing CNN/FULL18 receipts are reused.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import tarfile
import time
import traceback

PROJECT = Path(__file__).resolve().parents[2]
TRAIN = PROJECT / 'search_artifacts/titantpp_gru_residual_cap_validation_20261014_v1/execution_manual2'
BUNDLE = PROJECT / 'search_artifacts/titantpp_gru_cap9_test_20261015_v1_repair2'
REPORT = PROJECT / 'reports/titantpp_gru_cap9_test_20261015_v1_repair2'
TRAIN_SHA = '274ff80ef771788dc4652749bd2875716c7ddeb59ba0a1a0e497358545fe1fdc'
TRAIN_CANONICAL = '9a7e553730e36879fa89f3732c5d4a5493ed60f6bfcb1a4f826cb0990693622f'
CONTROL_SHA = '64f143793b2134e0db2fc9dd2e9ca80c782f00b5304c32989d5305e6c4c32ef7'
COLLECTION_SHA = '56740fb9797c6ad076ed5ba156c393043bd194ae196cbb45dbb057a7681b0251'
CAP = 'titantpp_cnn_gru54_residual_cap025'
CNN = 'titantpp_cnn_gru54_cnn_only'
FULL = 'titantpp_cnn_gru54'
DATASETS = ('yellow_trip_hourly', 'raf_spare_parts', 'intermittent_frozen_5000')
SEEDS = (42, 52, 62)
HOSTS = ('5080', '5090')
USER_INSTRUCTION = 'ㅇㅇ 그 연결 오류도 수정·검증하고 진행해. 기존 원본·실패 기록·선택 checkpoint는 그대로 두고 별도 경로에서 Validation9 통과하면 CAP9 Test까지 끝내서, 기존 CNN·FULL과 데이터셋별 3seed RMSE·MAE랑 seed별 차이 정리해줘.'
REMOTE = {h: '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/' + BUNDLE.name + '_' + h for h in HOSTS}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def safe(root, name):
    root = Path(root).resolve()
    p = Path(name)
    require(not p.is_absolute() and '..' not in p.parts and p.parts, 'Unsafe relative path')
    target = root / p
    require(target.resolve().is_relative_to(root) and not target.is_symlink(), 'Path escapes root')
    return target


def copy_checked(source, destination, expected):
    require(sha(source) == expected, 'Source bytes changed: ' + str(source))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open('rb') as src, destination.open('xb') as dst:
        for block in iter(lambda: src.read(1024 * 1024), b''):
            dst.write(block)
    require(sha(destination) == expected, 'Copied bytes differ')


def evaluator():
    spec = importlib.util.spec_from_file_location('cap9_eval_operational', BUNDLE / 'operation/evaluate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def training_contract():
    require(sha(TRAIN / 'execution_contract.json') == TRAIN_SHA, 'Original training contract changed; remote0')
    c = read(TRAIN / 'execution_contract.json')
    require(canonical(c) == TRAIN_CANONICAL, 'Training canonical seal differs')
    expected = {f'{d}__{s}__{CAP}' for d in DATASETS for s in SEEDS}
    require(len(c['jobs']) == 9 and {j['id'] for j in c['jobs']} == expected, 'Foreign training cohort')
    return c


def check_receipt(receipt, contract, job, split, e):
    if split == 'validation':
        return e.verify_validation_receipt(receipt, contract, job)
    require(receipt['status'] == 'complete' and receipt['split'] == 'test'
            and receipt['contract_sha256'] == contract['contract_sha256']
            and receipt['job_id'] == job['id'] and receipt['selected_epoch'] == job['selected_epoch']
            and receipt['checkpoint_sha256'] == job['checkpoint']['sha256']
            and receipt['state_sha256'] == job['checkpoint_state_sha256']
            and receipt['source_closure'] == contract['sources']['components']['closure']
            and receipt['retrained'] is False and receipt['reselected'] is False
            and receipt['held_out_test_evaluated'] is True and receipt['full_population'] is True
            and receipt['native_runtime_checked'] is True, 'Test receipt is incomplete or foreign')
    for key in ('id', 'cohort', 'study', 'dataset', 'arm', 'seed', 'initial_state_sha256'):
        require(receipt[key] == job[key], 'Test model identity differs: ' + key)
    require(set(receipt['qualification']) == e.QUALIFICATION_KEYS
            and all(v is True for v in receipt['qualification'].values()), 'Test causal checks failed')
    population = contract['datasets'][job['dataset']]['test_population']
    require(receipt['metrics']['count'] == receipt['prediction_rows'] == population['count'], 'Test count differs')
    for key in ('target_identity_sha256', 'truth_sha256', 'time_sha256'):
        require(receipt[key] == population[key], 'Test population differs')
    e.compare_validation_metrics(receipt['metrics'], receipt['metrics'])  # finite six metrics; not a Test selection
    runtime = contract['runtimes'][job['evaluation_host']]
    e.check_runtime(receipt['runtime'], runtime, 'cuda', current_process=False)
    e.check_runtime(receipt['runtime_after'], runtime, 'cuda', current_process=False)
    require(receipt['runtime'] == receipt['runtime_after'], 'Runtime changed during Test')
    return receipt


def prepare():
    c = training_contract()
    require(not BUNDLE.exists(), 'Evaluation bundle already exists; no overwrite')
    BUNDLE.mkdir()
    REPORT.mkdir(exist_ok=True)
    files = {}
    def add(src, name, expected=None):
        expected = expected or sha(src)
        copy_checked(src, safe(BUNDLE, name), expected)
        files[name] = expected
    add(TRAIN / 'execution_contract.json', 'provenance/training_contract.json', TRAIN_SHA)
    for src, name in [('evaluate_titantpp_gru_cap9_frozen.py', 'evaluate.py'),
                      ('run_titantpp_gru_cap9_test_pipeline.py', 'pipeline.py')]:
        add(PROJECT / 'paper/scripts' / src, 'operation/' + name)
    for name, expected in c['source']['files'].items():
        add(TRAIN / 'source' / name, 'source/' + name, expected)
    for name, expected in c['input_files'].items():
        add(TRAIN / 'source' / name, 'source/' + name, expected)
    approval = {'approved': True, 'user_instruction': USER_INSTRUCTION,
        'user_instruction_sha256': hashlib.sha256(USER_INSTRUCTION.encode()).hexdigest(),
        'expected_condition_count': 9, 'job_ids': sorted(j['id'] for j in c['jobs']),
        'training_contract_sha256': TRAIN_CANONICAL, 'new_training': False,
        'checkpoint_reselection': False, 'local_root': str(BUNDLE), 'remote_roots': REMOTE,
        'test_independent': False, 'automatic_retry': False, 'approved_date_kst': '2026-10-15'}
    write(BUNDLE / 'approval.json', approval)
    files['approval.json'] = sha(BUNDLE / 'approval.json')
    datasets, runtimes, references, dependencies = {}, {}, [], {}
    definitions = {d['dataset_id']: d for d in c['datasets']}
    require(len(c['baseline_references']) == 18, 'Expected existing CNN/FULL18')
    for ref in c['baseline_references']:
        root = Path(ref['root'])
        for path_key, hash_key in [('parent_contract_path', 'parent_contract_raw_sha256'),
                                   ('results_manifest_path', 'results_manifest_sha256')]:
            path = Path(ref[path_key]); expected = ref[hash_key]
            require(sha(path) == expected, 'Baseline evidence changed')
            dependencies[str(path)] = expected
        old = read(ref['parent_contract_path'])
        manifest = read(ref['results_manifest_path'])
        require(manifest['contract_sha256'] == old['contract_sha256'], 'Baseline manifest contract differs')
        old_job = next(j for j in old['jobs'] if j['id'] == ref['id'])
        require(old_job['arm'] in (CNN, FULL) and old_job['selected_epoch'] == ref['selected_epoch']
                and old_job['checkpoint']['sha256'] == ref['checkpoint_sha256'], 'Baseline checkpoint changed')
        name = 'results/' + ref['id'] + '/test_metrics.json'
        exit_name = 'results/' + ref['id'] + '/test_worker_exit.json'
        for n in (name, exit_name):
            expected = manifest['files'][n]
            require(sha(root / n) == expected, 'Baseline Test proof bytes differ')
            dependencies[str(root / n)] = expected
            add(root / n, 'references/' + ref['id'] + '/' + Path(n).name, expected)
        actual = read(root / exit_name)
        require(actual['actual_child_exit_observed'] is True and type(actual['exit_code']) is int
                and actual['exit_code'] == 0 and actual['job_id'] == ref['id']
                and actual['contract_sha256'] == old['contract_sha256'], 'Baseline Test worker exit not verified')
        r = check_receipt(read(root / name), old, old_job, 'test', evaluator())
        spec = copy.deepcopy(old['datasets'][ref['dataset']])
        require(spec['definition'] == definitions[ref['dataset']], 'Baseline preprocessing differs from CAP training')
        if ref['dataset'] in datasets:
            require(datasets[ref['dataset']] == spec, 'Cross-seed dataset contract differs')
        else:
            datasets[ref['dataset']] = spec
            for pk, hk in [('data_path', 'data_sha256'), ('training_data_path', 'training_data_sha256'),
                           ('split_manifest_path', 'split_manifest_sha256')]:
                if spec[pk] not in files:
                    add(root / spec[pk], spec[pk], spec[hk])
        h = old_job['evaluation_host']
        require(h not in runtimes or runtimes[h] == old['runtimes'][h], 'Cross-cohort host Runtime differs')
        runtimes[h] = old['runtimes'][h]
        references.append({'id': ref['id'], 'dataset': ref['dataset'], 'seed': ref['seed'],
            'arm': ref['arm'], 'cohort': r['cohort'], 'selected_epoch': ref['selected_epoch'],
            'checkpoint_sha256': ref['checkpoint_sha256'], 'contract_sha256': old['contract_sha256'],
            'receipt_path': 'references/' + ref['id'] + '/test_metrics.json',
            'receipt_sha256': sha(root / name), 'original_receipt_path': str(root / name)})
    require({(r['dataset'], r['seed'], r['arm']) for r in references}
            == {(d, s, a) for d in DATASETS for s in SEEDS for a in (CNN, FULL)}, 'Baseline matching set differs')
    ops = {k: v for k, v in files.items() if k.startswith('operation/')}
    jobs = [{**j, 'cohort': 'gru_cap9_source134', 'study': 'components',
             'training_host': j['host'], 'evaluation_host': j['host'],
             'frozen_source_closure': c['source']['files_sha256']} for j in c['jobs']]
    draft = {'schema': 'titantpp_gru_cap9_frozen_evaluation_v1', 'cohort': 'gru_cap9_source134',
        'evaluation_scope': 'test_gru_cap9_followup', 'metrics_schema': 'quantity_time_tail_v1',
        'expected_seeds': list(SEEDS), 'expected_jobs': 9, 'binding_complete': False,
        'sources': {'components': {'root': 'source', 'files': c['source']['files'],
                    'closure': c['source']['files_sha256'], 'revision': c['source']['base_git_revision']}},
        'jobs': jobs, 'datasets': datasets, 'runtimes': runtimes,
        'evaluator_integrity': {'files': ops, 'closure': canonical(ops)},
        'training_contract_path': 'provenance/training_contract.json', 'training_contract_sha256': TRAIN_SHA,
        'partial_approval_path': 'approval.json', 'partial_approval_sha256': files['approval.json'],
        'baseline_references': references, 'test_is_independent': False, 'new_training': False,
        'checkpoint_reselection': False, 'automatic_retry': False, 'bound_files': {}}
    draft['contract_sha256'] = canonical(draft)
    write(BUNDLE / 'draft_contract.json', draft)
    # Synthetic qualification uses a separate root and an unbound copy; final contract remains exclusive.
    files['draft_contract.json'] = sha(BUNDLE / 'draft_contract.json')
    write(BUNDLE / 'preparation_inventory.json', {'files': files, 'external_dependencies': dependencies})
    return {'status': 'prepared_unbound', 'candidate_count': 9, 'baseline_count': 18, 'remote_calls': 0}


def verify_bootstrap(expected):
    require(sha(BUNDLE / 'bootstrap_integrity.json') == expected, 'Bootstrap integrity differs; remote0')
    seal = read(BUNDLE / 'bootstrap_integrity.json')
    for name, value in seal['files'].items():
        require(sha(name) == value, 'Pinned dependency changed; remote0: ' + name)
    c = training_contract()
    inventory = read(BUNDLE / 'preparation_inventory.json')
    for name, value in inventory['files'].items():
        require(sha(safe(BUNDLE, name)) == value, 'Prepared artifact changed; remote0')
    for name, value in inventory['external_dependencies'].items():
        require(sha(name) == value, 'Existing baseline proof changed; remote0')
    require(sha(TRAIN / 'parallel3_v3_control/integrity.json') == CONTROL_SHA, 'Training control seal changed')
    return c


def collection_gate(c):
    path = TRAIN / 'retrieval/final_receipt.json'
    if not path.exists():
        return None
    r = read(path)
    require(r['status'] == 'complete' and r['candidate_fit_count'] == 9 and r['seed_count'] == 3
            and r['baseline_reference_count'] == 18 and r['held_out_test_evaluated'] is False
            and r['contract_sha256'] == canonical(c) and r['contract_raw_sha256'] == TRAIN_SHA
            and r['collection_integrity_raw_sha256'] == COLLECTION_SHA,
            'Training final receipt incomplete/foreign; remote0')
    require(canonical(r['local_files']) == r['local_files_sha256'], 'Collection file list changed')
    for name, value in r['local_files'].items():
        p = Path(name)
        require(p.is_absolute() and p.resolve().is_relative_to(PROJECT) and not p.is_symlink()
                and sha(p) == value, 'Collected original/archive/summary SHA differs; remote0')
    require(sha(r['round_path']) == r['round_raw_sha256'], 'Training round differs')
    from paper.scripts import advance_titantpp_gru_cap_parallel3_v3 as monitor
    native, _, observer, collector = monitor.gate(CONTROL_SHA)
    observer, collector = monitor.helpers(native, observer, collector)
    launcher = monitor.launch.launcher_for(TRAIN)
    round_value = read(r['round_path'])
    require(round_value['all_training_terminal_verified'] is True, 'Nine training jobs not terminal')
    for host in HOSTS:
        proof = r['terminal_cache_evidence'][host]
        require(sha(proof['path']) == proof['raw_sha256']
                and sha(proof['snapshot_path']) == proof['snapshot_raw_sha256'], 'Training terminal cache changed')
        cached = observer.load_cache(TRAIN, monitor.OUTPUT, native, host, launcher)
        require(cached is not None and str(cached[0]) == proof['snapshot_path']
                and cached[1]['observed_unix'] == proof['original_observed_unix'], 'Training terminal cache differs')
        expected = collector.terminal_files(native, host, cached[1], launcher)
        for name, value in expected.items():
            require(sha(safe(TRAIN / 'retrieval/originals', name)) == value, 'Native terminal binary/original changed')
    return r


def validation_reference(endpoint):
    """Add evaluator split metadata to a copy of a verified native endpoint."""
    require(isinstance(endpoint, dict)
            and endpoint.get('evaluation_scope') == 'validation_only'
            and endpoint.get('held_out_test_evaluated') is False
            and endpoint.get('split', 'validation') == 'validation',
            'Selected endpoint is not Validation-only')
    return {**copy.deepcopy(endpoint), 'split': 'validation'}


def bind(c, receipt):
    import torch  # Metadata only; no model construction, data loader or inference.
    e = evaluator()
    contract = read(BUNDLE / 'draft_contract.json')
    contract.pop('contract_sha256')
    index = read(TRAIN / 'retrieval/retrieval_index.json')
    require(set(index['jobs']) == {j['id'] for j in c['jobs']}, 'Collected job registry differs')
    proof = 'provenance/training_final_receipt.json'
    copy_checked(TRAIN / 'retrieval/final_receipt.json', BUNDLE / proof, sha(TRAIN / 'retrieval/final_receipt.json'))
    bound = {proof: sha(BUNDLE / proof)}
    for j in contract['jobs']:
        item = index['jobs'][j['id']]
        root = TRAIN / 'retrieval/originals'
        for key in ('checkpoint', 'history', 'terminal_manifest', 'endpoint_replays', 'input_receipt', 'derived_validation_replay'):
            name = 'originals/' + item[key + '_path']
            copy_checked(safe(root, item[key + '_path']), safe(BUNDLE, name), item[key + '_sha256'])
            bound[name] = item[key + '_sha256']
        checkpoint = BUNDLE / ('originals/' + item['checkpoint_path'])
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        metadata = {k: v for k, v in payload.items() if k != 'model_state_dict'}
        metadata_name = 'metadata/' + j['id'] + '.json'
        write(BUNDLE / metadata_name, metadata)
        bound[metadata_name] = sha(BUNDLE / metadata_name)
        history_name = 'originals/' + item['history_path']
        epoch, value = e.strict_selected_epoch(e.read_selection_history(BUNDLE / history_name))
        endpoint = read(root / item['endpoint_replays_path'])['selected']
        require(epoch == payload['best_epoch'] and value == payload['selected_metric_value']
                and item['state_tensor_sha256'] == payload['model_state_sha256'] == endpoint['state_sha256'],
                'Selected checkpoint differs from immutable Validation history')
        j.update(selected_epoch=epoch, selected_metric_value=value,
            initial_state_sha256=payload['initial_state_sha256'], source_revision_history=payload['source_revision_history'],
            checkpoint_state_sha256=item['state_tensor_sha256'],
            checkpoint={'relative_path': 'originals/' + item['checkpoint_path'], 'sha256': item['checkpoint_sha256']},
            checkpoint_metadata_path=metadata_name, checkpoint_metadata_sha256=bound[metadata_name],
            selection_history_path=history_name, selection_history_sha256=item['history_sha256'],
            validation_reference=validation_reference(endpoint), binding_status='bound_terminal_immutable_originals')
        e.validate_payload(payload, j, contract['sources']['components'], contract['datasets'][j['dataset']]['definition'], metadata)
        del payload
    contract.update(binding_complete=True, training_receipt_path=proof,
                    training_receipt_sha256=bound[proof], bound_files=bound)
    contract['contract_sha256'] = canonical(contract)
    write(BUNDLE / 'contract.json', contract)
    e.verify_contract(BUNDLE)
    return contract


def transport(host, label, code, payload=None, archive=None, timeout=120):
    from paper.scripts.titantpp_inter62_transport_evidence import recorded_run
    write(BUNDLE / 'control' / (label + '_intent.json'), {'host': host, 'started_unix': time.time(),
          'automatic_retry': False, 'training_root_writes': 0})
    argv = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=30',
            '-o', 'ServerAliveCountMax=3', host, '/usr/bin/python3 -c ' + shlex.quote(code)]
    if archive:
        with Path(archive).open('xb') as stream:
            result = recorded_run(BUNDLE, label, argv, input=json.dumps(payload).encode(), stdout=stream, timeout=timeout)
    else:
        result = recorded_run(BUNDLE, label, argv, input=json.dumps(payload).encode(), timeout=timeout)
    require(result.returncode == 0, 'Remote operation failed; no retry: ' + label + ': ' + result.stderr.decode(errors='replace')[-2000:])
    return result.stdout


def pack(names, path):
    names = sorted(set(names))
    require(all(safe(BUNDLE, n).is_file() for n in names), 'Packet file missing')
    manifest = {n: sha(BUNDLE / n) for n in names}
    with Path(path).open('xb') as out, tarfile.open(fileobj=out, mode='w:gz') as tar:
        raw = json.dumps(manifest, sort_keys=True).encode()
        info = tarfile.TarInfo('packet_manifest.json'); info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
        for n in names:
            tar.add(BUNDLE / n, arcname=n, recursive=False)
    return manifest


DEPLOY = r'''
import hashlib,json,sys,tarfile
from pathlib import Path
p=json.load(sys.stdin); r=Path(p['root'])
assert str(r) in p['allowed_roots'] and r.name.startswith('titantpp_gru_cap9_test_20261015_v1_')
archive=r/'incoming'/p['archive_name']; assert hashlib.sha256(archive.read_bytes()).hexdigest()==p['archive_sha256']
with tarfile.open(archive) as t:
 members=t.getmembers(); manifest=json.load(t.extractfile('packet_manifest.json'))
 assert len(members)==len(manifest)+1 and {m.name for m in members}==set(manifest)|{'packet_manifest.json'}
 raws={}
 for m in members:
  if m.name=='packet_manifest.json':continue
  n=Path(m.name); target=r/n
  assert m.isfile() and not n.is_absolute() and '..' not in n.parts and target.resolve().is_relative_to(r.resolve())
  raw=t.extractfile(m).read(); assert hashlib.sha256(raw).hexdigest()==manifest[m.name]; raws[m.name]=raw
  assert not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest()==manifest[m.name]
 for n,raw in raws.items():
  target=r/n
  if target.exists():continue
  target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as f:f.write(raw)
 with (r/(p['archive_name']+'.receipt.json')).open('x') as f:json.dump({'files':manifest,'archive_sha256':p['archive_sha256']},f)
print(json.dumps({'status':'deployed','file_count':len(manifest)}))
'''


def deploy(host, names, label, reserve=False):
    from paper.scripts.titantpp_inter62_transport_evidence import recorded_run
    archive = BUNDLE / (label + '.tar.gz')
    pack(names, archive)
    root = REMOTE[host]
    if reserve:
        transport(host, label + '_reserve',
                  "import json,sys;from pathlib import Path;p=json.load(sys.stdin);r=Path(p['root']);"
                  "assert str(r) in p['allowed_roots'];r.mkdir(exist_ok=False);(r/'incoming').mkdir()",
                  {'root': root, 'allowed_roots': list(REMOTE.values())})
    write(BUNDLE / 'control' / (label + '_upload_intent.json'), {'started_unix': time.time(), 'automatic_retry': False})
    result = recorded_run(BUNDLE, label + '_upload', ['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                          str(archive), host + ':' + root + '/incoming/' + archive.name], timeout=300)
    require(result.returncode == 0, 'Packet upload failed; no retry')
    transport(host, label + '_deploy', DEPLOY, {'root': root, 'allowed_roots': list(REMOTE.values()),
        'archive_name': archive.name, 'archive_sha256': sha(archive)}, timeout=300)


SUPERVISE = r'''
import importlib.util,json,os,signal,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin); r=Path(p['root']); s=p['split']
spec=importlib.util.spec_from_file_location('native_cap9_pipeline',r/'operation/pipeline.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
e=m.load_evaluator(r);c=e.verify_contract(r);a=e.verify_partial_metadata(r,c)
assert str(r)==a['remote_roots'][p['host']] and c['contract_sha256']==p['contract_sha256']
with (r/(s+'_supervision.claim')).open('x') as f:json.dump({'argv':p['argv'],'pid':os.getpid(),'started_unix':time.time()},f)
child=None;outcome=None;start=time.time()
def term(sig,frame):raise KeyboardInterrupt('evaluation supervisor interrupted')
signal.signal(signal.SIGTERM,term);signal.signal(signal.SIGHUP,term)
try:
 child=subprocess.Popen(p['argv'],cwd=r,start_new_session=True);code=child.wait(timeout=43200)
 outcome={'actual_child_exit_observed':True,'exit_code':code}
except BaseException:
 signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGHUP,signal.SIG_IGN)
 if child is not None:outcome=m.stop_owned_child(child)
 raise
finally:
 if outcome is not None:
  with (r/(s+'_pipeline_actual_exit.json')).open('x') as f:json.dump({'contract_sha256':c['contract_sha256'],'host':p['host'],'split':s,'started_unix':start,'completed_unix':time.time(),**outcome},f)
sys.exit(code)
'''


RECOVER = r'''
import hashlib,io,json,sys,tarfile
from pathlib import Path
p=json.load(sys.stdin);r=Path(p['root']);s=p['split']
assert hashlib.sha256((r/'contract.json').read_bytes()).hexdigest()==p['contract_raw_sha256']
c=json.loads((r/'contract.json').read_text());a=json.loads((r/(s+'_pipeline_actual_exit.json')).read_text())
assert a['actual_child_exit_observed'] and type(a['exit_code']) is int and a['exit_code']==0
names=[s+'_pipeline_actual_exit.json',s+'_pipeline_complete.json',s+'_supervision.claim',s+'_pipeline.claim']
for j in c['jobs']:
 if j['evaluation_host']!=p['host']:continue
 q='validation_replay' if s=='validation' else 'test_metrics'
 names += ['results/'+j['id']+'/'+q+'.json','results/'+j['id']+'/'+q+'_claim.json','results/'+j['id']+'/'+s+'_worker_exit.json',
           'worker_launches/'+s+'_'+j['id']+'.json','logs/'+s+'_'+j['id']+'.log']
raws={n:(r/n).read_bytes() for n in names};manifest={n:hashlib.sha256(v).hexdigest() for n,v in raws.items()}
raws['result_manifest.json']=json.dumps(manifest,sort_keys=True).encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for n,raw in raws.items():
  i=tarfile.TarInfo(n);i.size=len(raw);t.addfile(i,io.BytesIO(raw))
'''


def extract_results(archive, host, split, contract):
    expected = {split + suffix for suffix in ('_pipeline_actual_exit.json', '_pipeline_complete.json',
                                            '_supervision.claim', '_pipeline.claim')}
    for j in contract['jobs']:
        if j['evaluation_host'] != host:
            continue
        q = 'validation_replay' if split == 'validation' else 'test_metrics'
        expected.update({'results/' + j['id'] + '/' + q + '.json',
                         'results/' + j['id'] + '/' + q + '_claim.json',
                         'results/' + j['id'] + '/' + split + '_worker_exit.json',
                         'worker_launches/' + split + '_' + j['id'] + '.json',
                         'logs/' + split + '_' + j['id'] + '.log'})
    with tarfile.open(archive) as tar:
        manifest = json.load(tar.extractfile('result_manifest.json'))
        members = tar.getmembers()
        require(set(manifest) == expected and len(members) == len(manifest) + 1 and {m.name for m in members}
                == set(manifest) | {'result_manifest.json'}, 'Result archive membership differs')
        require(sum(m.size for m in members) <= 256 * 1024 ** 2, 'Result archive expansion limit exceeded')
        # Verify the entire packet before writing any recovered result.
        for m in members:
            require(m.isfile(), 'Unsafe archive member')
            safe(BUNDLE, m.name)
            if m.name != 'result_manifest.json':
                require(hashlib.sha256(tar.extractfile(m).read()).hexdigest() == manifest[m.name], 'Result archive SHA differs')
        for m in members:
            require(m.isfile(), 'Unsafe archive member')
            if m.name == 'result_manifest.json':
                continue
            raw = tar.extractfile(m).read()
            require(hashlib.sha256(raw).hexdigest() == manifest[m.name], 'Result archive SHA differs')
            prefix = '' if m.name.startswith(('results/', 'logs/', 'worker_launches/')) else 'native/' + host + '/'
            path = safe(BUNDLE, prefix + m.name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as f:
                f.write(raw)
    write(BUNDLE / 'native' / host / (split + '_result_manifest.json'), manifest)
    actual = read(BUNDLE / 'native' / host / (split + '_pipeline_actual_exit.json'))
    complete = read(BUNDLE / 'native' / host / (split + '_pipeline_complete.json'))
    jobs = [j for j in contract['jobs'] if j['evaluation_host'] == host]
    require(actual['actual_child_exit_observed'] is True and type(actual['exit_code']) is int
            and actual['exit_code'] == 0 and actual['contract_sha256'] == contract['contract_sha256']
            and complete['status'] == 'complete' and set(complete['completed']) == {j['id'] for j in jobs},
            'Native pipeline completion not verified')
    e = evaluator()
    for j in jobs:
        exit_record = read(BUNDLE / 'results' / j['id'] / (split + '_worker_exit.json'))
        require(exit_record['actual_child_exit_observed'] is True and type(exit_record['exit_code']) is int
                and exit_record['exit_code'] == 0 and exit_record['job_id'] == j['id']
                and exit_record['contract_sha256'] == contract['contract_sha256'], 'Worker actual exit not zero')
        name = 'validation_replay.json' if split == 'validation' else 'test_metrics.json'
        r = check_receipt(read(BUNDLE / 'results' / j['id'] / name), contract, j, split, e)
        if split == 'test':
            require(r['validation_gate_sha256'] == sha(BUNDLE / 'validation_gate.json'), 'Test gate SHA differs')


def evaluate_stage(c, training, split):
    # Host time is kept separate. These are bounded evaluations, not training concurrency changes.
    for h in HOSTS:
        begin = time.time()
        permit = {'schema': 'titantpp_gru_cap9_evaluation_permit_v1', 'authorized': True,
            'new_training': False, 'frozen_selection': True, 'stage': split,
            'contract_sha256': c['contract_sha256'], 'evaluator_sha256': sha(BUNDLE / 'operation/evaluate.py'),
            'host': h, 'job_ids': sorted(j['id'] for j in c['jobs'] if j['evaluation_host'] == h),
            'not_before_unix': begin, 'deadline_unix': begin + 43200, 'automatic_retry': False}
        name = 'permits/' + h + '_' + split + '.json'
        write(BUNDLE / name, permit)
        names = [name]
        if split == 'test':
            names += ['validation_gate.json']
            for j in c['jobs']:
                names += ['results/' + j['id'] + '/validation_replay.json',
                          'results/' + j['id'] + '/validation_worker_exit.json']
        deploy(h, names, h + '_' + split + '_gate')
        root = REMOTE[h]
        env = dict(training['hosts'][h]['environment'])
        env.update(PYTHONPATH=root + '/source', MPLCONFIGDIR=root + '/runtime_cache/matplotlib',
                   XDG_CACHE_HOME=root + '/runtime_cache', PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        argv = ['env', *[k + '=' + str(v) for k, v in sorted(env.items())], c['runtimes'][h]['python'],
                root + '/operation/pipeline.py', '--root', root, '--host', h, '--split', split]
        transport(h, h + '_' + split + '_execute', SUPERVISE,
                  {'root': root, 'host': h, 'split': split, 'argv': argv, 'contract_sha256': c['contract_sha256']},
                  timeout=43320)
        archive = BUNDLE / (h + '_' + split + '_results.tar.gz')
        transport(h, h + '_' + split + '_recover', RECOVER,
                  {'root': root, 'host': h, 'split': split, 'contract_raw_sha256': sha(BUNDLE / 'contract.json')}, archive=archive)
        extract_results(archive, h, split, c)


def make_gate(c):
    e = evaluator()
    receipts, exits = {}, {}
    for j in c['jobs']:
        stem = BUNDLE / 'results' / j['id']
        e.verify_validation_receipt(read(stem / 'validation_replay.json'), c, j)
        receipts[j['id']] = sha(stem / 'validation_replay.json')
        exits[j['id']] = sha(stem / 'validation_worker_exit.json')
    write(BUNDLE / 'validation_gate.json', {'schema': 'titantpp_gru_cap9_validation_gate_v1',
          'cohort': c['cohort'], 'status': 'PASS', 'contract_sha256': c['contract_sha256'],
          'job_count': 9, 'validation_receipts': receipts, 'validation_worker_exits': exits, 'created_unix': time.time()})
    return e.verify_global_validation_gate(BUNDLE, c)


def aggregate(rows):
    keys = [('whole_qty_rmse', None, 'qty_rmse'), ('whole_qty_mae', None, 'qty_mae'),
            ('tail_qty_rmse', 'tail', 'qty_rmse'), ('tail_qty_mae', 'tail', 'qty_mae'),
            ('whole_time_nll', None, 'time_nll'), ('tail_time_nll', 'tail', 'time_nll')]
    require(len(rows) == 27 and len({(r['dataset'], r['arm'], r['seed']) for r in rows}) == 27,
            'Expected nine CAP and eighteen original CNN/FULL receipts')
    by_key = {(r['dataset'], r['arm'], r['seed']): r for r in rows}
    grouped, deltas = [], []
    for d in DATASETS:
        for arm in (CNN, FULL, CAP):
            subset = [by_key[(d, arm, s)] for s in SEEDS]
            metrics = {}
            for name, group, key in keys:
                values = [(r['metrics'] if group is None else r['metrics'][group])[key] for r in subset]
                require(all(type(v) in (int, float) and math.isfinite(v) for v in values), 'Nonfinite metric')
                metrics[name] = {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values)}
            grouped.append({'dataset': d, 'arm': arm, 'seed_count': 3, 'metrics': metrics})
        for seed in SEEDS:
            cap = by_key[(d, CAP, seed)]
            for other in (CNN, FULL):
                ref = by_key[(d, other, seed)]
                for key in ('target_identity_sha256', 'truth_sha256', 'time_sha256'):
                    require(cap[key] == ref[key], 'Matched baseline target population differs')
                metrics = {}
                for name, group, key in keys:
                    left = (cap['metrics'] if group is None else cap['metrics'][group])[key]
                    right = (ref['metrics'] if group is None else ref['metrics'][group])[key]
                    metrics[name] = {'cap_minus_baseline': left - right,
                                     'relative_percent': 100 * (left - right) / right if right != 0 else None}
                deltas.append({'dataset': d, 'seed': seed, 'baseline': other, 'metrics': metrics})
    return {'status': 'complete', 'split': 'test', 'candidate_fit_count': 9, 'baseline_reference_count': 18,
            'seed_count': 3, 'grouped': grouped, 'seed_differences': deltas, 'seed_results': rows,
            'test_independent': False, 'interpretation': 'Exploratory follow-up after previously observed Test reversal; no causal or statistical significance claim.',
            'runtime_limitation': 'Intermittent42/52:5090 cuDNN92000; seed62:5080 cuDNN91900. Host times kept separate.',
            'new_training': False, 'checkpoint_reselection': False}


def summarize(c):
    e = evaluator()
    rows = []
    for j in c['jobs']:
        rows.append(check_receipt(read(BUNDLE / 'results' / j['id'] / 'test_metrics.json'), c, j, 'test', e))
    for ref in c['baseline_references']:
        require(sha(BUNDLE / ref['receipt_path']) == ref['receipt_sha256'], 'Reused Test receipt changed')
        rows.append(read(BUNDLE / ref['receipt_path']))
    result = aggregate(rows)
    write(REPORT / 'THREE_SEED_TEST.json', result)
    labels = {CNN: 'CNN', FULL: 'CNN+GRU', CAP: 'CAP(ρ=0.25)',
              'yellow_trip_hourly': 'Taxi', 'raf_spare_parts': 'RAF', 'intermittent_frozen_5000': 'Intermittent'}
    lines = ['# CAP9 고정 checkpoint Test 비교', '',
             'Validation 전체 raw 수량 RMSE 최초 strict finite 최소 epoch를 고정했다. 기존 CNN/FULL18은 재실행하지 않았다.', '',
             '| 데이터셋 | 구조 | RMSE 평균±표본SD | MAE 평균±표본SD |', '|---|---|---:|---:|']
    for row in result['grouped']:
        m = row['metrics']
        def fmt(k):
            return f"{m[k]['mean']:.6f}±{m[k]['sample_sd']:.6f}"
        lines.append(f"| {labels[row['dataset']]} | {labels[row['arm']]} | {fmt('whole_qty_rmse')} | {fmt('whole_qty_mae')} |")
    lines += ['', 'CAP−기존 모델: 음수는 개선, 양수는 악화다.', '',
              '| 데이터셋 | seed | 비교 | ΔRMSE | ΔMAE |', '|---|---:|---|---:|---:|']
    for row in result['seed_differences']:
        m = row['metrics']
        lines.append(f"| {labels[row['dataset']]} | {row['seed']} | CAP−{labels[row['baseline']]} | {m['whole_qty_rmse']['cap_minus_baseline']:+.6f} | {m['whole_qty_mae']['cap_minus_baseline']:+.6f} |")
    lines += ['', '전체·큰수량 수량 RMSE/MAE와 recorded positive-integer TimeNLL의 여섯 지표 및 개별 seed 원문은 JSON에 모두 보존했다.',
              '이번 후보는 기존 Test 순위 반전을 본 뒤 설계한 탐색적 후속 실험이다. 새 독립 평가나 원인 확정으로 해석하지 않는다.',
              'Intermittent42/52는 5090/cuDNN92000, seed62는 5080/cuDNN91900이다. 시간은 host별로 분리한다.', '']
    with (REPORT / 'THREE_SEED_TEST.md').open('x') as f:
        f.write('\n'.join(lines))
    files = {str(p): sha(p) for p in BUNDLE.rglob('*') if p.is_file()}
    files.update({str(REPORT / n): sha(REPORT / n) for n in ('THREE_SEED_TEST.json', 'THREE_SEED_TEST.md')})
    receipt = {'status': 'complete', 'candidate_test_count': 9, 'baseline_reference_count': 18,
        'seed_count': 3, 'validation_replay_count': 9, 'actual_native_worker_exit0_count': 18,
        'actual_pipeline_exit0_count': 4, 'pipeline_wrapper_own_exit_directly_observed': False,
        'files': files, 'files_sha256': canonical(files), 'contract_sha256': c['contract_sha256'],
        'completed_unix': time.time(), 'held_out_test_evaluated': True}
    write(BUNDLE / 'final_receipt.json', receipt)
    return receipt


def advance(expected):
    training = verify_bootstrap(expected)
    final_path = BUNDLE / 'final_receipt.json'
    if final_path.exists():
        r = read(final_path)
        require(r['status'] == 'complete' and r['candidate_test_count'] == 9 and r['seed_count'] == 3
                and r['baseline_reference_count'] == 18 and canonical(r['files']) == r['files_sha256'], 'Final receipt invalid')
        for n, value in r['files'].items():
            require(sha(n) == value, 'Final local evidence SHA differs')
        return {'status': 'complete', 'receipt_sha256': sha(final_path), 'remote_calls': 0}
    require(not (BUNDLE / 'advance.claim').exists() and not (BUNDLE / 'failure.json').exists(),
            'Existing attempt or failure: no automatic retry; remote0')
    receipt = collection_gate(training)
    if receipt is None:
        return {'status': 'waiting_for_verified_training9_and_original_collection', 'remote_calls': 0,
                'held_out_test_evaluated': False, 'checkpoint_reselection': False}
    write(BUNDLE / 'advance.claim', {'started_unix': time.time(), 'bootstrap_sha256': expected, 'automatic_retry': False})
    try:
        c = bind(training, receipt)
        names = list(read(BUNDLE / 'preparation_inventory.json')['files']) + list(c['bound_files']) + ['contract.json']
        names = [n for n in names if not n.startswith('references/') and n != 'draft_contract.json']
        for host in HOSTS:
            deploy(host, names, host + '_initial', reserve=True)
        evaluate_stage(c, training, 'validation')
        make_gate(c)
        evaluate_stage(c, training, 'test')
        result = summarize(c)
        return {'status': result['status'], 'candidate_test_count': 9, 'baseline_reference_count': 18,
                'receipt_sha256': sha(BUNDLE / 'final_receipt.json')}
    except BaseException as error:
        write(BUNDLE / 'failure.json', {'status': 'failed', 'error': str(error), 'traceback': traceback.format_exc(),
              'automatic_retry': False, 'completed_unix': time.time()})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('prepare', 'advance'), required=True)
    p.add_argument('--integrity-sha')
    args = p.parse_args()
    if args.mode == 'prepare':
        value = prepare()
    else:
        require(bool(args.integrity_sha), 'Integrity SHA required')
        value = advance(args.integrity_sha)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))
