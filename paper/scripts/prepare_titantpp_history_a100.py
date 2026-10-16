"""Prepare an isolated history27 A100 draft; stdlib only, no remote execution.

The generated draft cannot authorize GPU work. Actual pod/GPU/runtime identity,
monetary guard, permits, parent ownership reservation and seals are a separate step.
"""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

PROJECT = Path(__file__).resolve().parents[2]
PARENT = PROJECT / 'search_artifacts/titantpp_contribution_history_20261008_v1'
SCHEMA = 'titantpp_contribution_history_a100_20261008_v1'
TARGET = PROJECT / 'search_artifacts' / SCHEMA
REMOTE_ROOT = '/workspace/' + SCHEMA + '_a100'
PARENT_SHA = '7594a484d29dbe5efb633d2e78d08734d5c24661761639874ed64e77b2c11f92'
SOURCE_SHA = 'c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a'
PARENT_OPERATION_SHA = 'e47bea24d796c5eeeff4f4fcc3c702308b461ac4c446cca1aab354d951d4af09'
DATA_ORDER = ('yellow_trip_hourly', 'raf_spare_parts', 'intermittent_frozen_5000')
SCIENCE_FIELDS = ('stage', 'arms', 'seeds', 'runner_modules', 'datasets', 'dataset_sha256',
 'source', 'input_files', 'training', 'architecture', 'design_sha256', 'diagnostic',
 'measurement_arms', 'cost_limits', 'evaluation_scope', 'held_out_test_evaluated',
 'claim_contract_v2_rawSHA', 'policy')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe(root, name):
    p = Path(name)
    require(not p.is_absolute() and '..' not in p.parts, 'Unsafe relative path')
    result = (Path(root) / p).resolve()
    require(result.is_relative_to(Path(root).resolve()), 'Path escape')
    return result


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    if path.exists():
        require(path.read_bytes() == payload, 'Existing artifact differs: ' + str(path))
    else:
        with path.open('xb') as stream:
            stream.write(payload)


def verify_parent(parent=PARENT):
    c = read(parent / 'execution_contract.json')
    require(sha_json(c) == PARENT_SHA, 'Parent canonical drift')
    pointer = read(parent / 'current.json')
    require(pointer['canonical_sha256'] == PARENT_SHA, 'Parent pointer drift')
    require(c['source']['files_sha256'] == SOURCE_SHA and sha_json(c['source']['files']) == SOURCE_SHA,
        'Parent scientific source closure drift')
    require(len(c['source']['files']) == 129 and len(c['input_files']) == 7, 'Parent file count drift')
    for name, digest in {**c['source']['files'], **c['input_files']}.items():
        require(sha_file(safe(parent / 'source', name)) == digest, 'Parent source/input drift: ' + name)
    require(sha_json(c['operation']['files']) == c['operation']['files_sha256'], 'Parent operation closure drift')
    for name, digest in c['operation']['files'].items():
        require(sha_file(safe(parent, name)) == digest, 'Parent operation drift: ' + name)
    require(sha_file(parent / 'control/integrity_receipt.json') == pointer['control_integrity_raw_sha256'],
        'Parent control integrity drift')
    for name, digest in read(parent / 'control/integrity_receipt.json')['project_files'].items():
        require(sha_file(safe(PROJECT, name)) == digest, 'Parent project helper drift: ' + name)
    for host in ('5080', '5090'):
        for rel in (f'start_permit_{host}.json', f'training_permit_{host}.json',
                    f'deployment/launch_receipt_{host}.json', f'deployment/launch_intent_{host}.json'):
            require(not (parent / rel).exists(), 'Parent already owns execution: ' + rel)
    return c


def scientific_fingerprint(c):
    return sha_json({key: c[key] for key in SCIENCE_FIELDS})


def runtime_text(parent):
    original = (PROJECT / 'paper/scripts/titantpp_contribution_runtime.py').read_text()
    require(hashlib.sha256(original.encode()).hexdigest() == PARENT_OPERATION_SHA, 'Base operation drift')
    result = original.replace("require(c['stage'] in STAGES and c['schema']=='titantpp_contribution_'+c['stage']+'_20261008_v1','Foreign stage')",
        "require(c['stage']=='history' and c['schema']=='" + SCHEMA + "','Foreign stage')\n    validate_a100_scope(c)")
    result = result.replace("require(c['limits']['total_wall_seconds']==168*3600 and c['limits']['per_condition_seconds']==36*3600",
        "require(0<c['limits']['total_wall_seconds']<=168*3600 and 0<c['limits']['per_condition_seconds']<=36*3600")
    result = result.replace("choices=('5080','5090')", "choices=('a100',)")
    result = result.replace("require(s['deadline_unix']-s['started_at_unix']==c['limits']['total_wall_seconds'],'Extended lease')",
        "require(s['deadline_unix']-s['started_at_unix']==c['limits']['total_wall_seconds'],'Extended lease')\n    require(s['deadline_unix']<=c['budget']['hard_deadline_unix'],'Start permit exceeds monetary deadline')")
    scope = '''
def validate_a100_scope(c):
    require(set(c['hosts'])=={'a100'} and set(c['assignments'])=={'a100'},'Foreign A100 host scope')
    expected_groups=[[d,s] for d in DATA_ORDER for s in SEEDS]
    require(c['assignments']['a100']==expected_groups,'A100 queue order drift')
    require(sha_json({k:c[k] for k in SCIENCE_FIELDS})==SCIENCE_FINGERPRINT,'Scientific identity drift')
    spec=c['hosts']['a100']
    require(spec['root']==OWNED_ROOT,'Foreign A100 owned root')
    require(spec['gpu_name_contains']=='A100-SXM4-80GB','Foreign GPU type')
    require(isinstance(spec['gpu_uuid'],str) and spec['gpu_uuid'].startswith('GPU-')
        and len(spec['gpu_uuid'])==40 and spec['gpu_uuid'] not in (
        'GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6','GPU-c9adc246-ef66-7906-bc07-6e152fa9952f'),'Unbound/foreign GPU UUID')
    require(Path(spec['python']).is_absolute() and spec['runtime_expected']
        and all(v is not None for v in spec['runtime_expected'].values()),'Unbound native runtime')
    budget=c['budget']
    require(budget['maximum_total_usd']==50 and budget['independent_controller_required'] is True,'Budget authority drift')
    keys=('confirmed_gpu_hourly_usd','reserved_non_compute_usd','billing_started_at_unix','hard_deadline_unix')
    require(all(type(budget.get(k)) in (int,float) and math.isfinite(budget[k]) for k in keys),'Unbound monetary budget')
    duration=budget['hard_deadline_unix']-budget['billing_started_at_unix']
    require(budget['confirmed_gpu_hourly_usd']>0 and 0<=budget['reserved_non_compute_usd']<50
        and 0<duration<=168*3600,'Invalid monetary budget')
    require(duration/3600*budget['confirmed_gpu_hourly_usd']+budget['reserved_non_compute_usd']<=50,'Monetary cap exceeded')
    require(c['limits']['total_wall_seconds']<=duration,'Training lease exceeds billing window')
    require(c.get('deployment_ready') is True,'Draft cannot execute')
'''
    constants = '\nSCIENCE_FIELDS = ' + repr(SCIENCE_FIELDS) + '\nSCIENCE_FINGERPRINT = ' + repr(scientific_fingerprint(parent)) + '\nDATA_ORDER = ' + repr(DATA_ORDER) + '\nOWNED_ROOT = ' + repr(REMOTE_ROOT) + '\n'
    result = result.replace("ROOT = None", constants + '\nROOT = None')
    result = result.replace('\ndef bootstrap(c,host):', scope + '\ndef bootstrap(c,host):')
    # Every scientific/training function must remain text-identical. Only the
    # contract/lease/host wrapper is allowed to differ.
    old_tree, new_tree = ast.parse(original), ast.parse(result)
    changed = {'validate', 'authorization', 'main'}
    old_nodes = {n.name: ast.get_source_segment(original, n) for n in old_tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    new_nodes = {n.name: ast.get_source_segment(result, n) for n in new_tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for name in old_nodes.keys() - changed:
        require(old_nodes[name] == new_nodes[name], 'Unapproved function change: ' + name)
    return result


def make_draft(parent, runtime_sha):
    c = deepcopy(parent)
    c['schema'] = SCHEMA
    c['assignments'] = {'a100': [[d,s] for d in DATA_ORDER for s in c['seeds']]}
    c['jobs'] = [{**j, 'host':'a100'} for j in parent['jobs']]
    h = deepcopy(parent['hosts']['5080'])
    for key, suffix in [('root',''),('source_root','/source'),('operation_root','/operation'),('output_dir','/run')]:
        h[key] = REMOTE_ROOT + suffix
    h.update(python='/usr/local/bin/python', gpu_uuid=None, gpu_name_contains='A100-SXM4-80GB',
        runtime_expected={'torch':None,'cuda':None,'cudnn':None,'numpy':None,'polars':None},
        tmux_binary='/usr/bin/tmux', tmux=SCHEMA+'_a100', assigned_datasets=list(DATA_ORDER),old_owner_roots=[])
    h['environment'].pop('LD_LIBRARY_PATH',None)
    h['environment'].update(PYTHONPATH=REMOTE_ROOT+'/source',MPLCONFIGDIR=REMOTE_ROOT+'/cache/matplotlib',XDG_CACHE_HOME=REMOTE_ROOT+'/cache')
    c['hosts'] = {'a100':h}
    c['operation'] = {'files':{'operation/campaign.py':runtime_sha}}
    c['operation']['files_sha256'] = sha_json(c['operation']['files'])
    c['limits']['total_wall_seconds'] = 0
    c['limits']['per_condition_seconds'] = 36*3600
    c['budget'] = {'maximum_total_usd':50,'confirmed_gpu_hourly_usd':None,
        'reserved_non_compute_usd':None,'billing_started_at_unix':None,'hard_deadline_unix':None,
        'independent_controller_required':True}
    c['deployment_ready'] = False
    c['parent_ownership'] = {'contract_sha256':PARENT_SHA,'bundle':str(PARENT.relative_to(PROJECT)),
        'reservation_required_before_launch':True,'original_execution_ids_preserved':True}
    c['prior_stage'] = None
    c['operational_change'] = {'reason':'User approved standalone A100 SXM history27 with USD 50 total cap',
        'Instacart_dependency_waived':True,'retrieval_27_not_in_this_authorization':True,
        'science_unchanged':True,'runtime_hardware_changed':True}
    require(scientific_fingerprint(c) == scientific_fingerprint(parent),'Scientific contract changed')
    def identities(x): return [{k:v for k,v in j.items() if k!='host'} for j in x['jobs']]
    require(identities(c)==identities(parent),'Canonical 27 identities changed')
    return c


def prepare():
    parent = verify_parent()
    for name, digest in {**parent['source']['files'], **parent['input_files']}.items():
        src, dst = safe(PARENT/'source',name),safe(TARGET/'source',name)
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists(): require(sha_file(dst)==digest,'Existing copied file drift: '+name)
        else: shutil.copyfile(src,dst)
        require(sha_file(dst)==digest,'Copy failed: '+name)
    runtime = runtime_text(parent)
    op = TARGET/'operation/campaign.py';op.parent.mkdir(parents=True,exist_ok=True)
    if op.exists(): require(op.read_text()==runtime,'Existing A100 operation differs')
    else: op.write_text(runtime)
    c = make_draft(parent,sha_file(op))
    write_new(TARGET/'execution_contract.draft.json',c)
    receipt = {'status':'local_preparation_passed_not_execution_authorized',
        'parent_canonical_sha256':PARENT_SHA,'draft_canonical_sha256':sha_json(c),
        'source_files':129,'input_files':7,'source_closure_sha256':SOURCE_SHA,
        'input_closure_sha256':sha_json(c['input_files']),
        'operation_closure_sha256':c['operation']['files_sha256'],
        'scientific_fingerprint':scientific_fingerprint(c),'canonical_execution_ids':27,
        'all_scientific_runtime_functions_identical':True,'parent_launch_and_permits_absent':True,
        'cpu_data_admission_executed':False,'native_qualification_executed':False,
        'remote_calls':0,'GPU_training_calls':0,'prepared_helper_raw_sha256':sha_file(__file__),
        'outstanding':['actual pod/GPU UUID and Python/runtime versions','independent USD50 guard and concrete deadline',
         'parent exclusive ownership reservation','final contract/current/approval/control seals',
         'native qualification and start/training permits']}
    write_new(TARGET/'preparation_receipt.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args();prepare()
