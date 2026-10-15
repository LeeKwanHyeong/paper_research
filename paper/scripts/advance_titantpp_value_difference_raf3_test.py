"""RAF V adjacent-difference CNN3 frozen Validation gate and Test; reuse historical CNN3.
No training, checkpoint reselection or automatic retry. Exclusive isolated outputs.
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
TRAIN = PROJECT / 'search_artifacts/titantpp_backbone_value_difference_raf3_20261016_v1'
BUNDLE = TRAIN / 'evaluation_v1'
REPORT = PROJECT / 'reports/titantpp_backbone_value_difference_raf3_20261016_v1/evaluation_v1'
TRAIN_SHA = TRAIN_CANONICAL = TRAIN_RECEIPT_SHA = None
TRAIN_INTEGRITY_SHA = None
SELECTION_RULE = 'single_candidate:all_four_user_checks'
BASELINE = PROJECT / 'search_artifacts/titantpp_components_taxi_raf24_test_20261014_v1'
VALUE = 'titantpp_cnn_value_difference_v1'
CNN = 'titantpp_cnn'
LEGACY_CNN = 'titantpp_cnn_gru54_cnn_only'
REGISTRY = PROJECT / 'reports/titantpp_backbone_value_difference_raf3_20261016_v1/EXTERNAL_BASELINES.json'
DATASETS = ('raf_spare_parts',)
SEEDS = (42, 52, 62)
HOSTS = ('5090',)
USER_INSTRUCTION = 'ㅇㅇ 진행하자. 승인할것들에 대해서는 미리 승인한다. 지금부터 진행시켜!'
APPROVAL_CONTEXT = 'RAF V 관측 차분 CNN ×3seed, 고정 checkpoint Validation3 모두 재현 후 Test3. 기존 CNN3·외부 비교군은 원결과만 재사용한다. 기존 CNN 재학습·재평가0, Test checkpoint 재선택0.'
REMOTE = {h: '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_backbone_value_difference_raf3_20261016_v1_test_' + h for h in HOSTS}


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
    spec = importlib.util.spec_from_file_location('raf3_eval_operational', BUNDLE / 'operation/evaluate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def training_contract():
    global TRAIN_SHA,TRAIN_CANONICAL
    require(TRAIN_INTEGRITY_SHA is not None,'Training integrity pin is required before any work')
    require(sha(TRAIN/'followup_integrity.json')==TRAIN_INTEGRITY_SHA,'Training integrity differs; remote0')
    integrity=read(TRAIN/'followup_integrity.json')
    for name,digest in integrity['files'].items():require(sha(PROJECT/name)==digest,'Training local SHA changed; remote0: '+name)
    TRAIN_SHA=integrity['contract_raw_sha256']
    require(sha(TRAIN / 'execution_contract.json') == TRAIN_SHA, 'Original training contract changed; remote0')
    c = read(TRAIN / 'execution_contract.json')
    TRAIN_CANONICAL=canonical(c)
    expected = {f'{d}__{s}__{a}' for d in DATASETS for s in SEEDS for a in (VALUE,)}
    require(len(c['jobs']) == 3 and {j['id'] for j in c['jobs']} == expected, 'Foreign training cohort')
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








def prepare(training_integrity_sha, registry_sha):
    global TRAIN_INTEGRITY_SHA
    TRAIN_INTEGRITY_SHA=training_integrity_sha
    c = training_contract()
    receipt=collection_gate(c)
    if receipt is None:
        return {'status':'waiting_for_verified_training3_and_original_collection','remote_calls':0,'Test':False}
    from paper.scripts.compare_titantpp_value_difference_loop import verify_registry
    require(registry_sha == '57e88f4d5d338c6da7f54fe38d2b9492b06476fa93161331481de3c4ee1a6b49', 'Unapproved external registry; remote0')
    registry=verify_registry(REGISTRY, registry_sha, root=PROJECT)
    require(not BUNDLE.exists(), 'Evaluation bundle already exists; no overwrite')
    BUNDLE.mkdir(); REPORT.mkdir(parents=True,exist_ok=True)
    files, dependencies = {}, {}
    def add(src, name, expected=None):
        expected = expected or sha(src)
        copy_checked(src, safe(BUNDLE, name), expected); files[name] = expected
    add(TRAIN / 'execution_contract.json', 'provenance/training_contract.json', TRAIN_SHA)
    for src, name in [('evaluate_titantpp_value_difference_raf3_frozen.py', 'evaluate.py'),
                      ('run_titantpp_gru_cap9_test_pipeline.py', 'pipeline.py')]:
        add(PROJECT / 'paper/scripts' / src, 'operation/' + name)
    lineage={'source_closure':c['source']['files_sha256'],'source_revision':c['source']['base_git_revision'],
        'training_canonical':TRAIN_CANONICAL,'training_raw_sha256':TRAIN_SHA,'training_integrity_sha256':training_integrity_sha}
    write(BUNDLE/'operation/lineage.json',lineage);files['operation/lineage.json']=sha(BUNDLE/'operation/lineage.json')
    for name, expected in {**c['source']['files'], **c['input_files']}.items():
        add(TRAIN / 'source' / name, 'source/' + name, expected)
    approval = {'approved': True, 'user_instruction': USER_INSTRUCTION, 'scope_context': APPROVAL_CONTEXT,
        'user_instruction_sha256': hashlib.sha256(USER_INSTRUCTION.encode()).hexdigest(),
        'expected_condition_count': 3, 'job_ids': sorted(j['id'] for j in c['jobs']),
        'training_contract_sha256': TRAIN_CANONICAL, 'new_training': False,
        'checkpoint_reselection': False, 'local_root': str(BUNDLE), 'remote_roots': REMOTE,
        'test_independent': False, 'automatic_retry': False, 'approved_date_kst': '2026-10-15'}
    write(BUNDLE / 'approval.json', approval); files['approval.json'] = sha(BUNDLE / 'approval.json')
    old = read(BASELINE / 'contract.json'); manifest = read(BASELINE / 'evaluation_results_manifest.json')
    require(old['contract_sha256'] == canonical({k:v for k,v in old.items() if k!='contract_sha256'})
            and manifest['contract_sha256'] == old['contract_sha256'], 'Baseline contract/manifest differs')
    for name in ('contract.json', 'evaluation_results_manifest.json'):
        dependencies[str(BASELINE / name)] = sha(BASELINE / name)
    spec = copy.deepcopy(old['datasets']['raf_spare_parts'])
    require(spec['definition'] == c['datasets'][0], 'RAF preprocessing differs from original CNN')
    require(registry['data'] == {k:spec[k] for k in ('data_sha256','training_data_sha256','split_manifest_sha256')}, 'Registry input SHA differs')
    require(registry['expected_population'] == {split:spec[split+'_population'] for split in ('validation','test')}, 'Registry target population differs')
    for pk, hk in [('data_path','data_sha256'), ('training_data_path','training_data_sha256'),
                   ('split_manifest_path','split_manifest_sha256')]:
        add(BASELINE / spec[pk], spec[pk], spec[hk])
    references=[]
    for ref in c['baseline_jobs']:
        j = next(j for j in old['jobs'] if j['id']==ref['id'])
        require(j['dataset']=='raf_spare_parts' and j['arm']==LEGACY_CNN and j['seed'] in SEEDS
                and j['checkpoint']['sha256']==ref['original']['checkpoint']['sha256']
                and j['selected_epoch']==ref['original']['selected_epoch'], 'Wrong historical CNN baseline')
        for suffix in ('test_metrics.json', 'test_worker_exit.json'):
            name='results/'+j['id']+'/'+suffix; expected=manifest['files'][name]
            add(BASELINE/name, 'references/'+j['id']+'/'+suffix, expected)
            dependencies[str(BASELINE/name)] = expected
        actual=read(BASELINE/'results'/j['id']/'test_worker_exit.json')
        require(actual['actual_child_exit_observed'] is True and type(actual['exit_code']) is int
                and actual['exit_code']==0 and actual['job_id']==j['id']
                and actual['contract_sha256']==old['contract_sha256'], 'CNN Test exit not verified')
        historical=check_receipt(read(BASELINE/'results'/j['id']/'test_metrics.json'),old,j,'test',evaluator())
        fixed=[r for r in registry['rows'] if r['model']=='historical_CNN'
               and r['split']=='test' and r['seed']==j['seed']]
        require(len(fixed)==1 and fixed[0]['checkpoint_sha256']==j['checkpoint']['sha256']
                and fixed[0]['source_sha256']==manifest['files']['results/'+j['id']+'/test_metrics.json'],
                'Historical CNN differs from frozen external registry')
        evaluator().compare_validation_metrics(ref['original']['validation_reference'],j['validation_reference'])
        for split in ('validation','test'):
            fixed_pair=[r for r in registry['rows'] if r['model']=='historical_CNN' and r['split']==split and r['seed']==j['seed']]
            require(len(fixed_pair)==1 and fixed_pair[0]['selected_epoch']==j['selected_epoch']
                and fixed_pair[0]['checkpoint_sha256']==j['checkpoint']['sha256'], 'CNN selection differs from registry')
            metric_reference=ref['original']['validation_reference'] if split=='validation' else historical['metrics']
            require(all(math.isclose(fixed_pair[0]['metrics'][key],metric_reference[key],rel_tol=1e-5,abs_tol=1e-5) for key in ('qty_rmse','qty_mae')), 'CNN endpoint differs from registry')
        path='references/'+j['id']+'/test_metrics.json'
        references.append({'id':j['id'],'dataset':j['dataset'],'seed':j['seed'],'arm':j['arm'],
            'comparison_arm':CNN,'role':'historical_immutable_reference','retrained':False,'reevaluated':False,
            'selected_epoch':j['selected_epoch'],'checkpoint_sha256':j['checkpoint']['sha256'],
            'contract_sha256':old['contract_sha256'],'receipt_path':path,'receipt_sha256':files[path],
            'validation_reference':validation_reference(ref['original']['validation_reference'])})
    require({r['seed'] for r in references}==set(SEEDS) and len(references)==3, 'CNN3 references incomplete')
    add(REGISTRY, 'references/EXTERNAL_BASELINES.json', registry_sha)
    dependencies[str(REGISTRY)]=registry_sha
    for script in ('compare_titantpp_backbone_loop.py','compare_titantpp_value_difference_loop.py'):
        dependencies[str(PROJECT/'paper/scripts'/script)]=sha(PROJECT/'paper/scripts'/script)
    amendment={'schema':'raf3_historical_comparison_scope_v1','registry_sha256':registry_sha,
        'historical_CNN':'immutable reference only; no new training or reevaluation',
        'same_host_CNN_control_performed':False,'same_host_CNN_control_reason':'excluded from approved three-candidate scope',
        'intervention':'V-only adjacent observed differences; local CNN weights576->512; includes degrees of freedom, short-history boundary and optimizer parameterization',
        'candidate_selection_rule':SELECTION_RULE,
        'user_performance_gate':'strictly lower 3seed means than historical CNN and per-metric external frontier for RMSE and MAE on both Validation and Test',
        'runtime_confound':'historical CNN/external 5080/cuDNN91900; candidates 5090/cuDNN92000',
        'test_independent':False}
    write(BUNDLE/'comparison_scope_amendment.json',amendment)
    files['comparison_scope_amendment.json']=sha(BUNDLE/'comparison_scope_amendment.json')
    h=c['hosts']['5090']; q=read(TRAIN/'retrieval/originals/qualification/receipt.json')['runtime']
    runtimes={'5090':{'host':'5090','python':h['python'],'resolved_python_executable':'/opt/miniconda3/envs/ai_env/bin/python3.12',
        'python_version':q['python'],**{k:q[k] for k in ('torch','cuda','cudnn','numpy','polars')},
        'gpu_uuid':h['gpu_uuid'],'torch_num_threads':4,'interop_threads':1,'dtype':'float32',
        'deterministic_algorithms':True,'TF32':False,
        'native':{k:q[k] for k in ('gpu','cudnn_benchmark','cudnn_deterministic','float32_matmul_precision','reduced_precision_reduction','runtime_helper_sha256')}}}
    ops={k:v for k,v in files.items() if k.startswith('operation/')}
    e=evaluator()
    jobs=[{**j,'cohort':e.COHORT,'study':'components','training_host':j['host'],'evaluation_host':j['host'],
           'frozen_source_closure':c['source']['files_sha256']} for j in c['jobs']]
    draft={'schema':e.KIND,'cohort':e.COHORT,'evaluation_scope':'test_value_difference_raf3_followup',
        'metrics_schema':'quantity_time_tail_v1','expected_seeds':list(SEEDS),'expected_jobs':3,'binding_complete':False,
        'sources':{'components':{'root':'source','files':c['source']['files'],'closure':c['source']['files_sha256'],
                               'revision':c['source']['base_git_revision']}},
        'jobs':jobs,'datasets':{'raf_spare_parts':spec},'runtimes':runtimes,
        'evaluator_integrity':{'files':ops,'closure':canonical(ops)},
        'training_contract_path':'provenance/training_contract.json','training_contract_sha256':TRAIN_SHA,
        'partial_approval_path':'approval.json','partial_approval_sha256':files['approval.json'],
        'baseline_references':references,
        'candidate_selection_rule':SELECTION_RULE,'external_registry':{'path':'references/EXTERNAL_BASELINES.json','sha256':registry_sha},
        'comparison_scope_amendment_sha256':files['comparison_scope_amendment.json'],
        'test_is_independent':False,'new_training':False,
        'checkpoint_reselection':False,'automatic_retry':False,'bound_files':{}}
    draft['contract_sha256']=canonical(draft);write(BUNDLE/'draft_contract.json',draft)
    files['draft_contract.json']=sha(BUNDLE/'draft_contract.json')
    write(BUNDLE/'preparation_inventory.json',{'files':files,'external_dependencies':dependencies})
    e.verify_contract(BUNDLE,require_bound=False)
    synthetic=BUNDLE/'control/synthetic';synthetic.mkdir(parents=True)
    command=[sys.executable,str(BUNDLE/'operation/evaluate.py'),'--root',str(BUNDLE),'--mode','qualify-synthetic']
    outcome=subprocess.run(command,cwd=PROJECT,capture_output=True,timeout=180,
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''})
    for name,raw in [('stdout',outcome.stdout),('stderr',outcome.stderr)]:
        with (synthetic/name).open('xb') as stream:stream.write(raw)
    write(synthetic/'actual_exit.json',{'actual_child_exit_observed':True,'actual_exit':outcome.returncode,
        'argv':command,'stdout_sha256':sha(synthetic/'stdout'),'stderr_sha256':sha(synthetic/'stderr')})
    require(outcome.returncode==0,'Local frozen evaluation synthetic checks failed; no Test or retry')
    result=json.loads(outcome.stdout)
    require(result['status']=='PASS' and result['six_metrics_hand_calculation_checked']
        and result['checkpoint_reads']==0 and result['real_data_loaded'] is False,'Synthetic evidence invalid')
    write(synthetic/'result.json',result)
    paths=[p for p in BUNDLE.rglob('*') if p.is_file()]
    paths += [Path(__file__),PROJECT/'paper/scripts/evaluate_titantpp_value_difference_raf3_frozen.py',
        PROJECT/'paper/scripts/compare_titantpp_value_difference_loop.py',
        PROJECT/'paper/scripts/compare_titantpp_backbone_loop.py',
        PROJECT/'paper/scripts/run_titantpp_gru_cap9_test_pipeline.py',
        PROJECT/'paper/scripts/titantpp_inter62_transport_evidence.py',TRAIN/'followup_integrity.json',
        TRAIN/'retrieval/final_receipt.json']
    paths += list((PROJECT/'paper/tests').glob('test_titantpp_value_difference_raf3_test.py'))
    write(BUNDLE/'bootstrap_integrity.json',{'files':{str(p):sha(p) for p in paths},
        'training_integrity_sha256':training_integrity_sha,'training_receipt_raw_sha256':TRAIN_RECEIPT_SHA,
        'held_out_test_evaluated':False,'synthetic_passed':True})
    return {'status':'prepared_unbound','candidate_count':3,'baseline_count':3,'remote_calls':0,
        'bootstrap_integrity_sha256':sha(BUNDLE/'bootstrap_integrity.json')}

def verify_bootstrap(expected):
    global TRAIN_INTEGRITY_SHA,TRAIN_RECEIPT_SHA
    require(sha(BUNDLE / 'bootstrap_integrity.json') == expected, 'Bootstrap integrity differs; remote0')
    seal = read(BUNDLE / 'bootstrap_integrity.json')
    for name, value in seal['files'].items():
        require(sha(name) == value, 'Pinned dependency changed; remote0: ' + name)
    TRAIN_INTEGRITY_SHA=seal['training_integrity_sha256']
    TRAIN_RECEIPT_SHA=seal['training_receipt_raw_sha256']
    c = training_contract()
    inventory = read(BUNDLE / 'preparation_inventory.json')
    for name, value in inventory['files'].items():
        require(sha(safe(BUNDLE, name)) == value, 'Prepared artifact changed; remote0')
    for name, value in inventory['external_dependencies'].items():
        require(sha(name) == value, 'Existing baseline proof changed; remote0')
    from paper.scripts.compare_titantpp_value_difference_loop import verify_registry
    draft=read(BUNDLE/'draft_contract.json')
    registry=draft['external_registry']
    verify_registry(safe(BUNDLE,registry['path']),registry['sha256'],root=PROJECT)
    collection_gate(c)
    return c


def verify_terminal_snapshot(c,snapshot):
    require(snapshot.get('contract_sha256')==canonical(c)
            and snapshot.get('all_training_terminal_verified') is True
            and snapshot.get('failure') is False, 'Training terminal contract/status differs')
    keys={'pipeline','start','qualification','training_permit','dispatcher'}
    proof=snapshot.get('proof_bindings',{})
    require(set(proof)==keys and all(v is True for v in proof.values()), 'Training identity proof binding missing')
    rows=snapshot.get('jobs',[])
    expected={j['id']:j for j in c['jobs']}
    require(len(rows)==len(expected)==3 and {r['job']['id'] for r in rows}==set(expected)
            and all(r['job']==expected[r['job']['id']] and r['state']=='complete'
                    and r['selected_binary_present'] is True and r['terminal_binary_verified'] is True for r in rows),
            'Training three scientific completions/binary seals missing')
    probes=snapshot.get('owner_probes',{})
    require(bool(probes) and all(p.get('state')=='absent' for p in probes.values()),
            'Training explicit owner PID absence missing')
    gpu=snapshot.get('gpu_compute_pids')
    require(isinstance(gpu,list) and all(type(p) is int and p>0 for p in gpu)
            and not {int(p) for p in probes}.intersection(gpu), 'Owned training GPU/PID presence remains')
    return snapshot


def collection_gate(c):
    global TRAIN_RECEIPT_SHA
    path=TRAIN/'retrieval/final_receipt.json'
    require(not (TRAIN/'ACTION_REQUIRED.json').exists()
            and not (REPORT.parent/'ACTION_REQUIRED.json').exists(), 'Training requires user action; remote0')
    if not path.exists():
        require(not (TRAIN/'retrieval').exists(), 'Incomplete training retrieval; no automatic retry; remote0')
        return None
    if TRAIN_RECEIPT_SHA is not None:require(sha(path)==TRAIN_RECEIPT_SHA, 'RAF training receipt changed; remote0')
    r=read(path)
    require(r['status']=='complete' and r['new_fit_count']==3 and r['seed_count']==3
            and r['baseline_reference_count']==3 and r['held_out_test_evaluated'] is False
            and r['all_training_terminal_verified'] is True, 'Training completion invalid')
    for name,digest in r['files'].items():
        require(sha(safe(PROJECT,name))==digest, 'Training original/archive/summary changed: '+name)
    pointer=read(TRAIN/'retrieval/intent.json')
    require(sha(pointer['round'])==pointer['round_sha256'] and sha(pointer['summary'])==pointer['summary_sha256'],
            'Terminal observation bytes changed')
    snapshot=read(pointer['round'])
    verify_terminal_snapshot(c,snapshot)
    local=TRAIN/'retrieval/originals'
    from paper.scripts.advance_titantpp_value_difference_raf3 import parse_control
    for name in ('dispatcher_exit.json','pipeline_exit.json'):
        exit_record=parse_control((local/name).read_bytes())
        require(exit_record['actual_child_exit_observed'] is True and type(exit_record['returncode']) is int
                and exit_record['returncode']==0 and exit_record['contract_sha256']==TRAIN_CANONICAL,
                'Training pipeline/dispatcher exit0 not verified')
    require(read(local/'train_complete.json')['scientific_worker_count']==3, 'Training count differs')
    for j in c['jobs']:
        ex=read(local/'worker_exits'/ (j['id']+'.json'))
        require(ex['actual_child_exit_observed'] is True and type(ex['returncode']) is int and ex['returncode']==0
                and ex['contract_sha256']==TRAIN_CANONICAL and ex['job']==j,
                'Scientific worker exit0 missing')
        folder=local/'run'/j['id'];m=read(folder/'terminal_manifest.json')
        require(m['contract_sha256']==TRAIN_CANONICAL and m['job']==j and m['scientific_success'] is True,
                'Terminal manifest identity differs')
        for name,digest in m['files'].items():
            require(sha(safe(folder,name))==digest, 'Terminal binary/original changed')
    TRAIN_RECEIPT_SHA=sha(path)
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
    import torch
    e=evaluator(); contract=read(BUNDLE/'draft_contract.json');contract.pop('contract_sha256')
    proof='provenance/training_final_receipt.json'
    copy_checked(TRAIN/'retrieval/final_receipt.json',BUNDLE/proof,TRAIN_RECEIPT_SHA)
    bound={proof:TRAIN_RECEIPT_SHA}
    for j in contract['jobs']:
        folder=TRAIN/'retrieval/originals/run'/j['id'];m=read(folder/'terminal_manifest.json')
        prefix='runs/'+j['arm']+'/count_only_log_regression/seed_'+str(j['seed'])+'/'
        checkpoint_name='originals/'+j['id']+'/'+prefix+'best_val_qty_rmse_model.pt'
        history_name='originals/'+j['id']+'/'+prefix+'history.json'
        for relative in (prefix+'best_val_qty_rmse_model.pt',prefix+'history.json',prefix+'endpoint_replays.json','input_receipt.json','terminal_manifest.json'):
            name='originals/'+j['id']+'/'+relative
            digest=sha(folder/relative) if relative=='terminal_manifest.json' else m['files'][relative]
            copy_checked(folder/relative,BUNDLE/name,digest);bound[name]=digest
        payload=torch.load(BUNDLE/checkpoint_name,map_location='cpu',weights_only=False)
        metadata={k:v for k,v in payload.items() if k!='model_state_dict'}
        metadata_name='metadata/'+j['id']+'.json';write(BUNDLE/metadata_name,metadata);bound[metadata_name]=sha(BUNDLE/metadata_name)
        epoch,value=e.strict_selected_epoch(e.read_selection_history(BUNDLE/history_name))
        endpoint=read(folder/(prefix+'endpoint_replays.json'))['selected']
        require(epoch==payload['best_epoch'] and value==payload['selected_metric_value']
                and payload['model_state_sha256']==endpoint['state_sha256'], 'Frozen selection history differs')
        j.update(selected_epoch=epoch,selected_metric_value=value,initial_state_sha256=payload['initial_state_sha256'],
            source_revision_history=payload['source_revision_history'],checkpoint_state_sha256=payload['model_state_sha256'],
            checkpoint={'relative_path':checkpoint_name,'sha256':bound[checkpoint_name]},
            checkpoint_metadata_path=metadata_name,checkpoint_metadata_sha256=bound[metadata_name],
            selection_history_path=history_name,selection_history_sha256=bound[history_name],
            validation_reference=validation_reference(endpoint),binding_status='bound_terminal_immutable_originals')
        e.validate_payload(payload,j,contract['sources']['components'],contract['datasets'][j['dataset']]['definition'],metadata)
        del payload
    contract.update(binding_complete=True,training_receipt_path=proof,training_receipt_sha256=bound[proof],bound_files=bound)
    contract['contract_sha256']=canonical(contract);write(BUNDLE/'contract.json',contract)
    e.verify_contract(BUNDLE);return contract

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
assert str(r) in p['allowed_roots'] and r.name.startswith('titantpp_backbone_value_difference_raf3_20261016_v1_test_')
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
        permit = {'schema': 'titantpp_value_difference_raf3_evaluation_permit_v1', 'authorized': True,
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
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
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
    write(BUNDLE / 'validation_gate.json', {'schema': 'titantpp_value_difference_raf3_validation_gate_v1',
          'cohort': c['cohort'], 'status': 'PASS', 'contract_sha256': c['contract_sha256'],
          'job_count': 3, 'validation_receipts': receipts, 'validation_worker_exits': exits, 'created_unix': time.time()})
    return e.verify_global_validation_gate(BUNDLE, c)














def aggregate(rows, split='test'):
    keys = [('whole_qty_rmse',None,'qty_rmse'),('whole_qty_mae',None,'qty_mae'),
        ('tail_qty_rmse','tail','qty_rmse'),('tail_qty_mae','tail','qty_mae'),
        ('whole_time_nll',None,'time_nll'),('tail_time_nll','tail','time_nll')]
    expected = {(d,arm,seed) for d in DATASETS for arm in (CNN,VALUE) for seed in SEEDS}
    by_key = {(r['dataset'],r['arm'],r['seed']):r for r in rows}
    require(len(rows) == len(by_key) == 6 and set(by_key) == expected, 'Expected candidate3 and historical CNN3 only')
    grouped,deltas = [],[]
    for dataset in DATASETS:
        for arm in (CNN,VALUE):
            subset = [by_key[(dataset,arm,s)] for s in SEEDS]
            metrics = {}
            for name,group,key in keys:
                values = [(r['metrics'] if group is None else r['metrics'][group])[key] for r in subset]
                require(all(type(v) in (int,float) and math.isfinite(v) for v in values), 'Nonfinite metric')
                metrics[name] = {'mean':statistics.mean(values),'sample_sd':statistics.stdev(values)}
            grouped.append({'dataset':dataset,'arm':arm,'seed_count':3,'metrics':metrics})
        for seed in SEEDS:
            candidate,reference = by_key[(dataset,VALUE,seed)],by_key[(dataset,CNN,seed)]
            for key in ('target_identity_sha256','truth_sha256','time_sha256'):
                require(candidate[key] == reference[key], 'Matched baseline target population differs')
            metrics = {}
            for name,group,key in keys:
                left = (candidate['metrics'] if group is None else candidate['metrics'][group])[key]
                right = (reference['metrics'] if group is None else reference['metrics'][group])[key]
                metrics[name] = {'candidate_minus_baseline':left-right,
                    'relative_percent':100*(left-right)/right if right != 0 else None}
            deltas.append({'dataset':dataset,'seed':seed,'candidate':VALUE,'baseline':CNN,'metrics':metrics})
    from paper.scripts.compare_titantpp_value_difference_loop import RUNTIME_LIMITATION,INTERVENTION_LIMITATION
    return {'status':'complete','split':split,'candidate_fit_count':3,'baseline_reference_count':3,
        'comparison_model_count':6,'seed_count':3,'grouped':grouped,'seed_differences':deltas,'seed_results':rows,
        'test_independent':False,'runtime_limitation':RUNTIME_LIMITATION,'intervention_limitation':INTERVENTION_LIMITATION,
        'new_training':False,'checkpoint_reselection':False}


def compare_completed(c):
    from paper.scripts.compare_titantpp_value_difference_loop import (
        compare_historical_rows,normalized_native_row,verify_registry)
    e = evaluator(); e.verify_global_validation_gate(BUNDLE,c)
    spec = c['external_registry']
    registry = verify_registry(safe(BUNDLE,spec['path']),spec['sha256'],root=PROJECT)
    rows = []
    for j in c['jobs']:
        for split,name in (('validation','validation_replay.json'),('test','test_metrics.json')):
            folder = BUNDLE/'results'/j['id']; ex = folder/(split+'_worker_exit.json'); record = read(ex)
            require(record['actual_child_exit_observed'] is True and type(record['exit_code']) is int
                and record['exit_code'] == 0 and record['job_id'] == j['id']
                and record['contract_sha256'] == c['contract_sha256'], 'Comparison worker exit0 missing')
            raw = check_receipt(read(folder/name),c,j,split,e)
            if split == 'test':
                require(raw['validation_gate_sha256'] == sha(BUNDLE/'validation_gate.json'), 'Comparison Test gate changed')
            row = normalized_native_row(raw,j['evaluation_host'])
            row['native_evidence'] = {'receipt_path':str((folder/name).relative_to(BUNDLE)),
                'receipt_sha256':sha(folder/name),'worker_exit_path':str(ex.relative_to(BUNDLE)),
                'worker_exit_sha256':sha(ex),'validation_gate_sha256':sha(BUNDLE/'validation_gate.json')}
            rows.append(row)
    result = compare_historical_rows(registry,rows)
    result.update(registry_sha256=spec['sha256'],normalized_candidate_rows=rows,
        comparison_scope_amendment_sha256=c['comparison_scope_amendment_sha256'])
    require(c['candidate_selection_rule'] == result['candidate_selection']['selection_rule'], 'Candidate selection rule changed')
    write(REPORT/'EXTERNAL_COMPARISON.json',result)
    lines = ['# RAF V 관측 차분 CNN과 기존 CNN·외부 비교군','',
        'Validation에서 선택한 checkpoint3개를 고정했다. 기존 CNN3·외부 모델은 저장 결과만 재사용했다.',
        '사용자 기준은 두 split의 RMSE·MAE 평균이 기존 CNN 및 외부 지표별 최우수보다 모두 엄격하게 낮은지다.','',
        '| split | 지표 | 후보 평균±표본SD | CNN 평균±표본SD | 외부 최우수 평균 | 외부 모델 | 통과 |',
        '|---|---|---:|---:|---:|---|---|']
    for split,metrics in result['summaries'].items():
        for metric,row in metrics.items():
            fmt = lambda x:f"{x['mean']:.6f}±{x['sample_sd']:.6f}"
            lines.append(f"| {split} | {metric} | {fmt(row['candidate'])} | {fmt(row['historical_CNN'])} | {row['external_frontier']['mean']:.6f} | {', '.join(row['external_frontier']['models'])} | {row['user_performance_pass']} |")
    lines += ['',f"네 기준 전체 통과: {result['user_performance_gate']}. 다음 데이터셋 적용 후보: {result['candidate_selection']['selected_candidate']}.",
        '외부 모델은 공통 수량·시간 head를 사용한 과거 비교 구현이며 원 논문 성능과 동일시하지 않는다.',
        '기존 CNN/외부는5080/cuDNN91900, 신규후보는5090/cuDNN92000이다. 동일 서버 CNN을 새로 학습·평가하지 않았다.',
        'CNN 계수576→512로 자유도, 짧은 이력 경계 처리, 차분 계수의 학습·정규화 방식이 함께 달라진다. 원인 기여를 분리해 입증한 비교가 아니다.',
        '기존 Test를 참고한 탐색적 반복이다. Test는 확장 자격 판단에 사용되며 새 독립 평가·보편적 개선·원인 확정·통계적 유의성을 주장하지 않는다.',
        '모든 seed별 절대·상대차이 및 악화는 JSON에 보존했다.','']
    with (REPORT/'EXTERNAL_COMPARISON.md').open('x') as f:f.write('\n'.join(lines))
    return result


def write_summary_markdown(result, path):
    labels = {CNN:'기존 CNN(원결과 재사용)',VALUE:'V 관측 차분 CNN'}
    lines = [f"# RAF 3seed {result['split']} 비교",'',
        '선택은 fullValidation raw 수량 RMSE 최초 strict finite 최소 epoch다. 같은 선택 모델의 여섯 지표를 비교한다.',
        '신규3개만 평가하고 기존 CNN3은 원결과를 재사용했다. 표본SD는 ddof1이다.','',
        '| 구조 | RMSE 평균±표본SD | MAE 평균±표본SD |','|---|---:|---:|']
    for row in result['grouped']:
        m=row['metrics'];fmt=lambda key:f"{m[key]['mean']:.6f}±{m[key]['sample_sd']:.6f}"
        lines.append(f"| {labels[row['arm']]} | {fmt('whole_qty_rmse')} | {fmt('whole_qty_mae')} |")
    lines += ['','| seed | 구조 | 선택 epoch | RMSE | MAE |','|---:|---|---:|---:|---:|']
    for row in sorted(result['seed_results'],key=lambda r:(r['seed'],r['arm'])):
        lines.append(f"| {row['seed']} | {labels[row['arm']]} | {row['selected_epoch']} | {row['metrics']['qty_rmse']:.6f} | {row['metrics']['qty_mae']:.6f} |")
    lines += ['','차이는 후보−CNN이며 음수가 개선이다.','',
        '| seed | ΔRMSE | 상대차(%) | ΔMAE | 상대차(%) |','|---:|---:|---:|---:|---:|']
    for row in result['seed_differences']:
        a,b=[row['metrics'][key] for key in ('whole_qty_rmse','whole_qty_mae')]
        pct=lambda v:'N/A' if v is None else f'{v:+.4f}'
        lines.append(f"| {row['seed']} | {a['candidate_minus_baseline']:+.6f} | {pct(a['relative_percent'])} | {b['candidate_minus_baseline']:+.6f} | {pct(b['relative_percent'])} |")
    lines += ['','| 구조 | 큰수량 RMSE | 큰수량 MAE | 전체 TimeNLL | 큰수량 TimeNLL |','|---|---:|---:|---:|---:|']
    for row in result['grouped']:
        m=row['metrics'];fmt=lambda key:f"{m[key]['mean']:.6f}±{m[key]['sample_sd']:.6f}"
        lines.append('| '+labels[row['arm']]+' | '+' | '.join(fmt(k) for k in ('tail_qty_rmse','tail_qty_mae','whole_time_nll','tail_time_nll'))+' |')
    lines += ['', '큰수량 q>200은 Train에서 고정했다. 시간 지표는 recorded positive-integer TimeNLL이다. 단순 기준선의 TimeNLL은 N/A다.',
        '기존 CNN5080/cuDNN91900과 후보5090/cuDNN92000의 차이가 남는다. 계수576→512, 짧은 이력 경계와 매개화 차이도 포함된 제약 전체의 비교다.',
        'Test를 참고한 탐색적 후속 검증이며 독립 일반화·인과관계·통계적 유의성을 주장하지 않는다. 모든 seed의 악화와 여섯 지표를 보존했다.','']
    with Path(path).open('x') as f:f.write('\n'.join(lines))


def validation_summary(c):
    e = evaluator()
    rows = [check_receipt(read(BUNDLE/'results'/j['id']/'validation_replay.json'),c,j,'validation',e) for j in c['jobs']]
    population = c['datasets'][DATASETS[0]]['validation_population']
    for ref in c['baseline_references']:
        metrics = ref['validation_reference']
        require(metrics['count'] == population['count'], 'Historical CNN Validation population count differs')
        e.compare_validation_metrics(metrics,metrics)
        rows.append({'dataset':ref['dataset'],'arm':CNN,'seed':ref['seed'],'selected_epoch':ref['selected_epoch'],
            'checkpoint_sha256':ref['checkpoint_sha256'],'split':'validation','metrics':copy.deepcopy(metrics),
            **{key:population[key] for key in ('target_identity_sha256','truth_sha256','time_sha256')},
            'reference_role':ref['role'],'retrained':False,'reevaluated':False,
            'metric_origin':'original_training_selected_validation_endpoint',
            'population_provenance':'frozen common dataset/loader/full count; not a new historical inference'})
    result = aggregate(rows,'validation');write(REPORT/'THREE_SEED_VALIDATION.json',result)
    write_summary_markdown(result,REPORT/'THREE_SEED_VALIDATION.md')
    return result


def summarize(c):
    e = evaluator(); validation_summary(c)
    rows = [check_receipt(read(BUNDLE/'results'/j['id']/'test_metrics.json'),c,j,'test',e) for j in c['jobs']]
    for ref in c['baseline_references']:
        require(sha(BUNDLE/ref['receipt_path']) == ref['receipt_sha256'], 'Reused CNN Test changed')
        raw = read(BUNDLE/ref['receipt_path'])
        require(raw['arm'] == LEGACY_CNN and ref['role'] == 'historical_immutable_reference'
                and ref['retrained'] is False and ref['reevaluated'] is False, 'Historical reference role differs')
        rows.append({**raw,'arm':CNN,'historical_original_arm':raw['arm'],'reference_role':ref['role'],
                     'reused_original_receipt_sha256':ref['receipt_sha256']})
    result = aggregate(rows);write(REPORT/'THREE_SEED_TEST.json',result)
    write_summary_markdown(result,REPORT/'THREE_SEED_TEST.md')
    comparison = compare_completed(c)
    files = {str(p):sha(p) for p in BUNDLE.rglob('*') if p.is_file()}
    files.update({str(REPORT/n):sha(REPORT/n) for n in ('THREE_SEED_TEST.json','THREE_SEED_TEST.md',
        'THREE_SEED_VALIDATION.json','THREE_SEED_VALIDATION.md','EXTERNAL_COMPARISON.json','EXTERNAL_COMPARISON.md')})
    receipt = {'status':'complete','candidate_test_count':3,'baseline_reference_count':3,'comparison_model_count':6,
        'seed_count':3,'validation_replay_count':3,'actual_native_worker_exit0_count':6,'actual_pipeline_exit0_count':2,
        'historical_cnn_retrained':False,'historical_cnn_reevaluated':False,'external_comparison_complete':True,
        'candidate_user_performance_gate':comparison['user_performance_gate'],'candidate_selection':comparison['candidate_selection'],
        'pipeline_wrapper_own_exit_directly_observed':False,'files':files,'files_sha256':canonical(files),
        'contract_sha256':c['contract_sha256'],'completed_unix':time.time(),'held_out_test_evaluated':True}
    write(BUNDLE/'final_receipt.json',receipt)
    return receipt


def advance(expected):
    training = verify_bootstrap(expected)
    final_path = BUNDLE / 'final_receipt.json'
    if final_path.exists():
        r = read(final_path)
        require(r['status'] == 'complete' and r['candidate_test_count'] == 3 and r['seed_count'] == 3
                and r['baseline_reference_count'] == 3 and r['comparison_model_count'] == 6
                and r['candidate_selection']['selection_rule'] == SELECTION_RULE and r['validation_replay_count'] == 3
                and r['actual_native_worker_exit0_count'] == 6 and r['actual_pipeline_exit0_count'] == 2
                and r['held_out_test_evaluated'] is True and r['external_comparison_complete'] is True
                and r['historical_cnn_retrained'] is False and r['historical_cnn_reevaluated'] is False
                and canonical(r['files']) == r['files_sha256'], 'Final receipt invalid')
        for n, value in r['files'].items():
            require(sha(n) == value, 'Final local evidence SHA differs')
        return {'status': 'complete', 'receipt_sha256': sha(final_path), 'remote_calls': 0}
    require(not (BUNDLE / 'advance.claim').exists() and not (BUNDLE / 'failure.json').exists(),
            'Existing attempt or failure: no automatic retry; remote0')
    receipt = collection_gate(training)
    if receipt is None:
        return {'status': 'waiting_for_verified_training3_and_original_collection', 'remote_calls': 0,
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
        return {'status': result['status'], 'candidate_test_count': 3, 'baseline_reference_count': 3,
                'receipt_sha256': sha(BUNDLE / 'final_receipt.json')}
    except BaseException as error:
        write(BUNDLE / 'failure.json', {'status': 'failed', 'error': str(error), 'traceback': traceback.format_exc(),
              'automatic_retry': False, 'completed_unix': time.time()})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('prepare', 'advance'), required=True)
    p.add_argument('--integrity-sha')
    p.add_argument('--training-integrity-sha')
    p.add_argument('--registry-sha')
    args = p.parse_args()
    if args.mode == 'prepare':
        require(bool(args.training_integrity_sha),'Training integrity SHA required')
        require(bool(args.registry_sha),'Frozen external registry SHA required')
        value = prepare(args.training_integrity_sha,args.registry_sha)
    else:
        require(bool(args.integrity_sha), 'Integrity SHA required')
        value = advance(args.integrity_sha)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))
