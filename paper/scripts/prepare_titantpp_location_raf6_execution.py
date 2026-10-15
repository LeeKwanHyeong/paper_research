"""Exclusive preparation/sealing for the approved RAF-first backbone loop.

Only two new model/runner files are added to the pinned legacy source snapshot.
Training remains Validation-only; the separately gated fixed-model evaluation
stage is authorized by the same user request, not by a trainer-side Test path.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='titantpp_backbone_location_raf6_20261015_v1'
BUNDLE=ROOT/'search_artifacts'/SCHEMA
REPORT=ROOT/'reports'/SCHEMA
PARENT=ROOT/'search_artifacts/titantpp_backbone_nonlinear_raf6_20261015_v1'
LEGACY=ROOT/'search_artifacts/titantpp_cnn_history_raf6_20261015_v1'
PARENT_REPORT=ROOT/'reports/titantpp_backbone_nonlinear_raf6_20261015_v1'
PARENT_INTEGRITY_SHA='e3d214ca7c32c1a23fa4800cdc10c7013852897cfb5b857e38765fd16e8eed3c'
PARENT_RECEIPT_SHA='5b4268e10e62cdd66290af811d239b2844949fa6a4d77701a69ca2aa6e865cdf'
PARENT_CONTRACT_SHA='ecd9dc293cd3fe13767182aa68e8a79622f0b96992a983714ba5516b37cb34e4'
ARMS=('titantpp_cnn_query_nonlinear_v1','titantpp_cnn_value_nonlinear_v1')
SEEDS=(42,52,62)
EXTENSIONS=('models/TPPs/CountAwareTitanNonlinearLocationCNN.py','paper/scripts/run_titantpp_nonlinear_location_cnn.py')
USER_INSTRUCTION='기존 CNN 제외하고 하던 거 마저 진행해.'
APPROVAL_CONTEXT='기존 CNN은 학습·재평가하지 않고 CNN backbone 두 후보×3seed를 RAF에서 5090 최대3 병렬로 검증한다. 공통 입력·손실·head·예산을 유지하고 Validation 선택 checkpoint를 고정해 Validation/Test RMSE·MAE를 비교한다. 미달 시 분석 후 별도 계약의 다음 루프로 이어가며 원본·실패 기록은 보존한다.'

def require(value,message):
    if not value:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()

def read(path):return json.loads(Path(path).read_text())

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')

def checked_copy(source,destination,digest):
    require(sha(source)==digest,'Source SHA mismatch '+str(source))
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    with Path(source).open('rb') as src,destination.open('xb') as dst:shutil.copyfileobj(src,dst)
    require(sha(destination)==digest,'Copied SHA mismatch')

def stored_channel_references(parent_contract):
    receipt_path=PARENT/'retrieval/final_receipt.json'
    require(sha(receipt_path)==PARENT_RECEIPT_SHA,'Parent training receipt drift')
    receipt=read(receipt_path)
    require(receipt['status']=='complete' and receipt['new_fit_count']==6
        and receipt['held_out_test_evaluated'] is False and receipt['all_training_terminal_verified'] is True,
        'Parent training retrieval not complete')
    pins={str(receipt_path.relative_to(ROOT)):sha(receipt_path),**receipt['files']}
    for n,d in pins.items():require(sha(ROOT/n)==d,'Parent training archive/result drift '+n)
    validation_path=PARENT_REPORT/'THREE_SEED_VALIDATION.json'
    require(str(validation_path.relative_to(ROOT)) in pins,'Parent Validation report not receipt-bound')
    validation=read(validation_path)
    require(validation['split']=='validation' and validation['dataset']=='raf_spare_parts'
        and validation['held_out_test_evaluated'] is False,'Mixed or foreign reference results')
    selected=[r for r in validation['rows'] if r['arm']=='Channel']
    require(len(selected)==3 and {r['seed'] for r in selected}==set(SEEDS),'Stored Channel seed inventory drift')
    jobs=[]
    for row in sorted(selected,key=lambda r:r['seed']):
        jid=f"raf_spare_parts__{row['seed']}__titantpp_cnn_channel_nonlinear_v1"
        job=next(j for j in parent_contract['jobs'] if j['id']==jid)
        folder=PARENT/'retrieval/originals/run'/jid
        endpoint_path=folder/'runs'/job['arm']/'count_only_log_regression'/f"seed_{row['seed']}"/'endpoint_replays.json'
        require(str(endpoint_path.relative_to(ROOT)) in pins,'Stored endpoint not receipt-bound')
        endpoint=read(endpoint_path)
        require(endpoint['best_epoch']==row['best_epoch'] and endpoint['selected']['count']==6690
            and endpoint['selected']['qty_rmse']==row['RMSE'] and endpoint['selected']['qty_mae']==row['MAE'],
            'Stored Channel report/endpoint mismatch')
        jobs.append({'id':jid,'arm':job['arm'],'seed':row['seed'],'dataset':'raf_spare_parts','host':'5090',
            'reference_role':'stored_prior_loop_qkv_channel_not_fresh_primary_control',
            'original':{'selected_epoch':endpoint['best_epoch'],'validation_reference':endpoint['selected'],
                'endpoint_path':str(endpoint_path.relative_to(ROOT)),'endpoint_sha256':sha(endpoint_path),
                'receipt_path':str(receipt_path.relative_to(ROOT)),'receipt_sha256':PARENT_RECEIPT_SHA}})
    return jobs,pins

def prepare():
    require(sha(PARENT/'execution_contract.json')==PARENT_CONTRACT_SHA,'Legacy RAF contract drift')
    pc=read(PARENT/'execution_contract.json')
    legacy_integrity=LEGACY/'followup_integrity.json'
    require(sha(legacy_integrity)=='ce6dda0a7dc53373651fd7988b70978ebb7ab30a89c90e2d1b607dd0f8b9153c','Legacy baseline integrity drift')
    legacy_ready=ROOT/'reports/titantpp_cnn_history_raf6_20261015_v1/LOCAL_READY.json'
    require(sha(legacy_ready)==read(legacy_integrity)['files'][str(legacy_ready.relative_to(ROOT))],'Legacy baseline manifest drift')
    baseline_pins=read(legacy_ready)['baseline_pins']
    require(len(baseline_pins)==15,'Historical CNN3 evidence inventory drift')
    for n,d in baseline_pins.items():require(sha(ROOT/n)==d,'Historical CNN evidence drift '+n)
    require(sha(PARENT/'followup_integrity.json')==PARENT_INTEGRITY_SHA,'Parent integrity drift')
    for n,d in read(PARENT/'followup_integrity.json')['files'].items():require(sha(ROOT/n)==d,'Parent evidence drift '+n)
    references,reference_pins=stored_channel_references(pc)
    parent_manifest=read(PARENT/'deployment_manifest.json')
    for n,d in parent_manifest['files'].items():require(sha(PARENT/n)==d,'Legacy deployment drift '+n)
    for n in EXTENSIONS:require((ROOT/n).is_file(),'Model/runner not ready: '+n)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for name in (*EXTENSIONS,'paper/scripts/titantpp_location_raf6_runtime.py'):
        tracked=subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT)
        require(hashlib.sha256(tracked).hexdigest()==sha(ROOT/name),'Commit the new implementation before preparation: '+name)
    policy_paths=[REPORT/n for n in ('EXPERIMENT_PLAN.md','SCHEDULER.md','EXTERNAL_BASELINES.json','EXTERNAL_BASELINES.md','DIAGNOSTIC_BASIS.json')]
    policy_paths += [ROOT/'paper/scripts/compare_titantpp_backbone_loop.py',ROOT/'paper/tests/test_titantpp_backbone_loop_comparison.py']
    policy_paths += [legacy_integrity,legacy_ready,PARENT/'followup_integrity.json',*[ROOT/n for n in baseline_pins],*[ROOT/n for n in reference_pins]]
    for path in policy_paths:require(path.is_file(),'Required experiment policy/registry is missing: '+str(path))
    require(not BUNDLE.exists(),'Existing bundle; never overwrite or reprepare')
    BUNDLE.mkdir();REPORT.mkdir(parents=True,exist_ok=True)
    source={}
    for n,d in pc['source']['files'].items():
        checked_copy(PARENT/'source'/n,BUNDLE/'source'/n,d);source[n]=d
    for n in EXTENSIONS:
        require(n not in source,'Extension overwrites old source')
        d=sha(ROOT/n);checked_copy(ROOT/n,BUNDLE/'source'/n,d);source[n]=d
    for n,d in pc['input_files'].items():checked_copy(PARENT/'source'/n,BUNDLE/'source'/n,d)
    require(len(pc['source']['files'])==137 and len(source)==139,'Parent/extended source inventory changed')
    design={'local_scope':'first attention block; original linear Q/K/V causal CNN preserved; one additional recent-3-observed-event branch on Q or V',
        'arms':list(ARMS),'added_active_parameters':{a:1024 for a in ARMS},
        'activation':'GELU','query_control':'per-channel nonlinear temporal branch on Q only',
        'value_candidate':'same per-channel nonlinear temporal branch on V only',
        'branch_initialization':'identical Q/Value temporal and pointwise tensors for the same seed; zero pointwise output; preserved common initialization and RNG',
        'unchanged_keys':['K projection','memory tokens','attention temperature and normalization'],
        'preserved':['log1p input','quantity/time heads','loss','memory','optimizer','training budget','strict earliest fullValidation RMSE selection'],
        'interpretation':'Post-Test-informed exploratory follow-up; not independent confirmatory evidence. Stored QKV Channel differs in branch count and parameter count; it is descriptive, not the primary location control.'}
    prep={'schema':SCHEMA+'_preparation','dataset':pc['datasets'][0],
        'baseline_jobs':pc['baseline_jobs'],'stored_channel_reference_jobs':references,'stored_channel_reference_pins':reference_pins,'source_files':source,'design':design,
        'legacy_contract_raw_sha256':PARENT_CONTRACT_SHA,'legacy_source_closure':pc['source']['files_sha256'],
        'legacy_source_revision':pc['source']['base_git_revision'],'implementation_revision':revision,
        'policy_files':{str(p.relative_to(ROOT)):sha(p) for p in policy_paths},
        'new_fit_count':6,'fresh_same_runtime_baseline_count':0,'baseline_reference_count':3,'stored_channel_reference_count':3,'held_out_test_evaluated':False}
    write(BUNDLE/'preparation/contract.json',prep)
    checked_copy(ROOT/'paper/scripts/titantpp_location_raf6_runtime.py',BUNDLE/'operation/runtime.py',sha(ROOT/'paper/scripts/titantpp_location_raf6_runtime.py'))
    checked_copy(PARENT/'operation/frozen_runtime.py',BUNDLE/'operation/frozen_runtime.py',sha(PARENT/'operation/frozen_runtime.py'))
    checked_copy(PARENT/'preparation/original_5090_qualification.json',BUNDLE/'preparation/original_5090_qualification.json',sha(PARENT/'preparation/original_5090_qualification.json'))
    c=copy.deepcopy(pc);remote='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/'+SCHEMA+'_5090'
    host=c['hosts']['5090'];host.update(root=remote,source_root=remote+'/source',operation_root=remote+'/operation',output_dir=remote+'/run',tmux=SCHEMA+'_5090')
    host['environment'].update(PYTHONPATH=remote+'/source',MPLCONFIGDIR=remote+'/cache/matplotlib',XDG_CACHE_HOME=remote+'/cache',SOURCE_REVISION=revision)
    c.update(schema=SCHEMA,arms=list(ARMS),seeds=list(SEEDS),stored_channel_reference_jobs=references,stored_channel_reference_count=3,
        jobs=[{'id':f'raf_spare_parts__{s}__{a}','host':'5090','dataset':'raf_spare_parts','seed':s,'arm':a} for a in ARMS for s in SEEDS],
        source={**pc['source'],'base_git_revision':revision,'frozen_parent_revision':pc['source']['base_git_revision'],
            'files':source,'files_sha256':canonical(source)},
        design_sha256=canonical(design),architecture=design,
        preparation_contract_sha256=sha(BUNDLE/'preparation/contract.json'),
        comparison_runtime_limitation='Candidates use 5090/cuDNN92000; historical CNN3 uses 5080/cuDNN91900. Candidate−CNN includes this runtime difference; Value−Query is same-runtime.',
        runner_modules={a:'paper.scripts.run_titantpp_nonlinear_location_cnn' for a in ARMS},
        primary_contrast='Value−Query: one identically initialized 1024-parameter branch; only Q versus V location differs',
        subsequent_evaluation={'authorized':True,'candidate_test_count':6,'validation_replay_count':6,
            'all_native_validation_required_before_any_test':True,'checkpoint_reselection':False})
    c['parallelism']['grouping']='each_arm_three_seeds_then_next_arm'
    c['operation']={'files':{str(p.relative_to(BUNDLE)):sha(p) for p in (BUNDLE/'operation').glob('*.py')}}
    write(BUNDLE/'execution_contract.json',c)
    write(BUNDLE/'approval.json',{'approved':True,'hosts':['5090'],'contract_sha256':canonical(c),
        'user_instruction':USER_INSTRUCTION,'scope':APPROVAL_CONTEXT,'fresh_fit_count':6,
        'maximum_concurrent_workers':3,'new_training_authorized':True,'fixed_checkpoint_test_authorized':True,
        'automatic_retry':False,'loop_index':2})
    files={str(p.relative_to(BUNDLE)):sha(p) for p in sorted(BUNDLE.rglob('*')) if p.is_file()}
    write(BUNDLE/'deployment_manifest.json',{'files':files,'contract_sha256':canonical(c)})
    write(REPORT/'LOCAL_READY.json',{'status':'prepared_native_checks_pending',
        'contract_raw_sha256':sha(BUNDLE/'execution_contract.json'),'contract_canonical_sha256':canonical(c),
        'deployment_manifest_sha256':sha(BUNDLE/'deployment_manifest.json'),'source_files':len(source),
        'unchanged_inherited_source_files':len(pc['source']['files']),'source_extension_files':list(EXTENSIONS),
        'scientific_fits_started':0,'held_out_test_evaluated':False,
        'frozen_scientific_fit_sha256':sha(BUNDLE/'operation/frozen_runtime.py')})
    print(json.dumps({'status':'prepared','bundle':str(BUNDLE),'contract_sha256':canonical(c)}))

def seal(tests_receipt,model_tests_receipt):
    require(tests_receipt is not None,'A real local test receipt is required')
    test=Path(tests_receipt).resolve();result=read(test)
    require(result['actual_exit']==0 and result.get('failed',0)==0 and result.get('passed',0)>0,'Local tests did not pass')
    require(model_tests_receipt is not None,'A real model/runner test receipt is required')
    model_test=Path(model_tests_receipt).resolve();model_result=read(model_test)
    require(model_result['actual_exit']==0 and model_result.get('failed',0)==0 and model_result.get('passed',0)>0,'Model tests did not pass')
    require(not (BUNDLE/'followup_integrity.json').exists(),'Existing integrity; never reseal')
    c=read(BUNDLE/'execution_contract.json')
    for n,d in read(BUNDLE/'deployment_manifest.json')['files'].items():require(sha(BUNDLE/n)==d,'Bundle drift before seal '+n)
    for n in EXTENSIONS:require(sha(ROOT/n)==c['source']['files'][n],'Model/runner changed after preparation')
    for n,d in read(BUNDLE/'preparation/contract.json')['policy_files'].items():require(sha(ROOT/n)==d,'Policy/registry changed before seal')
    paths=[p for p in BUNDLE.rglob('*') if p.is_file()]
    paths += [ROOT/'paper/scripts'/n for n in ('prepare_titantpp_location_raf6_execution.py',
        'titantpp_location_raf6_runtime.py','deploy_titantpp_location_raf6.py','advance_titantpp_location_raf6.py')]
    paths += [ROOT/n for n in EXTENSIONS]
    paths += [ROOT/'paper/tests/test_titantpp_location_raf6_execution.py']
    paths += [ROOT/'paper/tests/test_titantpp_nonlinear_location_cnn.py',test,model_test,REPORT/'LOCAL_READY.json']
    paths += [ROOT/n for n in read(BUNDLE/'preparation/contract.json')['policy_files']]
    files={str(p.relative_to(ROOT)):sha(p) for p in paths}
    write(BUNDLE/'followup_integrity.json',{'schema':SCHEMA+'_integrity','files':files,
        'local_tests_receipt':str(test.relative_to(ROOT)),'local_tests_receipt_sha256':sha(test),
        'model_tests_receipt':str(model_test.relative_to(ROOT)),'model_tests_receipt_sha256':sha(model_test),
        'contract_raw_sha256':sha(BUNDLE/'execution_contract.json'),'source_closure':c['source']['files_sha256'],
        'held_out_test_evaluated':False})
    print(json.dumps({'status':'sealed','integrity_sha256':sha(BUNDLE/'followup_integrity.json')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','seal'],required=True)
    p.add_argument('--tests-receipt');p.add_argument('--model-tests-receipt');a=p.parse_args()
    prepare() if a.mode=='prepare' else seal(a.tests_receipt,a.model_tests_receipt)
