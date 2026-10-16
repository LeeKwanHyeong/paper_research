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
SCHEMA='titantpp_backbone_nonlinear_raf9_20261015_v1'
BUNDLE=ROOT/'search_artifacts'/SCHEMA
REPORT=ROOT/'reports'/SCHEMA
PARENT=ROOT/'search_artifacts/titantpp_cnn_history_raf6_20261015_v1'
PARENT_CONTRACT_SHA='7f67baa9a6e67383d47353d7b6e563f0c25eed27d6849f9847be0a0960cd2ca2'
ARMS=('titantpp_cnn','titantpp_cnn_channel_nonlinear_v1','titantpp_cnn_joint_nonlinear_v1')
SEEDS=(42,52,62)
EXTENSIONS=('models/TPPs/CountAwareTitanNonlinearLocalCNN.py','paper/scripts/run_titantpp_nonlinear_local_cnn.py')
USER_INSTRUCTION='ㅇㅇ 진행하자. 승인할것들에 대해서는 미리 승인한다. 지금부터 진행시켜!'
APPROVAL_CONTEXT='CNN backbone 3구조×3seed를 RAF에서 5090 최대3 병렬로 검증한다. 공통 입력·손실·head·예산을 유지하고 Validation 선택 checkpoint를 고정해 Validation/Test RMSE·MAE를 비교한다. 미달 시 분석 후 별도 계약의 다음 루프로 이어가며 원본·실패 기록은 보존한다.'

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

def prepare():
    require(sha(PARENT/'execution_contract.json')==PARENT_CONTRACT_SHA,'Legacy RAF contract drift')
    pc=read(PARENT/'execution_contract.json')
    parent_manifest=read(PARENT/'deployment_manifest.json')
    for n,d in parent_manifest['files'].items():require(sha(PARENT/n)==d,'Legacy deployment drift '+n)
    for n in EXTENSIONS:require((ROOT/n).is_file(),'Model/runner not ready: '+n)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for name in (*EXTENSIONS,'paper/scripts/titantpp_nonlinear_raf9_runtime.py'):
        tracked=subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT)
        require(hashlib.sha256(tracked).hexdigest()==sha(ROOT/name),'Commit the new implementation before preparation: '+name)
    policy_paths=[REPORT/n for n in ('EXPERIMENT_PLAN.md','SCHEDULER.md','EXTERNAL_BASELINES.json','EXTERNAL_BASELINES.md')]
    policy_paths += [ROOT/'paper/scripts/compare_titantpp_backbone_loop.py',ROOT/'paper/tests/test_titantpp_backbone_loop_comparison.py']
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
    design={'local_scope':'first attention block Q/K/V projections; recent 3 observed events',
        'arms':list(ARMS),'added_active_parameters':{ARMS[0]:0,ARMS[1]:3072,ARMS[2]:3072},
        'activation':'GELU','channel_control':'per-channel nonlinear temporal processing',
        'joint_candidate':'joint temporal/channel nonlinear processing',
        'preserved':['log1p input','quantity/time heads','loss','memory','optimizer','training budget','strict earliest fullValidation RMSE selection'],
        'interpretation':'Post-Test-informed exploratory follow-up; not independent confirmatory evidence.'}
    prep={'schema':SCHEMA+'_preparation','dataset':pc['datasets'][0],
        'baseline_jobs':pc['baseline_jobs'],'source_files':source,'design':design,
        'legacy_contract_raw_sha256':PARENT_CONTRACT_SHA,'legacy_source_closure':pc['source']['files_sha256'],
        'legacy_source_revision':pc['source']['base_git_revision'],'implementation_revision':revision,
        'policy_files':{str(p.relative_to(ROOT)):sha(p) for p in policy_paths},
        'new_fit_count':9,'fresh_same_runtime_baseline_count':3,'held_out_test_evaluated':False}
    write(BUNDLE/'preparation/contract.json',prep)
    checked_copy(ROOT/'paper/scripts/titantpp_nonlinear_raf9_runtime.py',BUNDLE/'operation/runtime.py',sha(ROOT/'paper/scripts/titantpp_nonlinear_raf9_runtime.py'))
    checked_copy(PARENT/'operation/frozen_runtime.py',BUNDLE/'operation/frozen_runtime.py',sha(PARENT/'operation/frozen_runtime.py'))
    checked_copy(PARENT/'preparation/original_5090_qualification.json',BUNDLE/'preparation/original_5090_qualification.json',sha(PARENT/'preparation/original_5090_qualification.json'))
    c=copy.deepcopy(pc);remote='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/'+SCHEMA+'_5090'
    host=c['hosts']['5090'];host.update(root=remote,source_root=remote+'/source',operation_root=remote+'/operation',output_dir=remote+'/run',tmux=SCHEMA+'_5090')
    host['environment'].update(PYTHONPATH=remote+'/source',MPLCONFIGDIR=remote+'/cache/matplotlib',XDG_CACHE_HOME=remote+'/cache',SOURCE_REVISION=revision)
    c.update(schema=SCHEMA,arms=list(ARMS),seeds=list(SEEDS),
        jobs=[{'id':f'raf_spare_parts__{s}__{a}','host':'5090','dataset':'raf_spare_parts','seed':s,'arm':a} for a in ARMS for s in SEEDS],
        source={**pc['source'],'base_git_revision':revision,'frozen_parent_revision':pc['source']['base_git_revision'],
            'files':source,'files_sha256':canonical(source)},
        design_sha256=canonical(design),architecture=design,
        preparation_contract_sha256=sha(BUNDLE/'preparation/contract.json'),
        comparison_runtime_limitation='All three arms are fresh fits on the same 5090/cuDNN92000 runtime. Historical CNN results are retained separately.',
        primary_contrast='Joint−Channel with exactly matched added active parameter count',
        subsequent_evaluation={'authorized':True,'candidate_test_count':9,'validation_replay_count':9,
            'all_native_validation_required_before_any_test':True,'checkpoint_reselection':False})
    c['parallelism']['grouping']='each_arm_three_seeds_then_next_arm'
    c['operation']={'files':{str(p.relative_to(BUNDLE)):sha(p) for p in (BUNDLE/'operation').glob('*.py')}}
    write(BUNDLE/'execution_contract.json',c)
    write(BUNDLE/'approval.json',{'approved':True,'hosts':['5090'],'contract_sha256':canonical(c),
        'user_instruction':USER_INSTRUCTION,'scope':APPROVAL_CONTEXT,'fresh_fit_count':9,
        'maximum_concurrent_workers':3,'new_training_authorized':True,'fixed_checkpoint_test_authorized':True,
        'automatic_retry':False,'loop_index':1})
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
    paths += [ROOT/'paper/scripts'/n for n in ('prepare_titantpp_nonlinear_raf9_execution.py',
        'titantpp_nonlinear_raf9_runtime.py','deploy_titantpp_nonlinear_raf9.py','advance_titantpp_nonlinear_raf9.py')]
    paths += [ROOT/n for n in EXTENSIONS]
    paths += [ROOT/'paper/tests/test_titantpp_nonlinear_raf9_execution.py']
    paths += [ROOT/'paper/tests/test_titantpp_nonlinear_local_cnn.py',test,model_test,REPORT/'LOCAL_READY.json']
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
