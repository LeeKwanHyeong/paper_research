"""Create a fresh execution overlay, preserving the reviewed preparation bundle."""
from pathlib import Path
import base64, json, shutil
from paper.scripts import titantpp_history_taxi6_runtime as op
ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'search_artifacts/titantpp_cnn_history_adaptation_matched_20261015_v1'
BUNDLE=PREP/'execution_v1'
REPORT=ROOT/'reports/titantpp_cnn_history_adaptation_matched_20261015_v1/execution_v1'
USER='ㅇㅇ 그럼 Taxi 두 후보 × 3seed, 총 6개 학습 승인할게. 기존 CNN 결과와 공통 조건은 유지하고, GPU·Runtime·입력 SHA 검증 통과하면 5080에서 최대 3개씩 병렬로 돌려. 끝나면 기존 CNN과 두 후보의 Validation RMSE·MAE, seed별 차이를 비교해줘.'

def main():
    op.require(op.sha_file(PREP/'contract.json')==op.PREPARATION_SHA,'Preparation contract mismatch')
    receipt=op.read(REPORT.parent/'FINAL_RECEIPT.json')
    for name,digest in {**receipt['files'],**receipt['baseline_binary_and_selection_files_verified']}.items():
        op.require(op.sha_file(ROOT/name)==digest,'Preparation/baseline drift: '+name)
    prepared=op.read(PREP/'contract.json');BUNDLE.mkdir(exist_ok=False);REPORT.mkdir(exist_ok=False)
    shutil.copytree(PREP/'source',BUNDLE/'source')
    adapter='paper/scripts/run_titantpp_adaptive_history_cnn.py'
    shutil.copy2(ROOT/adapter,BUNDLE/'source'/adapter)
    source={**prepared['source_files'],adapter:op.sha_file(ROOT/adapter)}
    (BUNDLE/'operation').mkdir();(BUNDLE/'preparation').mkdir()
    shutil.copy2(PREP/'contract.json',BUNDLE/'preparation/contract.json')
    old=ROOT/'search_artifacts/titantpp_gru_residual_cap_validation_20261014_v1/execution_manual2'
    shutil.copy2(old/'operation/frozen_runtime.py',BUNDLE/'operation/frozen_runtime.py')
    shutil.copy2(ROOT/'paper/scripts/titantpp_history_taxi6_runtime.py',BUNDLE/'operation/runtime.py')
    data=prepared['dataset'];inputs={}
    for key in ('data','split_manifest'):
        item=data['inherited_data_identity'][key];name=item['path'];digest=item['sha256']
        op.require(op.sha_file(old/'source'/name)==digest,'Input changed')
        target=BUNDLE/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/'source'/name,target);inputs[name]=digest
    sentinel=BUNDLE/'source/sample_data/.frozen_root_sentinel';sentinel.parent.mkdir(parents=True,exist_ok=True);sentinel.touch();inputs[str(sentinel.relative_to(BUNDLE/'source'))]=op.sha_file(sentinel)
    snapshot=op.read(ROOT/'search_artifacts/titantpp_cnn_gru_components_20261012_v5/control/qualification_cache/5080/snapshot.json')
    qb=base64.b64decode(snapshot['files']['qualification/receipt.json']['base64'])
    op.require(__import__('hashlib').sha256(qb).hexdigest()==snapshot['files']['qualification/receipt.json']['raw_sha256'],'Original runtime cache mismatch')
    (BUNDLE/'preparation/original_cnn_qualification.json').write_bytes(qb);qr=json.loads(qb)['runtime']
    root='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/'+op.SCHEMA+'_5080'
    host=op.read(old/'execution_contract.json')['hosts']['5080'];host.update(root=root,source_root=root+'/source',operation_root=root+'/operation',output_dir=root+'/run',tmux=op.SCHEMA+'_5080',assigned_datasets=['yellow_trip_hourly'])
    host['environment'].update(PYTHONPATH=root+'/source',MPLCONFIGDIR=root+'/cache/matplotlib',XDG_CACHE_HOME=root+'/cache')
    # The preparation runtime list was inference metadata (threads1). Match
    # original CNN scientific training's directly recorded threads4 instead.
    keys=('python','torch','numpy','polars','cuda','cudnn','threads','interop_threads','default_dtype','deterministic_algorithms','deterministic_warn_only','cuda_matmul_tf32','cudnn_tf32','gpu')
    host['runtime_expected']={k:qr[k] for k in keys}
    c={'schema':op.SCHEMA,'arms':list(op.ARMS),'seeds':list(op.SEEDS),'jobs':op.expected_jobs(),
       'datasets':[data],'baseline_jobs':prepared['baseline_jobs'],'training':op.TRAINING,'parallelism':op.PARALLELISM,
       'limits':{'total_wall_seconds':168*3600,'per_condition_seconds':36*3600,'qualification_seconds':1800,'automatic_retry':False},
       'source':{'files':source,'files_sha256':op.sha_json(source),'base_git_revision':'57da14a2a7a5687e1bdef7d3164f3e8c0051da59'},
       'input_files':inputs,'hosts':{'5080':host},'evaluation_scope':'validation_only','held_out_test_evaluated':False,'test_evaluation':False,
       'design_sha256':prepared['design_sha256'],'architecture':{'prepared_contract_sha256':op.PREPARATION_SHA,'added_parameters_each':243},
       'preparation_contract_sha256':op.PREPARATION_SHA,'original_training_runtime_receipt_sha256':op.sha_file(BUNDLE/'preparation/original_cnn_qualification.json')}
    c['operation']={'files':{str(p.relative_to(BUNDLE)):op.sha_file(p) for p in (BUNDLE/'operation').glob('*.py')}}
    op.write_json(BUNDLE/'execution_contract.json',c,exclusive=True)
    op.write_json(BUNDLE/'approval.json',{'approved':True,'hosts':['5080'],'contract_sha256':op.sha_json(c),'user_instruction':USER,'fresh_fit_count':6,'new_training_authorized':True,'Test':False},exclusive=True)
    files={str(p.relative_to(BUNDLE)):op.sha_file(p) for p in sorted(BUNDLE.rglob('*')) if p.is_file()}
    op.write_json(BUNDLE/'deployment_manifest.json',{'files':files,'contract_sha256':op.sha_json(c)},exclusive=True)
    op.write_json(REPORT/'LOCAL_READY.json',{'status':'ready_for_native_validation','contract_raw_sha256':op.sha_file(BUNDLE/'execution_contract.json'),
        'contract_canonical_sha256':op.sha_json(c),'deployment_manifest_sha256':op.sha_file(BUNDLE/'deployment_manifest.json'),
        'source_count':len(source),'preparation_verified_count':len(receipt['files']),'baseline_verified_count':len(receipt['baseline_binary_and_selection_files_verified']),
        'thread_setting':'original native scientific training threads4; preparation inference threads1 is not substituted',
        'scientific_fits_started':0,'held_out_test_evaluated':False},exclusive=True)
    print(json.dumps({'bundle':str(BUNDLE),'contract_sha256':op.sha_json(c),'files':len(files)}))

if __name__=='__main__':main()
