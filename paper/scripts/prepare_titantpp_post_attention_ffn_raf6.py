"""Exclusive preparation of six RAF post-attention FFN fits; never executes training."""
from __future__ import annotations
import argparse, copy, hashlib, json, shutil, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SCHEMA='titantpp_backbone_post_attention_ffn_raf6_20261016_v1'
BUNDLE=ROOT/'search_artifacts'/SCHEMA
REPORT=ROOT/'reports'/SCHEMA
PARENT=ROOT/'search_artifacts/titantpp_backbone_qk_location_raf6_20261016_v1'
LEGACY=ROOT/'search_artifacts/titantpp_cnn_history_raf6_20261015_v1'
PARENT_CONTRACT_SHA='255fb0b322a71d3bc86568fd99fa10606dd6a8f095e0fa50ca20622be632bed8'
PARENT_INTEGRITY_SHA='919855db1ad3690483624742fb20e3907c5e92636ad25513f6a71335e0681fac'
PARENT_RECEIPT_SHA='ac680e7665de247681a9e223b0f5e635217f342461104c31c10221e2e6c41342'
PARENT_START_SHA='b3faca44824ce90972a62a740bb338b2cf5f54b0dfeb0d3fb291e06bed2a7f5a'
SCIENCE_SHA='4222d04567f7764fcb5bb47177f692e189d56fbd42707ddad83861006672ced1'
REGISTRY_SHA='57e88f4d5d338c6da7f54fe38d2b9492b06476fa93161331481de3c4ee1a6b49'
ARMS=('titantpp_cnn_ffn_current_v1','titantpp_cnn_ffn_temporal_v1')
SEEDS=(42,52,62)
EXTENSIONS=('models/TPPs/CountAwareTitanPostAttentionFFN.py','paper/scripts/run_titantpp_post_attention_ffn.py')
OPS=('prepare_titantpp_post_attention_ffn_raf6.py','titantpp_post_attention_ffn_raf6_runtime.py','deploy_titantpp_post_attention_ffn_raf6.py','advance_titantpp_post_attention_ffn_raf6.py')
USER_INSTRUCTION='1순위 진행하자'
DESIGN_MANIFEST_SHA='c72536e50cd5dec00e52ed97341333bc5511eff1afb36f7d4c5f4d5ffc5345ff'
APPROVAL_CONTEXT='CNN backbone 반복 개선 승인 범위의 RAF5090 Current/Temporal FFN 두 후보×3seed 총6fit, 최대3병렬. 기존CNN 학습/재평가0. 공통 입력·손실·head·메모리·예산 유지. 고정 선택checkpoint의 별도 Validation/Test 평가 승인. 기존 절대기한 연장0, 자동retry0.'

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def checked_copy(src,dst,digest):
    require(sha(src)==digest,'Source SHA mismatch '+str(src))
    dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
    with Path(src).open('rb') as a,dst.open('xb') as b:shutil.copyfileobj(a,b)
    require(sha(dst)==digest,'Copied SHA mismatch')

def parent_training_pins():
    """Read only the completed parent training/Validation evidence, never Test."""
    receipt_path=PARENT/'retrieval/final_receipt.json'
    require(sha(receipt_path)==PARENT_RECEIPT_SHA,'Parent training receipt drift')
    r=read(receipt_path)
    require(r.get('status')=='complete' and r.get('new_fit_count')==6
        and r.get('held_out_test_evaluated') is False and r.get('all_training_terminal_verified') is True,
        'Parent training is not complete Validation-only evidence')
    pins={str(receipt_path.relative_to(ROOT)):PARENT_RECEIPT_SHA,**r['files']}
    for n,d in pins.items():require(sha(ROOT/n)==d,'Parent training artifact drift '+n)
    start_path=PARENT/'preparation/inherited_5090_start_permit.json'
    require(read(PARENT/'followup_integrity.json')['files'].get(str(start_path.relative_to(ROOT)))==PARENT_START_SHA and sha(start_path)==PARENT_START_SHA,
        'Inherited host deadline is not receipt-bound')
    start=read(start_path)
    require(start['host']=='5090' and start['contract_sha256']=='34b026b0c871695a3302709a8cb7007780c19cc1384b431e6da51c60fb65f7a6'
        and start['deadline_unix']-start['started_at_unix']==168*3600,'Parent absolute deadline drift')
    pins[str(start_path.relative_to(ROOT))]=PARENT_START_SHA
    return pins,start

def design_pins():
    """Verify the frozen research design and original evidence before any writes."""
    require(sha(REPORT/'SCIENCE_SPEC.json')==SCIENCE_SHA,'Scientific specification drift')
    manifest=REPORT/'DESIGN_MANIFEST.json'
    require(sha(manifest)==DESIGN_MANIFEST_SHA,'Design manifest drift')
    pins={str(manifest.relative_to(ROOT)):DESIGN_MANIFEST_SHA,**read(manifest)['files']}
    require(len(read(manifest)['files'])==9,'Design inventory drift')
    for n,d in pins.items():require(sha(ROOT/n)==d,'Design evidence drift '+n)
    for source in ('EVIDENCE.json','VERIFIED_INPUTS.json'):
        x=read(REPORT/source);entries=x['files'] if source=='EVIDENCE.json' else x
        require(bool(entries),'Empty design evidence')
        for n,d in entries.items():
            require(sha(ROOT/n)==d,'Design input drift '+n)
            pins[n]=d
    require(sha(REPORT/'SCIENCE_SPEC.json')==SCIENCE_SHA,'Scientific specification drift')
    science=read(REPORT/'SCIENCE_SPEC.json')
    require(science['schema']=='post_attention_ffn_science_spec_v1'
        and list(science['arms'])==list(ARMS) and science['candidate_fit_count']==6 and science['seeds']==list(SEEDS)
        and science['historical_baseline']['reuse_saved_results_only'] is True
        and all(science['historical_baseline'][k] is False for k in ('training','inference','reevaluation'))
        and science['dimensions']['new_registered_and_trainable_parameters']==1152,
        'Scientific scope drift')
    require(read(REPORT/'DESIGN_REVIEW.json')['status']=='PASS_final_design_documents_only', 'Design review missing')
    return science,pins

def independent_review_pins():
    """Verify flat independent-review evidence, excluding self-referential QA JSONs."""
    reviews={'LOCAL_INDEPENDENT_QA.json':'PASS','MODEL_SCIENCE_REVIEW.json':'PASS_static_science_alignment'}
    forbidden={(REPORT/name).resolve() for name in reviews}
    pins={}
    for name,status in reviews.items():
        record=read(REPORT/name)
        require(record.get('status')==status,'Independent review did not pass: '+name)
        entries=record.get('files')
        require(isinstance(entries,dict) and bool(entries),'Empty independent review inventory: '+name)
        for key,digest in entries.items():
            require(type(key) is str and key and type(digest) is str and len(digest)==64,'Invalid independent review evidence')
            path=Path(key);target=(ROOT/path).resolve()
            require(not path.is_absolute() and '..' not in path.parts
                and target.is_relative_to(ROOT.resolve()),'Unsafe independent review evidence path')
            require(target not in forbidden,'Independent review inventory cannot contain QA JSONs')
            require(target.is_file() and sha(target)==digest,'Independent review evidence drift '+key)
            require(key not in pins or pins[key]==digest,'Conflicting independent evidence SHA')
            pins[key]=digest
    return pins

def prepare():
    science,design_evidence=design_pins()
    review_evidence=independent_review_pins()
    require(sha(REPORT/'EXTERNAL_BASELINES.json')==REGISTRY_SHA,'External registry drift')
    require(sha(PARENT/'execution_contract.json')==PARENT_CONTRACT_SHA,'Parent contract drift')
    pc=read(PARENT/'execution_contract.json')
    require(sha(PARENT/'followup_integrity.json')==PARENT_INTEGRITY_SHA,'Parent integrity drift')
    for n,d in read(PARENT/'followup_integrity.json')['files'].items():require(sha(ROOT/n)==d,'Parent sealed evidence drift '+n)
    parent_pins,start=parent_training_pins()
    legacy_integrity=LEGACY/'followup_integrity.json'
    require(sha(legacy_integrity)=='ce6dda0a7dc53373651fd7988b70978ebb7ab30a89c90e2d1b607dd0f8b9153c','Legacy CNN integrity drift')
    legacy_ready=ROOT/'reports/titantpp_cnn_history_raf6_20261015_v1/LOCAL_READY.json'
    require(sha(legacy_ready)==read(legacy_integrity)['files'][str(legacy_ready.relative_to(ROOT))],'CNN baseline manifest drift')
    baseline_pins=read(legacy_ready)['baseline_pins'];require(len(baseline_pins)==15,'CNN3 reference inventory drift')
    for n,d in baseline_pins.items():require(sha(ROOT/n)==d,'CNN reference drift '+n)
    for n,d in read(PARENT/'deployment_manifest.json')['files'].items():require(sha(PARENT/n)==d,'Parent deployment drift '+n)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for n in (*EXTENSIONS,*('paper/scripts/'+name for name in OPS)):
        tracked=subprocess.check_output(['git','show',revision+':'+n],cwd=ROOT)
        require(hashlib.sha256(tracked).hexdigest()==sha(ROOT/n),'Commit the new implementation before preparation: '+n)
    policies=[REPORT/n for n in ('EXTERNAL_BASELINES.json','EXTERNAL_BASELINES.md','LOCAL_INDEPENDENT_QA.json','MODEL_SCIENCE_REVIEW.json','IMPLEMENTATION_AUTHORIZATION.json')]
    policies += [ROOT/n for n in design_evidence]
    policies += [ROOT/n for n in review_evidence]
    policies += [legacy_integrity,legacy_ready,PARENT/'followup_integrity.json',*[ROOT/n for n in baseline_pins],*[ROOT/n for n in parent_pins]]
    for p in policies:require(p.is_file(),'Missing policy/evidence '+str(p))
    require(not BUNDLE.exists(),'Existing bundle; never overwrite or reprepare')
    BUNDLE.mkdir();source={}
    for n,d in pc['source']['files'].items():checked_copy(PARENT/'source'/n,BUNDLE/'source'/n,d);source[n]=d
    for n in EXTENSIONS:
        require(n not in source,'Extension overwrites inherited source')
        d=sha(ROOT/n);checked_copy(ROOT/n,BUNDLE/'source'/n,d);source[n]=d
    for n,d in pc['input_files'].items():checked_copy(PARENT/'source'/n,BUNDLE/'source'/n,d)
    require(len(pc['source']['files'])==145 and len(source)==147,'Source inventory drift')
    checked_copy(REPORT/'SCIENCE_SPEC.json',BUNDLE/'preparation/science_spec.json',SCIENCE_SHA)
    checked_copy(PARENT/'preparation/inherited_5090_start_permit.json',BUNDLE/'preparation/inherited_5090_start_permit.json',PARENT_START_SHA)
    design={key:copy.deepcopy(science[key]) for key in ('arms','placement','formula','dimensions','initialization','observation_contract','unchanged','interpretation_limits','excluded')}
    prep={'schema':SCHEMA+'_preparation','dataset':pc['datasets'][0],'baseline_jobs':pc['baseline_jobs'],
        'source_files':source,'design':design,'science_spec_sha256':SCIENCE_SHA,
        'legacy_contract_raw_sha256':PARENT_CONTRACT_SHA,'legacy_source_closure':pc['source']['files_sha256'],
        'implementation_revision':revision,'policy_files':{str(p.relative_to(ROOT)):sha(p) for p in policies},
        'parent_training_pins':parent_pins,'baseline_pins':baseline_pins,'new_fit_count':6,
        'fresh_same_runtime_baseline_count':0,'baseline_reference_count':3,'held_out_test_evaluated':False,
        'inherited_absolute_host_deadline_unix':start['deadline_unix'],'inherited_start_permit_sha256':PARENT_START_SHA}
    write(BUNDLE/'preparation/contract.json',prep)
    for src,dst in [(ROOT/'paper/scripts/titantpp_post_attention_ffn_raf6_runtime.py','runtime.py'),(PARENT/'operation/frozen_runtime.py','frozen_runtime.py')]:
        checked_copy(src,BUNDLE/'operation'/dst,sha(src))
    checked_copy(PARENT/'preparation/original_5090_qualification.json',BUNDLE/'preparation/original_5090_qualification.json',sha(PARENT/'preparation/original_5090_qualification.json'))
    c=copy.deepcopy(pc);remote='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/'+SCHEMA+'_5090';host=c['hosts']['5090']
    host.update(root=remote,source_root=remote+'/source',operation_root=remote+'/operation',output_dir=remote+'/run',tmux=SCHEMA+'_5090')
    host['environment'].update(PYTHONPATH=remote+'/source',MPLCONFIGDIR=remote+'/cache/matplotlib',XDG_CACHE_HOME=remote+'/cache',SOURCE_REVISION=revision)
    c.pop('stored_channel_reference_jobs',None);c.pop('stored_channel_reference_count',None)
    c['limits'].update(inherited_absolute_host_deadline_unix=start['deadline_unix'],inherited_start_permit_sha256=PARENT_START_SHA)
    c.update(schema=SCHEMA,arms=list(ARMS),seeds=list(SEEDS),jobs=[{'id':f'raf_spare_parts__{s}__{a}','host':'5090','dataset':'raf_spare_parts','seed':s,'arm':a} for a in ARMS for s in SEEDS],
        source={**pc['source'],'base_git_revision':revision,'frozen_parent_revision':pc['source']['base_git_revision'],'files':source,'files_sha256':canonical(source)},
        architecture=design,design_sha256=canonical(design),science_spec_sha256=SCIENCE_SHA,
        preparation_contract_sha256=sha(BUNDLE/'preparation/contract.json'),
        comparison_runtime_limitation='Current/Temporal candidates use5090/cuDNN92000; historical CNN3 uses5080/cuDNN91900. This runtime difference is not removed by retraining the baseline.',
        runner_modules={a:'paper.scripts.run_titantpp_post_attention_ffn' for a in ARMS},
        primary_contrast='Temporal−Current: same1152 added parameters/initialization/availability, explicit past-prefix content versus current-prefix nonlinear expansion. Common QKV/head remain trainable and memory retrieval may change; equal counts do not imply equal function spaces or causal isolation.',
        subsequent_evaluation={'authorized':True,'candidate_test_count':6,'validation_replay_count':6,'all_native_validation_required_before_any_test':True,'checkpoint_reselection':False})
    c['parallelism']['grouping']='arm_major_three_seeds_barrier'
    c['operation']={'files':{str(p.relative_to(BUNDLE)):sha(p) for p in (BUNDLE/'operation').glob('*.py')}}
    write(BUNDLE/'execution_contract.json',c)
    write(BUNDLE/'approval.json',{'approved':True,'hosts':['5090'],'contract_sha256':canonical(c),'user_instruction':USER_INSTRUCTION,'scope':APPROVAL_CONTEXT,
        'fresh_fit_count':6,'maximum_concurrent_workers':3,'new_training_authorized':True,'fixed_checkpoint_test_authorized':True,'automatic_retry':False,'loop_index':6})
    write(BUNDLE/'deployment_manifest.json',{'files':{str(p.relative_to(BUNDLE)):sha(p) for p in sorted(BUNDLE.rglob('*')) if p.is_file()},'contract_sha256':canonical(c)})
    write(REPORT/'LOCAL_READY.json',{'status':'prepared_native_checks_pending','contract_raw_sha256':sha(BUNDLE/'execution_contract.json'),
        'contract_canonical_sha256':canonical(c),'deployment_manifest_sha256':sha(BUNDLE/'deployment_manifest.json'),
        'source_files':len(source),'unchanged_inherited_source_files':145,'source_extension_files':list(EXTENSIONS),
        'design_manifest_sha256':DESIGN_MANIFEST_SHA,'native_context_length':pc['datasets'][0]['loader']['max_seq_len'],
        'scientific_fits_started':0,'held_out_test_evaluated':False,'inherited_absolute_host_deadline_unix':start['deadline_unix'],
        'frozen_scientific_fit_sha256':sha(BUNDLE/'operation/frozen_runtime.py')})
    print(json.dumps({'status':'prepared','bundle':str(BUNDLE),'contract_sha256':canonical(c)}))

def verify_test_receipt(path,required_sources):
    """Bind a successful local run to current code and preserved output bytes."""
    path=Path(path).resolve();require(path.is_relative_to(ROOT),'Test receipt must be preserved inside project')
    r=read(path)
    require(type(r.get('actual_exit')) is int and r['actual_exit']==0
        and type(r.get('failed')) is int and r['failed']==0
        and type(r.get('passed')) is int and r['passed']>0,'Local tests did not pass')
    files=r.get('files')
    require(isinstance(files,dict) and files and set(required_sources)<=set(files),'Missing tested source inventory')
    for key in ('stdout_path','stderr_path'):
        require(isinstance(r.get(key),str) and r[key] in files,'Missing preserved test '+key)
    require(r['stdout_path']!=r['stderr_path'],'Test output streams must remain separate')
    for name,digest in files.items():
        p=Path(name)
        require(not p.is_absolute() and '..' not in p.parts and (ROOT/p).resolve().is_relative_to(ROOT), 'Unsafe test evidence path')
        require(type(digest) is str and len(digest)==64 and sha(ROOT/p)==digest,'Test evidence drift '+name)
    return r

def seal(tests_receipt,model_tests_receipt):
    require(tests_receipt and model_tests_receipt,'Both real local test receipts are required')
    receipts=[Path(p).resolve() for p in (tests_receipt,model_tests_receipt)]
    required_ops=['paper/scripts/'+n for n in OPS]+['paper/tests/test_titantpp_post_attention_ffn_raf6_ops.py']
    required_model=[*EXTENSIONS,'paper/tests/test_titantpp_post_attention_ffn_model.py']
    verified_receipts=[verify_test_receipt(p,required) for p,required in zip(receipts,(required_ops,required_model))]
    require(not (BUNDLE/'followup_integrity.json').exists(),'Existing integrity; never reseal')
    c=read(BUNDLE/'execution_contract.json')
    for n,d in read(BUNDLE/'deployment_manifest.json')['files'].items():require(sha(BUNDLE/n)==d,'Bundle drift before seal '+n)
    for n in EXTENSIONS:require(sha(ROOT/n)==c['source']['files'][n],'Model/runner changed after preparation')
    policies=read(BUNDLE/'preparation/contract.json')['policy_files']
    for n,d in policies.items():require(sha(ROOT/n)==d,'Policy drift before seal '+n)
    paths=[p for p in BUNDLE.rglob('*') if p.is_file()]+[ROOT/'paper/scripts'/n for n in OPS]+[ROOT/n for n in EXTENSIONS]
    paths += [ROOT/n for record in verified_receipts for n in record['files']]
    paths += [ROOT/'paper/tests/test_titantpp_post_attention_ffn_raf6_ops.py',ROOT/'paper/tests/test_titantpp_post_attention_ffn_model.py',*receipts,REPORT/'LOCAL_READY.json',*[ROOT/n for n in policies]]
    write(BUNDLE/'followup_integrity.json',{'schema':SCHEMA+'_integrity','files':{str(p.relative_to(ROOT)):sha(p) for p in paths},
        'local_tests_receipt':str(receipts[0].relative_to(ROOT)),'local_tests_receipt_sha256':sha(receipts[0]),
        'model_tests_receipt':str(receipts[1].relative_to(ROOT)),'model_tests_receipt_sha256':sha(receipts[1]),
        'contract_raw_sha256':sha(BUNDLE/'execution_contract.json'),'source_closure':c['source']['files_sha256'],'held_out_test_evaluated':False})
    print(json.dumps({'status':'sealed','integrity_sha256':sha(BUNDLE/'followup_integrity.json')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare','seal'],required=True)
    p.add_argument('--tests-receipt');p.add_argument('--model-tests-receipt');a=p.parse_args()
    prepare() if a.mode=='prepare' else seal(a.tests_receipt,a.model_tests_receipt)
