"""Seal the approved RAF9 x Train/Validation18 inference-only inputs."""
from pathlib import Path
import copy
import hashlib
import json
import shutil
import time

PROJECT=Path('/Users/igwanhyeong/PycharmProjects/paper_research')
PARENT=PROJECT/'search_artifacts/titantpp_history_train_fit_20261012_v1'
VALIDATION=PROJECT/'search_artifacts/titantpp_history_diagnostics_20261012_v2'
ROOT=PROJECT/'search_artifacts/titantpp_raf_log_raw_20261012_v1'
REPORT=PROJECT/'reports/titantpp_raf_log_raw_20261012_v1'
PARENT_CANONICAL='76248e2174d7bd651d821f649ad860030da985d59da88a8ce50725b476f55ccf'
PARENT_RAW='db4096cae1c1889eb1f7f7b248acc21a66f36dc1e8e023acac76a68507b6f342'
VALIDATION_CANONICAL='70e8e22dbe85aa93cb5dc555b06303511b0e5e7f511771b455b4b243ddb3ad1d'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1048576),b''):h.update(data)
    return h.hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def main():
    assert sha(PARENT/'contract.json')==PARENT_RAW
    parent=json.loads((PARENT/'contract.json').read_bytes())
    assert canonical({k:v for k,v in parent.items() if k!='contract_sha256'})==PARENT_CANONICAL==parent['contract_sha256']
    assert sha(PARENT/'originals/manifest.json')=='705e863e6cbcbdab71a6f6ec105d747318ee7ce3146ddc55d6070e59869e3600'
    assert sha(PARENT/'analysis/completion_receipt.json')=='8fadf88dd1983f74dbebc23118bcade67893a26800c4dac0af6f26e6b2f6f9e2'
    assert json.loads((PARENT/'originals/LOCAL_VERIFICATION.json').read_bytes())['status']=='verified'
    assert not ROOT.exists(), 'No preparation overwrite/retry'
    jobs=[j for j in parent['jobs'] if j['dataset']=='raf_spare_parts']
    assert len(jobs)==9 and len({j['id'] for j in jobs})==9
    ids={j['id'] for j in jobs}
    inputs=json.loads((PARENT/'input_integrity.json').read_bytes())['files']
    selected={}
    for rel,pin in inputs.items():
        parts=Path(rel).parts
        include=rel.startswith('sources/history/') or rel.startswith('provenance/history_')
        include|=rel=='data/full/raf_spare_parts.parquet'
        include|=rel.startswith('checkpoints/') and Path(rel).stem in ids
        include|=rel.startswith('provenance/checkpoint_metadata/') and Path(rel).stem in ids
        include|=len(parts)>2 and parts[:2]==('provenance','jobs') and parts[2] in ids
        if include:
            src,dest=PARENT/rel,ROOT/rel
            assert src.is_file() and not src.is_symlink() and sha(src)==pin,rel
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
            assert sha(dest)==pin;selected[rel]=pin
    references={}
    for j in jobs:
        references[j['id']]={}
        for split,base,name,parent_sha in [('train',PARENT,'train_metrics.json',PARENT_CANONICAL),
                ('validation',VALIDATION,'validation_replay.json',VALIDATION_CANONICAL)]:
            src=base/'results'/j['id']/name
            ref=json.loads(src.read_bytes())
            assert ref['status']=='complete' and ref['split']==split and ref['contract_sha256']==parent_sha
            assert ref['checkpoint_sha256']==j['checkpoint']['sha256'] and ref['state_sha256']==ref['state_after_sha256']==j['checkpoint_state_sha256']
            rel=f'reference/native_results/{j["id"]}/{split}_metrics.json'
            dest=ROOT/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
            pin=sha(src);assert sha(dest)==pin;selected[rel]=pin
            references[j['id']][split]={'relative_path':rel,'sha256':pin,'contract_sha256':parent_sha}
    c=copy.deepcopy(parent);c.pop('contract_sha256')
    for key in ('train_fit','reused_diagnostics','diagnostics'):c.pop(key,None)
    c.update(schema='titantpp_raf_log_raw_20261012_v1',created_unix=time.time(),jobs=jobs,
        datasets={'raf_spare_parts':parent['datasets']['raf_spare_parts']},
        parent_contract_sha256=PARENT_CANONICAL,parent_contract_raw_sha256=PARENT_RAW,
        evaluation_scope='frozen_RAF_Train_Validation_actual_log_raw_error_diagnostics',
        instrumentation_revision='read-only actual per-target log quantity loss')
    c['authorization']={'human_request':'RAF18부터 이어서 Intermittent18, 새 학습 없이 Train·Validation만 평가하자',
        'training':False,'test_inference':False,'test_reselection':False,'new_dataset':False,
        'checkpoint_or_hyperparameter_reselection':False,'model_forward_changes':False,
        'encoder_path_interventions':False,'shared_Runtime_or_auth_changes':False,'manuscript_edit':False}
    c['execution'].update(remote_root='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_raf_log_raw_20261012_v1_5080',
        gate='native_synthetic3_before_18_Train_Validation_workers',expected_worker_count=18,expected_model_split_passes=18)
    spec=c['datasets']['raf_spare_parts']
    c['h1']={'evaluation_splits':['train','validation'],'new_fits':0,'test_inference':0,
        'expected_populations':{'raf_spare_parts':{s:spec['definition']['inherited_data_identity']['populations'][s] for s in ('train','validation')}},
        'reference_receipts':references,'actual_per_target_loss':True,'raw_prediction_rows_written':False,
        'same_native_tolerance':{'relative':1e-5,'absolute':1e-5},'raw_tail_threshold':200,
        'train_Validation_only':True,'log_error_from_mean_raw_prediction':False}
    c['contract_sha256']=canonical(c)
    write(ROOT/'contract.json',c);selected['contract.json']=sha(ROOT/'contract.json')
    write(ROOT/'input_integrity.json',{'schema':'frozen_raf_log_raw_inputs_v1','contract_sha256':c['contract_sha256'],'files':selected})
    write(REPORT/'APPROVAL.json',{'authorized':True,'human_request':c['authorization']['human_request'],
        'contract_sha256':c['contract_sha256'],'scope':c['authorization'],'model_split_passes':18,
        'actual_planned_workers':18,'new_fits':0,'new_Test_inference':0,'observed_unix':time.time()})
    print(json.dumps({'status':'prepared','canonical':c['contract_sha256'],'input_files':len(selected),
        'checkpoint_count':9,'model_split_passes':18,'actual_worker_count':18}))

if __name__=='__main__':main()
