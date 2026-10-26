#!/usr/bin/env python3
"""Read-only evidence checks; writes a NEW final inventory only after all QA passes.

No network, model construction, tensor arithmetic, training or Test data access.
The v4 failed dispatcher remains failed; only its independently verified child is reused.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
COUNTS = {"train":393824, "validation":86285}
EPOCHS = {(42,"best"):1, (42,"last"):41, (52,"best"):5, (52,"last"):45}
EXPECTED = {(seed,kind,split):COUNTS[split] for seed,kind in EPOCHS for split in COUNTS}
GATES_V6 = {"pred_quantity":{"atol":1e-5,"rtol":1e-5}, "time_nll":{"atol":1e-4,"rtol":1e-4},
            "log_qty_mse":{"atol":1e-7,"rtol":1e-5}, "topk_similarity":{"atol":1e-6,"rtol":1e-5}}
GATES_V8 = {**GATES_V6,"log_qty_mse":{"atol":1e-6,"rtol":1e-5}}
NEW_RUNS = [('v6','5080',480109,GATES_V6),('v8','5090',873933,GATES_V8)]
SEEN = {}


def require(ok, message):
    if not ok: raise AssertionError(message)


def raw_record(path):
    p=Path(path).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return {"path":str(p),"bytes":p.stat().st_size,"sha256":h.hexdigest()}


def record(path):
    result=raw_record(path);prior=SEEN.setdefault(result['path'],result)
    require(result==prior,"File changed during inventory verification: "+result['path'])
    return result


def read(path):
    record(path);return json.loads(Path(path).read_text())


def verify(ref, path=None):
    actual=record(path or ref['path'])
    require(all(actual[k]==ref[k] for k in ('bytes','sha256')),"File binding differs: "+actual['path'])
    return actual


def local_remote(ref, originals, remote_root):
    remote=Path(ref['path']);rel=remote.relative_to(remote_root)
    require('..' not in rel.parts,"Unsafe remote path")
    return verify(ref,originals/rel)


def check_qa(path, required_status='PASS'):
    qa=read(path);require(qa['status']==required_status,"Independent QA not passed: "+str(path))
    for rel,digest in qa.get('evidence',{}).items():
        p=Path(rel) if Path(rel).is_absolute() else ROOT/rel
        require(record(p)['sha256']==digest,"QA evidence SHA differs: "+str(p))
    return qa


def check_identity(frames, metadata, split):
    data=pd.concat(frames,ignore_index=True)
    expected=metadata.loc[metadata.split==split].sort_values('target_index').reset_index(drop=True)
    require(len(data)==COUNTS[split],"Wrong cohort row count")
    require(not data.target_index.duplicated().any(),"Duplicate target index")
    require(not data[['entity_id','target_seq']].duplicated().any(),"Duplicate target key")
    data=data.sort_values('target_index').reset_index(drop=True)
    for field in ['target_index','entity_id','target_seq','history_length','raw_quantity','target_gap']:
        require(np.array_equal(data[field].to_numpy(),expected[field].to_numpy()),"Metadata mismatch: "+field)
    require((data.split==split).all(),"Mixed split in cohort")
    return len(data)


def archive_members(originals, manifest_name):
    path=originals/manifest_name;manifest=read(path);members=manifest['files']
    require(bool(members),"Empty archive member manifest")
    expected=set(members)|{manifest_name}
    actual={str(p.relative_to(originals)) for p in originals.rglob('*') if p.is_file()}
    require(actual==expected,"Archive member inventory differs")
    for rel,ref in members.items():
        require(not Path(rel).is_absolute() and '..' not in Path(rel).parts,"Unsafe archive member")
        verify(ref,originals/rel)
    return record(path)


def cohort_from_job(job, originals, remote_root, materials, metadata, source_label, outer_exit, qa_path, expected_spec_sha):
    job_id=job['id'];d=originals/'output_full'/job_id
    child=read(originals/'output_full'/(job_id+'.ACTUAL_EXIT.json'))
    require(child['actual_child_exit']==0,"Selected scientific child failed")
    require(child['spec_sha256']==expected_spec_sha,"Child spec binding differs")
    for stream in ('stdout','stderr'):local_remote(child[stream],originals,remote_root)
    result=read(d/'COMPLETE.json')
    require(result['runner_sha256']==child['collector_sha256'],"Child runner binding differs")
    native=next(j for j in materials['jobs'] if j['id']==job_id)
    require(result['status']=='complete',"Job output not complete")
    require(result['seed']==native['seed'] and result['checkpoint_kind']==native['checkpoint_kind'],"Wrong checkpoint identity")
    require(result['epoch']==native['expected_epoch']==EPOCHS[(native['seed'],native['checkpoint_kind'])],"Wrong checkpoint epoch")
    require(result['checkpoint_sha256']==native['checkpoint_sha256'],"Wrong binary checkpoint")
    require(result['canonical_checkpoint_state_sha256']==native['expected_model_state_sha256'],"Wrong tensor state")
    require(result['parameter_buffer_sha_before']==result['parameter_buffer_sha_after'],"State changed")
    for field in ['context_state_unchanged','input_and_model_immutability_verified','parameter_grad_fields_all_none']:
        require(result[field] is True,"State safeguard failed: "+field)
    for field in ['held_out_materialized','new_training_epochs','optimizer_steps']:
        require(result[field] in (False,0),"Unadmitted operation: "+field)
    require(result['precision']['mu_sigma']=='native_float64_lognormal_time_terms_return' and
            result['precision']['time_nll']=='native_float64_recorded_bin_mass',"Native precision differs")
    require(set(result['splits'])==set(job['splits']),"Unexpected split inventory")
    cohorts=[]
    for split,v in result['splits'].items():
        manifest=read(d/(split+'_manifest.json'));require(v==manifest,"Job and split manifests differ")
        require(v['rows']==v['population_rows']==COUNTS[split],"Wrong manifest count")
        require(v['head_hidden_saved']==(native['checkpoint_kind']=='best' and split=='train'),"Unexpected hidden storage scope")
        frames=[];chunks=[];offset=0;hidden_rows=0
        for c in v['chunks']:
            require(c['row_offset']==offset,"Noncontiguous chunk row offset")
            obs=local_remote(c['observations'],originals,remote_root)
            require(pq.ParquetFile(obs['path']).metadata.num_rows==c['rows'],"Chunk physical row count differs")
            frames.append(pd.read_parquet(obs['path'],columns=['target_index','entity_id','target_seq','history_length','raw_quantity','target_gap','split']))
            entry={'chunk':c['chunk'],'rows':c['rows'],'observations':obs}
            if v['head_hidden_saved']:
                entry['head_hidden']=local_remote(c['head_hidden'],originals,remote_root)
                with np.load(entry['head_hidden']['path'],allow_pickle=False) as z:
                    require(z['head_hidden'].shape==(c['rows'],64) and z['head_hidden'].dtype==np.float32,"Wrong hidden shape or precision")
                    require(np.array_equal(z['target_index'],frames[-1].target_index.to_numpy()),"Hidden and observation target order differ")
                hidden_rows+=c['rows']
            chunks.append(entry);offset+=c['rows']
        require(offset==v['rows'],"Chunk counts do not cover cohort")
        require(check_identity(frames,metadata,split)==v['rows'],"Target coverage differs")
        local_remote(v['progress'],originals,remote_root)
        cohorts.append({'seed':native['seed'],'checkpoint':native['checkpoint_kind'],'epoch':result['epoch'],
            'split':split,'rows':v['rows'],'source':source_label,'local_path':str(d),'manifest':record(d/(split+'_manifest.json')),
            'job_complete':record(d/'COMPLETE.json'),'child_exit':record(originals/'output_full'/(job_id+'.ACTUAL_EXIT.json')),
            'actual_child_exit':0,'outer_dispatcher_exit':outer_exit,'checkpoint_sha256':result['checkpoint_sha256'],
            'model_state_unchanged':True,'native_log_mse_available':True,'hidden_rows':hidden_rows,
            'precision':result['precision'],'chunks':chunks,'qa':record(qa_path),
            'target_identity_mismatches':0,'target_duplicates':0})
    return cohorts


def required_files():
    paths=[ROOT/('POSTRUN_FULL_'+v.upper()+'_'+h+'_QA.json') for v,h,_,_ in NEW_RUNS]
    paths += [ROOT/('execution_'+v)/'retrieval'/h/'full/COLLECTED.json' for v,h,_,_ in NEW_RUNS]
    return paths


def build():
    missing=[str(p) for p in required_files() if not p.is_file()]
    require(not missing,"Final gate unavailable: "+', '.join(missing))
    partial=read(ROOT/'PARTIAL_COLLECTION.json');require(partial['complete'] is False and partial['verified_rows']==566394,"Partial history changed")
    materials=read(ROOT/'MATERIALS.json')
    meta_path=ROOT/'support/results_v1/target_metadata.parquet'
    require(record(meta_path)['sha256']=='dcee4a92d729a1937b1f47e64b317e49e4043adf6dfce882fa0ace8398b5edb8',"Support metadata changed")
    metadata=pd.read_parquet(meta_path)
    check_qa(ROOT/'POSTRUN_SUPPORT_QA.json')
    reuse_qa=check_qa(ROOT/'POSTRUN_REUSE_QA.json')
    reuse=read(ROOT/'reuse/results_v1/PROVENANCE.json');cohorts=[]
    require(reuse['status']=='complete' and reuse_qa['source_run_actual_exit']==reuse_qa['qa_actual_exit']==0,"Reuse failed")
    require(reuse['source_pre']==reuse['source_post'],"Original reuse sources changed during conversion")
    for ref in reuse['source_pre']:verify(ref)
    for j in reuse['jobs']:
        ref=verify(j['output']);df=pd.read_parquet(ref['path'])
        require(df.log_qty_mse.isna().all() and not df.head_hidden_available.any(),"Missing native values invented")
        require((df.mu_sigma_precision=='exported_float32').all() and (df.time_nll_precision=='original_float64').all(),"Reuse precision relabelled")
        df=df.rename(columns={'true_recorded_gap':'target_gap'})
        rows=check_identity([df],metadata,'validation')
        cohorts.append({'seed':j['seed'],'checkpoint':'best','epoch':j['selected_epoch'],'split':'validation','rows':rows,
            'source':'historical_validation_reuse','parquet':ref,'native_log_mse_available':False,'hidden_rows':0,
            'precision':{'mu_sigma':'exported_float32_original','time_nll':'original_float64'},
            'missing_native_fields':['log_qty_mse','quantity_train_loss','head_preactivation','head_hidden'],
            'checkpoint_sha256':j['checkpoint_sha256'],'model_state_unchanged':True,'actual_child_exit':0,
            'qa':record(ROOT/'POSTRUN_REUSE_QA.json'),'target_identity_mismatches':0,'target_duplicates':0})
    old_qa_path=ROOT/'POSTRUN_FULL_V4_FAILURE_QA.json'
    old_qa=check_qa(old_qa_path,'FAIL_WHOLE_COLLECTION_PARTIAL_ARTIFACT_PASS')
    require(old_qa['actual_collector_exit']==1 and old_qa['partial_saved_arithmetic']['status']=='PARTIAL_ARTIFACT_PASS_PARENT_FAILED',"Old dispatcher failure was erased")
    old=ROOT/'failed_full_v4_5080';tr=read(old/'TRANSPORT.json');require(tr['actual_exit']==0,"Failure archive transfer failed")
    verify(tr['stdout']);verify(tr['stderr']);archive_members(old/'originals','FAILURE_MEMBERS.json')
    old_guard=read(ROOT/'execution_v4/5080_GUARD.json')
    native=next(j for j in materials['jobs'] if j['seed']==42 and j['checkpoint_kind']=='best')
    cohorts+=cohort_from_job({**native,'splits':['train']},old/'originals',Path(old_guard['remote_root']),materials,metadata,
                           'v4_completed_child_of_failed_dispatcher',1,old_qa_path,old_qa['partial_saved_arithmetic']['spec_sha256'])
    approval=read(ROOT/'execution_v6/NUMERIC_GATE_AUTHORIZATION_V6.json')
    require(approval['approved'] is True and approval['field']=='topk_similarity.atol' and approval['previous']==1e-7
            and approval['approved_atol']==1e-6 and approval['rtol']==1e-5,"Unadmitted numeric gate change")
    verify(approval['original_approval'])
    log_approval=read(ROOT/'LOG_MSE_GATE_AUTHORIZATION_V8.json')
    require(log_approval['approved'] is True and log_approval['field']=='log_qty_mse.atol' and log_approval['previous']==1e-7
            and log_approval['approved_atol']==1e-6 and log_approval['rtol']==1e-5,"Unadmitted log-MSE gate change")
    verify(log_approval['original_approval'])
    failed_v6=check_qa(ROOT/'POSTRUN_GPU5090_QUALIFICATION_V6_FAILURE_QA.json','FAIL')
    require(failed_v6['actual_collector_exit']==1 and failed_v6['failure']['field']=='log_qty_mse',"Prior v6 failure was erased")
    for rel,ref in failed_v6['failure_snapshot_inventory'].items():
        verify(ref,ROOT/'failed_gpu_qualification_v6/5090'/rel)
    new_rows=0
    for version,host,expected_rows,gates in NEW_RUNS:
        qa_path=ROOT/('POSTRUN_FULL_'+version.upper()+'_'+host+'_QA.json');qa=check_qa(qa_path)
        require(qa['actual_collector_exit']==0 and qa['post_guard_passed'] is True and qa['inference_by_qa']==0,"New run QA failed")
        require(qa['saved_rows_checked']==expected_rows,"Wrong QA scope count")
        e=ROOT/('execution_'+version);spec_path=e/(host+'_FULL_SPEC_'+version+'.json');spec=read(spec_path)
        require(spec['comparison_tolerances']==gates,"A non-approved gate changed")
        guard=read(e/(host+'_GUARD.json'));remote=Path(guard['remote_root'])
        target=e/'retrieval'/host/'full';originals=target/'originals'
        collected=read(target/'COLLECTED.json');require(collected['actual_transport_exit']==0,"Collection transport failed")
        verify(collected['archive']);verify(collected['manifest'])
        transfer=read(target/'TRANSFER.receipt.json');require(transfer['actual_exit']==0 and transfer.get('failure') is None,"Transfer not clean")
        verify(transfer['archive']);verify(transfer['stderr']);archive_members(originals,'full_RESULT_MEMBERS.json')
        parent=read(originals/'control/full.ACTUAL_EXIT.json')
        require(parent['actual_collector_exit']==0 and parent['post_guard_passed'] is True,"Dispatcher failed")
        require(parent['spec']['sha256']==record(spec_path)['sha256'],"Dispatcher spec differs")
        for stream in ('stdout','stderr'):local_remote(parent[stream],originals,remote)
        complete=read(originals/'output_full/COMPLETE.json')
        require(complete['actual_child_exits_all_zero'] and complete['spec_sha256']==record(spec_path)['sha256'],"Whole run incomplete")
        require(complete['collector_sha256']==spec['runner_sha256'] and len(complete['receipts'])==len(spec['jobs']),"Dispatcher child inventory differs")
        for ref in complete['receipts']:local_remote(ref,originals,remote)
        current=[]
        for job in spec['jobs']:current+=cohort_from_job(job,originals,remote,materials,metadata,'execution_'+version,0,qa_path,record(spec_path)['sha256'])
        require(sum(c['rows'] for c in current)==expected_rows,"Wrong new host rows")
        require({(r['job'],r['split'],r['rows']) for r in qa['records']}==
                {(j['id'],split,COUNTS[split]) for j in spec['jobs'] for split in j['splits']},"QA cohort coverage differs")
        cohorts+=current;new_rows+=expected_rows
    observed={(c['seed'],c['checkpoint'],c['split']):c['rows'] for c in cohorts}
    require(len(cohorts)==8 and observed==EXPECTED,"Final cohort inventory is missing or duplicated")
    require(new_rows==1354042 and sum(c['rows'] for c in cohorts)==1920436,"Final count differs")
    observed_files=[];hidden_files=[]
    for c in cohorts:
        if 'parquet' in c:observed_files.append(c['parquet'])
        for chunk in c.get('chunks',[]):
            observed_files.append(chunk['observations'])
            if 'head_hidden' in chunk:hidden_files.append(chunk['head_hidden'])
    all_data_files=observed_files+hidden_files
    require(len({r['path'] for r in all_data_files})==len(all_data_files),"Repeated physical data file in final cohorts")
    hidden_rows=sum(c['hidden_rows'] for c in cohorts)
    require(hidden_rows==787648,"Wrong best Train hidden coverage")
    storage={'observation_parquet_files':len(observed_files),'observation_parquet_bytes':sum(r['bytes'] for r in observed_files),
             'head_hidden_npz_files':len(hidden_files),'head_hidden_npz_bytes':sum(r['bytes'] for r in hidden_files),
             'head_hidden_rows':hidden_rows,'combined_bytes':sum(r['bytes'] for r in all_data_files),
             'scope':'Unique observation Parquet plus best Train head_hidden NPZ only; excludes archives, sources, logs and manifests.',
             'native_log_mse_missing_rows':172570,'native_log_mse_available_rows':1747866}
    require(all(raw_record(p)==ref for p,ref in SEEN.items()),"A source changed during final verification")
    return {'schema':'phen_full_observation_final_collection_v1','status':'collection_complete_independently_verified',
        'complete':True,'verified_rows':1920436,'reused_historical_validation_rows':172570,
        'verified_old_completed_child_rows':393824,'new_execution_v6_5080_rows':480109,
        'new_execution_v8_5090_rows':873933,'new_followup_rows':1354042,'remaining_rows':0,
        'cohorts':sorted(cohorts,key=lambda c:(c['seed'],c['checkpoint'],c['split'])),
        'partial_collection_preserved':record(ROOT/'PARTIAL_COLLECTION.json'),'old_dispatcher_status':'FAILED_PRESERVED',
        'numeric_gate_authorization':record(ROOT/'execution_v6/NUMERIC_GATE_AUTHORIZATION_V6.json'),
        'log_mse_gate_authorization':record(ROOT/'LOG_MSE_GATE_AUTHORIZATION_V8.json'),
        'v6_5090_qualification_status':'FAILED_PRESERVED','v6_5090_failure_qa':record(ROOT/'POSTRUN_GPU5090_QUALIFICATION_V6_FAILURE_QA.json'),
        'v7_observations_included':0,
        'data_storage':storage,
        'new_inference_by_inventory_builder':0,'training_epochs':0,'optimizer_steps':0,'test_materialized':False,
        'model_improvement_or_causal_conclusion_claimed':False,'source_bindings_unchanged':list(SEEN.values()),
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}


def render_summary(result):
    require(len(result['cohorts'])==8,'Summary requires all eight cohorts')
    text='# 전체 관측 회수·검증 완료\n\n총 1,920,436건을 확인했다. 기존 best Validation 172,570건, 실패한 v4 dispatcher 안에서 정상 종료·독립 검산된 seed42 best Train 393,824건, 승인된 별도 기준의 v6 5080 관측 480,109건과 v8 5090 관측 873,933건이다. v4 전체 실행 및 v6 5090 qualification의 실패 판정과 PARTIAL_COLLECTION은 보존한다. v7 관측은 포함하지 않는다.\n\n| seed | checkpoint | epoch | split | rows | source |\n| --- | --- | ---: | --- | ---: | --- |\n'
    for c in result['cohorts']:text+=f"| {c['seed']} | {c['checkpoint']} | {c['epoch']} | {c['split']} | {c['rows']:,} | {c['source']} |\n"
    text+='\n재사용 Validation의 μ·σ는 float32 export, NLL은 원래 float64다. 신규 기록의 μ·σ·NLL은 native float64다. 재사용 자료에 없는 native log-MSE·학습 손실·head preactivation·hidden은 채우지 않았다. 전체 회수 완료는 Prototype의 추가 설명력이나 시간 손실 악화 원인이 확인됐다는 뜻이 아니다. 정량 해석은 다음 분석 단계에 남긴다.\n'
    s=result['data_storage']
    text+=f"\n관측 Parquet {s['observation_parquet_files']}개 {s['observation_parquet_bytes']:,} bytes와 best Train hidden NPZ {s['head_hidden_npz_files']}개 {s['head_hidden_npz_bytes']:,} bytes를 합하면 **{s['combined_bytes']:,} bytes ({s['combined_bytes']/1048576:.2f} MiB)**다. hidden은 seed42·52 best Train 각393,824건, 합계 {s['head_hidden_rows']:,}건이다. archive·소스 사본·로그·manifest 중복은 이 용량에 포함하지 않는다.\n"
    return text


def verify_saved_final(result):
    destination=ROOT/'FINAL_COLLECTION.json';summary=ROOT/'FINAL_COLLECTION_SUMMARY.md'
    require(destination.exists()==summary.exists(),'Final inventory/summary pair is incomplete')
    if not destination.exists():return False
    saved=json.loads(destination.read_text())
    require('created_utc' in saved and isinstance(saved['created_utc'],str),'Saved creation timestamp missing')
    datetime.datetime.fromisoformat(saved['created_utc'])
    actual={k:v for k,v in saved.items() if k!='created_utc'}
    expected={k:v for k,v in result.items() if k!='created_utc'}
    require(actual==expected,'Stored FINAL_COLLECTION differs from complete source recalculation')
    require(summary.read_text()==render_summary(result),'Stored final summary differs from eight-cohort inventory')
    return True


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['readiness','check-only','write'],default='readiness');a=ap.parse_args()
    if a.mode=='readiness':
        missing=[str(p) for p in required_files() if not p.is_file()]
        print(json.dumps({'status':'pending' if missing else 'ready_for_verification','missing':missing,'complete_claimed':False},indent=2));return
    result=build()
    saved_verified=verify_saved_final(result)
    if a.mode=='write':
        destination=ROOT/'FINAL_COLLECTION.json';summary=ROOT/'FINAL_COLLECTION_SUMMARY.md'
        require(not destination.exists() and not summary.exists(),'Final inventory already exists; never overwrite')
        with destination.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
        with summary.open('x') as f:f.write(render_summary(result))
        saved_verified=verify_saved_final(result)
    print(json.dumps({'status':'verified','mode':a.mode,'rows':result['verified_rows'],'cohorts':len(result['cohorts']),
                      'old_dispatcher_status':result['old_dispatcher_status'],'source_files_checked':len(result['source_bindings_unchanged']),
                      'stored_final_and_summary_match_recalculation':saved_verified,'data_storage':result['data_storage']}))


if __name__=='__main__':main()
