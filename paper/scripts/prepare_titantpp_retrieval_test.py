"""Seal ten Validation-selected retrieval checkpoints for authorized Test inference."""
from pathlib import Path
import hashlib,json,math,shutil,time,datetime,subprocess
ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'search_artifacts/titantpp_contextual_retrieval_pilot_repair_20261007_v1'
OUT=ROOT/'search_artifacts/titantpp_retrieval_test_20261008_v1'
C2='bf1215ba8d6cfff14573cdcf9a98101846561238f1df5416c0a14249318cd9bf'
SOURCE='918f7cb883b62cbdc6f3f178cc53b6e20aac2e71a49b6e036769aa2fea76155a'
ARMS=tuple('titantpp_retrieval_'+x for x in ('mean','weighted','fixed','adaptive','generic'))
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def require(v,msg):
 if not v:raise ValueError(msg)
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def copy(src,dst,expected):
 require(sha(src)==expected,'Original SHA changed: '+str(src));dst.parent.mkdir(parents=True,exist_ok=True)
 with dst.open('xb') as f:f.write(src.read_bytes())
 require(sha(dst)==expected,'Copy SHA differs')
def main():
 require(not OUT.exists(),'Preserve existing Test campaign; no overwrite')
 c=read(OLD/'execution_contract.json');done=read(OLD/'analysis/completion_receipt.json')
 require(canonical(c)==C2==done['contract_sha256'],'Training contract changed')
 require(done['status']=='all10_Validation_originals_complete' and done['local_originals_SHA_verified'] and done['three_owner_roots_terminated'],'Training originals/termination incomplete')
 require(canonical(c['source']['files'])==SOURCE,'Frozen source closure changed')
 for name in ('Validation_summary.json','binary_originals_manifest.json'):require(sha(OLD/'analysis'/name)==done['outputs'][name],'Completion output SHA changed: '+name)
 rows=[]; originals=read(OLD/'analysis/binary_originals_manifest.json')['binary_originals'];selected_evidence={r['id']:r for r in read(OLD/'analysis/Validation_summary.json')['selected_metrics']}
 for host in ('5090','5080'):
  for arm in ARMS:
   matches=[r for r in originals if r['role']=='selected' and r['id']==f'{host}__yellow_trip_hourly__42__{arm}'];require(len(matches)==1,'Selected checkpoint not unique');r=matches[0]
   base=OLD/'retrieved'/r['owner']/'original';cp=base/r['path'];folder=base/'run'/r['id'];terminal=read(folder/'terminal_manifest.json');ep=read(cp.parent/'endpoint_replays.json');hist=read(cp.parent/'history.json')['history']
   require(terminal['status']=='complete' and terminal['scientific_success'],'Fit not complete')
   require(sha(folder/'terminal_manifest.json')==selected_evidence[r['id']]['terminal_manifest_sha256'],'Terminal evidence SHA changed')
   require([e['epoch'] for e in hist]==list(range(1,len(hist)+1)) and all(e['train_all_finite'] and math.isfinite(e['val_qty_rmse']) for e in hist),'History not finite/contiguous')
   for name in ('history.json','endpoint_replays.json'):require(sha(cp.parent/name)==terminal['files'][(cp.parent/name).relative_to(folder).as_posix()],'Selection evidence original SHA changed: '+name)
   best=min(hist,key=lambda e:e['val_qty_rmse']);require(best['epoch']==ep['best_epoch']==r['epoch'],'Selected earliest minimum changed')
   rel=cp.relative_to(folder).as_posix();require(terminal['files'][rel]==r['sha256']==sha(cp),'Selected original file changed')
   selected=ep['selected'];require(selected['count']==8268 and selected['tail']['count']==79 and selected['quantity_boundaries']==[7,686,1562,3449],'Validation population changed')
   row={'id':r['id'],'host':host,'arm':arm,'dataset':'yellow_trip_hourly','seed':42,'selected_epoch':r['epoch'],'selected_metric_value':best['val_qty_rmse'],'checkpoint_path':f'checkpoints/{r["id"]}.pt','checkpoint_sha256':r['sha256'],'state_sha256':selected['state_sha256'],'validation_reference':selected,'training_owner':r['owner'],'source_revision':c['source']['git_revision'],'training_terminal_manifest_sha256':sha(folder/'terminal_manifest.json')}
   rows.append(row)
 require(len(rows)==len({r['id'] for r in rows})==10,'Expected unique10')
 OUT.mkdir(parents=True)
 for name,digest in c['source']['files'].items():copy(OLD/'source'/name,OUT/'source'/name,digest)
 (OUT/'source/sample_data').mkdir(parents=True,exist_ok=True);(OUT/'source/sample_data/.frozen_root_sentinel').touch()
 spec=c['parent_contract']['datasets'][0];data=spec['inherited_data_identity'];src=OLD/'source'/data['data']['path']
 copy(src,OUT/'data/yellow_trip_hourly_with_split.parquet',data['data']['sha256']);copy(OLD/'source'/data['split_manifest']['path'],OUT/'data/yellow_trip_hourly_split_manifest.json',data['split_manifest']['sha256'])
 for row in rows:
  orig=next(r for r in originals if r['role']=='selected' and r['id']==row['id']);copy(OLD/'retrieved'/orig['owner']/'original'/orig['path'],OUT/row['checkpoint_path'],row['checkpoint_sha256'])
 metadata=read('/private/tmp/retrieval_test_populations.json');populations={k:metadata['populations'][k] for k in ('validation','test')}
 require(populations['validation']['target_count']==8268 and populations['test']['target_count']==8327,'Population count changed')
 require(metadata['frozen_contract_sha256']==C2 and metadata['source_closure_sha256']==SOURCE and metadata['data_sha256']==data['data']['sha256'] and metadata['loader']==spec['loader'],'Population metadata lineage changed')
 require(metadata['model_construction'] is False and metadata['model_inference'] is False and metadata['validation_sha_matches_original'] is True and metadata['train_sha_matches_original'] is True,'Population metadata unexpectedly evaluates models')
 for split in ('train','validation'):
  for key,value in data['populations'][split].items():require(metadata['populations'][split][key]==value,'Inherited population identity changed: '+key)
  require(metadata['populations'][split]['target_dt_sha256']==spec['time_statistics']['expected_'+split+'_target_dt_sha256'],'Population duration identity changed')
 copy(ROOT/'paper/scripts/evaluate_titantpp_retrieval_test.py',OUT/'operation/evaluate.py',sha(ROOT/'paper/scripts/evaluate_titantpp_retrieval_test.py'))
 preflight=read('/private/tmp/retrieval_test_preflight.json')
 hosts={}
 for host,h in c['hosts'].items():
  require(preflight[host]['gpu_processes']=='' and preflight[host]['runtime']==h['runtime_expected'] and h['gpu_uuid'] in preflight[host]['gpu'],'Native runtime/idle preflight differs')
  target='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_retrieval_test_20261008_v1_'+host
  env={**h['environment'],'PYTHONPATH':target+'/source','XDG_CACHE_HOME':target+'/cache','MPLCONFIGDIR':target+'/cache/matplotlib'}
  hosts[host]={'root':target,'python':h['python'],'gpu_uuid':h['gpu_uuid'],'runtime_expected':h['runtime_expected'],'environment':env}
 approval={'test_inference_authorized':True,'user_request':'Test까지 뽑아봐','approved_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'conditions':10,'datasets':['yellow_trip_hourly'],'seeds':[42],'arms':list(ARMS),'new_training':False,'checkpoint_reselection':False,'raw_predictions_written':False,'automatic_retry':False,'new_rental':False,'shared_runtime_changed':False}
 contract={'schema':'titantpp_retrieval_frozen_test_20261008_v1','operational_git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'approval':approval,'training_contract_sha256':C2,'source_root':'source','source_files':c['source']['files'],'source_closure_sha256':SOURCE,'source_revision':c['source']['git_revision'],'data_spec':spec,'data':{'path':'data/yellow_trip_hourly_with_split.parquet','sha256':data['data']['sha256'],'split_manifest_path':'data/yellow_trip_hourly_split_manifest.json','split_manifest_sha256':data['split_manifest']['sha256']},'population_references':populations,'hosts':hosts,'resources':{'threads':4,'host_split_timeout_seconds':900,'rss_limit_bytes':17179869184,'gpu_memory_limit_bytes':8589934592},'rows':rows,'evaluator_sha256':sha(OUT/'operation/evaluate.py'),'test_exposure_lineage':{'dataset_Test_previously_accessed':True,'these10_checkpoint_Test_previously_evaluated':False,'independent_untouched_Test':False},'evidence':{'training_current_sha256':sha(OLD/'current.json'),'training_completion_sha256':sha(OLD/'analysis/completion_receipt.json'),'Validation_summary_sha256':sha(OLD/'analysis/Validation_summary.json'),'binary_originals_manifest_sha256':sha(OLD/'analysis/binary_originals_manifest.json'),'population_references_derivation':'frozen loader metadata only; no model prediction','preflight_path':'evidence/native_preflight.json'}}
 write(OUT/'approval.json',approval);write(OUT/'evidence/native_preflight.json',preflight);write(OUT/'evidence/populations_metadata.json',metadata);write(OUT/'execution_contract.json',contract)
 for src,dst in ((OLD/'analysis/completion_receipt.json',OUT/'evidence/Validation_completion.json'),(OLD/'analysis/Validation_summary.json',OUT/'evidence/Validation_summary.json')):copy(src,dst,sha(src))
 files={p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
 write(OUT/'deployment_manifest.json',{'files':files,'bytes':sum((OUT/p).stat().st_size for p in files),'contract_sha256':sha(OUT/'execution_contract.json'),'source_closure_sha256':SOURCE})
 print(json.dumps({'status':'sealed','conditions':10,'contract_sha256':sha(OUT/'execution_contract.json'),'bytes':sum((OUT/p).stat().st_size for p in files)}))
if __name__=='__main__':main()
