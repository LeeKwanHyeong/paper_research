"""Bounded CPU validation samples through original frozen models and loader.

This preparation entry point deliberately has no test-split or GPU mode.
"""
import argparse,hashlib,json,os,platform,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run_bundle(name,limit):
 registry=read(OUT/'evaluation_registry.json');bundle=registry['bundles'][name]
 work=ROOT/bundle['source_root']
 for rel,h in bundle['source_files'].items():assert sha(work/rel)==h
 os.environ['CUDA_VISIBLE_DEVICES']=''
 os.environ['MPLCONFIGDIR']=str(OUT/'runtime_cache/matplotlib')
 os.environ['XDG_CACHE_HOME']=str(OUT/'runtime_cache')
 sys.path.insert(0,str(work))
 import numpy as np,polars as pl,torch
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 from models.TPPs.CountAwareFactory import build_count_aware_model,validate_checkpoint_route
 from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame,target_outputs
 from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
 from data_loader.event_seq_data_module import collate_week_lookback
 from simple_lab_test.search.common.runner import canonical_state_dict_sha256
 identities=read(ROOT/'reports/titantpp_independent_evaluation_protocol_20261001_v1/dataset_manifest.json')
 paths={d['dataset']:ROOT/d['identities'].get('deployed_data',d['identities']['data'])['path'] for d in identities['datasets']}
 paths_sha={d['dataset']:d['identities'].get('deployed_data',d['identities']['data'])['sha256'] for d in identities['datasets']}
 rows=[r for r in registry['rows'] if r['evaluator_source_bundle']==name]
 specs={d['dataset_id']:d for d in bundle['datasets']}
 cached={};records=[];checks=[];started=time.monotonic()
 for r in rows:
  ds=r['dataset'];data=specs[ds]
  if ds not in cached:
   path=paths[ds];assert sha(path)==paths_sha[ds]
   eligible=pl.scan_parquet(path).filter(pl.col('chronological_split')=='validation').select('oper_part_no').unique().sort('oper_part_no').limit(2).collect()['oper_part_no'].to_list()
   frame=pl.scan_parquet(path).filter(pl.col('chronological_split').is_in(['train','validation']) & pl.col('oper_part_no').is_in(eligible)).collect().sort(['oper_part_no','seq'])
   assert set(frame['chronological_split'].unique())<= {'train','validation'}
   assert frame.select(pl.struct(['oper_part_no','seq']).is_duplicated().any()).item() is False
   frame=prepare_count_frame(frame)
   loader=make_loader(frame,target_split='validation',shuffle=False,generator=None,**{k:data['loader'][k] for k in ['batch_size','lookback_weeks','max_seq_len']})
   dataset=loader.dataset;n=min(limit,len(dataset));assert n>0
   batch=collate_week_lookback([dataset[i] for i in range(n)])
   meta=[]
   for idx in range(n):
    pi,end=dataset.index[idx];entity=str(dataset.parts[pi]);seq=int(dataset.seq_lists[pi][end+1]);assert dataset.split_lists[pi][end+1]=='validation'
    row=frame.filter((pl.col('oper_part_no')==entity)&(pl.col('seq')==seq)).row(0,named=True)
    meta.append({'dataset':ds,'entity_id':entity,'seq':seq,'split':'validation','target_id':hashlib.sha256(json.dumps([ds,entity,seq,'validation'],separators=(',',':')).encode()).hexdigest(),'time_bucket':str(row['time_bucket']) if 'time_bucket'in row else None,'site_id':entity.split('::',1)[0] if ds=='intermittent_frozen_5000' else None})
   cached[ds]=(batch,meta)
  batch,meta=cached[ds];_,dts,mask,_,quantities=batch
  cpfile=ROOT/r['checkpoint_path'];assert sha(cpfile)==r['checkpoint_file_sha256']
  cp=torch.load(cpfile,map_location='cpu',weights_only=False);validate_checkpoint_route(cp,r['model'])
  config={k:v for k,v in data['model'].items() if k not in ('backbone','lambda_log_qty','lambda_tail','time_head_lr_multiplier')}
  model,_=build_count_aware_model(r['model'],**config,train_log_mean=data['statistics']['train_log_mean'],train_log_std=data['statistics']['train_log_std'],max_seq_len=data['loader']['max_seq_len'])
  model.load_state_dict(cp['model_state_dict'],strict=True);model.eval()
  before=canonical_state_dict_sha256(model.state_dict());assert before==r['state_tensor_sha256']
  with torch.no_grad():
   value=target_outputs(model,dts,mask,quantities,lambda_log_qty=1.)
   altered_qty=quantities.clone();altered_qty[:,-1]+=12345.
   qtest=target_outputs(model,dts,mask,altered_qty,lambda_log_qty=1.)
   altered_dt=dts.clone()
   top_code=data['model']['time_observation_contract'].get('top_code')
   # Perturb within the recorded observation domain (Instacart is top-coded at 30).
   altered_dt[:,-1]=torch.where(dts[:,-1]>1,torch.ones_like(dts[:,-1]),torch.full_like(dts[:,-1],2.)) if top_code else dts[:,-1]+123.
   ttest=target_outputs(model,altered_dt,mask,quantities,lambda_log_qty=1.)
   single=target_outputs(model,dts[:1],mask[:1],quantities[:1],lambda_log_qty=1.)
  assert torch.allclose(value['pred_qty'],qtest['pred_qty'],rtol=1e-6,atol=1e-6)
  assert torch.allclose(value['time_loss'],qtest['time_loss'],rtol=1e-6,atol=1e-6)
  assert torch.allclose(value['pred_qty'],ttest['pred_qty'],rtol=1e-6,atol=1e-6)
  assert torch.allclose(value['pred_qty'][:1],single['pred_qty'],rtol=1e-5,atol=1e-5)
  assert torch.allclose(value['time_loss'][:1],single['time_loss'],rtol=1e-5,atol=1e-5)
  assert canonical_state_dict_sha256(model.state_dict())==before
  for key in ['true_qty','pred_qty','time_loss']:assert torch.isfinite(value[key]).all()
  for i,m in enumerate(meta):
   q=float(value['true_qty'][i]);pred=float(value['pred_qty'][i]);nll=float(value['time_loss'][i])
   records.append({**m,'model':r['model'],'seed':r['seed'],'raw_quantity':q,'predicted_raw_quantity':pred,'recorded_gap':float(dts[i,-1]),'time_nll':nll,'history_length':int(value['history_length'][i]),'checkpoint_file_sha256':r['checkpoint_file_sha256'],'state_tensor_sha256':r['state_tensor_sha256'],'evaluation_scope':'validation_smoke'})
  checks.append({'dataset':ds,'model':r['model'],'seed':r['seed'],'targets':len(meta),'target_quantity_perturbation_preserves_prediction_and_time_loss':True,'target_gap_perturbation_preserves_quantity_prediction':True,'single_vs_batch_matches':True,'parameters_unchanged':True,'finite_outputs':True})
  assert time.monotonic()-started<180,'Bounded validation smoke exceeded 180 seconds per source bundle'
 result={'status':'passed','bundle':name,'checks':checks,'records':records,'elapsed_seconds':time.monotonic()-started,'runtime':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__},'device':'cpu','sample_rule':'first up to eight validation targets from first two sorted validation entities','full_endpoint_replay':False,'heldout_read':False}
 dest=OUT/'validation_smoke';dest.mkdir(exist_ok=True);(dest/f'{name}.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'bundle':name,'conditions':len(checks),'status':'passed','seconds':round(result['elapsed_seconds'],2)}),flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bundle');ap.add_argument('--targets',type=int,default=8);ap.add_argument('--split',choices=['validation'],default='validation');a=ap.parse_args()
 assert 1<=a.targets<=8,'Preparation is limited to eight validation targets per condition'
 if a.bundle:return run_bundle(a.bundle,a.targets)
 registry=read(OUT/'evaluation_registry.json');outputs=[]
 for name in registry['bundles']:
  subprocess.run([sys.executable,__file__,'--bundle',name,'--targets',str(a.targets)],check=True,timeout=240)
  outputs.append(read(OUT/'validation_smoke'/f'{name}.json'))
 checks=[c for o in outputs for c in o['checks']];records=[r for o in outputs for r in o['records']]
 assert len(checks)==84
 for ds in {r['dataset'] for r in records}:
  groups={}
  for r in records:
   if r['dataset']==ds:groups.setdefault((r['model'],r['seed']),[]).append((r['target_id'],r['raw_quantity'],r['recorded_gap']))
  assert len(groups)==21 and all(v==next(iter(groups.values())) for v in groups.values())
 result={'status':'passed','conditions':84,'prediction_rows':len(records),'target_identity_truth_and_gap_pairing_across_all_models_and_seeds':True,'records':records,'full_endpoint_replay':False,'performance_conclusion':'none_sample_is_for_adapter_correctness_only','heldout_read':False}
 (OUT/'validation_smoke.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='records'}))
if __name__=='__main__':main()
