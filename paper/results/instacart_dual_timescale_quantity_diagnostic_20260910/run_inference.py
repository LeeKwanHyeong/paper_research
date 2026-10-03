#!/usr/bin/env python3
"""Pinned CPU-only same-population frozen inference; train/validation only."""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys, time
from datetime import datetime, timezone
import numpy as np
import polars as pl
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
LARGE=ROOT/'search_artifacts/instacart_dual_timescale_quantity_diagnostic_20260910'
CONTRACT_SHA='0f7a37b537778ab71d4910ff4a47e624360f9be8767f53ebaf8b5070ab2a1152'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def require(ok,msg):
 if not ok:raise ValueError(msg)
def save(p,x):
 Path(p).parent.mkdir(parents=True,exist_ok=True);t=Path(str(p)+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def population(ds,split,meta):
 h=hashlib.sha256();h.update(b'hard_lmm_target_identity_v1\0');h.update(split.encode()+b'\0');h.update(json.dumps([str(x) for x in ds.parts],ensure_ascii=False,separators=(',',':')).encode())
 for col in ['series_index','target_position','target_seq']:h.update(np.asarray(meta[col],dtype='<i8').tobytes())
 q=hashlib.sha256();q.update(b'hard_lmm_target_quantity_v1\0');q.update(np.asarray(meta['true_qty'],dtype='<f8').tobytes())
 return {'count':len(ds),'identity_sha256':h.hexdigest(),'quantity_sha256':q.hexdigest()}
def run(split):
 started=time.monotonic();require(sha(HERE/'contract.json')==CONTRACT_SHA,'contract drift');c=json.loads((HERE/'contract.json').read_text());ex=c['execution'];src=Path(ex['frozen_source_root'])
 require(subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip()==ex['frozen_source_revision'],'source revision drift')
 require(not subprocess.check_output(['git','-C',str(src),'status','--porcelain'],text=True).strip(),'dirty source')
 for f,d in ex['source_file_sha256'].items():require(sha(src/f)==d,f'source drift {f}')
 for p,h in [('path','sha256'),('split_manifest','split_manifest_sha256')]:require(sha(c['data'][p])==c['data'][h],'data drift')
 if split=='train':
  require((HERE/'validation_hypotheses.json').is_file(),'freeze validation explanation before train')
  require(json.loads((HERE/'numeric_compatibility_audit.json').read_text())['status']=='passed','validation replay compatibility must pass')
 sys.path.insert(0,str(src))
 from paper.scripts.count_aware_tpp_backbone.core import load_train_validation_frame,prepare_count_frame,target_outputs
 from paper.scripts.run_matched_frozen_lognormal_duration import build_source_model
 from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
 from paper.scripts.run_hard_lmm_bounded_qk_error_diagnostic import build_validation_metadata
 from simple_lab_test.search.common.runner import canonical_state_dict_sha256
 torch.set_num_threads(ex['threads']);torch.set_num_interop_threads(1);torch.manual_seed(42)
 models={};audit={'split':split,'started_at':datetime.now(timezone.utc).isoformat(),'contract_sha256':CONTRACT_SHA,'device':'cpu','torch_version':torch.__version__,'source_revision':ex['frozen_source_revision'],'source_files_verified':True,'held_out_test_evaluated':False,'training':False,'model_audits':{}}
 for role,spec in c['models'].items():
  require(sha(spec['checkpoint'])==spec['checkpoint_file_sha256'],f'{role} file drift');require(sha(spec['summary'])==spec['summary_file_sha256'],f'{role} summary drift')
  payload=torch.load(spec['checkpoint'],map_location='cpu',weights_only=True)
  require(payload['model_state_sha256']==spec['checkpoint_state_sha256'],f'{role} payload state drift');require(payload['backbone']==spec['backbone'],'route drift')
  m=build_source_model(payload,max_seq_len=64).cpu().eval().requires_grad_(False)
  audit['model_audits'][role]={'checkpoint_file_sha256':spec['checkpoint_file_sha256'],'before_state_sha256':canonical_state_dict_sha256(m.state_dict()),'strict_restore':True,'weights_only':True,'best_epoch':spec['best_epoch']}
  if role!='b':models[role]=m
  del m,payload
 print(json.dumps({'stage':'restored','elapsed':time.monotonic()-started}),flush=True)
 frame=prepare_count_frame(load_train_validation_frame(Path(c['data']['path'])))
 if split=='train':frame=frame.filter(pl.col('chronological_split')=='train')
 loader=make_loader(frame,target_split=split,batch_size=ex['batch_size'],lookback_weeks=52,max_seq_len=64,shuffle=False,generator=None);ds=loader.dataset;del frame
 require(len(ds)==c['populations'][split]['count'],'count drift')
 cache=c['b_caches'][split];require(sha(cache['path'])==cache['sha256'],'B cache digest drift')
 if split=='validation':
  old=pl.read_parquet(cache['path']);meta={x:old[x].to_numpy() for x in old.columns if not x.startswith(('pred_','error_','abs_','sq_','delta_'))};b=old['pred_b'].to_numpy();del old
 else:
  print(json.dumps({'stage':'train_metadata','count':len(ds)}),flush=True)
  meta=build_validation_metadata(ds,lookback_weeks=52,max_seq_len=64)
  old=np.load(cache['path'],allow_pickle=False)
  for key,target in [('quantity','true_qty'),('series_index','series_index'),('context_end','context_end'),('target_index','dataset_index'),('history_length','history_length')]:require(np.array_equal(old[key],meta[target]),f'B cache {key} mismatch')
  b=old['prediction'].copy();old.close()
 # Independently compare metadata to loader identities and target quantities.
 for key,values in [('series_index',(i for i,j in ds.index)),('context_end',(j for i,j in ds.index)),('target_position',(j+1 for i,j in ds.index)),('target_seq',(int(ds.seq_lists[i][j+1]) for i,j in ds.index)),('true_qty',(float(ds.val_lists[i][j+1]) for i,j in ds.index))]:
  actual=np.fromiter(values,dtype=np.float64 if key=='true_qty' else np.int64,count=len(ds));require(np.array_equal(actual,meta[key]),f'canonical {key} mismatch')
 pop=population(ds,split,meta);require(pop==c['populations'][split],'population digest drift');audit['population']=pop
 predictions={role:np.empty(len(ds),np.float64) for role in models};logloss={role:np.empty(len(ds),np.float64) for role in models}
 cursor=0;infer_start=time.monotonic();last_log=infer_start
 with torch.inference_mode():
  for batch,(_,dts,mask,parts,qty) in enumerate(loader):
   end=cursor+len(dts)
   require(np.array_equal(parts.numpy().astype(np.int64),np.asarray(meta['series_index'])[cursor:end]),'batch series mismatch')
   for role,m in models.items():
    o=target_outputs(m,dts,mask,qty,lambda_log_qty=1.0)
    require(np.array_equal(o['true_qty'].numpy().astype(np.float64),np.asarray(meta['true_qty'])[cursor:end]),f'{role} target mismatch')
    require(np.array_equal(o['history_length'].numpy(),np.asarray(meta['history_length'])[cursor:end]),f'{role} history mismatch')
    p=o['pred_qty'].numpy().astype(np.float64);l=o['log_qty_loss'].numpy().astype(np.float64)
    require(np.isfinite(p).all() and np.isfinite(l).all(),f'{role} nonfinite')
    predictions[role][cursor:end]=p;logloss[role][cursor:end]=l
   cursor=end
   if time.monotonic()-last_log>=30 or batch==0:
    elapsed=time.monotonic()-infer_start;progress={'stage':'inference','split':split,'count':cursor,'total':len(ds),'seconds':elapsed,'targets_per_second':cursor/max(elapsed,1e-6),'eta_seconds':(len(ds)-cursor)*elapsed/cursor};print(json.dumps(progress),flush=True);save(HERE/f'{split}_progress.json',progress);last_log=time.monotonic()
 require(cursor==len(ds),'incomplete inference')
 predictions['b']=b;true=np.asarray(meta['true_qty'],dtype=np.float64);metrics={}
 for role,p in predictions.items():
  err=p-true;metrics[role]={'count':len(p),'qty_mae':float(np.abs(err).mean()),'qty_rmse':float(np.sqrt(np.mean(err**2))),'qty_bias':float(err.mean()),'log_qty_mse':float(logloss[role].mean()) if role!='b' else float(((np.log1p(p)-np.log1p(true))**2).mean())}
  spec=c['models'][role]
  if role in models:
   after=canonical_state_dict_sha256(models[role].state_dict());require(after==spec['checkpoint_state_sha256'],f'{role} state mutated');audit['model_audits'][role]['after_state_sha256']=after
  else:audit['model_audits'][role]['cache_file_sha256']=cache['sha256']
  if split=='validation':
   summary=json.loads(Path(spec['summary']).read_text());deltas={k:metrics[role][k]-summary['best_val_'+k] for k in ['qty_mae','qty_rmse','log_qty_mse']};audit['model_audits'][role]['summary_replay_deltas']=deltas
   audit['model_audits'][role]['original_absolute_replay_passed']=all(abs(v)<=ex['validation_metric_absolute_tolerance'] for v in deltas.values())
  audit['model_audits'][role]['state_unchanged']=True
 keep=['series_id','dataset_index','series_index','context_end','target_position','target_seq','history_length','true_qty','target_dt','last_qty','history_mean_qty','history_mean_log_qty','recent3_minus_history_mean_log_qty','last_minus_history_mean_log_qty']
 df=pl.DataFrame({k:meta[k] for k in keep}).with_columns(pl.lit(split).alias('split'),*[pl.Series('prediction_'+r,p) for r,p in predictions.items()])
 require(df.select(pl.struct(['series_id','target_position']).n_unique()).item()==len(ds),'duplicate target key')
 for k in keep:
  if k!='series_id':require(np.isfinite(df[k].to_numpy()).all(),f'nonfinite metadata {k}')
 out=LARGE/f'{split}_predictions.parquet';require(not out.exists(),'refuse output overwrite');df.write_parquet(out,compression='zstd');original_passed=split=='validation' and all(x['original_absolute_replay_passed'] for x in audit['model_audits'].values());audit.update({'metrics':metrics,'prediction_path':str(out),'prediction_sha256':sha(out),'completed_at':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-started,'inference_seconds':time.monotonic()-infer_start,'validation_replay_passed':original_passed,'status':'passed' if split=='train' or original_passed else 'requires_numeric_compatibility_audit'})
 save(HERE/f'{split}_inference_audit.json',audit);print(json.dumps(audit),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['validation','train'],required=True);a=p.parse_args();run(a.split)
