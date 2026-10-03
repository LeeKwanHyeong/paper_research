#!/usr/bin/env python3
"""Apply validation-frozen descriptive signs separately to each train-series fold."""
from pathlib import Path
import json,hashlib
import numpy as np
import polars as pl
import analyze_predictions as ap
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
HYP_SHA='d410b4978962d3fd3fa522c0babf1cd6e7a5fc3a52013d46c87f999d8cb43127'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok,message):
 if not ok:raise ValueError(message)
def main():
 require(sha(HERE/'validation_hypotheses.json')==HYP_SHA,'hypotheses drift')
 hypotheses=json.loads((HERE/'validation_hypotheses.json').read_text())
 require(sha(HERE/'validation/summary.json')==hypotheses['source_validation_summary_sha256'],'validation summary drift')
 require(sha(HERE/'validation/strata.csv')==hypotheses['source_validation_strata_sha256'],'validation strata drift')
 path=ROOT/'search_artifacts/instacart_dual_timescale_quantity_diagnostic_20260910/train_predictions.parquet'
 audit=json.loads((HERE/'train_inference_audit.json').read_text());require(audit['status']=='passed' and sha(path)==audit['prediction_sha256'],'train inference audit/prediction drift')
 contract=json.loads((HERE/'contract.json').read_text());require(sha(HERE/'contract.json')==audit['contract_sha256'] and audit['population']==contract['populations']['train'],'train contract/population drift')
 data=ap.prepare_frame(pl.read_parquet(path),split='train');rows=[];checks=[];concentration=[]
 def check(fold,hyp,name,value,operator):
  good={'positive':value>0,'negative':value<0,'nonnegative':value>=0,'less_than_half':value<0.5}[operator]
  checks.append({'fold':fold,'hypothesis_id':hyp,'check':name,'value':float(value),'criterion':operator,'passed':bool(good)})
 for fold in [0,1]:
  base=data.folds==fold;cm={}
  for family,values,labels in [('overall',None,['all']),('quantity_band',data.quantity_band,ap.QUANTITY_LABELS),('history_band',data.history_band,ap.HISTORY_LABELS)]:
   for label in labels:
    mask=base if values is None else base&(values==label)
    if not mask.any():continue
    for ref in ap.REFERENCE_ROLES:
     c=ap.comparison_metrics(data.target[mask],data.predictions['candidate'][mask],data.predictions[ref][mask]);cm[(family,label,ref)]=c
     rows.append({'fold':fold,'family':family,'group':label,'reference':ref,'count':int(mask.sum()),**{k:c[k] for k in ['delta_mse','delta_mae','delta_rmse','delta_bias_squared','delta_centered_mse','mean_prediction_shift']}})
  for ref in ['rmtpp','thp']:
   c=cm[('overall','all',ref)];check(fold,'H1_benchmark_quantity_level',ref+' overall mean_prediction_shift',c['mean_prediction_shift'],'negative');check(fold,'H1_benchmark_quantity_level',ref+' overall delta_bias_squared',c['delta_bias_squared'],'positive')
   for label in ['(8,20]','(20,25]','(25,35]']:check(fold,'H1_benchmark_quantity_level',ref+' '+label+' delta_mse',cm[('quantity_band',label,ref)]['delta_mse'],'positive')
   check(fold,'H1_benchmark_quantity_level',ref+' <=8 delta_mse',cm[('quantity_band','<=8',ref)]['delta_mse'],'negative')
  c=cm[('history_band','8-15','rmtpp')]
  for field,sign in [('delta_mse','positive'),('mean_prediction_shift','negative')]:check(fold,'H2_rmtpp_history','rmtpp h8-15 '+field,c[field],sign)
  check(fold,'H2_rmtpp_history','rmtpp h2-3 delta_mse',cm[('history_band','2-3','rmtpp')]['delta_mse'],'negative')
  for label,msesign,shiftsign in [('2-3','positive','negative'),('8-15','negative','positive')]:
   c=cm[('history_band',label,'b')]
   check(fold,'H3_B_history_tradeoff','b h'+label+' delta_mse',c['delta_mse'],msesign);check(fold,'H3_B_history_tradeoff','b h'+label+' mean_prediction_shift',c['mean_prediction_shift'],shiftsign)
  c=cm[('quantity_band','(8,20]','b')]
  for field,sign in [('delta_mse','positive'),('delta_centered_mse','positive'),('mean_prediction_shift','nonnegative')]:check(fold,'H4_B_mid_quantity','b q8-20 '+field,c[field],sign)
  delta=(data.predictions['candidate'][base]-data.target[base])**2-(data.predictions['rmtpp'][base]-data.target[base])**2
  totals=pl.DataFrame({'series_id':data.series_id[base].tolist(),'delta_sse':delta}).group_by('series_id').agg(pl.col('delta_sse').sum()).sort(['delta_sse','series_id'],descending=[True,False])
  values=totals['delta_sse'].to_numpy();n=int(np.ceil(len(values)*0.01));positive=float(np.maximum(values,0).sum());top=float(np.maximum(values[:n],0).sum());require(positive>0,'H5 not applicable: no positive series SSE');fraction=top/positive;require(np.isfinite(fraction),'H5 nonfinite fraction')
  concentration.append({'fold':fold,'series_count':len(values),'top_series_count':n,'gross_positive_sse':positive,'top1pct_positive_sse':top,'fraction':fraction});check(fold,'H5_series_concentration','top1pct share of gross positive SSE',fraction,'less_than_half')
 byhyp={};expected_checks={'H1_benchmark_quantity_level':24,'H2_rmtpp_history':6,'H3_B_history_tradeoff':8,'H4_B_mid_quantity':6,'H5_series_concentration':2}
 require({h['id'] for h in hypotheses['hypotheses']}==set(expected_checks),'hypothesis IDs drift')
 for h in hypotheses['hypotheses']:
  selected=[x for x in checks if x['hypothesis_id']==h['id']];require(len(selected)==expected_checks[h['id']],'missing hypothesis checks');passed=sum(x['passed'] for x in selected);byhyp[h['id']]={'passed_checks':passed,'total_checks':len(selected),'status':'replicated' if passed==len(selected) else 'partially_replicated' if passed else 'not_replicated'}
 (HERE/'train').mkdir(exist_ok=True)
 pl.DataFrame(rows).write_csv(HERE/'train/fold_strata.csv')
 out={'status':'completed','hypotheses_sha256':HYP_SHA,'train_prediction_sha256':sha(path),'row_count':len(data.target),'fold_series_disjoint':not set(data.series_id[data.folds==0])&set(data.series_id[data.folds==1]),'interpretation':'descriptive in-sample replication, not OOF; preserve all failed signs','hypotheses':byhyp,'checks':checks,'series_concentration':concentration}
 (HERE/'train_hypothesis_checks.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(byhyp,indent=2))
if __name__=='__main__':main()
