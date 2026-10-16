"""Read frozen input context; no models, inference, training or remote calls."""
from pathlib import Path
import json,hashlib,math,datetime
import numpy as np
import polars as pl
P=Path(__file__).resolve().parents[2]
R=Path(__file__).resolve().parent
C24=P/'search_artifacts/titantpp_components_taxi_raf24_test_20261014_v1'
C4=P/'search_artifacts/titantpp_components_inter62_test_20261014_v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,allow_nan=False,indent=2)
def stats(a):
 return {'n':len(a),'mean':float(np.mean(a)),'population_sd':float(np.std(a)),'min':float(np.min(a)),'median':float(np.median(a)),'p95':float(np.quantile(a,.95)),'p99':float(np.quantile(a,.99)),'max':float(np.max(a)),'second_moment':float(np.mean(a*a))}

def main():
 rows=[];sources={};contracts={}
 for ds in ('yellow_trip_hourly','raf_spare_parts','intermittent_frozen_5000'):
  b=C4 if ds=='intermittent_frozen_5000' else C24;c=read(b/'contract.json');spec=c['datasets'][ds];definition=spec['definition'];p=b/spec['data_path'];assert sha(p)==spec['data_sha256'];sources[str(p.relative_to(P))]=sha(p);contracts[str((b/'contract.json').relative_to(P))]=sha(b/'contract.json')
  f=pl.read_parquet(p).sort(['oper_part_no','seq']);bounds=definition['quantity_boundaries_all_train_rows'];qtrain=f.filter(pl.col('chronological_split')=='train')['demand_qty'].to_numpy().astype(np.float64)
  # Frozen statistics refer to every Train row, including first per-series rows.
  digest=hashlib.sha256(b'quantity_comparison_all_train_row_quantity_v1\0'+qtrain.astype('<f8').tobytes()).hexdigest();assert digest==definition['statistics']['all_train_quantity_sha256']
  assert math.isclose(float(np.log1p(qtrain).mean()),definition['statistics']['train_log_mean'],abs_tol=1e-12,rel_tol=0)
  assert math.isclose(float(np.log1p(qtrain).std()),definition['statistics']['train_log_std'],abs_tol=1e-12,rel_tol=0)
  splitprofiles={};entities={}
  for split in ('train','validation','test'):
   sub=f.filter(pl.col('chronological_split')==split);q=sub['demand_qty'].to_numpy().astype(np.float64);cells=[]
   # np.searchsorted(side=left) retains all strict Train boundaries, q>threshold.
   idx=np.searchsorted(np.asarray(bounds),q,side='left')
   for k in range(5):
    a=q[idx==k];cells.append({'bin':k,'count':len(a),'share':len(a)/len(q),'quantity':stats(a) if len(a) else None})
   entities[split]=set(sub['oper_part_no'].to_list());profile={'all_rows_count':len(q),'target_scope':('all_Train_rows_not_loader_targets' if split=='train' else 'frozen_loader_target_rows'), 'entity_count':len(entities[split]),'quantity':stats(q),'quantity_cells':cells,'recorded_date_min':int(sub['demand_dt'].min()),'recorded_date_max':int(sub['demand_dt'].max())}
   positive=sub.filter(pl.col('delta_t')>0)['delta_t'].to_numpy().astype(np.float64);profile['positive_stored_delta_t']=stats(positive)
   if split!='train':
    assert len(q)==spec[split+'_population']['count'];expected=next(x for x in c['jobs'] if x['dataset']==ds)
    rr=read(b/'results'/expected['id']/('validation_replay.json' if split=='validation' else 'test_metrics.json'))
    assert [x['count'] for x in cells]==[x['count'] for x in rr['metrics']['quantity_cells']]
   else:profile['first_sequence_row_count']=sub.filter(pl.col('seq')==0).height;profile['nonfirst_rows_count']=sub.filter(pl.col('seq')>0).height
   splitprofiles[split]=profile
  seqsplit=f.group_by('oper_part_no','chronological_split').agg(pl.col('seq').min().alias('first'),pl.col('seq').max().alias('last'),pl.col('demand_dt').min().alias('datefirst'),pl.col('demand_dt').max().alias('datelast')).to_dicts();lookup={}
  for z in seqsplit:lookup.setdefault(z['oper_part_no'],{})[z['chronological_split']]=z
  violations=[]
  for part,z in lookup.items():
   for before,after in [('train','validation'),('validation','test'),('train','test')]:
    if before in z and after in z and not(z[before]['last']<z[after]['first'] and z[before]['datelast']<=z[after]['datefirst']):violations.append({'entity':part,'before':before,'after':after})
  assert not violations
  v,t=splitprofiles['validation'],splitprofiles['test'];rows.append({'dataset':ds,'frozen_data_sha256':spec['data_sha256'],'thresholds':bounds,'time_unit':definition['model']['time_observation_contract']['unit'],'profiles':splitprofiles,'per_entity_chronology_violations':violations,'global_date_ranges_may_overlap_across_entities':True,'Validation_Test_entity_intersection':len(entities['validation']&entities['test']),'Validation_only_entities':len(entities['validation']-entities['test']),'Test_only_entities':len(entities['test']-entities['validation']),'Test_minus_Validation_bin_share':[t['quantity_cells'][k]['share']-v['quantity_cells'][k]['share'] for k in range(5)],'Test_Validation_raw_quantity_second_moment_ratio':t['quantity']['second_moment']/v['quantity']['second_moment']})
 result={'status':'ANALYZED','schema':'rank_reversal_input_context_v1','new_training':False,'model_inference':False,'remote_calls':0,'data_statistics_refit_for_model':False,'model_or_split_changed':False,'rows':rows,'source_files':sources,'contract_files':contracts,'methodology':['Fixed Train boundaries reused, no Test optimized thresholds','Train profiles include first rows and must not be called training loader prediction errors','Validation/Test counts and exact 5bin membership counts verified against native metric receipts','Per-entity seq/date chronological boundaries verified; global date overlap across series is not alone evidence of leakage','Quantity/gap context differences are descriptive observations, not a cause of model performance']}
 write(R/'INPUT_CONTEXT.json',result)
 lines=['# 고정 입력의 구간별 분포 맥락','','## Material Passport','','Origin Skill: academic-research-suite / experiment-agent validate. Verification Status: ANALYZED. 기존 입력·영수증만 읽었으며 학습·모델 추론·원격 호출·모델 통계 재추정은 없다. Train 표는 첫 시점 포함 전체 행이며 Train 예측오류가 아니다. Validation/Test 표의 전체·5개 수량bin 개수는 실제 평가 대상과 일치한다.','', '| 데이터셋 | split | 행 수 | 수량 평균 | p95 | p99 | 최대 | 큰 수량 비율 | 양의 저장 gap 평균 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for row in rows:
  for split,p in row['profiles'].items():lines.append(f"| {row['dataset']} | {split} | {p['all_rows_count']} | {p['quantity']['mean']:.5g} | {p['quantity']['p95']:.5g} | {p['quantity']['p99']:.5g} | {p['quantity']['max']:.5g} | {p['quantity_cells'][4]['share']*100:.4f}% | {p['positive_stored_delta_t']['mean']:.5g} |")
 lines+=['','세 데이터셋 모두 각 entity의 Train→Validation→Test 순서 위반은 0이다. 일부 dataset은 entity별 분할이므로 global 날짜 범위가 겹친다. 이 겹침만으로 data leakage라고 해석하지 않는다. 날짜/개체별 모델오류 위치는 신규CNN raw prediction 미저장으로 이 맥락 표에서 알 수 없다.','', '수량·gap·bin share의 변화가 관측돼도 과적합·분포변화의 인과적 기전은 입증되지 않는다. 조건부 모델오류 변화와 함께 해석해야 한다.','']
 with (R/'INPUT_CONTEXT.md').open('x') as out:out.write('\n'.join(lines))
 print(json.dumps({'status':'ANALYZED','datasets':3,'per_entity_order_violations':0,'new_training':False,'new_model_inference':False,'source_files':len(sources)}))
if __name__=='__main__':main()
