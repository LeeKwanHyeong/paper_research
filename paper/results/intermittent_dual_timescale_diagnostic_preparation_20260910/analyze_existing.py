"""Read-only analysis of frozen aggregate validation evidence; no model execution."""
import json,hashlib,math,csv
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
BASE=ROOT/'paper/results/final_backbone_and_baselines_20260909'
EXT=ROOT/'paper/results/dual_timescale_intermittent_quantity_extension_20260910'
BP=BASE/'B_reference/intermittent_frozen_5000/summary.json'
CP=next((EXT/'remote').rglob('summary.json'))
paths=[BP,CP,BASE/'seed42_validation_comparison.json',EXT/'remote/audit.json']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
b,c=read(BP),read(CP)
reference=next(r for r in read(paths[2])['rows'] if r['dataset']=='intermittent_frozen_5000' and r['model']=='TitanTPP(B)')
audit=read(paths[3]);assert sha(BP)==reference['summary_sha256'] and sha(CP)==audit['summary_sha256']
assert audit['status']=='passed' and not audit['held_out_test_evaluated']
assert b['best_epoch']==77 and c['best_epoch']==37
assert b['resume_identity']['quantity_contract']==c['resume_identity']['quantity_contract']
N=86285
DMSE=c['best_val_qty_rmse']**2-b['best_val_qty_rmse']**2
DMAE=c['best_val_qty_mae']-b['best_val_qty_mae']
rows=[];overall={}
for axis,key in [('quantity','quantity_rows'),('history','history_rows')]:
 br={r['stratum']:r for r in b[key]};cr={r['stratum']:r for r in c[key]}
 assert br.keys()==cr.keys()
 for role,data in [('B',br),('candidate',cr)]:
  assert sum(r['count'] for r in data.values())==N
  model=b if role=='B' else c
  close(sum(r['count']*r['qty_mae'] for r in data.values())/N,model['best_val_qty_mae'])
  close(sum(r['count']*r['qty_rmse']**2 for r in data.values())/N,model['best_val_qty_rmse']**2)
  bias=sum(r['count']*r['qty_bias'] for r in data.values())/N
  if role in overall:close(overall[role]['bias'],bias)
  overall[role]={'bias':bias,'mse':model['best_val_qty_rmse']**2,'mae':model['best_val_qty_mae'],'centered_mse':model['best_val_qty_rmse']**2-bias*bias}
 for k,r in br.items():
  s=cr[k];assert r['count']==s['count'] and r['stratum_label']==s['stratum_label'];w=r['count']/N
  d=s['qty_rmse']**2-r['qty_rmse']**2;bias2=s['qty_bias']**2-r['qty_bias']**2
  row={'axis':axis,'stratum':k,'label':r['stratum_label'],'count':r['count'],'sample_share':w,'b_rmse':r['qty_rmse'],'candidate_rmse':s['qty_rmse'],'b_mae':r['qty_mae'],'candidate_mae':s['qty_mae'],'b_bias':r['qty_bias'],'candidate_bias':s['qty_bias'],'mean_prediction_shift':s['qty_bias']-r['qty_bias'],'delta_mse':d,'weighted_delta_mse':w*d,'net_delta_mse_share_percent':100*w*d/DMSE,'weighted_delta_mae':w*(s['qty_mae']-r['qty_mae']),'weighted_bias_squared_delta':w*bias2,'weighted_centered_mse_delta':w*(d-bias2)}
  close(row['weighted_delta_mse'],row['weighted_bias_squared_delta']+row['weighted_centered_mse_delta']);rows.append(row)
 group=[r for r in rows if r['axis']==axis]
 close(sum(r['weighted_delta_mse'] for r in group),DMSE);close(sum(r['weighted_delta_mae'] for r in group),DMAE)
dbias2=overall['candidate']['bias']**2-overall['B']['bias']**2
result={'status':'aggregate_diagnosis_passed','scope':'seed42_validation_only_existing_summaries','target_count':N,'checkpoint_epochs':{'B':77,'candidate':37},'overall':overall,'delta_mse':DMSE,'delta_mae':DMAE,'global_bias_squared_delta':dbias2,'global_centered_mse_delta':DMSE-dbias2,'global_bias_squared_fraction_of_delta_mse':dbias2/DMSE,'rows':rows,'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths},'checks':['source_summary_hashes_match_frozen_receipts','same_train_quantile_contract','same_stratum_labels_counts','both_axes_reconstruct_each_model_RMSE_and_MAE','both_axes_reconstruct_same_signed_bias','both_axes_reconstruct_total_delta_MSE_and_MAE','bias_squared_plus_centered_MSE_identity'],'limitations':['Axes are separate marginal partitions; no intersection or series-level causality identified.','Net contribution shares may be negative; not gross deterioration shares.','Bias decomposition is descriptive, not evidence that calibration or a memory module change will help.'],'new_inference':False,'remote_execution':False,'training':False}
(OUT/'aggregate_diagnosis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
with (OUT/'strata_comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# Intermittent 기존 증적의 구간별 진단','', '동일 validation 86,285건, seed42, B epoch77 / 후보 epoch37. 원본 summary만 사용했으며 새 추론·학습을 실행하지 않았다. 오차는 예측−실제 수량이다.','',f'전체 ΔMSE는 {DMSE:.6f}, ΔMAE는 {DMAE:.6f}다. 전체 평균 오차는 B {overall["B"]["bias"]:.6f} → 후보 {overall["candidate"]["bias"]:.6f}다. bias² 차이는 전체 ΔMSE의 {dbias2/DMSE*100:.2f}%이며, 나머지는 centered MSE 차이다. 이 분해는 원인이나 보정 효과를 입증하지 않는다.']
for axis,title in [('quantity','수량 구간'),('history','관측 이력 구간')]:
 lines+=['',f'## {title}','', '| 구간 | 표본 수 | RMSE B → 후보 | MAE B → 후보 | 평균 오차 B → 후보 | 전체 ΔMSE 기여 | 순악화 기여 비중 |','| --- | --- | --- | --- | --- | --- | --- |']
 for r in rows:
  if r['axis']==axis:lines.append(f'| {r["label"]} | {r["count"]:,} | {r["b_rmse"]:.6f} → {r["candidate_rmse"]:.6f} | {r["b_mae"]:.6f} → {r["candidate_mae"]:.6f} | {r["b_bias"]:.6f} → {r["candidate_bias"]:.6f} | {r["weighted_delta_mse"]:+.6f} | {r["net_delta_mse_share_percent"]:+.2f}% |')
lines+=['','수량 구간과 이력 구간은 같은 표본을 각각 나눈 결과다. 두 표의 기여를 서로 더하지 않는다. 기여는 (구간 표본 수/전체 표본 수)×구간 ΔMSE로 계산하며 RMSE를 가중합하지 않는다. 순악화 기여 비중과 양의 오차 증가 총량 비중을 구분한다.','', '각 축의 합계로 양 모델의 전체 RMSE·MAE·평균 오차와 모델 간 차이를 독립 복원했다. 수량×이력 교차조건, 개별 예측 이동 분포와 series 집중도는 이 집계만으로 알 수 없다. 원본 파일 SHA와 검증 항목은 [aggregate_diagnosis.json](aggregate_diagnosis.json)에 기록했다.']
(OUT/'aggregate_findings.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
