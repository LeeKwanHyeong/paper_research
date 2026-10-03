"""Integrate frozen validation JSON only; no inference or checkpoint loading."""
import json, hashlib, math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
EXT=OUT.parent
BASE=ROOT/'paper/results/final_backbone_and_baselines_20260909'
REMOTE=EXT/'remote'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
state=read(REMOTE/'status.json');audit=read(REMOTE/'audit.json')
assert state['status']=='completed' and audit['status']=='passed'
assert sha(REMOTE/'audit.json')==state['audit_sha256']
assert sha(REMOTE/'run_extension.py')=='86ad8d7f905eab67f4ed2dd6bae4c547a44488d006a62775ac52a8eb0fe7dc74'
assert sha(REMOTE/'execution_contract.json')==sha(EXT/'execution_contract.json')
contract=read(REMOTE/'execution_contract.json')
assert sha(REMOTE/'benchmark_reference.json')==contract['benchmark_reference_sha256']
summary_file=next(REMOTE.rglob('summary.json'));summary=read(summary_file)
assert sha(summary_file)==audit['summary_sha256']
history=read(summary_file.with_name('history.json'))['history']
assert len(history)==summary['completed_epochs']==77
assert all(r['train_all_finite'] and r['train_event_count']==393824 and r['train_batch_count']==3077 for r in history)
assert all(math.isfinite(v) for r in history for v in r.values() if isinstance(v,(int,float)))
best=min(history,key=lambda r:r['val_qty_rmse'])
assert best['epoch']==summary['best_epoch']==audit['best_epoch']==37
assert len(history)-best['epoch']==40
close(best['val_qty_rmse'],audit['metrics']['raw_rmse'])
assert not summary['held_out_test_evaluated'] and not audit['held_out_test_evaluated']
assert audit['selected_restore']['strict_model_restore'] and audit['selected_restore']['finite_forward']
assert audit['last_restore']['optimizer_restore']
reference=read(BASE/'seed42_validation_comparison.json')
campaign=read(BASE/'final_campaign_decision.json')
assert reference==read(REMOTE/'benchmark_reference.json')
rows=[]; sources={};comparisons=[]
def record(p):sources[str(p.relative_to(ROOT))]=sha(p)
for p in [BASE/'seed42_validation_comparison.json',BASE/'final_campaign_decision.json',REMOTE/'status.json',REMOTE/'audit.json',REMOTE/'execution_contract.json',summary_file,summary_file.with_name('history.json')]:record(p)
# Verify each reused reference summary against its frozen receipt.
for r in reference['rows']:
 p=BASE/r['summary_file'];assert sha(p)==r['summary_sha256'];record(p)
 rows.append(dict(r,evidence_source=str(p.relative_to(ROOT))))
metrics=['raw_rmse','overall_mae','body_mae','gt_p99_mae']
for dataset in ['yellow_trip_hourly','insta_market_basket','intermittent_frozen_5000']:
 refs=[r for r in reference['rows'] if r['dataset']==dataset]
 b=next(r for r in refs if r['model']=='TitanTPP(B)')
 if dataset=='intermittent_frozen_5000':
  ap=REMOTE/'audit.json';a=audit;s=summary;sp=summary_file;lp=next(REMOTE.rglob('launch_contract.json'));gate=a['original_gate_for_transparency']
 else:
  c=next(r for r in campaign['rows'] if r['dataset']==dataset)
  ap=BASE/'5090/final'/f'seed42_screening_{dataset}_audit.json';a=read(ap)
  assert sha(ap)==c['audit_sha256']
  candidates=list((BASE/'5090').rglob('summary.json'))
  sp=next(p for p in candidates if sha(p)==a['summary_sha256']);s=read(sp)
  lp=next(p for p in (BASE/'5090').rglob('launch_contract.json') if read(p)['dataset']==dataset and read(p)['completed_run_count']==1)
  gate=c['performance_gate']
  for key in metrics:close(c['metrics'][key]['candidate'],a['metrics'][key])
 assert a['status']=='passed' and not a['held_out_test_evaluated']
 assert a['validation_target_identity_sha256']==b['validation_target_identity_sha256']
 launch=read(lp);pop=launch['validation_target_population']
 assert pop['target_quantity_sha256']==a['validation_target_quantity_sha256']
 assert pop['target_count']==b['validation_target_count']==a['metrics']['validation_target_count']
 strata=s['quantity_rows'];body=[r for r in strata if r['stratum'] in ['le_p50','p50_p90','p90_p95']]
 assert sum(r['count'] for r in strata)==b['validation_target_count']
 close(sum(r['qty_mae']*r['count'] for r in strata)/b['validation_target_count'],a['metrics']['overall_mae'])
 close(math.sqrt(sum(r['qty_rmse']**2*r['count'] for r in strata)/b['validation_target_count']),a['metrics']['raw_rmse'])
 close(sum(r['qty_mae']*r['count'] for r in body)/sum(r['count'] for r in body),a['metrics']['body_mae'])
 close(next(r['qty_mae'] for r in strata if r['stratum']=='gt_p99'),a['metrics']['gt_p99_mae'])
 for ref in refs:
  rs=read(BASE/ref['summary_file'])
  assert rs['resume_identity']['quantity_contract']==s['resume_identity']['quantity_contract']
  assert [(r['stratum'],r['count']) for r in rs['quantity_rows']]==[(r['stratum'],r['count']) for r in strata]
  # Target quantities and population are bound by the original launch evidence.
  launches=[p for p in (BASE/ref['summary_file']).parents if (p/'launch_contract.json').exists()]
  rl=read(launches[0]/'launch_contract.json')
  assert rl['validation_target_population']['target_quantity_sha256']==pop['target_quantity_sha256']
  comparisons.append({'dataset':dataset,'reference':ref['model'],'relative_change_percent':{key:(a['metrics'][key]/ref[key]-1)*100 for key in metrics}})
 row={'dataset':dataset,'model':'Dual-timescale candidate','seed':42,'best_epoch':a['best_epoch'],'completed_epochs':a['completed_epochs'],**{key:a['metrics'][key] for key in metrics},'legacy_clamped_time_loss':a['metrics']['clamped_time_loss'],'validation_target_count':pop['target_count'],'validation_target_identity_sha256':a['validation_target_identity_sha256'],'validation_target_quantity_sha256':a['validation_target_quantity_sha256'],'source_revision':a['source_revision'],'evidence_source':str(ap.relative_to(ROOT)),'execution_audit':'passed','performance_gate':gate,'body_definition':'quantity <= train p95','tail_definition':'quantity > train p99','quantity_contract':launch['quantity_contract']}
 rows.append(row)
 for p in [ap,sp,lp]:record(p)
result={'status':'evidence_integration_passed','scope':'seed42_validation_only','selector':'earliest_strict_validation_raw_RMSE','intermittent_completed_at':state['completed_at'],'rows':rows,'comparisons':comparisons,'source_sha256':sources,'checks':['source_receipt_hashes','77_finite_full_train_epochs','earliest_strict_best37_patience40','validation_identity_quantity_and_count','identical_train_quantile_boundaries_and_stratum_counts','strata_reconstruct_RMSE_MAE_body_tail','server_checkpoint_restore_audit_reused_without_monitor_deserialization'],'conclusion':'Keep B; no common improvement across three datasets; preserve Taxi time guardrail failure. Intermittent cause requires separate diagnosis.','new_training_executed':False,'held_out_test_evaluated':False}
(OUT/'comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
labels={'yellow_trip_hourly':'Taxi','insta_market_basket':'Instacart','intermittent_frozen_5000':'Intermittent'}
lines=['# Dual-timescale 후보 최종 validation 비교','', '**상태: 완료.** Intermittent는 2026-09-10 09:50:14 KST에 77 epoch로 종료했고, 최적 epoch37의 실행 감사가 통과했다.','', '동일 seed42 validation 표본과 최초 strict raw-RMSE 최적 checkpoint 기준이다. Body는 train p95 이하, tail은 train p99 초과다. 실행 감사 통과와 성능 기준 통과를 구분한다.','', '| 데이터셋 | 모델 | 최적/종료 epoch | RMSE | 전체 MAE | Body MAE | tail MAE | legacy time loss |','| --- | --- | --- | --- | --- | --- | --- | --- |']
for dataset,label in labels.items():
 for r in [x for x in rows if x['dataset']==dataset]:
  lines.append(f"| {label} | {r['model']} | {r['best_epoch']}/{r['completed_epochs']} | {r['raw_rmse']:.6f} | {r['overall_mae']:.6f} | {r['body_mae']:.6f} | {r['gt_p99_mae']:.6f} | {r['legacy_clamped_time_loss']:.6f} |")
lines+=['','## B 대비 후보 변화','', '음수는 개선, 양수는 악화다.','', '| 데이터셋 | RMSE | 전체 MAE | Body MAE | tail MAE |','| --- | --- | --- | --- | --- |']
for c in comparisons:
 if c['reference']=='TitanTPP(B)':lines.append('| '+labels[c['dataset']]+' | '+' | '.join(f'{c["relative_change_percent"][k]:+.2f}%' for k in metrics)+' |')
lines+=['','## 판단과 남은 작업','', '- Taxi: 수량 지표는 개선했으나 legacy time loss 10.825870은 B 1.473391 대비 기존 허용 기준을 충족하지 못한다. 원래 판정을 유지한다.', '- Instacart: B보다 소폭 개선했으나 RMSE는 RMTPP·THP보다 높다. 기존 CUDA 결과를 유지하며 CPU 진단 재현값으로 대체하지 않았다.', '- Intermittent: B 대비 전체·Body·tail 수량 오차 모두 악화했다. RMSE와 tail MAE는 RMTPP·THP보다 낮으나, 전체 MAE는 THP보다 높고 Body MAE는 두 모델보다 높다.', '- 세 데이터셋 공통 개선 근거가 없으므로 B 유지가 타당하다. 단일 seed validation 결과이며 최종 후보 채택이나 통계적 유의성을 선언하지 않는다.', '- 다음 작업: Intermittent 오차를 수량 구간·이력 길이별로 B와 대조할 진단 계획을 구체화한다. Body와 tail 모두 악화했으므로 tail만의 문제로 가정하지 않는다. Instacart 원인을 일반화하거나 모델 수정 대상을 아직 확정하지 않는다.', '- 새로운 진단 추론·모델/loss 변경·재학습·추가 seed·held-out 평가는 실행하지 않았다. 기존 캠페인 판정은 보존했다.','', '원본 경로·SHA·전체 비교와 검증 항목은 [comparison.json](comparison.json), 서버 실행 감사는 [audit.json](../remote/audit.json)에 기록했다.']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
latest={'observed_at':read(REMOTE/'retrieval_receipt.json')['observed_at'],'status':'completed_audit_passed_performance_gate_failed','completed_at':state['completed_at'],'completed_epochs':77,'best_epoch':37,'metrics':audit['metrics'],'integration':'paper/results/dual_timescale_intermittent_quantity_extension_20260910/integration/README.md','checkpoint_loaded_by_monitor':False,'remote_mutations':False,'next_action':'Pause completed heartbeat; prepare Intermittent diagnosis plan before any model modification.'}
(EXT/'monitor/terminal_20260910.json').write_text(json.dumps(latest,indent=2)+'\n')
(EXT/'monitor/latest.json').write_text(json.dumps(latest,indent=2)+'\n')
print(f'PASS: {len(rows)} rows; {len(comparisons)} candidate comparisons; {len(sources)} hashed sources.')
