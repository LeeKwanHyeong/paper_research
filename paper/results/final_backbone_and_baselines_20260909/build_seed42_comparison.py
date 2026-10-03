"""Reproduce the nine-row validation table using audited JSON; no model loading."""
from pathlib import Path
import csv, hashlib, json, math

ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
contract=read(ROOT/'contracts/hard_lmm_dual_timescale_v1.json')
specs={s['dataset']:s for s in contract['data_bindings']}
bref_path=ROOT/'B_reference/a_vs_b_metrics.json'
assert sha(bref_path)==contract['B_validation_reference']['sha256']
brefs={r['dataset']:r for r in read(bref_path)['datasets']}
close=lambda a,b: math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)
rows=[]

def admit(dataset,model,summary_path,launch_path,history_path,audit=None):
    spec=specs[dataset];s=read(summary_path);launch=read(launch_path);h=read(history_path)['history']
    assert s['status']=='success'
    if audit is not None:
        assert launch['status']=='complete'
    else:
        # Instacart's old B/C launcher remained running after C was discontinued.
        # Require the exact frozen launcher and the independently complete B run.
        assert launch['status']==brefs[dataset]['evidence']['launch_status']
    assert s['seed']==42 and s['checkpoint_monitor']=='validation_raw_quantity_rmse'
    assert s['evaluation_scope']==launch['evaluation_scope']=='validation_only'
    assert s['held_out_test_evaluated'] is launch['held_out_test_evaluated'] is False
    assert launch['data_sha256']==spec['data_sha256'] and launch['split_manifest_sha256']==spec['split_manifest_sha256']
    pop=launch['validation_target_population']
    assert pop['target_count']==spec['expected_validation_targets']
    assert pop['target_identity_sha256']==spec['expected_validation_target_identity_sha256']
    assert pop['target_quantity_sha256']==spec['expected_validation_target_quantity_sha256']
    for key,value in [('batch_size',128),('lr',.001),('hidden_dim',64),('lambda_log_qty',1.),('lambda_tail',0.),('grad_clip',1.)]:assert launch[key]==value
    # B shared a launch with C; its nonadaptive variant has no adaptive model kwargs.
    assert s['variant']=='count_only_log_regression'
    assert 'quantile_adaptive_contract' not in s['interface_meta']
    if audit is not None: assert launch['quantile_adaptive_strength']==0
    early=launch['early_stopping'];assert early['min_epochs']==early['patience']==40 and early['monitor']=='validation_raw_quantity_rmse'
    time=launch['time_head'];assert time['mode']=='legacy_clamped_rmtpp' and time['time_intercept_limit']==300 and time['time_scale']==3 and close(time['time_w_max'],10/3)
    assert [r['epoch'] for r in h]==list(range(1,s['completed_epochs']+1))
    assert all(r['train_all_finite'] and r['train_event_count']==spec['expected_train_targets'] for r in h)
    assert all(math.isfinite(v) for r in h for v in r.values() if isinstance(v,float))
    best=min(h,key=lambda r:r['val_qty_rmse']);assert best['epoch']==s['best_epoch']
    assert s['completed_epochs']==300 or s['completed_epochs']-s['best_epoch']==40
    q=s['quantity_rows'];count=sum(r['count'] for r in q);assert count==spec['expected_validation_targets']
    body=[r for r in q if r['stratum'] in ('le_p50','p50_p90','p90_p95')]
    metrics={'raw_rmse':math.sqrt(sum(r['count']*r['qty_rmse']**2 for r in q)/count),
      'overall_mae':sum(r['count']*r['qty_mae'] for r in q)/count,
      'body_mae':sum(r['count']*r['qty_mae'] for r in body)/sum(r['count'] for r in body),
      'gt_p99_mae':next(r['qty_mae'] for r in q if r['stratum']=='gt_p99'),
      'legacy_clamped_time_loss':sum(r['count']*r['time_nll'] for r in q)/count}
    assert close(metrics['raw_rmse'],best['val_qty_rmse']) and close(metrics['overall_mae'],best['val_qty_mae'])
    if audit is not None:
        assert audit['status']=='passed' and audit['checkpoint_state_sha256']==audit['resume_best_state_sha256']==s['checkpoint_state_sha256']
        for key,path in [('summary',summary_path),('history',history_path),('launch_contract',launch_path)]:assert sha(path)==audit['artifact_sha256'][key]
        canonical=dict(audit['metrics']);canonical['legacy_clamped_time_loss']=canonical.pop('time_nll')
    else:
        ref=brefs[dataset];e=ref['evidence']['B_t0_raw_rmse']
        assert sha(summary_path)==e['summary_sha256'] and sha(history_path)==e['history_sha256']
        assert sha(launch_path)==ref['evidence']['launch_contract_sha256']
        assert s['checkpoint_state_sha256']==e['checkpoint_state_sha256']
        canonical={k:v for k,v in ref['metrics']['B_t0_raw_rmse'].items() if k in metrics or k=='time_nll'}
        canonical['legacy_clamped_time_loss']=canonical.pop('time_nll')
    for key,value in canonical.items():assert close(metrics[key],value),(dataset,model,key)
    metrics.update(canonical)
    rows.append({'dataset':dataset,'model':model,'seed':42,'best_epoch':s['best_epoch'],'completed_epochs':s['completed_epochs'],**metrics,
      'validation_target_count':count,'validation_target_identity_sha256':pop['target_identity_sha256'],
      'source_revision':s['source_revision'],'checkpoint_state_sha256':s['checkpoint_state_sha256'],
      'summary_file':str(summary_path.relative_to(ROOT)),'summary_sha256':sha(summary_path),
      'launch_sha256':sha(launch_path),'history_sha256':sha(history_path)})

for dataset in specs:
    d=ROOT/'B_reference'/dataset
    admit(dataset,'TitanTPP(B)',d/'summary.json',d/'launch_contract.json',d/'history.json')
for job in sorted((ROOT/'5080/completed/jobs').iterdir()):
    a=read(job/'audit_receipt.json');p=job/'training/runs'/a['backbone']/'count_only_log_regression/seed_42'
    admit(a['dataset'],a['backbone'].upper(),p/'summary.json',job/'training/launch_contract.json',p/'history.json',a)
for model in ('rmtpp','thp'):
    job=ROOT/'Instacart_existing'/model;p=job/'training/runs'/model/'count_only_log_regression/seed_42'
    admit('insta_market_basket',model.upper(),p/'summary.json',job/'training/launch_contract.json',p/'history.json',read(job/'audit_receipt.json'))
order=['TitanTPP(B)','RMTPP','THP']; rows.sort(key=lambda r:(list(specs).index(r['dataset']),order.index(r['model'])))
assert len(rows)==9 and len({(r['dataset'],r['model']) for r in rows})==9
out={'status':'passed','status_meaning':'evidence_audit_passed_not_model_performance_gate','scope':'seed42_validation_only','selector':'earliest_strict_validation_raw_RMSE','held_out_test_evaluated':False,
     'additional_seeds_executed':False,'reused_B_reference_sha256':sha(bref_path),'rows':rows,
     'checks':['pinned_source_artifact_hashes','same_target_population_and_quantities','same_loss_optimizer_selector_contract','finite_full_train_counts','earliest_checkpoint_and_patience40','strata_reconstruct_metrics','baseline_checkpoint_resume_receipts_and_pinned_B_reference']}
(ROOT/'seed42_validation_comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
with (ROOT/'seed42_validation_comparison.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
labels={'intermittent_frozen_5000':'Intermittent','yellow_trip_hourly':'Taxi','insta_market_basket':'Instacart'}
lines=['# TitanTPP(B) · RMTPP · THP: seed42 validation 비교','',
       '세 모델 모두 validation raw RMSE의 가장 이른 최솟값으로 checkpoint를 선택했다. 기존 결과를 재사용하고 빠진 네 baseline만 추가했다. 학습은 최대300/minimum40/patience40, 같은 데이터·대상 표본·수량 loss 조건이다.','']
for key,label in [('raw_rmse','Raw RMSE ↓'),('overall_mae','전체 MAE ↓')]:
    lines += ['## '+label,'','| 데이터셋 | TitanTPP(B) | RMTPP | THP |','|---|---:|---:|---:|']
    for dataset in specs:
        ds={r['model']:r[key] for r in rows if r['dataset']==dataset};minimum=min(ds.values())
        values=[('**'+f'{ds[m]:.6f}'+'**') if ds[m]==minimum else f'{ds[m]:.6f}' for m in order]
        lines.append('| '+labels[dataset]+' | '+' | '.join(values)+' |')
    lines.append('')
lines += ['TitanTPP(B)는 Intermittent와 Taxi의 RMSE에서 두 baseline보다 낮다. Instacart에서는 RMTPP와 THP가 더 낮다. 전체 MAE의 최솟값은 Intermittent에서 THP, Taxi와 Instacart에서 RMTPP다.','',
          '현재 증적은 단일 seed의 validation 비교이며 다중 seed 또는 held-out 우위는 확정하지 않는다. NHP·SAHP의 raw-RMSE selector 정렬 비교는 이 표에 포함되지 않았다. 신규 dual-timescale Backbone은 이 표의 범위 밖이다.','',
          'CSV/JSON에는 body·>p99 MAE, legacy clamped time loss, 선택·종료 epoch와 출처 SHA도 기록했다. 이 시간 점수를 정상화된 Time NLL이라고 해석하지 않는다. 9개 행의 대상 identity·quantity hash, 처리 건수, 선택 규칙과 지표 재구성 검증을 통과했다.']
lines += ['', 'B 증적은 원래 B/C 공용 launcher에서 개별 B run을 선별했다. 공용 adaptive strength=1과 Instacart launcher의 running 상태를 그대로 보존했으며, B 자체의 success·비적응형 log-regression variant·완료 이력과 고정 summary/history/launch/checkpoint-state SHA를 확인했다. JSON의 passed는 증적 감사 통과를 뜻하며 모델 성능 gate 통과를 뜻하지 않는다.']
(ROOT/'seed42_validation_comparison.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
