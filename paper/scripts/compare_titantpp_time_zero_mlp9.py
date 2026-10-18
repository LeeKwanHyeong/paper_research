"""Compare completed alpha0 fits against frozen alpha.1/1/10 Validation endpoints."""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import json,math,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATASETS=('yellow_trip_hourly','raf_spare_parts','intermittent_frozen_5000')
SEEDS=(42,52,62)
ALPHAS=(0.,.1,1.,10.)
METRICS=('qty_rmse','qty_mae','time_nll')
COHORTS=('overall','cap_hit','tail')
PAIRS=((0.,1.),(0.,.1),(0.,10.),(.1,1.),(10.,1.),(.1,10.))

def require(ok,message):
    if not ok:raise ValueError(message)

def aggregate(records):
    indexed={(r['dataset'],r['seed'],float(r['time_alpha'])):r for r in records}
    expected={(d,s,a) for d in DATASETS for s in SEEDS for a in ALPHAS}
    require(len(records)==len(indexed)==36 and set(indexed)==expected,'Expected exact36 conditions including alpha0 controls')
    summaries,differences,decisions=[],[],[]
    for d in DATASETS:
        for s in SEEDS:
            base=indexed[d,s,1.]
            for a in ALPHAS:
                r=indexed[d,s,a]
                require(r['target_identity']==base['target_identity'] and r['host']==base['host'],'Target or host differs from matching reference')
                require(r['time_head_trained'] is (a!=0.),'Time-head training flag mismatch')
        for cohort in COHORTS:
            for a in ALPHAS:
                group=[indexed[d,s,a]['metrics'][cohort] for s in SEEDS]
                require(len({r['count'] for r in group})==1,'Unequal cohort populations')
                n=group[0]['count'];require(type(n) is int and n>=0,'Invalid cohort count')
                for metric in METRICS:
                    vals=[r[metric] for r in group]
                    if n==0:
                        require(all(v is None for v in vals),'Empty cohort metrics must be N/A')
                        mean=sd=None
                    else:
                        require(all(type(v) in (int,float) and math.isfinite(v) for v in vals),'Nonfinite metric')
                        mean,sd=statistics.mean(vals),statistics.stdev(vals)
                    summaries.append({'dataset':d,'cohort':cohort,'time_alpha':a,'metric':metric,'seed_count':3,'target_count_per_seed':n,
                        'mean':mean,'sample_sd_ddof1':sd,'time_head_trained':a!=0.,'time_metric_diagnostic_only':metric=='time_nll' and a==0.})
            for s in SEEDS:
                for left,right in PAIRS:
                    lhs=indexed[d,s,left]['metrics'][cohort];rhs=indexed[d,s,right]['metrics'][cohort]
                    require(lhs['count']==rhs['count'],'Different populations across weights')
                    for metric in METRICS:
                        x,y=lhs[metric],rhs[metric]
                        delta=None if x is None or y is None else x-y
                        relative=None if delta is None or y==0 else 100*delta/abs(y)
                        differences.append({'dataset':d,'seed':s,'cohort':cohort,'metric':metric,'left_time_alpha':left,'right_time_alpha':right,
                            'left_value':x,'right_value':y,'absolute_difference':delta,'relative_difference_percent':relative,
                            'worsened':None if delta is None else delta>0,'untrained_time_head_in_comparison':metric=='time_nll' and (left==0. or right==0.)})
        def avg(a,k):
            return next(r['mean'] for r in summaries if r['dataset']==d and r['cohort']=='overall' and r['time_alpha']==a and r['metric']==k)
        for a in (0.,.1,10.):
            better=all(avg(a,k)<avg(1.,k) for k in ('qty_rmse','qty_mae'))
            time_ok=avg(a,'time_nll')<=avg(1.,'time_nll')
            decisions.append({'dataset':d,'time_alpha':a,'quantity_both_means_improved_vs1':better,
                'time_head_trained':a!=0.,'time_nll_difference_diagnostic_only':a==0.,
                'observed_three_metric_means_improved_vs1':better and time_ok,
                'three_metric_joint_model_improvement':None if a==0. else better and time_ok,
                'quantity_only_control_not_jointly_optimized_forecaster':a==0.})
    return {'summary':summaries,'seed_differences':differences,'decisions':decisions}

def verify_zero_proof(receipt):
    proof=receipt.get('time_head_no_training_verification',{})
    require(receipt.get('quantity_only') is True and receipt.get('time_head_trained') is False,'Missing quantity-only receipt flags')
    require(proof.get('initial_time_head_sha256')==proof.get('final_time_head_sha256')
        and isinstance(proof.get('initial_time_head_sha256'),str) and len(proof['initial_time_head_sha256'])==64,'Time-head state changed or missing')
    require(proof.get('optimizer_steps_checked',0)>0 and proof.get('time_head_parameter_names')
        and all(proof.get(k) is True for k in ('all_time_head_gradients_none','optimizer_state_absent','time_head_parameters_unchanged')),'Time head was trained or unverified')
    return proof

def main():
    from paper.scripts import manage_titantpp_time_zero_mlp9 as zero
    from paper.scripts import manage_titantpp_time_weight_mlp18 as previous
    c0=zero.verify_local();cp=previous.verify_local()
    require(c0['evaluation_scope']==cp['evaluation_scope']=='validation_only','Wrong split scope')
    refs=previous.baseline_validation_references()
    require(refs==cp['baseline_validation_references']==c0['baseline_validation_references'],'Stored baseline binding changed')
    records,bindings=[],{}
    for manager,contract,alpha_zero in ((zero,c0,True),(previous,cp,False)):
        for host in ('5080','5090'):
            originals=manager.BUNDLE/'retrieval'/host/'originals'
            manager.verify_collected(contract,host,originals)
            for job in [j for j in contract['jobs'] if j['host']==host]:
                path=originals/'run'/job['id']/'validation_receipt.json';r=manager.read(path)
                require(r['status']=='complete' and r['validation_replay_passed'] and r['time_alpha']==job['time_alpha'] and r['held_out_test_evaluated'] is False,'Invalid selected Validation endpoint')
                proof=verify_zero_proof(r) if alpha_zero else None
                records.append({'dataset':r['dataset'],'host':host,'seed':r['seed'],'time_alpha':r['time_alpha'],'selected_epoch':r['selected_epoch'],
                    'metrics':r['metrics'],'target_identity':r['target_identity'],'time_head_trained':not alpha_zero,'time_head_no_training_verification':proof})
                bindings[str(path)]=manager.sha(path)
    for r in refs:
        records.append({**r,'time_head_trained':True})
        bindings[r['path']]=r['sha256']
    result=aggregate(records)
    result.update(status='complete',scope='validation_only',created_at=datetime.now(timezone.utc).isoformat(),records=records,source_bindings=bindings,
        new_alpha0_fit_count=9,previous_weight_fit_count=18,stored_reference_count=9,comparison_model_count=36,seed_count=3,
        held_out_test_evaluated=False,historical_cnn_changed=False,baseline_reevaluated=False,
        interpretation_limits=['Alpha0 is a quantity-only control; its untrained time-head NLL is diagnostic, not a jointly trained time forecaster.',
            'Quantity-only optimization includes changes in global gradient clipping; it does not establish gradient conflict as a unique cause.',
            'Do not pool different dataset raw errors or compare differently weighted joint objectives.',
            'Validation checkpoint selection remains first strict finite minimum raw quantity RMSE; no new Test.'])
    out=zero.REPORT/'comparison_v1';out.mkdir(exist_ok=False)
    zero.write(out/'FOUR_WEIGHT_VALIDATION.json',result)
    for name,rows in (('SUMMARY.csv',result['summary']),('SEED_DIFFERENCES.csv',result['seed_differences'])):
        with (out/name).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# MLP16 시간 가중치0·0.1·1·10 Validation 비교','',
        '전체 수량RMSE로 선택한 같은 checkpoint의3seed 평균±표본SD(ddof1)다. α0은 수량만 학습했으므로 시간NLL은 미학습 시간head의 진단값이다.','',
        '| 데이터 | 시간 가중치 | RMSE | MAE | TimeNLL | 시간 head 학습 |','|---|---:|---:|---:|---:|---|']
    for d in DATASETS:
        for a in ALPHAS:
            vals=[]
            for metric in METRICS:
                x=next(r for r in result['summary'] if r['dataset']==d and r['cohort']=='overall' and r['time_alpha']==a and r['metric']==metric)
                vals.append(f"{x['mean']:.6f}±{x['sample_sd_ddof1']:.6f}")
            lines.append(f"| {d} | {a:g} | "+' | '.join(vals)+f" | {'미학습(진단)' if a==0 else '학습'} |")
    lines+=['','모든seed별 절대/상대차이·악화·상한도달 집단과 빈 상한초과 집단은 JSON/CSV에 보존했다. 수량전용 대조는 backbone 변경이나 원인 확정이 아니다.']
    (out/'FOUR_WEIGHT_VALIDATION.md').write_text('\n'.join(lines)+'\n')
    zero.write(zero.BUNDLE/'retrieval/final_receipt.json',{'status':'complete','new_fit_count':9,'validation_replay_count':9,'previous_weight_fit_count':18,
        'baseline_reference_count':9,'comparison_model_count':36,'seed_count':3,'held_out_test_evaluated':False,
        'time_head_no_training_verified_count':9,'source_bindings':bindings,'report_files':{str(p):zero.sha(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps({'status':'complete','comparison_model_count':36,'output':str(out)}))

if __name__=='__main__':main()
