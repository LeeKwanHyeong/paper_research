"""Recompare existing completed results; never run a model or remote command."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics as st
from datetime import datetime
from zoneinfo import ZoneInfo

O=Path(__file__).resolve().parent;R=O.parents[1]
F={};checks=0
def pin(p,h=None):
    p=Path(p);p=p if p.is_absolute() else R/p
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    assert h is None or digest==h,(str(p),digest,h)
    F[str(p.relative_to(R))]=digest
    return p
def read(p,h=None):return json.loads(pin(p,h).read_text())
def eq(a,b):
    global checks
    assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-9),(a,b)
    checks+=1
def save(name,data):
    with (O/name).open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
assert not (O/'RESULT.json').exists()
pin(Path(__file__))
tp=R/'reports/titantpp_history_taxi6_test_20261015_v1'
qa=read(tp/'FINAL_VERIFICATION_20261015_2247KST.json')
tr=read('search_artifacts/titantpp_history_taxi6_test_20261015_v1/final_receipt.json',qa['final_receipt_sha256'])
assert tr['status']=='complete' and tr['candidate_test_count']==6 and tr['validation_replay_count']==6 and tr['raf_comparison_complete']
for file,h in tr['files'].items():pin(file,h)
for file,h in qa['reports'].items():pin(tp/file,h)
c=read(tp/'TAXI_RAF_COMPARISON.json',qa['reports']['TAXI_RAF_COMPARISON.json'])
for file,h in c['sources'].items():pin(file,h)
rf=read('search_artifacts/titantpp_history_raf6_test_20261015_v1/final_receipt.json','dc85fbdab7b41c067b1a53a6823a0fbc68b612bcfd22e00f62ab1cb4b347ade0')
assert rf['status']=='complete'
for file,h in rf['files'].items():pin(file,h)
old=R/'reports/titantpp_three_dataset_final_comparison_20261005_v1'
oq=read(old/'final_verification.json');orc=read(old/'comparison_receipt.json',oq['comparison_receipt_sha256'])
cp=pin(old/'metrics_per_seed.csv',orc['outputs']['metrics_per_seed.csv'])
all_ext=list(csv.DictReader(cp.open()))
prior=read('reports/titantpp_cnn_external_error_audit_20261016_v1/RECEIPT.json')
pr=R/'reports/titantpp_cnn_external_error_audit_20261016_v1'
prior_result=read(pr/'RESULT.json',prior['output_sha256']['RESULT.json'])
pin(pr/'RESULT.md',prior['output_sha256']['RESULT.md'])
pin('reports/titantpp_cnn_history_adaptation_matched_20261015_v1/EXPERIMENT_CONTRACT.md')
arms={'titantpp_cnn_gru54_cnn_only':'CNN','titantpp_cnn_history_adaptive_v1':'Adaptive','titantpp_cnn_history_capacity_v1':'Capacity'}
metricmap={'RMSE':('whole','qty_rmse'),'MAE':('whole','qty_mae'),'Large_RMSE':('tail','qty_rmse'),'Large_MAE':('tail','qty_mae'),'TimeNLL':('whole','time_nll'),'Large_TimeNLL':('tail','time_nll')}
testmap={'RMSE':'whole_qty_rmse','MAE':'whole_qty_mae','Large_RMSE':'tail_qty_rmse','Large_MAE':'tail_qty_mae','TimeNLL':'whole_time_nll','Large_TimeNLL':'tail_time_nll'}
rows=[];aggregates=[];contrasts=[];tail=[];external=[];frontiers=[];acceptance=[]
dsmap={'Taxi':'yellow_trip_hourly','RAF':'raf_spare_parts'}
extmodels=['rmtpp','thp','nhp','sahp','s2p2_matched_head','attnhp_matched_head']
for ds,dataset in dsmap.items():
    for x in c['validation'][ds]['rows']:
        rows.append({'dataset':ds,'split':'validation','arm':x['arm'],'seed':x['seed'],'selected_epoch':x['best_epoch'],'metrics':{k:x[k] for k in metricmap}})
    for x in c['datasets'][ds]['seed_results']:
        assert x['status']=='complete' and x['split']=='test'
        m=x['metrics'];vals={k:(m if scope=='whole' else m['tail'])[source] for k,(scope,source) in metricmap.items()}
        rows.append({'dataset':ds,'split':'test','arm':arms[x['arm']],'seed':x['seed'],'selected_epoch':x['selected_epoch'],'metrics':vals})
    for split in ['validation','test']:
        for arm in ['CNN','Adaptive','Capacity']:
            rr=[x for x in rows if (x['dataset'],x['split'],x['arm'])==(ds,split,arm)]
            assert sorted(x['seed'] for x in rr)==[42,52,62]
            g={k:{'mean':st.mean(x['metrics'][k] for x in rr),'sample_sd':st.stdev(x['metrics'][k] for x in rr)} for k in metricmap}
            ag=c['validation'][ds]['aggregate'][arm] if split=='validation' else next(x['metrics'] for x in c['datasets'][ds]['grouped'] if arms[x['arm']]==arm)
            for k in metricmap:
                oldkey=k if split=='validation' else testmap[k]
                for stat in ['mean','sample_sd']:eq(g[k][stat],ag[oldkey][stat])
            aggregates.append({'dataset':ds,'split':split,'arm':arm,'metrics':g})
        for a,b in [('Adaptive','CNN'),('Capacity','CNN'),('Adaptive','Capacity')]:
            for seed in [42,52,62]:
                aa=next(x for x in rows if (x['dataset'],x['split'],x['arm'],x['seed'])==(ds,split,a,seed))
                bb=next(x for x in rows if (x['dataset'],x['split'],x['arm'],x['seed'])==(ds,split,b,seed))
                dif={k:{'absolute':aa['metrics'][k]-bb['metrics'][k],'relative_percent':100*(aa['metrics'][k]/bb['metrics'][k]-1) if bb['metrics'][k]!=0 else None} for k in metricmap}
                contrasts.append({'dataset':ds,'split':split,'contrast':a+'−'+b,'seed':seed,'metrics':dif})
        for model in extmodels:
            er=[x for x in all_ext if x['dataset']==dataset and x['split']==split and x['model']==model and x['view']=='overall']
            assert sorted(int(x['seed']) for x in er)==[42,52,62]
            for x in er:
                pin(x['source'],x['source_sha256'])
                if split=='test':
                    receipt=read(x['source'])
                    ref=c['datasets'][ds]['seed_results'][0]
                    assert receipt['status']=='complete' and receipt['target_identity_sha256']==ref['target_identity_sha256']
                    assert receipt['data_file_sha256']==ref['data_sha256'] and receipt['loader']==ref['loader']
                    # Truth hash field lists differ; do not claim the raw digests are equal.
            ms={m:{'mean':st.mean(float(x['qty_'+m.lower()]) for x in er),'sample_sd':st.stdev(float(x['qty_'+m.lower()]) for x in er)} for m in ['RMSE','MAE']}
            external.append({'dataset':ds,'split':split,'model':model,'metrics':ms,'seed_rows':er})
        for m in ['RMSE','MAE']:
            best=min([x for x in external if x['dataset']==ds and x['split']==split],key=lambda x:x['metrics'][m]['mean'])
            frontiers.append({'dataset':ds,'split':split,'metric':m,'model':best['model'],**best['metrics'][m]})
    for arm in ['Adaptive','Capacity']:
        gates=[]
        for split in ['validation','test']:
            aa=next(x for x in aggregates if (x['dataset'],x['split'],x['arm'])==(ds,split,arm))
            bb=next(x for x in aggregates if (x['dataset'],x['split'],x['arm'])==(ds,split,'CNN'))
            for m in ['RMSE','MAE']:
                f=next(x for x in frontiers if (x['dataset'],x['split'],x['metric'])==(ds,split,m))
                a=aa['metrics'][m]['mean'];b=bb['metrics'][m]['mean']
                gates.append({'split':split,'metric':m,'candidate':a,'cnn':b,'external_best':f['mean'],'external_model':f['model'],'pass_cnn':a<b,'pass_external':a<f['mean'],'pass_both':a<min(b,f['mean']),'candidate_minus_cnn':a-b,'candidate_vs_cnn_percent':100*(a/b-1),'candidate_minus_external':a-f['mean'],'candidate_vs_external_percent':100*(a/f['mean']-1)})
        acceptance.append({'dataset':ds,'arm':arm,'gates':gates,'passed_count':sum(x['pass_both'] for x in gates),'all_passed':all(x['pass_both'] for x in gates)})
    ts=c['datasets'][ds]['seed_results']
    for seed in [42,52,62]:
        b=next(x['metrics'] for x in ts if arms[x['arm']]=='CNN' and x['seed']==seed)
        for arm in ['Adaptive','Capacity']:
            a=next(x['metrics'] for x in ts if arms[x['arm']]==arm and x['seed']==seed)
            tail.append({'dataset':ds,'seed':seed,'arm':arm,'whole_sse_delta':a['qty_sse']-b['qty_sse'],'tail_sse_delta':a['tail']['qty_sse']-b['tail']['qty_sse'],'body_sse_delta':(a['qty_sse']-a['tail']['qty_sse'])-(b['qty_sse']-b['tail']['qty_sse']),'tail_count':a['tail']['count']})
            eq(tail[-1]['whole_sse_delta'],tail[-1]['tail_sse_delta']+tail[-1]['body_sse_delta'])
for path,h in F.items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==h,path
out={'status':'complete_stored_result_integration','created_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),'scope':'completed Taxi/RAF Adaptive/Capacity Validation and previously authorized Test; RAF stored Train/Validation error audit as separate evidence','aggregates':aggregates,'seed_rows':rows,'seed_differences':contrasts,'external_models':external,'external_frontiers':frontiers,'user_criteria':acceptance,'test_tail_decomposition':tail,'source_files':F,'verified_scalar_equalities':checks,'new_training':0,'new_inference':0,'new_remote_execution':0,'baseline_modified':False,'automation_created':False,'limitations':['No raw-error averaging across datasets','3 seeds descriptive only; no significance claim','Capacity control adds learned nonlinear history residual; not parameter count alone','Historical external source/training differences remain; Test target/input/loader agreement checked, truth digest schemas differ','Same Taxi5080 environment; RAF candidates5090 versus historical CNN5080','No new candidate Train row predictions extracted','Existing Test informed exploratory development; not independent generalization']}
save('RESULT.json',out)

n={'CNN':'기존 CNN','Adaptive':'이력 적응 CNN','Capacity':'용량 대조군'}
L=[]
def add(x=''):L.append(x)
add('# Taxi 개선 실험과 RAF 결과의 통합 검토')
add('\n**Taxi에서는 개선 신호가 남아 있다. 다만 이력 적응 방식의 공통 효과와 두 데이터셋에서의 사용자 기준 충족은 아직 확인되지 않았다.** RAF만으로 전체 백본 개선 가능성을 닫는 결론은 과도하다. 반대로 Taxi 평균 개선만으로 다음 학습을 자동으로 정당화하지 않는다.')
add('\n## 현재 기준선과 범위 — 완료')
add('\nMaterial Passport: academic-research-suite / stored-result integration / 2026-10-16. Verification Status: ANALYZED. 어제 완료된 Taxi·RAF의 기존CNN/Adaptive/Capacity를 같은 표에 놓고, 앞선 RAF Train·Validation 오류 분석을 별도 근거로 연결했다. 이미 승인·완료된 고정 checkpoint Validation/Test 결과만 재사용했다. 새 학습·모델 추론·원격 실행·checkpoint 변경·기준선 교체·자동화 생성은0이다.')
add(f'\nTaxi 평가323파일·RAF 평가321파일의 기록SHA와 보고서 원문을 검증했다. 외부 모델은 기존6종의 저장된3seed 결과만 비교했다. 고유{len(F)}파일을 확인하고 {checks}개 수치 일치를 검산했다. 외부Test는 같은 target identity·입력SHA·loader를 확인했다. truth hash는 필드목록이 서로 달라 rawSHA 동일로 표현하지 않는다. Validation은 기존 동결 endpoint·공유 population 연결을 재사용하며 새 평가로 부르지 않는다.')
add('\n용량 대조군은 단순히 숫자상 파라미터 수만 늘린 모델이 아니다. Adaptive와 같은 현재/차분 입력,3→16→1 MLP,추가243개 파라미터를 사용한다. Adaptive는 입력에 따라tap 비율을 조절하고, Capacity는 비선형 잔차를 더한다. 따라서 Capacity가 좋다는 결과를 “용량 증가가 원인”으로 확정할 수 없다. 공통 입력/log1p·손실·head·메모리·선택 규칙은 유지했다.')
add('\n## 데이터셋별 전체 수량 성능')
add('\n3seed 평균±표본SD(ddof1). 같은 단위라도 서로 다른 데이터셋의 raw오류를 평균내지 않았다. 낮을수록 좋다.')
add('\n| 데이터 | 구조 | Validation RMSE | Validation MAE | Test RMSE | Test MAE |\n|---|---|---:|---:|---:|---:|')
for ds in dsmap:
    for arm in n:
        vals=[]
        for split in ['validation','test']:
            x=next(x for x in aggregates if (x['dataset'],x['split'],x['arm'])==(ds,split,arm))
            for m in ['RMSE','MAE']:
                v=x['metrics'][m];vals.append(f"{v['mean']:.6f} ± {v['sample_sd']:.6f}")
        add('| '+ds+' | '+n[arm]+' | '+' | '.join(vals)+' |')
add('\n## Taxi에서 남는 개선 근거와 한계')
add('\n- Adaptive는 기존CNN 대비 Validation RMSE−4.00%,MAE−4.55%이며 세 seed 모두 두 지표가 개선됐다. Test평균도 RMSE−4.28%,MAE−4.71%다. 이 개선을 없는 것으로 취급하면 안 된다.\n- Capacity도 CNN 대비 양 split의 두 평균 지표가 모두 낮다. Test평균은 RMSE−13.72%,MAE−11.26%로 Adaptive보다 낮다. 그러나 Capacity의 Test RMSE 개선은 seed42에서만 발생했다. 기존CNN42가 특히 큰 오차를 보인 영향이 평균에 크다.\n- Adaptive−Capacity는 Validation 평균RMSE−1.688994/MAE−0.808377로 유리하지만 Test에서는+12.767772/+2.658555로 불리하다. Test에서 seed42·62는 Adaptive가 낫고52는 크게 악화한다. 두 seed가 이겨도 한 seed의 큰 손실 때문에 평균 순위가 바뀐다.\n- Adaptive52는 자체Validation RMSE82.469324로 세 seed 중 가장 낮지만 Test166.783347로 가장 높다. 이는 선택된 모델의 split별 순위 변화이며 과적합 원인이나 선택 오류를 확정한 것이 아니다.\n- Test에서 Capacity를 보고 지금 새 기준선으로 교체하지 않는다. 기존CNN은 계속 고정된 비교 기준선이다.')
add('\n### Taxi Test의 seed별 실제 결과')
add('\n| seed | CNN RMSE / MAE | Adaptive RMSE / MAE | Capacity RMSE / MAE |\n|---:|---:|---:|---:|')
for seed in [42,52,62]:
    vv=[]
    for arm in n:
        x=next(x['metrics'] for x in rows if (x['dataset'],x['split'],x['arm'],x['seed'])==('Taxi','test',arm,seed));vv.append(f"{x['RMSE']:.3f} / {x['MAE']:.3f}")
    add('| '+str(seed)+' | '+' | '.join(vv)+' |')
add('\n### 큰수량과 시간 지표도 유리한 결과만 고르지 않는다')
add('\nTaxi Test의 큰수량(q>3449) 평균RMSE/MAE는 CNN735.979104/580.234081, Adaptive704.856457/565.834068, Capacity587.667931/466.297437이다. 그러나 Adaptive62는 전체RMSE가 낮아져도 큰수량 제곱오차는 증가했다. Adaptive52의 전체 제곱오차 악화 중 큰수량이 차지하는 부분은 약29.4%여서 큰수량만의 문제라고 할 수 없다.\n전체 Test TimeNLL은 CNN0.767969,Adaptive1.239752,Capacity0.908386으로 두 후보가 모두 악화했다. 여섯 지표 전체와 seed별 차이는 아래 부록과 RESULT.json에 보존했다. 작은 TimeNLL의 상대%는 분모가 작아 과장되므로 본문 판단에는 절대값을 사용한다.')
add('\n## 외부 지표별 최우수까지 포함한 사용자 기준')
add('\n외부6종은 동일 관측 수치형 입력·공통 head에 맞춘 저장 비교 구현이다. 원논문 전체 구현/최적튜닝의 보편적 우열로 확대하지 않는다. 각 split에서RMSE·MAE 모두 기존CNN과 외부 지표별최우수보다 엄격히 낮아야 한다.')
add('\n| 데이터 | split | 외부 최저RMSE(모델) | 외부 최저MAE(모델) |\n|---|---|---:|---:|')
for ds in dsmap:
    for split in ['validation','test']:
        a=next(x for x in frontiers if (x['dataset'],x['split'],x['metric'])==(ds,split,'RMSE'));b=next(x for x in frontiers if (x['dataset'],x['split'],x['metric'])==(ds,split,'MAE'))
        add(f"| {ds} | {split} | {a['mean']:.6f} ({a['model']}) | {b['mean']:.6f} ({b['model']}) |")
add('\n| 데이터 | 후보 | Val RMSE | Val MAE | Test RMSE | Test MAE | 네조건 전체 |\n|---|---|---|---|---|---|---|')
for x in acceptance:add('| '+x['dataset']+' | '+n[x['arm']]+' | '+' | '.join('통과' if g['pass_both'] else '미달' for g in x['gates'])+' | '+('통과' if x['all_passed'] else '미달')+' |')
add('\nTaxi Capacity가 네조건 중3개로 가장 가깝다. Test MAE36.043058은 S2P2의36.835852보다 낮지만 RMSE116.692983은115.541513보다1.151470(약1.00%) 높다. 근소한 차이라도 사용자의 엄격한 기준에서는 미달이며,3seed만으로 유의하거나 동등하다고 판단하지 않는다. Taxi Adaptive는Validation두조건을 통과하지만Test외부최우수의 두지표를 넘지 못했다.')
add('\n## RAF와 합친 방법론 판단')
add('\n- 같은 Adaptive 구조는RAF에서 Validation/Test의RMSE·MAE 평균이 모두 기존CNN보다 높다. Capacity대비도 두split 네평균지표 모두 높다. Taxi의일부이득을 두데이터셋 공통이력적응기여로 바꿀 수 없다.\n- RAF의 저장 Train·Validation 오류 분석에서는외부모델도RMSE·MAE우위가갈렸고, 한관측조건의RMTPP이득은소수사례에민감했다. 이결과는RAF의다음구조선정근거가약함을보여주지만Taxi에서실제로관측된개선을무효화하지않는다.\n- 따라서현재결론은 **Taxi개선신호는있음 / 이력적응의추가효과는불안정 / 공통구조의두데이터셋성공은미확보 / 사용자네조건전체통과는없음**이다. 앞선RAF중단권고를전체CNN연구의가능성부재로넓히지않는다.')
add('\n## 다음 작업을 좁힌다 — 판단 완료')
add('\n무작위구조변경과자동반복학습을멈춘상태는유지한다. 다음분석의우선순위가필요하다면Taxi에서기존CNN42와Adaptive52의큰오류가어느관측사례에집중되는지,Capacity가이를어떻게바꿨는지저장행으로비교하는것이다. 이번통합은완료성능·수량구간집계이며새후보의Train/Validation행예측을추출하거나오류원인을확정하지않았다. 관측가능한일관된차이가확인되기전새CNN모듈을선정하지않는다. 기존CNN기준선·선택checkpoint·학습예산은그대로다. 새학습이나자동화재개를이보고서에서실행하지않았다.')
add('\n## 전체 seed별 차이')
add('\n차이는 후보−비교대상,음수는개선이다. 전체원값/상대%/여섯지표는RESULT.json에있다.')
add('\n| 데이터 | split | seed | 대조 | ΔRMSE | ΔMAE |\n|---|---|---:|---|---:|---:|')
for x in contrasts:add(f"| {x['dataset']} | {x['split']} | {x['seed']} | {x['contrast']} | {x['metrics']['RMSE']['absolute']:+.6f} | {x['metrics']['MAE']['absolute']:+.6f} |")
add('\n## 여섯 지표의 평균±표본SD')
add('\n| 데이터 | split | 구조 | RMSE | MAE | 큰수량RMSE | 큰수량MAE | TimeNLL | 큰수량TimeNLL |\n|---|---|---|---:|---:|---:|---:|---:|---:|')
for x in aggregates:
    vals=[]
    for k in metricmap:
        z=x['metrics'][k];vals.append(f"{z['mean']:.6g} ± {z['sample_sd']:.6g}")
    add('| '+x['dataset']+' | '+x['split']+' | '+n[x['arm']]+' | '+' | '.join(vals)+' |')
add('\n## 한계와 보존')
add('\nTaxi세구조는5080/cuDNN91900이다. RAF신규후보는5090/cuDNN92000,기존CNN은5080/cuDNN91900이므로CNN대비환경차이가있다. 같은5090의Adaptive−Capacity대조도함께보고했다. 같은파라미터수는같은함수공간/최적화난도라는뜻이아니다. Test를본뒤설계한탐색이며새독립일반화·원인확정·통계유의성을주장하지않는다. 서로다른데이터셋/모델의유리한결과를합쳐가상의공통모델로만들지않는다. 선택은원래ValidationrawRMSE최초strictfinite최소epoch로고정했으며이번비교에서재선택하지않았다.')
add('\n주요원본: reports/titantpp_history_taxi6_test_20261015_v1/TAXI_RAF_COMPARISON.json, reports/titantpp_three_dataset_final_comparison_20261005_v1/metrics_per_seed.csv 및해당receipt, reports/titantpp_cnn_external_error_audit_20261016_v1/RESULT.json. 기존산출물은수정하지않고별도통합보고서를썼다.')
with (O/'RESULT.md').open('x') as f:f.write('\n'.join(L)+'\n')
save('RECEIPT.json',{'status':'complete','source_files_verified_unchanged':len(F),'scalar_checks':checks,'result_sha256':hashlib.sha256((O/'RESULT.json').read_bytes()).hexdigest(),'report_sha256':hashlib.sha256((O/'RESULT.md').read_bytes()).hexdigest(),'new_training':0,'new_inference':0,'remote_execution':0,'baseline_mutation':0})
print(json.dumps({'status':'complete','sources':len(F),'checks':checks,'acceptance':[{k:x[k] for k in ['dataset','arm','passed_count','all_passed']} for x in acceptance]},ensure_ascii=False))
