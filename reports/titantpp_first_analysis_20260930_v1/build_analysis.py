"""First synthesis of approved validation experiments; no model/data execution."""
from pathlib import Path
import csv
import datetime as dt
import hashlib
import json
import math
import statistics as st

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CORE = ROOT / 'search_artifacts/titantpp_core_ablation_20260928_v1'
LEDGER = CORE / 'aggregation/20260929T232132255352Z/ledger.json'
SNAPSHOT = CORE / 'monitor/20260929T232108Z/5090/snapshot.json'
EXTERNAL = ROOT / 'reports/local_detail_3seed_final_20260927_v1/condition_results.json'
GATE = ROOT / 'reports/titantpp_mlp_gate_execution_20260929_v1/comparison.json'
DATASETS = {'yellow_trip_hourly': 'Taxi', 'intermittent_frozen_5000': 'Intermittent', 'insta_market_basket': 'Instacart'}
LABELS = {'titantpp':'B · 이력 보완 없음', 'titantpp_local_detail':'Full', 'titantpp_history_mlp':'이력 MLP', 'titantpp_level_only':'수준만', 'titantpp_change_only':'변화만', 'titantpp_no_static_lmm':'정적 검색 제거', 'rmtpp':'RMTPP', 'thp':'THP', 'nhp':'NHP', 'sahp':'SAHP'}
METRICS = ['qty_mae', 'qty_rmse', 'time_nll']
SOURCES = {}
def read(p):
    p=Path(p); raw=p.read_bytes(); SOURCES[str(p.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest(); return json.loads(raw)
def stats(v):
    return {'n':len(v), 'mean':st.mean(v), 'sample_sd':st.stdev(v) if len(v)>1 else None}
def kst(t):
    return dt.datetime.fromtimestamp(t,dt.timezone(dt.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST')
def fmt(x):
    return f'{x["mean"]:.4f} ± {x["sample_sd"]:.4f}'
def mdtable(headers,rows):
    return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows]

def main():
    ledger=read(LEDGER); snapshot=read(SNAPSHOT); external=read(EXTERNAL); gate=read(GATE)
    contract=read(CORE/'frozen_execution/execution_contract.json')
    assert hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()==ledger['contract_sha256']
    assert ledger['counts']['new_audited']==36 and ledger['counts']['reuse_audited']==18 and ledger['counts']['audit_failures']==0
    assert snapshot['files']['recovery_5090_v1/status.json']['status']=='complete'
    assert snapshot['files']['run/partition_summary.json']['new_fits']==12
    assert not snapshot['owned_processes'] and not snapshot['gpu_processes']['stdout'].strip()
    core=ledger['conditions']; assert len(core)==54
    index={(x['dataset'],x['seed'],x['arm']):x for x in core}; assert len(index)==54
    # Verify the reused Full/B entries are identical to the external campaign entries.
    overlap=0
    for e in external:
        if e['model'] in ['titantpp','titantpp_local_detail']:
            r=index[e['dataset'],e['seed'],e['model']]
            for endpoint in ['selected','last']:
                assert all(e[endpoint][m]==r[endpoint][m] for m in METRICS)
                assert e[endpoint]['state_sha256']==r[endpoint]['state_sha256']
            overlap+=1
    assert overlap==18
    ext=[{**x,'arm':x['model'],'campaign':'external_controlled'} for x in external if x['model'] in ['rmtpp','thp','nhp','sahp']]
    rows=[{**x,'campaign':'core'} for x in core]+ext
    assert len(rows)==90 and len({(r['dataset'],r['seed'],r['arm']) for r in rows})==90
    for r in rows:
        for endpoint in ['selected','last']:
            v=r[endpoint];assert v['evaluation_scope']=='validation_only' and not v['held_out_test_evaluated']
            assert all(math.isfinite(v[m]) for m in METRICS)
            assert math.isclose(v['qty_rmse']**2*v['count'],v['qty_sse'],rel_tol=1e-10)
        if r['campaign']=='core':
            assert r['execution_status']=='complete' and r['audit_status']=='passed'
            h=read(ROOT/r['source']/'history.json')['history']
            r['first40']=stats([x['val_qty_rmse'] for x in h[:40]])
            assert min(range(len(h)),key=lambda i:h[i]['val_qty_rmse'])+1==r['best_epoch']
            assert len(h)==r['completed_epochs']
    idx={(r['dataset'],r['seed'],r['arm']):r for r in rows}
    aggregates=[]; amap={}; segments=[]; costs=[]
    for ds in DATASETS:
        for m in LABELS:
            rr=[idx[ds,s,m] for s in [42,52,62]]
            a={'dataset':ds,'dataset_label':DATASETS[ds],'model':m,'model_label':LABELS[m],'seeds':[42,52,62], 'selected':{k:stats([r['selected'][k] for r in rr]) for k in METRICS},'last':{k:stats([r['last'][k] for r in rr]) for k in METRICS},'validation_targets_per_seed':rr[0]['selected']['count']}
            for r in rr: assert r['selected']['count']==a['validation_targets_per_seed']
            aggregates.append(a);amap[ds,m]=a
            for group in ['body','tail','quantity_cells','history_cells','additional_history_cells']:
                vv=[r['selected'][group] for r in rr]
                for i in range(len(vv[0]) if isinstance(vv[0],list) else 1):
                    cells=[v[i] if isinstance(v,list) else v for v in vv];n=cells[0]['count'];assert all(c['count']==n for c in cells)
                    segments.append({'dataset':ds,'model':m,'group':group,'bin':i if isinstance(vv[0],list) else None,'count_per_seed':n,**{k:stats([c[k] for c in cells]) if n else None for k in METRICS}})
            if m in list(LABELS)[:6]:
                costs.append({'dataset':ds,'model':m,'parameters':rr[0]['parameters'],'recorded_fit_seconds':stats([r['fit_elapsed_seconds'] for r in rr]),'completed_epochs':stats([r['completed_epochs'] for r in rr]),'peak_allocated_mib':stats([r['peak_allocated_bytes']/2**20 for r in rr]),'standalone_eligible_seeds':[r['seed'] for r in rr if not r['timing_interference_recorded'] and not r.get('restoration')], 'notes':'Recorded train_one intervals; recovery can omit pre-reboot time. Concurrent intervals overlap. Not inference cost or additive GPU device hours.'})
    comparisons=[]
    for ds in DATASETS:
        for reference in list(LABELS)[:6]:
            if reference=='titantpp_local_detail':continue
            pairs=[]
            for seed in [42,52,62]:
                full,ref=idx[ds,seed,'titantpp_local_detail'],idx[ds,seed,reference];x,y=full['selected'],ref['selected']
                checks={'overall_rmse':x['qty_rmse']<y['qty_rmse'],'overall_mae':x['qty_mae']<=1.01*y['qty_mae'],'body_mae':x['body']['qty_mae']<=1.02*y['body']['qty_mae'],'tail_rmse':x['tail']['qty_rmse']<y['tail']['qty_rmse'],'tail_mae':x['tail']['qty_mae']<=1.02*y['tail']['qty_mae'],'time_nll':x['time_nll']<=y['time_nll']+.01,'last30_mean':full['last30']['mean']<=1.05*ref['last30']['mean'],'last30_sd':full['last30']['sd']<=1.05*ref['last30']['sd']}
                for i in range(1,4):
                    if x['quantity_cells'][i]['count']:
                        for metric in ['qty_mae','qty_rmse']:checks[f'quantity_bin{i}_{metric}']=x['quantity_cells'][i][metric]<=1.02*y['quantity_cells'][i][metric]
                pairs.append({'seed':seed,'differences':{k:x[k]-y[k] for k in METRICS},'checks':checks,'passed_all':all(checks.values())})
            comparisons.append({'dataset':ds,'candidate':'titantpp_local_detail','reference':reference,'pairs':pairs,'mae_wins':sum(p['differences']['qty_mae']<0 for p in pairs),'rmse_wins':sum(p['differences']['qty_rmse']<0 for p in pairs),'gate_passes':sum(p['passed_all'] for p in pairs),'mae_reduction_pct':100*(1-amap[ds,'titantpp_local_detail']['selected']['qty_mae']['mean']/amap[ds,reference]['selected']['qty_mae']['mean']),'rmse_reduction_pct':100*(1-amap[ds,'titantpp_local_detail']['selected']['qty_rmse']['mean']/amap[ds,reference]['selected']['qty_rmse']['mean'])})
    external_effects=[]
    for ds in DATASETS:
        for model in ['titantpp','titantpp_local_detail','titantpp_history_mlp']:
            row={'dataset':ds,'model':model}
            for metric in ['qty_mae','qty_rmse']:
                best=min(['rmtpp','thp','nhp','sahp'],key=lambda m:amap[ds,m]['selected'][metric]['mean']);rmean=amap[ds,best]['selected'][metric]['mean'];cmean=amap[ds,model]['selected'][metric]['mean']
                row[metric]={'reference':best,'reference_mean':rmean,'candidate_mean':cmean,'reduction_pct':100*(1-cmean/rmean),'paired_wins':sum(idx[ds,s,model]['selected'][metric]<idx[ds,s,best]['selected'][metric] for s in [42,52,62]),'wins_against_all_four_per_seed':sum(all(idx[ds,s,model]['selected'][metric]<idx[ds,s,b]['selected'][metric] for b in ['rmtpp','thp','nhp','sahp']) for s in [42,52,62])}
            external_effects.append(row)
    observed=kst(snapshot['observed_at_unix']);ended=kst(snapshot['files']['recovery_5090_v1/status.json']['completed_at_unix'])
    evidence_status={'as_of':observed,'core_5090_finished':ended,'core_completed_conditions':54,'core_new_conditions':36,'core_reused_conditions':18,'core_endpoint_roles':108,'external_additional_conditions':36,'deduplicated_core_external_conditions':90,'MLP_gate_conditions':4,'Full_gate_completed':2,'Full_gate_held_not_started':2,'core_json_evidence_audit_failures':0,'final_core_binary_source_audit_complete':False,'no_new_training_or_GPU_replay':True,'initial_sandbox_SSH_failure':'Operation not permitted, local sandbox; authorized read succeeded outside sandbox. Not a training failure.'}
    payload={'status':'first_analysis_not_final_core_binary_audit','evaluation_scope':'validation_only','evidence_status':evidence_status,'selection':'earliest strict validation raw quantity RMSE minimum; MAE/time/strata at identical checkpoint','seed_statistic':'equal weight seeds42,52,62; sample SD ddof1, not standard error or confidence interval','aggregates':aggregates,'core_contrasts':comparisons,'external_effects':external_effects,'strata':segments,'costs':costs,'core_and_external_conditions':rows,'gate_results':gate,'acceptance':contract['comparison']['acceptance'],'sources_sha256':SOURCES}
    (OUT/'analysis.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'three_seed_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['dataset','model','seed_count']+[f'{k}_{s}' for k in METRICS for s in ['mean','sample_sd']]);writer.writeheader()
        for a in aggregates:writer.writerow({'dataset':a['dataset'],'model':a['model'],'seed_count':3,**{f'{k}_{s}':a['selected'][k][s] for k in METRICS for s in ['mean','sample_sd']}})
    lines=['# TitanTPP 1차 분석: 수량 예측의 경쟁력과 구조별 한계','','## 핵심 판단','',
      '- **외부 TPP 대비 수량 개선은 유지된다.** Full과 이력 MLP 각각 하나의 고정 모델로 Taxi·Intermittent에서 기존 RMTPP·THP·NHP·SAHP보다 MAE·RMSE가 낮다. 데이터별로 유리한 모델을 골라야만 성립하는 결과가 아니다.',
      '- **Full의 분리 구조나 조건부 Gate의 보편적 우월성은 확인되지 않았다.** Taxi 평균 RMSE는 Full이 MLP보다1.17% 낮지만, Intermittent는 MLP가 Full보다7.77% 낮다. Gate seed42 탐색도 조건부의 수량 개선을 보여주지 못했다.',
      '- **5090 결과까지 포함해도 Instacart의 이득은 제한적이다.** 변화만 모델의 평균 RMSE5.8785가 core 중 가장 낮지만, 외부 THP5.8582보다 낮지 않다. 이력 보완 구조의 차이가 세 데이터에 똑같이 작동하지 않는다.',
      '- 현재 논문 중심 주장은 **사건 이력의 시간·수량 정보를 이용한 수량 예측 표현의 경쟁력과 적용 범위**로 잡을 수 있다. “모든 데이터·지표에서 우월”, “조건부 Gate가 핵심 개선 원인”, “원본 Titans보다 가볍고 빠름”은 현재 증거로 확정하지 않는다.','',
      '## 현재 실행 상태와 분석 범위','',f'5090 관측 기준: **{observed}**. 마지막 정적 검색 제거 seed62는129epoch(선택89)에서 끝났으며, 서버 전체 core 종료 시각은 **{ended}**다. 학습 소유 프로세스와 GPU compute process는 없다. 5080 Gate는2026-09-29 23:25:25 KST에 완료됐다.','',
      'core54조건/108 selected·last 역할은 신규36조건과 재사용 B·Full18조건으로 구성된다. 소형 기록의 선택·초기화·노출·배치 prefix·재평가 수집 감사는54조건 모두 통과했다. **5090 checkpoint binary 및 최종 source 회수 감사는 아직 별도 남아 있어 이 문서는1차 분석이다.** 5080 core와 Gate의 기존 terminal 감사를 재사용했다.','',
      '외부 비교는 완료된4모델×3데이터×3seed=36조건을 추가한다. 외부 캠페인의 B·Full18조건이 core와 정확히 겹침을 endpoint 값·state SHA로 확인하고 중복 제외했다. core+외부는90개 고유 조건이다. Gate는 Full2조건/MLP4조건의 별도 seed42 탐색이며 위3seed 평균에 섞지 않았다.','',
      '## 평가 기준과 모델의 의미','',
      '평가 split은 validation만이다. 모든 MAE·RMSE·시간 NLL은 각 조건의 **최초 strict 수량 RMSE 최소 checkpoint**에서 함께 읽었다. MAE 최소 epoch로 다시 선택하지 않았다. ±는3seed 표본표준편차(ddof=1)이며 통계적 유의성·동등성 또는 독립 검증을 의미하지 않는다. 데이터별 수량 척도가 달라 RMSE를 데이터 간 합산하지 않았다.','',
      '이력 MLP는 TitanTPP 전체를 단순 MLP로 바꾼 모델이 아니다. **동일 CountAwareTitanTPP encoder 안의 이력 보완 모듈**을 현재/직전 표현을 결합하는 MLP로 바꾼 비교다. Full과 MLP의 추가 파라미터는각6,144개로 같지만 FLOPs는 맞추지 않았다. 수준만·변화만은각4,096개로 용량도 줄어든다. 정적 검색 제거는 마지막 검색만 제거하고 encoder persistent memory는 유지한다.','',
      'RMTPP·THP·NHP·SAHP는 공통 예측 head/loss, 같은 데이터 분할·관측 입력·batch128/max300/min40/patience40·고정 최적화/선택 기준을 사용한 비교 구현이다. 이는 해당 조건의 표현 비교이며 원논문의 최적 튜닝 native 모델 전체에 대한 보편적 순위가 아니다. 시간은 현재 계약의 recorded_positive_integer_time_nll이며 v0.7의 과거 clamped 점수와 혼합하지 않았다.','']
    for ds,label in DATASETS.items():
        lines += [f'## {label}: 3seed 동일 선택 checkpoint','']
        lines += mdtable(['모델','수량 MAE ↓','수량 RMSE ↓','시간 NLL ↓'],[[LABELS[m]]+[fmt(amap[ds,m]['selected'][k]) for k in METRICS] for m in LABELS])
        if label=='Taxi':lines += ['','Full이 수량 RMSE 평균 최저이고, 정적 검색 제거가 MAE 평균 최저다. Full은 MLP 대비 RMSE 평균1.17% 개선이지만 seed52에서는 MLP가 더 좋다. MAE는 Full이 3개 seed 중 1개에서만 낮다. RMSE 개선을 모든 지표 우위나 분리 구조의 필수성으로 해석하지 않는다. 시간 NLL은 Full1.1822, MLP1.0387로 B0.7247보다 높다.','']
        elif label=='Intermittent':lines += ['','이력 MLP가 core의 MAE·RMSE 평균 최저다. Full 대비 RMSE는3seed 모두 낮지만 MAE는 MLP가 3개 seed 중 2개에서 낮다. 수준만도 Full보다 낮은 평균 RMSE를 보인다. 기존 MLP 자체의 경쟁력은 유지되며 조건부 Gate를 붙여야 성립하는 결과가 아니다. 시간 NLL은 수량 순위와 다르며 외부 RMTPP의0.2748이 MLP0.4971보다 낮다.','']
        else:lines += ['','새로운 최종 결과에서 변화만이 core 평균 MAE·RMSE 최저지만 격차는 작다. 모든 core 변형의 RMSE 평균은5.8785–5.8896 범위다. 지표별 외부 최저는 MAE RMTPP3.9852, RMSE THP5.8582, 시간 SAHP2.8018이다. 수치가 가깝다는 것은 통계적 동등성 입증이 아니다. 마지막 정적 검색 제거 seed62는MAE3.9953/RMSE5.9008이며 불리한seed도 포함했다.','']
    lines += ['## 외부 비교군 대비 효과 크기','', '아래 감소율은 (외부 비교군 평균−후보 평균)/외부 평균×100이다. 음수는 후보가 나쁘다는 뜻이다. 지표별 외부 최저 모델이 다르면 각각 명시했다.','']
    lines += mdtable(['데이터','후보','MAE 기준 비교군','MAE 감소','RMSE 기준 비교군','RMSE 감소','같은seed 승리 MAE/RMSE'],[[DATASETS[x['dataset']],LABELS[x['model']],LABELS[x['qty_mae']['reference']],f"{x['qty_mae']['reduction_pct']:+.2f}%",LABELS[x['qty_rmse']['reference']],f"{x['qty_rmse']['reduction_pct']:+.2f}%",f"{x['qty_mae']['paired_wins']}/3 · {x['qty_rmse']['paired_wins']}/3"] for x in external_effects])
    lines += ['','Full과 이력 MLP는 Taxi·Intermittent에서 두 수량 지표 모두 외부4모델에 대해 같은seed3/3 개선이다. 현재 비교 구현의 일관된 효과를 뒷받침하지만 최신 강한 비교군·native head·모델별 공정한 튜닝 예산까지 검증한 결과는 아니다.','', '## Full 고유 구조의 기여와 반례','']
    lines += mdtable(['데이터','Full 대비 기준','Full MAE 감소','Full RMSE 감소','MAE/RMSE 승리seed','전체 보호기준 통과'],[[DATASETS[x['dataset']],LABELS[x['reference']],f"{x['mae_reduction_pct']:+.2f}%",f"{x['rmse_reduction_pct']:+.2f}%",f"{x['mae_wins']}/3 · {x['rmse_wins']}/3",f"{x['gate_passes']}/3"] for x in comparisons])
    lines += ['','전체 보호기준은 원래 계약의 수량 전체/body/tail·중간수량bin1–3·시간NLL(+0.01)·last30 평균/표준편차 한도를 그대로 적용했다. Full 대 MLP가 핵심 사전 비교이며 다른 구조에 대한 표는 같은 기준의 탐색적 적용이다. 실행 완료와 성능 기준 통과는 별개다. 모든seed의 실패 항목은 analysis.json에 보존한다.','', '### 구간별 수량 오차','']
    def seg(ds,m,g,i=None):return next(x for x in segments if x['dataset']==ds and x['model']==m and x['group']==g and x['bin']==i)
    lines += mdtable(['데이터','모델','body MAE','tail MAE','tail RMSE','tail 표본/seed'],[[DATASETS[ds],LABELS[m]]+[f"{seg(ds,m,g)[k]['mean']:.4f}" for g,k in [('body','qty_mae'),('tail','qty_mae'),('tail','qty_rmse')]]+[seg(ds,m,'tail')['count_per_seed']] for ds in DATASETS for m in ['titantpp','titantpp_local_detail','titantpp_history_mlp','titantpp_no_static_lmm']])
    lines += ['','tail·수량 경계는 train에서 고정한 값이며 데이터마다 다르다. Instacart의 기본 이력>64 구간은 비어 있으므로0점으로 채우지 않았고, 추가1/3/7/15/31 경계의 모든 원값은 기계 판독 결과에 보존했다. 평균 성능 개선이 모든 구간 개선을 뜻하지 않는다.','', '## Gate 탐색: seed42만 별도 비교','']
    gr=[]
    for g in gate['MLP_Gate_new_conditions']:
        v=g['endpoint_replays']['selected']; gr.append([DATASETS[g['dataset']],g['queue_key'],f'{v["qty_mae"]:.6f}',f'{v["qty_rmse"]:.6f}',f'{v["time_nll"]:.6f}',f'{g["best_epoch"]}/{g["completed_epochs"]}',f'{g["vs_plain_mlp"]["mae_change_pct"]:+.2f}%',f'{g["vs_plain_mlp"]["rmse_change_pct"]:+.2f}%'])
    lines+=mdtable(['데이터','조건','MAE','RMSE','시간 NLL','선택/종료epoch','plain대비 MAE변화','plain대비 RMSE변화'],gr)
    lines += ['','MLP A/C는 조건부, B/D는 학습 가능한 상수다. 비교 기준도 동일seed42 plain MLP이며3seed 평균과 비교하지 않았다. Intermittent plain의MAE0.638480/RMSE1.578487, Taxi plain의26.499368/82.483934를 기준으로 변화율을 계산했다. 양수는 악화다.','',
      '조건부 Gate는 두 데이터에서 기존 MLP보다 MAE·RMSE가 높다. Intermittent A는5epoch가 최종best이고45epoch에 종료됐다. Taxi D 상수는 수량 MAE1.42%·RMSE2.76% 개선했지만 시간NLL은0.842735→1.304540으로 악화했다. Taxi Full seed42의MAE25.040551/RMSE77.616782도 상수보다 낮다. 따라서 상수 결과만으로 최종 대표 모델을 교체할 근거도 아직 충분하지 않다.','',
      '기존 Full Gate A/B는 Intermittent 변화 경로만 조절한 별도 실험이다. 조건부MAE0.731695/RMSE1.782706, 상수0.668810/1.699489였다. Full Taxi C/D는 미시작 보류이며 실패로 집계하지 않는다. Gate 새fit은 baseline warm start 없이 공통 초기화에서 시작했다.','',
      '## Contribution을 현재 증거에 맞춰 정리하면','']
    lines += mdtable(['주장','현재 판단','필요한 근거 또는 범위'],[
      ['사건 시간·수량 이력을 이용하는 TitanTPP 표현이 수량 예측에 경쟁력이 있다','지지됨 — 현재 공통head validation 범위','Full·MLP 각각 Taxi/Intermittent 외부4모델 대비 같은seed3/3 수량 개선'],
      ['수준·변화 분리가 필수다','전반적 지지 없음','Taxi Full의 일부 이점과 Intermittent MLP 우위를 함께 보고'],
      ['조건부 Gate가 핵심 성능 기여다','이번 탐색에서 지지되지 않음','두 데이터seed42에서 plain MLP 수량 지표 미개선; 시간 이점은 별도'],
      ['정적 검색을 덜어도 표현 경쟁력을 유지할 수 있다','제한적·탐색적 근거','Full 대비 전체 RMSE 차이가 작지만 수량/시간·seed 반례 존재; persistent memory는 남음'],
      ['원본 Titans의 무거운 연산을 줄여 더 가볍고 빠르다','아직 입증하지 못함','현재 파라미터/allocator 기록만으로 원본 Titans 대비 효율 입증 불가; 동일장치/입력/정밀도 속도·메모리 비교 필요'],
      ['Instacart의 짧은 이력·약한 인접 관계가 성능 한계 원인이다','설명 가설','관측 상관은 인과 증명 아님; 현재 보고서는 적용 범위의 한계로 기술']])
    lines += ['','**논문용 잠정 문장:** 사건 이력의 시간·수량 정보를 활용하는 TitanTPP 표현을 구성하고, 공통 예측 head와 통제된 학습 조건에서 Taxi와 Intermittent의 다음 사건 수량 예측이 기존 TPP 비교 구현보다 일관되게 개선됨을 확인했다. 구조 비교는 복잡한 분리·게이팅이 항상 추가 이득을 주지는 않으며, Instacart의 제한적 개선과 시간 예측의 상충관계를 보여준다.','',
      '방법 차별성은 “기존 Titan 적용”이라는 이름만으로 확보되지 않는다. 최종 설명에는 관측 가능한 사건 이력만 사용하는 입력·인과적 마스킹, 수량에 맞춘 표현/보정의 삽입 위치, 원본 Titans와 다른 메모리 업데이트/검색 설계를 코드·수식·제거 실험으로 연결해야 한다. 현재 MLP는 독립 MLP benchmark가 아니라 동일 encoder 내 대안이라는 점을 명시한다. 최종 대표 구조는 validation 검토 후 전체 데이터에 공통으로 고정하며 데이터별 사후 선택을 하지 않는다.','',
      '## 실측 비용과 측정 한계','']
    lines += mdtable(['데이터','core모델','파라미터','기록fit 평균분','종료epoch 평균','peak allocated 평균MiB','단독속도비교 가능seed수'],[[DATASETS[c['dataset']],LABELS[c['model']],c['parameters'],f"{c['recorded_fit_seconds']['mean']/60:.2f}",f"{c['completed_epochs']['mean']:.1f}",f"{c['peak_allocated_mib']['mean']:.2f}",len(c['standalone_eligible_seeds'])] for c in costs])
    lines += ['','기록fit은 train_one 구간이며 별도 endpoint replay를 제외한다. 5090의 병렬·진단 간섭 및 복구 조건은 단독 속도 비교에 넣지 않는다. 복구 summary는 복구 이후 시간만 기록할 수 있으므로 이전 구간과 구분한다. 겹친fit 합을 GPU장치 사용시간으로 해석하거나 누락 구간을 추정 보정하지 않았다. 장치와 NumPy/cuDNN/커널 환경도5080/5090 사이에 차이가 있으므로 데이터 간 속도 비교는 하지 않는다.','',
      '5080 MLP Gate4fit 합은14,401.646초(약4시간)이며 Full Gate 최초 시작부터 새 큐 종료까지14.356시간은 준비·실패·대기·복구를 포함한다. 개인5080 추가cloud 임대료는0달러, 전기료는 미측정이다. 5090의 비용은 실측 시간·메모리로 기록하며 이번 분석에서 전기료를 추정하지 않았다.','',
      '## 남은 작업 순서','',
      '**전체 core 원본 회수·최종 감사 — 다음 작업**','',
      '- 5090의 원본 소스·계약·selected/last checkpoint와 종료 증거를 보존 회수하고, 기존5080/재사용 감사를 결합해54조건108역할 최종 보고를 확정한다. 현재1차 분석 수치의 소형 증적 감사와 binary감사 완료를 혼동하지 않는다.','',
      '**대표 모델과 외부 비교 계약 확정 — 다음 작업**','',
      '- Full·이력 MLP·간소화 변형 중 논문 대표 구조와 주장 범위를 공통 기준으로 정리한다. 현 RMTPP/THP/NHP/SAHP 공통head 비교를 유지하면서 관련 강한 TPP 비교군과 공정한 튜닝 예산의 필요성을 검토한다. 이번 보고서에서 새모델을 임의 지정하거나 실행하지 않았다.','',
      '**독립 held-out와 추가 실험 — 승인 필요**','',
      '- 모델·평가 규칙을 고정한 뒤 별도 승인 범위에서 독립 검증과 필요한 효율 비교를 수행해야 한다. 현재 결과를보고held-out를 열거나seed·모델·설정을 늘리지 않았다.','',
      '## 출처와 재현','',f'- core ledger: `{LEDGER.relative_to(ROOT)}`',f'- 5090 최신 snapshot: `{SNAPSHOT.relative_to(ROOT)}`',f'- 외부36조건: `{EXTERNAL.relative_to(ROOT)}`',f'- Gate 감사 결과: `{GATE.relative_to(ROOT)}`','- 분석 코드 `build_analysis.py`, 전체조건·구간·seed별비교·원본SHA `analysis.json`, 집계표 `three_seed_metrics.csv`를 함께 보존한다. 기존결과·실패이력·계약은 수정하지 않았다.','']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    queryrows=[]
    for a in aggregates:
        queryrows.append({'dataset':a['dataset_label'],'model':a['model_label'],'model_id':a['model'],'n_seeds':3,**{f'{k}_{s}':a['selected'][k][s] for k in METRICS for s in ['mean','sample_sd']}})
    # Bounded, non-secret provenance for the local report app.
    source={'type':'local_files','title':'승인된 validation 실험 집계','files':[str(LEDGER.relative_to(ROOT)),str(EXTERNAL.relative_to(ROOT)),str(GATE.relative_to(ROOT))],'executedAt':dt.datetime.now(dt.timezone.utc).isoformat(),'evidenceFlow':[{'title':'동일 checkpoint와 중복 확인','detail':'core54와 외부36을 결합; 기존 B/Full18의 selected/last metric 및 state SHA 동일성을 확인하고 중복 제외.'},{'title':'3seed 집계','detail':'각 모델에서42/52/62 seed의 산술평균과 표본표준편차 ddof1. Gate는seed42 별도.'}],'metricDefinitions':[{'label':'수량 MAE/RMSE·시간 NLL','definition':'각 seed의 최초 strict validation raw 수량 RMSE 최소 checkpoint에서 함께 산출. 낮을수록 좋음. mean과 sample_sd는3seed 등가중 집계.','componentIds':['core-table','external-table','gate-table']}]}
    app={'title':'TitanTPP 1차 분석','subtitle':'수량 예측의 경쟁력과 구조별 한계','status':'reviewed','generatedAt':dt.datetime.now(dt.timezone.utc).isoformat(),'report':{'asOf':observed},'filters':[],'queries':{'three_seed':{'rows':queryrows,'source':source},'external_effects':{'rows':[{'dataset':DATASETS[x['dataset']],'model':LABELS[x['model']],'mae_reference':LABELS[x['qty_mae']['reference']],'rmse_reference':LABELS[x['qty_rmse']['reference']],'mae_reduction_pct':x['qty_mae']['reduction_pct'],'rmse_reduction_pct':x['qty_rmse']['reduction_pct']} for x in external_effects],'source':source},'gate':{'rows':[{'dataset':DATASETS[g['dataset']],'condition':g['queue_key'],'mode':'조건부' if g['queue_key'] in ['mlp_A','mlp_C'] else '상수','mae':g['endpoint_replays']['selected']['qty_mae'],'rmse':g['endpoint_replays']['selected']['qty_rmse'],'time_nll':g['endpoint_replays']['selected']['time_nll'],'mae_change_pct':g['vs_plain_mlp']['mae_change_pct'],'rmse_change_pct':g['vs_plain_mlp']['rmse_change_pct'],'best_epoch':g['best_epoch'],'stop_epoch':g['completed_epochs']} for g in gate['MLP_Gate_new_conditions']],'source':{**source,'metricDefinitions':[{'label':'Gate 변화율','definition':'같은 seed42 plain MLP 대비 (Gate/plain−1)×100. 양수는오차악화. 각 조건의 RMSE선택checkpoint.','componentIds':['gate-table']}]}}},'analysis_context':evidence_status}
    (OUT/'reviewed_snapshot.json').write_text(json.dumps(app,ensure_ascii=False,indent=2)+'\n')
    checks={'status':'passed','unique_core':54,'unique_external_additional':36,'overlap_reused_verified':18,'three_seed_aggregates':len(aggregates),'stratum_aggregate_rows':len(segments),'finite_endpoint_metrics':True,'endpoint_rmse_sse_reconciles':True,'earliest_strict_selection_core':True,'held_out_read':False,'new_GPU_execution':False,'final_core_binary_audit_complete':False,'source_sha256':SOURCES,'report_sha256':hashlib.sha256((OUT/'report.md').read_bytes()).hexdigest()}
    (OUT/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'report':str(OUT/'report.md'),'as_of':observed,'finished':ended,'aggregates':len(aggregates),'guardrail_passes':[(DATASETS[x['dataset']],LABELS[x['reference']],x['gate_passes']) for x in comparisons if x['reference'] in ['titantpp','titantpp_history_mlp']]},ensure_ascii=False))

if __name__=='__main__': main()
