"""Join audited terminal records to existing validation baselines, with source links."""
import csv,json,math,statistics,time
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
from collect import ROOT,REPORT,BUNDLE,DEST,EXPECTED,CLOSURE,sha,read
METRICS=('qty_mae','qty_rmse','time_nll')
BASE='titantpp_history_mlp'
CURRENT='titantpp_current_only_param_matched'
ALL='titantpp_all_available_history_mlp'
DRP='deep_renewal_event_native_nb'
LABEL={BASE:'기존 History MLP',CURRENT:'현재 상태 전용·파라미터 일치',ALL:'전체 분기 사용·고정 /8',DRP:'Deep Renewal (event-native NB)'}
DATA={'yellow_trip_hourly':'Taxi','intermittent_frozen_5000':'Intermittent','insta_market_basket':'Instacart'}
def rel(p):return str(p.relative_to(ROOT))
def link(p,label=None):return f'[{label or p.name}]({p})'
def emit(p,v):
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def close(a,b):return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8)
scope=read(REPORT/'scope.json');rows=[];audits={};retrievals={}
for host in ('5080','5090'):
    p=DEST/host/'terminal_audit.json';a=read(p);r=read(DEST/host/'retrieval_receipt.json')
    assert a['status']=='passed' and a['contract_sha256']==EXPECTED and a['source_closure_sha256']==CLOSURE
    assert a['audit_script_sha256']==sha(REPORT/'audit.py') and r['script_sha256']==sha(REPORT/'collect.py')
    assert a['scope_sha256']==sha(REPORT/'scope.json')==r['scope_sha256']
    assert a['verified_terminal_checkpoints']==len(a['conditions'])*2
    audits[host]={'path':rel(p),'sha256':sha(p),'status':a['status']};retrievals[host]=r
    for v in a['conditions']:
        j=v['job'];original=DEST/host/'original';manifest=read(original/'run'/j['id']/'terminal_manifest.json')
        ep=next(k for k in manifest['files'] if k.endswith('/endpoint_replays.json'))
        ep=original/'run'/j['id']/ep;e=read(ep)['selected']
        assert sha(ep)==manifest['files'][str(ep.relative_to(original/'run'/j['id']))]
        assert all(close(e[k],v['selected_validation'][k]) for k in METRICS)
        rows.append({'dataset':j['dataset'],'model':j['arm'],'seed':j['seed'],'host':host,
                     'completed_epochs':v['completed_epochs'],'selected_epoch':v['selected_epoch'],
                     'status':'completed_binary_CPU_audited','evaluation_scope':'validation_only',
                     **v['selected_validation'],'count':e['count'],'source':rel(ep),'source_sha256':sha(ep),
                     'model_tensor_sha256':v['model_tensor_sha256']['selected'],
                     'audit':rel(p),'audit_sha256':sha(p),'newly_audited':True})
assert len(rows)==13
needed={(r['dataset'],r['seed']) for r in rows}
basefile=ROOT/'reports/titantpp_dataset_appendix_20261002_v1/selected_conditions.csv'
baseverification=ROOT/'reports/titantpp_dataset_appendix_20261002_v1/verification.json'
old=read(baseverification);assert old['status']=='passed' and old['selected_conditions']==84
allbase=list(csv.DictReader(basefile.open()));assert len(allbase)==84
for r in allbase:
    if r['model']!=BASE or (r['dataset'],int(r['seed'])) not in needed:continue
    source=ROOT/r['source'];assert sha(source)==r['source_sha256']
    endpoint=read(source)['selected'];assert endpoint['evaluation_scope']=='validation_only' and not endpoint['held_out_test_evaluated']
    assert endpoint['state_sha256']==r['model_tensor_sha256']
    assert all(close(endpoint[k],float(r[k])) for k in METRICS) and endpoint['count']==int(r['count'])
    rows.append({'dataset':r['dataset'],'model':BASE,'seed':int(r['seed']),'host':'existing_baseline',
                 'completed_epochs':None,'selected_epoch':int(r['selected_epoch']),'status':'existing_audit_reused',
                 'evaluation_scope':'validation_only',**{k:float(r[k]) for k in METRICS},'count':int(r['count']),
                 'source':r['source'],'source_sha256':r['source_sha256'],'model_tensor_sha256':r['model_tensor_sha256'],
                 'audit':rel(baseverification),'audit_sha256':sha(baseverification),'newly_audited':False})
assert len(rows)==18
lookup={(r['dataset'],r['model'],r['seed']):r for r in rows};assert len(lookup)==18
paired=[]
for r in rows:
    if r['model']==BASE:continue
    b=lookup[(r['dataset'],BASE,r['seed'])];assert r['count']==b['count']
    paired.append({'dataset':r['dataset'],'model':r['model'],'seed':r['seed'],
                   'delta_definition':'variant_minus_original_MLP; lower_is_better',
                   **{k:{'baseline':b[k],'variant':r[k],'delta':r[k]-b[k],
                          'variant_relative_change_pct':100*(r[k]-b[k])/b[k]} for k in METRICS}})
groups=defaultdict(list)
for r in rows:groups[(r['dataset'],r['model'])].append(r)
aggregates=[]
for (dataset,model),rs in groups.items():
    seeds=sorted(r['seed'] for r in rs)
    if seeds!=[42,52,62]:continue
    aggregates.append({'dataset':dataset,'model':model,'seeds':seeds,'n':3,
                       'aggregation':'mean_of_seed_metrics; sample_SD_ddof_1; no_ensemble',
                       **{k:{'mean':statistics.mean(r[k] for r in rs),'sample_sd':statistics.stdev(r[k] for r in rs)} for k in METRICS}})
assert len(aggregates)==4 and all(g['dataset']=='yellow_trip_hourly' for g in aggregates)
ag={g['model']:g for g in aggregates};comparisons=[]
for model in (CURRENT,ALL,DRP):
    ps=[r for r in paired if r['dataset']=='yellow_trip_hourly' and r['model']==model]
    comparisons.append({'dataset':'yellow_trip_hourly','model':model,'baseline':BASE,
                        **{k:{'variant_minus_baseline_mean':ag[model][k]['mean']-ag[BASE][k]['mean'],
                               'variant_relative_change_pct':100*(ag[model][k]['mean']/ag[BASE][k]['mean']-1),
                               'variant_lower_seed_count':sum(r[k]['delta']<0 for r in ps),
                               'original_MLP_reduction_pct_relative_to_variant':100*(1-ag[BASE][k]['mean']/ag[model][k]['mean'])} for k in METRICS}})
reused_paths=[basefile,baseverification,
 ROOT/'reports/titantpp_independent_evaluation_preparation_20261001_v1/checkpoint_audit.json',
 ROOT/'reports/titantpp_independent_evaluation_preparation_20261001_v1/evaluation_registry.json',
 ROOT/'reports/titantpp_efficiency_5080_20260930_v1/verification.json']
assert read(reused_paths[-1])['status']=='passed'
result={'status':'completed13_CPU_audit_and_validation_comparison_complete','created_unix':time.time(),
        'evaluation_scope':'validation_only','scope':scope,'contract_sha256':EXPECTED,'source_closure_sha256':CLOSURE,
        'newly_audited_conditions':13,'verified_terminal_binary_checkpoints':26,'verified_source_files_per_host':109,
        'verified_replay_endpoint_records':26,'retrieved_files_excluding_collection_manifests':sum(r['files'] for r in retrievals.values()),
        'retrieval_bytes':sum(r['bytes'] for r in retrievals.values()),'audits':audits,'retrievals':retrievals,
        'reused_evidence':[{'path':rel(p),'sha256':sha(p)} for p in reused_paths],
        'conditions':rows,'paired_comparisons':paired,'complete_three_seed_groups':aggregates,
        'three_seed_comparisons':comparisons,'remaining_contract_conditions_outside_pinned_scope':23,
        'scientific_scope_notes':['Intermittent and Instacart controls are seed42 only, no three-seed inference.',
          'Deep Renewal uses its native NB time and quantity objectives, not the shared quantity loss/head.',
          'All-available history still reads the immediate predecessor and divides by eight; it is not active-count normalization.',
          'Taxi Deep Renewal selected the maximum epoch 300 for all three seeds; completion under budget is not evidence of convergence.',
          'Other contract conditions may have progressed since the pinned observation; this collection is not a new overall progress poll.',
          'Validation comparisons do not resolve final independent-evaluation provenance or demonstrate statistical significance.'],
        'new_training_updates':0,'new_forward_or_replay_calls':0,'heldout_metrics_or_predictions_read':False,
        'scientific_source_or_training_processes_changed':False,'manuscript_edited':False}
emit(REPORT/'comparison.json',result)
rows.sort(key=lambda r:(list(DATA).index(r['dataset']),[BASE,CURRENT,ALL,DRP].index(r['model']),r['seed']))
with (REPORT/'conditions.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
now=datetime.fromtimestamp(result['created_unix'],ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S KST')
lines=['# TitanTPP 후속 완료 13조건 원본 감사 및 validation 비교', '',f'작성: {now}', '',
 '**완료 13조건의 checkpoint 26개와 서버별 동결 source 109개를 회수하여 CPU 검증을 통과했다.** '
 '범위는 2026-10-02 10:51:58 KST 관측에서 완료된 Taxi 9조건, Intermittent 2조건, Instacart 2조건이다. '
 '학습 전체 완료 보고가 아니며, 이후 완료 조건은 이 감사에 포함하지 않았다.', '',
 '## 1. 회수 및 검증 결과', '',
 '| 서버 | 완료 조건 | binary checkpoint | 회수 파일¹ | 회수 byte | 결과 |','|---|---:|---:|---:|---:|---|']
for host in ('5080','5090'):
 r=retrievals[host];lines.append(f"| {host} | {r['jobs']} | {r['jobs']*2} | {r['files']} | {r['bytes']:,} | CPU 검증 통과 |")
lines += ['', '¹ collection_manifest.json 자체를 제외한 SHA 검증 파일 수. 두 서버의 동일 source 사본은 각각 검증했다.', '',
 '- 계약 SHA, source 109개 closure, 승인·실행 permit, native qualification, 종료 명세와 회수 전후 파일 SHA를 연결했다.',
 '- selected/last checkpoint를 frozen source로 구성한 모델에 CPU strict loading했다. tensor SHA·유한성·선택 epoch·embedded best를 대조했다.',
 '- last checkpoint의 optimizer tensor/step·RNG·shuffle state를 복원했고 전체 train 노출·기존 MLP와 공통 batch prefix·validation 모집단을 확인했다.',
 '- 최초 최소 raw 수량 RMSE 선택, 종료 epoch, selected/last validation 재평가 기록의 MAE·RMSE·시간 NLL 및 구간 합계를 대조했다.',
 '- 마지막 epoch의 exposure 영수증은 학습 종료 후 추가 validation 1회를 제외한 당시 상태와 일치한다. 원본 파일은 수정하지 않았다.',
 '- 서버에서는 종료된 조건의 파일만 읽었다. 새 학습·모델 forward·재평가·GPU 측정은 수행하지 않았다. 기존 84조건 및 효율 감사는 재사용했다.', '',
 'CPU strict loading과 기록 대조를 완료한 것이며 예측을 새로 계산한 것은 아니다. CUDA 초기화·정확성 검증은 SHA로 연결된 기존 native 증거를 재사용했다. '
 '서버별 qualification의 합성 75updates(정확성 15 + 비용 측정 60)는 scientific fit에 포함하지 않는다.', '',
 '## 2. Taxi — 세 seed 완료 비교', '',
 '세 seed 42·52·62의 validation 지표 평균 ± 표본 표준편차(ddof=1). 모든 지표는 같은 RMSE 선택 epoch에 대응하며 낮을수록 좋다.', '',
 '| 모델 | MAE | RMSE | 시간 NLL |','|---|---:|---:|---:|']
for model in (BASE,CURRENT,ALL,DRP):
 g=ag[model];lines.append('| '+LABEL[model]+' | '+' | '.join(f"{g[k]['mean']:.4f} ± {g[k]['sample_sd']:.4f}" for k in METRICS)+' |')
cc={c['model']:c for c in comparisons}
lines += ['',
 f"- 기존 MLP는 현재 상태 전용 변형 대비 MAE {cc[CURRENT]['qty_mae']['original_MLP_reduction_pct_relative_to_variant']:.2f}%, RMSE {cc[CURRENT]['qty_rmse']['original_MLP_reduction_pct_relative_to_variant']:.2f}% 낮다. 수량 두 지표에서 세 seed 모두 같은 방향이며, 인접 상태 결합의 유용성을 뒷받침한다. 시간 NLL은 현재 상태 전용 변형이 평균적으로 낮다.",
 f"- 전체 분기 사용 변형은 기존 MLP보다 평균 MAE {-cc[ALL]['qty_mae']['variant_relative_change_pct']:.2f}%, RMSE {-cc[ALL]['qty_rmse']['variant_relative_change_pct']:.2f}% 낮다. seed별 개선은 MAE {cc[ALL]['qty_mae']['variant_lower_seed_count']}/3, RMSE {cc[ALL]['qty_rmse']['variant_lower_seed_count']}/3이며 시간 NLL 평균은 악화한다. 단계적 분기 가용성 규칙이 항상 우월하다는 주장은 지지하지 않는다.",
 '- 기존 MLP는 이번 Deep Renewal adapter보다 수량 오차가 크게 낮고, Deep Renewal은 시간 NLL이 낮다. Deep Renewal은 세 seed 모두 최대 300epoch에서 선택됐으므로 수렴 완료나 모든 Deep Renewal 설정에 대한 우월성으로 확대하지 않는다.',
 '- 두 내부 대조군은 기존 MLP와 파라미터 수·공통 head·수량 loss를 맞췄다. Deep Renewal은 고유 NB 시간·수량 손실을 쓰므로 동일 head/loss 비교로 표현하지 않는다.', '',
 '## 3. Intermittent·Instacart — seed42 완료분', '',
 '| 데이터 | 모델 | 선택/완료 epoch² | MAE | RMSE | 시간 NLL |','|---|---|---:|---:|---:|---:|']
for dataset in ('intermittent_frozen_5000','insta_market_basket'):
 for model in (BASE,CURRENT,ALL):
  r=lookup[(dataset,model,42)];ep=f"{r['selected_epoch']}/{r['completed_epochs']}" if r['newly_audited'] else f"{r['selected_epoch']}/기존 감사"
  lines.append(f"| {DATA[dataset]} | {LABEL[model]} | {ep} | "+' | '.join(f'{r[k]:.6f}' for k in METRICS)+' |')
lines += ['', '² 기존 MLP의 완료 epoch는 이번 신규 감사 범위에 포함하지 않았으며 이미 확인된 선택 결과를 재사용했다.', '',
 '- Intermittent seed42에서는 기존 MLP가 두 변형보다 MAE·RMSE·시간 NLL 모두 낮다. 아직 한 seed 결과이므로 세 seed 결론을 만들지 않는다.',
 '- Instacart seed42에서 두 변형의 RMSE는 기존 MLP보다 약 0.11% 낮은 수준이다. MAE·시간 NLL의 방향은 변형별로 다르며, 전반적인 개선으로 묶지 않는다.',
 '- 이 두 데이터의 Deep Renewal과 seed52·62, RAF는 이번 고정 13조건 분석 대상이 아니다. 결과를 누락한 성공 집계가 아니라 별도 대기 범위로 보존한다.', '',
 '## 4. 주장에 미치는 영향', '',
 'Taxi의 같은 파라미터 수 대조는 기존 이력 결합 구조의 수량 예측 효과를 보강한다. '
 '반면 전체 분기 사용 변형이 평균 수량 오차에서 앞서므로, 특정 단계적 활성화 규칙을 개선의 필수 원인으로 제시하지 않는다. '
 '기존 외부 여섯 비교군 및 효율 측정 결과는 이번 감사로 변경되지 않았다. 구조 전체의 유용성, 분기 규칙의 효과, 데이터별 적용 범위를 분리해 원고에 반영할 수 있다. '
 '통계적 유의성이나 독립 test 결과를 새로 확보한 것은 아니다.', '',
 '## 5. 원본 및 재현 경로', '',
 f"- 고정 범위: {link(REPORT/'scope.json')}.",f"- 조건별 표와 원본 경로: {link(REPORT/'conditions.csv')}; 수치·짝지은 비교: {link(REPORT/'comparison.json')}."]
for host in ('5080','5090'):
 lines.append(f"- {host}: {link(DEST/host/'retrieval_receipt.json','회수 영수증')} · {link(DEST/host/'terminal_audit.json','최종 CPU 감사')} · {link(DEST/host/'original/collection_manifest.json','원본 파일 SHA 목록')}.")
lines += [f"- 기존 84조건: {link(ROOT/'reports/titantpp_independent_evaluation_preparation_20261001_v1/report.md','완료된 회수·검증')} 및 {link(basefile,'validation 기준선')}. 효율: {link(reused_paths[-1],'기존 검증 영수증')}.",
 '', '최초 retrieval_receipt의 binary_cpu_audit=pending은 전송 직후 이력이다. 후속 terminal_audit.json이 현재 완료 판정이며, 최초 영수증은 덮어쓰지 않았다.', '',
 '## 남은 작업 순서', '',
 '**1. 이번 확정 비교를 원고에 반영한다 — 다음 작업**',
 '- 대상: paper/titantpp_pakdd_2027_draft/main.tex의 Appendix A 및 연결 본문. Taxi 3seed와 다른 데이터 seed42를 구분하고, 단계적 분기 규칙에 대한 해석을 위 결과에 맞춘다. 기존 주 비교표 84조건과 효율 결과는 기준선으로 유지한다.', '',
 '**2. 진행 중인 후속 조건을 관측하고 완료분을 같은 방식으로 검증한다 — 진행 중**',
 '- 대상: 승인된 5080·5090 후속 큐. 이번 범위 밖 23조건의 학습 상태는 기존 시간별 관측을 따르며, 이번 보고서는 새로운 전체 상태 조회가 아니다. 데이터·모델별 3seed가 모두 확인된 뒤 평균과 표본 표준편차를 확정한다. 원고 반영과 병행할 수 있다.', '',
 '**3. 최종 평가 범위와 원고의 핵심 주장을 확정한다 — 이후 작업**',
 '- 남은 후속 비교와 기존 데이터 계보 검토를 연결한다. 최종 평가의 성격·실행 계약을 확인하고, 필요한 평가 후 초록·결론을 정리한다. 이번 원본 감사는 독립 평가나 새 평가 실행 승인을 대신하지 않는다.', '']
with (REPORT/'report.md').open('x') as f:f.write('\n'.join(lines))
verification={'status':'passed','new_conditions':13,'binary_checkpoints':26,'complete_three_seed_groups_including_reused_MLP':4,
 'partial_seed42_groups_including_reused_MLP':6,'all_13_pinned_conditions_preserved':True,'all_compared_endpoints_file_SHA_verified':True,
 'all_metrics_same_RMSE_selected_epoch':True,'seed_aggregation':'sample_SD_ddof_1','no_partial_three_seed_aggregates':True,
 'existing_84_and_efficiency_audits_reused':True,'no_new_fit_forward_replay_or_GPU':True,'heldout_read':False,
 'manuscript_edited':False,'remaining_23_outside_pinned_scope_not_assumed_complete':True,
 'generated_unix':time.time(),'analysis_script_sha256':sha(__file__)}
emit(REPORT/'verification.json',verification)
emit(REPORT/'artifact_manifest.json',{'created_unix':time.time(),'files':{p.name:sha(p) for p in REPORT.iterdir() if p.is_file() and p.name!='artifact_manifest.json'}})
print(json.dumps({'report':str(REPORT/'report.md'),'conditions':13,'checkpoints':26,'status':'passed'},ensure_ascii=False))
print(json.dumps(comparisons,ensure_ascii=False))
