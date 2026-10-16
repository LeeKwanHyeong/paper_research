"""Render the verified stored-prediction analysis; no model execution."""
from pathlib import Path
import hashlib
import json
import statistics as st

O=Path(__file__).resolve().parent
r=json.loads((O/'RESULT.json').read_text())
v=json.loads((O/'VERIFICATION.json').read_text())
assert v['status']=='PASS_local_second_implementation'
names={'titantpp_cnn_gru54_cnn_only':'기존 TitanTPP CNN','rmtpp':'RMTPP','s2p2_matched_head':'S2P2 matched-head','nhp':'NHP','thp':'THP','sahp':'SAHP','attnhp_matched_head':'AttNHP matched-head'}
cnn=r['models'][0]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fmt(x):return f"{x['mean']:.6f} ± {x['sample_sd']:.6f}"
lines=[]
def add(x=''):lines.append(x)
add('# RAF 저장 예측의 CNN·외부 비교군 오류 분석')
add('\n**판정: 현재 근거로 새 CNN 백본 학습을 시작하지 않는다. 자동 구조 탐색을 여기서 중단하는 편이 타당하다.**')
add('\n외부 모델이 더 잘 맞히는 사례는 분명히 있다. 그러나 그 차이가 특정 CNN 처리 결함으로 연결되거나, 앞선 이력·필터·FFN 변경과 구별되는 다음 구조를 선택할 만큼 안정적인 근거는 확보되지 않았다. 개선 가능성이 없다는 뜻이나 데이터의 예측 한계를 증명한 결과는 아니다. 이 판정은 현재 탐색의 관문인 RAF에 한정하며 Taxi·Intermittent 전체로 확대하지 않는다.')
add('\n## 분석 범위와 검증 — 완료')
add('\n- Material Passport: 사용자 소유 RAF의 동결 모델 7개(기존 CNN + 외부 6개), seed42·52·62의 저장된 Train·Validation 예측. Train 25,779개·Validation 6,690개 표적, 총42개 모델/seed/split 묶음이다. 같은 표적을 seed마다 새로운 독립 표본으로 세지 않는다.')
add('- Verification Status: **ANALYZED**. 저장 예측의 출처·행 대응·수치를 재계산했다. 모델 추론, 학습, checkpoint 재선택, 원격 실행, 기존 CNN 수정은 모두0이다. 기존 CNN은 byte hash 확인만 했으며 모델로 불러오지 않았다.')
add('- 저장 행의 worker 실제exit0, 선택 epoch·checkpoint file/tensor SHA, 원본 manifest/shard SHA, 동일 target/truth/history, 평가 전후 모델 상태 불변을 확인했다. CNN의 별도 Validation 원본 결과도 일치한다.')
add(f"- 별도 산식 구현으로 {v['checks']['cohort_metrics']}개 전체/큰수량 지표 묶음, {v['checks']['paired_cohorts']}개 대응 비교, {v['checks']['history_cell_cohorts']}개 조건 집계, {v['checks']['scalar_equalities']:,}개 수치 일치를 검산하고 {v['files_verified_unchanged']}개 파일 SHA가 유지됨을 확인했다. 같은 에이전트의 두 번째 구현 검산이며 독립 연구자/에이전트 검토로 표현하지 않는다.")
add('- 이 비교의 저장 예측은 모두5080·cuDNN91900·동일 GPU/Python/torch 조건이다. 최근5090 후보 실험과의 Runtime 차이를 이 비교에 잘못 적용하지 않는다. 모델별 구조·용량·학습 경로 차이는 여전히 남는다.')
add('- 계산에는 순수 Train·Validation 입력과 해당 split의 저장 예측만 사용했다. Test 예측·평가 파일과 원고는 열지 않았다. Notion 누적 기록을 위한 기존 페이지 조회에는 과거 Test 서술이 포함됐으나, 이번 계산·조건 설정·판정에는 사용하지 않았다. 원본 계약의 Test 경로/대상 metadata도 수치 분석 입력이 아니다.')
add('- 새 결과는 별도 경로에 기록했다. 기존 결과와 병합·대체하지 않았고, 삭제된 반복 자동화를 재생성하지 않았다. PyArrow의 macOS CPU cache sysctl 조회 경고는 stderr에 보존했으며 분석·검산은 실제exit0으로 종료했다.')
add('\n## 전체 성능: 외부 모델도 두 지표를 동시에 개선하지는 않았다')
add('\n각 행은 seed별 전체 raw 수량 지표의 평균±표본표준편차(ddof=1)다. 세 seed의 예측을 먼저 평균한 앙상블 성능이 아니다. 낮을수록 좋다.')
add('\n| 모델 | Train RMSE | Train MAE | Validation RMSE | Validation MAE |\n|---|---:|---:|---:|---:|')
for model in r['models']:
    a=next(x for x in r['summary'] if x['model']==model and x['split']=='train')['whole']
    b=next(x for x in r['summary'] if x['model']==model and x['split']=='validation')['whole']
    add(f"| {names[model]} | {fmt(a['rmse'])} | {fmt(a['mae'])} | {fmt(b['rmse'])} | {fmt(b['mae'])} |")
add('\nValidation에서 S2P2는 CNN보다 RMSE가 0.436704 낮지만 MAE는 0.054717 높다. RMTPP는 MAE가 0.082066 낮지만 RMSE는 0.118919 높다. 다른 네 모델까지 포함해 두 평균 지표가 모두 CNN보다 낮은 외부 모델은 없다. 사용자의 목표는 서로 다른 모델이 가진 지표별 최우수를 동시에 넘는 것이며, 이를 한 모델이 이미 달성한 성능처럼 취급하지 않는다. 목표를 완화하거나 달성 불가능하다고 단정하지 않는다.')
add('\n## seed별 차이와 어떤 사례에서 오류가 갈리는지')
add('\n아래 차이는 **CNN−외부 모델**이다. 양수는 외부 모델 우세, 음수는 CNN 우세다. 전체 모집단을 사용했다.')
add('\n| Split | 비교군 | seed | ΔRMSE | ΔMAE |\n|---|---|---:|---:|---:|')
for split in ['train','validation']:
    for model in ['s2p2_matched_head','rmtpp']:
        for x in r['pairwise']:
            if x['split']==split and x['model']==model:add(f"| {split} | {names[model]} | {x['seed']} | {x['delta_rmse']:+.6f} | {x['delta_mae']:+.6f} |")
add('\n### S2P2: 일부 큰 오차를 줄이지만 다른 수량 구간에서 잃는다')
add('\nValidation의 60<q≤200 및 q>200 구간에서 S2P2의 AE와 SE 합이 모두 낮았다. 반면2<q≤60 구간의 AE·SE 합은 모두 높았다. 3seed 평균으로 보면 다음과 같다. AE는 절대오차, SE는 제곱오차다.')
add('\n| 실제 수량 구간 | 표적 수 | CNN−S2P2 절대오차 합 | CNN−S2P2 제곱오차 합 |\n|---|---:|---:|---:|')
for i,label in enumerate(['q≤2','2<q≤30','30<q≤60','60<q≤200','q>200']):
    xs=[x for x in r['partitions'] if x['split']=='validation' and x['model']=='s2p2_matched_head' and x['cell']==f'target_quantity:{i}']
    add(f"| {label} | {xs[0]['rows']} | {st.mean(x['delta_ae_sum'] for x in xs):+.3f} | {st.mean(x['delta_se_sum'] for x in xs):+.3f} |")
add('\nS2P2가 제곱오차를 줄이는 사례 중 상위67개(Validation 전체의 약1%)가 **외부 모델에 유리한 제곱오차 감소분의 87.3~89.0%**를 차지한다. 이는 악화분을 차감하기 전의 양의 기여 합에 대한 비율이다. 전체 CNN 오차의89%를 해결했다는 뜻이 아니다.')
add('\nRMTPP는60<q≤200에서 평균 절대오차 합을788.274 낮추지만 q>200에서는491.902 높인다. 같은 큰수량 구간의 제곱오차 합은274,613.171 증가해 전체 RMSE가 악화된다. 특정 모델이 모든 사례에 우월한 표현을 배웠다는 해석은 맞지 않는다.')
add('\n실제 수량으로 나눈 구간은 사후 설명이다. 예측 시 알 수 없는 다음 수량을 CNN의 입력이나 분기 조건으로 사용할 수 없다.')
add('\n### CNN만 실패하는 문제인지: 큰 오류의 대부분은 공유된다')
add('\nTrain의 CNN 절대오차90백분위20.418005를 고정하고, 양 모델 모두 이 기준을 넘는 사례를 확인했다. Validation에서 이 공통 고오류 사례가 CNN 전체 제곱오차의97.16~97.59%(S2P2 비교),97.24~97.65%(RMTPP 비교)를 차지한다. q>200인50개 표적만으로도 CNN 제곱오차의68.18~69.73%가 발생한다. 세 seed 모두 같은 표적 집합이다.')
add('\n이 결과는 어려운 사례가 겹친다는 기술통계다. 관측 정보가 부족하다거나 어떤 모델도 예측할 수 없다는 증거는 아니다. 다만 외부 모델의 우위만 보고 CNN의 특정 처리 단계를 실패 원인으로 지정하기 어렵다.')
add('\n### 실제 사례: 같은 개체의 비슷한 이력에서도 우세 방향이 뒤집힌다')
add('\n아래는 원본 행에서 뽑은 예시다. 예측 수치는 설명을 위한3seed 평균이고, 성능 집계는 위의 seed별 산식을 유지한다. 같은 이력 조건에서 RMTPP의 가장 큰 이득과 손실 사례를 함께 보존했다.')
add('\n| Split/개체/순서 | 이력 길이 | 직전 수량 / 이력 평균 | 실제 수량 | CNN 예측 | RMTPP 예측 |\n|---|---:|---:|---:|---:|---:|')
for entity,seq in [('RAF::02318',71),('RAF::04329',59),('RAF::04329',67)]:
    x=next(x for x in v['signal_examples'] if x['split']=='validation' and x['entity_id']==entity and x['seq']==seq)
    add(f"| Validation / {entity} / {seq} | {x['history_length']} | {x['last_quantity']:.1f} / {x['history_mean']:.1f} | {x['truth']:.0f} | {st.mean(x['cnn_predictions']):.3f} | {st.mean(x['rmtpp_predictions']):.3f} |")
add('\nRAF::04329에서는 실제100을 예측할 때 RMTPP가 낫고, 이후 실제200을 예측할 때 CNN이 낫다. 둘 다 직전 수량은100이고 같은 고정 이력 조건에 속한다. 완전히 같은 이력이나 인과 대조는 아니며, 낮게 예측하는 쪽을 항상 선택하면 된다는 규칙도 나오지 않는다. target ID와 모든 seed 예측, 양방향 상위 사례는 RESULT.json 및 VERIFICATION.json에 있다.')
add('\n## 이력으로 설명할 수 있는 안정적인 차이는 있었나')
add('\n기존15개 패턴/변화폭 조건에 이력 길이·직전 수량·관측 변동계수·이력 평균 대비 직전값을 더해29개 조건을 보고했다. 서로 포함되거나 겹치는 조건이므로29개 독립 실험이 아니다. 구간 경계는 Train에서만 계산해 Validation에 적용했다. S2P2와 RMTPP를 주 비교군으로 고정하고, 다른 네 모델의 전체 결과도 빠짐없이 보고했다.')
add('\n**RMTPP의 한 조건에서만 양 split·모든 seed·두 가중 방식의 RMSE와 MAE 우위가 함께 유지됐다.** 최근 세 관측의 두 인접 변화 중 하나가0이고, 나머지 log1p 변화의 절댓값이 Train Q25 초과~Q75 이하인 조건이다. 경계는0.4054651081과1.2527629685다. 이력을 모두 평탄하게 유지한 경우라는 뜻은 아니다.')
add('\n| Split | 전체 조건 표적 수 | CNN RMSE / MAE | RMTPP RMSE / MAE | 공통 개체 / 남는 표적 수 |\n|---|---:|---:|---:|---:|')
cell='pattern_magnitude:stagnation:middle'
for split in ['train','validation']:
    xs=[x for x in r['history_cells'] if x['cell']==cell and x['model']=='rmtpp' and x['split']==split and x['weighting']=='row']
    sp=next(x for x in r['history_support'] if x['cell']==cell and x['split']==split)
    add(f"| {split} | {sp['rows']} | {st.mean(x['cnn']['rmse'] for x in xs):.6f} / {st.mean(x['cnn']['mae'] for x in xs):.6f} | {st.mean(x['external']['rmse'] for x in xs):.6f} / {st.mean(x['external']['mae'] for x in xs):.6f} | {sp['common_entities']} / {sp['common_rows']} |")
add('\n공통 개체 가중은 같은246개 개체를 양 split에 남겨 각 개체의 총 가중치를 같게 했다. Train1,363행·Validation427행을 제외하므로 원래 전체 표본과 다른 추정 대상이다. 수량 수준·변화폭·개체 구성을 동시에 정확히 맞춘 대조라고 표현하지 않는다.')
add('\n### 이 신호를 바로 새 학습 근거로 삼지 않는 이유')
add('\n- 조건 내에서도 RMTPP 우세와 CNN 우세가 섞인다. 전체 Validation MAE 차이에 대한 이 조건의 가산 기여는0.016837이다. 이 수치는 새로운 CNN이 실현할 수 있는 개선량이 아니다.')
add('- 결과를 본 뒤 실시한 민감도 부록에서, 각 seed의 가장 큰 AE 이득 사례 한 건을 제외하면 Train·Validation 모두 seed52·62의 MAE 우위가 사라졌다. 원래 신호 판정을 지우거나 새 통과 기준으로 바꾸지 않았고, 원래 전체 표본의 우위와 이 집중도를 함께 기록했다.')
add('- 이 조건에서 제곱오차의 양의 이득 상위1%는 Train80.3~87.2%,Validation87.5~92.8%를 차지한다. 단일 사례의 오류도 실제 오류이므로 삭제해서 성능을 재정의하지 않는다. 다만 조건 전체에 널리 적용될 기전이라고 해석하기에는 약하다.')
add('- 동결 코드의 RMTPP는 관측 입력을 GRU로 처리하고, S2P2는 상태공간 인코더를 사용한다. CNN과는 여러 구조·용량·최적화 특성이 동시에 다르다. 저장 출력의 차이만으로 “GRU 효과”, “상태공간 효과”, “CNN의 어느 부분 결함”을 분리할 수 없다.')
add('- 이 조건은 이미 검토한 최근 변화·수준·이력 적응과 겹친다. 새로운 gate·차분·긴 필터를 제안하면 이전 시도를 다른 이름으로 반복할 위험이 있다. 관측 가능한 조건을 찾았다는 것만으로 구별 가능한 새 backbone 가설이 완성된 것은 아니다.')
add('\n| Split | seed | 원래 조건 ΔMAE | 최대 AE 이득 한 건 제외 후 ΔMAE | 원래 조건 ΔMSE | 최대 SE 이득 한 건 제외 후 ΔMSE |\n|---|---:|---:|---:|---:|---:|')
for x in v['signal_sensitivity']:
    add(f"| {x['split']} | {x['seed']} | {x['delta_mae']:+.6f} | {x['delta_mae_after_removing_single_largest_ae_gain']:+.6f} | {x['delta_mse']:+.6f} | {x['delta_mse_after_removing_single_largest_se_gain']:+.6f} |")
add('\n각 열의 최대 기여 행을 별도로 제외한 기술적 민감도다. 유리한 행을 제외한 결과를 본 실험의 새 성능으로 사용하지 않는다. Validation이 이미 여러 설계에 활용됐고 다수 조건을 살폈으므로 새 독립 검증이나 통계적 유의성으로 해석하지 않는다.')
add('\n## 다음 작업 판단')
add('\n**현재 백본 반복 탐색을 마무리한다 — 완료**\n- 이 분석에서 추가 학습을 정당화할 새 구조는 선정하지 않았다. 기존 CNN과 외부 모델을 고정하고 현재 결과·실패·진단을 보존했다. 새 학습, 새 추론, 다음 루프와 데이터셋 확장은 실행하지 않았다.\n- 중단 권고의 이유는 승인이 없어서가 아니라, 모델을 선택할 기전 근거가 부족하기 때문이다. 지표 개선 사례가 전혀 없거나 CNN 개선이 불가능하다는 판정은 아니다.')
add('\n**재개 여부는 새로운 근거가 생겼을 때만 결정한다 — 다음 작업 조건**\n- 재개하려면 예측 전에 확인 가능한 반복 오류가 어떤 정보 처리 단계에서 생기는지 연결하고, 기존 이력·Q/K/V·FFN 실험과 다른 한 가지 변경 및 같은 조건 대조를 제시해야 한다. 그 가설이 맞다면 어느 사례에서 두 지표가 함께 좋아질지를 미리 설명할 수 있어야 한다.\n- 지금 그 조건을 충족한 후보는 없다. 이 문서는 새 진단이나 학습을 자동으로 이어갈 계획이 아니다. 원고의 주장이나 투고 가능성은 이번 분석만으로 판정하지 않는다.')
add('\n## 여섯 지표 전체와 불리한 결과 보존')
add('\n큰수량은 기존 Train 기준q>200이다. TimeNLL은 recorded positive-integer 시간의 음의 로그가능도다. 모든 지표는 낮을수록 좋다. 평균±표본SD를 그대로 보고한다.')
for split in ['train','validation']:
    add(f'\n### {split}')
    add('\n| 모델 | 전체 RMSE | 전체 MAE | 큰수량 RMSE | 큰수량 MAE | 전체 TimeNLL | 큰수량 TimeNLL |\n|---|---:|---:|---:|---:|---:|---:|')
    for x in r['summary']:
        if x['split']!=split:continue
        vals=[x['whole']['rmse'],x['whole']['mae'],x['tail']['rmse'],x['tail']['mae'],x['whole']['time_nll'],x['tail']['time_nll']]
        add('| '+names[x['model']]+' | '+' | '.join(fmt(q) for q in vals)+' |')
add('\n## 해석 검토와 재현 자료')
add('\n11개 해석 위험을 점검했다. ①관찰 차이를 인과로 바꾸지 않음 ②seed를 독립 표적 수로 중복 계산하지 않음 ③표본SD를 신뢰구간/유의성으로 바꾸지 않음 ④RMSE와MAE의 서로 다른 집계 기여 구분 ⑤전체·부분집단 추정 대상 구분 ⑥다중 조건에서 발견된 결과의 탐색성 명시 ⑦단일 큰 기여 사례와 표본 제외 내역 공개 ⑧사후 실제 수량을 배포 가능한 입력으로 사용하지 않음 ⑨반복 사용한 Validation의 독립성 주장 금지 ⑩무효/미선정이 효과 부재의 증명은 아님 ⑪외부 모델의 우위를 특정 CNN 기전/다른 데이터셋/논문 기여로 확대하지 않음.')
add('\n- PLAN.json: 사전 고정한 범위·조건·판정 원칙.\n- RESULT.json:42개 원본 연결,14개 전체 요약,36개 쌍 비교,432개 오차 분해,696개 이력 조건 집계,58개 신호 판정,80개 양방향 사례.\n- paired_train.parquet / paired_validation.parquet: 기존 예측과 관측 이력을 같은 표적에 연결한 감사용 표. 새 예측이 아니다.\n- INPUT_MANIFEST.json / VERIFICATION.json: 원본 SHA와 두 번째 산식 검산, 후속 민감도/사례.\n- analysis.stdout/.stderr 및 verification.stdout/.stderr: 실제 실행 원문.\n- RECEIPT.json: 보고서와 결과의 최종SHA 및 작업 범위.')
add('\n동결 구조 확인: CNN source/models/TPPs/CountAwareTitanCNNGRU.py의 _encode_base, 외부 evaluator_sources/raf/models/TPPs/CountAwareTPP.py의 CountAwareRMTPP, CountAwareAdditionalTPP.py의 CountAwareS2P2를 확인했다. 현재 작업트리 모델이나 논문 이름의 일반적 구현을 이 실험의 실제 구현으로 대신하지 않았다.')
add(f"\nRESULT rawSHA `{sha(O/'RESULT.json')}`; VERIFICATION rawSHA `{sha(O/'VERIFICATION.json')}`.")
with (O/'RESULT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
print('Report saved; original data unchanged.')
