import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
read = lambda p: json.loads(Path(p).read_text())
c = read(OUT / 'contract.json')
a = read(OUT / 'analysis.json')
names = {'yellow_trip_hourly': 'Taxi', 'raf_spare_parts': 'RAF', 'intermittent_frozen_5000': 'Intermittent', 'insta_market_basket': 'Instacart'}
receipts = {(r['dataset'], r['model']): r for r in [read(p) for p in (OUT / 'runs').glob('*/receipt.json')]}


def metric(ds, model, split, group):
    return next(x for x in a['metrics'] if (x['dataset'], x['model'], x['split'], x['bin']) == (ds, model, split, group))


lines = [
    '# 고정 checkpoint의 큰 수량 train·validation 적합도 진단',
    '',
    '2026-10-03. **진단 완료 / 후속 방법론 설계 제안.**',
    '',
    '큰 수량에서 남는 오차는 validation에만 나타나는 문제가 아니다. 다만 Taxi·RAF·Instacart의 과소예측과 Intermittent의 과대예측을 하나의 원인으로 묶을 수 없다. 이번 결과는 MLP의 표현력 부족이나 로그 손실의 인과적 책임을 확정하지 않는다.',
    '',
    '## 범위와 실행',
    '',
    '- 네 데이터 × 기존 History MLP / All-available /8 × seed42의 **기존 선택 checkpoint 8개**를 사용했다. 모델마다 원래 validation raw RMSE로 선택한 epoch를 유지했다.',
    '- train은 학습 당시의 다음 사건 loader와 target 모집단을 재구성하고, 기존 TRAIN 수량 경계의 각 구간에서 최대2,048건을 단순무작위 비복원 추출했다. 추출 seed와 경계는 추론 전 계약에 고정했다. 두 모델은 동일 표본을 평가했다.',
    '- 전체 지표는 `N구간/n표본` 가중 추정이다. train은 전수 평가가 아니다. 표본추출 오차의 설계 기반 표준오차를 metrics.csv에 별도로 기록했다. 이는 seed·개체·미래 모집단의 일반화 불확실성을 나타내지 않는다.',
    '- 기존 MLP의 전체 validation 예측604,976건은 원본 SHA와 checkpoint 연결을 검증해 재사용했다. All-available의 원본 전체 validation 예측은 이 경로에 없어 고정 층화 표본을 새 CPU 추론했다. 모델 간 직접 비교는 동일 validation 표본으로 한다.',
    '- 로컬 CPU2스레드, eval/inference 모드, 고정 동결 소스 사용. 원격 호출·임대·새 학습·optimizer update·checkpoint 재선택·Test 성능/예측 열람은 없다. 데이터 파일은 byte SHA를 확인한 뒤 train/validation 행만 필터링하여 읽었다.',
    '- 본 세션의 기존 수정 파일과 원고를 변경하지 않았다. 새 파일은 이 진단 폴더에만 작성했다.',
    '',
    '## 1. 기존 MLP는 큰 수량을 train에서도 놓치는가?',
    '',
    '아래는 각 데이터의 미리 정해진 **최상위 수량 구간**이다. train은 고정 표본, validation은 기존 MLP의 전체 validation 중 해당 구간이다. RMSE는 낮을수록 좋고, 편향은 예측−정답이다. 서로 다른 데이터의 RMSE 절댓값을 직접 비교하지 않는다.',
    '',
    '| 데이터 / 구간 | train 평가수 / 해당 구간 전체수 | train RMSE | validation RMSE | train 편향 | validation 편향 |',
    '|---|---:|---:|---:|---:|---:|',
]
for ds in c['datasets']:
    r = receipts[ds, c['models'][0]]
    tr = metric(ds, c['models'][0], 'train', 4)
    va = next(x for x in r['full_validation_metrics'] if x['bin'] == 4)
    N = r['splits']['train']['strata'][4]['population_n']
    lines.append(f"| {names[ds]} >{r['quantity_bounds'][-1]:g} | {tr['n_evaluated']:,} / {N:,} | {tr['rmse']:.3f} | {va['rmse']:.3f} | {tr['bias']:+.3f} | {va['bias']:+.3f} |")
lines += [
    '',
    '**Taxi:** >3449의 train383건과 validation79건은 모두 평가했다. 큰 수량의 오차와 음의 편향은 train에도 남는다. 그러나 같은 구간의 직전 수량 기준선보다 MLP가 좋으므로 “수량 수준을 전혀 학습하지 못했다”고 해석하면 안 된다.',
    '',
    '**RAF:** >200의 train205건과 validation50건을 모두 평가했다. train 실제 수량 평균433.89에 대해 예측 평균83.17로, 큰 수량에서 상당한 축소 예측이 이미 train에서 나타난다. 이 구간의 train MLP RMSE458.81은 직전 수량426.56, 관측창 평균412.18보다 높다. 다만 전체 validation에서는 MLP33.90이 직전 수량44.57·평균36.85보다 낮다. 큰 정답 구간만 보고 기준선으로 전부 대체하면 안 된다.',
    '',
    '**Instacart:** >35 구간의 train RMSE23.91과 전체 validation 구간 RMSE24.37이 가깝고 양쪽 편향은 약−21이다. validation에서만 발생한 급격한 붕괴라는 설명보다, train에서도 남는 수량 수준 추정 문제를 먼저 검토할 근거다. train의 큰 수량 사건은19,228건이므로 “아예 학습 사례가 없다”는 설명은 맞지 않는다. 같은 개체·맥락 조합의 충분성까지 확인한 것은 아니다.',
    '',
    '**Intermittent:** >187에서 기존 MLP는 train·validation 모두 양의 편향이다. 실제 수량 평균은 train표본262.82·validation270.64이고 RMSE는9.47·7.50이다. 다른 데이터와 같은 “대수량 과소예측” 처방을 적용하면 안 된다.',
    '',
    'train과 validation의 모집단·수량분포·이력분포는 같지 않다. 특히 RAF >200 실제 수량 평균은433.89 대368.66이다. RMSE 비율을 그대로 일반화 격차나 학습 부족의 증거로 해석하지 않는다. 이 결과는 선택 checkpoint 한 시점의 학습 적합도이며 전체 학습 궤적이나 모델이 달성할 수 있는 최적 적합도를 측정한 것이 아니다.',
    '',
    '## 2. 보정 방식만 달랐던 기존 대조에서는 무엇이 달라지는가?',
    '',
    '아래 전체 RMSE는 두 모델 모두 **같은 층화 표본의 가중 추정**이다. 앞 표의 전체 validation 기반 최상위 구간 수치와 평가 범위를 구분한다.',
    '',
    '| 데이터 | MLP train | All-available train | MLP validation | All-available validation |',
    '|---|---:|---:|---:|---:|',
]
for ds in c['datasets']:
    vals = [metric(ds, m, s, -1)['rmse'] for s in ['train', 'validation'] for m in c['models']]
    lines.append('| ' + names[ds] + ' | ' + ' | '.join(f'{v:.4f}' for v in vals) + ' |')
lines += [
    '',
    '- **Taxi:** 전체와 최상위 수량 구간 모두 train·validation이 개선된다. 최상위 구간은 train489.25→425.71, validation430.41→356.60이다. CNN/RNN을 도입하지 않고도 개선된 구체적 사례다. 기존 분기 활성화·보정 용량·보정 크기·최적화의 결합을 우선 분석할 근거이며 어느 하나의 순수 효과로 단정하지 않는다.',
    '- **RAF:** train 전체와 최상위 구간은 거의 그대로이고 validation 전체도 뚜렷한 이득이 없다. 단순 분기 활성화 확대를 공통 해결책으로 채택할 근거가 없다.',
    '- **Instacart:** 동일 표본에서 두 모델의 전체 차이가 작다. 뚜렷한 효과나 통계적 동등성을 주장하지 않는다.',
    '- **Intermittent:** train은 개선되지만 validation 전체는 악화한다. 최상위 구간 편향도 기존 MLP의 양수에서 All-available의 음수로 바뀐다. “더 강한 보정이 항상 좋다”는 주장을 지지하지 않는다.',
    '',
    '이는 seed42의 서로 다른 원래 선택 epoch 비교다. 세 seed 효과, 새로운 후보의 우월성, 순수 용량 효과 또는 효율 비교가 아니다.',
    '',
    '## 3. 손실 불일치 가설은 어디까지 뒷받침되는가?',
    '',
    '- 현재 수량 학습의 log-MSE와 평가의 raw RMSE가 서로 다른 목적이라는 코드 사실은 유지된다.',
    '- Taxi 최상위 train 구간은 log-MSE0.01526에 raw RMSE489.25가 함께 나타난다. 단위가 다르므로 수치의 크기를 직접 비교할 수는 없지만 상대 오차가 작은 큰 수량에서도 절대오차가 커질 수 있음을 보여준다.',
    '- 반면 RAF 최상위 train 구간 log-MSE도4.6906으로 크다. 따라서 모든 실패를 “log 공간에서는 이미 잘 맞추지만 raw RMSE만 나쁘다”로 설명할 수 없다.',
    '- 큰 **정답**으로 구간을 나누면 조건부 평균을 잘 추정하는 모델에서도 그 구간의 과소예측이 생길 수 있다. 음의 편향만으로 모델의 조건부 보정 실패나 로그 손실의 인과성을 확정하지 않는다.',
    '- 이를 보완해 예측값 기준의 고정 구간에서도 평균 예측과 평균 정답을 계산했다(prediction_calibration.csv). RAF·Instacart는 여러 구간에서 평균 예측이 평균 정답보다 낮다. 다만 표본 추정이며 희소 예측 구간도 있어 정식 보정 검정이나 원인 확정은 아니다.',
    '',
    '## 4. 결과에 맞춘 방법론 재정렬',
    '',
    '**수량 정보를 출력부가 활용하지 못하는지 먼저 분리한다 — 다음 작업 / 설계 제안**',
    '- 대상: `paper_research`의 동결 seed42 Backbone과 수량 출력부 사본. Backbone과 시간 출력부는 고정하고, 동일한 기존 수량 head 구조를 log-MSE와 train 고정 스케일로 정규화한 raw-MSE로 각각 재적합하는 진단을 설계한다. 같은 시작 head, 같은 train 표본·예산, validation 기준을 사용한다.',
    '- 목적: Backbone을 바꾸지 않아도 현재 표현에서 더 나은 수량 예측이 가능한지 확인한다. 이는 원인 확인용 head 재적합 대조이며 논문 비교군을 바꾸거나 새 모델 성과로 보고하는 제안이 아니다. 이번 작업에서는 재적합하지 않았다.',
    '- 판정: 같은 표현·head 용량에서 raw 목적의 대조만 validation까지 개선되면 손실/출력 적합 문제가 실제로 관여한다는 근거가 된다. 양쪽 재적합이 좋아지면 공동 학습 또는 head 최적화 문제도 검토한다. 둘 다 실패해도 비선형 정보가 없거나 데이터가 독립이라고 단정할 수 없다.',
    '- raw 손실의 gradient·스케일·clipping을 기록한다. 정규화·손실 변경을 최종 방법으로 채택할 경우에는 그 정책을 비교군에도 동일하게 적용해 구조 효과와 분리해야 한다.',
    '',
    '**보정의 사용 방식과 표현 용량을 분리한다 — 다음 작업 / 위 진단 이후**',
    '- 대상: 기존 History Correction. Taxi의 실제 개선을 출발점으로, 분기 가용성 변경과 폭4→16 변경을 별도 조건으로 둔다. 입력·출력부·손실을 고정하고 큰 수량뿐 아니라 작은 수량·전체 RMSE·MAE·시간 NLL을 함께 평가한다.',
    '- Intermittent는 보정 방향이 바뀌고 validation이 악화한 반례로 함께 유지한다. 단순히 대수량 예측을 일괄 상향하거나 보정 크기를 키우는 방식은 우선 후보로 삼지 않는다.',
    '- 판정: train 적합과 validation이 함께 좋아지는 구조만 후속 후보로 삼는다. 폭만 커진 결과를 새로운 기법이라고 부르지 않는다.',
    '',
    '**CNN과 GRU는 실패 양상에 맞춰 하나씩 선택한다 — 다음 작업 / 표현 병목 근거가 생긴 경우**',
    '- 최근 수량 수준·증가율·변화의 순서가 잔차를 설명하지만 현재 잠재 표현이 이를 활용하지 못한다는 근거가 있으면 Encoder1의 인과적 CNN을 검토한다.',
    '- 긴 관측 이력의 누적·유지가 별도로 필요하다는 근거가 있으면 작은 GRU 보정 경로를 검토한다. 현재 결과만으로 GRU 필요성은 확인되지 않았다.',
    '- CNN+GRU 동시 교체는 첫 단계에서 보류한다. 원인을 두 가지씩 바꾸면 개선·악화의 원인을 해석하기 어렵다. 어떤 후보든 공통 출력부를 유지하고 추가 파라미터·연산비용을 별도로 기록한다.',
    '',
    '위 설계는 순차적으로 해석해야 한다. 고정 계약 아래의 독립 조건 계산은 병렬화할 수 있지만, 현재 원격 학습·임대·다른 세션 작업에는 아무 변경도 가하지 않았다. 새 학습을 시작하는 계획은 이번 진단 완료와 구분한다.',
    '',
    '## 검증과 산출물',
    '',
    '- checkpoint8개의 파일 SHA·tensor SHA·선택 epoch와 동결 import source를 확인했다. 실행 후 파라미터·buffer SHA가 동일했다.',
    '- train/validation target 수·identity SHA·quantity SHA가 원래 학습 계약과 일치했다. 두 모델의 각 split별 표본 identity·정답·이력 길이·가중치 일치8건을 확인했다.',
    '- 실제 forward에서 target 수량·시간을 바꿔도 예측 수량이 변하지 않는지 검증했다. 기존 MLP validation은 데이터마다64건을 기존 전체 예측과 수치 대조했다.',
    '- 독립적인 scalar math.fsum 집계768건이 원 집계와 일치했다. 기존 작업 파일8개와 원고의 시작/종료 SHA가 같았다.',
    '- Instacart의 첫 합성 경계 검사는 진단 코드가 시간 상한30을 넘는 가짜 target을 넣어 실패했다. 원래 모델이 올바르게 거부한 것이다. 지원 범위 안의1/2로 바꾼 뒤 통과했다. 학습 실패가 아니며 모델·관측 정의를 수정하지 않았다(execution_notes.json).',
    '',
]
for name, label in [('contract.json', '사전 고정 진단 계약'), ('metrics.csv', '모든 수량 구간 지표와 표본추출 표준오차'), ('population.csv', 'train·validation 수량 모집단'), ('paired.csv', '동일 표본의 두 모델 비교'), ('prediction_calibration.csv', '예측값 기준 구간의 평균 정답'), ('verification.json', '독립 검증 기록'), ('diagnose.py', '동결 checkpoint 진단 코드')]:
    lines.append(f'- [{label}]({OUT / name})')
lines += ['', '**남은 작업:** 수량 head만 고정 조건에서 재적합하는 진단 명세 확정 → 결과에 따른 보정 가용성/폭 대조 → 필요성이 확인된 CNN 또는 GRU 후보 설계. 현재 완료된 표본 진단을 반복 실행할 필요는 없다.', '']
(OUT / 'README.md').write_text('\n'.join(lines))
print(OUT / 'README.md')
