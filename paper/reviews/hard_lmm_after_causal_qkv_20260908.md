# Causal QKV 이후 Backbone 검토 — 탐색 메모

상태: 기존 코드·학습 이력·선행연구 검토. 새 후보의 확정 계약이나 구현 결과가 아니다.
새 학습·GPU 실행·held-out 평가·checkpoint 재선택은 수행하지 않았다.

## 확인된 문제

Causal QKV 후보는 Taxi의 MAE/RMSE를 각각 5.7566%/2.6737% 개선했지만,
선택 epoch113의 legacy time loss가 B의 1.473391에서 25.407218로 증가했다.
Intermittent는 MAE가 2.8499% 개선된 대신 RMSE가 3.2568% 악화했고,
시간 loss도 −3.483794에서 −1.602208로 증가했다. 기존 Instacart 기준 미달도 유지된다.

학습 이력을 검토하면 두 데이터셋의 시간 악화 양상이 다르다.

- Taxi: validation 시간 loss가 e1의1.367691에서 e16의1.364235로 낮아졌다가,
  e49에 B+0.01을 처음 초과하고 후반에 크게 증가했다. 선택 e113에서 train은
  1.221637, validation은25.407218이다. 기록된 어떤 epoch도 B 대비 raw RMSE
  개선과 시간 guardrail을 동시에 충족하지 않았다.
- Intermittent: raw RMSE 최솟값은 e7의1.548392이며 B의1.499555보다 높다.
  e7 이후 시간 loss는 e47의−3.297113까지 매 epoch 개선됐지만,
  validation raw RMSE는 최종3.004042로 악화했다. Train 수량 loss는 감소했다.

따라서 단순 checkpoint 재선택으로 이번 후보를 통과시킬 수 없었다.
또한 두 지표의 반대 움직임만으로 공유 encoder의 gradient 충돌을 단정할 수 없다.
일반화 격차, 시간 head 입력의 민감도, 시간 간격/수량 구간의 영향을 분리해야 한다.

## 코드에서 확인할 경로

후보는 첫 attention block에서 Q/K/V 각각에 별도의 causal kernel3 residual을
추가한다. 따라서 같은 checkpoint 안에서 Q, K, V의 추가 경로를 개별적으로
끄는 비교가 가능하다. 이는 해당 checkpoint의 민감도 진단이며, 새로 학습한
Q/K-only 모델의 성능이나 원래 B 출력과의 동일성을 뜻하지 않는다.

기존 시간 계산은 hidden의 선형 투영을 지수 함수에 넣고 duration 항을 결합한다.
따라서 hidden 변화가 시간 loss에 얼마나 확대되는지, 큰 손실이 일부 표본과
clamp 영역에 집중되는지를 확인할 이유가 있다. 현재 증적은 V 경로가 원인임을
입증하지 않았으며, Q/K와 V의 상호작용도 남아 있다.

## 비교할 두 구조 방향

| 방향 | Backbone에서 달라지는 것 | 선택에 필요한 근거 |
| --- | --- | --- |
| Q/K-only causal mixing | V는 기존 경로를 유지하고 encoder의 사건 attention Q/K만 시간적으로 혼합 | Q/K의 수량 기여와 V의 시간 악화 기여를 분리할 수 있어야 함 |
| 시간·수량에 따라 일부 표현을 분리하는 encoder | encoder 내부 attention 또는 memory read에 작업별 경로를 두되 같은 구조를 세 데이터셋에 적용 | 공통 표현의 충돌/부족이 확인되고 추가 비용이 허용 범위여야 함 |

두 번째 방향은 데이터셋별 모델 분기가 아니라, 모든 데이터셋에서 같은 두 예측
작업을 수행하는 하나의 구조다. 다만 마지막 head 앞에 선형층만 추가한 경우는
Backbone 개선으로 해석하지 않는다. 내부 표현 경로의 실질적인 변경과 효과가 필요하다.

공유 expert와 작업별 gate를 함께 학습하는 설계의 일반적인 선행 사례는
[MMoE, Ma et al., KDD 2018](https://research.google/pubs/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-of-experts/)다.
이는 Hard-LMM에서의 개선 증거가 아니며, 새로운 논문 기여도 적용만으로 성립하지 않는다.
Task gradient 충돌 진단의 참고는
[PCGrad, Yu et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)다.
Gradient 조정 자체는 학습 방법의 변경이므로 Backbone 기여와 구분한다.

## 다음 진단의 목적

관측 이력으로 고정한 train 표본에서 Q/K/V 개별·조합 ablation, hidden 변화,
시간 head 입력과 duration/clamp 항, 수량·시간 loss 기여를 비교한다.
별도로 같은 batch의 작업별 gradient norm·방향을 기존 공유 layer와 새 kernel에
대해 측정하면, 단순 궤적 상관과 gradient 충돌을 구분할 수 있다.

모든 계산은 동결된 source/checkpoint와 미래 target·padding이 차단된 입력을
사용해야 한다. 기존 validation 이력은 원인 가설을 좁히는 탐색 증적이며,
새 후보의 일반화나 통계적 유의성을 증명하지 않는다.

우선은 이 진단으로 두 방향 중 하나를 선택할 근거를 확보한다. 새 e300이나
dataset별 구조·loss·selector 변경은 이 메모에서 확정하지 않는다.
