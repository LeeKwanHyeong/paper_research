# Frozen-B shared-block objective-transfer audit

## 결론

H1 또는 H2에서 공통 task-specific 분기를 구현할 필요조건이 성립하지 않았다. 따라서 이 후보는 구현하거나 GPU에서 학습하지 않는다.

이 판정은 고정 B의 train-only 1차 미분 방향을 비교한 필요조건 검사다. 실제 validation 성능이나 인과적 성능 향상을 뜻하지 않는다.

## 공통 경계 판정

| 경계 | Intermittent | Taxi | Instacart | 세 데이터셋 공통 |
|---|:---:|:---:|:---:|:---:|
| H1 | 실패 | 실패 | 실패 | 실패 |
| H2 | 실패 | 실패 | 실패 | 실패 |
| fused | 실패 | 실패 | 실패 | 실패 |

`fused`는 기존 post-LMM readout 실험과 겹치는 report-only 대조 경계이므로 단독으로 통과해도 새 Backbone 후보를 열지 않는다.

## 측정 계약

- 기존 4,096개 train target과 series-disjoint 2개 fold를 그대로 재사용했다.
- 각 방향의 normalized K=1 시간 head는 source fold 2,048개에서만 100 epoch 적합했고 validation selector나 held-out test를 사용하지 않았다.
- B encoder, Hard-LMM bank, quantity head와 checkpoint state는 전후 bitwise 동일하다.
- 각 경계에 가상 zero-init `64×64` linear residual을 두고 전체 valid sequence의 gradient를 계산했다. 실제 forward나 모델 코드는 바꾸지 않았다.
- separated upper bound는 quantity와 time 두 adapter의 합산 update norm을 shared adapter 하나와 동일하게 맞췄다.

## 해석 한계

- B checkpoint 자체는 과거 validation raw RMSE로 선택됐다. 이번 실행은 그 고정 checkpoint 안의 objective conflict를 재는 감사다.
- shared 비교는 normalized K=1 time objective를 쓰는 counterfactual이며 B가 과거에 학습한 legacy time update를 재현한 것이 아니다.
- 결과는 infinitesimal one-step 방향이다. 유한 epoch 학습, optimizer dynamics, top-4 membership 변화는 포함하지 않는다.
- 시간 head는 source-fold measurement instrument이며 최종 모델 후보가 아니다.

정량 directional dot, shared controls, cyclic alignment control과 fold별 head 적합 기록은 `analysis.json`과 `metrics.csv`에 저장했다.
