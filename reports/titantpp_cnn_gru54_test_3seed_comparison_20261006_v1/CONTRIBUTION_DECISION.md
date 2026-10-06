# CNN+GRU54의 Contribution 판단

현재 목표는 TitanTPP 개선 모델 가운데 외부 비교군에 우세한 일관된 모델을 찾아 논문을 강화하는 것이다. 이번 Test3seed 결과는 CNN+GRU54를 그 대표 모델로 확정할 근거를 제공하지 못했다. Validation의 수량 개선과 Test의 수량/시간 결과를 혼합하거나 좋은 데이터만으로 전체 우월성을 주장하지 않는다.

## S2P2에 대한 동일 seed 비교

전체·큰 수량 모두 seed42/52/62의 같은 checkpoint와 모집단을 사용했다. Δ는 CNN−S2P2이며 음수가 좋다.

| 데이터 | 전체 RMSE 평균 Δ ± paired 표본 SD | 개선 seed | 큰 수량 RMSE 평균 Δ ± paired 표본 SD | 개선 seed | 전체 TimeNLL 평균 Δ | 개선 seed |
|---|---:|---:|---:|---:|---:|---:|
| Taxi | +14.8587 ± 36.7034 | 1/3 | +243.3063 ± 227.6781 | 0/3 | +0.6399 | 0/3 |
| Intermittent | −0.3909 ± 0.4947 | 2/3 | −8.6671 ± 9.1088 | 3/3 | +4.4044 | 0/3 |
| RAF | +0.1834 ± 0.4468 | 1/3 | +3.4707 ± 9.4918 | 1/3 | +0.2599 | 1/3 |

Taxi는 Validation에서 수량 평균 우위가 관찰됐지만 Test에서 유지되지 않았다. seed52의 수량 RMSE는 Validation76.8463→Test164.3154이고, Test의 S2P2 대비 +56.0488로 손해가 컸다. 이를 제외하거나 더 좋은 seed/epoch를 재선택하지 않는다.

Intermittent는 외부 six 모두보다 전체·큰 수량 RMSE 평균이 낮았다. 특히 큰 수량은 S2P2를 포함한 외부 six 모두에 대해 3/3 seed가 낮았다. S2P2 대비 전체 MAE도 3/3 seed가 낮았다. 그러나 전체 RMSE는 S2P2 대비 2/3 개선이고 전체 TimeNLL은 0/3 개선이다. 이를 수량에 국한된 관찰로 남긴다.

RAF의 S2P2 대비 평균 전체 RMSE 차이 +0.46%, 큰 수량 +0.85%는 실질적으로 작은 차이이며, 우월성을 입증한 결과가 아니다. 큰 수량 target은 Test39개뿐이고 결정적 이력 평균도 RMSE가 낮다. 강한 단순 기준선과 수량 MAE/시간 지표를 함께 확인해야 한다.

## 다른 외부 모델과 강한 내부 모델

Taxi·RAF는 외부 six 중 각각 5개보다 전체 RMSE 평균이 낮았지만 S2P2보다 높았다. Taxi 큰 수량 RMSE는 S2P2·AttNHP보다 높았다. 외부 비교를 약한 모델만으로 구성해 CNN+GRU54의 우월성을 만들지 않는다.

MLP16 대비 전체 Test RMSE 변화는 Taxi+15.00%, Intermittent−3.79%, RAF−0.72%다. 같은 seed의 전체 RMSE 개선은 각각0/3·2/3·2/3이다. CNN+GRU54가 강한 내부 모델보다 일관되게 우월하다는 근거도 부족하다.

Intermittent 전체 수량 RMSE는 TitanTPP B·모든 가용 분기 MLP·MLP4·MLP12가 CNN+GRU54보다 평균이 낮았고, 큰 수량 RMSE도 MLP12·모든 가용 분기 MLP가 낮았다. Taxi에는 MLP12 등 더 강한 수량 기준선이 존재한다. 단일 지표의 데이터별 1등을 서로 다른 모델에서 골라 하나의 모델 성능처럼 제시하지 않는다.

## 가능한 주장과 아직 불가능한 주장

**현재 기록할 수 있는 결과 — 완료**

- 고정된 CNN+GRU54는 Intermittent Test의 큰 수량 RMSE에서 외부 six보다 3seed 모두 낮았다. S2P2 대비 평균41.01% 감소를 전체 시간 손해와 함께 보고한다.
- Taxi Test에서는 수량 일반화와 seed 안정성 문제가 남고 RAF에서는 S2P2에 대한 우위가 확인되지 않았다. 시간 보정/130개 head 재적합의 Validation 결과도 별도 기록으로 보존한다.

**현재 논문 주장으로 확정하지 않는 범위**

- S2P2보다 대부분 데이터·수량/시간 지표에서 우세한 통합 개선 모델.
- CNN·GRU 각각의 독립 효과 또는 두 구조의 결합 시너지.
- 원고에 사용할 미접근 독립 평가의 재현성·통계적 유의성·동일 GPU 효율 우위.

**다음 대조의 목표를 고정한다 — 다음 작업**

- Taxi 수량 Test 불안정과 Intermittent 낮은 수량 구간 시간 일반화를 해결할 수 있는지 검증하는 계약이 우선이다. 더 강한 MLP12/16·모든 가용 분기 대조·S2P2를 기준선으로 유지한다.
- 수량과 시간 표현/선택의 영향을 분리하되, 이번 Test에 맞춰 checkpoint를 다시 고르거나 같은 Test를 새 독립 검증처럼 사용하지 않는다. 새 학습과 독립 평가를 이번 완료 범위에서 추가 실행하지 않았다.

외부 six는 이 프로젝트의 공통 출력부·기록 정수 간격 likelihood 계약으로 평가된 비교군이다. 각 원 논문의 native 구현보다 우월하다는 뜻으로 확대하지 않는다. Deep Renewal native NB는 연구 기록으로만 보존하며 공통 출력부 비교/원고에서는 제외한다. seed42의 A100 계보와 seed52/62의 RTX 학습 차이도 유지한다.

근거: [전체/큰 수량 Test 정렬표](TEST_TABLES.md), [각 seed](metrics_per_seed.csv), [같은 seed 차이](paired_candidate_deltas.csv), [고정된 판단 기준과 실패 seed](strict_test_gate.json), [시간 일반화 위치 확인](TIME_GENERALIZATION_READOUT.md). CPU 전체 binary 재추론 감사와 새 독립 평가는 미완료다.
