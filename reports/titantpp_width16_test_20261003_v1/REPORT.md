# Taxi·RAF 폭16 Test 재평가 — 완료

사용자 요청에 따라 기존 Test 분할을 재평가했습니다. 모든 값은 seed42 한 조건이며, 과거에 보고한 3seed 평균과 다릅니다. Validation의 전체 raw 수량 RMSE로 선택한 checkpoint를 고정했습니다(Taxi 폭4/16: epoch60/27, RAF: epoch9/9). Test를 보고 epoch를 바꾸지 않았습니다.

| 데이터 | 폭 | Test N | MAE | RMSE | Time NLL | 큰 수량 N | 큰 수량 MAE | 큰 수량 RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| yellow_trip_hourly | 4 | 8327 | 42.613020 | 140.461827 | 0.812122 | 83 | 765.052864 | 874.432514 |
| yellow_trip_hourly | 16 | 8327 | 32.236180 | 95.931543 | 0.713981 | 83 | 358.370555 | 457.061607 |
| raf_spare_parts | 4 | 5226 | 10.105697 | 39.865993 | 3.991689 | 39 | 360.353786 | 408.167220 |
| raf_spare_parts | 16 | 5226 | 10.246340 | 40.406735 | 3.998955 | 39 | 372.569315 | 415.640768 |

큰 수량은 학습 자료에서 정한 고정 경계 Taxi >3449, RAF >200입니다. 지표는 모두 낮을수록 좋습니다. 시간 점수는 기존 recorded positive-integer duration NLL입니다.

## 결과 해석

- Taxi: 폭16은 전체 RMSE 31.70%, 전체 MAE 24.35%, 전체 Time NLL 12.08% 감소. 큰 수량 RMSE 47.73%, MAE 53.16% 감소. 큰 수량의 평균 prediction-minus-target 오차는 −746.883에서 −84.467로 변했습니다. 이 Test 조건에서 과소예측 완화가 관측되지만 원인 자체를 증명하지는 않습니다.
- RAF: 전체 RMSE 1.36%, MAE 1.39%, Time NLL 0.18% 증가. 큰 수량 RMSE 1.83%, MAE 3.39% 증가. 폭 확대의 이득이 관측되지 않았습니다.
- Taxi Validation의 큰 수량 개선 신호가 기존 Test 분할에서도 나타났으며, 전체 오차 개선까지 이어졌습니다. 두 데이터 모두에서 우월한 구조라는 결론은 지지하지 않습니다.
- 두 개의 검증된 seed42 결과를 보고한 탐색적 비교입니다. 새로운 독립 미접근 평가나 3seed 우월성, 유의성, GPU 효율성으로 표현하지 않습니다. 사후 모델·checkpoint 재선택을 수행하지 않았습니다.
- 극히 작은 Taxi tail 시간 NLL에 대한 상대 변화율은 해석하지 않습니다. 본문은 전체 Time NLL을 사용합니다.

## 검증과 실행

- 완료된 selected checkpoint와 소형 원본 26개를 5080에서 읽기 전용으로 회수하고 manifest SHA를 대조했습니다. source113개 closure와 checkpoint tensor identity가 일치합니다.
- 평가기의 계약·출력·누출·실패 보존 테스트 13개가 통과했습니다. 기존 공통 평가기에 동결 폭16 route 등록 두 줄만 추가했습니다.
- Taxi 8268·RAF 6690건의 full Validation을 로컬 CPU에서 재현했습니다. 원래 CUDA 선택 기록과 rtol=atol=1e-5 내 일치하며 최초 최소 validation RMSE 선택 epoch도 확인했습니다.
- 두 조건 모두 타깃 수량/시간 변조 검사, single/batch 검사, 모델 state와 checkpoint/data/source SHA 불변 검사를 통과했습니다.
- 기존 폭4 Test parquet의 SHA와 target/truth digest를 다시 계산했습니다. 폭16과 모든 대상·정답·과거 이력 필드 및 loader/data SHA가 정확히 일치합니다.
- 로컬 CPU의 단일 worker, 4threads, 각 추론 1800초 상한으로 순차 실행했습니다. 원격 학습·Runtime·예산·원고는 변경하지 않았습니다.
- 각 Test 실행은 Taxi 약16.5초, RAF 약9.5초였습니다. 이는 이번 CPU 작업의 실행 시간 기록이며 논문 효율 비교가 아닙니다.
- 이번 두 selected binary의 로딩·identity·validation 재현을 확인했습니다. Last binary 및 캠페인 전체 원본 회수·CPU 감사는 이 평가의 완료 범위와 별도입니다.

## 남은 작업

1. **진행 중 — 폭16 Intermittent·Instacart 학습 완료를 확인한다.** 현재 동일한 validation 계약에서 진행 중인 나머지 두 조건을 기다립니다.
2. **다음 작업 — 네 데이터의 폭 확대 효과와 한계를 함께 정리한다.** Taxi Test 개선과 RAF 악화를 모두 포함하고, 추가 seed 검증 필요성을 판단합니다. 새 seed 학습이나 추가 Test 평가를 자동 실행하지 않습니다.

근거: execution_contract.json, retrieval_receipt.json, comparison_binding.json, validation_gate.json, runs/test/*/receipt.json, results.json, metrics.csv, completion_receipt.json.

독립 재검증: 원본 예측 parquet 6개와 기준 receipt 2개의 SHA, 개별 target ID 및 part/full target·truth digest를 재계산했습니다. 13,553개 paired target의 13개 대상·정답·이력 필드가 일치하고 모든 집계 수치가 재현됐습니다. 폭4는 기존 GPU 예측을 재사용했고 폭16은 CPU 추론이며, 폭16 full Validation의 GPU/CPU 수치 재현 gate를 통과했습니다. 자세한 범위는 independent_prediction_verification.json에 기록했습니다.
