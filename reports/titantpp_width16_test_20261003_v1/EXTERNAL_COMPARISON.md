# 폭16 대 외부 비교군 — 기존 Test, seed42

Same existing Test split and seed42; selected validation RMSE checkpoint; common observed-time/quantity heads; exploratory comparison, not three-seed mean

모든 지표는 각 모델의 validation raw RMSE 선택 checkpoint에서 얻었습니다. 재학습·새 추론 없이 기존 외부 비교군 예측을 재사용했습니다. 기존 3seed 평균과 섞지 않습니다. 공통 수량·관측 시간 head를 사용한 비교이며 native 원논문 모델 전체 재현으로 표현하지 않습니다.

## Taxi

| 모델 | 선택 epoch | MAE | RMSE | Time NLL | 큰 수량 MAE | 큰 수량 RMSE |
|---|---:|---:|---:|---:|---:|---:|
| TitanTPP MLP 폭16 | 27 | 32.236 | 95.932 | 0.714 | 358.371 | 457.062 |
| S2P2 (공통 head) | 70 | 39.518 | 120.317 | 0.701 | 424.456 | 525.614 |
| AttNHP (공통 head) | 59 | 37.599 | 132.801 | 0.692 | 621.620 | 755.448 |
| SAHP | 22 | 38.562 | 133.818 | 0.688 | 548.171 | 734.537 |
| TitanTPP MLP 폭4 | 60 | 42.613 | 140.462 | 0.812 | 765.053 | 874.433 |
| THP | 70 | 46.841 | 165.881 | 0.680 | 1224.999 | 1314.104 |
| RMTPP | 123 | 54.716 | 190.526 | 0.908 | 1244.258 | 1303.835 |
| NHP | 8 | 127.539 | 410.851 | 0.684 | 2610.583 | 2660.200 |

## RAF

| 모델 | 선택 epoch | MAE | RMSE | Time NLL | 큰 수량 MAE | 큰 수량 RMSE |
|---|---:|---:|---:|---:|---:|---:|
| S2P2 (공통 head) | 21 | 10.259 | 39.637 | 3.649 | 353.803 | 399.636 |
| TitanTPP MLP 폭4 | 9 | 10.106 | 39.866 | 3.992 | 360.354 | 408.167 |
| AttNHP (공통 head) | 50 | 9.961 | 40.167 | 4.839 | 370.006 | 413.887 |
| THP | 50 | 10.078 | 40.343 | 3.444 | 372.245 | 416.777 |
| TitanTPP MLP 폭16 | 9 | 10.246 | 40.407 | 3.999 | 372.569 | 415.641 |
| RMTPP | 30 | 9.867 | 40.585 | 4.302 | 378.886 | 422.101 |
| SAHP | 30 | 10.309 | 41.142 | 4.259 | 380.613 | 424.688 |
| NHP | 4 | 9.896 | 41.941 | 3.431 | 395.738 | 437.772 |

## 비교 범위와 해석

- Taxi seed42: 폭16은 이 8개 모델에서 전체 RMSE·MAE와 고수량 RMSE·MAE가 가장 낮습니다. S2P2보다 전체 RMSE 약20.27%, MAE 약18.43%, 고수량 RMSE 약13.04% 낮습니다. Time NLL은 S2P2와 일부 다른 모델보다 높습니다.
- RAF seed42: S2P2가 전체 및 고수량 RMSE가 가장 낮고, 전체 MAE는 RMTPP가 가장 낮습니다. 폭16은 S2P2보다 RMSE 약1.94% 높습니다.
- 각 데이터 내 target/truth/loader/data digest가 모든 모델에서 폭16과 일치하며 예측 parquet SHA를 대조했습니다. Taxi N8327·고수량83, RAF N5226·고수량39입니다.
- 폭16은 CPU에서 새 평가했고 기존 비교군은 과거 GPU 예측을 재사용했습니다. 폭16 CPU full Validation은 native GPU 원본을 사전 지정 허용오차 안에서 재현했습니다. 효율 비교로 사용하지 않습니다.
- 단일 seed·기존 Test 분할의 탐색적 비교입니다. 이미 접근한 Test 결과이며, 다중seed 우월성·유의성·새 독립 평가를 주장하지 않습니다.

원본 경로·receipt SHA는 external_benchmark_comparison.json에 보존했습니다.
