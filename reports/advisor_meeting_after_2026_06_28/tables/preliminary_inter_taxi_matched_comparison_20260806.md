# Intermittent 및 Taxi preliminary validation 비교

- 집계 시점: 2026-08-06 KST
- 집계 단위: seeds 42, 52, 62의 mean +/- sample standard deviation
- checkpoint: 각 run의 best validation total NLL
- 공통 조건: fixed split, batch size 128, learning rate 1e-3, hybrid loss, residual value input, target-only training
- 신규 비교군: 5080 strict e300 validation-only 실행의 RMTPP-matched 및 THP-matched
- Titan 참고 결과: 기존 Intermittent V2 e200 및 Taxi V3b e50 artifact의 validation 지표
- 주의: 기존 Titan artifact와 신규 baseline은 epoch budget, source revision 및 strict 실행 계약이 같지 않으므로 이 표는 preliminary evidence이며 최종 공정 비교표가 아니다.

## 1. Intermittent

낮을수록 좋은 지표는 NLL과 MAE이며, mark accuracy는 높을수록 좋다.

| Model | Budget | Total NLL | Marker NLL | Time NLL | Quantity MAE | Delta-t MAE | Mark accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| RMTPP-matched | e300 | 5.6683 +/- 0.0115 | 0.9873 +/- 0.0041 | 4.6809 +/- 0.0126 | 2.7408 +/- 0.0493 | 41.8872 +/- 0.5030 | **55.183% +/- 0.236%p** |
| THP-matched | e300 | 5.6417 +/- 0.0305 | 0.9931 +/- 0.0033 | 4.6486 +/- 0.0273 | 2.8812 +/- 0.0177 | **40.5947 +/- 0.3284** | 54.235% +/- 0.637%p |
| TitanTPP V2 | e200 | **5.6046 +/- 0.0097** | **0.9869 +/- 0.0048** | **4.6177 +/- 0.0131** | **2.7162 +/- 0.0720** | 41.1990 +/- 0.4479 | 54.697% +/- 0.577%p |

### TitanTPP V2의 상대 변화

| Baseline | Total NLL | Marker NLL | Time NLL | Quantity MAE | Delta-t MAE | Mark accuracy |
|---|---:|---:|---:|---:|---:|---:|
| vs RMTPP-matched | -1.123% (3/3) | -0.040% (2/3) | -1.352% (3/3) | -0.899% (2/3) | -1.643% (3/3) | -0.486%p (1/3) |
| vs THP-matched | -0.657% (3/3) | -0.618% (3/3) | -0.665% (3/3) | -5.728% (3/3) | +1.488% (0/3) | +0.462%p (3/3) |

괄호는 TitanTPP가 해당 seed에서 더 좋았던 횟수다. Intermittent에서 TitanTPP V2는 total NLL과 time NLL이 두 baseline보다 세 seed 모두 낮았다. THP 대비 quantity MAE 개선도 세 seed에서 일관됐다. 다만 RMTPP 대비 quantity MAE 개선은 평균 0.899%이고 2/3 seeds에 그쳤으며, mark accuracy 평균은 RMTPP보다 0.486%p 낮았다. 따라서 이 결과만으로 RMTPP 대비 수량 예측의 확실한 우월성을 주장하기는 어렵고, joint likelihood 및 시간 분포 모델링의 개선 근거가 더 강하다.

THP는 delta-t MAE가 가장 낮지만 TitanTPP보다 time NLL은 높다. 이는 점 예측 오차와 확률분포 적합도가 서로 다른 평가 축이기 때문에 생길 수 있는 결과다.

## 2. Taxi

| Model | Budget | Total NLL | Marker NLL | Time NLL | Quantity MAE | Delta-t MAE | Mark accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| RMTPP-matched | e300 | 1.5803 +/- 0.0032 | 0.2200 +/- 0.0024 | **1.3603 +/- 0.0018** | 65.8580 +/- 2.4748 | **0.7326 +/- 0.0085** | 91.800% +/- 0.117%p |
| THP-matched | e300 | 1.5998 +/- 0.0087 | 0.2283 +/- 0.0085 | 1.3714 +/- 0.0012 | 87.7508 +/- 2.6771 | 0.7528 +/- 0.0224 | 91.461% +/- 0.202%p |
| TitanTPP V3b | e50 | **1.5553 +/- 0.0016** | **0.1915 +/- 0.0020** | 1.3638 +/- 0.0005 | **31.5918 +/- 2.9018** | 0.7411 +/- 0.0106 | **92.211% +/- 0.117%p** |

### TitanTPP V3b의 상대 변화

| Baseline | Total NLL | Marker NLL | Time NLL | Quantity MAE | Delta-t MAE | Mark accuracy |
|---|---:|---:|---:|---:|---:|---:|
| vs RMTPP-matched | -1.581% (3/3) | -12.977% (3/3) | +0.262% (0/3) | -52.030% (3/3) | +1.155% (1/3) | +0.411%p (3/3) |
| vs THP-matched | -2.780% (3/3) | -16.145% (3/3) | -0.555% (3/3) | -63.998% (3/3) | -1.552% (2/3) | +0.750%p (3/3) |

Taxi에서는 TitanTPP V3b가 total NLL, marker NLL, quantity MAE 및 mark accuracy에서 두 baseline을 모든 seed에서 앞섰다. 특히 RMTPP 대비 quantity MAE가 52.0%, THP 대비 64.0% 낮아 현재까지 가장 강한 개선 근거다. 반면 RMTPP와 비교하면 time NLL은 0.262%, delta-t MAE는 1.155% 악화됐다. 따라서 Taxi의 정확한 해석은 전 지표의 보편적 우월성이 아니라, 작은 시간 예측 trade-off와 함께 mark 및 수량 모델링을 크게 개선했다는 것이다.

## 3. 현재 자격 판정

### Preliminary result로 사용 가능

- TitanTPP는 Intermittent와 Taxi 모두에서 mean total validation NLL이 가장 낮고, 세 seed 방향도 일관된다.
- Taxi에서는 quantity MAE와 marker 성능 개선이 크고 세 seed에서 모두 재현된다.
- THP를 표에서 유지해도 Taxi의 TitanTPP 결론은 약해지지 않는다.
- Intermittent에서는 TitanTPP의 total NLL 개선은 분명하지만 RMTPP 대비 quantity MAE 차이는 작고 seed별로 혼재한다.

### 최종 우월성 주장에는 아직 부족

- 신규 baseline은 current source revision의 strict e300 run이지만 Titan 결과는 이전 source revision의 e200/e50 artifact다.
- Titan V3b는 Taxi validation을 사용한 개발 과정을 거쳤으므로 fresh frozen run으로 선택 편향을 통제해야 한다.
- 현재 표는 validation-only 해석이며 held-out test 결론이 아니다.
- seed가 3개뿐이므로 mean +/- std와 방향 일관성을 기술하고 강한 유의성 표현은 사용하지 않는다.

## 4. 수렴 및 다음 실행 판단

- Intermittent RMTPP best epochs: 16, 33, 78
- Intermittent THP best epochs: 14, 25, 35
- Taxi RMTPP best epochs: 61, 79, 138
- Taxi THP best epochs: 43, 24, 42
- Intermittent TitanTPP V2 best epochs: 27, 90, 168 of 200
- Taxi TitanTPP V3b best epochs: 42, 32, 49 of 50

신규 RMTPP와 THP는 네 조합 모두 best epoch가 138 이하이므로 현재 validation evidence만 보면 e800 연장의 우선순위가 낮다. 반면 Taxi TitanTPP V3b는 두 seed가 e50 경계에 가까워 current strict 계약으로 e300까지 fresh rerun하는 것이 우선이다. Intermittent TitanTPP V2도 같은 source revision과 strict 설정으로 e300을 실행해야 최종 직접 비교가 성립한다.

현재 18-run queue가 완료된 뒤에는 Intermittent V2와 Taxi V3b의 3-seed strict e300, 총 6 runs를 우선 실행하고, best epoch와 곡선 경계 여부를 보고 e800 연장을 결정한다. 이 단계까지 held-out test는 계속 잠근다.

## 5. 근거 artifact

- New matched baselines: `/home/leekwanhyeong/workspace/paper_research/search_artifacts/final_fair_matched_rmtpp_thp_e300_20260805`
- Intermittent TitanTPP V2: `search_artifacts/model_enhancement_v2_hybrid_e200_0705`
- Taxi TitanTPP V3b: `search_artifacts/model_enhancement_v3b_taxi_multiseed_e50_0710`
- Final comparison contract: `reports/advisor_meeting_after_2026_06_28/tables/final_rmtpp_titantpp_thp_comparison_contract.csv`
