# TitanTPP 외부 비교: 두 데이터의 수량 우위와 Instacart의 한계

증거 기준: 2026-10-01 07:07 KST. 추가 TPP는 07:02:47 KST 관측을 사용한다. 현재 실행 중인 조건을 완료 결과에 포함하지 않는다.

## 결론

**대표 모델은 기존 TitanTPP History MLP를 유지한다.** Taxi·Intermittent에서는 공통 head로 비교한 외부 TPP 6개 모두보다 3seed 평균 MAE·RMSE가 낮으며, 같은 seed별 비교 36쌍 모두에서 두 수량 지표가 낮다. 시간 NLL까지 일관되게 우월하지는 않다. Instacart에서는 최상위 외부 모델 대비 수량 우위를 확보하지 못했다.

활성 분기 정규화는 seed42에서 Taxi의 수량 지표를 개선했으나 시간 NLL은 악화됐고, Intermittent는 세 지표 모두 악화됐다. Instacart의 기존 MLP 대비 개선은 MAE 0.025%, RMSE 0.120%로 작으며 모든 외부 모델을 앞서지는 못한다. 대표 구조를 정규화로 교체할 근거는 부족하다.

## 동일 조건의 3seed 전체 비교

아래 값은 seed42·52·62 평균 ± 표본표준편차(ddof=1). 모든 지표는 낮을수록 좋다. MAE·시간 NLL도 RMSE로 선택한 동일 checkpoint 값이다. 서로 다른 데이터의 절대 오차나 시간 NLL을 직접 비교하지 않는다.

### Taxi

| 모델 | MAE ↓ | RMSE ↓ | 시간 NLL ↓ |
|---|---:|---:|---:|
| TitanTPP MLP | 25.623661 ± 0.832006 | 79.711120 ± 2.825983 | 1.038668 ± 0.284331 |
| RMTPP | 28.512359 ± 0.634433 | 94.728347 ± 3.400600 | 1.579776 ± 0.940037 |
| THP | 32.409513 ± 0.814403 | 107.597656 ± 4.504045 | 0.662929 ± 0.017446 |
| NHP | 94.269352 ± 1.875179 | 318.814784 ± 5.301108 | 0.651643 ± 0.002059 |
| SAHP | 34.226026 ± 0.453948 | 120.527337 ± 1.999633 | 0.684821 ± 0.014588 |
| S2P2 (공통 head) | 33.023150 ± 0.556866 | 106.354045 ± 1.913530 | 0.665765 ± 0.024178 |
| AttNHP (공통 head) | 35.981089 ± 1.608774 | 127.129033 ± 8.991916 | 0.687637 ± 0.019429 |

### Intermittent

| 모델 | MAE ↓ | RMSE ↓ | 시간 NLL ↓ |
|---|---:|---:|---:|
| TitanTPP MLP | 0.700523 ± 0.054055 | 1.673034 ± 0.081887 | 0.497111 ± 0.209463 |
| RMTPP | 0.894885 ± 0.049405 | 2.549928 ± 0.160402 | 0.274759 ± 0.004744 |
| THP | 0.899331 ± 0.066103 | 2.753352 ± 0.178427 | 0.291066 ± 0.004536 |
| NHP | 4.518767 ± 0.170590 | 12.875455 ± 1.040504 | 0.461293 ± 0.096945 |
| SAHP | 1.861945 ± 0.089926 | 7.029832 ± 0.767268 | 0.344800 ± 0.013771 |
| S2P2 (공통 head) | 0.935886 ± 0.008569 | 2.575842 ± 0.076563 | 0.234933 ± 0.018928 |
| AttNHP (공통 head) | 1.041263 ± 0.093761 | 3.079937 ± 0.298175 | 0.294622 ± 0.056852 |

### Instacart

| 모델 | MAE ↓ | RMSE ↓ | 시간 NLL ↓ |
|---|---:|---:|---:|
| TitanTPP MLP | 3.991574 ± 0.002592 | 5.882742 ± 0.004771 | 2.807087 ± 0.004149 |
| RMTPP | 3.985184 ± 0.005197 | 5.877541 ± 0.021322 | 2.805148 ± 0.004840 |
| THP | 4.001303 ± 0.002485 | 5.858178 ± 0.025304 | 2.805649 ± 0.001553 |
| NHP | 4.492282 ± 0.030410 | 6.761279 ± 0.053816 | 2.813234 ± 0.008020 |
| SAHP | 3.997910 ± 0.009945 | 5.877591 ± 0.006729 | 2.801849 ± 0.001259 |

S2P2·AttNHP의 Instacart 3seed는 미완료여서 이 평균표에 포함하지 않는다. 아래 seed42 표에서만 확정된 동일 seed 결과를 비교한다.

## 외부 비교군 대비 대표 MLP의 개선율과 seed별 승패

개선율 = (비교군 평균 − MLP 평균) / 비교군 평균 × 100. 음수는 MLP 악화. 승수는 완료된 동일 seed 비교만 센다.

| 데이터 | 비교군 | 완료 seed | MAE 개선율 | RMSE 개선율 | MAE 승 | RMSE 승 | 시간 NLL 승 |
|---|---|---:|---:|---:|---:|---:|---:|
| Taxi | RMTPP | 3 | 10.13% | 15.85% | 3/3 | 3/3 | 2/3 |
| Taxi | THP | 3 | 20.94% | 25.92% | 3/3 | 3/3 | 0/3 |
| Taxi | NHP | 3 | 72.82% | 75.00% | 3/3 | 3/3 | 0/3 |
| Taxi | SAHP | 3 | 25.13% | 33.86% | 3/3 | 3/3 | 0/3 |
| Taxi | S2P2 (공통 head) | 3 | 22.41% | 25.05% | 3/3 | 3/3 | 0/3 |
| Taxi | AttNHP (공통 head) | 3 | 28.79% | 37.30% | 3/3 | 3/3 | 0/3 |
| Intermittent | RMTPP | 3 | 21.72% | 34.39% | 3/3 | 3/3 | 0/3 |
| Intermittent | THP | 3 | 22.11% | 39.24% | 3/3 | 3/3 | 0/3 |
| Intermittent | NHP | 3 | 84.50% | 87.01% | 3/3 | 3/3 | 2/3 |
| Intermittent | SAHP | 3 | 62.38% | 76.20% | 3/3 | 3/3 | 0/3 |
| Intermittent | S2P2 (공통 head) | 3 | 25.15% | 35.05% | 3/3 | 3/3 | 0/3 |
| Intermittent | AttNHP (공통 head) | 3 | 32.72% | 45.68% | 3/3 | 3/3 | 0/3 |
| Instacart | RMTPP | 3 | -0.16% | -0.09% | 0/3 | 1/3 | 0/3 |
| Instacart | THP | 3 | 0.24% | -0.42% | 3/3 | 1/3 | 1/3 |
| Instacart | NHP | 3 | 11.15% | 12.99% | 3/3 | 3/3 | 3/3 |
| Instacart | SAHP | 3 | 0.16% | -0.09% | 2/3 | 1/3 | 0/3 |
| Instacart | S2P2 (공통 head) | 2 | 3seed 미완료 | 3seed 미완료 | 1/2 | 0/2 | 0/2 |
| Instacart | AttNHP (공통 head) | 1 | 3seed 미완료 | 3seed 미완료 | 1/1 | 0/1 | 1/1 |

## 활성 분기 정규화와 외부 모델: seed42만 비교

Taxi·Intermittent 정규화는 RTX4090, Instacart 정규화는 RTX4090 2epoch 이후 개인 RTX5080으로 승인된 복구를 수행했다. 하드웨어가 다르고 한 seed뿐이므로 일반적인 우월성이나 속도 차이로 해석하지 않는다.

### Taxi — seed42

| 모델 | 선택 epoch | MAE ↓ | RMSE ↓ | 시간 NLL ↓ |
|---|---:|---:|---:|---:|
| TitanTPP MLP | 60 | 26.499368 | 82.483934 | 0.842735 |
| 활성 분기 정규화 | 129 | 25.361965 | 79.297432 | 1.171106 |
| RMTPP | 123 | 28.230795 | 94.219012 | 1.350468 |
| THP | 70 | 32.607404 | 108.883264 | 0.653281 |
| NHP | 8 | 96.186537 | 312.810943 | 0.653590 |
| SAHP | 22 | 34.279016 | 120.805646 | 0.668265 |
| S2P2 (공통 head) | 70 | 32.645699 | 104.412232 | 0.692848 |
| AttNHP (공통 head) | 59 | 36.974967 | 136.554973 | 0.668677 |

정규화의 기존 MLP 대비 수량 개선율: qty_mae 4.2922%, qty_rmse 3.8632%. 시간 NLL 변화(정규화−MLP): +0.328371.

### Intermittent — seed42

| 모델 | 선택 epoch | MAE ↓ | RMSE ↓ | 시간 NLL ↓ |
|---|---:|---:|---:|---:|
| TitanTPP MLP | 10 | 0.638480 | 1.578487 | 0.346835 |
| 활성 분기 정규화 | 12 | 0.712465 | 1.707505 | 0.357442 |
| RMTPP | 26 | 0.909257 | 2.690288 | 0.269831 |
| THP | 68 | 0.975047 | 2.657052 | 0.285882 |
| NHP | 2 | 4.383098 | 12.114985 | 0.380285 |
| SAHP | 35 | 1.857280 | 7.886148 | 0.333986 |
| S2P2 (공통 head) | 60 | 0.940676 | 2.487482 | 0.253593 |
| AttNHP (공통 head) | 36 | 1.136850 | 3.383346 | 0.264412 |

정규화의 기존 MLP 대비 수량 개선율: qty_mae -11.5877%, qty_rmse -8.1735%. 시간 NLL 변화(정규화−MLP): +0.010607.

### Instacart — seed42

| 모델 | 선택 epoch | MAE ↓ | RMSE ↓ | 시간 NLL ↓ |
|---|---:|---:|---:|---:|
| TitanTPP MLP | 30 | 3.991502 | 5.878730 | 2.803218 |
| 활성 분기 정규화 | 30 | 3.990506 | 5.871661 | 2.802719 |
| RMTPP | 6 | 3.979249 | 5.858126 | 2.799876 |
| THP | 30 | 3.998591 | 5.852548 | 2.804355 |
| NHP | 28 | 4.495765 | 6.758617 | 2.806673 |
| SAHP | 30 | 3.999184 | 5.883638 | 2.800402 |
| S2P2 (공통 head) | 72 | 4.005765 | 5.838054 | 2.796148 |
| AttNHP (공통 head) | 72 | 3.995990 | 5.847649 | 2.811169 |

정규화의 기존 MLP 대비 수량 개선율: qty_mae 0.0249%, qty_rmse 0.1202%. 시간 NLL 변화(정규화−MLP): -0.000499.

## 내부 구조 비교와 Gate 탐색 — 외부 비교와 분리

기존의 내부 대안과 불리한 탐색 결과도 보존한다. Full이 Taxi 평균 수량 지표에서 MLP보다 조금 좋았다는 사실은 유지하며, MLP 채택을 모든 내부 대안 대비 수치상 1위라는 주장으로 바꾸지 않는다.

| 데이터 | 구조 | seed 수 | MAE 평균 | RMSE 평균 | 시간 NLL 평균 |
|---|---|---:|---:|---:|---:|
| Taxi | B: 이력 보완 없음 | 3 | 28.754505 | 90.509264 | 0.724678 |
| Taxi | Full: 수준·변화 분리 | 3 | 25.362274 | 78.777378 | 1.182236 |
| Intermittent | B: 이력 보완 없음 | 3 | 0.758760 | 1.780563 | 0.384621 |
| Intermittent | Full: 수준·변화 분리 | 3 | 0.749498 | 1.814042 | 0.471416 |
| Instacart | B: 이력 보완 없음 | 3 | 3.993169 | 5.879573 | 2.806474 |
| Instacart | Full: 수준·변화 분리 | 3 | 3.992555 | 5.885932 | 2.806797 |
| Taxi | TitanTPP MLP | 3 | 25.623661 | 79.711120 | 1.038668 |
| Taxi | 수준만 | 3 | 27.491678 | 85.099490 | 0.892129 |
| Taxi | 변화만 | 3 | 26.501030 | 83.103437 | 0.924542 |
| Taxi | Full 정적 검색 제거 | 3 | 25.275426 | 79.845416 | 1.073298 |
| Intermittent | TitanTPP MLP | 3 | 0.700523 | 1.673034 | 0.497111 |
| Intermittent | 수준만 | 3 | 0.715913 | 1.701751 | 0.414476 |
| Intermittent | 변화만 | 3 | 0.742189 | 1.820891 | 0.725432 |
| Intermittent | Full 정적 검색 제거 | 3 | 0.736530 | 1.817652 | 0.315958 |
| Instacart | TitanTPP MLP | 3 | 3.991574 | 5.882742 | 2.807087 |
| Instacart | 수준만 | 3 | 3.994353 | 5.889603 | 2.806863 |
| Instacart | 변화만 | 3 | 3.989909 | 5.878483 | 2.806435 |
| Instacart | Full 정적 검색 제거 | 3 | 3.993352 | 5.886773 | 2.806466 |

Gate 6완료조건은 seed42 탐색이며 위 3seed 표에 합산하지 않는다. Full Taxi C/D는 미시작 보류다.

| 데이터 | 변형 | 선택 epoch | MAE | RMSE | 시간 NLL |
|---|---|---:|---:|---:|---:|
| Intermittent | MLP 조건부 Gate | 5 | 0.734496 | 1.631734 | 0.316678 |
| Intermittent | MLP 상수 Gate | 4 | 0.708235 | 1.649104 | 0.363903 |
| Taxi | MLP 조건부 Gate | 27 | 27.722204 | 86.737720 | 0.678927 |
| Taxi | MLP 상수 Gate | 134 | 26.122403 | 80.208327 | 1.304540 |
| Intermittent | Full 조건부 Gate | 41 | 0.731695 | 1.782706 | 0.304817 |
| Intermittent | Full 상수 Gate | 57 | 0.668810 | 1.699489 | 0.365952 |

## 해석과 다음 실험

- 논문에서 방어 가능한 현재 주장: 동일한 관측 정보·공통 예측 head·RMSE 선택 규칙 아래, TitanTPP MLP는 Taxi와 Intermittent의 validation 다음 사건 수량 예측에서 외부 6개 TPP 비교군보다 반복적으로 낮은 MAE·RMSE를 보였다.
- Instacart에서 모든 비교군보다 우월하다는 주장은 지지되지 않는다. MAE와 RMSE의 비교 순위가 다르다. 정규화의 작은 수치 개선은 통계적 유의성이나 동등성 검증이 아니다.
- 시간 NLL 악화는 제외하지 않는다. 수량 예측 개선과 발생 간격 분포 적합도의 상충을 보고한다.
- 비교군은 공통 head·loss·입력으로 연결한 통제 비교다. 원 논문의 모든 native head 및 각 모델별 최적 튜닝 결과를 이겼다는 주장이 아니다.
- 다음 RAF 24조건은 대표 MLP, 정규화 MLP, 외부 6개를 같은 새로운 데이터에서 비교한다. 결과가 불리해도 모델·seed를 제외하지 않는다.

## 검증 범위와 남은 감사

모든 표는 validation-only이다. held-out/test 성능을 읽거나 추가 평가하지 않았다. 선택 규칙은 최초 최소 raw 수량 RMSE이며 MAE 최저 epoch를 별도로 고르지 않았다.

기존 core·외부4종 최종 감사와 Gate/RunPod 정규화 CPU 감사는 기존 증적을 재사용한다. 추가 TPP는 terminal/replay 소형 기록 SHA 검증 완료 결과이며 checkpoint binary CPU 감사는 아직 미완료다. 방금 완료한 5080 Instacart 정규화는 성공 manifest와 selected/last 재평가를 확인했으나 로컬 binary 회수·CPU 감사가 남았다. 이 비교표 작성이 그 감사를 대체하지 않는다.

추가 TPP Instacart는 관측시점 S2P2 seed42·52, AttNHP seed42가 완료, AttNHP seed52가 진행 중, seed62 두 조건은 미시작이다. 진행 중 best는 이 완료 성능표에서 제외했다.

개인5080 추가 cloud 임대료는 0이고 전기료는 미측정이다. RunPod 정규화의 계정잔액 감소 $3.036663은 Pod별 최종 청구서가 아니다. 이종 GPU·복구·다른 종료 epoch의 학습 시간을 단독 속도 우위로 비교하지 않는다.

## 원본 및 재현

`selected_conditions.csv`는 조건별 원본 수치, `comparison.json`은 집계·승패, `verification.json`은 검증 및 파일 SHA를 담는다. `build_analysis.py`로 저장된 원본에서 재생성할 수 있다.

- [reports/titantpp_core_ablation_execution_20260928_v1/final_conditions.csv](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_core_ablation_execution_20260928_v1/final_conditions.csv) — SHA256 `3501b98f0f3c5d250c6ddb7f8309b3455ec1bb908f0844928ec504d71c826765`
- [search_artifacts/titantpp_additional_tpp_20260930_v1/hourly_comparison/20260930T220247154140Z/comparison.json](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_additional_tpp_20260930_v1/hourly_comparison/20260930T220247154140Z/comparison.json) — SHA256 `c51d24c9baaa2f12a73e1f6317f3276f257c945bdab0318d94e4145f56672dca`
- [reports/titantpp_active_branch_norm_runpod4090_20260930_v1/comparison.json](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_active_branch_norm_runpod4090_20260930_v1/comparison.json) — SHA256 `273065333b9818c2e7af3965ba4e7ac03ca8905adf28fc6b5e3bb86f496bb880`
- [search_artifacts/titantpp_instacart_norm_5080_20260930_v1/monitor/20260930T220711276848Z/snapshot.json](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_instacart_norm_5080_20260930_v1/monitor/20260930T220711276848Z/snapshot.json) — SHA256 `812778a1bac2abc63f372196a48a00da73130c5c6844e992aa33bebd49acb950`
- [reports/titantpp_mlp_gate_execution_20260929_v1/comparison.json](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_mlp_gate_execution_20260929_v1/comparison.json) — SHA256 `cc22f093e5e13161e80d7bf1886d7e48751384a26d7e1f0a156c56ab0d38b07c`
