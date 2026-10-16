# Taxi·Intermittent·RAF 최종 Validation·Test 비교

**로컬 집계 완료:** 폭4·8·12·16과 기존 벤치마크·구조 대조군 123학습 조건을 연결했다. seed42·52·62의 두 split × 전체·TRAIN 기준 tail은 학습 모델 492지표행이며 결정적 수량 기준선 24행을 더해 총 516행이다.

**MLP 폭16은 사용자 기준선으로 유지한다.** 폭4는 비교군이다. 아래 결과는 이미 접근했던 기존 Test split에 대한 사후 비교이며, 새로운 독립 Test나 Test를 사용한 모델 선택으로 주장하지 않는다.

- [Validation 전체](VALIDATION_TABLES.md) / [Validation tail](VALIDATION_TAIL_TABLES.md) / [Validation paired Δ](VALIDATION_PAIRED_DELTAS.md)
- [Test 전체](TEST_TABLES.md) / [Test tail](TEST_TAIL_TABLES.md) / [Test paired Δ](TEST_PAIRED_DELTAS.md)
- [모든 seed 지표](metrics_per_seed.csv) / [평균·표본 SD](metrics_aggregated.csv) / [선택 epoch와 동결 source](selected_epochs.csv)

## 폭4·8·12·16의 전체·tail RMSE

| 데이터 | split | 범위 | 폭4 | 폭8 | 폭12 | 폭16 · 기준선 |
|---|---|---|---:|---:|---:|---:|
| Taxi | Validation | overall | 79.711 ± 2.826 | 77.735 ± 0.681 | 79.510 ± 2.198 | 77.826 ± 4.161 |
| Taxi | Validation | tail | 384.982 ± 39.407 | 376.660 ± 10.552 | 378.661 ± 17.727 | 376.664 ± 27.048 |
| Taxi | Test | overall | 118.624 ± 19.570 | 139.415 ± 16.272 | 103.916 ± 9.105 | 113.393 ± 16.502 |
| Taxi | Test | tail | 646.423 ± 197.467 | 849.404 ± 54.131 | 468.622 ± 44.440 | 594.683 ± 121.973 |
| Intermittent | Validation | overall | 1.673 ± 0.082 | 1.664 ± 0.009 | 1.732 ± 0.036 | 1.687 ± 0.048 |
| Intermittent | Validation | tail | 8.189 ± 0.598 | 7.711 ± 0.125 | 8.807 ± 0.247 | 7.924 ± 0.469 |
| Intermittent | Test | overall | 2.761 ± 0.079 | 2.903 ± 0.180 | 2.799 ± 0.190 | 2.936 ± 0.083 |
| Intermittent | Test | tail | 14.962 ± 4.429 | 13.079 ± 2.499 | 11.700 ± 1.972 | 13.785 ± 1.881 |
| RAF | Validation | overall | 33.951 ± 0.059 | 33.938 ± 0.071 | 34.050 ± 0.085 | 34.083 ± 0.087 |
| RAF | Validation | tail | 325.837 ± 1.235 | 326.227 ± 2.112 | 327.743 ± 1.946 | 329.857 ± 1.087 |
| RAF | Test | overall | 39.795 ± 0.075 | 39.927 ± 0.412 | 40.011 ± 0.245 | 40.248 ± 0.169 |
| RAF | Test | tail | 406.965 ± 1.253 | 408.309 ± 7.281 | 409.415 ± 4.986 | 414.097 ± 2.615 |

## 폭8·12와 기준선의 이득·손해를 함께 읽기

수량 열의 변화율은 각 split에서 폭16 평균 대비다. 전체·tail Time NLL은 절대 Δ(비교 모델 평균 − 폭16 평균)로 표시한다. 특히 거의 0인 tail Time NLL을 분모로 한 비율은 해석을 왜곡하므로 절대 Δ와 과학 표기를 사용했다. 계산된 모든 비율과 분모는 CSV에 보존했다.

| 데이터 | split | 모델 | 전체 RMSE 변화율 | 전체 MAE 변화율 | 전체 Time NLL Δ | tail RMSE 변화율 | tail MAE 변화율 | tail Time NLL Δ | 전체 RMSE 개선 seed | 전체 RMSE 최대 Δ seed / Δ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Taxi | Validation | TitanTPP MLP 폭8 | -0.12% | -0.55% | +0.2345 | -0.00% | -1.26% | -1.299e-37 | 1/3 | 52 / +3.679 |
| Taxi | Validation | TitanTPP MLP 폭12 | +2.16% | +1.78% | -0.0952 | +0.53% | -0.65% | +7.579e-25 | 1/3 | 52 / +3.125 |
| Taxi | Test | TitanTPP MLP 폭8 | +22.95% | +15.66% | +0.2297 | +42.83% | +50.01% | -9.766e-29 | 0/3 | 42 / +32.489 |
| Taxi | Test | TitanTPP MLP 폭12 | -8.36% | -5.33% | -0.0921 | -21.20% | -23.30% | +2.534e-22 | 2/3 | 42 / +11.072 |
| Intermittent | Validation | TitanTPP MLP 폭8 | -1.33% | +1.18% | -0.0349 | -2.68% | -1.05% | -1.5699 | 2/3 | 42 / +0.024 |
| Intermittent | Validation | TitanTPP MLP 폭12 | +2.70% | -0.33% | -0.0563 | +11.14% | +12.38% | -1.5856 | 0/3 | 42 / +0.067 |
| Intermittent | Test | TitanTPP MLP 폭8 | -1.13% | +1.93% | -20.2135 | -5.12% | -11.38% | -1.9461 | 2/3 | 62 / +0.231 |
| Intermittent | Test | TitanTPP MLP 폭12 | -4.65% | -7.33% | -23.1505 | -15.12% | -16.94% | -2.0793 | 2/3 | 62 / +0.109 |
| RAF | Validation | TitanTPP MLP 폭8 | -0.43% | +0.02% | +0.0524 | -1.10% | -1.52% | +0.1585 | 2/3 | 42 / +0.025 |
| RAF | Validation | TitanTPP MLP 폭12 | -0.10% | -0.53% | +0.0362 | -0.64% | -0.61% | +0.0860 | 3/3 | 52 / -0.027 |
| RAF | Test | TitanTPP MLP 폭8 | -0.80% | -0.15% | +0.1806 | -1.40% | -2.33% | +0.4318 | 3/3 | 42 / -0.060 |
| RAF | Test | TitanTPP MLP 폭12 | -0.59% | -0.46% | +0.2108 | -1.13% | -1.87% | +0.4065 | 3/3 | 52 / -0.063 |

각 지표의 부호와 seed 수를 별도로 판단한다. 평균 RMSE가 낮아도 모든 seed가 좋아졌다는 뜻은 아니며, tail은 일부 사건 집합이므로 전체 성능을 대신하지 않는다. 최대 Δ가 음수라면 세 seed 모두 개선한 상태에서 가장 작은 개선을 뜻한다. 전체 수량·tail 수량·시간이 같은 방향인지 paired 표에서 확인한다.

**Taxi:** 폭12 Test RMSE 103.916 ± 9.105와 tail RMSE 468.622 ± 44.440는 폭16 평균보다 낮다. 전체 RMSE 개선은 2/3 seed이며 seed42 Δ는 +11.072다. 폭8은 폭16 대비 전체 RMSE가 세 seed 모두 악화했다. 폭12의 Time NLL 평균 0.8882는 S2P2 0.6900보다 높으며, 모든 가용 분기 MLP의 Test RMSE 99.304도 보존한다.

**Intermittent:** 폭12 Test RMSE 2.799 ± 0.190와 tail RMSE 11.700 ± 1.972는 폭16 평균보다 낮지만 전체 RMSE 개선은 2/3 seed다. 폭12의 전체 Time NLL 평균 22.639는 폭16 45.789보다 낮고, 폭4 15.378와 S2P2 13.674보다 높다. 폭4의 전체 RMSE 평균 2.761도 폭12보다 낮다. Validation에서 폭12의 전체·tail RMSE는 폭16보다 높은 방향이므로 split 간 방향 차이를 함께 기록한다.

**RAF:** 폭8·12의 Test RMSE 평균은 39.927·40.011로 폭16 40.248보다 세 seed 모두 낮지만, 폭4 39.795와 S2P2 39.773보다 평균이 높다. 폭8·12의 Time NLL 평균 4.067·4.097는 폭16 3.886보다 높다. 수량 이득과 시간 손해를 함께 유지한다.

## 전체·tail 평가 집합과 출처

| 데이터 | split | 전체 target | tail target | TRAIN threshold · 엄격 초과 | tail 비율 |
|---|---|---:|---:|---:|---:|
| Taxi | Validation | 8,268 | 79 | > 3449 | 0.96% |
| Taxi | Test | 8,327 | 83 | > 3449 | 1.00% |
| Intermittent | Validation | 86,285 | 1,141 | > 187 | 1.32% |
| Intermittent | Test | 88,019 | 243 | > 187 | 0.28% |
| RAF | Validation | 6,690 | 50 | > 200 | 0.75% |
| RAF | Test | 5,226 | 39 | > 200 | 0.75% |

폭8·12의 신규 평가 18조건·36 split은 원본 계약·checkpoint binary/tensor SHA·선택 epoch를 유지한 전체 Validation gate 이후 Test 결과다. 이 보고서에서 실행한 추론은 0회다. 기존 폭4·16과 비교군은 기존 공통 집계 CSV 및 개별 원본 SHA를 검증해 재사용했다.

Validation 원본에는 전체 population receipt와 frozen selected endpoint가 함께 있다. frozen endpoint는 156지표행(전체·tail 포함)이다. endpoint의 split·state·epoch·count·TRAIN threshold·수치를 검증했고, 원래 전체 Validation reference의 target/truth SHA 및 loader/data를 동일 checkpoint의 Test receipt와 동결 source 계약에 연결했다. endpoint 자체에는 개별 target/truth digest가 없으므로 모든 Validation 조건을 새 raw 예측으로 재집계한 것으로 표현하지 않는다. full-population receipt가 있는 행에서는 target/truth/loader SHA를 직접 대조했다.

원GPU 학습 endpoint와 후속 full-population Validation replay는 같은 선택 epoch·source·loader를 사용해도 runtime에 따른 미세한 수치차가 있다. 독립 검토의 전체·tail 180지표셀 중 33셀에서 차이가 관측됐지만 사전 허용오차 abs/rel 1e-5 안에 있었다. 가장 큰 예는 Taxi 폭16 seed42 tail RMSE 406.191034904 → 406.191874862(차이 약 0.000840, 상대차 약 2.07e-6)다. 이번 표는 후속 replay 원본을 유지하며, 두 보고서의 값이 byte 수준에서 동일하다고 주장하지 않는다. 선택 epoch·코드·loader·선택 규칙 변경은 없다.

원래 frozen 결과와 현재 코드 재실행을 구분했다. source revision이 registry에 직접 기록된 조건은 그대로 기재하고, 없는 조건은 source closure SHA와 원본 checkpoint 연결을 기록했다. 현재 runtime 기본값으로 결과를 다시 만들지 않았다.

Time NLL은 기록 정수 간격의 확률질량 점수이며 데이터 간 NLL을 합산해 순위를 만들지 않는다. Deep Renewal native shifted-NB는 공통 출력부 비교와 다른 연구군으로 포함했으며 원고 제외를 유지한다. 결정적 수량 기준선은 seed SD·시간 likelihood가 없고, RAF의 TitanTPP B는 비교 가능한 원본이 없어 —로 표시했다.

## 사전에 구분한 비교 범위

과거 A100 단일 seed42 탐색 arm `titantpp_history_cross_product`(cross-product), `titantpp_history_recent4_mean`(recent-four mean), `titantpp_history_recent4_attention`(recent-four attention/routing), `titantpp_history_post_block`(post-block/placement)는 이번 3seed 폭·벤치마크 비교의 입력에서 제외했다. 서로 다른 seed 범위·탐색 계약을 구분한 것이며 성능 값에 따라 불리한 조건을 제거한 것이 아니다. 과거 보고서는 보존한다. 현재 A100 CNN/GRU는 별도의 Validation-only 분석 범위이고 합치지 않았다. Instacart 폭8·12는 후속 작업 범위로 남긴다.

이번 집계는 A100 개별 Validation/Test 파일·예측을 열람하지 않았다. 모든 포함된 비교군의 불리한 결과, Time NLL, 모든 seed를 보존했다.

## 검증과 남은 작업 순서

**3개 데이터의 split별 최종 비교와 지표 검증 — 완료**

- 대상: paper_research의 이 보고서. 516지표행·123조건, 3seed ddof=1, population·threshold·선택 epoch·원본 SHA와 paired Δ를 확인했다.
- [원본·선택·계산 영수증](comparison_receipt.json), [독립 집계 검토](final_verification.json), [재생성 스크립트](build_report.py)를 보존한다.

**현재 A100 잔여 조건과 원본 회수를 기다린다 — 외부 작업 대기**

- 대상: paper_research의 별도 [A100 CNN·GRU Validation-only 보고서](../titantpp_cnn_gru_completed_review_20261005_v1/README.md). 잔여 조건 종료와 원본 binary 회수·CPU 검증은 해당 작업에서 추적한다.
- A100의 Test는 미접근 상태를 유지하고, 이 3seed Validation·Test 표와 결합하지 않는다.

**Instacart 폭8·12의 후속 연결은 보류한다 — 다음 작업 / 현재 우선 범위 밖**

- 대상: paper_research의 Instacart 후속 실험. 현재 사용자 우선 작업 이후 별도 범위에서 기존 원본·Validation 선택을 유지해 연결한다.
- 이번 세 데이터 결과에서 Instacart 폭8·12 수치를 만들거나 기존 기준선을 교체하지 않는다.

**CNN·GRU와 원고 주장 연결 — 다음 작업 / 별도 검증 이후**

- 대상: paper_research의 A100 Validation-only 구조 대조와 논문 원고. 구조 효과·용량 효과를 구분하고 모든 확인된 이득·손해를 반영한다.
- 새 학습·원격 평가·commit·push는 이 로컬 집계 작업에서 수행하지 않았다.
