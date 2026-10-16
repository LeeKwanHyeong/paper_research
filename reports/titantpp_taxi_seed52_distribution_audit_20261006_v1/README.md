# Taxi seed52 데이터 분포·오차 감사

**결론: seed52에만 다른 데이터나 이상치가 들어갔다는 증거는 없다.** 세 seed는 같은 Taxi 파일·분할·학습/평가 대상·정규화값을 사용했다. 확인된 차이는 같은 Test 대상에 대한 seed52의 더 큰 수량 과소예측이다. 개별 오류 예측이나 학습 원인의 확정까지 수행한 분석은 아니다.

## 대상과 현재 기준선 — 완료

- 대상은 원래 CNN+GRU54의 Taxi seed42/52/62이다. 원래 수량 Validation RMSE로 선택한 checkpoint를 유지했다.
- 입력은 `search_artifacts/titantpp_cnn_gru54_test_3seed_20261006_v1`에 보존된 동결 원본이다. 기존 Test 재평가의 집계 영수증을 읽었으며 새 학습·추론·평가를 실행하지 않았다.
- seed42는 A100, seed52/62는 5080에서 학습했다. 선택/마지막 epoch는 각각 126/166, 157/197, 54/94이다.
- 기존 Test 노출이 있는 분할이며 이번 분석은 새 독립 평가가 아니다. 시간 출력부 재적합 결과와 원래 CNN+GRU54를 혼합하지 않았다.

## 세 seed의 입력·분할·학습 노출 대조 — 완료

| 항목 | seed42 | seed52 | seed62 |
|---|---:|---:|---:|
| 전체 Train 행(정규화 통계 대상) | 38,524 | 38,524 | 38,524 |
| 실제 Train 예측 대상 | 38,393 | 38,393 | 38,393 |
| Validation 예측 대상 | 8,268 | 8,268 | 8,268 |
| Test 예측 대상 | 8,327 | 8,327 | 8,327 |
| Test 수량 >3,449 대상 | 83 | 83 | 83 |

- Parquet SHA `b47e98e9fdb75d4274a18e3f8a5d8f463418a1d56a6db4db7d9b834c9d89ca46`, split manifest SHA `4a005d4a77a89f7ca793d8de56afb9267a3ca4a5e60c53e09465c0494d60ed85`가 모두 같다. 보존 파일의 실제 SHA를 재검산했다.
- Train/Validation target identity·quantity SHA와 정규화 통계가 같다. 원본 Parquet에서 native 규약의 identity·quantity SHA를 독립 재구성하여 일치를 확인했다.
- fresh Validation/Test의 target identity·truth SHA도 split별로 모두 같다. raw 대상의 evaluator identity를 독립 재구성하여 일치를 확인했다. 학습용 identity와 평가용 identity는 해시 규약이 다르므로 서로 직접 비교하지 않았다.
- 정규화 log 평균 `2.837734592747447`, 표준편차 `2.1840596349132664`, raw scale `685.6777796861172`가 모두 같다.
- 학습은 모든 epoch에서 Train 38,393건·300 batch를 사용했다. Validation은 모든 회차 8,268건·65 batch이며 batch digest가 동일하다.
- 원본 `input_receipt.json` 전체 내용은 실행 서버의 경로 두 개를 제외하면 같다. 원본 terminal manifest의 입력·초기화·exposure·summary SHA 연결도 독립 검증됐다.

seed는 이 실행 경로에서 데이터의 무작위 분할이나 bootstrap 표본을 만들지 않는다. 고정 데이터를 준비한 다음 모델 초기화와 Train shuffle generator에 적용한다. 실제 첫 Train epoch의 batch digest와 초기 상태 SHA는 세 seed에서 다르다. digest는 tensor의 dtype·shape·전체 bytes 해시이며 target 인덱스 순서가 직접 저장된 것은 아니다.

## 공통 데이터의 수량·시간 분포 확인 — 완료

아래는 세 seed에 공통인 **실제 예측 대상**의 분포다. Train 정규화 대상 38,524행과 예측 대상 38,393행을 구분했다. entity별 첫 global 행 131개는 과거 문맥이 없어 Train target에서 제외되며 Validation/Test 첫 행을 다시 제외하지 않는다.

| 분할 | 대상 수 | 수량 평균 | 중앙값 | 95% 분위수 | 최대 | 수량 >3,449 |
|---|---:|---:|---:|---:|---:|---:|
| Train | 38,393 | 229.03 | 7 | 1,562.00 | 6,322 | 383 (1.00%) |
| Validation | 8,268 | 213.46 | 6 | 1,420.95 | 6,489 | 79 (0.96%) |
| Test | 8,327 | 229.71 | 7 | 1,576.10 | 5,952 | 83 (1.00%) |

- 55,119행에서 결측 셀·중복 entity/seq 키·음수/비유한 수량·음수 시간 간격은 0건이다. 실제 예측 대상의 비양성 시간 간격도 0건이다.
- entity별 첫 행 131개의 `delta_t=0`은 이전 관측이 없는 최초 행의 기록이다. 세 seed에 공통이며 target에서 제외된다. 전체 원본의 비양성 gap이 0이라고 표현하지 않는다. 데이터 감사 guard를 최초 행과 실제 target으로 분리해 이 사실을 보존했다.
- entity 안에서 시간 순서 역전과 Train→Validation→Test 분할 역전은 0건이다. 시간 간격과 timestamp/seq 차이도 모든 target에서 일치한다.
- seq가 한 칸보다 크게 증가한 9,799쌍은 실제 시간 간격과 정확히 일치한다. 이 수치를 누락 target이나 데이터 손상으로 집계하지 않았다.
- 시간 간격 중앙값은 세 분할 모두 1시간, 99% 분위수는 Train 15시간·Validation/Test 16시간이다. 최대는 각각 115/67/60시간이다.
- Validation에는 raw Train 최대 6,322를 넘는 6,489 한 건이 있다. 세 seed가 모두 같은 대상을 보며 이것만으로 데이터 오류라고 판정하지 않았다. Test 최대는 raw Train 최대보다 작다.
- 분할의 전역 날짜 범위는 일부 겹치지만 분할은 entity별 시간 순서다. 전역 날짜 중첩을 곧바로 동일 entity의 시간 역전이나 누출로 해석하지 않았다.

분위수는 numpy의 기본 linear 보간으로 계산했다. 원래 평가 구간은 동결된 Train 경계 7/686/1,562/3,449를 그대로 유지했다.

## 같은 정답에 대한 seed52 오차 집중 확인 — 완료

| seed | 선택 epoch | Validation 전체 RMSE | Test 전체 RMSE | Test >3,449 RMSE | Test >3,449 평균 편향 |
|---|---:|---:|---:|---:|---:|
| 42 | 126 | 78.3198 | 105.9398 | 589.0411 | −381.5777 |
| 52 | 157 | 76.8463 | 164.3154 | 1,012.3871 | −922.8348 |
| 62 | 54 | 82.0222 | 120.9454 | 890.8372 | −637.5892 |

평균 편향은 `예측−정답`이며 음수는 과소예측이다. seed52는 Validation 전체 수량 RMSE가 세 seed 중 가장 낮았지만 Test의 큰 수량에 대한 일반화가 나빴다.

seed52의 Test SSE 224,825,271.32에서 seed42/62 SSE 평균 107,630,786.53을 뺀 추가 SSE는 **117,194,484.80**이다. 이 증가분의 **96.4663%**가 수량 686 초과 821건(전체 Test의 9.86%)에 집중됐다. 1,562–3,449 구간은 증가분의 49.89%, >3,449 구간은 32.20%를 차지했다. 분모는 seed52의 전체 SSE가 아닌 비교 seed 평균 대비 증가분이다.

작은 수량 ≤7에서 seed52의 수량 RMSE는 3.1698로 seed42 3.1877과 유사하다. 따라서 모든 수량 구간에서 균일하게 악화된 결과는 아니다. 원본 영수증 기반 전체/구간 SSE·RMSE·MAE·bias·TimeNLL은 `seed_metrics.csv`에 따로 저장했다.

## 학습 원인과 미확인 범위

- 원래 세 seed의 457개 학습 epoch는 `train_all_finite=true`이며 기록된 수치도 finite이다. 선택 epoch는 각 이력의 최초 최소 full Validation 수량 RMSE epoch와 정확히 일치했다. 이 점검에서 수치적 폭발이나 selector 오류를 확인하지 못했다.
- seed52의 더 이른 epoch93은 Validation RMSE 78.7840 / TimeNLL 1.4824이고, 선택 epoch157은 76.8463 / 2.1639다. 후반부의 시간 일반화 악화는 관찰되지만 Test 수량 손해의 원인을 단독으로 설명하지 않는다. 해당 earlier epoch의 Test를 새로 평가하거나 재선택하지 않았다.
- 초기화·batch 순서·학습 길이·seed42의 GPU 차이가 남아 있다. 어느 요인이 seed52의 과소예측을 유발했는지는 이번 감사로 분리되지 않는다. 첫 몇 batch에 큰 수량이 몰렸는지도 저장된 digest만으로 판정할 수 없다.
- 구간별 집계는 개별 예측 오류의 최대값이나 특정 행의 영향도를 제공하지 않는다. 입력 분포가 같다는 결과를 모든 개별 예측이 정상이라는 결론으로 확대하지 않는다.
- seed52를 제외하거나 Test를 보고 checkpoint를 바꾸지 않는다. 세 seed 결과를 현재 기준선으로 보존한다.

## 재현과 증거

프로젝트 root에서 `/usr/local/bin/python3 reports/titantpp_taxi_seed52_distribution_audit_20261006_v1/audit.py`를 실행한다. companion `audit.ipynb`도 같은 코드를 호출한다. 이 코드는 로컬 입력/기존 집계만 읽고 이 보고서 폴더만 쓴다. torch·checkpoint 역직렬화·SSH·GPU·새 평가·학습·raw prediction 저장은 없다.

- `data_distribution.csv`: 공통 분할별 수량/시간 분포.
- `seed_inputs.csv`: seed별 입력 SHA·대상 수·초기 상태·첫 학습 batch digest.
- `seed_metrics.csv`: 원래 선택 checkpoint의 split별 전체/수량 구간 성능.
- `test_sse_decomposition.csv`: seed52 추가 제곱오차의 구간별 분해.
- `training_history.csv`: 원래 학습 이력의 주요 수치, split 의미를 유지.
- `taxi_seed_comparison.png`: 동일 Test 대상의 RMSE/bias와 원래 Validation 이력.
- `analysis_receipt.json`: 읽은 원본의 SHA와 생성물 SHA 및 실제 실행시각.
- 독립 QA는 원본 3run 입력/exposure/source와 회수된 Validation/Test 집계를 별도로 검산했다. manifest 연결·동일 모집단·수량 구간 합계·추가 SSE 계산에 일치했다.

## 남은 작업 순서

**seed52의 과소예측을 낳은 학습 차이를 분리한다 — 다음 작업**
- 같은 데이터 기준으로 초기화·학습 순서·원래 선택 시점의 영향을 구분한다. 이번 감사에서는 데이터가 같고 큰 수량 오차가 커졌다는 사실까지 확인됐다.
- 새로운 실험이 필요하면 원래 checkpoint와 세 seed 결과를 보존한 별도 계약으로 검증한다. 현재 감사가 새 학습을 실행한 것은 아니다.

**외부 비교군에 대한 논문 주장 범위를 검토한다 — 다음 작업**
- 현재 동결된 3seed Validation/Test와 변동을 기준으로 CNN+GRU54의 수량 장점과 시간/일반화 손해를 함께 판단한다. seed52를 임의로 제외해 우월성을 주장하지 않는다.
