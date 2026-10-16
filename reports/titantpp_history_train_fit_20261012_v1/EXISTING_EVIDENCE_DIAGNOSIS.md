# 저장 모델의 수량 적합 상태와 시간 분포를 분리한다

## 이미 완료한 계측을 현재 기준선으로 둔다 — 완료

- 대상은 Taxi·Intermittent·RAF의 full·보정 제거·current-only, seed42/52/62 선택 checkpoint27개다. Validation27·Test27은 2026-10-12 10:51:06 KST에 정상 종료했고 원래 대상·truth/history·여섯 지표를 재현했다. 이 문서의 기존 증거 분석은 로컬 충분통계만 읽으며 새 추론·학습·SSH·경로 개입·재선택은0이다.
- [기존 native 분석](../../search_artifacts/titantpp_history_diagnostics_20261012_v2/analysis/DIAGNOSTIC_REPORT.md), [전체 계산 JSON](../../search_artifacts/titantpp_history_diagnostics_20261012_v2/analysis/diagnostic_conditions.json), [독립 QA](../titantpp_history_diagnostics_20261012_v2/FINAL_QA.md)를 기준으로 둔다. 같은 선택 모델의 새 Train27 평가는 2026-10-12 11:47:48 KST에 실제 pipeline OS exit0으로 종료했다. 새27worker의 exit0,559개 원본의 archive/manifest·로컬 두 사본 SHA,같은 checkpoint/state·원래 eligible Train 대상과 기존 Validation/Test truth-history 조인을 검증해 총81개 split entry를 분석했다. 새 Validation/Test 추론·신규 학습은0이다.
- 현재 노출된 Test는 탐색적 진단 자료다. 사후 Test 기준으로 checkpoint·seed·전처리·calibration·모델을 선택하지 않는다. 열린 원고는 이 작업에서 변경하지 않는다.

## raw 수량의 과소예측과 log 수량 학습 목표를 구분한다 — 완료 / 원인 미확정

큰 수량 구간은 모든 split에 같은 기존 Train 경계(Taxi>3449, Intermittent>187, RAF>200)를 사용한다. 평균 오차는 예측−실제이며 실제 목표 수량으로 나눈 사후 설명 구간이다. 큰 수량 평균 오차가 음수인 사실을 전체 데이터의 calibration 또는 인과 원인으로 확대하지 않는다.

| full 모델 | seed | 전체 RMSE: Train / Val / Test | 전체 MAE: Train / Val / Test | 큰 수량 RMSE: Train / Val / Test | 큰 수량 MAE: Train / Val / Test | 큰 수량 bias: Train / Val / Test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Taxi | 42 | 72.720 / 78.320 / 105.940 | 22.683 / 25.232 / 33.696 | 391.140 / 370.080 / 589.041 | 304.184 / 276.912 / 450.037 | -195.933 / -118.688 / -381.578 |
| Taxi | 52 | 77.128 / 79.082 / 155.413 | 23.918 / 24.876 / 46.330 | 413.310 / 370.760 / 1017.487 | 324.391 / 286.125 / 904.656 | -199.263 / -109.256 / -888.407 |
| Taxi | 62 | 80.009 / 83.825 / 107.897 | 24.300 / 26.797 / 34.324 | 471.034 / 443.443 / 637.063 | 373.334 / 355.058 / 555.406 | -301.421 / -256.822 / -521.736 |
| Intermittent | 42 | 1.415 / 1.667 / 2.943 | 0.615 / 0.716 / 1.236 | 7.411 / 7.560 / 14.272 | 5.185 / 6.048 / 13.064 | -1.941 / -0.166 / +12.910 |
| Intermittent | 52 | 1.397 / 1.676 / 2.663 | 0.618 / 0.689 / 1.055 | 7.231 / 8.518 / 14.250 | 4.952 / 6.433 / 12.111 | +0.708 / -0.009 / +10.968 |
| Intermittent | 62 | 1.406 / 1.744 / 2.732 | 0.600 / 0.712 / 1.097 | 7.501 / 8.812 / 8.959 | 5.267 / 6.919 / 7.799 | -1.229 / -3.208 / +7.232 |
| RAF | 42 | 45.706 / 34.129 / 39.655 | 10.637 / 9.284 / 10.143 | 461.981 / 327.325 / 404.312 | 354.742 / 279.662 / 354.648 | -354.742 / -279.662 / -354.648 |
| RAF | 52 | 46.276 / 34.401 / 40.277 | 10.552 / 9.124 / 9.936 | 468.428 / 334.422 / 417.018 | 364.103 / 288.787 / 369.052 | -364.103 / -288.787 / -369.052 |
| RAF | 62 | 45.635 / 33.983 / 39.794 | 10.496 / 9.198 / 10.127 | 458.951 / 321.124 / 405.468 | 351.001 / 273.896 / 355.860 | -350.862 / -273.896 / -355.821 |

**확인:** Taxi·RAF의 큰 수량 과소예측은 Validation에도 있다. Intermittent는 Test 큰 수량에서 세 seed 모두 과대예측한다. 따라서 세 데이터셋을 모두 같은 “큰 수량 과소예측 문제”로 묶지 않는다. Intermittent42의 두 대조 대비 수량 손해 대부분은 2<q≤31의 비절단33,183건에서 발생했다. Taxi52는 Test tail4series의 손해가 나머지 구간 이득을 상쇄한다. RAF의 Test tail39건은 모두 이력14개 이하이고32건은 Validation에도 등장한 series다.

**같은 충분통계로 확인한 오류 구조:** tail MSE를 평균 오차²와 그 평균을 뺀 오차 분산으로 나누면, RAF full의 평균 과소예측² 비중은 Validation72.75~74.57%, Test76.94~78.32%다. Taxi52는 Validation8.68%에서 Test76.24%로 커진다(좋은 seed42의 Test41.96%, seed62는67.07%). 이는 각 구간의 관측된 집단 오차 분해이며 편향 원인이나 독립 표본의 통계적 편향 추정이 아니다. Train 결과에도 같은 분해를 적용하되 raw 차이와 모집단 변화를 함께 읽는다.

**코드로 확인한 목표:** 봉인된 `CountAwareTPP.py`의 `quantity_outputs`는 `count_only_log_regression`에서 `softplus(quantity_head(hidden))`와 `log1p(target)`의 제곱오차를 학습한다. raw 예측은 그 log1p 위치의 `expm1`이고, 선택 checkpoint 기준은 full Validation raw 수량 RMSE다. lambda_log_qty=1, lambda_tail=0을 유지한다.

**decode의 cap/scale 여부 — 확인:** frozen full과 두 대조는 같은 `SharedTimeCountModel.predict_quantity`를 상속하며 raw 예측은 `expm1(softplus(logit))`다. 예측에 finite 상한 clamp·tail cap·raw_scale의 재곱셈을 적용하지 않는다. 실수 연산의 수학적 값은 `exp(logit)`와 같고 양수이면서 위로 제한되지 않는다(실제 float32에서 두 식의 bitwise 동일성을 주장하지 않는다). `target_outputs`는 이 `point_prediction`을 그대로 반환하며 `QuantityMetrics.add`는 그 raw 배열에서 오차를 계산한다. Taxi의 raw_scale685.678, RAF54.025는 이 decode의 clipping 경계가 아니다. factory의 tail_clip_cap 기본값187·tail_normalization_scale46은 다른 tail-aux variant의 학습 손실 분기에만 등장하고, 현재 log-regression은 그 분기 전에 반환한다. 목표·입력의 `clamp_min(0)`은 nonnegative frozen 수량에서 큰 수량을 자르지 않는다. **현재 tail 악화를 “출력 cap이 수량>200/3449를 잘라서 생겼다”로 설명할 코드 근거는 없다.**

**수학적 해석:** 조건부 log1p 제곱오차의 이상적 최적 위치는 `E[log(1+Q)|입력]`이고 그 역변환은 `exp(E[log(1+Q)|입력])−1`이다. Jensen 부등식에 따라 이는 raw 제곱오차의 이상적 최적값 `E[Q|입력]`보다 크지 않다. 이를 아무 가정 없이 조건부 median이라고 부르지는 않는다. 지수 역변환은 큰 수량에서 작은 log 위치 차이를 크게 만들 수 있다. 이는 log 목표와 raw 선택의 상충 가능성을 설명하지만, 현재 학습 모델이 그 이상적 최적값에 도달했거나 이번 실패가 그 상충 때문에 발생했다는 증명은 아니다. 기존 충분통계에는 target별 log1p 오차합·제곱합이 없으므로 평균 raw 예측이나 평균 head logit에 log를 취해 log 학습 오차를 만들어내지 않는다.

**Train까지 추가해 확인한 적합 상태 — 완료**

- **RAF는 Train부터 큰 수량 point를 충분히 재현하지 못한다.** full의 Train tail205건 RMSE458.95~468.43·MAE351.00~364.10·bias−350.86~−364.10이다. 같은 대상의 직전 수량 RMSE426.56/MAE303.72, 과거 평균 RMSE412.18/MAE293.29보다 세 seed 모두 나쁘다. 평균 실제 수량433.89에 평균 예측69.78~83.03이다. 반면 전체 Train RMSE45.63~46.28은 과거 평균47.71·직전 수량57.69보다 좋다. 전체와 tail의 손해/이득 상충이 학습 대상에도 있으며, 평가 구간에서만 새로 생긴 tail 실패라고 할 수 없다. 두 보정 대조도 Train tail 손해가 크고 seed52 full의 대조 대비 RMSE 손해는 Train부터 있으므로 보정 경로만의 고유 원인으로 단정하지 않는다.
- **Taxi52의 주요 증가 구간은 Test다.** Train tailRMSE413.31은 두 대조480.38/446.47·직전 수량539.19·과거 평균1693.42보다 좋다. Train tailbias−199.26/실제 평균4167.38(약−4.8%)에서 Test−888.41/실제 평균4309.99(약−20.6%)로 커졌다. Val tail370.76에서는 제거 대조412.83보다 좋지만 current-only357.06보다 이미 나쁘다. seed62 역시 Train tail471.03은 두 대조445.43/440.84보다 나쁘므로 모든 tail 대조가 Test에서 처음 반전한 것은 아니다.
- **Intermittent42는 평가 구간에서 수량·시간 악화가 커진다.** 전체 rawRMSE1.4146→1.6667→2.9429, TimeNLL0.2193→0.6187→62.5721이다. Train에서 전체 RMSE는 두 대조보다 좋으며 full 세 seed의 전체 Train RMSE는1.397~1.415로 직전 수량1.645보다 좋다. 큰 수량 Train의 평균 오차는−1.94/+0.71/−1.23이고 Test는 모든 seed가 양의 평균 오차다. 단순히 원래부터 모든 모델의 수량 과소예측이 심했다는 설명과 다르다.
- **모집단 차이를 함께 보존한다.** RAF tail의 실제 수량은 Train433.89±293.80(N205), Val368.66±178.24(N50), Test453.79±192.19(N39)다. 따라서 RAF의 raw Train RMSE가 Test보다 크다는 것만으로 과적합을 배제하거나 하나의 일반화 원인을 확정하지 않는다. Taxi tail은4167.38±640.96(N383)→4296.25±743.92(N79)→4309.99±694.06(N83), Intermittent는262.30±55.79(N3972)→270.64±61.74(N1141)→231.75±33.81(N243)로 다르다.

Train의 true tail에 조건을 건 음의 평균 오차는 평균회귀·조건부 확률성도 포함할 수 있다. 위 관측은 학습 대상의 point 재현 부족과 평가 구간에서 커진 손해를 구분하며, 용량 부족·최적화 실패·순수한 과적합을 단독 증명하지 않는다. 학습 중 dropout을 켠 mini-batch loss나 마지막 epoch loss와 이번 선택 checkpoint의 eval Train 지표도 구분한다.

## 시간 location·scale와 정수 관측 likelihood를 별도로 본다 — 완료 / 해결 효과 미확정

| Intermittent full | seed | Train TimeNLL / 평균 σ | Validation TimeNLL / 평균 σ | Test TimeNLL / 평균 σ |
| --- | ---: | ---: | ---: | ---: |
| full | 42 | 0.2193 / 0.06781 | 0.6187 / 0.04684 | 62.5721 / 0.03959 |
| full | 52 | 0.2171 / 0.08027 | 0.2878 / 0.06634 | 16.0294 / 0.05414 |
| full | 62 | 0.2148 / 0.08921 | 0.3227 / 0.07335 | 19.5514 / 0.06492 |

- **확인:** seed42 Test 평균 σ0.03959는 제거0.10683/current-only0.09455보다 작다. Test의 목표 gap>5주4,985건(5.66%)이 제거/current-only 대비 순 TimeNLL 손해의60.20%/67.51%를 차지한다. 작은 scale과 시간 위치 불일치가 함께 나타난 취약성이다. 수량 tail의 TimeNLL은 같은 Intermittent42에서 약2.10e−48인 반면 전체 TimeNLL은62.57이다. 시간 악화와 큰 수량 오류의 대상을 같은 집단으로 가정하지 않는다.
- **정의:** σ는 log(gap/time_scale)의 분포 scale이며 주 단위 표준편차나 수량 불확실성이 아니다. location μ와 σ는 저장 가중치의 실제 shared head hidden에서 계산됐다. 구조 경로와 가중치를 바꾸거나 head를 다시 학습하지 않았다.
- **floor 여부:** Intermittent42 Test의 실제 최소 σ0.004499도 고정 floor0.001보다 크다. 현재 heteroscedastic 경로는 `σ=0.001+softplus(raw_scale)`이며 legacy 시간 intercept/지수 cap 경로가 아니다. 좁은 scale가 확인된다고 해서 기존 floor에 hard clipping돼 발생한 손해나 floor를 올리면 해결되는 효과를 주장하지 않는다.
- **관측 계약:** 실제 점수는 `D=max(1,round(T))`, `T=time_scale·exp(N(μ,σ))`의 정수 확률질량이다. 코드1은 `P(T≤1.5)`, 일반 정수d는 `P(d−0.5<T≤d+0.5)`를 사용한다. 현재 세 dataset·checkpoint metadata27개와 양 canonical 계약을 대조한 `top_code`는 모두 `None`이므로 현재 점수에 최상위 top-code cell은 없다. 원문 helper는 top-code K가 지정된 계약에 한해 `D=min(K,max(1,round(T)))`이며 최상위 code K에서 `P(T>K−0.5)`의 survival 질량을 사용한다. 후속2×2 score도 현재 계약의 round/clamp·top_code=None을 그대로 보존한다. 연속 밀도로 바꾸거나 σ의 평균을 모든 예측에 대입해 NLL을 재구성하지 않는다.
- **한계:** 저장된 standardized log residual의 평균·분산은 위치와 scale가 결합된 기술 통계다. 기록된 정수d는 잠재 연속T가 아니므로 |z|>1.96 비율을 정규분포 coverage로 해석하지 않는다. 현재 집계만으로 “σ만 바꾸면 해결” 또는 “location만 틀림”을 분리하는 효과를 계산할 수 없다. 두 출력을 실제 개입한 paired likelihood가 필요하며 이번에 수행하지 않는다.

## 유력한 가설을 하나씩 검증할 최소 계약을 정한다 — 다음 작업 / 추가 실행은 승인 필요

1. **먼저 Train 적합 상태를 읽는다 — 완료.** 새 Train27과 기존 Validation/Test54의81개 split entry를 같은 dataset·arm·seed·선택 epoch·checkpoint/state SHA·source closure로 join했다. 실제27exit0·pipeline OS exit0·559원본 manifest/archive·두 로컬 사본 SHA·원래 eligible Train 대상과 raw 여섯 지표 계산을 검증했다. [전체 적합 상태 보고서](../../search_artifacts/titantpp_history_train_fit_20261012_v1/analysis/TRAIN_FIT_REPORT.md)를 기준선으로 둔다.
2. **H1: 수량 목표 상충의 증거를 저장 모델에서 확인한다 — 다음 작업 / 추가 실행 승인 필요.** Train 결과에서 RAF tail 손해가 이미 나타나면 RAF의 full·보정 제거·current-only×3seed 선택 checkpoint9개를 Train·Validation에 한 번씩 평가하는 **새18split pass·새 학습0·새 Test 추론0**을 후보로 둔다. 기존 raw 수량 여섯 지표·bias·목표/history digest를 재현하면서 frozen `log_qty_loss`의 실제 log1p MSE와 signed log 오차의 충분통계를 전체/원래 수량 bin/tail에 추가한다. 같은 대상에서 raw 손해와 log 손해의 방향이 다른지 읽되, 평균 raw 예측/logit로 log 오차를 만들지 않는다. 모델·예측·loss·selector를 바꾸지 않는다. 이는 log/raw 상충의 관측이며 loss를 바꾸면 개선된다는 causal 증명이나 gradient 분석이 아니다. Taxi 확대·loss/LR/tail weight 탐색·새 학습은 자동 실행하지 않는다.
3. **H2: 시간 scale와 location의 점수 의존성을 분리한다 — 다음 작업 / 추가 실행 승인 필요.** Intermittent의 같은9개 checkpoint×Train·Validation의 **새18split pass·새 학습0·새 Test 추론0**을 후보로 둔다. 같은 event에서 full과 각 대조의 실제 μ·σ를 정확한 target/history digest로 맞춘 뒤 `(μfull,σfull)`, `(μfull,σcontrol)`, `(μcontrol,σfull)`, `(μcontrol,σcontrol)`의2×2 정수 확률질량을 충분통계로만 기록한다. 두 원래 셀의 native NLL·수량 지표·state 불변을 먼저 재현한다. 6pair×2split×4셀=48 paired score cells이며 full 원래 셀의 중복을 포함한다. 그중 새 교차 score cells는24개다. 임의 σ floor/temperature·Test fit·grid search·head 재학습은0이다. 다른 학습모델의 관측 출력을 교차한 **counterfactual score**이므로 구조/head 인과 증명·배포 가능한 새 단일모델·완료된 calibration으로 부르지 않는다. Train/Validation에서 scale 교체와 location 교체의 손해/이득을 분리하지만, 기존 Test 시간 반전이 해결됐다는 증거는 아니다. 평균 μ·σ에서 이 교차 점수를 재구성할 수 없어 추가 추론이 필요하며 이번에는 실행하지 않는다.
4. **원고 주장에 필요한 경우만 학습 실험을 설계한다 — 조건부 / 승인 필요.** 관측만으로 causal 구성요소 필요성을 주장하지 않는다. head 또는 objective를 바꿔야 한다면 단일 가설·공통 Runtime·새 초기화와 동일 계약의 최소 paired fit을 별도로 설계한다. 현재 Test로 방향을 정했으면 후속 개발 증거로 표기하고, 최종 모델 동결 뒤 미노출 확인을 정한다. 신규 검색·자동 신규 학습은0이다.

## 근거 파일과 불변 조건을 고정한다 — 완료

- 새 Train canonical contract: `76248e2174d7bd651d821f649ad860030da985d59da88a8ce50725b476f55ccf`, operation rawSHA `157791af8ce197cfc858659e195eda314b2d1de7ca139971d627e51d6b91819b`
- Train archive rawSHA `26cd5a94ac02efcebdb8eb063e4397acc9c426cdd1289f35446dbb5f37523ed7`, manifest rawSHA `705e863e6cbcbdab71a6f6ec105d747318ee7ce3146ddc55d6070e59869e3600`.
- 새 [81개 split 계산 JSON](../../search_artifacts/titantpp_history_train_fit_20261012_v1/analysis/train_fit_conditions.json) rawSHA `afd2bfd8a8fefb4a3bc8b3c20b8cb54537e4a7ccf0de5495c8d427c4084cfa75`, [전체 보고서](../../search_artifacts/titantpp_history_train_fit_20261012_v1/analysis/TRAIN_FIT_REPORT.md) rawSHA `c102dad941eb3088fa5849b3420414e29210c27d7bf16ac1fd4b534e1a728b24`.
- 기존 v2 canonical contract: `70e8e22dbe85aa93cb5dc555b06303511b0e5e7f511771b455b4b243ddb3ad1d`.
- 기존 분석 JSON rawSHA: `8590a30c122818c94c4f11f196b63095ef6bcb42e9fedf4fd8ac1bee33462b6f`.
- 기존 completion receipt rawSHA: `f689243b21dcd556af81b33117f994191bc6b2d0a139634153f187172560f5ea`.
- 과학 source129 closure: `c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a`, revision `57da14a2a7a5687e1bdef7d3164f3e8c0051da59`.
- [봉인된 수량·시간 head 원문](../../search_artifacts/titantpp_history_diagnostics_20261012_v2/sources/history/models/TPPs/CountAwareTPP.py), [실제 최종 target 출력](../../search_artifacts/titantpp_history_diagnostics_20261012_v2/sources/history/paper/scripts/count_aware_tpp_backbone/core.py), [raw 오차 집계](../../search_artifacts/titantpp_history_diagnostics_20261012_v2/sources/history/paper/scripts/evaluate_titantpp_history_width.py), [factory의 대체 tail variant 기본값](../../search_artifacts/titantpp_history_diagnostics_20261012_v2/sources/history/models/TPPs/CountAwareFactory.py), [봉인된 정수 확률질량](../../search_artifacts/titantpp_history_diagnostics_20261012_v2/sources/history/models/TPPs/positive_integer_time.py), [첫 진단의 예외와 실패 보존 기록](../titantpp_history_diagnostics_20261012_v2/FOCUSED_INTERPRETATION.md).

좋은 seed·모든 대조·불리한 시간/큰 수량 결과를 유지한다. 대상 event가 서로 독립이라고 가정하거나3seed를 통계 유의성·채택 확률로 환산하지 않는다. 이 문서는 실제 관측, 수학적 후보, 추가로 필요한 검증을 구분한다.
