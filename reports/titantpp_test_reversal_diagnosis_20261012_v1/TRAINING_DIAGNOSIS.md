# 선택·학습 궤적과 Runtime에 대한 반전 진단

범위: 이력 보정27의 원본 Train/Validation 로그, 동결 checkpoint 메타데이터, 승인된 Test36의 이력 보정27 결과. 원격 호출·추론·학습·checkpoint 재선택은 수행하지 않았다. 원본 `history.json` 27개는 각 terminal manifest SHA와 대조했다. 기존 Test 접근 이력을 유지하며 이 분석을 새로운 독립 평가로 표현하지 않는다.

## 선택과 학습 종료를 확인한다 — 완료

- 27개 모두 학습 telemetry가 finite이고, 선택 epoch은 전체 Validation raw 수량 RMSE의 최초 strict 최소값과 정확히 일치한다. 선택된 checkpoint는 마지막 epoch의 checkpoint가 아니다.
- 원래 계약은 최대300 epoch, 종료 최소40 epoch, patience40이다. 실제 27개 모두 `final_epoch = selected_epoch + 40`이다. `min_epochs=40`는 조기종료 시점 하한이며 선택 epoch의 하한이 아니다. RAF·Intermittent의 epoch5/9/11 등의 선택은 이 계약 위반이 아니다.
- 마지막 epoch까지 Train log 수량 loss는 하락했고, 선택 epoch 이후 Validation raw 수량 RMSE는 악화했다. 하지만 두 값은 서로 다른 domain·지표이므로 이 사실만으로 raw 수량의 과적합을 입증할 수 없다.

세 arm 순서는 full / 보정 제거 / current-only이다.

| 자료·seed | 선택 epoch 3개 | 실제 마지막 epoch 3개 | full Validation RMSE | full 마지막 Validation RMSE | full Test RMSE |
|---|---:|---:|---:|---:|---:|
| Taxi 42 | 126 / 141 / 82 | 166 / 181 / 122 | 78.319802 | 85.710723 | 105.939821 |
| Taxi 52 | 74 / 68 / 19 | 114 / 108 / 59 | 79.081725 | 111.319817 | 155.412595 |
| Taxi 62 | 84 / 92 / 43 | 124 / 132 / 83 | 83.825131 | 96.518963 | 107.896910 |
| Intermittent 42 | 15 / 30 / 6 | 55 / 70 / 46 | 1.666661 | 2.064845 | 2.942942 |
| Intermittent 52 | 11 / 13 / 56 | 51 / 53 / 96 | 1.676304 | 2.168949 | 2.663178 |
| Intermittent 62 | 13 / 5 / 38 | 53 / 45 / 78 | 1.744248 | 2.711740 | 2.731651 |
| RAF 42 | 9 / 9 / 9 | 49 / 49 / 49 | 34.128845 | 37.174736 | 39.655336 |
| RAF 52 | 5 / 5 / 43 | 45 / 45 / 83 | 34.401169 | 35.639395 | 40.277494 |
| RAF 62 | 20 / 26 / 22 | 60 / 66 / 62 | 33.983366 | 38.519520 | 39.793665 |

## 늦은 선택이 seed별 반전을 설명하는지 비교한다 — 완료

- Taxi full은 seed42/52/62에서 각각126/74/84 epoch을 선택했다. Test 반전 seed52가 가장 늦은 선택이 아니며, 가장 늦은 seed42는 Test에서 양대조보다 개선했다. 따라서 늦은 선택 하나로 Taxi52의 실패를 설명할 수 없다.
- Intermittent full은15/11/13 epoch을 선택했다. 반전 seed42는 약간 늦지만 모두 초기 선택이다. 세 full의 선택 전후±5 epoch Validation RMSE 중앙값은2.2692/2.2669/2.1920으로 선택값1.6667/1.6763/1.7442보다 높다. seed42만 특별히 안정적이거나 불안정했다고 결론 내릴 근거가 없다.
- RAF full은9/5/20 epoch으로 선택 시점이 다르지만 세 seed 모두 Test 수량 RMSE가 두 대조보다 나쁘다. RAF42는 세 arm 모두 epoch9에서 비교되므로 epoch 차이 없는 경우에도 악화한다. RAF52는 이미 Validation에서도 full이 양대조보다 약간 불리했다.
- Taxi full의±5 epoch Validation RMSE 중앙값은91.9682/95.5907/92.1946이다. 선택값78.3198/79.0817/83.8251의 주변에서 등락이 확인된다. 이는 선택의 민감성에 대한 설명 후보이지만, 해당 주변 checkpoint의 Test 성과는 비교하지 않았고 Test로 선택을 바꾸면 안 된다.

## 대조 조건과 평가 환경을 확인한다 — 완료

- 같은 dataset/seed의 세 arm은 원본 input SHA, Train/Validation target population, initialization map, 공통 학습 인자가 동일하다. arm 역할 식별자만 필요한 차이로 남는다.
- 초기 state의 raw hash는 세 arm에서 다르다. 대조의 identity buffer 때문에 전체 hash가 달라지는 설계이며, frozen `initial_states`는 원본의 공통 tensor와 구성 RNG를 정확히 비교한 뒤 arm별 hash를 기록한다. 원본 initialization map27개와 summary의 각 arm hash가 모두 일치했다. 이번 분석에서는 초기 tensor를 재생성하거나 재학습하지 않았다.
- 모든 arm의 optimizer는 단일 `all_parameters` group, lr0.001이며 공통 인자는 동일하다. 보정 제거의 활성 파라미터가25,308개 감소하는 것은 의도된 개입이다. full과 current-only의 활성 수는 동일하다. Taxi·Intermittent는115,167/89,859/115,167개, RAF는104,159/78,851/104,159개다. RAF는 최대 sequence84이므로 position 관련 공통 파라미터 수도 다르다.
- Test36은 하나의 동일5080 Runtime에서 추론했고, 원래 history27 전체 Validation 지표를 재현했다. 전체 RMSE/MAE/TimeNLL의 최대 상대오차는2.6501e-7, 최대 절대오차는2.3548e-5(RMSE88.8586 기준)다. 현재 Test 추론 Runtime 전환만으로 반전이 발생했다는 설명을 지지하지 않는다. 이것이 학습 Runtime 차이의 영향을 모두 제거했다는 뜻은 아니다.
- RAF42 full/current-only는 물리 A100 이관 기록이 있고 보정 제거는 원래 A100이었다. 그러나 RAF52 세 arm은 같은 복구 A100, RAF62 세 arm은 같은5080에서 학습했으며 이들 역시 악화했다. 이관만으로 RAF 전체 실패를 설명할 수 없다.
- 이 검증은 기록·선택·입력·Runtime 불일치의 근거를 찾지 못했다는 뜻이다. 모든 구현 버그의 부재를 증명하지 않는다.

## 손실과 수량 선택의 상충을 확인한다 — 완료

- 학습 objective는 recorded 시간 loss + log-domain 수량 loss이고, 선택은 raw 수량 RMSE다. 로그에서는 이 세 지표의 최소 epoch이 일치하지 않는다. 이 불일치는 계약상 설계로 확인된다.
- Taxi full의 선택 epoch TimeNLL은1.3660/1.3686/1.0818이지만 각 fit의 최저 Validation TimeNLL은0.6535/0.6559/0.6568이며 epoch2/3/2에서 발생했다. raw 수량 선택이 시간 개선을 보장하지 않았음을 Test 전 Validation만으로도 확인할 수 있다.
- Intermittent42 full은 선택 시 Validation TimeNLL0.6187로 보정 제거/current-only0.3197/0.3197보다 이미 불리했다. 다른 full seed의 선택 TimeNLL은0.2878/0.3227이다. Test의62.5721이라는 큰 시간 손해에 대해 시간 head의 취약성을 의심할 근거는 있지만, duration별 위치·scale·오류 기여가 저장되지 않아 정확한 원인을 확정할 수 없다.
- full 선택 epoch의 gradient clipping 비율은 Taxi67% 등의 대조보다 높은 경우가 있고, full 자체는 Taxi94.3–99.0%, Intermittent95.8–97.2%, RAF76.2–87.1%다. clipping은 세 seed에서 모두 많이 발생한다. seed 실패와 clipping의 인과관계, learning rate를 바꾸면 해결되는지는 현재 기록으로 증명되지 않는다.

## 구조 대조가 분리하지 못하는 원인을 기록한다 — 완료

- 보정 제거는 CNN과GRU 경로를 함께 우회하며 활성 수25,308개를 제거한다. 이 비교는 결합 경로의 효과를 보여주지만 CNN·GRU 각각의 원인을 분리하지 않는다.
- current-only는 같은25,308개 활성 보정 파라미터를 유지하되 CNN은 현재 hidden에 identity/tanh/GELU basis, GRU는 위치별 독립 zero state에서 현재 입력을 두 번 처리한다. full의 과거 convolution 및 prefix recurrence와 활성화·깊이·계산 경로가 다르다.
- current-only의 입력 hidden 자체는 encoder의 causal prefix 정보를 포함한다. 이를 모든 이력을 없앤 대조라고 해석할 수 없다.
- full은 zero-initialized residual을 학습하며 fixed-alpha1, GRU dropout0이고 별도 learned scale은 없다. correction norm, CNN·GRU별 출력을 원본 결과에서 직접 관측할 수 없다. 특정 유형에서 큰 correction이 오류를 키웠는지는 아직 가설이다.

## 기존 checkpoint의 원인 진단을 보완한다 — 다음 작업 / 평가 실행 승인 필요

- 새 fit보다 먼저 원래 동결 history27의 full·보정 제거·current-only를 같은 규칙으로 계측한다. 이미 본 Test는 탐색적 오류 진단으로만 표시하고 선택·튜닝에는 사용하지 않는다.
- 최소 필요한 기록은 target identity에 연결된 수량 오류, duration과 시간 head 위치/scale, prefix 길이, 직전·과거 수량 요약, series/time 구간별 오차 기여다. Train/Validation/Test의 수량·duration·history 분포 변화와 모델 오류를 함께 분리해야 한다.
- full9의 CNN residual norm, GRU correction norm, 공통 hidden 대비 비율을 seed·자료·stratum별로 요약하면 결합 경로의 과도한 개입 여부를 확인할 수 있다.
- 필요하면 동일한 full9 checkpoint에 CNN만 우회 / GRU만 우회하는18개 추가 추론 설정을 사용한다. 이것은 해당 저장 모델의 출력 의존성 진단이며, 각 구성요소를 제거하고 학습한 대조나 구조의 필요성 증명은 아니다. 원래 Test 최종 비교와 별도 표로 기록한다.
- 독립적인 학습 ablation이 필요한 경우에도 원인 계측을 먼저 끝낸다. 단순 재학습·seed 추가·검색 확대는 원인을 분리하는 최소 실험이 아니다. 새 학습 계획은 특정 가설을 검증할 때에만 공통 계약을 새로 고정하고 시행한다.
- 완료 조건은 어느 seed·자료·수량/duration/이력 유형이 SSE/시간 손해를 만들었는지와 correction 경로의 출력 의존성을 구분하는 것이다. 개선된 모델의 우월성이나 독립 일반화를 확보한 것으로 표시하지 않는다.

## 증적

- [기계 판독 가능한27 fit 진단](training_diagnosis.json)
- 원본 history와 summary의 절대 경로·SHA는 위 JSON의 각 `rows`에 기록했다.
- Test 원본: `search_artifacts/titantpp_history_instacart_test_20261012_v1/analysis/test_conditions.json` (계약 `f7fe042869e239a369c4ca77aa0c519bad1cde40d10688f6538e51ef803c183c`).
- 선택/종료: frozen `sources/history/paper/scripts/count_aware_tpp_backbone/training.py` 613–627, 882–893, 939–955.
- loss: frozen `sources/history/paper/scripts/count_aware_tpp_backbone/core.py` 66–90.
- 대조 및 초기화: frozen `sources/history/paper/scripts/run_titantpp_cnn_history_controls.py` 113–130, `sources/history/models/TPPs/CountAwareTitanCNNHistoryControls.py` 19–80, 176–236.
