# Q3 Cap·원본 MLP·외부 비교군의 저장 결과 비교

상태: **저장 결과 비교 완료 / 새 학습·추론·서버 작업0 / 기존 CNN·checkpoint·원본 불변**.

**비교 대상과 한계 — 확인 완료**

Cap과 원본은 MLP4·MLP16·AllAvailable·CurrentOnly의 동일한 네 구조로 범위를 맞췄다. 아래 “원본 최저”는 이 네 구조 안의 값이며 과거 모든 CNN·GRU·폭8/12·후속 후보를 포함한 전역 최저가 아니다. 외부 비교군은 기존 matched-head6개 모델이다. 각 결과는 seed42·52·62의 평균±표본SD(ddof1)다. RMSE·MAE 모두 Validation 수량RMSE로 선택한 같은 checkpoint의 값을 사용한다. 모델은 재평가하거나 재선택하지 않았다. 이번에는 저장된 결과의 모델 간 최소값을 따로 표시한다.

Cap의 Train Q3 상한은 Taxi65·RAF10·Intermittent20이다. 모든 split의 관측 이력과 정답에 같은 상한을 적용했고 log1p는 유지했다. 예측 출력을 사후 clipping한 결과가 아니다. **Cap과 원본은 입력과 정답이 다르므로 점수 차이·개선율·우월성은 계산하지 않는다.** 외부 모델의 같은 Cap 학습·평가 결과는 확인한 고정 등록부와 보고 범위에 없어 N/A다. 원본 외부 결과를 Cap 기준선으로 대체하지 않는다.

**베스트의 의미 — 고정**

첫 표는 각 조건에서 Validation 평균RMSE가 가장 낮은 한 구조를 정해 그 구조의 Validation/Test 두 지표를 함께 보여준다. 이는 이미 알려진 결과의 기술적 선택이며 새로운 독립 선택·검증이 아니다. 그 뒤 지표별 최저는 고정된 비교 모델들 중 해당 split 평균이 가장 낮은 값이다. 특히 Test 최저는 **사후 최저**이므로 검증된 단일 채택 모델로 부르지 않는다. RMSE 최저 모델의 값과 다른 MAE 최저 모델의 값을 하나의 모델 결과로 합치지 않는다.

**Validation 평균RMSE 기준으로 한 구조를 고른 결과 — 완료**

| 데이터 | 조건 | Validation으로 고른 구조 | Validation RMSE / MAE | 같은 모델 Test RMSE / MAE |
| --- | --- | --- | --- | --- |
| Taxi | Cap MLP | MLP4 | 6.043985 / 3.090209 | 6.165356 / 3.437163 |
| Taxi | 원본 MLP | MLP16 | 77.825957 / 25.177656 | 113.392687 / 36.039467 |
| Taxi | 원본 외부 | RMTPP | 94.728347 / 28.512359 | 246.647279 / 65.276984 |
| RAF | Cap MLP | MLP4 | 2.606551 / 1.863357 | 2.702453 / 1.915335 |
| RAF | 원본 MLP | MLP4 | 33.950679 / 9.250602 | 39.795096 / 10.056276 |
| RAF | 원본 외부 | S2P2 | 33.994704 / 9.272988 | 39.772935 / 10.095805 |
| Intermittent | Cap MLP | AllAvailable | 0.414230 / 0.261810 | 1.485032 / 0.633899 |
| Intermittent | 원본 MLP | MLP4 | 1.673034 / 0.700523 | 2.761218 / 1.133524 |
| Intermittent | 원본 외부 | RMTPP | 2.549928 / 0.894885 | 4.211471 / 1.692320 |

위 표의 원본 외부 행은 Validation RMSE로 고른 모델이다. Test에서 가장 낮은 외부 모델은 아래와 다를 수 있다. 따라서 원본 모델의 경쟁력을 해석할 때 아래의 지표별 외부 최저도 함께 확인한다.

**Cap의 지표별 최저 — 완료 / 외부 Cap 비교 N/A**

| 데이터 | split | 최저 RMSE 모델·값 | 최저 MAE 모델·값 | 동일 Cap 외부 |
| --- | --- | --- | --- | --- |
| Taxi | validation | MLP4: 6.043985 ± 0.064436 | CurrentOnly: 3.026535 ± 0.077972 | N/A |
| Taxi | test | MLP16: 6.107686 ± 0.313573 | MLP4: 3.437163 ± 0.141957 | N/A |
| RAF | validation | MLP4: 2.606551 ± 0.011188 | MLP16: 1.835865 ± 0.046722 | N/A |
| RAF | test | MLP16: 2.693792 ± 0.013963 | MLP16: 1.887946 ± 0.036415 | N/A |
| Intermittent | validation | AllAvailable: 0.414230 ± 0.004588 | AllAvailable: 0.261810 ± 0.005689 | N/A |
| Intermittent | test | CurrentOnly: 1.454467 ± 0.156889 | AllAvailable: 0.633899 ± 0.019446 | N/A |

**원본 MLP 최저와 원본 외부 최저 — 동일 목표에서 비교 완료**

후보−외부이며 음수는 후보의 점수가 낮다는 뜻이다. 각 metric별로 모델 이름을 보존한다. 특히 Test 행은 사후 비교이며 통계적 유의성이나 새 독립 일반화 증거가 아니다.

RMSE

| 데이터 | split | 원본4MLP 내 최저 | 원본 외부6 내 최저 | 평균 차이 (상대%) | 낮은 seed 수 |
| --- | --- | --- | --- | --- | --- |
| Taxi | validation | MLP16: 77.825957 ± 4.160967 | RMTPP: 94.728347 ± 3.400600 | -16.902390 (-17.843%) | 3/3 |
| Taxi | test | AllAvailable: 99.304085 ± 12.846499 | S2P2: 115.541513 ± 6.402269 | -16.237428 (-14.053%) | 3/3 |
| RAF | validation | MLP4: 33.950679 ± 0.058804 | S2P2: 33.994704 ± 0.126119 | -0.044025 (-0.130%) | 2/3 |
| RAF | test | CurrentOnly: 39.464394 ± 0.224821 | S2P2: 39.772935 ± 0.160914 | -0.308542 (-0.776%) | 2/3 |
| Intermittent | validation | MLP4: 1.673034 ± 0.081887 | RMTPP: 2.549928 ± 0.160402 | -0.876894 (-34.389%) | 3/3 |
| Intermittent | test | AllAvailable: 2.660208 ± 0.097299 | S2P2: 3.215355 ± 0.274250 | -0.555147 (-17.265%) | 3/3 |

MAE

| 데이터 | split | 원본4MLP 내 최저 | 원본 외부6 내 최저 | 평균 차이 (상대%) | 낮은 seed 수 |
| --- | --- | --- | --- | --- | --- |
| Taxi | validation | MLP16: 25.177656 ± 1.412918 | RMTPP: 28.512359 ± 0.634433 | -3.334703 (-11.696%) | 3/3 |
| Taxi | test | AllAvailable: 32.124680 ± 3.415025 | S2P2: 36.835852 ± 2.381474 | -4.711172 (-12.790%) | 3/3 |
| RAF | validation | MLP4: 9.250602 ± 0.085484 | RMTPP: 9.136205 ± 0.055678 | +0.114397 (+1.252%) | 1/3 |
| RAF | test | CurrentOnly: 10.015831 ± 0.118046 | RMTPP: 9.906347 ± 0.037018 | +0.109484 (+1.105%) | 0/3 |
| Intermittent | validation | AllAvailable: 0.697524 ± 0.031214 | RMTPP: 0.894885 ± 0.049405 | -0.197361 (-22.054%) | 3/3 |
| Intermittent | test | MLP4: 1.133524 ± 0.019477 | THP: 1.312281 ± 0.027045 | -0.178757 (-13.622%) | 3/3 |

**다른 출력부·단순 참조의 결과 — 별도 보존**

주 비교의 외부6종은 공통 출력부를 맞춘 집합이다. 부모 결과에 있는 Deep Renewal native NB는 다른 출력부의 연구 기록이며 기존 원고 제외 상태를 유지한다. 마지막값·이력평균은 결정적 수량 기준선으로 학습 seed의 SD와 TimeNLL이N/A다. 이를 외부6종의 평균·SD 집합에 섞지 않는다. 다만 불리한 참조를 숨기지 않도록 아래에 그대로 보존한다.

| 데이터 | split | 보조 참조 | RMSE | MAE | 집계 |
| --- | --- | --- | --- | --- | --- |
| Taxi | validation | Deep Renewal native NB | 404.786354 | 91.474695 | 3seed |
| Taxi | validation | 마지막 관측 수량 | 134.393100 | 41.195090 | 결정적1회 |
| Taxi | validation | 관측 이력 평균 | 368.784314 | 120.377075 | 결정적1회 |
| Taxi | test | Deep Renewal native NB | 435.999175 | 105.833015 | 3seed |
| Taxi | test | 마지막 관측 수량 | 137.582195 | 42.611024 | 결정적1회 |
| Taxi | test | 관측 이력 평균 | 364.652067 | 120.968709 | 결정적1회 |
| RAF | validation | Deep Renewal native NB | 34.653847 | 10.261750 | 3seed |
| RAF | validation | 마지막 관측 수량 | 44.574172 | 11.313602 | 결정적1회 |
| RAF | validation | 관측 이력 평균 | 36.849906 | 10.856359 | 결정적1회 |
| RAF | test | Deep Renewal native NB | 40.799218 | 11.187499 | 3seed |
| RAF | test | 마지막 관측 수량 | 45.855031 | 12.203597 | 결정적1회 |
| RAF | test | 관측 이력 평균 | 39.357906 | 11.301169 | 결정적1회 |
| Intermittent | validation | Deep Renewal native NB | 2.505718 | 0.582465 | 3seed |
| Intermittent | validation | 마지막 관측 수량 | 1.707803 | 0.564733 | 결정적1회 |
| Intermittent | validation | 관측 이력 평균 | 13.782119 | 4.553695 | 결정적1회 |
| Intermittent | test | Deep Renewal native NB | 3.811988 | 1.682679 | 3seed |
| Intermittent | test | 마지막 관측 수량 | 3.206282 | 1.263579 | 결정적1회 |
| Intermittent | test | 관측 이력 평균 | 18.856628 | 5.751676 | 결정적1회 |

중요한 예외: Intermittent Validation에서 Deep Renewal MAE0.582465와 마지막값 MAE0.564733은 네 MLP 구조의 최저 MAE0.697524보다 낮다. RAF Test의 이력평균 RMSE39.357906도 네 MLP 구조의 최저39.464394보다 낮지만 MAE11.301169는 더 높다. 따라서 외부6종에 대한 일부 수량 우위를 모든 기준선·모든 지표의 우위로 확대하지 않는다.

**결과 해석 — 완료**

Cap에서 수량 오류 숫자가 작아진 것은 예측 대상의 상한이 바뀐 결과를 포함하므로 원본 문제의 해결로 해석하지 않는다. Cap에서도 세 데이터셋·Validation/Test·RMSE/MAE를 모두 개선한 공통 구조는 확인되지 않았다. RAF 폭16의 Test 개선과 Intermittent AllAvailable의 평균 이득은 보존하되 seed 예외와 Validation의 다른 순위를 생략하지 않는다. 원본 외부 대비 개선 여부는 위의 raw 조건 안에서만 판단한다. 외부 비교군의 Cap 결과가 없으므로 Cap 모델의 외부 우월성·열등성은 판정하지 않는다.

**표본·출처·계산 검증 — 완료**

Cap36개의 Validation/Test checkpoint epoch·binary/state 동일성과24그룹144개 평균/SD를 확인했다. 원본 MLP는기존3dataset 비교표와 원본checkpoint 결속을 재사용하고 평균/SD를 재계산했다. 외부는 같은 원본 비교표 및 고정 등록부로 연결했다. 세 자료는 같은 dataset/split의 대상 개수가 같지만 동일 개수만으로 같은 truth를 증명하지 않는다. Cap의quantity digest는 별도이며 각 부모 receipt의 target/loader/input lineage와 누락된 digest 범위를 RAW_MLP_RECEIPT.json과 EXTERNAL_RECEIPT.json에 남겼다.

원본 표는2026-10-05 고정 비교 자료를 일관되게 사용했다. 일부 후속 CPU/native Validation 재현과 원래 GPU endpoint에는 작은 수치 차이가 있어 이전2026-10-18 endpoint 분석과 자릿수 끝이 다를 수 있다. 서로 다른 관측 원본을 임의로 교체하거나 기존 보고서를 수정하지 않았다. 과거 환경·서버·평가 관측시점 차이를 완전히 통제한 새 실험이 아니며, Cap Intermittent42는5080/cuDNN91900, 52·62는5090/cuDNN92000이다. 같은 dataset·seed의 Cap 네 모델은 같은 서버에서 비교했다.

모든 평균은 세 개별 seed의 산술평균이다. 사건을 seed 간 합쳐RMSE를 계산하거나 ensemble한 값이 아니다. 서로 다른 데이터셋의 raw 오류는 평균내지 않는다. p값·유의성 검정은0이며, 방법 오류11종을 검토해 선택편향·다중탐색·집단구성·범위외일반화를 한계로 남겼다. 기존 Test를 참고한 탐색이며 한 구조의 평균 이득을 모든 seed·데이터셋의 우월성으로 확대하지 않는다.

**남은 작업 — 완료 및 미확인 구분**

이번 요청의 저장 결과 비교는 완료했다. 기존 CNN·source·checkpoint·결과는 변경하지 않았다. 같은 Cap 조건의 외부 비교군 결과는 미확인이므로 해당 비교는N/A로 남긴다. 새 학습·재추론·서버 실행·자동화 변경은 수행하지 않았다.
