# Shared15 논문 근거 원장

작성일: 2026-10-20 · 상태: **기존 Test 결과의 기술적 정리 완료 / 원고 작성은 외부 작업 대기**

이 문서는 원고 초안이 아니다. 기존 RAF·Taxi Test6와 Intermittent 임시 Test3의 결과 JSON·receipt, Shared15와 직접 연결되는 모델 코드를 읽어 논문에 사용할 수 있는 사실과 한계를 정리했다. 이번 추가 문서 작업에서는 SSH·학습·추론을 실행하지 않았고, 신규 extra42 학습 파일과 기존 원고 파일을 읽거나 수정하지 않았다.

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate — 기존 산출물의 수치 검산과 기술적 해석만 수행
- Origin Date: 2026-10-20
- Verification Status: **ANALYZED** — 이번 작업에서 학습·추론 재실행을 통한 재현 검증은 하지 않음
- Version Label: paper_evidence_ledger_v1
- 연구 기준선: original_uncapped, seed 42·52·62. v0.7 validation freeze 및 새 seed 72·82 결과와 혼합하지 않음

## 1. 근거 파일과 평가 범위

| 근거 ID | 원본 파일 | SHA-256 | 이 문서에서 사용하는 범위 |
|---|---|---|---|
| E1 | [reports/titantpp_shared_raf_taxi6_test_20261020_v1/THREE_SEED_TEST.json](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_shared_raf_taxi6_test_20261020_v1/THREE_SEED_TEST.json) | `1ee5d2ee0322d1ca89a0b1495a7052b5db6bcfcb2e902c9e4a689c8612320b1e` | RAF·Taxi Test 각각 Shared15 3개, 저장된 외부6종 및 MLP16 참조 |
| E2 | [reports/titantpp_shared_inter3_interim_test_20261020_v1/THREE_SEED_TEST.json](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_shared_inter3_interim_test_20261020_v1/THREE_SEED_TEST.json) | `018fc63d83b7c848a6b03cab55b1b15fb1e99ac7e4deed58a0f7411e9b9521e8` | Intermittent 임시 best5/5/11 Test3와 저장된 비교군 |
| M1 | [models/TPPs/CountAwareTitanSharedHistoryMLP.py](/Users/igwanhyeong/PycharmProjects/paper_research/models/TPPs/CountAwareTitanSharedHistoryMLP.py) | `eee18d3abe199f0c71d80da35acf01cf19acd8f56291ff7cfd33fd86d8bc82a2` | 공유 관계 기저·분기·마스크·초기화·checkpoint 정체성 |
| M2 | [models/TPPs/CountAwareTitanHistoryWidth.py](/Users/igwanhyeong/PycharmProjects/paper_research/models/TPPs/CountAwareTitanHistoryWidth.py) | `2a903290810070bdfe72a355a98ff09790b02858a6853e94a7a4a87336afc6f7` | 내부 ablation MLP16의 8개 독립 width16 보정과 파라미터 계산 |

평가 완료 범위는 각 `search_artifacts/<operation>/final_receipt.json` 및 `retrieval/receipt.json`에 결속된다. E1은 qualification6·Validation6·Test6의 native worker 18개와 pipeline 1개 exit0를 기록한다. E2는 qualification3·Validation3·Test3의 native worker 9개와 pipeline 1개 exit0를 기록한다. 이는 두 평가 작업의 완료이며, 원래 5090 raw9 학습 전체의 완료를 뜻하지 않는다.

E1의 `baseline_reference_count=42`는 두 데이터셋 × (외부6종+MLP16) × 3seed다. E2 Test JSON의 참조는 21개이며, E2 최종 receipt의 42개는 Validation21+Test21을 합친 수다. 중복 계산해 독립 모델·실험 수로 늘리지 않는다.

| 데이터셋 | Test 관측 수 / seed | 큰수량 관측 수 / seed | Train 고정 큰수량 규칙 | Shared15 선택 epoch · seed42/52/62 | 결과의 지위 |
|---|---:|---:|---|---|---|
| RAF | 5,226 | 39 | q > 200 | 9/5/19 | 완료된 해당 학습의 Validation 선택 checkpoint |
| Taxi | 8,327 | 83 | q > 3449 | 112/148/127 | 완료된 해당 학습의 Validation 선택 checkpoint |
| Intermittent · 임시 고정본 | 88,019 | 243 | q > 187 | 5/5/11 | 캡처 시점 임시 best, 최종 선택 미확인 |

큰수량 규칙은 엄격한 `>`이며 수량을 상한으로 자른 자료가 아니다. 서로 단위·규모가 다른 데이터셋의 raw RMSE·MAE를 하나의 평균으로 합치지 않는다. 3seed가 같은 Test 관측을 반복 사용하므로 관측 수에 3을 곱해 독립 표본 수로 해석하지 않는다.

## 2. 외부 비교를 우선한 수치 근거

모든 값은 **Test**, original_uncapped, seed 42·52·62의 산술평균 ± 표본 표준편차(ddof=1)다. 여섯 지표 모두 낮을수록 좋다. 표준편차는 seed 간 변동이며 신뢰구간·유의성 검정이 아니다. 외부 모델 이름은 이 프로젝트에서 공통 head에 연결한 구현을 뜻한다.

### RAF — E1

| 모델 | 전체 RMSE | 전체 MAE | 전체 TimeNLL | 큰수량 RMSE | 큰수량 MAE | 큰수량 TimeNLL |
|---|---:|---:|---:|---:|---:|---:|
| Shared15 | 39.775389 ± 0.186122 | 10.150475 ± 0.098714 | 3.815814 ± 0.163216 | 405.165825 ± 3.709314 | 356.589070 ± 2.733991 | 4.014403 ± 0.341314 |
| RMTPP · 공통 head | 40.463166 ± 0.241581 | 9.906347 ± 0.037018 | 3.899754 ± 0.354955 | 420.601915 ± 3.979029 | 376.801310 ± 4.439999 | 3.999902 ± 0.264186 |
| THP · 공통 head | 40.555091 ± 0.295255 | 10.122950 ± 0.230475 | 3.456260 ± 0.024639 | 420.364414 ± 4.601256 | 376.021398 ± 4.724764 | 3.596128 ± 0.052094 |
| NHP · 공통 head | 42.176330 ± 0.300481 | 9.969145 ± 0.093177 | 3.590199 ± 0.138511 | 438.796256 ± 2.347396 | 396.578091 ± 2.194620 | 3.771375 ± 0.232514 |
| SAHP · 공통 head | 41.049344 ± 0.156112 | 10.261643 ± 0.103236 | 4.232558 ± 0.155261 | 425.478719 ± 3.412326 | 382.054038 ± 4.221969 | 3.822507 ± 0.123359 |
| S2P2 · 공통 head | 39.772935 ± 0.160914 | 10.095805 ± 0.145493 | 3.741535 ± 0.088344 | 406.404871 ± 6.277029 | 360.721446 ± 6.575346 | 3.871412 ± 0.065285 |
| AttNHP · 공통 head | 40.203624 ± 0.204402 | 9.983934 ± 0.028442 | 4.950772 ± 0.165720 | 414.764778 ± 1.847932 | 371.264445 ± 1.671447 | 5.023036 ± 0.264926 |

### Taxi — E1

| 모델 | 전체 RMSE | 전체 MAE | 전체 TimeNLL | 큰수량 RMSE | 큰수량 MAE | 큰수량 TimeNLL |
|---|---:|---:|---:|---:|---:|---:|
| Shared15 | 107.084863 ± 11.559813 | 34.323697 ± 3.020096 | 1.148959 ± 0.051299 | 546.872240 ± 81.248362 | 431.967496 ± 78.318050 | 6.00356e-27 ± 1.03985e-26 |
| RMTPP · 공통 head | 246.647279 ± 82.506426 | 65.276984 ± 16.103485 | 0.915580 ± 0.160426 | 1829.231403 ± 727.800728 | 1741.897869 ± 695.784650 | 0.000013 ± 0.000023 |
| THP · 공통 head | 202.174002 ± 36.218246 | 58.348636 ± 11.075891 | 0.689140 ± 0.009083 | 1264.894382 ± 458.384748 | 1148.160352 ± 527.148843 | 1.85373e-12 ± 3.20908e-12 |
| NHP · 공통 head | 381.076245 ± 26.234870 | 116.332384 ± 9.712797 | 0.683595 ± 0.001830 | 2534.174466 ± 122.974775 | 2474.317548 ± 128.748094 | 1.54768e-10 ± 2.37402e-10 |
| SAHP · 공통 head | 148.249483 ± 30.215681 | 40.900788 ± 4.923976 | 0.706186 ± 0.018891 | 996.720752 ± 355.944101 | 854.055000 ± 385.650763 | 6.07701e-06 ± 0.000011 |
| S2P2 · 공통 head | 115.541513 ± 6.402269 | 36.835852 ± 2.381474 | 0.690022 ± 0.011527 | 587.448847 ± 118.104835 | 486.680913 ± 117.068551 | 7.14106e-16 ± 1.23687e-15 |
| AttNHP · 공통 head | 131.319808 ± 2.427940 | 38.752772 ± 1.056452 | 0.702218 ± 0.008485 | 703.987470 ± 79.383554 | 565.601770 ± 71.977219 | 8.83089e-13 ± 1.4208e-12 |

### Intermittent · 임시 고정본 — E2

| 모델 | 전체 RMSE | 전체 MAE | 전체 TimeNLL | 큰수량 RMSE | 큰수량 MAE | 큰수량 TimeNLL |
|---|---:|---:|---:|---:|---:|---:|
| Shared15 | 2.966841 ± 0.248890 | 1.197329 ± 0.036735 | 22.553775 ± 15.114807 | 15.713591 ± 5.266404 | 14.632456 ± 5.632029 | 2.363478 ± 4.093078 |
| RMTPP · 공통 head | 4.211471 ± 0.212624 | 1.692320 ± 0.180771 | 24.132675 ± 11.994692 | 31.159555 ± 6.022845 | 28.682735 ± 6.235932 | 6.85037e-06 ± 5.19578e-06 |
| THP · 공통 head | 3.483944 ± 0.123200 | 1.312281 ± 0.027045 | 8.326340 ± 1.229744 | 30.543947 ± 4.719733 | 28.522792 ± 5.558141 | 4.17225e-06 ± 5.21816e-06 |
| NHP · 공통 head | 5.906457 ± 0.850743 | 2.327751 ± 0.159345 | 18.582785 ± 15.763645 | 61.365661 ± 14.409557 | 60.115823 ± 14.814254 | 8.03253e-10 ± 1.32043e-09 |
| SAHP · 공통 head | 12.252441 ± 1.518177 | 3.676981 ± 0.689162 | 14.911412 ± 2.725862 | 73.731660 ± 3.121596 | 68.023244 ± 3.203538 | 3.08096e-06 ± 4.97453e-06 |
| S2P2 · 공통 head | 3.215355 ± 0.274250 | 1.319690 ± 0.026354 | 13.673777 ± 4.592186 | 21.134082 ± 6.122699 | 19.016264 ± 6.993992 | 0.009035 ± 0.015649 |
| AttNHP · 공통 head | 12.289099 ± 2.689065 | 4.349284 ± 0.701125 | 19.486687 ± 2.046830 | 63.112194 ± 23.690402 | 58.926697 ± 20.855670 | 0.050603 ± 0.087643 |

### 외부6종의 지표별 최저 평균과의 차이

아래 상대 차이는 `(Shared15 평균 − 해당 지표의 외부 최저 평균) / 외부 최저 평균 × 100`이다. 음수는 Shared15의 오차가 작다는 뜻이다. 지표별 최선 모델이 다를 수 있으므로 이들을 한 모델의 성능으로 합치지 않는다. 외부 최저 모델을 Test 평균에서 골랐다는 선택 절차도 공개한다.

| 데이터셋 | 지표 | 외부 최저 평균 · 모델 | Shared15 평균 | 절대 차이 | 상대 차이(%) | 확인 가능한 해석 |
|---|---|---|---:|---:|---:|---|
| RAF | qty_rmse | 39.772935 · s2p2_matched_head | 39.775389 | +0.002454 | +0.0062 | 외부 최저 평균보다 오차가 높음 |
| RAF | qty_mae | 9.906347 · rmtpp | 10.150475 | +0.244129 | +2.4644 | 외부 최저 평균보다 오차가 높음 |
| Taxi | qty_rmse | 115.541513 · s2p2_matched_head | 107.084863 | -8.456650 | -7.3191 | 3seed 평균 오차가 낮음 |
| Taxi | qty_mae | 36.835852 · s2p2_matched_head | 34.323697 | -2.512155 | -6.8199 | 3seed 평균 오차가 낮음 |
| Intermittent · 임시 고정본 | qty_rmse | 3.215355 · s2p2_matched_head | 2.966841 | -0.248514 | -7.7290 | 3seed 평균 오차가 낮음 |
| Intermittent · 임시 고정본 | qty_mae | 1.312281 · thp | 1.197329 | -0.114952 | -8.7597 | 3seed 평균 오차가 낮음 |

**확인된 사실:** Taxi와 현재 Intermittent 고정본은 평균 전체 RMSE·MAE가 각각 외부6종의 지표별 최저 평균보다 낮다. RAF는 전체 RMSE가 S2P2보다 0.002454 높고, 전체 MAE가 RMTPP보다 0.244129 높다. 따라서 RAF 전체 성능 우세 또는 세 데이터셋 전체 우세로 서술할 근거는 없다. RAF에서 단순 이력 평균의 RMSE 39.357906도 Shared15 평균 39.775389보다 낮다. 다만 그 단순 기준선의 MAE 11.301169는 Shared15보다 높고 시간 모델이 없어 TimeNLL은 N/A다. E1 `simple_quantity_baselines`가 이 비교의 근거다.

**시간 지표의 반대 방향:** Taxi Shared15 전체 TimeNLL 1.148959는 외부6종 모두보다 높다(외부 최저 NHP 0.683595). RAF Shared15 3.815814도 THP 3.456260보다 높고, Intermittent 임시 Shared15 22.553775도 THP 8.326340보다 높다. 수량 RMSE·MAE의 개선을 시간 예측·확률 보정 또는 결합 성능의 일관된 개선으로 확대할 수 없다. 큰수량 TimeNLL의 매우 작은 수는 원래 값의 크기를 유지해 과학 표기로 적었으며 0으로 반올림해 동일하다고 판단하지 않는다.

## 3. seed 예외와 내부 ablation

외부 모델에 대한 전체 수량 오차의 악화 사례를 모두 남긴다. 차이는 Shared15−동일 seed 비교군이며 양수는 악화다. 전체·큰수량 TimeNLL 및 나머지 모든 seed별 차이는 E1/E2의 `seed_differences`에 보존돼 있다.

| 데이터셋 | 비교군 | seed | 지표 | Shared15 | 비교군 | 절대 차이 | 상대 차이(%) |
|---|---|---:|---|---:|---:|---:|---:|
| RAF | rmtpp | 42 | qty_mae | 10.224523 | 9.866792 | +0.357732 | +3.6256 |
| RAF | rmtpp | 52 | qty_mae | 10.038404 | 9.940155 | +0.098249 | +0.9884 |
| RAF | rmtpp | 62 | qty_mae | 10.188499 | 9.912093 | +0.276406 | +2.7886 |
| RAF | thp | 42 | qty_mae | 10.224523 | 10.077917 | +0.146606 | +1.4547 |
| RAF | thp | 52 | qty_mae | 10.038404 | 9.918316 | +0.120088 | +1.2108 |
| RAF | nhp | 42 | qty_mae | 10.224523 | 9.895699 | +0.328824 | +3.3229 |
| RAF | nhp | 52 | qty_mae | 10.038404 | 9.937777 | +0.100626 | +1.0126 |
| RAF | nhp | 62 | qty_mae | 10.188499 | 10.073959 | +0.114540 | +1.1370 |
| RAF | s2p2_matched_head | 42 | qty_rmse | 39.875356 | 39.637374 | +0.237982 | +0.6004 |
| RAF | s2p2_matched_head | 52 | qty_rmse | 39.890167 | 39.730669 | +0.159498 | +0.4014 |
| RAF | s2p2_matched_head | 62 | qty_mae | 10.188499 | 9.978645 | +0.209854 | +2.1030 |
| RAF | attnhp_matched_head | 42 | qty_mae | 10.224523 | 9.961348 | +0.263176 | +2.6420 |
| RAF | attnhp_matched_head | 52 | qty_mae | 10.038404 | 9.974579 | +0.063824 | +0.6399 |
| RAF | attnhp_matched_head | 62 | qty_mae | 10.188499 | 10.015874 | +0.172624 | +1.7235 |
| Taxi | s2p2_matched_head | 52 | qty_rmse | 109.916402 | 108.266626 | +1.649776 | +1.5238 |
| Intermittent · 임시 고정본 | s2p2_matched_head | 42 | qty_rmse | 2.907007 | 2.898904 | +0.008103 | +0.2795 |

Taxi의 S2P2 대비 전체 RMSE는 seed52에서 +1.5238%, Intermittent 임시 고정본은 seed42에서 +0.2795% 악화한다. 따라서 두 데이터셋의 평균 우세도 모든 seed의 우세를 뜻하지 않는다. RAF의 S2P2 대비 큰수량 RMSE 역시 seed42·52에서 각각 +1.7956%·+0.0546% 악화한다. Taxi 큰수량 RMSE·MAE는 S2P2 대비 seed42·52에서 악화하며, Intermittent 임시 큰수량 RMSE·MAE도 S2P2 대비 seed52에서 각각 +11.8890%·+39.6989% 악화한다.

MLP16은 내부 구조 대조(ablation)다. 외부6종과의 1차 비교를 대신하는 기준으로 쓰지 않는다.

| 데이터셋 | Shared15 / MLP16 전체 RMSE | Shared15 / MLP16 전체 MAE | Shared15 / MLP16 큰수량 RMSE | Shared15 / MLP16 큰수량 MAE | 보존할 예외 |
|---|---:|---:|---:|---:|---|
| RAF | 39.775389 / 40.247819 | 10.150475 / 10.148947 | 405.165825 / 414.097369 | 356.589070 / 369.902563 | 전체 MAE 평균은 Shared15가 약간 높음; seed62 MAE +1.2089% |
| Taxi | 107.084863 / 113.392687 | 34.323697 / 36.039467 | 546.872240 / 594.683335 | 431.967496 / 481.486066 | seed42 전체 RMSE +21.9263%, MAE +15.0383%; 전체 TimeNLL은 세 seed 모두 악화 |
| Intermittent · 임시 고정본 | 2.966841 / 2.935730 | 1.197329 / 1.222756 | 15.713591 / 13.784552 | 14.632456 / 12.739638 | 전체 RMSE·큰수량 RMSE·MAE 평균이 Shared15에서 높음; seed42 전체 RMSE·MAE, seed52 전체 RMSE 악화 |

Shared15의 관계 보정은 9,600개 파라미터이며 기존 MLP16은 24,576개다(M1·M2 및 기존 E1/E2 동명 Markdown 결과표). 보정 파라미터만 60.9375% 줄었다는 계산은 가능하지만, 두 모델은 공유 여부와 용량이 함께 바뀐다. 성능 차이를 공유 하나의 인과 효과로 분리할 수 없다. 코드에 있는 prefix-capacity 독립 모드는 활성 분기 prefix마다 파라미터 수를 맞추는 별도 대조 구조이며, 이 문서의 결과표에는 그 실험 결과가 없다. 같은 파라미터 수가 같은 함수 공간을 뜻하지도 않는다.

## 4. 코드와 receipt가 지지하는 방법 설명

| 항목 | 확인된 구현 | 근거 | 서술 시 경계 |
|---|---|---|---|
| 예측 입력 | 관측된 시간 간격과 수량을 각각 log1p로 변환한다. 목표 수량은 0으로 가리고 목표 위치의 memory write를 막은 뒤 직전 관측 상태에서 다음 값을 예측한다. | M1의 상속 경로, `CountAwareTPP.py:continuous_features`, `count_aware_tpp_backbone/core.py:target_outputs`, Train receipt의 `time_input_transform`·`quantity_input_transform` | 미래 목표 간격을 예측기 입력으로 사용한다고 서술하지 않음 |
| 관계 보정 위치 | hidden_dim 64의 기존 두 encoder block 사이에 residual correction을 더한다. 기존 persistent memory 16개와 static Hard-LMM prototype 64개/top4 산술평균 retrieval을 유지한다. | `CountAwareTitanMultiLagDetail.py:_encode_base`, `encode_task_states`, metadata; M1 `CountAwareTitanSharedHistoryMLP` | 이 보정이 전체 encoder 또는 출력 head를 교체한다고 서술하지 않음 |
| 공유 관계 기저 | 하나의 bias-free 128→15 입력 projection으로 현재 상태와 바로 전 관측 상태를 결합하고 exact GELU를 한 번 계산한다. 8개의 bias-free 15→64 출력 projection이 이를 재사용한다. | M1 `SharedHistoryCorrection.__init__`, `forward` | 8개 서로 다른 실제 lag의 값을 읽는 모델로 설명하지 않음 |
| 분기 활성 조건 | 실제 관계 입력은 모든 분기에서 바로 전 관측 하나다. 기존 가용성 임계값 1·2·4·8·16·32·64·128을 유지하고, 관측 prefix count가 임계값보다 클 때 해당 분기를 켠다. 합산 residual은 활성 분기 수가 아니라 항상 8로 나눈다. | M1 및 `CountAwareTitanMultiLagDetail.py:lag_source_indices(mode="local")` | 임계값에 도달했다는 조건과 서로 다른 lag 내용을 조회한다는 주장을 혼동하지 않음 |
| 수식 | z_i=GELU(W_c h_i+W_p h_prev), Δh_i=(1/8)Σ_b a_ib V_b z_i; W=[W_c,W_p]∈R^(15×128), V_b∈R^(64×15). | M1 `SharedHistoryCorrection.forward` | padding/withheld reset을 포함한 a_ib의 가용성 조건을 생략해 전 구간 사용으로 읽히지 않게 함 |
| 초기화와 수명 | 입력 projection은 기본 Linear 초기화, 출력 projection은 0 초기화다. forked CPU RNG로 공통 encoder/head 초기화 흐름을 보존한다. 보정은 sample-local forward이며 온라인 메모리 쓰기는 없다. | M1 `__init__`; 상속 metadata의 `multilag_detail_lifetime`, `online_writes` | 장기 온라인 적응이나 학습 이후 영구 기록 효과를 주장하지 않음 |
| 수량 head와 학습 목표 | u=softplus(linear(h)), q_hat=expm1(u). 직접 log-MSE (u−log1p(q))²와 시간 NLL을 계수1로 합산한다. lambda_tail=0, quantity normalization=none이다. | `CountAwareTPP.py:predict_quantity`, `quantity_outputs`; `core.py:target_outputs`; E1/E2의 alpha1·log_mse·joint proof | 수량의 확률분포 NLL을 학습했다고 부르지 않음 |
| 시간 head와 평가 단위 | 관측 간격을 양의 정수로 기록하는 likelihood에 연결한 이분산 lognormal duration head다. 보고한 TimeNLL은 `recorded_positive_integer_time_nll`이다. | E1/E2 각 record `time_head.observation_likelihood`, selected train receipt | v0.7의 과거 clamped time loss 또는 native marked intensity NLL과 혼합하지 않음 |
| 학습과 선택 | batch128, hidden64, 최대300 epoch, 최소40, patience40, AdamW lr0.001, grad clip1. 전체 Validation raw quantity RMSE의 최초 엄격한 유한 최소 epoch를 고정하며 MAE도 같은 checkpoint를 사용한다. | E1/E2 records의 selection_rule; selected train receipt의 resume arguments 및 optimizer proof | Test RMSE/MAE에 따른 epoch 재선택이나 지표별 checkpoint 선택을 암시하지 않음 |
| 외부6종의 공통 조건 | RMTPP·THP·NHP·SAHP·S2P2·AttNHP encoder를 같은 수량·시간 head와 관측 이력 인터페이스에 연결했다. S2P2/AttNHP는 observed t_i+ 상태를 사용하는 matched-head adaptation이다. | E1/E2 `time_head`, 조건 검증 필드; `CountAwareAdditionalTPP.py` 모듈 설명·metadata | 원논문 native marked NLL 및 모든 원본 튜닝 조건의 재현 또는 전반적 우열로 일반화하지 않음 |

RAF 이력 창은 84/max_seq_len84, Taxi는 168/256, Intermittent는 520/256이다. 이 값과 단위는 실험 계약의 `lookback_weeks` 필드 및 각 데이터셋의 기록 단위에 결속되므로, 세 자료가 같은 달력 단위를 가진다고 해석하지 않는다. 숫자만 임의 환산해 논문에 쓰지 않는다.

## 5. 재현 범위와 Intermittent 최종화 gate

RAF·Taxi6는 5090/cuDNN92000에서 학습한 선택 checkpoint를 5080/cuDNN91900에서 평가했다. Test 전에 저장된 Validation 개별 수량 예측·시간 손실과 여섯 지표를 고정 rtol=atol=1e-5로 재생 확인했다. 각 Test receipt는 `parameters_unchanged=true`, `context_unchanged=true`, `native_runtime_checked=true`를 기록한다. 외부·MLP16의 과거 참조는 재학습·재추론하지 않았고 일부 과거 CPU 재현 결과가 있어 동일 Runtime에서의 새 비교로 볼 수 없다.

Intermittent3는 snapshot current epoch42/40/38 안의 best_state_dict 중 best epoch5/5/11을 고정한 임시 평가다. 저장된 best history와 전체 Validation RMSE·MAE·TimeNLL의 재생은 확인했지만, 과거 큰수량 지표·개별 예측 파일이 없어 해당 재생의 동등성까지 검증했다고 말할 수 없다. 현재 상태 시간 head proof와 선택된 best 상태가 초기와 다르다는 사실은 확인됐지만, 선택 epoch 자체의 개별 gradient proof가 있는 것은 아니다.

다음 canonical tensor SHA는 **임시 고정본**의 정체성이다. 최종 학습 종료 후 원본 회수와 Validation 재검증이 끝나면 최종 선택 epoch와 이 SHA를 대조해야 한다. 직렬화 방식 차이로 checkpoint binary SHA가 달라도 canonical tensor SHA가 같을 수 있다. 현재 이 문서는 최종 동등성을 확인하지 않는다.

| seed | 임시 선택 epoch | canonical model tensor SHA-256 | 개별 원본 Test receipt |
|---:|---:|---|---|
| 42 | 5 | `2a7e6f923aae9d1e909d8d0422e1de58e21ccf2c928e1fedecbbc1f00350f021` | [search_artifacts/titantpp_shared_inter3_interim_test_20261020_v1/retrieval/originals/results/intermittent_frozen_5000__titantpp_history_mlp_shared15_v1__seed42__losslog_mse/test_receipt.json](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_shared_inter3_interim_test_20261020_v1/retrieval/originals/results/intermittent_frozen_5000__titantpp_history_mlp_shared15_v1__seed42__losslog_mse/test_receipt.json) |
| 52 | 5 | `613f6e815d086336fb47adc87abd553b979262f73371ea528da5bccb33ce314b` | [search_artifacts/titantpp_shared_inter3_interim_test_20261020_v1/retrieval/originals/results/intermittent_frozen_5000__titantpp_history_mlp_shared15_v1__seed52__losslog_mse/test_receipt.json](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_shared_inter3_interim_test_20261020_v1/retrieval/originals/results/intermittent_frozen_5000__titantpp_history_mlp_shared15_v1__seed52__losslog_mse/test_receipt.json) |
| 62 | 11 | `36e0074ce3d0e1d0542b1bec51493c1c377747dd1afe591681ba1f672ae34977` | [search_artifacts/titantpp_shared_inter3_interim_test_20261020_v1/retrieval/originals/results/intermittent_frozen_5000__titantpp_history_mlp_shared15_v1__seed62__losslog_mse/test_receipt.json](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_shared_inter3_interim_test_20261020_v1/retrieval/originals/results/intermittent_frozen_5000__titantpp_history_mlp_shared15_v1__seed62__losslog_mse/test_receipt.json) |

**논문 준비 자료의 최종화 조건 — 외부 작업 대기:** Intermittent 최종 선택 epoch가 5/5/11이고 canonical tensor SHA도 위 임시 고정본과 모두 같다는 gate가 확인되기 전에는 원고 본문·초록·TeX 초안을 작성하지 않는다. gate 통과 후에도 아래 PAKDD 2027 작성 규정에 따라 자동 본문 생성으로 넘어가지 않고 인간 작성용 결과표·근거 자료·빈 구조를 준비한다. 다른 경우에는 결과 지위를 먼저 다시 정리해야 하며, 임시 수치를 최종 결과로 바꾸거나 Test를 보고 유리한 checkpoint를 고르지 않는다. 새 seed72·82의 추가42개 결과는 이 원장의 3seed 값에 아직 합치지 않는다.

## 6. 반드시 보존할 해석의 한계

- 반복 Test를 확인하며 개발한 탐색적 후속 평가다. `independent_new_test=false`이며, 기존 Test에서의 checkpoint 비재선택만으로 새로운 독립 평가가 되지는 않는다. 추가 seed도 같은 Test 자료를 쓰면 이 한계를 해소하지 못한다.
- 외부6종 비교는 공통 head를 사용하는 encoder 비교다. 특히 S2P2의 affine-doubling recurrence 구현은 원래 순차식과 맞추기 위한 로컬 adaptation이며, 코드상 O(N log N) element work / O(log N) stage다. 원논문의 work-efficient scan이나 native head의 효율·성능을 대신 측정했다고 할 수 없다.
- RAF 전체 RMSE·MAE에서 외부 최저 평균에 미치지 못한다. Taxi·Intermittent의 수량 평균 우세와 별개로 seed 예외, 큰수량 악화, TimeNLL 악화를 함께 제시해야 한다.
- 3seed의 평균·표본SD만 있으며 p-value, 신뢰구간, 검정력, 다중 비교 보정 또는 독립 반복 데이터셋 추정은 없다. 통계적 유의성·일반적 우월성·인과적 기전으로 표현하지 않는다.
- 큰수량 지표의 support는 RAF39, Taxi83, Intermittent243이다. 전체 support와 분리해 기록하며 tail 결과의 높은 변동을 숨기지 않는다. TimeNLL의 near-zero 기준에 대한 상대 백분율은 크게 불안정하므로 원래 절대 값도 함께 확인한다.
- 서로 다른 학습·평가 호스트와 Runtime, 과거 평가 경로가 남는다. Validation 고정 허용오차 재생은 해당 예측 경로의 수치적 결속 근거이며 하드웨어 효과가 전혀 없다는 증명이 아니다.
- Shared15 대 MLP16은 공유와 보정 용량 감소가 함께 바뀐다. 어떤 요인이 관측된 차이를 만들었는지 분리하는 대조 결과가 아직 없다.
- Intermittent 표는 최종 학습 선택과의 동등성 gate 이전의 임시 결과다. 이 값으로 원본 raw9 전체 종료나 최종3데이터셋 성능을 선언하지 않는다.

## 7. 효율: 구조상 확인된 것과 아직 측정하지 않은 것

| 항목 | 현재 확인 가능한 사실 | 측정 상태 / 필요한 기록 | 지금 할 수 없는 주장 |
|---|---|---|---|
| 보정 파라미터 | Shared15 9,600 = 128×15 + 8×15×64. MLP16 24,576 대비 60.9375% 감소. | 코드와 기존 결과에서 확인. 전체 모델 파라미터 감소율과 구분. | 전체 모델이 60.9375% 작다고 주장 |
| 비선형 관계 계산 | 공유 입력 기저와 GELU를 한 번 만든 뒤 분기별 출력 projection에 재사용. | 코드상 계산 재사용은 확인. 실제 kernel 수/FLOPs/MACs는 미측정. | 학습·추론이 특정 배수 빨라졌다고 주장 |
| 추론 latency | 동일 GPU·라이브러리·batch·sequence 길이에서의 정식 비교 없음. | 미측정: warm-up, CUDA synchronize, 반복 p50/p95, batch1/128/256 및 길이84/256을 구분할 필요. | 낮은 지연 또는 실시간 처리 보장 |
| throughput | 공정하게 맞춘 targets/sec 또는 events/sec 비교 없음. | 미측정: padding 포함/실관측 처리량, precision·batch·길이·동시 worker 수를 기록. | S2P2·AttNHP·기존 MLP16보다 고속이라고 주장 |
| 학습 비용 | 각 fit의 epoch/elapsed 로그가 있더라도 중단·호스트·선택 epoch가 다름. | 공정 비교 미측정: epoch 시간, 선택까지 시간, 총 wall-clock/GPU-hours를 동일 조건에서 분리. | 작은 보정 파라미터 수만으로 총 학습 비용 감소 단정 |
| GPU/CPU 메모리 | 기존 receipt별 peak가 있더라도 모델·배치·Runtime 차이가 있어 공통 비용 실험이 아님. | 공정 비교 미측정: allocated/reserved peak와 CPU RSS, optimizer 상태·training/inference를 구분. | 항상 낮은 메모리 또는 큰 batch 수용 보장 |
| 확장성·에너지 | 길이 또는 batch 증가에 따른 정식 곡선, 소비전력·에너지 기록 없음. | 미측정: 공통 조건의 길이별 비용과 joules/target, 측정 장비·동시 작업 영향을 기록. | 전체 모델의 선형 확장성·에너지 효율·배포 비용 우세 |

위 항목은 측정이 필요한 목록이며 이번 작업에서 benchmark를 실행하거나 새 효율 실험을 승인한 기록은 아니다. GPU 합성 자격 검증의 peak memory도 실행 허용 여부를 확인하는 증적이며 모델 간 정식 효율 비교와 구분한다.

## 8. 기술적 검산과 통계 해석 점검

이번 작업에서 E1/E2의 각 dataset/model/group/metric에 대해 records 3개의 평균과 표본SD를 다시 계산해 JSON summary와 abs_tol=1e-9, rel_tol=1e-12로 대조했다. 전체 수량의 외부 최저 평균·절대/상대 차이도 재계산했다. 두 `final_receipt.json`에 결속된 결과 JSON SHA와 현재 파일 SHA가 일치함을 확인했다. 이는 기존 산출물의 로컬 검산이며 새로운 추론 결과가 아니다.

ARS 통계 오류 점검은 **11/11 범위를 확인**했다. 검정·회귀가 없는 항목은 적용 불가이며, 원자료 전체의 통계적 가정을 새로 검증했다는 뜻은 아니다.

| 점검 항목 | 이 근거에 적용한 판단 |
|---|---|
| Simpson 역전 | dataset·overall/tail·seed를 분리한다. 평균 우세가 seed 또는 tail에서 뒤집히는 예외를 위에 보존했다. |
| 생태학적 오류 | seed 평균으로 개별 target·상품·지역의 개선을 주장할 수 없다. |
| Berkson 선택 편향 | frozen 데이터·tail 조건에 결속된 비교다. 이 선택된 자료에서 다른 수요 모집단으로 일반화하지 않는다. |
| Collider 편향 | 별도 공변량 조정·인과 회귀가 없어 해당 분석에는 적용 불가. |
| 기저율 무시 | 분류 성능 분석은 아니지만 tail support와 전체 support를 함께 보존했다. |
| 평균 회귀 | pre/post 대상 선정 실험이 없어 직접 적용 불가. 낮은 Validation epoch를 골랐다는 선택은 별도 공개한다. |
| 생존자 편향 | RAF·Taxi 완료6과 Intermittent 임시3을 구분한다. raw9 전체 완료 결과처럼 집계하지 않는다. |
| 다중 탐색 | 외부6종·6지표·3dataset의 여러 비교와 불리한 결과를 보존한다. 유의성 주장은 하지 않는다. |
| 분석 경로 선택 | 반복 Test 참고와 후속 구조·용량 선택의 탐색적 성격을 명시했다. |
| 상관·인과 혼동 | 모델 비교의 관측 차이를 공유 기전 하나의 인과 효과나 실제 수요의 인과 관계로 해석하지 않는다. |
| 역인과 | 관측 이력의 다음 target 예측이며 수요 발생 원인에 관한 방향성 인과 주장은 없다. |

## 9. 남은 작업 순서

**Intermittent 최종 학습 선택을 임시 고정본과 대조한다 — 외부 작업 대기**

- 대상: 5090 원본 raw9의 Intermittent seed42·52·62. 종료·회수·Validation 재검증 뒤 epoch5/5/11 및 위 canonical tensor SHA가 모두 같은지 확인한다. 이 문서는 원격 상태를 새로 확인하지 않았다.

**gate 결과를 반영해 인간 작성용 논문 준비 자료를 정리한다 — 다음 작업**

- 앞 단계 통과 후 외부6종의 결과표·근거 자료와 사람이 채울 빈 구조를 준비한다. MLP16은 내부 ablation으로 두고, 수량 결과와 TimeNLL·seed·RAF 반례를 함께 유지한다. 본문·서론·초록·novelty 문장은 인간 저자가 작성한다. 도구의 논문 문장 지원은 아래 공식 규정의 문법 오류 교정 범위를 따른다.

**추가 seed72·82의 결과를 원래3seed와 구분해 반영한다 — 외부 작업 대기**

- 대상: 5080 extra42 별도 작업. 학습·전체 Validation gate·Test 증적이 완료되기 전에는 5seed 결과를 표기하지 않는다. 결과 반영 시 기존3seed 표와 source·Runtime·평가 범위를 추적하고 반복 Test의 한계를 유지한다.

## 10. PAKDD 2027 작성 경계 — 2026-10-20 공식 페이지 확인

공식 [Author’s Kit](https://www.pakdd2027.org/pages/calls/authors-kit)의 생성형 AI 규정에는 “may only be employed for grammatical error correction”이라고 명시돼 있다. submission form에서 도구 사용을 명시하고 설명해야 한다. 이 원장은 연구 원자료의 수치·코드 사실·근거 경로·한계를 정리한 자료로 유지하며, 투고용 본문·서론·초록·novelty 표현을 생성하거나 인간 저자의 원고처럼 사용하지 않는다. 해당 페이지는 최종 내용의 정확성·표절 확인에 관한 저자 책임도 명시한다.

최종 checkpoint gate를 통과한 뒤에도 도구가 준비할 대상은 결과표·근거 자료·인간 작성용 빈 구조다. 인간이 작성한 문장의 문법 오류 교정을 넘어서는 자동 원고 작성으로 확대하지 않는다. 사용 기록의 공개 범위와 투고 규정 충족 여부는 저자가 최종 확인한다. 이 문서 자체가 학회 규정 준수를 보장하지는 않는다.

형식상 Research Track은 참고문헌·부록 등을 포함해 최대 12쪽이며 Springer LNCS/LNAI 형식을 사용한다. [공식 일정](https://www.pakdd2027.org/pages/dates)의 논문 제출 마감은 **2026-11-20 23:59 AoE**다. 형식·일정과 생성형 AI 규정은 제출 전에 공식 페이지에서 다시 확인할 항목으로 보존한다.
