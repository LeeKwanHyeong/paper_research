# GRU 보정량 제한: 로컬 준비 완료

GRU 보정 출력만 제한하는 후보의 계약·모델·연결부와 로컬 검증을 완료했다. 기존 CNN·FULL 결과는 그대로 보존했다. 새 후보의 실제 학습과 Validation 결과는 아직 없다.

**보정 상한과 비교 조건을 고정한다 — 완료**

- 후보는 `titantpp_cnn_gru54_residual_cap025` 하나다. GRU 보정 벡터의 L2 크기를, 보정을 더하기 직전 block1 표현 크기의 25% 이내로 제한한다. 수량 예측값 자체의 25% 제한이 아니다.
- 추가 학습 파라미터 없이 기존 FULL의 초기 가중치·난수·CNN·head·손실·학습 예산을 유지한다. 상한 안에서는 원래 보정 출력을 그대로 사용한다.
- Taxi·RAF·Intermittent 각각 seed42·52·62를 사용해 후보 9개를 계획했다. 기존 구성요소 캠페인의 CNN·FULL Validation 18개를 읽기 전용 기준선으로 결속했다. 다른 캠페인의 FULL이나 Test 결과를 섞지 않았다.
- 최소40·최대300 epoch, patience40, batch128, AdamW lr0.001·weight decay0.01·gradient clip1을 유지한다. 선택은 전체 Validation raw 수량 RMSE의 최초 strict finite 최소 epoch다.
- `rho=0.25`는 이번 탐색 비교를 위해 미리 고정한 값이다. 제한을 구현했다는 사실만으로 일반화 개선이나 기존 순위 반전의 원인이 입증되지는 않는다.

**구현·계약·원본 보존을 검증한다 — 완료**

- 기존 구성요소 호환14개, 상한 모델10개, 연결부8개, 준비 계약12개: 최종 합성 CPU 검사 총44개가 통과했다. 실제 연구 데이터 행을 읽거나 학습하지 않았다.
- 상한·작은 값·0 초기화·유효 gradient, 인과적 이력 mask와 target 차단, batch 일치, checkpoint 신원과 변조 차단을 확인했다. 실제 공통 인자 생성기를 합성 metadata로 연결해 손실·head·예산도 확인했다.
- 독립 QA에서 원본 참조241개, 복제 source134개, 현재 신규 모듈3개의 SHA와 기존 Validation 18개를 재검증했다. 확인된 결함은 없다.
- 신규 후보는 계획9개·시작0개다. 원격 호출·실제 학습·신규 Validation 평가·Test는0이며 실행 승인·학습 permit도 생성하지 않았다.

준비 bundle: `search_artifacts/titantpp_gru_residual_cap_validation_20261014_v1`.

| 증적 | 고정 값 |
|---|---|
| execution_contract raw SHA | `fae92b1695483ee4c978a6f72ddff7628881fc9bb2af42b246508835bddf06c9` |
| execution_contract canonical SHA | `6b5564173eb6f9feed35bc89142bfdde50944c285c4a402d2a4f5b311e2983f0` |
| preparation_receipt raw SHA | `709dc99c777d5b199820fc1d40baf65a4a534d7e26808a018c64ab678004cb68` |
| 후보 신원 버퍼 SHA | `c006d86bbc5d2590426f217b2d97ee78e34d3528c44544b047e3891ae44af71a` |

계약의 source134는 기존 source131을 바꾸지 않고 신규 모델·runner·prepare 세 파일을 추가한 것이다. 입력 Parquet bytes는 이번 준비에서 다시 읽거나 복사하지 않았다. 입력 SHA·split·Train 통계의 원래 계약을 상속했고, 실제 native 입력 검증은 다음 단계다.

**native 실행 경계를 준비한다 — 다음 작업·실행 승인 필요**

- 대상은 기존 seed와 같은 환경이다: Taxi·RAF와 Intermittent62는5080/cuDNN91900, Intermittent42·52는5090/cuDNN92000이다. 이 값은 계획이며 현재 Runtime·GPU 점유 상태를 새로 조회하지 않았다.
- 실제 실행 승인 후 별도 native 경로에서 source·입력 bytes·GPU UUID·Runtime·초기화·합성 검증을 확인하고, 소유권·dispatcher·qualification·permit 연결을 마련해야 한다. 현재 `run_fit` 진입은 차단돼 있고 CLI는 합성 CPU 검사만 제공한다.
- 상속한 시간 예산은 조건당36시간·host당168시간이다. 새 후보의 시작시각·절대 deadline은 아직 발급되지 않았다. 기존 캠페인의 deadline·claim·failure는 변경하지 않았다.
- 고정된 조건으로 후보9개를 fresh fit으로 학습한다. 기존 CNN·FULL 재학습, warm start, 자동 retry·resume, 상한 sweep은 하지 않는다.

**새 Validation 결과를 기존 기준선과 비교한다 — 후속 작업**

- 후보9개가 끝난 뒤 전체 raw 수량 RMSE·MAE, 큰 수량 RMSE·MAE, 전체·큰 수량 recorded positive-integer TimeNLL을 같은 선택 checkpoint에서 평가한다.
- 데이터셋별 3seed 평균·표본SD와 seed별 후보−FULL, 후보−CNN 차이를 따로 보고한다. 불리한 지표와 seed 예외도 포함한다.
- 후보−FULL은 파라미터 수가 같은 보정 제한 비교다. 후보−CNN은 활성 파라미터 수가 달라 별도 비교로 해석한다. 이미 Test를 본 뒤 제안한 후속이므로 새 독립 확증 실험으로 부르지 않는다.
- 후보 Test 실행과 원고 반영은 이번 준비 범위에 포함하지 않는다.

문서: [실험 계약](EXPERIMENT_CONTRACT.md), [로컬 검증](LOCAL_VALIDATION.json), [독립 QA](FINAL_QA.md).
