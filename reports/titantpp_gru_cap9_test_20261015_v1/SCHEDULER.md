# CAP9 고정 checkpoint Test 후속 실행

## 승인과 목적 — 확정

사용자 원문: “ㅇㅇ 그럼 seed62까지 끝나면 Validation 기준으로 선택한 checkpoint를 고정해서 CAP 9개 Test도 진행하자. 실행 승인할게. 기존 CNN·FULL과 데이터셋별 3seed RMSE·MAE, seed별 차이까지 비교해줘.”

대상은 rho=0.25 CAP의 Taxi·RAF·Intermittent × seed42/52/62, 고유9개다. 기존 CNN·FULL18 Test는 같은 데이터셋·seed의 기존 고정 결과를 재사용한다. 다른 FULL 캠페인과 합치거나 좋은 결과로 바꾸지 않는다. 새 학습·추가 seed/rho·checkpoint 재선택은 없다.

이 승인은 이전 Validation 전용 지침의 Test 금지와 Validation 완료 직후 자동화 삭제 조건을 이 범위에서만 대체한다. 원 학습 계약·source134·입력·선택·손실·head·Runtime·claim/failure·실패 원본과 기존18 결과는 보존한다. 원고와 다른 자동화는 변경하지 않는다.

## 원 학습 종료와 회수 — 외부 작업 대기

기존 hourly_monitor_parallel3_v3/SCHEDULER.md와 봉인을 그대로 적용한다. 기존 관측·회수 helper만 회차당 한 번 실행한다. 각9 scientific worker의 실제 exit0와 binary SHA, 원순차 dispatcher의 계획된 행정exit1, 새 dispatcher 실제exit0·complete·owner/GPU 부재를 검증한 뒤 기존 collector가 host별 한 번 회수한다.

기존 retrieval/final_receipt.json의 complete/9후보/3seed/18참조/held_out_test_evaluated=false 및 모든 로컬 원본·archive·manifest·Validation 집계 SHA를 검증한다. 새 Test helper는 종료 cache와 native terminal manifest를 로컬에서 재검증한다. 완료 전에는 remote0·회수0·Test0으로 기다린다.

## 선택 모델을 고정해 같은 서버에서 평가 — 다음 작업

새 bundle: search_artifacts/titantpp_gru_cap9_test_20261015_v1.
새 native root: /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_gru_cap9_test_20261015_v1_5080 및 _5090.

1. 전체 fullValidation raw 수량 RMSE 최초 strict finite 최소 epoch를 원본 history에서 재검증한다. 선택 binary·전체 metadata·tensor SHA·source134·입력·전처리를 연결한다. 종료 checkpoint는 원 collector archive에 그대로 보존한다.
2. 기존 모델 및 지표 구현을 그대로 사용한다. 5080은 Taxi/RAF6+Inter62, 5090은 Inter42/52이며 기존 Python·GPU UUID·cuDNN을 유지한다. 같은 고정 입력의 여섯 지표와 대상 지문을 native Validation9에서 재현하고 각 실제 worker exit0를 검증한다.
3. 전9개의 Validation receipt·지문·실제 exit0가 모두 통과한 gate9 뒤에만 Test9를 평가한다. Test 결과로 재선택·보정·학습하지 않는다. 실제 Test로 실행 도구를 디버깅하지 않는다.
4. 각 host·stage의 실제 pipeline child exit와 worker exit를 보존한다. timeout·조회 실패와 평가 실패를 구분하며 자동 retry/resume는 하지 않는다. supervisor wrapper 자체의 OS exit를 직접 관측했다고 주장하지 않는다. 실패·기존 intent/claim이면 원문을 남기고 멈춘다. 원 학습 프로세스에 signal을 보내지 않는다. 새 평가 child의 시간 제한과 정리는 소유한 별도 process group만 대상으로 한다.

## 결과 비교와 자동화 종료 — 다음 작업

같은 checkpoint에서 전체 raw 수량 RMSE(주 지표)·MAE, Train 고정 큰수량 기준 Taxi q>3449/RAF q>200/Intermittent q>187의 RMSE·MAE, 전체/큰수량 recorded positive-integer TimeNLL을 계산한다. 단순 직전수량·이력평균 기준선의 시간 지표는 N/A다.

데이터셋별 CAP/CNN/FULL의 3seed 평균·표본SD(ddof=1)와 모든 seed별 CAP−CNN/CAP−FULL 절대·상대 차이를 보고한다. 음수는 개선, 양수는 악화이며 불리한 결과를 제외하지 않는다. 여섯 지표와 원문은 THREE_SEED_TEST.json, 주요 RMSE/MAE 비교는 THREE_SEED_TEST.md에 기록한다. Intermittent42/52=5090/cuDNN92000,62=5080/cuDNN91900의 차이와 host별 시간을 보존한다.

이번 CAP는 이전 Test 순위 반전을 확인한 뒤 설계한 탐색적 후속 실험이다. 새 독립 평가·일반화의 보편적 개선·통계 유의성·반전 원인 확정으로 확대하지 않는다.

final_receipt.status=complete,candidate_test_count=9,baseline_reference_count=18,seed_count=3,validation_replay_count=9,actual_native_worker_exit0_count=18,actual_pipeline_exit0_count=4 및 archive/manifest/원본/결과/THREE_SEED_TEST.json·md 모든 SHA를 재검증한 후 최종 결과를 보고하고 gru-cap-9-validation 하나만 삭제한다. 학습·Validation만 완료됐을 때는 삭제하지 않는다. 실패나 미확정이면 유지하고 필요한 조치를 한 번 보고한다. 동일 실패 중 원격 재실행과 자동 수정·재봉인은 하지 않는다.

## 로컬 준비 검증 — 완료

합성 입력만으로 CAP 구조 식별·다른 구조 거부·정답 차단·단건/배치 일치·평가 전후 파라미터 불변·여섯 지표 수계산을 검증했다. 실제 데이터/선택 checkpoint 읽기/optimizer update는 0이다. 별도 계약 테스트13개는 대기 시 remote0, 중복 실행 차단, 전9 Validation gate, 실제 exit0, 승인된 host/기간, 최초 최소 epoch, 합성9 checkpoint 결속, 3seed 표본SD·seed차이·대상 불일치 차단을 검증했다. native CUDA 평가와 실제 Test는 아직 실행하지 않았다.

## 실행 명령과 봉인

기존 학습 helper 실행 뒤 같은 회차에서 아래 후속 helper를 한 번 실행한다. 학습/회수 미완료라면 로컬 검증 후 remote0으로 대기한다. 기존 평가 attempt/failure가 있으면 자동 재실행하지 않는다.

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 -m paper.scripts.advance_titantpp_gru_cap9_test --mode advance --integrity-sha 3bd54e896b3fbcfe70cd6f0097a9140ba1be4ef335ed309788d718d5ebb7276d
```

신규 bootstrap rawSHA: `3bd54e896b3fbcfe70cd6f0097a9140ba1be4ef335ed309788d718d5ebb7276d`.
기존 training control rawSHA: `64f143793b2134e0db2fc9dd2e9ca80c782f00b5304c32989d5305e6c4c32ef7`.

로컬 실제 advance 검증: `waiting_for_verified_training9_and_original_collection`, `remote_calls=0`, `held_out_test_evaluated=false`, `checkpoint_reselection=false`. 미래 완료를 지금 완료된 것으로 보고하지 않는다.
