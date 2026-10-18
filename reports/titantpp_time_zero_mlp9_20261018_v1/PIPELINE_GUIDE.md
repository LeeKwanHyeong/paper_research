# 시간 가중치0 수량 전용 대조 — 후속 순서

## 기존 작업을 보존한다 — 현재 기준선

기존 시간0.1/10의18개 학습·Validation은 그대로 완료하도록 둔다. 외부 Cap27은 정지 상태를 유지한다. 기존 CNN·MLP16 기준선·선택 checkpoint를 재학습·재추론·재평가하지 않는다. 새0은 시간 입력·공유 backbone을 유지하고 시간 손실의 backward만 제외하는 수량 전용 학습9개다. 새 Test는 없다.

## 부모 큐 종료 후 새 학습에 진입한다 — 진행 중

새 관리 도구는 `paper.scripts.manage_titantpp_time_zero_mlp9`이다. 이미 prepare/deploy된 경우 반복하지 않는다. `START_RECEIPT.json`, `PREPARED.json`, `LAUNCHED_<host>.json`과 해당 observation/SHA를 먼저 읽는다. 새 큐는 각 host의 시간18 전체 train/Validation·dispatcher exit0·원본·소유PID/GPU부재 확인 뒤 native 검증과 최대3병렬 학습으로 진입한다. 대기는 CPU에서만 수행하며 기존 학습 기한을 연장하지 않는다.

현재 상태 요청 시 프로젝트 root에서 해당 관리 모듈의 `--mode status --host 5080` 및 `--host 5090`을 각각 한 번 실행한다. 두 조회는 병렬 가능하다. waiting 상태를 실제 α0 학습 시작으로 보고하지 않는다. 현재 신원이 불명확하거나 실패/미완료attempt가 있으면 원문을 보존하고 자동 retry/resume/deploy/kill을 하지 않는다. 사용자 승인 범위의 새 실험 경로만 사용하며 공용 Runtime·driver·인증은 변경하지 않는다.

## 새 학습 및 Validation 종료 후 원본을 회수한다 — 다음 작업

새 host의 전체 train/Validation 실제worker exit0·dispatcher actualexit0·명시적소유PID/GPU부재·기존정지상태·선택binary/입력/원본SHA가 통과하면 관리 모듈 `--mode collect --host <host>`로 한 번 회수한다. 이미 유효한 회수본이 있으면 재회수하지 않는다. 원본 archive/manifest/receipt는 별도 retrieval 경로에 보존한다. supervisor 자신의 OSexit와 직접 관측된 childexit를 구분한다.

같은 host의 기존 시간18 원본이 아직 로컬에 없으면, 새 α0 큐의 GPU 작업까지 모두 끝난 뒤 기존 `paper.scripts.manage_titantpp_time_weight_mlp18 --mode collect --host <host>`를 사용한다. 기존 collector는 α0 GPU가 실행 중이면 미확인 점유로 거부하므로 그 중간에 회수를 시도하지 않는다. 기존 시간18을 다시 실행하거나 종료 cache 시각을 현재 유휴 시각으로 바꾸지 않는다.

## 네 가중치의 수량·시간 지표를 비교한다 — 다음 작업

각 host의 기존18/신규9 원본을 검증한 뒤 기존18의 `paper.scripts.compare_titantpp_time_weight_mlp18`을 완료한다. 유효한 비교 결과가 이미 있으면 재실행하지 않는다. 이어 신규 `paper.scripts.compare_titantpp_time_zero_mlp9`을 한 번 실행해0/0.1/1/10 총36개 모델의 Validation을 비교한다.

신규9의 time_head_no_training_verification에는 시간 전용 head 초기/선택 SHA 동일, 모든 실제 optimizer step의 grad None·optimizer state 미생성·parameter불변 증적이 있어야 한다. 시간NLL은 미학습head의 진단값으로 표시한다. 수량 RMSE/MAE 개선과 joint모델 성능을 구분하며0을 시간 예측까지 학습한 모델의 Pareto 개선으로 분류하지 않는다.

모든 데이터셋별3seed 평균±표본SD(ddof1), seed별 절대·상대차이와 악화·예외·전체/상한도달 집단·빈 상한초과 집단을 보존한다. 서로 다른 데이터의 raw오류를 평균내지 않는다. 최종 완료는 신규 retrieval/final_receipt.status=complete,new_fit_count9,validation_replay_count9,previous_weight_fit_count18,baseline_reference_count9,comparison_model_count36,seed_count3,held_out_test_evaluated=false 및 모든 원본/선택/종료/결과SHA까지다.
