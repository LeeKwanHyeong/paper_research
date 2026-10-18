# MLP16 시간 가중치18 후속 작업

## 현재 기준선을 다시 실행하지 않는다 — 완료

배포·qualification·학습시작은 완료했다. `START_RECEIPT.json`과 각 host의 관측 원문/SHA를 먼저 읽는다. 기존 외부27은 계속 정지 상태다. 기존 CNN·기준 MLP16은 재학습·재추론·재평가하지 않는다. 이 문서는 이미 승인된 Validation18 회수·비교만 설명하며 새 Test나 실험 재개 승인이 아니다.

## 현재 진행과 실패 여부를 확인한다 — 진행 중

프로젝트 root에서 관리 도구의 `--mode status --host 5080` 및 `--host 5090`을 각 한 번 호출해 원문을 보존한다. 모듈은 `paper.scripts.manage_titantpp_time_weight_mlp18`이다. 서로 독립인 두 서버 조회는 병렬 가능하다. 원래 정지 PID 신원·T 상태와 새 dispatcher/worker/GPU를 구분한다. 조회 실패는 학습 실패가 아니며 실패·기존 attempt·미완료 회수에서 자동 retry/resume/deploy/kill을 하지 않는다.

## host 원본을 한 번 회수한다 — 다음 작업

해당 host `final_receipt.json status=complete`, 배정 모든 train/validation 실제 worker exit0, dispatcher actual exit0, 선택/입력/원본 SHA, 새 소유 worker/PID 부재, 기존 외부 실험 정지를 모두 검증한 뒤에만 관리 도구 `--mode collect --host <host>`로 회수한다. 이미 유효한 로컬 retrieval receipt가 있으면 다시 회수하지 않는다. 미완료 attempt는 원문을 보존하고 진단한다. supervisor 자신의 OS exit는 직접 관측하지 않았음을 유지한다.

## 저장된 기준선과 비교한다 — 다음 작업

두 host의 회수·manifest 검증이 모두 통과하면 `paper.scripts.compare_titantpp_time_weight_mlp18`을 한 번 실행한다. 새 Validation18과 기존 Validation9개를 검증해3seed 평균·표본SD(ddof1), 전체·상한도달 집단 RMSE/MAE/시간NLL, 모든 seed별 절대·상대차이와 수량/세 지표 개선 판정을 기록한다. 상한 초과 집단은 빈 집단/N/A다. 서로 다른 데이터셋 raw오류를 평균내지 않는다. 단순 기준선 시간NLL은N/A다. Test를 열거나 평가하지 않는다.

완료는 `search_artifacts/titantpp_time_weight_mlp18_20261018_v1/retrieval/final_receipt.json`의 complete/new_fit_count18/validation_replay_count18/baseline_reference_count9/seed_count3/held_out_test_evaluated=false와 모든 원본·manifest·결과SHA 재검증까지다. 현재는 학습 시작 단계이며 완료가 아니다.
