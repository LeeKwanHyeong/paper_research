# 폭8·12 Test 평가와 Intermittent 소유권 이관

**Taxi·RAF의 동결 checkpoint로 기존 Test를 평가한다 — 완료**

- 대상: Taxi·RAF × 폭8/12 × seed42/52/62의12조건. 학습 당시 최초 최소 전체 Validation raw 수량 RMSE epoch를 그대로 사용했습니다. 재학습·checkpoint 재선택은 없습니다.
- 5080에서 전체 Validation12의 동일 population/target/truth/history·전체/큰 수량 RMSE·MAE·Time NLL을 먼저 대조했습니다. gate12 통과 이후에만 Test12를 실행했습니다. 최종24 population 평가는96.51초에 완료됐습니다.
- [Test 전체 비교표](../titantpp_width8_12_test_20261004_v1/results_taxi_raf/TEST_TABLES.md), [큰 수량 비교표](../titantpp_width8_12_test_20261004_v1/results_taxi_raf/TEST_TAIL_TABLES.md), [seed별 원본 집계](../titantpp_width8_12_test_20261004_v1/results_taxi_raf/new_width_metrics_per_seed.csv)를 저장했습니다. 기존 폭4/16·비교군은 원본CSV의 split=test와 SHA를 확인해 재사용했습니다. 원시 예측은 저장·회수하지 않았습니다.
- 기존 Test 분할 재평가입니다. 새 독립 미접근 평가로 표현하지 않으며 Test 결과로 폭이나 checkpoint를 다시 선택하지 않습니다. 폭16 개발 기준선과 CNN/GRU 대조는 유지합니다.
- 첫 두 배치의 데이터 경로·프로젝트 인식 marker 오류는 성능 계산 전에 발생했습니다. 각각의 계약·배치와 원인을 보존했고, 세 번째 배치는 경로/SHA·동결114소스 import 사전 검사 후 통과했습니다. 과학 소스와 모델·손실은 바꾸지 않았습니다.

**5090의 미시작 seed62 두 조건을 idle5080에 연결한다 — 완료**

- 대상: Intermittent seed62 폭8·12 두 조건. 원래5090의 실행 중 seed42와 대기seed52 네 조건은 유지합니다. 원래5080 Taxi/RAF12 완료 + 새5080두 + 5090네 =18조건이며 새 과학 조건을 추가하지 않았습니다.
- [운영 계약과 절차](../../search_artifacts/titantpp_history_capacity_handoff_20261004_v1/README.md), native5080 qualification,5090 두 O_EXCL reservation claim과 원본SHA,5080 training permit을 연결했습니다. 사용자 승인 원문은 [approval.json](approval.json)에 있습니다.
- 2026-10-04 20:16:29KST 실제 관측에서5080 seed62 폭8 supervisor/worker/GPU PID와 저장4/best4 checkpoint가 일치했습니다. 최초 저장 checkpoint 이후 학습 진행을 확인했습니다. 폭12는 같은서버의 다음 조건입니다. 5090은20:04:26KST에 seed42 폭12 저장47/best24, 완료1·진행1·대기2로 확인됐습니다.
- 새5080 전용root: `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_capacity_handoff_20261004_v1_5080`. 원래 부모 계약/114source/자료SHA/학습/손실/선택/조건36h·전체168h 마감을 유지합니다. GPU 출처를 표시하고 이종GPU 속도로 효율성을 주장하지 않습니다.
- 원래5090 dispatcher는 소유한 네 조건 완료 후 첫seed62 예약claim에서 의도한 FileExistsError로 종료합니다. 정확한 error.type/active_job/claimpath·네 terminal·소유프로세스 부재를 모두 확인하기 전에는 행정 경계라고 선언하지 않습니다. 원래 failure/status는 보존합니다.

**Intermittent 여섯 조건의 Test를 완료 뒤 연결한다 — 외부 작업 대기**

- 새 통합 observer는 서버별 소유 조건과 terminal을 검증합니다. 여섯 조건 전체 완료 후 원본SHA회수 → 동일 선택checkpoint의 full Validation6 gate → 기존 Test6 순서로 처리합니다. 미완료 중에는 Test를 실행하지 않습니다.
- [시간별 모니터](../titantpp_three_gpu_hourly_monitor_20261004_v1/SCHEDULER.md)의 별도 후속 실행기를 사용합니다. 중복 실행과 자동 retry를 금지하고 평가 시간·메모리 상한을 별도 계약에 묶습니다.
- Instacart와 A100 CNN/GRU는 현재 Test 배치에 추가하지 않습니다. 원본 binary SHA 회수·nativeGPU Validation 확인과 binary CPU 재추론 감사는 별개입니다.
- 이관 Runtime·관측기·후속 평가 드라이버·평가 파이프라인의 로컬 테스트59개가 통과했습니다. 실제 미완료 상태의 후속 드라이버 호출은 원격 호출0으로 대기했으며, 학습 완료 전에 Test를 시작하지 않았습니다.
