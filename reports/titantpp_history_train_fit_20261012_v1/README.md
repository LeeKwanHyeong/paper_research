# 같은 선택 모델의 Train·Validation·Test 적합 상태 확인

**고정 모델의 적합 상태를 확인한다 — 완료**

- **2026-10-12 11:47:48 KST**에 Train27과 pipeline 실제 OS exit0을 확인했다. 기존 같은 선택 checkpoint의 Validation27·Test27을 재사용해81개 split entry를 연결했고, 원본559개·archive/manifest·두 로컬 사본 SHA를 검증했다. [실제 완료 상태](RUN_STATUS_COMPLETE.json), [전체·큰 수량 여섯 지표와 모든 seed/대조](../../search_artifacts/titantpp_history_train_fit_20261012_v1/analysis/TRAIN_FIT_REPORT.md).

- 사용자의 2026-10-12 “이거 먼저 진행하자”는 선택 history27 checkpoint의 Train·Validation·Test raw 수량 RMSE·MAE와 큰 수량 오류 비교, 기존 계측의 수량/시간 분리 분석 및 최소 후속 계약 정리를 승인한다. 신규 학습·경로 개입·새 자료·checkpoint 재선택·원고 편집은 수행하지 않는다.
- 기존 v2 Validation27·Test27은 같은27 checkpoint의 완료 원본을 SHA로 재사용한다. Train selected-checkpoint raw 지표는 기존 원본에 없으므로 Train27만 추가 추론한다. optimizer가 갱신되던 중의 학습 log loss를 선택 모델의 적합 지표로 대신하지 않는다.
- 계약 canonical은 `76248e2174d7bd651d821f649ad860030da985d59da88a8ce50725b476f55ccf`, 과학 source129 closure는 `c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a`, frozen revision은 `57da14a2a7a5687e1bdef7d3164f3e8c0051da59`다. 같은 model.eval·inference_mode·B128·shuffle0·drop_last0·원래 lookback/max sequence cap을 적용한다.
- Train의 실제 다음 사건 대상은 Taxi38,393·Intermittent393,824·RAF25,779다. 첫 사건은 예측 대상에서 제외하며 원래 Train target identity/quantity digest와 정확히 일치해야 한다. 큰 수량 경계는 기존 Train에서 정한 Taxi3449·Intermittent187·RAF200을 그대로 쓴다.

**현재 원본과 5080의 실행 조건을 확인한다 — 완료**

- 기존 완료54의 source·selected checkpoint·원본746·archive/manifest 및 로컬 사본 SHA와 새 입력359를 확인했다. 참조 v2 canonical은 `70e8e22dbe85aa93cb5dc555b06303511b0e5e7f511771b455b4b243ddb3ad1d`이며 Validation/Test 결과를 새 관측시각의 신규 평가로 바꾸지 않는다.
- [읽기 전용 건강 확인](INITIAL_5080_HEALTH.json)에서 GPU UUID `GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6`, GPU compute PID0·reboot-required0·패키지 업데이트 프로세스0·기존 ai_env 및 부모 pipeline 실제 exit0를 확인했다. 별도 5090·Pod·Runtime·인증은 변경하지 않는다.
- 새 remote root는 `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_train_fit_20261012_v1_5080`다. 이전 root는 읽기만 하며 새 isolated root에 승인된 운영 코드·입력 복사본·결과를 별도로 저장한다.

**같은 환경에서 Train27을 추론하고 기존 두 split과 연결한다 — 완료**

- synthetic3route의 실제 GPU qualification과 hook 출력 불변성을 먼저 확인한다. 같은 dataset/seed의 세 arm을 최대3개 동시에 평가하고 자동 retry/resume0, worker3시간·전체12시간 상한을 유지한다. 신규 Validation/Test 추론은0이다.
- Train 전체/큰 수량 여섯 지표와 bias·보정/시간 head의 관측을 기록한다. state SHA 불변·finite·Train population 일치·metric/telemetry 자체 재현을 확인한 경우에만 결과를 사용한다. Train에는 기존 선택 모델 평가 reference가 없다는 사실을 명시한다.
- 실제27 worker·pipeline OS exit0 및 소유 PID 부재를 확인한 뒤 immutable 원본을 한 번 회수하고 archive·manifest·로컬 SHA를 검증한다. 기존 v2 원본54와 checkpoint 단위로 정확히 연결한다.
- 실제 native3route와 Train27은 모두 exit0이었다. 전체/큰 수량의 metric과 telemetry 자체 계산·원래 Train population·checkpoint state 불변을 확인했다. Validation/Test의 원관측시각과 값은 유지한다. Train에는 과거 selected-checkpoint 평가 reference가 없어 기존 Train 지표를 재현했다고 주장하지 않는다.
- 원본 회수 전 검사는 launcher argv의 Python 실행 경로와 worker sys.argv의 형식 차이로 멈췄다. 당시27개의 PID/PPID·명령 인자·exit0는 정확히 일치했고 claim/archive는 아직 생성되지 않았다. [검사 실패 원문](RETRIEVAL_CHECK_FAILURE.json)·[읽기 전용 진단](RETRIEVAL_DIAGNOSIS_OBSERVATION.json)을 보존하고, 독립 QA·회귀4개 뒤 별도의 행정 검증 코드로 회수만 완료했다. 원래 봉인 finalizer·worker·scientific source는 불변이고 추론 재실행은0이다. [수정 판단](RETRIEVAL_CORRECTION_DECISION.json).

**수량과 시간의 원인 후보를 분리하고 다음 실험을 최소화한다 — 완료 / 추가 실행은 승인 필요**

- [기존 증거 분석](EXISTING_EVIDENCE_DIAGNOSIS.md)에서 이미 계산할 수 있는 관측을 먼저 정리한다. Train부터의 큰 수량 부족과 평가 구간 추가 악화를 구분하되 split의 대상 구성 차이를 순수 과적합으로 단정하지 않는다.
- 수량 objective·raw selector와 시간 scale/location 취약성을 서로 다른 질문으로 다룬다. 후보 하나를 검증할 추가 추론·학습은 결과를 보고 최소 계약만 정하며 이번 평가 승인으로 자동 실행하지 않는다.
- 구현·분석·독립 QA는 서로 다른 파일로 병렬 진행하고, 계약 봉인·GPU 실행·원본 회수·공통 작업 순서는 직렬 진행한다. 원고는 현재 상태를 유지한다.
- RAF full의 큰 수량 Train RMSE는 세 seed에서458.95~468.43, 평균 오차는−350.86~−364.10이다. Train tail205건에서도 직전 수량426.56·과거 평균412.18보다 tail RMSE가 나쁘지만 전체 Train RMSE는 두 기준선보다 좋다. 큰 수량 point 재현 부족은 학습 대상에도 있으며 용량/최적화 underfitting의 원인 증명은 아니다.
- Taxi52의 큰 수량 RMSE는 Train413.31·Validation370.76·Test1017.49다. Train에서 두 대조와 단순 기준선보다 좋아 Test 증가가 두드러진다. Intermittent42의 전체 수량 RMSE는1.4146→1.6667→2.9429, TimeNLL은0.2193→0.6187→62.5721이다. 이 세 경우를 같은 실패 원인으로 묶지 않는다.
- [최소 후속 계약 후보](MINIMUM_FOLLOWUP_CONTRACT.md)는 RAF9×Train/Validation의 log 수량 오차 계측18pass, 그 다음 Intermittent9×Train/Validation의 시간 위치/scale 교차 score18pass로 좁힌다. 두 후보 모두 새 학습0·새 Test 추론0이며 이번 실행은0이다. 추가 학습6/18fresh와 기존 구성18/RAF12·검색은 필요성 판단과 별도 승인 전 자동 실행하지 않는다.
