# Taxi 이력 CNN 6개 승인 실행

Material Passport: academic-research-suite / experiment-agent / execution / 2026-10-15 v1.

사용자 승인: Taxi 두 후보 × seed42·52·62, 5080 최대3 병렬, 종료 후 기존 CNN과 Validation 비교.

현재 기준선은 보존된 기존 CNN 3개다. 새 학습은 이력 적응 CNN 3seed를 함께 실행한 뒤 용량 대조군 3seed를 함께 실행한다. 모든 fit은 새 초기화이며 checkpoint를 이어 학습하지 않는다. 공통 데이터·입력 SHA·log1p 수량 손실·시간 head·AdamW·batch128·300epoch/min40/patience40을 유지한다. epoch당300 step이며 전체 Validation raw RMSE의 최초 strict finite 최소를 선택한다. Test는 이번 승인에 포함되지 않는다.

준비 계약과 모델 코드는 수정하지 않았다. 별도 실행 overlay는 5080의 원 CNN 학습 qualification에 기록된 torch2.11.0+cu130/cuDNN91900/Python3.12.13/threads4/interop1/TF32 off/strict deterministic을 계승한다. 준비 문서 runtimes의 threads1은 기존 추론 기록이며 이를 학습 설정으로 치환하지 않는다. 원 qualification 원문과 SHA를 함께 보관한다.

학습 전에 입력/소스/운영 SHA, 실제 GPU UUID·Runtime, 원 CNN 공통 초기 tensor, 전체 Train/Validation 대상과 시간 통계를 검증한다. 실제 batch128×length256 합성 입력으로 두 후보의 optimizer·gradient·target차단을 검증한다. 메모리 gate는 실제 측정 peak reserved에 25% 여유와 프로세스당512MiB를 더한 최대3개 합계가 GPU80% 안인지 확인한다. 이 gate를 동시 합성 실행으로 표현하지 않는다. 실제 fit의 각 process allocator 한도30%, 전체GPU 한도80%를 적용한다.

원격 실행은 독립 tmux에서 각 job의 timeout·lease와 실제 종료 코드를 기록한다. 조건36h, host168h를 넘기지 않으며 실패 자동재시도·resume은 없다. 기존 프로세스와 결과는 변경하지 않는다.

종료 조건은 6개 worker 실제 exit0, selected/last checkpoint와 endpoint replay 및 terminal manifest 전체 SHA, dispatcher 실제 exit0, train_complete와 소유 process/GPU 부재다. 이후 원본을 로컬 별도 archive/manifest로 회수한다. 선택 checkpoint의 Validation RMSE·MAE, 큰수량(q>3449) RMSE·MAE, 전체·큰수량 TimeNLL을 기존 CNN과 3seed 평균/표본SD 및 seed별 차이로 비교한다. adaptive−capacity, adaptive−CNN, capacity−CNN을 모두 보고한다. 탐색적 후속 실험이며 원인 확정이나 Test 개선으로 해석하지 않는다.
