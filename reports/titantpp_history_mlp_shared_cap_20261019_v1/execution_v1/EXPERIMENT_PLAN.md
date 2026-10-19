# Cap 공유형 이력 MLP18 — 승인된5080 학습과 Validation 비교

**현재 기준선과 승인 범위 — 완료**

- 2026-10-19 사용자는 두 후보×Cap 세 데이터셋×seed42·52·62, 총18개 학습을 승인했다. 5080의 실제 유휴를 확인한 뒤 최대3개씩 병렬로 진행한다. 기존5080/5090 작업과 원본·실패 기록·선택 checkpoint는 변경하지 않는다.
- 준비 계약 `../PREPARED.json`과 소스966개는 그대로 상속한다. 공유형 폭15와 독립 폭 `[15,5,5,5,5,5,5,5]` 모두 보정부9,600개이며 활성 분기 수별 가중치 수도 맞췄다. 같은 함수 공간이나 순수한 공유의 인과효과를 보장하는 대조는 아니다.
- Cap Taxi65·RAF10·Intermittent20, 시간·수량 log1p, 수량 log-MSE, 시간 가중치0, 기존 head·memory·Train 통계·입력6개를 유지한다. 기존 MLP16 Validation9개는 저장 결과만 사용한다. CNN·MLP16을 재학습·재평가하지 않는다.
- 예산300epoch/최소40/patience40, batch128, AdamW lr0.001/weight_decay0.01/clip1과 최초 strict finite fullValidation raw 수량 RMSE 최소 epoch 선택을 유지한다. MAE를 별도로 고른 checkpoint와 섞지 않는다.

**기존 작업을 보존하면서 실행 경로를 연결한다 — 진행 중**

- paper_research의 `codex/contextual-retrieval-pilot-20261007` 브랜치에 별도 실행 worker·병렬 Runtime·배포/회수 manager·비교기만 추가한다. 합성 전용 준비 도구나 기존 봉인 파일은 수정하지 않는다.
- 5080 초기 실제 관측은2026-10-19 22:01:48 KST다. 이전 late-path5080 dispatcher 실제 exit0와 진행 중인 이전 학습 부재, GPU 사용률0%를 확인했다. 기존에 중지된 외부 비교군 GPU 프로세스3개가 약4.5GB를 유지한다. 이 프로세스와 중지된 dispatcher는 그대로 둔다. 이 과거 관측을 배포 시점의 현재 유휴 증명으로 재사용하지 않는다.
- 실제 시작 직전 GPU UUID·Runtime·기존 중지 PID/start_ticks/argv·입력 SHA를 확인하고 새 두 후보 각각3개 동시 합성 CUDA 검증을 통과해야 한다. 새 후보의 메모리 측정과 남은 GPU 메모리로3개 동시 실행을 확인한다. 미확인 GPU 소유나 기존 작업의 재개가 있으면 새 학습을 시작하지 않는다.
- 기존 Python3.12.13/torch2.11.0+cu130/cuDNN91900, GPU `GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6`, 학습4threads를 유지한다. 라이브러리·driver·인증·공용 Runtime은 변경하지 않는다.
- 새 원격 root는 `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_mlp_shared_cap_20261019_v1_execution_v1_5080`다. 입력은 검증된 원본을 새 경로에 복사하며 기존 원격 root를 덮어쓰지 않는다.

**RAF·Taxi·Intermittent를 같은 계약으로 학습한다 — 다음 작업**

- RAF의 공유형3seed → 독립 대조3seed → Taxi의 같은6개 → Intermittent의 같은6개 순서다. 실제 동시 작업 수는 최대3이며 완료한 자리에 다음 고유 작업을 넣는다. 모든 작업은 fresh fit이다.
- 작업별36시간과 상속된 host 절대 기한 `1792888885.911408`을 유지하며 연장하지 않는다. 실패·timeout·기존 attempt/claim이면 원문과 실제 종료 코드를 보존하고 자동 retry/resume하지 않는다. 새 실행기의 소유 child에만 고정 timeout과 정리를 적용한다.
- 실제 PID/argv/start_ticks·GPU에서 진행을 확인한다. 조회 실패를 학습 실패로 바꾸지 않고, 실행 요청만으로 실제 학습 시작이라고 보고하지 않는다.

**학습 종료 후 같은 checkpoint로 Validation을 재현하고 원본을 회수한다 — 학습 종료 대기**

- 18개 학습이 모두 성공한 뒤 선택 checkpoint18개를 별도 fresh process에서 Validation 재현한다. 동일 대상·개별 예측/시간손실·여섯 지표·binary/state SHA가 맞아야 통과한다. Test 입력·추론·평가는 포함하지 않는다.
- scientific child36개와 dispatcher 실제 exit0, 새 소유 PID/GPU 부재, binary·manifest SHA를 확인한 뒤 한 번 원본 archive를 회수한다. supervisor 자신의 OS exit는 직접 관측한 종료 코드와 구분한다.
- 새 로컬 유한 완료 절차는 한 번만 실행한다. 원격 종료를 기다린 뒤 회수·검증·집계를 수행하며, 반복 백본 개선 자동화나 새 학습 루프를 만들지 않는다. 기존5090 복구 완료 절차도 변경하지 않는다.

**데이터셋별3seed 결과와 모든 seed 차이를 비교한다 — 원본 회수 대기**

- 기존 MLP16·공유형·독립 대조군의 전체 raw 수량 RMSE·MAE 평균±표본SD(ddof1)를 데이터셋별로 보고한다. 공유−MLP16/대조−MLP16/공유−대조의 모든 seed 절대·상대차이와 악화를 보존한다.
- `q==cap` 집단 RMSE·MAE와 전체/상한 도달 recorded positive-integer TimeNLL도 같은 checkpoint에서 보고한다. `q>cap`은 빈 집단이므로 N/A다. 시간 head는 학습하지 않았으므로 TimeNLL은 진단값이다.
- 기존 Taxi와 Intermittent42 MLP16은5080, 기존 RAF와 Intermittent52·62 MLP16은5090의 저장 결과다. 새 후보는 모두5080/cuDNN91900이다. 기존 host를 바꿔 표기하거나 재평가하지 않고 cuDNN92000과의 차이를 한계로 기록한다.
- 새 결과는 기존 결과를 대체하거나 합치지 않는다. 데이터셋 간 원단위 오류를 평균내지 않으며, Validation 개선을 Test·독립 일반화·통계유의성·원인 확정으로 확대하지 않는다.

**완료 조건과 남은 순서**

1. 별도 실행 코드의 합성·계약 검사와 독립 검토를 완료하고 커밋·실행 계약을 봉인한다.
2. 5080 fresh 유휴·native Runtime/GPU·3개 동시 합성 검증을 통과한 뒤18개 학습을 시작한다.
3. 학습18·Validation 재현18의 성공 종료와 원본 SHA를 검증하고 회수한다.
4. `THREE_SEED_VALIDATION.json/.md`와 `retrieval/final_receipt.json`의 `status=complete,new_fit_count=18,baseline_reference_count=9,seed_count=3,validation_replay_count=18,held_out_test_evaluated=false` 및 모든 원본·보고서 SHA를 확인한다.

이번18개 학습과 Validation 비교는 승인된 범위다. 결과에 따른 추가 학습·다른 서버 사용·Test는 이 계약에 포함되지 않는다.
