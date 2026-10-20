# 원본 공유형 MLP 9개: 현재 실행 서버 5090

**5090 전환과 실제 학습 시작 — 완료**

- 사용자 지시에 따라 5080 후속 대기만 해제했고, 현재 5080 Cap 학습은 유지했다. 기존 대기·실패·계약 원문은 보존했다.
- 실제 2026-10-20 10:02:52 KST 관측에서 5090 RAF seed42/52/62의 worker PID1611342/1611343/1611344, 시작 tick, 정확한 명령과 GPU 일치를 확인했다. 완료0·진행3·대기6·실패0·미확정0이다.
- 실제 서버의 GPU·Runtime·원본 입력 검증과 동시 합성 worker 3개의 실제 exit0 확인 뒤 학습이 시작됐다. Python3.12.13/torch2.11+cu130/cuDNN92000/GPU-c9adc246-ef66-7906-bc07-6e152fa9952f이다.
- qualification에 상속된 필드 이름 predecessor_5080_gate의 내용은 5090의 새 유휴 검사이며, 5080 Cap 완료 증거가 아니다. 5080 완료를 기다리지 않는다.

**원본 세 데이터셋 학습 — 진행 중**

- 공유형 하나 × RAF·Taxi·Intermittent × seed42/52/62 = 9개이며, 5090에서 최대 3개씩 병렬로 진행한다. RAF 3개 다음 Taxi 3개와 Intermittent 3개를 이어간다.
- 상한을 적용하지 않은 원본, 시간·수량 log1p, 시간 가중치1 + log 수량 MSE, 공통 head·메모리·학습 예산·선택 기준을 유지한다. 기존 CNN·MLP16·외부 모델은 재학습하거나 재추론하지 않는다.

**선택 checkpoint 검증·회수·비교 — 다음 작업**

- 학습 9개 종료 후, 전체 Validation raw 수량 RMSE의 최초 strict finite 최소 epoch checkpoint 9개를 고정한다. 같은 서버에서 Validation 9개를 재현한 뒤 원본을 한 번 회수한다.
- 기존 원본 MLP16 저장 결과 9개와 전체·큰수량 RMSE·MAE·TimeNLL의 3seed 평균±표본SD, seed별 절대·상대 차이와 악화를 보고한다.
- 종료까지 이어가는 유한 완료 프로세스 PID71592를 등록했다. Test·추가 모델·반복 개선 자동화·실패 자동 재시도는 없다.
- 최신 상태는 상위 LATEST_EXECUTION.json이 가리키는 실행 경로로 확인한다. 이전 CURRENT_EXECUTION.json/queue_v2는 취소된 5080 후속 대기의 과거 기록이다.
