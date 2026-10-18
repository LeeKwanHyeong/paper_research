# 시간 가중치0 수량 손실 비교 — 현재 상태

**실제 학습 시작 — 진행 중**

실제 관측: 2026-10-19 08:44:25 KST. 완료0·진행6·대기12다. 이 과거 관측을 새 현재 관측으로 바꾸지 않는다.

| 서버 | 신규 배정 | 실제 진행 | 대기 | 완료 | 저장 |
|---|---:|---:|---:|---:|---|
|5080|8|3|5|0|첫3fit 첫epoch 저장 전|
|5090|10|3|7|0|첫3fit 각4epoch|

각 서버의 첫3개는 seed42 rawMSE·seed42 rawMAE·seed52 rawMSE다. 두 손실을 각각3개씩 동시에 검증한 native 검사는 양 서버에서 각각6개 child의 실제 exit0로 통과했다. GPU·worker·Runtime·입력 일치와 기존 외부 실험의 정지T 상태도 확인했다. 현재 성능 우열은 판정하지 않았다.

**기존 기준선 보존 — 완료**

Cap·MLP16·시간 가중치0·seed42/52/62와 나머지 조건은 유지한다. 기존 log 손실9개는 원본 검증 후 재사용하며, 새 rawMSE9개와 rawMAE9개를 학습한다. 입력 log1p나 수량 head를 바꾸지 않고 Cap 정답의 실수량 단위로 손실을 계산한다. 추가 정규화나 예측값 상한 clipping은 없다. 기존 CNN·모델·checkpoint·결과·실패 기록과 외부 Cap27의 정지 상태는 그대로다. 새 Test는0이다.

**운영 수정과 검증 — 완료**

v1은 사전 GPU 신원 확인 단계에서 종료했으며 실데이터 학습은0이었다. 원문·source950·실패 경로와 합성 child12개의 exit0 기록을 보존했다. v2는 검증된 qualification 프로세스 자신의 GPU context 등록만 수정했고, 과학 명세와 worker는 동일 바이트다. 로컬70tests·국소 독립16tests·실제 배포 archive CPU 검증과 새 native 검증을 통과한 뒤에만 첫 실데이터 학습을 시작했다.

**원본 회수와 최종 Validation 비교 — 종료 대기**

로컬 일회성 완료 작업PID69408은08:45:23 KST에 시작한 뒤 생존을 확인했다. 반복 스케줄러가 아니다. 이미 시작한 dispatcher의 동일 신원·종료 증적을 서버당 한 연결에서 기다린다. 정상 종료 후 원본을 서버별 한 번 회수하고 SHA를 검증한 뒤 양 서버 결과를 집계한다. 실패·timeout·기존 attempt에 자동 재시도하지 않는다.

MAE 학습도 fullValidation 실수량RMSE의 최초 strict finite 최소epoch를 선택한다. 같은 checkpoint의3seed 평균±표본SD와 모든 seed의 절대·상대차이·악화를 보고한다. 시간NLL은 미학습 시간 head의 진단값이며, 손실 스케일·clipping 영향을 포함하는 비교다. 백본 효과나 Test 개선으로 부르지 않는다.

[상세계획](EXPERIMENT_PLAN.md) · [시작증적](START_RECEIPT.json) · [이어가기](PIPELINE_GUIDE.md)
