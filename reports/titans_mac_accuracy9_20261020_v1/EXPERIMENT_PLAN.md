# MAC9 원본 데이터 정확도 직접 대조

**현재 기준선과 비교 범위를 고정한다 — 완료**
- 원본 RAF·Taxi·Intermittent의 확정 Shared15 seed42·52·62 및 5090 MAC 비용 측정을 재사용한다. 기존 CNN/MLP16/외부6종/추가2seed/과거 MAC 실패·checkpoint·결과는 보존한다.
- 새 학습은 MAC 이벤트 adapter만 세 데이터×세 seed=9개다. 오늘 비용 측정과 동일한 clip1·의미 보존 최적화를 사용한다. 원논문 native head 재현과는 구분한다.

**같은 학습 조건의 실행 계약을 만든다 — 진행 중**
- 입력은 관측 시간·수량 log1p, 원본 uncapped 데이터다. 공통 수량 head·시간 lognormal 관측 likelihood, logMSE+시간 가중치1, AdamW .001/wd.01/clip1, batch128, 최대300/최소40/patience40을 유지한다.
- 전체 Validation 실수량 RMSE의 최초 strict finite 최소 epoch를 고정하고 MAE도 같은 checkpoint에서 산출한다.
- 부모 source983과 MAC helper2를 그대로 보존하고 새 운영4파일만 추가한다. 역할 연결은 새 worker 안에서만 임시 확장하고 finally 복원한다.

**검증 후 5090에서 최대3개씩 시작한다 — 다음 작업**
- 로컬 합성·계약 검사와 독립 검토 후 커밋·SHA 봉인을 한다. 새5090 전용 root의 유휴·GPU·기존 Python/torch/cuDNN·입력 SHA를 확인하고 실제3개 동시 합성 admission을 통과해야 학습한다.
- RAF42/52/62 → Taxi42/52/62 → Intermittent42/52/62 순서로 배정한다. 사용자는 실행을 승인했으며 재설치·공용 Runtime 변경은 하지 않는다.
- 조건36h/평가2h 및 기존 절대 host 기한1792888885.815062를 연장하지 않는다. GPU 측정 반복3회와 과학 학습3seed는 다르다.

**고정 checkpoint의 Validation과 Test를 검증한다 — 학습 종료 후 다음 작업**
- 전체9 학습 종료 후 nativeValidation9를 동일 대상·여섯지표·원형 예측으로 재현한다. 전부 actualexit0인 gate 뒤 Test9를 실행한다. Test 재선택·자동 retry/resume은 하지 않는다.
- 실제 worker/dispatcher exit·선택 binary/tensor/source/input SHA·소유PID/GPU부재 뒤 host1회 원본 archive/manifest를 회수한다. supervisor 자체 OS exit는 직접 관측 여부를 분리한다.

**Shared15와의3seed 정확도 비교를 기록한다 — 다음 작업**
- 데이터셋별 Validation/Test RMSE·MAE 평균±표본SD(ddof1), 모든 seed의 Shared−MAC 절대·상대차이와 악화, Train고정 큰수량 RMSE/MAE 및 전체/큰수량 TimeNLL을 보존한다.
- 기존 Shared3와 외부 결과는 저장본만 재사용한다. 기존 MAC legacy 시간 손실/joint 선택 결과를 새3seed에 합치지 않는다. 원고는 수정하지 않는다.
