# CNN prototype 메모리 반영량 진단

## 승인과 목적

2026-10-15 사용자는 "이거 먼저 진행하자. 그리고 학습은 직전 실험과 동일하게 5080/5090에서 세개씩 병렬로 돌릴 수 있게 학습하는 방식으로"라고 승인했다.

현재 이력과 맞지 않는 prototype을 강하게 더하는 것이 CNN 오류를 키우는지 먼저 확인한다. 메모리가 원인이라고 가정하지 않는다. 기존 Test를 본 뒤 정한 탐색적 후속이며 Test 일반화 개선의 독립 증거로 부르지 않는다.

**현재 선택 CNN 9개를 고정한다 — 완료**
- Taxi·RAF·Intermittent × seed42/52/62. source131 및 각 학습의 전체 Validation raw 수량 RMSE 최초 strict finite 최소 checkpoint를 유지한다.
- 원본 binary·tensor·metadata·선택 이력·Train/Validation 입력을 별도 진단 bundle에 SHA로 결속했다. 기존 원본·실패 기록·CNN/FULL/CAP 결과와 원고는 보존한다.
- 계약: `search_artifacts/titantpp_cnn_memory_sensitivity_20261015_v1/contract.json`, raw SHA `a1cd61a7a38d6174d8511c0316cf82ea58fbf3fcd5ca08ac15dc0a0fc2121a98`.

**고정 모델에서 메모리 반영량만 바꾼다 — 다음 작업**
- alpha=1, 0.5, 0.25, 0; top-k 선택·평균·CNN·attention·head·입력·손실·선택 epoch를 바꾸지 않는다. 총 9모델 × 2 split × 4 alpha = 72조건이다.
- 실제 `retrieve`가 반환한 residual을 사용하며 alpha=1은 원래 출력을 그대로 반환한다. 전체 모델·원본 checkpoint SHA를 전후 확인한다.
- Train과 Validation만 사용한다. 각 서버의 모든 Validation 원래 성능 재현이 통과한 뒤 Train 진단에 진입한다. 실패 시 신규 worker를 넣지 않고 원문·실제 종료 코드를 보존한다.
- 원래 native 배정: 5080은 Taxi3·RAF3·Inter62의7개, 5090은 Inter42/52의2개. 두 서버 모두 최대 동시3개다. 현재 5090 진단에는 두 모델만 있으므로 두 개가 동시 실행된다.
- 5080/cuDNN91900과5090/cuDNN92000을 유지하고 시간은 서버별로 구분한다. 새 학습에서는 두 서버 각각 최대3개 병렬로 구성하며 현재 진단을 새 학습으로 세지 않는다.

**오류 변화와 관측 가능한 조건을 연결한다 — 다음 작업**
- 전체 RMSE·MAE, Train고정 큰 수량 RMSE·MAE, 전체·큰 수량 recorded positive-integer TimeNLL의 3seed 평균·표본SD와 seed별 차이를 모두 기록한다.
- 개별 행에 과거 수량 평균·변동·최근 변화, encoder와 memory norm·cosine, top-k 유사도·직전 검색과의 겹침을 기록한다. 미래 target은 오류 계산에만 쓰며 gate 입력으로 쓰지 않는다.
- 특징 구간은 Train 사분위로 정하고 Validation에 적용한다. 과거 관측이 부족하거나 norm=0으로 정의되지 않는 특징은 별도로 표시한다.
- 상수 감쇠 근거: 어떤 alpha<1이 데이터셋 평균 Validation RMSE를 1% 이상 줄이고 3seed 중2개 이상에서 개선한다. 모든 데이터셋과 여섯 지표의 악화도 함께 보고한다.
- 상황별 gate 근거: 위 조건과 함께 같은 alpha·특징의 특정 구간에서는 감쇠가 도움, 다른 구간에서는 손해라는 양상이 Validation2seed 이상에서 반복되는지 본다. 이는 탐색적 패턴이며 통계적 유의성 판정이 아니다.

**근거에 맞춰 학습 대조를 확정한다 — 진단 결과 대기**
- 근거가 없으면 gate 학습을 진행하지 않는다. 상수 감쇠만 유용하면 복잡한 gate가 필요한지 별도로 판단한다.
- 근거가 있으면 기존 CNN·고정 감쇠·상황별 gate를 같은 손실/head/예산/선택 기준으로 비교한다. 구체적인 배율·gate 입력·fit 수는 진단 후 Test를 사용하지 않고 고정한다.
- 실제 메모리 보정 크기를 기록해 학습 중 prototype 크기 증가가 감쇠를 상쇄하는지 확인한다. 이를 방지했다고 미리 주장하지 않는다.
- 새 학습 스케줄러는 **5080·5090 각각 최대3개**를 유지한다. 기존 결과를 새 학습으로 합치거나 대체하지 않는다.

서로 독립인 제어 모듈·실행기·합성 검증은 병렬로 준비했다. 계약, native 진단, 집계, 후속 학습 결정은 의존관계에 따라 순서대로 진행한다.
