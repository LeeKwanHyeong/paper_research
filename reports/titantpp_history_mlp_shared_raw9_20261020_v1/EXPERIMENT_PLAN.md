# 원본 데이터 공유형 MLP 9개 후속 실험

**현재 Cap18의 종료·검증·회수를 보존한다 — 진행 중**

- 기존 Intermittent와 유한 완료 절차를 그대로 둔다. 18개 학습·18개 선택 checkpoint Validation·원본 회수와 기존 비교가 모두 완료돼야 원본 후속을 시작한다.
- 원본·실패 기록·기존 MLP16·CNN·외부 비교군·공용 Runtime은 변경하거나 재실행하지 않는다.

**원본 비교 조건을 고정한다 — 완료**

- 사용자 2026-10-20 승인: 공유형 MLP 하나 × RAF·Taxi·Intermittent × seed42·52·62 = 9개. 추가 용량 대조군은 없다.
- 사용자가 선택한 시간 가중치는 1이다. 상한 적용 전 원본 MLP16의 log1p 시간·수량 입력, log 수량 MSE와 기존 시간 손실·head·Train 통계·학습 예산을 따른다.
- 공유형 구조는 Cap 때의 shared15 구현 그대로다. 새 모델을 설계하지 않는다. 원본 입력은 기존 Train/Validation 부분만 복사하며 수량 상한을 적용하지 않는다.
- 큰 수량은 기존 Train 고정 기준 Taxi q>3449, RAF q>200, Intermittent q>187이다. 기존 Q3 Cap값65/10/20과 혼동하지 않는다.

**5080 후속을 유한 순서로 연결한다 — 준비 중**

- 로컬 구현·계약·합성 검사·독립 검토·커밋·입력 봉인을 완료한다.
- 기존 Cap18 완료 증적의 archive/manifest/모든 SHA를 확인한 뒤 fresh GPU/소유·입력·Runtime와 3개 동시 합성 검증을 거쳐 딱 한 번 시작한다.
- RAF3 → Taxi3 → Intermittent3, 각 최대3병렬이다. 이전 host 절대 기한을 연장하지 않는다.
- 기존 Cap finisher를 수정하지 않는 단일 유한 대기 프로세스를 사용한다. 반복 개선 자동화는 만들지 않는다. 실패·미완료attempt에서는 자동재시도하지 않는다.

**고정 checkpoint Validation을 비교한다 — 학습 종료 후**

- 전체 Validation 실수량 RMSE의 최초 strict finite 최소epoch를 선택하고 같은 checkpoint의 MAE를 사용한다.
- 신규9 Validation 재현·실제 child18/dispatcher exit0·소유PID/GPU부재·선택binary SHA 뒤 새 경로에 한 번 회수한다.
- 기존 원본 MLP16 저장9개와 전체/큰수량 RMSE·MAE·TimeNLL의 3seed 평균±표본SD 및 모든 seed 절대·상대차이·악화를 보존한다.
- 기존 CPU/CUDA 및 cuDNN 차이를 표시한다. 새 Test는 이번 승인 범위에 없다. Cap과 원본 raw오류를 같은 문제의 개선량으로 계산하지 않는다.
