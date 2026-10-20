# 원본 RAF·Taxi 공유형 MLP의 3seed Test 우선 확인

**5080 Cap 작업을 중지하고 기존 결과를 보존한다 — 완료**

- 사용자 최신 지시: “우선 먼저 5080에서 실험 중지하고, 3-seed 기준으로 먼저 RAF, Taxi 먼저 Test 결과 산출하자. 그리고 그 결과 먼저 확인하자”.
- 2026-10-20 13:34:07 KST에 Cap 소유 dispatcher·worker·supervisor와 로컬 finisher 종료를 확인했다. 기존 중단 외부 비교군은 그대로 보존한다.
- 중지는 사용자 요청이며 과학적 실패나 정상 학습 완료로 해석하지 않는다. 기존 원본·checkpoint·실패·claim을 삭제하거나 수정하지 않는다.
- 기존 RAF 전용 로컬 대기만 종료해 중복 Test를 막는다. 원본5090 Intermittent와 원본9개 회수 finisher는 계속 유지한다.

**완료된 RAF·Taxi의 선택 원본만 별도로 회수한다 — 진행 중**

- 원본 uncapped Shared15의 RAF seed42/52/62 epoch9/5/19, Taxi seed42/52/62 epoch112/148/127로 고정한다.
- 원본5090의 성공 종료6개만 actual worker exit0·launch·history·summary·binary/tensor·training Validation prediction SHA로 결속하고 새 subset archive에 보존한다. 원본9개 전체 종료를 주장하거나 기다리지 않는다.
- 입력시간·수량 log1p, log-MSE+시간가중치1, head·메모리·선택 기준을 유지한다. 새 학습·5seed 확장은 이번 범위에 없다.

**비워진5080에서 Validation6 재현 후 Test6을 실행한다 — 다음 작업**

- 5080의 기존 Python3.12.13/torch2.11.0+cu130/cuDNN91900/4threads/GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6을 유지한다.
- 원본checkpoint는5090/cuDNN92000에서 학습됐다. 다른 GPU에서 평가한다는 한계를 명시하며 허용오차를 결과에 맞춰 바꾸지 않는다.
- 신규 합성 qualification6 → 개별 prediction/time-loss와 여섯 지표의 Validation6 재현 → 전6개 gate → Test6 순서로 최대3개씩 실행한다. 고정 허용오차는 기존 rtol=atol=1e-5다.
- Test에서 epoch·모델·조건을 다시 선택하지 않는다. 실제 Test를 도구 디버깅에 사용하지 않으며 실패·미완료 attempt를 자동 재시도하지 않는다.

**외부 비교군을 중심으로 데이터셋별 결과를 정리한다 — 다음 작업**

- 저장된 RAF·Taxi의 MLP16 및 공통 head 외부6종 × 각3seed, 총42개 Test를 검증해 재사용한다. 기존 CNN·MLP16·외부 모델은 재학습·재평가하지 않는다.
- 데이터셋별 RMSE·MAE 평균±표본SD(ddof1), 모든 seed의 절대·상대차이와 악화를 보존한다. 외부 지표별 최우수와 비교하고 MLP16은 구조 해석용 대조로 보고한다.
- Train고정 큰수량 RAF q>200/Taxi q>3449의 RMSE·MAE와 전체/큰수량 recorded positive-integer TimeNLL도 기록한다. 단순 수량 기준선의 TimeNLL은 N/A다.
- 실제 평가 child18개(합성6+Validation6+Test6)·pipeline exit0·소유PID/GPU부재와 archive/manifest SHA 확인 후 최종 결과를 보고한다. 기존 중단 CUDA 문맥의 실제 PID 목록은 숨기지 않는다.
- 과거 Test를 참고한 탐색적 후속 비교이며 새 독립 평가·통계적 유의성·인과관계로 확대하지 않는다.

반복 자동화를 만들지 않고 이번 Test 결과 확인까지 수행한다. 원고·공용 Runtime·driver·인증·다른 실험은 변경하지 않는다.
