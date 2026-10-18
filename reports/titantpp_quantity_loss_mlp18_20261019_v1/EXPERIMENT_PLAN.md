# 시간 가중치0에서 수량 손실을 비교한다

## Material Passport

- Workflow: academic-research-suite / experiment-agent / run.
- User authorization: 2026-10-19, 시간 가중치0·Cap 세 데이터셋·MLP16·3seed와 나머지 조건을 유지한 기존 log 손실 대 실수량 MSE·MAE의 Validation 비교.
- Initial status: 새 실행 코드 구현·검증 중. 실제 학습 시작은 START_RECEIPT와 원격 신원 관측으로 별도 확인한다.
- Scope: 신규18 fit와 선택 checkpoint Validation18, 저장 log 손실/시간 가중치0 결과9개 재사용. Test·기존 CNN 변경·기존 결과 덮어쓰기·외부 Cap27 재개는 포함하지 않는다.

**완료된 기준 모델을 보존한다 — 현재 기준선**

기존 `titantpp_time_zero_mlp9_20261018_v1`의9개 log 손실 모델을 재학습하거나 재추론하지 않고 저장된 Validation 결과만 재사용한다. 부모 최종 receipt SHA는 `6820861b68f9d44fc66b1faf6d614f89b3d8405a11e871fac076af46148d86fb`다. checkpoint·선택 epoch·tensor·입력·source·원본 회수 manifest와 시간 head 미학습 증적을 검증한다. 모델은 동일한 이력 MLP16이며 Titan attention·정적 메모리·공통 수량/시간 head를 그대로 사용한다.

**실수량 단위의 손실 두 개를 추가한다 — 진행 중**

시간·수량 입력의 log1p, 전처리와 모델/head를 유지한다. 기존 수량 head의 log 공간 출력이 u이고 같은 예측의 실수량 복원이 p=expm1(u), Cap 정답이 q일 때:

| 조건 | 학습 수량 손실 | 실행 |
| --- | --- | --- |
| log_mse | mean((u−log1p(q))²) | 시간 가중치0 저장9개 재사용 |
| raw_mse | mean((p−q)²) | 신규9개 |
| raw_mae | mean(abs(p−q)) | 신규9개 |

실수량은 상한 적용 전 데이터가 아니라 **Cap 데이터의 실수량 단위**다. Cap 상한은 Train Q3의 Taxi65·RAF10·Intermittent20이다. 기존 target/mask/reduction과 differentiable 예측 p를 그대로 사용한다. cap으로 나누는 추가 정규화·예측값의 상한 clipping·sqrt·1/2 배율·Huber 평활화는 적용하지 않는다. 수량 가중치1이며 새 학습은 기존 checkpoint warmstart 없이 같은 초기화/RNG를 사용한다.

시간 가중치는0으로 고정한다. 목적함수는 수량 loss tensor 자체이며 시간 loss를0배해 graph에 연결하지 않는다. 시간 관측 입력과 공유 백본은 사용한다. 매 optimizer step의 시간 전용 parameter gradNone·optimizer state 부재·parameter 불변 및 선택 checkpoint 증적을 확인한다. TimeNLL은 기록하되 미학습 시간 head의 진단값으로 표시한다. 시간 head가 Train 통계로 초기화되므로 무작위 예측으로 부르지 않는다.

**나머지 학습·선택 조건을 고정한다 — 다음 작업**

- 각 조건 Taxi·RAF·Intermittent × seed42/52/62, 두 조건18개다.
- batch128·AdamW lr0.001/weight_decay0.01·clip1·최대300/min40/patience40과 기존 데이터 순서·이력 처리·정적 메모리 초기 상태를 유지한다.
- MAE 학습도 전체 Validation 실수량 RMSE의 최초 strict finite 최소epoch(동률 첫epoch)로 선택한다. MAE나 각자 학습 loss로 checkpoint를 다시 고르지 않는다.
- 선택 모델의 native Validation 재현은 기존 대상·여섯지표 및 전체/상한도달·빈 상한초과 집단을 같은 규칙으로 확인한다. 새 Test 입력을 포장하거나 평가하지 않는다.

**두 서버에서 최대3개씩 실행한다 — 다음 작업**

| 서버 | 각 손실의 배정 | 신규 fit 수 |
| --- | --- | ---: |
| 5080 | Taxi42/52/62·Intermittent42 |8|
| 5090 | RAF42/52/62·Intermittent52/62 |10|

각 seed를 기존 log0과 같은 서버에 두며 Python3.12.13·torch2.11.0+cu130·4threads,5080 cuDNN91900/5090 cuDNN92000과 기존 GPU UUID를 유지한다. 외부 Cap27은 정지한 신원과 T 상태를 유지하고 신호를 보내지 않는다. 그 CUDA context를 제외한 미확인 GPU 점유는 새 시작을 막는다. 공용 Runtime·driver·인증은 바꾸지 않고 별도 새 실험 경로만 설치한다.

기존 source947개는 동일바이트로 상속하고 새 실행 파일만 추가한다. 기존 작업의 완전 종료와 원본 회수·모든 로컬 SHA를 확인한 뒤 실제 native GPU/Runtime·입력·두 loss 합성·최대3동시 admission 검증을 통과해야 처음 한 번 시작한다. 기존 root·attempt·claim·실패가 있으면 자동 재시작/retry/resume하지 않는다. 종료한 부모 helper를 반복 실행하지 않는다.

host 절대기한5080=1792888885.911408/5090=1792888885.815062를 상속하고 연장하지 않는다. fit별36h와 절대기한 중 빠른 것을 적용한다. recurring scheduler를 새로 만들지 않는다. 실제 시작 후 PID·argv·start_ticks·lease·GPU와 저장 epoch를 확인한다. 조회 실패와 학습 수치 실패를 구분하고 원문을 보존한다.

**원본을 회수하고 세 손실을 비교한다 — 종료 후 다음 작업**

각 host의 모든 scientific worker·Validation worker 및 dispatcher 실제exit0, 선택 binary SHA와 명시적 소유PID/GPU부재가 확인되면 host별 한 번 원본 archive/manifest를 새 경로에 exclusive 회수한다. 정지한 외부 context는 그대로 유지한다. wrapper 자신의 OS exit는 직접 관측 여부를 구분한다.

실제 시작 신원을 확인한 뒤 로컬 `finish_titantpp_quantity_loss_mlp18` 일회성 완료 작업을 연결한다. host별 한 SSH 연결에서 같은 dispatcher 신원과 종료 파일을 기다리며 학습·GPU를 재조회하거나 새 worker를 시작하지 않는다. 성공한 종료 뒤에만 기존 collector를 host별 한 번 실행하고, 두 원본이 모두 검증되면 순수 집계를 한 번 수행한다. 연결 실패·기한 초과·기존 회수 attempt는 원문을 남기고 자동 재시도하지 않는다. 반복 자동화가 아니며 기존 학습과 정지한 외부 프로세스에는 신호를 보내지 않는다.

저장 log0 9개와 신규 raw18개, 총27개를 데이터셋별로 보고한다. 전체·상한도달 집단의 수량 RMSE/MAE와 recorded positive-integer TimeNLL,3seed 평균±표본SD(ddof1), 모든 seed의 rawMSE−logMSE/rawMAE−logMSE/rawMAE−rawMSE 절대·상대차이·악화·예외를 보존한다. q>cap 집단은 빈 집단/N/A다. 서로 다른 데이터셋의 raw 오류를 평균내지 않는다.

손실 형태뿐 아니라 절대 스케일·gradient clipping·AdamW 경로가 함께 달라진다. 개선이 있어도 log1p 입력의 문제·백본 개선·유일 원인·독립 일반화나 통계 유의성으로 확대하지 않는다. 현재 고정된 조건에서 어느 수량 학습 목적이 Validation 오류를 줄이는지 확인하는 비교다.

완료는 신규18fit/Validation18과 저장9참조의 모든 종료·선택·원본 SHA를 확인하고 `comparison_v1/THREE_LOSS_VALIDATION.json/.md` 및 seed별 표를 생성·독립 검산한 때다. 학습 시작만으로 비교 완료라고 보고하지 않는다.
