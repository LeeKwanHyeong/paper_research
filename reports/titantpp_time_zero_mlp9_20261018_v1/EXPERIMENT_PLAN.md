# MLP16 시간 가중치0 수량 전용 대조

## 기존 결과와 실행 중인 실험을 보존한다 — 완료

사용자는 시간 가중치0으로 수량만 학습하는 대조군을 추가 비교하도록 요청했다. 공식 Cap 데이터(Train Q3: Taxi65/RAF10/Intermittent20), MLP16, log1p 입력, 기존 수량 log-MSE, 모델·메모리·수량/시간 head·초기 tensor/RNG·optimizer·예산·선택을 유지한다. 기존 CNN 및 MLP16 기준선은 변경·재학습·재추론·재평가하지 않는다.

기존 시간 가중치0.1/10의18개는 그대로 완료하도록 둔다. 외부 Cap27은 정지 상태를 유지하며 신호를 보내지 않는다. 기존 source941와 시간18의 source944·입력·계약·선택 checkpoint·실패 기록은 그대로 보존한다.

## 시간 손실의 gradient를 완전히 제외한다 — 진행 중

- 새 목적함수는 기존 `quantity_train_loss` tensor 자체다. `0 * time_loss + quantity_loss`로 만들지 않는다. 시간 전용 parameter의 gradient가 None이고 AdamW의 weight decay·momentum state도 해당 parameter를 바꾸지 않는지 검증한다.
- 시간 입력과 공유 백본은 정상 학습한다. 시간 head는 모델의 공통 초기 상태를 유지하며, 시간 손실을 통해 학습하지 않는다. 원래 시간 head가 Train 통계로 초기화됐을 수 있으므로 무작위 예측으로 단정하지 않는다.
- 시간 NLL 계산은 진단용으로 보존하지만 시간 예측을 학습한0.1/1/10과 같은 목적의 학습 모델로 해석하지 않는다. 시간 head 불변이 TimeNLL 불변을 항상 뜻하지는 않는다. 공유 표현은 변할 수 있다.
- 사용하지 않는 시간 진단에 비유한값이 생기면0으로 덮지 않는다. 원문과 진단 실패를 보존하고 수량 gradient 수치 실패로 단정하지 않는다.
- 신규9개 = Taxi·RAF·Intermittent × seed42/52/62. 기존checkpoint를 불러오지 않는 fresh fit이다. 배치128·AdamW lr0.001/weight_decay0.01·clip1·최대300/min40/patience40을 유지한다.
- 선택은 fullValidation raw수량RMSE의 최초 strict finite 최소epoch이며, MAE나 시간NLL로 다시 선택하지 않는다. 새 Test는 포함하지 않는다.

## 기존 서버 큐 종료 후 같은 환경에서 실행한다 — 다음 작업

5080에 Taxi3와 Intermittent42 한 개(총4),5090에 RAF3와 Intermittent52·62 두 개(총5)를 배정한다. 각 서버의 기존 시간18 train/Validation·dispatcher 실제exit0와 소유PID/GPU부재를 확인한 뒤에만 새 native 검사·학습으로 진입한다. 기존 정지 외부 실험의 GPU context는 허용하되 신원·T 상태를 보존한다. 실행 중인 시간18은 미확인 GPU를 거부하므로 그 작업과 새 CUDA 작업을 겹치지 않는다.

새 경로의 유한 CPU 대기 큐가 기존 작업의 완료를 기다리고 native 검증 뒤 최대3병렬 학습을 시작한다. 반복 자동화·재시작·자동 retry/resume은 만들지 않는다. 사용자 서버를 공용 서비스로 재배포하지 않고 별도 실험 경로만 사용한다.

동일 Python3.12.13/torch2.11.0+cu130/GPU UUID/4threads,5080 cuDNN91900·5090 cuDNN92000을 유지한다. 기존 host 절대기한5080=1792888885.911408/5090=1792888885.815062와 fit36h 중 빠른 기한을 적용하고 연장하지 않는다. 대기 중에도 절대기한을 적용한다.

## 수량 전용 대조를 검증한다 — 다음 작업

신규 worker·운영 파일만 구현하고, 수량-only oracle과의 공유 gradient·여러 AdamW step 상태 일치, 시간 전용 head의 gradient/parameter/optimizer state 불변, 공통 초기화·checkpoint결속·선택 모델 Validation 재현을 합성 입력으로 검증한다. 운영은 부모 완료/실패·대기·GPU 진입·최대3·중복 시작 방지·종료/회수 원문을 검사한다. 신규 파일을 paper_research의 codex/contextual-retrieval-pilot-20261007 브랜치에 커밋한 뒤 실제 native 검사와 큐 시작으로 이어간다.

## 네 가중치의 동일 조건 Validation을 비교한다 — 다음 작업

새0의9개, 기존1 저장9개, 시간0.1/10 신규18개의 종료·원본·선택·Validation 재현을 검증한 뒤 총36모델을 비교한다. 전체·상한도달 집단의 수량 RMSE/MAE와 시간NLL, 데이터셋별3seed 평균±표본SD(ddof1), seed별 절대/상대차이·악화·예외를 보존한다. q>cap은 빈 집단/N/A다.

0의 시간NLL에는 시간손실 미학습 head라는 표시를 붙이며, 공동 모델의 수량·시간 성능을 동시에 달성한 후보로 취급하지 않는다. 주질문은 시간 손실 없이 수량 최적화할 때 수량 오류가 달라지는지다. 효과가 있어도 공동 최적화·clip 변화가 포함되므로 gradient 충돌의 원인 확정이나 백본 기여로 확대하지 않는다. 서로 다른 데이터셋 raw오류를 평균내지 않는다.

원본 회수는 각 host의 새 큐가 종료한 뒤 진행한다. 새 작업 중에는 기존 시간18 collector가 새 GPU worker를 미확인 점유로 거부하므로, 기존 시간18의 미회수 원본도 새 큐 종료 후 회수·검증한다. 기존18에 이미 유효한 로컬 회수본이 있으면 재회수하지 않는다.

## Material Passport

- Workflow: academic-research-suite / experiment-agent / run.
- Status at plan creation: implementation in progress; no new alpha0 training started.
- Objective: quantity-only training control; not a backbone modification or a universal optimal-weight search.
- Evaluation: Validation only, existing checkpoints and results immutable; no new Test.
