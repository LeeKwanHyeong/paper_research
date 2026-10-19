# 원시간 입력과 로그 시간 입력의 같은 조건 비교

## 현재 손실 비교와 기존 결과를 보존한다 — 현재 기준선

현재 실행 중인 quantity_loss_mlp18 v2와 종료 후 회수·집계 작업은 수정하지 않는다.
Cap 데이터·MLP16·시간 가중치0의 log 손실9개와 신규 실수량 MSE·MAE18개,
총27개 결과가 다음 입력 대조의 기준선이다. 기존 CNN·외부 모델·checkpoint·실패 기록은 그대로 둔다.

## 시간 입력 한 곳만 바꾸는 대조를 고정한다 — 준비 중

사용자는 2026-10-19 세 손실 각각의 대조27개를 승인했다. Cap 세 데이터셋,
seed42/52/62, MLP16, 수량 log1p 입력, 공통 head·메모리·학습 조건을 유지한다.
encoder가 받는 관측 시간 채널만 log1p(Δt)에서 원래 Δt로 바꾼다.
원 관측 tensor를 직접 사용하며 로그의 역변환으로 복원하지 않는다.
추가 정규화·스케일 조절·상한 clipping은 적용하지 않는다.
시간 정답의 단위, 정수시간 관측법칙과 시간 head의 Train time_scale은 바꾸지 않는다.

| 항목 | 고정 조건 |
|---|---|
| 모델·입력 | MLP16, 수량 log1p, 같은 Cap Train/Validation |
| 수량 손실 | log-MSE·실수량 MSE·실수량 MAE 각각9개 |
| 시간 손실 | 가중치0, 시간 전용 head의 gradient·optimizer state·parameter 불변 확인 |
| 학습 | fresh fit, 동일 seed·공통 초기 tensor/RNG, AdamW lr0.001/weight decay0.01 |
| 예산 | batch128, 최대300epoch, 최소40epoch, patience40, gradient clip1 |
| 선택 | 전체 Validation 실수량 RMSE의 최초 strict finite 최소epoch |
| 평가 | 고정 선택 checkpoint의 native Validation 재현, Test0 |

MAE로 학습한 경우에도 선택 지표는 RMSE다. 입력 크기가 달라지며 발생하는
최적화·gradient clipping 차이까지 포함한 입력 변환 비교로 해석한다.
모델 구조의 개선이나 시간 정보의 인과효과로 확대하지 않는다.

## 현재 비교의 전체 완료를 확인한 뒤 새 학습을 시작한다 — 종료 대기

양 서버 부모18개 학습·Validation 재현·원본 회수·최종27개 집계가 모두 끝나고,
최종 receipt와 archive·manifest·원본·선택·결과 SHA가 확인되어야 새 원격 작업에 들어간다.
기존 회수 도구를 중복 실행하지 않고, 이미 연결된 로컬 완료 작업의 성공 증적을 기다린다.
새 일회성 후속 작업은 반복 스케줄러가 아니다.

기존 대조와 같은 host로 배정한다.

| 서버 | 신규 학습 | 동시 실행 상한 |
|---|---|---:|
| 5080 | Taxi3seed×3loss, Intermittent42×3loss:12개 | 3 |
| 5090 | RAF3seed×3loss, Intermittent52·62×3loss:15개 | 3 |

각 서버의 기존 Python·torch·cuDNN·GPU를 유지한다. 기존 외부 Cap27의
정지T 상태를 바꾸거나 신호를 보내지 않는다. 원래 host 절대기한을 상속하며 연장하지 않는다.
새 격리 경로의 source·입력·GPU·Runtime·세 손실 native 합성 검증 뒤 한 번만 시작한다.
실패·timeout·기존 attempt가 있으면 원문을 보존하고 자동 retry/resume하지 않는다.

## 같은 손실·seed의 시간 입력 차이를 비교한다 — 다음 작업

전27개 종료 및 선택 binary·소유 process 종료·원본 SHA를 검증한 뒤,
저장 log-time27개와 신규 raw-time27개를 dataset·loss·seed·host·정답 대상별로 짝짓는다.
데이터셋별 RMSE·MAE의3seed 평균±표본SD(ddof1), 모든 seed의 원시간−로그 시간
절대·상대차이와 악화를 기록한다. 서로 다른 손실의 best끼리 비교해 시간 입력 효과라고 부르지 않는다.
Cap 상한 도달 집단·상한 초과 빈 집단(N/A)과 TimeNLL 진단도 보존한다.
시간 head는 시간 손실로 학습하지 않으므로 TimeNLL을 실제 시간 MAE/RMSE로 설명하지 않는다.
서로 다른 데이터셋의 원단위 오류는 평균내지 않는다.

결과 경로는 comparison_v1/PAIRED_TIME_INPUT_VALIDATION.json/.md,
SUMMARY.csv, SEED_DIFFERENCES.csv다. 완료 receipt는 신규27/저장27/비교54,
Validation 재현27, 실제 과학 child exit0 54개, dispatcher exit0 2개와 Test=false를 결속한다.

## Material Passport

- Type: authorized code experiment, Validation only.
- Status: preparation; no new native training or evaluation at this document creation.
- Source lineage: quantity_loss_mlp18 v2 source952, preserved byte-identically, plus isolated new adapters.
- Boundary: existing parent execution untouched; no Test or manuscript changes.
