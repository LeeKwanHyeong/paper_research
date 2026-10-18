# MLP16 시간 손실 가중치 대조

## 현재 기준선을 보존한다 — 완료

- 사용자 요청: 기존 외부 비교 실험을 잠시 중단하고, MLP16의 시간·수량 균형 진단에 이어 시간 손실 비중만 바꾸는 학습 대조를 진행한다.
- 공식 기준 데이터는 Train Q3 상한을 적용한 Cap이다. 기존 Cap36의 MLP16·seed42/52/62·세 데이터셋 Validation 결과9개를 시간 가중치1 참조로 재사용한다. 기존 CNN 및 참조 MLP16을 재학습·재평가하지 않는다.
- 직전 진단은 고정 가중치의 국소 Train 기울기다. 세 데이터에 같은 최적 가중치를 알아낸 결과로 해석하지 않는다.
- 외부27의 완료11·정지6·대기10 상태와 원본·실패·선택 checkpoint를 보존한다. 신규 실험은 원 외부 실행기에 신호를 보내거나 재개하지 않는다.

## 시간 항만 양방향으로 바꾼다 — 진행 중

- 목적함수는 `alpha_time * existing_time_NLL + existing_log_quantity_MSE`다. alpha_time은0.1과10이며 기존1을 사이에 둔 공통 두 조건을 세 데이터에 동일하게 적용한다. 합으로 나누거나 상대 수량 가중치로 치환하지 않는다.
- 신규18개 = 두 가중치 × Taxi·RAF·Intermittent × seed42/52/62. 기존 checkpoint warm start 없이 새로 학습하며 공통 초기 tensor/RNG·log1p 입력·모델·메모리·수량/시간 head·손실 형태·optimizer·clip1·배치128·300epoch/최소40/patience40을 유지한다.
- 기존 source941파일을 동일 바이트로 복제하고 새 worker/운영 파일에서만 process-local 가중치 결합을 처리한다. alpha는 계약·job·interface·resume identity·checkpoint·summary·receipt에 결속한다.
- Train/Validation의 joint는 실제 가중 목적함수로 기록하고, 원 시간 NLL과 수량 MSE·가중치1 joint를 별도 기록한다. 서로 다른 alpha의 가중 joint 숫자로 모델 순위를 정하지 않는다.
- 선택은 각 fit의 fullValidation raw수량RMSE 최초 strict finite 최소epoch(동률첫epoch)다. 시간 NLL이나 MAE에 맞춰 재선택하지 않는다. 이 단계는 Validation-only이며 Test 진입점·입력·평가를 추가하지 않는다.

## 소유권과 여유 메모리를 확인한 뒤 새 학습을 시작한다 — 다음 작업

- 대상5080: Taxi6 + Intermittent seed42의2 = 8개. 대상5090: RAF6 + Intermittent seed52·62의4 = 10개. 서버당 신규 학습 최대3병렬을 유지한다.
- 정지한 기존 GPU context의 VRAM을 유지한 채 그 PID·start_ticks·argv·경로·T 상태와 현재 GPU 점유를 확인한다. 알려진 정지 PID 외 미확인 프로세스가 있으면 시작하지 않는다. 새3개에 필요한 실제 합성 메모리 admission을 통과해야 한다.
- 같은 Python3.12.13/torch2.11.0+cu130/cuDNN91900(5080)·92000(5090)/4threads/GPU UUID를 사용한다. 공용 Runtime·driver·인증은 변경하지 않는다.
- 현재 Cap27의 실제 시작+168h 절대 기한을 상속한다: 5080 `1792888885.911408`,5090 `1792888885.815062` Unix seconds. 원 Cap27 기한·정지시간을 수정하거나 새168h를 더하지 않는다. 새 fit의36h 상대 제한과 상속 절대 기한 중 빠른 쪽을 적용한다.
- 로컬 합성·계약 검사와 독립검토 → paper_research 현재 브랜치 로컬 커밋 → 새 격리 경로 설치 → native 합성·메모리·입력 검사 → exclusive 시작 → 실제 worker/GPU·첫 epoch 확인 순서다. Test 평가나 반복개선 자동화는 만들지 않는다.

## 같은 선택 모델의 Validation을 비교한다 — 다음 작업

- 각 서버 전체 학습과 실제 종료·원본 binary 확인 뒤 선택 checkpoint의 native Validation을 재현한다. 원본은 새 archive/manifest로 회수하고 source·input·선택·target·예측·종료 SHA를 검증한다.
- 데이터셋별 alpha0.1/1/10의 전체 및 q==cap RMSE·MAE·recorded positive-integer TimeNLL 평균±표본SD(ddof1), 모든seed의0.1−1·10−1·0.1−10 절대·상대차이를 보존한다. q>cap은 빈 집단/N/A, 단순 기준선 TimeNLL도N/A다. 예측은 자르지 않는다.
- 수량 두 평균이 모두 낮으면 수량 개선으로 기록한다. 시간 NLL 평균까지 나빠지지 않아야 세 지표 Pareto 개선이라고 부른다. 시간 악화나 seed 예외는 tradeoff로 별도 기록하고 숨기지 않는다. 서로 다른 데이터의 raw오류를 평균내지 않는다.
- Intermittent42는5080,52·62는5090이다. 같은 데이터·seed의 참조와 신규는 같은 서버·Runtime이며, 시간·기울기·속도 변화와 통계 유의성·원인 확정을 구분한다. 이번 대조는 백본 기여가 아니라 공동 목적함수 가중치 효과 검증이다.

## Material Passport

- Workflow: academic-research-suite / experiment-agent / run.
- Status: implementation and validation in progress; no new training started when this plan was written.
- Prior evidence: reports/titantpp_time_quantity_balance_20261018_v1/REPORT.md and FINAL_RECEIPT.json.
- Dataset/source reference: search_artifacts/titantpp_mlp_cap36_20261017_v2/execution_contract.json.
- Existing campaign stays paused; no automatic retry/resume or original overwrite.
