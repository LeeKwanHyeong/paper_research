# PHEN 고정 위치 인코딩 RAF·Taxi seed 42 시험

## Material Passport

- 요청: 5080에서 RAF·Taxi seed 42의 고정 위치 인코딩 PHEN을 시험하고 Train/Validation learning curve를 산출한다.
- 상태: 준비 중. 이 문서는 학습 완료 또는 개선을 뜻하지 않는다.
- 범위: 새로운 PHEN 학습 2회, Train·Validation만 평가. Test는 사용하지 않는다.
- 보존: 기존 PHEN·외부 비교군·실패 기록·checkpoint, 5080 중지 작업, 5090 Instacart를 변경하지 않는다. MAC은 보류한다.
- 과학적 질문: 다른 조건을 유지한 채 학습형 위치 테이블을 고정 정현파 인코딩으로 바꾸면 학습 후 수량 일반화의 변화가 달라지는가?

## 실험 조건을 고정한다 — 진행 중

- 원 seed 42 PHEN의 실행 소스 `b1455baec1c084e422ce044dec7cd454acd95631`과 동결된 RAF·Taxi 입력을 기준선으로 삼는다.
- 새 adapter에서 기존 모델을 같은 난수 순서로 구성한 뒤 위치 parameter만 고정 sin/cos buffer로 바꾼다. 다른 초기 가중치의 동일성을 검증한다.
- base 10000의 sin/cos를 위치 0부터 사용하며, 임의의 미학습 벡터를 동결하는 방법은 사용하지 않는다.
- 이력 범위, 입력 길이, 인코더·이력 보정·Prototype·head·loss를 유지한다. AdamW lr 0.001, weight decay 0.01, clip 1, batch 128 및 기존 epoch·선택·조기종료 동작을 그대로 사용한다.
- 최초 strict Validation raw-quantity RMSE 최저 epoch를 선택한다. 결과를 보고 seed나 설정을 바꾸지 않는다.

## 검증 후 5080에 새 실험을 실행한다 — 다음 작업

- 두 학습은 RAF, Taxi 순서로 단독 실행한다. 기존 중지 작업은 메모리 점유 상태까지 보존한다.
- 로컬 단위·계약 검증과 5080 synthetic/capacity 검증을 먼저 수행한다. 실행 소스와 입력 SHA를 봉인하고 신규 경로에서만 실행한다.
- 기존 시도나 부분 산출물이 있으면 덮어쓰지 않는다. 학습 실패는 보존하고 자동 재학습하지 않는다.

## 같은 epoch 종료 모델의 Train·Validation 곡선을 산출한다 — 다음 작업

- 학습 중 누적한 online Train 손실을 원래 형식으로 보존한다.
- 각 epoch 종료 모델을 eval 모드로 고정하여 전체 Train·Validation을 평가한다. 수량 RMSE·MAE·log1p MSE, 시간 NLL, 결합 손실과 대상 수를 기록한다.
- 추가 Train 평가는 별도 비셔플 loader를 사용한다. 전후 model mode·state와 Python/NumPy/Torch/CUDA 난수 상태를 보존하여 학습 과정 및 checkpoint 선택에 개입하지 않는다.
- 같은 가중치의 Train·Validation 곡선과 online Train 곡선을 구분한다. 추가 평가 시간을 포함한 실제 wall time은 순수 epoch 학습 속도라고 부르지 않는다.
- 미완료 곡선은 partial로 표시하고, 학습 완료·선택 checkpoint 재현·실제 child 종료 확인 후 최종 비교를 작성한다.

## 해석 범위

- 한 seed의 탐색적 대조이다. 개선 여부를 다른 seed나 Intermittent 전체로 일반화하지 않는다.
- 고정 인코딩은 위치별 미학습 parameter 문제를 없애지만 긴 이력 일반화 개선을 보장하지 않는다.
- 정현파 인코딩은 표현값과 크기 및 학습 가능성 모두를 바꾼다. 개선이 나타나더라도 미노출 행 하나만의 인과 효과라고 주장하지 않는다.
- 기존 결과는 보존하며 새 모델의 정식 논문 결과로 자동 대체하지 않는다.
