# 이력 적응 CNN과 용량 대조군의 준비 계약

Material Passport: academic-research-suite / experiment-agent / implementation preparation / 2026-10-15 v1.

## 무엇을 검증하는가

같은 개체·관측 수량·변화 폭을 맞춘 Taxi 표본에서도 최근 관측 하락과 상승 뒤의 signed bias 차이가 일부 seed에서 반복됐다. CNN의 현재·직전·두 번째 이전 관측을 항상 같은 학습된 커널로 조합하는 방식이 이러한 조건 차이에 대응하는 데 제약이 있는지 **새 학습으로 검증할 후보를 준비**한다. 현재 분석은 필터가 원인이라는 증명이 아니며, 새 모델이 개선된 결과도 아니다. seed62의 공통 개체 Train 반전, 낮은 Validation 포함률, RAF의 매칭 부족을 함께 보존한다.

## 비교 구조 — 구현 대상

공통 기반은 GRU가 비활성화된 기존 CNN이다. 첫 attention block의 event Q/K/V 투영에 적용되는 3-tap depthwise CNN만 확장한다. 두 번째 block, attention 참조 범위, prototype 검색/더하기, persistent memory, 입력, 손실과 두 출력 head는 그대로 둔다. RevIN이나 새 관측 통계 입력을 추가하지 않는다.

- 기존 CNN: 고정된 기존 선택 checkpoint와 결과는 그대로 보존한다.
- **이력 적응 CNN:** Q/K/V 각각의 채널에 공유하는 작은 `3→16→1` MLP를 둔다. 입력은 해당 투영의 현재 값, 현재−직전 값, 직전−그 이전 값이다. 관측되지 않은 lag의 차이는 0이다. `u=tanh(MLP)`로 현재 tap 계수를 `1+0.5u`, 사용 가능한 과거 tap 각각을 `1−0.5u/m`로 조절한다(m은 1 또는 2). 원래 학습되는 kernel에 이 계수를 곱한다. 한 관측만 있으면 조절하지 않는다. 사용 가능한 tap 계수의 평균은 1, 범위는 [0.5,1.5]다. 커널 부호와 크기는 원 CNN처럼 학습하므로 이것이 전체 보정 벡터 norm 상한은 아니다.
- **용량 대조군:** 같은 입력·MLP·파라미터 수를 사용하되 tap 계수를 만들지 않고 `0.5*tanh(MLP)`를 채널별 CNN 잔차에 더한다. 최소 두 관측 조건은 동일하다. 고정 가중치의 비선형 로컬 변환이므로 입력에 따라 출력은 변하지만, 명시적으로 입력별 tap 비율을 만드는 경로는 없다.

두 후보의 추가 학습 파라미터는 각각 243개다(Q/K/V별 81개). 모든 추가 파라미터가 실제 연산 경로에 참여한다. dummy 파라미터로 수를 맞추지 않는다. 파라미터 수·입력·수용범위를 맞췄어도 함수군, gradient 및 최적화 난도가 같다는 뜻은 아니다.

초기 MLP 마지막 층은 0, 첫 층은 기존 RNG를 소비하지 않는 fork 안에서 생성한다. 공통 기존 tensor와 RNG는 정확히 보존한다. 초기 두 후보 출력은 기존 CNN과 같다. GRU는 계속 비활성·동결한다. 새 identity와 별도 checkpoint metadata로 기존 CNN과 후보를 혼동하거나 서로 다른 후보를 불러오는 것을 차단한다.

## 고정 조건과 추후 학습 비교 — 승인 필요

첫 비교는 근거가 확보된 **Taxi, seed42·52·62, 두 새 구조 총6 fit**이다. 기존 CNN 3개는 역사적 기준선이다. 머신 판독 계약에 기존 dataset definition·입력 SHA·선택 checkpoint SHA를 그대로 묶는다. Train/Validation 분리, log1p 입력과 수량 손실, recorded positive-integer 시간 head, optimizer, loader, head 초기화, 300 epoch/epoch당300 step, min40/patience40을 유지한다. 선택은 전체 Validation raw 수량 RMSE의 최초 strict finite 최소 epoch다. Test를 선택에 쓰지 않는다.

5080/5090 모두 최대 동시3 정책을 유지한다. 첫 Taxi6은 기존 CNN과 동일한 5080/cuDNN91900에서 seed별 두 후보를 같은 host에 배치하는 계획이다. 5090/cuDNN92000으로 임의 분산해 GPU/Runtime 차이를 구조 효과에 섞지 않는다. 두 서버를 동시에 활용하는 확장이 필요하면 배정과 같은 host의 CNN 대조 필요성을 실행 전에 명시한다. 현재 접속·배포·학습은 없다.

주 비교는 **전체 Validation raw RMSE**이며 adaptive−capacity, adaptive−기존CNN을 각 seed와 3seed 평균·표본SD로 보고한다. 전체 MAE, Train 고정 큰 수량 RMSE/MAE, 전체·큰 수량 TimeNLL도 같은 선택 checkpoint에서 보존한다. 여기에 이번 고정 구간의 signed bias, 하락/상승 RMSE·MAE, tap 계수 분포와 실제 CNN 잔차 norm을 보고한다. 좋은 구간만 선택하지 않는다.

적응 구조가 용량 대조군보다 전체 Validation RMSE 평균에서 좋지 않으면 ‘상황별 tap 조절의 추가 이득’을 주장하지 않는다. 평균만 좋아지고 seed별 악화나 시간/큰 수량 악화가 있으면 함께 보고한다. 학습 채택, Test 실행과 다른 데이터셋 확장은 사용자가 이 준비 결과를 보고 결정한다. 기존 Test를 본 뒤 나온 탐색적 후보이며 독립 일반화 검증으로 부르지 않는다.

## 구현 검증과 완료 경계

합성 입력으로 공통 초기 tensor/RNG·출력, 실제 추가 파라미터 수·gradient 연결, tap 범위·가용 lag 평균, target/미래/패딩 차단·withheld reset, batch 구성 불변, 평가 전후 state 불변, checkpoint 구조 혼동 거부를 확인한다. 기존 과학 파일 SHA를 전후 검증한다. 실제 Train/Validation 추론·native GPU 검증·학습·Test는 이 준비 단계에서 실행하지 않는다. `run_fit`은 승인 전 항상 거부한다.
