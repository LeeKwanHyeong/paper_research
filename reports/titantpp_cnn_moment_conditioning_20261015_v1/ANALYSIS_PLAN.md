# 관측 μ·σ CNN 비교 계획

상태: 준비 단계. 실제 학습·Train/Validation 추론·Test 미실행.

## 주 비교

μ·σ 조건 CNN−같은 μ·σ 용량 대조군의 전체 Validation raw 수량 RMSE를 먼저 비교한다. 이어 조건 CNN−블록 RevIN, 블록 RevIN−기존 CNN, 조건 CNN−기존 CNN을 보고한다. 입력 정규화 효과, 통계 접근 효과, 구조 조절 효과를 혼합하지 않는다. 파라미터 수가 같아도 함수군·최적화·복원 scale의 작용이 같지는 않다.

조건 CNN이 용량 대조군과 블록 RevIN 모두보다 3seed 평균 RMSE가 좋지 않으면 상황별 tap 조절의 추가 이득을 주장하지 않는다. 같은 기존 CNN보다 나쁜 경우도 별도로 명시한다. 표본 3seed의 개선을 통계 유의성이나 보편적 일반화로 표현하지 않는다.

## 고정 집계

데이터셋별로 원래 단위의 전체 RMSE·MAE를 중심으로 3seed 평균±표본SD(ddof1), seed별 절대·상대 차이를 기록한다. 상대차이의 분모가0이면 N/A다. Train에서 고정한 큰 수량 RMSE·MAE, 전체·큰 수량 recorded positive-integer TimeNLL도 같은 Validation 선택 checkpoint에서 모두 보고한다. 큰 수량 대상이 없으면 N/A와 수량0을 기록한다. 서로 다른 데이터셋의 raw 오류를 평균내지 않는다.

선택은 전체 fullValidation raw 수량 RMSE 최초 strict finite 최소 epoch이며 동률이면 첫 epoch다. MAE나 구간 결과로 checkpoint를 재선택하지 않는다. 새 Test는 이번 범위에 없다.

## 작동 확인

고정 구간에서 관측 μ·σ, 가용 이력 길이, tap 비중, CNN 잔차/기본 표현 norm 비율, 상승·하락 조건의 signed bias·RMSE·MAE를 기록한다. μ·σ 구간은 Train 기준으로만 고정한다. 같은 checkpoint에서 조절 계수를1로 되돌리는 진단은 추가 학습 없는 기전 점검으로만 해석하며 학습된 일반 RevIN 대조군을 대체하지 않는다.

블록 RevIN의 잠재 sigma 상쇄와 조건 네트워크가 사용하는 관측 수량 sigma를 구분한다. 수량 sigma 조건 효과가0으로 학습되거나 tap이 항상 경계에 붙는 경우도 보고한다. 유리한 seed·구간만 고르지 않는다.

## 실행 전 남은 항목

1. 사용자가 준비 결과를 보고 학습 대상·규모를 결정한다.
2. 승인 범위에서만 baseline binary·실제 입력 SHA, source closure, GPU UUID·Python·torch·cuDNN과 native 합성검증을 결속한다.
3. source/계약 수정 없이 지정된 fresh fit과 최대3 병렬 dispatcher를 준비한 뒤 승인된 학습만 시작한다.
4. actual worker/dispatcher 종료·terminal binary SHA·원본 회수를 확인한 뒤 위 계획대로 비교한다.

기존 진행 중인 Taxi/RAF 실험과 자동화는 별도 캠페인으로 유지한다. 이번 계약이 기존 실행 승인이나 Test 승인을 승계하지 않는다.
