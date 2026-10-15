# 루프3 완료와 다음 진단 범위

**RAF ValueDifference 검증 — 완료, 사용자 성능 기준 미달**
- 고정 선택 checkpoint 3개의 학습·원본 회수·Validation3/Test3를 완료했다. 네 기준 중 Test RMSE만 기존 CNN과 외부 지표별 최우수보다 낮다.
- 기존 CNN 모델·checkpoint·결과를 재학습하거나 재평가하지 않았다.

**같은 대상의 Train·Validation 오류 비교 — 완료**
- 현재 후보에 한해 새 격리 경로에서 예측 행을 추출했다. 2026-10-16 02:47:58 KST에 회수 완료했다. Train 25,779개와 Validation 6,690개를 seed마다 동일 대상으로 결속했다.
- Train 평균 ΔRMSE −0.201754, ΔMAE −0.008222이나 Validation은 +0.014398, +0.032349다. Validation MAE는 세 seed 모두 악화했다. 공통 개체 균등 가중에서도 Validation 악화가 남았다.
- 고정15조건에서 Train/Validation·세 seed·두 가중 방식 모두 일관되게 악화하는 단일 조건은 없다. 특정 패턴의 실패나 과적합을 원인으로 확정하지 않는다.
- 학습된 차분 계수의 읽기 전용 응답 감사도 완료했다. 잠재 V의 상수 응답은 설계대로 보존되지만 최종 raw 수량의 편향을 없앤다는 뜻은 아니다. 응답 크기에 임의 문턱을 두어 새 방법을 선정하지 않는다.

**현재 후보 V 보정 분기의 직접 영향 확인 — 다음 작업**
- NEXT_MECHANISM_SPEC.json과 NEXT_MECHANISM_ACCEPTANCE.json에 따라 별도 진단 코드를 구현하고 합성검사·독립 검토한다. 같은 후보의 V 잔차 켬/끔 두 조건만 다루며 공통 가중치·입력·head·메모리·선택 checkpoint를 보존한다.
- 5090의 현재 유휴 상태와 고정 환경을 새로 확인하고, 원래 on 조건의 Validation을 지표와 개별 행 모두 재현한 뒤 off Train·Validation을 평가한다. 최대3 병렬이며 기존 CNN·Test·새 학습은 실행하지 않는다.
- 이 진단의 off 조건을 기존 CNN이나 V 없이 새로 학습한 모델로 해석하지 않는다. 현재 후보 가중치에서의 분기 민감도만 확인한다.

**다음 backbone 학습 계약 — 진단 근거 확보 후**
- 근거가 있는 단일 CNN 연산 가설과 대조를 새 격리 경로에 준비한다. 기존 Adaptive·MultiLag·LevelHistory를 이름만 바꿔 재시도하지 않는다.
- 사용자 사전승인은 유지한다. 새로운 baseline 학습이나 손실·head·메모리·RevIN·GRU 변경으로 범위를 넓히지 않는다. 루프4 학습은 아직 시작하지 않았다.
- 반복 Test를 참고한 탐색이라는 한계와 5080/5090 Runtime 차이를 유지한다. RAF 기준 통과 전에는 Taxi·Intermittent로 확장하지 않는다.
