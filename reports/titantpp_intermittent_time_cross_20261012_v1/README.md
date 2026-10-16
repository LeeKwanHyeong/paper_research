# Intermittent 시간 위치와 분포 폭의 점수 영향을 분리

**승인된 동일 선택 모델과 실행 순서를 유지한다 — 완료**

- 사용자는 RAF18 이후 이 단계18평가를 승인했다. 같은 full/두 대조×3seed 선택9checkpoint×Train/Validation=18model-split pass이며 새 학습·새 Test 추론·checkpoint 재선택·floor/temperature 탐색·원고 편집은0이다.
- source129·선택 checkpoint·전처리·원래 float64 정수 시간 확률질량·top_code=None을 유지한다. 서로 다른 모델의 실제 시간 위치 μ와 분포 폭 σ를 섞은 점수는 배포 가능한 새 모델이나 완료된 calibration이 아니다.

**같은 사건의 세 모델 출력을 메모리에서 정렬한다 — 완료**

- 원별 μ·σ·목표 배열을 디스크에 남기지 않고 정확히 정렬하기 위해 seed×split의6개 joint worker에서 각 세 모델을 같은 batch로 순차 forward한다. 실제 모델 평가 수는18이며 실제 OS worker 수는6이다. 이전 후보의 “18worker” 표현은 이 운영 방식으로 구체화하고 두 수를 구분해 기록한다.
- joint worker는 최대1개, resident 모델은3개다. 원래9모델×양 split의18개 metric·대상·truth/history digest·state 불변을 재현하고, 각 모델 평가 receipt는 공동 PID와 실제 시작/완료를 정직하게 기록한다. 원래 model forward나 가중치는 바꾸지 않는다.
- full−제거/full−current-only의6pair×2split×4셀=48 paired score cells에는 원래 full의 중복이 있고, 새 교차 score cells는24개다. 위치만 교체·폭만 교체·상호작용의 손해/이득을 전체·기존 Train-derived 시간 구간에서 함께 보고한다.

**선행 완료와 결과의 과학 범위를 확인한다 — 완료**

- RAF18 actual exit0·immutable 원본·로컬 SHA 및 분석 완료를 새 계약의 선행 조건으로 결합한 뒤 launch한다. native3route와6joint OS worker의 actual exit0·18모델 receipt·pipeline OS exit0·archive/manifest·로컬 SHA·독립 QA를 확인한다.
- actual 두 원래 score가 기존 NLL과 같고 새 교차 score도 finite여야 한다. 원래 정수 질량을 연속 밀도나 평균 μ·σ로 대신하지 않는다. Test 구간을 다시 평가하거나 어떤 score를 Test에 fit하지 않는다.
- 결과는 저장 모델의 출력 의존성 검증이다. CNN/GRU 학습의 인과 원인·새 모델 일반화·기존 Test 반전의 해결로 확대하지 않는다. 불리한 seed·대조·score와 실패 원본을 보존한다.

**실제 완료 결과를 제한된 해석에 연결한다 — 완료**

- RAF18 원본·분석·독립 QA 뒤에만 시작했다. 2026-10-12 13:13:49 KST에 실제 pipeline 종료, six joint worker exit0·18모델 평가·native3 PASS와309개 원본 archive/manifest·두 로컬 복사본 SHA·독립 최종 QA가 확인됐다.
- Validation의 세 seed×두 대조6조건에서 full 위치를 유지한 실제 대조 σ 교체는 모두 NLL을 낮췄고, μ만 교체하면 모두 높아졌다. Train에서는 두 단일 교체 모두6조건에서 악화했다. seed42 Validation은0.618713→σ 교체0.243837/0.253718이며 μ 교체0.858383/0.686191이다.
- 기록된 간격(1,3]에서 σ 교체 이득이 두드러졌지만 ≤1과 일부 >7 구간에는 손해가 있었다. 상호작용을 남기며 원인을 유일한 비율로 나누거나 임의 σ 확대·floor 변경·새 calibration의 효과를 확정하지 않는다.
- [최종 원본 분석](../../search_artifacts/titantpp_intermittent_time_cross_20261012_v1/analysis/TIME_CROSS_REPORT.md), [독립 최종 QA](FINAL_QA.md), [논문 주장과 후속 순서](DIAGNOSTIC_DECISION.md)를 기준으로 사용한다.

**완료 진단을 원고와 후속 결정에 반영한다 — 다음 작업**

- 같은 열린 main.tex에 예외·진단·한계를 반영하는 문서 작업을 우선한다. 이번 실행에서는 원고 편집·추가 학습·Test 평가·Runtime 변경0이다.
- 원고를 보완하는 데 실제 개입 검증이 꼭 필요한 경우만 단일 가설의 새 Train/Validation 계약을 확정하고 별도 실행 승인을 받는다. 빈 GPU를 이유로 학습·검색을 시작하지 않는다.
