# 완료 13조건의 LNCS 원고 반영

현재 열린 `paper/titantpp_pakdd_2027_draft/main.tex`를 제자리에서 수정했다. 원고의 Appendix A, 실험 설정, 이력 보정 결과 해석, Discussion 및 두 번째 Contribution에 검증된 후속 결과를 연결했다. 편집기 내장 컴파일러의 최종 결과는 `success`다.

## 요청과 반영 범위

사용자 지시:

> Appendix A: 검증을 마친 Taxi 3seed 평균·표준편차를 반영합니다. Intermittent·Instacart는 완료된 seed42 결과로 표시하고 나머지는 진행 중으로 구분합니다.
>
> 관련 본문: 인접 상태 결합의 효과와 단계적 분기 활성화의 효과를 분리합니다.
>
> Contribution: 이력 보정 구조·수량 예측 효과·계산 효율이라는 중심은 유지하되, 모든 데이터에서 효과적이라는 해석을 피합니다.

관측 기준은 2026-10-02 10:51:58 KST의 완료 13조건이다. 이 시각을 현재 진행 상태로 바꾸어 표현하지 않았다. 원본 근거는 `reports/titantpp_extension_completed13_audit_20261002_v1/comparison.json`과 그 회수·CPU 검증 결과다.

## 수정 내용

- Appendix A의 07:14 KST·11조건·checkpoint 검증 대기 설명을 10:51:58 KST·13조건·검증 완료로 갱신했다.
- Taxi의 기존 MLP와 세 후속 모델을 평균 ± 표본 표준편차로 제시했다. Intermittent·Instacart는 seed42의 기존 MLP와 두 구조 변형을 별도 표로 배치했다.
- 인접 상태를 직접 결합하는 구조의 Taxi 수량 개선과, 전체 분기 사용 변형의 더 낮은 수량 오차를 함께 설명했다. 가용성 규칙의 우월성을 구조 전체의 유용성과 동일시하지 않았다.
- 현재 상태 전용 변형도 causal encoder의 이력을 유지한다는 설명을 본문에 추가했다. Deep Renewal의 고유 NB 손실과 공통 회귀 손실을 구분했다.
- 다른 그룹의 진행 중·대기 상태를 관측 시각에 한정해 기록했다. 미완료 그룹의 3seed 추정치는 제시하지 않았다.

## 주장과 근거 연결

| 원고 주장 | 검증된 근거 | 반영 위치 |
|---|---|---|
| Taxi에서 기존 MLP는 같은 파라미터 수의 현재 상태 전용 변형보다 수량 오차가 낮다. | MAE 11.07%, RMSE 11.48% 감소; 두 지표 모두 3/3 paired seeds | Results 5.3, Appendix A |
| 전체 분기 사용은 Taxi 수량 오차를 더 낮추지만 시간 NLL은 높인다. | MAE 1.15%, RMSE 1.99% 감소; 개선 seed 1/3·2/3; 평균 NLL 증가 | Results 5.3, Appendix A, Discussion |
| Intermittent의 기존 MLP는 두 구조 변형보다 세 지표가 낮다. | 검증된 seed42만 비교 | Appendix A의 별도 단일 seed 표 |
| Instacart의 두 변형은 기존 MLP보다 RMSE가 약 0.11% 낮으며 지표별 방향이 다르다. | 검증된 seed42만 비교 | Results 5.3, Appendix A |
| Deep Renewal의 Taxi 비교는 고유 NB 손실과 최대 300epoch 예산에서의 결과다. | 세 seed 모두 선택 epoch 300; 수량 오차와 시간 NLL의 순위 상충 | 설정, Appendix A |
| 수량 개선의 크기와 방향은 데이터에 따라 다르다. | 기존 84조건 주 비교 및 새 구조 대조 결과 | Contribution 2, 본문 해석 |

## 검증

`verify.py`로 Taxi 4행의 평균·표준편차와 단일 seed 6행의 모든 수치를 원본 감사 결과와 대조했다. 두 쌍의 상대 개선율과 paired-seed 방향도 확인했다. 기존 84조건 주 비교표, 구성 제거 표, 효율 표, Appendix B/C, 수식·그림·저자·참고문헌 및 초록·결론은 이전 원본과 동일하다. 새 label은 중복되지 않고 모든 ref가 연결된다.

학술 표현은 숫자에 관한 사실과 구조에 관한 해석을 분리했다. 새 문단의 동사는 `compare`, `reduce`, `support`, `distinguish`, `achieve` 등을 사용했고, 결론을 데이터·seed 범위에 맞췄다. 원문은 `main_before.tex`, 전체 변경은 `main.diff`, 검증 결과는 `verification.json`, 편집기 컴파일 결과는 `compilation_receipt.json`에 보존했다. 새 PDF를 별도로 생성하거나 다른 편집기 탭을 열지 않았다.

## 남은 작업

**1. 남은 후속 그룹의 완료 결과를 검증한다 — 진행 중**

- 승인된 5080·5090 큐의 완료분을 원본과 대조한다. 데이터·모델별 세 seed가 모이면 부분 결과를 평균·표본 표준편차로 교체한다.

**2. 후속 비교와 최종 평가 범위에 맞춰 원고를 확정한다 — 다음 작업**

- 학습 결과의 취합과 평가 데이터 계보 정리를 병행한다. 공통 결과가 확정된 뒤 초록·결론과 전체 해석을 함께 정리한다.
