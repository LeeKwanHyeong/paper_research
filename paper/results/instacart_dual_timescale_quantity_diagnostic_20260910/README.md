# Instacart 동일 표본 진단 — 현재 dual-timescale 후보

현재 dual-timescale 후보의 Instacart 약점은 비교 상대에 따라 구분됩니다. RMTPP 대비 validation RMSE는 0.53% 높고 MAE는 사실상 동률입니다. 제곱오차 차이의 83.62%가 평균 과소예측 차이 항으로 분해되며, 수량 8–35와 관측 이력 8–15 구간이 주된 불리 조건입니다. 반면 B 대비로는 전체 RMSE·MAE가 소폭 좋아졌지만 이력 2–3 구간에서 예측이 내려가며 오차가 커집니다. 이력 8–15에서는 B보다 개선되므로 짧은 이력 문제를 모든 benchmark 차이의 설명으로 적용하면 안 됩니다.

## 완료 범위

- Validation 503,733건 / 206,195 series, train 1,991,192건 / 205,934 series를 비교했다.
- B는 검증된 원본 예측 캐시를 재사용하고, 현재 후보·RMTPP·THP는 고정 checkpoint에서 로컬 CPU로 추론했다. 모든 모델 state SHA가 유지되었다.
- source `ca8823e` 및 원본 checkpoint·데이터·모집단 SHA를 검증했다. Held-out test를 사용하지 않았다.

## 동일 validation 결과

| 모델 | Raw RMSE | MAE | Body MAE (≤25) | >p99 MAE (>35) | 평균 오차 |
|---|---:|---:|---:|---:|---:|
| TitanTPP(B) | 5.872217 | 3.993781 | 3.438602 | 21.917795 | -0.849631 |
| 현재 dual-timescale 후보 | 5.868058 | 3.985497 | 3.433833 | 21.781933 | -0.863504 |
| RMTPP | 5.836975 | 3.985848 | 3.440219 | 21.889153 | -0.664380 |
| THP | 5.852341 | 3.991003 | 3.454442 | 21.773307 | -0.649639 |

음의 평균 오차는 과소예측이다. Body는 train p95 경계 25 이하, tail은 train p99 경계 35 초과다. MAE와 RMSE는 같은 수량 예측을 사용한다.

## 실제 차이의 위치

1. **RMTPP와의 차이는 중간 수량·이력 구간에 있다.** 전체 ΔMSE +0.363835 중 bias² 차이가 +0.304238(83.62%), centered MSE 차이가 +0.059597이다. 수량 8–35의 기여 +0.834117을 ≤8의 개선 −0.456384와 >35의 개선 −0.013899가 일부 상쇄한다. 이력 8–15가 +0.273830으로 가장 큰 불리 기여이며, 2–3은 −0.030739로 유리하다.
2. **B와의 차이는 이력별 trade-off다.** 이력 2–3에서는 예측 이동 −0.186239, ΔMSE +0.271425이고, 8–15에서는 예측 이동 +0.095191, ΔMSE −0.306128이다. 전체 예측 이동은 −0.013873에 그친다. 특히 수량 (8,20]에서는 예측 이동 +0.002286임에도 centered MSE가 +0.210465 악화되어 이전 후보의 전체 하향 이동 설명을 그대로 사용할 수 없다.
3. **소수 series 하나로 설명되지 않는다.** RMTPP보다 SSE가 작은 series는 50.34%다. 최악 상위 1%는 gross positive SSE의 29.40%, 상위 5%는 59.74%를 차지한다. Gross positive 640,822와 개선량 457,546이 상쇄되어 net 183,275가 남는다. Net과 gross 비중을 혼동하지 않는다.
4. **MAE의 RMTPP 우세 주장은 보류한다.** 후보−RMTPP MAE 차이는 −0.000351이고 series bootstrap 구간은 [−0.001704,+0.001091]이다. 고정 seed42에서 사실상 동률이다.

## Train의 분리된 series 집단에서 재현

Train 1,991,192건을 series가 겹치지 않는 두 집단으로 나눈 결과, validation에서 먼저 고정한 5개 가설의 46개 방향 조건이 모두 재현됐습니다. 다만 train 전체에서는 후보의 RMSE가 THP보다 좋고 validation에서는 나빠집니다. RMTPP 대비 centered MSE도 train에서는 개선되고 validation에서는 악화됩니다. 따라서 조건별 오차 패턴의 재현과 전체 순위·일반화의 재현을 구분해야 합니다. 두 train 집단은 원래 모델 학습에 사용되었으며 out-of-fold 검증이 아닙니다.

| 비교·조건 | Train fold0 ΔMSE | Train fold1 ΔMSE | 방향 |
|---|---:|---:|---|
| 후보−RMTPP, 이력 2–3 | −0.081462 | −0.076642 | 후보 유리 |
| 후보−RMTPP, 이력 8–15 | +0.420739 | +0.525499 | 후보 불리 |
| 후보−B, 이력 2–3 | +0.079395 | +0.092262 | 후보 불리 |
| 후보−B, 이력 8–15 | −0.124836 | −0.141327 | 후보 유리 |

Fold0은 998,509건/103,121 series, fold1은 992,683건/102,813 series다. 전역 THP 순위 반전이나 centered MSE 방향까지 모두 재현되었다고 주장하지 않는다.

## 검사·한계

- 분석기 단위·계약 테스트 6개 통과. 구간별 가중 기여, bias²/centered MSE 분해, fold 분리와 합계 복원을 검사했다. 독립 계산 감사는 `independent_audit.json`에 기록한다.
- 첫 RMTPP CPU 재현은 기존 CUDA 기록과 RMSE +0.00001404 차이로 절대1e-5 기준에 실패했다. 원래 실패를 유지하고, 행 분석 전 동결한 별도 수치 호환성 기준으로 통과했다. 세 raw-RMSE 비교의 최대 왜곡은 원래 차이의 0.0473%다. 구간별 최대 MAE/RMSE 재현 차이는 0.0006062/0.0004378이고 모든 30개 비교 방향이 보존되었다.
- Bootstrap은 같은 validation과 고정 모델에서의 series 표본 변동만 설명한다. 이미 checkpoint 선택에 쓰인 validation이므로 독립 확증 검정이 아니며, 추가 seed·held-out 결과를 대신하지 않는다.
- Bias 분해는 회계적 분해다. 예측 시 알 수 없는 target 수량 구간을 경로 선택에 사용하거나, 일괄 상향 보정의 효과가 입증되었다고 해석하지 않는다.

## 다음 작업 / 로컬·5090 기존 평가

현재 근거는 어떤 조건에서 예측 수준과 오차가 달라지는지를 설명합니다. 과소예측의 발생 위치를 local/global memory, 공통 encoder 또는 quantity head 중 하나로 특정하는 인과 증거는 아직 없습니다. 일괄 상향 보정의 효과도 검증하지 않았습니다. 현재 승인된 Intermittent 평가와 이 진단을 합쳐 기존 후보의 최종 비교를 완료한 뒤 수정 여부를 결정하는 순서를 유지합니다. 이번 작업에서는 모델 변경·재학습·보정·checkpoint 재선택·held-out 평가를 실행하지 않았습니다.

## 보고서 표시 검증

보고서의 canonical 데이터 일치, 구조 검증과 self-contained HTML 패키징을 통과했다. 호환 Chromium이 없어 브라우저 렌더링·배치 검증은 실행하지 못했다(`structural_only`). 표와 차트의 의미를 보존하는 HTML fallback 표는 포함되어 있다. 이는 분석 수치의 독립 감사와 별도의 표시 검증 한계다.

## 산출물

- `report/report.html`: 표·그림·재현 조건을 담은 보고서
- `validation/`, `train/`: 전체·구간·fold별 집계
- `validation_hypotheses.json`, `train_hypothesis_checks.json`: train 확인 전 가설과 46개 판정
- `methodology.md`: 수식, source·cache 재사용, 실행 명령, 수치 재현 정책
- `search_artifacts/instacart_dual_timescale_quantity_diagnostic_20260910/`: 고정 checkpoint, 동일 표본 예측, series 상세 기여
