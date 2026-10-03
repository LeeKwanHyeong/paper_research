# 기존 수량 손실·출력 연구의 재사용 범위

2026-10-03 로컬 validation-only 기록 검토. 아래는 과거 B의 결과이며 현재 History-MLP 결과가 아니다. checkpoint binary를 이번에 재검증하거나 replay하지 않았다.

## 수량 손실 × 출력 link 비교 — 완료 기록 재사용

범위: Taxi·Intermittent·Instacart × 4안, seed42, 각120epoch, 12조건. backbone=`titantpp` B, hidden64, 시간손실=`legacy_clamped_rmtpp`, intercept cap300. 입력 log1p 고정. RAF·History-MLP 폭 변경·현재 관측시간 likelihood는 포함하지 않는다.

다음은 **validation raw 수량 RMSE**, 각 조건의 최소 raw RMSE checkpoint 기준이다.

| 데이터 | B log/original | raw/original | log/softplus | raw/softplus |
|---|---:|---:|---:|---:|
| Taxi | 88.1802 | 77.2189 | 102.7071 | 78.1711 |
| Intermittent | 1.4996 | 1.5595 | 1.8057 | 2.0463 |
| Instacart | 5.8722 | 5.7743 | 5.9018 | 5.7744 |

raw/original은 Taxi·Instacart RMSE를 개선하고 Intermittent를 악화했다. Instacart MAE는 3.9938→4.0423으로 악화했다. Taxi raw/original의 마지막120epoch RMSE는97.6934로 선택 checkpoint보다 나빴다. 선택 결과만으로 학습 안정성을 주장하지 않는다. 출력 link만 softplus로 바꾼 결과는 세 데이터 모두 악화했다.

- [설계](/Users/igwanhyeong/PycharmProjects/paper_research/paper/contracts/quantity_objective_output_comparison_v1.json)
- [완료 결과 원본](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/quantity_partition_preparation_v1/monitor/final_results_20260912T092321Z.json)
- [12조건 JSON 메타데이터 감사](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/quantity_partition_preparation_v1/monitor/audit_20260912T092321Z_merged.json)

실행 계약 v1/v2/v3는 독립된 완료 연구 세 개가 아니다. 첫 v2 계열 시도는 CUDA synthetic qualification에서 본학습 전에 실패했고, 수정 v3와 서버별 분담 이후 위12조건이 완료됐다. Taxi는5080, Intermittent·Instacart는5090이었다. 계약의 옛 `pending_approval` 문구와 실제 완료 receipt를 구분한다.

## log/raw 혼합 손실 — 완료 기록 재사용

범위: 위3데이터 × 3안, seed42, 각120epoch, 9조건. 같은 B/legacy 시간손실 계열과 원래 출력 함수를 사용한다. `Llog + α·Lraw_scaled`와 초기 quantity-head gradient 크기를 맞춘 `c(Llog + α·Lraw_scaled)`를 비교했다. α/c는 train-only 초기 calibration으로 고정했으며 epoch별 적응값이 아니다.

**validation raw 수량 RMSE**, 해당 연구의 선택 checkpoint 기준:

| 데이터 | 해당 연구 B | mixed | gradient-matched mixed |
|---|---:|---:|---:|
| Taxi | 85.6070 | 69.7954 | 73.2785 |
| Intermittent | 1.4996 | 1.5618 | 1.5719 |
| Instacart | 5.8728 | 5.8391 | 5.8324 |

두 혼합 후보 모두 당시 공통 채택 기준에서 탈락했다. Taxi matched 혼합만 데이터별 수치 기준을 통과했다. Intermittent RMSE·MAE는 모두 악화했다. Instacart matched는5090, B/mixed는5080이므로 작은 차이를 손실 scaling만의 효과로 해석하지 않는다. 이 표의 B를 앞 표의 B로 바꾸어 비교하지 않는다.

- [validation 비교 원본](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/mixed_quantity_interpretation_20260913_v1/comparison.json)
- [독립 해석 검토](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/mixed_quantity_interpretation_20260913_v1/interpretation_review.json)
- [5080 실행 계약](/Users/igwanhyeong/PycharmProjects/paper_research/paper/contracts/mixed_quantity_execution_5080_v1.json)
- [5090 실행 계약](/Users/igwanhyeong/PycharmProjects/paper_research/paper/contracts/mixed_quantity_execution_5090_v1.json)

## 현재 설계에 적용

1. raw 손실·혼합 손실·softplus link를 처음 시험하는 것처럼 제안하지 않는다. 데이터에 따라 이득과 손해가 갈린다는 선행 증거를 보존한다.
2. 현재 MLP 폭 문제나 입력 log1p 제거의 효과를 위 결과로 확정하지 않는다. 두 연구 모두 그 요인을 바꾸지 않았다.
3. 현재 시간 likelihood·checkpoint 선택·학습 상한에서의 효과는 별도 검증 대상이다. 이전 B와 현재 MLP의 성능 수치를 한 표의 동일 조건으로 섞지 않는다.
4. 우선순위는 독립 검증이 없는 분기 폭 대조, 그 다음 현행 모델의 폭×손실 상호작용과 입력 변환 대조다. 새 backbone 설계는 이 결과에 조건부로 둔다.
