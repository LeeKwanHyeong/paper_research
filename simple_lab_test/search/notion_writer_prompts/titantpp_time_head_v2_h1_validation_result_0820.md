# Notion Result Source: Time Head v2 H1 Validation Safety Screening

## 대상

- `5. Model Design Enhancement`
- 기존 페이지: `2026-08-20 TitanTPP Stable Exact Time Head v2`
- 기존 `H1 Validation Safety Screening > 결과` 아래에만 반영한다.

## 완료 상태

- epoch 62 early stopping, best epoch 22
- seed 42 validation only
- held-out test 미사용
- H1 safety gate 실패
- H0 time head 유지

## 핵심 결과

| Head | Time NLL | Quantity MAE | Quantity RMSE | `<=p95` MAE |
| --- | ---: | ---: | ---: | ---: |
| H0 | 0.820917 | 0.645580 | 1.733665 | 0.464500 |
| H1 | 1.563867 | 0.884585 | 2.895075 | 0.473229 |

- Time NLL: `+0.742950`, 기준 실패
- MAE: `+37.02%`, 기준 실패
- RMSE: `+66.99%`, 기준 실패
- `<=p95` MAE: `+1.88%`, 기준 통과
- H1은 train loss와 gradient clipping을 안정화했지만 p95 초과 tail과 중간·긴 history
  quantity 오차가 크게 악화됐다.
- H1을 채택하지 않고 H0를 유지한다. Memory 재비교는 열지 않는다.
