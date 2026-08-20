# Notion Result Source: TitanTPP Final Time Head + T1 Integration

## 위치

- `5. Model Design Enhancement`
- 기존 페이지: `2026-08-20 TitanTPP Time-Quantity Gradient Audit and T1 Integration`
- 상태: 완료, H3 미채택

## 통합 검증 결과

F0와 F1은 모두 62 epoch에서 early stopping됐고 best epoch는 22였다. 두 run은
time-head family 외 조건이 같고 validation only로 평가했으며 held-out test는 사용하지
않았다.

| Variant | Time head | Time NLL | Quantity MAE | Quantity RMSE | <=p95 MAE | >p99 MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| F0 | H0 scaled exact | 0.822590 | 0.699481 | 1.650459 | 0.539278 | 6.347091 |
| F1 | H3 log-normal duration | 0.474637 | 1.116539 | 3.643457 | 0.597421 | 22.377996 |

H3는 Time NLL을 `0.347953` 낮췄다. 그러나 F0 대비 quantity MAE는 `59.62%`, RMSE는
`120.75%`, `<=p95` MAE는 `10.78%`, `>p99` MAE는 `252.57%` 악화됐다. 모든 값은
finite였지만 quantity safety limit `2%`를 모두 초과했다.

H0는 train time loss와 gradient가 반복적으로 폭증했고, H3는 loss spike를 제거한 대신
평균 gradient clipping 비율이 `99.94%`였다. 따라서 H3의 낮은 Time NLL을 전체 모델의
안정적인 개선으로 보지 않는다.

**판정:** H3는 현재 Hard-LMM + T1 통합 구조에 채택하지 않는다. 현 비교에서는 H0를
유지하지만 H0의 학습 불안정성도 해결된 것으로 간주하지 않는다. H3 상수나 checkpoint를
validation quantity에 맞춰 사후 조정하지 않는다.
