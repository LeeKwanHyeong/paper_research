# Count-aware Tail-aware Auxiliary 5080 Fresh Rerun Result

## 판정

- 상태: **PASS**
- 선택 Variant: `count_only_log_mse_tail_shared` (T1-tail-shared)
- 실행 완료: `2026-08-17 18:21:35 KST`
- Comparator 재검증: `2026-08-17 18:28:06 KST`
- 범위: Intermittent validation only, seed 42
- Held-out test: 미사용

## Best Validation 결과

| Variant | Best epoch | MAE | RMSE | <=p95 MAE | >p99 MAE | Time NLL | 판정 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| T0-logMSE | 200 | 0.762085 | 1.722403 | 0.603369 | 6.067983 | -3.592856 | control |
| T1-tail-shared | 300 | 0.685375 | 1.689794 | 0.531487 | 5.467451 | -3.593712 | pass |
| T2-tail-head-only | 250 | 0.674438 | 1.794892 | 0.489200 | 7.198653 | -3.592021 | fail |

T1은 T0 대비 전체 MAE를 10.07%, RMSE를 1.89%, p99 초과 MAE를 9.90% 개선했다.
p95 이하 MAE도 11.91% 개선했고 Time NLL은 0.000856 낮아져 모든 acceptance
조건을 통과했다.

T2는 전체 MAE와 p95 이하 MAE가 개선됐지만 RMSE가 4.21% 악화됐고 p99 초과 MAE도
18.63% 악화됐다. Tail loss를 quantity head에만 전달하는 방식은 body fitting에는
도움이 됐으나 극단 수량 오차를 줄이지 못했다.

## Artifact 검증

- Source revision: `7de638a5c9290f79dae02a40fd22839aba9802e7`
- 완료 runs: 3/3
- Traceback, NaN, Inf: 없음
- Source와 contract checksum: 일치
- Test summary: validation-only 계약으로 생성하지 않음
- Histories와 quantity/history scale-wise metrics: 생성 및 검증 완료
- Plots: 현재 runner 계약에서 생성하지 않음
- Comparator: 로컬에서 재실행했으며 서버 판정과 동일

## 결론

Seed-42 screening에서는 T1-tail-shared를 다음 단계 후보로 선택한다. 다만 단일 seed와
validation 결과이므로 모델 확정이 아니라 multi-seed 재현성 검증 대상으로만 본다.
T2-tail-head-only는 이번 계약에서 종료한다.
