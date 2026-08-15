# Count-aware Tail Lambda Train-only Calibration

- 상태: **COMPLETE**
- 실행 서버: `5080`, CUDA
- Source revision: `3faa1610060294d3eb81ad07261f61fd07c1b010`
- Warm-up / probe batches: `128 / 64`
- Probe targets / tail targets: `8,192 / 388`
- Mean log-MSE quantity-head gradient norm: `0.4048230874`
- Mean unweighted tail quantity-head gradient norm: `0.4443048940`
- Frozen `lambda_tail`: `0.09111380335463036`
- Weighted tail/main gradient ratio: `0.10`
- Validation/test rows read: `0 / 0`

계수는 Intermittent train split만 사용해 고정했다. T1과 T2에 같은 값을 적용한다.
