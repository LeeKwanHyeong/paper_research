# Instacart Count-aware T0 3-seed e300 결과

## 결론

Instacart validation에서는 Count-aware RMTPP가 joint objective, Time NLL,
Quantity MAE와 RMSE에서 모두 가장 낮았다. 현재 Hard-LMM 기반 Count-aware
TitanTPP는 THP 및 SAHP와 가까운 최상위권이지만, Instacart에서 독립적인 backbone
우위를 보이지 않았다.

따라서 이 결과는 RMTPP, THP, NHP, SAHP의 Instacart T0 비교군을 동결하는 근거로
사용한다. 이후 memory 구조 실험에서는 동일 artifact를 재사용하고 TitanTPP의
B0/B1/B2 후보만 fresh 학습한다.

## Artifact 및 계약 검증

- 5080 recovery artifact의 5개 backbone x seeds 42, 52, 62, 총 15개 run이 모두 완료됐다.
- source revision은 `28293c43521615be2ed8fad5b043dc9df8e5e457`이며 source manifest의 5개 SHA-256이 해당 revision과 일치했다.
- RMTPP seed 52는 최초 실행의 정상 완료 run을 보존했고, seed 42와 62는 fresh initialization으로 다시 실행했다.
- fixed split, validation-only, `t0_common_control`, direct `log1p(quantity)` MSE, `legacy_clamped_rmtpp` 계약을 확인했다.
- 최대 e300, 최소 e40, patience 40, batch 128, learning rate 0.001, lookback 52일, max sequence 64가 일치했다.
- 15개 summary와 history의 값은 모두 finite이며 recovery launcher log에 Traceback, CUDA error, NaN 또는 Infinity가 없다.
- held-out test 관련 artifact는 없고 `held_out_test_evaluated=false`다.
- runner가 별도 plot을 생성하지 않아 plot artifact는 0개이며, 판정에는 summary 및 quantity/history scale-wise CSV를 사용했다.

## 3-seed Validation 결과

| Backbone | Joint objective | Time NLL | Quantity MAE | Quantity RMSE |
| --- | ---: | ---: | ---: | ---: |
| RMTPP | **3.448557 +/- 0.001010** | **3.205289 +/- 0.000846** | **4.026994 +/- 0.030269** | **5.990086 +/- 0.066938** |
| THP | 3.451637 +/- 0.001669 | 3.206215 +/- 0.000547 | 4.036790 +/- 0.027036 | 6.001171 +/- 0.062483 |
| NHP | 3.495006 +/- 0.002664 | 3.222689 +/- 0.002019 | 4.456324 +/- 0.020013 | 6.713594 +/- 0.040996 |
| SAHP | 3.450706 +/- 0.001069 | 3.206419 +/- 0.000726 | 4.049824 +/- 0.023528 | 6.035738 +/- 0.058942 |
| Count-aware TitanTPP | 3.450881 +/- 0.000830 | 3.206516 +/- 0.000151 | 4.045027 +/- 0.016616 | 6.015904 +/- 0.037398 |

TitanTPP는 RMTPP보다 joint objective 0.067%, Time NLL 0.038%, Quantity MAE
0.448%, RMSE 0.431% 높았다. THP와 비교하면 joint objective는 0.022% 낮지만,
Time NLL은 0.009%, MAE는 0.204%, RMSE는 0.246% 높아 사실상 근접한 결과다.

## Quantity 구간 해석

THP 대비 TitanTPP의 MAE는 `p50-p90`에서 2.081% 낮았다. 반면 `<=p50`은
0.122%, `p90-p95`는 2.353%, `p95-p99`는 3.974%, `>p99`는 3.102%
높았다. RMTPP 대비로도 `<=p50`만 0.652% 낮고 나머지 구간은 0.266-2.248%
높았다.

따라서 Instacart에서 현재 Hard-LMM의 memory retrieval이 body 또는 tail 전체를
일관되게 개선했다고 해석할 수 없다. 특히 validation target 503,733개가 모두
`history <= 64`에 속하므로, 이 데이터셋은 대규모 short-history basket-count
일반화 근거이지 long-history memory 이점의 검증 근거는 아니다.

## 최종 판정

- Instacart T0 validation 비교는 완료됐으며 다섯 backbone의 3-seed 결과를 동결한다.
- RMTPP가 Instacart의 공식 validation 선두다.
- 현재 Hard-LMM TitanTPP는 경쟁력은 있지만 architecture contribution을 입증하지 못했다.
- 다음 비교는 고정된 외부 backbone artifact를 재사용하고 B0 Current Hard-LMM, B1 Faithful Titans-MAC, B2 TPP-specific Gated Memory만 fresh 학습한다.
- Held-out test는 최종 구조와 checkpoint 규칙을 동결할 때까지 계속 잠근다.
