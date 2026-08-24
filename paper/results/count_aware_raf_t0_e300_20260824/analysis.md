# RAF Spare Parts Count-aware T0 3-seed e300 결과

- 상태: 완료
- 실행 서버: 5080
- source revision: `c7d362e206d8e5d78d96917f03b17b9ad0f50374`
- 실행 시간: 2026-08-24 15:11:42-16:03:31 KST
- artifact: `search_artifacts/count_aware_raf_t0_e300_20260824`
- 평가 범위: validation-only, held-out test 미사용

## 계약 및 Artifact 검증

- 5개 backbone과 seeds 42, 52, 62의 총 15개 run이 모두 완료됐다.
- RAF parquet과 split manifest SHA, source manifest와 실행 revision이 시작 계약과 일치했다.
- T0 공통 계약인 mark-free 입력, direct `log1p(quantity)` MSE, `legacy_clamped_rmtpp`, batch 128, learning rate 0.001, lookback/max sequence 84를 확인했다.
- 15개 summary와 history에서 NaN, Inf, Traceback은 발견되지 않았다.
- 모든 run은 최소 40 epoch 이후 patience 40 조건으로 early stopping됐다.
- Validation-only runner이므로 `test_summary`는 생성되지 않았다. 별도 plot artifact도 생성되지 않았으며, 수치 판정은 summary와 scale-wise CSV를 사용했다.

## 3-seed Validation 결과

| Backbone | Joint objective | Time NLL | Quantity MAE | Quantity RMSE |
| --- | ---: | ---: | ---: | ---: |
| RMTPP | 3.879833 +/- 0.008668 | 3.316661 +/- 0.001616 | 9.146853 +/- 0.210922 | 36.405589 +/- 1.126228 |
| THP | 3.838855 +/- 0.010263 | 3.275795 +/- 0.006350 | **9.140449 +/- 0.073295** | 35.919295 +/- 0.874236 |
| NHP | 3.881560 +/- 0.024216 | 3.310739 +/- 0.020930 | 9.303430 +/- 0.047603 | 36.903886 +/- 0.151358 |
| SAHP | 3.963785 +/- 0.087334 | 3.383700 +/- 0.091683 | 9.394907 +/- 0.162642 | 37.939680 +/- 0.843255 |
| Count-aware TitanTPP | **3.826442 +/- 0.022538** | **3.264143 +/- 0.024193** | 9.148543 +/- 0.103179 | **35.900488 +/- 0.907829** |

Count-aware TitanTPP는 joint objective와 Time NLL 평균이 가장 낮았다. RMTPP 대비 joint objective는 1.376%, Time NLL은 1.583%, RMSE는 1.387% 낮았다. NHP와 SAHP보다도 네 주요 지표가 모두 낮았다.

THP와의 차이는 작다. TitanTPP의 joint objective와 Time NLL은 각각 0.323%, 0.356% 낮았지만 Quantity MAE는 0.089% 높고 RMSE는 0.052% 낮았다. Seed별 TitanTPP-THP 차이의 방향이 일관되지 않아 RAF 하나만으로 통계적 또는 보편적 우위를 주장하지 않는다.

## Quantity 구간 해석

THP 대비 TitanTPP의 MAE는 `<=p50`에서 4.655%, `p50-p90`에서 3.929%, `>p99`에서 0.761% 낮았다. 반면 `p95-p99`에서는 6.350% 높았고 `p90-p95`는 사실상 동일했다.

전체 RMSE가 근소하게 낮은 것은 모든 상위 수량 구간의 일관된 개선이 아니라 body와 extreme-tail 일부의 상쇄 결과다. RAF에서 TitanTPP의 quantity 이점을 tail 전반의 강점으로 확대 해석하지 않는다.

## History 길이와 결론

Validation target 6,690개가 모두 `history <= 64`에 속했다. 따라서 RAF는 실제 spare-parts 수요에서 T0의 데이터셋 일반성을 확인하는 근거는 되지만, Titan memory의 long-history 이점을 검증하는 데이터셋은 아니다.

최종 판정은 다음과 같다.

- Count-aware TitanTPP-T0는 RAF에서 경쟁력 있는 최상위권 결과를 보였다.
- Time NLL과 joint objective 평균은 5개 backbone 중 가장 낮았다.
- Quantity는 THP와 사실상 동률이며, TitanTPP의 명확한 단독 우위로 표현하지 않는다.
- RAF 결과는 T0 주 모델을 유지할 근거에는 포함하되, 보편적 우월성은 Instacart 등 추가 matched T0 결과와 함께 판단한다.
