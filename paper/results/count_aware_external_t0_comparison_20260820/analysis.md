# Count-aware External T0 3-seed 결과 해석

## 판정

공통 T0 조건에서 Adapted THP를 외부 backbone 기준선으로 유지한다. Adapted SAHP는
RMTPP보다 우수했지만 THP를 넘지 못했고, Adapted NHP는 quantity 예측에서 뚜렷한
열세를 보여 추가 확장 대상으로 선택하지 않는다.

## 핵심 결과

| 모델 | Quantity MAE | Quantity RMSE | Time NLL |
| --- | ---: | ---: | ---: |
| Adapted RMTPP | 2.902523 +/- 0.170252 | 10.578742 +/- 0.604664 | -3.599494 +/- 0.000000 |
| Adapted THP | **0.666380 +/- 0.081872** | **2.150737 +/- 0.555155** | -3.599485 +/- 0.000004 |
| Adapted NHP | 5.282690 +/- 0.164652 | 15.424367 +/- 0.738889 | -3.599492 +/- 0.000001 |
| Adapted SAHP | 1.081459 +/- 0.008461 | 3.772380 +/- 0.175035 | **-3.599494 +/- 0.000001** |

Time NLL의 모델 간 차이는 약 0.00001 이내다. 따라서 joint objective 차이는 거의
전적으로 log-quantity MSE에서 발생했다. SAHP는 RMTPP보다 quantity MAE와 RMSE를
각각 약 62.7%, 64.3% 줄였지만, THP보다는 각각 약 62.3%, 75.4% 높았다.

## 수량 구간별 해석

- `quantity <= 2`에서는 RMTPP와 NHP의 MAE가 가장 낮았지만 전체 표본의 다른 구간으로 일반화되지 않았다.
- `(2, 31]`부터 `p95-p99`까지 THP가 가장 낮은 MAE를 기록했고 SAHP가 뒤를 이었다.
- `> p99` MAE는 THP 11.2654, SAHP 17.3515, RMTPP 77.6442, NHP 108.0919였다.
- NHP는 중간 및 상위 수량 구간에서 큰 음의 bias를 보여 지속적인 과소 예측이 전체 MAE와 RMSE를 악화시켰다.

## History 구간별 해석

- `History <= 64`와 `65-128`에서는 THP가 가장 안정적인 quantity 성능을 보였다.
- `History > 128`에서도 THP MAE 0.1078로 가장 낮았고 SAHP 0.1126, RMTPP 0.1381, NHP 0.2429 순이었다.
- 긴 history에서도 NHP 또는 SAHP가 THP를 일관되게 앞선다는 근거는 확인되지 않았다.

## 계약 검증

- dataset, fixed split, seeds, maximum epoch, batch size, learning rate, lookback과 max sequence length가 일치한다.
- 네 모델 모두 `t0_common_control`, direct log-MSE와 `legacy_clamped_rmtpp` time head를 사용했다.
- checkpoint는 minimum validation joint objective로 선택했다.
- 모든 summary와 scale-wise metric은 finite였다.
- held-out test는 사용하지 않았다.
- runner가 별도 plot artifact를 생성하지 않아 CSV와 history를 기준으로 판정했다.

TitanTPP-T1은 개선 방법 후보이므로 이 T0 backbone 통제표와 분리해 보고한다. 최종
논문 비교에서는 T0 표로 backbone 기준 성능을 제시하고, TitanTPP-T1은 proposed
method 표에서 별도로 비교해야 한다.
