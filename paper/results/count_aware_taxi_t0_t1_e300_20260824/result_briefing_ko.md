# Taxi mark-free T0·TitanTPP-T1 3-seed 결과

## 결론

Taxi에서는 tail-aware auxiliary loss를 적용한 TitanTPP-T1이 matched
TitanTPP-T0보다 좋아지지 않았다. T1은 T0 대비 validation quantity MAE가
`42.5805`에서 `48.4493`으로 13.8%, RMSE가 `143.9931`에서 `165.8092`로
15.2% 악화됐다. Time NLL 차이는 `+0.000384`로 작아, 주된 손상은 quantity
예측에서 발생했다.

따라서 Intermittent에서 선택한 T1 objective를 Taxi의 공통 모델로 확장하지 않는다.
Taxi에서는 T0 direct log-MSE를 유지하고, 데이터셋별 tail objective가 필요하다는
근거로 해석한다.

## 전체 비교

| 역할 | Backbone | Validation joint objective | Time NLL | Quantity MAE | Quantity RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| T0 | RMTPP | 1.5447 ± 0.0022 | 1.3598 ± 0.0020 | **40.2893 ± 3.2321** | 144.8148 ± 14.0054 |
| T0 | THP | 1.5573 ± 0.0090 | 1.3624 ± 0.0028 | 41.5884 ± 2.9369 | 147.4963 ± 10.2086 |
| T0 | NHP | 1.6238 ± 0.0110 | 1.3638 ± 0.0033 | 95.5995 ± 6.0161 | 327.0676 ± 24.3055 |
| T0 | SAHP | 1.5979 ± 0.0099 | 1.3617 ± 0.0034 | 56.1938 ± 7.6475 | 222.4728 ± 36.0360 |
| T0 | TitanTPP | 1.5565 ± 0.0010 | 1.3658 ± 0.0015 | 42.5805 ± 8.0971 | **143.9931 ± 32.5312** |
| T1 | TitanTPP | 1.5566 ± 0.0030 | 1.3662 ± 0.0009 | 48.4493 ± 7.4778 | 165.8092 ± 33.5499 |

T0 backbone 비교에서는 RMTPP가 joint objective와 MAE에서 가장 좋았다.
TitanTPP-T0는 RMSE가 가장 낮았지만 seed 간 편차가 커서 Taxi에서 일관된 backbone
우위라고 보기는 어렵다.

## Quantity 구간별 해석

TitanTPP-T1은 TitanTPP-T0 대비 `<=p50` MAE만 2.6% 개선했다. 반면
`p50-p90`은 2.8%, `p90-p95`는 9.8%, `p95-p99`는 20.5%, `>p99`는
25.5% 악화됐다. Tail loss가 직접 겨냥한 상위 구간에서도 평균 개선이 나타나지
않았고 seed 간 편차도 컸다.

이는 Intermittent train 분포에서 고정한 tail-aware 학습 가설을 Taxi에 동일하게
적용하는 방식이 적절하지 않음을 보여준다. 현재 결과는 loss 자체의 보편적 개선이나
TitanTPP의 일관된 우월성을 뒷받침하지 않는다.

## Artifact 검증

- 18개 run이 모두 성공했고 seeds 42·52·62가 포함됐다.
- source revision, fixed split, batch 128, learning rate 0.001, 최대 300 epoch,
  최소 40 epoch, patience 40, lookback 168시간, max sequence length 256이 일치했다.
- 모든 run은 validation joint objective 최저 checkpoint를 사용했다.
- held-out test artifact는 없으며 test set은 사용하지 않았다.
- JSON metric에서 NaN/Inf가 없었고 Traceback도 확인되지 않았다.
- runner가 별도 plot을 생성하지 않아 plot artifact는 없었다. 수치 판정은 summary,
  quantity/history scale-wise CSV를 사용했다.
