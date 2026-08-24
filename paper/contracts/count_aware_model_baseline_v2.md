# Count-aware TitanTPP 공식 T0 계약 v2

- 동결일: 2026-08-24
- 논문 표기: `Count-aware TitanTPP`
- 내부 실험명: `TitanTPP-T0`
- 적용 범위: mark-free event-time 및 continuous quantity 연구 트랙
- Held-out test: 모델과 논문 주장을 고정하기 전까지 잠금

## 주 모델

논문의 주 모델은 TitanTPP-T0다. Titan Hard-LMM encoder에 mark-free count
formulation을 적용하고, 다음 사건의 시간과 연속 수량을 함께 예측한다.

| 축 | 고정 값 |
| --- | --- |
| encoder | `count_titan_small_lmm`, hidden dimension 64 |
| memory | static Hard-LMM, persistent 16, memory 64, top-k 4 |
| history input | `log1p(delta_t)`, `log1p(raw_quantity)` |
| mark/residual | 사용하지 않음 |
| quantity target | `log1p(raw_quantity)` |
| quantity loss | direct MSE |
| point prediction | `distribution_median_expm1_location` (`expm1` 복원) |
| time head | `legacy_clamped_rmtpp` |
| tail loss | 사용하지 않음, `lambda_tail=0` |

## 공정한 Backbone 비교

RMTPP, THP, NHP, SAHP와 TitanTPP에는 동일한 history feature, quantity head,
direct log-MSE, time head와 checkpoint selection을 적용한다. T0 표의 차이는 encoder
backbone 차이로 해석한다.

모든 모델은 seeds 42·52·62, 최대 300 epoch, 최소 40 epoch, patience 40,
batch 128, learning rate 0.001, hidden dimension 64, gradient clipping 1.0과
minimum validation joint objective checkpoint를 사용한다. 첫 비교는
validation-only이며 held-out test는 사용하지 않는다.

## 데이터셋별 허용 차이

데이터의 관측 길이가 다르므로 context window만 아래처럼 다르게 둔다. Time unit은
튜닝 대상이 아니라 각 데이터셋에 고정된 속성이다.

| Dataset | Time unit | Lookback | Max sequence length |
| --- | --- | ---: | ---: |
| Intermittent v2 | week | 520 | 256 |
| RAF Spare Parts | month | 84 | 84 |
| Taxi Hourly | hour | 168 | 256 |
| Instacart | day | 52 | 64 |

이외의 head, loss, optimizer, seed, checkpoint selection과 평가 범위는 동일하게
유지한다.

## T1의 역할

TitanTPP-T1은 TitanTPP-T0에 tail-aware auxiliary loss만 추가한 objective
ablation이다. Backbone이나 model head가 달라지는 새 아키텍처로 해석하지 않는다.

T1은 Intermittent에서 T0 대비 quantity MAE와 RMSE를 개선했지만 Taxi에서는 모두
악화됐다. 따라서 T1은 주 모델 비교표에서 제외하고 dataset-specific long-tail
ablation으로만 보고한다.

## 해석 경계

- Backbone 기여는 T0 공통 loss 조건의 모델 간 비교로만 판단한다.
- T0와 T1의 차이는 objective 효과다.
- RMSE 또는 extreme-tail 이점을 전체 MAE나 time modeling 우위로 확대하지 않는다.
- 데이터셋별 validation 결과를 확인하기 전에 보편적 우월성을 주장하지 않는다.
- 구조와 논문 주장을 고정한 뒤 held-out test를 한 번만 평가한다.
