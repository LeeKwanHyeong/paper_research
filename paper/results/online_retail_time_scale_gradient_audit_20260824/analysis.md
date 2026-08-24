# Online Retail II Train-Only Time-Scale 및 Gradient Clipping 감사 결과

## 판정

- **Train-only stability gate: 0/5 PASS**
- **현재 결정: Online Retail II를 `legacy_clamped_rmtpp` e300 비교에서 보류**
- 실행 서버: 5080, RTX 5080
- source revision: `00df9c91b855cd0465f2a0b08cb3d8b5de89a6f1`
- validation / held-out test: 사용하지 않음

이 판정은 데이터셋을 영구 제외한다는 뜻이 아니다. 단순한 delta-time 단위 변환만으로
현재 legacy time head의 gradient clipping을 해결할 수 없으므로, 이 head를 유지한
정식 비교에 바로 넣지 않는다는 의미다.

## Artifact 확인

1. `manifest.json`: 5개 Variant가 모두 성공 종료했고 passing Variant는 0개였다.
2. `logs/run.log`: focused test `5 passed`, Traceback·NaN·Infinity 없이 모든 Variant가 완료됐다.
3. `variant_summary.csv`와 `decision.json`: 최종 결정은 `stop_online_retail_under_legacy_time_head`다.
4. `evaluation_scope.json`: train만 평가했으며 validation과 held-out test artifact는 없다.
5. `runs/*/history.json`: 각 Variant는 3 epoch, epoch당 16 batch를 완료했고 모든 값이 finite했다.
6. `target_delta_time_summary.json`: train target 563,938건의 p50/p95/p99/max는 21/215/642/9,336시간이다.
7. `plots`: time-scale별 clipping 비율과 최대 per-event Time NLL을 확인했다.

## 핵심 결과

| Variant | Divisor | 초기 clamp 포화율 | 최대 per-event Time NLL | Time-only threshold 초과율 | Joint clip 비율 | 최종 hourly-corrected Time NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 raw hour | 1.0 | 5.2266% | 419,467.69 | 100.00% | 100.00% | 8.5450 |
| S1 calendar day | 24.0 | 0.0121% | 1,463.42 | 95.83% | 100.00% | 5.1205 |
| S2 train median | 21.0 | 0.0158% | 2,767.75 | 93.75% | 100.00% | 5.4212 |
| S3 train mean | 55.3569 | 0.0000% | 154.68 | 91.67% | 100.00% | 4.6496 |
| S4 train p95 | 215.0 | 0.0000% | 75.16 | 93.75% | 100.00% | 4.6488 |

Raw hour는 첫 epoch 평균 time-only gradient norm이 `126,161.77`로 폭증했다. S3는
마지막 epoch 값을 `2.37`까지 낮췄지만 clipping threshold `1.0`을 넘는 batch가
여전히 `81.25%`였고, 3 epoch 전체 기준은 `91.67%`였다. S4처럼 더 큰 divisor를
사용해도 gradient 안정성이 단조롭게 좋아지지 않았다.

Quantity-only gradient도 S3에서 전체 batch의 `83.33%`가 threshold를 넘었다.
따라서 joint clipping 100%는 time path만의 문제가 아니라 quantity path와의 결합
문제도 포함한다. 시간 단위 변환으로 legacy `w·delta_t` clamp 포화와 Time NLL 크기는
대폭 줄일 수 있지만, 전체 최적화 안정성을 해결하기에는 충분하지 않다.

## 해석과 다음 결정

- Online Retail II는 현재 공통 legacy-head e300 표에 포함하지 않는다.
- Intermittent v2와 RAF Spare Parts의 matched validation 준비는 이 결과와 독립적으로 이어갈 수 있다.
- Online Retail II를 다시 포함하려면 train-only scale을 입력 전처리로만 추가하는 대신, 안정적인 duration head와 time/quantity gradient routing을 모든 backbone에 동일하게 적용하는 별도 계약이 필요하다.
- S3는 이번 후보 중 Time NLL과 gradient 규모가 가장 안정적이었지만 gate를 통과하지 못했으므로 공식 time-scale로 채택하지 않는다.
