# Frozen-B Hard-LMM prototype update 필요조건 검사

## 판정

**사용량 정규화만으로 prototype update를 바꾸는 후보는 공통 gate에 실패했다.**
따라서 계약한 `usage-normalized support-confidence prototype update`는 모델에
구현하거나 GPU에서 학습하지 않는다.

검색 사용량 불균형은 분명했다. 하지만 불균형을 줄인 update는 세 데이터셋 모두
두 fold 방향에서 원본 B보다 log-MSE와 raw squared error를 함께 개선하지 못했다.
문제는 prototype이 고르게 학습되지 않는다는 한 가지 현상으로 설명되지 않는다.
데이터셋별로 충돌하는 목적이 달랐다.

이 판정은 Hard-LMM Backbone 개선이 불가능하다는 뜻이 아니다. 이번에 결과를 보기
전에 고정한 한 가지 **정적 prototype 학습 규칙**이 공통 후보가 되기 위한
필요조건을 충족하지 못했다는 뜻이다.

## 현재 B에서 실제로 일어나는 update

B의 `lmm.mem`은 하나의 `64 × 64` static bank이며 key와 value가 같은 parameter다.
각 표본은 cosine similarity로 네 행을 고르고 그 value를 산술평균한다.

- hard index 선택에는 미분 경로가 없으므로 query/key의 **검색 순위 방향**에는
  gradient가 전달되지 않는다.
- 선택된 네 행에는 readout gradient가 각각 `1/4`씩 전달된다.
- 이 bank는 test-time에 쓰이는 memory가 아니다. 학습 중 AdamW가 갱신하는
  prototype parameter다.
- 따라서 이번 검사는 inference 구조가 아니라 선택된 prototype value에 전달되는
  optimizer update를 다룬다.

## 검사 계약

- 기준선: seed 42의 B, 즉 Hard-LMM 구조를 validation raw quantity RMSE의 가장
  이른 최솟값으로 선택한 checkpoint
- 데이터: 데이터셋별 train target 4,096개, whole-series-disjoint fold당 2,048개
- 진단 metric: log quantity MSE, raw quantity squared error, train p95 이하 body MAE,
  legacy Time NLL
- 범위: validation과 held-out test를 materialize하지 않았고 optimizer step도
  수행하지 않음
- 검사 위치: 선택 value bank의 clip 전 gradient와 B의 global gradient clipping
  `1.0`을 반영한 전체 모델 update

후보는 batch에서 prototype 행 `j`의 선택 횟수를 `n_j`, 평균 선택 횟수를
`n̄=4B/64`라 할 때 다음과 같이 고정했다.

```text
Traw_j = Σ(selected credit / 4) / (n_j + n̄)
T      = ||G_log||F · Traw / ||Traw||F
```

즉 자주 선택된 행이 표본 수만큼 큰 update를 독점하지 않게 하고, 전체 Frobenius
norm은 원본 B의 log-quantity memory gradient와 같게 했다. 후보와 동일한 row
update 집합을 1~63칸 순환 이동한 모든 63개 control도 함께 검사했다.

채택하려면 세 데이터셋, 두 fold 방향, clip 전·후 모두에서 다음 조건을 동시에
통과해야 했다.

1. 후보 자체의 log/raw 방향이 양수일 것
2. 후보와 B의 차이가 log/raw를 추가로 개선할 것
3. body MAE와 Time NLL 방향을 악화하지 않을 것
4. B의 양수 방향을 보존할 것
5. log/raw 추가 효과가 63개 shuffled control의 95 percentile보다 클 것

첫 complete run 뒤 독립 QA에서 계약 문구와 export의 감사 가능성을 보완했다.
이 개정은 기존 계약 hash `022bfda5…`와 첫 analysis·decision hash를 새 계약에
남겼다. 이미 계산에 사용하던 gradient parity 분모를 명시하고 행×수량구간 export와
충돌 행 id를 보존한 변경이며, 후보 수식·control·metric·fold·gate·실패 조치는
바꾸지 않았다. 개정 계약으로 전체 진단을 다시 실행해 아래 판정이 같음을 확인했다.

## 검색 사용량과 수량 구간

`유효 행 수`는 선택 분포 entropy를 동일한 entropy의 균등 분포 행 수로 환산한
값이다. `상위 4행 질량`은 가장 자주 선택된 네 prototype 행이 전체 선택에서
차지한 비율이다.

| 데이터셋 | 활성 행 수, fold 0/1 | 유효 행 수, fold 0/1 | 상위 4행 선택 질량, fold 0/1 |
|---|---:|---:|---:|
| Intermittent | 45 / 45 | 14.06 / 14.05 | 49.60% / 50.09% |
| Taxi | 53 / 52 | 22.15 / 20.93 | 36.83% / 39.99% |
| Instacart | 31 / 33 | 7.24 / 7.10 | 83.09% / 83.48% |

불균형은 두 fold에서 재현됐고 Instacart가 가장 심했다. 수량 구간별로 한 번이라도
선택된 prototype 행 수는 다음과 같다.

| 데이터셋 | `≤p50` | `p50–p90` | `p90–p95` | `p95–p99` | `>p99` |
|---|---:|---:|---:|---:|---:|
| Intermittent | 42–43 | 24–28 | 5 | 5–6 | 5–6 |
| Taxi | 48–49 | 40–42 | 4 | 4 | 4 |
| Instacart | 29–32 | 19–21 | 12–13 | 8–10 | 7 |

특히 Taxi의 p90 이상 표본은 각 fold에서 사실상 네 행만 사용했다. Instacart의
`>p99`도 일곱 행에 집중됐다. 따라서 tail 표본이 소수 prototype을 갱신한다는
필요조건 자체는 확인됐다.

## 같은 prototype에서 충돌하는 metric

아래 `충돌 질량`은 두 metric의 행별 aggregate gradient 내적이 음수인 prototype이
전체 선택에서 차지한 비율이다. cosine은 bank 전체 gradient 사이의 값이다. 각
칸은 `fold 0 / fold 1`이다.

| 데이터셋 | log ↔ raw 충돌 질량 (cosine) | log ↔ time 충돌 질량 (cosine) | body ↔ time 충돌 질량 (cosine) |
|---|---:|---:|---:|
| Intermittent | 2.76% / 10.62% (0.947 / 0.964) | 55.13% / 63.31% (-0.027 / -0.033) | 63.96% / 75.01% (-0.040 / -0.041) |
| Taxi | 59.67% / 43.69% (0.152 / 0.559) | 26.76% / 48.82% (0.166 / 0.140) | 12.45% / 53.88% (0.026 / -0.006) |
| Instacart | 97.41% / 96.75% (-0.909 / -0.966) | 3.88% / 9.92% (0.023 / 0.378) | 4.16% / 11.34% (0.016 / 0.377) |

관계는 데이터셋마다 다르다.

- Intermittent에서는 log와 raw 수량 방향이 잘 맞지만, 수량과 시간 방향이
  대부분의 사용량에서 충돌한다.
- Taxi에서는 log, raw, body, time의 관계가 fold 사이에서도 크게 바뀐다.
- Instacart에서는 log-MSE와 body MAE가 잘 맞는 반면 log-MSE와 raw squared
  error가 거의 반대 방향이다. 이 때문에 log loss로 prototype을 더 잘 갱신해도
  raw RMSE가 함께 좋아진다는 보장이 없다.

## 고정 후보의 두 fold 판정

아래 값은 clip 전 선택 value bank에서
`dot(held metric gradient, candidate update − B update)`이다. 양수만 후보가 B보다
좋은 1차 방향이다. 각 칸은 `fold0→fold1 / fold1→fold0`이다.

| 데이터셋 | Δ log | Δ raw | Δ body | Δ time | 판정 |
|---|---:|---:|---:|---:|---|
| Intermittent | -2.11e-6 / -2.01e-6 | -0.00310 / -0.00533 | -0.000275 / -0.000281 | +1.38e-7 / +1.12e-7 | 실패 |
| Taxi | -9.64e-6 / -9.67e-6 | -3.420 / -5.772 | -0.00155 / +0.000816 | -1.21e-6 / -6.75e-7 | 실패 |
| Instacart | -2.56e-4 / -3.06e-4 | +0.0503 / +0.00748 | -0.00242 / -0.00263 | -8.76e-5 / -3.50e-6 | 실패 |

Intermittent와 Taxi에서는 후보가 B의 유용한 log/raw 방향을 약화했다. Instacart는
B 대비 raw 내적이 커졌지만 후보의 **절대 raw 내적이 두 방향 모두 음수**였고,
추가 효과도 shuffled-control 95 percentile을 넘지 못했다. global clip을 반영한
전체 모델 update에서도 모든 데이터셋의 최종 판정은 같았다.

## 무결성과 해석 한계

- frozen revision `f75243473adc25d622319dbca9bda7e076d8240f`에서 로드된
  project Python 파일 48개의 hash를 검증했다.
- top-4 index는 cache와 bitwise하게 같고 target quantity와 duration 차이는 0이다.
- analytic selected-value gradient는 native frozen-B autograd gradient와
  uncancelled scale 기준 최대 `9.42e-8` 상대 오차로 일치했다.
- 강한 상쇄 뒤 남은 native gradient norm만 분모로 사용하면 최악 상대 오차는
  `9.16e-6`이다. 이는 scatter 누적 순서의 반올림이 상쇄로 확대된 감사값이며,
  판정에 쓰인 log gradient의 같은 값은 최대 `3.79e-7`이었다.
- 후보의 batch별 norm match 최대 상대 오차는 `1.65e-15`보다 작다.
- checkpoint state hash는 실행 전후 동일하다.

이 결과는 frozen seed42 checkpoint 한 점에서의 train-only 1차 방향 검사다.
AdamW moment, dropout, weight decay, 여러 step 뒤의 검색 변화 또는 e300 성능을 직접
추정하지 않는다. 그래서 통과했다면 구현 근거만 되었을 것이며, 실패한 현재
결과는 이 한 규칙만 중단하는 근거다.

정식 수치는 [analysis.json](analysis.json), prototype별 export는
[per_slot.csv](per_slot.csv), 수량 구간별 prototype mapping은
[per_slot_stratum.csv](per_slot_stratum.csv), 판정은
[evidence_decision.json](evidence_decision.json), 실행·hash 증적은
[execution_manifest.json](execution_manifest.json)에 저장했다.
