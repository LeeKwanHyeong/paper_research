# TitanTPP Direct Raw Quantity Q0/Q1/Q2 Instacart Top-20 e1 Smoke 결과

기존 Notion 세부 페이지를 갱신한다.

- 상위 history: `https://app.notion.com/p/2e8bbe40561380a88b5eef94e834892e`
- 결과 페이지: `https://app.notion.com/p/39cbbe4056138156af0eec9d2ae7f9eb`
- 관련 contract: `https://app.notion.com/p/39cbbe40561381dda378d65257d6719c`
- 상위 Step: `Step 10. Q0/Q1/Q2 Instacart Top-20 e1 Integration Smoke`

새 페이지를 만들지 않는다. 결과 페이지의 시작 기록 아래에 완료 결과를 추가하고,
상위 history의 Step 10과 현재 의사결정, contract 페이지의 구현 상태와 다음 작업을
갱신한다.

## 실행 정보

- 실행일: `2026-07-13 KST`
- 시작/종료: `16:40:15 / 16:40:30 KST`
- 실행 서버: `5090`, GPU `NVIDIA GeForce RTX 5090`
- Python: `/opt/miniconda3/envs/ai_env/bin/python`
- tmux session: `insta_raw_q012_e1_0713`
- artifact: `search_artifacts/model_enhancement_direct_raw_qty_q012_insta_smoke_e1_0713`
- 통합 상태: `SMOKE_SUCCESS`, Q0/Q1/Q2 exit code `0`

## Artifact 확인 순서

1. root/variant `experiment_manifest.json`: 실행 환경과 normalization identity 확인
2. root/variant `logs/run.log`: 세 변형 순차 완료와 runtime error 여부 확인
3. `leaderboard/summary.csv`: validation과 raw/context diagnostics 확인
4. `leaderboard/test_summary.csv`: smoke export와 finite 여부만 확인
5. `leaderboard/histories.csv`: epoch 1 loss decomposition 확인
6. validation/test `scale_wise_metrics.csv`: bucket count와 weighted metric 확인
7. validation/test scale-wise plot 및 learning curve: 표와 시각화 일치 여부 확인

## Matched 실행 계약

Q0/Q1/Q2는 normalization mode만 다르다.

| 항목 | 공통 값 |
| --- | --- |
| dataset / split | `insta_market_basket` top-20 / `fixed` |
| model / candidate | `TitanTPP / small_lmm` |
| epoch / seed | `1 / 42` |
| LR / batch | `1e-3 / 16` |
| lookback / max sequence | `10 / 16` |
| decoder / magnitude domain | `direct_raw_qty / raw_qty` |
| train loss scope / mode | `target_only / hybrid` |
| marker loss | plain CE |
| lambda magnitude / quantity | `1.0 / 0.25` |
| RevIN epsilon / shrinkage k | `1e-5 / 8` |

- DataLoader samples: train `1,380`, validation `300`, test `300`
- series / real marks: `20 / 7`
- 세 run 모두 같은 fixed split parquet, seed, sample count를 사용
- train-only raw event count: `1,400`
- raw mean / variance / std: `13.770000 / 45.937100 / 6.777691`
- effective sigma floor: `0.006778`
- validation/test row는 normalization 통계에 포함되지 않음

## Validation 결과

| Metric | Q0 global | Q1 causal RevIN | Q2 shrinkage RevIN |
| --- | ---: | ---: | ---: |
| train loss | `4.989364` | `98.802982` | `4.931408` |
| score | `0.454835` | `0.464290` | `0.447440` |
| total NLL | `3.256910` | `3.256777` | `3.288182` |
| marker NLL | `1.124705` | `1.128241` | `1.159789` |
| time NLL | `2.132204` | `2.128536` | `2.128393` |
| quantity MAE | `5.265013` | `5.017135` | `4.951339` |
| quantity RMSE | `7.067073` | `6.785914` | `6.439588` |
| quantity WAPE | `0.368440` | `0.351094` | `0.346493` |
| log2 quantity MAE | `0.571535` | `0.549313` | `0.538954` |
| magnitude loss | `0.415787` | `24.640870` | `0.377636` |
| mark accuracy | `47.333%` | `48.333%` | `46.667%` |
| DT MAE | `1.323375` | `1.402583` | `1.427573` |
| normalized target abs p99 | `3.279879` | `25.059479` | `2.729008` |
| scale p01 | `6.777691` | `0.471415` | `5.602724` |
| non-finite target count | `0` | `0` | `0` |

Validation context count는 `n=1: 2`, `n=2-4: 250`, `n=5-8: 48`,
`n>=9: 0`으로 합계 `300`이다. `n=1`은 두 건뿐이므로 별도 성능 근거로 사용하지
않는다.

## Held-Out Test Smoke 결과

Test 수치는 export와 finite 여부 확인용이며 모델 선택에 사용하지 않는다.

| Metric | Q0 global | Q1 causal RevIN | Q2 shrinkage RevIN |
| --- | ---: | ---: | ---: |
| total NLL | `3.167962` | `3.167433` | `3.186840` |
| quantity MAE | `4.920631` | `5.074657` | `4.796129` |
| quantity RMSE | `6.552651` | `6.586758` | `6.123572` |
| quantity WAPE | `0.353409` | `0.364472` | `0.344467` |
| log2 quantity MAE | `0.565853` | `0.584968` | `0.551051` |
| magnitude loss | `0.377973` | `38.034946` | `0.365144` |
| mark accuracy | `52.667%` | `49.333%` | `52.333%` |
| DT MAE | `1.167884` | `1.200100` | `1.216596` |
| normalized target abs p99 | `2.837249` | `1268.074251` | `2.475928` |
| scale p01 | `6.777691` | `0.003162` | `5.535644` |
| non-finite target count | `0` | `0` | `0` |

Test context count는 `n=1: 3`, `n=2-4: 223`, `n=5-8: 74`, `n>=9: 0`으로
합계 `300`이다. e1이므로 `best_val_nll`, `best_score`, `final`은 같은 epoch와
checkpoint를 가리키며 세 selection row는 `selection` 필드 외에 동일하다.

## Scale-Wise Quantity MAE

| Split / bucket | Q0 | Q1 | Q2 | Count / share |
| --- | ---: | ---: | ---: | ---: |
| Validation `1-9` | `4.672331` | `5.241086` | `6.669998` | `78 / 26.0%` |
| Validation `10-99` | `5.473252` | `4.938450` | `4.347481` | `222 / 74.0%` |
| Test `1-9` | `4.795723` | `6.250209` | `7.020659` | `76 / 25.3%` |
| Test `10-99` | `4.963011` | `4.675810` | `4.041383` | `224 / 74.7%` |

- Q2의 overall MAE 이점은 표본의 약 `74%`를 차지하는 `10-99` bucket에서 발생
- Q2는 Q0 대비 `1-9` MAE가 validation `+42.75%`, test `+46.39%` 악화
- Q2는 Q0 대비 `10-99` MAE가 validation `-20.57%`, test `-18.57%` 개선
- top-20 smoke에는 `100+` target이 한 건도 없어 tail 성능 근거가 없음
- 빈 `100+` bucket의 NaN은 의도된 empty-bucket 표기이며 runtime non-finite가 아님

## Numeric 및 Artifact 검산

- 세 run 모두 `NLL = marker NLL + time NLL`이 `3.2e-8` 이하 오차로 성립
- validation/test context count와 scale bucket count가 각각 정확히 `300`
- scale bucket count 가중 MAE가 summary MAE와 `7.4e-8` 이하 오차로 일치
- Q0/Q1/Q2 모두 pre-clamp negative prediction share `0`
- checkpoint, history, summary/test summary, scale-wise tables, report와 plots 생성 확인
- runtime log에 Traceback, RuntimeError, CUDA error/OOM, NaN/Inf loss 없음
- e1 learning curve는 단일 point를 line-only로 그려 실선이 보이지 않으므로 추세
  증거로 사용할 수 없고, scale-wise plot만 표와 같은 방향을 확인

## 판정

- 5090 actual-data integration smoke gate: `PASS`
- matched split/statistics/artifact contract: `PASS`
- Q1 numeric diagnostic: `finite, but material scale-collapse warning`
- Q2 shrinkage stabilization: `observed at integration level`
- 모델 성능 및 RevIN benefit: `not evaluated`

Q1은 모든 값이 finite해서 smoke gate 자체는 통과했지만, 짧거나 상수인 context에서
`scale=sqrt(eps)=0.003162`까지 내려갔다. 그 결과 normalized target p99와 normalized
Huber loss가 크게 증가했다. 이는 plain causal RevIN의 예상된 zero-variance failure
mode가 실제 DataLoader에서도 재현된 것으로 기록한다.

Q2는 train loss와 magnitude loss가 Q0 수준을 유지하고 normalized target tail을
안정화했다. 다만 e1에서 Q2의 낮은 수량 `1-9` 오차가 크게 악화됐고 validation NLL
차이는 주로 marker NLL에서 발생했다. 수렴 전 단일 epoch이고 `100+` tail 표본이 없어
Q2를 선택하거나 RevIN benefit을 주장하지 않는다.

## 다음 작업

1. 5090에서 matched Q0/Q1/Q2 Intermittent seed-42 e50 validation-only screening 실행
2. frozen V2 validation reference 대비 quantity, short-context, mark/time, numeric gate 판정
3. Q1은 canonical diagnostic으로 유지하고 Q2를 primary stabilized candidate로 평가
4. validation decision 전 held-out test artifact를 열지 않음
