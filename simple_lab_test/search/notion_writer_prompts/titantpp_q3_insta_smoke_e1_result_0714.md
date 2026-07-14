# TitanTPP Q3 Factorial Instacart Top-20 e1 Smoke 결과

기존 Notion 세부 페이지를 갱신한다.

- 상위 history: `https://app.notion.com/p/2e8bbe40561380a88b5eef94e834892e`
- 결과 페이지: `https://app.notion.com/p/39cbbe4056138110865dd2ec2f6fae3d`
- 상위 Step: `Step 1. Q3 Factorial Instacart Top-20 e1 Smoke`
- 새 페이지를 만들지 않는다.

## 실행 정보

- 시작/종료: `2026-07-14 08:45:33 / 08:45:53 KST`
- 실행 서버: `5090`, host `RTX5090-server`
- GPU/runtime: NVIDIA GeForce RTX 5090, PyTorch `2.11.0+cu130`, CUDA `13.0`
- tmux session: `titantpp_q3_insta_e1_0714`
- source revision: `d552b7749c0e3836c277338dc44d82de50589e82`
- Q3 implementation revision: `14c297892ac725e272fb610772dd555d948ef055`
- artifact: `search_artifacts/model_enhancement_titantpp_q3_insta_smoke_e1_0714`
- 완료 상태: `SMOKE_SUCCESS`, aggregate/variant exit code 모두 `0`
- 분석 시각: `2026-07-14 13:15 KST`

## 분석 범위

Protocol 순서대로 아래 artifact를 확인했다.

1. Root/variant manifest와 source sync manifest
2. Root/variant log와 status
3. Validation summary
4. Held-out test summary의 export/finite 계약
5. Epoch history
6. Validation/test scale-wise metrics
7. Generated report
8. Plot
9. Checkpoint와 resume/cache identity

이 실험은 actual-data integration smoke다. e1 validation 수치와 test export는
성능 순위, Q3 효과, 후보 선택에 사용하지 않는다.

## 실행 및 데이터 계약

- Instacart top-20 fixed split, 총 `2,000 rows / 20 series`
- DataLoader samples: train `1,380`, validation `300`, test `300`
- model/candidate: `TitanTPP / small_lmm`
- epoch/seed/LR/batch: `1 / 42 / 1e-3 / 16`
- lookback/max sequence: `10 weeks / 16`
- decoder/normalization: `direct_raw_qty / causal_shrinkage_revin`
- train-only raw count/mean/std: `1,400 / 13.77 / 6.777691347354202`
- requested sigma floor: `0.0550124034288891`
- actual effective floor: `0.0067776913473542024`
- quantity reconstruction max error: `1.42109e-14`

Requested sigma floor는 CLI identity 값이고, actual effective floor는 Instacart
train-global std로 계산한 `max(0.001*std, 1e-4)`다. 네 variant는 동일한 train-only
통계와 effective floor를 사용했으며 validation/test row는 통계에 포함되지 않았다.

## Validation 통합 지표

아래 수치는 finite/export 확인용이며 variant 순위를 뜻하지 않는다.

| Variant | NLL | Mark acc | DT MAE | Qty MAE | Log2 MAE | Magnitude loss | Log aux |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q2 | 3.307908 | 46.667% | 1.600904 | 4.998507 | 0.542354 | 0.383444 | 0.000000 |
| Q3a | 3.277243 | 46.667% | 1.443068 | 4.972133 | 0.539305 | 0.378849 | 0.000000 |
| Q3b | 3.253297 | 46.333% | 1.346864 | 5.025736 | 0.544206 | 0.386639 | 0.220780 |
| Q3c | 3.256368 | 46.667% | 1.301929 | 4.955254 | 0.538276 | 0.375911 | 0.221140 |

- Q2/Q3a의 train/validation log auxiliary는 정확히 `0`이다.
- Q3b/Q3c의 log auxiliary는 positive finite다.
- History의 `NLL = marker NLL + time NLL` 최대 차이는 `3.50e-8`이다.
- Score는 `mark_acc - 0.01*dt_mae - 0.001*qty_mae`와 정확히 일치한다.
- History와 summary의 대응 값은 exact match다.

## Held-out Test Export 확인

`best_val_nll`, `best_score`, `final`은 e1에서 같은 checkpoint state이므로 모든 test
selection 값이 동일하다. 아래는 `best_val_nll` export만 표시한다.

| Variant | NLL | Marker NLL | Time NLL | Mark acc | DT MAE | Qty MAE | Log2 MAE | Log aux |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q2 | 3.179842 | 1.131791 | 2.048051 | 52.333% | 1.356881 | 4.812427 | 0.554174 | 0.000000 |
| Q3a | 3.182994 | 1.146202 | 2.036792 | 52.333% | 1.225982 | 4.751106 | 0.546272 | 0.000000 |
| Q3b | 3.173150 | 1.127271 | 2.045879 | 52.333% | 1.179200 | 4.811877 | 0.553641 | 0.236168 |
| Q3c | 3.181095 | 1.121946 | 2.059150 | 52.333% | 1.155615 | 4.744576 | 0.545708 | 0.236131 |

- 적용 가능한 test 숫자는 모두 finite다.
- Test NLL split 재합산 최대 차이는 `6.36e-9`다.
- Direct magnitude branch에서 legacy `value_mae`는 코드 계약상 `NaN/N/A`이며
  raw quantity metric으로 재해석하지 않는다.
- 이 표는 held-out export가 생성되고 finite라는 확인에만 사용하며 후보 선택에는
  사용하지 않는다.

## Scale-wise 및 Plot

- Validation bucket count: `1-9=78`, `10-99=222`, 나머지 `0`
- Test bucket count: `1-9=76`, `10-99=224`, 나머지 `0`
- 각 split count 합은 `300`, share 합은 `1`이다.
- Bucket 가중 quantity MAE, DT MAE, mark accuracy, log error는 overall metric과
  일치한다. 최대 절대 차이는 validation `8.9e-8`, test `1.03e-7`이다.
- `100+` empty bucket의 NaN은 의도된 N/A이며 model non-finite가 아니다.
- Variant별 scale plot 3개와 learning curve 1개, 총 PNG `16`개가 정상 생성됐다.
- e1 learning curve는 단일 점을 marker 없이 line으로 그려 선이 보이지 않는다.
  파일 손상이나 학습 실패로 보지 않는다.

## Checkpoint 및 Manifest 정합성

- Variant별 `best_val_nll`, `best_score`, `final`, `last_epoch_state`가 존재한다.
- e1이므로 세 selection checkpoint는 variant 내부에서 tensor-level exact match다.
- Resume state의 current/best states는 대응 checkpoint와 exact match다.
- 네 variant는 tensor `40`개, parameter `77,626`개, 동일 key/shape를 사용한다.
- 모든 checkpoint tensor가 finite다.
- `magnitude_head`, `magnitude_input_proj`가 존재하고 legacy `value_head`는 없다.
- 허용된 gradient/aux factor와 base path를 제외한 run config는 동일하다.

Root manifest의 `expected_parameter_count=78,111`은 CUDA synthetic gate의
`num_marks=12` 참조값을 복사한 메타데이터 오기다. Actual Instacart checkpoint는
PAD 포함 `num_marks=7`이며, 차이 `485`는 다음과 정확히 일치한다.

```text
(12 - 7) * (mark_emb_dim 32 + mark_head_weight 64 + bias 1) = 485
78,111 - 485 = 77,626
```

따라서 state 누락이나 variant budget 불일치는 아니다. 원 artifact는 변경하지 않고
재실행 runner에 actual expected count `77,626`, synthetic reference `78,111`을 분리해
기록하도록 수정했다.

## 판정

- actual-data backward/runtime: `PASS`
- factorial config와 loss routing: `PASS`
- checkpoint/resume/cache identity: `PASS`
- summary/history/scale-wise/report/plot contract: `PASS`
- parameter-count manifest metadata: `corrected in runner; non-blocking`
- 종합 integration gate: **PASS**
- 성능 우위 및 후보 선택: **not evaluated**

Q3 factorial branch는 Intermittent seed-42 e50 validation-only screening을 준비할 수
있다. 이 결과만으로 Q3a/Q3b/Q3c 중 하나를 선호하지 않으며 multi-seed와 held-out
선택은 계속 잠근다.

## 다음 작업

1. Q2/Q3a/Q3b/Q3c Intermittent seed-42 e50 runner, contract, 시작 기록 준비
2. 5090 source checksum, CUDA/data preflight 후 tmux 실행
3. 사용자 요청 시 완료 여부를 한 번 확인하고 artifact 동기화
4. Validation-only artifact gate와 factorial mechanism 분석
5. Full gate 통과 후보가 있을 때만 strict multi-seed 준비
