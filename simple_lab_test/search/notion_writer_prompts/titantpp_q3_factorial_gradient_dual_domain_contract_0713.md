# TitanTPP Q3 Factorial Gradient Routing And Dual-Domain Loss Contract

## Notion 위치

- 상위 페이지: `5. Model Design Enhancement`
- 날짜 섹션: `2026-07-13 | Direct Magnitude Regression과 RevIN Track`
- History 제목: `Step 12. Q3 Gradient Routing과 Dual-Domain Loss Contract`
- 상세 페이지 제목:
  `TitanTPP Q3 Factorial Gradient Routing And Dual-Domain Loss Contract`
- 작성된 상세 페이지:
  `https://app.notion.com/p/39cbbe40561381cbb5f3c3d2d226a0d7`
- 현재 상태: `CUDA·Instacart integration PASS; Intermittent seed-42 runner와 acceptance contract 준비 완료; 실행 전`

## 작성 목적

Q2는 raw quantity MAE와 짧은 history MAE를 크게 개선했지만 log2 MAE,
validation의 대부분을 차지하는 `1-9` 구간, mark accuracy를 동시에 보호하지
못했다. 다음 모델은 Q2의 shrinkage RevIN을 바꾸지 않고 두 실패 가설을 분리한다.

1. magnitude loss가 shared Titan encoder를 압박해 marker representation을 훼손했는지
2. raw absolute error 중심 loss가 저수량 예측을 충분히 보호하지 못했는지

한 번에 구조를 모두 바꾸지 않고 두 요인을 `2 x 2` factorial로 비교한다.

## 기준 결과

| Metric | V2 | Q2 |
| --- | ---: | ---: |
| Overall raw MAE | `3.060182` | `2.606458` |
| History `<=4` raw MAE | `2.296124` | `1.955420` |
| Log2 MAE | `0.588742` | `0.631778` |
| Mark accuracy | `57.249%` | `53.996%` |
| `1-9` raw MAE | `0.979752` | `1.054374` |
| Predicted mark-0 share | `45.030%` | `61.235%` |
| Mark-1 recall | `49.616%` | `17.730%` |

Q2는 overall/short raw MAE를 `14.827%/14.838%` 개선했지만 log2 MAE
`7.310%`, `1-9` MAE `7.616%` 악화와 mark accuracy `3.253%p` 하락으로
seed-42 gate를 실패했다. Held-out test는 계속 잠근다.

## Q3 Variant 계약

| Variant | Magnitude-to-encoder gradient | Log2 auxiliary | 목적 |
| --- | --- | --- | --- |
| Q2 control | coupled | off | 같은 코드에서 다시 실행하는 control |
| Q3a | detached | off | encoder gradient 간섭만 분리 |
| Q3b | coupled | on | 저수량 loss 효과만 분리 |
| Q3c | detached | on | 두 효과의 결합과 interaction 확인 |

공통 조건:

- `qty_decoder_mode=direct_raw_qty`
- `magnitude_norm_mode=causal_shrinkage_revin`
- Q2 `k=8`, raw global moments, sigma floor 그대로 유지
- 동일 Titan `small_lmm`, parameter count, seeded initialization, forward output
- fixed split, target-only, plain CE, e50, seed 42
- V3/V5, ordinal, class prior, statistic context, learnable affine 비활성
- primary checkpoint는 계속 `best_val_nll`

## Gradient Routing

새 옵션은 V3의 `value_encoder_gradient_mode`를 재사용하지 않고 direct magnitude
전용으로 분리한다.

```text
magnitude_encoder_gradient_mode = coupled | detached

coupled:  h_mag = h_j
detached: h_mag = stop_gradient(h_j)

u_hat = magnitude_head(h_mag)
```

Detached에서는 magnitude head는 계속 학습하지만 magnitude loss가 encoder와
magnitude input projection으로 전달되지 않는다. 반면 marker/time NLL은 기존처럼
encoder와 input projection을 학습하므로 과거 raw quantity feature는 제거되지 않는다.

## Dual-Domain Loss

```text
u_target = (q_target - center) / scale
q_affine = center + scale * u_hat

L_norm = Huber(u_hat, u_target)
L_raw  = Huber(q_affine, q_target)
L_log  = Huber(log2(max(q_affine, 1)), log2(max(q_target, 1)))

L_total = marker_ce + time_nll
        + 1.00 * L_norm
        + 0.25 * L_raw
        + 0.25 * L_log
```

- Q2/Q3a는 `L_log=0`
- Q3b/Q3c는 `lambda_log_qty=0.25`
- Huber delta `1`, log floor `1` 고정
- log2는 auxiliary error에만 사용하며 입력·RevIN 통계·decoder target은 raw 유지
- log floor 아래에서는 기존 unclamped raw loss가 음수·저수량 예측을 복구
- probabilistic NLL은 계속 `marker NLL + time NLL`

## 구현 전 Acceptance

1. 네 모델의 parameter key/count/initialization과 forward output exact match
2. Q2/Q3a scalar loss exact match, Q3b/Q3c scalar loss exact match
3. Detached magnitude loss는 magnitude head만 학습
4. Coupled magnitude loss는 기존 Q2 encoder/input gradient 유지
5. Marker/time NLL gradient route 불변
6. Masked log loss 수식과 target/padding isolation 검증
7. Negative affine 예측도 raw loss gradient 유지
8. CLI, path, manifest, checkpoint, cache, resume, history, summary identity 분리
9. Local focused/full test 이후 5090 CUDA model-test
10. CUDA 통과 후 Instacart top-20 e1 fixed-split smoke

## Seed-42 Validation Gate

같은 코드 revision에서 Q2/Q3a/Q3b/Q3c를 모두 실행한다. Q3a/Q3b가 실패해도
interaction 가능성이 있으므로 Q3c를 중간에 중단하지 않는다.

Fresh Q2 reproduction:

- frozen Q2 대비 total NLL/raw MAE/log2 MAE 차이 `<=1%`
- mark accuracy 차이 `<=0.25%p`
- sample count, train raw moments, config, checkpoint policy 일치

Full candidate gate:

| Gate | 기준 |
| --- | ---: |
| Overall raw MAE | `<=2.736781` |
| History `<=4` raw MAE | `<=2.053191` |
| Log2 MAE | `<=0.600517` |
| `1-9` raw MAE | `<=0.999348` |
| Marker NLL | `<=1.001186` |
| Total NLL | `<=5.694853` |
| Time NLL | `<=4.698623` |
| Mark accuracy | `>=56.999%` |
| DT MAE | `<=42.905873` |
| Predicted mark-0 absolute share error | `<=5.850%p` |
| Mark-1 recall | `>=44.616%` |
| Pre-clamp negative share | `<=1%` |

그 외 validation share `>=5%` quantity bucket은 V2 대비 `5%` 이상 악화되면
실패한다. 모든 loss, prediction, context, gradient는 finite여야 한다.

## 해석과 승격 규칙

- Q3a가 V2-Q2 mark-accuracy gap의 절반 이상을 회복하면 shared-gradient
  interference 근거로 본다.
- Q3b가 Q2 log2 MAE를 `5%` 이상 개선하고 `1-9` gate를 통과하면 low-scale
  auxiliary 효과로 본다.
- `interaction=(Q3c-Q3a)-(Q3b-Q2)`를 주요 metric별로 기록한다.
- Full gate를 통과한 모델 중 single intervention을 우선한다.
- Q3a와 Q3b가 모두 통과하면 loss hyperparameter가 없는 Q3a를 우선한다.
- 결합이 필요한 경우에만 Q3c를 선택한다.
- 모두 실패하면 V2 baseline을 유지한다.

선택 모델만 V2/Q2와 seeds `42,52,62`, e50 strict matched comparison으로
확장한다. Multi-seed gate를 통과하기 전에는 held-out test를 열지 않는다.

## 현재 상태와 다음 작업

- 완료: architecture/loss/gradient/artifact/acceptance contract, Q3 구현, focused
  `19/19`, search 전체 `104/104`, CPU/CUDA model-test, Instacart top-20 e1
  actual-data integration gate
- 준비 완료: Intermittent fresh Q2/Q3a/Q3b/Q3c seed-42 e50 runner,
  frozen V2/Q2 reference identity와 runtime `acceptance_contract.json`
- 미완료: 5090 source sync/preflight, Intermittent seed-42 실행과 validation 판정
- 다음: 준비 commit을 5090에 checksum 동기화하고 CUDA/data/reference SHA
  preflight 후 tmux 실행

## Intermittent 실행 계약

- runner:
  `simple_lab_test/search/scripts/run_titantpp_q3_inter_seed42_e50_0714.sh`
- artifact:
  `search_artifacts/model_enhancement_titantpp_q3_inter_seed42_e50_0714`
- tmux: `titantpp_q3_inter_e50_0714`
- Notion start record:
  `https://app.notion.com/p/39dbbe405613814ab90acb3f61406daf`
- fresh Q2와 Q3a/Q3b/Q3c를 `e50 / seed 42 / batch 128`로 모두 실행
- V2 checkpoint, five fixed-split source files, frozen Q2 summary SHA를 학습 전 검증
- unrounded baseline과 gate ceiling은 runtime `acceptance_contract.json`에 저장
- seed-42 판정 전 merged/test/report/test-plot artifact는 읽지 않음
- Q3a/Q3b 결과와 관계없이 Q3c까지 완료한 뒤 interaction을 계산

## 로컬 근거

- `.agents/results/architecture/adr-titantpp-q3-factorial-gradient-dual-domain.md`
- `.agents/results/architecture/adr-titantpp-raw-quantity-revin-q0-q1-q2.md`
- `simple_lab_test/search/model_enhancement_strategy.md`
- `models/RMTPPs/TitanTPP.py`
- `simple_lab_test/search/tests/test_titantpp_direct_raw_quantity.py`
- `simple_lab_test/search/scripts/run_titantpp_q3_inter_seed42_e50_0714.sh`
- `simple_lab_test/search/notion_writer_prompts/titantpp_q3_inter_seed42_e50_start_0714.md`
