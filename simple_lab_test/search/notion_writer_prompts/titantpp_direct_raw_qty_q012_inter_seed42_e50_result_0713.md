# TitanTPP Direct Raw Quantity Q0/Q1/Q2 Intermittent Seed-42 e50 결과

기존 Notion 페이지를 갱신한다.

- 상위 history: `https://app.notion.com/p/2e8bbe40561380a88b5eef94e834892e`
- 결과 페이지: `https://app.notion.com/p/39cbbe405613812b8a44eba91ea82e92`
- 관련 contract: `https://app.notion.com/p/39cbbe40561381dda378d65257d6719c`
- 상위 Step: `Step 11. Q0/Q1/Q2 Intermittent Seed-42 e50 Validation-Only Screening`

새 페이지를 만들지 않는다. Model Enhancement 결과이므로 `2. Confirm and Refine
Topic`에는 작성하지 않는다. 확인된 수치와 해석을 분리하고, held-out test 수치는
작성하지 않는다.

## 실행 상태

- 서버 / tmux: `5090` / `inter_raw_q012_e50_0713`
- 시작 / 종료: `2026-07-13 17:14:05 / 17:37:12 KST`
- 전체 runtime: `23m 07s`
- completion marker: `SCREENING_SUCCESS`
- Q0/Q1/Q2 exit code: 모두 `0`
- 완료 epoch: 모두 `50/50`
- best validation NLL epoch: Q0 `48`, Q1 `42`, Q2 `46`
- NaN / Inf / Traceback / RuntimeError / CUDA OOM: 없음
- held-out lock: 유지. `test_*`, merged `runs.csv`, `paper_outputs/report.md`,
  test plot의 내용은 열지 않음

## Matched 계약 확인

- dataset / split: `intermittent / fixed`
- model / candidate: `TitanTPP / small_lmm`
- seed / epochs: `42 / 50`
- LR / batch: `1e-3 / 128`
- lookback / max sequence: `52 / 16`
- decoder / domain: `direct_raw_qty / raw_qty`
- train loss scope / loss mode: `target_only / hybrid`
- marker objective: plain CE
- value/encoder gradient route: `coupled / coupled`
- Q0/Q1/Q2는 normalization mode만 `global`, `causal_revin`,
  `causal_shrinkage_revin`으로 다름
- train / validation / test samples: `136,256 / 41,901 / 41,344`
- normalization statistics source: fixed train events `159,643`
- raw train mean / std / Q2 sigma floor: `6.845856 / 55.012403 / 0.055012`

## V2 기준선

Frozen V2 `best_val_nll`, epoch `19`를 validation target `41,901`개에만
재평가했다. Manifest는 `held_out_test_read=false`를 기록한다.

| Metric | V2 |
| --- | ---: |
| Total NLL | `5.666520` |
| Marker NLL | `0.991274` |
| Time NLL | `4.675246` |
| Raw quantity MAE | `3.060182` |
| History count <=4 raw MAE | `2.296124` |
| Log2 quantity MAE | `0.588742` |
| Mark accuracy | `57.249%` |
| DT MAE | `42.064581` |

## Validation 결과

모든 값은 각 variant의 `best_val_nll` checkpoint 기준이다. Change는 V2 대비이며,
raw MAE는 개선율, 나머지 regression은 양수가 악화다.

| Metric | Q0 global | Q1 causal RevIN | Q2 shrinkage RevIN |
| --- | ---: | ---: | ---: |
| Best epoch | `48` | `42` | `46` |
| Total NLL | `5.611214` (`-0.976%`) | `5.757298` (`+1.602%`) | `5.625528` (`-0.723%`) |
| Marker NLL | `0.994210` (`+0.296%`) | `1.061742` (`+7.109%`) | `0.991452` (`+0.018%`) |
| Time NLL | `4.617004` (`-1.246%`) | `4.695556` (`+0.434%`) | `4.634076` (`-0.881%`) |
| Raw quantity MAE | `2.820415` (`7.835%` 개선) | `2.639700` (`13.740%` 개선) | `2.606458` (`14.827%` 개선) |
| History <=4 raw MAE | `2.224180` (`3.133%` 개선) | `1.945620` (`15.265%` 개선) | `1.955420` (`14.838%` 개선) |
| Log2 quantity MAE | `0.638690` (`+8.484%`) | `0.596929` (`+1.391%`) | `0.631778` (`+7.310%`) |
| Mark accuracy gap | `-2.809%p` | `-2.814%p` | `-3.253%p` |
| DT MAE change | `-1.707%` | `+2.858%` | `-0.946%` |

## Validation Gate

| Gate | Q0 | Q1 | Q2 |
| --- | --- | --- | --- |
| Overall raw MAE improvement >=3% | PASS | PASS | PASS |
| History <=4 raw MAE improvement >=3% | PASS | PASS | PASS |
| Log2 MAE regression <=2% | FAIL | PASS | FAIL |
| Share >=5% bucket regression <=5% | FAIL | PASS | FAIL |
| Marker NLL regression <=1% | PASS | FAIL | PASS |
| Total NLL regression <=0.5% | PASS | FAIL | PASS |
| Time NLL regression <=0.5% | PASS | PASS | PASS |
| Mark accuracy gap >=-0.25%p | FAIL | FAIL | FAIL |
| DT MAE regression <=2% | PASS | FAIL | PASS |
| Numeric safety | PASS | PASS | PASS |
| Overall eligibility | **FAIL** | **FAIL** | **FAIL** |

## Scale-Wise Quantity Safety

Validation share가 5% 이상인 bucket만 gate에 사용했다.

| Bucket | Share | Q0 MAE change | Q1 MAE change | Q2 MAE change |
| --- | ---: | ---: | ---: | ---: |
| `1-9` | `88.666%` | `+6.266%` FAIL | `+1.518%` PASS | `+7.616%` FAIL |
| `10-99` | `10.723%` | `-7.031%` PASS | `-8.555%` PASS | `-5.606%` PASS |

Q0와 Q2의 overall raw MAE 개선은 `10-99`와 작은 tail의 큰 absolute error 감소가
반영된 결과다. Validation의 대부분인 `1-9`에서는 두 모델 모두 V2보다 나빠졌다.

## RevIN과 Numeric 진단

- Q1은 Q0 대비 overall raw MAE `6.407%`, history `<=4` raw MAE `12.524%`,
  log2 MAE `6.539%`를 개선했다.
- Q2는 Q0 대비 overall raw MAE `7.586%`, history `<=4` raw MAE `12.084%`,
  log2 MAE `1.082%`를 개선했다.
- Q1 scale p01은 `0.003162`, normalized-target abs p99는 `1897.3666`,
  magnitude loss는 `183.5540`이다. 값은 finite지만 canonical RevIN의 scale-collapse가
  실제 e50에서도 지속됐다.
- Q2 scale p01은 `35.3003`, normalized-target abs p99는 `0.6668`, magnitude
  loss는 `0.00673`으로 Q1의 numeric tail을 안정화했다.
- 세 variant 모두 pre-clamp negative prediction share와 normalized-target non-finite
  count가 `0`이다.

## Marker 진단

- 실제 validation mark-0 share는 `41.180%`다.
- predicted mark-0 share는 Q0 `59.154%`, Q1 `60.462%`, Q2 `61.235%`다.
- mark-1 recall은 Q0 `20.834%`, Q1 `25.142%`, Q2 `17.730%`다.
- 세 variant 모두 raw quantity objective를 shared encoder에 coupled로 전달하면서
  mark-0 쏠림과 overall mark accuracy 하락을 보였다. 한 seed 결과이므로 인과로
  확정하지 않고 gradient interference 가설로 기록한다.

## 판정

- Q0 seed-42 candidate eligibility: `FAIL`
- Q1 seed-42 candidate eligibility: `FAIL`
- Q2 seed-42 candidate eligibility: `FAIL`
- Raw MAE와 short-context MAE 개선만으로 RevIN benefit을 주장하지 않는다.
- Q1은 canonical RevIN scale-collapse diagnostic으로 종료한다.
- Q2는 Q1보다 numeric하게 안정적이고 raw MAE가 가장 낮지만 log2/low-quantity와
  marker safety를 함께 실패했으므로 승격하지 않는다.
- Q0/Q1/Q2 matched multi-seed를 실행하지 않는다.
- held-out test를 unlock하지 않고 Intermittent baseline은 V2를 유지한다.

## 다음 판단

Q2b detached magnitude-to-encoder route는 marker interference만 진단할 수 있고,
현재 Q2의 log2 MAE와 `1-9` bucket 실패를 직접 해결하지 못한다. 따라서 즉시
multi-seed로 가지 않고 다음 두 문제를 함께 다루는 후속 구조 계약을 먼저 설계한다.

1. magnitude gradient가 marker representation에 주는 영향을 차단하거나 제한
2. raw tail 개선을 유지하면서 low-quantity/log2 error를 보호하는 dual-domain objective

V5b class-prior correction은 direct raw branch와 분리된 fallback으로 유지한다.

## Artifact

```text
server: /home/leekwanhyeong/workspace/paper_research/search_artifacts/model_enhancement_direct_raw_qty_q012_inter_seed42_e50_0713
local: /Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_direct_raw_qty_q012_inter_seed42_e50_0713
V2 reference: /Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_v2_inter_validation_reference_raw_q012_0713
```
