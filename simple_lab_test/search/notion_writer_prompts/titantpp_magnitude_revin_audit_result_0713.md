다음 Intermittent train-only magnitude RevIN audit 결과를 Notion에 정리해주세요.

사후 범위 정정 (`2026-07-13`): 이 audit이 제안한 M0-M4는 모두
`log2(qty)` 기반 log-domain family다. 아래 M3/M4 권고는 M0 실행 전의 설계
기록이며, M0 gate 실패 후에는 종속된 log-domain M1-M4를 실행하지 않는다.
Raw-quantity RevIN은 별도 Q0/Q1/Q2 계약이 필요한 미검증 가설로 유지한다.

## 작성 위치

- `5. Model Design Enhancement > Enhancement & Validation History`
- 기존 `TitanTPP Parallel Magnitude Decoder And Causal Shrinkage RevIN Design`
  페이지가 있으면 결과 섹션을 추가합니다.
- 기존 페이지가 아직 없으면 위 제목으로 만들고 설계 요약과 audit 결과를 함께
  작성합니다.
- Model Enhancement 결과이므로 `2. Confirm and Refine Topic`에는 작성하지 않습니다.
- 중복 페이지를 만들지 않습니다.

## 문체

- 데이터 audit 기록처럼 간결하게 작성합니다.
- 확인된 수치와 향후 모델 가설을 분리합니다.
- audit만으로 RevIN 성능이 검증됐다고 쓰지 않습니다.
- 과장된 표현이나 일반적인 GPT 문장은 넣지 않습니다.

## 상태

- 상태: `train-only audit completed; constants frozen; implementation not started`
- 실행 시각: `2026-07-13 10:31:58-10:32:00 KST`
- 실행 환경: local deterministic analysis
- source split: train only
- validation/held-out test read: `false`
- GPU/tmux 실험이 아니며 5090은 사용하지 않음

## 목적

- 실제 weekly target-only context에서 plain RevIN의 mean/std가 안정적인지 확인
- series 간 scale 차이와 series 내부 level shift를 분리
- M3/M4 shrinkage에 사용할 `k`, variance floor, exp2 clamp를 validation 전에 고정
- audit context가 실제 `RMTPPWeekLookbackDataset`과 일치하는지 검증

## Source And Contract Validation

| Item | Result |
| --- | --- |
| Decoded train rows | `159,643` |
| Series | `23,387` |
| Train events | `159,643` |
| Train next-event targets | `136,256` |
| Lookback/max sequence | `52 / 16` |
| DataLoader target count | exact match |
| DataLoader context-length distribution | exact match |
| Required nulls | `0` |
| Duplicate part/seq keys | `0` |
| Non-positive/non-finite quantity | `0` |
| Decoded non-train rows | `0` |
| Non-train context violations | `0` |
| Max `log2(qty)` reconstruction error | `0` |
| Quality gate | `PASS` |

## History Length And Variance

| Metric | Result |
| --- | --- |
| Context count p50/p95/max | `3 / 11 / 12` |
| One-event context | `22.66%` |
| Context count <=2 | `41.50%` |
| Context count <=4 | `67.63%` |
| Zero-variance context | `35.23%` |
| Train series count <=4 | `61.06%` |
| Zero-variance train series | `38.61%` |
| History std p50/p95 | `0.4949 / 1.1610` |

## Scale Heterogeneity And Level Shift

| Metric | Result |
| --- | --- |
| Train global log2 mean/std | `1.2662 / 1.4535` |
| Train log2 range | `0.0000-12.2877` |
| Between-series variance share | `73.23%` |
| Window/global mean absolute gap p50/p95 | `0.7662 / 2.6514` |
| Target/history mean absolute gap p50/p95 | `0.6000 / 1.9227` |
| Target outside history range | `28.70%` |
| Window half-shift p50/p95 | `0.5000 / 1.4534` |
| Series early/late shift p50/p95 | `0.2925 / 1.3838` |

## Shrinkage Candidate Result

| k | alpha p50 | scale p01 | one-event scale p50 | target abs(z_norm) p99 | >3 share |
| --- | --- | --- | --- | --- | --- |
| `1` | `0.7500` | `0.7017` | `1.2072` | `2.2453` | `0.0961%` |
| `2` | `0.6000` | `0.8468` | `1.3285` | `1.9222` | `0.0139%` |
| `4` | `0.4286` | `0.9871` | `1.3953` | `1.7773` | `0.0073%` |
| `8` | `0.2727` | `1.1244` | `1.4270` | `1.8125` | `0.0103%` |
| `16` | `0.1579` | `1.2421` | `1.4413` | `1.9686` | `0.0477%` |

Train-only stability rule에서 p99가 가장 낮고 scale/local-weight gate를 통과한
`k=4`를 선택했습니다.

## Frozen Constants

```text
shrinkage_k=4
magnitude_sigma_floor=0.0014535461338152059
magnitude_exp_clamp_min=-2
magnitude_exp_clamp_max=15
```

## Interpretation

- Plain M2 RevIN은 zero-variance context가 많아 primary candidate로 사용하지 않음
- M1 per-series scaler도 짧거나 zero-variance인 series가 많아 global fallback 필요
- Between-series variance가 크므로 scale normalization 가설은 타당함
- 동시에 local/early-late shift가 존재하므로 static global/per-series scaler만으로는
  부족할 가능성이 있음
- 당시 사전 결정은 M0 global baseline을 먼저 구현하고, 통과할 경우에만 M3/M4
  log-domain shrinkage 후보를 검증하는 것이었음
- 이후 M0 gate가 실패했으므로 기존 log-domain M1-M4는 실행하지 않으며, 이
  audit 결과를 raw-quantity RevIN의 성능 근거 또는 실패 근거로 사용하지 않음
- 이 audit은 normalization 안정성 근거이며 quantity MAE 개선 근거가 아님

## Artifact

```text
script: /Users/igwanhyeong/PycharmProjects/paper_research/simple_lab_test/search/analyze_magnitude_revin_audit.py
artifact: /Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_magnitude_revin_audit_0713
report: /Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_magnitude_revin_audit_0713/report.md
```

페이지 마지막 Next:

```text
Next: frozen constants를 사용하는 M0 direct log2-magnitude baseline과 shared magnitude-context contract를 구현한다.
```
