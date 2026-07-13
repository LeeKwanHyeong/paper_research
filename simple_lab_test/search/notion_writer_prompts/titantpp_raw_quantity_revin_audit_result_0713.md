# TitanTPP Raw-Quantity RevIN Train-Only Audit And Q2 Constants Freeze

## Notion 위치

- 상위 페이지: `5. Model Design Enhancement`
- 날짜 섹션: `2026-07-13`
- 세부 단계: `Intermittent train-only raw history, variance, tail audit and Q2 constants freeze`
- 같은 제목의 상세 페이지를 검색해 기존 페이지가 있으면 업데이트하고, 없으면 상위 페이지 아래에 생성한다.

## 실행 정보

- 상태: `completed`
- 시작: `2026-07-13 15:10:45 KST`
- 종료: `2026-07-13 15:10:48 KST`
- 서버: `5090` (`RTX5090-server`)
- tmux: `titantpp_raw_revin_audit_0713`
- Python: `/opt/miniconda3/envs/ai_env/bin/python`
- split: fixed `train` only
- validation/test read: `false`
- lookback: `52 weeks`
- max sequence length: `16` including target

## 실행 목적

M0 log2 direct regression 실패를 raw-quantity RevIN 실패로 일반화하지 않고, 실제 raw 수량에서 history length, variance, level 차이와 tail을 다시 확인했다. Q0 raw/global과 Q1 plain masked RevIN을 참조로 두고, Q2 history/global moment shrinkage에 사용할 `k`와 global moments를 validation 전에 고정하는 것이 목적이다.

## 데이터 및 계약 검증

- train event: `159,643`
- train series: `23,387`
- weekly train target: `136,256`
- null, duplicate key, non-positive/non-finite quantity, non-train row: 모두 `0`
- `demand_qty = 2 ** (mark + scale_residual)` 최대 상대 오차: `1.150e-15`
- `RMTPPWeekLookbackDataset` target 수와 context-length distribution: 일치
- quality gate: `PASS`
- loader contract gate: `PASS`

## Raw history와 tail

| 항목 | 결과 |
| --- | ---: |
| raw mean / median | `6.8459 / 2` |
| p95 / p99 / p99.9 / max | `17 / 65 / 623.938 / 5000` |
| global population std | `55.0124` |
| top 1% / 0.1% quantity sum share | `42.59% / 20.69%` |
| between-series raw variance share | `78.10%` |
| context count p50 / p95 / max | `3 / 11 / 12` |
| one-event / n<=4 context share | `22.66% / 67.63%` |
| zero-variance context / series share | `35.23% / 38.61%` |
| target outside history range | `28.70%` |

raw 수량은 낮은 값에 대부분 몰려 있지만 긴 tail이 전체 moment를 크게 끌어올린다. 동시에 분산의 대부분이 자재 간 level 차이에서 발생한다. 따라서 scale 차이를 다루는 normalization은 필요하지만, 짧고 분산이 0인 context가 많아 history mean/std만 쓰는 방식은 안전하지 않다.

## Q0과 Q1 진단

| Variant | scale p01 | scale p50 | target abs norm p99 | abs norm > 3 |
| --- | ---: | ---: | ---: | ---: |
| Q0 raw/global | `55.012403` | `55.012403` | `1.1844` | `0.4073%` |
| Q1 causal masked RevIN | `0.003162` | `0.500010` | `2846.0499` | `22.0636%` |

Q1은 one-event와 zero-variance context에서 scale이 `sqrt(1e-5)`로 축소되며 normalized target이 폭발했다. Q1은 plain RevIN failure mode를 확인하는 diagnostic ablation으로 남기고 primary candidate로 사용하지 않는다.

## Q2 candidate gate

| k | Eligible | alpha p50 | one-event scale / global std | target abs norm p99 | abs norm > 3 |
| ---: | --- | ---: | ---: | ---: | ---: |
| `1` | true | `0.7500` | `0.7085` | `1.0058` | `0.0594%` |
| `2` | true | `0.6000` | `0.8176` | `0.8776` | `0.0448%` |
| `4` | true | `0.4286` | `0.8951` | `0.8081` | `0.0448%` |
| `8` | true, selected | `0.2727` | `0.9432` | `0.7968` | `0.0514%` |
| `16` | false | `0.1579` | `0.9704` | `0.8417` | `0.0837%` |

`k=8`은 eligible 후보 중 전체 target absolute normalized p99가 가장 낮았다. `k=16`은 tail 수치가 유한하고 Q0보다 안정적이지만 median local weight `alpha`가 사전 기준 `0.25`보다 낮아 제외했다.

## Frozen Q2 constants

```text
shrinkage_k=8
revin_eps=0.00001
sigma_floor_raw=0.0550124034288891
global_mean_raw=6.8458560663480394
global_var_raw=3026.3645310228494
global_std_raw=55.0124034288891
```

결론은 Q2 normalization constants를 구현용으로 고정한다는 것이다. 이번 audit는 입력·target normalization의 수치 안정성을 확인한 것이며, Q2가 Q0/V2보다 예측 정확도가 좋다는 근거는 아니다. 모델 성능 판단은 동일 parameter, initialization, budget, split으로 구현한 뒤 validation-only screening에서 별도로 진행한다.

## 검증 및 artifact

- local focused formula test: `5 passed`
- local syntax check와 `git diff --check`: 통과
- 5090 `ai_env` 원격 pytest: `pytest` 미설치로 미실행
- 5090 실제 audit entrypoint: 정상 완료, non-finite/runtime error 없음
- artifact: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_raw_quantity_revin_audit_0713`
- source draft: `simple_lab_test/search/notion_writer_prompts/titantpp_raw_quantity_revin_audit_result_0713.md`

## 다음 작업

1. frozen constants를 사용하는 matched Q0/Q1/Q2 `direct_raw_qty` 경로 구현
2. parameter/init equivalence, target/padding leakage, normalize-denormalize round-trip focused test
3. 5090 CUDA model-test
4. Instacart top-20 e1 fixed-split smoke
5. Intermittent seed-42 e50 validation-only screening
