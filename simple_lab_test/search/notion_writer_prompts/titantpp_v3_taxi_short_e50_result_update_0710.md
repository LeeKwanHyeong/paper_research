다음 Taxi V2/V3 e50 screening 결과를 기존 Notion 문서에 업데이트해주세요.

기본 위치:
- `5. Model Design Enhancement`

기본 대상 페이지:
- `TitanTPP V3: Mark-Conditioned Value Head`
- 기존 페이지를 먼저 찾아 업데이트하고 같은 제목의 페이지를 새로 만들지 마세요.

연결할 실험 페이지:
- `TitanTPP V3 Mark-Conditioned Value Head Smoke And Short Screening e50`
- 해당 실험 페이지가 이미 있으면 Taxi 결과를 동일하게 반영해주세요.
- 실험 페이지가 없다면 이번 작업에서는 중복 페이지를 만들지 말고 기본 대상 페이지에만 반영해주세요.

업데이트 목적:
- Taxi V3 e50 실행 완료 상태를 반영합니다.
- 동일 조건의 V2 shared value head와 V3 mark-conditioned experts를 비교합니다.
- V3의 quantity 개선과 marker 성능 trade-off를 분리해 기록합니다.
- Intermittent 결과 분석 전에는 전체 V3 screening의 최종 결론을 내리지 않습니다.

## 1. Status Update

기존 상태를 다음과 같이 갱신해주세요.

- V3 implementation: `implemented`
- focused verification: `passed`
- Instacart mini smoke: `completed`
- Intermittent V2/V3 execution: `completed`, analysis pending
- Taxi V2/V3 execution: `completed`
- Taxi result analysis: `completed`
- 전체 short screening 판정: `in progress`

Taxi V3 완료 정보:
- 시작 시각: `2026-07-10 10:58:20 KST`
- 종료 시각: `2026-07-10 11:07:32 KST`
- 실행 서버: `5080`
- tmux session: `titantpp_v3_smoke_screen_e50_0710`
- conda env: `ai_env`
- dataset: `yellow_trip_hourly`
- candidate: `mid_lmm`
- epochs: `50`
- seed: `42`
- device: `cuda`

Short Screening Status 표의 Taxi V3 행을 다음과 같이 변경해주세요.

| Dataset | Version | Candidate | Epochs | Seed | Status |
| --- | --- | --- | ---: | ---: | --- |
| Yellow Trip Hourly | V3 experts | `mid_lmm` | 50 | 42 | completed at `11:07:32 KST` |

상단 callout은 아래 의미가 드러나도록 수정해주세요.

```text
Design confirmed / Implemented / Focused tests passed /
Short screening execution completed / Result analysis in progress
```

## 2. Experiment Integrity

확인된 실행 상태:
- planned `50/50` epochs completed
- held-out test evaluation completed
- `experiment_manifest.json` 생성 확인
- `logs/run.log` 정상 완료 마커 확인
- `summary.csv`, `test_summary.csv`, `histories.csv` 생성 확인
- validation/test scale-wise summary 생성 확인
- `paper_outputs/report.md`와 plots 생성 확인
- NaN, Traceback, RuntimeError, runtime ERROR 없음

artifact는 아래 순서로 확인했습니다.
1. `experiment_manifest.json`
2. `logs/run.log`
3. `leaderboard/summary.csv`
4. `leaderboard/test_summary.csv`
5. `leaderboard/histories.csv`
6. `leaderboard/scale_wise_summary.csv`
7. `leaderboard/test_scale_wise_summary.csv`
8. `paper_outputs/report.md`
9. `paper_outputs/plots/`

## 3. Fair Comparison Contract

V2와 V3의 공통 조건:
- dataset: `yellow_trip_hourly`
- model: `titantpp`
- candidate: `mid_lmm`
- seed: `42`
- epochs: `50`
- lr: `1e-3`
- batch size: `128`
- lookback: `168`
- max sequence length: `256`
- scale base: `10`
- split mode: `fixed`
- value input: `residual`
- train loss scope: `target_only`
- loss mode: `hybrid`
- value head activation: `identity`
- test-time memory: `none`

변경된 조건:
- V2: `value_head_mode=shared`
- V3: `value_head_mode=mark_conditioned_experts`

이번 비교에서 Titan backbone, LMM, marker head, time head, dataset, split, lookback 및 loss 구성은 변경하지 않았습니다.

## 4. Checkpoint Selection Rule

본문의 primary held-out test 비교는 `best_val_nll` checkpoint를 사용해주세요.

- V2 best validation NLL: `1.583074`, epoch `42`
- V3 best validation NLL: `1.598806`, epoch `46`
- final epoch 또는 `best_score` checkpoint를 primary 결과로 바꾸지 마세요.
- `best_score` 결과는 checkpoint trade-off를 설명하는 보조 관찰로만 작성해주세요.

## 5. Primary Held-out Test Result

`best_val_nll` checkpoint 기준:

| Metric | V2 shared | V3 experts | V3 change | Direction |
| --- | ---: | ---: | ---: | --- |
| Test score | `0.854053` | `0.850308` | `-0.003745` | worse |
| Total NLL | `1.633429` | `1.674005` | `+2.484%` | worse |
| Marker NLL | `0.235469` | `0.276678` | `+17.501%` | worse |
| Time NLL | `1.397960` | `1.397327` | `-0.045%` | effectively unchanged |
| Quantity MAE | `50.530877` | `46.999812` | `-6.988%` | improved |
| DT MAE | `0.798947` | `0.782031` | `-2.117%` | improved |
| Mark accuracy | `0.912574` | `0.905128` | `-0.745%p` | worse |
| Value MAE | `0.201250` | `0.178963` | `-11.075%` | improved |

## 6. Secondary Checkpoint Observation

`best_score` checkpoint에서는 V3 score가 높았습니다.

- V2 best-score test score: `0.853932`
- V3 best-score test score: `0.862135`
- V2 best-score total NLL: `1.655289`
- V3 best-score total NLL: `1.673059`
- V2 best-score marker NLL: `0.254570`
- V3 best-score marker NLL: `0.267735`
- V2 best-score quantity MAE: `44.537631`
- V3 best-score quantity MAE: `42.934047`

해석:
- V3는 score와 quantity MAE가 좋은 checkpoint를 만들 수 있었습니다.
- 그러나 해당 checkpoint에서도 total NLL과 marker NLL은 V2보다 나빴습니다.
- 따라서 checkpoint selection을 바꿔 V3가 우세하다고 표현하지 마세요.

## 7. Scale-wise Quantity Result

held-out test `best_val_nll` 기준:

| Quantity scale | Test share | V2 MAE | V3 MAE | Change | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| `1-9` | `54.23%` | `1.615639` | `1.159566` | `-28.229%` | improved |
| `10-99` | `24.33%` | `13.739594` | `14.649610` | `+6.623%` | worse |
| `100-999` | `13.89%` | `117.353863` | `136.572765` | `+16.377%` | worse |
| `1000-9999` | `7.54%` | `397.865534` | `315.980934` | `-20.581%` | improved |

Confirmed finding:
- 전체 quantity MAE는 개선됐지만 모든 scale에서 일관된 개선은 아닙니다.
- 개선은 `1-9`와 `1000-9999` scale에서 발생했습니다.
- `10-99`와 `100-999`에서는 quantity MAE가 악화됐습니다.

## 8. Mark-wise Confusion Finding

held-out test `best_val_nll` 기준 true-mark accuracy:

| True mark | V2 accuracy | V3 accuracy | Change |
| ---: | ---: | ---: | ---: |
| `0` | `96.61%` | `98.10%` | `+1.48%p` |
| `1` | `84.45%` | `79.96%` | `-4.49%p` |
| `2` | `83.33%` | `80.22%` | `-3.11%p` |
| `3` | `89.31%` | `89.00%` | `-0.32%p` |

Confirmed finding:
- V3는 가장 낮은 mark `0`의 정확도는 개선했습니다.
- mark `1`과 `2`의 정확도가 크게 하락해 전체 mark accuracy와 marker NLL을 악화시켰습니다.
- 이 패턴은 `10-99`, `100-999` quantity scale의 MAE 악화와 함께 관찰됩니다.

## 9. Interpretation

Confirmed:
- V3 mark-conditioned experts는 value MAE를 `11.08%`, 전체 quantity MAE를 `6.99%` 개선했습니다.
- time NLL은 사실상 동일하고 DT MAE는 소폭 개선됐습니다.
- marker NLL은 `17.50%` 악화됐고 mark accuracy는 `0.745%p` 감소했습니다.
- quantity MAE 개선은 scale 전반에서 일관되지 않았습니다.
- 실행과 artifact 생성은 정상 완료됐습니다.

Inferred:
- V3의 conditional residual 구조 자체는 quantity/value modeling에 유효한 신호를 보였습니다.
- 하지만 V3a가 V2와 동일하게 유지한 quantity-to-mark gradient coupling이 marker head를 계속 압박했을 가능성이 있습니다.
- mark `1`, `2`에서 발생한 분류 악화 때문에 predicted-mark expert 선택 오류와 중간 scale quantity 오차가 함께 증가한 것으로 해석할 수 있습니다.

추론임을 명확히 표시하고, causal mechanism으로 확정해 쓰지 마세요.

## 10. Taxi Screening Decision

결론:
- Taxi V3a는 `partial success`입니다.
- quantity/value modeling gate는 통과했습니다.
- marker NLL과 mark accuracy guardrail은 통과하지 못했습니다.
- 현재 단계에서 V3a를 Taxi의 V2 replacement로 채택하지 않습니다.
- Taxi baseline은 계속 V2 shared hybrid `mid_lmm`으로 유지합니다.
- Intermittent V2/V3 분석을 완료한 뒤 V3b detached-gate 진행 여부를 최종 확정합니다.

V3b 후보:
- mark-conditioned experts는 유지합니다.
- expected quantity loss를 계산할 때 mark probability를 detach합니다.
- 목표는 V3a의 quantity/value 이득을 유지하면서 marker NLL과 accuracy degradation을 줄이는 것입니다.
- V3b는 아직 구현 또는 검증된 결과가 아니라 다음 설계 가설입니다.

## 11. Limitations

- 이 결과는 seed `42` 하나의 e50 short screening입니다.
- 최종 우월성, 일반화 성능 또는 multi-seed 안정성을 주장하지 마세요.
- Intermittent V3 결과와 함께 분석하기 전에는 V3 전체 판정을 완료하지 마세요.
- V3가 Taxi marker 문제를 해결했다고 표현하지 마세요.
- overall score 하나만 사용해 V3가 우세하다고 표현하지 마세요.
- primary result는 `best_val_nll` checkpoint 기준입니다.

## 12. Artifact Paths

Local:
- V2: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_v2_taxi_short_e50_0710`
- V3: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_v3_taxi_short_e50_0710`

Server:
- V2: `~/workspace/paper_research/search_artifacts/model_enhancement_v2_taxi_short_e50_0710`
- V3: `~/workspace/paper_research/search_artifacts/model_enhancement_v3_taxi_short_e50_0710`

주요 파일:
- `leaderboard/test_summary.csv`
- `leaderboard/test_scale_wise_summary.csv`
- `leaderboard/histories.csv`
- `paper_outputs/report.md`
- `paper_outputs/plots/yellow_trip_hourly_learning_curves.png`
- `paper_outputs/plots/test/yellow_trip_hourly_best_val_nll_scale_wise_qty_errors.png`

## 13. Page Update Structure

기본 대상 페이지에 아래 순서로 반영해주세요.
1. 상단 status callout 갱신
2. Short Screening Status 표의 Taxi V3 행 완료 처리
3. `Taxi V2-vs-V3 e50 Result` 섹션 추가
4. Primary Held-out Test Result
5. Scale-wise Quantity Result
6. Mark-wise Confusion Finding
7. Confirmed Findings
8. Inferred Interpretation
9. Taxi Screening Decision
10. Limitations
11. Next Action

작성 주의사항:
- confirmed metric과 inferred explanation을 분리해주세요.
- quantity 개선과 marker 악화를 모두 같은 비중으로 보여주세요.
- V3a를 성공 또는 실패 하나로 단순화하지 말고 `partial success`로 기록해주세요.
- `best_score` checkpoint 결과를 primary 결과로 사용하지 마세요.
- 기존 V3 architecture, implementation, focused verification 섹션은 삭제하거나 축약하지 마세요.
- 실험 결과를 추가하기 위해 기존 child page나 표를 삭제하지 마세요.

Next Action:
1. Intermittent V2/V3 e50 artifact 분석
2. Intermittent와 Taxi short screening 종합 판정
3. V3b detached-gate 설계 확정
4. V3b focused implementation test
5. V3b Intermittent/Taxi short screening
