다음 TitanTPP V3a Intermittent·Taxi e50 short screening 종합 결과를 기존 Notion 문서에 업데이트해주세요.

기본 위치:
- `5. Model Design Enhancement`

기본 대상 페이지:
- `TitanTPP V3: Mark-Conditioned Value Head`
- 기존 페이지를 먼저 찾아 업데이트하고 같은 제목의 페이지를 새로 만들지 마세요.

연결할 실험 페이지:
- `TitanTPP V3 Mark-Conditioned Value Head Smoke And Short Screening e50`
- 해당 페이지가 이미 있으면 종합 결과와 최종 screening decision을 함께 반영해주세요.
- 실험 페이지가 없다면 이번 작업에서는 새 중복 페이지를 만들지 말고 기본 대상 페이지에만 반영해주세요.

업데이트 목적:
- Intermittent와 Yellow Trip Hourly의 V2 shared head와 V3a mark-conditioned experts 결과를 같은 기준으로 종합합니다.
- V3a의 공통 개선점과 공통 실패 지점을 구분합니다.
- dataset별 상반된 marker NLL 결과를 평균이나 단일 score로 덮지 않습니다.
- V3a를 최종 모델로 채택하지 않는 이유와 V3b detached-gate로 이어지는 설계 근거를 기록합니다.

## 1. Status Update

기존 상태를 다음과 같이 갱신해주세요.

- V3 architecture design: `confirmed`
- V3a implementation: `implemented`
- focused verification: `passed`
- Instacart mini smoke: `completed`
- Intermittent V2/V3 e50 execution: `completed`
- Taxi V2/V3 e50 execution: `completed`
- artifact synchronization: `completed`
- Intermittent result analysis: `completed`
- Taxi result analysis: `completed`
- V3a short screening decision: `completed`
- V3b detached-gate: `proposed`, not implemented

상단 callout은 다음 의미가 드러나도록 갱신해주세요.

```text
V3a Design Confirmed / Implemented / Focused Tests Passed /
Short Screening Completed / Combined Decision Completed /
Next Architecture: V3b Detached-Gate
```

## 2. Experiment Scope And Environment

실행 환경:
- 실행 서버: `5080`
- tmux session: `titantpp_v3_smoke_screen_e50_0710`
- conda env: `ai_env`
- device: `cuda`

실행 시간:

| Run | Start | End | Status |
| --- | --- | --- | --- |
| Intermittent V2 | `2026-07-10 10:36:12 KST` | `2026-07-10 10:42:40 KST` | completed |
| Intermittent V3a | `2026-07-10 10:42:41 KST` | `2026-07-10 10:49:10 KST` | completed |
| Taxi V2 | `2026-07-10 10:49:11 KST` | `2026-07-10 10:58:18 KST` | completed |
| Taxi V3a | `2026-07-10 10:58:20 KST` | `2026-07-10 11:07:32 KST` | completed |

실험 범위:
- datasets: `intermittent`, `yellow_trip_hourly`
- model: `titantpp`
- epochs: `50`
- seed: `42`
- lr: `1e-3`
- batch size: `128`
- split mode: `fixed`
- value input mode: `residual`
- train loss scope: `target_only`
- loss mode: `hybrid`
- value head activation: `identity`
- eval selections: `best_val_nll`, `best_score`, `final`

Dataset-specific setting:

| Dataset | Candidate | Scale base | Lookback | Max sequence length |
| --- | --- | ---: | ---: | ---: |
| Intermittent | `small_lmm` | `2` | `52` | `16` |
| Yellow Trip Hourly | `mid_lmm` | `10` | `168` | `256` |

V2/V3a에서 달라진 항목:
- V2: `value_head_mode=shared`
- V3a: `value_head_mode=mark_conditioned_experts`

변경하지 않은 항목:
- Titan backbone과 memory mode
- marker head와 continuous-time head
- hybrid loss와 `lambda_qty`
- value input 및 target-only loss scope
- dataset preprocessing와 fixed split
- lookback과 max sequence length
- checkpoint selection rule

## 3. Artifact Integrity

확인된 상태:
- planned `4/4` screening runs completed
- 각 run의 planned `50/50` epochs completed
- held-out test evaluation completed
- 모든 run에서 manifest, log, summary, test summary, histories 생성 확인
- validation/test scale-wise metrics 생성 확인
- report와 learning/scale-wise plots 생성 확인
- NaN, Traceback, RuntimeError, runtime ERROR 없음

artifact는 각 run에서 아래 순서로 확인했습니다.
1. `experiment_manifest.json`
2. `logs/run.log`
3. `leaderboard/summary.csv`
4. `leaderboard/test_summary.csv`
5. `leaderboard/histories.csv`
6. `leaderboard/scale_wise_summary.csv`
7. `leaderboard/test_scale_wise_summary.csv`
8. `paper_outputs/report.md`
9. `paper_outputs/plots/`

Data coverage:
- Intermittent train/validation/test samples: `136,256 / 41,901 / 41,344`
- Taxi train/validation/test samples: `38,393 / 8,268 / 8,327`
- Intermittent test scale bucket count 합계: `41,344`, share 합계 `1.0`
- scale-wise weighted quantity MAE를 다시 계산해 test summary의 aggregate MAE와 일치하는 것을 확인했습니다.

## 4. Checkpoint Selection Rule

본문의 primary 비교는 기존 protocol대로 `best_val_nll` checkpoint를 사용해주세요.

| Dataset | V2 best epoch | V2 validation NLL | V3a best epoch | V3a validation NLL |
| --- | ---: | ---: | ---: | ---: |
| Intermittent | `19` | `5.666520` | `47` | `5.642602` |
| Taxi | `42` | `1.583074` | `46` | `1.598806` |

작성 규칙:
- final epoch 결과를 primary result로 바꾸지 마세요.
- `best_score` 결과를 primary result로 바꾸지 마세요.
- secondary checkpoint 결과는 selection sensitivity를 설명할 때만 사용해주세요.
- held-out test를 사용해 checkpoint나 candidate를 다시 선택하지 마세요.

## 5. Combined Primary Held-out Test Result

`best_val_nll` checkpoint 기준 V3a 변화:

| Dataset | Metric | V2 shared | V3a experts | V3a change | Result |
| --- | --- | ---: | ---: | ---: | --- |
| Intermittent | Test score | `0.286767` | `0.277922` | `-3.085%` | worse |
| Intermittent | Total NLL | `5.071916` | `5.062077` | `-0.194%` | improved |
| Intermittent | Marker NLL | `1.016321` | `1.006122` | `-1.004%` | improved |
| Intermittent | Time NLL | `4.055595` | `4.055955` | `+0.009%` | unchanged |
| Intermittent | Quantity MAE | `3.528298` | `3.058517` | `-13.315%` | improved |
| Intermittent | Value MAE | `0.153685` | `0.117189` | `-23.748%` | improved |
| Intermittent | DT MAE | `25.430574` | `25.339001` | `-0.360%` | improved |
| Intermittent | Mark accuracy | `0.544601` | `0.534370` | `-1.023%p` | worse |
| Taxi | Test score | `0.854053` | `0.850308` | `-0.439%` | worse |
| Taxi | Total NLL | `1.633429` | `1.674005` | `+2.484%` | worse |
| Taxi | Marker NLL | `0.235469` | `0.276678` | `+17.501%` | worse |
| Taxi | Time NLL | `1.397960` | `1.397327` | `-0.045%` | unchanged |
| Taxi | Quantity MAE | `50.530877` | `46.999812` | `-6.988%` | improved |
| Taxi | Value MAE | `0.201250` | `0.178963` | `-11.075%` | improved |
| Taxi | DT MAE | `0.798947` | `0.782031` | `-2.117%` | improved |
| Taxi | Mark accuracy | `0.912574` | `0.905128` | `-0.745%p` | worse |

## 6. Dataset-level Interpretation

### Intermittent

Confirmed:
- total NLL은 `0.19%`, marker NLL은 `1.00%` 개선됐습니다.
- time NLL은 사실상 동일합니다.
- quantity MAE는 `13.32%`, value MAE는 `23.75%` 개선됐습니다.
- mark accuracy는 `1.023%p` 감소했고 score는 `3.09%` 감소했습니다.
- V3a best validation NLL은 epoch `47`에서 발생해 V2 epoch `19`보다 늦었습니다.
- validation final-minus-best NLL gap은 V2 `8.59%`, V3a `4.00%`로 V3a가 더 작았습니다.

Interpretation:
- Intermittent V3a는 NLL, quantity/value modeling과 late-epoch stability 측면에서 긍정적입니다.
- 하지만 argmax mark classification과 composite score guardrail은 통과하지 못했습니다.
- marker NLL 개선과 mark accuracy 악화가 동시에 발생했으므로 probability calibration과 argmax decision boundary를 분리해 해석해야 합니다.

### Taxi

Confirmed:
- quantity MAE는 `6.99%`, value MAE는 `11.08%` 개선됐습니다.
- time NLL은 사실상 동일하고 DT MAE는 소폭 개선됐습니다.
- total NLL은 `2.48%`, marker NLL은 `17.50%` 악화됐습니다.
- mark accuracy는 `0.745%p` 감소했고 score도 감소했습니다.

Interpretation:
- Taxi V3a의 conditional residual 구조는 quantity/value modeling에는 유효한 신호를 보였습니다.
- 그러나 Taxi의 primary marker guardrail을 명확하게 통과하지 못했습니다.
- V3a를 V2 Taxi baseline의 replacement로 채택할 근거는 없습니다.

## 7. Scale-wise Quantity Result

### Intermittent

held-out test `best_val_nll` 기준:

| Quantity scale | Test share | V2 MAE | V3a MAE | Change | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| `1-9` | `88.67%` | `1.061286` | `1.159043` | `+9.211%` | worse |
| `10-99` | `10.66%` | `9.787868` | `8.669663` | `-11.424%` | improved |
| `100-999` | `0.58%` | `120.963119` | `98.963354` | `-18.187%` | improved |
| `1000-9999` | `0.09%` | `912.686605` | `576.466961` | `-36.838%` | improved |

Confirmed finding:
- aggregate quantity MAE 개선은 tail quantity 구간이 주도했습니다.
- 전체 test event의 `88.67%`인 `1-9` 구간 mean MAE는 악화됐습니다.
- 따라서 Intermittent V3a를 모든 quantity scale에서 개선됐다고 표현하지 마세요.
- `100-999`, `1000-9999`는 sample 수가 각각 `241`, `38`이므로 개선 폭의 안정성을 과장하지 마세요.

### Taxi

held-out test `best_val_nll` 기준:

| Quantity scale | Test share | V2 MAE | V3a MAE | Change | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| `1-9` | `54.23%` | `1.615639` | `1.159566` | `-28.229%` | improved |
| `10-99` | `24.33%` | `13.739594` | `14.649610` | `+6.623%` | worse |
| `100-999` | `13.89%` | `117.353863` | `136.572765` | `+16.377%` | worse |
| `1000-9999` | `7.54%` | `397.865534` | `315.980934` | `-20.581%` | improved |

Confirmed finding:
- Taxi quantity 개선도 scale 전반에서 일관되지 않습니다.
- 가장 낮은 scale과 가장 높은 scale은 개선됐지만 중간 scale `10-999`는 악화됐습니다.

## 8. Mark Confusion Findings

### Intermittent

Intermittent model mark는 log2 magnitude class이고, 위 quantity scale bucket은 decimal quantity 범위입니다. 두 구분을 같은 것으로 표현하지 마세요.

주요 true-mark accuracy 변화:

| True mark | V2 accuracy | V3a accuracy | Change |
| ---: | ---: | ---: | ---: |
| `0` | `69.41%` | `85.37%` | `+15.97%p` |
| `1` | `47.62%` | `23.06%` | `-24.55%p` |
| `2` | `40.24%` | `32.72%` | `-7.52%p` |
| `3` | `51.25%` | `54.35%` | `+3.10%p` |

주요 confusion 변화:
- true mark `1`이 mark `0`으로 예측된 비율: `40.11% → 67.17%`
- true mark `2`가 mark `0`으로 예측된 비율: `11.68% → 25.34%`

Confirmed finding:
- V3a는 mark `0` 적중률을 크게 높였지만 mark `1`, `2`를 mark `0`으로 보내는 오류도 증가했습니다.
- 이 변화로 전체 marker NLL은 소폭 개선됐지만 argmax mark accuracy는 감소했습니다.

Inferred:
- V3a의 mark probability가 true class에 더 나은 확률 질량을 배분하면서도 argmax boundary는 mark `0` 쪽으로 이동했을 가능성이 있습니다.
- 이는 confusion과 NLL/accuracy 방향 차이에 기반한 해석이며 calibration 분석으로 직접 검증한 causal conclusion은 아닙니다.

### Taxi

주요 true-mark accuracy 변화:

| True mark | V2 accuracy | V3a accuracy | Change |
| ---: | ---: | ---: | ---: |
| `0` | `96.61%` | `98.10%` | `+1.48%p` |
| `1` | `84.45%` | `79.96%` | `-4.49%p` |
| `2` | `83.33%` | `80.22%` | `-3.11%p` |
| `3` | `89.31%` | `89.00%` | `-0.32%p` |

Confirmed finding:
- Taxi에서도 가장 낮은 mark 정확도는 개선됐지만 중간 mark의 정확도가 감소했습니다.
- Taxi에서는 이 변화가 marker NLL의 `17.50%` 악화와 함께 나타났습니다.

## 9. Cross-dataset Confirmed Findings

두 데이터셋에서 공통으로 확인된 내용:
- V3a는 quantity MAE를 개선했습니다.
- V3a는 value MAE를 개선했습니다.
- time NLL은 거의 변하지 않았습니다.
- DT MAE는 소폭 개선됐습니다.
- mark accuracy는 감소했습니다.
- quantity 개선은 모든 scale에서 일관되지 않았습니다.
- 낮은 mark의 예측 비중 또는 정확도가 증가하고 일부 중간 mark accuracy가 감소했습니다.

데이터셋별로 달랐던 내용:
- Intermittent total NLL과 marker NLL은 소폭 개선됐습니다.
- Taxi total NLL과 marker NLL은 악화됐습니다.
- Intermittent aggregate quantity 개선은 tail이 주도했고 가장 많은 `1-9` 구간은 악화됐습니다.
- Taxi는 `1-9`, `1000-9999`가 개선되고 `10-999`가 악화됐습니다.

## 10. Cross-dataset Inference

Inferred interpretation:
- mark-conditioned residual experts는 mark별 quantity/value residual을 분리해 학습하는 데 유효한 구조적 신호를 보였습니다.
- 하지만 V3a가 유지한 differentiable expected quantity의 quantity-to-mark gradient coupling이 mark decision boundary에 영향을 줬을 가능성이 있습니다.
- 특히 낮은 mark 방향으로의 confusion 증가가 두 데이터셋의 mark accuracy 저하와 연결됩니다.
- Taxi에서 marker NLL까지 크게 악화된 점은 이 coupling이 long-sequence/high-scale regime에서 더 위험할 가능성을 시사합니다.

작성 주의사항:
- 위 내용은 실험 결과로부터 추론한 hypothesis입니다.
- quantity loss가 marker degradation의 유일한 원인이라고 확정하지 마세요.
- class imbalance, single-seed optimization noise, checkpoint timing도 대안 설명으로 남겨주세요.

## 11. V3a Screening Gate Decision

Gate 결과:

| Gate | Intermittent | Taxi | Overall |
| --- | --- | --- | --- |
| Runtime/artifact integrity | pass | pass | pass |
| Quantity MAE | pass | pass | pass |
| Value MAE | pass | pass | pass |
| Total NLL | pass | fail | fail |
| Marker NLL | pass | fail | fail |
| Mark accuracy | fail | fail | fail |
| Scale consistency | fail | fail | fail |

최종 V3a screening decision:
- V3a는 `partial success`입니다.
- mark-conditioned expert architecture 전체를 폐기하지 않습니다.
- 현재 V3a를 V2 shared hybrid baseline의 replacement로 채택하지 않습니다.
- V2 shared hybrid는 Intermittent와 Taxi의 공식 pre-V3 baseline으로 유지합니다.
- V3a의 quantity/value 이득을 유지하면서 marker coupling을 줄이는 V3b detached-gate로 진행합니다.
- full multi-seed 본실험으로 바로 확장하지 않습니다.

## 12. Next Architecture: V3b Detached-Gate

V3b hypothesis:
- shared-plus-mark-delta experts 구조는 유지합니다.
- expected quantity 계산에 사용하는 mark probability를 detach합니다.
- quantity loss는 value experts와 quantity reconstruction을 계속 학습합니다.
- quantity loss가 mark probability를 통해 marker head로 직접 전달되는 gradient는 차단합니다.
- marker NLL과 mark accuracy는 marker CE가 중심이 되어 학습되도록 합니다.

V3b에서 변경하지 않을 항목:
- Titan backbone과 memory mode
- marker/time head architecture
- value expert architecture
- dataset, split, lookback, max sequence length
- hybrid loss coefficient와 checkpoint selection rule

제안하는 V3b screening acceptance criteria:
- V2 대비 aggregate quantity MAE가 악화되지 않을 것
- V3a quantity 이득의 방향이 유지될 것
- Taxi marker NLL의 대규모 악화가 제거될 것
- mark accuracy가 V2에 근접하거나 개선될 것
- Intermittent `1-9` 및 Taxi `10-999` scale regression이 축소될 것
- NaN/Traceback 없이 동일 e50 seed `42` screening을 완료할 것

위 acceptance criteria는 다음 실험의 proposed gate이며 아직 달성된 결과가 아닙니다.

## 13. Limitations And Confidence

- 각 결과는 seed `42` 하나의 e50 short screening입니다.
- multi-seed variance는 평가하지 않았습니다.
- e50 결과를 final paper performance로 사용하지 마세요.
- tail bucket은 sample 수가 적고 magnitude가 커 aggregate quantity MAE에 큰 영향을 줍니다.
- Intermittent mark class는 log2 class이고 quantity scale 표는 decimal quantity bucket입니다.
- `best_score`와 final checkpoint에서 일부 방향이 달라도 primary comparison은 `best_val_nll`입니다.
- 현재 종합 판정의 confidence는 `Share with caveats`입니다.

## 14. Artifact Paths

Local:
- Intermittent V2: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_v2_inter_short_e50_0710`
- Intermittent V3a: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_v3_inter_short_e50_0710`
- Taxi V2: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_v2_taxi_short_e50_0710`
- Taxi V3a: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_v3_taxi_short_e50_0710`

Server:
- Intermittent V2: `~/workspace/paper_research/search_artifacts/model_enhancement_v2_inter_short_e50_0710`
- Intermittent V3a: `~/workspace/paper_research/search_artifacts/model_enhancement_v3_inter_short_e50_0710`
- Taxi V2: `~/workspace/paper_research/search_artifacts/model_enhancement_v2_taxi_short_e50_0710`
- Taxi V3a: `~/workspace/paper_research/search_artifacts/model_enhancement_v3_taxi_short_e50_0710`

주요 evidence:
- 각 artifact의 `experiment_manifest.json`
- 각 artifact의 `logs/run.log`
- 각 artifact의 `leaderboard/summary.csv`
- 각 artifact의 `leaderboard/test_summary.csv`
- 각 artifact의 `leaderboard/histories.csv`
- 각 artifact의 `leaderboard/test_scale_wise_summary.csv`
- 각 artifact의 `paper_outputs/report.md`
- 각 artifact의 `paper_outputs/plots/`

## 15. Page Update Structure

기본 대상 페이지를 아래 순서로 갱신해주세요.
1. 상단 status callout을 screening completed 상태로 변경
2. Short Screening Status 표의 모든 V2/V3 행을 completed로 변경
3. `V3a Short Screening Combined Result` 섹션 추가
4. Experiment Integrity
5. Checkpoint Selection Rule
6. Combined Primary Held-out Test Result
7. Intermittent Result
8. Taxi Result
9. Scale-wise Comparison
10. Mark Confusion Findings
11. Cross-dataset Confirmed Findings
12. Cross-dataset Inference
13. V3a Screening Gate Decision
14. Next Architecture: V3b Detached-Gate
15. Limitations And Confidence
16. Next Action

작성 주의사항:
- 기존 Architecture, Training Contract, Inference Contract, Initialization, Implementation Map, Focused Verification 섹션을 삭제하거나 축약하지 마세요.
- confirmed metric과 inferred mechanism을 별도 소제목으로 분리해주세요.
- Intermittent와 Taxi의 marker NLL 방향이 다르다는 점을 숨기지 마세요.
- quantity 개선과 mark accuracy 저하를 같은 비중으로 보여주세요.
- V3a를 완전 성공 또는 완전 실패로 표현하지 말고 `partial success`로 기록해주세요.
- `best_score` 결과로 primary conclusion을 바꾸지 마세요.
- V3b는 proposed architecture이며 구현 완료처럼 쓰지 마세요.
- child page, 기존 표, 기존 설계 근거를 삭제하지 마세요.

Next Action:
1. V3b detached-gate architecture contract 확정
2. V3b 최소 구현
3. focused gradient-routing test 추가
4. shared/V3a/V3b zero-init 및 forward equivalence 확인
5. V3b Intermittent·Taxi e50 seed `42` screening
6. gate 통과 시 seeds `42,52,62` 검증으로 확장
