다음 기존 Notion 실험 페이지에 V1/V2 최종 결과를 업데이트해주세요.

위치:
- `2. Confirm and Refine Topic > Model Validation`

대상 페이지:
- `TitanTPP V1/V2 Baseline Finalization e200`
- 같은 제목 또는 동일 실험의 기존 페이지를 먼저 찾아 업데이트하고, 중복 페이지는 만들지 마세요.

문서 상태:
- `implemented`
- V1/V2 실험, artifact 동기화, 결과 분석이 모두 완료된 상태입니다.

실험 시간 및 실행 환경:
- 실험 시작: `2026-07-05 20:34:30 KST`
- V1 종료: `2026-07-07 16:01:47 KST`
- V2 종료: `2026-07-09 12:14:48 KST`
- 실행 서버: `5090`
- tmux session: `titantpp_v1_v2_baseline_e200_0705`
- conda env: `ai_env`

실험 목적:
- V3 mark-conditioned value head를 추가하기 전에 TitanTPP의 immediate baseline을 확정합니다.
- V1 `residual_only`와 V2 `hybrid`를 같은 데이터, split, candidate, seed 조건에서 비교합니다.
- V2가 quantity MAE를 개선하면서 score/NLL과 학습 안정성을 유지하는지 확인합니다.

공통 실험 조건:
- datasets: `intermittent,yellow_trip_hourly,insta_market_basket`
- model: `titantpp`
- candidates: `small_lmm,mid_lmm`
- epochs: `200`
- seeds: `42,52,62`
- lr: `1e-3`
- batch_size: `128`
- split_mode: `fixed`
- value_input_mode: `residual`
- train_loss_scope: `target_only`
- value_head_activation: `identity`
- eval selections: `best_val_nll,best_score,final`
- V1 loss_mode: `residual_only`
- V2 loss_mode: `hybrid`

완료 상태:
- V1 `18/18` runs 완료
- V2 `18/18` runs 완료
- NaN, Traceback, runtime ERROR 없음
- 로그의 `max_abs_error`는 quantity reconstruction 계약 검증값이며 실행 오류가 아닙니다.

분석 방법:
- artifact는 `experiment_manifest.json`, `logs/run.log`, `summary.csv`, `test_summary.csv`, `histories.csv`, scale-wise summaries, report, plots 순서로 확인했습니다.
- candidate는 held-out test를 보기 전에 validation mean best NLL로 선택했습니다.
- 최종 test 비교는 `best_val_nll` checkpoint를 사용했습니다.

Validation으로 선택한 candidate:

| Dataset | V1 selected | V2 selected | V1 validation NLL | V2 validation NLL |
| --- | --- | --- | ---: | ---: |
| Instacart | `mid_lmm` | `small_lmm` | `4.379298` | `4.381901` |
| Intermittent | `small_lmm` | `small_lmm` | `5.629190` | `5.604595` |
| Yellow Trip Hourly | `small_lmm` | `mid_lmm` | `1.568134` | `1.576568` |

Held-out test 결과:

| Dataset | V1 score | V2 score | Score delta | V1 qty MAE | V2 qty MAE | Qty MAE change | Test NLL change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Instacart | `0.436657` | `0.437034` | `+0.000377` | `4.452236` | `4.405873` | `-1.04%` | `+0.04%` |
| Intermittent | `0.279412` | `0.287139` | `+0.007727` | `3.365281` | `3.030658` | `-9.94%` | `-0.32%` |
| Yellow Trip Hourly | `0.824321` | `0.840379` | `+0.016058` | `79.172297` | `58.700291` | `-25.86%` | `+1.23%` |

Confirmed findings:
- Intermittent는 V2에서 3개 seed 모두 score와 quantity MAE가 개선됐습니다.
- Intermittent는 모든 populated quantity scale에서 MAE가 개선됐습니다.
- Instacart는 3개 seed 모두 score와 quantity MAE가 개선됐지만 개선 폭은 작습니다.
- Instacart의 quantity 개선은 `1-9` bucket의 `-4.87%` 개선이 주도하고, `10-99` bucket은 `+1.47%` 악화됐습니다.
- Instacart `100-999` bucket은 test share가 약 `0.0036%`라 안정적인 결론 근거로 사용하지 않습니다.
- Yellow Trip V2는 평균 quantity MAE를 `25.86%` 개선했습니다.
- Yellow Trip V1 selected `small_lmm`은 best validation NLL이 epoch `4-13`에서 나온 뒤 final NLL이 악화됐습니다.
- Yellow Trip V2 `mid_lmm`은 best validation NLL이 epoch `148-184`에서 나왔고 final-minus-best NLL 차이가 약 `0.012-0.017`로 안정적이었습니다.
- Yellow Trip V2 marker NLL은 약 `13.2%` 악화됐고 mark accuracy는 `0.004764` 감소했습니다.
- Yellow Trip V2는 seed 52/62에서 크게 개선됐지만 seed 42에서는 score와 quantity MAE가 악화됐습니다.

Interpretation:
- V2 hybrid objective는 quantity reconstruction 개선뿐 아니라 긴 weekly sequence인 taxi에서 late-epoch time-NLL collapse를 완화한 것으로 해석합니다.
- Intermittent에서는 V2 개선이 seed와 scale 전반에 나타나 가장 신뢰도가 높은 positive result입니다.
- Taxi는 평균적으로 강한 개선이지만 seed variance와 marker 성능 trade-off가 남아 있어 완전히 안정적이라고 과장하지 않습니다.
- Instacart는 V1/V2 모두 plateau에 가까우며, V2 효과는 작고 low-scale quantity에 집중됩니다.

Baseline decision:
- Primary pre-V3 baseline: V2 hybrid TitanTPP
- Dataset candidates: `intermittent=small_lmm`, `insta_market_basket=small_lmm`, `yellow_trip_hourly=mid_lmm`
- Auxiliary guardrail: V1 `residual_only`, 특히 taxi marker NLL과 mark accuracy

다음 모델 강화 작업:
- V3 mark-conditioned value head를 설계 및 구현합니다.
- V3 목표는 V2의 quantity/stability 개선을 유지하면서 taxi marker NLL, mark accuracy, seed variance를 개선하는 것입니다.

Artifact 경로:
- V1 local: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_v1_residual_e200_0705`
- V2 local: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_v2_hybrid_e200_0705`
- V1 server: `~/workspace/paper_research/search_artifacts/model_enhancement_v1_residual_e200_0705`
- V2 server: `~/workspace/paper_research/search_artifacts/model_enhancement_v2_hybrid_e200_0705`
- 분석 notebook: `/Users/igwanhyeong/PycharmProjects/paper_research/simple_lab_test/notebooks/experiments/titantpp_v1_v2_baseline_analysis.ipynb`
- 전략 문서: `/Users/igwanhyeong/PycharmProjects/paper_research/simple_lab_test/search/model_enhancement_strategy.md`

페이지 구성:
1. 상단에 `implemented` 상태와 baseline decision을 callout으로 표시
2. Experiment Objective
3. Configuration
4. Completion And Data Integrity
5. Validation-Only Candidate Selection
6. Held-out Test Results
7. Dataset-wise Interpretation
8. Convergence And Stability
9. Risks And Caveats
10. Baseline Decision
11. Next Action: V3

작성 주의사항:
- confirmed metric과 inferred interpretation을 구분해서 작성해주세요.
- 이 실험에는 RMTPP 비교가 포함되지 않았으므로 TitanTPP가 RMTPP보다 우월하다고 새롭게 주장하지 마세요.
- 평균값만 보고 taxi 결과를 완전히 안정적이라고 표현하지 마세요.
- final epoch가 아니라 `best_val_nll` checkpoint 결과를 본문 주요 표에 사용하세요.
