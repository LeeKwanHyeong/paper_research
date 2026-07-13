다음 실험 결과를 기존 Notion 페이지에 업데이트해주세요.

대상 페이지:
- `TitanTPP V3b Detached Quantity Gate Smoke And Short Screening e50`
- `5. Model Design Enhancement`의 V3b 설계 페이지에도 결과 요약과 상호 링크 추가

실험 상태:
- `completed`

실험 시간:
- 실험 시작 시각: `2026-07-10 14:15:50 KST`
- 실험 종료 시각: `2026-07-10 14:31:40 KST`
- 총 소요시간: `15분 50초`
- 실행 서버: `5080`
- tmux session: `titantpp_v3b_screen_e50_0710`
- conda env: `ai_env`

결과 artifact:
- local: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_v3b_insta_smoke_e1_0710`
- local: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_v3b_inter_short_e50_0710`
- local: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_v3b_taxi_short_e50_0710`
- server: `~/workspace/paper_research/search_artifacts/model_enhancement_v3b_insta_smoke_e1_0710`
- server: `~/workspace/paper_research/search_artifacts/model_enhancement_v3b_inter_short_e50_0710`
- server: `~/workspace/paper_research/search_artifacts/model_enhancement_v3b_taxi_short_e50_0710`

실행 완료 및 무결성:
- Instacart top-20 e1 smoke, Intermittent e50, Taxi e50 모두 완료
- 세 run 모두 `qty_mark_gradient_mode=detached`와 독립 `qtymarkgrad_detached` path 확인
- NaN 및 Traceback 없음
- local focused pytest `9/9` 통과
- 5080 V3a/V3b GPU model-test에서 forward/loss 수치 exact equivalence 확인
- Intermittent best validation NLL epoch: `49`
- Taxi best validation NLL epoch: `32`

분석 기준:
- `best_val_nll` checkpoint의 held-out test를 기본 비교로 사용
- V2는 shared/coupled, V3a는 experts/coupled, V3b는 experts/detached
- 동일 dataset, candidate, seed, lr, split, lookback, max_seq_len 조건만 비교
- artifact reading order: manifest, run.log, summary, test_summary, histories, validation scale-wise, test scale-wise, report, plots

V3b held-out test 절대값:

| Dataset | Total NLL | Marker NLL | Time NLL | Qty MAE | Value MAE | Mark accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Intermittent | `5.058310` | `1.004198` | `4.054112` | `3.463607` | `0.150965` | `53.367%` |
| Taxi | `1.600917` | `0.197501` | `1.403416` | `32.274835` | `0.153194` | `92.314%` |

V3b vs V2:

| Dataset | Total NLL | Marker NLL | Qty MAE | Value MAE | Mark accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intermittent | `-0.268%` | `-1.193%` | `-1.833%` | `-1.770%` | `-1.093%p` |
| Taxi | `-1.990%` | `-16.124%` | `-36.128%` | `-23.879%` | `+1.057%p` |

V3b vs V3a:

| Dataset | Total NLL | Marker NLL | Qty MAE | Value MAE | Mark accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intermittent | `-0.074%` | `-0.191%` | `+13.245%` | `+28.822%` | `-0.070%p` |
| Taxi | `-4.366%` | `-28.617%` | `-31.330%` | `-14.399%` | `+1.801%p` |

Scale-wise held-out test 해석:
- Intermittent `1-9` bucket share `88.67%`: V2 대비 MAE `+3.640%`
- Intermittent `10-99` bucket share `10.66%`: V2 대비 MAE `-1.621%`
- Taxi `1-9`: V2 대비 MAE `-14.165%`
- Taxi `10-99`: V2 대비 MAE `-24.917%`
- Taxi `100-999`: V2 대비 MAE `-32.027%`
- Taxi `1000-9999`: V2 대비 MAE `-40.248%`
- Taxi에서는 sample share가 5% 이상인 모든 bucket이 개선됨

Acceptance gate 판정:

| Gate | Result |
| --- | --- |
| V2 대비 total NLL 악화 <= 0.5% | PASS, Intermittent `-0.268%`, Taxi `-1.990%` |
| Taxi marker NLL 악화 <= 2% | PASS, `-16.124%` |
| V2 대비 mark accuracy gap <= 0.25%p | FAIL, Intermittent `-1.093%p`; Taxi는 PASS `+1.057%p` |
| V3a aggregate quantity gain의 절반 이상 유지 | PASS, 평균 gain V3a `10.151%` vs V3b `18.981%`, 유지율 `186.98%` |
| share >= 5% bucket의 V2 대비 MAE 악화 <= 5% | PASS, 두 데이터셋의 모든 대상 bucket 통과 |

핵심 해석:
- V3b는 Taxi에서 V3a의 marker degradation을 제거하면서 quantity 성능도 크게 개선한 강한 성공 신호입니다.
- Intermittent에서는 direct quantity-to-mark-head gradient 차단만으로 mark accuracy가 회복되지 않았고 V3a quantity gain 대부분이 사라졌습니다.
- 따라서 Intermittent의 conflict는 mark probability gate뿐 아니라 shared encoder로 전달되는 value/quantity gradient 또는 dataset-specific mark imbalance에도 존재할 가능성이 있습니다.
- 공식 aggregate quantity-retention gate는 통과했으며, 전체 gate의 유일한 실패 항목은 Intermittent mark accuracy입니다.
- V3b는 `dataset-specific success`이며 전 데이터셋 공통 replacement로는 아직 채택하지 않습니다.
- seed-42 e50 screening이므로 최종 우월성 주장은 보류합니다.

결정 및 다음 액션:
- V2를 공통 baseline으로 유지
- Taxi V3b만 seeds `42,52,62` multi-seed 확인 대상으로 승격
- Intermittent는 V3b를 승격하지 않고 shared-encoder value-gradient routing을 조절하는 V3c 설계 검토
- Taxi multi-seed에서도 NLL, marker NLL, quantity MAE, mark accuracy, scale-wise 개선이 유지될 때 최종 variant 승격 검토
