다음 실험 결과를 기존 Notion 페이지에 업데이트해주세요.

대상 페이지:
- `TitanTPP V3b Taxi Multi-Seed Confirmation e50`
- 기존 `TitanTPP V3b Detached Quantity Gate Smoke And Short Screening e50` 페이지와 상호 링크
- `5. Model Design Enhancement`의 V3b 설계 페이지에도 최종 판정 요약 추가

실험 상태:
- `completed`

실험 시간:
- 실험 시작 시각: `2026-07-10 14:54:27 KST`
- 실험 종료 시각: `2026-07-10 15:21:44 KST`
- 총 소요시간: `27분 17초`
- 실행 서버: `5080`
- tmux session: `taxi_v3b_multiseed_e50_0710`
- conda env: `ai_env`

서버 선택 기록:
- 5090 GPU는 비어 있었으나 tmux 바이너리가 설치되어 있지 않았음
- 공유 서버에 시스템 패키지를 임의 설치하거나 nohup으로 우회하지 않음
- 프로토콜의 보조 서버 규칙에 따라 tmux와 ai_env가 검증된 5080에서 실행

결과 artifact:
- local: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_v3b_taxi_multiseed_e50_0710`
- server: `~/workspace/paper_research/search_artifacts/model_enhancement_v3b_taxi_multiseed_e50_0710`

실행 완료 및 무결성:
- seeds `42,52,62`의 `3/3` run 완료
- NaN 및 Traceback 없음
- dataset `yellow_trip_hourly`, candidate `mid_lmm`, epochs `50`
- `value_head_mode=mark_conditioned_experts`
- `qty_mark_gradient_mode=detached`
- fixed split, target-only, hybrid loss, residual input 확인
- local/5080/5090 preflight model-test 통과
- local과 5080 핵심 실행 파일 checksum 일치

분석 기준:
- `best_val_nll` checkpoint의 held-out test 사용
- 비교 baseline은 `model_enhancement_v2_hybrid_e200_0705`의 Taxi `mid_lmm`, seeds `42,52,62`
- V2 baseline은 e200이고 V3b는 e50이므로 epoch budget mismatch를 한계로 명시
- artifact reading order: manifest, run.log, summary, test_summary, histories, validation scale-wise, test scale-wise, report, plots

Validation 결과:
- mean best validation NLL: `1.555298 ± 0.001582`
- best epoch: seed 42=`42`, seed 52=`32`, seed 62=`49`
- mean best validation quantity MAE: `31.591845 ± 2.901839`
- mean final validation NLL: `1.637165`
- validation final-minus-best NLL degradation: `+5.264%`

Held-out test 평균:

| Metric | Mean | Std |
| --- | ---: | ---: |
| Total NLL | `1.611892` | `0.000590` |
| Marker NLL | `0.208618` | - |
| Time NLL | `1.403274` | - |
| Quantity MAE | `38.312184` | `4.635932` |
| Value MAE | `0.151485` | - |
| Mark accuracy | `91.902%` | - |

V3b mean vs V2 multi-seed e200:

| Total NLL | Marker NLL | Time NLL | Quantity MAE | Value MAE | Mark accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `-2.357%` | `-17.015%` | `+0.277%` | `-34.733%` | `-24.247%` | `+1.209%p` |

Seed별 best-val-NLL test:

| Seed | NLL | Marker NLL | Qty MAE | Mark accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 42 | `1.611390` | `0.209287` | `40.618267` | `91.858%` |
| 52 | `1.611744` | `0.208358` | `41.342844` | `91.882%` |
| 62 | `1.612541` | `0.208210` | `32.975441` | `91.966%` |

Seed-matched V2 e200 대비 변화:

| Seed | Total NLL | Marker NLL | Quantity MAE | Mark accuracy | 동시개선 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 42 | `-2.174%` | `-15.637%` | `-51.918%` | `+0.733%p` | PASS |
| 52 | `-2.278%` | `-16.105%` | `+13.017%` | `+1.045%p` | FAIL: quantity |
| 62 | `-2.618%` | `-19.219%` | `-40.092%` | `+1.849%p` | PASS |

Scale-wise held-out test mean vs V2:

| Scale | Share | V3b Qty MAE | Change vs V2 |
| --- | ---: | ---: | ---: |
| `1-9` | `54.23%` | `1.561491` | `-8.095%` |
| `10-99` | `24.33%` | `11.434228` | `-27.444%` |
| `100-999` | `13.89%` | `81.922968` | `-37.833%` |
| `1000-9999` | `7.54%` | `308.954200` | `-34.612%` |

Stability:
- test NLL CV: V3b `0.037%`, V2 `0.275%`
- quantity MAE CV: V3b `12.100%`, V2 `41.152%`
- V2 대비 NLL std `-86.99%`, quantity MAE std `-80.81%`
- final test NLL은 best-val-NLL test보다 평균 `+4.758%` 악화
- seed 52 final validation NLL은 best보다 `+12.913%` 악화
- 동일 seed 42 재실행에서도 quantity 값이 달라 CUDA-level non-determinism 가능성을 기록

Confirmation gate:

| Gate | Result |
| --- | --- |
| 3/3 완료, NaN/Traceback 없음 | PASS |
| V2 대비 mean total NLL 악화 <= 0.5% | PASS, `-2.357%` |
| V2 대비 mean marker NLL 악화 <= 2% | PASS, `-17.015%` |
| V2 대비 mean mark accuracy regression <= 0.25%p | PASS, `+1.209%p` |
| V2 대비 mean quantity MAE 개선 >= 18.06% | PASS, `-34.733%` |
| seed-matched 동시개선 2/3 이상 | PASS, `2/3` |
| share >= 5% bucket의 mean MAE 5% 초과 regression 없음 | PASS, 모든 bucket 개선 |

핵심 해석:
- V3b Taxi 효과는 seed 42 단독 현상이 아니며 likelihood, marker, accuracy, 평균 quantity 관점에서 재현됐습니다.
- total NLL 개선은 time NLL이 아니라 marker NLL 개선이 주도했습니다.
- quantity는 평균과 scale-wise에서 강하게 개선됐지만 seed 52에서는 seed-matched V2보다 악화되어 완전한 seed 독립성은 아닙니다.
- best epoch이 `32~49`에 분포하고 final degradation이 커 validation checkpoint 또는 early stopping이 필수입니다.
- V3b는 Taxi 전용 next-stage candidate로 승격하되 전체 데이터셋 공통 replacement로 채택하지 않습니다.
- V2와 V3b의 epoch budget이 다르므로 논문 최종 주장 전 동일 e50 multi-seed matched comparison이 필요합니다.

결정 및 다음 액션:
- Taxi V3b를 차기 본실험 후보로 승격
- 동일 조건의 Taxi V2 e50 seeds `42,52,62`를 실행해 strict matched-budget comparison 구성
- 이후 필요하면 V3b e100/e200에 early stopping을 적용해 장기 안정성 확인
- Intermittent는 V2 baseline을 유지하고 V3c shared-encoder gradient routing 설계로 진행
