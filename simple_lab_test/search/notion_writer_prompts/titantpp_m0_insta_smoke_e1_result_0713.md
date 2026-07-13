다음 실험 결과를 기존 Notion 페이지에 업데이트해주세요.

위치:
- `5. Model Design Enhancement > Enhancement & Validation History`
- `2. Confirm and Refine Topic`에는 작성하지 않음
- 기존 `TitanTPP M0 Instacart Top-20 e1 Smoke (2026-07-13)` 페이지에 결과 추가

결과 artifact:
- local path: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_m0_insta_smoke_e1_0713`
- server path: `/home/leekwanhyeong/workspace/paper_research/search_artifacts/model_enhancement_m0_insta_smoke_e1_0713`
- completion marker: `SMOKE_SUCCESS`

실험 시간과 환경:
- 시작: `2026-07-13 11:16:45 KST`
- 종료: `2026-07-13 11:16:50 KST`
- 총 소요시간: 약 `5초`
- exit code: `0`
- server/tmux: `5090` / `titantpp_m0_insta_e1_0713`
- GPU: NVIDIA GeForce RTX 5090
- Python: `/opt/miniconda3/envs/ai_env/bin/python`

실행 범위:
- dataset: Instacart top `20` series, 총 `2,000` rows
- fixed split rows: train `1,400`, validation `300`, test `300`
- DataLoader samples: train `1,380`, validation `300`, test `300`
- model/candidate: `TitanTPP / small_lmm`
- epoch/seed: `1 / 42`
- batch size/lookback/max sequence length: `16 / 10 / 16`
- quantity decoder/normalization: `direct_log_qty / global`
- value input: `none`
- marker loss: plain CE, `lambda_ordinal=0`
- train loss scope/loss mode: `target_only / hybrid`

Train-only global magnitude statistics:

| 항목 | 값 |
| --- | ---: |
| source split | `train` |
| train event count | `1,400` |
| log2 mean | `3.611660` |
| population variance | `0.537870` |
| population std | `0.733396` |

- 동일 값이 run manifest의 `marked_meta`, `rmtpp_config`와 checkpoint config에 저장됨
- validation/test event는 통계 계산에 포함되지 않음

학습 및 validation 결과:

| Metric | Validation |
| --- | ---: |
| epoch train loss | `4.964815` |
| score | `0.467243` |
| total NLL | `3.252680` |
| marker NLL | `1.122828` |
| time NLL | `2.129852` |
| magnitude loss | `0.398124` |
| quantity MAE | `5.215548` |
| log2 quantity MAE | `0.567363` |
| log2 quantity RMSE | `0.704898` |
| mark accuracy | `48.667%` |
| DT MAE | `1.420777` |
| evaluated targets | `300` |

Held-out test smoke 결과:

| Metric | Test |
| --- | ---: |
| score | `0.489654` |
| total NLL | `3.153552` |
| marker NLL | `1.117854` |
| time NLL | `2.035698` |
| magnitude loss | `0.411177` |
| quantity MAE | `4.902802` |
| log2 quantity MAE | `0.564945` |
| log2 quantity RMSE | `0.773779` |
| mark accuracy | `50.667%` |
| DT MAE | `1.210981` |
| evaluated targets | `300` |

Scale-wise quantity MAE:

| Split | `1-9` | `10-99` |
| --- | ---: | ---: |
| Validation | `5.121273` (`78`) | `5.248671` (`222`) |
| Test | `5.117551` (`76`) | `4.829940` (`224`) |

- populated bucket에는 NaN/Inf가 없음
- `100+` bucket은 count `0`이므로 NaN은 의도된 empty-bucket 표기

Artifact와 구조 확인:
- `SMOKE_SUCCESS`, experiment/run manifest, logs, checkpoint 4종, summary/test summary, history, validation/test scale-wise, confusion/per-class, report, plots 생성 확인
- final runtime log에 NaN, Inf, Traceback, RuntimeError, `[ERROR]`, FAILED 없음
- checkpoint state에 `magnitude_head`, `magnitude_input_proj` 존재
- checkpoint state에 legacy `value_head` 없음
- `value_mae`, `val_value_loss`는 null이며 direct metric으로 재사용되지 않음
- run path에 `qtydecoder_direct_log_qty/magnorm_global`이 포함되어 legacy artifact와 분리됨
- e1 단일 epoch plot은 line이 보이지 않지만 direct/global label과 9개 metric panel이 정상 생성됨

판정:
- 5090 actual-data integration smoke gate: `PASS`
- train-only global normalization persistence: `PASS`
- direct decoder exclusive checkpoint contract: `PASS`
- runtime/artifact integrity: `PASS`
- 모델 성능 우위: `not evaluated`

해석:
- M0는 실제 fixed-split DataLoader에서 backward, checkpoint selection, validation/test direct quantity evaluation까지 정상 동작함
- 이번 e1 결과는 실행 경로 확인용이며 V2 대비 성능 우위나 held-out test 기반 후보 선택 근거로 사용하지 않음
- 다음 단계는 frozen V2 validation reference와 비교하는 Intermittent M0 seed-42 e50 validation-only screening

문체:
- smoke 통과와 성능 개선을 구분
- test metric은 artifact integrity 확인용임을 명시
- 확인된 수치만 사용하고 과장하지 않음
