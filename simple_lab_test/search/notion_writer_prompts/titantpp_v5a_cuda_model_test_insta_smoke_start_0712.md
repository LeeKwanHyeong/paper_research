다음 TitanTPP V5a integration smoke 완료 결과를 기존 Notion 페이지에 업데이트해주세요.

## 작성 위치

- `5. Model Design Enhancement > Enhancement & Validation History`
- 기존 `TitanTPP V5a Ordinal Marker Loss Contract And Acceptance Gate` 페이지를
  업데이트하거나 그 하위에 연결합니다.
- Model Enhancement 작업이므로 `2. Confirm and Refine Topic`에는 작성하지 않습니다.
- 같은 제목의 페이지가 있으면 업데이트하고 중복 페이지는 만들지 않습니다.

## 페이지 제목

- `TitanTPP V5a CUDA Model-Test And Instacart e1 Smoke`

## 기준 시각과 실행 환경

- 기록 준비 시각: `2026-07-12 20:12:30 KST`
- 실험 시작 시각: `2026-07-12 20:16:32 KST`
- 실험 종료 시각: `2026-07-12 20:16:38 KST`
- 결과 확인 시각: `2026-07-12 20:37:39 KST`
- 전체 실행 시간: 약 `6초`
- 실행 서버: `5090` (`192.168.0.71:22`)
- SSH user: `leekwanhyeong`
- project root: `/home/leekwanhyeong/workspace/paper_research`
- Python: `/opt/miniconda3/envs/ai_env/bin/python`
- conda env: `ai_env`
- tmux session: `titantpp_v5a_smoke_e1_0712`

## 현재 상태

- 상태: `completed`
- smoke gate: `PASS`
- local V5a focused tests: `20 passed`
- 기존 V3 계열과 diagnostic tests를 포함한 전체 local tests: `42 passed`
- local V5a CPU `small_lmm` model-test: `success`
- 5090 CUDA model-test: `success`
- Instacart top-20 e1: `1/1 epoch completed`
- 초기 확인 시각: `2026-07-12 20:16:35 KST`
- 초기 확인 PID: `2329053`; GPU memory: `500 MiB`
- 초기 실행 확인 후 지속 polling하지 않음
- 결과 확인 시 tmux session과 GPU compute process가 종료된 상태
- 5090 artifact를 local `search_artifacts/`로 동기화 완료

초기 확인 내용:

- model-test hidden shape: `[2, 16, 64]`
- model-test total NLL: `4.944328`
- model-test artifact와 log 생성 확인
- Instacart dataset: `2,000` rows, `20` series, `num_marks=7`
- fixed split samples: train `1,380`, validation `300`, test `300`
- `batch_size=16`, `lookback=10`, `max_seq_len=16`
- 실행 config에 `shared/coupled/coupled`, `ce_rps`, `lambda_ordinal=0.1` 기록 확인

## 실험 목적

- normalized RPS와 `marker_loss_mode=ce_rps`가 5090 CUDA에서 finite하게 동작하는지 확인
- V2 구조인 `shared/coupled/coupled`를 유지하고 marker objective만 바뀌었는지 확인
- Instacart fixed-split top-20 e1에서 train/eval/report artifact가 생성되는지 확인
- Intermittent e50을 시작하기 전에 runtime, schema, path collision 문제를 차단

## 실험 계획

| Order | Stage | Dataset | Candidate | Epochs | Seed | 실행 조건 |
| ---: | --- | --- | --- | ---: | ---: | --- |
| 1 | CUDA model-test | synthetic | `small_lmm` | - | 42 | 실패 시 즉시 중단 |
| 2 | integration smoke | Instacart top 20 series | `small_lmm` | 1 | 42 | model-test 통과 후 실행 |

공통 V5a contract:

- model: `titantpp`
- value head: `shared`
- quantity mark gradient: `coupled`
- value encoder gradient: `coupled`
- marker loss: `ce_rps`
- `lambda_ordinal=0.10`
- existing marker head와 inference 변경 없음
- `nll_marker`는 CE, `nll`은 CE + time NLL 의미 유지

Instacart smoke 조건:

- lr: `1e-3`
- batch size: `16`
- lookback: `10`
- max sequence length: `16`
- max series: `20`
- split mode: `fixed`
- value input: `residual`
- train loss scope: `target_only`
- loss mode: `hybrid`
- value head activation: `identity`
- eval selections: `best_val_nll,best_score,final`
- force rerun, stop on error: enabled

## 실행 명령어

5090 tmux 실행:

```bash
ssh 5090
cd /home/leekwanhyeong/workspace/paper_research
/opt/miniconda3/envs/ai_env/bin/tmux new-session -s titantpp_v5a_smoke_e1_0712
```

tmux 안에서 실행:

```bash
env \
  PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
  PYTHON_BIN=/opt/miniconda3/envs/ai_env/bin/python \
  bash /home/leekwanhyeong/workspace/paper_research/simple_lab_test/search/scripts/run_v5a_cuda_model_test_insta_smoke_0712.sh
```

실제 단계별 명령은 아래 local script와 동일합니다.

```text
/Users/igwanhyeong/PycharmProjects/paper_research/simple_lab_test/search/scripts/run_v5a_cuda_model_test_insta_smoke_0712.sh
```

## Artifact 경로

5090:

```text
/home/leekwanhyeong/workspace/paper_research/search_artifacts/model_enhancement_v5a_model_test_0712
/home/leekwanhyeong/workspace/paper_research/search_artifacts/model_enhancement_v5a_insta_smoke_e1_0712
```

완료 후 local sync 대상:

```text
/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_v5a_model_test_0712
/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_v5a_insta_smoke_e1_0712
```

## Smoke Gate

- model-test status `success`
- CUDA hidden/output/loss finite
- `marker_loss_mode=ce_rps`, `lambda_ordinal=0.10` 기록
- `marker_train_loss = nll_marker + 0.10 * ordinal_marker_loss` 수치 일치
- Instacart e1 정상 종료
- NaN, Traceback, RuntimeError, ERROR 없음
- run path에 `markloss_ce_rps/lambdaord_0p1` 포함
- manifest, log, summary, test summary, histories, validation/test scale-wise,
  report, plots 생성
- V2 architecture fields가 `shared/coupled/coupled`로 기록

## CUDA Model-Test 결과

| Metric | Value |
| --- | ---: |
| Status | `success` |
| Hidden shape | `[2, 16, 64]` |
| Parameter count | `77,583` |
| Total NLL | `4.944328` |
| Marker CE | `2.500342` |
| Time NLL | `2.443986` |
| Normalized RPS | `0.185512` |
| Marker train loss | `2.518893` |

`marker_train_loss = marker CE + 0.10 * RPS`의 absolute error는
`6.41e-08`입니다. `shared/coupled/coupled`, `ce_rps`, lambda `0.1`, CUDA가
model-test summary에 기록됐습니다.

## Instacart e1 결과

| Metric | Validation | Held-out test |
| --- | ---: | ---: |
| Total NLL | `3.275237` | `3.161041` |
| Marker NLL | `1.146415` | `1.124429` |
| Time NLL | `2.128822` | `2.036612` |
| Normalized RPS | `0.078961` | `0.077667` |
| Mark accuracy | `46.667%` | `52.333%` |
| Balanced accuracy | `20.000%` | `16.667%` |
| Macro F1 | `12.727%` | `11.451%` |
| Mark MAE | `0.560000` | `0.523333` |
| Adjacent accuracy | `97.333%` | `96.333%` |
| Quantity MAE | `5.637924` | `5.266978` |
| Value MAE | `0.256561` | `0.267392` |
| Time MAE | `1.423680` | `1.214902` |

- epoch-1 train loss: `4.727945`
- best validation NLL epoch: `1`
- validation/test sample: 각각 `300`
- e1 checkpoint가 하나이므로 `best_val_nll`, `best_score`, `final` 결과는 동일
- 모든 validation/test prediction은 mark `3`이었음
- e1은 학습 가능성과 artifact contract 확인용이므로 mark-3 collapse를 V5a 성능
  판정이나 lambda 선택 근거로 사용하지 않음

Scale-wise quantity MAE:

| Split | `1-9` | `10-99` |
| --- | ---: | ---: |
| Validation | `3.754026` (`78`) | `6.299834` (`222`) |
| Test | `3.747793` (`76`) | `5.782416` (`224`) |

- `100+` bucket은 sample count가 `0`이므로 metric `NaN`은 의도된 empty-bucket 표기
- populated summary, test summary, history에는 NaN/Inf 없음

## Artifact 확인

Protocol 순서로 아래 항목을 확인했습니다.

1. `experiment_manifest.json`
2. `logs/run.log`
3. `leaderboard/summary.csv`
4. `leaderboard/test_summary.csv`
5. `leaderboard/histories.csv`
6. `leaderboard/scale_wise_summary.csv`
7. `leaderboard/test_scale_wise_summary.csv`
8. `paper_outputs/report.md`
9. `paper_outputs/plots/`

- NaN runtime, Traceback, RuntimeError, `[ERROR]`, FAILED 없음
- V5a run path `markloss_ce_rps/lambdaord_0p1` 확인
- checkpoint, manifest, summary, test, history, scale-wise, confusion, per-class,
  report, validation/test plot 생성 확인
- learning-curve plot은 e1 단일 point라 line이 보이지 않지만 7개 metric panel은 정상 생성
- scale-wise plot은 populated bucket 값이 정상 렌더링됨

## 판정

- 5090 CUDA integration gate: `PASS`
- V2 architecture와 V5a objective-only contract: `PASS`
- runtime/artifact integrity: `PASS`
- 모델 성능 판정: `not evaluated`; e1 결과로 V5a가 V2보다 낫다고 쓰지 않음
- held-out test는 smoke artifact 확인에만 사용했으며 coefficient 선택에 사용하지 않음

## 해석

- local CPU와 동일하게 CUDA에서도 CE, RPS, time, value, quantity 경로가 finite함
- V2의 parameterized architecture를 유지한 상태에서 CE+RPS objective가 실행됨
- e1의 단일 mark 예측은 장기 학습 전 초기 상태이며 ordinal benefit을 말해주지 않음
- 다음 단계는 predeclared gate를 사용하는 Intermittent seed-42 e50 validation screening

페이지 마지막 Next:

```text
Next: smoke gate 통과 시 Intermittent V5a seed-42 e50 validation-only screening 준비
```
