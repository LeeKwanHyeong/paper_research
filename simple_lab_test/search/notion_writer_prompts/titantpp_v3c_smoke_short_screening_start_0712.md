다음 V3c 실험 완료 결과를 기존 Notion 페이지에 반영해주세요.

## 작성 원칙

- 사람이 실험 전에 남기는 기술 연구노트처럼 간결하게 작성합니다.
- 아래에 제공된 코드, test 결과, 기존 artifact에서 확인된 사실만 사용합니다.
- V3c가 성능을 개선할 것이라고 미리 단정하지 않습니다.
- `획기적인`, `강력한`, `유의미한`, `주목할 만한`, `명확히 입증`,
  `종합적으로` 같은 홍보성·상투적 표현은 사용하지 않습니다.
- 실험 목적, 가설, acceptance gate를 같은 말로 반복하지 않습니다.
- confirmed fact와 아직 검증할 hypothesis를 별도 문단으로 구분합니다.
- metric과 CLI option은 아래 표기 그대로 유지하고 백틱으로 표시합니다.
- 현재 실험은 artifact 분석까지 끝났으므로 `completed`로 기록합니다.
- V3c short gate는 `FAIL`이며 multi-seed로 승격하지 않습니다.
- 같은 제목의 페이지가 있으면 업데이트하고, 없을 때만 새 페이지를 만듭니다.

## Notion 위치

- `5. Model Design Enhancement > Enhancement & Validation History`
- `TitanTPP V3: Mark-Conditioned Value Head` 설계 페이지와 상호 링크
- 기존 `TitanTPP V2/V3b Taxi Strict Matched-Budget Comparison e50` 결과 페이지와 연결
- Model Enhancement 범위이므로 `2. Confirm and Refine Topic` 아래에는 생성하지 않음
- 중복 페이지를 만들지 않습니다.

## 페이지 제목

- `TitanTPP V3c Detached Value-Encoder Route Smoke And Intermittent Screening e50`

## 기준 시각과 실행 환경

- 문서 작성 기준 시각: `2026-07-12 08:48:09 KST`
- 실험 시작 시각: `2026-07-12 08:59:54 KST`
- 실험 종료 시각: `2026-07-12 09:06:20 KST`
- 결과 확인 시각: `2026-07-12 11:02:34 KST`
- 실행 서버: `5090` (`192.168.0.71:22`)
- SSH user: `leekwanhyeong`
- project root: `/home/leekwanhyeong/workspace/paper_research`
- Python: `/opt/miniconda3/envs/ai_env/bin/python`
- tmux: `/opt/miniconda3/envs/ai_env/bin/tmux`
- tmux session: `titantpp_v3c_screen_e50_0712`
- conda env: `ai_env`

## 실험 상태

- 현재 상태: `completed`
- short gate 판정: `FAIL`
- 5090 코드 동기화 및 CUDA model-test preflight: `completed`
- preflight 확인 시각: `2026-07-12 08:55:37 KST`
- tmux session 시작: `2026-07-12 08:59:54 KST`
- Instacart e1 smoke: `completed`
- Intermittent seed-42 e50: `completed` (`50/50` epochs)
- 초기 실행 확인 시각: `2026-07-12 09:00:23 KST`
- 초기 확인 이후에는 지속적으로 polling하지 않음
- tmux와 GPU process가 종료된 상태에서 결과 확인
- 5090에 문제가 있어도 5080으로 자동 전환하지 않음

## 실험 배경

동일 seed-42 e50, `best_val_nll` held-out test의 Intermittent 결과는 아래와
같습니다.

| Variant | Total NLL | Marker NLL | Quantity MAE | Value MAE | Mark accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| V2 | `5.071916` | `1.016321` | `3.528298` | `0.153685` | `54.460%` |
| V3a | `5.062077` | `1.006122` | `3.058517` | `0.117189` | `53.437%` |
| V3b | `5.058310` | `1.004198` | `3.463607` | `0.150965` | `53.367%` |
| V3c | `5.143232` | `1.023542` | `3.613536` | `0.128854` | `53.674%` |

확인된 사실:

- V3a와 V3b는 V2보다 total NLL과 marker NLL이 낮았습니다.
- mark accuracy는 V2보다 V3a가 `1.023%p`, V3b가 `1.093%p` 낮았습니다.
- V3b의 mark-probability gate detachment만으로 Intermittent accuracy가
  회복되지 않았습니다.
- test sample의 `88.67%`를 차지하는 `1-9` quantity bucket은 V2 대비
  V3a `+9.21%`, V3b `+3.64%` MAE regression을 보였습니다.
- 이 결과는 shared-encoder gradient conflict를 증명하지 않습니다. V3c는 해당
  경로를 다음 가설로 분리해 확인하는 실험입니다.

## V3c 구조

공식 variant contract:

| Variant | `value_head_mode` | `qty_mark_gradient_mode` | `value_encoder_gradient_mode` |
| --- | --- | --- | --- |
| V2 | `shared` | `coupled` | `coupled` |
| V3a | `mark_conditioned_experts` | `coupled` | `coupled` |
| V3b | `mark_conditioned_experts` | `detached` | `coupled` |
| V3c | `mark_conditioned_experts` | `detached` | `detached` |

V3c gradient route:

```text
h_main = h_j
h_value = stop_gradient(h_j)

mark_logits = mark_head(h_main)
time_density = time_head(h_main)
value_by_mark = value_heads(h_value)
```

- marker/time loss는 Titan encoder를 계속 학습
- value/quantity loss는 shared value head와 mark-delta experts를 계속 학습
- value/quantity loss의 encoder, embedding, LMM gradient는 차단
- V3b의 detached mark-probability gate를 유지하므로 quantity loss는
  `mark_head`도 직접 학습하지 않음
- V3b와 V3c의 parameter, initialization, forward output, loss 값은 동일
- backward graph만 변경
- inference, time head, Titan memory, input feature, split, lookback은 변경하지 않음

## 구현 및 사전 검증

- `value_encoder_gradient_mode=coupled|detached` config/CLI/model propagation 완료
- V3c run path: `valueencgrad_detached`
- manifest, checkpoint, cache, history, validation/test, scale-wise, confusion,
  model-test 및 report grouping에 새 mode 반영
- 불완전한 V3c 조합은 실행 전에 fail-fast
- local `py_compile` 통과
- local `git diff --check` 통과
- static LMM을 포함한 focused pytest `18/18` 통과
- V3b/V3c state dictionary, forward, loss exact equivalence 통과
- isolated value/quantity loss에서 value-head gradient 유지 및
  encoder/embedding/LMM gradient 차단 확인
- marker/time/full loss에서 encoder와 해당 prediction head gradient 유지 확인
- local V3c `small_lmm` CPU model-test 통과
- 기본 `coupled` RMTPP/TitanTPP/THP regression model-test 통과
- 5090 `RTX5090-server`, NVIDIA GeForce RTX 5090 32 GB 확인
- 5090 remote Bash syntax와 대상 모듈 `py_compile` 통과
- 5090 `ai_env`에는 `pytest`가 없어 remote focused pytest는 미실행;
  별도 dependency 설치는 하지 않음
- 5090 CUDA model-test `success`: hidden shape `[2, 16, 64]`, NLL
  `4.944328`, parameter count `78,298`
- model-test summary에서 `mark_conditioned_experts / detached / detached`와
  device `cuda` 기록 확인
- model-test 종료 후 GPU compute process가 남지 않은 것을 확인
- Instacart와 Intermittent fixed-split 원본, train, validation, test, manifest
  입력 파일 존재 확인
- 5090 tmux `titantpp_v3c_screen_e50_0712`를 `08:59:54 KST`에 시작
- tmux 안에서 CUDA model-test 재실행 `success`: NLL `4.944328`
- Instacart e1 smoke는 `08:59:56 KST`에 시작해 `09:00:00 KST`에 완료
- Instacart fixed split: train `1,380`, validation `300`, test `300`,
  `max_seq_len=16`, lookback `10`
- Instacart epoch 1: train loss `4.672574`, validation NLL `3.257967`,
  validation accuracy `47.333%`, validation quantity MAE `6.135745`
- NaN, Traceback, ERROR 없이 Instacart run과 summary 생성을 완료
- Intermittent e50은 `09:00:00 KST`에 시작했으며 fixed split과 run config
  진입을 확인
- 초기 확인 시 PID `1885208`, GPU memory `712 MiB` 사용 확인
- 초기 확인 이후 지속적으로 polling하지 않음
- Intermittent e50은 `09:06:20 KST`에 `50/50` epochs로 완료
- best validation NLL은 epoch `5`의 `5.608073`
- manifest, log, summary, test summary, histories, validation/test scale-wise,
  report, plots 순서로 artifact 확인 완료
- NaN, Traceback, RuntimeError, ERROR 없음
- 완료 artifact를 local `search_artifacts/model_enhancement_v3c_inter_short_e50_0712`
  경로로 동기화

관련 설계 문서:

- `/Users/igwanhyeong/PycharmProjects/paper_research/.agents/results/architecture/adr-titantpp-v3c-detached-value-encoder-route.md`
- `/Users/igwanhyeong/PycharmProjects/paper_research/simple_lab_test/search/model_enhancement_strategy.md`

준비된 실행 스크립트:

- local: `/Users/igwanhyeong/PycharmProjects/paper_research/simple_lab_test/search/scripts/run_v3c_smoke_short_screening_0712.sh`
- 5090: `/home/leekwanhyeong/workspace/paper_research/simple_lab_test/search/scripts/run_v3c_smoke_short_screening_0712.sh`
- local/5090 SHA-256:
  `df221eb38877e16fdd071689c847dbf55157d07e47894766d0eaf077503b4398`
- `set -euo pipefail`을 사용하므로 model-test 또는 Instacart smoke가 실패하면
  다음 단계로 넘어가지 않음

## 실험 목적

- value/quantity loss가 shared Titan encoder를 업데이트하는 경로가 Intermittent
  mark accuracy 하락의 주요 원인인지 분리합니다.
- marker/time objective에 encoder update ownership을 주었을 때 mark accuracy가
  V2 수준으로 회복되는지 확인합니다.
- encoder detachment가 quantity/value 성능을 과도하게 훼손하지 않는지 전체 및
  scale-wise metric으로 확인합니다.
- 이 실험은 Intermittent 전용 V3c screening이며 Taxi V3b 결정을 변경하지 않습니다.

## 실험 계획

| Order | Stage | Dataset | Candidate | Epochs | Seed | 실행 조건 |
| ---: | --- | --- | --- | ---: | ---: | --- |
| 0 | 5090 preflight | synthetic | `small_lmm` | - | 42 | `PASS` at `2026-07-12 08:55:37 KST` |
| 1 | integration smoke gate | Instacart top 20 series | `small_lmm` | 1 | 42 | `PASS`; completed at `09:00:00 KST` |
| 2 | short screening | Intermittent | `small_lmm` | 50 | 42 | `completed`; short gate `FAIL` |

실행 순서 규칙:

- preflight가 실패하면 smoke를 시작하지 않습니다.
- Instacart smoke에서 NaN, Traceback, artifact/schema 오류가 발생하면
  Intermittent e50을 시작하지 않습니다.
- Intermittent seed 42 short gate를 분석하기 전에는 multi-seed를 실행하지 않습니다.
- 모든 실행은 5090에서만 진행합니다.

## 공통 실험 조건

- model: `titantpp`
- lr: `1e-3`
- split_mode: `fixed`
- value_input_mode: `residual`
- train_loss_scope: `target_only`
- loss_mode: `hybrid`
- value_head_activation: `identity`
- value_head_mode: `mark_conditioned_experts`
- qty_mark_gradient_mode: `detached`
- value_encoder_gradient_mode: `detached`
- eval selections: `best_val_nll,best_score,final`
- device: `cuda`
- force_rerun: enabled
- stop_on_error: enabled

Dataset-specific 조건:

- Instacart: top 20 series, batch size `16`, lookback `10`, max_seq_len `16`
- Intermittent: batch size `128`, lookback `52`, effective max_seq_len `16`

## 실행 명령어

5090 접속 및 tmux 생성:

```bash
ssh 5090
cd /home/leekwanhyeong/workspace/paper_research
/opt/miniconda3/envs/ai_env/bin/tmux new-session -s titantpp_v3c_screen_e50_0712
```

tmux 안에서 준비된 전체 순서 실행:

```bash
/home/leekwanhyeong/workspace/paper_research/simple_lab_test/search/scripts/run_v3c_smoke_short_screening_0712.sh
```

tmux 안에서 사용할 공통 환경:

```bash
set -euo pipefail
export PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research
export PYTHON_BIN=/opt/miniconda3/envs/ai_env/bin/python
export ENTRYPOINT="$PROJECT_ROOT/simple_lab_test/search/tpp_experiment.py"
export ARTIFACT_ROOT="$PROJECT_ROOT/search_artifacts"
export MPLCONFIGDIR="/tmp/matplotlib-${USER}"
mkdir -p "$MPLCONFIGDIR"
```

5090 V3c preflight model-test:

```bash
"$PYTHON_BIN" "$ENTRYPOINT" model-test \
  --models titantpp \
  --titan-candidates small_lmm \
  --device cuda \
  --seq-len 16 \
  --num-marks 12 \
  --rmtpp-hidden-dim 64 \
  --value-head-mode mark_conditioned_experts \
  --qty-mark-gradient-mode detached \
  --value-encoder-gradient-mode detached \
  --left-pad \
  --stop-on-error \
  --output-dir "$ARTIFACT_ROOT/model_enhancement_v3c_model_test_0712"
```

Instacart top-20 e1 smoke:

```bash
mkdir -p "$ARTIFACT_ROOT/model_enhancement_v3c_insta_smoke_e1_0712/logs"
"$PYTHON_BIN" "$ENTRYPOINT" long-epoch \
  --base-dir "$ARTIFACT_ROOT/model_enhancement_v3c_insta_smoke_e1_0712" \
  --datasets insta_market_basket \
  --models titantpp \
  --titan-candidates small_lmm \
  --epochs 1 \
  --seeds 42 \
  --lr 1e-3 \
  --batch-size 16 \
  --lookback-weeks 10 \
  --max-seq-len 16 \
  --insta-max-series 20 \
  --split-mode fixed \
  --value-head-activation identity \
  --value-head-mode mark_conditioned_experts \
  --qty-mark-gradient-mode detached \
  --value-encoder-gradient-mode detached \
  --value-input-mode residual \
  --train-loss-scope target_only \
  --loss-mode hybrid \
  --eval-selections best_val_nll,best_score,final \
  --device cuda \
  --force-rerun \
  --stop-on-error \
  2>&1 | tee -a "$ARTIFACT_ROOT/model_enhancement_v3c_insta_smoke_e1_0712/logs/run.log"
```

Instacart smoke 통과 후 Intermittent seed-42 e50:

```bash
mkdir -p "$ARTIFACT_ROOT/model_enhancement_v3c_inter_short_e50_0712/logs"
"$PYTHON_BIN" "$ENTRYPOINT" long-epoch \
  --base-dir "$ARTIFACT_ROOT/model_enhancement_v3c_inter_short_e50_0712" \
  --datasets intermittent \
  --models titantpp \
  --titan-candidates small_lmm \
  --epochs 50 \
  --seeds 42 \
  --lr 1e-3 \
  --batch-size 128 \
  --lookback-weeks 52 \
  --max-seq-len 16 \
  --split-mode fixed \
  --value-head-activation identity \
  --value-head-mode mark_conditioned_experts \
  --qty-mark-gradient-mode detached \
  --value-encoder-gradient-mode detached \
  --value-input-mode residual \
  --train-loss-scope target_only \
  --loss-mode hybrid \
  --eval-selections best_val_nll,best_score,final \
  --device cuda \
  --force-rerun \
  --stop-on-error \
  2>&1 | tee -a "$ARTIFACT_ROOT/model_enhancement_v3c_inter_short_e50_0712/logs/run.log"
```

## Artifact 경로

5090:

- `/home/leekwanhyeong/workspace/paper_research/search_artifacts/model_enhancement_v3c_model_test_0712`
- `/home/leekwanhyeong/workspace/paper_research/search_artifacts/model_enhancement_v3c_insta_smoke_e1_0712`
- `/home/leekwanhyeong/workspace/paper_research/search_artifacts/model_enhancement_v3c_inter_short_e50_0712`

완료 후 local sync 대상:

- `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_v3c_model_test_0712`
- `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_v3c_insta_smoke_e1_0712`
- `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_v3c_inter_short_e50_0712`

비교 기준 local artifact:

- V2: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_v2_inter_short_e50_0710`
- V3a: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_v3_inter_short_e50_0710`
- V3b: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_v3b_inter_short_e50_0710`

## Smoke Gate

- model-test와 Instacart e1 run 모두 정상 종료
- NaN, Traceback, ERROR 없음
- manifest와 model-test output에 아래 조합 기록
  `mark_conditioned_experts / detached / detached`
- run path에 `valuehead_mark_conditioned_experts`, `qtymarkgrad_detached`,
  `valueencgrad_detached`가 모두 포함
- validation/test/scale-wise/report artifact 생성 확인
- smoke gate를 통과한 경우에만 Intermittent e50 진행

## Intermittent Seed-42 Short Gate

비교 기준은 V2 e50 `best_val_nll` held-out test입니다.

- NaN, Traceback, ERROR, artifact collision 없음
- mark accuracy gap vs V2: `>= -0.25%p`
- total NLL regression vs V2: `<= 0.5%`
- marker NLL regression vs V2: `<= 2%`
- quantity MAE regression vs V2: `<= 2%`; matching or improvement가 목표
- value MAE regression vs V2: `<= 2%`
- test share `>= 5%`인 quantity bucket regression vs V2: `<= 5%`
- validation과 held-out test에서 marker accuracy와 quantity safety 방향 일치

실제 판정:

| Gate | V3c vs V2 | Result |
| --- | ---: | --- |
| runtime/artifact integrity | `50/50`, 오류 없음 | PASS |
| mark accuracy gap `>= -0.25%p` | `-0.786%p` | FAIL |
| total NLL regression `<= 0.5%` | `+1.406%` | FAIL |
| marker NLL regression `<= 2%` | `+0.710%` | PASS |
| quantity MAE regression `<= 2%` | `+2.416%` | FAIL |
| value MAE regression `<= 2%` | `-16.157%` | PASS |
| share `>= 5%` bucket regression `<= 5%` | `1-9: +12.935%` | FAIL |
| validation/test direction agreement | aggregate quantity 방향 불일치 | FAIL |

주요 scale:

| Scale | Test share | V2 MAE | V3c MAE | Change |
| --- | ---: | ---: | ---: | ---: |
| `1-9` | `88.67%` | `1.061286` | `1.198562` | `+12.935%` |
| `10-99` | `10.66%` | `9.787868` | `9.469296` | `-3.255%` |

## 판정 분기

- marker gate와 quantity safety가 모두 통과하면 Intermittent seeds
  `42,52,62` e50 confirmation 대상으로 승격합니다.
- marker gate는 통과하지만 quantity safety가 실패하면 V3c를 승격하지 않고
  partial encoder-gradient 설계를 검토합니다.
- marker gate가 실패하면 추가 encoder detachment는 중단하고 class imbalance 또는
  ordinal marker objective 분석의 우선순위를 높입니다.
- quantity만 개선되고 marker gate가 실패하면 V2를 유지합니다.
- Taxi는 strict matched confirmation을 통과한 V3b를 유지하며 이번 V3c 실험에
  포함하지 않습니다.

## 결과 작성란

- 결과 상태: `completed`
- V3c short gate: `FAIL`
- Intermittent multi-seed: `진행하지 않음`
- 아래 artifact reading order를 그대로 따라 확인했습니다.

```text
1. experiment_manifest.json
2. logs/run.log
3. leaderboard/summary.csv
4. leaderboard/test_summary.csv
5. leaderboard/histories.csv
6. leaderboard/scale_wise_summary.csv
7. leaderboard/test_scale_wise_summary.csv
8. paper_outputs/report.md
9. paper_outputs/plots/
```

Primary `best_val_nll` held-out test:

- total NLL `5.143232`: V2 대비 `+1.406%`
- marker NLL `1.023542`: V2 대비 `+0.710%`
- time NLL `4.119690`: V2 대비 `+1.580%`
- quantity MAE `3.613536`: V2 대비 `+2.416%`
- value MAE `0.128854`: V2 대비 `-16.157%`
- mark accuracy `53.674%`: V2 대비 `-0.786%p`

비교 해석:

- V3c는 V3b보다 mark accuracy를 `+0.307%p` 회복했지만 V2 gap을
  `0.25%p` 이내로 줄이지 못했습니다.
- V3c는 V3b 대비 total NLL `+1.679%`, marker NLL `+1.926%`, quantity MAE
  `+4.329%`로 악화됐습니다.
- validation에서는 aggregate quantity MAE가 V2 대비 `-1.601%`였지만 test에서는
  `+2.416%`여서 quantity safety 방향이 일치하지 않았습니다.
- final checkpoint는 primary checkpoint보다 좋아졌지만, V2 대비 total NLL
  `+0.861%`, mark accuracy `-0.295%p`, `1-9` MAE `+13.093%`로 strict gate를
  여전히 통과하지 못했습니다.
- primary checkpoint는 기존 protocol대로 validation NLL로 선택하며 held-out test나
  final 결과로 다시 선택하지 않습니다.

최종 결정:

- V3c를 Intermittent enhancement로 승격하지 않습니다.
- seeds `42,52,62` multi-seed confirmation을 실행하지 않습니다.
- Intermittent 공식 baseline은 V2를 유지합니다.
- full value-encoder detachment가 mark-accuracy 하락의 주원인이라는 가설의
  우선순위를 낮춥니다.
- 다음 모델 강화는 추가 encoder detachment보다 class imbalance 또는 ordinal marker
  objective를 우선 검토합니다.
- Taxi의 V3b confirmed decision은 변경하지 않습니다.

## 초기 해석 가설

- V3b의 Intermittent mark-accuracy 하락에 value/quantity encoder gradient가 주요하게
  기여했다면 V3c에서 V2 accuracy gap이 `0.25%p` 이내로 줄어야 합니다.
- mark accuracy가 회복되면서 quantity가 악화되면 full detachment가 너무 강한 것으로
  보고 partial gradient routing을 별도 설계합니다.
- mark accuracy가 회복되지 않으면 shared-encoder gradient 가설의 우선순위를 낮추고
  class imbalance와 ordinal marker objective를 검토합니다.
- seed 42 e50 결과는 screening 근거이며 최종 모델 우월성 주장에 사용하지 않습니다.

실제 결과는 세 번째 분기에 해당합니다. V3c의 accuracy gap은 `-0.786%p`로 남았고
quantity 및 major scale safety도 함께 실패했으므로 추가 encoder detachment는 중단합니다.

Notion 업데이트가 끝나면 작성한 페이지 제목, 현재 상태, 연결한 관련 페이지와
다음 모델 강화 작업만 짧게 보고해주세요.
