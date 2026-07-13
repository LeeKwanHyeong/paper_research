다음 실험 시작 내용을 Notion에 정리해주세요.

위치:
- `2. Confirm and Refine Topic > Model Validation`
- 기존 `5. Model Design Enhancement`의 TitanTPP V3/V3b 설계 페이지가 있으면 상호 링크

페이지 제목:
- `TitanTPP V3b Detached Quantity Gate Smoke And Short Screening e50`

기준 시각:
- 실험 시작 시각: `2026-07-10 14:15:50 KST`
- 실행 서버: `5080`
- tmux session: `titantpp_v3b_screen_e50_0710`
- conda env: `ai_env`

실험 상태:
- `in progress`

실험 목적:
- V3a mark-conditioned value head의 quantity 개선 가능성은 유지하면서 quantity loss가 mark probability gate를 직접 최적화해 marker 성능을 훼손하는 문제를 분리합니다.
- V3b는 V3a와 같은 순전파 값을 사용하되 expected quantity 계산에서 mark probability만 detach하여 quantity loss의 mark-head gradient를 차단합니다.
- Instacart mini smoke를 실행 gate로 사용한 뒤 Intermittent와 Yellow Trip Hourly에서 V2/V3a artifact 대비 V3b e50 성능을 screening합니다.

사전 검증:
- local `git diff --check` 및 `py_compile` 통과
- local focused pytest `9/9` 통과
- isolated quantity loss에서 V3a mark-head gradient 존재, V3b mark-head gradient 없음 확인
- V3b에서도 value head, mark-delta expert, encoder gradient 유지 확인
- full hybrid loss에서는 marker CE를 통해 V3b mark-head gradient가 유지됨을 확인
- local V3a/V3b forward 및 `nll()` exact equivalence 확인
- local shared RMTPP/TitanTPP/THP regression model-test 통과
- local Instacart top-20 e1 official long-epoch integration smoke 통과
- 5080 `py_compile` 통과
- 5080 V3a/V3b GPU model-test exact equivalence 확인: small NLL `4.261527`, mid NLL `4.160898`

V3b 구조:
- `value_head_mode=mark_conditioned_experts`
- `qty_mark_gradient_mode=detached`
- V3a와 동일한 shared residual plus per-mark zero-init delta experts 사용
- `softmax(mark_logits).detach()`는 reconstructed expected quantity gate에만 적용
- marker CE, time head, Titan memory backbone, data split, lookback은 변경하지 않음
- V2는 `shared/coupled`, V3a는 `experts/coupled`, V3b는 `experts/detached`로 구분

실험 계획:

| Order | Experiment | Dataset | Candidate | Epochs | Seed | Gradient mode |
| --- | --- | --- | --- | ---: | ---: | --- |
| 1 | V3b smoke gate | Instacart top 20 series | `small_lmm` | 1 | 42 | `detached` |
| 2 | V3b short screening | Intermittent | `small_lmm` | 50 | 42 | `detached` |
| 3 | V3b short screening | Yellow Trip Hourly | `mid_lmm` | 50 | 42 | `detached` |

공통 screening 조건:
- model: `titantpp`
- lr: `1e-3`
- split_mode: `fixed`
- value_input_mode: `residual`
- train_loss_scope: `target_only`
- loss_mode: `hybrid`
- value_head_activation: `identity`
- value_head_mode: `mark_conditioned_experts`
- qty_mark_gradient_mode: `detached`
- eval selections: `best_val_nll,best_score,final`
- device: `cuda`

Dataset-specific 조건:
- Instacart: top 20 series, batch size `16`, lookback `10`, max_seq_len `16`
- Intermittent: batch size `128`, lookback `52`, effective max_seq_len `16`
- Yellow Trip Hourly: batch size `128`, lookback `168`, max_seq_len `256`

실행 명령어:

```bash
ssh 5080
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai_env
tmux attach -t titantpp_v3b_screen_e50_0710
```

실제 tmux 실행 대상:

```bash
PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
PYTHON_BIN=/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python \
bash /home/leekwanhyeong/workspace/paper_research/simple_lab_test/search/scripts/run_v3b_smoke_short_screening_0710.sh
```

Artifact 경로:
- `~/workspace/paper_research/search_artifacts/model_enhancement_v3b_insta_smoke_e1_0710`
- `~/workspace/paper_research/search_artifacts/model_enhancement_v3b_inter_short_e50_0710`
- `~/workspace/paper_research/search_artifacts/model_enhancement_v3b_taxi_short_e50_0710`

비교 기준 artifact:
- V2 Intermittent: `model_enhancement_v2_inter_short_e50_0710`
- V3a Intermittent: `model_enhancement_v3_inter_short_e50_0710`
- V2 Taxi: `model_enhancement_v2_taxi_short_e50_0710`
- V3a Taxi: `model_enhancement_v3_taxi_short_e50_0710`

Acceptance gate:
- V2 대비 total NLL 악화 `<= 0.5%`
- Taxi marker NLL 악화 `<= 2%`
- V2 대비 mark accuracy gap `<= 0.25%p`
- V3a가 얻은 quantity MAE 개선 폭의 절반 이상 유지
- sample share가 `>= 5%`인 scale bucket에서 quantity MAE 악화 `<= 5%`

결과 작성란:
- 결과는 실행 완료 후 artifact reading order에 따라 업데이트 예정
- 확인 순서: manifest, `logs/run.log`, summary, test_summary, histories, scale-wise validation, scale-wise test, report, plots
- 확인 지표: validation/test total NLL, marker NLL, time NLL, mark accuracy, quantity MAE, value MAE, best epoch, final-minus-best degradation, scale-wise quantity MAE, NaN/Traceback

초기 해석 가설:
- V3a의 Taxi marker NLL 악화가 quantity loss의 gate gradient 간섭 때문이라면 V3b에서 marker NLL과 mark accuracy가 V2 방향으로 회복되어야 합니다.
- detach는 순전파 함수를 바꾸지 않으므로 학습 전 값은 같고, 학습 trajectory와 최종 파라미터에서만 차이가 발생해야 합니다.
- quantity 개선이 크게 사라지면 coupled gate gradient가 V3a quantity gain의 주요 원인이었다고 해석합니다.
- e50 seed 42는 screening 근거이며 최종 우월성 주장의 근거로 사용하지 않습니다.
