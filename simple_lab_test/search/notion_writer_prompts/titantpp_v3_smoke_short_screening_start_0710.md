다음 실험 시작 내용을 Notion에 정리해주세요.

위치:
- `2. Confirm and Refine Topic > Model Validation`
- 기존 TitanTPP Model Enhancement 페이지가 있으면 그 하위에 생성

페이지 제목:
- `TitanTPP V3 Mark-Conditioned Value Head Smoke And Short Screening e50`

기준 시각:
- 실험 시작 시각: `2026-07-10 10:33:38 KST`
- 실행 서버: `5080`
- tmux session: `titantpp_v3_smoke_screen_e50_0710`
- conda env: `ai_env`

실험 상태:
- `in progress`

실험 목적:
- V3 shared-plus-mark-delta value head가 공식 long-epoch runner에서 정상 작동하는지 확인합니다.
- Instacart mini smoke를 gate로 사용하고, 성공한 경우에만 Intermittent/Yellow Trip V2-vs-V3 e50 screening을 순차 실행합니다.
- V2 quantity/stability 이득을 유지하면서 V3가 mark-conditioned residual을 학습할 수 있는지 초기 방향을 확인합니다.

사전 검증:
- local py_compile 통과
- local focused pytest `6/6` 통과
- local shared model-test 통과
- local V3 small/mid model-test 통과
- Titan-only shared/V3 zero-init equivalence 확인: NLL `3.890020`, qty mean `90.139053` 동일
- local Instacart 20-series e1 long-epoch integration smoke 통과
- 5080 py_compile 통과
- 5080 V3 small/mid model-test 통과
- 5080 `ai_env`에는 pytest가 없어 원격 pytest는 미실행이며 새 dependency는 설치하지 않음

V3 구조:
- `value_head_mode=mark_conditioned_experts`
- 기존 shared residual에 real mark별 zero-init delta를 추가
- residual metric은 true-mark expert 사용
- reconstructed quantity는 predicted-mark expert 사용
- time head, Titan memory, lookback, split은 변경하지 않음

실험 계획:

| Order | Experiment | Dataset | Candidate | Epochs | Seed | Value head |
| --- | --- | --- | --- | ---: | ---: | --- |
| 1 | V3 smoke | Instacart top 20 series | `small_lmm` | 1 | 42 | `mark_conditioned_experts` |
| 2 | V2 short baseline | Intermittent | `small_lmm` | 50 | 42 | `shared` |
| 3 | V3 short screening | Intermittent | `small_lmm` | 50 | 42 | `mark_conditioned_experts` |
| 4 | V2 short baseline | Yellow Trip Hourly | `mid_lmm` | 50 | 42 | `shared` |
| 5 | V3 short screening | Yellow Trip Hourly | `mid_lmm` | 50 | 42 | `mark_conditioned_experts` |

공통 screening 조건:
- model: `titantpp`
- lr: `1e-3`
- batch_size: `128`
- split_mode: `fixed`
- value_input_mode: `residual`
- train_loss_scope: `target_only`
- loss_mode: `hybrid`
- value_head_activation: `identity`
- eval selections: `best_val_nll,best_score,final`
- device: `cuda`

Dataset-specific 조건:
- Intermittent: lookback `52`, effective max_seq_len `16`
- Yellow Trip Hourly: lookback `168`, max_seq_len `256`

실행 명령어:

```bash
ssh 5080
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai_env
tmux attach -t titantpp_v3_smoke_screen_e50_0710
```

실제 tmux 실행 대상:

```bash
PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
PYTHON_BIN=/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python \
bash /home/leekwanhyeong/workspace/paper_research/simple_lab_test/search/scripts/run_v3_smoke_short_screening_0710.sh
```

Artifact 경로:
- `~/workspace/paper_research/search_artifacts/model_enhancement_v3_insta_smoke_e1_0710`
- `~/workspace/paper_research/search_artifacts/model_enhancement_v2_inter_short_e50_0710`
- `~/workspace/paper_research/search_artifacts/model_enhancement_v3_inter_short_e50_0710`
- `~/workspace/paper_research/search_artifacts/model_enhancement_v2_taxi_short_e50_0710`
- `~/workspace/paper_research/search_artifacts/model_enhancement_v3_taxi_short_e50_0710`

결과 작성란:
- 결과는 실행 완료 후 artifact reading order에 따라 업데이트 예정
- 확인 지표: validation/test score, total NLL, marker NLL, time NLL, quantity MAE, mark accuracy, best epoch, final-minus-best degradation, scale-wise quantity MAE, NaN/Traceback

초기 해석 가설:
- Zero-init 때문에 V3는 초기에는 V2와 같은 함수에서 시작하고, 학습 중 mark별 residual delta가 분화됩니다.
- Intermittent에서 V3가 V2 quantity MAE를 유지하거나 개선하면 conditional residual의 긍정적 신호로 봅니다.
- Taxi에서 V3가 quantity 이득을 유지하면서 marker NLL 또는 seed 안정성을 개선할 가능성을 확인합니다.
- e50 seed 42는 screening 근거이며 최종 우월성 주장의 근거로 사용하지 않습니다.
