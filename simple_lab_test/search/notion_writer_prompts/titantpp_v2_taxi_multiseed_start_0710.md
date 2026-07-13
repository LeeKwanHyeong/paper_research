다음 실험 시작 내용을 Notion에 정리해주세요.

위치:
- `2. Confirm and Refine Topic > Model Validation`
- 기존 `TitanTPP V3b Taxi Multi-Seed Confirmation e50` 페이지 하위에 생성
- `5. Model Design Enhancement`의 V3b 결과 페이지와 상호 링크

페이지 제목:
- `TitanTPP Taxi V2 vs V3b Matched-Budget Baseline - V2 Multi-Seed e50`

기준 시각:
- 실험 시작 시각: `2026-07-10 16:42:36 KST`
- 실행 서버: `5090` (`192.168.0.71:22`)
- tmux session: `taxi_v2_multiseed_e50_0710`
- conda env: `/opt/miniconda3/envs/ai_env`

실험 상태:
- `in progress`

실험 목적:
- Taxi V3b seeds `42,52,62` e50 결과와 epoch budget까지 동일한 V2 baseline을 생성합니다.
- 기존 비교 baseline은 V2 e200이어서 best-NLL checkpoint의 quantity 성능이 epoch budget에 영향을 받을 수 있었습니다.
- 이번 V2 e50 결과로 V2/V3b의 architecture 차이만 남긴 strict matched-budget 비교를 구성합니다.

사전 검증:
- local script `bash -n` 통과
- local V2 shared/coupled `mid_lmm` model-test 통과
- 5090 V2 GPU model-test 통과: NLL `4.376733`
- 5090 GPU: RTX 5090, utilization `0%`, used memory `41 MiB`
- 5090 endpoint와 절대 Python/tmux 경로를 `TEST_SESSION_PROTOCOL.md`에 기록

실험 계획:
- dataset: `yellow_trip_hourly`
- model: `titantpp`
- candidate: `mid_lmm`
- epochs: `50`
- seeds: `42,52,62`
- lr: `1e-3`
- batch_size: `128`
- lookback: `168`
- max_seq_len: `256`
- split_mode: `fixed`
- value_head_activation: `identity`
- value_head_mode: `shared`
- qty_mark_gradient_mode: `coupled`
- value_input_mode: `residual`
- train_loss_scope: `target_only`
- loss_mode: `hybrid`
- eval selections: `best_val_nll,best_score,final`
- device: `cuda`

V2/V3b matched contract:

| Setting | V2 | V3b |
| --- | --- | --- |
| Dataset/candidate | Taxi `mid_lmm` | Taxi `mid_lmm` |
| Epochs/seeds | e50, `42,52,62` | e50, `42,52,62` |
| Input/loss | residual, target-only, hybrid | residual, target-only, hybrid |
| Value head | `shared` | `mark_conditioned_experts` |
| Quantity gate gradient | `coupled` | `detached` |

실행 명령어:

```bash
ssh 5090
/opt/miniconda3/envs/ai_env/bin/tmux attach -t taxi_v2_multiseed_e50_0710
```

실제 tmux 실행 대상:

```bash
PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
PYTHON_BIN=/opt/miniconda3/envs/ai_env/bin/python \
bash /home/leekwanhyeong/workspace/paper_research/simple_lab_test/search/scripts/run_v2_taxi_multiseed_e50_0710.sh
```

Artifact 경로:
- server: `~/workspace/paper_research/search_artifacts/model_enhancement_v2_taxi_multiseed_e50_0710`
- local sync target: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_v2_taxi_multiseed_e50_0710`

비교 대상:
- V3b matched artifact: `model_enhancement_v3b_taxi_multiseed_e50_0710`
- 이전 V2 long-budget reference: `model_enhancement_v2_hybrid_e200_0705`

결과 작성란:
- 결과는 사용자가 확인을 요청한 뒤 artifact reading order에 따라 업데이트 예정
- 확인 순서: manifest, run.log, summary, test_summary, histories, validation scale-wise, test scale-wise, report, plots
- mean/std/CV, seed별 best epoch, final degradation, NLL split, quantity MAE, mark accuracy, scale-wise MAE를 V3b와 비교

모니터링 정책:
- 초기 설정, GPU process, 첫 학습 진입까지만 확인
- 이후 지속 polling은 수행하지 않음
- 완료 여부와 결과 분석은 사용자 요청 시 확인

초기 해석 가설:
- 동일 e50 budget에서도 V3b가 marker NLL과 quantity MAE를 함께 개선하면 detached gate의 Taxi 효과를 더 강하게 지지합니다.
- V2 e50 quantity가 V2 e200보다 크게 좋아지면 기존 V3b quantity gain 일부는 epoch-budget mismatch 영향이었던 것으로 분리합니다.
- test 결과는 validation-selected checkpoint로만 비교하며 final checkpoint는 별도 안정성 지표로 기록합니다.
