다음 실험 시작 내용을 Notion에 정리해주세요.

위치:
- `2. Confirm and Refine Topic > Model Validation`
- 기존 `TitanTPP V3b Detached Quantity Gate Smoke And Short Screening e50` 결과 페이지 하위에 생성
- `5. Model Design Enhancement`의 V3b 설계 페이지와 상호 링크

페이지 제목:
- `TitanTPP V3b Taxi Multi-Seed Confirmation e50`

기준 시각:
- 실험 시작 시각: `2026-07-10 14:54:27 KST`
- 실행 서버: `5080`
- tmux session: `taxi_v3b_multiseed_e50_0710`
- conda env: `ai_env`

서버 선택 사유:
- 프로토콜 우선순위에 따라 5090을 먼저 확인했고 GPU는 비어 있었음
- 그러나 5090에는 tmux 바이너리가 설치되어 있지 않아 표준 장시간 실행 조건을 만족하지 못함
- 공유 서버의 시스템 패키지를 임의 설치하거나 nohup으로 우회하지 않고, tmux와 ai_env가 검증된 보조 서버 5080으로 전환

실험 상태:
- `in progress`

실험 목적:
- seed 42 Taxi screening에서 확인한 V3b detached quantity gate의 NLL, marker, quantity, mark-accuracy 개선이 seed에 독립적인지 확인합니다.
- V3b를 Taxi 전용 승격 후보로 유지할 수 있는지 평균, 표준편차, 최악 seed, scale-wise 일관성으로 판정합니다.
- 기존 V2 multi-seed e200 `mid_lmm` 결과를 보수적인 baseline으로 사용합니다.

사전 검증:
- local script `bash -n` 통과
- local V3b `mid_lmm` model-test 통과
- 5090 V3b code `py_compile` 및 GPU model-test 통과
- 5080 V3b GPU model-test 통과
- 5090과 5080 model-test NLL `4.376733`으로 일치

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
- value_head_mode: `mark_conditioned_experts`
- qty_mark_gradient_mode: `detached`
- value_input_mode: `residual`
- train_loss_scope: `target_only`
- loss_mode: `hybrid`
- eval selections: `best_val_nll,best_score,final`
- device: `cuda`

실행 명령어:

```bash
ssh 5080
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai_env
tmux attach -t taxi_v3b_multiseed_e50_0710
```

실제 tmux 실행 대상:

```bash
PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
PYTHON_BIN=/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python \
bash /home/leekwanhyeong/workspace/paper_research/simple_lab_test/search/scripts/run_v3b_taxi_multiseed_e50_0710.sh
```

Artifact 경로:
- server: `~/workspace/paper_research/search_artifacts/model_enhancement_v3b_taxi_multiseed_e50_0710`
- local sync target: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_v3b_taxi_multiseed_e50_0710`

비교 기준 artifact:
- V2 multi-seed baseline: `model_enhancement_v2_hybrid_e200_0705`, Taxi `mid_lmm`, seeds `42,52,62`
- V3b seed-42 screening: `model_enhancement_v3b_taxi_short_e50_0710`

Confirmation gate:
- seeds `42,52,62`의 `3/3` run 완료, NaN/Traceback 없음
- V2 multi-seed 대비 mean total NLL 악화 `<= 0.5%`
- V2 multi-seed 대비 mean marker NLL 악화 `<= 2%`
- V2 multi-seed 대비 mean mark accuracy regression `<= 0.25%p`
- seed-42 quantity 개선폭의 절반 이상 유지: V2 multi-seed 대비 mean quantity MAE 개선 `>= 18.06%`
- seed-matched V2 대비 total NLL, marker NLL, quantity MAE를 동시에 개선한 seed가 `2/3` 이상
- test share가 `>= 5%`인 모든 scale bucket에서 V2 multi-seed mean 대비 quantity MAE `5%` 초과 regression 없음
- mean뿐 아니라 standard deviation, coefficient of variation, worst seed도 함께 기록

결과 작성란:
- 결과는 완료 후 artifact reading order에 따라 업데이트 예정
- 확인 순서: manifest, run.log, summary, test_summary, histories, validation scale-wise, test scale-wise, report, plots
- validation/test total NLL, marker NLL, time NLL, quantity MAE, value MAE, mark accuracy, best epoch, final degradation, scale-wise MAE를 seed별·평균으로 기록

초기 해석 가설:
- V3b 효과가 구조적이면 세 seed 평균에서도 V2 대비 marker NLL과 quantity MAE가 함께 개선되어야 합니다.
- seed 42만 강하고 52/62에서 효과가 사라지면 V3b는 seed-sensitive screening 결과로 판정합니다.
- mean 개선과 함께 worst seed 및 scale-wise 방향이 유지될 때만 Taxi 전용 variant 승격을 검토합니다.
