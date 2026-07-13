다음 TitanTPP V5a Intermittent seed-42 e50 validation screening 계획을 Notion에 정리해주세요.

## 작성 위치

- `5. Model Design Enhancement > Enhancement & Validation History`
- 기존 `TitanTPP V5a Ordinal Marker Loss Contract And Acceptance Gate`와
  `TitanTPP V5a CUDA Model-Test And Instacart e1 Smoke` 페이지에 연결합니다.
- Model Enhancement 작업이므로 `2. Confirm and Refine Topic`에는 작성하지 않습니다.
- 같은 제목의 페이지가 있으면 업데이트하고 중복 페이지는 만들지 않습니다.

## 페이지 제목

- `TitanTPP V5a Intermittent Seed-42 e50 Validation Screening`

## 기준 시각과 실행 환경

- 문서 준비일: `2026-07-12 KST`
- 실험 시작 시각: `2026-07-12 21:16:02 KST`
- 실행 서버: `5090` (`192.168.0.71:22`)
- project root: `/home/leekwanhyeong/workspace/paper_research`
- Python: `/opt/miniconda3/envs/ai_env/bin/python`
- conda env: `ai_env`
- tmux session: `inter_v5a_rps_e50_0712`

## 현재 상태

- 상태: `in progress`
- V5a local focused tests: `20 passed`
- 5090 CUDA model-test와 Instacart e1 smoke: `PASS`
- V2 validation-only reference freeze: `completed`
- V5a Intermittent seed-42 e50: training loop 진입 확인
- 5090 preflight 완료 시각: `2026-07-12 21:11:43 KST`
- GPU: NVIDIA GeForce RTX 5090, idle
- tmux `inter_v5a_rps_e50_0712`: 실행 중
- 초기 확인 시각: `2026-07-12 21:16:24 KST`
- 초기 GPU process: PID `2368792`, memory `710 MiB`
- 초기 확인 이후 지속 polling하지 않음
- 모든 실행은 5090에서만 진행
- 초기 config, GPU process, 첫 epoch 진입까지만 확인하고 지속 polling하지 않음

Preflight 확인:

- V2 `best_val_nll` checkpoint와 marked cache를 local에서 5090 원래 경로로 동기화
- checkpoint SHA-256:
  `1a901eb2ac912537e25b6c798978870a6f650857b41642f2a0b773030cc103c0`
- marked cache SHA-256:
  `dab4d8a7217f9c14d1c2336f649aef9ddaf2ba440d074e446d8fd5cc41506a30`
- validation reference evaluator local/remote SHA-256:
  `8022c35d1d17fe752b28625eeb429a91310d804c8fecd30aa6ab3f61bd6dc017`
- e50 실행 script local/remote SHA-256:
  `92c39351d61400eab1d019e20e593e2b347b57ea8fc9ca455bc74f9fc8b0c1bb`
- remote Bash syntax와 Python `py_compile` 통과
- 5090 CUDA dry-run에서 V2 checkpoint `state_dict=strict_ok`
- Intermittent fixed split with/train/validation/test/manifest 입력 파일 존재
- dry-run 종료 후 GPU compute process 없음

실행 진입 확인:

- V2 reference validation samples: `41,901`
- V2 total/marker NLL: `5.666520 / 0.991274`
- V2 normalized RPS: `0.035283`
- V2 mark accuracy/MAE: `0.572492 / 0.487411`
- 기존 best epoch-19 validation metric을 재현
- V5a fixed split samples: train `136,256`, validation `41,901`, test `41,344`
- V5a config: `shared/coupled/coupled`, `ce_rps`, lambda `0.1`
- `batch_size=128`, `lookback=52`, `max_seq_len=16`
- 아직 완료 epoch와 validation gate 결과는 확인하지 않음

## 실험 목적

- ordered magnitude mark에 CE+normalized RPS를 추가했을 때 V2보다 validation RPS와
  mark MAE가 개선되는지 확인합니다.
- marker accuracy, CE likelihood, time likelihood, quantity/value 성능이 허용 범위 안에
  남는지 확인합니다.
- 이 단계는 seed-42 validation-only screening이며 multi-seed 결론이 아닙니다.

## Matched Variant Contract

| Field | V2 reference | V5a |
| --- | --- | --- |
| dataset/split | Intermittent fixed | same |
| candidate | `small_lmm` | same |
| seed/epochs | `42 / 50` | same |
| lr/batch | `1e-3 / 128` | same |
| lookback/max sequence | `52 / 16` | same |
| value input/loss scope | `residual / target_only` | same |
| quantity loss | `hybrid` | same |
| value head | `shared` | same |
| quantity mark gradient | `coupled` | same |
| value encoder gradient | `coupled` | same |
| marker loss | `ce` | `ce_rps` |
| ordinal weight | `0.0` | `0.10` |

V3 mark-conditioned expert와 gradient detachment는 사용하지 않습니다.

## Pre-Run V2 Validation Reference Freeze

기존 V2 `best_val_nll` checkpoint를 현재 evaluator로 validation에만 재평가합니다.
V5a 학습보다 먼저 실행하며 held-out test dataset은 생성하거나 읽지 않습니다.

고정된 기존 값:

| Metric | V2 validation reference |
| --- | ---: |
| Best epoch | `19` |
| Total NLL | `5.666520` |
| Marker NLL | `0.991274` |
| Time NLL | `4.675246` |
| Mark accuracy | `57.249%` |
| Balanced accuracy | `42.664%` |
| Macro F1 | `43.302%` |
| Mark MAE | `0.487411` |
| Adjacent accuracy | `94.377%` |
| Mark-0 recall | `75.543%` |
| Mark-1 recall | `49.616%` |
| Quantity MAE | `3.060182` |
| Value MAE | `0.146300` |

Normalized RPS `0.035283`을 V5a metric을 보기 전에 고정했습니다.

## 실행 순서

1. V2 `best_val_nll` checkpoint validation-only 재평가
2. normalized RPS와 기존 NLL/accuracy 재현 여부 확인
3. V5a seed-42 e50 실행
4. 완료 후 validation artifact만 분석
5. held-out test metric은 coefficient와 multi-seed candidate가 고정될 때까지 잠금

## 실행 명령어

5090 tmux:

```bash
ssh 5090
cd /home/leekwanhyeong/workspace/paper_research
/opt/miniconda3/envs/ai_env/bin/tmux new-session -s inter_v5a_rps_e50_0712
```

tmux 안에서:

```bash
env \
  PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
  PYTHON_BIN=/opt/miniconda3/envs/ai_env/bin/python \
  bash /home/leekwanhyeong/workspace/paper_research/simple_lab_test/search/scripts/run_v5a_inter_seed42_e50_0712.sh
```

Local script:

```text
/Users/igwanhyeong/PycharmProjects/paper_research/simple_lab_test/search/scripts/run_v5a_inter_seed42_e50_0712.sh
```

## Artifact 경로

V2 validation reference:

```text
server: /home/leekwanhyeong/workspace/paper_research/search_artifacts/model_enhancement_v2_inter_validation_reference_v5a_0712
local: /Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_v2_inter_validation_reference_v5a_0712
```

V5a e50:

```text
server: /home/leekwanhyeong/workspace/paper_research/search_artifacts/model_enhancement_v5a_inter_short_e50_0712
local: /Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_v5a_inter_short_e50_0712
```

## Seed-42 Validation Gate

Ordinal benefit:

- normalized RPS improvement `>= 1%`
- mark MAE improvement `>= 1%`
- balanced accuracy 또는 macro F1 중 하나 `>= +0.50%p`, 다른 하나 `>= -0.25%p`

Classification safety:

- mark accuracy gap `>= -0.25%p`
- mark-1 recall gap `>= -1.00%p`
- mark-0 recall gap `>= -2.00%p`
- adjacent accuracy gap `>= -0.25%p`

Likelihood/task safety:

- marker NLL regression `<= 1%`
- total NLL regression `<= 0.5%`
- time NLL regression `<= 0.5%`
- quantity MAE와 value MAE regression 각각 `<= 2%`
- validation share `>= 5%` bucket의 quantity MAE regression `<= 5%`

## Validation-Only Lambda Branch

- 전체 gate 통과: lambda `0.10` 고정 후 strict multi-seed 준비
- safety 통과, ordinal benefit 실패: lambda `0.20` 한 번만 screening
- ordinal benefit 통과, safety 실패: lambda `0.05` 한 번만 screening
- benefit과 safety가 모두 실패하거나 단일 조정도 실패: V5a 중단

## Test Lock

- runner가 held-out test artifact를 생성하더라도 metric content를 읽지 않습니다.
- runtime 확인은 file existence까지만 허용합니다.
- coefficient와 multi-seed candidate는 validation만으로 결정합니다.
- held-out test는 strict matched multi-seed validation 이후 frozen audit에서만 읽습니다.

## 결과 작성란

- 실험 종료 시각, 완료 epoch, NaN/Traceback 여부를 추가
- V2 reference RPS를 먼저 기록
- V5a best validation NLL epoch 기준으로 gate 표 작성
- `nll_marker`, `nll_time`, normalized RPS, mark MAE와 class `0/1` recall을 분리
- validation scale-wise quantity safety를 확인
- test metric은 기록하지 않음

페이지 마지막 Next:

```text
Next: 5090 tmux에서 V2 validation reference freeze 후 V5a seed-42 e50 실행
```
