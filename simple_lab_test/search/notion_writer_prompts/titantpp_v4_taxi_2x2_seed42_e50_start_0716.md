# TitanTPP V4 Taxi 2x2 Seed-42 e50 Screening

Notion의 `5. Model Design Enhancement` 아래 제목 2
`2026-07-16 | V4 Mark-Conditioned Time Head`, 제목 3
`Step 4. Taxi V2/V3b/V4a/V4b Validation-Only Screening`으로 정리한다.

## 상태

- 상태: `실행 준비 - CUDA/Instacart gate 대기`
- 실행 서버 / tmux: `5090 / titantpp_v4_taxi_2x2_e50_0716`
- 실행 시작 시각: `integration gate 통과 후 기록`

## 목적

- V4 time head의 효과를 V2와 Taxi V3b 각각에서 분리해 확인한다.
- validation 기준으로 V4b의 multi-seed 승격 여부를 결정한다.
- 이 단계에서는 held-out test metric을 생성하거나 읽지 않는다.

## Factorial 계약

| Variant | Value head | Quantity-mark gradient | Time head | 역할 |
| --- | --- | --- | --- | --- |
| V2 | shared | coupled | shared | 공통 control |
| V3b | mark-conditioned experts | detached | shared | Taxi value-head control |
| V4a | shared | coupled | mark-conditioned | time-head 단독 효과 |
| V4b | mark-conditioned experts | detached | mark-conditioned | Taxi 승격 후보 |

## 고정 조건

- Taxi fixed split, `mid_lmm`, seed `42`, e50
- batch `128`, lookback `168`, max sequence `256`, learning rate `1e-3`
- residual input, hybrid loss, target-only, plain marker CE
- strict reproducibility, `best_val_nll` checkpoint
- `evaluation_scope=validation_only`
- V4 pair별 time NLL `0.5%` 이상 개선
- total NLL `0.5%`, DT MAE `1%`, marker NLL `2%`, mark accuracy `-0.25%p`,
  quantity MAE `5%` guardrail

## 실행 명령어

```bash
ssh 5090 '/opt/miniconda3/envs/ai_env/bin/tmux new-session -d \
  -s titantpp_v4_taxi_2x2_e50_0716 \
  "env PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
  PYTHON_BIN=/opt/miniconda3/envs/ai_env/bin/python \
  PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  SOURCE_REVISION=<sync_commit_sha> \
  bash /home/leekwanhyeong/workspace/paper_research/simple_lab_test/search/scripts/run_titantpp_v4_taxi_2x2_seed42_e50_0716.sh"'
```

## 결과
