# TitanTPP V4 CUDA Model-Test And Instacart Smoke

Notion의 `5. Model Design Enhancement` 아래 제목 2
`2026-07-16 | V4 Mark-Conditioned Time Head`, 제목 3
`Step 3. 5090 CUDA Model-Test And Instacart e1 Smoke`로 정리한다.

## 상태

- 상태: `실행 준비`
- 실행 서버 / tmux: `5090 / titantpp_v4_cuda_insta_smoke_0716`
- 실행 시작 시각: `5090 동기화 후 기록`

## 목적

- V4a/V4b의 CUDA forward, loss, sampling 경로를 확인한다.
- Instacart top-20 e1에서 backward, checkpoint, validation artifact를 확인한다.
- validation-only 실행에서 test metric이 생성되지 않는지 확인한다.

## Factorial 계약

| Variant | Value head | Quantity-mark gradient | Time head |
| --- | --- | --- | --- |
| V4a | shared | coupled | mark-conditioned |
| V4b | mark-conditioned experts | detached | mark-conditioned |

## 고정 조건

- server: 5090, conda: `ai_env`, device: CUDA
- candidate: `mid_lmm`, seed: `42`
- Instacart: top-20, fixed split, e1, batch `16`, lookback `10`, max sequence `64`
- training: residual input, hybrid loss, target-only
- evaluation: `validation_only`, held-out test metric 미생성
- smoke 결과는 성능 순위나 V4 승격 근거로 사용하지 않음

## 실행 명령어

```bash
ssh 5090 '/opt/miniconda3/envs/ai_env/bin/tmux new-session -d \
  -s titantpp_v4_cuda_insta_smoke_0716 \
  "env PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
  PYTHON_BIN=/opt/miniconda3/envs/ai_env/bin/python \
  PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  SOURCE_REVISION=<sync_commit_sha> \
  bash /home/leekwanhyeong/workspace/paper_research/simple_lab_test/search/scripts/run_titantpp_v4_cuda_insta_smoke_0716.sh"'
```

## 결과
