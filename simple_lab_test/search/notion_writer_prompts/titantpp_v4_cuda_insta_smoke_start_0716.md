# TitanTPP V4 CUDA Model-Test And Instacart Smoke

Notion의 `5. Model Design Enhancement` 아래 제목 2
`2026-07-16 | V4 Mark-Conditioned Time Head`, 제목 3
`Step 3. 5090 CUDA Model-Test And Instacart e1 Smoke`로 정리한다.

## 상태

- 상태: `완료 - corrected validator 재검증 PASS`
- 실행 서버 / tmux: `5090 / titantpp_v4_cuda_insta_smoke_0716`
- 실행 시작 시각: `2026-07-16 18:32:31 KST`
- 실행 종료 시각: `2026-07-16 18:32:44 KST`
- 학습 source revision: `51cbad647c27118c18576338aa3ce536ddfbddf0`
- 재검증 source revision: `c5e9cca4241a5579ba0af655c884d6692484ba5a`

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

- V4a/V4b CUDA model-test와 Instacart e1 smoke 네 단계가 모두 exit code `0`으로 종료됐다.
- CUDA model-test NLL은 두 variant 모두 `3.760483`이었다. 파라미터 수는 V4a
  `328,236`, V4b `328,752`였다.
- Instacart validation NLL은 V4a `3.328981`, V4b `3.324357`이었다. Quantity
  MAE는 각각 `5.617047`, `5.448310`이었다.
- loader sample은 train/validation/test `1,380/300/300`으로 계약과 일치했고,
  `held_out_test_evaluated=false`이며 test metric artifact는 생성되지 않았다.
- 최초 validator는 실제 `manifest/run_config.json`을 `manifests/run_config.json`으로
  조회해 실패 표시를 남겼다. 학습 재실행 없이 경로만 수정해 재검증했고 `PASS`했다.
- 이 결과는 통합 동작 확인용이며 V4 성능 승격 근거로 사용하지 않는다. 다음 단계인
  Taxi validation-only 2x2 screening 진행 조건만 충족했다.
