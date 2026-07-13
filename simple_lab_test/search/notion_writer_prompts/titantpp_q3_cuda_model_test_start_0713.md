# TitanTPP Q3 Factorial CUDA Model-Test 시작

`5. Model Design Enhancement` 아래에 세부 페이지
`TitanTPP Q3 Factorial CUDA Model Test`를 생성하거나 같은 제목의 기존 페이지를
갱신한다. 상위 history에는 제목 3
`Step 13. Q3 Factorial 5090 CUDA Model-Test`로 연결한다.

## 실험 시작 정보

- 시작 기록 시각: `2026-07-13 22:53 KST`
- 실행 서버: `5090` (`192.168.0.71`, user `leekwanhyeong`)
- 프로젝트 경로: `/home/leekwanhyeong/workspace/paper_research`
- Conda 환경: `ai_env`
- tmux session: `titantpp_q3_cuda_0713`
- artifact root: `search_artifacts/model_enhancement_titantpp_q3_cuda_model_test_0713`
- 구현 기준 commit: `14c2978`
- 실행 준비 commit: `f4cc2235e16ae75433bdf6be1767a38b328cbaec`
- 실제 실행 시작 시각: `2026-07-13 23:04:19 KST`
- 상태: `in progress; Q3c entry confirmed, continuous monitoring stopped`

## 실험 목적

동일 synthetic batch와 Q2 causal shrinkage RevIN 조건에서 Q2/Q3a/Q3b/Q3c의
CUDA forward, loss, artifact identity를 검증한다. Q3가 바꾸는 축은 magnitude
encoder gradient와 log2 auxiliary뿐이며, 이 단계에서는 실제 데이터 성능이나 Q3
우위를 판정하지 않는다.

## Variant 계약

| Variant | magnitude encoder gradient | log2 auxiliary | 역할 |
|---|---|---|---|
| Q2 | `coupled` | `none` | fresh factorial control |
| Q3a | `detached` | `none` | gradient isolation |
| Q3b | `coupled` | `log_huber` | low-quantity auxiliary |
| Q3c | `detached` | `log_huber` | combined interaction |

## 고정 조건

| 항목 | 값 |
|---|---:|
| model / candidate | `TitanTPP / small_lmm` |
| device / seed | `cuda / 42` |
| batch / sequence | `4 / 16` |
| marks / hidden dim | `12 / 64` |
| scale base | `2` |
| decoder / normalization | `direct_raw_qty / causal_shrinkage_revin` |
| magnitude input embedding | `8` |
| `lambda_magnitude / lambda_qty` | `1.0 / 0.25` |
| shrinkage k / sigma floor | `8 / 0.0550124034288891` |
| `lambda_log_qty` / Huber delta / log floor | `0.25 / 1.0 / 1.0` |
| center / affine / stat context | `mean / false / none` |
| padding stress | `left_pad=true` |
| expected parameter count | `78,111` |

## 실험 계획

1. 준비 commit을 5090에 동기화하고 remote revision을 확인한다.
2. RTX 5090, CUDA library, `ai_env` PyTorch CUDA availability를 preflight한다.
3. Q2, Q3a, Q3b, Q3c model-test를 같은 seed와 batch로 순차 실행한다.
4. variant별 `model_test_summary.json/csv`, 통합 `run.log`, status TSV를 생성한다.
5. 네 variant의 parameter/forward identity와 paired scalar-loss 계약을 확인한다.
6. gate를 통과한 뒤에만 Instacart top-20 e1 actual-data smoke를 준비한다.

## 실행 명령어

```bash
ssh 5090
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ai_env
/opt/miniconda3/envs/ai_env/bin/tmux new-session -d \
  -s titantpp_q3_cuda_0713 \
  "env PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
  PYTHON_BIN=/opt/miniconda3/envs/ai_env/bin/python \
  bash /home/leekwanhyeong/workspace/paper_research/simple_lab_test/search/scripts/run_titantpp_q3_cuda_model_test_0713.sh"
```

## 실행 시작 확인

- `f4cc223`까지의 변경 파일 13개를 rsync한 뒤 checksum dry-run에서 차이가
  없음을 확인했다.
- Remote working copy에는 `.git` metadata가 없으므로
  `source_sync_manifest.json`에 full revision과 검증 시각을 별도로 기록했다.
- Preflight: RTX 5090 idle `41 MiB`, PyTorch `2.11.0+cu130`, CUDA `13.0`, 실제
  CUDA tensor allocation 통과.
- 초기 로그: Q2, Q3a, Q3b는 각각 `exit_code=0`; Q3c model-test 진입과 GPU
  process를 확인했다.
- `/tmp/xdg_cache_paper_research/torch/kernels` 생성 실패 경고는 kernel cache만
  비활성화하며, 해당 경고 이후에도 앞선 variant 계산은 정상 진행됐다.
- 사용자 요청 전까지 추가 polling이나 결과 판정은 수행하지 않는다.

## 사전 Acceptance Gate

- 네 variant가 모두 `status=success`, `device=cuda`여야 한다.
- hidden state, NLL, magnitude/raw/log auxiliary/total loss와 예측이 finite여야 한다.
- parameter count는 모두 `78,111`, hidden shape은 모두 `[4,16,64]`여야 한다.
- Q2/Q3a의 NLL, magnitude loss, raw quantity loss, total loss가 같아야 한다.
- Q3b/Q3c의 NLL, magnitude loss, raw quantity loss, log auxiliary, total loss가
  같아야 한다.
- Q2/Q3a `log_qty_aux_loss=0`; Q3b/Q3c는 같은 finite positive 값이어야 한다.
- 실패 시 Instacart smoke로 이동하지 않고 source/runtime drift를 먼저 확인한다.

## 해석 제한

Synthetic model-test는 CUDA integration gate다. Q3의 marker recovery, 저수량 개선,
raw MAE 개선은 판단하지 않는다. 성능 판단은 Instacart integration 이후 Intermittent
seed-42 validation-only screening에서만 수행하며 held-out test는 계속 잠근다.

## Notion 반영 기록

- 최초 반영 시각: `2026-07-13 22:59 KST`
- 실행 상태 반영 시각: `2026-07-13 23:05 KST`
- 세부 페이지: [TitanTPP Q3 Factorial CUDA Model Test](https://app.notion.com/p/39cbbe40561381e591b0d021d028c4bd)
- 상위 위치: `5. Model Design Enhancement > 2026-07-13 > Step 13`
- 검증: 상위 page block, Q3 계약 상태, Model Enhancement Strategy의 다음 작업을
  다시 fetch해 반영 내용을 확인했다.
