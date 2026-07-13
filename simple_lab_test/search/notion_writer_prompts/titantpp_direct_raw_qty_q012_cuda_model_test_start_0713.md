# TitanTPP Direct Raw Quantity Q0/Q1/Q2 CUDA Model-Test 시작

`5. Model Design Enhancement` 아래에 세부 페이지
`TitanTPP Direct Raw Quantity Q0 Q1 Q2 CUDA Model Test`를 생성하거나 같은 제목의
기존 페이지를 갱신한다. 상위 history에는 제목 3
`Step 9. 5090 CUDA Q0/Q1/Q2 Model-Test`로 연결한다.

## 실험 시작 정보

- 시작 기록 시각: `2026-07-13 16:26 KST`
- 실행 서버: `5090` (`192.168.0.71`, user `leekwanhyeong`)
- 프로젝트 경로: `/home/leekwanhyeong/workspace/paper_research`
- Conda 환경: `ai_env`
- tmux session: `titantpp_raw_q012_cuda_0713`
- artifact root: `search_artifacts/model_enhancement_direct_raw_qty_q012_cuda_model_test_0713`
- 상태: `prepared; 5090 preflight and launch pending`

## 실험 목적

동일 synthetic batch, seed, Titan candidate, hidden size, quantity head와 loss 설정에서
Q0 `global`, Q1 `causal_revin`, Q2 `causal_shrinkage_revin`만 바꿔 CUDA runtime
contract를 검증한다. 이 단계는 validation accuracy나 RevIN benefit을 판정하는 성능
실험이 아니다.

## 고정 조건

| 항목 | 값 |
|---|---:|
| model / candidate | `TitanTPP / small_lmm` |
| device | `cuda` |
| seed | `42` |
| batch / sequence | `4 / 16` |
| marks / hidden dim | `12 / 64` |
| scale base | `2` |
| decoder | `direct_raw_qty` |
| magnitude input embedding | `8` |
| lambda magnitude | `1.0` |
| RevIN epsilon | `1e-5` |
| shrinkage k | `8` |
| center / affine / stat context | `mean / false / none` |
| padding stress | `left_pad=true` |

Q0/Q1/Q2는 각각 별도 artifact 디렉터리에 기록하며 한 변형 실패가 나머지 변형의
실행을 취소하지 않도록 한다.

## 실험 계획

1. 5090의 GPU, Python, CUDA 13 runtime library와 원격 source 반영 상태를 확인한다.
2. 동일 조건에서 Q0, Q1, Q2 model-test를 순차 실행한다.
3. 각 variant의 `model_test_summary.json/csv`와 통합 `run.log`를 확인한다.
4. 세 variant 모두 success, finite forward/loss, 동일 parameter count인지 검증한다.
5. 결과를 이 페이지와 상위 Enhancement history에 반영한다.

## 실행 명령어

```bash
ssh 5090
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ai_env
/opt/miniconda3/envs/ai_env/bin/tmux new-session -d \
  -s titantpp_raw_q012_cuda_0713 \
  "bash /home/leekwanhyeong/workspace/paper_research/simple_lab_test/search/scripts/run_direct_raw_qty_q012_cuda_model_test_0713.sh"
```

## 사전 Acceptance Gate

- 세 variant가 모두 `status=success`여야 한다.
- hidden state, total/marker/time/magnitude loss와 예측 tensor가 모두 finite여야 한다.
- Q0/Q1/Q2 parameter count가 정확히 같아야 한다.
- output이 실제 `device=cuda`로 기록되어야 한다.
- 실패 시 actual-data smoke로 이동하지 않고 원인을 먼저 수정한다.

## 해석 제한

synthetic model-test loss의 절대값과 variant 간 차이는 normalization 좌표계가 다르므로
성능 우열로 해석하지 않는다. RevIN 효과는 이후 fixed-split actual-data validation-only
screening에서 Q0와 V2를 함께 비교해야 판단할 수 있다.
