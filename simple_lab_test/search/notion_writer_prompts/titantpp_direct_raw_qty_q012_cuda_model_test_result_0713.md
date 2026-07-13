# TitanTPP Direct Raw Quantity Q0/Q1/Q2 CUDA Model-Test 결과

기존 Notion 세부 페이지를 갱신한다.

- 상위 history: `https://app.notion.com/p/2e8bbe40561380a88b5eef94e834892e`
- 결과 페이지: `https://app.notion.com/p/39cbbe4056138192ab4bf374455775fb`
- 상위 Step: `Step 9. 5090 CUDA Q0/Q1/Q2 Model-Test`

새 페이지를 만들지 않는다. 결과 페이지의 시작 기록 아래에 완료 결과를 추가하고,
상위 history의 Step 9와 `현재 의사결정`을 완료 상태로 갱신한다.

## 실행 정보

- 실행일: `2026-07-13 KST`
- 시작/종료: `16:30:26 / 16:30:31 KST`
- 실행 서버: `5090`, host `RTX5090-server`, GPU `NVIDIA GeForce RTX 5090`
- Python: `/opt/miniconda3/envs/ai_env/bin/python`
- tmux session: `titantpp_raw_q012_cuda_0713`
- artifact: `search_artifacts/model_enhancement_direct_raw_qty_q012_cuda_model_test_0713`
- 통합 상태: `MODEL_TEST_SUCCESS`, exit code `0`

## Artifact 확인 순서

1. `experiment_manifest.json`: server/device/seed/batch/sequence/variant identity 확인
2. `logs/run.log`: Q0/Q1/Q2 순차 실행, 각 exit code 0, traceback/NaN/Inf 없음
3. variant별 `model_test_summary.json/csv`: status, device, loss, shape, parameter 확인
4. `variant_status.tsv`: 세 variant exit code 0 확인
5. model-test이므로 `test_summary`, histories, scale-wise metrics, plots는 생성 대상 아님

## 결과

| Variant | Norm | Status | NLL | Marker NLL | Time NLL | Magnitude loss | Qty hat mean | Params |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Q0 | `global` | success | 3.900679 | 2.497418 | 1.403261 | 0.126255 | 213.319992 | 78,111 |
| Q1 | `causal_revin` | success | 3.909228 | 2.501662 | 1.407567 | 0.143572 | 196.889984 | 78,111 |
| Q2 | `causal_shrinkage_revin` | success | 3.903186 | 2.498845 | 1.404341 | 0.127688 | 205.008331 | 78,111 |

- 세 variant 모두 `device=cuda`, decoder `direct_raw_qty`
- hidden shape는 모두 `[4,16,64]`, target-only step은 모두 `4`
- 검증 대상 loss와 prediction tensor는 모두 finite
- parameter count는 모두 `78,111`로 동일
- normalization mode를 제외한 RMTPP config와 encoder config는 동일
- seed `42`, batch `4`, sequence `16`, `left_pad=true`, epsilon `1e-5`, `k=8` 동일

`nll`은 기존 marker/time likelihood identity이며 magnitude loss는 별도로 기록된다.
Synthetic loss 수치 차이는 성능 순위로 사용하지 않는다.

## 경고와 범위 제한

실행 중 `/tmp/xdg_cache_paper_research/torch/kernels`를 만들 수 없어 CUDA kernel
cache가 비활성화됐다는 warning이 있었지만 세 실행 모두 정상 종료했다. Runtime error,
traceback, NaN, Inf는 없었다.

이 model-test는 synthetic history에서 raw global moments를 다시 계산한다.

- synthetic global mean: `244.2063446044922`
- synthetic global std: `409.29852294921875`
- effective sigma floor: `0.40929852294921876`

따라서 이번 gate는 Intermittent train-only 고정 mean/std/floor 자체를 검증한 것이 아니다.
그 값의 data-path 반영은 다음 actual-data fixed-split smoke에서 확인한다. 이번 결과만으로
Q0/Q1/Q2 정확도나 RevIN benefit을 판단하지 않는다.

## 판정

5090 CUDA runtime gate는 통과했다. Q0/Q1/Q2 모두 동일 parameter budget에서 finite하게
forward/loss/artifact를 생성했으므로 actual-data integration smoke로 이동할 수 있다.

## 다음 작업

1. Q0/Q1/Q2 Instacart top-20 e1 fixed-split smoke 실행
2. train-only raw global moments, checkpoint/history/summary/scale-wise artifact와 backward 확인
3. smoke 통과 후 Intermittent seed-42 e50 validation-only screening 실행
4. V2와 Q0를 함께 사용해 Q1/Q2 RevIN benefit gate 판정
