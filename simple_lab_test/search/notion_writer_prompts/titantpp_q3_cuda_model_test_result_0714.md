# TitanTPP Q3 Factorial CUDA Model-Test 결과

기존 Notion 세부 페이지를 갱신한다.

- 상위 history: `https://app.notion.com/p/2e8bbe40561380a88b5eef94e834892e`
- 결과 페이지: `https://app.notion.com/p/39cbbe40561381e591b0d021d028c4bd`
- 상위 Step: `Step 13. Q3 Factorial 5090 CUDA Model-Test`
- 새 페이지를 만들지 않는다.

## 실행 정보

- 시작/종료: `2026-07-13 23:04:19 / 23:04:26 KST`
- 실행 서버: `5090`, host `RTX5090-server`
- GPU / runtime: `NVIDIA GeForce RTX 5090`, PyTorch `2.11.0+cu130`, CUDA `13.0`
- tmux session: `titantpp_q3_cuda_0713`
- source revision: `f4cc2235e16ae75433bdf6be1767a38b328cbaec`
- artifact: `search_artifacts/model_enhancement_titantpp_q3_cuda_model_test_0713`
- 통합 상태: `MODEL_TEST_SUCCESS`, aggregate exit code `0`
- 분석 시각: `2026-07-14 08:00 KST`

Remote working copy에는 `.git` metadata가 없어 `experiment_manifest.json`의
`code_revision`은 `unknown`이다. 대신 실행 전에 rsync checksum을 확인하고
`source_sync_manifest.json`에 full revision을 기록했으므로 source trace는 유지된다.

## Artifact 확인 순서

1. `experiment_manifest.json`, `source_sync_manifest.json`: server, device, seed,
   model, candidate, factorial variant, source revision 확인
2. `logs/run.log`: 네 variant 순차 실행, CUDA runtime, warning, 종료 상태 확인
3. `variant_status.tsv`: 네 고유 variant와 exit code 확인
4. variant별 `model_test_summary.json/csv`: schema, config, loss, prediction, shape,
   parameter identity 확인
5. model-test이므로 test summary, histories, scale-wise metrics, plots는 생성 대상 아님

## Manifest와 Log

- seed `42`, batch `4`, sequence `16`, marks `12`, candidate `small_lmm`
- decoder `direct_raw_qty`, normalization `causal_shrinkage_revin`, `k=8`
- `lambda_magnitude=1.0`, `lambda_qty=0.25`, `lambda_log_qty=0.25`
- Q2/Q3a/Q3b/Q3c factorial mapping이 사전 계약과 일치
- 네 variant와 aggregate exit code가 모두 `0`
- NaN, Inf, Traceback, runtime error 없음
- `/tmp/xdg_cache_paper_research/torch/kernels` 생성 실패로 kernel cache가
  비활성화됐지만 계산과 artifact 생성에는 영향 없음

## Variant 결과

| Variant | Gradient | Log aux | NLL | Magnitude loss | Qty loss | Log aux loss | Total loss | Qty hat mean | Params |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Q2 | `coupled` | `none` | 3.903185844 | 0.127688080 | 192.696228027 | 0.000000000 | 52.204929352 | 205.008331299 | 78,111 |
| Q3a | `detached` | `none` | 3.903185844 | 0.127688080 | 192.696228027 | 0.000000000 | 52.204929352 | 205.008331299 | 78,111 |
| Q3b | `coupled` | `log_huber` | 3.903185844 | 0.127688080 | 192.696228027 | 3.797908068 | 53.154407501 | 205.008331299 | 78,111 |
| Q3c | `detached` | `log_huber` | 3.903185844 | 0.127688080 | 192.696228027 | 3.797908068 | 53.154407501 | 205.008331299 | 78,111 |

공통 값:

- marker NLL `2.498844624`, time NLL `1.404341221`
- hidden shape `[4,16,64]`, target-only steps `4`
- dt hat mean `0.643254161`
- marker mode `ce`, `lambda_ordinal=0`, value loss `0`
- 48개 검증 scalar가 모두 finite

## Identity 검증

- 네 summary의 schema와 encoder config는 동일하다.
- 허용된 gradient/aux factor와 output path를 제외한 CLI/RMTPP config는 동일하다.
- NLL, marker/time split, magnitude/raw quantity loss, quantity/time prediction은 네
  variant에서 exact match다.
- Q2/Q3a row는 gradient mode를 제외하면 exact match다.
- Q3b/Q3c row는 gradient mode를 제외하면 exact match다.
- Q2/Q3a `log_qty_aux_loss=0`이다.
- Q3b/Q3c는 같은 positive log auxiliary를 기록했다.
- Total-loss 수식 재계산의 최대 절대 오차는 `1.58e-6`으로 FP32 누적 오차 범위다.
- Q3b-Q2 total 증가량과 `0.25 * log_qty_aux_loss`의 차이는 `1.13e-6`이다.
- 네 CSV의 scalar/status/device/parameter 값은 대응 JSON과 일치한다.

## Synthetic Scope 제한

Model-test는 synthetic batch에서 train statistics를 다시 계산한다.

- synthetic global mean: `244.2063446044922`
- synthetic global std: `409.29852294921875`
- CLI sigma floor: `0.0550124034288891`
- effective synthetic sigma floor: `0.40929852294921876`

따라서 이번 결과는 Intermittent train-only raw moments 자체를 검증하지 않는다.
Actual-data fixed-split 경로의 constants, backward, checkpoint, history, scale-wise
artifact는 Instacart e1 smoke에서 확인한다.

CUDA artifact에는 full hidden/prediction tensor hash가 저장되지 않아 tensor-level
bitwise equality를 artifact만으로 다시 계산할 수는 없다. 이 항목은 로컬 focused
contract test `19/19`의 exact tensor/state 검증과 CUDA summary의 exact paired scalar
identity를 함께 근거로 사용한다.

## 판정

5090 CUDA model-test gate는 **통과**다.

네 factorial variant가 같은 parameter budget과 공통 config에서 finite CUDA
forward/loss/artifact를 생성했고, paired gradient routing과 log auxiliary 계약이
사전 정의와 일치했다. 따라서 Q2/Q3a/Q3b/Q3c를 동일 Instacart top-20 fixed split의
e1 actual-data integration smoke로 이동할 수 있다.

이 판정은 runtime과 model contract에 한정한다. Q3의 mark recovery, low-quantity
개선, raw MAE 또는 모델 우위는 판단하지 않는다.

## 다음 작업

1. Q2/Q3a/Q3b/Q3c Instacart top-20 e1 fixed-split smoke 준비 및 실행
2. actual-data backward, train-only raw statistics, checkpoint/resume, history,
   summary, scale-wise artifact 확인
3. integration gate 통과 후 Intermittent seed-42 e50 validation-only screening
4. seed-42 full gate 통과 후보만 strict multi-seed로 승격하고 held-out test는 계속 잠금
