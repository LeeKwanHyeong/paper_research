# TitanTPP Count-Aware Log-Normal K=1 Screening

## 작성 위치

- `5. Model Design Enhancement > Enhancement & Validation History`
- 날짜 구역: `2026-08-15 | Mark-Free Quantity Distribution Enhancement`
- 세부 Step: `Step 1. Log-Normal K=1 Quantity Head Screening`
- 동일 제목의 상세 페이지가 있으면 새로 만들지 않고 업데이트한다.

## 상태

- 상태: 실험 완료 · K=1 gate 실패 · log-MSE 기준선 유지
- 실험 시작 시각: `2026-08-15 11:40:56 KST`
- 실험 종료 시각: `2026-08-16 02:17:39 KST`
- 실행 서버 / tmux: `5080 / count_lognormal_k1_e300_0815`

## 목적

- Mark-free direct log-MSE에서 확인된 TitanTPP의 p99 초과 수량 강점을 유지하면서 p99 이하 일반 수량 구간의 MAE를 개선한다.
- 동일한 THP와 TitanTPP backbone에서 fresh log-MSE control과 K=1 log-normal head를 비교해 quantity distribution head의 효과를 분리한다.

## Variant 계약

| Variant | Quantity head | 학습 loss | 역할 |
| --- | --- | --- | --- |
| `log_mse` | 단일 positive log-count | `log1p(q)` MSE | fresh matched control |
| `lognormal_k1` | positive `mu`와 `sigma` | Gaussian NLL + location Huber | 일반 구간 calibration 후보 |

두 variant는 quantity head와 quantity loss만 다르며 data, backbone, time head, optimizer, seed, epoch, checkpoint 규칙은 동일하다. K=1의 point prediction은 median 하나로 고정하고 MAE와 RMSE에 공통 사용한다.

## 고정 조건

- dataset: `intermittent_frozen_5000`
- model: Count-aware THP, Count-aware TitanTPP
- epochs / seeds: `300 / 42`
- lr / batch size: `1e-3 / 128`
- lookback / max sequence length: `520 / 256`
- split mode: fixed validation-only
- 주요 model/loss 옵션: mark-free `log1p(dt), log1p(q)` 입력, shared RMTPP time head, `sigma_floor=1e-3`, Huber delta `0.25`
- artifact: `search_artifacts/count_aware_lognormal_k1_screening_e300_20260815`

## 실행 명령어

```bash
SOURCE_REVISION=a70d6af517ac8be6a3631679fe62393544834da5 \
PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
PYTHON_BIN=/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python \
OUTPUT_ROOT=/home/leekwanhyeong/workspace/paper_research/search_artifacts/count_aware_lognormal_k1_screening_e300_20260815 \
MODELS=thp,titantpp \
VARIANTS=log_mse,lognormal_k1 \
SEEDS=42 \
EXECUTION_ROLE=seed42_validation_screening_5080 \
bash simple_lab_test/search/scripts/run_count_aware_lognormal_k1_screening_e300_20260815.sh
```

## 결과

- 네 run 모두 정상 종료했으며 validation `86,285`건, held-out test 미사용 조건이 일치했다. NaN, Traceback, OOM은 없었다.
- TitanTPP K=1은 log-MSE control 대비 전체 MAE를 `1.8637%` 개선해 최소 `5%` 기준에 미달했다.
- 전체 RMSE는 `33.2453%`, `q > p99` MAE는 `75.0980%`, Time NLL은 `3.944258` 악화되어 세 safety gate를 모두 실패했다.
- TitanTPP K=1은 `q <= p95` MAE를 개선했지만 `p95` 이상 tail과 long-history 오차를 손상했다.
- THP에서도 K=1은 MAE `12.0091%`, RMSE `52.1536%`, Time NLL `0.442550` 악화되어 Titan encoder에 국한된 문제가 아니었다.
- Negative Gaussian NLL이 joint objective를 지배하고 shared encoder의 time 학습을 손상했다. TitanTPP K=1은 전체 epoch에서 time safety를 한 번도 만족하지 못했다.
- 최종 판정은 K=1 미채택, multi-seed 및 held-out 중단, mark-free direct log-MSE 기준선 유지다.
