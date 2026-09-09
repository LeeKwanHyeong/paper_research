# VNC-Hard-LMM 5090 검증

최종 상태: **실행 계약 통과, 성능 후보 탈락**. 커밋 `956603f16ca3540e2012a461a494ea1ec905102d`의 VNC-Hard-LMM 후보는 RTX 5090에서 CUDA·gradient·비용·세 데이터셋 full-data e1·저장/복원 계약을 충족했다. 후속 Instacart seed42 screening에서는 raw RMSE 기준을 충족하지 못해 확장을 중단했다.

e1 validation 수치로 성능 채택을 판단하지 않았다. 성능 판정은 후속 seed42 screening에서 수행했으며 held-out test는 실행하지 않았다.

## Seed42 성능 판정

Instacart 학습은 epoch 112에서 조기 종료됐고 epoch 72가 선택됐다. VNC의 validation raw RMSE는 `5.8743577925`, B는 `5.8722169305`로 VNC가 약 `0.0365%` 높았다. 다른 네 guardrail은 통과했지만 raw RMSE를 엄격히 개선해야 한다는 사전 기준을 충족하지 못했다.

따라서 VNC용 normalized-duration fit, Taxi·Intermittent seed42 및 추가 seed는 실행하지 않는다. 상세 결과와 CUDA 재감사는 [seed42 screening 기록](seed42_screening_5090/README.md)에 있다.

## Source·데이터 무결성

- Git archive SHA-256: `effaa12f18b7e6ef126218c8b2e6cbeee91a8a1c2159df677955ccc8e6ee404d`
- archive embedded revision: `956603f16ca3540e2012a461a494ea1ec905102d`
- 추출 파일 manifest: `177286e0e405e21885214ee0c6acb0fada158465a1715743f399facfd5a9af9f` (`2,130`개 파일 모두 일치)
- Intermittent·Taxi·Instacart parquet와 split manifest 6개의 SHA-256이 [후보 계약](../../contracts/hard_lmm_value_norm_consistent_v1.json)과 모두 일치했다.
- Runtime: NVIDIA GeForce RTX 5090, Python 3.12.13, PyTorch 2.11.0+cu130, CUDA 13.0, Polars 1.39.3.

## CUDA·gradient·비용 계약

CUDA qualification은 `71 passed`, failure/error/skip `0`이었다. VNC 전용 CUDA case에서 초기 B 출력 동일성, finite gradient, 극단값 안정성, checkpoint·optimizer roundtrip을 확인했다. 비용 profile의 모든 candidate worker에서 `alpha_raw`의 유한·비영 gradient와 학습 후 출력 변화도 확인했다.

| 이력 길이 | B 대비 median step | B 대비 peak allocation | 계약 (`≤1.25×`, `≤1.10×`) |
|---:|---:|---:|---|
| 8 | 1.103× | 1.006× | 통과 |
| 64 | 1.131× | 1.008× | 통과 |
| 256 | 1.137× | 1.005× | 통과 |

## Full-data e1

| 데이터셋 | train target | validation target | 실행 시간 | CUDA peak | 학습된 `alpha_raw` | 상태 |
|---|---:|---:|---:|---:|---:|---|
| Instacart | 1,991,192 | 503,733 | 167.7s | 358.7 MiB | +0.127371 | 통과 |
| Taxi | 38,393 | 8,268 | 10.3s | 1810.2 MiB | +0.014946 | 통과 |
| Intermittent | 393,824 | 86,285 | 87.6s | 1810.2 MiB | -0.004615 | 통과 |

세 job 모두 다음 조건을 충족했다.

- full train target을 정확히 한 번 처리했고 validation identity·quantity SHA가 계약값과 일치했다.
- `train_all_finite=true`, pre-clip gradient norm이 양수였고 `alpha_raw`의 AdamW state가 비영이었다.
- raw-RMSE selector가 epoch 1을 선택했고 candidate route metadata가 일치했다.
- best/last model state를 `strict=True`로 복원했고 canonical state digest가 일치했다.
- optimizer state를 실제 load했다. 저장된 Python·NumPy·Torch·CUDA RNG를 두 번 복원했을 때 다음 난수열이 정확히 같았고, train-loader generator를 두 번 복원했을 때 다음 표본 순서가 정확히 같았다.
- `evaluation_scope=validation_only`, `held_out_test_evaluated=false`였다.

## 감사 기록

커밋 `956603f`에는 VNC 전용 GPU 비용·통합 감사기가 없어서, source snapshot과 분리한 감사 도구를 SHA-256으로 고정해 artifact에 보존했다. 비용 감사 v1은 첫 warmup 전 인자 순서 오류로 중단됐고 v2가 새 경로에서 통과했다. e1 감사 v1/v2는 checkpoint 검사 전 archive stream 처리 때문에 중단됐고 v3는 model·optimizer 복원까지 통과했다. 독립 검토에서 v3의 RNG·loader 검사가 payload 존재 확인에 그친 점을 발견해, 실제 복원과 결정적 다음 상태를 확인한 v4를 최종 판정으로 고정했다. 학습 job은 재실행하지 않았다.

e1의 machine-readable 판정은 [completion.json](completion.json)과 [remote/e1_audit_v4.json](remote/e1_audit_v4.json)에 있다. 이 두 파일은 seed42 screening 전의 e1 완료 시점 기록이다. 성능 판정은 [seed42_screening_5090/decision.json](seed42_screening_5090/decision.json)을 따른다. 대용량 `.pt` checkpoint는 5090 원격 artifact에 보존하고 로컬 Git 증적에서는 제외했다.

## 남은 작업

이 VNC 후보의 조건부 실험은 종료됐다. 다음 Backbone 후보를 정한다면 이번 결과에서 확인된 “평균 절대오차와 tail은 보존되지만 제곱오차가 소폭 증가하는 현상”을 새 가설의 근거로 사용하되, VNC의 탈락 기준이나 결과를 소급 변경하지 않는다.
