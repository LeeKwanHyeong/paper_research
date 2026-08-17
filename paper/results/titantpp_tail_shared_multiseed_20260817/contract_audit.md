# TitanTPP T1 Multi-seed Baseline Contract Audit

## 판정

- 상태: **PASS**
- 재사용 baseline: `paper/results/count_aware_tpp_backbone_control_20260812/source_5080`
- 비교 대상: RMTPP, THP seeds 42/52/62
- 범위: Intermittent validation only
- Held-out test: 미사용

## 계약 대조

| 항목 | 기존 RMTPP·THP artifact | TitanTPP T1 artifact | 판정 |
| --- | --- | --- | :---: |
| Dataset | intermittent frozen-5000 | intermittent frozen-5000 | 일치 |
| Data SHA-256 | `85d1fe3a...ffffc3f` | `85d1fe3a...ffffc3f` | 일치 |
| Split SHA-256 | `393158a5...64c1c04` | `393158a5...64c1c04` | 일치 |
| Split rows | 398,824 / 86,285 / 88,019 | 398,824 / 86,285 / 88,019 | 일치 |
| Seeds | 42, 52, 62 | 42 완료, 52·62 추가 예정 | 확장 가능 |
| Epoch / batch / LR | 300 / 128 / 1e-3 | 300 / 128 / 1e-3 | 일치 |
| Lookback / max sequence | 520 / 256 | 520 / 256 | 일치 |
| Hidden dimension | 64 | 64 | 일치 |
| Input / target | log1p time·quantity / log1p quantity | 동일 | 일치 |
| Point prediction | softplus location → expm1 | 동일 | 일치 |
| Selection | best validation joint objective | best validation joint objective | 일치 |
| Evaluation | validation only | validation only | 일치 |

## 실행 경로 재현성

기존 backbone-control artifact의 TitanTPP-T0 seed 42와 새 tail-aware artifact의
T0 seed 42가 다음 값에서 완전히 동일했다.

| Metric | 기존 artifact | 새 artifact |
| --- | ---: | ---: |
| Completed / best epoch | 240 / 200 | 240 / 200 |
| Validation joint | -3.585705690087701 | -3.585705690087701 |
| Time NLL | -3.592856040674848 | -3.592856040674848 |
| Log quantity MSE | 0.007150351058535544 | 0.007150351058535544 |
| Quantity MAE | 0.762085075221471 | 0.762085075221471 |
| Quantity RMSE | 1.7224032516352568 | 1.7224032516352568 |
| Parameter count | 89,795 | 89,795 |

기존 artifact는 이전 schema라 `point_prediction`과 `train_target_std`를 metadata에
명시하지 않았지만, 동일한 T0 결과가 재현되어 quantity forward와 selection 실행
경로가 같음을 확인했다.

## 결론

기존 RMTPP·THP 3-seed validation 결과를 최종 방법 비교에 재사용한다. TitanTPP-T1은
완료된 seed 42를 재사용하고 seeds 52·62만 동일 계약으로 추가 실행한다. 이 비교는
완성된 방법 간 validation 성능 비교이며 Titan backbone 단독 효과의 인과 판정으로는
사용하지 않는다.
