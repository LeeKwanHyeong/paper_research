# Count-aware TPP 공식 기준선 계약 v1

- 동결일: 2026-08-20
- 적용 범위: mark-free event-time 및 continuous quantity 연구 트랙
- Held-out test: 사용하지 않음

## 역할 구분

| 역할 | 고정 구성 | 사용 목적 |
| --- | --- | --- |
| `T0 common control` | 공통 `legacy_clamped_rmtpp` time head + direct `log1p(quantity)` MSE | RMTPP·THP·NHP·SAHP 및 TitanTPP의 공정한 backbone/손실 비교 |
| `T1 incumbent` | TitanTPP Hard-LMM + T1 tail-shared quantity loss + T1 3-seed 당시 time head | 현재 TitanTPP 대표 모델과 향후 backbone 강화의 fresh control |
| `H0/H3 diagnostic` | H0 scaled exact 또는 H3 log-normal duration | time-head 실패 원인 분석만 수행하며 최종 모델 비교표에서 제외 |

과거 mark-residual V2와 Taxi V3b는 이전 formulation의 이력으로 보존한다. 현재
count-aware 논문 트랙의 공식 기준선이나 incumbent로 재사용하지 않는다.

## T0 공통 비교 계약

- 적용 backbone: RMTPP, THP, NHP, SAHP, TitanTPP-T0
- quantity variant: `count_only_log_regression`
- quantity loss: `MSE(predicted_log1p_qty, target_log1p_qty)`
- point prediction: `expm1(predicted_log1p_qty)`
- time head: `legacy_clamped_rmtpp`
- `lambda_tail=0`

외부 backbone 비교에서는 모든 모델에 이 계약을 동일하게 적용한다. T1 loss를
RMTPP·THP·NHP·SAHP에 붙인 결과는 최종 모델 순위가 아니라 loss attribution을 위한
별도 ablation으로만 다룬다.

## T1 Incumbent 계약

- backbone: `titantpp`
- encoder candidate: `count_titan_small_lmm`
- memory: Hard-LMM, memory size 64, top-k 4, persistent memory 16
- quantity variant: `count_only_log_mse_tail_shared`
- base loss: direct `log1p(quantity)` MSE
- auxiliary loss: train-only tail normalized raw-quantity Huber
- tail gradient: quantity head와 Titan encoder에 전달
- `lambda_tail=0.09111380335463036`
- threshold/normalization/cap/Huber delta: `46/46/187/1`
- time head: `legacy_clamped_rmtpp`

T1은 Intermittent validation 3-seed에서 T0보다 quantity MAE `6.43%`, RMSE
`6.24%`를 개선하고 Time NLL을 유지했다. 따라서 현재 TitanTPP 대표 모델로 사용한다.
다만 THP보다 MAE가 낮다는 결론은 확인되지 않았으므로 전체 backbone 우월성을
의미하지 않는다.

## 공통 학습 조건

| 항목 | 값 |
| --- | --- |
| seeds | 42, 52, 62 |
| maximum/minimum epochs | 300 / 40 |
| early-stopping patience | 40 |
| batch size | 128 |
| learning rate | 0.001 |
| lookback | 520 weeks |
| max sequence length | 256 |
| hidden dimension | 64 |
| gradient clipping | 1.0 |
| checkpoint | minimum validation joint objective |
| evaluation | validation-only |

새 Titan backbone 후보는 위 조건에서 `T1 incumbent + candidate`를 fresh matched로
같이 실행한다. 기존 T1 수치를 새 후보의 exact control로 재사용하지 않는다.

## 실행 역할

공식 실행은 runner의 `--model-role`을 명시한다.

```bash
# RMTPP/THP/NHP/SAHP 공통 T0 비교
--model-role t0_common_control \
--backbones rmtpp,thp,nhp,sahp \
--quantity-variants log_mse \
--time-head-mode legacy_clamped_rmtpp

# 현재 T1 incumbent 재현
--model-role t1_incumbent \
--backbones titantpp \
--quantity-variants tail_shared \
--lambda-tail 0.09111380335463036 \
--time-head-mode legacy_clamped_rmtpp

# 향후 Titan backbone fresh matched 비교
--model-role t1_backbone_comparison \
--backbones titantpp,<candidate> \
--quantity-variants tail_shared \
--lambda-tail 0.09111380335463036 \
--time-head-mode legacy_clamped_rmtpp
```

`H0/H3`는 `--model-role time_head_diagnostic`으로만 실행한다. 이 결과는 time-head
진단 표에는 포함할 수 있지만 최종 RMTPP·THP·NHP·SAHP·TitanTPP 비교표에는 넣지 않는다.
