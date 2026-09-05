# TitanTPP v0.7 validation evidence freeze

**판정: 원본 mark-free Hard-LMM T0는 Taxi와 Instacart에 동일한 모델 계열로 적용된 3-seed validation 증적이 있다. 그러나 두 데이터셋에서 공통 우위를 보이지 않으므로, 이 결과는 조건부 관찰로만 사용한다. Held-out test는 사용하지 않았다.**

## 통합 결과

- 입력, 수량 head, loss, time head와 checkpoint 규칙을 고정하고 RMTPP, THP, 원본 Hard-LMM만 비교했다. 각 데이터셋은 seeds 42, 52, 62와 e300 상한을 사용한다.
- Taxi에서 Hard-LMM의 평균 RMSE는 `143.993141`로 RMTPP `144.814842`보다 `0.567%` 낮다. 하지만 paired seed 승수는 `2/3`이고, MAE는 `42.580526`로 RMTPP `40.289319`보다 악화했다. Clamped time loss와 log-MSE도 세 seed 모두 RMTPP보다 높다.
- Instacart에서 Hard-LMM의 평균 MAE/RMSE는 `4.045027` / `6.015904`이고 RMTPP는 `4.026994` / `5.990086`이다. Hard-LMM은 MAE에서 `0/3`, RMSE에서 `1/3` seed만 이겼다.
- Taxi의 `history >128`에서도 Hard-LMM RMSE 개선은 RMTPP 대비 2/3 seed이며 MAE는 1/3 seed다. Instacart validation target은 전부 `history <=64`이다. 따라서 **긴 이력 자체가 개선 원인이라는 주장은 이 결과로 지지되지 않는다.**
- 허용되는 서술은 “Taxi에서 평균 RMSE 이점이 관찰됐지만 seed-stable한 전 지표 우위는 아니며, 짧은 Instacart에서는 backbone 우위가 확립되지 않았다”이다.

상세 표는 [paper_tables.md](paper_tables.md), seed 원자료는 [validation_seed_metrics.csv](validation_seed_metrics.csv), paired 차이는 [paired_seed_deltas.csv](paired_seed_deltas.csv)에 있다. Quantity/history 구간 집계는 [quantity_strata_aggregate.csv](quantity_strata_aggregate.csv)와 [history_strata_aggregate.csv](history_strata_aggregate.csv), 같은 seed 안의 Hard-LMM 차이는 [quantity_strata_paired_seed_deltas.csv](quantity_strata_paired_seed_deltas.csv)와 [history_strata_paired_seed_deltas.csv](history_strata_paired_seed_deltas.csv)에 있다.

## 공통 최종 구조 판정

공통 최종 구조 판정은 **FREEZE_ORIGINAL_MARK_FREE_HARD_LMM_T0**이다. 원본 mark-free Hard-LMM T0는 Taxi와 Instacart 모두 seeds 42/52/62가 있다. Separate-key Hard-LMM은 두 데이터셋 모두 seed 42만 있고, 사전 고정한 교차 데이터셋 gate에서 Instacart가 실패해 추가 seed 확장도 거부됐다. 따라서 separate-key에는 두 데이터셋 공통 최종 구조로 사용할 3-seed 증적이 없으며, single-seed 진단으로만 남긴다. 기계 판독 가능한 판정은 [final_model_evidence_audit.json](final_model_evidence_audit.json)에 있다.

## 데이터셋 identity 재감사

v0.7의 3개 데이터셋 통합 identity 판정은 **FAIL_INTERMITTENT_POPULATION_MISMATCH**이다. Taxi와 Instacart는 `T1_dataset_statistics.csv`의 data/split SHA-256이 이번 T0 평가 계약과 일치하므로, 이 freeze의 통합표는 **Taxi와 Instacart만** 같은 artifact identity로 묶었다.

Intermittent는 일치하지 않는다. 기존 T1 행은 `intermittent` 23,387 sequences / 242,888 events이고 split event 수는 159,643 / 41,901 / 41,344이다. 반면 mark-free 비교 계약은 `intermittent_frozen_5000` 5,000 series / 573,128 rows이며 split row 수는 398,824 / 86,285 / 88,019이다. data SHA-256과 split-manifest SHA-256도 모두 다르다. 따라서 기존 T1 Intermittent 통계와 frozen-5000 결과를 같은 population으로 인용할 수 없다. 후속 v0.7 통합 단계에서는 frozen artifact에서 [새 T1 행](../../tables/T1_v0_7_dataset_statistics.md)을 다시 계산해 이 문제를 해소했다. 상세 탐지 결과는 [dataset_identity_audit.json](dataset_identity_audit.json)에 보존한다.

## Quantity-interface 재감사

기존 Taxi direct-log 대 categorical 결과의 v0.7 계약 판정은 **FAIL**이며 논문 자격은 `diagnostic_only`이다.

충족한 항목은 동일 Taxi split, RMTPP 계열, seeds 42/52/62, e300, batch와 optimizer 설정, train-only bin fitting 및 train median reconstruction, validation-only/held-out lock이다.

차단 항목은 다음과 같다.

1. Categorical 경로는 다시 만든 quantity mark만 입력하고, direct-log 경로는 기존 magnitude mark와 within-mark residual을 입력한다. 현재 mark-free 토큰 `[log1p(delta_t), log1p(raw_quantity)]`을 공유하지 않는다.
2. 저장된 categorical 후보는 raw-scale uniform bin과 raw-scale train-quantile bin이다. v0.7이 요구한 순수 log-transformed categorical binning이 없다.
3. 두 계약은 source revision이 다르다.
4. Selector 이름은 모두 `best_val_nll`이지만 category-dependent mark NLL의 의미가 variant별로 달라지고 direct-log quantity MSE는 선택에 포함되지 않는다. 현재 T0의 validation joint selector와도 다르다.
5. 재구성 수량의 raw MAE/RMSE는 있지만 supporting log1p-MSE가 없다.
6. 이 controlled interface 결과는 Taxi만 다룬다.

따라서 기존 결과는 direct log가 유망하다는 진단에는 사용할 수 있지만, mark-free formulation의 empirical superiority를 입증하는 v0.7 ablation으로 사용할 수 없다. 전체 판정과 정확한 누락은 [quantity_interface_audit.json](quantity_interface_audit.json)에 기록했다.

## 증적 경계

- 모든 집계는 저장된 validation summary와 validation stratum row에서 수행했다. Test row는 읽거나 평가하지 않았다.
- Builder는 각 run의 `held_out_test_evaluated=false`, `evaluation_scope=validation_only`, mark-free interface와 joint-objective 산식을 확인한다.
- 원천 필드 `time_nll`은 `legacy_clamped_rmtpp` score의 음의 평균이다. `w * delta_t` 제한 때문에 exact likelihood가 아니므로 표에서는 clamped time loss로 부른다.
- [time_head_revision_audit.json](time_head_revision_audit.json)은 세 실행 revision의 intercept cap 300 / `w * delta_t` cap 10과 현재 기본값 30의 차이를 Git blob에서 확인한다.
- [independent_verification.json](independent_verification.json)은 별도 CSV 경로로 모든 seed/aggregate/stratum 수치를 재계산한다.
- [source_manifest.json](source_manifest.json)은 읽은 계약, summary, CSV, 구현 근거의 SHA-256을 고정한다.

## 후속 논문 동기화 결과와 남은 작업

원본 Hard-LMM T0, RMTPP·THP·TitanTPP 모델 행, metric과 checkpoint 규칙은
[최종 계약](../../contracts/titantpp_v0_7_final_model_claim_contract_v1.md)에 동결됐다.
Quantity-interface empirical superiority는 현재 기여에서 제외했고, mark-free F1,
dataset-level validation F2와 [artifact manifest](../../manifests/titantpp_v0_7_validation_freeze_manifest.md)를
생성했다. 이 freeze 이후 남은 필수 실험은 별도 승인을 받은 one-time held-out
평가뿐이다. 평가 후에는 결과에 따라 주장을 유지하거나 축소하고, 모델이나
checkpoint를 다시 선택하지 않는다.
