# TitanTPP v0.7 revision notes

- 동결일: 2026-09-05 KST
- 상태: **validation freeze 완료, held-out test 잠금 유지**
- 최종 모델: original mark-free Hard-LMM / Count-aware TitanTPP-T0
- 본 비교: Count-aware RMTPP, THP, TitanTPP
- 평가 범위: fixed-split validation, seeds 42·52·62

## 1. 이번 revision의 질문

v0.7은 다음 질문만 검증한다.

> 동일한 mark-free continuous-count 입출력과 학습 계약을 사용할 때,
> Titan-inspired history encoder의 수량 예측 결과는 recurrent RMTPP 및
> attention-based THP와 어떻게 다른가?

v0.6의 magnitude mark, exponent residual, mark-conditioned decoder와 tail-aware
objective는 최종 구조에서 제외한다. v0.7은 `log1p(delta_t)`와
`log1p(raw_quantity)`를 관측 event token으로 사용하고, 공통 time head와 direct
log-count head로 다음 사건을 예측한다. 모델 사이에서 바뀌는 축은 history
encoder뿐이다.

상세 identity와 허용 주장은
[최종 모델·주장 계약](../contracts/titantpp_v0_7_final_model_claim_contract_v1.md)에
동결했다.

## 2. 현재 기준선 — 완료

### 최종 모델과 비교 행

- **TitanTPP-T0**를 세 데이터셋에 적용하는 공통 최종 구조로 확정했다.
- **RMTPP, THP, TitanTPP**를 본 비교 행으로 고정했다.
- 모든 행은 hidden dimension 64, direct log-count MSE, 같은 event-time head,
  seeds 42·52·62, e300 ceiling과 minimum validation joint-objective checkpoint
  선택을 공유한다.
- 원천 artifact의 `time_nll`은 `legacy_clamped_rmtpp` score의 음의 평균이다.
  `w * delta_t` cap이 활성화되면 normalized density가 아니므로 원고와 표에서는
  **clamped time loss**로 표기한다. 세 validation source revision의 실행 수식은
  intercept cap 300과 `w * delta_t` cap 10을 사용했다. 현재 코드의 effective
  intercept cap 30은 이후 변경이므로 held-out replay에 그대로 사용하지 않는다.
  이 차이는 [source-level 감사](../results/titantpp_v0_7_validation_freeze_20260905/time_head_revision_audit.json)의
  23개 검사로 확인했다.
- separate-key, elapsed-age와 Instacart raw-history probe는 최종 모델 후보가 아니라
  탐색적 진단으로 분리했다.

### 데이터셋 역할

| Dataset | 논문에서의 역할 | 현재 validation 해석 |
| --- | --- | --- |
| Intermittent-5000 | recurrent RMTPP 대비 주된 양의 증거와 THP 경계 | TitanTPP는 RMTPP보다 MAE·RMSE가 낮지만 THP보다 MAE가 높다. |
| Taxi | 다른 event semantics와 넓은 수량 범위에서의 일반화 경계 | TitanTPP는 평균 RMSE만 소폭 낮고 MAE·clamped time loss·log-count MSE는 RMTPP보다 높으며 seed 변동이 크다. |
| Instacart | 대규모 짧은 이력에서의 null/negative boundary | RMTPP가 네 지표 모두 가장 낮고 TitanTPP는 THP와 가까운 수준이다. |

이 배치는 데이터셋별로 다른 모델을 선택하기 위한 것이 아니다. 같은 T0 구조의
상대 성능이 데이터셋과 지표에 따라 달라지는 범위를 보여주기 위한 것이다.

### 세 seed validation 결과

| Dataset | TitanTPP vs RMTPP | TitanTPP vs THP |
| --- | --- | --- |
| Intermittent-5000 | MAE -74.3%, RMSE -81.9%; 각각 3/3 seeds 개선 | RMSE -10.8%, 3/3 개선; MAE +12.1%, 0/3 개선 |
| Taxi | RMSE -0.57%, 2/3 개선; MAE +5.69%, 1/3 개선 | RMSE -2.38%, 2/3 개선; MAE +2.39%, 2/3 개선 |
| Instacart | RMSE +0.43%, 1/3 개선; MAE +0.45%, 0/3 개선 | RMSE +0.25%, 1/3 개선; MAE +0.20%, 1/3 개선 |

절대 지표, sample standard deviation과 paired delta는
[T3 validation 표](../tables/T3_v0_7_backbone_validation.md)와
[T4 paired 비교표](../tables/T4_v0_7_paired_titan_deltas.md)에 기록했다.
Train에서 경계를 고정한 수량 구간별 3-seed 결과는
[T5](../tables/T5_v0_7_quantity_strata_validation.md)에 세 데이터셋을 함께 통합했다.

## 3. 기존 증적 통합과 감사 결정 — 완료

### 데이터 identity

Taxi와 Instacart의 기존 통계 행은 모델 평가에 사용한 parquet 및 split manifest와
일치했다. 과거 Intermittent 통계 행은 전체 23,387-series corpus를 가리켜 실제
Backbone 실험의 frozen 5,000-series population과 일치하지 않았다. v0.7
[T1](../tables/T1_v0_7_dataset_statistics.md)은 실험에 사용한 정확한 frozen-5000
파일에서 다시 계산했으며, 과거 행을 대체한다. 모집단·split row 수만 전체 frozen
identity에서 가져오고 sequence-length와 quantity 분포는 train rows에서만 계산했다.

### Quantity interface

[Quantity-interface 재감사](../results/titantpp_v0_7_validation_freeze_20260905/quantity_interface_audit.json)는
기존 비교를 v0.7 계약 기준 `FAIL`로 판정했다. 기존 실험은 marked input, 다른
hidden dimension과 head 조합, event-NLL checkpoint 선택을 사용했으며 정확한
train-fitted log-binned control도 포함하지 않았다.

따라서 v0.7에서는 direct log regression을 세 Backbone의 공통 인터페이스로
고정하되 categorical interface보다 경험적으로 우수하다는 주장은 제외한다. 이
결정에 따라 현재 논문 범위에는 추가 quantity-interface 3-seed 학습이 필요하지
않다. 해당 주장을 향후 다시 도입할 때만 별도 계약과 validation 실험이 필요하다.

### 독립 검산

[독립 검산 결과](../results/titantpp_v0_7_validation_freeze_20260905/independent_verification.json)는
329개 행, 6,327개 cell과 41개 source SHA-256을 재계산했다. 64개 검사가 모두
통과했고 최대 수치 오차는 0이었다. 검산 과정에서 held-out test는 사용하지 않았다.

## 4. v0.7 원고와 시각 자료 — 완료

- [v0.7 manuscript](../titantpp_short_paper_draft_v0_7_manuscript.md)는 v0.6의
  magnitude-mark/residual 설명을 공통 mark-free T0 구조로 교체했다.
- [Figure 1](../figures/F1_v0_7_mark_free_architecture.svg)은 event token, 공통
  encoder, static Hard-LMM retrieval, time head와 direct log-count head를 표시한다.
- [Figure 2](../figures/F2_v0_7_dataset_validation_errors.svg)는 데이터셋 안에서
  RMTPP를 1로 둔 세 모델의 validation MAE와 RMSE 비율을 표시한다.
- [v0.7 artifact manifest](../manifests/titantpp_v0_7_validation_freeze_manifest.md)는
  원천 파일 hash, 생성 표·그림 hash, v0.7 manuscript·루트 README hash, 27개
  qualified validation row와 held-out 잠금 상태를 기록한다.

## 5. 허용 주장과 금지 주장

### 허용

- TitanTPP는 mark-free continuous-count TPP이며 세 Backbone은 동일한 입력,
  prediction head, loss와 checkpoint rule을 사용한다.
- Intermittent-5000 validation에서 TitanTPP는 recurrent RMTPP보다 수량 MAE와 RMSE가
  크게 낮다.
- Intermittent-5000의 THP 비교와 Taxi 비교는 MAE와 RMSE 사이의 trade-off다.
- Instacart에서는 TitanTPP가 RMTPP보다 개선되지 않았으며 THP와 가까운 수준이다.
- 관측된 상대 성능은 데이터셋과 지표에 따라 달라진다.

### 금지

- 모든 데이터셋·지표에서 TitanTPP가 우수하다는 주장
- 긴 이력 기억, heavy-tail 개선 또는 특정 encoder 병목에 대한 인과 주장
- direct log regression이 categorical interface보다 우수하다는 경험적 주장
- validation 결과를 held-out generalization으로 표현하는 문구
- separate-key나 elapsed-age를 채택된 Backbone 개선으로 표현하는 문구

## 6. 남은 작업 순서

**승인 필요 / 5090 — one-time held-out 평가**

- 세 데이터셋의 RMTPP·THP·TitanTPP validation-selected checkpoint 27개만 평가한다.
- 데이터 hash, checkpoint SHA-256, metric implementation과 실행 범위를 먼저
  고정한다.
- 각 checkpoint의 source revision을 직접 사용하거나, intercept cap 300을 명시한
  compatibility path가 원본 validation 출력과 일치함을 확인한다.
- Test 결과로 모델, seed, context, checkpoint 또는 표 구성을 다시 선택하지 않는다.
- 완료 조건은 27개 결과의 완전성, identity 검증과 독립 재계산이다.

**다음 작업 / 로컬 — held-out 결과 반영과 최종 주장 감사**

- 승인된 test 결과를 validation 표와 분리해 원고에 추가한다.
- validation 결론과 다르면 주장을 축소하고, 모델 identity나 계약은 바꾸지 않는다.
- 표·그림·원고·manifest의 수치와 링크를 다시 검산한다.
- 완료 조건은 허용 주장마다 직접 대응하는 표 또는 artifact가 있고 금지 주장이
  원고에 없는 상태다.

**다음 작업 / `paper_research/master` — 제출본 정리**

- 최종 렌더링, reference와 caption을 확인한다.
- validation freeze와 held-out 결과를 구분한 독립 커밋으로 정리한다.
- Push 또는 제출은 별도 요청 범위에서 진행한다.

추가 Backbone 탐색과 quantity-interface 재실험은 현재 v0.7 제출 경로의 필수
작업이 아니다.
