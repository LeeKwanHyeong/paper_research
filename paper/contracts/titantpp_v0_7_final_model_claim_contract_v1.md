# TitanTPP v0.7 최종 모델·주장 계약 v1

- 동결일: 2026-09-05 KST
- 대상 저장소·브랜치: `paper_research/master`
- 논문 모델명: **Count-aware TitanTPP**
- 내부 실험명: **TitanTPP-T0 / original mark-free Hard-LMM**
- 개발 평가 범위: fixed-split validation, seeds 42·52·62
- held-out test: **잠금 유지**

## 1. 계약 목적과 적용 범위

이 계약은 TitanTPP v0.7 원고에서 사용할 최종 모델 identity와 현재 증거가
허용하는 주장 범위를 고정한다. Taxi와 Instacart에 서로 다른 head, loss 또는
memory 경로를 수동 선택하지 않는다. 두 데이터셋에는 같은 TitanTPP-T0 구조와
학습 계약을 적용하고, 데이터에 고정된 시간 단위와 context window만 다르게 둔다.

이 계약은 기존 validation artifact를 변경하지 않는다. 이후 새 Backbone 후보를
주 모델로 채택하려면 이 계약을 명시적으로 개정하고, held-out test를 열기 전에
동일한 비교 계약으로 validation을 다시 완료해야 한다.

## 2. 최종 모델 identity

논문의 주 모델은 [Count-aware TitanTPP 공식 T0 계약 v2](count_aware_model_baseline_v2.md)의
TitanTPP-T0다. `Hard-LMM`과 `static_hard_lmm`은 같은 정적 Hard Local Memory
Matcher를 가리키는 과거 artifact 명칭이다. 이 모델은 원본 Titans의 test-time
updated long-term memory를 구현한 것으로 서술하지 않는다.

| 축 | 동결 값 |
| --- | --- |
| Encoder | `count_titan_small_lmm`, hidden dimension 64 |
| Memory | static Hard-LMM, persistent tokens 16, prototypes 64, top-k 4 |
| 관측 이력 입력 | `log1p(delta_t)`, `log1p(raw_quantity)` |
| Quantity-derived mark/residual | 사용하지 않음 |
| 수량 target·loss | `log1p(raw_quantity)`, direct MSE |
| 원수량 point prediction | location을 `expm1`으로 복원 |
| Time head | `legacy_clamped_rmtpp`; 실행 수식은 intercept cap 300, `w * delta_t` cap 10 |
| Tail auxiliary loss | 사용하지 않음, `lambda_tail=0` |

`legacy_clamped_rmtpp`는 RMTPP log-density 식의 intercept를 최대 300으로,
`w * delta_t`를 최대 10으로 제한한 score를 사용한다. 뒤의 제한이 활성화되면 이
score는 정규화된 density를 보장하지 않는다. 기존 artifact의 `time_nll` 필드명은
provenance를 위해 유지하지만, v0.7 본문과 표에서는 **clamped time loss**로
표기하고 exact likelihood 또는 NLL로 해석하지 않는다.

실행 당시 source와 현재 `master`의 기본값은 구분한다.
[Source-level time-head 감사](../results/titantpp_v0_7_validation_freeze_20260905/time_head_revision_audit.json)는
아래 세 source revision이 계산 함수 안에서 intercept cap 300을 사용했음을
확인했다. 이후 `b1d9e63`에서 현재 코드가
`time_intercept_limit` 기본값 30을 실제 계산에 적용하도록 변경됐다. 따라서 아래
checkpoint의 held-out 평가는 현재 기본값으로 실행하지 않고, 실행 revision을 직접
사용하거나 cap 300 호환 경로가 원본 출력과 일치함을 먼저 검증해야 한다.

| Dataset | Validation 실행 source revision |
| --- | --- |
| Intermittent-5000 | `044add1f3de768d804d9f0269fd0013bd9658a35` |
| Taxi | `6a01aea9024db9e3ef6cfdd2c3d0219ceb320856` |
| Instacart | `28293c43521615be2ed8fad5b043dc9df8e5e457` |

세 데이터셋에서 위 구조, head, loss와 hidden dimension은 동일하다.
데이터셋별로 허용하는 값은 다음 context 범위와 train-only quantity-head bias
초기화 값뿐이다. Bias 값은 각 train split의 mean `log1p(quantity)`를 같은 규칙으로
변환하며, validation이나 test를 보아 선택하지 않는다.

| Dataset | 고정 시간 단위 | Lookback | Max sequence length |
| --- | --- | ---: | ---: |
| Intermittent-5000 | week | 520 | 256 |
| Taxi | hour | 168 | 256 |
| Instacart | day | 52 | 64 |

## 3. 본 비교와 공정성 계약

주 Backbone 비교 행은 **RMTPP, THP, TitanTPP-T0**다. NHP와 SAHP 결과는 보조
비교표나 부록에 유지할 수 있지만 핵심 encoder 대비를 대신하지 않는다.

세 주 비교 모델에는 같은 관측 이력 feature, direct log-quantity head와 loss,
time head, split, seed, optimizer 설정, checkpoint 선택을 적용한다. 따라서 이
T0 비교에서만 모델 간 차이를 history encoder 차이로 해석한다. TitanTPP-T1이나
탐색 후보와 T0 baseline의 차이를 Backbone 효과라고 부르지 않는다.

| 학습·선택 축 | 동결 값 |
| --- | --- |
| Seeds | 42, 52, 62 |
| Maximum / minimum epochs | 300 / 40 |
| Early-stopping patience | 40 |
| Batch size | 128 |
| Learning rate | 0.001 |
| Gradient clipping | 1.0 |
| Checkpoint | minimum validation joint objective |
| 개발 평가 | validation-only |

Checkpoint는 quantity MAE, RMSE, tail 결과 또는 held-out test를 보고 다시 고르지
않는다. 모델별 결과는 세 seed의 산술평균과 sample standard deviation으로
집계한다. 비교 방향은 같은 seed끼리도 확인하며, 서로 다른 데이터셋의 target을
합쳐 하나의 pooled metric을 만들지 않는다.

## 4. 데이터셋 역할과 현재 validation 결론

### Intermittent-5000 — 주된 양의 RMTPP 대비 증거와 THP 경계

[3-seed Backbone qualification](../results/count_aware_tpp_backbone_control_20260812/qualification_briefing.md)은
TitanTPP가 RMTPP보다 quantity MAE 74.3%, RMSE 81.9% 낮았음을 보여준다.
THP와 비교하면 TitanTPP는 RMSE가 10.8% 낮지만 MAE는 12.1% 높고, 사전 general
gate와 long-history gate는 모두 실패했다. 따라서 Intermittent-5000은 recurrent
RMTPP 대비 수량 오차 감소의 주된 증거이지만, THP를 포함한 보편적 우위나
long-history 우위의 증거가 아니다.

### Taxi — 긴 이력·넓은 수량 범위에서의 혼합 결과

[Taxi 3-seed 결과](../results/count_aware_taxi_t0_t1_e300_20260824/result_briefing_ko.md)은
RMTPP가 validation joint objective와 quantity MAE에서 가장 낮고,
TitanTPP-T0가 평균 quantity RMSE에서 가장 낮음을 보여준다. TitanTPP-T0의 seed
간 RMSE 편차가 크므로 Taxi를 일관된 Backbone 우위로 해석하지 않는다. Taxi는
서로 다른 데이터 특성에서 같은 T0 구조의 metric trade-off와 안정성을 보여주는
일반화 경계다.

### Instacart — 대규모 짧은 이력에서의 null/negative boundary

[Instacart 3-seed 결과](../results/count_aware_instacart_t0_e300_20260824/result_briefing_ko.md)은
RMTPP가 validation joint objective, clamped time loss, quantity MAE와 RMSE에서 모두
가장 낮고, TitanTPP-T0가 THP와 가까운 수준임을 보여준다. 모든 validation
target의 history가 64 이하이므로 Instacart는 짧은 이력의 basket-count
일반화 경계이며 long-history memory를 검증하는 데이터셋으로 사용하지 않는다.

세 데이터셋의 결과는 하나의 구조가 서로 다른 데이터 분포에서 다른 상대 성능을
보인다는 **dataset-dependent validation 결론**으로 함께 보고한다. 데이터셋별
상대 성능 차이를 특정 구조적 병목의 인과 증거로 확대하지 않는다.

## 5. 논문 지표와 보고 규칙

| 지표 | 역할 |
| --- | --- |
| Raw quantity RMSE | 큰 수량 오차에 민감한 주 quantity 지표 |
| Raw quantity MAE | 평균 절대 수량 오차를 나타내는 보조 quantity 지표 |
| Log-quantity MSE | 공통 수량 학습 공간의 보조 지표 |
| Clamped time loss | 공통 `legacy_clamped_rmtpp` time-score의 음의 평균. Artifact field는 `time_nll`이지만 exact NLL로 해석하지 않음 |
| Validation joint objective | checkpoint 선택 감사용; 단독 우월성 주장에 사용하지 않음 |

Quantity/history 구간 분석을 보고할 때 경계는 train split에서만 계산하고 세 모델에
같은 target membership을 적용한다. 한 지표의 우위를 다른 지표의 우위로 바꾸어
서술하지 않는다. Validation 결과는 held-out test 성능으로 표현하지 않는다.

## 6. 허용 주장

다음 주장은 validation 또는 이후 계약에 맞는 held-out 결과라는 평가 범위를 함께
표시할 때만 허용한다.

1. TitanTPP는 mark를 quantity bin으로 사용하지 않고 관측 시간 간격과 연속 수량을
   입력받아 다음 사건 시간과 log 수량을 함께 예측하는 count-aware TPP다.
2. 동일한 direct count-regression interface에서 TitanTPP는 Intermittent-5000의
   quantity MAE와 RMSE를 recurrent RMTPP보다 크게 줄였다.
3. Intermittent-5000에서 TitanTPP와 THP의 비교는 MAE와 RMSE 사이의 trade-off다.
4. Taxi에서 TitanTPP-T0는 평균 RMSE가 가장 낮았지만 RMTPP보다 MAE와 joint
   objective가 낮지 않았고 seed 변동도 컸다.
5. Instacart에서 현재 TitanTPP-T0는 RMTPP보다 우수하지 않았고 THP와 가까운
   성능을 보였다.
6. 전체 결과는 TitanTPP의 효과가 데이터셋과 평가 지표에 따라 달라짐을 보여준다.

## 7. 금지 주장

다음 주장은 현재 증거로 사용하지 않는다.

1. TitanTPP 또는 Hard-LMM이 모든 데이터셋, 모든 TPP 또는 모든 지표에서 우수하다.
2. Hard-LMM이 긴 이력을 더 잘 기억하거나 long-term dependency를 더 잘 포착한다.
3. Taxi의 결과가 heavy-tail 개선, long-history 개선 또는 separate-key retrieval의
   일반적 효과를 입증한다.
4. Instacart의 결과가 데이터에 예측 신호가 없거나 더 이상의 성능 개선이
   불가능함을 뜻한다.
5. 데이터셋별 head, loss 또는 memory 경로를 수동 선택한 모델을 동일한 공통
   Backbone의 결과로 제시한다.
6. Validation 결과를 held-out generalization 또는 인과적 mechanism 증거로 부른다.

## 8. 탐색 실험의 논문상 위치

다음 실험은 최종 주 모델 identity를 바꾸지 않으며, 필요할 경우 제한된 진단 또는
향후 연구로만 보고한다.

- [Separate-key retrieval](../results/hard_lmm_key_value_screening_5090_20260904/README.md):
  seed42에서 Taxi는 사전 기준을 통과했지만 Instacart는 실패해 1/2 dataset으로
  종료됐다. 일반 Backbone 개선이나 3-seed 채택 결과가 아니다.
- [Elapsed-age encoding](../results/hard_lmm_elapsed_age_screening_5090_20260905/README.md):
  seed42 Taxi·Instacart 동시 채택 기준을 충족하지 못해 0/2로 종료됐다.
- [Instacart raw-history accessibility](../results/hard_lmm_instacart_raw_history_20260905/README.md):
  raw 이력 자체에는 다음 수량 신호가 있었지만, 고정한 동일 용량 probe에서 raw가
  encoder 상태 `h`보다 낫지 않았고 frozen model 잔차도 줄이지 못했다. 이는 제한된
  train-only 접근성 진단이며 정보 부재나 개선 불가능성의 증거가 아니다.

TitanTPP-T1 tail-aware objective도 주 모델이 아니다. Intermittent-5000에서의 양의 결과와
Taxi에서의 악화를 함께 제시하는 objective ablation으로만 다룬다.

## 9. Quantity-interface 감사 결정과 held-out 잠금

[기존 quantity-interface 증적 감사](../results/titantpp_v0_7_validation_freeze_20260905/quantity_interface_audit.json)는
v0.7 공정 비교 계약을 충족하지 못해 `FAIL`로 판정됐다. 기존 실험은 marked input,
다른 hidden dimension, 다른 head 조합과 event-NLL checkpoint 선택을 사용했고,
train-fitted log-binned categorical control도 아니다. 따라서 v0.7에서는 direct
log-quantity regression을 세 Backbone의 **공통 고정 인터페이스**로만 사용하며,
categorical interface 대비 경험적 우위는 기여와 주장 목록에서 제외한다.

이 결정으로 현재 논문 범위를 완성하기 위한 추가 quantity-interface 학습은 요구하지
않는다. 향후 해당 주장을 다시 도입하려면 RMTPP backbone, mark-free history input,
split, seeds, 학습 예산과 joint-objective selector를 고정한 뒤 train-fitted log bins와
numeric representatives를 별도의 개정 계약 아래 검증해야 한다. 그 실험은 현재
held-out 평가 대상이나 모델 선택 후보에 포함하지 않는다.

Held-out test는 다음 조건을 모두 충족할 때까지 잠근다.

1. 최종 model rows, checkpoint 목록, 지표, 표와 그림 구성이 고정된다.
2. 각 validation artifact의 source, data, split과 checkpoint identity 감사가 끝난다.
3. quantity-interface 우위 주장이 현재 범위에서 제외됐음이 원고에 반영된다.
4. source revision 직접 실행 또는 cap 300 compatibility replay의 출력 동일성이
   validation sample에서 검증된다.
5. one-time held-out evaluation의 범위와 실행이 별도로 승인된다.

잠금 해제 후에는 validation에서 선택된 checkpoint만 한 번 평가한다. Test 결과로
모델을 다시 선택하거나 head, loss, context, seed, checkpoint 또는 주장을 유리하게
바꾸지 않는다. Test 결과가 validation 주장과 다르면 결과를 숨기지 않고 주장을
그에 맞게 축소한다.

## 10. 현재 논문 완료 경로

현재 논문에는 추가 Backbone 탐색이나 quantity-interface 학습을 필수 작업으로 두지
않는다. 새 공통 후보를 탐색하거나 interface 우위 주장을 다시 도입하려면 이 계약과
held-out 잠금을 유지한 별도 연구 주기로 분리한다. v0.7의 남은 순서는 최종
validation 표·그림과 checkpoint 목록 동결, 승인된 one-time held-out 평가, test
결과를 반영한 주장 축소 여부 확인과 최종 원고 감사다.
