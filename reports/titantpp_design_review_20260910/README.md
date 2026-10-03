# TitanTPP 다음 설계 검토: 시간·수량 분리의 근거와 대안

검토일: 2026-09-10. 대상: `paper_research`, `codex/hard-lmm-causal-qkv`.
방법: OMA Architecture의 대안 비교와 독립적인 증거·코드·평가 설계 검토.
상태: **설계 검토 완료 / 분리 모델 성능 미검증 / 신규 학습·제품 코드 변경 없음**.

## 결론

**시간·수량 완전 분리는 진단 대조군으로 검토할 가치가 있지만 다음 본모델로 확정할 근거는 부족하다.** 공동 학습·수량 전용·시간 전용을 동일한 B 구조와 초기화에서 비교해, 공동 학습이 각 목표의 성능을 실제로 제한하는지 먼저 확인한다. 수량 전용과 시간 전용 모델의 조합이 완전 분리 시스템이므로, 첫 진단을 위해 네 번째 거대 모델을 만들 필요는 없다.

앞선 대화의 “공동 표현의 간섭을 우선 원인으로 보고 분리 구조를 구현하자”는 권고는 낮춘다. 확인한 과거 실험에는 분리 가설을 지지하지 않은 결과가 있으며, 최신 Taxi의 실패에는 checkpoint 선택과 학습 궤적이 크게 관련된다. 시간분포의 scale, 관측 likelihood, loss 규모와 gradient clipping도 별도 설명으로 남아 있다.

이번 검토에서 확정한 것은 **다음에 반증할 질문과 구현 계약의 범위**다. 실제 성능 증거 없이 분리 모델의 효과가 검증됐다고 간주하거나 원격 학습을 시작하지 않았다.

## 현재 모델에서 실제로 공유하는 것

- 현재 count-aware B는 시간·수량 head 자체는 이미 분리되어 있지만 동일 encoder 출력을 사용한다. 입력 projection·위치 표현·attention/FFN·persistent memory·최종 static Hard-LMM bank는 두 목표의 gradient를 받는다. [CountAwareTPP.py](../../models/TPPs/CountAwareTPP.py)
- 두 과제 모두 관측된 시간 간격과 수량 이력을 입력받는다. “목표 분리”를 “시간 모델에는 시간만, 수량 모델에는 수량만 제공”으로 바꾸면 입력 정보의 차이가 추가된다.
- 현재 학습 목적은 time loss + log1p 수량 MSE이며 AdamW와 전체 parameter 대상 norm clipping을 사용한다. encoder를 두 개로 나눠도 두 branch를 함께 clipping하면 최적화 결합이 남는다. [core.py](../../paper/scripts/count_aware_tpp_backbone/core.py), [training.py](../../paper/scripts/count_aware_tpp_backbone/training.py)
- quantity head는 0으로 초기화되므로 초기 순간의 quantity-to-encoder gradient는 0이다. 초기 한 번의 gradient 측정만으로 지속적인 시간 우세나 간섭 부재를 판단하면 안 된다.
- 현재 RMTPP·THP 비교군도 동일 입력·head를 사용하도록 만든 count-aware adaptation이다. 원 논문의 모든 기본 설정을 그대로 실행한 모델이라고 표현하지 않는다.

## 중요한 반례와 대안 설명

### 1. Taxi에서 수량과 시간이 함께 개선된 시점이 존재한다

원본 history를 직접 대조했다. 아래는 validation 기록이며 새 checkpoint를 선택하거나 복원한 결과가 아니다.

| 모델·epoch | Raw RMSE | 전체 MAE | legacy clamped time loss |
| --- | ---: | ---: | ---: |
| B, epoch45 — 기존 선택 | 88.194997 | 28.674020 | 1.473391 |
| 후보, epoch45 — 설명용 history | 80.781331 | 25.718790 | 1.440061 |
| B, epoch83 — 설명용 history | 93.164080 | 29.370771 | 8.007974 |
| 후보, epoch83 — 기존 선택 | 78.255427 | 25.048308 | 10.825870 |

따라서 최종 선택값의 시간 악화를 “공동 표현은 두 목표를 동시에 잘 처리할 수 없다”는 증거로 사용할 수 없다. B도 같은 후기 epoch에서 시간 loss가 악화됐다. 후보 epoch45의 Body/tail, checkpoint 보존과 전체 gate 통과는 여기서 확인되지 않았으므로, 이 시점을 새 채택 결과로 바꾸지 않는다. 원래 Taxi 실패와 B 유지 판정은 그대로다.

동일 조건의 설명용 history 지점은 Taxi15개, Instacart1개, Intermittent0개였다. 이 수치는 `candidate RMSE < selected B RMSE` 및 `candidate legacy time <= selected B time + 0.01`을 만족하는 지점 수다. 사후 집계이며 새로운 선택 규칙이나 성능 승격 근거가 아니다. [재계산과 SHA](selector_evidence.json)

### 2. 과거의 분리 진단은 강한 지지를 주지 않았다

| 증거 | 확인된 내용 | 현재 판단에 적용할 범위 |
| --- | --- | --- |
| [8/20 gradient 진단](../../paper/results/titantpp_time_quantity_gradient_audit_20260820/result_analysis.md) | Intermittent의 cosine 중앙값 +0.0466, 강한 충돌25%; 사전 분리 조건 미충족 | 다른 head·checkpoint 조건이다. 현재에도 간섭이 없다는 증거는 아니지만 기존 부정적 결과를 무시하지 않는다. |
| [8/20 H0/H3 attribution](../../paper/results/titantpp_h0_h3_gradient_attribution_20260820/result_analysis.md) | H3에서 time-head가 joint gradient 제곱norm의93.30%, clipping100%; encoder time/quantity norm7.06배, cosine−0.0066 | 방향 충돌보다 magnitude와 clipping의 결합이 강한 대안 설명이다. |
| [9/9 shared-block 진단](../../paper/results/hard_lmm_shared_block_transfer_probe_20260909/README.md) | H1/H2/fused 경계에서 공통 분리 경로의 필요조건을 세 데이터셋 모두 통과하지 못함 | frozen B의 한 단계 국소 진단이며 fresh 독립 학습을 반증하지 않는다. |
| [9/4 local-time screening](../../paper/results/hard_lmm_local_time_screening_20260904/README.md) | 시간에는 local state, 수량에는 memory residual을 제공한 routing이0/2 승격 | 공유 encoder를 유지한 이전 후보다. 완전 분리와 다르지만 “처음 해보는 시간·수량 routing 분리”가 아니다. |

### 3. 시간 head의 제한적 실패를 표현 정보의 부재로 단정할 수 없다

[Frozen time-head refit](../../paper/results/hard_lmm_time_head_refit_seed42_20260906/README.md)은 세 데이터셋에서 시간 점수를 개선했으나 원래 공통 gate를 통과하지 못했다. Intermittent는 epoch100의 상한에서도 개선 중이었다. 이 계약의 미달을 모든 head의 불가능성으로 일반화하지 않는다.

[Frozen log-normal 비교](../../paper/results/matched_frozen_lognormal_duration_seed42_20260906/README.md)는 A/B의 checkpoint selector도 다르다. 이후 [관측 및 scale 진단](../../paper/results/hard_lmm_causal_duration_adapter_seed42_20260906/README.md)의 교차 재생은 Taxi A/B gap의 대부분이 scale 차이와 연결됨을 보여준다. 이는 “시간 정보가 공유 encoder에서 이미 사라졌다”는 단정에 대한 대안 설명이다. [정렬된 scale adapter](../../paper/results/aligned_causal_duration_scale_adapter_seed42_20260907/README.md)와 [CDF calibration](../../paper/results/aligned_conditional_cdf_calibration_e300_20260907/README.md) 역시 공통 gate를 통과하지 못했으므로 이름만 바꿔 재실행하지 않는다.

### 4. 현재 B의 강점도 정확히 유지해야 한다

[동일 seed42 기준선](../../paper/results/final_backbone_and_baselines_20260909/seed42_validation_comparison.md)에서 B의 raw RMSE는 Intermittent와 Taxi에서 RMTPP·THP보다 낮다. Instacart에서는 두 모델보다 높고, MAE 순위는 다시 다르다. “TitanTPP 전체가 benchmark보다 전반적으로 나쁘다”를 연구의 출발점으로 삼지 않는다. 또한 추가 memory 후보의 실패를 기존 B의 실패와 혼동하지 않는다.

## 대안 비교와 결정

| 대안 | 장점 | 핵심 위험·비용 | 결정 |
| --- | --- | --- | --- |
| 완전 분리 encoder를 즉시 본모델로 개발 | 독립 학습이 명확함 | 전체 용량·계산·선택 기회 증가; 기존 반례를 설명하지 못함 | 본모델 채택 보류. Q/T 진단 모델의 조합으로 먼저 측정 |
| 추가 shared/private gate·memory 재구성 | 일부 공유로 비용을 줄일 가능성 | 효과 근거 없이 새 설계 축을 추가, 기존 routing 후보와 중복 | 진단에서 반복되는 분리 이득이 확인될 때 재검토 |
| 동일 B의 J/Q/T 목적 진단 | 작은 구현으로 “다른 목표를 추가한 효과”를 직접 비교 | loss 규모·clipping·selector를 통제하지 않으면 인과 설명이 불명확 | **다음 구현 대상으로 권고** |
| head·loss·selector를 동시에 교체 | 정상화된 과학적 평가 기반을 만들 수 있음 | 기존 B와 직접 matched 비교가 아니며 변화 원인 분리 불가 | 새 논문 비교 트랙으로 별도 계약; 분리 효과와 함께 주장하지 않음 |

## 다음 구현 계약: J/Q/T 진단

### 1. 현재 B와 연결되는 호환성 진단

- 모델: 새 memory 없이 기존 static Hard-LMM B 계열. `J = Lt + Lq`, `Q = Lq`, `T = Lt`. Q와 T는 각각 전체 encoder와 해당 head를 새로 학습한다.
- 각 군은 동일 초기 tensor를 복제한다. seed를 한 번 설정하고 세 생성자를 연속 호출하는 방식은 동일 초기화가 아니다. 데이터 표본·입력·batch 순서·dropout RNG 정책·optimizer·학습률·weight decay를 명시한다.
- 사용하지 않는 head를 optimizer에서 제외하고 gradient가 None인지 및 state 불변을 검사한다. 손실에 단순히0을 곱하면 AdamW decay 또는 `0 * NaN` 문제가 남는다.
- frozen 결과 연결을 위해 legacy cap300을 명시한다. 현재 로컬 기본 cap30을 그대로 쓰고 원본 재현이라 보고하지 않는다. legacy score는 clamped loss이며 normalized Time NLL이 아니다.
- 사전 고정된 동일 optimizer-step 예산에서 비교한다. 군마다 다른 early stopping으로 주 비교의 학습 노출을 바꾸지 않는다. 구체적인 step/epoch 예산은 실행 전 비용 측정과 함께 동결한다. 이번 검토는 학습 예산을 소모하거나 GPU 작업을 예약하지 않는다.
- 같은 시점의 J 대 Q 수량, J 대 T 시간을 주 비교로 둔다. J에도 수량 selector와 시간 selector의 보조 결과를 각각 제공한다. 두 전용 모델의 각자 최적 checkpoint를 J의 단일 raw-RMSE checkpoint와만 비교하지 않는다.
- Q의 시간 성능과 T의 수량 성능은 평가 대상이 아니다. 학습하지 않은 head의 오차를 실패로 세지 않는다.
- gradient cosine과 norm을 encoder·head별로 clipping 전에 기록하고 clip 계수·발생률을 함께 본다. 음수 cosine만으로 성능 저하의 원인을 확정하지 않는다.
- 초기·학습 중 여러 시점을 보며, 초기 quantity-head가0인 상태의 gradient만으로 판단하지 않는다.

### 2. 논문에서 정상화된 시간 성능을 주장할 트랙

시간 확률의 성능을 주장하려면 J/Q/T와 benchmark 모두 같은 관측 law를 사용해야 한다. 기존 [관측 정렬 head 계약](../../paper/results/aligned_frozen_lognormal_duration_local_20260907/README.md)을 재사용할 수 있는지 먼저 확인한다. Intermittent의 continuous 단위, Taxi의 positive-integer PMF, Instacart의 censoring 처리를 혼합하지 않는다. latent quantizer 가정을 실제 전처리의 반올림 사실로 설명하지 않는다.

관측 정렬 head는 기존 frozen-refit 구현이 있다는 사실과 fresh end-to-end 학습에 통합·검증됐다는 사실이 다르다. 새 head/loss로 실험할 경우 **새로운 J도 처음부터 학습**해야 하며 원래 B를 matched control처럼 재사용하지 않는다. legacy `+0.01`을 정상화된 likelihood의 동일 의미 허용폭이라고 자동 승계하지 않는다.

### 3. 진단 결과를 본모델 설계로 연결하는 조건

- J와 Q/T의 차이가 없거나 충분히 수렴하지 않았다면 분리의 필요성은 미확인이다. 짧은 예산의 실패를 구조적 불가능성으로 확대하지 않는다.
- 특정 데이터셋에서만 이득이면 공통 분리 Backbone이 아니라 적용 범위가 제한된 가설이다.
- 반복되는 이득이 있어도 우선 “해당 학습 규칙에서 다른 목적을 추가한 효과”로 설명한다. 방향 충돌·크기 불균형·global clipping·capacity 중 무엇이 원인인지 분리한다.
- 두 원래 크기 네트워크를 합친 Q/T 시스템은 대략 두 encoder분의 용량과 계산을 쓴다. 같거나 비슷한 총 parameter의 큰 shared control 및 계산 예산 통제가 필요하다. parameter 일치와 FLOPs 일치는 서로 다른 기준이다.
- 같은 J/Q/T 원칙을 THP·RMTPP에도 적용한다. 공통 분리 기법의 이득과 TitanTPP의 추가 기여를 구별한다.
- “세 데이터셋 모두 benchmark보다 낮은 raw RMSE”가 논문 가설이면 각 데이터셋에서 사전 지정 benchmark 모두를 넘어야 한다. B만 이기는 것을 대체 기준으로 쓰지 않는다. 서로 단위가 다른 RMSE를 단순 평균하지 않는다.
- 원래 raw RMSE 주 지표와 전체·Body·tail·시간 성능을 함께 기록한다. 새 트랙의 허용폭, 추가 seed와 최종 held-out 절차는 결과 열람 전에 확정한다. 이번 리뷰에서 그 기준을 사후 변경하지 않는다.

## 독립 검토와 남은 불확실성

- 과거 증거 검토: 공동 학습 간섭을 주원인으로 확정하기 어렵고, 부정적 분리 실험과 head/selector 대안 설명이 있음.
- 코드 검토: 실제로 공유하는 trainable 경로가 확인됨. 완전 분리 fresh training은 기존 detached/frozen 경로와 다르며 현재 CLI에 J/Q/T 모드가 없음.
- 비교 설계 검토: 독립 학습은 유효한 대조군이나 capacity·compute·selector·관측모형을 통제하지 않으면 논문 기여가 과장됨.
- 종합 판단은 검토자의 동의 개수가 아니라 위 근거에 따른 것이다. **공유 경로의 존재는 확인됐고, 공유로 인한 성능 손실은 미확인**이다.

관련 문헌도 한 방향만 지지하지 않는다. [Sener & Koltun](https://arxiv.org/abs/1810.04650)은 다중 목표의 상충을 명시적으로 다루지만, [Kurin et al.](https://arxiv.org/abs/2201.04122)은 단순 손실 합과 표준 안정화가 복잡한 다중 과제 최적화 기법에 경쟁적일 수 있음을 보인다. 이 일반 연구를 현재 TitanTPP에서의 간섭 유무를 증명하는 자료로 사용하지 않는다.

## 검증 범위와 주의 기록

- 이번 실행은 로컬 문서·코드 검토와 frozen validation history JSON 재계산뿐이다. 신규 inference·checkpoint 역직렬화·학습·원격 실행·스케줄 변경·커밋·Push·MR을 하지 않았다.
- 하나의 검토자가 과거 marked V3c 구조 문서를 읽던 중 주변의 역사적 held-out 수치를 우발적으로 보았다. 해당 문서 검토를 즉시 중단하고, 문서와 수치 모두 권고·후보 선택 근거에서 제외했다. 이번 검토 전체에 대해 “held-out 문구나 수치를 전혀 열람하지 않았다”고 주장하지 않는다. 자세한 경계 기록은 [scope_incident.json](scope_incident.json)에 수치 없이 보존한다.
- source 파일과 원래 성능 판정은 보존했다. 주요 입력의 SHA와 파일 존재 확인은 [review_receipt.json](review_receipt.json)에 기록한다.

## 남은 작업 순서

**진단 실행기와 재현 계약 구현 — 다음 작업**
- 현재 세션에서 별도 experimental runner의 J/Q/T objective, inactive head 제외, 동일 초기화·batch 및 resume identity, 두 selector를 구현한다. 기존 baseline 기본 동작을 보존한다.
- 초기 공동 objective/gradient 일치, 비활성 head 불변, target/padding 인과성, 동일 step 비교, strict earliest tie와 저장·복원 검증이 완료 조건이다.

**현재 B의 학습 간섭 진단 — 구현 검증 후 다음 작업**
- 현재 B와 dual-timescale의 train-only gradient·clipping을 같은 표본에서 비교하고, J/Q/T의 동일 budget 학습을 수행할 실행 계약을 동결한다. 서버와 비용·실행 한도를 그 계약에 적는다.
- 과거 학습이 승인됐다는 이유로 새 GPU 실험을 자동 재개하지 않는다. scheduler는 종료된 기존 평가의 PAUSED 상태를 유지한다.

**분리 구조와 benchmark 비교 확정 — 진단 결과 후 다음 작업**
- 이득이 반복될 때만 capacity/compute 통제와 RMTPP·THP 적용을 포함한 최종 구조를 설계한다. 초기 구현 결과나 단일 seed validation을 논문의 우월성 증거로 승격하지 않는다.
