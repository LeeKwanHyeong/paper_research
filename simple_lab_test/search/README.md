# search

이 폴더는 재현 가능한 실험 러너와, 다음 단계 실험 설계 문서를 함께 두는
공간입니다. 현재는 기존 탐색 스크립트 설명보다, 다음 핵심 ablation인
`qty-supervision` 실험 설계를 명확히 남기는 것이 더 중요합니다.

## 현재 들어 있는 스크립트

- `titan_hparam_search.py`
  - `yellow_trip`, `intermittent`에서 `scale_base`와 Titan preset을 자동 탐색
- `titan_rmtpp_ab_test.py`
  - 탐색 결과에서 고른 Titan 조합으로 RMTPP vs TitanTPP A/B 테스트
- `tpp_qty_loss_ablation.py`
  - `residual_only`, `hybrid`, `qty_only` quantity-supervision ablation
- `compare_log_bases_distribution.py`
  - `log10`, `log4`, `log2` mark 분포 비교

## 다음 실험: qty supervision ablation

### 문제 정의

현재 학습은 `qty`를 직접 맞추지 않고 `residual`을 회귀합니다.

```text
L_current = L_mark + lambda_dt * L_time + lambda_value * L_residual
```

여기서

- `L_mark`: next mark cross-entropy
- `L_time`: next inter-event time negative log-likelihood
- `L_residual`: true residual vs predicted residual Huber loss

입니다.

검증에서는 이 residual을 최종 quantity로 복원해 `qty_mae`를 계산합니다.
즉, `qty_mae`는 중요한 지표이지만 현재는 간접적으로만 최적화됩니다.

### 왜 후속 실험이 필요한가

지금까지의 curve를 보면 특히 TitanTPP에서 `qty_mae`의 epoch별 up/down이
상대적으로 크게 나타납니다. 이는 다음 구조적 이유와 맞닿아 있습니다.

- residual은 log-like space에서 작은 오차여도
- 최종 복원은 `qty = base^(mark + residual)`이라
- 원래 quantity space에서는 오차가 더 크게 증폭될 수 있음

또한 mark가 경계 근처에서 뒤집히면 quantity reconstruction이 불연속적으로
점프할 수 있습니다.

따라서 다음 질문은 자연스럽습니다.

```text
quantity 자체를 학습 objective에 직접 넣으면 더 안정적일까?
```

## 비교할 두 설계

### A. qty direct loss

가장 직접적인 아이디어는 residual loss를 빼고, quantity space에서 바로 loss를
거는 것입니다.

```text
L_qty_only = L_mark + lambda_dt * L_time + lambda_qty * L_qty
```

여기서 `L_qty`는 복원된 quantity와 실제 quantity의 Huber loss입니다.

하지만 구현에서 주의할 점이 있습니다. 학습 중 `argmax(mark)`로 quantity를
복원하면 mark head로 gradient가 흐르지 않습니다. 따라서 training-time quantity
loss는 다음처럼 `expected quantity`를 사용해야 합니다.

```text
p_k = softmax(mark_logits)_k
q_hat_expected = sum_k p_k * base^(k + residual_hat)
q_true = base^(mark_true + residual_true)
L_qty = Huber(q_hat_expected, q_true)
```

#### 장점

- validation `qty_mae`와 objective가 직접 맞닿음
- residual이 아닌 최종 복원 quantity를 바로 압박함
- large-quantity error를 더 직접적으로 줄일 가능성이 있음

#### 리스크

- heavy-tail quantity가 loss를 지배할 수 있음
- log-space의 안정성을 버리므로 학습이 더 거칠어질 수 있음
- mark/time 학습과 충돌하면 validation NLL이나 mark_acc가 나빠질 수 있음

### B. residual loss + qty loss hybrid

두 번째 설계는 현재 residual supervision을 유지한 채, quantity loss를 보조항으로
추가하는 방식입니다.

```text
L_hybrid = L_mark
         + lambda_dt * L_time
         + lambda_value * L_residual
         + lambda_qty * L_qty
```

이 방식 역시 `L_qty`는 differentiable expected-quantity 경로로 계산합니다.

#### 장점

- 현재의 안정적인 residual supervision을 유지함
- 동시에 최종 quantity metric과의 미스매치를 줄일 수 있음
- qty loss weight를 작게 시작하며 안전하게 tuning 가능

#### 리스크

- loss 항이 하나 더 늘어나므로 tuning이 필요함
- `lambda_qty`가 너무 크면 결국 qty-only처럼 tail-dominant가 될 수 있음

## 현재 추천

현재 상황에서는 **hybrid를 1순위**, `qty direct loss`를 **대조 ablation**으로
두는 편이 더 타당합니다.

이유는 간단합니다.

- 우리는 이미 residual-only 구조가 작동한다는 것을 알고 있음
- 지금 문제는 “작동하지 않음”이 아니라 “qty metric이 충분히 직접 최적화되지 않음”
- 따라서 baseline을 완전히 버리기보다, residual 안정성을 살린 채 qty objective를
  추가하는 편이 실패 비용이 작음

즉, 실험 우선순위는 아래처럼 두는 것을 권장합니다.

1. residual-only baseline
2. hybrid
3. qty-only

## 제안하는 구현 규칙

### 1. training-time quantity loss는 argmax가 아니라 expectation으로 계산

이 규칙은 반드시 지켜야 합니다. 그렇지 않으면 mark head가 quantity loss에서
사실상 학습되지 않습니다.

### 2. qty loss는 Huber로 시작

MSE는 큰 demand 몇 개에 지나치게 끌릴 가능성이 큽니다. 첫 실험은 Huber가
안전합니다.

### 3. qty loss weight는 작게 시작

첫 grid는 다음 범위를 권장합니다.

- `lambda_qty in {0.1, 0.25, 0.5, 1.0}`

특히 hybrid에서는 `lambda_value=1.0`을 유지한 채 `lambda_qty=0.25`부터
시작하는 것이 가장 무난합니다.

### 4. quantity loss 안정화를 위한 scaling

strict raw-quantity Huber도 가능하지만, dataset 간 스케일 차이가 큰 만큼
다음 보조 안정화 중 하나를 함께 고려하는 것이 좋습니다.

- training-set `qty_p95`로 나눈 뒤 Huber
- 또는 dataset별 fixed quantity scale로 나눈 뒤 Huber

이 방식은 여전히 quantity space loss이면서, 일부 extreme value가 objective를
완전히 지배하는 것을 막아줍니다.

## 실험 매트릭스

### 1차 실험 범위

우선은 가장 변동이 큰 TitanTPP 쪽에서 먼저 검증합니다.

- model: `TitanTPP`
- dataset: `intermittent`, `yellow_trip`
- loss mode:
  - `residual_only`
  - `hybrid`
  - `qty_only`
- epochs: `30`
- seeds: `42, 52, 62`
- scale base:
  - dataset best 조합 유지
  - 필요하면 second pass에서 overall best 조합도 같이 확인

### 2차 확장

TitanTPP에서 방향성이 확인되면 RMTPP에도 동일 loss mode를 붙여 비교합니다.

- model: `RMTPP`, `TitanTPP`
- loss mode: best two modes only
- datasets: same

## 저장 산출물

새 실험 스크립트는 아래 결과를 남기도록 설계합니다.

- run-level metrics
- epoch history
- paper table CSV/Markdown
- delta table
- learning curve plots
- qty-loss 전용 비교 보고서 Markdown

권장 경로:

```text
search_artifacts/tpp_qty_loss_ablation/
```

## 구현 파일

현재 설계는 아래 스크립트에 구현되어 있습니다.

```text
simple_lab_test/search/tpp_qty_loss_ablation.py
```

## 성공 기준

이 실험의 목적은 단순히 `qty_mae`만 내리는 것이 아닙니다. 아래를 함께 봐야
합니다.

- `qty_mae` 개선
- `val_nll` 유지 또는 개선
- `mark_acc` 급락 없음
- seed variance 감소
- TitanTPP의 epoch별 qty curve 출렁임 완화

## 예상 시나리오

현재 기준으로 가장 가능성이 높은 결과는 다음과 같습니다.

- `qty_only`
  - `qty_mae`는 일부 개선 가능
  - 하지만 validation NLL, mark stability가 흔들릴 리스크 큼
- `hybrid`
  - `qty_mae` 개선
  - residual-only 대비 더 안정적인 절충 가능성 큼

따라서 현재 가설은 아래처럼 정리할 수 있습니다.

```text
The most likely winner is not pure quantity-only supervision,
but residual supervision augmented with a moderate direct quantity loss.
```

이 문서를 기준으로 다음 단계 구현을 진행합니다.
