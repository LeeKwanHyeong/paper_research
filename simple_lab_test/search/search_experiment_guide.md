# search

이 폴더는 재현 가능한 실험 러너와, 다음 단계 실험 설계 문서를 함께 두는
공간입니다. 현재는 기존 탐색 스크립트 설명보다, 다음 핵심 ablation인
`qty-supervision` 실험 설계를 명확히 남기는 것이 더 중요합니다.

## 현재 들어 있는 스크립트

- `titan_hparam_search.py`
  - `yellow_trip`, `intermittent`에서 `scale_base`와 Titan preset을 자동 탐색
- `titan_rmtpp_ab_test.py`
  - 탐색 결과에서 고른 Titan 조합으로 RMTPP vs TitanTPP A/B 테스트
- `titan_rmtpp_long_epoch_scale_eval.py`
  - 교수님 피드백 대응용 long-epoch 수렴 검증 및 scale-wise quantity MAE 분석
- `tpp_overfit_diagnostic.py`
  - RMTPP/TitanTPP가 실제로 train distribution을 강하게 학습하고 overfitting까지 만들 수 있는지 확인하는 진단 실험
- `tpp_qty_loss_ablation.py`
  - `residual_only`, `hybrid`, `qty_only` quantity-supervision ablation
- `compare_log_bases_distribution.py`
  - `log10`, `log4`, `log2` mark 분포 비교

## 교수님 피드백 대응: long-epoch + scale-wise MAE

### 문제 정의

30 epoch learning curve에서 RMTPP와 TitanTPP 모두 validation NLL이 계속 낮아지는
패턴을 보였습니다. 이 경우 30 epoch 비교만으로는 학습이 충분히 끝났는지,
혹은 더 오래 학습했을 때 overfitting이 발생하는지 판단하기 어렵습니다.

또한 전체 `qty_mae`는 큰 수요 이벤트 몇 개에 강하게 끌릴 수 있습니다. 특히
현재 quantity reconstruction은

```text
qty = scale_base^(mark + residual)
```

형태이므로, 큰 scale에서 작은 log-space 오차가 quantity-space에서는 매우 큰
absolute error로 확대될 수 있습니다. 따라서 전체 평균 MAE만 보면 작은 수요
구간에서 모델이 얼마나 잘 맞추는지 해석하기 어렵습니다.

### 새 실험 스크립트

위 질문을 확인하기 위해 아래 스크립트를 추가했습니다.

```text
simple_lab_test/search/titan_rmtpp_long_epoch_scale_eval.py
```

이 스크립트는 기존 본실험 스크립트인 `titan_rmtpp_ab_test.py`의 데이터 준비
흐름과 Titan profile 설정을 재사용하지만, 산출물은 별도 폴더에 저장합니다.

### 핵심 기능

- RMTPP와 TitanTPP를 100 epoch 이상 장기 학습
- `best_score` checkpoint와 `best_val_nll` checkpoint를 분리 저장
- `final` checkpoint도 함께 저장하여 overfitting 여부 확인 가능
- epoch별 train loss, validation NLL, score, qty MAE learning curve 저장
- true quantity 기준 scale bucket별 MAE/WAPE/median AE 저장
- 논문 및 미팅용 CSV, Markdown, plot 자동 저장

### Scale-wise MAE 정의

scale bucket은 예측 mark가 아니라 실제 quantity 기준으로 계산합니다.

```text
scale_order = floor(log10(true_qty))
```

기본 bucket은 아래와 같습니다.

- `0`: 1-9
- `1`: 10-99
- `2`: 100-999
- `3`: 1000-9999
- `4`: 10000+

이렇게 나누면 전체 MAE가 큰 수요 구간에 의해 얼마나 왜곡되는지, 그리고 작은
수요 구간에서 RMTPP와 TitanTPP 중 어느 쪽이 안정적인지 분리해서 볼 수 있습니다.

### 실행 예시

기본 설정은 `intermittent`, `yellow_trip`, seed `42,52,62`, epoch `100`입니다.

```bash
python simple_lab_test/search/titan_rmtpp_long_epoch_scale_eval.py
```

먼저 빠르게 smoke test만 하고 싶다면 dataset, seed, epoch를 줄여 실행합니다.

```bash
python simple_lab_test/search/titan_rmtpp_long_epoch_scale_eval.py \
  --datasets intermittent \
  --seeds 42 \
  --epochs 3 \
  --intermittent-max-series 200
```

교수님 피드백에 맞춰 더 충분히 확인하려면 epoch를 늘립니다.

```bash
python simple_lab_test/search/titan_rmtpp_long_epoch_scale_eval.py \
  --epochs 200 \
  --seeds 42,52,62
```

### 저장 산출물

권장 확인 순서는 아래와 같습니다.

- `search_artifacts/titan_rmtpp_long_epoch_scale_eval/paper_outputs/long_epoch_scale_report.md`
- `search_artifacts/titan_rmtpp_long_epoch_scale_eval/paper_outputs/paper_table_long_epoch_metrics.csv`
- `search_artifacts/titan_rmtpp_long_epoch_scale_eval/paper_outputs/paper_table_scale_wise_mae.csv`
- `search_artifacts/titan_rmtpp_long_epoch_scale_eval/paper_outputs/plots/*learning_curves.png`
- `search_artifacts/titan_rmtpp_long_epoch_scale_eval/paper_outputs/plots/*scale_wise_qty_errors.png`

### 해석 기준

long-epoch 실험에서는 final epoch보다 `best_val_nll`을 1차 기준으로 봅니다.
교수님이 말씀하신 sweet spot은 validation NLL이 가장 낮은 epoch이기 때문입니다.

확인해야 할 질문은 아래 세 가지입니다.

- validation NLL이 충분한 epoch 이후 plateau 또는 상승으로 전환되는가
- `best_val_nll_epoch`가 30보다 훨씬 뒤라면 기존 30 epoch 결과가 under-training이었는가
- scale-wise MAE에서 전체 MAE가 특정 큰 scale bucket에 의해 지배되는가

## 학습 가능성 확인: overfitting diagnostic

### 왜 별도 실험이 필요한가

교수님 피드백의 핵심은 “성능이 좋다/나쁘다” 이전에 모델이 데이터를 제대로
학습하고 있는지 먼저 확인해야 한다는 점입니다. 모델 capacity가 충분하고
learning rate가 공격적으로 설정되어 있다면, 적어도 일부 설정에서는 train loss가
명확히 내려가고 validation NLL이 어느 시점 이후 나빠지는 overfitting 패턴이
나와야 합니다.

따라서 이 실험은 논문 성능 비교용이 아니라 **학습 가능성 검증용 stress test**입니다.

### 새 실험 스크립트

```text
simple_lab_test/search/tpp_overfit_diagnostic.py
```

### 기본 설정

- learning rate: `1e-3`
- model: `RMTPP`, `TitanTPP`
- RMTPP:
  - `rnn_type`: `rnn`, `gru`, `lstm`
  - `rnn_hidden_dim`: `64`, `128`, `256`
  - `mark_emb_dim`: `32`
- TitanTPP:
  - Titan preset: `small_lmm`, `mid_lmm`, `mid_deep_lmm`
  - `mark_emb_dim`: `32`
- `max_seq_len`: 기본 `64`, 필요 시 `32,64,128,256` 등으로 확장

### 실행 예시

먼저 작은 subset에서 빠르게 “과적합이 가능한지” 확인합니다.

```bash
python simple_lab_test/search/tpp_overfit_diagnostic.py \
  --datasets intermittent \
  --models rmtpp,titantpp \
  --epochs 80 \
  --lr 1e-3 \
  --seeds 42 \
  --intermittent-max-series 300 \
  --max-seq-lens 64 \
  --force-rerun
```

RMTPP의 RNN cell과 sequence length를 더 넓게 확인하려면 아래처럼 실행합니다.

```bash
python simple_lab_test/search/tpp_overfit_diagnostic.py \
  --datasets intermittent,yellow_trip \
  --models rmtpp \
  --epochs 100 \
  --lr 1e-3 \
  --rmtpp-rnn-types rnn,gru,lstm \
  --rmtpp-hidden-dims 64,128,256 \
  --rmtpp-mark-emb-dims 32,64 \
  --max-seq-lens 64,128
```

TitanTPP의 `max_seq_len`과 Titan preset을 확인하려면 아래처럼 실행합니다.

```bash
python simple_lab_test/search/tpp_overfit_diagnostic.py \
  --datasets intermittent,yellow_trip \
  --models titantpp \
  --epochs 100 \
  --lr 1e-3 \
  --titan-candidates small_lmm,mid_lmm,mid_deep_lmm \
  --titan-mark-emb-dims 32,64 \
  --max-seq-lens 64,128,256
```

### Yellow Trip 후속 preset

기존 100 epoch 결과에서 `yellow_trip / RMTPP`는 validation NLL이 final epoch까지
계속 개선되어 명확한 overfitting이 나타나지 않았습니다. 이 경우 intermittent
결과는 그대로 두고, yellow_trip만 별도 폴더에 follow-up으로 돌립니다.

Full yellow-trip 데이터를 더 오래 학습시켜 overfitting이 늦게 나오는지 보려면
아래 preset을 사용합니다.

```bash
python simple_lab_test/search/tpp_overfit_diagnostic.py \
  --preset yellow_trip_full_long
```

이 preset은 기본적으로 아래 설정을 사용합니다.

- dataset: `yellow_trip`
- epochs: `300`
- lr: `1e-3`
- max_seq_len: `250`
- RMTPP: `rnn,gru,lstm`, hidden `64,128,256`, mark embedding `32,64`
- TitanTPP: `small_lmm,mid_lmm,mid_deep_lmm`, mark embedding `32,64`
- output: `search_artifacts/tpp_overfit_yellow_trip_full_long/`

더 빠르게 과적합 신호를 만들기 위해 yellow_trip series 수를 줄인 stress test는
아래 preset을 사용합니다.

```bash
python simple_lab_test/search/tpp_overfit_diagnostic.py \
  --preset yellow_trip_subset_stress
```

이 preset은 기본적으로 아래 설정을 사용합니다.

- dataset: `yellow_trip`
- `yellow_max_series`: `120`
- epochs: `200`
- lr: `1e-3`
- max_seq_len: `64,128,250`
- RMTPP: `gru,lstm`, hidden `128,256`, mark embedding `32,64`
- TitanTPP: `mid_lmm,mid_deep_lmm`, mark embedding `32,64`
- output: `search_artifacts/tpp_overfit_yellow_trip_subset_stress/`

Preset을 쓰더라도 명시적으로 넘긴 옵션은 우선합니다. 예를 들어 subset 수만
바꾸고 싶으면 아래처럼 실행합니다.

```bash
python simple_lab_test/search/tpp_overfit_diagnostic.py \
  --preset yellow_trip_subset_stress \
  --yellow-max-series 80
```

### 저장 산출물

```text
search_artifacts/tpp_overfit_diagnostic/
```

주요 확인 파일은 아래와 같습니다.

- `paper_outputs/overfit_diagnostic_report.md`
- `paper_outputs/paper_table_overfit_summary.csv`
- `paper_outputs/plots/*_overfit_curves.png`
- `leaderboard/overfit_runs.csv`
- `leaderboard/overfit_histories.csv`

### 해석 기준

이 실험에서는 “validation metric이 가장 좋은가”보다 아래를 먼저 봅니다.

- train loss가 충분히 내려가는가
- validation NLL의 best epoch이 final epoch보다 앞에 있는가
- `final_val_nll - best_val_nll`이 양수로 커지는가
- RMTPP에서 `rnn`, `gru`, `lstm` 중 어떤 cell이 가장 잘 학습/과적합되는가
- TitanTPP에서 `max_seq_len`과 Titan preset을 키웠을 때 train loss 감소가 더 명확해지는가

즉, overfit이 관찰되면 “모델이 학습 자체를 못 하는 것이 아니라 capacity/regularization/
selection 문제”로 해석할 수 있습니다. 반대로 어떤 설정에서도 train loss가 충분히
내려가지 않는다면 모델 입력, loss, time scaling, sequence 구성 자체를 다시 봐야 합니다.

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
