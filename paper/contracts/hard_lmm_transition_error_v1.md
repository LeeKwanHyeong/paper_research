# Hard-LMM transition-error 후보 계약

이 계약은 Backbone 기여를 우선해 `transition-error Hard-LMM` 한 후보만 검토한다. 먼저 frozen T0의 train 이력에서 필요한 신호가 있는지 확인하고, 공통 gate를 통과한 경우에만 모델을 구현하고 RTX 5090에서 e1과 e300을 실행한다. Validation 성능을 보기 전에 진단·모델·비용·성능 기준을 모두 고정한다.

## 인과적 전이 값

현재 target window 안에서 각 관측 사건 직전의 T0 state가 만든 수량 preactivation을 `a`라 둔다. 이미 관측된 다음 수량에 대한 오차만 다음처럼 만든다.

```text
e_t = log1p(q_t) - softplus(stop_gradient(a_(t-1)))
```

`e_t`는 이전 state가 선택한 T0 top-4 prototype에 쓴다. Prototype별 합과 횟수를 시간축으로 누적하고, 현재 state가 선택한 top-4 bucket의 과거 평균을 산술평균해 보정 신호를 만든다. 현재 target, padding, write 금지 위치는 합계와 횟수에 들어가지 않는다. 이력이 한 사건뿐이면 값은 0이다.

과거 값을 preactivation 잔차 `log(q)-a`로 정의하지 않는다. 현재 세 데이터셋의 train 수량은 양수지만, 실제 학습 손실과 같은 `log1p` 잔차를 사용하면 0 수량에서도 유한하고 진단과 모델의 정의가 하나로 유지된다.

## 구현 전 train-only 판정

Intermittent, Taxi, Instacart에서 각각 최대 8,192개의 train target을 seed 42로 미리 선택한다. Series hash로 두 fold를 나누며 같은 series의 window는 같은 fold에 둔다. Validation과 held-out test는 읽지 않는다.

Registry의 원래 Intermittent checkpoint 파일 경로는 현재 로컬 artifact에서 사라졌으므로, model-state SHA256이 registry 값과 정확히 일치하는 후속 e300 보존본을 사용한다. 대체 checkpoint·launch contract·summary의 파일 hash와 공통 state hash를 JSON 계약에 함께 고정한다.

고정 대조군에는 현재 T0 예측, 이력 길이, 전이 수, 현재 top-4의 관측 가능성, 최근 오차와 전체 과거 평균 오차를 넣는다. 후보는 여기에 prototype-conditioned 오차 하나를 더한다. Sham은 같은 합법적 prefix 안에서 과거 오차와 write prototype의 대응만 고정 순환 이동하고 실제 현재 top-4 read는 유지한다.

각 fold의 보정기는 반대 fold에서만 표준화하고 적합한다. 선형 보정을 quantity preactivation에 더한 뒤 `softplus`를 거친 실제 log 예측의 summed squared error와 계수 L2 penalty를 최소화한다. 표준 Ridge의 `alpha=1` 의미이며 intercept는 penalize하지 않는다. 후보 선택이나 hyperparameter 탐색은 하지 않는다.

후보가 구현 단계로 넘어가려면 세 데이터셋 각각에서 다음 조건을 모두 만족해야 한다.

- Strong control과 sham 각각보다 pooled log1p MSE를 1% 이상 개선한다.
- 두 fold 모두 두 비교의 개선 방향이 양수다.
- 세 데이터셋 × 두 비교의 여섯 paired series-bootstrap 하한이 모두 0보다 크다. 10,000회와 one-sided `0.05/6` 분위수를 고정한다.
- Train p95 이하 body raw MAE가 T0, strong control, sham보다 pooled와 각 fold에서 악화되지 않는다.
- 비유한 값, 누락 행, fold당 512개 미만 target 또는 30개 미만 series는 보정해서 사용하지 않고 판정 불가로 처리한다.

통과는 구현 필요조건일 뿐 validation 개선이나 논문 채택을 뜻하지 않는다. 실패하면 이 후보의 구현과 GPU 학습을 중단한다.

## 통과 시 모델 계약

공통 후보 이름은 `titantpp_transition_error_memory`다. 기존 encoder, persistent token, static 64-prototype bank, cosine top-4, 산술평균 residual, head, direct log-MSE, optimizer와 checkpoint 선택은 유지한다. 시간 예측은 T0 fused state를 그대로 쓴다. 수량 경로에만 transition summary와 zero-initialized hidden direction의 곱을 더한다.

새 방향 벡터는 0으로 초기화하므로 같은 seed의 T0와 초기 parameter, RNG, memory state, 시간·수량 출력이 일치해야 한다. 과거 오차를 만드는 T0 예측은 새 보정과 무관한 base branch에서 계산하고 detach한다. 따라서 후보 보정이 자기 과거 오차를 재귀적으로 바꾸지 않는다.

로컬 계약 검증 후 RTX 5090에서 CUDA와 full-data e1을 확인한다. e1은 실행 가능성과 계약만 판정한다. 이어서 seed42 e300 validation을 실행하고, 세 데이터셋 raw RMSE가 paired T0보다 모두 개선되며 body·tail·time guardrail을 지킬 때만 seeds 52·62를 실행한다. 기존 body MAE 5% 개선 기준은 공통 raw-RMSE 목표와 별도로 유지해 결과에서 구분한다. Held-out test는 이 계약의 승인 범위가 아니다.

## 후속 작업

Backbone 후보를 판정한 뒤 세 데이터셋의 raw 성능을 우선할 경우 **quantile-adaptive loss와 checkpoint 정렬**을 진행한다. 이 항목은 진행 보고마다 계속 명시한다.
