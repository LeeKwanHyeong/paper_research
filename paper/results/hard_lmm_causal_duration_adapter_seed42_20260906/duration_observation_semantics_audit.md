# Duration observation semantics audit

## 판정

Taxi와 Instacart의 `delta_t`는 단일 사건의 연속 interarrival을 반올림해 저장한 값이 아니다. 두 데이터셋 모두 전처리된 **양의 정수 bucket gap**이다. 따라서 기존 continuous density NLL은 TPP 호환 점수로 유지할 수 있지만, 정수 관측에 맞는 primary likelihood를 주장하려면 합이 1인 positive-integer PMF를 별도로 정의해야 한다.

실행 당시 사용한 centered-bin 계산

\[
P(\max(0.5,d-0.5)<T\le d+0.5)
\]

은 `d=1,2,...`에 대해 합이 `S(0.5)`다. `F(0.5)`가 빠지므로 정규화된 PMF 또는 proper observation likelihood가 아니다. 예를 들어 `mu=0`, `sigma=1`인 log-normal에서는 합이 `0.755891`이고 `0.244109`가 누락된다. 기존 값은 legacy 민감도 점수로만 사용한다.

## Taxi 생성 과정

[`yellow_trip.ipynb`](../../../simple_lab_test/notebooks/preprocessing/yellow_trip.ipynb)은 pickup timestamp를 한 시간 단위로 truncate하고, grid cell과 hour별 positive pickup count를 한 event row로 합친다. 전체 시간 bucket의 순번 `seq`를 만든 뒤 같은 cell 안의 `seq` 차이를 `delta_t`로 저장한다.

- Pinned 기간: 2015-01-01 00시부터 2015-01-31 23시까지 연속 744시간
- 첫 sentinel `delta_t=0`: 131건으로 series 수와 동일
- 학습 target이 될 수 있는 nonfirst gap: 54,988건, 모두 정수 `1..115`
- Validation target: 8,268건, 이 중 `delta_t=1`은 6,790건(`82.12%`)

한 row가 여러 pickup을 합친 active-hour event이므로 `delta_t`는 개별 trip 사이의 반올림된 연속 시간이 아니라 occupied-hour bucket 사이의 이산 gap이다.

## Instacart 생성 과정

[`insta_market_basket.ipynb`](../../../simple_lab_test/notebooks/preprocessing/insta_market_basket.ipynb)은 user별 integer `days_since_prior_order`를 누적하고 `day_bucket`을 만든다. 같은 user와 day bucket의 주문을 active-day row 하나로 합친 뒤 day bucket 차이를 `delta_t`로 저장한다.

- 첫 sentinel `delta_t=0`: 206,209건으로 user series 수와 동일
- Nonfirst derived gap: 3,073,312건, 모두 정수 `1..30`
- Derived `delta_t=30`은 모두 raw code 30에서 옴
- Raw code 0부터 29까지는 요일 차이와 정확히 일치하지만, raw code 30은 요일 차이가 거의 균등하다. 이는 30이 exact 30일보다 `D>=30` top code라는 공급자 정의와 일치한다.
- Validation target: 503,733건, 이 중 code 30은 63,923건

공급자 data dictionary도 `days_since_prior_order`를 30에서 cap한다고 설명한다. 다만 연속 timestamp에서 integer day를 계산한 반올림 규칙은 공개하지 않는다. 따라서 원자료가 직접 지지하는 likelihood는 `d=1..29`의 discrete mass와 `d=30`의 `P(D>=30)`이다. `S(29.5)` 같은 연속 경계는 별도 quantizer convention을 선언할 때만 사용할 수 있다.

## Loader의 target support

[`event_seq_data_module.py`](../../../data_loader/event_seq_data_module.py)은 저장된 `delta_t`를 최소 1로 clip하므로 first sentinel도 context에서는 1이 된다. 하지만 target은 항상 series의 다음 row에서 만들기 때문에 first sentinel 0은 target이 되지 않는다. 고정된 target support는 Taxi와 Instacart 모두 양의 정수다.

## 사용할 수 있는 정규화 계약

### 직접 discrete PMF 또는 hazard

관측 의미에 가장 충실하다.

- Taxi: `p(D=d | H)`, `d>=1`
- Instacart: `p(C=d | H)=p(D=d | H)` for `d=1..29`, `p(C=30 | H)=P(D>=30 | H)`

시간 단위별 support를 다루는 공통 hazard network는 만들 수 있지만 기존 continuous TPP Time NLL과 같은 수치로 직접 비교할 수 없다.

### Positive-integer discretized continuous distribution

연속 TPP 형태를 유지하려면 `D=max(1, round(T))`를 명시적인 모델 convention으로 둔다.

\[
\begin{aligned}
p(D=1)&=F(1.5),\\
p(D=d)&=F(d+0.5)-F(d-0.5),\quad d\ge2,\\
p(C=30)&=S(29.5)\quad\text{for Instacart.}
\end{aligned}
\]

이 정의는 positive-integer support에서 합이 1이고 K=2 LogNormMix의 CDF로 정확히 계산할 수 있다. 전처리에서 확인된 반올림 규칙이 아니므로 논문에서 latent round-and-clamp observation model이라고 명시해야 한다.

대안은 기존 centered bin을 유지하면서 모든 mass와 top survival을 `S(0.5)`로 나누는 zero-truncated model이다. 두 정의를 결과를 본 뒤 선택해서는 안 된다.

## 다음 구현 전 필수 테스트

1. 단일 log-normal과 K=2 mixture에서 모든 유한 bin과 마지막 tail의 합이 float64 허용오차 안에서 1이어야 한다.
2. Instacart code 30은 선택한 model에 맞춰 `P(D>=30)` 또는 `S(29.5)` 하나로 고정한다.
3. 첫 bin과 top-code 경계의 density, CDF, survival gradient가 finite여야 한다.
4. Continuous NLL과 positive-integer NLL은 같은 checkpoint에서 별도 열로 보고하며 서로 직접 차감하지 않는다.
5. 기존 legacy centered-bin 결과는 새 positive-integer NLL로 이름을 바꾸지 않고 재계산 전 성능 근거로 사용하지 않는다.
