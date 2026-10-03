# Architecture Recommendation: 저비용 Hard-LMM 고도화

## Problem

현재 논문 최종 Backbone인 T0 Hard-LMM은 64개 정적 prototype에서 cosine top-4를 선택해 산술평균 residual을 마지막 encoder state에 더한다. 구조는 저렴하지만 prototype은 현재 series의 관측으로 갱신되지 않고, 같은 top-4 안에서는 query에 따른 memory 출력 변화가 없다.

B1 Faithful Titans-MAC은 test-time neural-memory update를 도입했지만, 사건마다 2층 fast-weight와 momentum을 갱신하고 그 경로를 역전파한다. 현재 5090의 동일 synthetic input 측정에서 이력 길이 16·64·255일 때 B1의 steady step은 T0 대비 각각 4.36·12.27·17.38배였고, prior-prefix 후보는 각각 6.76·17.09·22.13배였다. 이 비용은 compiler 최적화만으로 제거되지 않았다.

따라서 결정할 문제는 원본 Titans를 더 충실하게 재현하는 것이 아니라, T0의 비용 수준을 유지하면서 현재 관측 이력을 memory 출력에 반영할 최소 구조를 고르는 것이다.

## Constraints and Quality Attributes

- 세 데이터셋에 같은 계산 그래프와 학습 규칙을 사용한다. Dataset ID 분기를 두지 않는다.
- T0의 encoder, persistent token, static prototype residual, time/quantity head, loss와 checkpoint 선택을 유지한다.
- 미래 target과 padding을 memory에 쓰지 않는다.
- 초기 출력은 T0와 일치하고, 추가 경로가 유용하지 않으면 T0에 가까운 해를 학습할 수 있어야 한다.
- 사건별 neural-memory parameter gradient update와 긴 순차 계산 그래프를 금지한다.
- 목표 비용은 T0 step 1.5배 이하, peak memory 1.25배 이하, 추가 parameter 약 2% 이하로 둔다.
- 기존에 실패한 weighted top-k, 단순 separate-key, residual shrinkage gate, concat/MLP readout, elapsed-age bias, Surprise/FIFO gated memory를 이름만 바꿔 반복하지 않는다.

## Evidence

- Frozen T0에서 memory residual을 제거하면 Intermittent, Taxi, RAF, Instacart의 전체 MAE와 RMSE가 모두 악화됐다. 정적 memory를 제거하는 방향은 근거가 없다.
- Separate-key는 Taxi seed42에서 body MAE 26.77%, RMSE 34.23%를 개선했지만 Instacart body 개선은 0.05%에 그쳤다. Addressing 개선만으로 공통 성능을 확보하지 못했다.
- Instacart separate-key는 상위 네 prototype에 attention mass 99.53%가 집중됐고 평균 뒤 수량 방향 SD는 0.003228이었다. 다만 Taxi에서도 출력 분산이 줄면서 성능이 좋아졌으므로 다양성 자체는 목표가 될 수 없다.
- Weighted retrieval, smooth shrinkage, readout 확장, elapsed-age, B2 FIFO gated memory와 기존 Surprise/dual memory는 공통 gate를 통과하지 못했다.
- Instacart raw history에는 다음 수량 신호가 있지만 같은 용량 probe에서 encoder state가 raw history보다 더 잘 예측했다. 더 큰 encoder나 raw bypass를 우선 후보로 둘 근거가 없다.
- B1의 self-associative write는 다음 사건 예측과 직접 일치하지 않으며, segment read timing과 window reset 때문에 짧은 이력에서는 write가 예측에 보이지 않는 문제가 있었다.

## Options

### Option A: T0-anchored dual static retrieval

기존 tied top-4 residual을 기본 경로로 유지하고 separate-key residual과의 차이만 query-conditioned scalar로 더한다.

\[
z_i=h_i+r_i^{T0}+g_i\left(r_i^{KV}-r_i^{T0}\right).
\]

장점은 Taxi의 separate-key 양성 신호를 이용하면서 `g=0`으로 T0를 보존할 수 있다는 점이다. 두 static 검색을 병렬 계산하므로 B1보다 훨씬 저렴하다.

한계는 Instacart에서 separate-key 자체의 추가 이득이 거의 없었다는 점이다. 이 후보는 보존에는 유리하지만 세 데이터셋의 strict improvement를 만들 가능성은 제한적이다. 기존 soft/gated retrieval과 구분되는 계약도 필요하다.

### Option B: Prototype-conditioned transition-error memory

T0의 top-4 배정을 history 안의 관측된 예측오차를 모으는 bucket으로 재사용한다. 관측 `t-1`의 state에서 만든 T0 log-quantity 예측과 다음에 실제로 관측된 수량의 차이를 innovation으로 정의한다.

\[
I_t=Top4_j\cos(h_t,p_j),
\]

\[
e_t=\log(1+q_t)-\hat y_{t-1}^{T0},
\qquad
d_j=\frac{\sum_{t=2}^{H}\mathbf 1[j\in I_{t-1}]e_t}
{\epsilon+\sum_{t=2}^{H}\mathbf 1[j\in I_{t-1}]},
\qquad c_H=\frac14\sum_{j\in I_H}d_j,
\]

\[
\hat y_H^{adapt}=\hat y_H^{T0}+\alpha c_H.
\]

`alpha=0`으로 초기화해 T0 출력을 정확히 보존한다. H=1이면 관측된 전이가 없어 correction은 0이다. 첫 진단은 이 직접 수량 보정으로 transition-error signal의 존재만 판정한다. 신호가 확인된 뒤에만 같은 요약을 shared state에 주입하는 Backbone형 projection을 별도 후보로 계약한다.

이 설계는 현재 target을 쓰지 않는다. `e_t`는 다음 예측 시점에는 이미 결과가 관측된 과거 전이에서만 계산하고, 최종 query `h_H`가 예측하는 `q_(H+1)`은 집계에 포함하지 않는다. 기존 T0가 이미 모든 token의 top-4를 계산하므로 배정을 재사용하고 `scatter_add` 기반 합계와 count만 추가할 수 있다. 추가 계산은 대략 `O(BLk)`, 상태는 `O(BM)`이며 사건별 parameter update나 fast-weight 역전파 그래프가 없다.

장점은 raw 사건 평균보다 T0가 실제로 남긴 체계적 오차를 직접 겨냥한다는 점이다. H≥2부터 동작하므로 짧은 Instacart 이력에서도 adaptation 기회가 있다.

한계는 과거 예측오차가 현재 오차를 실제로 설명한다는 증거가 아직 없다는 점이다. Prototype별 평균에서 정보가 상쇄될 수 있고, top-4가 이력 전체에서 거의 같으면 단순 series 평균 오차로 퇴화한다. 따라서 constant·series 평균·last-error·prototype 없는 동일 용량 요약보다 나아야 한다. 기존 B2와 구분되는 핵심은 FIFO/gate가 아니라 이전 상태의 실제 next-event prediction error를 value로 쓴다는 점이다.

### Option C: Faithful Titans-MAC 또는 다른 fast-weight recurrence 유지

원본 Titans와의 구조적 연관성은 가장 강하다. 그러나 이미 비용 gate를 크게 넘었고, Surprise/dual/B2 계열의 공통 성능 실패도 있다. 전용 fused kernel을 개발해도 구조의 순차 inner update 자체는 남는다.

## Tradeoff Comparison

| Option | 공통 개선 가능성 | T0 보존 | 예상 비용 | 기존 실험 중복 | 판단 |
| --- | --- | --- | --- | --- | --- |
| A. Dual static retrieval | Taxi에는 근거, Instacart 개선 근거 약함 | 강함 | 낮음 | soft/gated retrieval과 일부 유사 | 보수적 대안 |
| B. Transition-error memory | 미검증이나 짧은 이력과 남은 예측오차를 직접 겨냥 | 강함 | 낮음 | B2와 value 정의가 다름 | 우선 검토 |
| C. Faithful fast-weight | 성능 미확정 | 약함 | 매우 높음 | 이미 다수 실행 | 채택 우선순위 낮음 |

## Recommendation

첫 후보는 Option B로 좁힌다. 다만 곧바로 e300을 시작하지 않는다. 먼저 train-only 분리 series에서 frozen T0 state와 top-4 trace를 사용해 prototype-conditioned past-error summary가 다음 수량 잔차를 설명하는지 확인한다.

다음 조건을 결과 열람 전에 동결한다.

1. 상수, 이력 길이, series 평균 오차, 마지막 오차, prototype을 쓰지 않은 동일 용량 error summary를 대조군으로 둔다.
2. 세 데이터셋에서 같은 rank, 집계법, optimizer와 fold 규칙을 사용한다.
3. 두 fold 모두에서 residual log-MSE가 개선되고 body MAE 방향이 충돌하지 않아야 한다.
4. 개선이 한 데이터셋에만 있으면 공통 Backbone 후보로 구현하지 않는다.
5. 통과한 경우에만 T0 identity initialization, H1/H2 인과성, padding 차단, gradient 개방, checkpoint 복원과 비용 gate를 검증한다.

## Risks

- Train-only probe가 통과해도 fresh end-to-end 학습과 validation 개선을 보장하지 않는다.
- Prototype bucket 평균이 오차의 다봉 분포를 뭉갤 수 있다.
- 첫 진단은 quantity correction이므로 Backbone 전체 개선 증거가 아니다. Shared state 주입은 신호 확인 뒤 별도 인과 실험이 필요하다.
- Instacart의 남은 오차가 수량 분포·loss 또는 추가 입력의 문제라면 Backbone 변경은 strict improvement를 만들지 못할 수 있다.
- `Hard-LMM`은 프로젝트의 Hard Local Memory Matcher이며 원본 Titans Long-term Memory Module로 표현하면 안 된다.

## Assumptions

- 현재 runner는 target window의 마지막 state로 다음 사건을 예측하며 모든 `x_2...x_H`는 관측 이력이다.
- T0가 계산하는 token별 top-4 trace를 추가 검색 없이 재사용할 수 있다.
- Prototype별 error 합계와 count가 GPU에서 vectorized scatter로 구현 가능하다.

## Validation Steps

1. 현재 B1 prior-prefix의 Taxi 결과를 별도 고비용 reference로 보존한다.
2. 로컬 train-only transition-signal 계약과 대조군을 동결한다.
3. 분리 series OOF probe를 세 데이터셋에서 실행한다.
4. 공통 gate 통과 시 별도 Backbone 경로를 구현하고 단위·계약 검증을 완료한다.
5. RTX 5090에서 T0 대비 step/peak 비용과 full-data e1을 검증한다.
6. 비용과 e1 통과 후에만 seed42 validation screening을 수행한다.
7. Seed42 공통 조건을 통과할 때만 추가 seed를 실행한다.
