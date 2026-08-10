# TitanTPP manuscript v0.6 revision notes

이 문서는 이창희 교수님 피드백이 v0.6 원고에 어떻게 반영되었는지와 제출 전 남은 근거를 기록한다.

## Introduction

- 첫 문장을 slow-moving item, spare part, long-tail retail product의 불규칙한 수요 상황으로 시작했다.
- classical TPP의 hand-designed history effect에서 deep neural TPP의 learned history representation으로 이동한 뒤 RMTPP를 소개하도록 흐름을 바꿨다.
- RMTPP 기반 수요 모델링의 도전 요소를 세 가지로 정리했다.
  1. 긴 이력이 하나의 recurrent state를 거쳐야 하는 표현 제약
  2. categorical mark와 continuous quantity 사이의 출력 공간 불일치
  3. rare large quantity가 만드는 raw-regression scale과 shared-objective gradient conflict
- TitanTPP를 contribution보다 먼저 정의하고, 각 구성 요소가 위 문제와 어떻게 연결되는지 설명했다.
- 결과 주장은 Taxi의 명확한 개선과 Instacart의 제한적인 quantity-MAE 개선으로 구분했다.

## Method

- Problem Setup에서 long-tail quantity를 raw regression 하나로 학습할 때 생기는 scale 문제를 먼저 설명했다.
- magnitude mark와 residual의 정의, tail-class clipping, original-scale reconstruction 관계를 명시했다.
- Titan-inspired causal memory-attention, persistent learned memory, static learned-memory module을 구분했다.
- 원본 Titans의 test-time learning을 구현한 모델로 오해하지 않도록 경계를 적었다.
- mark probability, RMTPP time density, mark-conditioned residual, expected quantity를 모두 수식으로 추가했다.
- mark CE, time NLL, residual Huber, quantity Huber와 최종 hybrid objective를 구현에 맞춰 정리했다.
- Taxi의 quantity-to-mark stop-gradient가 forward prediction을 바꾸지 않고 backward path만 분리한다는 점을 명시했다.
- Figure 1은 단순 블록 목록 대신 observed event sequence, tokenization, history encoding, three prediction paths, quantity reconstruction을 한 흐름으로 보여주는 현재 SVG/PNG를 사용했다.

## Experiments

- Intermittent는 메인 원고에서 제거했다. 기존 11-mark 결과와 cap5 sensitivity는 현재 주장을 강화하지 못하므로 main result로 사용하지 않는다.
- 완료된 Taxi와 Instacart e300 결과만 사용했다.
- RMTPP-Q와 THP-Q라는 내부 명칭을 제거하고 Adapted RMTPP와 Adapted THP로 변경했다.
- 두 baseline이 원 논문의 그대로인 모델이 아니라 demand quantity interface를 공유하도록 수정한 variant임을 설명했다.
- Taxi와 Instacart 결과를 별도 표로 나눴다.
- aggregate table을 반복하던 기존 Figure 2는 삭제했다.
- Figure 2의 역할을 quantity-tail stratified error로 재정의했다. Train split에서 얻은 p50, p90, p95, p99 경계만 사용해야 한다.
- 기존 모델 비교를 ablation이라고 부르지 않았다. 새 ablation은 quantity interface만 바꾸는 실험으로 정의했다.

## Format and references

- Abstract와 keywords를 포함한 IEEE conference paper 순서로 재구성했다.
- 최종 제출본은 공식 ICTC IEEE two-column template로 옮기고 4-page 제한 안에서 다시 편집해야 한다.
- active reference는 출판사나 공식 proceedings에서 확인한 12개만 남겼다.
- Daley and Vere-Jones의 2nd-edition Springer 식별자를 `10.1007/b97277`로 수정했다.
- Box-Cox의 article page를 공식 Oxford Academic metadata에 맞춰 211-243으로 수정했다.
- 용도가 약했던 factorial marked TPP와 mixed-type event reference는 active manuscript에서 제외했다.

## Evidence boundary

- 현재 표는 fixed chronological split의 validation 결과다.
- 모든 값은 seeds 42, 52, 62의 mean and sample standard deviation이다.
- checkpoint는 validation mark NLL plus time NLL의 합이 가장 작은 epoch다.
- held-out test는 사용하지 않았다.
- Taxi는 TitanTPP가 validation NLL, quantity MAE, mark accuracy에서 우세하지만 delta-time MAE는 Adapted RMTPP가 더 좋다.
- Instacart는 TitanTPP의 quantity MAE만 가장 낮고, Adapted RMTPP가 validation NLL, delta-time MAE, mark accuracy에서 우세하다.
- 따라서 모든 데이터셋과 모든 지표에서 TitanTPP가 우세하다는 문장은 사용할 수 없다.

## Submission blockers

1. **Quantity-interface ablation**
   - 동일 RMTPP encoder로 uniform categorical binning, train-quantile binning, direct raw regression, mark-residual을 비교해야 한다.
   - fixed split, seeds 42/52/62, e300 budget, best-validation-NLL checkpoint rule을 고정한다.

2. **Long-tail Figure 2**
   - aggregate bar chart를 다시 만들지 않는다.
   - validation events를 train-derived p50/p90/p95/p99 quantity strata로 나누고 strata별 MAE와 event count를 표시한다.
   - held-out test에서 quantile boundary를 추정하지 않는다.

3. **Final evaluation**
   - 모델, ablation, figure 설계가 모두 고정된 뒤 selected configuration의 held-out test를 한 번만 평가한다.

4. **IEEE two-column conversion**
   - v0.6 Markdown을 공식 ICTC template의 LaTeX source로 옮긴다.
   - Figure 1은 SVG/PDF vector source를 우선 사용하고, 표와 reference를 포함해 4-page fit을 확인한다.
