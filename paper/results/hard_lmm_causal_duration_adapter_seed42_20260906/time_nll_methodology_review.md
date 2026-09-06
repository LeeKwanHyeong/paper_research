# Time NLL 개선 방법론 재검토

## 현재 증적이 허용하는 결론

Frozen-B causal scale adapter는 B의 encoder, memory, quantity path, log-normal location과 time median을 고정하고 관측 `log1p(delta_t)` prefix만 읽었다. Taxi에서 continuous-density NLL은 B의 `0.266558`에서 `0.264790`으로 `0.001769` 낮아졌지만, history-free global-scale control의 `0.264511`보다 높았고 matched A의 `-0.443556 + 0.01` 기준을 크게 넘었다. 이 가설은 기각하며 다른 dataset과 추가 seed는 실행하지 않는다.

실행 당시 centered-bin 계산은 양의 정수 support에서 확률 합이 `S(0.5)`이므로 proper observation likelihood가 아니다. Taxi와 Instacart의 target은 연속 interarrival를 반올림한 값이 아니라 occupied-hour와 active-day bucket의 양의 정수 차이다. 기존 centered-bin 값은 legacy telemetry로만 남기고 다음 후보의 학습·선택·채택에는 사용하지 않는다. 관측 생성 근거는 [`duration_observation_semantics_audit.md`](duration_observation_semantics_audit.md)에 기록했다.

같은 checkpoint를 두 정규화 convention으로 사후 재생하면 다음과 같다.

| 경로 | Fold-to-one NLL | Zero-truncated NLL |
|---|---:|---:|
| A | 0.811792 | 0.811328 |
| Frozen-B | 0.853267 | 0.862033 |
| Global scale | 0.871290 | 0.878591 |
| Causal scale adapter | 0.864470 | 0.872286 |

후보는 global control보다 각각 `0.006821`, `0.006304` 낮지만 B보다 각각 `0.011202`, `0.010254` 높다. 후보와 control은 continuous-density NLL로 서로 다른 epoch에서 선택됐고 둘 다 B보다 악화했으며 length-only control도 없다. 따라서 이 사후 차이를 duration 값의 조건부 효과로 해석하지 않는다. 이 재생은 목적함수와 selector를 정렬한 새 실험이 필요한 이유만 제공한다.

A/B 출력의 location과 scale을 같은 Taxi validation row에서 교차한 결과는 더 직접적이다.

| Location / scale | Continuous-density NLL | Fold-to-one NLL |
|---|---:|---:|
| A / A | -0.443556 | 0.811792 |
| B / B | 0.266558 | 0.853267 |
| A / B | 0.242048 | 0.839292 |
| B / A | -0.370101 | 0.815147 |

B location에 A scale을 넣으면 B-A 격차의 `89.66%`를 continuous-density NLL에서, `91.91%`를 fold-to-one NLL에서 회복한다. A location/B scale의 회복률은 각각 `3.45%`, `33.69%`다. 이 결과는 Taxi 격차가 **표본별 scale 출력 축과 강하게 연결됨**을 보여 주지만 A scale을 재현할 입력이나 모델을 식별하지는 않는다. 다음 실험은 location을 풀거나 큰 decoder로 바로 넘어가기보다 scale objective, selector와 표현을 먼저 정렬해야 한다.

## 방법 우선순위

| 순위 | 방법 | 근거와 역할 | 판단 |
|---:|---|---|---|
| 1 | **Objective-aligned K=1 causal scale adapter** | 현재와 같은 273-parameter GRU8 scale adapter를 정규화된 positive-integer likelihood로 학습·선택한다. 모델과 관측 목적의 효과를 섞지 않는 최소 재검증이다. | **다음 단일 후보** |
| 2 | **B-anchored first-bin hurdle + conditional tail scale** | Taxi의 `delta_t=1` 비율 `82.12%`와 scale 축 진단을 직접 다룬다. Zero residual에서 B 분포를 정확히 복원하면서 first-bin mass와 오른쪽 tail을 분리할 수 있다. | K=1 objective 정렬 실패 시 구조 후보 |
| 3 | `K=2` shared-`mu_B` conditional log-normal scale mixture | Continuous TPP 형태와 B median을 유지하며 scale shape를 확장한다. 초기 동일성과 component symmetry가 충돌한다. | Hurdle보다 후순위 |
| 4 | Direct positive-integer hazard 또는 PMF | 실제 bucket gap과 Instacart code 30의 `P(D>=30)`을 가장 직접적으로 모델링한다. | 관측 의미에는 가장 충실하나 continuous TPP 비교 틀 변경 필요 |
| 5 | Recurrent spline, flow 또는 monotone cumulative hazard | 비대칭·다봉성·tail을 가장 유연하게 표현한다. | 비용과 계약 복잡도가 커 앞 후보 실패 후 검토 |

[Shchur et al. (2020)](https://arxiv.org/abs/1909.12127)은 TPP를 conditional inter-event density estimation으로 구성하고 log-normal mixture가 유연성과 tractable sampling 및 moments를 함께 제공함을 보였다. 이는 순위 3의 근거다. [Omi et al. (2019)](https://proceedings.neurips.cc/paper/2019/hash/39e4973ba3321b80f37d9b55f63ed8b8-Abstract.html)은 단조 cumulative-hazard network로 복잡한 intensity의 likelihood를 계산하는 방법을 제시하며 순위 5의 대안이 된다.

이 데이터에서는 모델 표현력보다 관측 law가 먼저다. [Kvamme and Borgan (2021)](https://arxiv.org/abs/1910.06724)은 discrete-time survival을 PMF와 hazard로 구성하는 방식을 비교한다. Right-censored 또는 top-coded 값은 density가 아니라 survival probability로 처리해야 하며, right-censored likelihood의 평가 원칙은 [Rindt et al. (2022)](https://proceedings.mlr.press/v151/rindt22a.html), discretized survival scoring의 properness 조건은 [Yanagisawa (2023)](https://proceedings.mlr.press/v202/yanagisawa23a.html)이 뒷받침한다.

## 다음 단일 후보의 계약 초안

> **상태: 제안됨, 미구현, 미학습. 구현 전에 수치와 checksum을 별도 JSON 계약으로 동결한다.**

### 1. 비교 기준선 정렬

현재 A와 B duration head는 continuous-density NLL로 학습·선택됐다. 다음 후보는 normalized positive-integer NLL을 primary objective로 사용하므로 기존 A/B 값을 matched selector 결과처럼 비교하지 않는다.

1. Frozen A와 Frozen B hidden cache에 동일한 K=1 log-normal duration head를 붙인다.
2. 두 head 모두 아래 normalized observation likelihood로 학습하고 가장 이른 validation strict finite minimum을 선택한다.
3. 동일 epoch budget, optimizer, batch order, train-only time statistics와 censor/top-code 규칙을 사용한다.
4. 이렇게 얻은 `aligned-A-K1`과 `aligned-B-K1`을 다음 후보의 비교 기준선으로 고정한다.
5. 기존 A/B 값은 continuous-selected external reference로만 남긴다.

이 정렬 과정은 encoder나 quantity model을 재학습하지 않는다. 필요한 epoch checkpoint가 이미 없으면 duration head만 다시 적합하며, held-out test는 사용하지 않는다.

`aligned-B-K1` 자체가 기존 B보다 normalized NLL을 `0.005` 이상 낮추고, `aligned-A-K1 + 0.01` 기준과 기존 B 대비 continuous-density non-worsening을 모두 만족하면 더 단순한 time-head 정렬 결과를 우선 채택하고 causal adapter를 실행하지 않는다. Aligned B가 이 기준을 충족하지 못할 때만 adapter를 붙인다. 이렇게 해야 loss/selector 정렬만으로 해결되는 문제에 불필요한 recurrent module을 추가하지 않는다.

### 2. 관측 likelihood

연속 LogNormal decoder를 유지하기 위해 다음 latent observation convention을 결과를 보기 전에 고정한다.

\[
D=\max(1,\operatorname{round}(T)).
\]

따라서

\[
p(D=1)=F(1.5), \qquad
p(D=d)=F(d+0.5)-F(d-0.5),\ d\ge2.
\]

Instacart code 30은 `p(C=30)=S(29.5)`로 처리한다. 이 정의는 양의 정수 support에서 합이 1이지만 전처리에서 확인된 반올림 규칙이 아니라 명시적인 latent continuous model convention이다. 이 convention을 논문에서 받아들이지 않으면 K=1 adapter를 구현하지 않고 direct discrete hazard/PMF로 전환한다.

모든 interval mass는 float64에서 stable log-CDF 또는 log-survival difference로 계산한다. 단일·혼합 log-normal, 첫 bin, 마지막 tail과 사전 정의한 극단 파라미터에서 전체 질량 합이 `1`과 absolute/relative tolerance `1e-10` 안에서 일치해야 한다.

### 3. 모델과 frozen 경계

- Base: `aligned-B-K1`.
- Trainable candidate: 기존과 같은 1-layer GRU(`input_size=1`, `hidden_size=8`)와 zero-initialized scale projection, 총 273개 parameter.
- Input: train-only normalized active-prefix `log1p(delta_t)`.
- Excluded input: next target, padding, mark, quantity, B hidden state.
- Frozen: B encoder, memory, quantity head, duration location `mu_B`, time scale와 나머지 모든 B parameter.
- Scale: 기존 smooth `[0.1, 10]` ratio bound와 `sigma_floor=0.001`을 유지한다.
- Epoch 0: normalized likelihood, continuous density, location, scale, median과 quantity prediction이 aligned B와 정확히 일치해야 한다.

현재 후보와 architecture는 같지만 objective와 selector가 다르다. 따라서 결과는 기존 실패의 재해석이 아니라 **관측 목적 정렬 가설**의 별도 판정이다.

### 4. 필수 control

1. **History-free global scale:** 하나의 global scale residual만 학습한다.
2. **Same-capacity length-only GRU:** candidate와 동일한 GRU·projection·mask·sequence length를 사용하되 active token을 normalized feature space에서 정확히 `0`으로 고정한다. 이 경로는 duration 값 없이 recurrent step 수만 본다.

Candidate와 length-only control은 GRU와 projection의 초기 parameter digest, minibatch permutation, optimizer 설정을 정확히 공유한다. Candidate가 global control만 이기면 recurrent capacity와 prefix length의 효과를 배제할 수 없다. Duration 값의 추가 효과는 두 control을 모두 사전 margin 이상 이길 때만 인정한다.

### 5. 학습·선택·판정

- Primary train objective: mean normalized positive-integer/top-code NLL.
- Primary selector와 early stopping: 가장 이른 strict finite minimum validation normalized NLL.
- Secondary report: 같은 checkpoint의 continuous-density NLL under the latent continuous convention.
- Selector에서 제외하는 guardrail: B quantity metrics, B time median error, continuous-density NLL, `d=1`과 `d>1` 구간 NLL.
- 공통 규칙: 세 dataset에 같은 architecture, optimizer, likelihood, margin과 normalization을 사용하고 dataset별 lambda나 구조 분기를 두지 않는다.

Taxi seed 42에서 다음을 모두 만족할 때만 Intermittent와 Instacart seed 42로 확장한다.

1. Candidate normalized NLL `<= aligned-B-K1 - 0.005`.
2. Candidate normalized NLL `<= aligned-A-K1 + 0.01`.
3. Candidate normalized NLL `<= global control - 0.005`.
4. Candidate normalized NLL `<= length-only control - 0.005`.
5. Candidate continuous-density NLL `<= aligned-B-K1`.
6. Candidate continuous-density NLL `<= aligned-A-K1 + 0.01`.
7. B source state, base location, quantity prediction과 모든 quantity metric이 정확히 일치한다.
8. Scale-only 경로이므로 B time median과 median MAE/RMSE가 정확히 일치한다.
9. `d=1`과 `d>1` normalized NLL을 따로 보고하고 각 구간은 aligned B보다 `2%`를 초과해 악화하지 않는다.

Epoch 0을 selector 후보에 포함하므로 단순 `candidate <= B`는 개선 증거가 아니다. 그래서 새 후보에는 `0.005`의 strict improvement margin을 둔다. CUDA e1은 실행과 계약만 판정한다. Taxi full seed-42가 하나라도 실패하면 다른 dataset, 추가 seed와 held-out test를 중단한다.

## K=1 실패 뒤의 구조 후보

### B-anchored first-bin hurdle

Aligned B의 `q_B=F_B(1.5)`와 두 conditional density를 이용해 base distribution을 다음처럼 분해한다.

\[
p_B(t)=q_B p_B(t\mid t\le1.5)+(1-q_B)p_B(t\mid t>1.5).
\]

GRU8 adapter는 `logit(q)` residual과 오른쪽 conditional tail의 scale residual만 출력한다. 두 residual이 0이면 B density, CDF, survival과 integer PMF가 정확히 복원된다. 이 구조는 Taxi의 큰 first-bin mass와 긴 tail, Instacart의 top code를 따로 조정하며 mixture component symmetry가 없다. Quantity는 그대로 고정한다. Time median은 바뀔 수 있으므로 exact median identity 대신 aligned B 대비 time median MAE와 RMSE 악화 `2% 이하`를 guardrail로 둔다.

이 방법은 관측 형태에 가장 직접적이지만 표준 log-normal head보다 설명이 늘어난다. Objective-aligned K=1이 실패한 뒤에만 구현해, 목적 정렬과 구조 변경의 효과를 분리한다.

### K=2 shared-mu scale mixture

두 log-normal이 `mu_B`를 공유하면 B median을 정확히 유지할 수 있다. 그러나 서로 다른 scale을 가진 혼합은 단일 B log-normal과 모든 `t`에서 정확히 같을 수 없다. `pi=(0.75,0.25)`, scale residual `(epsilon,-3epsilon)`은 1차항만 상쇄하며 exact identity가 아니다. 따라서 `1e-12` 초기 출력 동일성과 의미 있는 symmetry break를 동시에 요구하지 않는다.

K=2까지 진행한다면 exact epoch-0 identity를 요구하지 않고 **approximate parity 한 방식만** 사용한다. `epsilon=1e-4`, `pi=(0.75,0.25)`를 고정하고, Float64 초기 audit에서 mean normalized/continuous NLL의 B 대비 차이 `<=1e-5`, 관측점 최대 `|delta log p|<=1e-3`, stress grid 최대 `|delta CDF|`와 `|delta S|<=1e-6`을 요구한다.

고정 첫 batch에서 각 scale/logit head의 gradient L2 `>=1e-8`, 첫 update 최대 parameter 변화 `>=1e-7`을 구현 전에 확인한다. 어느 하나라도 충족하지 못하면 epsilon이나 optimizer를 결과에 맞춰 바꾸지 않고 K=2 후보를 보류한다. 종료 시 각 component의 평균 posterior responsibility가 `0.01` 이상이고 median effective log-scale gap이 `1e-3` 이상일 때만 두 성분을 실제로 사용했다고 해석한다. 미달이면 effectively K=1로 기록한다.

## 보고와 주장 경계

- Continuous-density NLL과 normalized positive-integer NLL은 서로 다른 수치 척도이므로 직접 차감하거나 dataset 사이에서 합산하지 않는다.
- Candidate, global, length-only와 aligned A/B는 같은 primary selector의 동일 checkpoint에서 모든 지표를 보고한다.
- Time calibration은 first-bin probability, `d>1` conditional NLL, top-code survival과 randomized PIT 또는 discrete calibration curve로 함께 확인한다.
- [Bosser and Ben Taieb (2023)](https://arxiv.org/abs/2306.17066)이 강조하듯 history encoder와 decoder parametrization 및 calibration을 분리해 해석한다.
- Validation 결과는 후보 screening과 가설 선택에만 사용한다. 논문 성능 주장은 모든 validation gate와 추가 seed가 통과한 뒤 model-unevaluated held-out test에서만 확정한다. Intermittent는 사후 수동 점검에서 target marginal aggregate가 노출됐으므로 analyst-blind라고 주장하지 않는다.
- 이 경로가 성공해도 우선 기여는 **frozen quantity backbone 위의 common causal time adapter와 observation interface**다. Backbone 전체의 시간 표현 개선이라고 주장하려면 candidate가 global과 length-only control을 모두 이기고 세 dataset에서 같은 방향을 보여야 한다.

현재 완료된 실행 범위는 single-log-normal scale-only adapter의 Taxi seed 42와 validation-only 재감사까지다. Objective-aligned K=1, hurdle, K=2, 다른 dataset, 추가 seed와 held-out test는 실행하지 않았다.
