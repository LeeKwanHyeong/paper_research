# TitanTPP manuscript section plan v0.1

> 목적: 본문을 길게 쓰기 전에 Abstract, Introduction, Related Work, Methodology,
> Experiments, Ablation에 들어갈 내용과 증거를 고정한다. 최종 성능 수치는 strict validation 실행과
> held-out test가 끝난 뒤 채운다.

## 1. Abstract

1. **연구 문제**
   - 간헐 수요 예측은 다음 positive-demand event의 발생 시점과 수요량을 함께 예측하는 문제이다.
2. **기존 TPP의 표현 한계**
   - 일반적인 categorical mark는 유한한 event type을 전제로 하므로 연속적이고
     heavy-tailed한 수요량 표현에 적합하지 않다.
3. **RMTPP와 단순 회귀 확장의 한계**
   - RMTPP는 긴 사건 이력을 recurrent state에 순차적으로 압축하므로 먼 과거 정보에 대한
     직접 접근이 제한된다.
   - Raw quantity regression은 수량의 scale 차이와 tail events에 민감해 categorical mark의
     한계를 충분히 보완하지 못한다.
4. **제안 방법**
   - TitanTPP는 log-magnitude mark와 continuous residual의 분해, differentiable quantity
     decoder, causal memory encoder를 결합한 quantity-aware TPP이다.
5. **평가 범위**
   - 평가 범위는 Intermittent, Taxi, Instacart에서의 RMTPP·THP 비교와 history-length 및
     quantity-scale breakdown이다.
6. **핵심 결과와 결론**
   - 최종 Abstract에는 대표 성능 결과와 breakdown 결과가 포함되며, 결론은 효과가 확인된
     데이터셋과 지표의 범위로 제한된다.

## 2. Introduction

Introduction은 여섯 문단으로 구성한다.

1. **문제 배경:** zero-heavy regular series를 positive-demand event sequence로 바꾸면
   대기 시간과 발생량을 직접 모델링할 수 있음을 설명한다.
2. **기존 접근:** neural TPP와 RMTPP가 event time과 mark를 함께 다루는 출발점임을 소개한다.
3. **연구 공백:** recurrent state에 긴 이력을 순차적으로 압축하는 점과 categorical mark만으로
   연속 수량의 구간 내 차이를 보존하기 어렵다는 점을 구분해 설명한다.
4. **제안 방법:** log-magnitude mark와 continuous residual의 결합, Titan encoder, differentiable
   quantity reconstruction을 짧게 소개한다.
5. **검증 범위:** 서로 다른 sequence length와 quantity tail을 가진 세 데이터셋, matched
   baselines, sequence/quantity breakdown을 소개한다.
6. **기여:** quantity-aware event formulation, TitanTPP architecture, fixed-split matched
   evaluation의 세 가지로 정리한다.

RMTPP의 long-history 한계는 설계 동기로 표현하고, 실험 전에는 경험적 실패로 단정하지 않는다.
주요 근거 문헌은 RMTPP, THP, recurrent long-context modeling, continuous-mark TPP,
intermittent-demand forecasting으로 구성한다.

## 3. Related Work

Related Work는 네 가지 연구 흐름과 본 연구의 위치를 연결하는 다섯 소절로 구성한다.

### 3.1 Recurrent neural temporal point processes

- RMTPP는 사건 이력을 recurrent state로 요약해 다음 사건의 시간과 mark를 함께 예측하는
  초기 neural marked TPP에 해당한다.
- Neural Hawkes Process는 continuous-time LSTM을 이용해 사건 이력이 event intensity에
  미치는 영향을 확장한 recurrent TPP에 해당한다.
- 두 연구는 recurrent history encoder의 출발점으로 사용하며, 긴 이력에서의 성능 저하는
  선행연구의 결론으로 단정하지 않고 본 연구의 history-length breakdown에서 확인한다.

### 3.2 Attention- and memory-based history encoders

- SAHP와 THP는 self-attention과 시간 정보를 결합해 과거 사건의 영향을 표현한 attention 기반
  TPP에 해당한다. 이 가운데 THP는 본 연구의 직접 비교군이다.
- Titans는 recurrent state와 제한된 attention context를 보완하기 위한 장기 메모리 설계의
  배경으로 사용한다.
- TitanTPP는 Titans에서 영향을 받은 구조이지만 원 논문의 test-time learning을 구현한 모델은
  아니다. 현재 frozen model은 causal memory attention, learnable persistent memory, static LMM을
  사용하며 validation과 test 중에는 memory를 갱신하지 않는다.

### 3.3 Event marks and continuous quantity

- 일반적인 marked TPP의 mark space는 범주형에 한정되지 않지만, 다수의 neural TPP 구현과
  benchmark는 유한한 event type을 예측 대상으로 사용한다.
- Factorial marked TPP는 하나의 사건을 여러 개의 discrete marker로 분해하며, mixed-type event
  model은 discrete attribute와 continuous attribute에 서로 다른 prediction head를 사용한다.
- 본 연구의 log-magnitude mark와 continuous residual 분해는 이 문제 흐름 위에 놓이지만,
  수량을 정확히 복원하기 위한 구체적인 factorization과 decoder는 본 연구의 방법이다.

### 3.4 Intermittent-demand forecasting as event prediction

- Croston 계열 접근은 간헐적 수요의 발생 간격과 수요 크기를 분리해 추정하는 고전적 출발점이다.
- Deep Renewal Processes는 비영 수요의 도착 간격과 크기를 확률적으로 함께 모델링하며,
  간헐적 수요를 renewal 또는 point-process 관점으로 연결한다.
- 본 연구는 다음 positive-demand event의 시간과 수량을 동시에 예측하고, 동일한 event
  formulation을 서로 다른 수요 특성을 가진 세 데이터셋에 적용한다.

### 3.5 Position of this work and baseline terminology

- `RMTPP-matched`와 `THP-matched`는 원 논문의 결과를 그대로 재현한 이름이 아니다. 각 원 논문의
  history encoder를 유지하면서 본 연구의 residual quantity input, quantity decoder, loss와 평가
  interface를 맞춘 adapted baseline이다.
- Intermittent와 Instacart의 TitanTPP V2는 두 matched baseline과 shared residual head 및 coupled
  quantity objective를 공유한다.
- Taxi의 순수 encoder 비교는 TitanTPP V2 control을 기준으로 한다. Taxi primary인 V3b는
  mark-conditioned experts와 detached quantity-to-mark gradient를 사용하므로, encoder와 head 설계의
  결합 효과로 해석한다.

Related Work 본문에서는 각 연구의 역할과 차이만 정리한다. TitanTPP의 우위와 데이터셋별 효과는
Experiments와 Ablation의 동일 조건 결과에서만 판단한다.

## 4. Methodology

### 4.1 Problem setup

Positive-demand event를 `e_i=(t_i,q_i)`로 두고, 관측 이력 `H_i`, inter-event time
`Delta t_i`, 다음 사건의 time과 quantity 예측 문제를 정의한다. zero-demand interval은
명시적 사건에서 제외되지만 `Delta t_i`에 보존된다는 점을 설명한다.

### 4.2 Quantity-aware mark formulation

다음 세 수식을 핵심으로 사용한다.

```text
m_i = min(floor(log_b q_i), M)
r_i = log_b q_i - m_i
q_i = b^(m_i + r_i)
```

`m_i`는 coarse magnitude class, `r_i`는 class 내부의 연속적 위치를 나타낸다. tail mark가
clipping되어도 residual을 1보다 크게 허용해 수량을 정확히 복원할 수 있음을 밝힌다.

### 4.3 Conditional event model

다음 mark probability와 RMTPP time intensity를 정의한다.

```text
p_i,k = P(m_(i+1)=k | H_i)
lambda(tau | h_i) = exp(a_i + w tau)
```

본문에는 conditional time density와 time NLL까지 제시하고, 기호가 구현과 일치하는지 최종
점검한다.

### 4.4 Matched encoders and TitanTPP

모든 비교군이 split, seed, epoch policy, checkpoint rule을 공유한다는 점을 먼저 명시한다.
Intermittent와 Instacart에서는 RMTPP-matched, THP-matched, TitanTPP V2가 residual quantity input,
shared head, hybrid quantity objective를 공유한다. Taxi의 순수 encoder 비교에는 같은 head를 사용하는
TitanTPP V2 control을 사용하고, V3b는 expert head와 detached gate를 포함한 최종 설계로 구분한다.
이후 GRU recurrent state, THP attention, TitanTPP의 causal memory-attention block을 차례로 설명한다.
TitanTPP에서는 persistent memory와 static Local Memory Matching의 역할을 구분하고,
validation/test 중 memory update가 없음을 밝힌다.

### 4.5 Prediction heads and objective

mark, time, residual head와 expected quantity reconstruction을 설명한다.

```text
q_hat_(i+1) = sum_k p_i,k * b^(k + r_hat_i,k)
L = L_mark + lambda_t L_time + lambda_r L_res + lambda_q L_qty
```

V2는 shared residual head와 coupled gradient를 사용한다. Taxi V3b는 mark-conditioned residual
experts를 사용하고 quantity loss에서 mark probability gate를 detach한다. 이 변경은 forward
quantity estimate를 바꾸지 않으면서 quantity loss가 mark logits를 직접 갱신하지 않도록 한다.

### 4.6 사용할 그림과 자료

- **F1 Problem formulation:** regular observations에서 event sequence로의 변환과
  mark/residual 분해 및 복원을 보여준다. Methodology의 problem setup 뒤에 배치한다.
- **F2 TitanTPP architecture:** RMTPP-matched와 TitanTPP의 encoder 차이, 공통 heads,
  differentiable decoder, Taxi V3b의 expert와 stop-gradient 경로를 보여준다.
- **T2 Model and training contract:** 모델별 encoder, input, objective, parameter 수를 요약한다.
  자세한 설정은 Experiments 또는 appendix로 이동한다.
- **구현 근거:** frozen source revision, model-source hash, loss/checkpoint contract를 appendix와
  artifact manifest에 연결한다.

## 5. Experiments

### 5.1 연구 질문

- **RQ1:** 동일한 quantity contract에서 TitanTPP V2가 RMTPP-matched와 THP-matched에 비해 어떤
  차이를 보이는가? Taxi에서는 V2 control을 이 비교에 사용한다.
- **RQ2:** history length와 quantity scale이 달라질 때 모델 간 차이가 어떻게 변하는가?
- **RQ3:** Taxi에서 V3b의 expert/detached 설계가 V2 대비 mark와 quantity 결과를 어떻게 바꾸는가?
- **RQ4:** 개선이 있다면 계산 비용과 수렴 시간은 어느 정도인가?

### 5.2 데이터와 비교군

- Intermittent, Taxi, Instacart를 사용한다. T1에는 fixed-split 통계와 dataset hash를,
  F3에는 sequence length와 positive quantity의 분포를 제시한다.
- primary comparison은 RMTPP-original, RMTPP-matched, THP-matched, dataset별 TitanTPP primary로
  구성한다.
- Taxi V2 control은 primary baseline이 아니라 encoder-matched comparison과 V3b ablation에
  사용하는 control이다.

### 5.3 학습 및 평가 계약

- seeds 42, 52, 62, strict reproducibility, fixed chronological split을 사용한다.
- 모든 모델은 e300에서 시작하고 사전에 정한 validation convergence rule을 만족한 dataset만
  e800까지 이어서 학습한다.
- checkpoint는 minimum validation total NLL로 선택한다.
- 개발 중에는 validation-only를 유지하고, 모델과 epoch 정책을 고정한 뒤 held-out test를
  한 번만 평가한다.

### 5.4 지표와 결과물

- likelihood: total, mark, time NLL
- event type: accuracy, macro F1, balanced accuracy
- time: inter-event-time MAE
- quantity: MAE, RMSE, WAPE, 필요 시 log-quantity MAE
- robustness: history-length 및 quantity-scale bucket별 오차와 sample count
- efficiency: parameter 수, sec/epoch, peak GPU memory, best epoch, wall-clock time

최종 결과에는 **T3 main comparison**, **F4 main comparison**, **T5/F5 breakdown**,
**T6 efficiency summary**를 사용한다. 현재 T1, T2, F1, F2, F3는 준비됐고, 성능 기반 표와
그림은 strict run 완료 후 만든다.

## 6. Ablation

### 6.1 Quantity interface

RMTPP-original과 RMTPP-matched를 비교해 residual input과 hybrid quantity objective를 함께
도입했을 때의 변화를 본다. 두 요소가 동시에 달라지므로 이 비교만으로 각각의 효과를 분리해
주장하지 않는다.

### 6.2 Encoder comparison

RMTPP-matched, THP-matched, TitanTPP를 동일 quantity contract 아래 비교한다. 전체 결과와
history-length breakdown을 함께 사용해 encoder 차이가 어느 구간에서 나타나는지 확인한다.
Taxi에서는 V3b가 아니라 shared/coupled head를 사용하는 TitanTPP V2 control을 사용한다.

### 6.3 Taxi V2 versus V3b

동일 Titan encoder에서 V2 shared/coupled head와 V3b expert/detached head를 비교한다.
**T4**에는 NLL, mark, time, quantity 결과를, **F6**에는 marker와 quantity의 trade-off를 제시한다.
현재 계약으로는 expert head와 gradient detachment의 결합 효과만 확인할 수 있다.

두 구성요소의 효과를 따로 주장해야 한다면 아래 2x2 실험을 추가한다.

- shared head + coupled gate: V2
- expert head + coupled gate
- shared head + detached gate
- expert head + detached gate: V3b

시간이 부족하면 2x2 실험은 appendix 또는 후속 실험으로 미루고, 본문에서는 V3b combined
design의 효과로만 표현한다. ablation 설정도 held-out test를 열기 전에 고정한다.

## 현재 자산과 남은 제작물

| 구분 | 사용 가능 | 실험 완료 후 제작 |
| :--- | :--- | :--- |
| 표 | T1 dataset statistics, T2 model/training contract | T3 main results, T4 ablation, T5 breakdown, T6 efficiency |
| 그림 | F1 problem formulation, F2 architecture, F3 distributions | F4 main comparison, F5 breakdown, F6 V2/V3b trade-off |

이 문서는 section별 역할과 증거를 정한 구성안이다. 실제 영문 본문은 이 순서를 기준으로 쓰되,
정량 결과 문장은 frozen artifact가 준비된 뒤 채운다.
