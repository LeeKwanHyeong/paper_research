# TitanTPP 논문 Draft

Notion page: https://app.notion.com/p/3b4bbe4056138126b2fecc26979f3f77

## Local working files

- `paper/drafts/introduction_v0_1.md`
- `paper/drafts/problem_formulation_v0_1.md`
- `paper/drafts/methodology_v0_1.md`
- `paper/drafts/related_work_outline_v0_1.md`
- `paper/drafts/manuscript_section_plan_v0_1.md`

## 1. Abstract plan

1. 연구 문제: 간헐 수요 예측은 다음 positive-demand event의 발생 시점과 수요량을 함께 예측하는 문제이다.
2. 기존 TPP의 표현 한계: 일반적인 categorical mark는 유한한 event type을 전제로 하므로 연속적이고 heavy-tailed한 수요량 표현에 적합하지 않다.
3. RMTPP와 단순 회귀 확장의 한계: RMTPP는 긴 사건 이력을 recurrent state에 순차적으로 압축한다. Raw quantity regression은 수량의 scale 차이와 tail events에 민감하다.
4. 제안 방법: TitanTPP는 log-magnitude mark와 continuous residual의 분해, differentiable quantity decoder, causal memory encoder를 결합한 quantity-aware TPP이다.
5. 평가 범위: Intermittent, Taxi, Instacart에서 RMTPP/THP 비교와 history-length 및 quantity-scale breakdown을 수행한다.
6. 핵심 결과와 결론: 최종 Abstract에는 대표 성능 결과와 breakdown 결과가 포함되며, 결론은 효과가 확인된 데이터셋과 지표의 범위로 제한된다.

## 2. Introduction plan

Original Notion version used six paragraphs:

1. Problem background
2. Existing neural TPP and RMTPP
3. Research gap
4. Proposed method
5. Evaluation scope
6. Contributions

After professor feedback, the preferred 4-page version compresses this to three paragraphs:

1. Problem and challenge
2. Limitation of existing TPP modeling and TitanTPP contribution
3. Dataset and experimental summary

## 3. Related Work

Sections:

- Recurrent neural temporal point processes
- Attention- and memory-based history encoders
- Event marks and continuous quantity
- Intermittent-demand forecasting as event prediction
- Position of this work and baseline terminology

Key terminology:

- `RMTPP-matched` and `THP-matched` are adapted baselines under the paper's residual quantity input, quantity decoder, loss, and evaluation interface.
- They are not claims of exact reproduction of the original papers.
- TitanTPP superiority is judged only from matched experiments, not from the Related Work section.

## 4. Methodology

### 4.1 Problem setup

Positive-demand event: `e_i = (t_i, q_i)`.

The history `H_i`, inter-event time `Delta t_i`, next event time, and next quantity are defined in this setup. Zero-demand intervals are excluded as explicit events but preserved through `Delta t_i`.

### 4.2 Quantity-aware mark formulation

Core equations:

```tex
m_i = min(floor(log_b q_i), M)
r_i = log_b q_i - m_i
q_i = b^{m_i + r_i}
```

`m_i` is a coarse magnitude class and `r_i` is a continuous within-class residual.

### 4.3 Conditional event model

Core definitions:

```tex
p_{i,k} = P(m_{i+1}=k | H_i)
lambda(tau | h_i) = exp(a_i + w tau)
```

Final manuscript must align these symbols with implementation details.

### 4.4 Matched encoders and TitanTPP

- RMTPP-matched: recurrent state encoder.
- THP-matched: attention-based history encoder.
- TitanTPP: causal memory-attention block with persistent memory/static LMM.
- Validation and test do not update memory.
- State is not passed across unrelated demand series.

### 4.5 Prediction heads and objective

Expected quantity reconstruction:

```tex
qhat_{i+1} = sum_k p_{i,k} b^{k + rhat_{i,k}}
```

Training objective:

```tex
L = L_mark + lambda_t L_time + lambda_r L_res + lambda_q L_qty
```

V2 uses shared residual head and coupled gradient. Taxi V3b uses mark-conditioned residual experts and detached quantity-to-mark gate.

## 5. Experiments

Research questions:

- RQ1: Under the same quantity contract, how does TitanTPP compare with RMTPP-matched and THP-matched?
- RQ2: How do history length and quantity scale change model differences?
- RQ3: On Taxi, how does V3b expert/detached design change mark and quantity results?
- RQ4: What are the compute cost and convergence properties?

Training/evaluation contract:

- Seeds 42, 52, 62.
- Strict reproducibility.
- Fixed chronological split.
- Start with e300.
- Continue to e800 only if validation convergence requires it.
- Select checkpoint by minimum validation total NLL.
- Keep held-out test locked until configuration is frozen.

## 6. Ablation

- Quantity interface: RMTPP-original vs RMTPP-matched. This comparison changes multiple elements and should not be over-interpreted.
- Encoder comparison: RMTPP-matched, THP-matched, TitanTPP V2.
- Taxi V2 vs V3b: expert/detached design effect, currently interpreted as a combined design effect unless a full 2x2 ablation is run.

## Current update

The e300 matched baseline result page now changes how this draft should phrase experiments:

- RMTPP/THP e300 rows are final-ready validation baselines.
- Existing TitanTPP rows are draft-only preliminary evidence.
- Taxi gives the strongest quantity result.
- Instacart is mixed and should not be used for strong superiority claims yet.
