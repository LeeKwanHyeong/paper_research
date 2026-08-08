# TitanTPP: quantity-aware temporal point process modeling for intermittent demand

> Draft version: v0.1  
> Date: 2026-08-08 KST  
> Target: ICTC AICA 2026 short-paper draft  
> Status: August 14 working draft. This version contains validation-only preliminary TitanTPP evidence. Held-out test results are not evaluated or reported.

## Abstract

Intermittent demand forecasting can be formulated as a sequence of positive-demand events, where a model predicts both the next event time and the associated demand quantity. This formulation differs from standard marked temporal point process settings because demand quantity is continuous, skewed, and often heavy-tailed. Recurrent marked temporal point processes such as RMTPP summarize event history through a fixed-dimensional recurrent state, and their categorical mark interface does not directly preserve within-class quantity variation. This paper proposes TitanTPP, a quantity-aware temporal point process architecture that combines long-history sequence encoding with a transformed quantity target and a differentiable quantity reconstruction path. The method represents demand quantity through a coarse magnitude mark and a continuous residual, then predicts event time, magnitude class, and within-class quantity residual jointly. We evaluate the approach on Intermittent, Taxi, and Instacart demand-event datasets using fixed chronological splits and three random seeds. Preliminary validation results indicate that TitanTPP yields lower validation NLL on Intermittent and strong quantity-error reductions on Taxi, while Instacart remains mixed under the current draft-only TitanTPP artifacts. Final fair comparison requires fresh TitanTPP e300 runs under the same frozen contract as the completed RMTPP and THP baselines.

## 1. Introduction

Many demand forecasting problems contain long intervals with no observed demand, followed by irregular positive-demand events. When these observations are forced into a regular time series, the model must represent both the waiting time and the nonzero demand amount indirectly. An event-based formulation makes this structure explicit. In this view, each positive-demand occurrence has a time stamp and a quantity, and the forecasting task asks when the next demand event will occur and how large it will be. This setting is particularly difficult when event histories are long, event times are irregular, and quantities vary across several scales.

Neural temporal point processes provide a natural starting point for this problem because they model event times and marks conditional on event history. RMTPP embeds history into a recurrent hidden state and predicts the next event time and mark from that state [1]. For demand-event sequences, however, this design raises three issues. First, a fixed-dimensional recurrent state may compress long histories in a way that weakens access to older demand patterns. Second, a categorical mark can represent a demand scale or regime, but it cannot reconstruct continuous variation within that class without an additional decoder. Third, adding a quantity regression head to a shared event representation may couple the gradients of likelihood and quantity objectives. TitanTPP addresses these issues by combining a Titan-style history encoder with a transformed quantity representation and a quantity-specific reconstruction path.

This study examines the resulting model on three datasets with different event-history and quantity characteristics. Intermittent demand contains many short part-level demand sequences, Taxi contains long grid-cell event histories with heavy-tailed pickup counts, and Instacart contains many user-level basket sequences. We compare RMTPP-matched, THP-matched, and TitanTPP variants under fixed split and validation-only conditions. Because the current TitanTPP results were produced under earlier e50/e200 budgets, they are treated as preliminary evidence rather than final fair-comparison rows. The draft therefore makes a bounded claim: TitanTPP appears promising for joint event-demand modeling, especially for Taxi quantity prediction, but final claims require fresh TitanTPP e300 runs under the same frozen contract.

## 2. Related work

### 2.1 Neural temporal point processes

Classical temporal point processes model the conditional intensity of event arrivals, while neural temporal point processes replace hand-crafted history effects with learned sequence representations. RMTPP is a representative recurrent marked temporal point process that embeds event history into a vector and predicts the next event time and mark [1]. Neural Hawkes Process extends this recurrent line with continuous-time LSTM dynamics [2]. These models motivate the present task formulation, but they also illustrate the standard recurrent-history interface against which TitanTPP is compared.

Attention-based temporal point process models were developed partly to improve how event histories are represented. Self-Attentive Hawkes Process and Transformer Hawkes Process employ self-attention with temporal encodings to represent dependencies among prior events [3,4]. THP is therefore a useful comparison model for TitanTPP because it replaces the recurrent encoder with attention while retaining a neural TPP objective. More recent memory architectures, including Titans, further motivate the study of long-history representation beyond fixed recurrent states and bounded attention contexts [5]. TitanTPP should be interpreted as a TPP architecture influenced by this memory-oriented design space, not as an implementation of every mechanism in the original Titans paper.

### 2.2 Marks, quantities, and intermittent demand

Marked TPPs often treat the mark as a finite event type, although the general formulation can allow richer event attributes. Reviews of neural TPPs and work on factorial marked TPPs provide useful background for decomposing event attributes beyond a single categorical label [6,7]. Mixed-type event sequence models further support the distinction between categorical and continuous attributes [8]. In the present problem, the mark must describe the scale of a demand event, while a continuous residual must preserve within-scale quantity variation.

Intermittent-demand forecasting has long separated demand occurrence from demand size. Croston's method estimates demand interval and nonzero demand size separately [9], and deep renewal process models extend this idea with probabilistic neural sequence modeling [10]. This paper follows the same broad motivation but formulates the problem as next-event prediction with an explicit event time and a continuous demand quantity. The main difference is that the proposed model is evaluated within a marked TPP interface rather than a regular zero-filled forecasting interface.

## 3. Method

### 3.1 Problem setup

For each demand series, let a positive-demand event be denoted by \(e_i=(t_i,q_i)\), where \(t_i\) is the event time and \(q_i>0\) is the observed demand quantity. The observed history before the next event is

\[
\mathcal{H}_i=\{(t_j,q_j)\}_{j=1}^{i}.
\]

The inter-event time is \(\Delta t_i=t_i-t_{i-1}\). Given \(\mathcal{H}_i\), the model predicts the next inter-event time \(\Delta t_{i+1}\) and the next quantity \(q_{i+1}\). Zero-demand intervals are not represented as explicit events, but their duration is retained through the inter-event time. This event-based construction is used for all three datasets.

Demand quantities can span several orders of magnitude. To reduce the burden on a single raw regression target, TitanTPP represents quantity through a log-magnitude class and a continuous residual. For a dataset-specific base \(b\), define

\[
m_i = \min(\lfloor \log_b q_i \rfloor, M),
\]

\[
r_i = \log_b q_i - m_i,
\]

where \(m_i\) is a coarse magnitude mark and \(r_i\) is the within-class residual. Quantity is reconstructed as

\[
q_i = b^{m_i+r_i}.
\]

This factorization lets the classifier represent demand scale while the residual decoder preserves continuous variation inside each scale. When the tail class is clipped at \(M\), the residual may exceed the unit interval, which allows large quantities to remain reconstructable.

### 3.2 Limitations of RMTPP-style modeling

RMTPP encodes the event history through a recurrent state. This is a compact and useful representation, but it may lose detail when the sequence contains long-range patterns and recent local changes. The limitation is architectural rather than absolute. For this reason, the paper treats long-history weakness as a design motivation and evaluates it empirically through datasets with different sequence-length distributions.

The mark interface introduces a second mismatch. A categorical event type can be predicted with a softmax head, but a demand quantity contains both scale and within-scale variation. A mark-only model discards this residual variation, while a raw regression model can be dominated by scale differences and tail events. The proposed mark-residual representation separates these two requirements.

A third issue arises during joint learning. Time likelihood, magnitude classification, and quantity reconstruction are not identical objectives. If all objectives update the same representation in the same way, quantity error can alter the representation used for event-time and mark likelihood. TitanTPP therefore separates the quantity prediction path where the dataset and architecture require it. In Taxi V3b, for example, mark-conditioned residual experts and a detached quantity-to-mark path are adopted as a combined design.

### 3.3 TitanTPP architecture

TitanTPP follows the same conditional event modeling interface as the matched baselines. Given an event history, an encoder produces a history representation \(h_i\). The model then predicts the mark distribution \(p_{i,k}=P(m_{i+1}=k \mid \mathcal{H}_i)\), the inter-event-time distribution, and a residual quantity estimate. The expected quantity prediction is computed through the differentiable reconstruction

\[
\widehat{q}_{i+1}=\sum_{k=0}^{M} p_{i,k} b^{k+\widehat{r}_{i,k}}.
\]

The training objective combines mark, time, residual, and quantity losses:

\[
\mathcal{L}=\mathcal{L}_{\mathrm{mark}}+\lambda_t\mathcal{L}_{\mathrm{time}}+\lambda_r\mathcal{L}_{\mathrm{res}}+\lambda_q\mathcal{L}_{\mathrm{qty}}.
\]

The baseline encoders differ only in the history representation module under the matched contract. RMTPP-matched uses a GRU encoder, THP-matched uses a Transformer encoder, and TitanTPP uses a Titan-style causal memory-attention encoder with persistent memory. Intermittent and Instacart currently use TitanTPP V2, which employs a shared residual head and coupled gradient flow. Taxi uses TitanTPP V3b as the current primary model, where mark-conditioned residual experts and detached quantity-to-mark gradients form the final draft architecture. Validation and test do not update memory, and state is not transferred across unrelated demand series.

## 4. Experiments

### 4.1 Datasets

The evaluation uses three fixed-split demand-event datasets. Intermittent contains 23,387 part-level sequences and 242,888 positive-demand events. Taxi contains 131 grid-cell sequences and 55,119 pickup events, with a median sequence length of 405 and a p95 length of 743. Instacart contains 206,209 user sequences and 3,279,521 order events. These datasets differ substantially in both event-history length and quantity scale, which makes them suitable for separating recurrent-history, attention-history, and quantity-modeling effects.

| Dataset | Sequences | Events | Seq. length med/p95/max | Quantity med/p95/max | Marks | Base |
|---|---:|---:|---:|---:|---:|---:|
| Intermittent | 23,387 | 242,888 | 6 / 35 / 110 | 2 / 16 / 5,000 | 11 | 2 |
| Taxi | 131 | 55,119 | 405 / 743 / 744 | 7 / 1,547 / 6,489 | 4 | 10 |
| Instacart | 206,209 | 3,279,521 | 10 / 50 / 100 | 8 / 25 / 177 | 8 | 2 |

The chronological split is applied within each sequence. Development uses validation only, and held-out test results remain locked until model identity, continuation policy, and checkpoint selection are fixed.

### 4.2 Baselines and protocol

RMTPP-matched and THP-matched are adapted baselines. They retain the original encoder family of each model, but they share the paper's residual quantity input, hybrid objective, output interface, fixed split, and checkpoint rule. This naming avoids presenting the runs as exact reproductions of the original RMTPP or THP papers. The comparison isolates the encoder family more cleanly than an unadapted baseline would, although RMTPP-original remains useful for future ablation.

All frozen e300 baseline runs use seeds 42, 52, and 62, AdamW with learning rate 0.001, batch size 128, strict reproducibility mode, and minimum validation total NLL as the checkpoint rule. The completed RMTPP/THP baseline queue contains 18 successful runs. The TitanTPP rows reported in this draft come from earlier e50/e200 artifacts and are therefore marked as draft-only. They support paper direction and August 14 writing, but they do not yet qualify for the final fair-comparison table.

### 4.3 Preliminary validation results

Table 2 reports the current validation-only comparison. Lower values are better for validation NLL, quantity MAE, and \(\Delta t\) MAE; higher values are better for mark accuracy. TitanTPP rows are preliminary because their epoch budgets and artifact contracts differ from the frozen e300 baseline contract.

| Dataset | Model | Budget | Val NLL | Qty MAE | Delta-t MAE | Mark acc |
|---|---|---:|---:|---:|---:|---:|
| Intermittent | RMTPP-matched | e300 | 5.6683 +/- 0.0115 | 2.7408 +/- 0.0493 | 41.8872 +/- 0.5030 | 55.183% +/- 0.236%p |
| Intermittent | THP-matched | e300 | 5.6417 +/- 0.0305 | 2.8812 +/- 0.0177 | 40.5947 +/- 0.3284 | 54.235% +/- 0.637%p |
| Intermittent | TitanTPP V2 | e200 draft | 5.6046 +/- 0.0097 | 2.7162 +/- 0.0720 | 41.1990 +/- 0.4479 | 54.697% +/- 0.577%p |
| Taxi | RMTPP-matched | e300 | 1.5803 +/- 0.0032 | 65.8580 +/- 2.4748 | 0.7326 +/- 0.0085 | 91.800% +/- 0.117%p |
| Taxi | THP-matched | e300 | 1.5998 +/- 0.0087 | 87.7508 +/- 2.6771 | 0.7528 +/- 0.0224 | 91.461% +/- 0.202%p |
| Taxi | TitanTPP V3b | e50 draft | 1.5555 +/- 0.0019 | 31.0775 +/- 2.8184 | 0.7591 +/- 0.0206 | 92.267% +/- 0.194%p |
| Instacart | RMTPP-matched | e300 | 4.3809 +/- 0.0007 | 4.3379 +/- 0.0131 | 5.6690 +/- 0.0094 | 49.940% +/- 0.034%p |
| Instacart | THP-matched | e300 | 4.3881 +/- 0.0009 | 4.3046 +/- 0.0081 | 5.7063 +/- 0.0059 | 49.793% +/- 0.091%p |
| Instacart | TitanTPP V2 | e200 draft | 4.3819 +/- 0.0009 | 4.3199 +/- 0.0084 | 5.6768 +/- 0.0129 | 49.941% +/- 0.027%p |

On Intermittent, TitanTPP V2 has lower validation NLL than RMTPP-matched and THP-matched. The quantity result is weaker: TitanTPP improves over THP but only slightly improves over RMTPP. This pattern suggests that the joint likelihood and time-distribution components may benefit from the TitanTPP formulation, while the quantity claim remains modest.

Taxi provides the strongest preliminary result. TitanTPP V3b reduces quantity MAE by 52.8% relative to RMTPP-matched and by 64.6% relative to THP-matched. It also obtains lower validation NLL and higher mark accuracy than both baselines. By contrast, its \(\Delta t\) MAE is slightly worse than the RMTPP value, so the result should be interpreted as a quantity and mark improvement with a small time-error trade-off rather than as universal metric superiority.

Instacart remains inconclusive. TitanTPP V2 is close to RMTPP-matched in validation NLL and lies between RMTPP and THP in quantity MAE. The current evidence therefore does not support a strong Instacart claim. Fresh TitanTPP e300 runs are required before this dataset can be included in a final claim.

![Validation NLL comparison](results/e300_matched_20260808/figures/validation_nll_comparison.png)

![Quantity MAE comparison](results/e300_matched_20260808/figures/quantity_mae_comparison.png)

### 4.4 Ablation and analysis plan

The final version will separate three effects. First, RMTPP-original and RMTPP-matched will measure the effect of the quantity interface, although this comparison changes multiple factors and must be interpreted cautiously. Second, RMTPP-matched, THP-matched, and TitanTPP V2 will compare encoder families under the same quantity contract. Third, Taxi V2 and V3b will examine the combined effect of mark-conditioned experts and detached quantity-to-mark gradients. If time permits, a full 2x2 ablation should separate expert heads from gradient detachment; otherwise, the paper will describe V3b as a combined design.

## 5. Conclusion

This draft formulates intermittent demand forecasting as a marked temporal point process with continuous quantity reconstruction. The formulation exposes two limitations of common neural TPP interfaces: recurrent history compression can be restrictive for long event histories, and categorical marks alone cannot reconstruct within-class quantity variation. TitanTPP addresses these issues by combining a Titan-style history encoder with a log-magnitude mark, a continuous residual, and a differentiable quantity decoder.

The current validation evidence supports a cautious interpretation. RMTPP-matched and THP-matched are now available as final-ready e300 validation baselines, and existing TitanTPP artifacts provide preliminary support on Intermittent and Taxi. The Taxi result is the most substantial because quantity MAE decreases sharply relative to both baselines. Instacart remains mixed. The next experimental step is therefore fixed: run TitanTPP V3b on Taxi and TitanTPP V2 on Intermittent and Instacart under the same frozen e300 contract, then regenerate the final validation table before opening the held-out test.

## References

[1] N. Du, H. Dai, R. Trivedi, U. Upadhyay, M. R. Gomez-Rodriguez, and L. Song, "Recurrent Marked Temporal Point Processes: Embedding Event History to Vector," KDD, 2016.

[2] H. Mei and J. Eisner, "The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process," NeurIPS, 2017.

[3] Q. Zhang, A. Lipani, O. Kirnap, and E. Yilmaz, "Self-Attentive Hawkes Process," ICML, 2020.

[4] S. Zuo, H. Jiang, Z. Li, T. Zhao, and H. Zha, "Transformer Hawkes Process," ICML, 2020.

[5] A. Behrouz, P. Zhong, and V. Mirrokni, "Titans: Learning to Memorize at Test Time," NeurIPS, 2025.

[6] O. Shchur, A. C. Türkmen, T. Januschowski, and S. Günnemann, "Neural Temporal Point Processes: A Review," IJCAI, 2021.

[7] Y. Wu, N. Li, R. B. Silva, and A. Smola, "Decoupled Learning for Factorial Marked Temporal Point Processes," KDD, 2018.

[8] F. Draxler, A. G. Baydin, and F. Wood, "Transformers for Mixed-type Event Sequences," NeurIPS, 2025.

[9] J. D. Croston, "Forecasting and Stock Control for Intermittent Demands," Operational Research Quarterly, 1972.

[10] A. C. Türkmen, Y. Wang, and T. Januschowski, "Forecasting intermittent and sparse time series: A unified probabilistic framework via deep renewal processes," PLOS ONE, 2021.

## Draft notes

### Claim-evidence map

| Claim | Evidence in this draft | Status |
|---|---|---|
| Event-based demand formulation is suitable for intermittent demand | Croston, Deep Renewal Processes, T1 dataset construction | Supported as formulation |
| RMTPP recurrent history compression motivates a long-history encoder | RMTPP, Neural Hawkes, THP, Titans, Taxi long sequence statistics | Supported as design motivation |
| Categorical marks alone do not reconstruct continuous demand quantity | Neural TPP review, mixed-type event work, mark-residual formulation | Supported as method motivation |
| TitanTPP improves Taxi quantity prediction | Preliminary Taxi validation table | Preliminary only |
| TitanTPP improves all datasets and metrics | Current results contain counterexamples | Not supported |

### Items to revise after fresh TitanTPP e300

- Replace draft-only TitanTPP rows with frozen e300 rows.
- Regenerate validation NLL and quantity MAE figures.
- Add convergence/best-epoch analysis for e800 continuation.
- Add final held-out test table only after model identity and epoch policy are frozen.
