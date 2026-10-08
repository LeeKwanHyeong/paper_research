# Explicit history correction and structured prototype retrieval for quantity-aware event prediction

**Research working draft — 8 October 2026.** This AI-assisted document is for internal method and experiment review. It is not a submission-ready PAKDD manuscript: the current [PAKDD 2027 Author Kit](https://pakdd2027.org/pages/calls/authors-kit) limits generative-AI use to grammatical correction and requires disclosure. Author review alone does not establish that generated prose satisfies that policy. Existing manuscripts and frozen experiment contracts remain separate.

**Evidence cutoff.** The completed MLP8 Validation study and native implementation-parity checks motivate this draft. The latest local campaign record consulted was observed at 14:38:06–07 KST on 8 October: Instacart was in progress; the representative-structure correction and retrieval stages had not launched. This is a writing-time snapshot, not a new GPU observation. The new stages' outcomes are deliberately left open. Sources and claim boundaries are recorded in the accompanying [writing notes](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_contribution_writing_20261008_v1/WRITING_NOTES.md).

## 1. Research question and scope

An event-history encoder can summarize earlier observations without exposing an explicit path from earlier states to the representation used for prediction. We ask whether adding such a path improves the prediction of event quantities under a fixed data and training protocol. Our primary question is: **Does explicit local and recurrent correction provide an additional benefit within an encoder that already models event history?** This operational question refines the motivating question about explicit immediately preceding-state correction. The representative CNN+GRU54 implementation uses a three-position local filter and a recurrent state; it is broader than a single lag-one correction.

We study this question through three versions of the same representative architecture: the full correction, a version with that correction disabled, and a current-position-only correction with the same active parameter count. The last control distinguishes access to earlier positions through the correction path from an increase in trainable parameters. It does not remove history from the model: its current representation after the first attention block already depends on the observed prefix. Nor does equal parameter count imply equal depth, computation, or functional capacity.

A secondary question concerns retrieval: **Does separately normalizing time- and quantity-conditioned queries, then combining their scores with fixed weights, improve on a general learned query under the same information and parameter budgets?** All retrieval variants use a train-learned static prototype bank. We do not propose an online store of past events. The study separates uniform aggregation, similarity weighting, split-query scoring, and a budget-matched general query. Retrieval remains a secondary claim, contingent on its own performance and cost comparisons.

The intended contribution is therefore a controlled assessment of these specific design choices for joint quantity and recorded-gap prediction. We do not claim to introduce event-history modeling, residual connections, learned memory, or joint time–quantity prediction. A single representative specification will be used across the core datasets; choosing a different best TitanTPP variant for each dataset would not constitute the performance of one model.

## 2. Related methods and the comparison boundary

Recurrent models already encode history in a hidden vector. RMTPP updates that vector using event information and predicts subsequent events from it [1]. Deep renewal models are particularly relevant to our task: they use recurrent representations to model demand sizes and interdemand intervals, including continuous-time extensions [2]. Consequently, the presence of historical quantity and timing information is a starting point, not a novelty claim. Although our inputs do not contain categorical event-type identifiers, quantities can still be interpreted as marks in point-process terminology.

Attention-based models provide another history representation. A-NHP combines temporal queries with attention and includes residual layers [3]. S2P2 instead uses continuous-time state dynamics with event-driven impulses and also employs residual transformations [4]. These methods rule out a general claim that existing point-process models lack memory or residual connections. Our question is about the added value of the particular local Q/K/V correction and recurrent residual specified below.

Learned memory is also established. Persistent-memory attention introduces trainable, input-independent keys and values [5], while product-key memory supports sparse top-k selection and weighted value aggregation [6]. Our small static bank differs in its construction and use, but neither static slots nor top-k weighting alone establishes novelty. The secondary experiment tests the inductive bias of separate time/quantity query normalization under matched budgets. A broader search of multi-view retrieval remains necessary before any priority claim.

**S2P2 adaptation.** In the inspected project source, the S2P2 adapter receives continuous projections of observed log-transformed gaps and quantities, rather than categorical mark IDs. It reads the state immediately after the last observed event, at its right-hand limit, and connects that state to the common quantity and recorded-time heads. This is a history-encoder adaptation, not a reproduction of S2P2's native marked likelihood. Its local scan implementation also differs from the original work-efficient implementation, so the original complexity or speed claims cannot simply be transferred. New S2P2 fits are not included in the current 63-fit campaign. Historical external comparisons require their own source, population, head, training, selection, and Runtime audit before supporting a claim about the final representative model.

## 3. Prediction task and shared output contract

Let the observed prefix be \(\mathcal H_j=\{(\Delta_i,q_i)\}_{i\leq j}\), where \(\Delta_i\) is the recorded gap and \(q_i\) the observed quantity. Position \(j\) denotes the last observed event, and \(j+1\) the target. The prediction state is read at \(j\). TitanTPP masks the target's quantity and duration in its encoder path and does not write the withheld target into its history summaries. Padding is excluded from observation counts. No target quantity is available to the predictor.

The input embedding is

\[
e_i=W_{\mathrm{in}}[\log(1+\Delta_i),\log(1+q_i)]^\top+b_{\mathrm{in}}+p_i,
\]

where \(p_i\) is a learned positional embedding. These input features are not Train-z-scored; the standardized retrieval summaries in Section 5 are a separate operation. The shared encoder has two pre-normalized causal attention blocks, width 64, four heads, feedforward width 128, and dropout 0.1. Each block contains 16 learned persistent attention tokens. These tokens are distinct from the 64-slot prototype bank applied after the encoder.

Let \(z_j\) be the representation after correction and retrieval. Both prediction heads use this representation. The quantity head predicts

\[
\widehat\ell_{j+1}=\operatorname{softplus}(w_q^\top z_j+b_q),\qquad
\widehat q_{j+1}=\operatorname{expm1}(\widehat\ell_{j+1}),
\]

and minimizes squared error against \(\log(1+q_{j+1})\). Raw-unit evaluation applies the displayed exponential transform, without a further Train-mean/standard-deviation inverse transform.

For timing, \(\mu_j\) and \(\sigma_j=10^{-3}+\operatorname{softplus}(w_\sigma^\top z_j+b_\sigma)\) are affine-head location and positive scale parameters. A latent duration satisfies \(T=s\exp(\mu_j+\sigma_j\epsilon)\), with \(\epsilon\sim\mathcal N(0,1)\) and a frozen dataset-specific scale \(s\). The recorded observation is \(D=\max(1,\operatorname{round}(T))\). Thus the time score is a probability mass for recorded integers, not a continuous density evaluated at a rounded time:

\[
p_j(1)=\Phi\left(\frac{\log(1.5/s)-\mu_j}{\sigma_j}\right),
\]
\[
p_j(d)=\Phi\left(\frac{\log((d+0.5)/s)-\mu_j}{\sigma_j}\right)
-\Phi\left(\frac{\log((d-0.5)/s)-\mu_j}{\sigma_j}\right),\quad d\ge2.
\]

The three core datasets have no upper top-code. Instacart has top-code 30: its upper category uses the survival mass \(P(T\ge29.5)\). The per-target objective is

\[
\mathcal L_j=-\log p_j(D_{j+1})+
(\widehat\ell_{j+1}-\log(1+q_{j+1}))^2.
\]

The quantity coefficient is one and no tail-weighted loss is added. Training this objective does not guarantee improvement in every reported metric. In particular, a checkpoint selected for raw quantity RMSE may have worse time negative log-likelihood.

## 4. Explicit local and recurrent correction

### 4.1 Full CNN+GRU54 path

The first attention block projects its normalized input into Q, K, and V. For each projection \(X\in\{Q,K,V\}\), the correction is

\[
\widetilde X_i=X_i+\sum_{\ell=0}^{2}a_{i\ell}\kappa^X_\ell\odot X_{s_\ell(i)}.
\]

Here \(s_0(i)=i\), while \(s_1(i)\) and \(s_2(i)\) are the preceding two observed positions in the current observed segment. The eligibility indicator \(a_{i\ell}\) requires at least \(\ell+1\) observed positions. Each bias-free depthwise kernel \(\kappa^X_\ell\) has 64 entries. Padding is skipped. Persistent K/V tokens are appended after this correction and are not temporally filtered. Attention then applies its causal softmax, value aggregation, output projection, residual, and feedforward sublayer.

Let \(b_i\) be the complete first-block output. A recurrent correction is computed as

\[
u_i=A b_i,\qquad r_i=\operatorname{GRU}(u_i,r_{i-1}),\qquad
c_i=\mathbf1[n_i\ge2]B r_i,
\]

where \(A\in\mathbb R^{54\times64}\), \(B\in\mathbb R^{64\times54}\), and the one-layer GRU has input and hidden width 54. Here \(n_i\) counts observed positions up to and including \(i\) in the current segment; the recurrent state starts from zero at each segment boundary. The second attention block receives \(b_i+c_i\). The GRU residual uses the state updated with the current observation, not the previous recurrent state alone. The label “54” refers to this latent width.

Padding preserves the recurrent state while contributing no correction. A valid-but-withheld position resets correction-state and local-window eligibility. This reset does not introduce a new segment mask into the shared attention mechanism, which retains its causal and observed-key masks.

### 4.2 Controls and what they identify

The **disabled** version bypasses and freezes the correction parameters while retaining the shared attention blocks, persistent tokens, static mean retrieval, and heads. The **current-only** version uses the same correction parameter tensors but does not dereference earlier positions or carry a recurrent state across positions. Its local path is

\[
\widetilde X_i=X_i+a_{i0}\kappa^X_0\odot X_i
+a_{i1}\kappa^X_1\odot\tanh(X_i)
+a_{i2}\kappa^X_2\odot\operatorname{GELU}(X_i).
\]

The distinct parameter-free bases prevent three identical current-position terms from collapsing to one summed kernel. Eligibility remains matched to the full path. For its recurrent module, each position starts from zero and processes its current \(u_i\) twice:

\[
r_{i,1}=\operatorname{GRU}(u_i,0),\quad
r_{i,2}=\operatorname{GRU}(u_i,r_{i,1}),\quad
c_i=\mathbf1[n_i\ge2]B r_{i,2}.
\]

The second step allows hidden-to-hidden weights to participate without carrying state between positions. This current \(b_i\) is already prefix-conditioned by the first attention block. In contrast, the current Q/K/V inputs to the local correction precede that attention operation and derive from the current input embedding. These are different points in the computation graph.

The full and current-only paths each contain 576 local-filter parameters and 24,732 GRU-path parameters, totaling **25,308 active correction parameters**. The disabled path has zero active correction parameters. Local kernels and the GRU output projection are initialized to zero; common tensors begin with matched values. “Active” describes functional participation, not a guarantee that every tensor receives a nonzero gradient on the first update. All arms are trained end to end; the encoder architecture is fixed, not its learned weights.

A full-versus-current-only advantage would support the specified history-dependent correction package under this control. It would not isolate a lag-one effect, prove independent contributions from CNN and GRU, or establish that the control has identical functional capacity. Those stronger claims would require additional ablations not included in the current campaign.

## 5. Prototype retrieval as a separate question

The shared bank contains 64 learned vectors \(m_k\in\mathbb R^{64}\). They are static with respect to an evaluation sequence and are optimized on Train; they are not stored event records. Write \(N(v)=v/\max(\|v\|_2,\varepsilon)\). The normalized vector is the key and the unnormalized same vector is its value. Let \(h_j\) denote the second-block output and \(I_j\) the four highest-scoring slots.

**Mean retrieval** uses \(s_{jk}=N(h_j)^\top N(m_k)\) and
\(z_j=h_j+\tfrac14\sum_{k\in I_j}m_k\).
**Weighted retrieval** uses the same score rule and a softmax over the selected four scores:
\(z_j=h_j+\sum_{k\in I_j}\operatorname{softmax}(s_{j,I_j})_k m_k\).
Neither adds parameters beyond the common architecture.

The two learned-query variants receive the same summary features. From the latest 32 observed positions, including the current event, we form standardized log-gap and log-quantity values using frozen Train statistics. Log gaps use \(\log(\Delta_i/s)\); log quantities use \(\log(1+q_i)\). Each view supplies last value, mean, population standard deviation, and last-minus-previous difference, giving \(\phi_j^t,\phi_j^q\in\mathbb R^4\). Nonpositive gaps are excluded from time summaries; zero quantities remain valid. Unavailable statistics are zero, padding consumes no window position, and withheld positions reset the summary segment.

**Fixed split-query retrieval** uses

\[
q_j^t=h_j+\phi_j^tW_t,\quad q_j^q=h_j+\phi_j^qW_q,\qquad
s_{jk}^{\mathrm{fixed}}=\tfrac12N(q_j^t)^\top N(m_k)+\tfrac12N(q_j^q)^\top N(m_k),
\]

with \(W_t,W_q\in\mathbb R^{4\times64}\). **Generic512** uses

\[
q_j=h_j+\tfrac12[\phi_j^t,\phi_j^q]W,\quad
s_{jk}^{\mathrm{generic}}=N(q_j)^\top N(m_k),\qquad W\in\mathbb R^{8\times64}.
\]

Both select top four and apply softmax aggregation with temperature one. Both add exactly 512 active parameters, initialized to zero without changing the common initialization stream. The previous pilot's 521-parameter generic arm is not this budget-matched control. Adaptive weighting is not an arm of the present follow-up campaign.

The fixed score can also be written

\[
s_{jk}^{\mathrm{fixed}}=\tfrac12[N(q_j^t);N(q_j^q)]^\top[N(m_k);N(m_k)].
\]

This identity exposes its relationship to a concatenated normalized representation. It does not imply equivalence of the full training procedures; fixed normalizes two queries separately whereas generic512 normalizes a joint query. Describing the views as “time” and “quantity” also does not establish semantic disentanglement or better prototype content. For separately trained arms, encoder and bank values can diverge, so an arm-level difference is broader than a post-hoc change of aggregation alone. The contracted same-score, same-top-four aggregation diagnostic addresses that narrower operation separately without extra fits.

## 6. Experimental design

### 6.1 Roles of completed and new evidence

The completed MLP8 study is motivation, not a result for CNN+GRU54. Its 54 Validation conditions showed lower overall quantity RMSE for the history-enabled branch in all 18 matched history-on/off comparisons. Yet the comparison includes a capacity change, and adverse timing results remain: on Taxi, mean overall TimeNLL increased by 53.17% and 69.64% for the two prototype settings at the quantity-selected checkpoints. Prototype addition did not uniformly improve quantity RMSE. These observations motivate the active-budget control and mandatory reporting of timing and tail metrics; they do not establish the representative model's contribution [E1].

A completed native check found bitwise agreement across 83 parity checks between the existing CNN+GRU54 implementation and the mean-retrieval implementation on one RTX 5080 Runtime. Its scope included initialization, specified synthetic cases, and a small number of shared-batch updates. It supports the new structural baseline, not equality of historical A100 training runs or of converged checkpoints [E2].

The authorized campaign consists of **63 fresh fits**, in three sequential stages. Completion of each stage includes successful scientific terminal records, process exit evidence, verified selected/last originals, and a Validation report before the next stage starts.

| Stage and purpose | Conditions | New fits | Reuse and allocation |
|---|---|---:|---|
| Instacart supplementary baseline | MLP width 8, width 12, CNN+GRU54 × seeds 42/52/62 | 9 | 5080: seeds 42/62; 5090: seed 52. Historical width 16 is reference only. |
| Primary correction question | Full, disabled, current-only × Taxi/Intermittent/RAF × three seeds | 27 | 5080: Taxi/RAF; 5090: Intermittent. All arms for a dataset/seed share a GPU and Runtime. |
| Secondary retrieval question | Weighted, fixed, generic512 × three core datasets × three seeds | 27 | Reuse the nine new full-history fits as mean, once each. No duplicate mean training. |

Instacart alone does not test the correction or retrieval contribution. The earlier single-seed two-GPU retrieval pilot is developmental context, not two seeds or an independent replication of the present question. The fixed follow-up hypothesis arose after earlier evaluation observations, so the current Validation study is explicitly developmental.

### 6.2 Data and optimization

The dataset artifacts, target identities, splits, Train statistics, and preprocessing are frozen by the three execution contracts [E3]. Table entries below are eligible target counts and contracted settings, not all raw rows.

| Dataset | Train targets | Validation targets | Sequence cap, including target | Lookback bins in native units | Recorded gap unit; scale | Large quantity |
|---|---:|---:|---:|---:|---|---|
| Taxi | 38,393 | 8,268 | 256 | 168 | hour; 1 | >3,449 |
| Intermittent | 393,824 | 86,285 | 256 | 520 | week; 3 | >187 |
| RAF | 25,779 | 6,690 | 84 | 84 | month; 6 | >200 |
| Instacart | 1,991,192 | 503,733 | 64 | 52 | day; 7 | >35 |

The legacy field `lookback_weeks` is measured in each dataset's native discrete time units. At the last observed event with coordinate \(v_j\), the loader retains observations in \([v_j-(W-1),v_j]\), appends the next-event target, and truncates the combined sequence from the left to the stated cap. The cap includes the target, leaving at most 255, 255, 83, and 63 observed events respectively. The lookback values denote inclusive bins, not a uniform number of calendar weeks. The large-quantity thresholds come from Train and are not tuned on Validation. Full Validation uses all eligible targets. Fixed, weighted Train diagnostics are kept separate from full-Validation metrics.

Every fit uses batch size 128, AdamW with learning rate 0.001 and weight decay 0.01, gradient clipping at 1, and no warm start. Training runs for at least 40 and at most 300 epochs, with patience 40. The selector takes the **first strictly improving finite minimum of full-Validation overall raw quantity RMSE**; ties retain the earliest checkpoint. Loss, selector, and stopping settings are shared across arms. Common initial tensors and batch-order contracts are checked; subsequent optimization trajectories are allowed to differ.

The two hosts retain their existing dedicated Runtimes. Each host runs one worker, with a 36-hour per-condition bound and a 168-hour stage/host bound. A timeout is an incomplete scientific fit, not a poor score or a successful terminal result. There is no automatic retry or resume in this campaign. These bounds are resource limits, not completion forecasts or equal numbers of optimizer updates.

### 6.3 Outcomes, cost, and decision rules

At each quantity-selected checkpoint, we report six metrics: quantity RMSE, quantity MAE, and TimeNLL over all Validation targets and over the large-quantity subset. We retain each seed and report the three-seed mean and sample standard deviation. Paired differences, unfavorable seeds, and inconsistent datasets remain visible. Large-quantity sample counts accompany tail results. An improvement in the primary metric alone is not evidence of improvement in all six.

For the primary question, developmental support requires the full correction to achieve strictly lower overall quantity RMSE than **both** disabled and current-only controls for **each of the three seeds in the same two or more core datasets**. This is a predeclared consistency criterion, not a significance test. If it fails, the claim is restricted to the supported dataset/seed observations.

For the secondary question, fixed must beat mean, weighted, and generic512 under the same all-three-seeds, same-two-or-more-datasets criterion. It must also meet the corresponding GPU's cost limits at both sequence lengths 64 and 256. Native cost measurements use batch 128, five warm-up iterations, and twenty measured iterations. Relative to mean retrieval, inference time must be at most +20%, training-step time +30%, and peak allocation +20%. Technical execution PASS and candidate cost eligibility are distinct: results from an over-budget arm are retained and reported.

Fit time, selected epoch, final saved epoch, and measured step/inference costs are reported by GPU. We do not pool RTX 5080 and 5090 timings into an architectural efficiency estimate. Validation overhead, checkpointing, and early stopping can affect total fit time. TimeNLL and large-quantity tradeoffs are reported at the same selected checkpoint without inventing a favorable time threshold after seeing results.

### 6.4 Interpretation and remaining verification

A positive correction study would support an additional local/recurrent path within this representative architecture and protocol. A positive retrieval study would support the tested split-normalization constraint against its specified controls. Neither result alone establishes novelty, statistical significance, universal superiority, or independent predictive validity.

The current campaign does not repeat all external baselines. Before a final external comparison, the selected representative model and adapted baselines must be aligned in population, observable inputs, heads, optimizer budget, selection rule, seeds, and Runtime provenance. Native-model reproduction and common-head adaptation should occupy distinct comparison categories. S2P2 must remain in the comparison even if it is stronger; an unverified claim of superiority over all other models is not a substitute for reporting the full comparison.

Finally, no unseen evaluation is claimed here. Candidate and selector choices must be frozen before a genuinely unexposed period or dataset is used, with access history checked first. A new Validation table on previously used data does not restore independence. CPU binary reinference, any separately authorized Test evaluation, and a new independent assessment remain distinct tasks. The manuscript's eventual empirical contribution must be limited to the evidence these steps actually produce.

## References

[1] N. Du et al. *Recurrent Marked Temporal Point Processes: Embedding Event History to Vector*. KDD, 2016, pp. 1555–1564. [DOI](https://doi.org/10.1145/2939672.2939875), [author manuscript](https://people.mpi-sws.org/~manuelgr/pubs/rmtpp.pdf).

[2] A. C. Türkmen, T. Januschowski, Y. Wang, and A. T. Cemgil. *Forecasting intermittent and sparse time series: A unified probabilistic framework via deep renewal processes*. PLOS ONE 16(11), e0259764, 2021. [Article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0259764).

[3] C. Yang, H. Mei, and J. Eisner. *Transformer Embeddings of Irregularly Spaced Events and Their Participants*. ICLR, 2022. [Author manuscript](https://www.cs.jhu.edu/~jason/papers/yang%2Bal.iclr22.pdf).

[4] Y. Chang et al. *Deep Continuous-Time State-Space Models for Marked Event Sequences*. NeurIPS 38, 2025. [Publication](https://papers.nips.cc/paper_files/paper/2025/hash/ee39348acc798915d2d15a8bbbd417b8-Abstract-Conference.html), [method text](https://arxiv.org/html/2412.19634v2).

[5] S. Sukhbaatar et al. *Augmenting Self-attention with Persistent Memory*. arXiv:1907.01470, 2019. [Manuscript](https://arxiv.org/pdf/1907.01470).

[6] G. Lample et al. *Large Memory Layers with Product Keys*. NeurIPS, 2019. [Publication](https://papers.neurips.cc/paper/9061-large-memory-layers-with-product-keys.pdf).

[E1] Completed MLP8 structure study, Validation only: [tables](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_structure_contribution_results_20261007_v1/VALIDATION_TABLES.md), [claim audit](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_structure_contribution_results_20261007_v1/CONTRIBUTION.md).

[E2] CNN+GRU54/mean native parity: [attempt 3 completion receipt](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_claims_and_parity_20261008_v1/evidence/attempt3/completion_receipt.json).

[E3] Frozen execution contracts: [Instacart](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_instacart_20261008_v1/execution_contract.json), [history correction](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_20261008_v1/execution_contract.json), [retrieval](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_retrieval_20261008_v1/execution_contract.json). File-level method provenance is in the accompanying writing notes and evidence manifest; these development references will need publication-appropriate anonymization and archival identifiers.
