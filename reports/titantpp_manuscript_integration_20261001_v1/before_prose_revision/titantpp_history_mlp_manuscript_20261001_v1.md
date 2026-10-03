# TitanTPP: Efficient History Correction for Next-Event Quantity Prediction

**Working manuscript — 1 October 2026.** This draft integrates the verified method, related work, evaluation protocol, completed validation evidence, and matched batch-cost measurements. *TitanTPP* denotes the fixed `titantpp_history_mlp` configuration throughout. It is separate from the historical v0.7 manuscript. Performance evidence uses the existing 1 October 2026, 07:07 KST comparison cutoff and subsequent completed-artifact audits; no new training, prediction, or server observation was performed for this draft. RAF and the incomplete three-seed Instacart comparison with S2P2/AttNHP are not presented as final results. The abstract and final conclusion await those results and an independently authorized evaluation.

## 1. Introduction

An event sequence records both when an observation occurs and the numerical quantity associated with it. For demand-related applications, the size of the next observed event can matter alongside its arrival time. Treating each size as a categorical event type discards the ordering of quantities, while forecasting an equally spaced series addresses a different target unless a mapping between event and calendar-time predictions is specified. We study the next recorded event directly: its duration distribution and a nonnegative quantity point prediction, conditional on observed gaps and quantities.

Temporal point processes provide established tools for representing irregular event histories. Renewal-process models have already connected this representation to intermittent demand, and recent event models accommodate numerical as well as categorical attributes. Accordingly, our question concerns the representation and computational cost of event history, rather than the introduction of numerical marks to point-process research. [Türkmen et al., 2021](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0259764); [Draxler et al., 2025](https://proceedings.nips.cc/paper_files/paper/2025/hash/a6c7515ac435277dc92b75a07bb2257c-Abstract-Conference.html).

We introduce TitanTPP, a causal event encoder with a bottleneck residual that combines adjacent contextual states between two attention blocks. The model retains persistent vectors and static prototype retrieval but omits online neural-memory optimization. This design is motivated by the cost of updating neural memory at inference in Titans and by the possibility that a small feed-forward correction can provide useful event-history information in bounded input windows. We evaluate that possibility through component comparisons and separately measure batch-processing cost against a Titans-MAC event adapter. [Behrouz et al., 2025](https://papers.nips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html).

The contributions supported by the current development evidence are:

1. **An event-history architecture for numerical quantity prediction.** TitanTPP combines causal gap–quantity encoding, a 6,144-parameter history correction, and fixed-at-inference learned banks with shared duration and quantity heads. The correction has explicit rules for missing predecessors and limited history, and it preserves the base function at initialization.
2. **A controlled evaluation of quantity–time trade-offs.** A common-input, common-head protocol compares six TPP encoders and internal structural alternatives. Completed validation comparisons show quantity improvements on Taxi and Intermittent, with limited gains on Instacart and counterexamples in time likelihood. These observations bound the empirical claim rather than establish general superiority.
3. **Measured computational evidence with a disclosed boundary.** A matched RTX 5080 experiment measures training-step time, evaluation-batch time, and allocated/reserved GPU memory. It identifies faster batch processing and lower training allocation, together with higher evaluation allocation, relative to the tested MAC adapter.

These contributions distinguish the proposed configuration, its empirical behavior, and its measured cost. The component operations are established building blocks; their combination alone does not prove novelty or that each component is necessary. The remaining evidence requirements are stated in Section 7.

## 2. Related work

### 2.1 Neural representations of event histories

RMTPP represents the influence of event history with a recurrent neural network and uses that state to model event times and marks. The Neural Hawkes Process instead employs a continuous-time LSTM whose state controls evolving event intensities. These approaches motivate recurrent comparators for event representations, but their original likelihoods and categorical outputs differ from the shared heads used in our quantity-prediction experiments. [Du et al., 2016](https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf); [Mei and Eisner, 2017](https://papers.nips.cc/paper_files/paper/2017/hash/6463c88460bd63bbe256e495c63aa40b-Abstract.html).

Attention-based models offer an alternative representation of event history. THP uses self-attention to capture temporal dependencies, whereas SAHP incorporates time intervals into its attention-based encoding. AttNHP replaces recurrent event embeddings with an attention architecture for irregular events. TitanTPP shares the use of causal attention; its distinguishing design in this comparison is the intermediate adjacent-state correction and static retrieval path. We do not attribute the ability to attend to irregular events itself to our method. [Zuo et al., 2020](https://proceedings.mlr.press/v119/zuo20a.html); [Zhang et al., 2020](https://proceedings.mlr.press/v119/zhang20q.html); [Yang et al., 2022](https://arxiv.org/abs/2201.00044).

S2P2 develops a continuous-time state-space representation using stochastic jumps and nonlinear transformations, with a parallel scan for efficient sequence processing. It therefore provides a comparator with a different temporal inductive bias from recurrent and attention encoders. Our adapted implementation preserves its state evolution while connecting the representation to common prediction heads; the resulting measurements are not evaluations of the original S2P2 intensity model or its published likelihood benchmark. [Chang et al., 2025](https://arxiv.org/abs/2412.19634).

### 2.2 Numerical marks and intermittent demand

Intermittent demand has previously been formulated through renewal processes that model demand arrivals and sizes. Deep renewal processes replace simple smoothing relations with neural state updates and also describe a continuous-time extension. FlexTPP more recently models heterogeneous event attributes with discrete and continuous output heads, including normalizing flows for continuous values. These precedents rule out framing TitanTPP as the first continuous-mark TPP or the first bridge between event models and intermittent demand. [Türkmen et al., 2021](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0259764); [Draxler et al., 2025](https://proceedings.nips.cc/paper_files/paper/2025/hash/a6c7515ac435277dc92b75a07bb2257c-Abstract-Conference.html).

Our present task is narrower: a point estimate of the next numerical quantity and a likelihood for the recorded duration. Quantities may themselves be integer-valued; using a real-valued regression output does not turn them into continuously observed marks. The quantity head does not define a normalized quantity density, so the combined training objective is not claimed to be a full joint marked-event log-likelihood. Likewise, next-event evaluation does not establish fixed-horizon cumulative demand accuracy. Deep renewal processes and FlexTPP are relevant methodological context, but they are not included in the current experimental panel. Their absence limits claims about the wider numerical-mark forecasting literature.

### 2.3 Neural memory and the efficiency question

Titans combines attention with a neural long-term memory that learns from the incoming context at test time. The MAC variant places memory in the attention context. TitanTPP retains learned memory vectors while removing this online optimization path from its own encoder; its history correction is an ordinary feed-forward residual. [Behrouz et al., 2025](https://arxiv.org/abs/2501.00663).

Whether that choice is useful is an empirical question. Our comparator is a documented Titans-MAC event adapter with online memory updates retained. The matched cost experiment measures two full adapted backbones, so it cannot assign the entire speed difference to memory removal alone. It also does not establish equivalent predictive accuracy, equal-convergence cost, or a universal advantage over Titans. This distinction connects the architectural motivation to the measurements in Section 6 without conflating them.

## 3. Method

### 3.1 Prediction task and observed inputs

Let an event history be \(\mathcal H_n=\{(t_i,q_i)\}_{i=1}^{n}\), where \(t_i\) is the recorded event time, \(\Delta_i\) is the supplied recorded inter-event duration, and \(q_i\geq0\) is the event quantity. We predict the duration and quantity of the next event from the observed history. Quantity is a numerical regression target; it need not be an event-type label. The present experiments evaluate next-event prediction and do not establish a method for arbitrary-horizon forecasts on a regular time grid.

Durations use the dataset’s recorded units and preprocessing conventions; the first history row uses the loader’s positive-gap convention. For a padded sequence, let \(v_i\) indicate a valid row and \(w_i\) indicate an observed row permitted as encoder input. Define \(o_i=v_iw_i\). The two input features and their embedding are

$$
f_i=[\log(1+\Delta_i),\log(1+q_i)]^\top,\qquad
x_i=o_i\left(W_{\mathrm{in}}f_i+b_{\mathrm{in}}+p_i\right). \tag{1}
$$

Here \(p_i\) is a learned positional embedding and the hidden dimension is \(d=64\). In implementation, unobserved inputs are replaced by zero **before** the logarithm and projection. Thus even invalid values at withheld or padded positions cannot enter the representation. Each training window contains an observed prefix and one target row. The target row is withheld from the encoder, and the heads read the last observed state. Its duration and quantity are used only to evaluate the loss.

### 3.2 Causal encoding with persistent vectors

TitanTPP contains two pre-normalized causal attention blocks. For block \(\ell\), each attention head projects the normalized event states into queries, keys and values. A learned persistent bank \(P^{(\ell)}\in\mathbb R^{16\times d}\) is split across the heads and prepended to their keys and values:

$$
K=[P^{(\ell)};K_{\mathrm{events}}],\quad
V=[P^{(\ell)};V_{\mathrm{events}}],\quad
A_i=\operatorname{softmax}\!\left(Q_iK^\top/\sqrt{d_h}+C_i\right)V. \tag{2}
$$

There are four heads and \(d_h=16\). Head indices are suppressed in (2), where the persistent bank denotes the corresponding 16-by-16 head slice. The additive mask \(C_i\) permits every persistent vector and only observed event keys at positions \(j\leq i\). The bank supplies the same raw vectors as keys and values; it is not passed through the event key/value projections. Attention dropout and a learned output projection precede the attention residual. A second residual contains a pre-normalized feed-forward network of width 128 with GELU and dropout. Dropout is 0.1; unobserved query states are zeroed after each residual. Denote the resulting block by \(\mathcal E_\ell\). The first block produces \(H^{(1)}=\mathcal E_1(X)\).

These persistent vectors are ordinary model parameters learned by the outer training optimizer. They remain fixed during inference. They do not implement an online neural-memory update.

### 3.3 A bottleneck correction from adjacent observed states

The correction between the two encoder blocks combines the current state with its immediate observed predecessor. Let \(\pi(i)\) be the preceding observed position, skipping padding, and let \(c_i\) count observed rows since the most recent withheld valid row, including position \(i\). For thresholds \(\tau=(1,2,4,8,16,32,64,128)\), define a deterministic availability mask

$$
a_{i,b}=o_i\,\mathbf 1\{c_i>\tau_b\},\qquad b=1,\ldots,8. \tag{3}
$$

Each branch contains bias-free matrices \(U_b\in\mathbb R^{4\times2d}\) and \(V_b\in\mathbb R^{d\times4}\). The correction and its insertion are

$$
r_i=\frac{1}{8}\sum_{b=1}^{8}a_{i,b}V_b\operatorname{GELU}\!\left(U_b[h_i^{(1)};h_{\pi(i)}^{(1)}]\right),
\qquad \widetilde h_i^{(1)}=h_i^{(1)}+r_i,
\qquad H^{(2)}=\mathcal E_2(\widetilde H^{(1)}). \tag{4}
$$

GELU follows the exact error-function formulation. Although the thresholds differ, all eight branches read the **same immediate predecessor**. The thresholds control availability, not retrieval lag. The divisor is always eight, including when only some branches are active. If no predecessor exists, its vector is defined as zero and every branch is unavailable. When no branch is available, the correction is exactly zero. A withheld valid row resets correction eligibility, whereas padding does not. This eligibility rule does not reset the causal attention history of the encoder.

The module adds \(8(2dr+rd)=6{,}144\) parameters for \(r=4\). We initialize every \(V_b\) to zero. Consequently the correction is zero at initialization; shared encoder and head parameters therefore retain the predictions of the configuration without this correction under the same stochastic state. Module construction preserves the caller's random-number stream. This establishes a common initialization, not evidence that zero initialization itself improves final accuracy. The implementation computes the current and predecessor projections before gathering their low-dimensional values; it is algebraically equivalent to (4).

The bottleneck provides a limited-capacity path for adjusting intermediate event representations. The predecessor state is already causally contextualized, so the complete model is not restricted to two raw events. Conversely, the correction is not an online learned memory and is not a bank of eight different temporal lags.

### 3.4 Static prototype retrieval and shared prediction state

A separate bank \(M\in\mathbb R^{64\times d}\) provides static retrieval after the second block. For each observed query, select four prototypes by cosine similarity, then add their unweighted mean:

$$
S_i=\operatorname{TopK}_{j,\,k=4}\operatorname{cos}(h_i^{(2)},m_j),\qquad
z_i=o_i\left(h_i^{(2)}+\frac14\sum_{j\in S_i}m_j\right). \tag{5}
$$

Normalization is used for selection only. The vectors added to the residual are the raw prototypes, not their normalized versions or a softmax-weighted average. Keys and values are tied. This bank is distinct from the persistent vectors inside attention. Both banks are learned during training and receive no online writes during inference. The last observed state \(z_n\) is shared by the time and quantity heads.

### 3.5 Numerical quantity and recorded-duration heads

The quantity head predicts on the log-transformed scale:

$$
\widehat y_{n+1}=\operatorname{softplus}(w_q^\top z_n+b_q),\qquad
\widehat q_{n+1}=\exp(\widehat y_{n+1})-1. \tag{6}
$$

This is a nonnegative point prediction, not a fitted probability density over quantity. At initialization, \(w_q=0\) and the bias gives \(\widehat y=\overline{\log(1+q)}_{\mathrm{train}}\). This differs from initialization at the arithmetic mean of raw quantities.

The time head parameterizes a positive latent duration \(T\):

$$
\mu_n=w_\mu^\top z_n+b_\mu,\quad
\sigma_n=\operatorname{softplus}(w_\sigma^\top z_n+b_\sigma)+10^{-3},\quad
\log(T/s)\mid\mathcal H_n\sim\mathcal N(\mu_n,\sigma_n^2). \tag{7}
$$

Recorded durations follow a positive-integer observation model, \(D=\max(1,\operatorname{round}(T))\), with \(D=\min(K,\max(1,\operatorname{round}(T)))\) when upper coding applies; \(K=30\) for Instacart. Writing \(F_n\) for the CDF of \(T\), the observed probability mass is

$$
p(D=d\mid\mathcal H_n)=
\begin{cases}
F_n(1.5),&d=1,\\
1-F_n(K-0.5),&d=K\text{ when upper coding applies},\\
F_n(d+0.5)-F_n(d-0.5),&\text{otherwise}.
\end{cases} \tag{8}
$$

The first bin begins at zero, rather than 0.5. The scale \(s\) is 1 hour for Taxi, 3 weeks for Intermittent, and 7 days for Instacart. The implementation evaluates log masses in float64 using stable CDF or survival-function differences. Reported time NLL is the negative log of (8), not the continuous lognormal density at the recorded integer.

### 3.6 Optimization and checkpoint selection

For a minibatch \(\mathcal B\) of history–target pairs, the training objective is

$$
\mathcal L=\frac1{|\mathcal B|}\sum_{n\in\mathcal B}
\left[-\log p(D_{n+1}\mid\mathcal H_n)
+\big(\widehat y_{n+1}-\log(1+q_{n+1})\big)^2\right]. \tag{9}
$$

The quantity term carries weight one, and the objective contains no additional tail loss. Each window contributes one final target. We optimize with AdamW at a learning rate \(10^{-3}\), weight decay 0.01, gradient clipping at norm 1, and batch size 128. Training permits at most 300 epochs and stops no earlier than epoch 40, with early-stopping patience of 40. The selected checkpoint is the earliest checkpoint attaining the strictly lowest finite validation RMSE on the raw quantity scale. MAE and time NLL are reported at that same checkpoint; their minima are not selected separately. Seeds are 42, 52 and 62.

The representative architecture was selected after examining validation results. These results support model development; they are not an untouched test estimate. The final evaluation requires separately frozen rules and independent authorization, as discussed in Section 7.


![TitanTPP architecture](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_method_efficiency_20260930_v1/architecture.png)

**Figure 1.** The fixed TitanTPP architecture. All eight correction branches read the same immediately preceding observed state; their thresholds control availability, not distinct retrieval lags. Persistent attention vectors and final static prototypes are learned during training and remain fixed at inference. The withheld target enters the loss only. The original [vector figure](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_method_efficiency_20260930_v1/architecture.pdf) and [editable diagram](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_method_efficiency_20260930_v1/architecture.mmd) are reused unchanged.

### 3.7 Design rationale and computational scope

The two observed features describe event spacing and magnitude without introducing unavailable future information. Placing the correction between the blocks lets the second block process an adjusted contextual representation. Its bottleneck restricts added capacity, while zero output initialization preserves the shared initial function. These statements describe the design and verified implementation properties; they do not establish that the insertion position, rank, or initialization is individually optimal.

A fixed divisor of eight reduces the aggregate correction when fewer branches are available. We retain this rule as part of the selected architecture. The tested alternative that divides by the active branch count is a separate exploratory variant, not a silent change to TitanTPP. Static retrieval is also retained, but the completed static-retrieval removal experiment uses the separate Full architecture and cannot establish that retrieval is necessary in TitanTPP.

For fixed width and branch count, the correction performs linear dense-projection work in sequence length. Full causal attention remains in the encoder, so the complete backbone is not claimed to have linear sequence complexity. The term “efficient” refers to the measured comparison in Section 6, not a complexity guarantee for arbitrarily long histories.

## 4. Evaluation protocol

### 4.1 Event construction, splits, and scope

The existing datasets use different definitions of a quantity-bearing event. Taxi counts pickup rows within a spatial cell and an active hour. Intermittent uses positive `order_qty` observations for a site–part sequence. Instacart counts product rows aggregated for a user on the same recorded activity day; it is a proxy for basket size rather than the number of physical units bought or necessarily one order. Zero-demand calendar periods are not separate quantity targets in these event representations.

Frozen splits follow event order within each sequence, approximately 70/15/15, rather than a common calendar cutoff across all sequences. The current analyses use training and validation only. Existing Taxi sequence selection and Intermittent activity filtering/stratification refer to the full observed sequence length; therefore cohort construction is retrospective and is not described as entirely train-only. Train-only feature initialization and the absence of target leakage in a window do not remove that cohort-selection limitation.

**Table 1.** Frozen train/validation target populations and input limits. Maximum sequence length includes the withheld target, so it is not the maximum number of observed inputs. The lookback values use the recorded units, despite the legacy loader field name `lookback_weeks`.

| Dataset | Train targets | Validation targets | Maximum rows | Lookback in recorded units | Duration scale |
|---|---:|---:|---:|---|---|
| Taxi | 38,393 | 8,268 | 256 | 168 hours | 1 hour |
| Intermittent | 393,824 | 86,285 | 256 | 520 weeks | 3 weeks |
| Instacart | 1,991,192 | 503,733 | 64 | 52 recorded days | 7 days; upper code 30 |

Models receive the same permitted gap and quantity history, training targets, validation targets, and quantity/time heads. They do not receive item identities, product composition, or calendar covariates in addition to those two features. Data and target identities, source revisions, and selection records are retained in the frozen contracts and audit trail. The upstream provenance and redistribution terms of the local Intermittent source still require independent documentation before public release; it is not described here as an independently verified public real-world dataset. RAF is a separate ongoing extension and is excluded from the completed-results tables below.

### 4.2 Common-head TPP comparisons

The experimental question is which encoder yields useful representations for this shared duration–quantity objective. We compare RMTPP, NHP, THP, SAHP, S2P2, and AttNHP after adapting their inputs and outputs to that question. Their results are labeled as **common-head adaptations**, rather than the best achievable performance of the original complete models. The common objective, checkpoint selector, and maximum epoch budget control several sources of variation, but do not equalize parameter count, optimization difficulty, runtime, or tuning opportunity across architectures.

**Table 2.** Disclosed adaptation boundaries for the two added comparators. Official upstream revisions are preserved in the local source receipt.

| Comparator | Retained computation | Connection to the common task | Implementation boundary |
|---|---|---|---|
| S2P2 | HiPPO initialization, complex diagonal state evolution, backward zero-order-hold discretization, event impulses, relative time, normalization/nonlinearity | Numerical gap–quantity projection replaces categorical input embedding; shared heads replace native intensity output | Width 64, two layers, state size 16; the affine-doubling recurrence was checked against the original sequential equations, but has O(L log L) work and is not the original work-efficient scan |
| AttNHP | Query/context attention, sinusoidal time encoding, and tanh query residual | Common numerical input and heads; query at the last observed event’s right limit, without accessing the next target time | Width 64, two layers, one attention head, time-encoding width 16, no feed-forward sublayer |

S2P2 is pinned to `UCIDataLab/state_space_point_process@c3933240f16a22b43d80d09bda272475526ff24b`; AttNHP to `yangalan123/anhp-andtt@78f754fde61547b84a0749fd7e7e020e7cd6b4d8`. Existing validation checks cover causality, masking, input/output shape, shared head initialization, and correspondence to the retained upstream equations. These checks establish the tested implementation boundary, not equivalence to each paper’s native modeling objective.

### 4.3 Selection, reporting, and reproducibility

Training loss, checkpoint selection, and reported metrics answer different questions. Equation (9) fits recorded-duration likelihood and squared log-quantity error. Checkpoints are selected by validation RMSE on the raw quantity scale, giving more influence to large absolute quantity errors. This choice does not assume that all events have large demand, encode an asymmetric inventory cost, or make MAE irrelevant. MAE remains a complementary measure of typical absolute error, and time NLL evaluates a different output. Their values are always taken from the same RMSE-selected checkpoint; separately choosing a favorable epoch for each metric would change the protocol.

For completed three-seed groups, we report the arithmetic mean and sample standard deviation over seeds 42, 52, and 62, together with same-seed comparisons. Three-seed consistency is descriptive evidence, not a significance test. The “B” control retains the causal encoder and learned banks but omits the intermediate correction. Full and its component removals are internal alternatives, while learned gates and active-branch normalization remain separately labeled exploratory variants. All completed conditions and adverse results remain in the source tables.

Terminal records preserve both selected and last checkpoints and their validation re-evaluations. Completed core results and the 5080 additional-comparator results have passed source/checkpoint CPU audits. Audit completion verifies integrity and correspondence of records; it does not establish generalization or empirical superiority. The study has used validation results for model development, so final independent evaluation remains necessary.

## 5. Completed evidence connecting design to behavior

### 5.1 What the history-correction control establishes

Table 3 compares TitanTPP with B, the same encoder without the intermediate correction. It gives direct evidence about adding the tested correction, while also adding 6,144 parameters. It cannot isolate the effect of adjacency from added capacity, or prove that rank four and the availability thresholds are optimal. Lower values are better for all three metrics.

**Table 3.** Completed validation results, mean ± sample standard deviation across three seeds. Each run contributes all three metrics from its RMSE-selected checkpoint.

| Dataset | Configuration | Quantity MAE | Quantity RMSE | Time NLL |
|---|---|---:|---:|---:|
| Taxi | B: no correction | 28.7545 ± 0.4526 | 90.5093 ± 0.5207 | 0.7247 ± 0.0668 |
| Taxi | TitanTPP | 25.6237 ± 0.8320 | 79.7111 ± 2.8260 | 1.0387 ± 0.2843 |
| Intermittent | B: no correction | 0.7588 ± 0.0658 | 1.7806 ± 0.0439 | 0.3846 ± 0.1034 |
| Intermittent | TitanTPP | 0.7005 ± 0.0541 | 1.6730 ± 0.0819 | 0.4971 ± 0.2095 |
| Instacart | B: no correction | 3.9932 ± 0.0018 | 5.8796 ± 0.0166 | 2.8065 ± 0.0045 |
| Instacart | TitanTPP | 3.9916 ± 0.0026 | 5.8827 ± 0.0048 | 2.8071 ± 0.0041 |

Adding the correction reduces mean quantity MAE/RMSE by 10.89%/11.93% on Taxi and 7.68%/6.04% on Intermittent. On Instacart, MAE decreases by only about 0.04% and RMSE increases by about 0.05%. Mean time NLL increases on all three datasets. This pattern supports a quantity-specific benefit on two datasets under the current setup, rather than an across-task benefit of the correction.

### 5.2 Relationship to the external comparison

The six completed common-head comparator groups on Taxi and Intermittent all have higher quantity MAE and RMSE than TitanTPP in the same-seed comparisons: 2 datasets × 6 models × 3 seeds = 36 paired comparisons for each metric. RMTPP has the lowest three-seed mean MAE and RMSE among those six comparators on both datasets. Table 4 places the current quantity claim against that stronger reference, instead of averaging it over weak baselines.

**Table 4.** Quantity reductions relative to the strongest completed external mean on each dataset. RMTPP is that comparator for both metrics here. Time NLL is shown to retain the trade-off; values are three-seed means, with full deviations and all six comparators in the source report.

| Dataset | MAE reduction | RMSE reduction | TitanTPP time NLL | RMTPP time NLL |
|---|---:|---:|---:|---:|
| Taxi | 10.13% | 15.85% | 1.0387 | 1.5798 |
| Intermittent | 21.72% | 34.39% | 0.4971 | 0.2748 |

Time likelihood does not have the same ranking. Several external encoders have lower time NLL, including S2P2 on both datasets. Instacart also does not support an overall quantity advantage: TitanTPP’s mean MAE is approximately 0.16% higher than RMTPP’s, and its RMSE is approximately 0.42% higher than THP’s in the completed original panel. Additional Instacart comparator groups are incomplete at the report cutoff and are not assigned three-seed means. These differences are numerical comparisons, not tests of equivalence.

Train-only characterization gives a possible explanation to investigate: the median usable Instacart history has five events, and the adjacent log-quantity correlation after subtracting each user’s mean is 0.037. The corresponding centered correlations are 0.749 on Taxi and 0.928 on Intermittent. Short histories and weak adjacent linear association may limit the correction’s benefit, but these descriptive statistics do not establish causation or rule out nonlinear predictive information. The learned contextual states also contain more information than raw adjacent quantities.

Full remains an alternative architecture, not a second representative selected per dataset. It has slightly lower quantity errors than TitanTPP on Taxi, whereas TitanTPP has lower errors on Intermittent. The paper therefore does not claim that TitanTPP wins every internal comparison. Gate and active-normalization results likewise remain in the experimental record; their current behavior does not justify replacing the fixed representative across all datasets.

Internal evidence: [complete comparison](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_completed_external_comparison_20261001_v1/report.md), [subsequent integrity audit](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_completed_5080_audit_20261001_v1/report.md), and [train-data characterization](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_data_characteristics_20260928_v1/README.md). These local links are development provenance and will be converted to anonymized supplementary references for submission.

## 6. Matched computational cost against Titans-MAC

### 6.1 Measurement boundary

We measured TitanTPP and the Titans-MAC event adapter on one RTX 5080, using identical cached training-input batches, padding length 256, batch size 128, and common heads, loss, outer optimizer, and precision. Each dataset uses 4,736 distinct target windows: five warm-up batches followed by 32 measured batches. A dedicated seed-42 permutation selects the windows; this is not a timing of an entire training epoch or the original training-prefix order. The measurement runtime uses Python 3.12.13, PyTorch 2.11.0+cu130, CUDA 13.0, float32 model computation, disabled TF32, and deterministic execution settings.

Each dataset/model/path combination has three independent fresh-process repetitions, all using the same seed. These are timing repetitions, not three accuracy seeds. Within each repetition, latency is the median of the 32 synchronized compute intervals. Tables report the mean and sample standard deviation of those three medians. Speed ratios are calculated within each paired repetition before averaging. The model order is reversed in the second repetition to reduce a fixed-order effect.

Training timing includes the forward pass, shared heads and loss, gradient clearing, backward pass, gradient clipping, and the AdamW update. Evaluation timing includes the forward pass, heads, and target loss, without an outer optimizer update. The MAC adapter retains observed-history memory updates and internally enables gradients where its update requires them. Thus “evaluation batch” is not pure deployment inference. Host-to-device transfer is measured separately and excluded from the principal compute interval; GPU synchronization bounds each measured interval.

The MAC adapter uses a fixed-shape compiled scan and inner-update gradient clipping at norm one. No new compiled graphs appeared during the 32 measured batches. Compilation and initialization remain separate costs: the first cold MAC training batch took 49.920 seconds and the first cold evaluation batch 10.744 seconds. These are first-batch observations including initialization, not pure compilation durations. Warm measurements below do not absorb them into an amortized epoch estimate.

### 6.2 Observed time and memory

**Table 5.** Synchronized compute time in milliseconds per batch; mean ± sample standard deviation across three repetitions of the within-repetition median. The final column is the paired MAC/TitanTPP time ratio.

| Dataset | Path | TitanTPP (ms) | MAC adapter (ms) | MAC / TitanTPP |
|---|---|---:|---:|---:|
| Taxi | Training step | 31.447 ± 0.106 | 180.582 ± 6.056 | 5.74 ± 0.18 |
| Taxi | Evaluation batch | 7.632 ± 0.009 | 64.827 ± 0.622 | 8.49 ± 0.08 |
| Intermittent | Training step | 32.157 ± 0.023 | 179.472 ± 0.555 | 5.58 ± 0.01 |
| Intermittent | Evaluation batch | 7.909 ± 0.030 | 64.916 ± 0.645 | 8.21 ± 0.07 |

**Table 6.** Mean peak GPU memory in MiB across the same three repetitions. Peak counters are reset after warm-up. Allocated and reserved memory are separate measurements; neither is a measurement of memory complexity at other input lengths.

| Dataset | Path | TitanTPP allocated | MAC allocated | TitanTPP reserved | MAC reserved |
|---|---|---:|---:|---:|---:|
| Taxi | Training | 1869.5 | 7631.3 | 2042.0 | 7670.0 |
| Taxi | Evaluation | 389.2 | 97.8 | 416.0 | 151.3 |
| Intermittent | Training | 1869.5 | 7631.0 | 2042.0 | 7646.0 |
| Intermittent | Evaluation | 389.2 | 97.8 | 416.0 | 130.0 |

For these inputs and implementations, the MAC/TitanTPP time ratio is 5.58–5.74 for training steps and 8.21–8.49 for evaluation batches. TitanTPP’s peak allocated training memory is about 75.5% lower, but its evaluation allocation is about 3.98 times that of the adapter. The two tested length-256 configurations have 96,003 and 122,310 trainable parameters respectively, a 21.5% reduction for TitanTPP. Parameter counts depend on the configured positional embedding length and should not be copied unchanged to Instacart.

![Matched batch cost and GPU memory](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_efficiency_5080_20260930_v1/paired_efficiency.png)

**Figure 2.** Matched RTX 5080 batch measurements. Timing bars summarize the three repetition medians; error bars show their sample standard deviations. The memory panel shows peak allocated memory and exposes the reversal between training and evaluation. Both datasets use identical input batches across the two models. Evaluation includes target-loss computation. The [vector figure](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_efficiency_5080_20260930_v1/paired_efficiency.pdf) and measurements are reused unchanged.

These findings support a computational advantage for the tested batch paths, coupled with a memory trade-off. They do not establish seconds per full epoch, end-to-end training duration, equal-accuracy efficiency, energy use, or serving latency. The measurement compares complete backbones; an isolated memory-update removal ablation would be required to attribute the full difference to that operation alone. Earlier MAC/B0 epoch ratios refer to another baseline and are not used as current TitanTPP speedups. The source [measurement report](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_efficiency_5080_20260930_v1/report.md) includes the per-batch records and verification receipt.

## 7. Limitations and evidence required for the final manuscript

The current evidence supports a specific architecture and a quantity–time–cost trade-off within a fixed protocol. It does not support a new continuous-mark paradigm, a fully probabilistic quantity model, or superiority on every dataset and metric. The correction, rank, availability thresholds, and static banks have not each been isolated in capacity-matched experiments. The base-control comparison adds parameters as well as a computational path, while the MAC cost study changes the complete backbone. These constraints limit causal statements about why a design works.

Validation results informed the representative architecture and exploratory alternatives. Their integrity audits do not make them an untouched test estimate. Before final evaluation, the representative, all comparator checkpoints, metrics, aggregation rules, and uncertainty procedure must be fixed; split construction and previous access history must also be checked to establish what independence can be claimed. This draft has not opened held-out predictions or metrics. Per-sequence temporal splits and retrospective cohort selection further limit extrapolation to prospective deployment with a global calendar cutoff.

The input representation omits potentially useful covariates and evaluates numerical quantities attached to observed events. Instacart’s basket-size proxy and upper-coded time observations limit the interpretation of purchase behavior. For Intermittent, upstream provenance and publication rights require documentation. No inventory-cost benefit or arbitrary-horizon forecasting performance is established by next-event MAE/RMSE. The six external common-head adaptations constitute a defined comparison panel rather than exhaustive coverage of numerical-mark or renewal forecasting methods.

The final results section must add the completed RAF study and remaining Instacart comparator groups after their terminal audits, retaining failures and counterexamples. It must also retain selected-versus-last behavior and the existing quantity-range/history-length analyses instead of replacing them with overall means alone. The final abstract and conclusion will be written after those tables and the separately authorized independent evaluation. No new ablation or training run is implied by this manuscript integration.

## Appendix A. Notation


| Symbol | Meaning | Fixed value or scope |
|---|---|---|
| \(\Delta_i,q_i\) | Recorded gap and numerical event quantity | Observed history only |
| \(v_i,w_i,o_i\) | Valid, permitted observation and effective observation masks | Boolean |
| \(p_i\) | Learned position embedding | Maximum sequence includes the target |
| \(d,H,d_h,d_{ff}\) | Hidden width, heads, head width, feed-forward width | 64, 4, 16, 128 |
| \(P^{(\ell)}\) | Persistent key/value bank in block \(\ell\) | 16 × 64 per block |
| \(\pi(i),c_i\) | Immediate observed predecessor; local observed count | Padding skipped; withheld eligibility reset |
| \(a_{i,b}\) | Deterministic branch-availability mask | Not a learned Gate |
| \(U_b,V_b,r\) | Input/output matrices and bottleneck width | 4 × 128; 64 × 4; 4 |
| \(M,S_i\) | Final static prototype bank and selected indices | 64 × 64; top four |
| \(z_n\) | Last observed shared head input | 64-dimensional |
| \(s,K\) | Duration scale and optional upper code | Dataset-specific; K=30 only for Instacart |


## References

- **Du et al. (2016).** Nan Du, Hanjun Dai, Rakshit Trivedi, Utkarsh Upadhyay, Manuel Gomez-Rodriguez, and Le Song. *Recurrent Marked Temporal Point Processes: Embedding Event History to Vector*. KDD. [Primary paper](https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf). DOI: 10.1145/2939672.2939875.
- **Mei and Eisner (2017).** Hongyuan Mei and Jason Eisner. *The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process*. NeurIPS 30. [Proceedings](https://papers.nips.cc/paper_files/paper/2017/hash/6463c88460bd63bbe256e495c63aa40b-Abstract.html).
- **Zuo et al. (2020).** Simiao Zuo, Haoming Jiang, Zichong Li, Tuo Zhao, and Hongyuan Zha. *Transformer Hawkes Process*. ICML, PMLR 119:11692–11702. [Proceedings](https://proceedings.mlr.press/v119/zuo20a.html).
- **Zhang et al. (2020).** Qiang Zhang, Aldo Lipani, Omer Kirnap, and Emine Yilmaz. *Self-Attentive Hawkes Process*. ICML, PMLR 119:11183–11193. [Proceedings](https://proceedings.mlr.press/v119/zhang20q.html).
- **Türkmen et al. (2021).** Ali Caner Türkmen, Tim Januschowski, Yuyang Wang, and Ali Taylan Cemgil. *Forecasting intermittent and sparse time series: A unified probabilistic framework via deep renewal processes*. PLOS ONE 16(11):e0259764. [Article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0259764).
- **Yang et al. (2022).** Chenghao Yang, Hongyuan Mei, and Jason Eisner. *Transformer Embeddings of Irregularly Spaced Events and Their Participants*. ICLR. [Author manuscript, arXiv:2201.00044](https://arxiv.org/abs/2201.00044).
- **Behrouz et al. (2025).** Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. *Titans: Learning to Memorize at Test Time*. NeurIPS 38. [Proceedings](https://papers.nips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html); [author manuscript, arXiv:2501.00663](https://arxiv.org/abs/2501.00663).
- **Chang et al. (2025).** Yuxin Chang, Alex Boyd, Cao Xiao, Taha Kass-Hout, Parminder Bhatia, Padhraic Smyth, and Andrew Warrington. *Deep Continuous-Time State-Space Models for Marked Event Sequences*. NeurIPS. [Author manuscript, arXiv:2412.19634v2](https://arxiv.org/abs/2412.19634v2).
- **Draxler et al. (2025).** Felix Draxler, Yang Meng, Kai Nelson, Lukas Laskowski, Yibo Yang, Theofanis Karaletsos, and Stephan Mandt. *Transformers for Mixed-type Event Sequences*. NeurIPS 38. [Proceedings](https://proceedings.nips.cc/paper_files/paper/2025/hash/a6c7515ac435277dc92b75a07bb2257c-Abstract-Conference.html). DOI: 10.52202/085713-3833.

## Internal reproducibility index — remove or anonymize for submission

- [Equation-to-code mapping](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_method_efficiency_20260930_v1/code_equation_map.md) and [existing CPU equation checks](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_method_efficiency_20260930_v1/verification.json).
- [Current core results and audit](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_core_ablation_execution_20260928_v1/final_report.md) and [additional comparator adaptation](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_additional_tpp_20260930_v1/README.md).
- [Claim–evidence map and writing notes](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_manuscript_integration_20261001_v1/claim_evidence_and_writing_notes.md), [integration verification](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_manuscript_integration_20261001_v1/verification.json), and [remaining work](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_baseline_reset_20261001_v1/README.md).
