# TitanTPP: Efficient History Correction for Next-Event Quantity Prediction

**Working draft — 1 October 2026.** TitanTPP denotes `titantpp_history_mlp`. The four-dataset comparison is complete: TitanTPP and six common-head TPP encoders each have three-seed validation results, including the finalized Instacart S2P2 and AttNHP runs. The RAF normalization variant is reported separately. Source and checkpoint audits are complete for the included runs. Editorial status and source checks are recorded in the [writing notes](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_manuscript_integration_20261001_v1/claim_evidence_and_writing_notes.md).

## 1. Introduction

Intermittent demand raises two linked forecasting questions: when will the next demand event occur, and how large will it be? In a replenishment setting, these questions concern the timing and magnitude of the next requirement. An event history records both quantities directly as gaps and sizes, making it a natural representation for learning from sporadic observations. Separating demand size from occurrence has a long history in intermittent-demand forecasting. [Croston, 1972](https://link.springer.com/article/10.1057/jors.1972.50). Renewal-process forecasting connects this event-based view to intermittent demand and provides a foundation for neural models of arrivals and sizes. [Türkmen et al., 2021](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0259764).

The representation of that history determines both what a model can learn and how much computation it requires. Attention combines information across observed events; neural memory offers another way to retain historical context. Titans integrates attention with a memory module that learns at test time. For bounded event histories, this raises a practical design question: how much predictive value can a compact feed-forward history path provide while avoiding online memory updates? [Behrouz et al., 2025](https://papers.nips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html).

We introduce TitanTPP, a causal encoder for next-event quantity and duration prediction. A bottleneck residual combines the current and immediately preceding contextual states between two attention blocks. This placement gives the second block a representation explicitly adjusted for their relationship. Persistent vectors and static prototype retrieval complement the correction, while all learned parameters remain fixed during inference. The resulting architecture adds a direct path for recent event information with 6,144 correction parameters.

Our experiments connect this design to prediction quality and computational cost. Across three seeds, TitanTPP improves quantity MAE and RMSE over six common-head TPP adaptations on Taxi and Intermittent. A matched batch-computation experiment also measures lower training-step time and peak allocated training memory than a Titans-MAC event adapter. On RAF spare-parts demand, quantity RMSE is close to the strongest external encoder while MAE favors a recurrent comparator. On Instacart, small MAE gains over S2P2 and AttNHP accompany higher RMSE. Together, these results characterize the trade-off between quantity accuracy, duration modeling, and computation across demand settings.

Our contributions are:

1. **A compact architecture for event-history correction.** We introduce TitanTPP, which refines causal gap–quantity representations through a bottleneck residual between encoder blocks. By combining adjacent contextual states with static retrieval, the model incorporates event-history information without online memory updates.
2. **An empirical study of event-quantity prediction.** Under a shared input, head, loss, and checkpoint-selection protocol, TitanTPP reduces mean quantity RMSE by 15.85% on Taxi and 34.39% on Intermittent relative to the strongest of six external comparators. Three-seed validation comparisons across four datasets and component controls characterize the quantity gains, duration-likelihood trade-offs, and variation across demand settings.
3. **An empirical characterization of computational efficiency.** Matched measurements show that the MAC adapter requires 5.58–5.74 times TitanTPP’s training-step compute time, while TitanTPP reduces peak allocated training memory by approximately 75.5%. Evaluation measurements further reveal how the memory requirements differ between training and evaluation.

## 2. Related work

### 2.1 Event-history representations and duration models

Hawkes processes describe event dependence through self-excitation and mutual excitation. [Hawkes, 1971](https://academic.oup.com/biomet/article-abstract/58/1/83/224809). Neural temporal point processes learn history representations and conditional event distributions. These two design choices provide a useful distinction between the history encoder and the prediction head. [Shchur et al., 2021](https://www.ijcai.org/proceedings/2021/623).

RMTPP summarizes event history with a recurrent neural network, whereas the Neural Hawkes Process maintains a continuous-time LSTM state that controls evolving event intensities. [Du et al., 2016](https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf); [Mei and Eisner, 2017](https://papers.nips.cc/paper_files/paper/2017/hash/6463c88460bd63bbe256e495c63aa40b-Abstract.html). Attention-based alternatives expose interactions between observed events directly. THP applies self-attention to temporal dependencies, SAHP incorporates time intervals into attention-based encoding, and AttNHP constructs attention-based embeddings for irregularly spaced events. [Zuo et al., 2020](https://proceedings.mlr.press/v119/zuo20a.html); [Zhang et al., 2020](https://proceedings.mlr.press/v119/zhang20q.html); [Yang et al., 2022](https://openreview.net/forum?id=Rty5g9imm7H).

State-space representations offer another way to compress history. HiPPO develops polynomial projections for online history compression; S2P2 combines continuous-time state evolution, stochastic jumps, and nonlinear transformations for marked event sequences. Its temporal inductive bias complements the recurrent and attention encoders in our comparison. Section 4 specifies the S2P2 operations retained in the common-head adaptation. [Gu et al., 2020](https://proceedings.neurips.cc/paper/2020/hash/102f0bb6efb3a6128a3c750dd16729be-Abstract.html); [Chang et al., 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ee39348acc798915d2d15a8bbbd417b8-Abstract-Conference.html).

The time distribution also affects likelihood evaluation. Omi et al. model integrated intensity with a neural network and obtain intensity by differentiation. Intensity-free TPPs instead parameterize the conditional inter-event-time distribution through flows or tractable mixtures. TitanTPP follows the direct-distribution approach with a conditional lognormal duration head and an observation model for recorded integer durations. The shared head lets the experiments focus on event-history encoding. [Omi et al., 2019](https://proceedings.neurips.cc/paper/2019/hash/39e4973ba3321b80f37d9b55f63ed8b8-Abstract.html); [Shchur et al., 2020](https://openreview.net/forum?id=HygOjhEYDH).

### 2.2 Numerical event attributes and demand forecasting

Intermittent-demand forecasting has long separated demand occurrence from demand size. Croston estimates these components separately, and subsequent work evaluates Croston-style estimators and develops demand-probability updates that respond to zero-demand periods and obsolescence. These methods motivate the distinction between when demand occurs and how much occurs. [Croston, 1972](https://link.springer.com/article/10.1057/jors.1972.50); [Syntetos and Boylan, 2005](https://www.sciencedirect.com/science/article/pii/S0169207004000792); [Teunter et al., 2011](https://www.sciencedirect.com/science/article/pii/S0377221711004437).

Deep renewal processes connect that distinction to neural arrival-and-size models, including continuous-time formulations. FlexTPP broadens event modeling to heterogeneous attributes, with discrete heads and normalizing flows for continuous values. Numerical event attributes therefore fit within an established research setting. TitanTPP addresses the representation of gap–quantity histories within this setting, pairing a quantity point estimate with a distribution over the next recorded duration. [Türkmen et al., 2021](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0259764); [Draxler et al., 2025](https://proceedings.nips.cc/paper_files/paper/2025/hash/a6c7515ac435277dc92b75a07bb2257c-Abstract-Conference.html).

Neural time-series forecasting provides related approaches to learning across demand histories. DeepAR trains an autoregressive recurrent model across related series for probabilistic forecasting. PatchTST organizes temporal observations into patches with channel-independent processing, while iTransformer attends across variate tokens. These methods illustrate different choices of forecasting target and tokenization. Our event representation concentrates on positive-demand observations and their gaps, and the task predicts the next event's size and duration. Extending it to a fixed calendar horizon would require a rule for accumulating intervening events. [Salinas et al., 2020](https://www.sciencedirect.com/science/article/pii/S0169207019301888); [Nie et al., 2023](https://openreview.net/forum?id=Jbdc0vTOcol); [Liu et al., 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/2ea18fdc667e0ef2ad82b2b4d65147ad-Abstract-Conference.html).

### 2.3 Residual correction and memory access

Residual learning adds a learned transformation to a shortcut path. Bottleneck adapters apply down-projection, a nonlinearity, and up-projection within such a residual, allowing compact task-specific additions to a pretrained network. TitanTPP specializes this structure to event histories by combining the current and immediately preceding contextual states between two jointly trained encoder blocks. Its branches participate according to observed-history availability. [He et al., 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html); [Houlsby et al., 2019](https://proceedings.mlr.press/v97/houlsby19a.html).

Initialization determines the residual's initial effect. ReZero starts each residual path with a zero-valued scalar gate. TitanTPP initializes the branch output matrices to zero, so the correction initially preserves the base encoder's function. The availability mask is deterministic, and the correction learns together with the encoder and prediction heads. [Bachlechner et al., 2021](https://proceedings.mlr.press/v161/bachlechner21a.html).

Memory reading and online parameter adaptation provide distinct computational paths. End-to-end memory networks perform attention-based reads over external memory, whereas Titans introduces neural memory that learns from incoming context at test time. TitanTPP retains learned persistent vectors and a static prototype bank alongside its feed-forward correction, with all parameters fixed during inference. This design motivates the matched comparison against a Titans-MAC event adapter in Section 6, which measures the complete training and evaluation paths. [Sukhbaatar et al., 2015](https://proceedings.neurips.cc/paper/2015/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html); [Behrouz et al., 2025](https://papers.nips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html).

## 3. Method

### 3.1 Prediction task and observed inputs

Let an event history be \(\mathcal H_n=\{(t_i,q_i)\}_{i=1}^{n}\), where \(t_i\) is the recorded event time, \(\Delta_i\) is the supplied recorded inter-event duration, and \(q_i\geq0\) is the event quantity. We predict the duration and quantity of the next event from the observed history. The outputs are a nonnegative quantity point estimate and a distribution over the next recorded duration.

Durations use the dataset’s recorded units and preprocessing conventions; the first history row uses the loader’s positive-gap convention. For a padded sequence, let \(v_i\) indicate a valid row and \(w_i\) indicate an observed row permitted as encoder input. Define \(o_i=v_iw_i\). The two input features and their embedding are

$$
f_i=[\log(1+\Delta_i),\log(1+q_i)]^\top,\qquad
x_i=o_i\left(W_{\mathrm{in}}f_i+b_{\mathrm{in}}+p_i\right). \tag{1}
$$

Here \(p_i\) is a learned positional embedding and the hidden dimension is \(d=64\). In implementation, unobserved inputs are replaced by zero **before** the logarithm and projection. Thus even invalid values at withheld or padded positions cannot enter the representation. Each training window contains an observed prefix and one target row. The target row is withheld from the encoder, and the heads read the last observed state. Its duration and quantity are used only to evaluate the loss.

### 3.2 Causal encoding with persistent vectors

TitanTPP contains two pre-normalized causal attention blocks, built from multi-head attention and layer normalization. [Vaswani et al., 2017](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html); [Ba et al., 2016](https://arxiv.org/abs/1607.06450). For block \(\ell\), each attention head projects the normalized event states into queries, keys and values. A learned persistent bank \(P^{(\ell)}\in\mathbb R^{16\times d}\) is split across the heads and prepended to their keys and values:

$$
K=[P^{(\ell)};K_{\mathrm{events}}],\quad
V=[P^{(\ell)};V_{\mathrm{events}}],\quad
A_i=\operatorname{softmax}\!\left(Q_iK^\top/\sqrt{d_h}+C_i\right)V. \tag{2}
$$

There are four heads and \(d_h=16\). Head indices are suppressed in (2), where the persistent bank denotes the corresponding 16-by-16 head slice. The additive mask \(C_i\) permits every persistent vector and only observed event keys at positions \(j\leq i\). The bank supplies the same raw vectors as keys and values; it is not passed through the event key/value projections. Attention dropout and a learned output projection precede the attention residual. A second residual contains a pre-normalized feed-forward network of width 128 with GELU and dropout. Dropout is 0.1; unobserved query states are zeroed after each residual. Denote the resulting block by \(\mathcal E_\ell\). The first block produces \(H^{(1)}=\mathcal E_1(X)\).

The outer training optimizer learns the persistent vectors, which remain fixed during inference.

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

GELU follows the exact error-function formulation. [Hendrycks and Gimpel, 2016](https://arxiv.org/abs/1606.08415). Although the thresholds differ, all eight branches read the **same immediate predecessor**. The thresholds control availability, not retrieval lag. The divisor is always eight, including when only some branches are active. If no predecessor exists, its vector is defined as zero and every branch is unavailable. When no branch is available, the correction is exactly zero. A withheld valid row resets correction eligibility, whereas padding does not. This eligibility rule does not reset the causal attention history of the encoder.

The module adds \(8(2dr+rd)=6{,}144\) parameters for \(r=4\). We initialize every \(V_b\) to zero. Consequently the correction is zero at initialization; shared encoder and head parameters therefore retain the predictions of the configuration without this correction under the same stochastic state. Module construction preserves the caller's random-number stream. The implementation computes the current and predecessor projections before gathering their low-dimensional values; it is algebraically equivalent to (4).

The bottleneck limits the capacity of the intermediate correction. Both inputs are contextual states from the first causal block, so their combination can incorporate information from the earlier observed history as well as the adjacent event.

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

Equation (6) produces a nonnegative quantity point estimate. At initialization, \(w_q=0\) and the bias gives \(\widehat y=\overline{\log(1+q)}_{\mathrm{train}}\). The initial prediction therefore matches the training mean on the log-transformed scale.

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

The first bin begins at zero, rather than 0.5. The scale \(s\) is 1 hour for Taxi, 3 weeks for Intermittent, 7 days for Instacart, and 6 months for RAF. The implementation evaluates log masses in float64 using stable CDF or survival-function differences. Reported time NLL is the negative log of (8), not the continuous lognormal density at the recorded integer.

### 3.6 Optimization and checkpoint selection

For a minibatch \(\mathcal B\) of history–target pairs, the training objective is

$$
\mathcal L=\frac1{|\mathcal B|}\sum_{n\in\mathcal B}
\left[-\log p(D_{n+1}\mid\mathcal H_n)
+\big(\widehat y_{n+1}-\log(1+q_{n+1})\big)^2\right]. \tag{9}
$$

The quantity term carries weight one, and the objective contains no additional tail loss. Each window contributes one final target. We optimize with AdamW ([Loshchilov and Hutter, 2019](https://openreview.net/forum?id=Bkg6RiCqY7)) at a learning rate \(10^{-3}\), weight decay 0.01, gradient clipping at norm 1, and batch size 128. Training permits at most 300 epochs and stops no earlier than epoch 40, with early-stopping patience of 40. The selected checkpoint is the earliest checkpoint attaining the strictly lowest finite validation RMSE on the raw quantity scale. MAE and time NLL are reported at that same checkpoint; their minima are not selected separately. Seeds are 42, 52 and 62.



![TitanTPP overview and expanded history-correction module](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_architecture_figure_20261001_v2/architecture.png)

**Figure 1.** TitanTPP and its intermediate history correction. **(a)** Observed gaps and quantities pass through two causal encoders, with the correction inserted between them. Each encoder has its own learned persistent bank, \(P^{(1)}\) or \(P^{(2)}\), and final retrieval uses a separate static prototype bank \(M\). Both heads read the last observed state. **(b)** Eight independent bottleneck branches receive the same current–predecessor pair. Each branch applies a bias-free \(128\to4\to64\) transformation with GELU and availability mask \(a_{i,b}\); the outputs are summed, divided by eight, and added to the current state. The ellipsis denotes branches 3–7. Thresholds govern availability, not retrieval lag. With no predecessor, all branches are unavailable and the correction is zero. The \(V_b\) matrices initialize to zero, and all parameters remain fixed during inference. The withheld next event contributes only to the loss. The dashed link expands the highlighted module. [Vector PDF](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_architecture_figure_20261001_v2/architecture.pdf), [editable SVG](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_architecture_figure_20261001_v2/architecture.svg), and [drawing source](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_architecture_figure_20261001_v2/draw.py) are retained for typesetting.

### 3.7 Design rationale and computational structure

TitanTPP separates the roles of contextual encoding and local adjustment. The first block forms representations from observed gaps and quantities. The correction then combines adjacent contextual states through a narrow bottleneck, and the second block integrates the adjusted representations across the history. This arrangement provides a direct feed-forward path for recent event relationships within a contextual encoder.

The residual construction preserves the base function at initialization and limits the added parameter count. Branch availability adapts the participating paths to the amount of observed history. With the fixed divisor of eight, fewer available branches contribute a smaller aggregate correction. The static banks supply shared learned vectors throughout this computation while remaining fixed at inference.

For fixed width and branch count, the correction requires linear dense-projection work in sequence length. The encoder retains the quadratic attention interactions of full causal attention. Section 6 measures the resulting batch-processing cost against the MAC adapter.

## 4. Evaluation protocol

### 4.1 Event construction, splits, and scope

The existing datasets use different definitions of a quantity-bearing event. Taxi counts pickup rows within a spatial cell and an active hour. Intermittent uses positive `order_qty` observations for a site–part sequence. Instacart counts product rows aggregated for a user on the same recorded activity day; it is a proxy for basket size rather than the number of physical units bought or necessarily one order. RAF represents positive monthly spare-parts demand as events, with the number of months between positive observations as the recorded duration. Zero-demand calendar periods are not separate quantity targets in these event representations.

Frozen splits follow event order within each sequence, approximately 70/15/15. All reported performance here is from validation, which also informed architecture development. Taxi sequence selection and Intermittent activity filtering/stratification refer to full observed sequence lengths, yielding retrospectively defined cohorts. Section 7 discusses the implications for prospective evaluation.

**Table 1.** Frozen train/validation target populations and input limits. Maximum sequence length includes the withheld target, so it is not the maximum number of observed inputs. The lookback values use the recorded units, despite the legacy loader field name `lookback_weeks`.

| Dataset | Train targets | Validation targets | Maximum rows | Lookback in recorded units | Duration scale |
|---|---:|---:|---:|---|---|
| Taxi | 38,393 | 8,268 | 256 | 168 hours | 1 hour |
| Intermittent | 393,824 | 86,285 | 256 | 520 weeks | 3 weeks |
| Instacart | 1,991,192 | 503,733 | 64 | 52 recorded days | 7 days; upper code 30 |
| RAF | 25,779 | 6,690 | 84 | 84 months | 6 months |

Models receive the same permitted gap and quantity history, training targets, validation targets, and quantity/time heads. The two observed features constitute the common input interface. Data identities, source revisions, and selection records are preserved for reproducibility. The completed-results tables include all eight RAF configurations across three seeds. The Instacart panel includes all six external encoders across the same three seeds.

### 4.2 Common-head TPP comparisons

Reproducible TPP benchmarking benefits from explicit data interfaces and evaluation programs, as developed in EasyTPP. [Xue et al., 2024](https://openreview.net/forum?id=PJwAkg0z7h). Our comparison isolates event representations through a shared duration–quantity objective for RMTPP, NHP, THP, SAHP, S2P2, and AttNHP. Each encoder receives the common numerical inputs and connects to the same prediction heads. We refer to these models as **common-head adaptations**. Training follows the common objective, checkpoint selector, and maximum epoch budget; Table 2 specifies the additional adapters, and Section 7 discusses the scope of this comparison.

**Table 2.** Disclosed adaptation boundaries for the two added comparators. Official upstream revisions are preserved in the local source receipt.

| Comparator | Retained computation | Connection to the common task | Implementation boundary |
|---|---|---|---|
| S2P2 | HiPPO initialization, complex diagonal state evolution, backward zero-order-hold discretization, event impulses, relative time, normalization/nonlinearity | Numerical gap–quantity projection replaces categorical input embedding; shared heads replace native intensity output | Width 64, two layers, state size 16; the affine-doubling recurrence was checked against the original sequential equations, but has O(L log L) work and is not the original work-efficient scan |
| AttNHP | Query/context attention, sinusoidal time encoding, and tanh query residual | Common numerical input and heads; query at the last observed event’s right limit, without accessing the next target time | Width 64, two layers, one attention head, time-encoding width 16, no feed-forward sublayer |

The retained HiPPO component follows the polynomial-projection framework of [Gu et al., 2020](https://proceedings.neurips.cc/paper/2020/hash/102f0bb6efb3a6128a3c750dd16729be-Abstract.html). S2P2 is pinned to `UCIDataLab/state_space_point_process@c3933240f16a22b43d80d09bda272475526ff24b`; AttNHP to `yangalan123/anhp-andtt@78f754fde61547b84a0749fd7e7e020e7cd6b4d8`. Existing validation checks cover causality, masking, input/output shape, shared head initialization, and correspondence to the retained upstream equations.

### 4.3 Selection, reporting, and reproducibility

Equation (9) fits recorded-duration likelihood and squared log-quantity error. We select checkpoints by validation RMSE on the raw quantity scale to emphasize large absolute quantity errors. MAE complements this criterion by reporting average absolute error, while time NLL measures the fit to recorded durations. All three metrics are evaluated at the same RMSE-selected checkpoint. MAE and RMSE retain the quantity scale, so comparisons are made within each dataset rather than by pooling raw errors across datasets. [Hyndman and Koehler, 2006](https://doi.org/10.1016/j.ijforecast.2006.03.001). Duration NLL evaluates the probability assigned to the recorded observation through the logarithmic score. [Gneiting and Raftery, 2007](https://doi.org/10.1198/016214506000001437).

For completed three-seed groups, we report the arithmetic mean and sample standard deviation over seeds 42, 52, and 62, together with same-seed comparisons. The “B” control retains the causal encoder and learned banks but omits the intermediate correction. Full and its component removals are internal alternatives, while learned gates and active-branch normalization remain separately labeled exploratory variants. All completed conditions and adverse results remain in the source tables.

We retain selected and last checkpoints, their validation re-evaluations, and source identities. The completed core, additional-comparator, and RAF artifacts have passed source and checkpoint integrity checks; the internal reproducibility index links those records.

## 5. Results on the completed validation comparisons

### 5.1 Effect of the history correction

Table 3 compares TitanTPP with B, the same encoder without the intermediate correction. This comparison evaluates the addition of the 6,144-parameter history path. Lower values are better for all three metrics.

**Table 3.** Completed validation results, mean ± sample standard deviation across three seeds. Each run contributes all three metrics from its RMSE-selected checkpoint.

| Dataset | Configuration | Quantity MAE | Quantity RMSE | Time NLL |
|---|---|---:|---:|---:|
| Taxi | B: no correction | 28.7545 ± 0.4526 | 90.5093 ± 0.5207 | 0.7247 ± 0.0668 |
| Taxi | TitanTPP | 25.6237 ± 0.8320 | 79.7111 ± 2.8260 | 1.0387 ± 0.2843 |
| Intermittent | B: no correction | 0.7588 ± 0.0658 | 1.7806 ± 0.0439 | 0.3846 ± 0.1034 |
| Intermittent | TitanTPP | 0.7005 ± 0.0541 | 1.6730 ± 0.0819 | 0.4971 ± 0.2095 |
| Instacart | B: no correction | 3.9932 ± 0.0018 | 5.8796 ± 0.0166 | 2.8065 ± 0.0045 |
| Instacart | TitanTPP | 3.9916 ± 0.0026 | 5.8827 ± 0.0048 | 2.8071 ± 0.0041 |

Adding the correction reduces mean quantity MAE/RMSE by 10.89%/11.93% on Taxi and 7.68%/6.04% on Intermittent. On Instacart, MAE decreases by only about 0.04% and RMSE increases by about 0.05%. Mean time NLL increases on all three datasets. The correction therefore improves quantity accuracy on Taxi and Intermittent while trading off duration likelihood.

### 5.2 Comparison with TPP encoders

Table 4 compares the same TitanTPP architecture with six common-head TPP encoders across four datasets. All 28 dataset–model groups contain three completed seeds, giving 84 runs in the common comparison.

**Table 4.** Validation results across four datasets, mean ± sample standard deviation over seeds 42, 52, and 62. Each run contributes MAE, RMSE, and time NLL from the same RMSE-selected checkpoint. Lower is better. All dataset–model groups are complete. Exploratory variants are reported separately in Section 5.3.

| Dataset | Model | Quantity MAE | Quantity RMSE | Time NLL |
|---|---|---:|---:|---:|
| Taxi | TitanTPP | 25.6237 ± 0.8320 | 79.7111 ± 2.8260 | 1.0387 ± 0.2843 |
| Taxi | RMTPP | 28.5124 ± 0.6344 | 94.7283 ± 3.4006 | 1.5798 ± 0.9400 |
| Taxi | THP | 32.4095 ± 0.8144 | 107.5977 ± 4.5040 | 0.6629 ± 0.0174 |
| Taxi | NHP | 94.2694 ± 1.8752 | 318.8148 ± 5.3011 | 0.6516 ± 0.0021 |
| Taxi | SAHP | 34.2260 ± 0.4539 | 120.5273 ± 1.9996 | 0.6848 ± 0.0146 |
| Taxi | S2P2 | 33.0231 ± 0.5569 | 106.3540 ± 1.9135 | 0.6658 ± 0.0242 |
| Taxi | AttNHP | 35.9811 ± 1.6088 | 127.1290 ± 8.9919 | 0.6876 ± 0.0194 |
| Intermittent | TitanTPP | 0.7005 ± 0.0541 | 1.6730 ± 0.0819 | 0.4971 ± 0.2095 |
| Intermittent | RMTPP | 0.8949 ± 0.0494 | 2.5499 ± 0.1604 | 0.2748 ± 0.0047 |
| Intermittent | THP | 0.8993 ± 0.0661 | 2.7534 ± 0.1784 | 0.2911 ± 0.0045 |
| Intermittent | NHP | 4.5188 ± 0.1706 | 12.8755 ± 1.0405 | 0.4613 ± 0.0969 |
| Intermittent | SAHP | 1.8619 ± 0.0899 | 7.0298 ± 0.7673 | 0.3448 ± 0.0138 |
| Intermittent | S2P2 | 0.9359 ± 0.0086 | 2.5758 ± 0.0766 | 0.2349 ± 0.0189 |
| Intermittent | AttNHP | 1.0413 ± 0.0938 | 3.0799 ± 0.2982 | 0.2946 ± 0.0569 |
| RAF | TitanTPP | 9.2506 ± 0.0855 | 33.9507 ± 0.0588 | 3.5446 ± 0.0669 |
| RAF | RMTPP | 9.1362 ± 0.0557 | 34.5503 ± 0.2506 | 3.5191 ± 0.1463 |
| RAF | THP | 9.3014 ± 0.1819 | 34.7445 ± 0.3204 | 3.3928 ± 0.0187 |
| RAF | NHP | 9.1793 ± 0.0882 | 36.2731 ± 0.3186 | 3.3889 ± 0.0556 |
| RAF | SAHP | 9.2332 ± 0.0209 | 35.4193 ± 0.2130 | 3.7415 ± 0.0745 |
| RAF | S2P2 | 9.2730 ± 0.0517 | 33.9947 ± 0.1261 | 3.4187 ± 0.0284 |
| RAF | AttNHP | 9.1917 ± 0.0434 | 34.8766 ± 0.0676 | 3.9707 ± 0.0609 |
| Instacart | TitanTPP | 3.9916 ± 0.0026 | 5.8827 ± 0.0048 | 2.8071 ± 0.0041 |
| Instacart | RMTPP | 3.9852 ± 0.0052 | 5.8775 ± 0.0213 | 2.8051 ± 0.0048 |
| Instacart | THP | 4.0013 ± 0.0025 | 5.8582 ± 0.0253 | 2.8056 ± 0.0016 |
| Instacart | NHP | 4.4923 ± 0.0304 | 6.7613 ± 0.0538 | 2.8132 ± 0.0080 |
| Instacart | SAHP | 3.9979 ± 0.0099 | 5.8776 ± 0.0067 | 2.8018 ± 0.0013 |
| Instacart | S2P2 | 3.9987 ± 0.0063 | 5.8523 ± 0.0150 | 2.7970 ± 0.0014 |
| Instacart | AttNHP | 3.9958 ± 0.0021 | 5.8648 ± 0.0194 | 2.8080 ± 0.0028 |

On Taxi and Intermittent, TitanTPP has lower quantity MAE and RMSE than every external encoder in all 36 same-seed comparisons (2 datasets × 6 models × 3 seeds). RMTPP has the lowest three-seed mean MAE and RMSE among the external models on both datasets. Relative to RMTPP, TitanTPP reduces MAE/RMSE by 10.13%/15.85% on Taxi and 21.72%/34.39% on Intermittent. The ranking changes for time likelihood, where several external encoders, including S2P2 on both datasets, achieve lower time NLL.

On RAF spare-parts demand, TitanTPP reaches a mean RMSE of 33.9507, close to S2P2’s 33.9947, the lowest external mean on this metric. The difference is 0.13%, with TitanTPP ahead in two of the three paired seeds. RMTPP has the lowest mean MAE at 9.1362, compared with TitanTPP’s 9.2506, which is 1.25% higher; NHP has the lowest mean time NLL. RAF thus provides a demand setting where the quantity gains are small and depend on the error metric.

On Instacart, RMTPP achieves the lowest mean MAE, while S2P2 achieves the lowest mean RMSE and time NLL. TitanTPP’s mean MAE is 0.16% higher than RMTPP’s. Relative to S2P2 and AttNHP, TitanTPP reduces mean MAE by 0.18% and 0.11%, but increases mean RMSE by 0.52% and 0.31%, respectively. TitanTPP has lower MAE in two of three paired seeds against each added comparator, whereas both comparators have lower RMSE in all three seeds. S2P2 also has lower time NLL than TitanTPP in every seed.

Quantity-stratified errors account for the MAE–RMSE trade-off against S2P2. Targets with quantities at most 20 make up 89.37% of the Instacart validation set, and TitanTPP has lower mean MAE and RMSE in each of the two corresponding quantity bins. S2P2 performs better in all three bins above 20. For quantities above 35, mean RMSE is 24.6419 for TitanTPP and 23.4686 for S2P2. The larger squared errors above 20 outweigh TitanTPP’s gains on smaller quantities, producing a higher overall RMSE despite its lower MAE.

Train-only characterization offers a possible explanation for the limited benefit of history correction on Instacart. The median usable Instacart history contains five events, and adjacent log quantities have a correlation of 0.037 after subtracting each user’s mean. The corresponding centered correlations are 0.749 on Taxi and 0.928 on Intermittent. Shorter histories and weaker adjacent association may limit the information available to the correction on Instacart; this interpretation remains a hypothesis about the observed pattern.

### 5.3 Structural alternatives

Among the internal alternatives, Full has slightly lower quantity errors on Taxi, whereas TitanTPP has lower errors on Intermittent. We retain TitanTPP as the common representative across datasets and report Full, Gate, and active-normalization variants separately.

Active normalization replaces the fixed divisor of eight in Equation (4) with the number of available branches, using one when none is available. Table 5 isolates the completed RAF comparison between these two normalization rules.

**Table 5.** RAF normalization comparison, mean ± sample standard deviation across three seeds at the RMSE-selected checkpoint. Active normalization is an exploratory variant; the representative TitanTPP in Table 4 uses the fixed divisor.

| Configuration | Quantity MAE | Quantity RMSE | Time NLL |
|---|---:|---:|---:|
| TitanTPP | 9.2506 ± 0.0855 | 33.9507 ± 0.0588 | 3.5446 ± 0.0669 |
| TitanTPP: active normalization | 9.2322 ± 0.0974 | 33.8979 ± 0.0804 | 3.5631 ± 0.0978 |

On RAF, active normalization reduces mean RMSE by 0.16% and MAE by 0.20%, while mean time NLL increases from 3.5446 to 3.5631. Each quantity metric improves in two of the three seeds. The variant also has the lowest mean RMSE among the eight RAF configurations. Its effects differ across datasets: the seed-42 comparisons improve quantity error on Taxi but worsen it on Intermittent. These results support treating normalization as a structural sensitivity analysis while keeping the same representative architecture throughout the external comparison.

Internal evidence: [Taxi, Intermittent, and RAF comparison](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_completed_external_comparison_20261001_v1/report.md), [final Instacart comparison and audit](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_instacart_final_analysis_20261001_v1/report.md), [RAF per-seed and stratified results](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_raf_execution_20261001_v1/report.md), [5080 integrity audit](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_completed_5080_audit_20261001_v1/report.md), and [train-data characterization](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_data_characteristics_20260928_v1/README.md). These local links are development provenance and will be converted to anonymized supplementary references for submission.

## 6. Matched computational cost against Titans-MAC

### 6.1 Measurement protocol

We measured TitanTPP and the Titans-MAC event adapter on one RTX 5080, using identical cached training-input batches, padding length 256, batch size 128, and common heads, loss, outer optimizer, and precision. Each dataset uses 4,736 distinct target windows: five warm-up batches followed by 32 measured batches. A dedicated seed-42 permutation selects this fixed set of windows. The measurement runtime uses Python 3.12.13, PyTorch 2.11.0+cu130 ([Paszke et al., 2019](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html)), CUDA 13.0, float32 model computation, disabled TF32, and deterministic execution settings.

Each dataset/model/path combination has three independent fresh-process repetitions, all using the same seed. Each repetition measures the same fixed computational workload. Within each repetition, latency is the median of the 32 synchronized compute intervals. Tables report the mean and sample standard deviation of those three medians. Speed ratios are calculated within each paired repetition before averaging. The model order is reversed in the second repetition to reduce a fixed-order effect.

Training timing includes the forward pass, shared heads and loss, gradient clearing, backward pass, gradient clipping, and the AdamW update. Evaluation timing includes the forward pass, heads, and target loss, without an outer optimizer update. The MAC adapter retains observed-history memory updates and internally enables gradients where its update requires them. The evaluation interval therefore covers the full forward-and-loss path. Host-to-device transfer is measured separately and excluded from the principal compute interval; GPU synchronization bounds each measured interval.

The MAC adapter uses a fixed-shape compiled scan and inner-update gradient clipping at norm one. No new compiled graphs appeared during the 32 measured batches. Compilation and initialization remain separate costs: the first cold MAC training batch took 49.920 seconds and the first cold evaluation batch 10.744 seconds. These cold first-batch costs include initialization and are reported separately from the warm batch measurements.

### 6.2 Observed time and memory

**Table 6.** Synchronized compute time in milliseconds per batch; mean ± sample standard deviation across three repetitions of the within-repetition median. The final column is the paired MAC/TitanTPP time ratio.

| Dataset | Path | TitanTPP (ms) | MAC adapter (ms) | MAC / TitanTPP |
|---|---|---:|---:|---:|
| Taxi | Training step | 31.447 ± 0.106 | 180.582 ± 6.056 | 5.74 ± 0.18 |
| Taxi | Evaluation batch | 7.632 ± 0.009 | 64.827 ± 0.622 | 8.49 ± 0.08 |
| Intermittent | Training step | 32.157 ± 0.023 | 179.472 ± 0.555 | 5.58 ± 0.01 |
| Intermittent | Evaluation batch | 7.909 ± 0.030 | 64.916 ± 0.645 | 8.21 ± 0.07 |

**Table 7.** Mean peak GPU memory in MiB across the same three repetitions. Peak counters are reset after warm-up. Allocated and reserved memory are reported separately.

| Dataset | Path | TitanTPP allocated | MAC allocated | TitanTPP reserved | MAC reserved |
|---|---|---:|---:|---:|---:|
| Taxi | Training | 1869.5 | 7631.3 | 2042.0 | 7670.0 |
| Taxi | Evaluation | 389.2 | 97.8 | 416.0 | 151.3 |
| Intermittent | Training | 1869.5 | 7631.0 | 2042.0 | 7646.0 |
| Intermittent | Evaluation | 389.2 | 97.8 | 416.0 | 130.0 |

For these inputs and implementations, the MAC/TitanTPP time ratio is 5.58–5.74 for training steps and 8.21–8.49 for evaluation batches. TitanTPP’s peak allocated training memory is about 75.5% lower, but its evaluation allocation is about 3.98 times that of the adapter. The two tested length-256 configurations have 96,003 and 122,310 trainable parameters respectively, a 21.5% reduction for TitanTPP. These parameter counts apply to the configured length-256 positional embeddings.

![Matched batch cost and GPU memory](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_efficiency_5080_20260930_v1/paired_efficiency.png)

**Figure 2.** Matched RTX 5080 batch measurements. Timing bars summarize the three repetition medians; error bars show their sample standard deviations. The memory panel shows peak allocated memory and exposes the reversal between training and evaluation. Both datasets use identical input batches across the two models. Evaluation includes target-loss computation. The [vector figure](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_efficiency_5080_20260930_v1/paired_efficiency.pdf) and measurements are reused unchanged.

TitanTPP processes the measured batches faster in both paths and requires substantially less allocated memory during training. Its higher evaluation allocation illustrates why training and evaluation costs should be reported separately. The [measurement report](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_efficiency_5080_20260930_v1/report.md) provides the per-batch records.

## 7. Limitations

The reported results come from validation partitions used during architecture development. An independent final evaluation requires fixed model and checkpoint choices, reporting metrics, and uncertainty procedures, together with a review of prior split access. Three-seed means, sample standard deviations, and paired wins describe repeatability; they are not a significance test. The quantity gains vary across datasets: RAF shows a small RMSE difference and a reversal under MAE, while Instacart shows small MAE gains over the added comparators alongside higher RMSE in every paired seed.

The component controls evaluate particular architectural choices rather than their separate causal contributions. Adding the correction changes capacity as well as the computational path. Rank, insertion position, availability thresholds, and zero initialization have not each been isolated in matched experiments. The existing static-retrieval removal experiment uses Full, so the necessity of that component in TitanTPP remains open. Similarly, the history statistics in Section 5 do not establish causation or exclude nonlinear predictive information on Instacart.

The external panel evaluates six encoders with adapted numerical inputs and common heads. Its shared objective and epoch limit do not equalize parameter counts, tuning opportunity, or runtime, and its rankings are specific to those adaptations. Deep renewal processes and FlexTPP provide relevant context beyond the present panel. Broader claims about numerical-mark forecasting would require comparisons addressing that wider setting.

The efficiency study measures short, warmed batch workloads on one GPU. Full-epoch duration, time to a matched accuracy, energy consumption, and deployment inference latency require separate measurements. Evaluation here includes target-loss computation. Because the experiment compares complete backbones, the observed time difference cannot be attributed solely to removing online memory updates. Evaluation memory is also higher for TitanTPP, despite the lower training allocation.

Finally, the task predicts the next recorded event rather than demand accumulated over a fixed calendar horizon. The training objective combines duration NLL with point-regression loss for quantity, so it is not a full joint marked-event likelihood. The input omits item identities, product composition, and calendar covariates. Instacart uses a basket-size proxy and upper-coded durations; the local Intermittent source still requires upstream provenance and redistribution documentation. Per-sequence temporal splits and retrospective cohort selection constrain prospective deployment claims. Inventory-cost improvements and long-horizon forecasting performance remain beyond the present evaluation.

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

1. J. D. Croston (1972). *Forecasting and Stock Control for Intermittent Demands*. Journal of the Operational Research Society 23(3):289–303. [Source](https://link.springer.com/article/10.1057/jors.1972.50).
2. Ali Caner Türkmen, Tim Januschowski, Yuyang Wang, and Ali Taylan Cemgil (2021). *Forecasting intermittent and sparse time series: A unified probabilistic framework via deep renewal processes*. PLOS ONE 16(11):e0259764. [Source](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0259764).
3. Ali Behrouz, Peilin Zhong, and Vahab Mirrokni (2025). *Titans: Learning to Memorize at Test Time*. Advances in Neural Information Processing Systems 38:125925–125962. [Source](https://papers.nips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html).
4. Alan G. Hawkes (1971). *Spectra of some self-exciting and mutually exciting point processes*. Biometrika 58(1):83–90. [Source](https://academic.oup.com/biomet/article-abstract/58/1/83/224809).
5. Oleksandr Shchur, Ali Caner Türkmen, Tim Januschowski, and Stephan Günnemann (2021). *Neural Temporal Point Processes: A Review*. Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence:4585–4593. [Source](https://www.ijcai.org/proceedings/2021/623).
6. Nan Du, Hanjun Dai, Rakshit Trivedi, Utkarsh Upadhyay, Manuel Gomez-Rodriguez, and Le Song (2016). *Recurrent Marked Temporal Point Processes: Embedding Event History to Vector*. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining:1555–1564. [Source](https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf).
7. Hongyuan Mei and Jason Eisner (2017). *The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process*. Advances in Neural Information Processing Systems 30. [Source](https://papers.nips.cc/paper_files/paper/2017/hash/6463c88460bd63bbe256e495c63aa40b-Abstract.html).
8. Simiao Zuo, Haoming Jiang, Zichong Li, Tuo Zhao, and Hongyuan Zha (2020). *Transformer Hawkes Process*. Proceedings of the 37th International Conference on Machine Learning 119:11692–11702. [Source](https://proceedings.mlr.press/v119/zuo20a.html).
9. Qiang Zhang, Aldo Lipani, Omer Kirnap, and Emine Yilmaz (2020). *Self-Attentive Hawkes Process*. Proceedings of the 37th International Conference on Machine Learning 119:11183–11193. [Source](https://proceedings.mlr.press/v119/zhang20q.html).
10. Chenghao Yang, Hongyuan Mei, and Jason Eisner (2022). *Transformer Embeddings of Irregularly Spaced Events and Their Participants*. International Conference on Learning Representations. [Source](https://openreview.net/forum?id=Rty5g9imm7H).
11. Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, and Christopher Ré (2020). *HiPPO: Recurrent Memory with Optimal Polynomial Projections*. Advances in Neural Information Processing Systems 33. [Source](https://proceedings.neurips.cc/paper/2020/hash/102f0bb6efb3a6128a3c750dd16729be-Abstract.html).
12. Yuxin Chang, Alex Boyd, Cao Xiao, Taha Kass-Hout, Parminder Bhatia, Padhraic Smyth, and Andrew Warrington (2025). *Deep Continuous-Time State-Space Models for Marked Event Sequences*. Advances in Neural Information Processing Systems 38:180555–180596. [Source](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ee39348acc798915d2d15a8bbbd417b8-Abstract-Conference.html).
13. Takahiro Omi, Naonori Ueda, and Kazuyuki Aihara (2019). *Fully Neural Network based Model for General Temporal Point Processes*. Advances in Neural Information Processing Systems 32. [Source](https://proceedings.neurips.cc/paper/2019/hash/39e4973ba3321b80f37d9b55f63ed8b8-Abstract.html).
14. Oleksandr Shchur, Marin Biloš, and Stephan Günnemann (2020). *Intensity-Free Learning of Temporal Point Processes*. International Conference on Learning Representations. [Source](https://openreview.net/forum?id=HygOjhEYDH).
15. Aris A. Syntetos and John E. Boylan (2005). *The accuracy of intermittent demand estimates*. International Journal of Forecasting 21(2):303–314. [Source](https://www.sciencedirect.com/science/article/pii/S0169207004000792).
16. Ruud H. Teunter, Aris A. Syntetos, and M. Zied Babai (2011). *Intermittent demand: Linking forecasting to inventory obsolescence*. European Journal of Operational Research 214(3):606–615. [Source](https://www.sciencedirect.com/science/article/pii/S0377221711004437).
17. Felix Draxler, Yang Meng, Kai Nelson, Lukas Laskowski, Yibo Yang, Theofanis Karaletsos, and Stephan Mandt (2025). *Transformers for Mixed-type Event Sequences*. Advances in Neural Information Processing Systems 38:127441–127473. [Source](https://proceedings.nips.cc/paper_files/paper/2025/hash/a6c7515ac435277dc92b75a07bb2257c-Abstract-Conference.html).
18. David Salinas, Valentin Flunkert, Jan Gasthaus, and Tim Januschowski (2020). *DeepAR: Probabilistic forecasting with autoregressive recurrent networks*. International Journal of Forecasting 36(3):1181–1191. [Source](https://www.sciencedirect.com/science/article/pii/S0169207019301888).
19. Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*. International Conference on Learning Representations. [Source](https://openreview.net/forum?id=Jbdc0vTOcol).
20. Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, and Mingsheng Long (2024). *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*. International Conference on Learning Representations. [Source](https://proceedings.iclr.cc/paper_files/paper/2024/hash/2ea18fdc667e0ef2ad82b2b4d65147ad-Abstract-Conference.html).
21. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun (2016). *Deep Residual Learning for Image Recognition*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition:770–778. [Source](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).
22. Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly (2019). *Parameter-Efficient Transfer Learning for NLP*. Proceedings of the 36th International Conference on Machine Learning 97:2790–2799. [Source](https://proceedings.mlr.press/v97/houlsby19a.html).
23. Thomas Bachlechner, Bodhisattwa Prasad Majumder, Henry Mao, Gary Cottrell, and Julian McAuley (2021). *ReZero is all you need: fast convergence at large depth*. Proceedings of the Thirty-Seventh Conference on Uncertainty in Artificial Intelligence 161:1352–1361. [Source](https://proceedings.mlr.press/v161/bachlechner21a.html).
24. Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus (2015). *End-To-End Memory Networks*. Advances in Neural Information Processing Systems 28. [Source](https://proceedings.neurips.cc/paper/2015/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html).
25. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems 30. [Source](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html).
26. Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton (2016). *Layer Normalization*. arXiv:1607.06450. [Source](https://arxiv.org/abs/1607.06450).
27. Dan Hendrycks and Kevin Gimpel (2016). *Gaussian Error Linear Units (GELUs)*. arXiv:1606.08415. [Source](https://arxiv.org/abs/1606.08415).
28. Ilya Loshchilov and Frank Hutter (2019). *Decoupled Weight Decay Regularization*. International Conference on Learning Representations. [Source](https://openreview.net/forum?id=Bkg6RiCqY7).
29. Siqiao Xue, Xiaoming Shi, Zhixuan Chu, Yan Wang, Hongyan Hao, Fan Zhou, Caigao Jiang, Chen Pan, James Y. Zhang, Qingsong Wen, Jun Zhou, and Hongyuan Mei (2024). *EasyTPP: Towards Open Benchmarking Temporal Point Processes*. International Conference on Learning Representations. [Source](https://openreview.net/forum?id=PJwAkg0z7h).
30. Rob J. Hyndman and Anne B. Koehler (2006). *Another look at measures of forecast accuracy*. International Journal of Forecasting 22(4):679–688. [Source](https://doi.org/10.1016/j.ijforecast.2006.03.001).
31. Tilmann Gneiting and Adrian E. Raftery (2007). *Strictly Proper Scoring Rules, Prediction, and Estimation*. Journal of the American Statistical Association 102(477):359–378. [Source](https://doi.org/10.1198/016214506000001437).
32. Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. Advances in Neural Information Processing Systems 32. [Source](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html).

## Internal reproducibility index — remove or anonymize for submission

- [Equation-to-code mapping](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_method_efficiency_20260930_v1/code_equation_map.md) and [existing CPU equation checks](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_method_efficiency_20260930_v1/verification.json).
- [Final Instacart three-seed results, selected/last comparisons, and original-file audit](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_instacart_final_analysis_20261001_v1/report.md).
- [RAF 24-condition results and audit](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_raf_execution_20261001_v1/report.md).
- [Current core results and audit](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_core_ablation_execution_20260928_v1/final_report.md) and [additional comparator adaptation](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_additional_tpp_20260930_v1/README.md).
- [Claim–evidence map and writing notes](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_manuscript_integration_20261001_v1/claim_evidence_and_writing_notes.md), [integration verification](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_manuscript_integration_20261001_v1/verification.json), and [remaining work](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_baseline_reset_20261001_v1/README.md).
