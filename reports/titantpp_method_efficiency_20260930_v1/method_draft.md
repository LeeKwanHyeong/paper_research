# TitanTPP: method draft

**Draft status:** implementation-aligned methods text, 30 September 2026. The paper name *TitanTPP* denotes the selected `titantpp_history_mlp` configuration. Historical experiment IDs remain unchanged. Equations below describe that configuration, without either of the subsequently tested gates. This draft does not assert an unmeasured speedup over Titans-MAC.

## 1. Prediction task and observed inputs

Let an event history be \(\mathcal H_n=\{(t_i,q_i)\}_{i=1}^{n}\), where \(t_i\) is the recorded event time, \(\Delta_i=t_i-t_{i-1}\) is the supplied inter-event duration, and \(q_i\geq0\) is the event quantity. We predict the duration and quantity of the next event from the observed history. Quantity is a numerical regression target; it need not be an event-type label. The present experiments evaluate next-event prediction and do not establish a method for arbitrary-horizon forecasts on a regular time grid.

For a padded sequence, let \(v_i\) indicate a valid row and \(w_i\) indicate an observed row permitted as encoder input. Define \(o_i=v_iw_i\). The two input features and their embedding are

$$
f_i=[\log(1+\Delta_i),\log(1+q_i)]^\top,\qquad
x_i=o_i\left(W_{\mathrm{in}}f_i+b_{\mathrm{in}}+p_i\right). \tag{1}
$$

Here \(p_i\) is a learned positional embedding and the hidden dimension is \(d=64\). In implementation, unobserved inputs are replaced by zero **before** the logarithm and projection. Thus even invalid values at withheld or padded positions cannot enter the representation. Each training window contains an observed prefix and one target row. The target row is withheld from the encoder, and the heads read the last observed state. Its duration and quantity are used only to evaluate the loss.

## 2. Causal encoding with persistent vectors

TitanTPP contains two pre-normalized causal attention blocks. For block \(\ell\), each attention head projects the normalized event states into queries, keys and values. A learned persistent bank \(P^{(\ell)}\in\mathbb R^{16\times d}\) is split across the heads and prepended to their keys and values:

$$
K=[P^{(\ell)};K_{\mathrm{events}}],\quad
V=[P^{(\ell)};V_{\mathrm{events}}],\quad
A_i=\operatorname{softmax}\!\left(Q_iK^\top/\sqrt{d_h}+C_i\right)V. \tag{2}
$$

There are four heads and \(d_h=16\). Head indices are suppressed in (2), where the persistent bank denotes the corresponding 16-by-16 head slice. The additive mask \(C_i\) permits every persistent vector and only observed event keys at positions \(j\leq i\). The bank supplies the same raw vectors as keys and values; it is not passed through the event key/value projections. Attention dropout and a learned output projection precede the attention residual. A second residual contains a pre-normalized feed-forward network of width 128 with GELU and dropout. Dropout is 0.1; unobserved query states are zeroed after each residual. Denote the resulting block by \(\mathcal E_\ell\). The first block produces \(H^{(1)}=\mathcal E_1(X)\).

These persistent vectors are ordinary model parameters learned by the outer training optimizer. They remain fixed during inference. They do not implement an online neural-memory update.

## 3. A bottleneck correction from adjacent observed states

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

## 4. Static prototype retrieval and shared prediction state

A separate bank \(M\in\mathbb R^{64\times d}\) provides static retrieval after the second block. For each observed query, select four prototypes by cosine similarity, then add their unweighted mean:

$$
S_i=\operatorname{TopK}_{j,\,k=4}\operatorname{cos}(h_i^{(2)},m_j),\qquad
z_i=o_i\left(h_i^{(2)}+\frac14\sum_{j\in S_i}m_j\right). \tag{5}
$$

Normalization is used for selection only. The vectors added to the residual are the raw prototypes, not their normalized versions or a softmax-weighted average. Keys and values are tied. This bank is distinct from the persistent vectors inside attention. Both banks are learned during training and receive no online writes during inference. The last observed state \(z_n\) is shared by the time and quantity heads.

## 5. Numerical quantity and recorded-duration heads

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

Recorded durations follow a positive-integer observation model, \(D=\max(1,\operatorname{round}(T))\), with an additional upper code \(K=30\) for Instacart. Writing \(F_n\) for the CDF of \(T\), the observed probability mass is

$$
p(D=d\mid\mathcal H_n)=
\begin{cases}
F_n(1.5),&d=1,\\
1-F_n(K-0.5),&d=K\text{ when upper coding applies},\\
F_n(d+0.5)-F_n(d-0.5),&\text{otherwise}.
\end{cases} \tag{8}
$$

The first bin begins at zero, rather than 0.5. The scale \(s\) is 1 hour for Taxi, 3 weeks for Intermittent, and 7 days for Instacart. The implementation evaluates log masses in float64 using stable CDF or survival-function differences. Reported time NLL is the negative log of (8), not the continuous lognormal density at the recorded integer.

## 6. Optimization and checkpoint selection

For a minibatch \(\mathcal B\) of history–target pairs, the training objective is

$$
\mathcal L=\frac1{|\mathcal B|}\sum_{n\in\mathcal B}
\left[-\log p(D_{n+1}\mid\mathcal H_n)
+\big(\widehat y_{n+1}-\log(1+q_{n+1})\big)^2\right]. \tag{9}
$$

The quantity term carries weight one, and the objective contains no additional tail loss. Each window contributes one final target. We optimize with AdamW at a learning rate \(10^{-3}\), weight decay 0.01, gradient clipping at norm 1, and batch size 128. Training permits at most 300 epochs and stops no earlier than epoch 40, with early-stopping patience of 40. The selected checkpoint is the earliest checkpoint attaining the strictly lowest finite validation RMSE on the raw quantity scale. MAE and time NLL are reported at that same checkpoint; their minima are not selected separately. Seeds are 42, 52 and 62.

The representative architecture was selected after examining validation results. These results support model development; they are not an untouched test estimate. An independently authorized final evaluation remains separate from this methods draft.

## 7. Relationship to Titans and scope of the efficiency claim

Titans introduces neural long-term memory that learns at test time. TitanTPP instead uses fixed learned banks and a feed-forward history correction, with no online associative optimization or surprise-momentum state during inference. This distinguishes the inference computation, but does not imply that all memory has been removed. The relevant experimental comparator is the repository's **Titans-MAC event adapter**, whose adaptation and implementation must be disclosed, rather than an asserted reproduction of every original Titans experiment. [Titans, original paper](https://arxiv.org/abs/2501.00663).

The correction requires linear dense projection work in sequence length for fixed width and branch count. The encoder still performs full causal attention; removing online memory updates does not make the whole backbone linear in sequence length. Existing timings do not establish a matched speedup, lower peak GPU memory, or equal-accuracy efficiency advantage over the MAC adapter. Those claims require the measurements specified in the accompanying efficiency audit.

## Notation and fixed dimensions

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

## Figure caption

**Figure 1.** TitanTPP encodes observed gaps and quantities with two causal attention blocks. An eight-branch bottleneck residual combines current and immediately preceding contextual states between the blocks. Branch availability depends on observed history length; all branches use lag one. A static top-four prototype residual supplies the shared state for duration likelihood and quantity regression. Persistent and retrieval banks are learned offline and fixed at inference. The target row is excluded from the encoder. Rendered figure: [architecture.svg](architecture.svg); [editable Mermaid source](architecture.mmd).

Implementation trace and source hashes: [code_equation_map.md](code_equation_map.md), [verification.json](verification.json). Fixed contract: `eff125f587f7a9a8097d5d43b3d688eac481abb78bb6006b5aa4ab1b0ec2a88d`.
