# Methodology

## Matched history encoders

The comparison separates the quantity formulation from the history encoder. RMTPP-matched, THP-
matched, and TitanTPP receive the same observed mark, inter-event time, and quantity residual, and
they share the mark-residual reconstruction described in the previous section. Their training split,
next-event target, output losses, and checkpoint rule are also fixed. Consequently, differences
between these models can be attributed to the encoder and its associated parameterization rather
than to a different quantity target. RMTPP-original remains a separate reference because it omits
the residual input and direct quantity term.

For an observed event \(i\), the input combines a mark embedding with time and quantity features.
TitanTPP constructs

$$
x_i=
\left[
E_m(m_i)\,\|\,\log(1+\Delta t_i)\,\|\,W_r r_i
\right],
$$

where \(E_m\) denotes the mark embedding and \(W_r\) projects the scalar residual. RMTPP-matched
uses the same mark and residual features with its recurrent time input. Padding positions are
masked, and the target residual appended by the data loader is replaced before encoding. Thus, each
prediction depends only on quantities already present in the observed history.

## RMTPP history representation

RMTPP maps the input sequence to recurrent states

$$
h_i=\operatorname{GRU}(x_i,h_{i-1}).
$$

The final state available at a transition summarizes all preceding events in a fixed-dimensional
vector. Mark logits, the time-intensity intercept, and the residual prediction are linear functions
of this state. This construction supplies a compact autoregressive baseline, but information from a
distant event must propagate through every intermediate recurrent update. RMTPP-matched retains
this encoder while adopting the proposed residual input and hybrid quantity objective.

## TitanTPP encoder

TitanTPP projects each input token to dimension \(d\), adds a learned positional embedding, and
passes the sequence through \(L\) pre-normalized causal memory-attention blocks. A block contains
multi-head attention followed by a feed-forward network, with residual connections around both
operations. For the frozen models, \(L=2\). Causal masking permits a token to attend to its observed
prefix but not to later events.

Each attention layer also contains learned persistent memory vectors. If \(Z\in\mathbb{R}^{n\times
d}\) denotes the current sequence representation and \(P\in\mathbb{R}^{n_p\times d}\) denotes
persistent memory, keys and values are constructed from \([P;Z]\), while queries are derived from
\(Z\). The sequence portion of attention remains causal; persistent memory is visible at every
position. The frozen comparison does not update contextual memory across mini-batches, which avoids
carrying information between unrelated demand series.

After the final encoder block, Local Memory Matching (LMM) retrieves vectors from a second learned
memory bank \(B\in\mathbb{R}^{n_b\times d}\). For each encoded token \(z_i\), cosine similarity
selects the \(K\) nearest memory entries,

$$
\mathcal{I}_i=\operatorname{TopK}_{j}
\frac{z_i^{\top}B_j}{\lVert z_i\rVert_2\lVert B_j\rVert_2},
$$

and the retrieved mean is added to the token representation,

$$
\widetilde z_i=z_i+\frac{1}{K}\sum_{j\in\mathcal{I}_i}B_j.
$$

This static memory path differs from test-time memory adaptation. All memory vectors are learned
during ordinary training and remain fixed during validation and test evaluation.

## Prediction heads

The mark head maps \(\widetilde z_i\) to logits over the real marks and the padding class. Padding
is removed before expected-quantity reconstruction. The time head follows the RMTPP intensity
parameterization with an intercept \(a_i=v_t^{\top}\widetilde z_i+b_t\) and a positive slope
\(w=\operatorname{softplus}(w_{\mathrm{raw}})+\epsilon\). This shared time head keeps the event-time
model comparable across RMTPP-matched and TitanTPP.

TitanTPP V2 predicts a single residual from each encoded state. Because the same residual is paired
with every candidate mark in the expected-quantity decoder, the mark probabilities determine the
coarse scale and the residual supplies within-scale variation. The hybrid objective combines the
marked point-process likelihood with residual and quantity Huber losses.

## Scale-aware residual experts and gradient routing

Taxi exhibits a much wider quantity range than the other datasets. TitanTPP V3b therefore replaces
the shared residual head with mark-conditioned experts,

$$
\hat r_{i,k}=g_{\mathrm{shared}}(\widetilde z_i)
+g_{\mathrm{delta},k}(\widetilde z_i).
$$

The delta heads are initialized at zero, so the model begins from the shared-head prediction. During
training, the residual loss selects the expert associated with the observed next mark. Quantity
reconstruction combines every expert with its predicted mark probability.

V3b detaches the probability gate only for the quantity loss:

$$
\widehat q^{\mathrm{V3b}}_{i+1}
=\sum_{k=0}^{M}\operatorname{sg}(p_{i,k})
\,b^{k+\hat r_{i,k}},
$$

where \(\operatorname{sg}(\cdot)\) is the stop-gradient operator. The forward quantity estimate is
unchanged, but \(\mathcal{L}_{\mathrm{qty}}\) cannot update the mark logits through the gate.
Categorical cross-entropy remains responsible for mark discrimination, while the residual experts
and their encoder path continue to receive quantity supervision. The Taxi V2 control retains the
same Titan encoder and memory dimensions with a shared residual head and coupled gate, which isolates
the contribution of the expert and routing changes.

## Optimization and model selection

All matched models are optimized with AdamW at a learning rate of \(10^{-3}\), a batch size of 128,
weight decay 0.01, and gradient clipping at 1.0. Training begins with a 300-epoch budget for seeds 42,
52, and 62. The selected checkpoint minimizes validation total NLL, defined as the sum of marker and
time NLL; quantity and point-error metrics do not determine checkpoint selection. A dataset is
extended to epoch 800 only when the preregistered late-convergence rule is met. Development remains
validation-only until model identities, continuation decisions, and checkpoint selection are frozen.

## Draft status

- The architecture and equations reflect source revision
  `726aa64ab0b5478646d11be36fc19dcb224d417e` and the frozen T2 contract.
- Performance comparisons will be added after strict e300 qualification and any required e800
  continuation.
- The held-out test remains outside this draft.
