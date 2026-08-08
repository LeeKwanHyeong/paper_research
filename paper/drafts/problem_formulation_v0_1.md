# Problem formulation

## Event representation

Let a demand sequence contain positive events
\(\mathcal{S}=\{e_i\}_{i=1}^{N}\), where \(e_i=(t_i,q_i)\),
\(t_i\in\mathbb{R}_{+}\), and \(q_i>0\). The inter-event time is

$$
\Delta t_i=t_i-t_{i-1}, \qquad i\geq 2.
$$

The observed history immediately before event \(i+1\) is
\(\mathcal{H}_i=\{(t_j,q_j)\}_{j=1}^{i}\). Given this history, the forecasting
task estimates the next inter-event time \(\Delta t_{i+1}\) and quantity \(q_{i+1}\).
Only positive-demand observations enter the event sequence. Zero-demand intervals remain encoded in
\(\Delta t\), so removing explicit zero rows does not remove the waiting time between events.

## Magnitude-factorized quantity mark

A categorical mark alone can identify a quantity range but cannot recover variation within that
range. We therefore decompose each positive quantity into a coarse mark and a continuous residual.
Given a dataset-specific base \(b>1\) and the largest regular mark index \(M\), define

$$
m_i=\min\left(\left\lfloor\log_b q_i\right\rfloor,M\right),
\qquad
r_i=\log_b q_i-m_i.
$$

For non-tail observations, \(r_i\in[0,1)\). When \(\lfloor\log_b q_i\rfloor>M\),
the mark is clipped to \(M\), but the residual is allowed to exceed one. The mapping remains
lossless for every positive quantity because

$$
q_i=b^{m_i+r_i}.
$$

Intermittent and Instacart employ \(b=2\), while Taxi employs \(b=10\). These bases determine the
resolution of the categorical scale and were fixed before the final comparison. A padding index is
added for batched sequence processing, but it is excluded from the next-mark distribution and
quantity decoder.

Under this construction, the sequence can be written as

$$
\widetilde{\mathcal{S}}
=\{(t_i,m_i,r_i)\}_{i=1}^{N}.
$$

The mark \(m_i\) describes the order of magnitude, and \(r_i\) retains the continuous location
within that scale. The two terms are predicted jointly rather than treating the mark as a proxy for
the complete quantity.

## Conditional event model

An encoder maps the observed marked history to a state \(h_i\). The next-mark head defines

$$
p_{i,k}=P(m_{i+1}=k\mid\mathcal{H}_i),
\qquad k\in\{0,\ldots,M\}.
$$

The time head follows the RMTPP conditional-intensity form

$$
\lambda(\tau\mid h_i)
=\exp\left(a_i+w\tau\right),
\qquad
a_i=v_t^{\top}h_i+b_t,
$$

where \(\tau\geq0\) denotes elapsed time after event \(i\), and \(w>0\) is enforced with a
softplus transformation. Its conditional log-density is

$$
\log f(\Delta t_{i+1}\mid h_i)
=a_i+w\Delta t_{i+1}
-\frac{\exp(a_i)}{w}
\left[\exp(w\Delta t_{i+1})-1\right].
$$

The shared residual head predicts \(\hat r_i=g_r(h_i)\). For the scale-aware variant, the model
instead predicts one candidate residual per real mark,
\(\hat r_{i,k}=g_{r,k}(h_i)\). Training evaluates residual error against the head selected by the
observed next mark.

## Differentiable quantity reconstruction

Argmax decoding would block the quantity loss from the categorical distribution. We instead form an
expected quantity over all real marks. With a shared residual head,

$$
\widehat q_{i+1}
=\sum_{k=0}^{M}p_{i,k}b^{k+\hat r_i}.
$$

The mark-conditioned expert variant replaces \(\hat r_i\) with \(\hat r_{i,k}\):

$$
\widehat q_{i+1}
=\sum_{k=0}^{M}p_{i,k}b^{k+\hat r_{i,k}}.
$$

Both expressions preserve a differentiable path from the reconstructed quantity to the residual
prediction. In the coupled configuration, quantity supervision also updates the mark logits through
\(p_{i,k}\). The detached configuration evaluates the same forward expression but treats
\(p_{i,k}\) as constant for the quantity term.

## Training objective

For a valid next-event transition, the marker and time losses are

$$
\mathcal{L}_{\mathrm{mark}}
=-\log p_{i,m_{i+1}},
\qquad
\mathcal{L}_{\mathrm{time}}
=-\log f(\Delta t_{i+1}\mid h_i).
$$

The continuous residual and reconstructed quantity are trained with Huber losses,

$$
\mathcal{L}_{\mathrm{res}}
=\operatorname{Huber}(\hat r_{i,m_{i+1}},r_{i+1}),
$$

$$
\mathcal{L}_{\mathrm{qty}}
=\operatorname{Huber}\left(
\frac{\widehat q_{i+1}}{s_q},
\frac{q_{i+1}}{s_q}
\right),
$$

where \(s_q\geq1\) is a fixed quantity scale. The hybrid objective is

$$
\mathcal{L}
=\mathcal{L}_{\mathrm{mark}}
+\lambda_t\mathcal{L}_{\mathrm{time}}
+\lambda_r\mathcal{L}_{\mathrm{res}}
+\lambda_q\mathcal{L}_{\mathrm{qty}}.
$$

The frozen comparison sets \(\lambda_t=1\), \(\lambda_r=1\), and \(\lambda_q=0.25\).
Losses are computed on the final valid transition of each training window, matching the next-event
validation target. History quantities may enter the encoder, but the appended target residual is
masked before the forward pass.

