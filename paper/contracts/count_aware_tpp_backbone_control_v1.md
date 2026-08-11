# Mark-Free Count-Aware TPP Backbone Control v1

## Purpose

This experiment separates the effect of the history backbone from categorical
quantity marks. It asks whether TitanTPP improves next-event time and count
prediction when RMTPP, THP, and TitanTPP receive the same continuous quantity
history and use the same output heads.

The experiment does not test exponent-residual factorization. It also does not
use a product-type mark. Each sequence already corresponds to one material, and
the internal event identifier only distinguishes a real event from padding.

## Frozen Data Contract

- Dataset: `intermittent_frozen_5000_with_split.parquet`
- Data SHA-256: `85d1fe3ade3ae5a90241018e99a3e9463828d5ba35bc374b56def0168ffffc3f`
- Split manifest SHA-256: `393158a54a8ca703dbf7e9311b9dff6d2825ef737e3e3de1c30a1f3ff64c1c04`
- Development scope: validation only
- Held-out test: locked until one manuscript configuration is frozen
- Seeds: 42, 52, and 62
- Maximum sequence length: 256 events
- History strata: at most 64, 65-128, and more than 128 observed events
- Quantity strata: train-derived p50, p90, p95, and p99 boundaries

## Input Contract

For each observed event, the model receives:

\[
x_i = \left[\log(1+\Delta t_i),\; \log(1+q_i)\right].
\]

- `q_i` is the observed raw count, not a categorical bin or residual.
- The appended prediction target is masked from the history quantity input.
- Quantity-derived `mark` and `scale_residual` are not model inputs.
- A constant internal event token is permitted only to preserve the existing
  padding interface. It has no quantity or product semantics, is fixed to zero,
  and is not trained.
- Product identity is not supplied as a feature. Each sample is formed within a
  single material sequence.

## Model Contract

The three models differ only in the history encoder:

| Model | History encoder | Hidden size |
|---|---|---:|
| Count-aware RMTPP | GRU | 64 |
| Count-aware THP | causal Transformer | 64 |
| Count-aware TitanTPP | causal Titan memory encoder | 64 |

Every model uses the same two prediction interfaces:

1. RMTPP-style conditional density for the next inter-event time.
2. Log-scale regression for the next count:

\[
\hat z_{i+1}=\operatorname{softplus}(g_q(h_i)), \qquad
\hat q_{i+1}=\exp(\hat z_{i+1})-1.
\]

There is no mark head, mark cross-entropy, mark accuracy, or mark NLL in this
contract.

## Training And Checkpoint Contract

The training and validation objective is:

\[
\mathcal L_{\mathrm{joint}}
= \mathcal L_{\mathrm{time\text{-}NLL}}
+ \lambda_q\,\operatorname{MSE}(\hat z,\log(1+q)),
\qquad \lambda_q=1.
\]

- Optimizer: AdamW, learning rate `1e-3`
- Gradient clipping: `1.0`
- Maximum epochs: 300
- Early stopping: minimum 40 epochs, patience 40
- Primary checkpoint: minimum validation joint objective
- The same objective selects checkpoints for every backbone.
- Time and quantity components are always reported separately; the joint
  objective must not be labelled simply as NLL.

## Reported Metrics

Primary metrics:

- Validation time NLL
- Validation log-quantity MSE
- Raw-scale quantity MAE and RMSE

Required breakdowns:

- Quantity MAE and RMSE by train-derived p50/p90/p95/p99 stratum
- Time NLL, quantity MAE, and quantity RMSE by history-length stratum
- Seed-level values and mean +/- sample standard deviation

## Pre-Registered Claim Gates

### General Count-Prediction Claim

TitanTPP qualifies only if all conditions hold:

- Mean quantity MAE and RMSE are lower than both RMTPP and THP.
- Paired quantity MAE and RMSE are lower in at least two of three seeds against
  each baseline.
- Mean time NLL is no more than `0.01` worse than the best baseline.

### Long-History Claim

The stronger long-history claim additionally requires:

- In the `history > 128` stratum, TitanTPP lowers both quantity MAE and RMSE
  against both baselines in at least two of three seeds.
- The relative RMSE improvement over RMTPP is larger for `history > 128` than
  for `history <= 64`.

If these gates fail, the manuscript must not claim that TitanTPP captures long
histories better. Held-out test evaluation is permitted only after a validation
claim qualifies and the final configuration is frozen.
