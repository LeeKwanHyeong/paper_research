# Count-Aware Log-Normal K=1 Screening v1

## Purpose

This validation-only screening tests whether a heteroscedastic quantity head can
improve the ordinary-count calibration of Count-aware TitanTPP without losing
its existing extreme-count and RMSE behavior. The history encoder, time head,
data split, optimizer, and training budget remain fixed.

The completed mark-free control showed the following error structure:

- TitanTPP has lower overall RMSE than THP but higher overall MAE.
- TitanTPP is better than THP only in the train-derived `q > p99` quantity
  stratum; THP is better throughout the `q <= p99` strata.
- The experiment therefore targets body calibration while preserving the p99
  tail result. It does not reopen categorical quantity marks, residual decoding,
  RevIN, or the long-history claim.

## Variant Contract

| Variant | Quantity output | Quantity training loss | Role |
| --- | --- | --- | --- |
| `log_mse` | `softplus(raw)`, then `expm1` | MSE on `log1p(q)` | Fresh matched control |
| `lognormal_k1` | location `mu`, scale `sigma` | Gaussian NLL on `log1p(q)` plus location Huber | Candidate |

For `lognormal_k1`:

\[
z=\log(1+q),\qquad
\mu=\operatorname{softplus}(g_\mu(h)),\qquad
\sigma=10^{-3}+\operatorname{softplus}(g_\sigma(h)).
\]

\[
\mathcal L_q
=-\log\mathcal N(z;\mu,\sigma^2)
+\operatorname{Huber}_{\delta=0.25}(\mu,z).
\]

The raw quantity point prediction is fixed before validation:

\[
\hat q=\exp(\mu)-1.
\]

This is the distribution median. The same `q_hat` is used for MAE and RMSE;
metric-specific point predictions are not allowed in this screening. `mu >= 0`
guarantees `q_hat >= 0`. The scale head is initialized from the train-split
standard deviation of `log1p(q)`, and the new scale parameters do not alter the
matched encoder initialization.

## Fixed Conditions

- Dataset: `intermittent_frozen_5000_with_split.parquet`
- Split: fixed chronological validation only
- Held-out test: locked and not read
- Backbones: Count-aware THP and Count-aware TitanTPP
- Seed: `42`
- Epoch ceiling: `300`
- Minimum epochs / patience: `40 / 40`
- Batch size: `128`
- Learning rate: `1e-3`
- Lookback / maximum sequence length: `520 / 256`
- Hidden dimension: `64`
- Time head: existing shared RMTPP conditional density
- Quantity input: causal `log1p(raw quantity)` history
- Checkpoint: minimum validation joint objective within each variant

Joint-objective values are not compared across `log_mse` and `lognormal_k1`
because their quantity losses use different sample-space scores. Common
comparison metrics are time NLL, raw quantity MAE, raw quantity RMSE, and
train-derived quantity-stratum MAE/RMSE.

## Acceptance Gate

`lognormal_k1` passes only if, relative to the fresh matched TitanTPP `log_mse`
control at seed 42:

- overall quantity MAE improves by at least `5%`;
- overall quantity RMSE regresses by no more than `2%`;
- `q > p99` quantity MAE regresses by no more than `2%`;
- time NLL regresses by no more than `0.01`;
- all losses, predictions, gradients, and exported metrics are finite.

THP is run under both variants as a strong attention boundary. It does not
replace the paired TitanTPP acceptance gate, but the report must disclose
whether the candidate closes TitanTPP's existing MAE gap to THP.

## Execution Boundary

- Server: `5080`
- tmux: `count_lognormal_k1_e300_0815`
- Artifact: `search_artifacts/count_aware_lognormal_k1_screening_e300_20260815`
- Completion check: only when requested; no continuous polling
