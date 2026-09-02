# Frozen Hard-LMM Minimal-Change Probe v1

## Purpose and boundaries

Use four immutable Hard-LMM seed-42 checkpoints to test whether a small readout
correction can improve body MAE without replacing the static prototype bank.
This is a diagnostic, not a matched fresh e300 experiment or a new model claim.
Historical MAC and online-event B2 artifacts are not inputs. The baseline bank,
local encoder, persistent tokens, quantity head, and time head remain frozen in
evaluation mode. The JSON contract pins their checkpoint, summary, launch,
dataset, split-manifest, and execution-code checksums.

## Two separate candidates

Let h be the local state, r the unchanged top-4 arithmetic-mean prototype
residual, z the original quantity logit, and w the frozen quantity-head weight.

- Calibration: z_new = z + 0.05 * tanh(MLP(features)).
- Shrinkage: z_new = z + (g - 1) * dot(w, r), where
  g = 1 - 0.2 * clamp(MLP(features), 0, 1).
- Both use softplus(z_new) followed by expm1, exactly as the original readout.
- Each candidate has its own 16-unit tanh MLP. Only the final layer is zero
  initialized; bounds remain nonzero so the first gradient is not blocked.
- Shrinkage starts at g=1 and stays in [0.8, 1]. The clamp has a tested nonzero
  boundary gradient at initialization; this is not an unconstrained sigmoid gate.
- The frozen time prediction remains on h+r for both probes. A local-only time
  route, similarity weighting, adaptive top-k, and null slots are excluded.
- Features contain only causal observed-history states and retrieval scores.
  Target quantity, target delta-time, error labels, and true body/tail membership
  cannot enter the MLP. Labels are used only in the train loss and reporting.
- Entropy is computed from top-4 cosine-score softmax at fixed temperature 1;
  it is not the constant entropy of the four uniform retrieval weights.
- A bounded logit change does not guarantee bounded raw-quantity absolute error.
  Tail guardrails must be evaluated rather than assumed.

## Diagnostic budget, distinct from baseline training

Baseline artifacts retain their original e300/min40/patience40 provenance.
The new small-module diagnostic uses seed 42, at most 65,536 uniformly sampled
train targets without replacement, and full validation. Sampling indices are
persisted. Extraction is performed once per dataset with the frozen model.
There is no train-only redefinition of the fixed split and no validation-fitted
normalization. Test rows are filtered out before feature materialization.

Adapters use Adam, lr=0.001, batch=128, clip=1, maximum 40 epochs, minimum 10,
and patience 10. The loss remains time NLL + direct log1p quantity MSE, with no
tail term or new regularizer. Time NLL is constant under this frozen probe.
The best validation joint objective selects the checkpoint, including the exact
identity at epoch 0 as a safe fallback. An epoch-0 winner is reported as no
selected change, not a successful improvement. No hyperparameter retuning by
dataset is permitted in this contract.

## Verification and reporting

Require base digest equality before/after fitting, finite forward/backward,
nonzero adapter gradients, exact identity output, padding/target/future
invariance, and adapter checkpoint prediction replay. Full baseline replay must
match the pinned official validation MAE, RMSE, time NLL and joint objective.
Report baseline and each candidate separately over overall, train-quantile,
body/tail and history strata, with event-wise error deltas and correction/gate
distributions. Empty strata are explicit and never represented as zero error.

The exploratory flag requires body MAE improvement >=5%, overall RMSE and >p99
MAE regression <=2%, and time NLL increase <=0.01 on each of the four datasets.
It does not authorize fresh e300, seed expansion, or opening held-out test.
Calibration gains are readout evidence; they are not a backbone-only gain.

## Execution and failure handling

Local paper_research/master is the source of truth. Commit contract, code and
tests before syncing an isolated source snapshot to 5080 without --delete.
Never alter shared models, running experiments, GDM, or external services.
Use one process per dataset and CPU feature-cache fitting; only frozen feature
extraction needs CUDA. Check free VRAM, GDM and compute occupancy before each
CUDA child. Fail closed on checksum, replay, finite, or resource mismatch.
Persist failure status, preserve artifacts, and do not retry automatically.
If runtime extends beyond the current turn, attach hourly monitoring with
measured stage timing, not an invented completion estimate.
