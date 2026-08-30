# Count-aware TitanTPP-MAC Primary Contract v1 Amendment 2

## Correction Recorded 2026-08-31

The causal attribution below was premature. Recovery2 failed at the same
Instacart batches after the time-head correction. Exact seed-42 replay located
the first non-finite values in the neural associative memory update, with
finite inputs and finite model parameters. The time-head bound correction is
a separate synthetic defect fix, not a demonstrated explanation of these
failures. The claim about why RAF/Taxi passed is also withdrawn. The original
text is retained as an audit trail; use
`count_aware_titantpp_mac_inner_gradient_stability_v1.json` for the new policy.

## Original Record (Causal Claims Superseded)

This amendment corrects a contract-implementation mismatch in the shared
`legacy_clamped_rmtpp` time head. The frozen launch metadata declared
`time_intercept_limit=30`, but the legacy density and survival implementation
used a hard-coded upper bound of `300`. In float32, the resulting exponential
can overflow before the configured bound is enforced.

The defect was independently observed on both servers during the first
Instacart epoch: seed 42 stopped at batch 3705 on the 5090 and seed 62 stopped
at batch 13439 on the 5080. Neither server reported an NVIDIA Xid or CUDA OOM.
RAF and Taxi completed because their epochs contain far fewer optimizer steps.

Revision `b1d9e638c68bc7bab9ace6b5c8fa9d0af989c8f7` applies the already
contracted limit to density, survival, and median prediction. It does not
change the Titans-MAC memory equations, quantity objective, density family,
data split, optimizer, or checkpoint selection. Normal-range values retain the
same formula; an explicit extreme-input regression test verifies finite
forward, backward, survival, and median calculations.

Pre-amendment Instacart runs are invalid because no epoch completed. Completed
RAF and Taxi runs remain preserved as historical evidence but are not imported
into the corrected grid. Held-out test remains locked. Existing seed-42
results may enter final three-seed reporting only after checkpoint replay shows
equivalence under the corrected bound.
