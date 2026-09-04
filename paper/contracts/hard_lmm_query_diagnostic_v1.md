# Hard-LMM causal-history query: prospective train-only diagnostic

Date: 2026-09-05. Source baseline: local `paper_research/master`, `1003033`.
The companion JSON fixes the procedure **before extraction or diagnostic results**.
This is a mechanism/decodability check, not a new predictive-model experiment.

## Authorized scope and sequence

The user authorized local diagnosis, conditional query contract/implementation,
local tests and an independent commit, then isolated **5090 CUDA/full e1** if
the evidence and tests pass. This supersedes the older protocol's 5080 default
for this task. e300, other seeds/datasets, held-out evaluation, unrelated service
changes and push are outside this authorization. Original and separate-key
parameters stay frozen throughout this diagnostic.

1. Verify original registry, data/split and both checkpoint digests. Read only
   train rows with a lazy split filter; preserve original loader semantics.
2. Extract both models on the same fixed seed-42 sample, at most 8,192 targets
   per dataset. Record target IDs and series-disjoint two-fold assignment.
3. Compare actual retrieval and quantity contribution; crossfit constant,
   h-neighborhood-only and h-neighborhood plus two causal history statistics.
4. Apply the fixed evidence gate. Only a pass permits defining/implementing the
   conditional query candidate. A failure leaves the model and server untouched.

## Evidence question and fixed method

Within similar normalized local-state neighborhoods, do observed history level
and latest deviation explain the correction still needed by separate-key?
The two statistics are mean observed log1p quantity and latest observed log1p
quantity minus that mean, limited to the legal context. Future/target/padding
values never enter them. One observed event gives zero deviation.

Use 64 opposite-fold cosine neighbors and a two-variable ridge fit with penalty
1 on summed squared error, unpenalized local intercept and opposite-fold-only
standardization. Compare out-of-fold residual MSE with the neighbor-mean and
constant controls. There is no tuning, validation fitting or model update.

Proceed only if separate-key Instacart improves pooled residual MSE by at least
1% over the h-only control, improves in **both** directions, beats the constant,
and has a positive lower 5% band from 500 fixed-seed series-cluster bootstrap
resamples. Taxi pooled degradation must be no more than 1%. Each fold must have
at least ten series. Record cosine-distance coverage and value-projection
variation alongside the numerical gate; they are mechanism evidence, not
performance or diversity acceptance criteria.

## Interpretation and limits

The backbone has already seen all train rows, so these folds assess a helper's
train-internal stability, **not generalization to unseen train/validation/test**.
The source checkpoints also used validation for selection. Fresh-model benefit
requires separately evaluated evidence. Missing support does not prove that a
nonlinear query could never work.

Original Hard-LMM remains the official baseline. Separate-key is a direct
mechanism control, not an accepted replacement. Prototype identifiers can drift
in meaning between fitted models. In particular, Instacart has no completed
weighted-only e300 control, so a key-only causal explanation is not justified.
There are no plots in this contract; generated tables and machine-readable
results will be audited, and no unrendered figure will be claimed inspected.
