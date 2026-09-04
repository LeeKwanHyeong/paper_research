# TitanTPP-HardLMM Separate-Key Sparse Retrieval

## Scope and authorization

Contract: `hard_lmm_key_value_v1.json`, frozen before implementation and local
synthetic tests on 2026-09-04. Work is limited to `paper_research/master`.
No server transfer, CUDA/e1, real-data fit, scheduler, held-out evaluation or push
is authorized. Implementation verification is recorded separately; this contract
is not a performance result or a primary-model adoption decision.

## One contrast, two references

The official reference remains `titantpp` (original uniform Hard-LMM). The
mechanism control is `titantpp_weighted_static_memory` (tied-bank weighted
retrieval). The candidate is `titantpp_key_value_static_memory`, with dedicated
role `t0_key_value_static_retrieval` and memory mode `static_key_value_lmm`.

For the original local representation h, keys K and values V:

1. Compute s = cosine(h, K) and choose the original hard top-4 indices I.
2. Compute a = softmax(s[I] / 1.0).
3. Retrieve r = sum(a[j] * V[I[j]]) and pass masked h+r to both original heads.

Only K/V tying differs from the weighted control. Comparing directly with the
uniform official reference changes both aggregation and tying; that contrast
cannot alone establish the cause of an improvement. Encoder, persistent tokens,
head, objective, selection and all other model settings remain unchanged.

Keys are a detached clone of the freshly initialized value bank, with independent
storage and no extra random draws. Initial candidate outputs exactly match the
weighted control on CPU, not the original uniform model. All parameters remain
trainable. The extra 64x64 = 4,096 parameters and different optimizer dynamics are
part of the intervention, not hidden capacity matching. No query/value/output
projection, gate, null slot, online update or calibration head is added.

## Gradient and causal boundaries

Autograd differentiates selected cosine scores and softmax weights. The discrete
top-k indices are not differentiated and no straight-through estimator is used.
An unselected key gets no loss gradient from that query; weight decay may still
change it. A singleton top-k, identical selected values or a zero downstream head
can yield zero key gradient. In particular the existing zero-initialized quantity
head is preserved, so tests must not demand a nonzero quantity-key gradient on
the very first forward. They must also check it after actual synthetic optimizer
steps, not only after manually changing the head.

The keys are static task parameters, not an online series state. Retrieval accepts
only the encoded observed history; the existing target evaluator masks target
quantity and predicts at the last observed event. No target-based routing, oracle
body/tail assignment, validation updates or cross-series mutable state is added.

## Prior variants are not renamed as a new invention

| Existing path | Overlap | Difference in this experiment |
| --- | --- | --- |
| Weighted static Hard-LMM | Cosine top-4, tau=1 softmax | Only untie keys from values |
| GatedSoftMemory | Independent keys and values | Existing path also has dense attention, projections and a zero-init gate |
| TPPSpecificGatedMemory | Sparse weighted retrieval | Existing path also has confidence/null selection, projections and online writes |

Key/value addressing is established prior art, including
[Miller et al., 2016](https://aclanthology.org/D16-1147/). No novelty or superiority
claim follows from implementing it. Existing prototype concentration motivates a
hypothesis; it does not prove a causal bottleneck. More diverse selection alone
does not satisfy the performance gate.

## Reuse and later evaluation

Original seed-42 references come from the pinned baseline registry in the JSON.
Reuse requires completed, finite, contract-matched artifacts and checkpoint/source
digests. Taxi and RAF weighted runs are completed candidates for reuse, subject to
that audit. Intermittent was interrupted and Instacart never completed a weighted
main run; neither is a reusable completed control. An isolated key/value-effect
claim on those datasets requires a separately approved matched weighted control.
Do not silently retrain controls or other benchmarks in this phase.

Future screening requires separate approval and full CUDA/e1 verification.
Taxi and Instacart are the first-stage datasets, followed conditionally by
Intermittent and RAF with identical candidate settings. Keep the original fixed
splits, seed 42, e300/min40/patience40, batch128, lr=.001, AdamW, clipping, direct
log-MSE, legacy time head and validation joint selector. Keep the original gate:
body MAE improvement >=5%, overall RMSE and >p99 MAE regression <=2%, Time NLL
increase <=.01, all finite. Held-out remains locked and benchmarks remain frozen.

## Implementation acceptance

Verify the manual formula, addressing/value gradients, identical weighted
initialization and RNG use, key-only and value-only intervention effects, original
path regression, future/padding/series isolation, finite extremes, synthetic
optimizer steps, checkpoint identity and strict replay. Store local test evidence
separately from this preregistered contract. No performance run is implied by
passing these tests.
