# Architecture Recommendation: Hard-LMM prototype value-norm consistency boundary

Date: 2026-09-09

Repository / branch: `paper_research / codex/hard-lmm-causal-qkv`

Method: Recommendation mode with a design-twice comparison

## Problem

The next candidate must be a real Hard-LMM backbone change while keeping the
current validation-raw-RMSE-selected Hard-LMM B as an exact nested state. The
most obvious combination change, a scalar gate on the aggregated memory
residual, is already covered by the frozen shrinkage and readout experiments.
The online write/update boundary is also represented by the earlier B1, B2, and
Surprise-memory paths. Repeating either boundary would spend another full
training budget without testing a new mechanism.

The remaining structural mismatch is inside static retrieval. Hard-LMM chooses
prototypes by cosine similarity, which discards prototype length, but then
averages the raw selected values, which restores their lengths as an implicit
value weight. Thus two prototypes with the same search relevance can contribute
different magnitudes solely because their learned row norms differ.

## Constraints and Quality Attributes

- Keep the same encoder, persistent tokens, 64-row static bank, cosine top-4
  selection, arithmetic aggregation, time and quantity heads, training loss,
  optimizer, and validation raw-RMSE selector.
- Use one graph and one parameterization for every dataset. Dataset names,
  fixed duration values, quantity strata, and per-dataset coefficients are not
  model inputs.
- Nest B exactly at initialization, preserve the common parameter tensors and
  random-number stream, and allow a strict B checkpoint to initialize the new
  route without reinterpretation.
- Keep target and padding causality, finite loss and gradients, checkpoint route
  isolation, and the existing held-out lock.
- Add little runtime and maintenance cost. A candidate that needs online
  per-series state or a second search is out of scope.

Non-goals are changing the quantity loss or checkpoint selector, calibrating the
output head, improving the time likelihood definition, and claiming benchmark
superiority from a seed-42 validation screening.

## Options

### 1. Scalar post-retrieval gate

Apply `g(h)` or a learned scalar `g` to the existing residual `r_B` before
`h + r_B`. For a linear quantity head this produces
`w^T(h + g r_B) + b`, so its added correction is `(g-1) w^T r_B`. This is the
same quantity direction already tested by residual shrinkage/readout probes.
The smooth adaptive and scalar controls passed 0/4 datasets; the broader frozen
readout factorial passed the joint gate in 0/24 fits. It is low cost, but does
not create a new memory mechanism.

Decision: reject as an experimental duplicate.

### 2. Confidence-gated online write/update

Write an observed-state summary into a per-series memory and gate the update by
surprise or retrieval confidence. This is a different boundary from a frozen
readout, but it overlaps the earlier Titans-MAC B1, circular read-before-write
B2, and Surprise/dual-memory routes. Those paths already showed high training
cost or unstable dataset effects. It is especially weak for Instacart histories
of length 2–3, where only one or two prior writes exist.

Decision: defer until a train-only, series-disjoint pre-write usefulness test
reproduces in both folds.

### 3. Aggregate residual or hidden-state normalization

Normalize the final retrieved residual or `h + r_B` by a token-level RMS or
LayerNorm before the heads. This is simple, but normalization after aggregation
still changes the same aggregate direction targeted by shrinkage/readout and
also perturbs the encoder state even when the bank geometry is not at fault.
Exact B identity needs another signed gate, leaving the mechanism weakly
identified.

Decision: retain only as a fallback if prototype-level normalization is
degenerate.

### 4. Prototype-level value-norm consistency

Keep the original search indices and similarities. Normalize only each selected
value to the bank-wide mean row norm, form a second arithmetic-mean residual,
and combine it with the original residual through one zero-initialized scalar.
This changes the vector geometry before aggregation and cannot generally be
reduced to scaling the aggregated residual.

Decision: select.

## Tradeoff Comparison

| Option | New mechanism | Exact B state | Added state | Runtime risk | Prior overlap | Main risk |
| --- | --- | --- | ---: | --- | --- | --- |
| Post-retrieval scalar gate | No | Yes | 1 scalar | Low | Direct | Repeats a failed direction |
| Online write confidence | Partial | Possible | Per-series state and parameters | High | Substantial | Cost and short-history scarcity |
| Aggregate normalization | Weak | Possible | At least 1 gate | Low | Moderate | Changes scale without isolating prototype cause |
| Prototype value-norm consistency | Yes | Yes | 1 scalar | Low | Adjacent to weighted static, no exact match | May erase useful norm information |

The selected option has the smallest distinct intervention. It is adjacent to
similarity-weighted retrieval because both act before prototype aggregation,
but the old candidate weights raw values by cosine score whereas this candidate
uses inverse row norm and exactly nests B. It keeps retrieval membership and
scores fixed and does not add an online state.

## Recommendation

For selected prototype values `m_j`, define

\[
s = \frac{1}{M}\sum_{i=1}^{M}\lVert m_i\rVert_2,\qquad
\widetilde m_j =
\begin{cases}
s\,m_j/\lVert m_j\rVert_2,&\lVert m_j\rVert_2>\epsilon\\
0,&\text{otherwise},
\end{cases}
\]

\[
r_B=\frac{1}{k}\sum_{j\in I(h)}m_j,\qquad
r_N=\frac{1}{k}\sum_{j\in I(h)}\widetilde m_j,
\]

\[
r=r_B+\tanh(\alpha_{raw})(r_N-r_B),\qquad z=h+r.
\]

Use `M=64`, `k=4`, `epsilon=1e-8`, detach `s` from gradient propagation, and
initialize `alpha_raw=0`. Detaching the reference preserves the original sparse
selected-row credit assignment; otherwise every query would update all 64 rows
through `s`. Rows with norm at or below epsilon contribute zero to the normalized
branch and receive zero normalization-branch gradient. The signed
coefficient is deliberate: an exact zero, finite smooth parameterization, and a
nonzero first-step gate gradient cannot all be obtained from a one-sided smooth
sigmoid interpolation at a finite parameter value. The model is therefore a
bounded signed normalization contrast rather than a claimed convex mixture.

This is a backbone change because `z` is the shared state consumed by both the
time and quantity heads. It is not an output calibration or checkpoint-selector
change. A strict B state can populate every inherited tensor; only
`lmm.alpha_raw` is new and zero.

Checkpoint-only geometry confirms that the contrast is non-degenerate for all
three frozen B banks. Prototype norm coefficient of variation is approximately
0.017 for Intermittent, 0.197 for Taxi, and 0.161 for Instacart. Across all
`64 choose 4` subsets, the normalized and raw means are non-collinear in every
finite subset; their median relative differences are approximately 0.006,
0.072, and 0.092 respectively. These values justify implementation only. They
are not evidence of lower prediction error.

## Risks

- Prototype norm may encode useful frequency or quantity magnitude. Equalizing
  it can harm predictions even if cosine search ignores it.
- This remains a candidate at the already explored top-4 aggregation boundary.
  Its inverse-norm rule is distinct from cosine-softmax weighting, but a failure
  would be evidence to stop further hand-designed static weighting rules.
- The single global gate may be too weak to correct heterogeneous prototype
  effects. Increasing gate capacity before the first screening would confound
  the mechanism and is excluded.
- Top-k is discrete. The candidate leaves selected indices unchanged for a
  fixed state, but end-to-end training can later move the shared bank and alter
  membership.
- The time head consumes the same changed state, so quantity gains can trade off
  against time quality. The existing time guardrail remains binding.
- Existing validation results have informed the candidate choice. Seed 42 is an
  exploratory screening and cannot establish generalization.

## Assumptions

- The three pinned seed-42 B checkpoints and their stored state digests are the
  correct current baseline identities.
- Bank row-norm variation reflects a real mismatch worth testing rather than a
  harmless learned scale convention.
- A single signed scalar is sufficient to determine whether the normalization
  contrast has useful gradient signal before adding capacity.
- The existing direct log1p-MSE training and validation raw-RMSE selection
  remain the intended comparison contract.

## Validation Steps

1. Reproduce the checkpoint-only norm and all-quartet geometry audit without
   loading train, validation, or held-out target rows.
2. Prove same-seed common-state and RNG identity with B, exact output and common
   gradient identity at `alpha_raw=0`, and nonzero finite gate gradient on a
   crafted non-collinear case.
3. Verify unchanged top-4 indices/similarities, causal and padding behavior,
   finite extreme-input loss/gradient, strict B initialization, dedicated
   checkpoint metadata, and save/restore including optimizer state.
4. Commit the source and prospective contract only after the local suite passes.
5. In a later authorized GPU stage, run CUDA and full-data e1 before Instacart
   seed-42 screening. Stop on any contract failure or Instacart performance-gate
   failure. Do not access held-out test data.
