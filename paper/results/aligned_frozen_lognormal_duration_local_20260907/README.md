# Observation-aligned frozen K=1 duration head: local verification

## Status

The implementation and local contract checks are complete. No CUDA run,
full-data fit, held-out evaluation, or performance decision was performed in
this stage.

## Frozen boundary and likelihood

- A and B use the same 130-parameter heteroscedastic K=1 log-normal duration
  head, train-only initialization, AdamW settings, batch order, epoch budget,
  calibration source revision, and earliest-finite-minimum selector.
- The source encoder, Hard-LMM memory, quantity head, and every non-time tensor
  remain frozen. Quantity predictions must be bitwise identical to the source.
- Intermittent retains original-unit continuous log-normal density likelihood.
- Taxi uses the normalized positive-integer law
  `P(D=1)=F(1.5)` and
  `P(D=d)=F(d+0.5)-F(d-0.5)` for `d>=2`.
- Instacart uses the same integer law, with recorded code 30 evaluated as
  `S(29.5)`.
- The selected checkpoint is the earliest strict finite minimum of validation
  NLL under the dataset's declared observation law. Continuous-density NLL is
  reported from the same checkpoint as a reference and never enters selection
  or the Taxi/Instacart acceptance decision.

This treats integer durations as interval observations and the Instacart cap
as right censoring, so every score is a normalized probability of the event
that was actually recorded. The design follows the likelihood and censoring
principles used by [Rindt et al.](https://proceedings.mlr.press/v151/rindt22a.html)
and [Yanagisawa](https://proceedings.mlr.press/v202/yanagisawa23a.html).

The complete frozen contract is
[`aligned_frozen_lognormal_duration_v1.json`](../../contracts/aligned_frozen_lognormal_duration_v1.json).

## Local evidence

- Contract SHA-256:
  `677cc3af91d84dfea8e3b4e71b38697fb467059f680678f0b8370a50f2366f16`.
- The normalized integer likelihood was checked against closed-form first-bin,
  regular-bin, and top-code values and sums to one with a closing survival bin.
- Loss and all four time-head gradients are finite. An in-place tensor update
  found by the first gradient test was removed before qualification.
- Interrupted and uninterrupted synthetic fits reproduce the same history,
  selected epoch, and selected model-state digest.
- Reusing a completed fit revalidates its history, earliest selected epoch,
  first valid early-stopping epoch, selected metric, checkpoint state digest,
  and observation contract before accepting the cached result.
- Existing continuous frozen-duration, matched-head, observation audit, and
  causal-adapter tests remain passing.
- All six A/B source checkpoints and all three prior-B reference checkpoints
  match their pinned file and model-state SHA-256 values.
- All three data files and split manifests match their pinned SHA-256 values.
- Replaying the prior B checkpoint may differ by at most `1e-6` in continuous
  NLL across the 5090 artifact and the 5080 execution; checkpoint, data, and
  state identities still have to match exactly.
- A one-train-batch, one-validation-batch Taxi CPU smoke passed for both A and
  B. Their initial time-head digest was identical
  (`dced6541bbef01d2fb2798ca566835efa9dfa0999cf16b989e0eecedf0ad8dff`),
  only the four declared time-head tensors changed, and quantity predictions
  remained bitwise identical.
- Repeating both smoke commands against the completed local artifacts also
  passed the cached-state NLL and quantity replay audit.
- The Taxi smoke is marked `qualified_full_data=false` and
  `qualified_full_fit=false`; its NLL values cannot
  be used for model selection or a performance claim.

The local regression command completed with `141 passed`. Held-out test rows
were not admitted or evaluated.

## Fixed validation decision

For each dataset, the prior continuous-selected B head is first replayed under
the same observation likelihood as aligned B. Aligned B is accepted without an
adapter only if all datasets satisfy all applicable gates:

1. Taxi and Instacart primary NLL improve over prior B by at least `0.005`.
2. Intermittent, whose likelihood and selected objective are unchanged from the
   prior B fit, replays without worsening beyond numerical tolerance `1e-6`.
3. Primary NLL is at most aligned A plus `0.01` on every dataset.
4. Source quantity predictions remain bitwise identical.

Continuous-density NLL remains a report-only reference metric. It is not a
second acceptance gate for Taxi or Instacart because their aligned selector is
the normalized integer observation likelihood.

The decision builder rejects incomplete epoch budgets, partial data, held-out
use, likelihood drift, source/cache/checkpoint identity drift, and A/B
initialization drift.

## Remaining execution

The next execution stage is six A/B CUDA e1 runs on the 5080, followed by the
full frozen-head fits if all runtime and contract checks pass. The fitted-head
results, rather than this local smoke, will determine whether a causal time
adapter is still needed.
