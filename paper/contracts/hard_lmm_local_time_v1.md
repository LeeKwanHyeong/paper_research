# Hard-LMM Quantity Memory / Local Time Contract

## Status and decision

Contract frozen on 2026-09-03 in local `paper_research/master`. Implementation,
CUDA tests, source transfer, training and scheduling have NOT started and are
not authorized by this contract-only task. This is one prospective diagnostic
candidate, not an adopted primary model or a successful improvement.

Authoritative values: `hard_lmm_local_time_v1.json`. The reserved backbone ID is
`titantpp_hard_memory_local_time`, role `t0_hard_memory_local_time`; neither is
registered yet. Preserve historical `titantpp` and its artifacts unchanged.

The question is whether the same static prototype residual should be used for
both event time and quantity. Weak weighted-retrieval effects and the observed
time/quantity objective tradeoff motivate testing this boundary. They do not
prove that memory harms body events or that this change will improve them.

## The single change

Let `h` be the masked output of the original `_encode_base`, which already
includes attention to persistent tokens. Let `r` be the arithmetic mean of the
same four cosine-selected vectors from the original 64-prototype static bank.

| Path | Original Hard-LMM | This candidate |
| --- | --- | --- |
| Quantity representation | `h+r` | `h+r` |
| Time representation | `h+r` | `h` |
| Persistent tokens within h | 16 per attention block | unchanged |
| Trainable parameters | original | identical names/shapes/count; zero added |

Use original UNIFORM hard top-4 retrieval, not the cancelled weighted candidate,
Titans-MAC, or the earlier multi-change gated-memory model. No projection, gate,
temperature, adapter, head calibration, local-encoder replacement or online write
is introduced. `titantpp_no_memory` is not a substitute: it also removes the
persistent-token configuration that must remain here.

The implementation must compute `h` once, return `(time_state, quantity_state)`
from `encode_task_states`, mask both states, and keep `encode()` returning the
time state. The existing evaluator already selects the last observed history
position separately from each returned state. Do not swap tuple order, re-encode
with a new dropout draw, or add/detach parameters to simulate routing.

## Gradient and causality boundaries

Quantity outputs and quantity-only gradients must match the original at identical
weights, inputs, mode and RNG. This is NOT a promise of unchanged quantity after
fresh training: the time gradient changes, the encoder/persistent parameters
remain shared, and global gradient clipping can change the update to other
parameters too. Loss weights, optimizer groups and clipping stay unchanged.

Time-only loss must have no gradient to the static prototype bank, while it still
trains the encoder and persistent tokens. Quantity loss must retain its path to
the bank. Verify this with a nonzero quantity-head fixture or a valid trained
checkpoint; the original zero-initialized quantity weights would otherwise make
a vacuous zero-gradient test pass. Static prototypes remain trainable parameters,
not per-series history buffers. There is no series-specific state to reset.

Target quantity is zeroed before encoding; target writes are disabled. Predictions
use position `length-2` after right-padding, before the target. Target duration
is used to evaluate the density, not as a feature available to that prediction.
The existing full tensor can contain the later target duration, so perturbing it
must not affect the history prediction through causal attention. Test future
events, padding, batch ordering and series isolation on both paths. Neither the
static nor persistent bank may be updated by validation or test inference.

## Baseline and data contract

Reuse original seed-42 Hard-LMM joint-selected results through the hash-pinned
`count_aware_hard_lmm_frozen_probe_v1.json` registry. Inherit ONLY dataset paths,
split/data hashes, original summary/checkpoint identities and context lengths;
do not inherit its calibration/shrinkage architecture, subset or frozen-fit plan.
No RMTPP, THP, NHP, SAHP or original Hard-LMM performance training is scheduled.
The weighted candidate and MAC are not this candidate's reference.

| Dataset | Lookback / unit | Maximum sequence | Train / validation targets | Parameters |
| --- | --- | --- | --- | --- |
| Taxi | 168 hours | 256 | 38,393 / 8,268 | 89,795 |
| RAF | 84 months | 84 | 25,779 / 6,690 | 78,787 |
| Intermittent v2 frozen-5000 | 520 weeks | 256 | 393,824 / 86,285 | 89,795 |
| Instacart | 52 days | 64 | 1,991,192 / 503,733 | 77,507 |

All four original summary and launch-contract identities were checked locally.
The registry's remote-style Intermittent artifact path is not present locally;
the hash-identical summary/contract mirror is under
`paper/results/count_aware_tpp_backbone_control_20260812/source_5080`. That copy
has no checkpoint file. Do not equate metadata availability with weight
availability or fabricate a replacement. Verify file and canonical-state hashes
against the registry before any future checkpoint test. No sync was done here.

Do not read or materialize held-out rows. Only fixed train/validation splits are
allowed; full target counts, train-derived quantity cutoffs and history strata
must match the reference. Missing Intermittent legacy fields such as unused
log-normal `train_target_std` are not fabricated or silently applied to direct
log-MSE. Resolve the pinned original source for legacy time-head metadata.

## Unchanged learning and selection

Future performance runs train ALL original parameters from fresh initialization,
seed 42, maximum e300, minimum e40, patience 40, batch 128, lr 0.001. No checkpoint
fine-tuning, readout-only fitting, new AMP/compile policy, PCGrad or LR schedule.
Keep the seeded loaders, zero workers and existing warn-only deterministic
policy; record backend flags, warnings and runtime versions rather than claiming
historical bitwise CUDA replay. Synthetic identical-weight regression is not
proof of identical training trajectories across runtimes.

Important documentation correction: the actual baseline code uses **AdamW**,
not plain Adam, despite the earlier weighted-contract prose. Preserve one group,
betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.01`, and time LR multiplier
`1.0`. Current `build_optimizer` and the pinned original Intermittent source both
call AdamW. This clarification does not change old results or edit old contracts.
Keep global L2 gradient clipping at 1.0 across all trainable parameters.

Keep direct MSE on `log1p(quantity)` with coefficient 1, tail coefficient 0, the
same softplus/expm1 quantity decoder and `legacy_clamped_rmtpp` head formula.
Only the time head's input representation changes. Preserve launch time scale
3.0, slope cap argument 10/3, intercept limit 30, safety argument 40 and legacy
initialization. Derived train time statistics are separate evidence, NOT new
legacy launch values. This avoids the prior expected/observed scale mix-up.

Select the first strict minimum of full-validation Time NLL + log-MSE over
completed training epochs; ties retain the earlier epoch. Epoch zero is not
eligible. Do not substitute a post-hoc raw-MAE/RMSE epoch, separately select
the heads, or infer body metrics where no breakdown exists. Routing cannot by
itself align log-MSE with raw MAE or remove all shared-encoder task conflict.

## Acceptance and staged execution, not launched

Required per-dataset gate against the original official seed-42 checkpoint:

- Count-weighted body MAE over `le_p50`, `p50_p90`, `p90_p95` improves at least 5%.
- Overall RMSE and `gt_p99` MAE each worsen by at most 2%.
- Time NLL increases by at most 0.01, using an absolute difference even if negative.
- All metrics are finite; matching counts/cutoffs and complete evidence are required.
- Empty/missing strata or a zero reference denominator are not an automatic pass.

The body formula is `sum(n_bin * MAE_bin) / sum(n_bin)`, not an unweighted bin
mean. Report overall MAE, RMSE, body/p95/p99/history metrics, time/log-MSE/joint,
selected epoch, epoch cost and peak VRAM. Existing thresholds are not lowered.

After a separate implementation and execution agreement, use **5080 only**:

1. Pass the listed local/CUDA route tests, then full train/validation e1 smoke
   for Taxi and RAF. Smoke is feasibility-only; never reuse its weights.
2. Run only the two fresh candidate seed-42 Taxi/RAF comparisons first. Keep
   e300/min40/patience40: the small screening scope limits DATASETS, not epochs.
   Both first-stage datasets are evaluated before the follow-up decision.
3. Only if both pass every original gate, request a separate decision for
   Intermittent and Instacart, each preceded by its own full e1 smoke. Any failed
   gate holds the candidate; it does not prove all memory architectures fail.
4. Only all-four success permits considering broader follow-up. Seed 42 remains
   exploratory; no automatic multi-seed run, primary-model adoption or held-out use.

This preserves the historical epoch/selector contract without truncating and
reselecting baseline histories to manufacture a budget match. If a run is stopped
for budget/user reasons, label it inconclusive, preserve its state and never emit
a synthetic complete summary. A different budget needs a prospective amendment,
not a post-hoc relaxation. No old baseline or frozen-cache experiment is repeated.

Before a future launch, verify source/data/checkpoint identity, no competing GPU
processes, inactive GDM/graphics, at least 12,000 MiB free VRAM and readable kernel
logs with no recent Xid/OOM. Do not alter services automatically. Use fresh output
paths and isolated processes; record failures atomically and never auto-resume.
Require route ID and contract metadata when loading weights: the equal state-dict
shapes alone cannot distinguish original and local-time models. Synchronize only
approved committed source without `--delete`. No command or runtime SHA is invented
before implementation. Hourly monitoring is created only with a later approved run.

## Verification boundary and next work

This phase validates the DOCUMENT contract, registry/hash references, architecture
constraints and unchanged gates. The JSON lists 16 required implementation checks;
their status remains `not_run_candidate_not_implemented`. In particular, no new
model has passed forward/backward, causal, checkpoint, CUDA or e1 tests yet.

Next work is serial in the same repository: implement the distinct factory/runner
route without changing legacy behavior, then pass local route tests. Server sync,
CUDA/e1 and the first two performance runs remain separate approval boundaries.
Local code/test/report commits may follow, but no push or server deployment now.

Alternatives intentionally excluded: weighted retrieval retuning, shrinking or
removing quantity memory, time-head calibration, loss changes and a new encoder.
If routing fails, local-encoder attribution is a later independent decision,
not a hidden second factor in this contract.

Evidence: analysis commit `2739f92` and
`paper/results/hard_lmm_weighted_static_20260903/mechanism_analysis.md`.
Code reviewed at `7394d9a990e82aaacf4046940951cfa6a30ef4c6`:
`models/TPPs/CountAwareTPP.py` (`_encode_base`, `encode_task_states`, `encode`),
`models/Titan/common/memory.py` (`HardLocalMemoryMatcher`),
`paper/scripts/count_aware_tpp_backbone/core.py` (`target_outputs`), and
`paper/scripts/count_aware_tpp_backbone/training.py` (`build_optimizer`, `train_one`).
