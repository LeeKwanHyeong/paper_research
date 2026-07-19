# ADR: TitanTPP V7 Causal Time-History Adapter

- Date: 2026-07-19
- Status: Stage-0 execution completed; artifact analysis pending; V7 model locked
- Scope: Taxi-first TitanTPP time modeling after V6 closure
- Method: Design-Twice followed by ADR
- Baselines: V2 common control and Taxi V3b incumbent

## Context

V2 remains the common TitanTPP baseline and V3b remains the Taxi-specific
incumbent. The post-V6 decision has two credible but materially different
directions:

- V5b changes the Intermittent marker objective to correct class priors.
- A time-history adapter gives the Taxi time head strictly past information that
  is currently outside the active 168-hour window.

V5b is motivated by a real Intermittent imbalance: marks `0-2` account for
`86.60%` of train next-event targets and the train effective-class count is
`4.31` across eleven real marks. That evidence does not establish that class
prior is the cause of the main error. V5a and the V3 variants mainly moved the
high-support mark `0/1` boundary, and raw inverse-frequency correction would
give unstable influence to very small tail classes. V5b also changes the
training posterior and therefore needs an explicit calibration and inference
contract before it can be compared through marker NLL.

V6 tested a generic pre-window memory hypothesis and correctly closed after its
frozen marker-CE primary gate failed. Its final train-only artifact nevertheless
contains a narrower secondary observation:

- final `log1p(dt)` MAE improved `2.4696%`;
- the series-bootstrap 95% interval was `[1.5236%, 3.5050%]`;
- `67.176%` of the 131 series improved;
- all eight selection candidates improved the time metric by
  `0.4581%` to `2.1265%`.

This does not retroactively pass V6, does not freeze `M=64/topk=4`, and is not
TitanTPP model-quality evidence. It is sufficient to formulate a new,
time-specific source-isolation hypothesis. V4 does not answer this hypothesis:
V4 conditioned the time intercept on the next mark but added no information
from before the active window, and its paired validation time-NLL gains
(`0.415%/0.321%`) missed the frozen gate.

## Decision Problem

Select one post-V6 enhancement to investigate next while preserving:

- a single identifiable mechanism;
- the marked-TPP marker and time likelihood definitions;
- exact nesting against V2 and V3b at initialization and with empty history;
- strict same-series causality and validation/test lock;
- fresh matched controls for every model-quality comparison;
- Model Enhancement emphasis on architecture rather than broad loss tuning.

## Options Considered

| Option | Positive evidence | Main risk | Decision |
| --- | --- | --- | --- |
| V5b capped class-prior correction | Intermittent class imbalance and macro/micro trade-off are confirmed | The main error is the high-support `0/1` boundary, not an isolated rare-class failure; posterior calibration and NLL semantics can shift | Defer as Intermittent fallback |
| Reopen V6 generic series memory | Pre-window coverage is sufficient | Frozen V6 primary and bootstrap gates failed; reopening would be post-hoc | Reject |
| Reopen V4 mark-conditioned time head | Existing implementation is available | It adds no new history and already failed validation | Reject |
| V7 causal time-history adapter | V6's secondary time signal is positive and series-wide; V3b still leaves time modeling unimproved | Existing probe mixed time, mark, and quantity features, so a time-only source must be isolated first | Select for Stage 0 audit |

## Decision

Select V7, a Taxi-first causal time-history adapter, as the next hypothesis.
Keep V5b `DEFERRED`; do not implement or screen both branches in parallel.

V7 is a new model line rather than a V6 retry:

- V6 would have fused generic pre-window event memory into the shared hidden
  representation and therefore affected marker, time, and value paths.
- V7 uses only temporal pre-window features and can change only the RMTPP time
  intercept.
- V7 does not use V6's selected `M=64`, `topk=4`, marker primary, or generic
  mark/quantity memory features.

The first V7 model contract, if Stage 0 passes, is:

```text
h_base = LMM_static(TitanEncoder(x_active_context))
t_pre  = TimeFeatures(x_strictly_pre_window)
r_time = MaskedTimeRetrieve(stop_gradient(h_base), t_pre, time_memory_mask)

a_base = v_t(h_base) + b_t
a_v7   = a_base + tanh(alpha_time) * delta_time(r_time)

alpha_time = 0 at initialization
lambda(dt | history) = exp(a_v7 + w * dt)
```

`TimeFeatures` may contain only observed temporal fields: `log1p(delta_t)`,
event age, and 24/168-hour phase encodings derived from observed sequence time.
It must not contain mark, quantity, series identity, target time, or a future
row. The adapter has its own projection parameters and must not share the Titan
input projection. Its query receives a stopped-gradient base hidden state so
the added retrieval route cannot directly update the shared encoder.

The base time path remains trainable as in V2/V3b. Because the combined time
likelihood can still change the shared encoder's ordinary time gradient during
end-to-end training, marker and quantity equality is an initialization and
empty-history contract, not a claim that independently trained checkpoints
will remain identical. Validation guardrails cover that indirect drift.

## Causality And Fallback Contract

For every next-event sample:

```text
pre_window_event_index < effective_context_start_index
effective_context_start_index <= context_end_index < target_index
```

- Every history event belongs to the same `oper_part_no` as the target.
- The target, future rows, and active-window rows are excluded from external
  time history.
- Padding is masked before retrieval and can never receive attention mass.
- No history and all-padding history produce `delta_time=0` exactly.
- `alpha_time=0` reproduces the paired V2/V3b time density, NLL components,
  marker logits, value predictions, quantity reconstruction, and sampling.
- Shuffled loader order and resume state cannot change a sample's history.
- Validation may use only events already observed before that validation target;
  no later validation event or test row is visible.
- Held-out test remains unread until a multi-seed candidate is frozen.

## Stage 0: Train-Only Time-Source Isolation

The prior V6 probe combined pre-window mark, quantity, and temporal features.
Therefore V7 model implementation remains locked until a new Taxi train-only
factorial establishes that the temporal source itself carries the signal.

Use identical eligible targets and rolling-origin folds to compare:

| Probe | Active-window input | Added pre-window input | Role |
| --- | --- | --- | --- |
| P0 | existing window summaries | none | baseline |
| P1 | same as P0 | temporal fields only | V7 source test |
| P2 | same as P0 | temporal + mark + quantity fields | attribution reference only |

The audit must predeclare one fixed pooling/retrieval rule. It may not select or
reuse V6's `M/topk` after observing fold results. Ridge and feature-scaler
parameters are fitted on each fold's expanding prefix only. Validation and test
parquets are not read.

Stage 0 passes only if P1 versus P0:

- improves out-of-fold `log1p(dt)` MAE by at least `1%` on the pooled targets;
- improves at least `2/3` rolling-origin evaluation folds;
- has a series-bootstrap 95% improvement interval with lower bound above zero;
- covers at least `35%` of train targets and `80%` of Taxi series with the
  predeclared minimum history requirement;
- remains finite and has no source, loader, or causal-ordering violation.

P2 quantifies whether non-temporal fields explain the old V6 signal but cannot
pass V7 on P1's behalf. The previously opened V6 final suffix is
hypothesis-generating evidence, not a new untouched holdout; the rolling-fold
result must be reported as train-only feasibility evidence rather than a final
statistical confirmation.

Failure closes V7 before model implementation and returns the next-selection
decision to V5b. Passing Stage 0 freezes the temporal feature contract and opens
focused implementation without changing the active incumbent.

### Stage-0 Implementation Status (2026-07-19)

The train-only P0/P1/P2 audit is implemented and its first 5080 execution is
complete. Artifact synchronization and protocol-order analysis remain pending:

- entrypoint: `simple_lab_test/search/analyze_taxi_time_source_isolation_audit.py`;
- 5080 runner: `simple_lab_test/search/scripts/run_titantpp_v7_taxi_time_source_audit_0719.sh`;
- fixed pooling: all strictly pre-window events summarized by predeclared
  temporal moments; V6 `M/topk` is not reused;
- fixed evaluation: three expanding rolling-origin folds, fold-local scaler and
  Ridge fit, and series bootstrap with seed `42`;
- focused tests cover source isolation, target/future causality, fold ordering,
  OOF uniqueness, bootstrap determinism, P1-only promotion, and the complete
  train-only artifact contract.

Local verification passed `6` V7 focused tests and `14` combined V6/V7 audit
tests. Source revision `ea874d2` passed the 5080 dependency, dataset, runner,
and `8/8` source-checksum preflight. Tmux execution started at
`2026-07-19 11:33:58 KST` and completed at `11:34:04 KST`. These execution facts
do not finalize the Stage-0 decision before the artifact-reading protocol is
completed.

## Taxi Factorial After Stage 0

| Variant | Value head | Quantity-mark route | Time-history adapter | Role |
| --- | --- | --- | --- | --- |
| V2 | shared | coupled | off | common/attribution control |
| V3b | mark-conditioned experts | detached | off | Taxi incumbent |
| V7a | shared | coupled | on | isolated time-history effect |
| V7b | mark-conditioned experts | detached | on | Taxi replacement candidate |

All variants keep `mid_lmm`, `static_lmm`, plain marker CE, residual value input,
hybrid quantity objective, target-only supervision, fixed split,
`lookback=168`, `max_seq_len=256`, optimizer, batch, seed, epoch budget,
checkpoint selection, and strict reproducibility settings matched. V7a and
V7b differ from their paired controls only by the time-history provider and
adapter.

## Implementation Gates

Focused tests must verify:

1. Zero-gate and empty-history exact equality against V2/V3b for forward,
   component NLLs, predictions, and sampled time with fixed `u`.
2. Target, future, active-window, cross-series, and padded rows cannot affect the
   retrieved time residual.
3. A valid temporal history can affect only the time intercept after a non-zero
   gate; mark/value heads receive no direct adapter gradient.
4. Gradients reach the gate, temporal projection, retrieval, and time delta.
5. The V3b detached quantity-to-mark contract remains unchanged.
6. Config, CLI, path, manifest, checkpoint, cache, resume, history, and summary
   identify the time-history mode and provider contract.
7. Strict shuffled-loader and resume runs reconstruct identical histories.

After focused tests, run a 5080 CUDA model-test and a one-epoch Taxi
validation-only integration smoke. These gates establish wiring only.

## Validation-Only Acceptance Gate

Run strict Taxi V2/V3b/V7a/V7b seed-42 e50 with fresh matched controls. At
`best_val_nll`, V7b advances only if all conditions hold against V3b:

- overall time NLL improves by at least `0.5%`;
- pre-window-eligible time NLL improves by at least `1%`;
- total NLL improves by at least `0.25%`;
- DT MAE regresses by no more than `1%`;
- marker NLL regresses by no more than `0.5%`;
- mark accuracy regresses by no more than `0.25%p`;
- quantity MAE and value MAE each regress by no more than `2%`;
- no quantity bucket with at least `5%` validation share regresses by over `5%`;
- at least `60%` of eligible Taxi series improve time NLL.

V7a is attribution-only and cannot replace V3b. If V7b passes, freeze the
architecture and run strict seeds `42,52,62`. Held-out test opens once only after
the multi-seed gate passes. A seed-42 failure closes the first V7 design without
validation-driven feature or gate tuning.

## V5b Deferred Contract

V5b remains a separate Intermittent branch from V2. If it is activated later,
its ADR must freeze train-only class-prior estimation, smoothing/capping,
train-versus-inference logit handling, and calibration metrics before any new
validation result is read. `nll_marker` must remain ordinary categorical CE for
reporting; a prior-corrected training objective must use a separate artifact
field. V5b must not be composed with V5a, V3, Q, or V7 in its first screen.

## Consequences And Risks

Positive consequences:

- The next experiment follows the only robust secondary signal left by V6
  without changing V6's failed decision.
- The adapter adds previously omitted information but limits its direct effect
  to the time likelihood.
- V2 and V3b remain exact nested controls at initialization and fallback.
- V5b remains available for the separate Intermittent problem.

Risks:

- The old time signal may depend on pre-window marks or quantities rather than
  temporal fields; Stage 0 detects this before model implementation.
- Only 131 Taxi series can make a learned retrieval branch overfit; series-level
  bootstrap and eligible-series gates are required.
- Dense all-history retrieval can add data-loader and GPU cost; implementation
  must report memory length and wall-clock overhead before e50.
- End-to-end time loss can indirectly move shared encoder parameters even though
  the adapter query is detached; marker and quantity guardrails remain binding.
- A single seed is only a resource gate, not promotion evidence.

## Non-Goals

- no retroactive V6 pass or reuse of `M=64/topk=4`
- no generic hidden-state memory fusion
- no mark, quantity, series-ID, or target/future feature in V7 time history
- no mark-conditioned V4 time delta
- no class-prior, ordinal, direct-quantity, or RevIN composition
- no stateful cross-window update or test-time parameter adaptation
- no 5090 execution while the temporary 5080 override is active
- no held-out access before model and coefficients are frozen

## Next Execution Order

1. On request, sync the completed 5080 artifacts locally.
2. Read manifest, log, summary, fold metrics, series metrics, and plots; apply
   the frozen Stage-0 gate.
3. Update the Notion result section only after the artifact analysis.
4. If Stage 0 passes, implement V7 and its focused contract tests. If it fails,
   close V7 and reopen the V5b design decision.
