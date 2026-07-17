# ADR: TitanTPP V6 Causal Pre-Window Series Memory

- Date: 2026-07-17
- Status: Train-only audit running on 5090 since 2026-07-17 09:27:41 KST; model adapter not implemented
- Scope: TitanTPP series-aware long-horizon memory
- Baselines: V2 common baseline and Taxi V3b confirmed enhancement

## Context

V2 remains the common TitanTPP baseline. V3b is the Taxi-specific incumbent,
where its matched multi-seed gain is driven by marker and quantity modeling while
time NLL remains slightly worse than V2. V4 tested a mark-conditioned time head
but failed its Taxi validation-only promotion gate. The next enhancement should
therefore add information that neither V2 nor V3b currently observes rather than
reopen a failed output-head or direct-quantity branch.

Taxi is the strongest first dataset for a long-horizon memory hypothesis:

- each series has mean length about `420.76`, median `405`, and maximum `744`
- the active window is `lookback=168`, `max_seq_len=256`
- the same `131` series recur over long chronological histories
- events before the current 168-hour window are omitted even though they are
  already observed and may contain series-specific weekly or level information

Intermittent and Instacart are not the first V6 quality targets. Their median or
p95 histories are mostly short enough to fit the active context, and Instacart
has `206,209` series, making identity-based memory especially risky.

The code currently exposes `memory_mode=series_lmm`, but that is only a hook:

- the common runner discards `part_idx` and never passes `series_memory`
- absent memory makes `series_lmm` an encoder-only fallback
- supplied memory replaces the learned static LMM bank rather than extending it
- `LMM.forward` adds the retrieved mean with fixed coefficient `1`
- there is no external-memory padding mask or leakage contract

The historical e800 memory-mode screening used `residual_only`, did not inject
series memory, and predates V2/V3b. It is useful only as a warning that
stateful/contextual memory can be unstable; it is not V6 model-quality evidence.

## Hypothesis

> Strictly past events from the same Taxi series that lie before the active
> 168-hour context contain complementary long-horizon information. A masked,
> zero-initialized residual retrieval adapter can use that information while
> preserving the exact V2/V3b function at initialization.

## Options Considered

| Option | Benefit | Main risk | Decision |
| --- | --- | --- | --- |
| V5b class-prior correction | Directly targets Intermittent imbalance | Objective calibration rather than Titan model enhancement; can distort likelihood | Defer |
| Deeper/wider Titan | Easy capacity increase | Hyperparameter change without an isolated mechanism | Reject as next hypothesis |
| Stateful contextual TTM across windows | Long online memory | Order-dependent training, resume/state complexity, leakage risk | Reject for first V6 |
| Learned series-ID embedding | Cheap persistent identity | Memorization, cold-start failure, prohibitive Instacart table | Reject |
| Existing `series_lmm` switch | Already scaffolded | Drops static LMM and lacks mask/gate/provider | Reject as a clean experiment |
| Causal pre-window memory adapter | Adds previously omitted same-series evidence and can nest V2/V3b exactly | Requires dataset metadata, retrieval mask, and leakage tests | Select |

## Decision

Select V6 as an orthogonal `causal_pre_window` series-memory adapter. Keep the
incumbent candidate and `memory_mode=static_lmm`; do not activate
`small_series_lmm` or `mid_series_lmm` directly.

For one target sample, let:

```text
pre-window memory = events [0, ..., j-1]
active context    = events [j, ..., i]
target            = event i+1
```

The model contract is:

```text
h_base = LMM_static(TitanEncoder(x_context))
m      = InputProjection(x_pre_window)             # [B, M, D]
r_mem  = MaskedRetrieve(h_base, m, memory_mask)
h_v6   = h_base + tanh(alpha_series) * r_mem

alpha_series = 0 at initialization
```

`r_mem` is the retrieved memory residual, not `h_base + residual`. A zero gate
must make V6 bitwise-equivalent to its paired V2 or V3b control even when valid
memory is supplied. The gate is bounded with `tanh` to prevent an unscaled
external residual from dominating the trained static LMM path.

The first implementation may reuse the existing event feature builder and
`MemoryEncoder.input_proj` to map observed pre-window events to `d_model`.
Exact memory length, sampling policy, and retrieval `topk` remain unfrozen until
the train-only audit. No full pre-window encoder or series-ID table is allowed in
V6a/V6b.

## Causality And Leakage Contract

For every sample, the following ordering must be asserted:

```text
memory_event_index < context_start_index <= context_end_index < target_index
```

- memory events must belong to the same `oper_part_no`
- the appended target and all future rows are excluded
- current-window events are excluded from external memory to isolate added context
- memory construction is stateless per sample, so shuffled training order cannot
  change the memory contents
- a padded memory token is never eligible for top-k retrieval
- an empty memory produces the exact paired baseline output
- validation may use earlier validation events only after they have become
  chronologically observed; no later validation or test event is visible
- held-out test rows and metrics remain unread during audit and first screening

The dataset already carries `part_idx`, but V6 additionally needs the target and
context-start indices or an equivalent precomputed memory slice. Those fields
must be included in cache/resume and strict reproducibility identity.

## Taxi Factorial Contract

| Variant | Value head | Quantity-mark gradient | Static LMM | Causal series adapter | Role |
| --- | --- | --- | --- | --- | --- |
| V2 | shared | coupled | on | off | common/attribution control |
| V3b | mark-conditioned experts | detached | on | off | Taxi incumbent |
| V6a | shared | coupled | on | on | isolated memory effect |
| V6b | mark-conditioned experts | detached | on | on | Taxi replacement candidate |

All four variants keep `mid_lmm`, seed, split, epoch, optimizer, `lookback=168`,
`max_seq_len=256`, residual input, hybrid objective, plain CE, target-only loss,
shared time head, and checkpoint selection fixed. V6a/V6b differ from their
paired controls only by the causal memory adapter.

## Stage 0: Taxi Train-Only Audit

No model implementation or validation run starts before this audit passes. Read
only fixed-split Taxi train rows and produce:

- pre-window event-count coverage at `>=1`, `>=8`, `>=16`, `>=32`, and `>=64`
- median/p90/p95 pre-window count and chronological span
- context truncation share caused by the 168-hour boundary versus `max_seq_len`
- support by series and target mark, so gains cannot come from a few long series
- train-internal chronological probes comparing current-window summaries against
  current-window plus pre-window summaries for marker CE, `log1p(dt)` error, and
  `log2(qty)` error
- candidate memory budgets `M={16,32,64,128}` and retrieval `topk={4,8}`, selected
  using coverage, compute, and train-internal evidence only

Proceed to constants freeze only if:

- at least `35%` of eligible train targets have `>=8` pre-window events
- at least `80%` of the 131 series contribute one eligible memory target
- the augmented train-internal probe improves at least one of marker CE,
  `log1p(dt)` MAE, or `log2(qty)` MAE by `>=1%`
- the other two probe metrics regress by no more than `1%`
- the series-bootstrap 95% interval for the improved metric has a positive lower
  bound

Audit failure closes V6 before implementation. The thresholds are feasibility
gates, not model-performance claims.

### Frozen Audit Implementation Contract

The audit implementation is available at:

```text
simple_lab_test/search/analyze_taxi_pre_window_memory_audit.py
simple_lab_test/search/scripts/run_titantpp_v6_taxi_train_memory_audit_0717.sh
simple_lab_test/search/tests/test_taxi_pre_window_memory_audit.py
```

It uses the exact `RMTPPWeekLookbackDataset` train target index and reconstructs
both the temporal start and the effective start after `max_seq_len` truncation.
Because `max_seq_len` includes the appended target, at most
`max_seq_len - 1` active context events remain. External memory contains only
indices strictly smaller than the effective context start.

The Taxi train targets are partitioned chronologically within each series into
`70% probe_fit / 15% probe_selection / 15% probe_audit`. The final suffix is not
used to select `M`, `topk`, the primary metric, or a probe hyperparameter.

The fixed low-capacity probes are:

- plain multinomial logistic regression for next-mark CE
- ridge regression for next `log1p(delta_t)` MAE
- ridge regression for next `log2(quantity)` MAE
- current-window summaries for the baseline input
- the same summaries plus top-k same-series pre-window aggregates for the
  augmented input

The retrieval proxy standardizes event keys from the allowed fit prefix only,
uses the last eight active context events as the query, and evaluates
`M={16,32,64,128}` with `topk={4,8}`. Candidate ranking first requires the
selection guard, then maximizes the worst metric improvement, maximizes mean
improvement, and prefers smaller `M` and `topk`. The strongest selection metric
is frozen as the final primary metric before opening `probe_audit`.

Focused synthetic tests verify temporal and `max_seq_len` boundaries, exact
loader target correspondence, chronological partition order, target/future
invariance, valid-memory sensitivity, coverage denominators, candidate
tie-breaking, finite probe metrics, and all acceptance checks. These tests do
not provide Taxi audit evidence. The 5090 train-only run passed source,
dependency, dataset, runner, and launch-conflict preflight and entered coverage
decoding at `2026-07-17 09:27:41 KST`. Result artifacts remain unread until a
user-requested completion check.

## Implementation Gate After Audit

If the audit passes, focused tests must verify:

1. V2/V6a and V3b/V6b parameter correspondence outside the adapter.
2. Zero-gate forward outputs, NLL components, quantity predictions, and sampling
   are exactly equal to paired controls with empty and non-empty memory.
3. No memory, all-padding memory, and masked padding produce identical fallback.
4. Changing target/future/cross-series rows cannot change a sample's memory.
5. Changing valid pre-window rows can change V6 only after a non-zero gate.
6. Gradients reach the gate and valid memory path without altering the V3b
   detached quantity-to-mark contract.
7. Strict shuffled-loader and resume runs reconstruct identical memory slices.
8. Config, path, manifest, checkpoint, cache, history, and summary record the
   memory source, budget, top-k, gate, and audit-constants identity.

## Validation-Only Promotion Gate

After local tests, run a 5090 CUDA model-test and one-epoch Taxi integration
smoke. The first quality screen is strict Taxi V2/V3b/V6a/V6b seed-42 e50,
validation-only.

V6b advances only if all conditions hold at `best_val_nll`:

- total NLL improves by at least `0.5%` relative to V3b
- memory-eligible target NLL improves by at least `1%` relative to V3b
- marker NLL and time NLL each regress by no more than `0.5%`
- mark accuracy regresses by no more than `0.25%p`
- DT MAE regresses by no more than `1%`
- quantity MAE regresses by no more than `5%`
- no quantity bucket with at least `5%` validation share regresses by over `5%`
- at least `60%` of memory-eligible series improve total NLL

V6a is an attribution control and does not replace Taxi V3b by itself. If V6b
passes, run strict seeds `42,52,62`; freeze the selected architecture and
constants before opening held-out test once. If it fails, retain V3b and close
the first V6 design without tuning on validation.

## Non-Goals

- no learned per-series ID embedding
- no cross-window mutable state or test-time parameter update
- no target/future/current-window duplication in external memory
- no removal of the trained static LMM bank
- no V4 time-head, V5 marker-loss, or Q-series direct-magnitude composition
- no Intermittent or Instacart model-quality claim in the first V6 screen
- no held-out test access before multi-seed promotion and architecture freeze

## Next Execution Order

1. Implement a Taxi train-only pre-window support and predictiveness audit - completed; no Taxi result read.
2. Checksum-sync commit `6d7ed32` to 5090 - completed, `7/7` files match.
3. Run dependency, dataset, source-revision, and command preflight; then start
   the audit once in 5090 tmux - completed; started `2026-07-17 09:27:41 KST`
   and initial coverage decoding confirmed.
4. Read manifest, log, summary, coverage, candidate metrics, final probe metrics,
   and plots; freeze or reject `M`, `topk`, and the coverage policy.
5. If the audit passes, implement the masked zero-init adapter and focused tests.
6. Run 5090 CUDA and Taxi e1 integration gates.
7. Run strict Taxi V2/V3b/V6a/V6b seed-42 e50 validation-only screening.
