# Count-aware TitanTPP-MAC B1 Audit Interpretation

## Audit boundary

This is validation-only, seed-42, same-checkpoint attribution. It compares the
frozen B1 checkpoint with online writes disabled and with long-term reads and
writes neutralized. It is not a retrained ablation, and held-out test data
remain locked. The neutral long-term-memory control preserves zero-valued MAC
retrieval slots, so it isolates learned memory content rather than removing the
MAC topology.

## Dataset findings

### Taxi

B1 improves over B0 by 16.14% on body MAE, 27.91% on overall RMSE, and 39.99%
on extreme-tail MAE. In the same B1 checkpoint, disabling online writes raises
overall MAE by 18.60, while neutralizing long-term memory raises it by 14.05.
The online-write gain is concentrated in histories longer than 128 events:
full B1 reduces MAE by 27.67 versus no-update there, compared with 0.13 for
histories 65--128 and 0.02 for histories up to 64. Online writes also reduce
extreme-tail MAE by 512.33. Learned long-term content helps body and p95--p99,
but raises >p99 MAE by 91.71 versus the neutral control. The broad Taxi gain is
therefore primarily consistent with causal online adaptation over long history,
not uniformly beneficial static memory content.

### Intermittent

B1 improves over B0 by 25.98% on body MAE and lowers overall MAE, but overall
RMSE regresses by 10.67% and >p99 MAE regresses by 22.12%. The same-checkpoint
controls do not support the hypothesis that the memory residual causes this
tail regression: disabling online writes raises overall MAE by 0.284 and >p99
MAE by 0.505, while neutralizing long-term content raises overall MAE by 9.585
and >p99 MAE by 160.906. However, online writes harm 52.50% of >p99 events even
though their mean effect is beneficial, indicating heterogeneous tail behavior.
The remaining B0-to-B1 tail gap is attributable to the learned B1 representation
and optimization trajectory rather than a uniformly harmful runtime residual.

### RAF Spare Parts

B1 is effectively neutral relative to B0: body MAE improves by 0.86%, while
overall RMSE and >p99 MAE regress by 1.67% and 1.76%. The online-write residual
norm and full-minus-no-update deltas are exactly zero for all 6,690 validation
events, and all events fall in the no-prior-visible-write surprise scope. Thus
this validation construction provides no causal opportunity for online memory
to affect a later segment. Neutralizing long-term content raises body MAE by
0.601 but slightly improves upper-tail error, yielding only a 0.529 overall MAE
benefit. RAF is evidence for a small initial-memory body effect, not for online
adaptation.

## Compute finding

The frozen seed-42 B1 runs cost 5.45x, 6.86x, and 7.23x B0 seconds per completed
epoch on Intermittent, Taxi, and RAF. Steady validation forward ratios were
6.39x, 6.63x, and 9.80x. Cold B1 forward included up to 8.33 seconds of compile
overhead. This supports optimizing recurrence scheduling and compile-shape reuse
without changing the Titans-MAC equations. The official optimization gate
remains at most 3x B0 steady training-step cost on RTX 5080.

## Decision

The audit supports retaining B1 as the primary candidate because its memory is
causally useful on Taxi and Intermittent and not the direct cause of the observed
Intermittent mean tail regression. It also limits the claim: benefit depends on
history that crosses a write/read boundary, and extreme-tail effects are not
uniform. The model remains subject to the frozen three-seed acceptance rule and
the 3x compute guardrail before held-out testing.
