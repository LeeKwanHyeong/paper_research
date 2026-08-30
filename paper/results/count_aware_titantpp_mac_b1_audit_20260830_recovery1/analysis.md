# Count-aware TitanTPP-MAC B1 Audit

This is a validation-only, same-checkpoint attribution audit. It is not a
retrained ablation and does not use held-out test data.

## Memory attribution

| Dataset | Full - no update MAE | Help share | Harm share | Full - neutral LTM MAE |
|---|---:|---:|---:|---:|
| intermittent_frozen_5000 | -0.284044 | 76.81% | 23.16% | -9.585105 |
| raf_spare_parts | +0.000000 | 0.00% | 0.00% | -0.528943 |
| yellow_trip_hourly | -18.602890 | 57.66% | 41.56% | -14.047230 |

Negative deltas mean that the enabled memory path reduced MAE. Surprise
strata use only writes completed before the prediction segment begins.
The neutral-LTM control leaves zero-valued retrieved token slots in MAC,
so it isolates the learned long-term content rather than deleting topology.

## Historical training cost

| Dataset | B1/B0 seconds per completed epoch |
|---|---:|
| intermittent_frozen_5000 | 5.448x |
| yellow_trip_hourly | 6.859x |
| raf_spare_parts | 7.233x |

These ratios describe the frozen seed-42 runs before the
semantics-preserving optimization pass.
