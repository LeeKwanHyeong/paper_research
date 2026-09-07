# Conditional CDF calibration e300 convergence decision

The e300 budget extension confirms that the Taxi candidate had converged at
the e100 boundary result. The candidate selected epoch 99 again, produced no
new validation minimum through epoch 119, and then stopped under the unchanged
20-epoch patience rule. The common candidate therefore remains rejected.

## Frozen extension

- Source revision: `f11fbe2cdff88ced113213459743d4d835bd7276`
- Contract SHA-256: `6c3b822458e850b2871dfa633e61b6d5f922841bafc3bcbe206b0fbf9111acec`
- Parent contract SHA-256: `86632f4f7ea22960c8f028cc8e2347986ae837bee29c9fee804fdf25b46c1a52`
- Maximum epoch budget: 300
- The model, initialization, optimizer, learning rate, batch order, selector,
  patience, controls, and acceptance thresholds are unchanged from v1.
- The run started again at epoch zero. Candidate epochs 0--100, both control
  histories, and all selected-state digests matched v1 exactly.
- The passed v1 full-data e1 evidence was reused because implementation and
  inputs were unchanged.
- Evaluation remained validation-only; held-out test was not evaluated.

## Convergence

| Path | Completed epoch | Selected epoch | Early stopped | Primary Time NLL |
| --- | ---: | ---: | --- | ---: |
| Candidate | 119 | 99 | yes | 0.6889874164 |
| Global-shape control | 20 | 0 | yes | 0.6956474254 |
| Permuted-hidden control | 49 | 29 | yes | 0.6934882532 |
| aligned-B | fixed | 0 | n/a | 0.6956474254 |
| aligned-A | fixed | fixed | n/a | 0.6506978539 |

The extra budget did not change any selected checkpoint or metric. Candidate
Time NLL improved over aligned-B by `0.0066600090`, but remained
`0.0382895625` above aligned-A. Its `0.0045008368` advantage over the
permuted-hidden control remained below the required `0.005` margin.

Time-median MAE remained `2.6829%` worse than aligned-B and failed the `2%`
guardrail. Time-median RMSE remained `0.1425%` worse and passed its guardrail.
The quantity path and frozen source state remained unchanged.

## Decision

The e100 failure was not caused by stopping at the epoch budget. It is now a
convergence-supported rejection under the fixed criteria. The Taxi-first
controller therefore did not run Intermittent, Instacart, additional seeds, or
held-out evaluation.
