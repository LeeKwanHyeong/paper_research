# Frozen-5000 Intermittent Result Qualification

## Contract Verification

The six Adapted RMTPP/THP runs from 5080 and the three TitanTPP runs from 5090 use the same frozen 5,000-series split, source revision, three seeds, 300-epoch ceiling, and validation-NLL checkpoint rule. Both contracts are complete, validation-only, and record `held_out_test_evaluated=false`. All nine run rows are successful.

- Source revision: `308cec0b9c383d4eab5aac8b9015dae663b0ad73`
- Split manifest SHA-256: `393158a54a8ca703dbf7e9311b9dff6d2825ef737e3e3de1c30a1f3ff64c1c04`
- Validation parquet SHA-256: `53ca2328d1c9408465100bead285cec5135066fd46afda58d39f23cb50ecd0e6`
- Shared interface: 10 marks, exponent + residual decoder, maximum sequence length 96

## Three-Seed Validation Summary

| Model | NLL | Time NLL | Mark NLL | Quantity MAE | Quantity RMSE | Mark accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Adapted RMTPP | -3.4270 +/- 0.0050 | -3.5155 +/- 0.0051 | 0.0886 +/- 0.0009 | 0.7091 +/- 0.0273 | 1.9723 +/- 0.0528 | 96.49 +/- 0.06% |
| Adapted THP | -3.4277 +/- 0.0060 | -3.5388 +/- 0.0049 | 0.1111 +/- 0.0029 | 0.8227 +/- 0.0395 | 2.1869 +/- 0.0755 | 95.82 +/- 0.05% |
| TitanTPP | -3.4469 +/- 0.0021 | -3.5379 +/- 0.0029 | 0.0910 +/- 0.0014 | 0.7255 +/- 0.0404 | 2.0159 +/- 0.0984 | 96.25 +/- 0.07% |

Lower is better for NLL and quantity errors; higher is better for mark accuracy. The reported `+/-` value is the sample standard deviation across seeds 42, 52, and 62.

## Evidence Assessment

TitanTPP lowers validation NLL against Adapted RMTPP in all three paired seeds (mean difference -0.0199). The time-likelihood term is lower by 0.0223, while the mark term is +0.0024 worse. This supports a narrow temporal-likelihood improvement over the recurrent baseline.

Against Adapted THP, TitanTPP also lowers total NLL in all three seeds (mean difference -0.0192), but its time-likelihood term is +0.0009 worse. The advantage comes primarily from the mark term, so this comparison does not establish a general temporal-dependency advantage over THP.

For quantity reconstruction, TitanTPP improves MAE by 11.8% relative to Adapted THP, but is 2.3% worse than Adapted RMTPP. It is therefore the middle-ranked model rather than a universal winner.

At quantities 64-127, TitanTPP has the lowest MAE (2.308; Adapted RMTPP: 2.519; Adapted THP: 2.430). At quantities >=128, Adapted RMTPP is best (3.978), followed by TitanTPP (4.292). The tail evidence is mixed and does not support an across-the-board long-tail quantity claim.

The delta-time MAE is exactly identical across all nine rows. It is not discriminative in this experiment and should not be used as evidence of backbone superiority until the point-prediction path is audited.

## Qualification Decision

**Share with caveats.** The result is qualified as validation evidence under a frozen contract. It can support a narrow statement that TitanTPP improves total event likelihood over both adapted baselines and temporal likelihood over Adapted RMTPP. It cannot support the stronger claims that TitanTPP uniformly improves temporal modeling over THP or that the exponent-residual representation yields the best quantity accuracy.

The next controlled experiment should place the same fair log-scale quantity head on all three backbones. That isolates the backbone contribution from the quantity representation and aligns this Intermittent result with the completed Taxi quantity-interface finding.
