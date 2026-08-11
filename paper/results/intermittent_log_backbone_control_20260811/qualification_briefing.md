# Intermittent Log-Head Backbone Qualification

## Contract

- Same frozen 5,000-series split, log1p regression head, seeds, optimizer, and checkpoint rule across all backbones.
- History ranges are fixed at <=64, 65-128, and >128 observed events.
- Held-out test remains locked; this report uses validation only.
- Event NLL is the sum of mark and time NLL; the log-quantity regression loss is reported separately and is not included in NLL.
- Source revision: `8099d07e8fc6d8bcbbd7990f6034fa367f3aee5d`.
- Data SHA-256: `85d1fe3ade3ae5a90241018e99a3e9463828d5ba35bc374b56def0168ffffc3f`.

## Overall Results

| Model | Event NLL (mark + time) | Time NLL | Quantity MAE | Mark accuracy | Parameters |
|---|---:|---:|---:|---:|---:|
| Adapted RMTPP | -3.5136 +/- 0.0010 | -3.5995 +/- 0.0000 | 0.6786 +/- 0.0466 | 0.9653 +/- 0.0002 | 21726 |
| Adapted THP | -3.4631 +/- 0.0078 | -3.5948 +/- 0.0047 | 1.8910 +/- 0.3320 | 0.9497 +/- 0.0012 | 101582 |
| TitanTPP | -3.4959 +/- 0.0080 | -3.5964 +/- 0.0052 | 0.6388 +/- 0.0243 | 0.9588 +/- 0.0012 | 93342 |

## History-Length Event NLL (Mark + Time)

| Model | <=64 | 65-128 | >128 |
|---|---:|---:|---:|
| Adapted RMTPP | -3.5238 | -3.4995 | -3.5180 |
| Adapted THP | -3.4895 | -3.4260 | -3.4751 |
| TitanTPP | -3.5090 | -3.4862 | -3.4942 |

## Pre-Registered Gates

- Overall event NLL best: **False**
- Overall time NLL best: **False**
- Paired overall event NLL consistency: **False**
- Long-history event NLL best: **False**
- Long-history time NLL best: **False**
- Quantity MAE within 10% of the best baseline: **True**

## Decision: **BACKBONE CLAIM NOT QUALIFIED**

This decision is a validation-stage go/no-go result. A qualified claim still requires one locked held-out test evaluation after the manuscript configuration is frozen.
