# T1. Dataset statistics and task construction

> Status: frozen fixed-split dataset audit PASS. This table does not use model predictions or held-out test performance.

## T1a. Dataset and split statistics

| Dataset | Sequences | Events | Events T/V/Test | Targets T/V/Test | Seq. length med/p95/max | Quantity med/p95/max | Marks K | Base b |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Intermittent | 23,387 | 242,888 | 159,643 / 41,901 / 41,344 | 136,256 / 41,901 / 41,344 | 6 / 35 / 110 | 2 / 16 / 5,000 | 11 | 2 |
| Taxi | 131 | 55,119 | 38,524 / 8,268 / 8,327 | 38,393 / 8,268 / 8,327 | 405 / 743 / 744 | 7 / 1,547 / 6,489 | 4 | 10 |
| Instacart | 206,209 | 3,279,521 | 2,197,401 / 503,733 / 578,387 | 1,991,192 / 503,733 / 578,387 | 10 / 50 / 100 | 8 / 25 / 177 | 8 | 2 |

`Events` counts every fixed-split event row. `Targets` excludes the first event of each sequence because next-event prediction requires an observed predecessor. T/V/Test denotes train/validation/test.
The nominal 70/15/15 split is applied chronologically within each sequence. Aggregate event shares can differ from the nominal ratio because sequence boundaries are integer-valued and many sequences are short.

## T1b. Task construction

| Dataset | Sequence unit | Time unit | Event | Quantity |
| :--- | :--- | :--- | :--- | :--- |
| Intermittent | part | week | Part-level positive-demand episode after intermittent-demand preprocessing | Demand quantity aggregated within the episode |
| Taxi | 0.02-degree pickup grid cell | hour | Active grid-cell-hour pickup event | Number of taxi pickups in the grid cell during the hour |
| Instacart | user | day | User order event on the cumulative relative-day timeline | Number of products in the order basket |

## Frozen dataset identity

| Dataset | with-split SHA-256 | manifest SHA-256 | contract SHA-256 |
| :--- | :--- | :--- | :--- |
| Intermittent | dab4d8a7217f | 49752a1bd4cc | 12c8e5b920e0 |
| Taxi | b47e98e9fdb7 | 4a005d4a77a8 | 27cb86673d1f |
| Instacart | 06296e48f5ca | 6c6cdd41f847 | 802310581daf |

The manuscript may cite the 12-character identifiers above. Full hashes, sizes and paths are stored in `paper/data/T1_dataset_hashes.csv` and `paper/data/T1_dataset_audit.json`.
