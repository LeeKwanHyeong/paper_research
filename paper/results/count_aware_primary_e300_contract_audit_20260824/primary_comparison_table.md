# Compatible e300 primary comparison

Only Intermittent v2 currently satisfies the official mark-free T0/T1 contract.
Body is quantity <= train p95, tail is quantity > train p95, and extreme tail is quantity > train p99.
All values are validation mean +/- sample standard deviation over seeds 42, 52, and 62.

| Role | Model | Time NLL | Quantity MAE | Quantity RMSE | Body MAE | Tail MAE | >p99 MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T0 backbone control | Adapted RMTPP | -3.5995 +/- 0.0000 | 2.9025 +/- 0.1703 | 10.5787 +/- 0.6047 | 1.1621 +/- 0.0758 | 34.1956 +/- 1.8800 | 77.6442 +/- 4.6299 |
| T0 backbone control | Adapted THP | -3.5995 +/- 0.0000 | 0.6664 +/- 0.0819 | 2.1507 +/- 0.5552 | 0.4270 +/- 0.0099 | 4.9703 +/- 1.7270 | 11.2654 +/- 6.1927 |
| T0 backbone control | Adapted NHP | -3.5995 +/- 0.0000 | 5.2827 +/- 0.1647 | 15.4244 +/- 0.7389 | 2.6941 +/- 0.0615 | 51.8262 +/- 2.5763 | 108.0919 +/- 5.6002 |
| T0 backbone control | Adapted SAHP | -3.5995 +/- 0.0000 | 1.0815 +/- 0.0085 | 3.7724 +/- 0.1750 | 0.6108 +/- 0.0245 | 9.5448 +/- 0.4544 | 17.3515 +/- 1.6783 |
| T0 backbone control | TitanTPP-T0 | -3.5931 +/- 0.0007 | 0.7469 +/- 0.0685 | 1.9195 +/- 0.2938 | 0.5349 +/- 0.0637 | 4.5586 +/- 1.1825 | 7.7458 +/- 3.1850 |
| Tail-aware objective effect | TitanTPP-T1 | -3.5932 +/- 0.0009 | 0.6989 +/- 0.0597 | 1.7997 +/- 0.1821 | 0.5032 +/- 0.0314 | 4.2182 +/- 1.0073 | 7.2109 +/- 2.2434 |

T0 rows isolate encoder/backbone effects under direct log-MSE. TitanTPP-T1 must be compared with TitanTPP-T0 to isolate the added tail-aware objective; it is not a pure backbone row.
Taxi and Instacart are excluded because their existing e300 checkpoints retain the marked-hybrid interface and use best event NLL rather than the official mark-free validation joint objective.
Held-out test data were not used.
