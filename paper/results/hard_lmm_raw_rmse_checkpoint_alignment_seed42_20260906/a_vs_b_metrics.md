# Hard-LMM Raw-RMSE Checkpoint Alignment: A vs B

- Common raw-RMSE goal: **MET**
- Scope: validation only; held-out test and additional seeds were not evaluated.
- C is excluded from the revised comparison and adoption; already completed or partial artifacts remain recorded.
- Body MAE, >p99 MAE, and time NLL are descriptive for A→B; the frozen C gate was not reinterpreted.

| Dataset | Arm | Raw RMSE | Body MAE | >p99 MAE | Time NLL | B improves RMSE |
| --- | --- | ---: | ---: | ---: | ---: | :---: |
| intermittent_frozen_5000 | A_t0_joint | 1.72240325 | 0.60336928 | 6.06798283 | -3.59285604 |  |
| intermittent_frozen_5000 | B_t0_raw_rmse | 1.49955507 | 0.46578960 | 5.14810801 | -3.48379369 | yes |
| yellow_trip_hourly | A_t0_joint | 181.53759358 | 23.16740661 | 1166.28242126 | 1.36643025 |  |
| yellow_trip_hourly | B_t0_raw_rmse | 88.19499665 | 18.99747812 | 332.76974449 | 1.47339063 | yes |
| insta_market_basket | A_t0_joint | 5.97416001 | 3.42454564 | 23.00099934 | 3.20644001 |  |
| insta_market_basket | B_t0_raw_rmse | 5.87221693 | 3.43860150 | 21.91779477 | 3.21552290 | yes |

## A→B effects

- `intermittent_frozen_5000`: raw RMSE improvement 12.9382%; body MAE change -22.8019%; >p99 MAE change -15.1595%; time NLL change 0.10906235.
- `yellow_trip_hourly`: raw RMSE improvement 51.4178%; body MAE change -17.9991%; >p99 MAE change -71.4675%; time NLL change 0.10696038.
- `insta_market_basket`: raw RMSE improvement 1.7064%; body MAE change 0.4104%; >p99 MAE change -4.7094%; time NLL change 0.00908288.
