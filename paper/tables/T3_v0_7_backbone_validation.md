# T3 v0.7. Three-seed mark-free backbone validation

> Mean +/- sample standard deviation. Lower is better. Held-out test data were not evaluated.

| Dataset | Model | Clamped time loss | Log-count MSE | Quantity MAE | Quantity RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| Intermittent-5000 | Count-aware RMTPP | -3.599494 +/- 0.000000 | 0.010517 +/- 0.000854 | 2.9025 +/- 0.1703 | 10.5787 +/- 0.6047 |
| Intermittent-5000 | Count-aware THP | -3.599485 +/- 0.000004 | 0.004184 +/- 0.000153 | 0.6664 +/- 0.0819 | 2.1507 +/- 0.5552 |
| Intermittent-5000 | Count-aware TitanTPP | -3.593078 +/- 0.000661 | 0.007050 +/- 0.000633 | 0.7469 +/- 0.0685 | 1.9195 +/- 0.2938 |
| Taxi | Count-aware RMTPP | 1.359763 +/- 0.001960 | 0.184898 +/- 0.001350 | 40.2893 +/- 3.2321 | 144.8148 +/- 14.0054 |
| Taxi | Count-aware THP | 1.362396 +/- 0.002819 | 0.194915 +/- 0.006842 | 41.5884 +/- 2.9369 | 147.4963 +/- 10.2086 |
| Taxi | Count-aware TitanTPP | 1.365797 +/- 0.001465 | 0.190733 +/- 0.001819 | 42.5805 +/- 8.0971 | 143.9931 +/- 32.5312 |
| Instacart | Count-aware RMTPP | 3.205289 +/- 0.000846 | 0.243269 +/- 0.000244 | 4.0270 +/- 0.0303 | 5.9901 +/- 0.0669 |
| Instacart | Count-aware THP | 3.206215 +/- 0.000547 | 0.245422 +/- 0.001697 | 4.0368 +/- 0.0270 | 6.0012 +/- 0.0625 |
| Instacart | Count-aware TitanTPP | 3.206516 +/- 0.000151 | 0.244365 +/- 0.000680 | 4.0450 +/- 0.0166 | 6.0159 +/- 0.0374 |
