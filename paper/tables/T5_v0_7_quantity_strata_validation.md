# T5 v0.7. Train-defined quantity-stratified validation errors

> Mean +/- sample standard deviation over seeds 42, 52, and 62. Stratum thresholds are fitted on each train split and then fixed for validation targets.

| Dataset | Validation target stratum | Share | Model | Quantity MAE | Quantity RMSE |
| --- | --- | ---: | --- | ---: | ---: |
| Intermittent-5000 | <= 2 | 52.19% | Count-aware RMTPP | 0.0479 +/- 0.0018 | 0.1418 +/- 0.0052 |
| Intermittent-5000 | <= 2 | 52.19% | Count-aware THP | 0.0676 +/- 0.0058 | 0.1697 +/- 0.0020 |
| Intermittent-5000 | <= 2 | 52.19% | Count-aware TitanTPP | 0.0899 +/- 0.0140 | 0.2274 +/- 0.0128 |
| Intermittent-5000 | (2, 31] | 34.28% | Count-aware RMTPP | 1.8886 +/- 0.0874 | 2.2907 +/- 0.0846 |
| Intermittent-5000 | (2, 31] | 34.28% | Count-aware THP | 0.7711 +/- 0.0180 | 1.2985 +/- 0.0223 |
| Intermittent-5000 | (2, 31] | 34.28% | Count-aware TitanTPP | 0.9494 +/- 0.0533 | 1.5885 +/- 0.0418 |
| Intermittent-5000 | (31, 46] | 8.26% | Count-aware RMTPP | 5.1897 +/- 0.5794 | 5.4427 +/- 0.5674 |
| Intermittent-5000 | (31, 46] | 8.26% | Count-aware THP | 1.2700 +/- 0.0835 | 2.1081 +/- 0.0477 |
| Intermittent-5000 | (31, 46] | 8.26% | Count-aware TitanTPP | 1.6277 +/- 0.5374 | 2.4083 +/- 0.4270 |
| Intermittent-5000 | (46, 187] | 3.95% | Count-aware RMTPP | 19.6362 +/- 1.1170 | 22.8737 +/- 1.2991 |
| Intermittent-5000 | (46, 187] | 3.95% | Count-aware THP | 2.8608 +/- 0.2984 | 4.2404 +/- 0.4234 |
| Intermittent-5000 | (46, 187] | 3.95% | Count-aware TitanTPP | 3.4905 +/- 0.6330 | 4.9744 +/- 0.8668 |
| Intermittent-5000 | > 187 | 1.32% | Count-aware RMTPP | 77.6442 +/- 4.6299 | 81.1077 +/- 4.7253 |
| Intermittent-5000 | > 187 | 1.32% | Count-aware THP | 11.2654 +/- 6.1927 | 14.7398 +/- 5.6727 |
| Intermittent-5000 | > 187 | 1.32% | Count-aware TitanTPP | 7.7458 +/- 3.1850 | 9.7330 +/- 3.5619 |
| Taxi | <= 7 | 52.78% | Count-aware RMTPP | 1.4337 +/- 0.0800 | 3.2961 +/- 0.2783 |
| Taxi | <= 7 | 52.78% | Count-aware THP | 1.4532 +/- 0.0783 | 3.3751 +/- 0.2765 |
| Taxi | <= 7 | 52.78% | Count-aware TitanTPP | 1.4021 +/- 0.0608 | 3.2603 +/- 0.3278 |
| Taxi | (7, 686] | 37.95% | Count-aware RMTPP | 25.9454 +/- 1.6103 | 50.9437 +/- 5.2733 |
| Taxi | (7, 686] | 37.95% | Count-aware THP | 27.9098 +/- 1.2862 | 56.0572 +/- 1.4089 |
| Taxi | (7, 686] | 37.95% | Count-aware TitanTPP | 29.6474 +/- 3.6401 | 62.7305 +/- 11.1963 |
| Taxi | (686, 1562] | 4.67% | Count-aware RMTPP | 188.1005 +/- 6.3807 | 242.5942 +/- 13.8451 |
| Taxi | (686, 1562] | 4.67% | Count-aware THP | 207.8030 +/- 23.8775 | 267.8249 +/- 24.3498 |
| Taxi | (686, 1562] | 4.67% | Count-aware TitanTPP | 208.4336 +/- 29.0443 | 271.8409 +/- 36.0027 |
| Taxi | (1562, 3449] | 3.64% | Count-aware RMTPP | 338.5358 +/- 62.8890 | 431.6059 +/- 49.1807 |
| Taxi | (1562, 3449] | 3.64% | Count-aware THP | 332.2865 +/- 6.1285 | 427.3078 +/- 7.3738 |
| Taxi | (1562, 3449] | 3.64% | Count-aware TitanTPP | 383.7385 +/- 108.5504 | 489.6231 +/- 99.8226 |
| Taxi | > 3449 | 0.96% | Count-aware RMTPP | 897.8846 +/- 167.2097 | 1041.8141 +/- 148.0761 |
| Taxi | > 3449 | 0.96% | Count-aware THP | 882.2762 +/- 127.1022 | 1049.4498 +/- 116.3896 |
| Taxi | > 3449 | 0.96% | Count-aware TitanTPP | 720.7966 +/- 411.4079 | 821.8885 +/- 395.1543 |
| Instacart | <= 8 | 49.16% | Count-aware RMTPP | 2.5380 +/- 0.0456 | 3.5447 +/- 0.1106 |
| Instacart | <= 8 | 49.16% | Count-aware THP | 2.5183 +/- 0.0233 | 3.5502 +/- 0.0981 |
| Instacart | <= 8 | 49.16% | Count-aware TitanTPP | 2.5214 +/- 0.0413 | 3.5055 +/- 0.0610 |
| Instacart | (8, 20] | 40.21% | Count-aware RMTPP | 3.7942 +/- 0.0417 | 4.7692 +/- 0.0657 |
| Instacart | (8, 20] | 40.21% | Count-aware THP | 3.8852 +/- 0.0927 | 4.8712 +/- 0.1202 |
| Instacart | (8, 20] | 40.21% | Count-aware TitanTPP | 3.8043 +/- 0.0376 | 4.7698 +/- 0.0408 |
| Instacart | (20, 25] | 5.42% | Count-aware RMTPP | 8.6386 +/- 0.3461 | 9.6095 +/- 0.2311 |
| Instacart | (20, 25] | 5.42% | Count-aware THP | 8.5522 +/- 0.3954 | 9.5897 +/- 0.2429 |
| Instacart | (20, 25] | 5.42% | Count-aware TitanTPP | 8.7534 +/- 0.1899 | 9.6999 +/- 0.1325 |
| Instacart | (25, 35] | 4.01% | Count-aware RMTPP | 12.6607 +/- 0.7488 | 13.8361 +/- 0.5130 |
| Instacart | (25, 35] | 4.01% | Count-aware THP | 12.4505 +/- 0.8176 | 13.7133 +/- 0.5525 |
| Instacart | (25, 35] | 4.01% | Count-aware TitanTPP | 12.9453 +/- 0.3437 | 14.0373 +/- 0.2425 |
| Instacart | > 35 | 1.20% | Count-aware RMTPP | 23.1776 +/- 1.3975 | 25.6084 +/- 1.0141 |
| Instacart | > 35 | 1.20% | Count-aware THP | 22.8422 +/- 1.4027 | 25.3961 +/- 1.0328 |
| Instacart | > 35 | 1.20% | Count-aware TitanTPP | 23.5508 +/- 0.5491 | 25.8340 +/- 0.3834 |

Target membership and count are identical across the three models within each dataset and stratum. Results remain validation-only.
