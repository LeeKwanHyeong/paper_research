# TitanTPP v0.7 validation tables

> Validation only; mean ± sample standard deviation over seeds 42, 52, and 62. Lower is better. No held-out test result is included.

## Taxi: common T0 aggregate

| Model | Clamped time loss | Log-quantity MSE | Quantity MAE | Quantity RMSE |
| --- | ---: | ---: | ---: | ---: |
| Count-aware RMTPP | 1.359763 ± 0.001960 | 0.184898 ± 0.001350 | 40.289319 ± 3.232066 | 144.814842 ± 14.005352 |
| Count-aware THP | 1.362396 ± 0.002819 | 0.194915 ± 0.006842 | 41.588394 ± 2.936870 | 147.496347 ± 10.208561 |
| TitanTPP Hard-LMM | 1.365797 ± 0.001465 | 0.190733 ± 0.001819 | 42.580526 ± 8.097106 | 143.993141 ± 32.531155 |

## Taxi: paired Hard-LMM comparison

| Reference | Clamped-time Δ / wins | Log-MSE Δ / wins | MAE Δ / wins | RMSE Δ / wins |
| --- | ---: | ---: | ---: | ---: |
| Count-aware RMTPP | 0.006034 / 0/3 | 0.005835 / 0/3 | 2.291208 / 1/3 | -0.821700 / 2/3 |
| Count-aware THP | 0.003401 / 1/3 | -0.004182 / 2/3 | 0.992132 / 2/3 | -3.503206 / 2/3 |

Δ is Hard-LMM minus reference; negative is better for Hard-LMM. Wins count paired seeds with a lower Hard-LMM value.

## Taxi: quantity-stratified MAE / RMSE

| Train-derived stratum | RMTPP | THP | Hard-LMM |
| --- | ---: | ---: | ---: |
| <= 7 | 1.434 / 3.296 | 1.453 / 3.375 | 1.402 / 3.260 |
| (7, 686] | 25.945 / 50.944 | 27.910 / 56.057 | 29.647 / 62.730 |
| (686, 1562] | 188.100 / 242.594 | 207.803 / 267.825 | 208.434 / 271.841 |
| (1562, 3449] | 338.536 / 431.606 | 332.287 / 427.308 | 383.739 / 489.623 |
| > 3449 | 897.885 / 1041.814 | 882.276 / 1049.450 | 720.797 / 821.889 |

## Instacart: common T0 aggregate

| Model | Clamped time loss | Log-quantity MSE | Quantity MAE | Quantity RMSE |
| --- | ---: | ---: | ---: | ---: |
| Count-aware RMTPP | 3.205289 ± 0.000846 | 0.243269 ± 0.000244 | 4.026994 ± 0.030269 | 5.990086 ± 0.066938 |
| Count-aware THP | 3.206215 ± 0.000547 | 0.245422 ± 0.001697 | 4.036790 ± 0.027036 | 6.001171 ± 0.062483 |
| TitanTPP Hard-LMM | 3.206516 ± 0.000151 | 0.244365 ± 0.000680 | 4.045027 ± 0.016616 | 6.015904 ± 0.037398 |

## Instacart: paired Hard-LMM comparison

| Reference | Clamped-time Δ / wins | Log-MSE Δ / wins | MAE Δ / wins | RMSE Δ / wins |
| --- | ---: | ---: | ---: | ---: |
| Count-aware RMTPP | 0.001227 / 0/3 | 0.001096 / 0/3 | 0.018033 / 0/3 | 0.025818 / 1/3 |
| Count-aware THP | 0.000302 / 1/3 | -0.001057 / 3/3 | 0.008236 / 1/3 | 0.014733 / 1/3 |

Δ is Hard-LMM minus reference; negative is better for Hard-LMM. Wins count paired seeds with a lower Hard-LMM value.

## Instacart: quantity-stratified MAE / RMSE

| Train-derived stratum | RMTPP | THP | Hard-LMM |
| --- | ---: | ---: | ---: |
| <= 8 | 2.538 / 3.545 | 2.518 / 3.550 | 2.521 / 3.506 |
| (8, 20] | 3.794 / 4.769 | 3.885 / 4.871 | 3.804 / 4.770 |
| (20, 25] | 8.639 / 9.610 | 8.552 / 9.590 | 8.753 / 9.700 |
| (25, 35] | 12.661 / 13.836 | 12.451 / 13.713 | 12.945 / 14.037 |
| > 35 | 23.178 / 25.608 | 22.842 / 25.396 | 23.551 / 25.834 |

## Taxi: history-stratified MAE / RMSE

| History stratum | RMTPP | THP | Hard-LMM |
| --- | ---: | ---: | ---: |
| History <= 64 | 0.504 / 0.829 | 0.499 / 0.838 | 0.476 / 0.857 |
| History 65-128 | 1.291 / 3.404 | 1.264 / 3.029 | 1.264 / 2.937 |
| History > 128 | 59.543 / 176.712 | 61.486 / 179.986 | 62.968 / 175.711 |

Instacart has one history stratum (history ≤64) for all validation targets, so it cannot test a long-history mechanism.
