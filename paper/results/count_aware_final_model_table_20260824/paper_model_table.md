# Count-aware TPP Three-seed Validation Results

- Dataset: Intermittent fixed split
- Seeds: 42, 52, 62
- Selection: minimum validation joint objective
- Held-out test: not evaluated
- T0 rows share direct log-MSE; TitanTPP-T1 alone adds the train-only tail-aware auxiliary loss.

| Role | Model | Encoder | Quantity objective | Time NLL | Quantity MAE | Quantity RMSE |
| --- | --- | --- | --- | ---: | ---: | ---: |
| T0 common control | Adapted RMTPP | GRU | Direct log-MSE | -3.599494 +/- 0.000000 | 2.902523 +/- 0.170252 | 10.578742 +/- 0.604664 |
| T0 common control | Adapted THP | Causal Transformer | Direct log-MSE | -3.599485 +/- 0.000004 | **0.666380 +/- 0.081872** | 2.150737 +/- 0.555155 |
| T0 common control | Adapted NHP | Continuous-time LSTM | Direct log-MSE | -3.599492 +/- 0.000001 | 5.282690 +/- 0.164652 | 15.424367 +/- 0.738889 |
| T0 common control | Adapted SAHP | Self-attention + decay | Direct log-MSE | -3.599494 +/- 0.000001 | 1.081459 +/- 0.008461 | 3.772380 +/- 0.175035 |
| T0 common control | TitanTPP-T0 | Titan Hard-LMM | Direct log-MSE | -3.593078 +/- 0.000661 | 0.746917 +/- 0.068508 | 1.919518 +/- 0.293766 |
| Proposed method | TitanTPP-T1 | Titan Hard-LMM | Log-MSE + tail-aware auxiliary | -3.593170 +/- 0.000919 | 0.698884 +/- 0.059655 | **1.799715 +/- 0.182126** |

**Reading rule.** Lower is better for Time NLL, Quantity MAE, and Quantity RMSE. The T0 rows isolate backbone differences. TitanTPP-T1 is the final proposed configuration, so its result reflects both the Titan backbone and the tail-aware training objective.
