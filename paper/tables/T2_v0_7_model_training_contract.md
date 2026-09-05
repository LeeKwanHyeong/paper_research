# T2 v0.7. Matched model and training contract

> Frozen for validation reporting. Models differ only in the history encoder within each dataset.

| Dataset | Model | Encoder | Lookback / max length | Shared input and quantity loss |
| --- | --- | --- | ---: | --- |
| Intermittent-5000 | Count-aware RMTPP | one-layer GRU | 520 week / 256 | `log1p(dt)`, `log1p(q)` / direct log-MSE |
| Intermittent-5000 | Count-aware THP | two-layer causal Transformer | 520 week / 256 | `log1p(dt)`, `log1p(q)` / direct log-MSE |
| Intermittent-5000 | Count-aware TitanTPP | two-layer causal memory attention + static Hard-LMM | 520 week / 256 | `log1p(dt)`, `log1p(q)` / direct log-MSE |
| Taxi | Count-aware RMTPP | one-layer GRU | 168 hour / 256 | `log1p(dt)`, `log1p(q)` / direct log-MSE |
| Taxi | Count-aware THP | two-layer causal Transformer | 168 hour / 256 | `log1p(dt)`, `log1p(q)` / direct log-MSE |
| Taxi | Count-aware TitanTPP | two-layer causal memory attention + static Hard-LMM | 168 hour / 256 | `log1p(dt)`, `log1p(q)` / direct log-MSE |
| Instacart | Count-aware RMTPP | one-layer GRU | 52 day / 64 | `log1p(dt)`, `log1p(q)` / direct log-MSE |
| Instacart | Count-aware THP | two-layer causal Transformer | 52 day / 64 | `log1p(dt)`, `log1p(q)` / direct log-MSE |
| Instacart | Count-aware TitanTPP | two-layer causal memory attention + static Hard-LMM | 52 day / 64 | `log1p(dt)`, `log1p(q)` / direct log-MSE |

All rows use hidden dimension 64, seeds 42/52/62, AdamW at 0.001, batch size 128, gradient clipping 1.0, an e300 ceiling, minimum 40 epochs, patience 40, and minimum validation joint objective checkpoint selection.
The executed legacy time score caps the intercept at 300 and `w * delta_t` at 10. The source-artifact field `time_nll` is reported as clamped time loss, not exact NLL.
