# External T0 Contract Audit

- Status: PASS
- RMTPP/THP source: existing matched T0 artifact
- NHP/SAHP source: fresh official `t0_common_control` artifact
- Dataset, split, epoch, seed, batch, learning rate, lookback, max sequence length: matched
- Quantity loss: direct log-MSE for all four models
- Time head: `legacy_clamped_rmtpp` for all four models
- Checkpoint: minimum validation joint objective
- Held-out test: not evaluated
- Base source revision: `044add1f3de768d804d9f0269fd0013bd9658a35`
- Extension source revisions: `b6831d30a60b83677cc438b3a560217bf343c75c`, `b6831d30a60b83677cc438b3a560217bf343c75c`, `b6831d30a60b83677cc438b3a560217bf343c75c`
