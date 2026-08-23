# Final Paper Table Contract Audit

- Status: **PASS**
- RMTPP and THP overlapping summaries matched exactly across both source results.
- All six rows use seeds 42, 52, and 62 on the same Intermittent fixed split.
- Maximum/minimum epochs, patience, batch size, learning rate, lookback, max sequence length, and checkpoint selection are matched.
- All models use the `legacy_clamped_rmtpp` time head.
- T0 uses direct log-MSE; TitanTPP-T1 intentionally adds the train-only tail-aware auxiliary loss.
- Evaluation is validation-only and the held-out test remains locked.
- H0/H3 time-head diagnostics and previous mark-residual V2/V3 variants are excluded.
