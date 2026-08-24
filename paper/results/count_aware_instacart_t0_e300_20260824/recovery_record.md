# Instacart T0 e300 recovery record

- Recovery date: 2026-08-24
- Execution server: 5080
- Original artifact: `search_artifacts/count_aware_instacart_t0_e300_20260824`
- Recovery artifact: `search_artifacts/count_aware_instacart_t0_e300_20260824_recovery1`
- Frozen source revision: `28293c43521615be2ed8fad5b043dc9df8e5e457`
- Held-out test evaluated: false

## Recovery decision

The original execution was interrupted twice by NVIDIA Xid 8 watchdog errors. RMTPP seed 52 completed without interruption and is retained. RMTPP seed 42, which resumed after the first interruption, and the partial RMTPP seed 62 run are excluded from the final artifact and rerun from fresh initialization.

The recovery artifact copies only the completed RMTPP seed 52 run. The runner then executes RMTPP seeds 42 and 62 from epoch 1 before continuing the remaining THP, NHP, SAHP, and TitanTPP runs under the original matched contract.

No model, dataset, split, optimizer, checkpoint-selection, or held-out test contract is changed by this recovery.
