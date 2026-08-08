# T2. Matched model and training configuration

> Status: frozen for validation. Parameter counts are derived from the declared model classes and dataset-specific mark cardinality on CPU.

## T2a. Model contract

| Dataset | Model | Encoder | Quantity input | Objective | Value head | Qty-to-mark grad. | Lookback / max len | Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | ---: | ---: |
| Intermittent | RMTPP-original | GRU, h=64 | none | residual_only | shared | coupled | 52 / 16 | 20,303 |
| Intermittent | RMTPP-matched | GRU, h=64 | residual | hybrid | shared | coupled | 52 / 16 | 21,855 |
| Intermittent | THP-matched | Transformer, d=64, L=2, H=4 | residual | hybrid | shared | coupled | 52 / 16 | 101,775 |
| Intermittent | TitanTPP V2 | Titan, d=64, L=2, H=4, static LMM | residual | hybrid | shared | coupled | 52 / 16 | 78,111 |
| Taxi | RMTPP-original | GRU, h=128 | none | residual_only | shared | coupled | 168 / 256 | 63,656 |
| Taxi | RMTPP-matched | GRU, h=128 | residual | hybrid | shared | coupled | 168 / 256 | 66,744 |
| Taxi | THP-matched | Transformer, d=128, L=3, H=4 | residual | hybrid | shared | coupled | 168 / 256 | 596,616 |
| Taxi | TitanTPP V3b | Titan, d=128, L=2, H=4, static LMM | residual | hybrid | mark_conditioned_experts | detached | 168 / 256 | 329,276 |
| Taxi | TitanTPP V2 control | Titan, d=128, L=2, H=4, static LMM | residual | hybrid | shared | coupled | 168 / 256 | 328,760 |
| Instacart | RMTPP-original | GRU, h=64 | none | residual_only | shared | coupled | 52 / 64 | 20,012 |
| Instacart | RMTPP-matched | GRU, h=64 | residual | hybrid | shared | coupled | 52 / 64 | 21,564 |
| Instacart | THP-matched | Transformer, d=64, L=2, H=4 | residual | hybrid | shared | coupled | 52 / 64 | 101,388 |
| Instacart | TitanTPP V2 | Titan, d=64, L=2, H=4, static LMM | residual | hybrid | shared | coupled | 52 / 64 | 80,892 |

RMTPP-matched and THP-matched use the same residual quantity input, hybrid quantity objective and output heads as the dataset's TitanTPP primary model. Taxi V2 control is an ablation row and is not a fifth primary baseline.
All RMTPP baselines use one GRU layer. The configured RNN dropout of 0.1 is therefore inactive under PyTorch's inter-layer dropout semantics; THP and TitanTPP retain dropout 0.1 in their multi-layer encoders.

## T2b. Shared training and evaluation protocol

| Item | Frozen rule |
| :--- | :--- |
| Data split | Nominal chronological 70/15/15 within each sequence; full observed history may be context, while the target split determines training or evaluation membership |
| Seeds | 42, 52, 62 |
| Optimization | AdamW; learning rate 0.001; weight decay 0.01; scheduler none; batch size 128; gradient clip 1.0 |
| Initial budget | e300 for every declared run |
| Continuation trigger | Dataset-level e800 continuation when any primary model has a best epoch in 241-300 for at least 2/3 seeds, or the 251-300 window improves best validation NLL by at least 0.5% in at least 2/3 seeds |
| Checkpoint | Minimum validation total NLL; final and composite-score checkpoints are diagnostic only |
| Development scope | Validation only; reproducibility mode strict |
| Held-out test | Locked until model identity, epoch continuation and checkpoint rules are frozen; evaluated once |
| Benchmark source revision | 726aa64ab0b5478646d11be36fc19dcb224d417e |

The earlier all-e800 meeting contract is retained as provenance, but its epoch policy is superseded by this approved e300-first continuation rule. Model identities, split, seeds, loss settings and test lock remain unchanged.
