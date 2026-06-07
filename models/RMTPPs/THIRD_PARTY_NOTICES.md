# Third-Party Notices

## Transformer Hawkes Process

`TransformerHawkesTPP.py` adapts the encoder design from the official
Transformer Hawkes Process implementation:

- Repository: https://github.com/SimiaoZuo/Transformer-Hawkes-Process
- Paper: Transformer Hawkes Process, ICML 2020
- License: MIT License

The adapted model keeps the official THP ideas of temporal sinusoidal encoding
and causal self-attention, while replacing the original repository's output
heads with this project's shared mark/time/value decoder for controlled
RMTPP/TitanTPP/THP comparison.
