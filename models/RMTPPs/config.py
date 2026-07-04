from dataclasses import dataclass
from typing import Literal


@dataclass
class RMTPPConfig:
    num_marks: int  # K
    mark_emb_dim: int = 32
    rnn_hidden_dim: int = 256
    rnn_type: Literal['rnn', 'gru', 'lstm'] = 'rnn'
    dropout: float = 0.1
    scale_base: float = 10.0
    use_value_head: bool = True
    value_head_activation: Literal['sigmoid', 'identity'] = 'sigmoid'
    # Optional value-conditioned marked TPP input.
    # - none: original baseline, input is mark + dt only
    # - residual: append past observed scale_residual_t
    # - log_qty: append past observed log_base(qty_t) = mark_t + scale_residual_t
    value_input_mode: Literal['none', 'residual', 'log_qty'] = 'none'
    value_input_emb_dim: int = 8
    # all: train on every valid transition inside a window
    # target_only: train only on the final next-event transition, matching fixed validation/test
    train_loss_scope: Literal['all', 'target_only'] = 'all'
    # Loss configuration:
    # - residual_only: current baseline objective
    # - hybrid: residual supervision + direct quantity supervision
    # - qty_only: direct quantity supervision without residual loss
    #
    # The main A/B benchmark keeps this fixed at residual_only so the reported
    # RMTPP vs TitanTPP comparison stays aligned with the legacy setup.
    loss_mode: Literal['residual_only', 'hybrid', 'qty_only'] = 'residual_only'
    lambda_qty: float = 0.25
    qty_scale_value: float = 1.0
    eps: float = 1e-8
    w_min: float = 1e-3     # w_t stabilization because of 0 (1/w)
    exp_clamp: float = 300.0 # exponential overflow guard


@dataclass
class THPConfig:
    """
    Transformer Hawkes Process encoder configuration.

    This follows the official THP implementation's main encoder choices:
    temporal sinusoidal encoding, causal self-attention, and position-wise FFN.
    The surrounding TPP decoder is kept compatible with this project so THP can
    be compared against RMTPP/TitanTPP under the same mark/time/value heads.
    """
    d_model: int = 128
    d_inner: int = 512
    n_layers: int = 3
    n_heads: int = 4
    dropout: float = 0.1
    normalize_before: bool = False
    add_temporal_encoding_each_layer: bool = True
    use_rnn: bool = False
    d_rnn: int = 128
