from dataclasses import dataclass
from typing import Literal


@dataclass
class RMTPPConfig:
    num_marks: int  # K
    mark_emb_dim: int = 32
    rnn_hidden_dim: int = 128
    rnn_type: Literal['rnn', 'gru', 'lstm'] = 'rnn'
    dropout: float = 0.1
    scale_base: float = 10.0
    use_value_head: bool = True
    value_head_activation: Literal['sigmoid', 'identity'] = 'sigmoid'
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
