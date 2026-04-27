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
    eps: float = 1e-8
    w_min: float = 1e-3     # w_t stabilization because of 0 (1/w)
    exp_clamp: float = 300.0 # exponential overflow guard
