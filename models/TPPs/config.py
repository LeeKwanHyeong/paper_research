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
    # shared: one residual prediction reused across all next-mark branches.
    # mark_conditioned_experts: shared residual plus a per-real-mark delta.
    value_head_mode: Literal['shared', 'mark_conditioned_experts'] = 'shared'
    # mark_conditioned: shared RMTPP intercept plus a zero-init real-mark delta.
    time_head_mode: Literal['shared', 'mark_conditioned'] = 'shared'
    # coupled: quantity loss updates mark logits through the probability gate.
    # detached: quantity loss treats mark probabilities as fixed gate weights.
    qty_mark_gradient_mode: Literal['coupled', 'detached'] = 'coupled'
    # coupled: value/quantity losses update the shared sequence encoder.
    # detached: those losses train value heads from a stopped-gradient hidden state.
    value_encoder_gradient_mode: Literal['coupled', 'detached'] = 'coupled'
    # ce_rps keeps categorical CE and adds a normalized ordinal RPS auxiliary.
    marker_loss_mode: Literal['ce', 'ce_rps'] = 'ce'
    lambda_ordinal: float = 0.0
    # Exactly one quantity decoder is active in a run. Direct modes keep the
    # marker/time likelihoods but bypass mark argmax for quantity prediction.
    qty_decoder_mode: Literal[
        'mark_residual', 'direct_log_qty', 'direct_raw_qty'
    ] = 'mark_residual'
    magnitude_norm_mode: Literal[
        'global', 'causal_revin', 'causal_shrinkage_revin'
    ] = 'global'
    magnitude_input_emb_dim: int = 8
    lambda_magnitude: float = 1.0
    # Q3 keeps the direct raw-quantity forward pass fixed while controlling
    # whether magnitude losses can update the shared Titan encoder.
    magnitude_encoder_gradient_mode: Literal['coupled', 'detached'] = 'coupled'
    # The optional log2 Huber term protects low quantities without moving
    # normalization, inputs, or decoding out of the raw-quantity domain.
    magnitude_aux_loss_mode: Literal['none', 'log_huber'] = 'none'
    lambda_log_qty: float = 0.25
    log_qty_huber_delta: float = 1.0
    log_qty_floor: float = 1.0
    magnitude_global_mean: float = 0.0
    magnitude_global_var: float = 1.0
    magnitude_global_std: float = 1.0
    magnitude_sigma_floor: float = 1e-3
    magnitude_revin_eps: float = 1e-5
    magnitude_shrinkage_k: float = 8.0
    magnitude_center_mode: Literal['mean'] = 'mean'
    magnitude_revin_affine: bool = False
    magnitude_stat_context_mode: Literal['none'] = 'none'
    magnitude_exp_clamp_min: float = -2.0
    magnitude_exp_clamp_max: float = 15.0
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
