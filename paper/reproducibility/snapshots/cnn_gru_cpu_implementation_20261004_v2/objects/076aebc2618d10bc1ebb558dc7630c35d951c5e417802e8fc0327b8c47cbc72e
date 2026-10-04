"""S2P2 and AttNHP history encoders with the existing common output heads.

These are matched-head adaptations, not reproductions of native marked NLL.
S2P2: UCIDataLab/state_space_point_process c3933240 (Apache-2.0).
AttNHP: yangalan123/anhp-andtt 78f754fd (MIT).
Pinned original kernels/licenses are under vendor/. Inputs contain observed
log-gap/log-quantity only. States describe t_i+; no query uses t_{i+1}.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from models.TPPs.CountAwareTPP import SharedTimeCountModel
from models.TPPs.vendor.s2p2.models import Int_Backward_LLH
from models.TPPs.vendor.attnhp.xfmr import EncoderLayer, MultiHeadAttention

ARMS = ("s2p2_matched_head", "attnhp_matched_head")
ROLE = "additional_tpp_matched_head_v1"


def affine_scan(a, b, initial):
    """Inclusive x_i=a_i*x_{i-1}+b_i, avoiding complex log(0).

    Associative composition (a2,b2) o (a1,b1)=(a2*a1,a2*b1+b2).
    Equivalent to the authors' sequential recurrence; O(N log N) element work,
    O(log N) stages. Do not claim the original work-efficient scan's complexity.
    """
    step = 1
    while step < a.shape[-2]:
        b = torch.cat((b[..., :step, :], b[..., step:, :] + a[..., step:, :] * b[..., :-step, :]), -2)
        a = torch.cat((a[..., :step, :], a[..., step:, :] * a[..., :-step, :]), -2)
        step *= 2
    return b + a * initial.unsqueeze(-2)


class StableBackwardLLH(Int_Backward_LLH):
    """Original backward ZOH LLH; only parallel recurrence arithmetic changes."""
    def _ssm(self, left_u_NH, right_u_NH, impulse_NP, dt_N, initial_state_P):
        rates = self.get_lambda(right_u_NH=right_u_NH, shift_u=True)
        lam = rates.get("lambda_rescaled_NP", rates.get("lambda_rescaled_P"))
        decay = torch.exp(dt_N.unsqueeze(-1) * lam)
        if left_u_NH is None:
            integrated, left_skip, right_skip = 0.0, 0.0, 0.0
        else:
            integrated = torch.einsum("...np,ph,...nh->...np", decay - 1.0,
                                     self.B_tilde_PH, left_u_NH.to(torch.complex64))
            left_skip = left_u_NH * self.D_HH
            right_skip = right_u_NH * self.D_HH
        right = affine_scan(decay, integrated + impulse_NP, initial_state_P)
        previous = torch.cat((initial_state_P.unsqueeze(-2), right[..., :-1, :]), -2)
        left = decay * previous + integrated
        left_y = 2 * torch.einsum("hp,...np->...nh", self.C_tilde_HP, left).real + left_skip
        right_y = 2 * torch.einsum("hp,...np->...nh", self.C_tilde_HP, right).real + right_skip
        return right, left_y, right_y


class CountAwareS2P2(SharedTimeCountModel):
    def __init__(self, hidden_dim, train_log_mean, **kwargs):
        super().__init__(hidden_dim, train_log_mean, **kwargs)
        self.input_projection = nn.Linear(2, hidden_dim)
        self.layers = nn.ModuleList([
            StableBackwardLLH(P=16, H=hidden_dim, dropout_rate=.1,
                              pre_norm=False, post_norm=True, relative_time=True,
                              complex_values=True, simple_mark=True, is_first_layer=i == 0)
            for i in range(2)
        ])

    def encode(self, dts, history_quantities, mask):
        # Production caller uses right-padding. Canonicalizing here also makes
        # left-padding and holes inert, then restores the original positions.
        n = mask.shape[1]
        positions = torch.arange(n, device=mask.device).expand_as(mask)
        order = torch.argsort((~mask).long() * n + positions, dim=1)
        inverse = torch.argsort(order, dim=1)
        valid = mask.gather(1, order)
        gaps = torch.where(valid, dts.gather(1, order).float().clamp_min(0), 0.)
        qty = torch.where(valid, history_quantities.gather(1, order), 0.)
        alpha = self.input_projection(self.continuous_features(gaps, qty, valid))
        alpha = alpha * valid.unsqueeze(-1)
        left = right = None
        for layer in self.layers:
            _, left, right = layer(left, right, alpha, gaps)
            left = left * valid.unsqueeze(-1)
            right = right * valid.unsqueeze(-1)
        return right.gather(1, inverse.unsqueeze(-1).expand(-1, -1, self.hidden_dim))


class CountAwareAttNHP(SharedTimeCountModel):
    """Author query/context two-stream attention, queried at observed t_i+.

    One attention head, width64, two layers, time dimension16; no FFN or
    attention residual, original tanh residual on the query stream.
    """
    def __init__(self, hidden_dim, train_log_mean, **kwargs):
        super().__init__(hidden_dim, train_log_mean, **kwargs)
        self.input_projection = nn.Linear(2, hidden_dim)
        self.d_time = 16
        self.register_buffer("div_term", torch.exp(torch.arange(0, self.d_time, 2)
                             * -(math.log(10000.) / self.d_time)).reshape(1, 1, -1))
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_dim + self.d_time,
                         MultiHeadAttention(1, hidden_dim + self.d_time, hidden_dim,
                                            dropout=.1, output_linear=False),
                         use_residual=False, dropout=.1) for _ in range(2)
        ])

    def temporal(self, times):
        phases = times.unsqueeze(-1) * self.div_term
        return torch.stack((phases.sin(), phases.cos()), -1).flatten(-2)

    def encode(self, dts, history_quantities, mask):
        gaps = torch.where(mask, dts.float().clamp_min(0), 0.)
        qty = torch.where(mask, history_quantities, 0.)
        nonpad = mask.unsqueeze(-1)
        temporal = self.temporal(gaps.cumsum(1)) * nonpad
        x = torch.tanh(self.input_projection(self.continuous_features(gaps, qty, mask))) * nonpad
        query = torch.zeros_like(x)
        b, n = mask.shape
        future = torch.ones(n, n, device=mask.device, dtype=torch.bool).triu(1)
        history_mask = future[None].expand(b, -1, -1) | ~mask[:, None, :]
        # Padded context rows have a private dummy self-key and emit zero;
        # no valid query can ever attend to that padded key.
        context_mask = history_mask.clone()
        idx = torch.arange(n, device=mask.device)
        context_mask[:, idx, idx] = False
        only_self = ~torch.eye(n, device=mask.device, dtype=torch.bool)[None].expand(b, -1, -1)
        combined_mask = torch.cat((torch.cat((context_mask, torch.ones_like(only_self)), -1),
                                   torch.cat((history_mask, only_self), -1)), 1)
        enc_input = torch.cat((x, temporal), -1)
        for layer in self.layers:
            z = layer(torch.cat((enc_input, torch.cat((query, temporal), -1)), 1), combined_mask)
            query = (query + torch.tanh(z[:, n:] * nonpad)) * nonpad
            enc_input = torch.cat((z[:, :n] * nonpad, temporal), -1)
        return query


def metadata(arm, hidden_dim, max_seq_len):
    shared = {"backbone_contract_id": ROLE, "adaptation": "observed_event_right_limit_common_heads",
              "d_model": hidden_dim, "n_layers": 2, "dropout": .1,
              "max_len": max_seq_len, "future_target_time_used": False,
              "native_mark_intensity_head": False, "input": "log1p_gap_log1p_quantity"}
    if arm == ARMS[0]:
        return {**shared, "encoder": "S2P2", "P": 16, "complex_values": True,
                "discretization": "backward_ZOH", "relative_time": True,
                "pre_norm": False, "post_norm": True, "scan": "affine_doubling",
                "upstream_commit": "c3933240f16a22b43d80d09bda272475526ff24b"}
    if arm == ARMS[1]:
        return {**shared, "encoder": "AttNHP", "n_heads": 1, "d_time": 16,
                "use_norm": False, "query": "observed_t_i_including_event_i",
                "upstream_commit": "78f754fde61547b84a0749fd7e7e020e7cd6b4d8"}
    raise ValueError("Unknown additional TPP")


def validate_checkpoint(payload, expected):
    m = payload.get("encoder_config", {})
    candidate = payload.get("backbone") in ARMS or m.get("backbone_contract_id") == ROLE
    if expected not in ARMS:
        if candidate:
            raise ValueError("Additional TPP checkpoint cannot load as another encoder")
        return False
    if payload.get("backbone") != expected or m.get("d_model") != 64:
        raise ValueError("Additional TPP checkpoint identity mismatch")
    wanted = metadata(expected, 64, m.get("max_len"))
    if any(m.get(k) != v for k, v in wanted.items()):
        raise ValueError("Additional TPP encoder metadata mismatch")
    if (payload.get("variant") != "count_only_log_regression"
            or m.get("time_head", {}).get("mode") != "heteroscedastic_lognormal_duration"
            or payload.get("evaluation_scope") != "validation_only"
            or payload.get("held_out_test_evaluated") is not False):
        raise ValueError("Additional TPP checkpoint objective/scope mismatch")
    return True
