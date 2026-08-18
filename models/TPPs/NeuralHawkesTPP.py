"""Continuous-time LSTM encoder for the adapted count-aware NHP control."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.TPPs.CountAwareTPP import SharedTimeCountModel


class CountAwareNHP(SharedTimeCountModel):
    """Continuous-time LSTM history encoder under shared time/count heads.

    This is the adapted NHP backbone used by the controlled count-aware
    experiment. It intentionally keeps the experiment's shared output heads
    instead of reproducing the original NHP likelihood.
    """

    def __init__(
        self,
        hidden_dim: int,
        train_log_mean: float,
        **quantity_kwargs: Any,
    ) -> None:
        super().__init__(hidden_dim, train_log_mean, **quantity_kwargs)
        self.input_projection = nn.Linear(2, hidden_dim * 7)
        self.recurrent_projection = nn.Linear(hidden_dim, hidden_dim * 7, bias=False)
        self.input_dropout = nn.Dropout(0.1)

    def encode(
        self,
        dts: torch.Tensor,
        history_quantities: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        features = self.input_dropout(
            self.continuous_features(dts, history_quantities, mask)
        )
        batch_size, seq_len, _ = features.shape
        state_shape = (batch_size, self.hidden_dim)
        cell = features.new_zeros(state_shape)
        cell_bar = features.new_zeros(state_shape)
        output_gate = features.new_zeros(state_shape)
        decay_rate = features.new_ones(state_shape)
        encoded_steps: list[torch.Tensor] = []

        for step in range(seq_len):
            active = mask[:, step].unsqueeze(-1)
            elapsed = dts[:, step].to(features.dtype).clamp_min(0.0).unsqueeze(-1)
            decay_factor = torch.exp(-decay_rate * elapsed)
            decayed_cell = cell_bar + (cell - cell_bar) * decay_factor
            decayed_hidden = output_gate * torch.tanh(decayed_cell)
            gates = (
                self.input_projection(features[:, step])
                + self.recurrent_projection(decayed_hidden)
            )
            (
                input_gate,
                forget_gate,
                candidate,
                next_output_gate,
                input_bar_gate,
                forget_bar_gate,
                next_decay_rate,
            ) = gates.chunk(7, dim=-1)
            input_gate = torch.sigmoid(input_gate)
            forget_gate = torch.sigmoid(forget_gate)
            candidate = torch.tanh(candidate)
            next_output_gate = torch.sigmoid(next_output_gate)
            input_bar_gate = torch.sigmoid(input_bar_gate)
            forget_bar_gate = torch.sigmoid(forget_bar_gate)
            next_decay_rate = F.softplus(next_decay_rate) + 1e-4
            next_cell = forget_gate * decayed_cell + input_gate * candidate
            next_cell_bar = forget_bar_gate * cell_bar + input_bar_gate * candidate
            next_hidden = next_output_gate * torch.tanh(next_cell)

            # Masked rows preserve the recurrent state and emit a zero state.
            cell = torch.where(active, next_cell, cell)
            cell_bar = torch.where(active, next_cell_bar, cell_bar)
            output_gate = torch.where(active, next_output_gate, output_gate)
            decay_rate = torch.where(active, next_decay_rate, decay_rate)
            encoded_steps.append(
                torch.where(active, next_hidden, torch.zeros_like(next_hidden))
            )

        return torch.stack(encoded_steps, dim=1)


__all__ = ["CountAwareNHP"]
