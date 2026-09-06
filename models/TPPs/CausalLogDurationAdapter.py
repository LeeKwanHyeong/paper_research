"""Causal scale adapter for a frozen log-normal duration head.

The adapter sees only the observed inter-event durations in the active
context.  ``lengths`` defines the active prefix of each right-padded row, so
the next-event target and every padded value stay outside the recurrent
computation by construction.

The base log-normal location is returned unchanged.  A one-layer GRU produces
one bounded multiplicative residual for the distance between the base scale
and its positive floor.  The final projection is initialized to zero, making
the complete density and survival functions identical to the base head at
initialization while still giving that projection a first-step gradient.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class CausalLogDurationAdapter(nn.Module):
    """Adjust only log-normal scale from a causal duration-history prefix."""

    INPUT_SIZE = 1
    HIDDEN_SIZE = 8
    NUM_LAYERS = 1
    DROPOUT = 0.0
    MAX_SCALE_RATIO = 10.0

    def __init__(
        self,
        *,
        log_duration_mean: float,
        log_duration_std: float,
        sigma_floor: float,
    ) -> None:
        super().__init__()
        if not math.isfinite(log_duration_mean):
            raise ValueError("log_duration_mean must be finite")
        if not math.isfinite(log_duration_std) or log_duration_std <= 0.0:
            raise ValueError("log_duration_std must be finite and positive")
        if not math.isfinite(sigma_floor) or sigma_floor <= 0.0:
            raise ValueError("sigma_floor must be finite and positive")

        # These values are supplied by the runner after being calculated on
        # train history only.  Buffers make their checkpoint and resume state
        # explicit without turning them into learned dataset parameters.
        self.register_buffer(
            "log_duration_mean",
            torch.tensor(float(log_duration_mean), dtype=torch.float64),
        )
        self.register_buffer(
            "log_duration_std",
            torch.tensor(float(log_duration_std), dtype=torch.float64),
        )
        self.register_buffer(
            "sigma_floor",
            torch.tensor(float(sigma_floor), dtype=torch.float64),
        )

        self.gru = nn.GRU(
            input_size=self.INPUT_SIZE,
            hidden_size=self.HIDDEN_SIZE,
            num_layers=self.NUM_LAYERS,
            batch_first=True,
            dropout=self.DROPOUT,
        )
        self.scale_projection = nn.Linear(self.HIDDEN_SIZE, 1)
        nn.init.zeros_(self.scale_projection.weight)
        nn.init.zeros_(self.scale_projection.bias)

    def _validate_history(
        self,
        observed_delta_t: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if observed_delta_t.ndim != 2:
            raise ValueError("observed_delta_t must have shape [batch, time]")
        if lengths.ndim != 1:
            raise ValueError("lengths must have shape [batch]")
        if observed_delta_t.shape[0] != lengths.shape[0]:
            raise ValueError("history and lengths batch sizes differ")
        if lengths.dtype == torch.bool or lengths.dtype.is_floating_point:
            raise TypeError("lengths must use an integer dtype")
        if observed_delta_t.device != self.log_duration_mean.device:
            raise ValueError("history and adapter must be on the same device")

        lengths_cpu = lengths.detach().to(device="cpu", dtype=torch.long)
        max_length = observed_delta_t.shape[1]
        if bool((lengths_cpu < 0).any()) or bool(
            (lengths_cpu > max_length).any()
        ):
            raise ValueError("lengths must stay within the padded time axis")

        positions = torch.arange(
            max_length,
            device=observed_delta_t.device,
        ).unsqueeze(0)
        active_mask = positions < lengths.to(
            device=observed_delta_t.device,
            dtype=torch.long,
        ).unsqueeze(1)
        active_values = observed_delta_t.masked_select(active_mask)
        if bool((~torch.isfinite(active_values)).any()):
            raise ValueError("active observed durations must be finite")
        if bool((active_values < 0.0).any()):
            raise ValueError("active observed durations must be nonnegative")
        return active_mask, lengths_cpu

    def normalized_history(
        self,
        observed_delta_t: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Return normalized ``log1p(delta_t)`` with all padding set to zero."""
        active_mask, _ = self._validate_history(observed_delta_t, lengths)
        safe_duration = torch.where(
            active_mask,
            observed_delta_t,
            torch.zeros_like(observed_delta_t),
        )
        values_64 = safe_duration.to(dtype=torch.float64)
        normalized_64 = (
            torch.log1p(values_64) - self.log_duration_mean
        ) / self.log_duration_std
        normalized_64 = torch.where(
            active_mask,
            normalized_64,
            torch.zeros_like(normalized_64),
        )
        return normalized_64.to(dtype=self.gru.weight_ih_l0.dtype).unsqueeze(-1)

    def raw_log_scale_residual(
        self,
        observed_delta_t: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Encode each non-empty active prefix into one unconstrained scalar."""
        normalized = self.normalized_history(observed_delta_t, lengths)
        lengths_cpu = lengths.detach().to(device="cpu", dtype=torch.long)
        nonempty_cpu = torch.nonzero(lengths_cpu > 0, as_tuple=False).flatten()

        states = normalized.new_zeros(
            (observed_delta_t.shape[0], self.HIDDEN_SIZE)
        )
        if nonempty_cpu.numel() > 0:
            nonempty = nonempty_cpu.to(device=observed_delta_t.device)
            packed = pack_padded_sequence(
                normalized.index_select(0, nonempty),
                lengths_cpu.index_select(0, nonempty_cpu),
                batch_first=True,
                enforce_sorted=False,
            )
            _, final_state = self.gru(packed)
            states = states.index_copy(0, nonempty, final_state[-1])

        raw = self.scale_projection(states).squeeze(-1)
        # An empty history has no temporal evidence.  It remains an exact base
        # route even after the shared projection bias has been trained.
        nonempty_mask = (lengths > 0).to(device=raw.device, dtype=raw.dtype)
        return raw * nonempty_mask

    def bounded_log_scale_residual(
        self,
        observed_delta_t: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Return a smooth log multiplier bounded by ``log(10)``."""
        raw = self.raw_log_scale_residual(observed_delta_t, lengths)
        limit = math.log(self.MAX_SCALE_RATIO)
        return limit * torch.tanh(raw / limit)

    def forward(
        self,
        observed_delta_t: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Return the scale multiplier, bounded to the range [0.1, 10]."""
        return torch.exp(
            self.bounded_log_scale_residual(observed_delta_t, lengths)
        ).clamp(
            min=1.0 / self.MAX_SCALE_RATIO,
            max=self.MAX_SCALE_RATIO,
        )

    def adjusted_scale(
        self,
        base_scale: torch.Tensor,
        observed_delta_t: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the bounded residual above ``sigma_floor``.

        ``base + (base-floor) * expm1(residual)`` is algebraically the same as
        ``floor + (base-floor) * exp(residual)``.  The first form adds an exact
        zero when the projection is zero, preserves the base scale bit for
        bit, and remains accurate for small learned residuals.
        """
        if base_scale.ndim != 1 or base_scale.shape[0] != observed_delta_t.shape[0]:
            raise ValueError("base_scale must have shape [batch]")
        if bool((~torch.isfinite(base_scale)).any()):
            raise ValueError("base_scale must be finite")
        floor = self.sigma_floor.to(
            device=base_scale.device,
            dtype=base_scale.dtype,
        )
        if bool((base_scale <= floor).any()):
            raise ValueError("base_scale must be strictly above sigma_floor")

        residual = self.bounded_log_scale_residual(
            observed_delta_t,
            lengths,
        ).to(dtype=base_scale.dtype)
        adjusted = base_scale + (base_scale - floor) * torch.expm1(residual)
        if bool((~torch.isfinite(adjusted)).any()):
            raise FloatingPointError("adjusted scale is non-finite")
        return adjusted

    def adjusted_location_scale(
        self,
        base_location: torch.Tensor,
        base_scale: torch.Tensor,
        observed_delta_t: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the exact base location and the history-adjusted scale."""
        if base_location.shape != base_scale.shape:
            raise ValueError("base location and scale shapes differ")
        if bool((~torch.isfinite(base_location)).any()):
            raise ValueError("base_location must be finite")
        return base_location, self.adjusted_scale(
            base_scale,
            observed_delta_t,
            lengths,
        )

    @staticmethod
    def _proper_terms(
        location: torch.Tensor,
        scale: torch.Tensor,
        target_delta_t: torch.Tensor,
        *,
        time_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not math.isfinite(time_scale) or time_scale <= 0.0:
            raise ValueError("time_scale must be finite and positive")
        if target_delta_t.shape != location.shape:
            raise ValueError("target_delta_t and location shapes differ")
        if bool((~torch.isfinite(target_delta_t)).any()) or bool(
            (target_delta_t <= 0.0).any()
        ):
            raise ValueError("target durations must be finite and positive")

        location_64 = location.to(dtype=torch.float64)
        scale_64 = scale.to(dtype=torch.float64)
        log_delta_t = torch.log(target_delta_t.to(dtype=torch.float64))
        standardized = (
            log_delta_t - math.log(time_scale) - location_64
        ) / scale_64
        return standardized, scale_64, log_delta_t

    def proper_log_density(
        self,
        base_location: torch.Tensor,
        base_scale: torch.Tensor,
        target_delta_t: torch.Tensor,
        observed_delta_t: torch.Tensor,
        lengths: torch.Tensor,
        *,
        time_scale: float,
    ) -> torch.Tensor:
        """Evaluate the normalized log-normal density in original time units."""
        location, scale = self.adjusted_location_scale(
            base_location,
            base_scale,
            observed_delta_t,
            lengths,
        )
        standardized, scale_64, log_delta_t = self._proper_terms(
            location,
            scale,
            target_delta_t,
            time_scale=time_scale,
        )
        return (
            -0.5 * torch.square(standardized)
            - torch.log(scale_64)
            - log_delta_t
            - 0.5 * math.log(2.0 * math.pi)
        )

    def proper_log_survival(
        self,
        base_location: torch.Tensor,
        base_scale: torch.Tensor,
        target_delta_t: torch.Tensor,
        observed_delta_t: torch.Tensor,
        lengths: torch.Tensor,
        *,
        time_scale: float,
    ) -> torch.Tensor:
        """Evaluate log survival for censor-aware duration training."""
        location, scale = self.adjusted_location_scale(
            base_location,
            base_scale,
            observed_delta_t,
            lengths,
        )
        standardized, _, _ = self._proper_terms(
            location,
            scale,
            target_delta_t,
            time_scale=time_scale,
        )
        return torch.special.log_ndtr(-standardized)

    @staticmethod
    def predict_median(
        base_location: torch.Tensor,
        *,
        time_scale: float,
    ) -> torch.Tensor:
        """Return the base median; scale adaptation cannot change it."""
        if not math.isfinite(time_scale) or time_scale <= 0.0:
            raise ValueError("time_scale must be finite and positive")
        if bool((~torch.isfinite(base_location)).any()):
            raise ValueError("base_location must be finite")
        return time_scale * torch.exp(base_location)


__all__ = ["CausalLogDurationAdapter"]
