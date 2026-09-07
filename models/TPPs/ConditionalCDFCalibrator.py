"""Identity-anchored conditional CDF calibration for duration models.

The module applies a Kumaraswamy CDF transform to an arbitrary continuous
base distribution.  It predicts two positive shape parameters from the
frozen time hidden state.  Zero initialization gives ``a=b=1`` and therefore
reproduces the complete base density, CDF, and survival function exactly.

All probability calculations use log space.  The transform is anchored to a
numerically evaluated identity transform so the initialized path returns the
provided base tensors bitwise, including in extreme tails.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _log1mexp(value: torch.Tensor) -> torch.Tensor:
    """Return log(1-exp(value)) for non-positive float64 inputs."""
    if value.dtype != torch.float64:
        raise TypeError("log1mexp requires float64 input")
    if bool((value > torch.finfo(value.dtype).eps).any()):
        raise ValueError("log1mexp input must be non-positive")
    value = value.clamp_max(0.0)
    split = -math.log(2.0)
    return torch.where(
        value < split,
        torch.log1p(-torch.exp(value)),
        torch.log(-torch.expm1(value)),
    )


class ConditionalKumaraswamyCDFCalibrator(nn.Module):
    """Predict a common two-shape CDF transform from frozen hidden states."""

    OUTPUT_SIZE = 2
    DEFAULT_MAX_SHAPE_RATIO = 10.0
    ASYMPTOTIC_LOG_THRESHOLD = -30.0

    def __init__(
        self,
        *,
        hidden_dim: int,
        max_shape_ratio: float = DEFAULT_MAX_SHAPE_RATIO,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not math.isfinite(max_shape_ratio) or max_shape_ratio <= 1.0:
            raise ValueError("max_shape_ratio must be finite and exceed one")
        self.hidden_dim = int(hidden_dim)
        self.max_shape_ratio = float(max_shape_ratio)
        self.shape_head = nn.Linear(self.hidden_dim, self.OUTPUT_SIZE)
        nn.init.zeros_(self.shape_head.weight)
        nn.init.zeros_(self.shape_head.bias)

    def _validate_hidden(self, hidden: torch.Tensor) -> None:
        if hidden.ndim != 2 or hidden.shape[1] != self.hidden_dim:
            raise ValueError(
                f"hidden must have shape [batch, {self.hidden_dim}]"
            )
        if bool((~torch.isfinite(hidden)).any()):
            raise ValueError("hidden contains non-finite values")

    def bounded_log_shapes(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return log(a), log(b) with one shared, finite ratio bound."""
        self._validate_hidden(hidden)
        raw = F.linear(
            hidden.to(torch.float64),
            self.shape_head.weight.to(torch.float64),
            self.shape_head.bias.to(torch.float64),
        )
        limit = math.log(self.max_shape_ratio)
        result = limit * torch.tanh(raw / limit)
        if bool((~torch.isfinite(result)).any()):
            raise FloatingPointError("non-finite bounded log shape")
        return result

    def shape_parameters(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_shapes = self.bounded_log_shapes(hidden)
        shapes = torch.exp(log_shapes)
        return shapes[:, 0], shapes[:, 1]

    @staticmethod
    def _validate_base_probabilities(
        base_log_cdf: torch.Tensor,
        base_log_survival: torch.Tensor,
        hidden: torch.Tensor,
    ) -> None:
        expected = (hidden.shape[0],)
        if base_log_cdf.shape != expected or base_log_survival.shape != expected:
            raise ValueError("base log-probability shape mismatch")
        if base_log_cdf.dtype != torch.float64 or base_log_survival.dtype != torch.float64:
            raise TypeError("base log probabilities must be float64")
        if bool((~torch.isfinite(base_log_cdf)).any()) or bool(
            (~torch.isfinite(base_log_survival)).any()
        ):
            raise ValueError("base log probabilities must be finite")
        if bool((base_log_cdf > 1e-14).any()) or bool(
            (base_log_survival > 1e-14).any()
        ):
            raise ValueError("a base log probability exceeds zero")
        complement_error = torch.logaddexp(
            base_log_cdf, base_log_survival
        ).abs()
        if bool((complement_error > 1e-10).any()):
            raise ValueError("base CDF and survival are not complementary")

    @classmethod
    def _raw_log_cdf_survival(
        cls,
        *,
        base_log_cdf: torch.Tensor,
        base_log_survival: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate the unanchored transform and log(1-F_base**a)."""
        log_u_to_a = a * base_log_cdf
        right_tail = base_log_survival < cls.ASYMPTOTIC_LOG_THRESHOLD
        log_one_minus_u_to_a = torch.empty_like(log_u_to_a)
        if bool(right_tail.any()):
            indices = torch.nonzero(right_tail, as_tuple=False).squeeze(-1)
            values = torch.log(a[indices]) + base_log_survival[indices]
            log_one_minus_u_to_a = log_one_minus_u_to_a.index_copy(
                0, indices, values
            )
        exact_right = ~right_tail
        if bool(exact_right.any()):
            indices = torch.nonzero(exact_right, as_tuple=False).squeeze(-1)
            values = _log1mexp(log_u_to_a[indices])
            log_one_minus_u_to_a = log_one_minus_u_to_a.index_copy(
                0, indices, values
            )
        raw_log_survival = b * log_one_minus_u_to_a
        left_tail = log_u_to_a < cls.ASYMPTOTIC_LOG_THRESHOLD
        raw_log_cdf = torch.empty_like(raw_log_survival)
        if bool(left_tail.any()):
            indices = torch.nonzero(left_tail, as_tuple=False).squeeze(-1)
            values = torch.log(b[indices]) + log_u_to_a[indices]
            raw_log_cdf = raw_log_cdf.index_copy(0, indices, values)
        exact_left = ~left_tail
        if bool(exact_left.any()):
            indices = torch.nonzero(exact_left, as_tuple=False).squeeze(-1)
            values = _log1mexp(raw_log_survival[indices])
            raw_log_cdf = raw_log_cdf.index_copy(0, indices, values)
        return raw_log_cdf, raw_log_survival, log_one_minus_u_to_a

    def transformed_log_cdf_survival(
        self,
        *,
        base_log_cdf: torch.Tensor,
        base_log_survival: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return calibrated log CDF and log survival in float64."""
        self._validate_hidden(hidden)
        self._validate_base_probabilities(base_log_cdf, base_log_survival, hidden)
        a, b = self.shape_parameters(hidden)
        transformed_cdf, transformed_survival, _ = self._raw_log_cdf_survival(
            base_log_cdf=base_log_cdf,
            base_log_survival=base_log_survival,
            a=a,
            b=b,
        )
        ones = torch.ones_like(a)
        identity_cdf, identity_survival, _ = self._raw_log_cdf_survival(
            base_log_cdf=base_log_cdf,
            base_log_survival=base_log_survival,
            a=ones,
            b=ones,
        )
        # Adding only the transform delta makes a=b=1 bitwise identical to
        # the caller's base tensors, even when the tail approximation is used.
        anchored_cdf = base_log_cdf + (transformed_cdf - identity_cdf)
        anchored_survival = base_log_survival + (
            transformed_survival - identity_survival
        )
        if bool((~torch.isfinite(anchored_cdf)).any()) or bool(
            (~torch.isfinite(anchored_survival)).any()
        ):
            raise FloatingPointError("non-finite calibrated CDF or survival")
        return anchored_cdf, anchored_survival

    def transformed_log_density(
        self,
        *,
        base_log_density: torch.Tensor,
        base_log_cdf: torch.Tensor,
        base_log_survival: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Return log(g(F_base(t))*f_base(t)) in float64."""
        self._validate_hidden(hidden)
        self._validate_base_probabilities(base_log_cdf, base_log_survival, hidden)
        if base_log_density.shape != base_log_cdf.shape:
            raise ValueError("base log-density shape mismatch")
        if base_log_density.dtype != torch.float64:
            raise TypeError("base log density must be float64")
        if bool((~torch.isfinite(base_log_density)).any()):
            raise ValueError("base log density must be finite")
        log_shapes = self.bounded_log_shapes(hidden)
        log_a, log_b = log_shapes[:, 0], log_shapes[:, 1]
        a, b = torch.exp(log_a), torch.exp(log_b)
        _, _, log_one_minus_u_to_a = self._raw_log_cdf_survival(
            base_log_cdf=base_log_cdf,
            base_log_survival=base_log_survival,
            a=a,
            b=b,
        )
        log_density_ratio = (
            log_a
            + log_b
            + (a - 1.0) * base_log_cdf
            + (b - 1.0) * log_one_minus_u_to_a
        )
        result = base_log_density + log_density_ratio
        if bool((~torch.isfinite(result)).any()):
            raise FloatingPointError("non-finite calibrated density")
        return result

    def base_quantile_level(
        self,
        probability: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Map a calibrated quantile level back to the base CDF level."""
        self._validate_hidden(hidden)
        if probability.shape != (hidden.shape[0],):
            raise ValueError("probability shape mismatch")
        probability = probability.to(torch.float64)
        if bool((~torch.isfinite(probability)).any()) or bool(
            ((probability <= 0.0) | (probability >= 1.0)).any()
        ):
            raise ValueError("probability must lie strictly inside (0,1)")
        a, b = self.shape_parameters(hidden)
        log_one_minus_probability = torch.log1p(-probability)
        log_inner = _log1mexp(log_one_minus_probability / b)
        raw_log_level = log_inner / a
        ones = torch.ones_like(a)
        identity_log_level = _log1mexp(log_one_minus_probability / ones) / ones
        raw_level = torch.exp(raw_log_level)
        identity_level = torch.exp(identity_log_level)
        level = probability + (raw_level - identity_level)
        if bool((~torch.isfinite(level)).any()) or bool(
            ((level <= 0.0) | (level >= 1.0)).any()
        ):
            raise FloatingPointError("invalid calibrated base quantile level")
        return level


class GlobalKumaraswamyCDFCalibrator(ConditionalKumaraswamyCDFCalibrator):
    """History-free two-scalar control with the identical CDF transform."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        max_shape_ratio: float = ConditionalKumaraswamyCDFCalibrator.DEFAULT_MAX_SHAPE_RATIO,
    ) -> None:
        nn.Module.__init__(self)
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not math.isfinite(max_shape_ratio) or max_shape_ratio <= 1.0:
            raise ValueError("max_shape_ratio must be finite and exceed one")
        self.hidden_dim = int(hidden_dim)
        self.max_shape_ratio = float(max_shape_ratio)
        self.raw_log_shapes = nn.Parameter(torch.zeros(self.OUTPUT_SIZE))

    def bounded_log_shapes(self, hidden: torch.Tensor) -> torch.Tensor:
        self._validate_hidden(hidden)
        limit = math.log(self.max_shape_ratio)
        values = limit * torch.tanh(
            self.raw_log_shapes.to(torch.float64) / limit
        )
        result = values.unsqueeze(0).expand(hidden.shape[0], -1)
        if bool((~torch.isfinite(result)).any()):
            raise FloatingPointError("non-finite bounded global log shape")
        return result


__all__ = [
    "ConditionalKumaraswamyCDFCalibrator",
    "GlobalKumaraswamyCDFCalibrator",
]
