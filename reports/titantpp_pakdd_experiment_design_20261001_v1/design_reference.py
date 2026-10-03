"""CPU-only mathematical references, not registered training models or launchers."""
import torch
from torch import nn
from torch.nn import functional as F

THRESHOLDS = (1, 2, 4, 8, 16, 32, 64, 128)


def eligibility(valid, write=None, *, all_available=False):
    if valid.dtype != torch.bool or valid.ndim != 2:
        raise ValueError("Expected a boolean batch by length mask")
    if write is not None and (write.dtype != torch.bool or write.shape != valid.shape):
        raise ValueError("Invalid write mask")
    observed = valid if write is None else valid & write
    count = observed.long().cumsum(1)
    reset = torch.where(valid & ~observed, count, 0).cummax(1).values
    threshold = torch.tensor((1,) * 8 if all_available else THRESHOLDS, device=valid.device)
    available = observed[..., None] & ((count - reset)[..., None] > threshold)
    positions = torch.arange(valid.shape[1], device=valid.device).expand_as(valid)
    latest = torch.where(observed, positions, -1).cummax(1).values
    predecessor = torch.cat((torch.full_like(latest[:, :1], -1), latest[:, :-1]), 1)
    return torch.where(available, predecessor[..., None], 0), available


class CorrectionReference(nn.Module):
    def __init__(self, mode="baseline"):
        super().__init__()
        if mode not in ("baseline", "current_only", "all_available"):
            raise ValueError(mode)
        self.mode = mode
        width, inputs = (6, 64) if mode == "current_only" else (4, 128)
        with torch.random.fork_rng(devices=[]):
            self.input_projections = nn.ModuleList(nn.Linear(inputs, width, bias=False) for _ in range(8))
            self.output_projections = nn.ModuleList(nn.Linear(width, 64, bias=False) for _ in range(8))
            for layer in self.output_projections:
                nn.init.zeros_(layer.weight)

    def forward(self, hidden, valid, write=None):
        if hidden.device.type != "cpu":
            raise ValueError("Design reference is CPU-only")
        observed = valid if write is None else valid & write
        safe = torch.where(observed[..., None], hidden, 0.)
        if not torch.isfinite(safe).all():
            raise ValueError("Nonfinite observed state")
        source, available = eligibility(valid, write, all_available=self.mode == "all_available")
        output = torch.zeros_like(safe)
        for branch, (u, v) in enumerate(zip(self.input_projections, self.output_projections)):
            inputs = safe
            if self.mode != "current_only":
                previous = safe.gather(1, source[..., branch, None].expand(-1, -1, 64))
                inputs = torch.cat((safe, previous), -1)
            features = torch.where(available[..., branch, None], u(inputs), 0.)
            output = output + v(F.gelu(features, approximate="none"))
        return output / 8


def nb_log_mass(k, mu, alpha):
    """GluonTS parameterization, for synthetic formula checks only."""
    if ((k < 0) | (k != k.round())).any() or (mu <= 0).any() or (alpha <= 0).any():
        raise ValueError("Negative binomial support or parameter violation")
    r = 1 / alpha
    return (torch.lgamma(k + r) - torch.lgamma(k + 1) - torch.lgamma(r)
            + k * (torch.log(alpha * mu) - torch.log1p(alpha * mu))
            - r * torch.log1p(alpha * mu))
