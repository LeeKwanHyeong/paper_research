#!/usr/bin/env python3
"""Train an objective-aligned causal duration scale adapter on immutable aligned-B.

The selected aligned-B checkpoint supplies every backbone, memory,
quantity, location, and base-scale tensor.  This runner freezes that complete
model and trains only a one-layer GRU that reads past observed durations.  The
GRU changes log-normal scale, never location, so B's quantity predictions and
time medians remain exact.

Only train and validation targets are admitted. Intermittent keeps its continuous
observation density; Taxi and Instacart use the aligned normalized integer law,
with Instacart code 30 represented by survival from 29.5.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data_loader.event_seq_data_module import (
    RMTPPWeekLookbackDataset,
    collate_week_lookback,
)
from models.TPPs.CausalLogDurationAdapter import (
    CausalLogDurationAdapter,
    LengthOnlyCausalLogDurationAdapter,
)
from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
from paper.scripts.run_hard_lmm_frozen_lognormal_duration import (
    OBSERVATION_LIKELIHOOD_CONTINUOUS,
    OBSERVATION_LIKELIHOOD_POSITIVE_INTEGER,
    FrozenFeatureCache,
    build_candidate_from_selected_checkpoint,
    censor_mask,
    load_admitted_frame,
    resolve_censor_threshold,
    target_dt_sha256,
    validate_observation_likelihood_contract,
)
from paper.scripts.run_hard_lmm_time_head_refit import (
    cached_quantity_predictions,
    exact_target_population_contract,
    finite_nested_tensors,
    quantity_metrics,
    require,
    save_json,
    stratified_quantity_metrics,
    tensor_sha256,
    validate_target_population,
)
from simple_lab_test.search.common.runner import (
    atomic_torch_save,
    canonical_state_dict_sha256,
    torch_load_checkpoint,
)


CONTRACT_ID = "aligned_causal_duration_scale_adapter_v1"
DEFAULT_CONTRACT = (
    PROJECT_ROOT / "paper/contracts/aligned_causal_duration_scale_adapter_v1.json"
)
SELECTED_CHECKPOINT_NAME = "best_validation_primary_nll_model.pt"
LAST_CHECKPOINT_NAME = "last_epoch_state.pt"
SUMMARY_NAME = "summary.json"
INSTACART_DATASET = "insta_market_basket"
CONTINUOUS_CENSOR_THRESHOLD = 30.0
INTERVAL_CENSOR_BOUNDARY = 29.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract_without_duplicate_keys(path: Path) -> dict[str, Any]:
    """Load the frozen experiment contract and reject ambiguous JSON."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        counts = Counter(name for name, _ in pairs)
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        require(not duplicates, f"Duplicate contract keys: {duplicates}")
        return dict(pairs)

    payload = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )
    require(isinstance(payload, dict), "Contract root must be an object")
    return payload


def _clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _tensor_digest(label: str, tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(label.encode("utf-8") + b"\0")
    digest.update(str(value.dtype).encode("ascii") + b"\0")
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def compact_masked_history(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move active left-padded history tokens to a right-padded sequence."""
    require(values.ndim == 2, "History durations must be rank two")
    require(mask.shape == values.shape, "History mask shape mismatch")
    require(mask.dtype == torch.bool, "History mask must be boolean")
    require(bool(torch.isfinite(values).all()), "Non-finite history duration")
    require(
        bool((values[mask] > 0.0).all()),
        "Active history durations must be strictly positive",
    )
    lengths = mask.sum(dim=1).to(dtype=torch.long)
    require(bool((lengths > 0).all()), "Every target needs observed history")
    max_length = int(lengths.max().item())
    compact = values.new_zeros((values.shape[0], max_length))
    positions = mask.long().cumsum(dim=1) - 1
    rows = (
        torch.arange(values.shape[0], device=values.device)
        .unsqueeze(1)
        .expand_as(values)
    )
    compact[rows[mask], positions[mask]] = values[mask]
    return compact, lengths


def split_appended_target(
    dts: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return causal context plus target, explicitly removing the append."""
    require(dts.ndim == 2, "Canonical duration batch must be rank two")
    require(mask.shape == dts.shape, "Canonical mask shape mismatch")
    require(mask.dtype == torch.bool, "Canonical mask must be boolean")
    require(
        bool(mask[:, -1].all()),
        "The canonical appended target must be the final active token",
    )
    require(
        bool((mask.sum(dim=1) >= 2).all()),
        "Each sample must contain history and an appended target",
    )
    target = dts[:, -1].contiguous()
    history_dt = dts[:, :-1].contiguous()
    history_mask = mask[:, :-1].contiguous()
    require(
        bool((history_mask.sum(dim=1) >= 1).all()),
        "The target was not preceded by an observed event",
    )
    return history_dt, history_mask, target


class GlobalScaleControl(nn.Module):
    """Bias-only attribution control; it is not an architecture candidate."""

    MAX_SCALE_RATIO = 10.0

    def __init__(self, *, sigma_floor: float) -> None:
        super().__init__()
        require(
            math.isfinite(sigma_floor) and sigma_floor > 0.0,
            "sigma_floor must be finite and positive",
        )
        self.register_buffer(
            "sigma_floor", torch.tensor(sigma_floor, dtype=torch.float64)
        )
        self.raw_bias = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def bounded_log_scale_residual(
        self,
        observed_delta_t: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        del lengths
        limit = math.log(self.MAX_SCALE_RATIO)
        residual = limit * torch.tanh(self.raw_bias / limit)
        return residual.expand(observed_delta_t.shape[0])

    def forward(
        self,
        observed_delta_t: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
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
        floor = self.sigma_floor.to(
            device=base_scale.device, dtype=base_scale.dtype
        )
        require(
            bool(torch.isfinite(base_scale).all())
            and bool((base_scale > floor).all()),
            "Base sigma must be finite and strictly above its floor",
        )
        residual = self.bounded_log_scale_residual(
            observed_delta_t, lengths
        ).to(dtype=base_scale.dtype)
        result = base_scale + (base_scale - floor) * torch.expm1(residual)
        require(
            bool(torch.isfinite(result).all()) and bool((result > floor).all()),
            "Control scale must be finite and above its floor",
        )
        return result


def adapted_scale(
    module: nn.Module,
    base_sigma: torch.Tensor,
    history_dt: torch.Tensor,
    history_mask: torch.Tensor,
) -> torch.Tensor:
    compact, lengths = compact_masked_history(history_dt, history_mask)
    result = module.adjusted_scale(
        base_sigma.to(torch.float64), compact, lengths
    )
    return result.to(torch.float64)


def continuous_lognormal_log_likelihood(
    *,
    location: torch.Tensor,
    sigma: torch.Tensor,
    target_dt: torch.Tensor,
    time_scale: float,
    is_right_censored: torch.Tensor,
) -> torch.Tensor:
    """Normalized censor-aware log likelihood in the original time unit."""
    require(location.shape == sigma.shape == target_dt.shape, "Term shape drift")
    require(
        is_right_censored.shape == target_dt.shape
        and is_right_censored.dtype == torch.bool,
        "Censor mask mismatch",
    )
    require(time_scale > 0.0 and math.isfinite(time_scale), "Invalid time scale")
    dt = target_dt.to(torch.float64)
    require(bool((dt > 0.0).all()), "Duration targets must be positive")
    mu = location.to(torch.float64)
    sigma = sigma.to(torch.float64)
    log_dt = torch.log(dt)
    standardized = (log_dt - math.log(time_scale) - mu) / sigma
    log_density = (
        -0.5 * torch.square(standardized)
        - torch.log(sigma)
        - log_dt
        - 0.5 * math.log(2.0 * math.pi)
    )
    log_survival = torch.special.log_ndtr(-standardized)
    result = torch.where(is_right_censored, log_survival, log_density)
    require(bool(torch.isfinite(result).all()), "Non-finite continuous likelihood")
    return result


def _logdiffexp(log_large: torch.Tensor, log_small: torch.Tensor) -> torch.Tensor:
    """Stable log(exp(log_large) - exp(log_small)) for ordered log values."""
    require(log_large.shape == log_small.shape, "logdiffexp shape mismatch")
    difference = log_small - log_large
    require(
        bool((difference <= torch.finfo(difference.dtype).eps).all()),
        "logdiffexp received reversed arguments",
    )
    return log_large + torch.log(-torch.expm1(difference.clamp_max(0.0)))


def interval_lognormal_log_likelihood(
    *,
    location: torch.Tensor,
    sigma: torch.Tensor,
    target_dt: torch.Tensor,
    time_scale: float,
    is_right_censored: torch.Tensor,
) -> torch.Tensor:
    """Stable integer-bin probability; Instacart top code uses S(29.5)."""
    require(location.shape == sigma.shape == target_dt.shape, "Term shape drift")
    dt = target_dt.to(torch.float64)
    mu = location.to(torch.float64)
    sigma = sigma.to(torch.float64)
    lower = (dt - 0.5).clamp_min(0.5)
    upper = dt + 0.5
    standardized_lower = (
        torch.log(lower) - math.log(time_scale) - mu
    ) / sigma
    standardized_upper = (
        torch.log(upper) - math.log(time_scale) - mu
    ) / sigma
    log_cdf_lower = torch.special.log_ndtr(standardized_lower)
    log_cdf_upper = torch.special.log_ndtr(standardized_upper)
    log_sf_lower = torch.special.log_ndtr(-standardized_lower)
    log_sf_upper = torch.special.log_ndtr(-standardized_upper)
    left_mass = _logdiffexp(log_cdf_upper, log_cdf_lower)
    right_mass = _logdiffexp(log_sf_lower, log_sf_upper)
    log_mass = torch.where(standardized_lower >= 0.0, right_mass, left_mass)
    censor_boundary = torch.full_like(dt, INTERVAL_CENSOR_BOUNDARY)
    censor_z = (
        torch.log(censor_boundary) - math.log(time_scale) - mu
    ) / sigma
    log_censor_survival = torch.special.log_ndtr(-censor_z)
    result = torch.where(is_right_censored, log_censor_survival, log_mass)
    require(bool(torch.isfinite(result).all()), "Non-finite interval likelihood")
    return result


def primary_observation_log_likelihood(
    *,
    location: torch.Tensor,
    sigma: torch.Tensor,
    target_dt: torch.Tensor,
    time_scale: float,
    is_right_censored: torch.Tensor,
    observation_likelihood_mode: str,
) -> torch.Tensor:
    """Evaluate the aligned dataset observation law from explicit terms."""
    if observation_likelihood_mode == OBSERVATION_LIKELIHOOD_CONTINUOUS:
        return continuous_lognormal_log_likelihood(
            location=location,
            sigma=sigma,
            target_dt=target_dt,
            time_scale=time_scale,
            is_right_censored=is_right_censored,
        )
    require(
        observation_likelihood_mode
        == OBSERVATION_LIKELIHOOD_POSITIVE_INTEGER,
        "Unsupported aligned observation likelihood",
    )
    require(location.shape == sigma.shape == target_dt.shape, "Term shape drift")
    require(
        is_right_censored.shape == target_dt.shape
        and is_right_censored.dtype == torch.bool,
        "Censor mask mismatch",
    )
    target = target_dt.to(torch.float64)
    require(
        bool(torch.isfinite(target).all())
        and bool((target >= 1.0).all())
        and bool(
            torch.isclose(target, torch.round(target), atol=1e-8, rtol=0.0).all()
        ),
        "Positive-integer target contract drift",
    )
    mu = location.to(torch.float64)
    scale = sigma.to(torch.float64)
    first_boundary = torch.full_like(target, 1.5)
    lower = target - 0.5
    upper = target + 0.5
    first_z = (
        torch.log(first_boundary) - math.log(time_scale) - mu
    ) / scale
    lower_z = (torch.log(lower) - math.log(time_scale) - mu) / scale
    upper_z = (torch.log(upper) - math.log(time_scale) - mu) / scale
    first_mass = torch.special.log_ndtr(first_z)
    lower_log_cdf = torch.special.log_ndtr(lower_z)
    upper_log_cdf = torch.special.log_ndtr(upper_z)
    lower_log_sf = torch.special.log_ndtr(-lower_z)
    upper_log_sf = torch.special.log_ndtr(-upper_z)
    left_mass = _logdiffexp(upper_log_cdf, lower_log_cdf)
    right_mass = _logdiffexp(lower_log_sf, upper_log_sf)
    regular_mass = torch.where(lower_z >= 0.0, right_mass, left_mass)
    uncensored = torch.where(target == 1.0, first_mass, regular_mass)
    censored = torch.special.log_ndtr(-lower_z)
    result = torch.where(is_right_censored, censored, uncensored)
    require(bool(torch.isfinite(result).all()), "Non-finite primary likelihood")
    return result


@dataclass(frozen=True)
class FrozenBaseTimeCache:
    location: torch.Tensor
    sigma: torch.Tensor
    median: torch.Tensor
    target_dt: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.target_dt.shape[0])

    def validate(self) -> None:
        require(self.target_dt.ndim == 1 and self.count > 0, "Empty base cache")
        for name in ("location", "sigma", "median"):
            value = getattr(self, name)
            require(value.shape == self.target_dt.shape, f"{name} shape drift")
            require(bool(torch.isfinite(value).all()), f"Non-finite {name}")
        require(bool((self.sigma > 0.0).all()), "Base sigma is not positive")
        require(bool((self.median > 0.0).all()), "Base median is not positive")

    def digest(self) -> str:
        return canonical_state_dict_sha256(
            {
                "location": self.location,
                "sigma": self.sigma,
                "median": self.median,
                "target_dt": self.target_dt,
            }
        )


@torch.no_grad()
def derive_frozen_base_time_cache(
    *,
    model: nn.Module,
    cache: FrozenFeatureCache,
    device: torch.device,
    batch_size: int,
) -> FrozenBaseTimeCache:
    """Evaluate immutable aligned-B distribution terms once."""
    model.eval()
    locations: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    medians: list[torch.Tensor] = []
    for start in range(0, cache.count, batch_size):
        end = min(start + batch_size, cache.count)
        hidden = cache.time_hidden[start:end].to(device)
        target = cache.target_dt[start:end].to(device)
        location, sigma, _ = model._lognormal_time_terms(hidden, target)
        median = model.predict_time_median(hidden)
        locations.append(location.detach().cpu().to(torch.float64))
        scales.append(sigma.detach().cpu().to(torch.float64))
        medians.append(median.detach().cpu().to(torch.float64))
    result = FrozenBaseTimeCache(
        location=torch.cat(locations).contiguous(),
        sigma=torch.cat(scales).contiguous(),
        median=torch.cat(medians).contiguous(),
        target_dt=cache.target_dt.detach().cpu().contiguous(),
    )
    result.validate()
    return result


class IndexedCanonicalDataset(Dataset):
    """Expose canonical sample index so row i remains paired with cache row i."""

    def __init__(self, dataset: Dataset, *, limit: int | None = None) -> None:
        self.dataset = dataset
        self.count = len(dataset) if limit is None else min(len(dataset), int(limit))
        require(self.count > 0, "Indexed canonical dataset is empty")

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> tuple[int, Mapping[str, torch.Tensor]]:
        return index, self.dataset[index]


def collate_indexed_week_lookback(
    batch: list[tuple[int, Mapping[str, torch.Tensor]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    indices = torch.tensor([item[0] for item in batch], dtype=torch.long)
    marks, dts, mask, _, values = collate_week_lookback(
        [item[1] for item in batch]
    )
    return indices, marks, dts, mask, values


def make_indexed_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    count: int,
    shuffle: bool,
    seed: int | None,
) -> DataLoader:
    generator = None
    if shuffle:
        require(seed is not None, "A shuffle seed is required")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
    return DataLoader(
        IndexedCanonicalDataset(dataset, limit=count),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_indexed_week_lookback,
    )


def audit_appended_target_exclusion(
    dataset: Dataset,
    *,
    count: int,
) -> dict[str, Any]:
    """Prove that changing the appended target cannot change adapter input."""
    loader = make_indexed_loader(
        dataset,
        batch_size=min(32, count),
        count=count,
        shuffle=False,
        seed=None,
    )
    indices, _, dts, mask, _ = next(iter(loader))
    history, history_mask, target = split_appended_target(dts, mask)
    compact, lengths = compact_masked_history(history, history_mask)
    changed = dts.clone()
    changed[:, -1] = changed[:, -1] + 12345.0
    changed_history, changed_mask, changed_target = split_appended_target(
        changed, mask
    )
    changed_compact, changed_lengths = compact_masked_history(
        changed_history, changed_mask
    )
    require(not torch.equal(target, changed_target), "Target perturbation failed")
    require(
        torch.equal(compact, changed_compact)
        and torch.equal(lengths, changed_lengths),
        "Appended target leaked into adapter history",
    )
    return {
        "verified": True,
        "audited_sample_count": int(indices.numel()),
        "target_perturbation": "+12345 in the appended final token",
        "adapter_input_bitwise_identical": True,
    }


def _batch_terms(
    *,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None],
    base_cache: FrozenBaseTimeCache,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    indices, _, dts, mask, _ = batch
    history_dt, history_mask, target = split_appended_target(dts, mask)
    expected = base_cache.target_dt[indices]
    require(torch.equal(target.cpu(), expected), "Dataset/cache target order drift")
    return (
        indices,
        history_dt.to(device),
        history_mask.to(device),
        target.to(device),
        base_cache.location[indices].to(device),
        base_cache.sigma[indices].to(device),
    )


def _censor_for_target(
    target: torch.Tensor,
    *,
    censor_threshold: float | None,
) -> torch.Tensor:
    return censor_mask(target, threshold=censor_threshold)


def canonical_frozen_time_median(
    *,
    base_cache: FrozenBaseTimeCache,
    count: int,
    time_scale: float,
) -> tuple[torch.Tensor, float]:
    """Return aligned-B's exact median tensor after checking its formula.

    ``base_cache.median`` is calculated by aligned-B on the requested runtime
    device. Re-evaluating ``exp(location)`` on CPU can differ from the CUDA
    result by one float64 ULP, even though the location is identical. Keep the
    canonical source tensor for the bitwise identity gate and use a tight
    numerical check only for the cross-device formula replay.
    """
    canonical = base_cache.median[:count]
    implied = time_scale * torch.exp(base_cache.location[:count].to(torch.float64))
    difference = torch.abs(implied - canonical)
    maximum = float(difference.max().item()) if difference.numel() else 0.0
    require(
        torch.allclose(implied, canonical, atol=1e-12, rtol=1e-12),
        "aligned-B median is inconsistent with its unchanged location",
    )
    return canonical, maximum


@torch.no_grad()
def evaluate_scale_module(
    *,
    module: nn.Module,
    dataset: Dataset,
    base_cache: FrozenBaseTimeCache,
    count: int,
    device: torch.device,
    batch_size: int,
    time_scale: float,
    censor_threshold: float | None,
    observation_likelihood_mode: str,
) -> dict[str, Any]:
    module.eval()
    primary_total = 0.0
    continuous_total = 0.0
    first_total = 0.0
    first_count = 0
    remainder_total = 0.0
    remainder_count = 0
    observed = 0
    deltas: list[torch.Tensor] = []
    for batch in make_indexed_loader(
        dataset,
        batch_size=batch_size,
        count=count,
        shuffle=False,
        seed=None,
    ):
        indices, history_dt, history_mask, target, location, base_sigma = (
            _batch_terms(batch=batch, base_cache=base_cache, device=device)
        )
        compact, lengths = compact_masked_history(history_dt, history_mask)
        delta = module.bounded_log_scale_residual(compact, lengths)
        sigma = module.adjusted_scale(
            base_sigma.to(torch.float64), compact, lengths
        ).to(torch.float64)
        censored = _censor_for_target(
            target, censor_threshold=censor_threshold
        ).to(device)
        primary = primary_observation_log_likelihood(
            location=location,
            sigma=sigma,
            target_dt=target,
            time_scale=time_scale,
            is_right_censored=censored,
            observation_likelihood_mode=observation_likelihood_mode,
        )
        continuous = continuous_lognormal_log_likelihood(
            location=location,
            sigma=sigma,
            target_dt=target,
            time_scale=time_scale,
            is_right_censored=censored,
        )
        first_mask = target.to(torch.float64) == 1.0
        remainder_mask = ~first_mask
        primary_total -= float(primary.sum().item())
        continuous_total -= float(continuous.sum().item())
        if bool(first_mask.any()):
            first_total -= float(primary[first_mask].sum().item())
            first_count += int(first_mask.sum().item())
        if bool(remainder_mask.any()):
            remainder_total -= float(primary[remainder_mask].sum().item())
            remainder_count += int(remainder_mask.sum().item())
        observed += int(indices.numel())
        deltas.append(delta.detach().cpu().to(torch.float64))
    require(observed == count, "Evaluation target count drift")
    delta_all = torch.cat(deltas).contiguous()
    canonical_median, median_formula_max_abs_error = canonical_frozen_time_median(
        base_cache=base_cache,
        count=count,
        time_scale=time_scale,
    )
    result = {
        "count": observed,
        "observation_likelihood_mode": observation_likelihood_mode,
        "primary_proper_time_nll": primary_total / observed,
        "continuous_reference_time_nll": continuous_total / observed,
        "first_bin_count": first_count,
        "first_bin_primary_nll": (
            first_total / first_count if first_count else None
        ),
        "remainder_count": remainder_count,
        "remainder_primary_nll": (
            remainder_total / remainder_count if remainder_count else None
        ),
        "bounded_log_scale_residual_mean": float(delta_all.mean().item()),
        "bounded_log_scale_residual_std": float(
            delta_all.std(unbiased=False).item()
        ),
        "bounded_log_scale_residual_min": float(delta_all.min().item()),
        "bounded_log_scale_residual_max": float(delta_all.max().item()),
        "bounded_log_scale_residual_sha256": _tensor_digest(
            "bounded_log_scale_residual", delta_all
        ),
        "time_median_sha256": _tensor_digest(
            "time_median", canonical_median
        ),
        "cross_device_median_formula_max_abs_error": (
            median_formula_max_abs_error
        ),
        "base_location_sha256": _tensor_digest(
            "base_location", base_cache.location[:count]
        ),
    }
    for name, value in result.items():
        if name.endswith("nll") and value is not None:
            require(math.isfinite(float(value)), f"Non-finite metric: {name}")
    require(
        all(
            math.isfinite(float(result[name]))
            for name in (
                "bounded_log_scale_residual_mean",
                "bounded_log_scale_residual_std",
                "bounded_log_scale_residual_min",
                "bounded_log_scale_residual_max",
                "cross_device_median_formula_max_abs_error",
            )
        ),
        "Non-finite evaluation telemetry",
    )
    return result


def _finite_optimizer(optimizer: torch.optim.Optimizer) -> None:
    finite_nested_tensors(optimizer.state_dict(), label="Adapter optimizer")


def train_scale_epoch(
    *,
    module: nn.Module,
    dataset: Dataset,
    base_cache: FrozenBaseTimeCache,
    count: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    time_scale: float,
    censor_threshold: float | None,
    observation_likelihood_mode: str,
    grad_clip: float,
    seed: int,
    epoch: int,
) -> dict[str, float]:
    module.train()
    parameters = tuple(module.parameters())
    total = 0.0
    observed = 0
    gradient_norms: list[float] = []
    loader = make_indexed_loader(
        dataset,
        batch_size=batch_size,
        count=count,
        shuffle=True,
        seed=int(seed) * 1_000_003 + int(epoch),
    )
    for batch in loader:
        indices, history_dt, history_mask, target, location, base_sigma = (
            _batch_terms(batch=batch, base_cache=base_cache, device=device)
        )
        compact, lengths = compact_masked_history(history_dt, history_mask)
        sigma = module.adjusted_scale(
            base_sigma.to(torch.float64), compact, lengths
        ).to(torch.float64)
        censored = _censor_for_target(
            target, censor_threshold=censor_threshold
        ).to(device)
        loss_terms = -primary_observation_log_likelihood(
            location=location,
            sigma=sigma,
            target_dt=target,
            time_scale=time_scale,
            is_right_censored=censored,
            observation_likelihood_mode=observation_likelihood_mode,
        )
        loss = loss_terms.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in module.named_parameters():
            require(parameter.grad is not None, f"Missing adapter gradient: {name}")
            require(
                bool(torch.isfinite(parameter.grad).all()),
                f"Non-finite adapter gradient: {name}",
            )
        norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
        require(bool(torch.isfinite(norm)), "Non-finite adapter gradient norm")
        optimizer.step()
        _finite_optimizer(optimizer)
        finite_nested_tensors(module.state_dict(), label="Adapter state")
        total += float(loss_terms.detach().sum().item())
        observed += int(indices.numel())
        gradient_norms.append(float(norm.detach().cpu().item()))
    require(observed == count, "Training target count drift")
    module.eval()
    return {
        "train_primary_proper_time_nll": total / observed,
        "pre_clip_gradient_norm_mean": float(np.mean(gradient_norms)),
        "pre_clip_gradient_norm_max": float(np.max(gradient_norms)),
    }


def earliest_strict_minimum(history: list[dict[str, Any]]) -> dict[str, Any]:
    require(bool(history), "Empty selection history")
    best = history[0]
    best_value = float(best["val_primary_proper_time_nll"])
    require(math.isfinite(best_value), "Non-finite epoch-zero metric")
    for row in history[1:]:
        value = float(row["val_primary_proper_time_nll"])
        require(math.isfinite(value), "Non-finite selection history")
        if value < best_value:
            best = row
            best_value = value
    return best


def early_stopping_exhausted(
    history: list[dict[str, Any]],
    *,
    minimum_epochs: int,
    patience: int,
) -> bool:
    current = int(history[-1]["epoch"])
    best = int(earliest_strict_minimum(history)["epoch"])
    return current >= minimum_epochs and current - best >= patience


def _fit_identity(
    *,
    role: str,
    contract_sha256: str,
    dataset_name: str,
    source_checkpoint_sha256: str,
    source_state_sha256: str,
    frozen_cache_sha256: str,
    base_cache_sha256: str,
    history_statistics: Mapping[str, Any],
    module_initial_sha256: str,
    train_count: int,
    validation_count: int,
    settings: Mapping[str, Any],
    source_revision: str,
    observation_likelihood_contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "contract_sha256": contract_sha256,
        "role": role,
        "dataset": dataset_name,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_state_sha256": source_state_sha256,
        "frozen_cache_sha256": frozen_cache_sha256,
        "base_cache_sha256": base_cache_sha256,
        "history_statistics": dict(history_statistics),
        "module_initial_state_sha256": module_initial_sha256,
        "train_count": train_count,
        "validation_count": validation_count,
        "seed": int(settings["seed"]),
        "optimizer": settings["optimizer"],
        "learning_rate": float(settings["learning_rate"]),
        "weight_decay": float(settings["weight_decay"]),
        "batch_size": int(settings["batch_size"]),
        "gradient_clip": float(settings["gradient_clip"]),
        "planned_epochs": int(settings["epochs"]),
        "minimum_epochs": int(settings["minimum_epochs"]),
        "early_stopping_patience": int(settings["early_stopping_patience"]),
        "selection": "earliest_strict_finite_minimum_validation_primary_nll",
        "observation_likelihood_contract": dict(observation_likelihood_contract),
        "source_revision": source_revision,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }


def _validate_resume_payload(
    payload: Mapping[str, Any], *, identity: Mapping[str, Any]
) -> None:
    require(
        payload.get("checkpoint_type") == "aligned_causal_scale_adapter_resume",
        "Wrong adapter resume type",
    )
    require(payload.get("resume_identity") == identity, "Adapter resume drift")
    state = payload.get("module_state_dict")
    require(isinstance(state, Mapping), "Adapter resume state is missing")
    require(
        canonical_state_dict_sha256(state) == payload.get("module_state_sha256"),
        "Adapter resume state digest drift",
    )
    finite_nested_tensors(payload, label="Adapter resume payload")


def fit_scale_module(
    *,
    role: str,
    module: nn.Module,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    train_base_cache: FrozenBaseTimeCache,
    validation_base_cache: FrozenBaseTimeCache,
    train_count: int,
    validation_count: int,
    output_dir: Path,
    device: torch.device,
    time_scale: float,
    censor_threshold: float | None,
    observation_likelihood_contract: Mapping[str, Any],
    contract_sha256: str,
    source_checkpoint_sha256: str,
    source_state_sha256: str,
    frozen_cache_sha256: str,
    history_statistics: Mapping[str, Any],
    source_revision: str,
    settings: Mapping[str, Any],
    run_epoch_limit: int | None = None,
) -> dict[str, Any]:
    """Fit the candidate or one attribution control with exact resume."""
    require(
        role in {"candidate", "length_only_control", "global_scale_control"},
        "Unknown fit role",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    module = module.to(device)
    module.eval()
    initial_state = _clone_state(module)
    initial_sha256 = canonical_state_dict_sha256(initial_state)
    expected_parameters = 1 if role == "global_scale_control" else 273
    require(
        sum(parameter.numel() for parameter in module.parameters())
        == expected_parameters,
        f"Unexpected {role} parameter count",
    )
    observation_mode = str(observation_likelihood_contract["mode"])
    identity = _fit_identity(
        role=role,
        contract_sha256=contract_sha256,
        dataset_name=str(settings["dataset"]),
        source_checkpoint_sha256=source_checkpoint_sha256,
        source_state_sha256=source_state_sha256,
        frozen_cache_sha256=frozen_cache_sha256,
        base_cache_sha256=(
            train_base_cache.digest() + ":" + validation_base_cache.digest()
        ),
        history_statistics=history_statistics,
        module_initial_sha256=initial_sha256,
        train_count=train_count,
        validation_count=validation_count,
        settings=settings,
        source_revision=source_revision,
        observation_likelihood_contract=observation_likelihood_contract,
    )
    selected_path = output_dir / SELECTED_CHECKPOINT_NAME
    last_path = output_dir / LAST_CHECKPOINT_NAME
    summary_path = output_dir / SUMMARY_NAME
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        require(summary.get("status") == "success", "Cached summary is incomplete")
        require(summary.get("resume_identity") == identity, "Cached summary drift")
        require(selected_path.is_file(), "Cached selected checkpoint is missing")
        selected = torch_load_checkpoint(selected_path, map_location="cpu")
        require(selected.get("resume_identity") == identity, "Cached checkpoint drift")
        require(
            canonical_state_dict_sha256(selected["module_state_dict"])
            == selected.get("module_state_sha256")
            == summary.get("selected_module_state_sha256"),
            "Cached selected state digest drift",
        )
        module.load_state_dict(selected["module_state_dict"], strict=True)
        replay = evaluate_scale_module(
            module=module,
            dataset=validation_dataset,
            base_cache=validation_base_cache,
            count=validation_count,
            device=device,
            batch_size=int(settings["batch_size"]),
            time_scale=time_scale,
            censor_threshold=censor_threshold,
            observation_likelihood_mode=observation_mode,
        )
        require(
            math.isclose(
                float(replay["primary_proper_time_nll"]),
                float(summary["selected_validation_metrics"]["primary_proper_time_nll"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "Cached selected primary NLL replay drift",
        )
        return summary

    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    planned_epochs = int(settings["epochs"])
    minimum_epochs = int(settings["minimum_epochs"])
    patience = int(settings["early_stopping_patience"])
    batch_size = int(settings["batch_size"])
    grad_clip = float(settings["gradient_clip"])
    seed = int(settings["seed"])
    if last_path.exists():
        payload = torch_load_checkpoint(last_path, map_location="cpu")
        _validate_resume_payload(payload, identity=identity)
        module.load_state_dict(payload["module_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        _finite_optimizer(optimizer)
        history = list(payload["history"])
        best_state = copy.deepcopy(payload["best_state_dict"])
        start_epoch = int(payload["epoch"]) + 1
    else:
        epoch_zero = evaluate_scale_module(
            module=module,
            dataset=validation_dataset,
            base_cache=validation_base_cache,
            count=validation_count,
            device=device,
            batch_size=batch_size,
            time_scale=time_scale,
            censor_threshold=censor_threshold,
            observation_likelihood_mode=observation_mode,
        )
        require(
            epoch_zero["bounded_log_scale_residual_min"] == 0.0
            and epoch_zero["bounded_log_scale_residual_max"] == 0.0,
            f"{role} epoch zero does not reproduce aligned-B",
        )
        history = [{
            "epoch": 0,
            "train_primary_proper_time_nll": None,
            "val_primary_proper_time_nll": epoch_zero["primary_proper_time_nll"],
            "val_continuous_reference_time_nll": epoch_zero[
                "continuous_reference_time_nll"
            ],
            "pre_clip_gradient_norm_mean": None,
            "pre_clip_gradient_norm_max": None,
        }]
        best_state = _clone_state(module)
        start_epoch = 1

    stopped_early = early_stopping_exhausted(
        history, minimum_epochs=minimum_epochs, patience=patience
    )
    call_last_epoch = planned_epochs
    if run_epoch_limit is not None:
        require(run_epoch_limit >= 0, "run_epoch_limit must be nonnegative")
        call_last_epoch = min(planned_epochs, start_epoch + run_epoch_limit - 1)
    for epoch in range(start_epoch, call_last_epoch + 1):
        if stopped_early:
            break
        train_metrics = train_scale_epoch(
            module=module,
            dataset=train_dataset,
            base_cache=train_base_cache,
            count=train_count,
            optimizer=optimizer,
            device=device,
            batch_size=batch_size,
            time_scale=time_scale,
            censor_threshold=censor_threshold,
            observation_likelihood_mode=observation_mode,
            grad_clip=grad_clip,
            seed=seed,
            epoch=epoch,
        )
        validation_metrics = evaluate_scale_module(
            module=module,
            dataset=validation_dataset,
            base_cache=validation_base_cache,
            count=validation_count,
            device=device,
            batch_size=batch_size,
            time_scale=time_scale,
            censor_threshold=censor_threshold,
            observation_likelihood_mode=observation_mode,
        )
        history.append({
            "epoch": epoch,
            **train_metrics,
            "val_primary_proper_time_nll": validation_metrics[
                "primary_proper_time_nll"
            ],
            "val_continuous_reference_time_nll": validation_metrics[
                "continuous_reference_time_nll"
            ],
        })
        selected_row = earliest_strict_minimum(history)
        if int(selected_row["epoch"]) == epoch:
            best_state = _clone_state(module)
        state = _clone_state(module)
        atomic_torch_save({
            "checkpoint_type": "aligned_causal_scale_adapter_resume",
            "checkpoint_schema_version": 1,
            "role": role,
            "epoch": epoch,
            "module_state_dict": state,
            "module_state_sha256": canonical_state_dict_sha256(state),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "best_epoch": int(selected_row["epoch"]),
            "best_state_dict": best_state,
            "best_state_sha256": canonical_state_dict_sha256(best_state),
            "resume_identity": identity,
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
        }, last_path)
        stopped_early = early_stopping_exhausted(
            history, minimum_epochs=minimum_epochs, patience=patience
        )

    current_epoch = int(history[-1]["epoch"])
    paused = (
        run_epoch_limit is not None
        and current_epoch < planned_epochs
        and not stopped_early
    )
    if paused:
        return {
            "status": "paused",
            "role": role,
            "completed_epochs": current_epoch,
            "resume_identity": identity,
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
        }

    selected_row = earliest_strict_minimum(history)
    module.load_state_dict(best_state, strict=True)
    module.eval()
    selected_metrics = evaluate_scale_module(
        module=module,
        dataset=validation_dataset,
        base_cache=validation_base_cache,
        count=validation_count,
        device=device,
        batch_size=batch_size,
        time_scale=time_scale,
        censor_threshold=censor_threshold,
        observation_likelihood_mode=observation_mode,
    )
    require(
        math.isclose(
            float(selected_metrics["primary_proper_time_nll"]),
            float(selected_row["val_primary_proper_time_nll"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Selected adapter metric replay drift",
    )
    selected_state = _clone_state(module)
    selected_sha256 = canonical_state_dict_sha256(selected_state)
    checkpoint = {
        "checkpoint_type": "selected_aligned_causal_duration_scale_adapter",
        "checkpoint_schema_version": 1,
        "role": role,
        "best_epoch": int(selected_row["epoch"]),
        "selected_metric_value": selected_metrics["primary_proper_time_nll"],
        "module_state_dict": selected_state,
        "module_state_sha256": selected_sha256,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_state_sha256": source_state_sha256,
        "resume_identity": identity,
        "selection": "earliest_strict_finite_minimum_validation_primary_nll",
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }
    atomic_torch_save(checkpoint, selected_path)
    summary = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "status": "success",
        "role": role,
        "best_epoch": int(selected_row["epoch"]),
        "completed_epochs": current_epoch,
        "stopped_early": bool(stopped_early),
        "epoch_zero_validation_primary_nll": float(
            history[0]["val_primary_proper_time_nll"]
        ),
        "selected_validation_metrics": selected_metrics,
        "module_initial_state_sha256": initial_sha256,
        "selected_module_state_sha256": selected_sha256,
        "trainable_parameter_count": expected_parameters,
        "history": history,
        "resume_identity": identity,
        "selected_checkpoint_path": str(selected_path.resolve()),
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }
    save_json(summary_path, summary)
    return summary


def load_verified_feature_cache(
    *,
    path: Path,
    split: str,
    dataset_name: str,
    dataset_spec: Mapping[str, Any],
    selected_payload: Mapping[str, Any],
    require_quantity: bool,
) -> FrozenFeatureCache:
    """Reuse the exact v1 cache and bind it to this canonical target order."""
    payload = torch_load_checkpoint(path, map_location="cpu")
    identity = payload.get("identity")
    require(isinstance(identity, Mapping), "Feature-cache identity is missing")
    require(identity.get("dataset") == dataset_name, "Feature-cache dataset drift")
    require(identity.get("split") == split, "Feature-cache split drift")
    require(
        identity.get("held_out_test_evaluated") is False,
        "Held-out data entered the feature cache",
    )
    require(
        identity.get("target_identity_sha256")
        == dataset_spec[f"expected_{split}_target_identity_sha256"],
        "Feature-cache target identity drift",
    )
    cache = FrozenFeatureCache.from_payload(
        payload,
        expected_identity=identity,
        require_quantity=require_quantity,
    )
    resume = selected_payload.get("resume_identity")
    require(isinstance(resume, Mapping), "aligned-B resume identity is missing")
    require(
        cache.count == int(dataset_spec[f"expected_{split}_targets"]),
        "Feature-cache target count drift",
    )
    require(
        cache.digest() == resume[f"{split}_cache_sha256"],
        "Feature-cache digest differs from selected aligned-B",
    )
    require(
        cache.digest()
        == dataset_spec[f"aligned_B_{split}_feature_cache_sha256"],
        "Feature-cache digest differs from adapter contract",
    )
    require(
        target_dt_sha256(cache.target_dt)
        == dataset_spec[f"expected_{split}_target_dt_sha256"],
        "Feature-cache duration identity drift",
    )
    return cache


def derive_train_history_statistics(
    dataset: RMTPPWeekLookbackDataset,
) -> dict[str, float | int]:
    """Exact sample-weighted active-context log1p moments, without targets."""
    count = 0
    total = 0.0
    square_total = 0.0
    lengths: list[int] = []
    for part_index, context_end in dataset.index:
        sequence = np.asarray(dataset.seq_lists[part_index], dtype=np.int32)
        durations = np.asarray(dataset.dt_lists[part_index], dtype=np.float64)
        left_sequence = int(sequence[context_end]) - (dataset.W - 1)
        start = int(np.searchsorted(sequence, left_sequence, side="left"))
        start = max(start, context_end - (dataset.max_len - 2))
        active = np.log1p(durations[start : context_end + 1])
        require(active.size > 0, "Empty canonical active context")
        require(bool(np.isfinite(active).all()), "Non-finite context transform")
        count += int(active.size)
        total += float(active.sum(dtype=np.float64))
        square_total += float(np.square(active).sum(dtype=np.float64))
        lengths.append(int(active.size))
    require(count > 0, "Empty train history population")
    mean = total / count
    std = math.sqrt(max(0.0, square_total / count - mean * mean))
    require(std > 0.0 and math.isfinite(std), "Degenerate train history std")
    return {
        "token_count": count,
        "log1p_mean": mean,
        "log1p_std": std,
        "context_length_min": min(lengths),
        "context_length_max": max(lengths),
        "context_length_mean": float(sum(lengths) / len(lengths)),
    }


def validate_history_statistics(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    for name in ("token_count", "context_length_min", "context_length_max"):
        require(int(observed[name]) == int(expected[name]), f"{name} drift")
    for name in ("log1p_mean", "log1p_std", "context_length_mean"):
        require(
            math.isclose(
                float(observed[name]),
                float(expected[name]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"{name} drift",
        )


def freeze_source_model(model: nn.Module) -> tuple[str, tuple[str, ...]]:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    trainable = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    require(not trainable, f"aligned-B still exposes trainable tensors: {trainable}")
    return canonical_state_dict_sha256(model.state_dict()), trainable


def validate_contract(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    require(contract.get("schema_version") == 1, "Contract schema drift")
    require(contract.get("contract_id") == CONTRACT_ID, "Wrong adapter contract")
    candidate = contract.get("candidate")
    require(isinstance(candidate, Mapping), "Candidate contract is missing")
    for name, expected in {
        "single_hypothesis": True,
        "source_contract_id": "aligned_frozen_lognormal_duration_v1",
        "source_contract_sha256": "677cc3af91d84dfea8e3b4e71b38697fb467059f680678f0b8370a50f2366f16",
        "backbone_change": False,
        "quantity_path_change": False,
        "location_change": False,
        "scale_only": True,
    }.items():
        require(candidate.get(name) == expected, f"Candidate drift: {name}")
    adapter = contract.get("adapter")
    require(isinstance(adapter, Mapping), "Adapter contract is missing")
    for name, expected in {
        "input_size": 1,
        "hidden_size": 8,
        "num_layers": 1,
        "bidirectional": False,
        "dropout": 0.0,
        "expected_trainable_parameter_count": 273,
        "appended_target_excluded": True,
        "marks_excluded": True,
        "quantities_excluded": True,
        "B_hidden_excluded": True,
        "sigma_floor": 0.001,
        "minimum_scale_ratio": 0.1,
        "maximum_scale_ratio": 10.0,
    }.items():
        require(adapter.get(name) == expected, f"Adapter drift: {name}")
    controls = contract.get("controls")
    require(isinstance(controls, Mapping), "Control contract is missing")
    length_only = controls.get("same_capacity_length_only")
    require(isinstance(length_only, Mapping), "Length-only control is missing")
    require(
        length_only.get("trainable_parameter_count") == 273
        and length_only.get("same_lengths_masks_initial_state_optimizer_batch_order")
        is True,
        "Length-only parity drift",
    )
    scope = contract.get("scope")
    require(isinstance(scope, Mapping), "Scope contract is missing")
    for name, expected in {
        "datasets": [
            "intermittent_frozen_5000",
            "yellow_trip_hourly",
            "insta_market_basket",
        ],
        "input_splits": ["train", "validation"],
        "evaluation_scope": "validation_only",
        "held_out_test": False,
        "seed": 42,
        "additional_seeds": False,
        "quantity_predictions_may_change": False,
        "time_median_may_change": False,
    }.items():
        require(scope.get(name) == expected, f"Scope drift: {name}")
    optimization = contract.get("optimization")
    require(isinstance(optimization, Mapping), "Optimization contract is missing")
    for name, expected in {
        "shared_across_all_datasets": True,
        "seed": 42,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "batch_size": 4096,
        "gradient_clip": 1.0,
        "epochs": 50,
        "minimum_epochs": 5,
        "early_stopping_patience": 8,
        "scheduler": None,
        "dataset_specific_hyperparameters": False,
    }.items():
        require(optimization.get(name) == expected, f"Optimization drift: {name}")
    selection = contract.get("checkpoint_selection")
    require(isinstance(selection, Mapping), "Selector contract is missing")
    require(
        selection.get("monitor")
        == "validation primary proper observation NLL"
        and selection.get("rule") == "earliest strict finite minimum"
        and selection.get("fallback") == "epoch 0"
        and selection.get("secondary_metrics_in_selector") is False,
        "Selector drift",
    )
    structured = contract.get("acceptance", {}).get("structured_gates")
    require(isinstance(structured, Mapping), "Acceptance gates are missing")
    for name, expected in {
        "maximum_candidate_minus_aligned_A_primary_nll": 0.01,
        "minimum_candidate_improvement_over_each_control_primary_nll": 0.005,
        "maximum_candidate_minus_aligned_B_continuous_nll": 0.01,
        "require_quantity_prediction_bitwise_identity": True,
        "require_time_median_bitwise_identity": True,
        "require_base_location_bitwise_identity": True,
        "require_source_model_state_digest_identity": True,
    }.items():
        require(structured.get(name) == expected, f"Acceptance drift: {name}")
    rows = contract.get("datasets")
    require(isinstance(rows, list) and len(rows) == 3, "Expected three datasets")
    datasets = {str(row["dataset"]): dict(row) for row in rows}
    require(tuple(datasets) == tuple(scope["datasets"]), "Dataset order drift")
    expected_improvements = {
        "intermittent_frozen_5000": -1e-6,
        "yellow_trip_hourly": 0.0,
        "insta_market_basket": 0.005,
    }
    for dataset_name, row in datasets.items():
        resolve_censor_threshold(dataset_name, row)
        validate_observation_likelihood_contract(row["observation_contract"])
        for key in (
            "data_sha256",
            "split_manifest_sha256",
            "aligned_B_checkpoint_file_sha256",
            "aligned_B_model_state_sha256",
            "aligned_B_train_feature_cache_sha256",
            "aligned_B_validation_feature_cache_sha256",
            "aligned_B_quantity_prediction_sha256",
        ):
            value = str(row.get(key, ""))
            require(
                len(value) == 64
                and all(character in "0123456789abcdef" for character in value),
                f"Missing {dataset_name} {key}",
            )
        for key in (
            "aligned_A_validation_primary_nll",
            "aligned_B_validation_primary_nll",
            "aligned_B_validation_continuous_nll",
        ):
            require(math.isfinite(float(row[key])), f"Invalid {dataset_name} {key}")
        require(
            float(row["minimum_candidate_primary_improvement"])
            == expected_improvements[dataset_name],
            f"Baseline threshold drift: {dataset_name}",
        )
    return datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-partial-contract", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-validation-batches", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--run-epoch-limit", type=int, default=None)
    return parser.parse_args()


def _effective_count(
    full_count: int,
    *,
    max_batches: int | None,
    batch_size: int,
) -> int:
    if max_batches is None:
        return full_count
    require(max_batches > 0, "max_batches must be positive")
    return min(full_count, max_batches * batch_size)


@torch.no_grad()
def _base_metrics(
    *,
    base_cache: FrozenBaseTimeCache,
    count: int,
    time_scale: float,
    censor_threshold: float | None,
    batch_size: int,
    observation_likelihood_mode: str,
) -> dict[str, Any]:
    primary_total = 0.0
    continuous_total = 0.0
    first_total = 0.0
    first_count = 0
    remainder_total = 0.0
    remainder_count = 0
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        target = base_cache.target_dt[start:end]
        censored = censor_mask(target, threshold=censor_threshold)
        primary = primary_observation_log_likelihood(
            location=base_cache.location[start:end],
            sigma=base_cache.sigma[start:end],
            target_dt=target,
            time_scale=time_scale,
            is_right_censored=censored,
            observation_likelihood_mode=observation_likelihood_mode,
        )
        continuous = continuous_lognormal_log_likelihood(
            location=base_cache.location[start:end],
            sigma=base_cache.sigma[start:end],
            target_dt=target,
            time_scale=time_scale,
            is_right_censored=censored,
        )
        first_mask = target.to(torch.float64) == 1.0
        remainder_mask = ~first_mask
        primary_total -= float(primary.sum().item())
        continuous_total -= float(continuous.sum().item())
        if bool(first_mask.any()):
            first_total -= float(primary[first_mask].sum().item())
            first_count += int(first_mask.sum().item())
        if bool(remainder_mask.any()):
            remainder_total -= float(primary[remainder_mask].sum().item())
            remainder_count += int(remainder_mask.sum().item())
    return {
        "count": count,
        "observation_likelihood_mode": observation_likelihood_mode,
        "primary_proper_time_nll": primary_total / count,
        "continuous_reference_time_nll": continuous_total / count,
        "first_bin_count": first_count,
        "first_bin_primary_nll": first_total / first_count if first_count else None,
        "remainder_count": remainder_count,
        "remainder_primary_nll": (
            remainder_total / remainder_count if remainder_count else None
        ),
        "time_median_sha256": _tensor_digest(
            "time_median", base_cache.median[:count]
        ),
        "base_location_sha256": _tensor_digest(
            "base_location", base_cache.location[:count]
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    contract = load_contract_without_duplicate_keys(args.contract)
    datasets = validate_contract(contract)
    require(args.dataset in datasets, "Dataset is outside the adapter contract")
    dataset_spec = datasets[args.dataset]
    partial = any(
        value is not None
        for value in (
            args.max_train_batches,
            args.max_validation_batches,
            args.max_epochs,
            args.run_epoch_limit,
        )
    )
    if partial:
        require(
            args.allow_partial_contract,
            "Batch/epoch limits require --allow-partial-contract",
        )
    require(
        len(args.source_revision) == 40
        and all(character in "0123456789abcdef" for character in args.source_revision),
        "source_revision must be a full SHA",
    )
    require(sha256_file(args.data) == dataset_spec["data_sha256"], "Data digest drift")
    require(
        sha256_file(args.split_manifest) == dataset_spec["split_manifest_sha256"],
        "Split-manifest digest drift",
    )
    checkpoint_sha256 = sha256_file(args.checkpoint)
    require(
        checkpoint_sha256 == dataset_spec["aligned_B_checkpoint_file_sha256"],
        "Aligned-B checkpoint file digest drift",
    )
    selected_payload = torch_load_checkpoint(args.checkpoint, map_location="cpu")
    require(
        selected_payload.get("checkpoint_type") == "selected_frozen_lognormal_duration",
        "Expected a selected aligned-B duration checkpoint",
    )
    require(
        selected_payload.get("model_role") == "B",
        "Adapter source is not aligned-B",
    )
    require(
        selected_payload.get("model_state_sha256")
        == dataset_spec["aligned_B_model_state_sha256"],
        "Aligned-B model-state digest drift",
    )
    observation_contract = validate_observation_likelihood_contract(
        dataset_spec["observation_contract"]
    )
    observation_mode = str(observation_contract["mode"])

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA was requested but unavailable")
        device_name = torch.cuda.get_device_name(requested_device)
        # PyTorch 2.11/CUDA 13 on the 5080 runtime must initialize the device
        # context before its allocator statistics can be reset.
        torch.cuda.reset_peak_memory_stats(requested_device)
    else:
        device_name = str(requested_device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    raw_frame = load_admitted_frame(args.data)
    frame = prepare_count_frame(raw_frame)
    lookback = int(dataset_spec["lookback"])
    max_sequence_length = int(dataset_spec["max_sequence_length"])
    train_dataset = RMTPPWeekLookbackDataset(
        frame,
        lookback_weeks=lookback,
        max_seq_len=max_sequence_length,
        mode="all",
        split_col="chronological_split",
        target_splits={"train"},
    )
    validation_dataset = RMTPPWeekLookbackDataset(
        frame,
        lookback_weeks=lookback,
        max_seq_len=max_sequence_length,
        mode="all",
        split_col="chronological_split",
        target_splits={"validation"},
    )
    train_population = exact_target_population_contract(
        frame,
        target_split="train",
        lookback=lookback,
        max_seq_len=max_sequence_length,
    )
    validation_population = exact_target_population_contract(
        frame,
        target_split="validation",
        lookback=lookback,
        max_seq_len=max_sequence_length,
    )
    validate_target_population(train_population, dataset_spec=dataset_spec)
    validate_target_population(validation_population, dataset_spec=dataset_spec)
    observed_history_statistics = derive_train_history_statistics(train_dataset)
    validate_history_statistics(
        observed_history_statistics,
        dataset_spec["train_active_context_log1p"],
    )

    train_cache = load_verified_feature_cache(
        path=args.feature_cache_dir / "train_features.pt",
        split="train",
        dataset_name=args.dataset,
        dataset_spec=dataset_spec,
        selected_payload=selected_payload,
        require_quantity=False,
    )
    validation_cache = load_verified_feature_cache(
        path=args.feature_cache_dir / "validation_features.pt",
        split="validation",
        dataset_name=args.dataset,
        dataset_spec=dataset_spec,
        selected_payload=selected_payload,
        require_quantity=True,
    )
    require(len(train_dataset) == train_cache.count, "Train dataset/cache count drift")
    require(
        len(validation_dataset) == validation_cache.count,
        "Validation dataset/cache count drift",
    )
    train_target_exclusion_audit = audit_appended_target_exclusion(
        train_dataset, count=train_cache.count
    )
    validation_target_exclusion_audit = audit_appended_target_exclusion(
        validation_dataset, count=validation_cache.count
    )

    source_model = build_candidate_from_selected_checkpoint(selected_payload).to(
        requested_device
    )
    source_state_sha256, trainable_source = freeze_source_model(source_model)
    require(not trainable_source, "Aligned-B parameter boundary failed")
    require(
        source_state_sha256 == dataset_spec["aligned_B_model_state_sha256"],
        "Rebuilt aligned-B digest drift",
    )
    batch_size = int(contract["optimization"]["batch_size"])
    train_base_cache = derive_frozen_base_time_cache(
        model=source_model,
        cache=train_cache,
        device=requested_device,
        batch_size=batch_size,
    )
    validation_base_cache = derive_frozen_base_time_cache(
        model=source_model,
        cache=validation_cache,
        device=requested_device,
        batch_size=batch_size,
    )
    assert validation_cache.source_quantity_prediction is not None
    assert validation_cache.target_quantity is not None
    cached_quantity_sha256 = tensor_sha256(
        "quantity_prediction", validation_cache.source_quantity_prediction
    )
    require(
        cached_quantity_sha256
        == dataset_spec["aligned_B_quantity_prediction_sha256"],
        "Aligned-B cached quantity prediction provenance drift",
    )
    runtime_quantity_before = cached_quantity_predictions(
        model=source_model,
        cache=validation_cache,
        device=requested_device,
        batch_size=batch_size,
    )
    tolerance = contract["identity_and_stability"]
    cached_vs_runtime_close = torch.allclose(
        runtime_quantity_before,
        validation_cache.source_quantity_prediction,
        atol=float(tolerance["cross_device_quantity_absolute_tolerance"]),
        rtol=float(tolerance["cross_device_quantity_relative_tolerance"]),
    )
    require(
        cached_vs_runtime_close,
        "Cached aligned-B quantity predictions drift from runtime replay",
    )

    settings = dict(contract["optimization"])
    settings["dataset"] = args.dataset
    if args.max_epochs is not None:
        require(args.max_epochs >= 0, "max_epochs must be nonnegative")
        settings["epochs"] = min(int(settings["epochs"]), int(args.max_epochs))
    train_count = _effective_count(
        train_cache.count,
        max_batches=args.max_train_batches,
        batch_size=batch_size,
    )
    validation_count = _effective_count(
        validation_cache.count,
        max_batches=args.max_validation_batches,
        batch_size=batch_size,
    )
    full_data = (
        train_count == train_cache.count and validation_count == validation_cache.count
    )
    qualified_full_fit = (
        full_data and args.max_epochs is None and args.run_epoch_limit is None
    )
    time_scale = float(dataset_spec["train_time_scale"])
    censor_threshold = resolve_censor_threshold(args.dataset, dataset_spec)
    base_metrics = _base_metrics(
        base_cache=validation_base_cache,
        count=validation_count,
        time_scale=time_scale,
        censor_threshold=censor_threshold,
        batch_size=batch_size,
        observation_likelihood_mode=observation_mode,
    )
    if full_data:
        require(
            math.isclose(
                float(base_metrics["primary_proper_time_nll"]),
                float(dataset_spec["aligned_B_validation_primary_nll"]),
                rel_tol=0.0,
                abs_tol=1e-6,
            ),
            "Aligned-B primary NLL replay drift",
        )
        require(
            math.isclose(
                float(base_metrics["continuous_reference_time_nll"]),
                float(dataset_spec["aligned_B_validation_continuous_nll"]),
                rel_tol=0.0,
                abs_tol=1e-6,
            ),
            "Aligned-B continuous NLL replay drift",
        )

    time_head_contract = selected_payload["encoder_config"]["time_head"]
    sigma_floor = float(time_head_contract["time_sigma_floor"])
    require(
        sigma_floor == float(contract["adapter"]["sigma_floor"]),
        "Aligned-B sigma floor differs from adapter contract",
    )
    adapter_kwargs = {
        "log_duration_mean": float(observed_history_statistics["log1p_mean"]),
        "log_duration_std": float(observed_history_statistics["log1p_std"]),
        "sigma_floor": sigma_floor,
    }
    torch.manual_seed(int(settings["seed"]))
    candidate = CausalLogDurationAdapter(**adapter_kwargs)
    initial_candidate_state = _clone_state(candidate)
    length_only = LengthOnlyCausalLogDurationAdapter(**adapter_kwargs)
    length_only.load_state_dict(initial_candidate_state, strict=True)
    require(
        canonical_state_dict_sha256(candidate.state_dict())
        == canonical_state_dict_sha256(length_only.state_dict()),
        "Candidate and length-only initial states differ",
    )
    global_control = GlobalScaleControl(sigma_floor=sigma_floor)

    common_fit = dict(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        train_base_cache=train_base_cache,
        validation_base_cache=validation_base_cache,
        train_count=train_count,
        validation_count=validation_count,
        device=requested_device,
        time_scale=time_scale,
        censor_threshold=censor_threshold,
        observation_likelihood_contract=observation_contract,
        contract_sha256=sha256_file(args.contract),
        source_checkpoint_sha256=checkpoint_sha256,
        source_state_sha256=source_state_sha256,
        frozen_cache_sha256=train_cache.digest() + ":" + validation_cache.digest(),
        history_statistics=observed_history_statistics,
        source_revision=args.source_revision,
        settings=settings,
        run_epoch_limit=args.run_epoch_limit,
    )
    candidate_summary = fit_scale_module(
        role="candidate",
        module=candidate,
        output_dir=args.output_dir / "candidate",
        **common_fit,
    )
    if candidate_summary.get("status") != "success":
        return candidate_summary
    length_summary = fit_scale_module(
        role="length_only_control",
        module=length_only,
        output_dir=args.output_dir / "length_only_control",
        **common_fit,
    )
    if length_summary.get("status") != "success":
        return length_summary
    global_summary = fit_scale_module(
        role="global_scale_control",
        module=global_control,
        output_dir=args.output_dir / "global_scale_control",
        **common_fit,
    )
    if global_summary.get("status") != "success":
        return global_summary

    require(
        canonical_state_dict_sha256(source_model.state_dict()) == source_state_sha256,
        "Aligned-B state changed during adapter fitting",
    )
    require(
        all(parameter.grad is None for parameter in source_model.parameters()),
        "Aligned-B received gradients",
    )
    runtime_quantity_after = cached_quantity_predictions(
        model=source_model,
        cache=validation_cache,
        device=requested_device,
        batch_size=batch_size,
    )
    quantity_bitwise = torch.equal(runtime_quantity_after, runtime_quantity_before)
    require(quantity_bitwise, "Aligned-B quantity predictions changed")
    runtime_quantity_before_sha256 = tensor_sha256(
        "quantity_prediction", runtime_quantity_before
    )
    runtime_quantity_after_sha256 = tensor_sha256(
        "quantity_prediction", runtime_quantity_after
    )
    require(
        runtime_quantity_before_sha256 == runtime_quantity_after_sha256,
        "Same-device aligned-B quantity digest drift",
    )
    candidate_metrics = candidate_summary["selected_validation_metrics"]
    length_metrics = length_summary["selected_validation_metrics"]
    global_metrics = global_summary["selected_validation_metrics"]
    time_median_bitwise = all(
        metrics["time_median_sha256"] == base_metrics["time_median_sha256"]
        for metrics in (candidate_metrics, length_metrics, global_metrics)
    )
    require(time_median_bitwise, "Scale path changed aligned-B time median")
    base_location_bitwise = all(
        metrics["base_location_sha256"] == base_metrics["base_location_sha256"]
        for metrics in (candidate_metrics, length_metrics, global_metrics)
    )
    require(base_location_bitwise, "Scale path changed aligned-B location")
    epoch_zero_exact = all(
        summary["epoch_zero_validation_primary_nll"]
        == base_metrics["primary_proper_time_nll"]
        for summary in (candidate_summary, length_summary, global_summary)
    )
    require(epoch_zero_exact, "Epoch-zero paths do not exactly reproduce aligned-B")

    candidate_primary = float(candidate_metrics["primary_proper_time_nll"])
    candidate_continuous = float(
        candidate_metrics["continuous_reference_time_nll"]
    )
    baseline_primary = float(base_metrics["primary_proper_time_nll"])
    baseline_continuous = float(base_metrics["continuous_reference_time_nll"])
    A_primary = float(dataset_spec["aligned_A_validation_primary_nll"])
    observed_deltas = {
        "candidate_primary_improvement_over_aligned_B": (
            baseline_primary - candidate_primary
        ),
        "candidate_minus_aligned_A_primary_nll": candidate_primary - A_primary,
        "candidate_primary_improvement_over_global_control": (
            float(global_metrics["primary_proper_time_nll"]) - candidate_primary
        ),
        "candidate_primary_improvement_over_length_only_control": (
            float(length_metrics["primary_proper_time_nll"]) - candidate_primary
        ),
        "candidate_minus_aligned_B_continuous_nll": (
            candidate_continuous - baseline_continuous
        ),
    }
    gates: dict[str, bool | None] = {
        "candidate_meets_dataset_aligned_B_rule": None,
        "candidate_within_aligned_A_plus_0_01": None,
        "candidate_beats_global_control_by_0_005": None,
        "candidate_beats_length_only_control_by_0_005": None,
        "candidate_continuous_within_aligned_B_plus_0_01": None,
        "quantity_prediction_bitwise_identical": quantity_bitwise,
        "time_median_bitwise_identical": time_median_bitwise,
        "source_model_state_unchanged": True,
        "base_location_bitwise_identical": base_location_bitwise,
    }
    if qualified_full_fit:
        structured = contract["acceptance"]["structured_gates"]
        control_margin = float(
            structured["minimum_candidate_improvement_over_each_control_primary_nll"]
        )
        gates.update(
            candidate_meets_dataset_aligned_B_rule=(
                observed_deltas["candidate_primary_improvement_over_aligned_B"]
                >= float(dataset_spec["minimum_candidate_primary_improvement"])
            ),
            candidate_within_aligned_A_plus_0_01=(
                observed_deltas["candidate_minus_aligned_A_primary_nll"]
                <= float(structured["maximum_candidate_minus_aligned_A_primary_nll"])
            ),
            candidate_beats_global_control_by_0_005=(
                observed_deltas["candidate_primary_improvement_over_global_control"]
                >= control_margin
            ),
            candidate_beats_length_only_control_by_0_005=(
                observed_deltas[
                    "candidate_primary_improvement_over_length_only_control"
                ] >= control_margin
            ),
            candidate_continuous_within_aligned_B_plus_0_01=(
                observed_deltas["candidate_minus_aligned_B_continuous_nll"]
                <= float(
                    structured["maximum_candidate_minus_aligned_B_continuous_nll"]
                )
            ),
        )
    performance_values = [
        value for name, value in gates.items() if name.startswith("candidate_")
    ]
    acceptance_status = (
        "passed"
        if qualified_full_fit and all(value is True for value in performance_values)
        else "failed"
        if qualified_full_fit
        else "not_evaluated_execution_check"
    )
    q_metrics = quantity_metrics(
        runtime_quantity_after, validation_cache.target_quantity
    )
    q_metrics.update(
        stratified_quantity_metrics(
            runtime_quantity_after,
            validation_cache.target_quantity,
            body_max=float(dataset_spec["reporting_body_max_train_p95"]),
            tail_min_exclusive=float(
                dataset_spec["reporting_tail_min_exclusive_train_p99"]
            ),
            require_nonempty=full_data,
        )
    )
    if full_data:
        require(
            math.isclose(
                float(q_metrics["mae"]),
                float(dataset_spec["aligned_B_quantity_mae"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(q_metrics["rmse"]),
                float(dataset_spec["aligned_B_quantity_rmse"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "Aligned-B quantity metrics drift",
        )
    summary = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "contract_sha256": sha256_file(args.contract),
        "source_revision": args.source_revision,
        "status": "success",
        "dataset": args.dataset,
        "seed": int(settings["seed"]),
        "observation_likelihood_contract": observation_contract,
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_model_state_sha256": source_state_sha256,
        "source_model_state_unchanged": True,
        "train_feature_cache_sha256": train_cache.digest(),
        "validation_feature_cache_sha256": validation_cache.digest(),
        "train_target_population": train_population,
        "validation_target_population": validation_population,
        "train_history_statistics": observed_history_statistics,
        "train_target_exclusion_audit": train_target_exclusion_audit,
        "validation_target_exclusion_audit": validation_target_exclusion_audit,
        "train_count": train_count,
        "validation_count": validation_count,
        "qualified_full_data": full_data,
        "qualified_full_fit": qualified_full_fit,
        "base_aligned_B_validation_metrics": base_metrics,
        "candidate": candidate_summary,
        "length_only_control": length_summary,
        "global_scale_control": global_summary,
        "aligned_A_validation_primary_nll": A_primary,
        "quantity_prediction_bitwise_identical": quantity_bitwise,
        "runtime_quantity_before_sha256": runtime_quantity_before_sha256,
        "runtime_quantity_after_sha256": runtime_quantity_after_sha256,
        "cached_quantity_prediction_sha256": cached_quantity_sha256,
        "cached_vs_runtime_quantity_close": bool(cached_vs_runtime_close),
        "quantity_metrics": q_metrics,
        "time_median_bitwise_identical": time_median_bitwise,
        "base_location_bitwise_identical": base_location_bitwise,
        "epoch_zero_exact_aligned_B": epoch_zero_exact,
        "candidate_length_only_initial_state_identical": True,
        "acceptance_gates": gates,
        "acceptance_observed_deltas": observed_deltas,
        "acceptance_status": acceptance_status,
        "taxi_first_stop_triggered": (
            args.dataset == "yellow_trip_hourly"
            and qualified_full_fit
            and acceptance_status != "passed"
        ),
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "legacy_nll_compared": False,
        "runtime": {
            "requested_device": str(requested_device),
            "device_name": device_name,
            "cuda_available": torch.cuda.is_available(),
            "elapsed_seconds": time.perf_counter() - started_at,
            "peak_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(requested_device))
                if requested_device.type == "cuda"
                else 0
            ),
            "peak_memory_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(requested_device))
                if requested_device.type == "cuda"
                else 0
            ),
        },
    }
    save_json(args.output_dir / SUMMARY_NAME, summary)
    return summary


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
