"""Label-free, bounded geometry of a static value bank and cached selections.

analyze_bank(values[M,D], indices[N,K], weights[N,K], quantity_weight[D],
time_weight[D]) returns (JSON summary, per-slot CPU tensors). Calculations use
float64. Weight sums within 1e-6 of one are normalized to remove storage
roundoff; the correction is disclosed. No parameters are fitted or modified.
"""

from __future__ import annotations

import math

import torch


WEIGHT_SUM_ATOL = 1e-6
MAX_TARGETS = 8192


def _float_cpu(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite floating values")
    return value.detach().to(torch.float64)


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


def _projection_stats(projection: torch.Tensor, mass: torch.Tensor) -> dict:
    anchor = projection[torch.where(mass > 0)[0][0]]
    mean = anchor + (mass * (projection - anchor)).sum()
    variance = (mass * (projection - mean).square()).sum()
    used = projection[mass > 0]
    return {
        "mean": float(mean), "std": float(variance.sqrt()),
        "variance": float(variance), "min": float(used.min()),
        "max": float(used.max()), "range": float(used.max() - used.min()),
    }


def _bank_distribution(
    values: torch.Tensor, mass: torch.Tensor, heads: dict[str, torch.Tensor]
) -> dict:
    anchor = values[torch.where(mass > 0)[0][0]]
    centroid = anchor + mass @ (values - anchor)
    centered = values - centroid
    weighted_centered = mass.sqrt()[:, None] * centered
    spectrum = torch.linalg.svdvals(weighted_centered).square()
    trace = (mass[:, None] * centered.square()).sum()
    if not torch.isfinite(spectrum).all() or not torch.isfinite(trace):
        raise FloatingPointError("nonfinite covariance geometry")
    if spectrum.sum() > 0:
        spectral_mass = spectrum[spectrum > 0] / spectrum.sum()
        effective_rank = float(torch.exp(-(spectral_mass * spectral_mass.log()).sum()))
    else:
        effective_rank = 0.0
    axes = {}
    for name, head in heads.items():
        norm = float(torch.linalg.vector_norm(head))
        raw = _projection_stats(values @ head, mass)
        normalized = _projection_stats(values @ (head / norm), mass) if norm > 0 else None
        axes[name] = {
            "raw": raw, "unit_head": normalized,
            "unit_axis_variance_over_vector_trace": (
                _ratio(normalized["variance"], float(trace)) if normalized else None
            ),
        }
    return {
        "positive_mass_slots": int((mass > 0).sum()),
        "mean_vector_norm": float(torch.linalg.vector_norm(centroid)),
        "mean_slot_norm": float((mass * torch.linalg.vector_norm(values, dim=1)).sum()),
        "mean_squared_slot_norm": float((mass[:, None] * values.square()).sum()),
        "covariance_trace": float(trace),
        "centered_rms_norm": float(trace.sqrt()),
        "covariance_effective_rank": effective_rank,
        "axes": axes,
    }


def _decomposition(total: torch.Tensor, between: torch.Tensor, within: torch.Tensor) -> dict:
    return {
        "total": float(total), "between_query": float(between),
        "mean_within_query": float(within),
        "closure_error": float(total - between - within),
        "between_over_total": _ratio(float(between), float(total)),
        "within_over_total": _ratio(float(within), float(total)),
    }


def analyze_bank(
    values: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor,
    quantity_weight: torch.Tensor, time_weight: torch.Tensor,
) -> tuple[dict, dict[str, torch.Tensor]]:
    """Describe bank support and exact selected-mixture variance identities.

    Active union means membership in cached top-k, including any zero-weight
    positions. Selected mass uses actual normalized aggregation weights.
    Unused slots are unused only in this sample, not proven untrained slots.
    Per-slot selection_count counts occurrences; repeated indices are allowed.
    """
    values = _float_cpu(values, "values")
    weights = _float_cpu(weights, "weights")
    heads = {"quantity": _float_cpu(quantity_weight, "quantity_weight"),
             "time": _float_cpu(time_weight, "time_weight")}
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("values must have nonempty shape [M,D]")
    m, d = values.shape
    if not isinstance(indices, torch.Tensor) or indices.device.type != "cpu":
        raise ValueError("indices must be a CPU tensor")
    if indices.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("indices must have an integer dtype")
    indices = indices.detach().to(torch.int64)
    if indices.ndim != 2 or not 1 <= indices.shape[0] <= MAX_TARGETS or indices.shape[1] < 1:
        raise ValueError(f"indices must have shape [N,K], 1 <= N <= {MAX_TARGETS}, K >= 1")
    if weights.shape != indices.shape or any(head.shape != (d,) for head in heads.values()):
        raise ValueError("weights must match indices and heads must have shape [D]")
    if (indices < 0).any() or (indices >= m).any():
        raise ValueError("indices outside bank bounds")
    if (weights < 0).any():
        raise ValueError("weights must be nonnegative")
    row_sums = weights.sum(dim=1, keepdim=True)
    sum_error = float((row_sums - 1).abs().max())
    if sum_error > WEIGHT_SUM_ATOL:
        raise ValueError("weights must sum to one in every row")
    weights = weights / row_sums
    n, k = indices.shape
    counts = torch.bincount(indices.flatten(), minlength=m)
    mass = torch.zeros(m, dtype=torch.float64).scatter_add_(
        0, indices.flatten(), weights.flatten()
    ) / n
    active = counts > 0
    heads_norm = {name: float(torch.linalg.vector_norm(head)) for name, head in heads.items()}
    projections = {name: values @ head for name, head in heads.items()}
    if not all(torch.isfinite(value).all() for value in projections.values()):
        raise FloatingPointError("nonfinite head projection")
    if not all(math.isfinite(norm) for norm in heads_norm.values()):
        raise FloatingPointError("nonfinite head norm")
    distributions = {
        "full_uniform": torch.full((m,), 1.0 / m, dtype=torch.float64),
        "active_union_uniform": active.double() / active.sum(),
        "selected_mass": mass,
    }
    description = {name: _bank_distribution(values, probability, heads)
                   for name, probability in distributions.items()}
    selected = values[indices]
    query_anchor = selected[:, 0]
    query_mean = query_anchor + (weights[:, :, None] * (selected - query_anchor[:, None])).sum(dim=1)
    selected_anchor = values[torch.where(mass > 0)[0][0]]
    selected_mean = selected_anchor + mass @ (values - selected_anchor)
    vector_total = (mass[:, None] * (values - selected_mean).square()).sum()
    vector_between = (query_mean - selected_mean).square().sum(dim=1).mean()
    vector_within = (weights * (selected - query_mean[:, None]).square().sum(dim=2)).sum(dim=1).mean()
    decomposition = {"vector_trace": _decomposition(vector_total, vector_between, vector_within)}
    averaging = {}
    for name, projection in projections.items():
        selected_projection = projection[indices]
        query_anchor_projection = selected_projection[:, 0]
        query_projection = query_anchor_projection + (
            weights * (selected_projection - query_anchor_projection[:, None])
        ).sum(dim=1)
        anchor = projection[torch.where(mass > 0)[0][0]]
        center = anchor + mass @ (projection - anchor)
        total = (mass * (projection - center).square()).sum()
        between = (query_projection - center).square().mean()
        within = (weights * (selected_projection - query_projection[:, None]).square()).sum(dim=1).mean()
        decomposition[name] = _decomposition(total, between, within)
        weighted_minus_uniform = query_projection - selected_projection.mean(dim=1)
        denominator = (weights * (selected_projection - center).abs()).sum(dim=1)
        defined = denominator > 0
        cancellation = 1 - (query_projection[defined] - center).abs() / denominator[defined]
        averaging[name] = {
            "weighted_minus_uniform_mean": float(weighted_minus_uniform.mean()),
            "weighted_minus_uniform_std": float(weighted_minus_uniform.std(unbiased=False)),
            "weighted_minus_uniform_rms": float(weighted_minus_uniform.square().mean().sqrt()),
            "cancellation_about_selected_mean_defined_queries": int(defined.sum()),
            "cancellation_about_selected_mean_average": float(cancellation.mean()) if defined.any() else None,
        }
    norm_product = heads_norm["quantity"] * heads_norm["time"]
    summary = {
        "protocol": "hard_lmm_bank_geometry_v1", "n_targets": n, "n_slots": m,
        "dimension": d, "topk": k, "input_weight_sum_max_error": sum_error,
        "weight_roundoff_policy": "row-normalize accepted sums within absolute 1e-6",
        "head_norms": heads_norm,
        "head_direction_cosine": float(heads["quantity"] @ heads["time"]) / norm_product if norm_product > 0 else None,
        "active_slots": torch.where(active)[0].tolist(),
        "unused_slots": torch.where(~active)[0].tolist(),
        "selected_mass_sum": float(mass.sum()),
        "distributions": description, "selected_variance_decomposition": decomposition,
        "averaging": averaging,
        "mean_effective_selected_count": float((1 / weights.square().sum(dim=1)).mean()),
        "mean_weight_entropy": float(-(weights * weights.clamp_min(torch.finfo(torch.float64).tiny).log()).sum(dim=1).mean()),
        "scope": "label-free sample geometry; slot utility and training-wide usage are unknown",
    }
    per_slot = {"selection_count": counts, "selection_mass": mass, "selected": active,
                **{f"{name}_projection": projection for name, projection in projections.items()}}
    def check_finite_scalars(item):
        if isinstance(item, dict):
            for value in item.values():
                check_finite_scalars(value)
        elif isinstance(item, (list, tuple)):
            for value in item:
                check_finite_scalars(value)
        elif isinstance(item, float) and not math.isfinite(item):
            raise FloatingPointError("nonfinite geometry summary")
    check_finite_scalars(summary)
    return summary, per_slot
