"""Fixed train-internal diagnostic; this does not fit or evaluate a backbone.

``analyze_cache(cache)`` returns a small JSON-serializable summary. With
``return_predictions=True`` it returns ``(summary, predictions)``; prediction
tensors follow input row order and may be saved separately for an audit.

Required CPU tensors: h [N,D], stats [N,2], log_residual [N], series_index [N],
fold [N] (0/1), target_index [N]. The caller supplies at most 8192 fixed paired
train targets and the series-disjoint split. ``stats`` must contain only the
observed-history mean log1p quantity and latest log1p quantity minus that mean.
``log_residual`` is target log1p quantity minus the frozen model prediction in
log1p space. This module cannot independently certify the caller's causality.

Each held fold uses 64 cosine-nearest h directions from the opposite fold.
The h-only correction is their residual mean. The augmented correction fits
two centered statistics by ridge (summed squared error + 1 * squared slopes),
with an unpenalized local intercept. Statistics use opposite-fold population
standard deviations; zero-variance dimensions use scale one. A constant
opposite-fold residual mean supplies the scalar control. There is no tuning.

The bootstrap describes series-cluster variability conditional on the frozen
model and sampled training targets. The backbone already saw training labels:
helper cross-fitting and these bands do NOT establish held-out generalization.
"""

from __future__ import annotations

import numpy as np
import torch


PROTOCOL = "hard_lmm_two_stat_query_train_diagnostic_v1"
MAX_TARGETS = 8192
NEIGHBORS = 64
MIN_SERIES_PER_FOLD = 10
RIDGE = 1.0
BOOTSTRAP_REPLICATES = 500
BOOTSTRAP_SEED = 20260905
QUERY_CHUNK_SIZE = 128
_FLOAT_FIELDS = ("h", "stats", "log_residual")
_INTEGER_FIELDS = ("series_index", "fold", "target_index")
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _validated_cache(cache: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name in _FLOAT_FIELDS + _INTEGER_FIELDS:
        value = cache.get(name)
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU tensor")
        if name in _INTEGER_FIELDS and value.dtype not in _INTEGER_DTYPES:
            raise ValueError(f"{name} must have an integer dtype")
        if name in _FLOAT_FIELDS and not value.is_floating_point():
            raise ValueError(f"{name} must have a floating dtype")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains nonfinite values")
        result[name] = value.detach().to(
            dtype=torch.float64 if name in _FLOAT_FIELDS else torch.int64
        )
    h = result["h"]
    if h.ndim != 2 or h.shape[1] < 1 or not 1 <= h.shape[0] <= MAX_TARGETS:
        raise ValueError(f"h must have shape [N,D], 1 <= N <= {MAX_TARGETS}, D >= 1")
    n = h.shape[0]
    if result["stats"].shape != (n, 2):
        raise ValueError("stats must have shape [N,2]")
    for name in ("log_residual",) + _INTEGER_FIELDS:
        if result[name].shape != (n,):
            raise ValueError(f"{name} must have shape [N]")
    if set(result["fold"].tolist()) != {0, 1}:
        raise ValueError("fold must contain both 0 and 1 only")
    groups = []
    for fold in (0, 1):
        selected = result["fold"] == fold
        if selected.sum().item() < NEIGHBORS:
            raise ValueError(f"fold {fold} needs at least {NEIGHBORS} targets")
        unique = set(result["series_index"][selected].tolist())
        if len(unique) < MIN_SERIES_PER_FOLD:
            raise ValueError(f"fold {fold} needs at least {MIN_SERIES_PER_FOLD} independent series")
        groups.append(unique)
    if groups[0] & groups[1]:
        raise ValueError("series appear in both folds; group separation is required")
    pairs = torch.stack((result["series_index"], result["target_index"]), dim=1)
    if torch.unique(pairs, dim=0).shape[0] != n:
        raise ValueError("duplicate (series_index, target_index) rows")
    return result


def _relative_gain(reference: float, candidate: float) -> float | None:
    if reference == 0.0:
        return 0.0 if candidate == 0.0 else None
    return (reference - candidate) / reference


def _distribution(values: torch.Tensor) -> dict[str, float]:
    quantiles = torch.quantile(values.flatten(), torch.tensor(
        [0.0, 0.05, 0.5, 0.95, 1.0], dtype=torch.float64
    ))
    return {
        "mean": float(values.mean()),
        **{name: float(value) for name, value in zip(
            ("min", "p05", "median", "p95", "max"), quantiles
        )},
    }


def _metrics(errors: dict[str, torch.Tensor], selected: torch.Tensor) -> dict:
    mse = {name: float(error[selected].mean()) for name, error in errors.items()}
    return {
        "n_targets": int(selected.sum()),
        "mse": mse,
        "two_stat_relative_improvement_vs_h_only": _relative_gain(
            mse["h_only"], mse["two_stat"]
        ),
        "two_stat_relative_improvement_vs_constant": _relative_gain(
            mse["constant"], mse["two_stat"]
        ),
        "paired_delta_h_only_minus_two_stat": mse["h_only"] - mse["two_stat"],
        "h_only_mse_is_zero": mse["h_only"] == 0.0,
    }


def _cluster_band(delta: torch.Tensor, series: torch.Tensor) -> dict:
    group_ids, inverse = np.unique(series.numpy(), return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    sums = np.bincount(inverse, weights=delta.numpy()).astype(np.float64)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = generator.integers(0, len(group_ids), size=len(group_ids))
        draws[replicate] = sums[sampled].sum() / counts[sampled].sum()
    return {
        "unit": "series",
        "n_series": int(len(group_ids)),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "paired_delta_lower_p05": float(np.quantile(draws, 0.05)),
        "paired_delta_median": float(np.quantile(draws, 0.5)),
        "paired_delta_upper_p95": float(np.quantile(draws, 0.95)),
        "interpretation": "descriptive train-internal conditional variability; not held-out inference",
    }


def analyze_cache(
    cache: dict[str, torch.Tensor], *, return_predictions: bool = False
) -> dict | tuple[dict, dict[str, torch.Tensor]]:
    """Apply the fixed protocol without selecting thresholds or model candidates.

    The optional prediction dictionary contains corrections, squared errors,
    mean neighbor cosine, fold, series_index and target_index in original row
    order. Neither inputs nor global random state are modified.
    """
    data = _validated_cache(cache)
    h, stats, residual = (data[name] for name in _FLOAT_FIELDS)
    n = h.shape[0]
    norms = torch.linalg.vector_norm(h, dim=1, keepdim=True)
    if not torch.isfinite(norms).all():
        raise FloatingPointError("nonfinite hidden-vector norm")
    normalized_h = h / norms.clamp_min(1e-12)
    corrections = {
        name: torch.empty(n, dtype=torch.float64)
        for name in ("constant", "h_only", "two_stat")
    }
    neighbor_cosines = torch.empty((n, NEIGHBORS), dtype=torch.float64)
    fold_fit_details = {}
    with torch.no_grad():
        for held_fold in (0, 1):
            query_ids = torch.where(data["fold"] == held_fold)[0]
            reference_ids = torch.where(data["fold"] != held_fold)[0]
            # Stable pair ordering makes exact cosine ties independent of cache order.
            ordering = np.lexsort((
                data["target_index"][reference_ids].numpy(),
                data["series_index"][reference_ids].numpy(),
            ))
            reference_ids = reference_ids[torch.from_numpy(ordering.copy())]
            reference_stats = stats[reference_ids]
            mean = reference_stats.mean(dim=0)
            raw_scale = reference_stats.std(dim=0, unbiased=False)
            if not torch.isfinite(mean).all() or not torch.isfinite(raw_scale).all():
                raise FloatingPointError("nonfinite reference statistic moments")
            zero_variance = raw_scale == 0.0
            scale = torch.where(zero_variance, torch.ones_like(raw_scale), raw_scale)
            standardized_stats = (stats - mean) / scale
            reference_y = residual[reference_ids]
            corrections["constant"][query_ids] = reference_y.mean()
            max_condition = 1.0
            for start in range(0, len(query_ids), QUERY_CHUNK_SIZE):
                current = query_ids[start:start + QUERY_CHUNK_SIZE]
                similarities = normalized_h[current] @ normalized_h[reference_ids].T
                positions = torch.argsort(
                    similarities, dim=1, descending=True, stable=True
                )[:, :NEIGHBORS]
                selected_ids = reference_ids[positions]
                neighbor_cosines[current] = torch.gather(similarities, 1, positions)
                local_stats = standardized_stats[selected_ids]
                local_y = residual[selected_ids]
                local_mean = local_stats.mean(dim=1, keepdim=True)
                local_y_mean = local_y.mean(dim=1)
                centered_stats = local_stats - local_mean
                centered_y = local_y - local_y_mean[:, None]
                gram = centered_stats.transpose(1, 2) @ centered_stats
                gram = gram + RIDGE * torch.eye(2, dtype=torch.float64)[None]
                rhs = (centered_stats.transpose(1, 2) @ centered_y[:, :, None])
                coefficients = torch.linalg.solve(gram, rhs).squeeze(-1)
                query_difference = standardized_stats[current] - local_mean[:, 0]
                corrections["h_only"][current] = local_y_mean
                corrections["two_stat"][current] = local_y_mean + (
                    query_difference * coefficients
                ).sum(dim=1)
                eigenvalues = torch.linalg.eigvalsh(gram)
                max_condition = max(max_condition, float(
                    (eigenvalues[:, -1] / eigenvalues[:, 0]).max()
                ))
            fold_fit_details[str(held_fold)] = {
                "reference_n_targets": int(len(reference_ids)),
                "reference_n_series": int(data["series_index"][reference_ids].unique().numel()),
                "held_n_series": int(data["series_index"][query_ids].unique().numel()),
                "reference_stat_mean": mean.tolist(),
                "reference_stat_raw_std": raw_scale.tolist(),
                "zero_variance_stat_dimensions": torch.where(zero_variance)[0].tolist(),
                "max_regularized_local_condition_number": max_condition,
            }
    errors = {name: (residual - value).square() for name, value in corrections.items()}
    if not all(torch.isfinite(value).all() for value in (
        *corrections.values(), *errors.values(), neighbor_cosines
    )):
        raise FloatingPointError("nonfinite diagnostic result")
    mean_cosine = neighbor_cosines.mean(dim=1)
    summary = {
        "protocol": PROTOCOL,
        "sample_limit": MAX_TARGETS,
        "neighbors": NEIGHBORS,
        "min_series_per_fold": MIN_SERIES_PER_FOLD,
        "ridge_summed_sse_penalty": RIDGE,
        "stat_standardization": "opposite-fold population mean/std; zero std uses one",
        "group_separation_verified": True,
        "n_series": int(data["series_index"].unique().numel()),
        "zero_norm_h_count": int((norms[:, 0] <= 1e-12).sum()),
        "pooled": _metrics(errors, torch.ones(n, dtype=torch.bool)),
        "folds": {
            str(fold): {
                **_metrics(errors, data["fold"] == fold),
                **fold_fit_details[str(fold)],
            } for fold in (0, 1)
        },
        "neighbor_cosine": {
            "all_selected": _distribution(neighbor_cosines),
            "mean_per_target": _distribution(mean_cosine),
            "targets_with_mean_at_least_0_9": int((mean_cosine >= 0.9).sum()),
            "fraction_with_mean_at_least_0_9": float((mean_cosine >= 0.9).double().mean()),
        },
        "cluster_bootstrap": _cluster_band(
            errors["h_only"] - errors["two_stat"], data["series_index"]
        ),
        "scope": "train-internal conditional decodability; no backbone training or predictive claim",
    }
    if not return_predictions:
        return summary
    predictions = {
        **{f"{name}_correction": value for name, value in corrections.items()},
        **{f"{name}_squared_error": value for name, value in errors.items()},
        "mean_neighbor_cosine": mean_cosine,
        **{name: data[name].clone() for name in _INTEGER_FIELDS},
    }
    return summary, predictions
