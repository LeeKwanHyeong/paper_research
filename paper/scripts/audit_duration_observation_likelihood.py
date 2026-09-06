#!/usr/bin/env python3
"""Audit continuous and integer-observation log-normal duration likelihoods.

The matched duration experiment fitted continuous log-normal densities to
durations recorded on integer grids.  This read-only utility replays those
selected heads from their frozen validation caches, evaluates the probability
of each centered integer observation bin, and exposes the terms that dominate
the continuous-density NLL.  It never trains, selects a checkpoint, or opens a
held-out split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from simple_lab_test.search.common.runner import (
    canonical_state_dict_sha256,
    torch_load_checkpoint,
)


CONTRACT_ID = "duration_observation_likelihood_audit_v1"
DEFAULT_CONTRACT = (
    PROJECT_ROOT
    / "paper/contracts/duration_observation_likelihood_audit_v1.json"
)
LOG_TWO = math.log(2.0)
LOG_TWO_PI_HALF = 0.5 * math.log(2.0 * math.pi)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def stable_log_difference(
    log_larger: torch.Tensor,
    log_smaller: torch.Tensor,
) -> torch.Tensor:
    """Return ``log(exp(log_larger) - exp(log_smaller))`` stably.

    Both inputs are already in log space.  The two branches are the standard
    stable ``log1mexp`` construction and avoid materializing either
    probability in ordinary space.
    """
    require(log_larger.shape == log_smaller.shape, "Log shapes differ")
    require(
        log_larger.dtype == torch.float64
        and log_smaller.dtype == torch.float64,
        "Stable log difference requires float64 inputs",
    )
    require(
        bool(torch.isfinite(log_larger).all())
        and bool(torch.isfinite(log_smaller).all()),
        "Stable log difference received a non-finite input",
    )
    delta = log_smaller - log_larger
    require(
        bool((delta <= 1e-14).all()),
        "Subtracted log probability exceeds the minuend",
    )
    delta = delta.clamp_max(0.0)
    log_complement = torch.where(
        delta < -LOG_TWO,
        torch.log1p(-torch.exp(delta)),
        torch.log(-torch.expm1(delta)),
    )
    result = log_larger + log_complement
    require(
        bool(torch.isfinite(result).all()),
        "Integer-bin probability is zero at float64 resolution",
    )
    return result


def centered_integer_bin_bounds(
    target_dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the centered width-one bin for a positive integer duration."""
    target = target_dt.to(dtype=torch.float64)
    require(target.ndim == 1, "Duration targets must be rank one")
    require(
        bool(torch.isfinite(target).all()) and bool((target >= 1.0).all()),
        "Integer duration targets must be finite and at least one",
    )
    require(
        bool(torch.isclose(target, torch.round(target), atol=1e-8, rtol=0.0).all()),
        "Duration targets are not on the declared integer grid",
    )
    lower = torch.maximum(
        target - 0.5,
        torch.full_like(target, 0.5),
    )
    upper = target + 0.5
    return lower, upper


def _standardized_log_boundary(
    boundary: torch.Tensor,
    location: torch.Tensor,
    scale: torch.Tensor,
    *,
    time_scale: float,
) -> torch.Tensor:
    require(boundary.shape == location.shape == scale.shape, "Boundary shape drift")
    require(
        math.isfinite(time_scale) and time_scale > 0.0,
        "time_scale must be finite and positive",
    )
    require(
        bool(torch.isfinite(location).all())
        and bool(torch.isfinite(scale).all())
        and bool((scale > 0.0).all())
        and bool((boundary > 0.0).all()),
        "Invalid log-normal parameters or boundary",
    )
    return (
        torch.log(boundary)
        - math.log(float(time_scale))
        - location
    ) / scale


def centered_integer_log_mass(
    *,
    target_dt: torch.Tensor,
    location: torch.Tensor,
    scale: torch.Tensor,
    time_scale: float,
    top_code: float | None = None,
) -> torch.Tensor:
    """Return log probability of each recorded integer duration.

    For bins whose lower standardized boundary is positive, subtraction is
    performed as ``log(S(lower) - S(upper))``.  Otherwise it is performed as
    ``log(F(upper) - F(lower))``.  A top-coded observation uses survival from
    the lower edge of its bin, e.g. Instacart 30 means ``S(29.5)``.
    """
    target = target_dt.to(dtype=torch.float64)
    location = location.to(dtype=torch.float64)
    scale = scale.to(dtype=torch.float64)
    require(target.shape == location.shape == scale.shape, "Input shape drift")
    lower, upper = centered_integer_bin_bounds(target)
    lower_z = _standardized_log_boundary(
        lower, location, scale, time_scale=time_scale
    )
    upper_z = _standardized_log_boundary(
        upper, location, scale, time_scale=time_scale
    )

    result = torch.empty_like(target)
    survival_branch = lower_z > 0.0
    if bool(survival_branch.any()):
        log_survival_lower = torch.special.log_ndtr(
            -lower_z[survival_branch]
        )
        log_survival_upper = torch.special.log_ndtr(
            -upper_z[survival_branch]
        )
        result[survival_branch] = stable_log_difference(
            log_survival_lower, log_survival_upper
        )
    cdf_branch = ~survival_branch
    if bool(cdf_branch.any()):
        log_cdf_lower = torch.special.log_ndtr(lower_z[cdf_branch])
        log_cdf_upper = torch.special.log_ndtr(upper_z[cdf_branch])
        result[cdf_branch] = stable_log_difference(
            log_cdf_upper, log_cdf_lower
        )

    if top_code is not None:
        require(
            math.isfinite(top_code)
            and top_code >= 1.0
            and float(top_code).is_integer(),
            "top_code must be a positive integer",
        )
        top_mask = target == float(top_code)
        if bool(top_mask.any()):
            top_boundary = torch.full_like(target[top_mask], top_code - 0.5)
            top_z = _standardized_log_boundary(
                top_boundary,
                location[top_mask],
                scale[top_mask],
                time_scale=time_scale,
            )
            result = result.clone()
            result[top_mask] = torch.special.log_ndtr(-top_z)

    require(bool(torch.isfinite(result).all()), "Non-finite integer log mass")
    require(bool((result <= 1e-14).all()), "An integer-bin probability exceeds one")
    return result


def continuous_lognormal_nll_components(
    *,
    target_dt: torch.Tensor,
    location: torch.Tensor,
    scale: torch.Tensor,
    time_scale: float,
) -> dict[str, torch.Tensor]:
    """Decompose the continuous log-normal density NLL per observation."""
    target = target_dt.to(dtype=torch.float64)
    location = location.to(dtype=torch.float64)
    scale = scale.to(dtype=torch.float64)
    require(target.shape == location.shape == scale.shape, "Input shape drift")
    require(
        bool(torch.isfinite(target).all()) and bool((target > 0.0).all()),
        "Continuous duration targets must be finite and positive",
    )
    standardized = (
        torch.log(target)
        - math.log(float(time_scale))
        - location
    ) / scale
    components = {
        "quadratic": 0.5 * torch.square(standardized),
        "log_sigma": torch.log(scale),
        "jacobian": torch.log(target),
        "constant": torch.full_like(target, LOG_TWO_PI_HALF),
    }
    components["total"] = sum(components.values())
    for name, value in components.items():
        require(bool(torch.isfinite(value).all()), f"Non-finite NLL component: {name}")
    return components


def continuous_observation_nll(
    *,
    target_dt: torch.Tensor,
    location: torch.Tensor,
    scale: torch.Tensor,
    time_scale: float,
    top_code: float | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Replay the matched experiment's density/top-code likelihood."""
    components = continuous_lognormal_nll_components(
        target_dt=target_dt,
        location=location,
        scale=scale,
        time_scale=time_scale,
    )
    nll = components["total"].clone()
    top_mask = torch.zeros_like(target_dt, dtype=torch.bool)
    if top_code is not None:
        top_mask = target_dt.to(dtype=torch.float64) == float(top_code)
        if bool(top_mask.any()):
            z = _standardized_log_boundary(
                target_dt.to(dtype=torch.float64)[top_mask],
                location.to(dtype=torch.float64)[top_mask],
                scale.to(dtype=torch.float64)[top_mask],
                time_scale=time_scale,
            )
            nll[top_mask] = -torch.special.log_ndtr(-z)
    require(bool(torch.isfinite(nll).all()), "Non-finite continuous observation NLL")
    return nll, components, top_mask


def empirical_integer_mode(target_dt: torch.Tensor) -> float:
    target = target_dt.to(dtype=torch.float64)
    centered_integer_bin_bounds(target)
    values, counts = torch.unique(target, sorted=True, return_counts=True)
    maximum = counts.max()
    return float(values[counts == maximum][0].item())


def _mean_or_none(value: torch.Tensor) -> float | None:
    return float(value.mean().item()) if value.numel() else None


def summarize_duration_likelihoods(
    *,
    target_dt: torch.Tensor,
    location: torch.Tensor,
    scale: torch.Tensor,
    time_scale: float,
    top_code: float | None,
) -> dict[str, Any]:
    """Summarize all observations and the empirical-mode partition."""
    target = target_dt.to(dtype=torch.float64)
    mode = empirical_integer_mode(target)
    integer_nll = -centered_integer_log_mass(
        target_dt=target,
        location=location,
        scale=scale,
        time_scale=time_scale,
        top_code=top_code,
    )
    continuous_nll, components, top_mask = continuous_observation_nll(
        target_dt=target,
        location=location,
        scale=scale,
        time_scale=time_scale,
        top_code=top_code,
    )

    partitions = {
        "all": torch.ones_like(target, dtype=torch.bool),
        "dt_equals_mode": target == mode,
        "dt_other": target != mode,
    }
    groups: dict[str, Any] = {}
    for name, mask in partitions.items():
        density_mask = mask & ~top_mask
        censored_mask = mask & top_mask
        component_means = {
            key: _mean_or_none(value[density_mask])
            for key, value in components.items()
        }
        groups[name] = {
            "count": int(mask.sum().item()),
            "integer_observation_nll": _mean_or_none(integer_nll[mask]),
            "continuous_observation_nll": _mean_or_none(continuous_nll[mask]),
            "integer_minus_continuous_nll": _mean_or_none(
                (integer_nll - continuous_nll)[mask]
            ),
            "continuous_density": {
                "count": int(density_mask.sum().item()),
                "component_means": component_means,
            },
            "continuous_top_code_survival": {
                "count": int(censored_mask.sum().item()),
                "nll": _mean_or_none(continuous_nll[censored_mask]),
            },
        }
    return {
        "count": int(target.numel()),
        "duration_mode": mode,
        "time_scale": float(time_scale),
        "top_code": top_code,
        "top_code_integer_observation_lower_bound": (
            None if top_code is None else float(top_code) - 0.5
        ),
        "groups": groups,
    }


def validate_contract(contract: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    require(contract.get("contract_id") == CONTRACT_ID, "Wrong audit contract")
    scope = contract.get("scope")
    require(isinstance(scope, Mapping), "Audit scope is missing")
    require(scope.get("evaluation_split") == "validation", "Scope is not validation")
    require(scope.get("held_out_test") is False, "Held-out access is enabled")
    require(scope.get("retraining") is False, "Retraining is enabled")
    require(scope.get("checkpoint_selection") is False, "Checkpoint selection is enabled")
    rows = contract.get("datasets")
    require(isinstance(rows, list) and len(rows) == 3, "Dataset scope drift")
    datasets = {str(row["dataset"]): row for row in rows}
    require(list(datasets) == scope.get("datasets"), "Dataset order drift")
    for dataset, row in datasets.items():
        require(row.get("observation_resolution") == 1, f"Resolution drift: {dataset}")
        require(row.get("integer_grid") is True, f"Integer-grid drift: {dataset}")
        expected_top_code = 30.0 if dataset == "insta_market_basket" else None
        require(row.get("top_code") == expected_top_code, f"Top-code drift: {dataset}")
    numerical = contract.get("numerical_contract")
    require(isinstance(numerical, Mapping), "Numerical contract is missing")
    require(numerical.get("calculation_dtype") == "float64", "Audit dtype drift")
    require(
        numerical.get("direct_cdf_subtraction") is False,
        "Direct CDF subtraction is enabled",
    )
    require(
        numerical.get("positive_lower_z_branch") == "log-survival difference",
        "Positive-tail branch drift",
    )
    references = contract.get("verified_reference_values")
    require(isinstance(references, Mapping), "Verified references are missing")
    expected_reference_keys = {
        f"{dataset}/{role}"
        for dataset in scope["datasets"]
        for role in scope["model_roles"]
        if not (dataset == "intermittent_frozen_5000" and role == "rmtpp")
    }
    require(
        set(references) == expected_reference_keys,
        "Auditable checkpoint/reference row set drift",
    )
    for key, reference in references.items():
        require(isinstance(reference, Mapping), f"Invalid reference row: {key}")
        value = reference.get("continuous_validation_nll")
        require(
            isinstance(value, (int, float)) and math.isfinite(float(value)),
            f"Continuous reference missing: {key}",
        )
    return datasets


def _payload_cache_digest(payload: Mapping[str, Any]) -> str:
    tensors = {
        name: value
        for name in (
            "time_hidden",
            "target_dt",
            "quantity_hidden",
            "target_quantity",
            "source_quantity_prediction",
        )
        if isinstance((value := payload.get(name)), torch.Tensor)
    }
    require("time_hidden" in tensors and "target_dt" in tensors, "Cache tensors missing")
    return canonical_state_dict_sha256(tensors)


def _conditional_parameters(
    *,
    hidden: torch.Tensor,
    state: Mapping[str, torch.Tensor],
    sigma_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    required = ("v_t.weight", "b_t", "w_raw", "time_scale_weight.weight")
    require(all(name in state for name in required), "Duration-head state is incomplete")
    hidden_64 = hidden.to(dtype=torch.float64)
    location = F.linear(
        hidden_64,
        state["v_t.weight"].to(dtype=torch.float64),
    ).squeeze(-1) + state["b_t"].to(dtype=torch.float64)
    raw_scale = F.linear(
        hidden_64,
        state["time_scale_weight.weight"].to(dtype=torch.float64),
    ).squeeze(-1) + state["w_raw"].to(dtype=torch.float64)
    scale = float(sigma_floor) + F.softplus(raw_scale)
    return location, scale


def _validate_checkpoint_role(
    checkpoint: Mapping[str, Any],
    *,
    model_role: str,
    resume_identity: Mapping[str, Any],
) -> None:
    """Bind the caller's role label to immutable checkpoint provenance.

    Frozen-B predates the matched comparison's explicit ``model_role`` field,
    so its originating contract is the authoritative role marker.  The
    matched A/RMTPP/THP checkpoints carry both their common source contract and
    an explicit role.  Accepting a missing marker for those rows would allow a
    valid checkpoint to be reported under the wrong model label.
    """
    checkpoint_role = checkpoint.get("model_role")
    source_contract = resume_identity.get("contract_id")
    require(
        resume_identity.get("planned_epochs") == 100,
        "Checkpoint is not from the full 100-epoch matched run",
    )
    if model_role == "B":
        require(
            source_contract == "hard_lmm_frozen_lognormal_duration_v1",
            "B checkpoint did not originate from the frozen-B contract",
        )
        require(
            checkpoint_role in (None, "B"),
            "B checkpoint carries a conflicting model role",
        )
        return

    require(
        model_role in {"A", "rmtpp", "thp"},
        f"Unsupported matched model role: {model_role}",
    )
    require(
        source_contract == "matched_frozen_lognormal_duration_v1",
        f"{model_role} checkpoint did not originate from the matched contract",
    )
    require(
        checkpoint_role == model_role,
        "Checkpoint model-role drift",
    )


def audit_checkpoint_cache(
    *,
    dataset: str,
    model_role: str,
    checkpoint_path: Path,
    cache_path: Path,
    dataset_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit one selected checkpoint/cache pair without modifying either."""
    checkpoint_sha = sha256_file(checkpoint_path)
    cache_sha = sha256_file(cache_path)
    checkpoint = torch_load_checkpoint(checkpoint_path, map_location="cpu")
    cache = torch_load_checkpoint(cache_path, map_location="cpu")
    require(isinstance(checkpoint, Mapping), "Checkpoint payload is not a mapping")
    require(isinstance(cache, Mapping), "Cache payload is not a mapping")
    require(
        checkpoint.get("checkpoint_type") == "selected_frozen_lognormal_duration",
        "Checkpoint is not a selected matched duration head",
    )
    require(
        checkpoint.get("time_head_mode") == "heteroscedastic_lognormal_duration",
        "Checkpoint duration family drift",
    )
    require(checkpoint.get("evaluation_scope") == "validation_only", "Checkpoint scope drift")
    require(checkpoint.get("held_out_test_evaluated") is False, "Checkpoint used held-out data")
    identity = cache.get("identity")
    require(isinstance(identity, Mapping), "Cache identity is missing")
    require(identity.get("split") == "validation", "Cache is not validation")
    require(identity.get("held_out_test_evaluated") is False, "Cache used held-out data")
    require(identity.get("dataset") == dataset, "Cache dataset drift")
    require(cache.get("cache_schema_version") == 1, "Cache schema drift")
    require(_payload_cache_digest(cache) == cache.get("cache_sha256"), "Cache digest drift")
    resume_identity = checkpoint.get("resume_identity")
    require(isinstance(resume_identity, Mapping), "Checkpoint resume identity missing")
    require(resume_identity.get("dataset") == dataset, "Checkpoint dataset drift")
    require(
        resume_identity.get("validation_cache_sha256") == cache.get("cache_sha256"),
        "Checkpoint/cache identity drift",
    )
    _validate_checkpoint_role(
        checkpoint,
        model_role=model_role,
        resume_identity=resume_identity,
    )

    state = checkpoint.get("model_state_dict")
    require(isinstance(state, Mapping), "Checkpoint state is missing")
    require(
        canonical_state_dict_sha256(state) == checkpoint.get("model_state_sha256"),
        "Checkpoint model-state digest drift",
    )
    hidden = cache.get("time_hidden")
    target = cache.get("target_dt")
    require(isinstance(hidden, torch.Tensor), "Cached hidden states are missing")
    require(isinstance(target, torch.Tensor), "Cached duration targets are missing")
    require(hidden.ndim == 2 and target.shape == (hidden.shape[0],), "Cache shape drift")

    head_contract = checkpoint.get("encoder_config", {}).get("time_head", {})
    statistics = checkpoint.get("train_time_statistics", {})
    time_scale = float(statistics.get("time_scale", head_contract.get("time_scale")))
    sigma_floor = float(head_contract.get("time_sigma_floor", 0.001))
    location, scale = _conditional_parameters(
        hidden=hidden,
        state=state,
        sigma_floor=sigma_floor,
    )
    result = summarize_duration_likelihoods(
        target_dt=target,
        location=location,
        scale=scale,
        time_scale=time_scale,
        top_code=dataset_contract.get("top_code"),
    )
    replay = float(result["groups"]["all"]["continuous_observation_nll"])
    selected = float(checkpoint["selected_metric_value"])
    require(
        math.isclose(replay, selected, rel_tol=0.0, abs_tol=1e-6),
        "Continuous selected-metric replay drift",
    )
    result.update(
        {
            "dataset": dataset,
            "model_role": model_role,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_file_sha256": checkpoint_sha,
            "cache_path": str(cache_path.resolve()),
            "cache_file_sha256": cache_sha,
            "cache_identity_sha256": cache.get("cache_sha256"),
            "selected_continuous_nll": selected,
            "selected_continuous_nll_replay_error": replay - selected,
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
            "trained_or_selected_by_audit": False,
            "checkpoint_role_binding_verified": True,
        }
    )
    return result


def _resolve(path: str, *, base: Path = PROJECT_ROOT) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def _load_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if args.input_spec is not None:
        spec_path = _resolve(args.input_spec)
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
        require(payload.get("schema_version") == 1, "Input-spec schema drift")
        require(isinstance(payload.get("rows"), list), "Input-spec rows missing")
        rows.extend(dict(row) for row in payload["rows"])
    for values in args.row or []:
        dataset, model_role, checkpoint_path, cache_path = values
        rows.append(
            {
                "dataset": dataset,
                "model_role": model_role,
                "checkpoint_path": checkpoint_path,
                "cache_path": cache_path,
            }
        )
    require(bool(rows), "At least one checkpoint/cache row is required")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument(
        "--input-spec",
        help="JSON file with schema_version=1 and checkpoint/cache rows",
    )
    parser.add_argument(
        "--row",
        action="append",
        nargs=4,
        metavar=("DATASET", "ROLE", "CHECKPOINT", "CACHE"),
        help="Audit one row; may be repeated",
    )
    parser.add_argument("--output", help="Write the aggregate JSON here; default stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract_path = _resolve(args.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    datasets = validate_contract(contract)
    rows = _load_rows(args)
    allowed_roles = set(contract["scope"]["model_roles"])
    results = []
    for row in rows:
        dataset = str(row["dataset"])
        model_role = str(row["model_role"])
        require(dataset in datasets, f"Dataset outside audit contract: {dataset}")
        require(model_role in allowed_roles, f"Model role outside audit contract: {model_role}")
        results.append(
            audit_checkpoint_cache(
                dataset=dataset,
                model_role=model_role,
                checkpoint_path=_resolve(str(row["checkpoint_path"])),
                cache_path=_resolve(str(row["cache_path"])),
                dataset_contract=datasets[dataset],
            )
        )

    references = contract.get("verified_reference_values", {})
    for result in results:
        key = f"{result['dataset']}/{result['model_role']}"
        reference = references.get(key)
        require(isinstance(reference, Mapping), f"Unsupported audit row: {key}")
        observed = result["groups"]["all"]
        comparison: dict[str, Any] = {}
        for source_name, result_name in (
            ("continuous_validation_nll", "continuous_observation_nll"),
            ("centered_integer_validation_nll", "integer_observation_nll"),
        ):
            if source_name in reference:
                delta = float(observed[result_name]) - float(reference[source_name])
                tolerance = float(reference.get("absolute_tolerance", 1e-6))
                comparison[source_name] = {
                    "expected": float(reference[source_name]),
                    "observed": float(observed[result_name]),
                    "delta": delta,
                    "matches": abs(delta) <= tolerance,
                    "absolute_tolerance": tolerance,
                }
                require(
                    comparison[source_name]["matches"] is True,
                    f"Verified reference mismatch for {key}: {source_name}",
                )
        result["verified_reference_comparison"] = comparison

    payload = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "trained_or_selected_by_audit": False,
        "rows": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        save_json(_resolve(args.output), payload)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
