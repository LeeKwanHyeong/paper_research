#!/usr/bin/env python3
"""Validation-only A/B location-scale decomposition for Taxi durations."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import torch


RESULT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RESULT_ROOT.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(RESULT_ROOT))

from paper.scripts.audit_duration_observation_likelihood import (  # noqa: E402
    _conditional_parameters,
)
from replay_normalized_integer_likelihood import (  # noqa: E402
    EXPECTED_DATA_SHA256,
    EXPECTED_MODEL_SOURCE_REVISION,
    EXPECTED_SPLIT_MANIFEST_SHA256,
    EXPECTED_VALIDATION_TARGET_IDENTITY_SHA256,
    ROLE_INPUTS,
    likelihoods,
)


OUTPUT = RESULT_ROOT / "taxi_location_scale_hybrid_replay_20260906.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(path)
    return payload


def parameters(role: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, dict]:
    cache_path, checkpoint_path = ROLE_INPUTS[role]
    cache = load(cache_path)
    checkpoint = load(checkpoint_path)
    identity = cache["identity"]
    assert identity["dataset"] == "yellow_trip_hourly"
    assert identity["split"] == "validation"
    assert identity["held_out_test_evaluated"] is False
    assert identity["data_sha256"] == EXPECTED_DATA_SHA256
    assert identity["split_manifest_sha256"] == EXPECTED_SPLIT_MANIFEST_SHA256
    assert identity["target_identity_sha256"] == EXPECTED_VALIDATION_TARGET_IDENTITY_SHA256
    assert identity["target_count"] == 8268
    assert checkpoint["evaluation_scope"] == "validation_only"
    assert checkpoint["held_out_test_evaluated"] is False
    head = checkpoint["encoder_config"].get("time_head", {})
    time_scale = float(
        checkpoint["train_time_statistics"].get(
            "time_scale", head.get("time_scale")
        )
    )
    location, scale = _conditional_parameters(
        hidden=cache["time_hidden"],
        state=checkpoint["model_state_dict"],
        sigma_floor=float(head.get("time_sigma_floor", 0.001)),
    )
    provenance = {
        "cache_path": cache_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix(),
        "cache_sha256": sha256_file(cache_path),
        "checkpoint_path": checkpoint_path.resolve()
        .relative_to(PROJECT_ROOT.resolve())
        .as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "source_checkpoint_selector": checkpoint.get("source_checkpoint_selector"),
    }
    return (
        cache["target_dt"].to(torch.float64),
        location.to(torch.float64),
        scale.to(torch.float64),
        time_scale,
        provenance,
    )


def continuous_density_nll(
    target: torch.Tensor,
    location: torch.Tensor,
    scale: torch.Tensor,
    time_scale: float,
) -> float:
    log_target = torch.log(target)
    z = (log_target - math.log(time_scale) - location) / scale
    log_density = (
        -0.5 * torch.square(z)
        - torch.log(scale)
        - log_target
        - 0.5 * math.log(2.0 * math.pi)
    )
    return float(-log_density.mean().item())


def metrics(
    target: torch.Tensor,
    location: torch.Tensor,
    scale: torch.Tensor,
    time_scale: float,
) -> dict[str, float | int]:
    row = likelihoods(target, location, scale, time_scale)
    median = time_scale * torch.exp(location)
    error = median - target
    row.update(
        {
            "continuous_density_nll": continuous_density_nll(
                target, location, scale, time_scale
            ),
            "time_median_mae": float(error.abs().mean().item()),
            "time_median_rmse": float(torch.sqrt(torch.square(error).mean()).item()),
        }
    )
    return row


def main() -> None:
    target_a, location_a, scale_a, time_scale_a, provenance_a = parameters("A")
    target_b, location_b, scale_b, time_scale_b, provenance_b = parameters("B")
    assert torch.equal(target_a, target_b)
    assert time_scale_a == time_scale_b == 1.0

    combinations = {
        "A_location_A_scale": (location_a, scale_a),
        "B_location_B_scale": (location_b, scale_b),
        "A_location_B_scale": (location_a, scale_b),
        "B_location_A_scale": (location_b, scale_a),
    }
    rows = {
        name: metrics(target_a, location, scale, time_scale_a)
        for name, (location, scale) in combinations.items()
    }

    payload = {
        "schema_version": 1,
        "scope": {
            "dataset": "yellow_trip_hourly",
            "split": "validation_only",
            "target_count": int(target_a.numel()),
            "held_out_test_evaluated": False,
            "training_or_checkpoint_selection": False,
        },
        "model_source_revision": EXPECTED_MODEL_SOURCE_REVISION,
        "purpose": (
            "Output-parameter swap diagnostic. It isolates how the selected A/B "
            "location and scale arrays affect the same validation targets; it is not "
            "a trainable model result or a causal attribution."
        ),
        "rows": rows,
        "inputs": {"A": provenance_a, "B": provenance_b},
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(rows, indent=2, sort_keys=True))
    print(f"output={OUTPUT}")
    print(f"output_sha256={sha256_file(OUTPUT)}")


if __name__ == "__main__":
    main()
