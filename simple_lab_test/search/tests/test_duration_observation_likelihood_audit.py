from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from paper.scripts.audit_duration_observation_likelihood import (
    DEFAULT_CONTRACT,
    _validate_checkpoint_role,
    centered_integer_bin_bounds,
    centered_integer_log_mass,
    continuous_lognormal_nll_components,
    stable_log_difference,
    summarize_duration_likelihoods,
    validate_contract,
)


@pytest.mark.parametrize("model_role", ["A", "rmtpp", "thp"])
def test_matched_checkpoint_role_must_be_explicit_and_exact(model_role: str) -> None:
    checkpoint = {"model_role": model_role}
    identity = {
        "contract_id": "matched_frozen_lognormal_duration_v1",
        "planned_epochs": 100,
    }
    _validate_checkpoint_role(
        checkpoint,
        model_role=model_role,
        resume_identity=identity,
    )
    with pytest.raises(ValueError, match="model-role drift"):
        _validate_checkpoint_role(
            {"model_role": None},
            model_role=model_role,
            resume_identity=identity,
        )
    wrong_role = "thp" if model_role != "thp" else "A"
    with pytest.raises(ValueError, match="model-role drift"):
        _validate_checkpoint_role(
            {"model_role": wrong_role},
            model_role=model_role,
            resume_identity=identity,
        )


def test_frozen_B_role_is_bound_to_its_originating_contract() -> None:
    _validate_checkpoint_role(
        {},
        model_role="B",
        resume_identity={
            "contract_id": "hard_lmm_frozen_lognormal_duration_v1",
            "planned_epochs": 100,
        },
    )
    with pytest.raises(ValueError, match="frozen-B contract"):
        _validate_checkpoint_role(
            {"model_role": "A"},
            model_role="B",
            resume_identity={
                "contract_id": "matched_frozen_lognormal_duration_v1",
                "planned_epochs": 100,
            },
        )


def test_role_binding_rejects_e1_checkpoint_even_with_correct_role() -> None:
    with pytest.raises(ValueError, match="full 100-epoch"):
        _validate_checkpoint_role(
            {"model_role": "thp"},
            model_role="thp",
            resume_identity={
                "contract_id": "matched_frozen_lognormal_duration_v1",
                "planned_epochs": 1,
            },
        )


def test_stable_log_difference_handles_extreme_log_probabilities() -> None:
    larger = torch.tensor([-10_000.0, -1.0], dtype=torch.float64)
    smaller = torch.tensor([-10_001.0, -1.0 - 1e-12], dtype=torch.float64)
    observed = stable_log_difference(larger, smaller)
    deltas = smaller - larger
    expected = larger + torch.tensor(
        [
            math.log1p(-math.exp(float(deltas[0].item()))),
            math.log(-math.expm1(float(deltas[1].item()))),
        ],
        dtype=torch.float64,
    )
    assert torch.isfinite(observed).all()
    assert torch.allclose(observed, expected, atol=1e-12, rtol=0.0)


def test_centered_bounds_apply_half_unit_floor() -> None:
    lower, upper = centered_integer_bin_bounds(
        torch.tensor([1.0, 2.0, 30.0])
    )
    assert torch.equal(lower, torch.tensor([0.5, 1.5, 29.5], dtype=torch.float64))
    assert torch.equal(upper, torch.tensor([1.5, 2.5, 30.5], dtype=torch.float64))
    with pytest.raises(ValueError, match="integer grid"):
        centered_integer_bin_bounds(torch.tensor([1.25]))


def test_centered_lognormal_mass_matches_known_unit_lognormal_bin() -> None:
    observed = centered_integer_log_mass(
        target_dt=torch.tensor([1.0]),
        location=torch.tensor([0.0]),
        scale=torch.tensor([1.0]),
        time_scale=1.0,
    )
    assert observed.item() == pytest.approx(-0.8835245213470986, abs=1e-14)


def test_positive_tail_bin_uses_a_finite_log_space_difference() -> None:
    observed = centered_integer_log_mass(
        target_dt=torch.tensor([1000.0]),
        location=torch.tensor([0.0]),
        scale=torch.tensor([0.01]),
        time_scale=1.0,
    )
    assert observed.shape == (1,)
    assert torch.isfinite(observed).all()
    assert observed.item() < -100_000.0


def test_instacart_top_code_is_survival_from_29_point_5() -> None:
    location = torch.tensor([0.0], dtype=torch.float64)
    scale = torch.tensor([1.0], dtype=torch.float64)
    observed = centered_integer_log_mass(
        target_dt=torch.tensor([30.0]),
        location=location,
        scale=scale,
        time_scale=1.0,
        top_code=30.0,
    )
    expected = torch.special.log_ndtr(
        -torch.tensor([math.log(29.5)], dtype=torch.float64)
    )
    assert torch.equal(observed, expected)


def test_continuous_nll_components_reconstruct_density() -> None:
    target = torch.tensor([1.0, 2.0, 7.0])
    location = torch.tensor([0.1, -0.2, 0.3])
    scale = torch.tensor([0.7, 1.1, 0.5])
    components = continuous_lognormal_nll_components(
        target_dt=target,
        location=location,
        scale=scale,
        time_scale=2.0,
    )
    expected = (
        0.5
        * torch.square(
            (
                torch.log(target.double())
                - math.log(2.0)
                - location.double()
            )
            / scale.double()
        )
        + torch.log(scale.double())
        + torch.log(target.double())
        + 0.5 * math.log(2.0 * math.pi)
    )
    assert torch.allclose(components["total"], expected, atol=1e-14, rtol=0.0)
    reconstructed = (
        components["quadratic"]
        + components["log_sigma"]
        + components["jacobian"]
        + components["constant"]
    )
    assert torch.equal(components["total"], reconstructed)


def test_summary_reports_mode_and_other_and_separates_top_code_survival() -> None:
    summary = summarize_duration_likelihoods(
        target_dt=torch.tensor([1.0, 1.0, 2.0, 30.0]),
        location=torch.zeros(4),
        scale=torch.ones(4),
        time_scale=1.0,
        top_code=30.0,
    )
    assert summary["duration_mode"] == 1.0
    assert summary["top_code_integer_observation_lower_bound"] == 29.5
    assert summary["groups"]["dt_equals_mode"]["count"] == 2
    assert summary["groups"]["dt_other"]["count"] == 2
    assert summary["groups"]["all"]["continuous_density"]["count"] == 3
    assert summary["groups"]["all"]["continuous_top_code_survival"]["count"] == 1


def test_checked_in_contract_is_read_only_and_covers_integer_datasets() -> None:
    contract = json.loads(DEFAULT_CONTRACT.read_text(encoding="utf-8"))
    datasets = validate_contract(contract)
    assert tuple(datasets) == (
        "intermittent_frozen_5000",
        "yellow_trip_hourly",
        "insta_market_basket",
    )
    assert contract["scope"]["retraining"] is False
    assert contract["scope"]["held_out_test"] is False
    assert datasets["insta_market_basket"]["top_code"] == 30.0
    assert contract["verified_reference_values"]["yellow_trip_hourly/B"][
        "centered_integer_validation_nll"
    ] == pytest.approx(0.8999188466)
    assert len(contract["verified_reference_values"]) == 11
    assert "intermittent_frozen_5000/rmtpp" not in contract[
        "verified_reference_values"
    ]
