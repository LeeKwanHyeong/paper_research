from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.TPPs.ConditionalCDFCalibrator import (
    ConditionalKumaraswamyCDFCalibrator,
)


CONTRACT = ROOT / "paper/contracts/aligned_conditional_cdf_calibration_v1.json"


def _normal_terms(z: torch.Tensor) -> tuple[torch.Tensor, ...]:
    log_cdf = torch.special.log_ndtr(z)
    log_survival = torch.special.log_ndtr(-z)
    log_density = -0.5 * z.square() - 0.5 * math.log(2.0 * math.pi)
    return log_cdf, log_survival, log_density


def _model() -> ConditionalKumaraswamyCDFCalibrator:
    torch.manual_seed(42)
    return ConditionalKumaraswamyCDFCalibrator(hidden_dim=64)


def _stable_logdiff(log_large: torch.Tensor, log_small: torch.Tensor) -> torch.Tensor:
    shape = log_large.shape
    large = log_large.reshape(-1)
    delta = (log_small.reshape(-1) - large).clamp_max(0.0)
    split = -math.log(2.0)
    first = delta < split
    complement = torch.empty_like(delta)
    if bool(first.any()):
        indices = torch.nonzero(first, as_tuple=False).squeeze(-1)
        complement = complement.index_copy(
            0, indices, torch.log1p(-torch.exp(delta[indices]))
        )
    second = ~first
    if bool(second.any()):
        indices = torch.nonzero(second, as_tuple=False).squeeze(-1)
        complement = complement.index_copy(
            0, indices, torch.log(-torch.expm1(delta[indices]))
        )
    return (large + complement).reshape(shape)


def test_contract_freezes_one_common_candidate_before_training() -> None:
    def reject_duplicates(pairs):
        counts = Counter(name for name, _ in pairs)
        duplicates = [name for name, count in counts.items() if count > 1]
        assert not duplicates
        return dict(pairs)

    contract = json.loads(
        CONTRACT.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
    )
    assert contract["contract_id"] == "aligned_conditional_cdf_calibration_v1"
    assert contract["candidate"]["single_hypothesis"] is True
    assert contract["candidate"]["dataset_specific_parameter"] is False
    assert contract["candidate"]["observation_specific_model_branch"] is False
    assert contract["cdf_transform"]["expected_trainable_parameter_count"] == 130
    assert contract["cdf_transform"]["same_parameterization_for_every_dataset"]
    assert contract["future_optimization_contract"]["dataset_specific_hyperparameters"] is False
    assert contract["future_optimization_contract"]["result_dependent_retuning"] is False
    assert contract["scope"]["held_out_test"] is False
    assert contract["scope"]["gpu_training_in_current_scope"] is False
    assert contract["current_completion"]["training_runner_implemented"] is False
    assert [row["dataset"] for row in contract["source_artifacts"]] == [
        "intermittent_frozen_5000",
        "yellow_trip_hourly",
        "insta_market_basket",
    ]
    for row in contract["source_artifacts"]:
        for name, value in row.items():
            if name.endswith("sha256"):
                assert len(value) == 64
                assert set(value) <= set("0123456789abcdef")


def test_zero_initialization_is_bitwise_identity_across_extreme_tails() -> None:
    model = _model()
    z = torch.linspace(-40.0, 40.0, 321, dtype=torch.float64)
    hidden = torch.randn(z.numel(), 64)
    base_log_cdf, base_log_survival, base_log_density = _normal_terms(z)
    a, b = model.shape_parameters(hidden)
    assert torch.equal(a, torch.ones_like(a))
    assert torch.equal(b, torch.ones_like(b))
    assert sum(parameter.numel() for parameter in model.parameters()) == 130

    calibrated_cdf, calibrated_survival = model.transformed_log_cdf_survival(
        base_log_cdf=base_log_cdf,
        base_log_survival=base_log_survival,
        hidden=hidden,
    )
    calibrated_density = model.transformed_log_density(
        base_log_density=base_log_density,
        base_log_cdf=base_log_cdf,
        base_log_survival=base_log_survival,
        hidden=hidden,
    )
    assert torch.equal(calibrated_cdf, base_log_cdf)
    assert torch.equal(calibrated_survival, base_log_survival)
    assert torch.equal(calibrated_density, base_log_density)
    probability = torch.linspace(0.01, 0.99, z.numel(), dtype=torch.float64)
    assert torch.equal(model.base_quantile_level(probability, hidden), probability)


def test_nonidentity_transform_is_normalized_monotone_and_finite() -> None:
    model = _model()
    with torch.no_grad():
        model.shape_head.bias.copy_(torch.tensor([0.4, -0.3]))
    z = torch.linspace(-40.0, 40.0, 50_001, dtype=torch.float64)
    hidden = torch.zeros(z.numel(), 64)
    base_log_cdf, base_log_survival, base_log_density = _normal_terms(z)
    calibrated_cdf, calibrated_survival = model.transformed_log_cdf_survival(
        base_log_cdf=base_log_cdf,
        base_log_survival=base_log_survival,
        hidden=hidden,
    )
    calibrated_density = model.transformed_log_density(
        base_log_density=base_log_density,
        base_log_cdf=base_log_cdf,
        base_log_survival=base_log_survival,
        hidden=hidden,
    )
    assert torch.isfinite(calibrated_cdf).all()
    assert torch.isfinite(calibrated_survival).all()
    assert torch.isfinite(calibrated_density).all()
    assert torch.logaddexp(calibrated_cdf, calibrated_survival).abs().max() < 1e-10
    assert torch.all(calibrated_cdf.diff() >= 0.0)
    integral = torch.trapezoid(torch.exp(calibrated_density), z)
    assert torch.allclose(
        integral, torch.tensor(1.0, dtype=torch.float64), atol=1e-4, rtol=0.0
    )


def test_positive_integer_and_top_code_probabilities_sum_to_one() -> None:
    model = _model()
    with torch.no_grad():
        model.shape_head.bias.copy_(torch.tensor([-0.5, 0.35]))
    maximum_code = 300
    boundaries = torch.arange(1.5, maximum_code, 1.0, dtype=torch.float64)
    mu, sigma = 0.35, 0.9
    z = (torch.log(boundaries) - mu) / sigma
    base_log_cdf, base_log_survival, _ = _normal_terms(z)
    hidden = torch.zeros(boundaries.numel(), 64)
    log_cdf, log_survival = model.transformed_log_cdf_survival(
        base_log_cdf=base_log_cdf,
        base_log_survival=base_log_survival,
        hidden=hidden,
    )
    masses = [torch.exp(log_cdf[0])]
    for index in range(1, boundaries.numel()):
        masses.append(torch.exp(_stable_logdiff(log_cdf[index], log_cdf[index - 1])))
    masses.append(torch.exp(log_survival[-1]))
    total = torch.stack(masses).sum()
    assert torch.allclose(
        total, torch.tensor(1.0, dtype=torch.float64), atol=1e-10, rtol=0.0
    )
    assert all(float(value.detach()) > 0.0 for value in masses)


def test_identity_preserves_integer_mass_bitwise() -> None:
    model = _model()
    boundaries = torch.arange(1.5, 50.0, 1.0, dtype=torch.float64)
    z = (torch.log(boundaries) - 0.2) / 0.7
    base_log_cdf, base_log_survival, _ = _normal_terms(z)
    hidden = torch.randn(boundaries.numel(), 64)
    log_cdf, log_survival = model.transformed_log_cdf_survival(
        base_log_cdf=base_log_cdf,
        base_log_survival=base_log_survival,
        hidden=hidden,
    )
    base_regular = _stable_logdiff(base_log_cdf[1:], base_log_cdf[:-1])
    calibrated_regular = _stable_logdiff(log_cdf[1:], log_cdf[:-1])
    assert torch.equal(log_cdf[0], base_log_cdf[0])
    assert torch.equal(calibrated_regular, base_regular)
    assert torch.equal(log_survival[-1], base_log_survival[-1])


@pytest.mark.parametrize("objective", ["continuous", "integer_censored"])
def test_epoch_zero_has_finite_nonzero_shape_head_gradients(objective: str) -> None:
    torch.manual_seed(7)
    model = _model()
    count = 257
    hidden = torch.randn(count, 64)
    z = torch.linspace(-12.0, 12.0, count, dtype=torch.float64)
    base_log_cdf, base_log_survival, base_log_density = _normal_terms(z)
    if objective == "continuous":
        terms = model.transformed_log_density(
            base_log_density=base_log_density,
            base_log_cdf=base_log_cdf,
            base_log_survival=base_log_survival,
            hidden=hidden,
        )
    else:
        log_cdf, log_survival = model.transformed_log_cdf_survival(
            base_log_cdf=base_log_cdf,
            base_log_survival=base_log_survival,
            hidden=hidden,
        )
        regular = _stable_logdiff(log_cdf[1:-1:2], log_cdf[:-2:2])
        terms = torch.cat((log_cdf[:1], regular, log_survival[-1:]))
    loss = -terms.mean()
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.norm()) > 0.0


def test_invalid_base_probability_contract_fails_closed() -> None:
    model = _model()
    hidden = torch.zeros(2, 64)
    with pytest.raises(ValueError, match="not complementary"):
        model.transformed_log_cdf_survival(
            base_log_cdf=torch.log(torch.tensor([0.2, 0.3], dtype=torch.float64)),
            base_log_survival=torch.log(
                torch.tensor([0.2, 0.3], dtype=torch.float64)
            ),
            hidden=hidden,
        )
    with pytest.raises(TypeError, match="float64"):
        model.transformed_log_cdf_survival(
            base_log_cdf=torch.log(torch.tensor([0.5, 0.5])),
            base_log_survival=torch.log(torch.tensor([0.5, 0.5])),
            hidden=hidden,
        )
