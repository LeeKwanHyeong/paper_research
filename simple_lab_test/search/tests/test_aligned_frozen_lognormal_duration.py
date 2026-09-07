from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.TPPs.CountAwareFactory import build_count_aware_model
from paper.scripts.run_hard_lmm_frozen_lognormal_duration import (
    CANDIDATE_TIME_HEAD_MODE,
    LAST_CHECKPOINT_NAME,
    OBSERVATION_LIKELIHOOD_POSITIVE_INTEGER,
    SELECTED_CHECKPOINT_NAME,
    TIME_HEAD_PARAMETER_NAMES,
    FrozenFeatureCache,
    build_frozen_lognormal_candidate,
    fit_time_head_from_cache,
    freeze_time_head_only,
    positive_integer_time_log_likelihood,
    state_partition_sha256,
    validate_observation_likelihood_contract,
)
from paper.scripts.run_aligned_frozen_lognormal_duration import (
    DEFAULT_CONTRACT as ALIGNED_CONTRACT_PATH,
    load_contract_without_duplicate_keys,
    validate_contract as validate_aligned_contract,
)
from simple_lab_test.search.common.runner import (
    canonical_state_dict_sha256,
)


def positive_integer_contract(*, top_coded: bool = False) -> dict[str, str | None]:
    return {
        "mode": OBSERVATION_LIKELIHOOD_POSITIVE_INTEGER,
        "selection_formula": (
            "mean negative normalized positive-integer log-normal "
            "observation log likelihood"
        ),
        "calculation_dtype": "float64",
        "first_bin": "P(T <= 1.5)",
        "regular_bin": "P(d - 0.5 < T <= d + 0.5), d >= 2",
        "top_code": "P(T > 29.5)" if top_coded else None,
    }


def train_time_statistics() -> dict[str, float | int | str]:
    return {
        "statistics_source_split": "train",
        "target_count": 31,
        "target_dt_min": 1.0,
        "target_dt_mean": 4.0,
        "target_dt_p50": 3.0,
        "target_dt_p99": 12.0,
        "target_dt_max": 14.0,
        "time_scale": 3.0,
        "wd_safety_limit": 40.0,
        "time_w_max": 12.0,
        "time_initial_intercept": math.log(0.75),
        "target_log_scaled_mean": 0.25,
        "target_log_scaled_std": 0.8,
    }


def build_source_and_candidate():
    torch.manual_seed(8719)
    source, encoder = build_count_aware_model(
        "titantpp",
        hidden_dim=64,
        train_log_mean=2.0,
        train_log_std=0.7,
        max_seq_len=16,
        quantity_variant="count_only_log_regression",
        lambda_tail=0.0,
        time_head_mode="legacy_clamped_rmtpp",
        time_scale=3.0,
        time_w_max=10.0 / 3.0,
        time_intercept_limit=300.0,
        time_initial_intercept=0.0,
        time_wd_safety_limit=40.0,
    )
    source.eval()
    state = copy.deepcopy(source.state_dict())
    payload = {
        "model_state_dict": state,
        "model_state_sha256": canonical_state_dict_sha256(state),
        "encoder_config": encoder,
        "interface_meta": {
            "mode": "mark_free_count_aware_log_regression",
            "target_quantity_masked_from_history": True,
            "train_target_mean": 2.0,
            "train_target_std": 0.7,
        },
    }
    candidate, _, metadata = build_frozen_lognormal_candidate(
        payload,
        train_time_statistics=train_time_statistics(),
        time_sigma_floor=0.001,
    )
    return source, payload, candidate, metadata


def zero_hidden(count: int) -> torch.Tensor:
    return torch.zeros(count, 64, dtype=torch.float32)


def standard_normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def standardized_boundary(candidate, boundary: float) -> float:
    location = float(candidate.b_t.detach().item())
    # The likelihood promotes the stored float32 parameters before applying
    # softplus, so reproduce that contract rather than its float32 telemetry.
    sigma = float(
        (
            candidate.time_sigma_floor
            + torch.nn.functional.softplus(candidate.w_raw.detach().double())
        ).item()
    )
    return (
        math.log(boundary / float(candidate.time_scale)) - location
    ) / sigma


def test_first_and_regular_integer_bins_match_closed_form() -> None:
    _, _, candidate, _ = build_source_and_candidate()
    observed = positive_integer_time_log_likelihood(
        candidate,
        zero_hidden(2),
        torch.tensor([1.0, 2.0]),
        is_right_censored=torch.zeros(2, dtype=torch.bool),
    ).exp()

    cdf_1_5 = standard_normal_cdf(
        standardized_boundary(candidate, 1.5)
    )
    cdf_2_5 = standard_normal_cdf(
        standardized_boundary(candidate, 2.5)
    )
    expected = torch.tensor(
        [cdf_1_5, cdf_2_5 - cdf_1_5], dtype=torch.float64
    )
    assert torch.allclose(observed, expected, atol=1e-14, rtol=0.0)


def test_instacart_top_code_is_survival_from_29_point_5() -> None:
    _, _, candidate, _ = build_source_and_candidate()
    observed = positive_integer_time_log_likelihood(
        candidate,
        zero_hidden(1),
        torch.tensor([30.0]),
        is_right_censored=torch.tensor([True]),
    )
    expected = torch.special.log_ndtr(
        torch.tensor(
            [-standardized_boundary(candidate, 29.5)],
            dtype=torch.float64,
        )
    )
    assert torch.allclose(observed, expected, atol=1e-14, rtol=0.0)


def test_instacart_top_coded_observation_law_normalizes_to_one() -> None:
    _, _, candidate, _ = build_source_and_candidate()
    targets = torch.arange(1, 31, dtype=torch.float32)
    censor = torch.zeros(targets.shape, dtype=torch.bool)
    censor[-1] = True
    probability = positive_integer_time_log_likelihood(
        candidate,
        zero_hidden(targets.numel()),
        targets,
        is_right_censored=censor,
    ).exp()
    assert probability.sum().item() == pytest.approx(1.0, abs=1e-12)


def test_positive_integer_observation_law_normalizes_to_one() -> None:
    _, _, candidate, _ = build_source_and_candidate()
    finite_codes = torch.arange(1, 101, dtype=torch.float32)
    # Code 101 closes the telescoping finite bins with S(100.5).
    targets = torch.cat([finite_codes, torch.tensor([101.0])])
    censor = torch.zeros(targets.shape, dtype=torch.bool)
    censor[-1] = True
    log_probability = positive_integer_time_log_likelihood(
        candidate,
        zero_hidden(targets.numel()),
        targets,
        is_right_censored=censor,
    )
    assert torch.isfinite(log_probability).all()
    assert log_probability.exp().sum().item() == pytest.approx(
        1.0, abs=1e-12
    )


@pytest.mark.parametrize("invalid", [0.0, 1.25, 2.0001])
def test_positive_integer_likelihood_rejects_off_contract_targets(
    invalid: float,
) -> None:
    _, _, candidate, _ = build_source_and_candidate()
    with pytest.raises(ValueError, match="integer|at least one"):
        positive_integer_time_log_likelihood(
            candidate,
            zero_hidden(1),
            torch.tensor([invalid]),
            is_right_censored=torch.tensor([False]),
        )


def test_positive_integer_likelihood_rejects_value_above_top_code() -> None:
    _, _, candidate, _ = build_source_and_candidate()
    with pytest.raises(ValueError, match="exceeds the right-censor code"):
        positive_integer_time_log_likelihood(
            candidate,
            zero_hidden(2),
            torch.tensor([30.0, 31.0]),
            is_right_censored=torch.tensor([True, False]),
        )


def test_integer_likelihood_has_finite_time_head_gradients() -> None:
    _, _, candidate, _ = build_source_and_candidate()
    parameters = freeze_time_head_only(candidate)
    hidden = torch.linspace(-2.0, 2.0, steps=6 * 64).reshape(6, 64)
    target = torch.tensor([1.0, 2.0, 3.0, 10.0, 30.0, 29.0])
    censor = torch.tensor([False, False, False, False, True, False])
    loss = -positive_integer_time_log_likelihood(
        candidate,
        hidden,
        target,
        is_right_censored=censor,
    ).mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert {
        name
        for name, parameter in candidate.named_parameters()
        if parameter.requires_grad
    } == set(TIME_HEAD_PARAMETER_NAMES)
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in parameters
    )


@pytest.mark.parametrize("location", [-10.0, 10.0])
def test_integer_likelihood_is_finite_in_both_distribution_tails(
    location: float,
) -> None:
    _, _, candidate, _ = build_source_and_candidate()
    parameters = freeze_time_head_only(candidate)
    with torch.no_grad():
        candidate.b_t.fill_(location)
    target = torch.tensor([1.0, 2.0, 29.0, 30.0])
    censor = torch.tensor([False, False, False, True])
    loss = -positive_integer_time_log_likelihood(
        candidate,
        zero_hidden(target.numel()),
        target,
        is_right_censored=censor,
    ).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in parameters
    )


def integer_cache(
    source,
    *,
    count: int,
    include_quantity: bool,
) -> FrozenFeatureCache:
    generator = torch.Generator().manual_seed(5701 + count)
    time_hidden = torch.randn(count, 64, generator=generator)
    target_dt = torch.round(
        3.0 * torch.exp(0.25 + 0.3 * time_hidden[:, 0])
    ).clamp(min=1.0, max=20.0)
    if not include_quantity:
        return FrozenFeatureCache(
            time_hidden=time_hidden,
            target_dt=target_dt,
        )
    quantity_hidden = torch.randn(count, 64, generator=generator)
    target_quantity = torch.arange(count, dtype=torch.float32).remainder(7)
    with torch.no_grad():
        source_prediction = source.predict_quantity(quantity_hidden)[1]
    return FrozenFeatureCache(
        time_hidden=time_hidden,
        target_dt=target_dt,
        quantity_hidden=quantity_hidden,
        target_quantity=target_quantity,
        source_quantity_prediction=source_prediction,
    )


def fit_kwargs(
    *,
    payload,
    candidate,
    metadata,
    train,
    validation,
    output_dir: Path,
    planned_epochs: int,
) -> dict:
    return {
        "model": candidate,
        "train_cache": train,
        "validation_cache": validation,
        "output_dir": output_dir,
        "dataset": "synthetic_integer",
        "censor_threshold": None,
        "contract_sha256": "c" * 64,
        "source_checkpoint_sha256": "f" * 64,
        "source_state_sha256": payload["model_state_sha256"],
        "source_non_time_state_sha256": state_partition_sha256(
            payload["model_state_dict"], time_head=False
        ),
        "candidate_initial_state_sha256": metadata[
            "candidate_initial_state_sha256"
        ],
        "train_time_statistics": train_time_statistics(),
        "calibration_source_revision": "a" * 40,
        "seed": 42,
        "planned_epochs": planned_epochs,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "batch_size": 8,
        "quantity_replay_batch_size": 8,
        "grad_clip": 1.0,
        "min_epochs": planned_epochs + 1,
        "patience": 20,
        "device": "cpu",
        "candidate_metadata": metadata,
        "source_metadata": {"calibration_source_revision": "a" * 40},
        "observation_likelihood_contract": positive_integer_contract(),
    }


def test_integer_contract_fit_and_resume_are_exact(tmp_path: Path) -> None:
    contract = positive_integer_contract()
    assert validate_observation_likelihood_contract(contract) == contract
    source, payload, candidate, metadata = build_source_and_candidate()
    train = integer_cache(source, count=25, include_quantity=False)
    validation = integer_cache(source, count=15, include_quantity=True)
    full = fit_time_head_from_cache(
        **fit_kwargs(
            payload=payload,
            candidate=candidate,
            metadata=metadata,
            train=train,
            validation=validation,
            output_dir=tmp_path / "full",
            planned_epochs=3,
        )
    )

    _, _, partial_candidate, partial_metadata = build_source_and_candidate()
    resumed_dir = tmp_path / "resumed"
    paused = fit_time_head_from_cache(
        **fit_kwargs(
            payload=payload,
            candidate=partial_candidate,
            metadata=partial_metadata,
            train=train,
            validation=validation,
            output_dir=resumed_dir,
            planned_epochs=3,
        ),
        run_epoch_limit=1,
    )
    assert paused["status"] == "paused"
    assert paused["completed_epochs"] == 1
    assert paused["resume_identity"]["observation_likelihood_contract"] == contract
    assert (resumed_dir / LAST_CHECKPOINT_NAME).is_file()

    _, _, resumed_candidate, resumed_metadata = build_source_and_candidate()
    resumed = fit_time_head_from_cache(
        **fit_kwargs(
            payload=payload,
            candidate=resumed_candidate,
            metadata=resumed_metadata,
            train=train,
            validation=validation,
            output_dir=resumed_dir,
            planned_epochs=3,
        )
    )
    assert resumed["history"] == full["history"]
    assert resumed["best_epoch"] == full["best_epoch"]
    assert resumed["selected_state_sha256"] == full["selected_state_sha256"]
    assert resumed["quantity_prediction_bitwise_identical"] is True
    assert resumed["observation_likelihood_contract"] == contract
    assert resumed["resume_identity"]["observation_likelihood_contract"] == contract
    assert resumed["observation_likelihood_mode"] == (
        OBSERVATION_LIKELIHOOD_POSITIVE_INTEGER
    )

    checkpoint = torch.load(
        resumed_dir / SELECTED_CHECKPOINT_NAME,
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["observation_likelihood_contract"] == contract
    assert checkpoint["resume_identity"]["observation_likelihood_contract"] == contract


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("history", "Cached selected metric drift"),
        (
            "history_after_early_stop",
            "Cached fit continued after early stopping",
        ),
        ("checkpoint_state_digest", "Cached selected state digest drift"),
        (
            "self_consistent_time_head_state",
            "Cached selected NLL replay drift",
        ),
    ],
)
def test_cached_completed_fit_revalidates_selector_and_checkpoint(
    tmp_path: Path,
    tamper: str,
    message: str,
) -> None:
    source, payload, candidate, metadata = build_source_and_candidate()
    train = integer_cache(source, count=9, include_quantity=False)
    validation = integer_cache(source, count=7, include_quantity=True)
    output_dir = tmp_path / tamper
    kwargs = fit_kwargs(
        payload=payload,
        candidate=candidate,
        metadata=metadata,
        train=train,
        validation=validation,
        output_dir=output_dir,
        planned_epochs=0,
    )
    kwargs["min_epochs"] = 0
    kwargs["patience"] = 1
    fit_time_head_from_cache(**kwargs)

    if tamper in {"history", "history_after_early_stop"}:
        summary_path = output_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if tamper == "history":
            summary["history"][0]["val_proper_time_nll"] += 0.1
        else:
            selected_nll = summary["best_validation_proper_time_nll"]
            summary["history"].extend(
                [
                    {"epoch": 1, "val_proper_time_nll": selected_nll + 0.1},
                    {"epoch": 2, "val_proper_time_nll": selected_nll + 0.2},
                ]
            )
            summary["completed_epochs"] = 2
            summary["stopped_early"] = True
        summary_path.write_text(
            json.dumps(summary, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif tamper == "checkpoint_state_digest":
        checkpoint_path = output_dir / SELECTED_CHECKPOINT_NAME
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        checkpoint["model_state_sha256"] = "0" * 64
        torch.save(checkpoint, checkpoint_path)
    else:
        checkpoint_path = output_dir / SELECTED_CHECKPOINT_NAME
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        changed_state = {
            name: value.detach().clone()
            for name, value in checkpoint["model_state_dict"].items()
        }
        changed_state["b_t"].add_(1.0)
        checkpoint["model_state_dict"] = changed_state
        checkpoint["model_state_sha256"] = canonical_state_dict_sha256(
            changed_state
        )
        checkpoint["selected_time_head_state_sha256"] = (
            state_partition_sha256(changed_state, time_head=True)
        )
        torch.save(checkpoint, checkpoint_path)
        summary_path = output_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["selected_state_sha256"] = checkpoint[
            "model_state_sha256"
        ]
        summary["selected_time_head_state_sha256"] = checkpoint[
            "selected_time_head_state_sha256"
        ]
        summary_path.write_text(
            json.dumps(summary, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    _, _, fresh_candidate, fresh_metadata = build_source_and_candidate()
    replay_kwargs = {
        **kwargs,
        "model": fresh_candidate,
        "candidate_metadata": fresh_metadata,
        "candidate_initial_state_sha256": fresh_metadata[
            "candidate_initial_state_sha256"
        ],
    }
    with pytest.raises(ValueError, match=message):
        fit_time_head_from_cache(**replay_kwargs)


def test_checked_in_aligned_contract_pins_dataset_observation_laws() -> None:
    contract = load_contract_without_duplicate_keys(ALIGNED_CONTRACT_PATH)
    datasets = validate_aligned_contract(contract)
    assert contract["acceptance"]["minimum_aligned_B_improvement_scope"] == (
        "datasets using positive_integer_round_clamp_lognormal only"
    )
    assert contract["acceptance"][
        "maximum_continuous_objective_replay_worsening"
    ] == 1e-6
    assert contract["comparison_policy"][
        "same_calibration_source_revision_for_all_rows"
    ] is True
    assert contract["comparison_policy"][
        "maximum_prior_B_continuous_replay_absolute_drift"
    ] == 1e-6
    assert datasets["intermittent_frozen_5000"]["observation_contract"][
        "mode"
    ] == "continuous_lognormal_density"
    assert datasets["yellow_trip_hourly"]["observation_contract"] == (
        positive_integer_contract()
    )
    assert datasets["insta_market_basket"]["observation_contract"] == (
        positive_integer_contract(top_coded=True)
    )
    assert all(
        len(source["source_non_time_state_sha256"]) == 64
        for dataset in datasets.values()
        for source in dataset["sources"].values()
    )


def test_aligned_contract_rejects_universal_integer_improvement_gate() -> None:
    contract = load_contract_without_duplicate_keys(ALIGNED_CONTRACT_PATH)
    contract["acceptance"]["minimum_aligned_B_improvement_scope"] = (
        "all datasets"
    )
    with pytest.raises(ValueError, match="improvement scope drift"):
        validate_aligned_contract(contract)
