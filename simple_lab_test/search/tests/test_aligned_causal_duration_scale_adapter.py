from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
import pytest
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.TPPs.CausalLogDurationAdapter import (
    CausalLogDurationAdapter,
    LengthOnlyCausalLogDurationAdapter,
)
from paper.scripts.build_aligned_causal_duration_scale_adapter_decision import (
    validate_full_summary,
)
from paper.scripts.run_aligned_causal_duration_scale_adapter import (
    CONTRACT_ID,
    FrozenBaseTimeCache,
    fit_scale_module,
    load_contract_without_duplicate_keys,
    primary_observation_log_likelihood,
    split_appended_target,
    validate_contract,
)
from paper.scripts.run_hard_lmm_frozen_lognormal_duration import (
    OBSERVATION_LIKELIHOOD_CONTINUOUS,
    OBSERVATION_LIKELIHOOD_POSITIVE_INTEGER,
)
from simple_lab_test.search.common.runner import (
    atomic_torch_save,
    canonical_state_dict_sha256,
)


CONTRACT = ROOT / "paper/contracts/aligned_causal_duration_scale_adapter_v1.json"


class SyntheticCanonicalDataset(Dataset):
    def __init__(self, rows: list[tuple[list[float], float]]) -> None:
        self.rows = rows
        self.max_len = max(len(history) + 1 for history, _ in rows) + 1

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        history, target = self.rows[index]
        active = torch.tensor([*history, target], dtype=torch.float32)
        pad = self.max_len - active.numel()
        return {
            "marks": torch.zeros(self.max_len, dtype=torch.long),
            "dts": torch.cat((torch.zeros(pad), active)),
            "mask": torch.cat(
                (
                    torch.zeros(pad, dtype=torch.bool),
                    torch.ones_like(active, dtype=torch.bool),
                )
            ),
            "part_idx": torch.tensor(index, dtype=torch.long),
            "values": torch.zeros(self.max_len),
        }


def make_dataset_and_cache(count: int) -> tuple[Dataset, FrozenBaseTimeCache]:
    rows = []
    targets = []
    for index in range(count):
        recent = 1.0 + float(index % 4)
        target = 1.0 if recent <= 2.0 else 4.0
        rows.append(([1.0, recent] if index % 2 else [recent], target))
        targets.append(target)
    dataset = SyntheticCanonicalDataset(rows)
    location = torch.full((count,), math.log(2.0), dtype=torch.float64)
    cache = FrozenBaseTimeCache(
        location=location,
        sigma=torch.full((count,), 0.8, dtype=torch.float64),
        median=torch.exp(location),
        target_dt=torch.tensor(targets, dtype=torch.float32),
    )
    cache.validate()
    return dataset, cache


def adapter() -> CausalLogDurationAdapter:
    torch.manual_seed(42)
    return CausalLogDurationAdapter(
        log_duration_mean=math.log1p(2.0),
        log_duration_std=0.5,
        sigma_floor=0.001,
    )


def test_contract_is_unambiguous_and_freezes_one_aligned_hypothesis() -> None:
    contract = load_contract_without_duplicate_keys(CONTRACT)
    datasets = validate_contract(contract)
    assert contract["contract_id"] == CONTRACT_ID
    assert contract["candidate"]["single_hypothesis"] is True
    assert contract["candidate"]["scale_only"] is True
    assert contract["execution_policy"]["taxi_first_full_stop"] is True
    assert list(datasets) == [
        "intermittent_frozen_5000",
        "yellow_trip_hourly",
        "insta_market_basket",
    ]

    def reject_duplicates(pairs):
        counts = Counter(name for name, _ in pairs)
        assert all(count == 1 for count in counts.values())
        return dict(pairs)

    json.loads(CONTRACT.read_text(), object_pairs_hook=reject_duplicates)


def test_positive_integer_likelihood_normalizes_with_folded_first_bin() -> None:
    maximum = 200
    target = torch.arange(1, maximum + 1, dtype=torch.float64)
    censored = torch.zeros(maximum, dtype=torch.bool)
    censored[-1] = True
    log_probability = primary_observation_log_likelihood(
        location=torch.full_like(target, 0.35),
        sigma=torch.full_like(target, 0.9),
        target_dt=target,
        time_scale=1.0,
        is_right_censored=censored,
        observation_likelihood_mode=OBSERVATION_LIKELIHOOD_POSITIVE_INTEGER,
    )
    assert torch.allclose(
        torch.exp(log_probability).sum(),
        torch.tensor(1.0, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    normal = torch.distributions.Normal(0.0, 1.0)
    expected_first = normal.cdf(
        torch.tensor((math.log(1.5) - 0.35) / 0.9, dtype=torch.float64)
    )
    assert torch.allclose(torch.exp(log_probability[0]), expected_first)


def test_continuous_likelihood_is_finite_with_censoring() -> None:
    value = primary_observation_log_likelihood(
        location=torch.tensor([-5.0, 0.0, 5.0], dtype=torch.float64),
        sigma=torch.tensor([0.2, 0.8, 0.2], dtype=torch.float64),
        target_dt=torch.tensor([1.0, 3.0, 30.0], dtype=torch.float64),
        time_scale=1.0,
        is_right_censored=torch.tensor([False, False, True]),
        observation_likelihood_mode=OBSERVATION_LIKELIHOOD_CONTINUOUS,
    )
    assert value.dtype == torch.float64
    assert torch.isfinite(value).all()


def test_target_is_excluded_and_length_control_ignores_duration_values() -> None:
    dts = torch.tensor([[0.0, 1.0, 2.0, 9.0], [3.0, 4.0, 5.0, 7.0]])
    mask = torch.tensor(
        [[False, True, True, True], [True, True, True, True]]
    )
    history, history_mask, target = split_appended_target(dts, mask)
    changed = dts.clone()
    changed[:, -1] = torch.tensor([999.0, 1234.0])
    changed_history, changed_mask, changed_target = split_appended_target(
        changed, mask
    )
    assert torch.equal(history, changed_history)
    assert torch.equal(history_mask, changed_mask)
    assert not torch.equal(target, changed_target)

    base = adapter()
    control = LengthOnlyCausalLogDurationAdapter(
        log_duration_mean=math.log1p(2.0),
        log_duration_std=0.5,
        sigma_floor=0.001,
    )
    control.load_state_dict(base.state_dict())
    lengths = torch.tensor([2, 3])
    assert torch.equal(
        control.normalized_history(history[:, -3:], lengths),
        torch.zeros((2, 3, 1)),
    )
    altered = history[:, -3:].clone()
    altered[altered > 0] *= 10.0
    assert torch.equal(
        control.bounded_log_scale_residual(history[:, -3:], lengths),
        control.bounded_log_scale_residual(altered, lengths),
    )


def test_candidate_and_length_control_share_initial_state_and_exact_base() -> None:
    candidate = adapter()
    length = LengthOnlyCausalLogDurationAdapter(
        log_duration_mean=math.log1p(2.0),
        log_duration_std=0.5,
        sigma_floor=0.001,
    )
    length.load_state_dict(candidate.state_dict())
    assert sum(parameter.numel() for parameter in candidate.parameters()) == 273
    assert canonical_state_dict_sha256(candidate.state_dict()) == (
        canonical_state_dict_sha256(length.state_dict())
    )
    history = torch.tensor([[1.0, 2.0], [4.0, 0.0]])
    lengths = torch.tensor([2, 1])
    base_scale = torch.tensor([0.7, 1.2], dtype=torch.float64)
    assert torch.equal(
        candidate.adjusted_scale(base_scale, history, lengths), base_scale
    )
    assert torch.equal(length.adjusted_scale(base_scale, history, lengths), base_scale)


def test_fit_has_finite_gradients_and_restores_exact_resume(tmp_path: Path) -> None:
    dataset, cache = make_dataset_and_cache(32)
    settings = {
        "dataset": "synthetic",
        "seed": 42,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "batch_size": 8,
        "gradient_clip": 1.0,
        "epochs": 2,
        "minimum_epochs": 1,
        "early_stopping_patience": 8,
    }
    kwargs = dict(
        role="candidate",
        train_dataset=dataset,
        validation_dataset=dataset,
        train_base_cache=cache,
        validation_base_cache=cache,
        train_count=32,
        validation_count=32,
        output_dir=tmp_path,
        device=torch.device("cpu"),
        time_scale=1.0,
        censor_threshold=None,
        observation_likelihood_contract={
            "mode": OBSERVATION_LIKELIHOOD_POSITIVE_INTEGER,
            "calculation_dtype": "float64",
            "first_bin": "P(T <= 1.5)",
            "regular_bin": "P(d - 0.5 < T <= d + 0.5), d >= 2",
            "top_code": None,
            "selection_formula": "synthetic test",
        },
        contract_sha256="a" * 64,
        source_checkpoint_sha256="b" * 64,
        source_state_sha256="c" * 64,
        frozen_cache_sha256="d" * 64,
        history_statistics={"log1p_mean": math.log1p(2.0), "log1p_std": 0.5},
        source_revision="e" * 40,
        settings=settings,
    )
    first = fit_scale_module(module=adapter(), run_epoch_limit=1, **kwargs)
    assert first["status"] == "paused"
    resumed = fit_scale_module(module=adapter(), run_epoch_limit=None, **kwargs)
    assert resumed["status"] == "success"
    assert resumed["completed_epochs"] == 2
    assert math.isfinite(
        resumed["selected_validation_metrics"]["primary_proper_time_nll"]
    )
    replay = fit_scale_module(module=adapter(), run_epoch_limit=None, **kwargs)
    assert replay["selected_module_state_sha256"] == (
        resumed["selected_module_state_sha256"]
    )


def test_decision_validator_recomputes_full_fit_gates(tmp_path: Path) -> None:
    contract = load_contract_without_duplicate_keys(CONTRACT)
    dataset_spec = validate_contract(contract)["yellow_trip_hourly"]
    contract_sha256 = hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
    source_revision = "e" * 40
    summary_dir = tmp_path / "yellow_trip_hourly"
    roles = {
        "candidate": 0.50,
        "length_only_control": 0.51,
        "global_scale_control": 0.52,
    }
    role_summaries = {}
    for index, (role, nll) in enumerate(roles.items()):
        state = {"weight": torch.tensor([float(index)])}
        digest = canonical_state_dict_sha256(state)
        atomic_torch_save(
            {
                "role": role,
                "module_state_dict": state,
                "module_state_sha256": digest,
            },
            summary_dir / role / "best_validation_primary_nll_model.pt",
        )
        role_summaries[role] = {
            "status": "success",
            "held_out_test_evaluated": False,
            "selected_module_state_sha256": digest,
            "selected_validation_metrics": {
                "count": dataset_spec["expected_validation_targets"],
                "primary_proper_time_nll": nll,
            },
        }
    gates = {
        "candidate_meets_dataset_aligned_B_rule": True,
        "candidate_within_aligned_A_plus_0_01": True,
        "candidate_beats_global_control_by_0_005": True,
        "candidate_beats_length_only_control_by_0_005": True,
        "candidate_continuous_within_aligned_B_plus_0_01": True,
        "quantity_prediction_bitwise_identical": True,
        "time_median_bitwise_identical": True,
        "source_model_state_unchanged": True,
        "base_location_bitwise_identical": True,
    }
    summary = {
        "contract_id": CONTRACT_ID,
        "contract_sha256": contract_sha256,
        "source_revision": source_revision,
        "status": "success",
        "dataset": dataset_spec["dataset"],
        "seed": 42,
        "qualified_full_data": True,
        "qualified_full_fit": True,
        "source_checkpoint_sha256": dataset_spec["aligned_B_checkpoint_file_sha256"],
        "source_model_state_sha256": dataset_spec["aligned_B_model_state_sha256"],
        "source_model_state_unchanged": True,
        "train_feature_cache_sha256": dataset_spec["aligned_B_train_feature_cache_sha256"],
        "validation_feature_cache_sha256": dataset_spec["aligned_B_validation_feature_cache_sha256"],
        "quantity_prediction_bitwise_identical": True,
        "time_median_bitwise_identical": True,
        "base_location_bitwise_identical": True,
        "epoch_zero_exact_aligned_B": True,
        "candidate_length_only_initial_state_identical": True,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "base_aligned_B_validation_metrics": {"primary_proper_time_nll": 0.51},
        "aligned_A_validation_primary_nll": 0.50,
        "acceptance_gates": gates,
        "acceptance_status": "passed",
        "quantity_metrics": {"mae": dataset_spec["aligned_B_quantity_mae"]},
        **role_summaries,
    }
    summary_path = summary_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    row = validate_full_summary(
        summary_path=summary_path,
        dataset_spec=dataset_spec,
        contract_sha256=contract_sha256,
        source_revision=source_revision,
    )
    assert row["acceptance_status"] == "passed"

    summary["acceptance_gates"]["candidate_beats_length_only_control_by_0_005"] = False
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="Gate drift"):
        validate_full_summary(
            summary_path=summary_path,
            dataset_spec=dataset_spec,
            contract_sha256=contract_sha256,
            source_revision=source_revision,
        )
