from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import sys

import pytest
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.TPPs.CausalLogDurationAdapter import CausalLogDurationAdapter
from paper.scripts.run_hard_lmm_causal_duration_adapter import (
    CONTRACT_ID,
    SELECTED_CHECKPOINT_NAME,
    FrozenBaseTimeCache,
    GlobalScaleControl,
    adapted_scale,
    canonical_frozen_time_median,
    compact_masked_history,
    continuous_lognormal_log_likelihood,
    fit_scale_module,
    interval_lognormal_log_likelihood,
    split_appended_target,
    validate_contract,
)
from simple_lab_test.search.common.runner import (
    canonical_state_dict_sha256,
    torch_load_checkpoint,
)


CONTRACT_PATH = ROOT / "paper/contracts/hard_lmm_causal_duration_adapter_v1.json"


class SyntheticCanonicalDataset(Dataset):
    def __init__(self, rows: list[tuple[list[float], float]]) -> None:
        self.rows = rows
        self.max_len = max(len(history) + 1 for history, _ in rows) + 2

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        history, target = self.rows[index]
        active = torch.tensor([*history, target], dtype=torch.float32)
        pad = self.max_len - active.numel()
        dts = torch.cat((torch.zeros(pad), active))
        mask = torch.cat(
            (torch.zeros(pad, dtype=torch.bool), torch.ones_like(active, dtype=torch.bool))
        )
        return {
            "marks": torch.zeros(self.max_len, dtype=torch.long),
            "dts": dts,
            "mask": mask,
            "part_idx": torch.tensor(index, dtype=torch.long),
            "values": torch.zeros(self.max_len),
        }


def make_dataset_and_cache(count: int, *, offset: int) -> tuple[Dataset, FrozenBaseTimeCache]:
    rows: list[tuple[list[float], float]] = []
    locations = []
    scales = []
    targets = []
    for index in range(count):
        recent = 1.0 + float((index + offset) % 4)
        history = [1.0, recent] if index % 2 else [recent]
        # The conditional spread changes with history, which gives the GRU a
        # real scale signal while leaving the base location fixed.
        target = 2.0 if recent <= 2.0 else (1.0 if index % 3 else 8.0)
        rows.append((history, target))
        locations.append(math.log(2.0))
        scales.append(0.8)
        targets.append(target)
    dataset = SyntheticCanonicalDataset(rows)
    target_tensor = torch.tensor(targets, dtype=torch.float32)
    base = FrozenBaseTimeCache(
        location=torch.tensor(locations, dtype=torch.float64),
        sigma=torch.tensor(scales, dtype=torch.float64),
        median=torch.full((count,), 2.0, dtype=torch.float64),
        target_dt=target_tensor,
    )
    base.validate()
    return dataset, base


def build_adapter() -> CausalLogDurationAdapter:
    torch.manual_seed(42)
    return CausalLogDurationAdapter(
        log_duration_mean=math.log(3.0),
        log_duration_std=0.5,
        sigma_floor=0.001,
    )


def test_contract_freezes_single_scale_only_hypothesis_and_taxi_stop() -> None:
    def reject_duplicate_keys(pairs):
        counts = Counter(name for name, _ in pairs)
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate JSON keys: {duplicates}")
        return dict(pairs)

    contract = json.loads(
        CONTRACT_PATH.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )
    datasets = validate_contract(contract)
    assert contract["contract_id"] == CONTRACT_ID
    assert contract["candidate"]["single_hypothesis"] is True
    assert contract["candidate"]["scale_only"] is True
    assert contract["execution_policy"]["taxi_first_stop"] is True
    assert contract["scope"]["additional_seeds"] is False
    assert set(datasets) == {
        "intermittent_frozen_5000",
        "yellow_trip_hourly",
        "insta_market_basket",
    }
    gates = contract["acceptance"]["structured_gates"]
    assert gates["maximum_candidate_minus_B_continuous_nll"] == 0.0
    assert gates["maximum_candidate_minus_A_continuous_nll"] == 0.01
    assert gates["minimum_candidate_improvement_over_global_control_continuous_nll"] == 0.005


def test_appended_target_is_excluded_and_target_perturbation_is_invariant() -> None:
    dts = torch.tensor(
        [[0.0, 1.0, 2.0, 9.0], [3.0, 4.0, 5.0, 7.0]],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [[False, True, True, True], [True, True, True, True]]
    )
    history, history_mask, target = split_appended_target(dts, mask)
    compact, lengths = compact_masked_history(history, history_mask)
    assert torch.equal(target, torch.tensor([9.0, 7.0]))
    assert torch.equal(lengths, torch.tensor([2, 3]))
    assert torch.equal(compact[0], torch.tensor([1.0, 2.0, 0.0]))
    assert torch.equal(compact[1], torch.tensor([3.0, 4.0, 5.0]))

    changed = dts.clone()
    changed[:, -1] = torch.tensor([999.0, 1234.0])
    changed_history, changed_mask, changed_target = split_appended_target(changed, mask)
    changed_compact, changed_lengths = compact_masked_history(
        changed_history, changed_mask
    )
    assert torch.equal(compact, changed_compact)
    assert torch.equal(lengths, changed_lengths)
    assert not torch.equal(target, changed_target)


def test_shared_adapter_zero_init_preserves_scale_location_and_median_exactly() -> None:
    adapter = build_adapter()
    assert sum(parameter.numel() for parameter in adapter.parameters()) == 273
    history = torch.tensor([[1.0, 2.0], [4.0, 0.0]], dtype=torch.float32)
    lengths = torch.tensor([2, 1], dtype=torch.long)
    base_location = torch.tensor([0.2, -0.4], dtype=torch.float64)
    base_scale = torch.tensor([0.7, 1.2], dtype=torch.float64)
    location, scale = adapter.adjusted_location_scale(
        base_location, base_scale, history, lengths
    )
    assert torch.equal(location, base_location)
    assert torch.equal(scale, base_scale)
    assert torch.equal(
        adapter.predict_median(base_location, time_scale=3.0),
        3.0 * torch.exp(base_location),
    )
    residual = adapter.bounded_log_scale_residual(history, lengths)
    assert torch.equal(residual, torch.zeros_like(residual))


def test_canonical_median_keeps_source_bits_across_formula_ulp_difference() -> None:
    _, base = make_dataset_and_cache(3, offset=0)
    implied = torch.exp(base.location)
    canonical = torch.nextafter(implied, torch.full_like(implied, math.inf))
    cache = FrozenBaseTimeCache(
        location=base.location,
        sigma=base.sigma,
        median=canonical,
        target_dt=base.target_dt,
    )
    observed, maximum_error = canonical_frozen_time_median(
        base_cache=cache,
        count=cache.count,
        time_scale=1.0,
    )
    assert torch.equal(observed, canonical)
    assert 0.0 < maximum_error <= 1e-12

    invalid = FrozenBaseTimeCache(
        location=base.location,
        sigma=base.sigma,
        median=canonical + 1e-4,
        target_dt=base.target_dt,
    )
    with pytest.raises(ValueError, match="inconsistent"):
        canonical_frozen_time_median(
            base_cache=invalid,
            count=invalid.count,
            time_scale=1.0,
        )


def test_global_control_uses_floor_preserving_bounded_scale_formula() -> None:
    control = GlobalScaleControl(sigma_floor=0.001)
    history = torch.tensor([[1.0], [2.0]])
    lengths = torch.ones(2, dtype=torch.long)
    base = torch.tensor([0.5, 1.5], dtype=torch.float64)
    assert torch.equal(control.adjusted_scale(base, history, lengths), base)
    with torch.no_grad():
        control.raw_bias.fill_(1000.0)
    high = control.adjusted_scale(base, history, lengths)
    assert torch.allclose(high, 0.001 + (base - 0.001) * 10.0)
    with torch.no_grad():
        control.raw_bias.fill_(-1000.0)
    low = control.adjusted_scale(base, history, lengths)
    assert torch.allclose(low, 0.001 + (base - 0.001) * 0.1)


def test_continuous_and_interval_likelihood_are_finite_in_both_tails() -> None:
    location = torch.tensor([-8.0, 0.0, 8.0, 0.5], dtype=torch.float64)
    sigma = torch.tensor([0.1, 0.8, 0.1, 1.1], dtype=torch.float64)
    target = torch.tensor([1.0, 3.0, 200.0, 30.0], dtype=torch.float64)
    censored = torch.tensor([False, False, False, True])
    continuous = continuous_lognormal_log_likelihood(
        location=location,
        sigma=sigma,
        target_dt=target,
        time_scale=1.0,
        is_right_censored=censored,
    )
    interval = interval_lognormal_log_likelihood(
        location=location,
        sigma=sigma,
        target_dt=target,
        time_scale=1.0,
        is_right_censored=censored,
    )
    assert continuous.dtype == torch.float64
    assert interval.dtype == torch.float64
    assert torch.isfinite(continuous).all()
    assert torch.isfinite(interval).all()


def fit_kwargs(
    *,
    module: CausalLogDurationAdapter,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    train_cache: FrozenBaseTimeCache,
    validation_cache: FrozenBaseTimeCache,
    output_dir: Path,
):
    settings = {
        "dataset": "synthetic",
        "seed": 42,
        "optimizer": "AdamW",
        "learning_rate": 0.003,
        "weight_decay": 0.0,
        "batch_size": 4,
        "gradient_clip": 1.0,
        "epochs": 3,
        "minimum_epochs": 99,
        "early_stopping_patience": 99,
        "scheduler": None,
        "shared_across_all_datasets": True,
        "dataset_specific_hyperparameters": False,
    }
    return {
        "role": "candidate",
        "module": module,
        "train_dataset": train_dataset,
        "validation_dataset": validation_dataset,
        "train_base_cache": train_cache,
        "validation_base_cache": validation_cache,
        "train_count": len(train_dataset),
        "validation_count": len(validation_dataset),
        "output_dir": output_dir,
        "device": torch.device("cpu"),
        "time_scale": 1.0,
        "censor_threshold": None,
        "contract_sha256": "c" * 64,
        "source_checkpoint_sha256": "f" * 64,
        "source_state_sha256": "s" * 64,
        "frozen_cache_sha256": "a" * 64 + ":" + "b" * 64,
        "history_statistics": {
            "token_count": 40,
            "log1p_mean": math.log(3.0),
            "log1p_std": 0.5,
        },
        "source_revision": "d" * 40,
        "settings": settings,
    }


def test_resume_matches_uninterrupted_adapter_training_exactly(tmp_path: Path) -> None:
    train_dataset, train_cache = make_dataset_and_cache(19, offset=0)
    validation_dataset, validation_cache = make_dataset_and_cache(11, offset=1)
    full = fit_scale_module(
        **fit_kwargs(
            module=build_adapter(),
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            train_cache=train_cache,
            validation_cache=validation_cache,
            output_dir=tmp_path / "full",
        )
    )
    resumed_dir = tmp_path / "resumed"
    paused = fit_scale_module(
        **fit_kwargs(
            module=build_adapter(),
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            train_cache=train_cache,
            validation_cache=validation_cache,
            output_dir=resumed_dir,
        ),
        run_epoch_limit=1,
    )
    assert paused["status"] == "paused"
    resumed = fit_scale_module(
        **fit_kwargs(
            module=build_adapter(),
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            train_cache=train_cache,
            validation_cache=validation_cache,
            output_dir=resumed_dir,
        )
    )
    assert resumed["status"] == "success"
    assert resumed["history"] == full["history"]
    full_checkpoint = torch_load_checkpoint(
        tmp_path / "full" / SELECTED_CHECKPOINT_NAME, map_location="cpu"
    )
    resumed_checkpoint = torch_load_checkpoint(
        resumed_dir / SELECTED_CHECKPOINT_NAME, map_location="cpu"
    )
    assert full_checkpoint["module_state_sha256"] == resumed_checkpoint[
        "module_state_sha256"
    ]
    assert canonical_state_dict_sha256(
        full_checkpoint["module_state_dict"]
    ) == canonical_state_dict_sha256(resumed_checkpoint["module_state_dict"])


def test_adapter_gradient_is_isolated_and_finite() -> None:
    adapter = build_adapter()
    history_dt = torch.tensor([[1.0, 2.0], [3.0, 1.0]], dtype=torch.float32)
    history_mask = torch.ones_like(history_dt, dtype=torch.bool)
    base_sigma = torch.tensor([0.8, 1.1], dtype=torch.float64)
    target = torch.tensor([1.0, 5.0], dtype=torch.float64)
    location = torch.zeros(2, dtype=torch.float64)
    sigma = adapted_scale(adapter, base_sigma, history_dt, history_mask)
    loss = -continuous_lognormal_log_likelihood(
        location=location,
        sigma=sigma,
        target_dt=target,
        time_scale=1.0,
        is_right_censored=torch.zeros(2, dtype=torch.bool),
    ).mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in adapter.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in adapter.parameters())
    assert adapter.scale_projection.weight.grad.abs().sum() > 0.0
