from __future__ import annotations

import pytest

from paper.scripts.count_aware_tpp_backbone.datasets import (
    DATASET_CONTRACTS,
    validate_dataset_runtime_contract,
)


@pytest.mark.parametrize(
    ("dataset_id", "lookback", "max_seq_len", "threshold", "cap"),
    [
        ("intermittent_frozen_5000", 520, 256, 46.0, 187.0),
        ("online_retail_ii", 8760, 256, 40.0, 144.0),
        ("raf_spare_parts", 84, 84, 60.0, 200.0),
        ("yellow_trip_hourly", 168, 256, 1562.0, 3449.0),
        ("insta_market_basket", 52, 64, 25.0, 35.0),
    ],
)
def test_official_dataset_contracts_accept_frozen_t1_constants(
    dataset_id: str,
    lookback: int,
    max_seq_len: int,
    threshold: float,
    cap: float,
) -> None:
    validate_dataset_runtime_contract(
        dataset_id=dataset_id,
        lookback=lookback,
        max_seq_len=max_seq_len,
        uses_tail_loss=True,
        tail_threshold=threshold,
        tail_normalization_scale=threshold,
        tail_clip_cap=cap,
        tail_huber_delta=1.0,
    )


def test_official_dataset_contract_rejects_context_drift() -> None:
    with pytest.raises(ValueError, match="lookback=8760"):
        validate_dataset_runtime_contract(
            dataset_id="online_retail_ii",
            lookback=520,
            max_seq_len=256,
            uses_tail_loss=False,
            tail_threshold=46.0,
            tail_normalization_scale=46.0,
            tail_clip_cap=187.0,
            tail_huber_delta=1.0,
        )


def test_official_dataset_contract_rejects_tail_drift() -> None:
    with pytest.raises(ValueError, match="tail contract mismatch"):
        validate_dataset_runtime_contract(
            dataset_id="raf_spare_parts",
            lookback=84,
            max_seq_len=84,
            uses_tail_loss=True,
            tail_threshold=46.0,
            tail_normalization_scale=46.0,
            tail_clip_cap=187.0,
            tail_huber_delta=1.0,
        )


def test_all_official_contracts_have_train_only_tail_constants() -> None:
    official = {
        name: contract
        for name, contract in DATASET_CONTRACTS.items()
        if contract["official_validation"]
    }
    assert set(official) == {
        "intermittent_frozen_5000",
        "online_retail_ii",
        "raf_spare_parts",
        "yellow_trip_hourly",
        "insta_market_basket",
    }
    assert all(contract["tail_contract"] is not None for contract in official.values())
