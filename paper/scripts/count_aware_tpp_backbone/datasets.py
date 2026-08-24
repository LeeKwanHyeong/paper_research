"""Frozen dataset contracts for matched count-aware validation runs."""

from __future__ import annotations

import math
from typing import Any


DATASET_CONTRACTS: dict[str, dict[str, Any]] = {
    "intermittent_frozen_5000": {
        "data_sha256": "85d1fe3ade3ae5a90241018e99a3e9463828d5ba35bc374b56def0168ffffc3f",
        "split_manifest_sha256": "393158a54a8ca703dbf7e9311b9dff6d2825ef737e3e3de1c30a1f3ff64c1c04",
        "time_unit": "week",
        "lookback": 520,
        "max_seq_len": 256,
        "tail_contract": {
            "threshold": 46.0,
            "normalization_scale": 46.0,
            "clip_cap": 187.0,
            "huber_delta": 1.0,
        },
        "official_validation": True,
    },
    "online_retail_ii": {
        "data_sha256": "4ac40e015c34314632d1f8b81c282c7b5b3309b40a180c9e38a38b8e01be2dc8",
        "split_manifest_sha256": "6ff3421707702c98fa7f99df708162bb4121778ac7408648c0a19507b9c95734",
        "time_unit": "hour",
        "lookback": 24 * 365,
        "max_seq_len": 256,
        "tail_contract": {
            "threshold": 40.0,
            "normalization_scale": 40.0,
            "clip_cap": 144.0,
            "huber_delta": 1.0,
        },
        "official_validation": True,
    },
    "raf_spare_parts": {
        "data_sha256": "439cc3bdd2f969f1121b149de5aa9ece4a7e1ce0c6da9e734a7687c54989780d",
        "split_manifest_sha256": "cd2458aa3809014aa74b8a121572a74e3d86dc520a690335d79a1e5150899e27",
        "time_unit": "month",
        "lookback": 84,
        "max_seq_len": 84,
        "tail_contract": {
            "threshold": 60.0,
            "normalization_scale": 60.0,
            "clip_cap": 200.0,
            "huber_delta": 1.0,
        },
        "official_validation": True,
    },
    "yellow_trip_hourly": {
        "data_sha256": "b47e98e9fdb75d4274a18e3f8a5d8f463418a1d56a6db4db7d9b834c9d89ca46",
        "split_manifest_sha256": "4a005d4a77a89f7ca793d8de56afb9267a3ca4a5e60c53e09465c0494d60ed85",
        "time_unit": "hour",
        "lookback": 168,
        "max_seq_len": 256,
        "tail_contract": {
            "threshold": 1562.0,
            "normalization_scale": 1562.0,
            "clip_cap": 3449.0,
            "huber_delta": 1.0,
        },
        "official_validation": True,
    },
    "insta_market_basket": {
        "data_sha256": "06296e48f5ca6c7e0c849f4b4a3c6d54a968ef892754f59369caf1d378424ef2",
        "split_manifest_sha256": "6c6cdd41f847878fbb405b73dfa038fbb7a88ad53df6843b0cc9e64531a8b71d",
        "time_unit": "day",
        "lookback": 52,
        "max_seq_len": 64,
        "tail_contract": {
            "threshold": 25.0,
            "normalization_scale": 25.0,
            "clip_cap": 35.0,
            "huber_delta": 1.0,
        },
        "official_validation": True,
    },
}


def validate_dataset_runtime_contract(
    *,
    dataset_id: str,
    lookback: int,
    max_seq_len: int,
    uses_tail_loss: bool,
    tail_threshold: float,
    tail_normalization_scale: float,
    tail_clip_cap: float,
    tail_huber_delta: float,
) -> None:
    """Reject context or train-only tail constants that drift from the registry."""
    contract = DATASET_CONTRACTS[dataset_id]
    expected_lookback = contract["lookback"]
    if expected_lookback is not None and lookback != expected_lookback:
        raise ValueError(
            f"{dataset_id} requires lookback={expected_lookback} "
            f"{contract['time_unit']} units"
        )
    if max_seq_len != contract["max_seq_len"]:
        raise ValueError(
            f"{dataset_id} requires max_seq_len={contract['max_seq_len']}"
        )
    if not uses_tail_loss:
        return
    expected_tail = contract["tail_contract"]
    if expected_tail is None:
        raise ValueError(f"{dataset_id} has no frozen T1 tail contract")
    observed = {
        "threshold": tail_threshold,
        "normalization_scale": tail_normalization_scale,
        "clip_cap": tail_clip_cap,
        "huber_delta": tail_huber_delta,
    }
    mismatches = {
        key: {"expected": expected_tail[key], "observed": value}
        for key, value in observed.items()
        if not math.isclose(
            float(value),
            float(expected_tail[key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    }
    if mismatches:
        raise ValueError(f"{dataset_id} tail contract mismatch: {mismatches}")


__all__ = ["DATASET_CONTRACTS", "validate_dataset_runtime_contract"]
