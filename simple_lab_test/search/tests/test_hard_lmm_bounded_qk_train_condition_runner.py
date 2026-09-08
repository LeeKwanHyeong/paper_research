from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from paper.scripts.run_hard_lmm_bounded_qk_train_condition import (
    PROJECT_ROOT,
    array_sha256,
    load_and_verify_b_cache,
    sha256_file,
    validate_contract,
)


CONTRACT = PROJECT_ROOT / "paper/contracts/hard_lmm_bounded_qk_train_condition_v1.json"


def test_contract_is_train_only_and_fails_closed() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    validate_contract(payload)

    validation = copy.deepcopy(payload)
    validation["scope"]["input_splits_materialized"] = ["train", "validation"]
    with pytest.raises(ValueError, match="Non-train input admitted"):
        validate_contract(validation)

    training = copy.deepcopy(payload)
    training["scope"]["training"] = True
    with pytest.raises(ValueError, match="Forbidden operation enabled: training"):
        validate_contract(training)

    changed_salt = copy.deepcopy(payload)
    changed_salt["analysis"]["fold_salt"] = "post-hoc"
    with pytest.raises(ValueError, match="Fold salt drift"):
        validate_contract(changed_salt)

    changed_criterion = copy.deepcopy(payload)
    changed_criterion["analysis"]["required_conditions"][
        "overall_mse_worsens"
    ] = "MSE(BOUNDED_QK) - MSE(B) < 0"
    with pytest.raises(ValueError, match="Required-condition drift"):
        validate_contract(changed_criterion)


def _small_cache() -> dict[str, np.ndarray]:
    return {
        "prediction": np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        "quantity": np.asarray([1.0, 3.0, 2.0], dtype=np.float64),
        "time_nll": np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
        "history_length": np.asarray([1, 2, 3], dtype=np.int64),
        "series_index": np.asarray([0, 0, 1], dtype=np.int64),
        "target_index": np.asarray([0, 1, 2], dtype=np.int64),
        "context_end": np.asarray([0, 1, 0], dtype=np.int64),
    }


def test_reusable_b_cache_requires_file_array_population_and_state_identity(
    tmp_path: Path,
) -> None:
    cache = _small_cache()
    cache_path = tmp_path / "train_predictions.npz"
    np.savez_compressed(cache_path, **cache)

    result = {
        "status": "success",
        "phase": "train",
        "dataset": "insta_market_basket",
        "model_role": "B",
        "held_out_test_evaluated": False,
        "source_checkpoint_file_sha256": "a" * 64,
        "source_model_state_sha256": "b" * 64,
        "target_population": {
            "train": {
                "target_count": 3,
                "target_identity_sha256": "c" * 64,
                "target_quantity_sha256": "d" * 64,
            }
        },
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    contract = {
        "dataset": {
            "expected_train_targets": 3,
            "expected_train_identity_sha256": "c" * 64,
            "expected_train_quantity_sha256": "d" * 64,
        },
        "models": {
            "B": {
                "checkpoint_file_sha256": "a" * 64,
                "checkpoint_state_sha256": "b" * 64,
                "train_cache_file_sha256": sha256_file(cache_path),
                "train_result_file_sha256": sha256_file(result_path),
                "train_cache_array_sha256": {
                    name: array_sha256(value, label=name)
                    for name, value in cache.items()
                },
            }
        },
    }

    observed, audit = load_and_verify_b_cache(
        cache_path, result_path, contract
    )
    assert np.array_equal(observed["prediction"], cache["prediction"])
    assert audit["all_array_hashes_verified"] is True

    # A numerically plausible replacement cache is still rejected because its
    # file identity no longer matches the frozen prospective contract.
    changed = dict(cache)
    changed["prediction"] = cache["prediction"].copy()
    changed["prediction"][0] += 1e-9
    np.savez_compressed(cache_path, **changed)
    with pytest.raises(ValueError, match="cache file drift"):
        load_and_verify_b_cache(cache_path, result_path, contract)
