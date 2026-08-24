from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = (
    PROJECT_ROOT / "paper/contracts/count_aware_model_baseline_v2.json"
)


def test_t0_is_the_primary_model_and_t1_is_an_ablation() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())

    assert contract["paper_model_name"] == "Count-aware TitanTPP"
    assert contract["internal_model_name"] == "TitanTPP-T0"
    assert contract["primary_model"]["quantity_loss"] == "mse_on_log1p_quantity"
    assert (
        contract["primary_model"]["point_prediction"]
        == "distribution_median_expm1_location"
    )
    assert contract["primary_model"]["time_head_mode"] == "legacy_clamped_rmtpp"
    assert contract["primary_model"]["lambda_tail"] == 0.0
    assert contract["ablation"]["name"] == "TitanTPP-T1"
    assert contract["ablation"]["main_model_table"] is False


def test_raf_context_and_matched_backbones_are_frozen() -> None:
    contract = json.loads(CONTRACT_PATH.read_text())

    assert contract["matched_controls"]["backbones"] == [
        "rmtpp",
        "thp",
        "nhp",
        "sahp",
        "titantpp",
    ]
    assert contract["dataset_context"]["raf_spare_parts"] == {
        "time_unit": "month",
        "lookback": 84,
        "max_sequence_length": 84,
    }
    assert contract["shared_training"]["seeds"] == [42, 52, 62]
    assert contract["shared_training"]["evaluation_scope"] == "validation_only"


def test_instacart_t0_runner_excludes_t1_and_freezes_the_factorial() -> None:
    runner_path = (
        PROJECT_ROOT
        / "simple_lab_test/search/scripts/"
        "run_count_aware_instacart_t0_e300_20260824.sh"
    )
    runner = runner_path.read_text()

    assert "--dataset-contract insta_market_basket" in runner
    assert "--model-role t0_common_control" in runner
    assert "--quantity-variants log_mse" in runner
    assert 'T0_BACKBONES="rmtpp,thp,nhp,sahp,titantpp"' in runner
    assert "--seeds 42,52,62" in runner
    assert "--lookback-weeks 52" in runner
    assert "--max-seq-len 64" in runner
    assert "t1_incumbent" not in runner
