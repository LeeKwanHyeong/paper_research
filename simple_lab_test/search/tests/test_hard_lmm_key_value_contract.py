"""Preregistered contrast and safe local runner registration."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from models.Titan.common.key_value_memory import (
    KEY_VALUE_BACKBONE, KEY_VALUE_CONTRACT, KEY_VALUE_MEMORY_MODE, KEY_VALUE_ROLE,
)
from paper.scripts.count_aware_tpp_backbone.constants import (
    MODEL_ROLES, SUPPORTED_BACKBONES, T0_COMMON_BACKBONES, VARIANT, validate_model_role_contract,
)
from paper.scripts.count_aware_tpp_backbone.core import load_train_validation_frame
from paper.scripts.count_aware_tpp_backbone.training import train_one


ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = ROOT / "paper/contracts/hard_lmm_key_value_v1.json"


def test_contract_identity_and_scope():
    contract = json.loads(CONTRACT_PATH.read_text())
    assert contract["contract_id"] == KEY_VALUE_CONTRACT
    c = contract["candidate"]
    assert (c["backbone"], c["model_role"], c["memory_mode"]) == (
        KEY_VALUE_BACKBONE, KEY_VALUE_ROLE, KEY_VALUE_MEMORY_MODE)
    assert c["official_reference"] == "titantpp"
    assert c["mechanism_control"] == "titantpp_weighted_static_memory"
    assert not c["adopted_primary_model"] and not c["novelty_claim"]
    assert [name for name, allowed in contract["authorization"].items() if allowed] == ["local_implementation_tests_commit"]
    assert contract["initialization"]["additional_parameters_h64"] == 4096
    assert contract["retrieval"]["temperature"] == 1.
    assert contract["unchanged"]["prototype_count"] == 64
    assert contract["unchanged"]["topk"] == 4
    assert contract["unchanged"]["persistent_tokens"] == 16
    assert contract["unchanged"]["lambda_tail"] == 0.
    assert contract["data_boundary"]["allowed_splits"] == ["train", "validation"]
    assert not contract["data_boundary"]["future_target_in_addressing_or_assignment"]
    reuse = contract["comparison_and_reuse"]
    registry = ROOT / reuse["original_registry"]
    assert hashlib.sha256(registry.read_bytes()).hexdigest() == reuse["original_registry_sha256"]
    assert set(reuse["weighted_available"]) == {"yellow_trip_hourly", "raf_spare_parts"}
    assert set(reuse["weighted_incomplete_not_reusable"]) == {"intermittent_v2", "insta_market_basket"}
    assert not reuse["retrain_other_benchmarks"]
    future = contract["prospective_screening_not_authorized"]
    assert (future["epochs"], future["min_epochs"], future["patience"], future["seed"]) == (300, 40, 40, 42)
    assert future["per_dataset_gate"] == {
        "body_le_p95_mae_improvement_min": .05, "overall_rmse_regression_max": .02,
        "gt_p99_mae_regression_max": .02, "time_nll_absolute_increase_max": .01,
        "finite_required": True,
    }


def test_dedicated_registration_and_original_role_remains_frozen():
    assert KEY_VALUE_BACKBONE in SUPPORTED_BACKBONES
    assert KEY_VALUE_ROLE in MODEL_ROLES
    assert KEY_VALUE_BACKBONE not in T0_COMMON_BACKBONES
    assert T0_COMMON_BACKBONES == ("rmtpp", "thp", "nhp", "sahp", "titantpp")
    valid = dict(model_role=KEY_VALUE_ROLE, backbones=(KEY_VALUE_BACKBONE,),
                 quantity_variants=(VARIANT,), time_head_mode="legacy_clamped_rmtpp", lambda_tail=0.)
    validate_model_role_contract(**valid)
    for change in ({"model_role": "experimental"}, {"model_role": "t0_common_control"},
                   {"backbones": ("titantpp",)}, {"backbones": (KEY_VALUE_BACKBONE, "thp")},
                   {"quantity_variants": ("tail_shared",)}, {"lambda_tail": .1},
                   {"lambda_tail": float("nan")}, {"time_head_mode": "scaled_exact_rmtpp"}):
        with pytest.raises(ValueError, match="Separate-key"):
            validate_model_role_contract(**(valid | change))


def test_runner_registration_uses_filtered_materialization():
    # Check the condition, not just the presence of a loader import.
    import ast
    from paper.scripts import run_count_aware_tpp_backbone_control as runner
    tree = ast.parse(Path(runner.__file__).read_text())
    branches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                and "KEY_VALUE_ROLE" in ast.unparse(node.test)]
    assert len(branches) == 1
    branch = branches[0]
    assert "load_train_validation_frame(args.data)" in ast.unparse(branch.body[0])
    assert "pl.read_parquet" in ast.unparse(branch.orelse[0])


def test_synthetic_split_filter_does_not_materialize_heldout(tmp_path):
    path = tmp_path / "synthetic.parquet"
    pl.DataFrame({"oper_part_no": ["x"] * 3, "seq": [1, 2, 3],
                  "chronological_split": ["train", "validation", "test"],
                  "demand_qty": [2., 3., float("nan")]}).write_parquet(path)
    frame = load_train_validation_frame(path)
    assert frame["chronological_split"].to_list() == ["train", "validation"]
    assert frame["demand_qty"].is_finite().all()


@pytest.mark.parametrize("force", [False, True])
def test_existing_candidate_artifact_rejected_before_any_training(tmp_path, force):
    run_dir = tmp_path / "runs" / KEY_VALUE_BACKBONE / VARIANT / "seed_42"
    run_dir.mkdir(parents=True)
    sentinel = run_dir / "keep.txt"
    sentinel.write_text("existing evidence")
    with pytest.raises(FileExistsError, match="fresh run"):
        train_one(args=SimpleNamespace(output_dir=tmp_path, force_rerun=force), frame=None,
                  quantity_contract={}, interface_meta={}, backbone=KEY_VALUE_BACKBONE,
                  quantity_variant=VARIANT, seed=42)
    assert sentinel.read_text() == "existing evidence"
    assert list(run_dir.iterdir()) == [sentinel]
