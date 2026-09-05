"""Independent candidate identity, opt-in runner and checkpoint boundaries."""

import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.TPPs.CountAwareFactory import build_count_aware_model, validate_checkpoint_route
from models.Titan.common.elapsed_age import (
    ELAPSED_AGE_BACKBONE, ELAPSED_AGE_BETA_KEY, ELAPSED_AGE_CONTRACT,
    ELAPSED_AGE_MEMORY_MODE, ELAPSED_AGE_ROLE,
)
from models.Titan.common.key_value_memory import KEY_VALUE_BACKBONE, KEY_VALUE_CONTRACT
from paper.scripts.count_aware_tpp_backbone.constants import (
    MODEL_ROLES, SUPPORTED_BACKBONES, T0_COMMON_BACKBONES, VARIANT, validate_model_role_contract,
)
from paper.scripts.count_aware_tpp_backbone.training import train_one


ROOT = Path(__file__).resolve().parents[3]


def build(backbone=ELAPSED_AGE_BACKBONE):
    return build_count_aware_model(backbone, hidden_dim=16, train_log_mean=1.5, max_seq_len=8)


def checkpoint():
    model, metadata = build()
    return model, {"backbone": ELAPSED_AGE_BACKBONE, "variant": VARIANT,
                   "encoder_config": metadata, "model_state_dict": model.state_dict(),
                   "evaluation_scope": "validation_only", "held_out_test_evaluated": False}


def test_contract_identity_geometry_and_preserved_reference_hashes():
    contract = json.loads((ROOT / "paper/contracts/hard_lmm_elapsed_age_v1.json").read_text())
    assert contract["contract_id"] == ELAPSED_AGE_CONTRACT
    candidate = contract["candidate"]
    assert (candidate["backbone"], candidate["model_role"], candidate["memory_mode"]) == (
        ELAPSED_AGE_BACKBONE, ELAPSED_AGE_ROLE, ELAPSED_AGE_MEMORY_MODE)
    assert candidate["mechanism_reference"] == KEY_VALUE_BACKBONE
    assert candidate["new_parameters"] == 8
    assert not candidate["adopted_primary_model"] and not candidate["novelty_claim"]
    assert contract["attention"]["parameter_state_key"] == ELAPSED_AGE_BETA_KEY
    assert contract["attention"]["beta_shape"] == [2, 4]
    assert contract["attention"]["initial_value"] == 0.
    assert contract["attention"]["persistent_bias"] == 0.
    assert not contract["authorization"]["performance_screening"]
    assert not contract["authorization"]["held_out_test"]
    assert not contract["authorization"]["push"]
    unchanged = contract["unchanged"]
    assert [unchanged[key] for key in ("n_layers", "n_heads", "prototype_count", "topk")] == [2, 4, 64, 4]
    assert unchanged["quantity_variant"] == VARIANT
    assert unchanged["lambda_tail"] == 0.
    assert contract["numerical_tolerances"]["cpu_initial_identity"] == {"rtol": 0, "atol": 0}
    for key, value in contract["references"].items():
        if not key.endswith("_sha256"):
            assert hashlib.sha256((ROOT / value).read_bytes()).hexdigest() == contract["references"][key + "_sha256"]


def test_candidate_opt_in_and_original_common_role_unchanged():
    assert ELAPSED_AGE_BACKBONE in SUPPORTED_BACKBONES
    assert ELAPSED_AGE_ROLE in MODEL_ROLES
    assert ELAPSED_AGE_BACKBONE not in T0_COMMON_BACKBONES
    assert T0_COMMON_BACKBONES == ("rmtpp", "thp", "nhp", "sahp", "titantpp")
    valid = dict(model_role=ELAPSED_AGE_ROLE, backbones=(ELAPSED_AGE_BACKBONE,),
                 quantity_variants=(VARIANT,), time_head_mode="legacy_clamped_rmtpp", lambda_tail=0.)
    validate_model_role_contract(**valid)
    for mutation in ({"model_role": "experimental"}, {"model_role": "t0_common_control"},
                     {"backbones": (KEY_VALUE_BACKBONE,)}, {"backbones": (ELAPSED_AGE_BACKBONE, "thp")},
                     {"quantity_variants": ("tail_shared",)}, {"lambda_tail": .1},
                     {"lambda_tail": float("nan")}, {"time_head_mode": "scaled_exact_rmtpp"}):
        with pytest.raises(ValueError, match="Elapsed-age"):
            validate_model_role_contract(**(valid | mutation))


def test_runner_candidate_uses_train_validation_filtered_materialization():
    from paper.scripts import run_count_aware_tpp_backbone_control as runner
    tree = ast.parse(Path(runner.__file__).read_text())
    branches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                and "ELAPSED_AGE_ROLE" in ast.unparse(node.test)]
    assert len(branches) == 1
    assert "load_train_validation_frame(args.data)" in ast.unparse(branches[0].body[0])
    assert "pl.read_parquet" in ast.unparse(branches[0].orelse[0])


@pytest.mark.parametrize("force", [False, True])
def test_existing_candidate_directory_rejected_before_training_even_with_force(tmp_path, force):
    run_dir = tmp_path / "runs" / ELAPSED_AGE_BACKBONE / VARIANT / "seed_42"
    run_dir.mkdir(parents=True)
    sentinel = run_dir / "keep.txt"
    sentinel.write_text("existing evidence")
    with pytest.raises(FileExistsError, match="fresh run"):
        train_one(args=SimpleNamespace(output_dir=tmp_path, force_rerun=force), frame=None,
                  quantity_contract={}, interface_meta={}, backbone=ELAPSED_AGE_BACKBONE,
                  quantity_variant=VARIANT, seed=42)
    assert sentinel.read_text() == "existing evidence"
    assert list(run_dir.iterdir()) == [sentinel]


def test_checkpoint_candidate_metadata_and_bidirectional_route_rejection():
    model, payload = checkpoint()
    validate_checkpoint_route(payload, ELAPSED_AGE_BACKBONE)
    metadata = payload["encoder_config"]
    assert metadata["backbone_contract_id"] == ELAPSED_AGE_CONTRACT
    assert metadata["static_retrieval_contract_id"] == KEY_VALUE_CONTRACT
    assert metadata["additional_parameter_count"] == 64 * 16 + 8
    for backbone in ("titantpp", KEY_VALUE_BACKBONE, "thp"):
        with pytest.raises(ValueError, match="Elapsed-age"):
            validate_checkpoint_route(payload, backbone)
    reference, reference_metadata = build(KEY_VALUE_BACKBONE)
    old = payload | {"backbone": KEY_VALUE_BACKBONE, "encoder_config": reference_metadata,
                     "model_state_dict": reference.state_dict()}
    validate_checkpoint_route(old, KEY_VALUE_BACKBONE)
    with pytest.raises(ValueError):
        validate_checkpoint_route(old, ELAPSED_AGE_BACKBONE)
    with pytest.raises(RuntimeError):
        reference.load_state_dict(model.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        model.load_state_dict(reference.state_dict(), strict=True)
    relabeled = payload | {"backbone": KEY_VALUE_BACKBONE, "encoder_config": reference_metadata}
    with pytest.raises(ValueError, match="Elapsed-age"):
        validate_checkpoint_route(relabeled, KEY_VALUE_BACKBONE)


@pytest.mark.parametrize("mutation", [
    {"backbone": "titantpp"}, {"variant": "tail_shared"}, {"evaluation_scope": "test"},
    {"held_out_test_evaluated": True}, {"encoder_config": {}}, {"model_state_dict": {}},
])
def test_checkpoint_rejects_identity_scope_and_state_drift(mutation):
    _, payload = checkpoint()
    with pytest.raises(ValueError):
        validate_checkpoint_route(payload | mutation, ELAPSED_AGE_BACKBONE)


@pytest.mark.parametrize("key,value", [
    ("elapsed_age_contract_id", "other"), ("elapsed_age_beta_shape", [1, 8]),
    ("elapsed_age_parameter_count", 0), ("elapsed_age_persistent_key_bias", 1.),
    ("elapsed_age_normalization", "final_window"), ("static_key_value_tied", True),
    ("static_retrieval_temperature", 2.), ("lmm_topk", 3),
    ("time_head", None), ("time_memory_route", "local"),
])
def test_checkpoint_rejects_geometry_retrieval_and_head_drift(key, value):
    _, payload = checkpoint()
    bad = copy.deepcopy(payload)
    bad["encoder_config"][key] = value
    with pytest.raises(ValueError):
        validate_checkpoint_route(bad, ELAPSED_AGE_BACKBONE)


@pytest.mark.parametrize("state_key", ["model_state_dict", "best_state_dict"])
@pytest.mark.parametrize("bad_beta", [None, torch.zeros(8), torch.full((2, 4), float("nan"))])
def test_checkpoint_rejects_invalid_beta_in_both_saved_states(state_key, bad_beta):
    _, payload = checkpoint()
    payload[state_key] = dict(payload["model_state_dict"])
    payload[state_key][ELAPSED_AGE_BETA_KEY] = bad_beta
    with pytest.raises(ValueError, match="finite beta"):
        validate_checkpoint_route(payload, ELAPSED_AGE_BACKBONE)


@pytest.mark.parametrize("override", [
    {"quantity_variant": "count_only_lognormal_k1"},
    {"time_head_mode": "scaled_exact_rmtpp"}, {"lambda_tail": .1},
])
def test_factory_rejects_objective_drift(override):
    with pytest.raises(ValueError, match="Elapsed-age"):
        build_count_aware_model(ELAPSED_AGE_BACKBONE, hidden_dim=16, train_log_mean=1.5,
                                max_seq_len=8, **override)
