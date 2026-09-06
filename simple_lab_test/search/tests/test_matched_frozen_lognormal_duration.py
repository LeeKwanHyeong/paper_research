from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTPP import LOG_MSE_VARIANT
from paper.scripts.run_hard_lmm_frozen_lognormal_duration import (
    FrozenFeatureCache,
    build_candidate_from_selected_checkpoint,
    build_frozen_lognormal_candidate,
    fit_time_head_from_cache,
    state_partition_sha256,
)
from paper.scripts.run_matched_frozen_lognormal_duration import (
    CONTRACT_ID,
    build_source_model,
    validate_contract,
    validate_source_checkpoint,
)
from paper.scripts.run_hard_lmm_time_head_refit import sha256_file
from simple_lab_test.search.common.runner import canonical_state_dict_sha256


CONTRACT_PATH = ROOT / "paper/contracts/matched_frozen_lognormal_duration_v1.json"
INPUT_ROOT = Path(
    os.environ.get("MATCHED_FROZEN_LOGNORMAL_INPUT_ROOT", str(ROOT))
).resolve()


def load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def available_rows():
    contract = load_contract()
    for row in contract["datasets"]:
        for role in ("A", "rmtpp", "thp"):
            source = row["sources"][role]
            if source["available"]:
                yield row, role, source


def test_contract_has_eight_new_rows_reuses_B_and_exposes_missing_rmtpp():
    contract = load_contract()
    datasets = validate_contract(contract)
    assert len(contract["execution_order"]) == 8
    assert [row["model_role"] for row in contract["execution_order"]].count("A") == 3
    assert [row["model_role"] for row in contract["execution_order"]].count("thp") == 3
    assert [row["model_role"] for row in contract["execution_order"]].count("rmtpp") == 2
    for row in datasets.values():
        assert row["sources"]["B"]["execution"] == "reuse_c04d32b"
    missing = datasets["intermittent_frozen_5000"]["sources"]["rmtpp"]
    assert missing["available"] is False
    assert missing["checkpoint_path"] is None
    assert contract["acceptance"]["four_way_claim"].startswith("prohibited")
    assert (
        contract["identity_and_stability"]["quantity_mae_and_rmse_required"]
        == "calculation-identical to each pinned source checkpoint"
    )


def test_contract_rejects_silently_admitting_missing_rmtpp_and_heldout():
    contract = load_contract()
    admitted = copy.deepcopy(contract)
    admitted["datasets"][0]["sources"]["rmtpp"]["available"] = True
    with pytest.raises(
        ValueError, match="silently admitted|Checkpoint hash|Execution drift"
    ):
        validate_contract(admitted)
    heldout = copy.deepcopy(contract)
    heldout["scope"]["held_out_test"] = True
    with pytest.raises(ValueError, match="Held-out"):
        validate_contract(heldout)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("scope", "additional_seeds", True, "Additional seeds"),
        (
            "comparison_policy",
            "B_result_policy",
            "refit B",
            "B reuse policy",
        ),
        (
            "comparison_policy",
            "cross_dataset_nll_aggregation",
            True,
            "Cross-dataset",
        ),
        ("time_head", "calculation_dtype", "float32", "calculation_dtype"),
    ],
)
def test_local_runner_rejects_comparison_policy_drift(
    section, field, value, message
):
    changed = copy.deepcopy(load_contract())
    changed[section][field] = value
    with pytest.raises(ValueError, match=message):
        validate_contract(changed)


def test_local_runner_rejects_censor_count_drift():
    changed = copy.deepcopy(load_contract())
    changed["datasets"][2]["expected_train_censored_targets"] -= 1
    with pytest.raises(ValueError, match="Train censor count"):
        validate_contract(changed)


def test_all_eight_source_checkpoints_match_pinned_file_state_and_lineage():
    assert sum(1 for _ in available_rows()) == 8
    for _, _, source in available_rows():
        path = INPUT_ROOT / source["checkpoint_path"]
        assert sha256_file(path) == source["checkpoint_file_sha256"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_source_checkpoint(payload, source_spec=source)
        assert (
            canonical_state_dict_sha256(payload["model_state_dict"])
            == source["checkpoint_state_sha256"]
        )


@pytest.mark.parametrize(
    ("dataset", "role"),
    [
        ("intermittent_frozen_5000", "A"),
        ("intermittent_frozen_5000", "thp"),
        ("yellow_trip_hourly", "rmtpp"),
    ],
)
def test_generic_candidate_preserves_each_backbone_states_and_quantity(dataset, role):
    row = validate_contract(load_contract())[dataset]
    source_spec = row["sources"][role]
    payload = torch.load(
        ROOT / source_spec["checkpoint_path"], map_location="cpu", weights_only=False
    )
    source = build_source_model(payload, max_seq_len=row["max_sequence_length"])
    stats = {
        "statistics_source_split": "train",
        "target_count": row["expected_train_targets"],
        "time_scale": row["train_time_scale"],
        "target_log_scaled_mean": row["train_log_scaled_mean"],
        "target_log_scaled_std": row["train_log_scaled_std"],
    }
    candidate, _, metadata = build_frozen_lognormal_candidate(
        payload,
        train_time_statistics=stats,
        time_sigma_floor=0.001,
        source_backbone=source_spec["backbone"],
        source_variant=source_spec["quantity_variant"],
        max_seq_len=row["max_sequence_length"],
        training_stage="matched_frozen_representation_duration_refit",
    )
    assert state_partition_sha256(
        candidate.state_dict(), time_head=False
    ) == state_partition_sha256(payload["model_state_dict"], time_head=False)
    assert metadata["source_non_time_state_sha256"] == state_partition_sha256(
        payload["model_state_dict"], time_head=False
    )
    dts = torch.tensor([[1.0, 2.0, 3.0, 1.0], [2.0, 1.0, 4.0, 1.0]])
    quantities = torch.tensor([[2.0, 3.0, 1.0, 0.0], [1.0, 4.0, 2.0, 0.0]])
    mask = torch.ones_like(dts, dtype=torch.bool)
    writes = mask.clone()
    writes[:, -1] = False
    with torch.no_grad():
        source_states = source.encode_task_states(
            dts, quantities, mask, memory_write_mask=writes
        )
        candidate_states = candidate.encode_task_states(
            dts, quantities, mask, memory_write_mask=writes
        )
    for source_hidden, candidate_hidden in zip(source_states, candidate_states):
        assert torch.equal(source_hidden, candidate_hidden)
        assert torch.equal(
            source.predict_quantity(source_hidden)[1],
            candidate.predict_quantity(candidate_hidden)[1],
        )


def synthetic_payload(backbone: str = "rmtpp"):
    torch.manual_seed(31)
    model, encoder = build_count_aware_model(
        backbone,
        hidden_dim=64,
        train_log_mean=1.2,
        train_log_std=0.7,
        max_seq_len=16,
        quantity_variant=LOG_MSE_VARIANT,
        lambda_tail=0.0,
        time_head_mode="legacy_clamped_rmtpp",
    )
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    payload = {
        "backbone": backbone,
        "variant": LOG_MSE_VARIANT,
        "seed": 42,
        "selection": "best_validation_joint_objective",
        "source_revision": "a" * 40,
        "source_revision_history": ["a" * 40],
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "encoder_config": encoder,
        "interface_meta": {
            "mode": "mark_free_count_aware_log_regression",
            "train_target_mean": 1.2,
            "train_target_std": 0.7,
            "target_quantity_masked_from_history": True,
        },
        "model_state_dict": state,
        "model_state_sha256": canonical_state_dict_sha256(state),
    }
    return model, payload


def synthetic_cache(source, count: int, *, quantity: bool):
    generator = torch.Generator().manual_seed(407 + count)
    hidden = torch.randn(count, 64, generator=generator)
    target_dt = torch.exp(0.2 + 0.4 * hidden[:, 0]).float()
    if not quantity:
        return FrozenFeatureCache(time_hidden=hidden, target_dt=target_dt)
    quantity_hidden = torch.randn(count, 64, generator=generator)
    target_quantity = torch.arange(count).remainder(7).float()
    with torch.no_grad():
        prediction = source.predict_quantity(quantity_hidden)[1]
    return FrozenFeatureCache(
        time_hidden=hidden,
        target_dt=target_dt,
        quantity_hidden=quantity_hidden,
        target_quantity=target_quantity,
        source_quantity_prediction=prediction,
    )


def test_generic_fit_records_role_backbone_variant_and_rebuilds_selected(tmp_path):
    source, payload = synthetic_payload("rmtpp")
    stats = {
        "statistics_source_split": "train",
        "target_count": 29,
        "time_scale": 1.0,
        "target_log_scaled_mean": 0.2,
        "target_log_scaled_std": 0.8,
    }
    candidate, _, metadata = build_frozen_lognormal_candidate(
        payload,
        train_time_statistics=stats,
        time_sigma_floor=0.001,
        source_backbone="rmtpp",
        source_variant=LOG_MSE_VARIANT,
        max_seq_len=16,
        training_stage="matched_frozen_representation_duration_refit",
    )
    source_state = payload["model_state_dict"]
    summary = fit_time_head_from_cache(
        model=candidate,
        train_cache=synthetic_cache(source, 29, quantity=False),
        validation_cache=synthetic_cache(source, 17, quantity=True),
        output_dir=tmp_path,
        dataset="synthetic",
        censor_threshold=None,
        contract_sha256="c" * 64,
        source_checkpoint_sha256="f" * 64,
        source_state_sha256=payload["model_state_sha256"],
        source_non_time_state_sha256=state_partition_sha256(
            source_state, time_head=False
        ),
        candidate_initial_state_sha256=metadata["candidate_initial_state_sha256"],
        train_time_statistics=stats,
        calibration_source_revision="b" * 40,
        seed=42,
        planned_epochs=2,
        learning_rate=0.001,
        weight_decay=0.0,
        batch_size=8,
        quantity_replay_batch_size=8,
        grad_clip=1.0,
        min_epochs=10,
        patience=20,
        device="cpu",
        candidate_metadata=metadata,
        source_metadata={"model_role": "rmtpp"},
        contract_id=CONTRACT_ID,
        model_role="rmtpp",
        source_backbone="rmtpp",
        source_variant=LOG_MSE_VARIANT,
    )
    assert summary["contract_id"] == CONTRACT_ID
    assert summary["model_role"] == "rmtpp"
    assert summary["source_backbone"] == "rmtpp"
    assert summary["source_variant"] == LOG_MSE_VARIANT
    assert summary["resume_identity"]["model_role"] == "rmtpp"
    assert summary["resume_identity"]["source_backbone"] == "rmtpp"
    assert summary["resume_identity"]["source_variant"] == LOG_MSE_VARIANT
    assert summary["trainable_parameter_count"] == 130
    assert summary["quantity_prediction_bitwise_identical"] is True
    selected = torch.load(
        tmp_path / "best_validation_proper_time_nll_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert selected["model_role"] == "rmtpp"
    assert selected["backbone"] == "rmtpp"
    rebuilt = build_candidate_from_selected_checkpoint(selected)
    assert canonical_state_dict_sha256(rebuilt.state_dict()) == selected["model_state_sha256"]


def test_source_admission_rejects_identity_and_selector_tampering():
    _, payload = synthetic_payload("thp")
    spec = {
        "available": True,
        "backbone": "thp",
        "quantity_variant": LOG_MSE_VARIANT,
        "seed": 42,
        "checkpoint_selection": "best_validation_joint_objective",
        "training_source_revision": "a" * 40,
        "training_source_revision_history": ["a" * 40],
        "checkpoint_state_sha256": payload["model_state_sha256"],
    }
    validate_source_checkpoint(payload, source_spec=spec)
    wrong = copy.deepcopy(payload)
    wrong["selection"] = "best_validation_raw_quantity_rmse"
    with pytest.raises(ValueError, match="selector"):
        validate_source_checkpoint(wrong, source_spec=spec)
    wrong = copy.deepcopy(payload)
    wrong["model_state_dict"]["b_t"] += 1.0
    with pytest.raises(ValueError, match="digest"):
        validate_source_checkpoint(wrong, source_spec=spec)
