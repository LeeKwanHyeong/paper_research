from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTPP import LOG_MSE_VARIANT
from models.TPPs.CountAwareTitanBoundedQK import BOUNDED_QK_BACKBONE
from models.TPPs.CountAwareTitanCausalQKV import CAUSAL_QKV_BACKBONE
from paper.scripts import run_backbone_normalized_duration as route
from paper.scripts.run_hard_lmm_frozen_lognormal_duration import (
    LAST_CHECKPOINT_NAME,
    OBSERVATION_LIKELIHOOD_CONTINUOUS,
    OBSERVATION_LIKELIHOOD_POSITIVE_INTEGER,
    build_frozen_lognormal_candidate,
    fit_time_head_from_cache,
    proper_time_log_likelihood,
    state_partition_sha256,
)
from paper.scripts.run_hard_lmm_time_head_refit import FrozenFeatureCache
from simple_lab_test.search.common.runner import (
    canonical_state_dict_sha256,
    torch_load_checkpoint,
)


CONTRACT_PATH = ROOT / "paper/contracts/backbone_normalized_duration_v1.json"
ALIGNED_PATH = ROOT / "paper/contracts/aligned_frozen_lognormal_duration_v1.json"
SHA64 = "a" * 64
SHA40 = "b" * 40


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def source_manifest(
    role: str, *, contract_sha: str, evaluation_source_revision: str = SHA40
) -> dict:
    contract = load(CONTRACT_PATH)
    aligned = load(ALIGNED_PATH)
    dataset = aligned["datasets"][0]
    role_spec = contract["roles"][role]
    return {
        "schema_version": 1,
        "manifest_id": route.JOB_MANIFEST_ID,
        "status": "frozen_before_normalized_duration_outputs",
        "execution_contract_sha256": contract_sha,
        "aligned_contract_sha256": route.ALIGNED_CONTRACT_SHA256,
        "dataset": dataset["dataset"],
        "model_role": role,
        "calibration_source_revision": (
            route.ALIGNED_CALIBRATION_SOURCE_REVISION
        ),
        "evaluation_source_revision": evaluation_source_revision,
        "evaluation_runner_file_sha256": route.sha256_file(
            Path(route.__file__)
        ),
        "runtime": route.runtime_identity(torch.device("cpu")),
        "data": {
            "path": dataset["data_path"],
            "sha256": dataset["data_sha256"],
            "split_manifest_path": dataset["split_manifest_path"],
            "split_manifest_sha256": dataset["split_manifest_sha256"],
        },
        "source": {
            "backbone": role_spec["backbone"],
            "model_role": role_spec["model_role"],
            "routing_contract_id": role_spec["routing_contract_id"],
            "quantity_variant": LOG_MSE_VARIANT,
            "seed": 42,
            "checkpoint_selection": "best_validation_raw_quantity_rmse",
            "checkpoint_path": "future/checkpoint.pt",
            "checkpoint_file_sha256": SHA64,
            "checkpoint_state_sha256": "c" * 64,
            "source_non_time_state_sha256": "d" * 64,
            "summary_path": "future/summary.json",
            "summary_sha256": "e" * 64,
            "history_path": "future/history.json",
            "history_sha256": "f" * 64,
            "training_source_revision": SHA40,
            "training_source_revision_history": [SHA40],
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
        },
    }


def source_artifacts(role: str) -> tuple[dict, dict, dict]:
    contract = load(CONTRACT_PATH)
    backbone = contract["roles"][role]["backbone"]
    model, encoder = build_count_aware_model(
        backbone,
        hidden_dim=64,
        train_log_mean=1.0,
        train_log_std=0.5,
        max_seq_len=8,
        quantity_variant=LOG_MSE_VARIANT,
        lambda_tail=0.0,
        time_intercept_limit=300.0,
    )
    interface = {
        "mode": "mark_free_count_aware_log_regression",
        "target_quantity_masked_from_history": True,
        "train_target_mean": 1.0,
        "train_target_std": 0.5,
        "time_head": {
            **model.time_head_contract(),
            "time_wd_safety_limit": 40.0,
            "time_head_lr_multiplier": 1.0,
        },
    }
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    state_sha = canonical_state_dict_sha256(state)
    dataset = load(ALIGNED_PATH)["datasets"][0]
    training_role, quantile_strength = {
        "B": ("quantile_checkpoint_alignment", 1.0),
        "FULL": ("hard_lmm_causal_qkv_candidate", 0.0),
        "BOUNDED": ("hard_lmm_bounded_qk_candidate", 0.0),
    }[role]
    resume_identity = {
        "arguments": {
            "dataset_contract": dataset["dataset"],
            "data_sha256": dataset["data_sha256"],
            "split_manifest_sha256": dataset["split_manifest_sha256"],
            "lookback_weeks": dataset["lookback"],
            "max_seq_len": dataset["max_sequence_length"],
            "max_train_batches": None,
            "max_val_batches": None,
            "max_series": None,
            "hidden_dim": 64,
            "batch_size": 128,
            "lr": 0.001,
            "grad_clip": 1.0,
            "epochs": 300,
            "min_epochs": 40,
            "early_stopping_patience": 40,
            "lambda_log_qty": 1.0,
            "lambda_tail": 0.0,
            "time_head_mode": "legacy_clamped_rmtpp",
            "time_scale": 3.0,
            "time_w_max": 10.0 / 3.0,
            "time_intercept_limit": 300.0,
            "time_wd_safety_limit": 40.0,
            "time_head_lr_multiplier": 1.0,
            "source_revision": SHA40,
            "model_role": training_role,
            "quantile_adaptive_strength": quantile_strength,
        }
    }
    payload = {
        "backbone": backbone,
        "variant": LOG_MSE_VARIANT,
        "seed": 42,
        "checkpoint_monitor": "validation_raw_quantity_rmse",
        "checkpoint_monitor_history_key": "val_qty_rmse",
        "checkpoint_selection": "best_validation_raw_quantity_rmse",
        "selection": "best_validation_raw_quantity_rmse",
        "selected_metric_value": 1.25,
        "selection_formula": (
            "sqrt(mean((predicted_raw_quantity - raw_quantity)^2))"
        ),
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "source_revision": SHA40,
        "source_revision_history": [SHA40],
        "best_epoch": 7,
        "encoder_config": encoder,
        "interface_meta": interface,
        "resume_identity": resume_identity,
        "model_state_dict": state,
        "model_state_sha256": state_sha,
    }
    summary = {
        "status": "success",
        **{
            name: payload[name]
            for name in (
                "backbone", "variant", "seed", "best_epoch",
                "evaluation_scope", "held_out_test_evaluated",
                "source_revision", "source_revision_history", "encoder_config",
                "interface_meta", "resume_identity",
                "checkpoint_monitor", "checkpoint_monitor_history_key",
                "checkpoint_selection", "selected_metric_value",
                "selection_formula",
            )
        },
        "checkpoint_state_sha256": state_sha,
        "completed_epochs": 47,
        "stopped_early": True,
        "best_val_qty_rmse": 1.25,
    }
    source_spec = {
        **source_manifest(role, contract_sha=route.sha256_file(CONTRACT_PATH))[
            "source"
        ],
        "checkpoint_state_sha256": state_sha,
        "source_non_time_state_sha256": state_partition_sha256(
            state, time_head=False
        ),
    }
    return payload, summary, source_spec


def synthetic_source_history() -> dict:
    rows = []
    for epoch in range(1, 48):
        if epoch < 7:
            value = 2.0 - 0.1 * epoch
        elif epoch == 7:
            value = 1.25
        else:
            value = 1.25 + 0.01 * (epoch - 7)
        rows.append({"epoch": epoch, "val_qty_rmse": value})
    return {"history": rows}


def test_contract_manifest_and_cache_identity_fail_closed() -> None:
    contract = load(CONTRACT_PATH)
    route.validate_execution_contract(contract)
    contract_sha = route.sha256_file(CONTRACT_PATH)
    assert contract_sha == route.EXPECTED_EXECUTION_CONTRACT_SHA256
    aligned = load(ALIGNED_PATH)
    identities = []
    for role in route.MODEL_ROLES:
        manifest = source_manifest(role, contract_sha=contract_sha)
        job = route.validate_job_manifest(
            manifest,
            contract=contract,
            aligned_dataset=aligned["datasets"][0],
            execution_contract_sha256=contract_sha,
            aligned_contract_sha256=route.ALIGNED_CONTRACT_SHA256,
            evaluation_source_revision=SHA40,
            expected_runtime_identity=route.runtime_identity(
                torch.device("cpu")
            ),
        )
        binding = route.job_binding_sha256(
            execution_contract_sha256=contract_sha,
            aligned_contract_sha256=route.ALIGNED_CONTRACT_SHA256,
            job_manifest_sha256=SHA64,
        )
        identities.append(
            route.build_cache_identity(
                dataset=manifest["dataset"],
                role=role,
                source_spec=job["source"],
                data_spec=job["data"],
                execution_contract_sha256=contract_sha,
                aligned_contract_sha256=route.ALIGNED_CONTRACT_SHA256,
                job_manifest_sha256=SHA64,
                job_binding_sha=binding,
                evaluation_source_revision=SHA40,
            )
        )
    assert len({json.dumps(value, sort_keys=True) for value in identities}) == 3

    unresolved = source_manifest("BOUNDED", contract_sha=contract_sha)
    unresolved["source"]["checkpoint_file_sha256"] = None
    with pytest.raises(ValueError, match="unresolved"):
        route.validate_job_manifest(
            unresolved,
            contract=contract,
            aligned_dataset=aligned["datasets"][0],
            execution_contract_sha256=contract_sha,
            aligned_contract_sha256=route.ALIGNED_CONTRACT_SHA256,
            evaluation_source_revision=SHA40,
            expected_runtime_identity=route.runtime_identity(
                torch.device("cpu")
            ),
        )


@pytest.mark.parametrize("role", route.MODEL_ROLES)
def test_source_checkpoint_summary_state_and_route_are_strict(role: str) -> None:
    payload, summary, source_spec = source_artifacts(role)
    observed = route.validate_source_artifacts(
        payload,
        summary,
        synthetic_source_history(),
        role=role,
        source_spec=source_spec,
        checkpoint_file_sha256=SHA64,
        aligned_dataset=load(ALIGNED_PATH)["datasets"][0],
    )
    assert observed == source_spec["source_non_time_state_sha256"]

    changed = copy.deepcopy(summary)
    changed["resume_identity"] = {"synthetic": False}
    with pytest.raises(ValueError, match="summary/checkpoint"):
        route.validate_source_artifacts(
            payload,
            changed,
            synthetic_source_history(),
            role=role,
            source_spec=source_spec,
            checkpoint_file_sha256=SHA64,
            aligned_dataset=load(ALIGNED_PATH)["datasets"][0],
        )


def test_source_checkpoint_cannot_cross_dataset_job_identity() -> None:
    payload, summary, source_spec = source_artifacts("FULL")
    wrong_dataset = load(ALIGNED_PATH)["datasets"][1]
    with pytest.raises(ValueError, match="training/data contract drift"):
        route.validate_source_artifacts(
            payload,
            summary,
            synthetic_source_history(),
            role="FULL",
            source_spec=source_spec,
            checkpoint_file_sha256=SHA64,
            aligned_dataset=wrong_dataset,
        )


def test_source_checkpoint_requires_explicit_full_scope_arguments() -> None:
    payload, summary, source_spec = source_artifacts("B")
    payload = copy.deepcopy(payload)
    summary = copy.deepcopy(summary)
    del payload["resume_identity"]["arguments"]["max_train_batches"]
    summary["resume_identity"] = copy.deepcopy(payload["resume_identity"])
    with pytest.raises(ValueError, match="training/data contract drift"):
        route.validate_source_artifacts(
            payload,
            summary,
            synthetic_source_history(),
            role="B",
            source_spec=source_spec,
            checkpoint_file_sha256=SHA64,
            aligned_dataset=load(ALIGNED_PATH)["datasets"][0],
        )


def test_source_history_replays_selector_and_early_stopping() -> None:
    payload, summary, source_spec = source_artifacts("BOUNDED")
    history = synthetic_source_history()
    changed = copy.deepcopy(history)
    changed["history"][0]["val_qty_rmse"] = 1.0
    with pytest.raises(ValueError, match="continued after|stopped before"):
        route.validate_source_artifacts(
            payload,
            summary,
            changed,
            role="BOUNDED",
            source_spec=source_spec,
            checkpoint_file_sha256=SHA64,
            aligned_dataset=load(ALIGNED_PATH)["datasets"][0],
        )

    changed = copy.deepcopy(history)
    changed["history"][6]["val_qty_rmse"] = 1.24
    with pytest.raises(ValueError, match="selected value drift"):
        route.validate_source_artifacts(
            payload,
            summary,
            changed,
            role="BOUNDED",
            source_spec=source_spec,
            checkpoint_file_sha256=SHA64,
            aligned_dataset=load(ALIGNED_PATH)["datasets"][0],
        )

    changed = copy.deepcopy(history)
    changed["history"][0]["val_qty_rmse"] = float("nan")
    with pytest.raises(ValueError, match="invalid raw-RMSE"):
        route.validate_source_artifacts(
            payload,
            summary,
            changed,
            role="BOUNDED",
            source_spec=source_spec,
            checkpoint_file_sha256=SHA64,
            aligned_dataset=load(ALIGNED_PATH)["datasets"][0],
        )


def test_full_B_fit_is_rejected_before_source_files_are_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract_sha = route.sha256_file(CONTRACT_PATH)
    revision = SHA40
    monkeypatch.setattr(route, "current_source_revision", lambda: revision)
    manifest = source_manifest(
        "B",
        contract_sha=contract_sha,
        evaluation_source_revision=revision,
    )
    manifest_path = tmp_path / "B_job.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    args = route.parse_args(
        [
            "--job-manifest", str(manifest_path),
            "--expected-job-manifest-sha256", route.sha256_file(manifest_path),
            "--output-dir", str(tmp_path / "output"),
            "--source-revision", revision,
        ]
    )
    with pytest.raises(ValueError, match="B full fit is forbidden"):
        route.run(args)

    disguised_full_args = route.parse_args(
        [
            "--job-manifest", str(manifest_path),
            "--expected-job-manifest-sha256", route.sha256_file(manifest_path),
            "--output-dir", str(tmp_path / "disguised_full_output"),
            "--source-revision", revision,
            "--allow-partial-contract",
            "--max-epochs", "100",
        ]
    )
    with pytest.raises(ValueError, match="B full fit is forbidden"):
        route.run(disguised_full_args)


def test_all_aligned_observation_likelihoods_are_finite_and_censor_aware() -> None:
    source, _, _ = source_artifacts("BOUNDED")
    candidate, _, _ = build_frozen_lognormal_candidate(
        source,
        train_time_statistics={
            "time_scale": 2.0,
            "target_log_scaled_mean": 0.0,
            "target_log_scaled_std": 0.7,
        },
        time_sigma_floor=0.001,
        source_backbone=BOUNDED_QK_BACKBONE,
        source_variant=LOG_MSE_VARIANT,
        max_seq_len=8,
        training_stage="synthetic_test",
    )
    hidden = torch.zeros(3, 64)
    continuous = proper_time_log_likelihood(
        candidate,
        hidden,
        torch.tensor([1.0, 2.0, 4.0]),
        is_right_censored=torch.zeros(3, dtype=torch.bool),
        observation_likelihood_mode=OBSERVATION_LIKELIHOOD_CONTINUOUS,
    )
    integer = proper_time_log_likelihood(
        candidate,
        hidden,
        torch.tensor([1.0, 2.0, 30.0]),
        is_right_censored=torch.tensor([False, False, True]),
        observation_likelihood_mode=OBSERVATION_LIKELIHOOD_POSITIVE_INTEGER,
    )
    assert continuous.dtype == integer.dtype == torch.float64
    assert torch.isfinite(continuous).all() and torch.isfinite(integer).all()


def test_tiny_cache_fit_preserves_quantity_selects_earliest_and_resumes(
    tmp_path: Path,
) -> None:
    source, _, source_spec = source_artifacts("BOUNDED")
    candidate, _, metadata = build_frozen_lognormal_candidate(
        source,
        train_time_statistics={
            "time_scale": 2.0,
            "target_log_scaled_mean": 0.1,
            "target_log_scaled_std": 0.7,
        },
        time_sigma_floor=0.001,
        source_backbone=BOUNDED_QK_BACKBONE,
        source_variant=LOG_MSE_VARIANT,
        max_seq_len=8,
        training_stage="synthetic_test",
    )
    generator = torch.Generator().manual_seed(42)
    train_hidden = torch.randn(12, 64, generator=generator)
    validation_hidden = torch.randn(8, 64, generator=generator)
    quantity_hidden = validation_hidden.clone()
    with torch.no_grad():
        source_prediction = candidate.predict_quantity(quantity_hidden)[1]
    train_cache = FrozenFeatureCache(
        time_hidden=train_hidden,
        target_dt=torch.linspace(1.0, 6.0, 12),
    )
    validation_cache = FrozenFeatureCache(
        time_hidden=validation_hidden,
        target_dt=torch.linspace(1.0, 5.0, 8),
        quantity_hidden=quantity_hidden,
        target_quantity=torch.linspace(1.0, 8.0, 8),
        source_quantity_prediction=source_prediction,
    )
    full_dir = tmp_path / "full"
    kwargs = dict(
        model=candidate,
        train_cache=train_cache,
        validation_cache=validation_cache,
        output_dir=full_dir,
        dataset="synthetic",
        censor_threshold=None,
        contract_sha256=SHA64,
        source_checkpoint_sha256="c" * 64,
        source_state_sha256=canonical_state_dict_sha256(source["model_state_dict"]),
        source_non_time_state_sha256=source_spec[
            "source_non_time_state_sha256"
        ],
        candidate_initial_state_sha256=metadata["candidate_initial_state_sha256"],
        train_time_statistics={
            "time_scale": 2.0,
            "target_log_scaled_mean": 0.1,
            "target_log_scaled_std": 0.7,
        },
        calibration_source_revision=route.ALIGNED_CALIBRATION_SOURCE_REVISION,
        seed=42,
        planned_epochs=2,
        learning_rate=0.001,
        weight_decay=0.0,
        batch_size=4,
        quantity_replay_batch_size=4,
        grad_clip=1.0,
        min_epochs=2,
        patience=20,
        device="cpu",
        candidate_metadata=metadata,
        source_metadata={"synthetic": True},
        contract_id=route.CONTRACT_ID,
        model_role="BOUNDED",
        source_backbone=BOUNDED_QK_BACKBONE,
        source_variant=LOG_MSE_VARIANT,
        observation_likelihood_contract={
            "mode": OBSERVATION_LIKELIHOOD_CONTINUOUS,
            "uncensored": "log-normal density in original time units",
            "top_code": None,
            "calculation_dtype": "float64",
            "selection_formula": "synthetic mean negative density",
        },
    )
    first = fit_time_head_from_cache(**kwargs)
    assert first["status"] == "success"
    assert first["quantity_prediction_bitwise_identical"] is True
    assert first["source_quantity_prediction_sha256"] == first[
        "selected_quantity_prediction_sha256"
    ]
    best = min(
        row["val_proper_time_nll"] for row in first["history"]
    )
    earliest = next(
        row["epoch"]
        for row in first["history"]
        if row["val_proper_time_nll"] == best
    )
    assert first["best_epoch"] == earliest
    selected = torch_load_checkpoint(
        full_dir / route.SELECTED_CHECKPOINT_NAME, map_location="cpu"
    )
    route._selected_fit_audit(
        summary=first,
        selected=selected,
        role="BOUNDED",
        source_non_time_sha256=source_spec["source_non_time_state_sha256"],
        observation_contract=kwargs["observation_likelihood_contract"],
        source_lineage={"synthetic": True},
    )
    paused_model, _, paused_metadata = build_frozen_lognormal_candidate(
        source,
        train_time_statistics=kwargs["train_time_statistics"],
        time_sigma_floor=0.001,
        source_backbone=BOUNDED_QK_BACKBONE,
        source_variant=LOG_MSE_VARIANT,
        max_seq_len=8,
        training_stage="synthetic_test",
    )
    resumed_dir = tmp_path / "resumed"
    paused_kwargs = {
        **kwargs,
        "output_dir": resumed_dir,
        "model": paused_model,
        "candidate_initial_state_sha256": paused_metadata[
            "candidate_initial_state_sha256"
        ],
        "candidate_metadata": paused_metadata,
    }
    paused = fit_time_head_from_cache(**paused_kwargs, run_epoch_limit=1)
    assert paused["status"] == "paused"
    assert paused["completed_epochs"] == 1

    resumed_model, _, resumed_metadata = build_frozen_lognormal_candidate(
        source,
        train_time_statistics=kwargs["train_time_statistics"],
        time_sigma_floor=0.001,
        source_backbone=BOUNDED_QK_BACKBONE,
        source_variant=LOG_MSE_VARIANT,
        max_seq_len=8,
        training_stage="synthetic_test",
    )
    resume_kwargs = {
        **paused_kwargs,
        "model": resumed_model,
        "candidate_initial_state_sha256": resumed_metadata[
            "candidate_initial_state_sha256"
        ],
        "candidate_metadata": resumed_metadata,
    }
    second = fit_time_head_from_cache(**resume_kwargs)
    assert second["history"] == first["history"]
    assert second["best_epoch"] == first["best_epoch"]
    assert second["best_validation_proper_time_nll"] == first[
        "best_validation_proper_time_nll"
    ]
    assert second["selected_state_sha256"] == first[
        "selected_state_sha256"
    ]
    assert second["quantity_prediction_bitwise_identical"] is True

    full_last = torch_load_checkpoint(
        full_dir / LAST_CHECKPOINT_NAME, map_location="cpu"
    )
    resumed_last = torch_load_checkpoint(
        resumed_dir / LAST_CHECKPOINT_NAME, map_location="cpu"
    )
    assert resumed_last["epoch"] == full_last["epoch"] == 2
    assert resumed_last["history"] == full_last["history"]
    assert resumed_last["model_state_sha256"] == full_last[
        "model_state_sha256"
    ]
    assert resumed_last["best_state_sha256"] == full_last[
        "best_state_sha256"
    ]
    assert resumed_last["optimizer_state_dict"]["param_groups"] == full_last[
        "optimizer_state_dict"
    ]["param_groups"]
    for parameter_id, state in full_last["optimizer_state_dict"]["state"].items():
        resumed_state = resumed_last["optimizer_state_dict"]["state"][
            parameter_id
        ]
        assert set(resumed_state) == set(state)
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(resumed_state[name], value)
            else:
                assert resumed_state[name] == value

    cached_model, _, cached_metadata = build_frozen_lognormal_candidate(
        source,
        train_time_statistics=kwargs["train_time_statistics"],
        time_sigma_floor=0.001,
        source_backbone=BOUNDED_QK_BACKBONE,
        source_variant=LOG_MSE_VARIANT,
        max_seq_len=8,
        training_stage="synthetic_test",
    )
    cached = fit_time_head_from_cache(
        **{
            **resume_kwargs,
            "model": cached_model,
            "candidate_initial_state_sha256": cached_metadata[
                "candidate_initial_state_sha256"
            ],
            "candidate_metadata": cached_metadata,
        }
    )
    assert cached == second
