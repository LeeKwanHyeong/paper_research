from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from paper.scripts.build_aligned_frozen_lognormal_duration_decision import (
    build_decision,
)
from paper.scripts.run_aligned_frozen_lognormal_duration import (
    DEFAULT_CONTRACT,
    load_contract_without_duplicate_keys,
)
from paper.scripts.run_hard_lmm_frozen_lognormal_duration import (
    CANDIDATE_TIME_HEAD_MODE,
    OBSERVATION_LIKELIHOOD_CONTINUOUS,
    OBSERVATION_LIKELIHOOD_POSITIVE_INTEGER,
    SELECTED_CHECKPOINT_NAME,
    SELECTION_RULE,
    state_partition_sha256,
)
from paper.scripts.run_hard_lmm_time_head_refit import sha256_file
from simple_lab_test.search.common.runner import (
    atomic_torch_save,
    canonical_state_dict_sha256,
    torch_load_checkpoint,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _source_state(dataset_index: int, role: str) -> dict[str, torch.Tensor]:
    role_offset = 0.0 if role == "A" else 1.0
    return {
        "encoder.weight": torch.full(
            (2, 2), float(dataset_index) + role_offset
        ),
        "v_t.weight": torch.zeros(1, 64),
        "b_t": torch.zeros(1),
        "w_raw": torch.zeros(1),
    }


def _initial_time_state(dataset_index: int) -> dict[str, torch.Tensor]:
    return {
        "v_t.weight": torch.zeros(1, 64),
        "b_t": torch.tensor([0.1 * (dataset_index + 1)]),
        "w_raw": torch.tensor([0.5]),
        "time_scale_weight.weight": torch.zeros(1, 64),
    }


def _selected_state(
    source_state: dict[str, torch.Tensor],
    *,
    dataset_index: int,
    role: str,
) -> dict[str, torch.Tensor]:
    state = {
        name: value.detach().clone()
        for name, value in source_state.items()
        if name not in {"v_t.weight", "b_t", "w_raw"}
    }
    learned = 0.01 * (dataset_index + 1) + (0.001 if role == "B" else 0.0)
    state.update(
        {
            "v_t.weight": torch.full((1, 64), learned),
            "b_t": torch.tensor([learned]),
            "w_raw": torch.tensor([0.5 + learned]),
            "time_scale_weight.weight": torch.full((1, 64), -learned),
        }
    )
    return state


def _train_statistics(dataset_spec: dict) -> dict:
    return {
        "statistics_source_split": "train",
        "target_count": dataset_spec["expected_train_targets"],
        "target_dt_min": 1.0,
        "target_dt_mean": 2.0,
        "target_dt_p50": dataset_spec["train_time_scale"],
        "target_dt_p99": 30.0,
        "target_dt_max": 100.0,
        "time_scale": dataset_spec["train_time_scale"],
        "wd_safety_limit": 40.0,
        "time_w_max": 0.25,
        "time_initial_intercept": -0.5,
        "target_log_scaled_mean": dataset_spec["train_log_scaled_mean"],
        "target_log_scaled_std": dataset_spec["train_log_scaled_std"],
    }


def _population(dataset_spec: dict, split: str) -> dict:
    return {
        "split": split,
        "target_count": dataset_spec[f"expected_{split}_targets"],
        "target_identity_sha256": dataset_spec[
            f"expected_{split}_target_identity_sha256"
        ],
        "target_quantity_sha256": dataset_spec[
            f"expected_{split}_target_quantity_sha256"
        ],
    }


def _observation_population(dataset_spec: dict, split: str) -> dict:
    return {
        "split": split,
        "target_count": dataset_spec[f"expected_{split}_targets"],
        "target_dt_sha256": dataset_spec[
            f"expected_{split}_target_dt_sha256"
        ],
        "right_censor_threshold": dataset_spec["right_censor_threshold"],
        "right_censored_count": dataset_spec[
            f"expected_{split}_censored_targets"
        ],
        "censor_mask_sha256": dataset_spec[
            f"expected_{split}_censor_mask_sha256"
        ],
    }


def _metrics(dataset_spec: dict, *, mode: str, nll: float) -> dict:
    return {
        "count": dataset_spec["expected_validation_targets"],
        "observation_likelihood_mode": mode,
        "proper_time_nll": nll,
        "right_censor_threshold": dataset_spec["right_censor_threshold"],
        "right_censored_count": dataset_spec[
            "expected_validation_censored_targets"
        ],
    }


def _history(primary_nll: float) -> list[dict]:
    rows = [{"epoch": 0, "val_proper_time_nll": primary_nll + 0.02}]
    rows.append({"epoch": 1, "val_proper_time_nll": primary_nll})
    rows.extend(
        {"epoch": epoch, "val_proper_time_nll": primary_nll + 0.01}
        for epoch in range(2, 22)
    )
    return rows


def _prepare_contract(tmp_path: Path) -> tuple[dict, Path, dict]:
    contract = copy.deepcopy(
        load_contract_without_duplicate_keys(DEFAULT_CONTRACT)
    )
    states: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for dataset_index, dataset_spec in enumerate(contract["datasets"]):
        dataset = dataset_spec["dataset"]
        states[dataset] = {}
        for role in ("A", "B"):
            source_state = _source_state(dataset_index, role)
            states[dataset][role] = source_state
            source_spec = dataset_spec["sources"][role]
            source_spec["checkpoint_file_sha256"] = _digest(
                f"source-file:{dataset}:{role}"
            )
            source_spec["checkpoint_state_sha256"] = (
                canonical_state_dict_sha256(source_state)
            )
            source_spec["source_non_time_state_sha256"] = (
                state_partition_sha256(source_state, time_head=False)
            )
        reference = dataset_spec["prior_B_reference"]
        reference["checkpoint_file_sha256"] = _digest(
            f"prior-file:{dataset}"
        )
        reference["checkpoint_state_sha256"] = _digest(
            f"prior-state:{dataset}"
        )
        reference["source_non_time_state_sha256"] = dataset_spec["sources"][
            "B"
        ]["source_non_time_state_sha256"]
        reference["validation_continuous_nll"] = (
            1.0
            if dataset_spec["observation_contract"]["mode"]
            == OBSERVATION_LIKELIHOOD_CONTINUOUS
            else 0.8
        )

    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    loaded = load_contract_without_duplicate_keys(contract_path)
    return loaded, contract_path, states


def _resume_identity(
    *,
    contract: dict,
    contract_sha256: str,
    dataset_spec: dict,
    role: str,
    candidate_initial_state_sha256: str,
    train_statistics: dict,
    caches: dict,
) -> dict:
    source_spec = dataset_spec["sources"][role]
    optimization = contract["optimization"]
    observation_contract = dataset_spec["observation_contract"]
    return {
        "schema_version": 1,
        "contract_id": contract["contract_id"],
        "contract_sha256": contract_sha256,
        "dataset": dataset_spec["dataset"],
        "source_checkpoint_sha256": source_spec["checkpoint_file_sha256"],
        "source_state_sha256": source_spec["checkpoint_state_sha256"],
        "source_non_time_state_sha256": source_spec[
            "source_non_time_state_sha256"
        ],
        "candidate_initial_state_sha256": candidate_initial_state_sha256,
        "train_cache_sha256": caches["train"]["sha256"],
        "validation_cache_sha256": caches["validation"]["sha256"],
        "train_cache_count": caches["train"]["count"],
        "validation_cache_count": caches["validation"]["count"],
        "train_target_dt_sha256": caches["train"]["target_dt_sha256"],
        "validation_target_dt_sha256": caches["validation"][
            "target_dt_sha256"
        ],
        "train_censor_mask_sha256": caches["train"]["censor_mask_sha256"],
        "validation_censor_mask_sha256": caches["validation"][
            "censor_mask_sha256"
        ],
        "right_censor_threshold": dataset_spec["right_censor_threshold"],
        "train_time_statistics": train_statistics,
        "calibration_source_revision": "c" * 40,
        "seed": optimization["seed"],
        "optimizer": optimization["optimizer"],
        "learning_rate": optimization["learning_rate"],
        "weight_decay": optimization["weight_decay"],
        "cached_state_batch_size": optimization["cached_state_batch_size"],
        "quantity_replay_batch_size": optimization["encoder_batch_size"],
        "gradient_clip": optimization["gradient_clip"],
        "minimum_epochs": optimization["minimum_epochs"],
        "early_stopping_patience": optimization[
            "early_stopping_patience"
        ],
        "planned_epochs": optimization["epochs"],
        "selection": SELECTION_RULE,
        "trainable_parameter_names": contract["time_head"][
            "trainable_parameter_names"
        ],
        "time_head_mode": CANDIDATE_TIME_HEAD_MODE,
        "likelihood": observation_contract["mode"],
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "legacy_nll_compared": False,
        "model_role": role,
        "source_backbone": source_spec["backbone"],
        "source_variant": source_spec["quantity_variant"],
        "observation_likelihood_contract": observation_contract,
    }


def _write_role_artifact(
    *,
    root: Path,
    contract: dict,
    contract_sha256: str,
    dataset_spec: dict,
    dataset_index: int,
    role: str,
    source_state: dict[str, torch.Tensor],
    primary_nll: float,
    continuous_nll: float,
) -> None:
    dataset = dataset_spec["dataset"]
    source_spec = dataset_spec["sources"][role]
    observation_contract = dataset_spec["observation_contract"]
    output_dir = root / dataset / role
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_time_state = _initial_time_state(dataset_index)
    candidate_initial_state = {
        name: value.detach().clone()
        for name, value in source_state.items()
        if name not in {"v_t.weight", "b_t", "w_raw"}
    }
    candidate_initial_state.update(initial_time_state)
    candidate_initial_state_sha256 = canonical_state_dict_sha256(
        candidate_initial_state
    )
    initial_time_sha256 = state_partition_sha256(
        candidate_initial_state, time_head=True
    )
    selected_state = _selected_state(
        source_state, dataset_index=dataset_index, role=role
    )
    selected_state_sha256 = canonical_state_dict_sha256(selected_state)
    selected_non_time_sha256 = state_partition_sha256(
        selected_state, time_head=False
    )
    selected_time_sha256 = state_partition_sha256(
        selected_state, time_head=True
    )
    train_statistics = _train_statistics(dataset_spec)
    caches = {
        split: {
            "count": dataset_spec[f"expected_{split}_targets"],
            "sha256": _digest(f"cache:{dataset}:{role}:{split}"),
            "target_dt_sha256": dataset_spec[
                f"expected_{split}_target_dt_sha256"
            ],
            "censor_mask_sha256": dataset_spec[
                f"expected_{split}_censor_mask_sha256"
            ],
        }
        for split in ("train", "validation")
    }
    identity = _resume_identity(
        contract=contract,
        contract_sha256=contract_sha256,
        dataset_spec=dataset_spec,
        role=role,
        candidate_initial_state_sha256=candidate_initial_state_sha256,
        train_statistics=train_statistics,
        caches=caches,
    )
    checkpoint = {
        "checkpoint_type": "selected_frozen_lognormal_duration",
        "checkpoint_schema_version": 1,
        "selection": SELECTION_RULE,
        "selection_formula": observation_contract["selection_formula"],
        "best_epoch": 1,
        "selected_metric_value": primary_nll,
        "model_state_dict": selected_state,
        "model_state_sha256": selected_state_sha256,
        "source_state_sha256": source_spec["checkpoint_state_sha256"],
        "source_non_time_state_sha256": source_spec[
            "source_non_time_state_sha256"
        ],
        "candidate_initial_state_sha256": candidate_initial_state_sha256,
        "candidate_initial_time_head_state_sha256": initial_time_sha256,
        "selected_time_head_state_sha256": selected_time_sha256,
        "backbone": source_spec["backbone"],
        "model_role": role,
        "variant": source_spec["quantity_variant"],
        "time_head_mode": CANDIDATE_TIME_HEAD_MODE,
        "source_checkpoint_lineage": {
            "model_role": role,
            "backbone": source_spec["backbone"],
            "checkpoint_file_sha256": source_spec[
                "checkpoint_file_sha256"
            ],
            "checkpoint_state_sha256": source_spec[
                "checkpoint_state_sha256"
            ],
            "checkpoint_selection": source_spec["checkpoint_selection"],
            "training_source_revision": source_spec[
                "training_source_revision"
            ],
            "training_source_revision_history": source_spec[
                "training_source_revision_history"
            ],
            "calibration_source_revision": identity[
                "calibration_source_revision"
            ],
        },
        "train_time_statistics": train_statistics,
        "right_censor_threshold": dataset_spec["right_censor_threshold"],
        "train_right_censored_count": dataset_spec[
            "expected_train_censored_targets"
        ],
        "validation_right_censored_count": dataset_spec[
            "expected_validation_censored_targets"
        ],
        "resume_identity": identity,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "legacy_nll_compared": False,
        "observation_likelihood_contract": observation_contract,
    }
    checkpoint_path = output_dir / SELECTED_CHECKPOINT_NAME
    atomic_torch_save(checkpoint, checkpoint_path)
    checkpoint_file_sha256 = sha256_file(checkpoint_path)

    quantity_sha256 = _digest(f"quantity:{dataset}:{role}")
    summary = {
        "schema_version": 1,
        "contract_id": contract["contract_id"],
        "contract_sha256": contract_sha256,
        "status": "success",
        "dataset": dataset,
        "model_role": role,
        "source_backbone": source_spec["backbone"],
        "source_variant": source_spec["quantity_variant"],
        "seed": 42,
        "qualified_full_data": True,
        "qualified_full_fit": True,
        "time_head_mode": CANDIDATE_TIME_HEAD_MODE,
        "source_checkpoint_sha256": source_spec["checkpoint_file_sha256"],
        "source_state_sha256": source_spec["checkpoint_state_sha256"],
        "source_non_time_state_sha256": source_spec[
            "source_non_time_state_sha256"
        ],
        "candidate_initial_state_sha256": candidate_initial_state_sha256,
        "candidate_initial_time_head_state_sha256": initial_time_sha256,
        "selected_state_sha256": selected_state_sha256,
        "selected_non_time_state_sha256": selected_non_time_sha256,
        "selected_time_head_state_sha256": selected_time_sha256,
        "selected_checkpoint_file_sha256": checkpoint_file_sha256,
        "changed_state_keys": list(
            contract["time_head"]["trainable_parameter_names"]
        ),
        "trainable_parameter_names": contract["time_head"][
            "trainable_parameter_names"
        ],
        "trainable_parameter_count": contract["time_head"][
            "expected_trainable_parameter_count"
        ],
        "encoder_mode_during_refit": "eval",
        "hidden_state_gradient": "detached_cache",
        "calculation_dtype": "float64",
        "observation_likelihood_mode": observation_contract["mode"],
        "observation_likelihood_contract": observation_contract,
        "selection": SELECTION_RULE,
        "best_epoch": 1,
        "completed_epochs": 21,
        "stopped_early": True,
        "best_validation_proper_time_nll": primary_nll,
        "validation_time_metrics": _metrics(
            dataset_spec,
            mode=observation_contract["mode"],
            nll=primary_nll,
        ),
        "validation_continuous_reference_metrics": _metrics(
            dataset_spec,
            mode=OBSERVATION_LIKELIHOOD_CONTINUOUS,
            nll=continuous_nll,
        ),
        "right_censor_threshold": dataset_spec["right_censor_threshold"],
        "train_right_censored_count": dataset_spec[
            "expected_train_censored_targets"
        ],
        "validation_right_censored_count": dataset_spec[
            "expected_validation_censored_targets"
        ],
        "quantity_prediction_bitwise_identical": True,
        "source_quantity_prediction_sha256": quantity_sha256,
        "selected_quantity_prediction_sha256": quantity_sha256,
        "train_time_statistics": train_statistics,
        "train_target_population": _population(dataset_spec, "train"),
        "validation_target_population": _population(
            dataset_spec, "validation"
        ),
        "train_observation_contract": _observation_population(
            dataset_spec, "train"
        ),
        "validation_observation_contract": _observation_population(
            dataset_spec, "validation"
        ),
        "train_cache": caches["train"],
        "validation_cache": caches["validation"],
        "history": _history(primary_nll),
        "resume_identity": identity,
        "selected_checkpoint_path": str(checkpoint_path),
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "legacy_nll_compared": False,
    }
    if role == "B":
        prior_primary = 1.0
        prior_continuous = (
            prior_primary
            if observation_contract["mode"]
            == OBSERVATION_LIKELIHOOD_CONTINUOUS
            else 0.8
        )
        improvement = prior_primary - primary_nll
        continuous_delta = continuous_nll - prior_continuous
        integer_mode = (
            observation_contract["mode"]
            == OBSERVATION_LIKELIHOOD_POSITIVE_INTEGER
        )
        minimum_improvement = 0.005 if integer_mode else -1e-6
        summary["prior_B_reference"] = {
            "checkpoint_file_sha256": dataset_spec["prior_B_reference"][
                "checkpoint_file_sha256"
            ],
            "checkpoint_state_sha256": dataset_spec["prior_B_reference"][
                "checkpoint_state_sha256"
            ],
            "primary_observation_metrics": _metrics(
                dataset_spec,
                mode=observation_contract["mode"],
                nll=prior_primary,
            ),
            "continuous_reference_metrics": _metrics(
                dataset_spec,
                mode=OBSERVATION_LIKELIHOOD_CONTINUOUS,
                nll=prior_continuous,
            ),
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
        }
        summary["within_B_gates"] = {
            "primary_requirement": (
                "improve_by_at_least_0.005"
                if integer_mode
                else "replay_nonworse_within_1e-6"
            ),
            "primary_nll_improvement": improvement,
            "minimum_accepted_primary_nll_improvement": minimum_improvement,
            "passes_primary_requirement": improvement >= minimum_improvement,
            "continuous_nll_delta": continuous_delta,
            "continuous_nll_is_report_only": True,
        }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_artifacts(
    root: Path,
    *,
    A_primary: float = 0.985,
    B_primary: float = 0.99,
) -> tuple[dict, Path, str]:
    contract, contract_path, states = _prepare_contract(root)
    contract_sha256 = sha256_file(contract_path)
    for dataset_index, dataset_spec in enumerate(contract["datasets"]):
        dataset = dataset_spec["dataset"]
        for role, primary, integer_continuous in (
            ("A", A_primary, 0.78),
            ("B", B_primary, 0.79),
        ):
            continuous = (
                primary
                if dataset_spec["observation_contract"]["mode"]
                == OBSERVATION_LIKELIHOOD_CONTINUOUS
                else integer_continuous
            )
            _write_role_artifact(
                root=root,
                contract=contract,
                contract_sha256=contract_sha256,
                dataset_spec=dataset_spec,
                dataset_index=dataset_index,
                role=role,
                source_state=states[dataset][role],
                primary_nll=primary,
                continuous_nll=continuous,
            )
    return contract, contract_path, contract_sha256


def test_decision_accepts_checkpoint_backed_evidence(tmp_path: Path) -> None:
    contract, _, contract_sha256 = _write_artifacts(tmp_path)
    decision = build_decision(
        contract=contract,
        contract_sha256=contract_sha256,
        results_root=tmp_path,
    )
    assert decision["all_datasets_pass"] is True
    assert decision["decision"] == "adopt_aligned_B_without_adapter"
    assert decision["evidence_integrity"] == "verified"
    assert all(row["passed"] for row in decision["datasets"])
    intermittent = decision["datasets"][0]
    assert (
        "aligned_B_replays_prior_continuous_objective_within_1e_6"
        in intermittent["gates"]
    )
    assert "aligned_B_continuous_nonworse_than_prior_B" not in intermittent[
        "gates"
    ]


def test_decision_rejects_B_that_misses_A_margin(tmp_path: Path) -> None:
    contract, _, contract_sha256 = _write_artifacts(
        tmp_path, A_primary=0.98, B_primary=0.994
    )
    decision = build_decision(
        contract=contract,
        contract_sha256=contract_sha256,
        results_root=tmp_path,
    )
    assert decision["all_datasets_pass"] is False
    assert decision["decision"].endswith("continue_to_adapter_gate")
    assert not any(row["passed"] for row in decision["datasets"])


def test_decision_rejects_qualified_full_fit_false(tmp_path: Path) -> None:
    contract, _, contract_sha256 = _write_artifacts(tmp_path)
    dataset = contract["datasets"][0]["dataset"]
    path = tmp_path / dataset / "B" / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["qualified_full_fit"] = False
    path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="partial run"):
        build_decision(
            contract=contract,
            contract_sha256=contract_sha256,
            results_root=tmp_path,
        )


def test_decision_rejects_actual_contract_digest_drift(tmp_path: Path) -> None:
    contract, _, _ = _write_artifacts(tmp_path)
    with pytest.raises(ValueError, match="contract digest drift"):
        build_decision(
            contract=contract,
            contract_sha256="0" * 64,
            results_root=tmp_path,
        )


def test_decision_rejects_likelihood_drift(tmp_path: Path) -> None:
    contract, _, contract_sha256 = _write_artifacts(tmp_path)
    dataset = contract["datasets"][1]["dataset"]
    path = tmp_path / dataset / "B" / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["observation_likelihood_contract"]["first_bin"] = (
        "P(0.5 < T <= 1.5)"
    )
    path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Observation likelihood contract drift"):
        build_decision(
            contract=contract,
            contract_sha256=contract_sha256,
            results_root=tmp_path,
        )


def test_decision_rejects_selected_checkpoint_tamper(tmp_path: Path) -> None:
    contract, _, contract_sha256 = _write_artifacts(tmp_path)
    dataset = contract["datasets"][0]["dataset"]
    checkpoint_path = tmp_path / dataset / "A" / SELECTED_CHECKPOINT_NAME
    with checkpoint_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="checkpoint file digest drift"):
        build_decision(
            contract=contract,
            contract_sha256=contract_sha256,
            results_root=tmp_path,
        )


def test_decision_rejects_history_continued_after_first_early_stop(
    tmp_path: Path,
) -> None:
    contract, _, contract_sha256 = _write_artifacts(tmp_path)
    dataset = contract["datasets"][0]["dataset"]
    summary_path = tmp_path / dataset / "A" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    # Epoch 21 is already the first valid stop: best=1 and patience=20.
    summary["history"].append(
        {
            "epoch": 22,
            "val_proper_time_nll": (
                summary["best_validation_proper_time_nll"] + 0.01
            ),
        }
    )
    summary["completed_epochs"] = 22
    summary["stopped_early"] = True
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="continued after early stopping"):
        build_decision(
            contract=contract,
            contract_sha256=contract_sha256,
            results_root=tmp_path,
        )


def test_decision_rejects_cross_row_calibration_revision_drift(
    tmp_path: Path,
) -> None:
    contract, _, contract_sha256 = _write_artifacts(tmp_path)
    dataset = contract["datasets"][0]["dataset"]
    output_dir = tmp_path / dataset / "B"
    summary_path = output_dir / "summary.json"
    checkpoint_path = output_dir / SELECTED_CHECKPOINT_NAME
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint = torch_load_checkpoint(checkpoint_path, map_location="cpu")

    drifted_revision = "d" * 40
    summary["resume_identity"][
        "calibration_source_revision"
    ] = drifted_revision
    checkpoint["resume_identity"][
        "calibration_source_revision"
    ] = drifted_revision
    checkpoint["source_checkpoint_lineage"][
        "calibration_source_revision"
    ] = drifted_revision
    atomic_torch_save(checkpoint, checkpoint_path)
    summary["selected_checkpoint_file_sha256"] = sha256_file(checkpoint_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="different calibration source revisions",
    ):
        build_decision(
            contract=contract,
            contract_sha256=contract_sha256,
            results_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("drift", "accepted"),
    [(5e-7, True), (2e-6, False)],
)
def test_prior_B_historical_replay_uses_contract_device_tolerance(
    tmp_path: Path,
    drift: float,
    accepted: bool,
) -> None:
    contract, _, contract_sha256 = _write_artifacts(tmp_path)
    dataset = contract["datasets"][1]["dataset"]
    summary_path = tmp_path / dataset / "B" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    replay = summary["prior_B_reference"]["continuous_reference_metrics"]
    replay["proper_time_nll"] += drift
    aligned_continuous = summary["validation_continuous_reference_metrics"][
        "proper_time_nll"
    ]
    summary["within_B_gates"]["continuous_nll_delta"] = (
        aligned_continuous - replay["proper_time_nll"]
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if accepted:
        decision = build_decision(
            contract=contract,
            contract_sha256=contract_sha256,
            results_root=tmp_path,
        )
        assert decision["all_datasets_pass"] is True
    else:
        with pytest.raises(ValueError, match="pinned continuous NLL drift"):
            build_decision(
                contract=contract,
                contract_sha256=contract_sha256,
                results_root=tmp_path,
            )
