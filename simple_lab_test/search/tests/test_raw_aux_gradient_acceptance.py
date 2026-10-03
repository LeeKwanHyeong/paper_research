"""CPU synthetic replay, partition accounting and fail-closed gate tests."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from paper.scripts import raw_aux_gradient_acceptance as acceptance
from paper.scripts import intermittent_cell_statistics as cells
from paper.scripts.mixed_quantity_objective import MixedQuantityObjective
from paper.scripts.quantity_comparison_engine import run_case
from paper.scripts.quantity_objective_comparison import QuantityCase
from paper.scripts.raw_aux_gradient_control import RawAuxGradientControl
from simple_lab_test.search.common.runner import canonical_state_dict_sha256
from simple_lab_test.search.tests.test_mixed_quantity_engine import make_model, statistics, tensors


@pytest.fixture(autouse=True)
def cpu_single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def rows(errors=None, truth=None, time=-2.):
    truth = np.asarray(truth if truth is not None else [1., 5., 40., 100., 200., 100.])
    error = np.asarray(errors if errors is not None else [1.] * len(truth))
    n = len(truth)
    return {"series_ids": np.asarray(["synthetic"] * n), "target_position": np.arange(n),
            "target_seq": np.arange(n), "true_qty": truth, "pred_qty": truth + error,
            "history_length": np.asarray([1, 64, 65, 128, 129, 255])[:n],
            "time_loss": np.full(n, time), "log_qty_loss": np.full(n, .1),
            "raw_qty_loss": np.full(n, .2), "quantity_train_loss": np.full(n, .3),
            "objective_loss": np.full(n, time + .3)}


def gate_inputs(candidate_errors=None, truth=None, baseline_errors=None, baseline_time=-2., candidate_time=-2.):
    arms = {}
    for name in acceptance.CASES:
        values = rows(baseline_errors if name == acceptance.CASES[0] else candidate_errors, truth,
                      baseline_time if name == acceptance.CASES[0] else candidate_time)
        summary = cells.summarize_rows(values, [2, 31, 46, 187], [64, 128])
        objective = MixedQuantityObjective("B_log_original" if name == acceptance.CASES[0] else "mixed_original",
                                           0. if name == acceptance.CASES[0] else .2, 1., "c" * 64)
        arm = {"arm_id": name, "design_sha256": acceptance.DESIGN_SHA256, "synthetic": True,
               "condition": {"case": "B_log_original", "mixed_objective": objective.to_dict(),
                             **({"raw_aux_control": RawAuxGradientControl().to_dict()} if name == acceptance.CASES[2] else {})},
               "evaluation_split": "validation", "held_out_evaluated": False,
               "execution_identity": {"runtime": "synthetic"}, "initial_state_sha256": "a" * 64,
               "training_contract_common_sha256": "f" * 64,
               "training_exposure": {"epochs": 2, "expected_global_steps": 4},
               "train_batch_order_sha256": ["b" * 64, "c" * 64],
               "selectors": {"raw_quantity_rmse": {"best_epoch": 1, "global_step": 2, "state_sha256": "d" * 64},
                             "legacy_time_loss": {"best_epoch": 2, "global_step": 4, "state_sha256": "e" * 64}}}
        for scope, epoch, state in (("primary", 1, "d"), ("last120", 2, "e")):
            arm[scope] = {"epoch": epoch, "global_step": epoch * 2, "state_sha256": state * 64,
                          "count": len(values["true_qty"]), "batches": 2,
                          "validation_batch_sha256": "b" * 64, "ordered_target_sha256": "c" * 64,
                          "metrics": acceptance._metrics(summary), "partitions": deepcopy(summary), "replay_passed": True}
        arms[name] = arm
    return arms


def test_exact_design_freeze_and_changed_field_rejected(tmp_path):
    design = acceptance.load_design()
    design["choice"]["cap_ratio"] = .5
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(design))
    with pytest.raises(ValueError, match="design hash"):
        acceptance.load_design(changed)


def test_all_gates_pass_but_synthetic_never_promoted():
    result = acceptance.assess_acceptance(gate_inputs(), synthetic=True)
    assert all(result["candidate_passed"].values())
    assert len(result["checks"]) == 64
    assert len(result["paired_comparisons"]) == 6
    assert result["eligible_candidate"] is None
    assert result["status"] == "synthetic_only"
    assert not result["benchmark_superiority_established"]
    with pytest.raises(ValueError, match="split mismatch"):
        acceptance.assess_acceptance(gate_inputs())


def test_quantity_bin_rmse_rejects_even_if_overall_and_bin_mae_improve():
    result = acceptance.assess_acceptance(gate_inputs(candidate_errors=[0, 0, 0, 1.5, 0, 0]), synthetic=True)
    checks = [x for x in result["checks"] if x["candidate"] == acceptance.CASES[2] and x["scope"] == "primary"]
    assert next(x for x in checks if x["metric"] == "raw_quantity_rmse")["passed"]
    assert next(x for x in checks if x["metric"] == "quantity_bin_3_MAE")["passed"]
    assert not next(x for x in checks if x["metric"] == "quantity_bin_3_RMSE")["passed"]
    assert not result["candidate_passed"][acceptance.CASES[2]]


@pytest.mark.parametrize("b,time", [(-2., -1.98), (-.1, -.09), (0., .01), (2., 2.02)])
def test_time_bound_is_additive_even_for_negative_loss(b, time):
    assert all(acceptance.assess_acceptance(gate_inputs(baseline_time=b, candidate_time=time), synthetic=True)["candidate_passed"].values())
    assert not any(acceptance.assess_acceptance(gate_inputs(baseline_time=b, candidate_time=time + 2e-7), synthetic=True)["candidate_passed"].values())


def test_empty_required_quantity_bin_fails_but_empty_cross_cells_do_not():
    result = acceptance.assess_acceptance(gate_inputs(truth=[1, 5, 40, 100, 100, 100]), synthetic=True)
    assert not any(result["candidate_passed"].values())
    assert any(x["metric"] == "quantity_bin_4_RMSE" and not x["passed"] for x in result["checks"])
    assert all(acceptance.assess_acceptance(gate_inputs(), synthetic=True)["candidate_passed"].values())


def test_zero_baseline_gate_tolerance_is_in_rmse_units():
    yes = acceptance.assess_acceptance(gate_inputs(candidate_errors=[0.] * 6, baseline_errors=[0.] * 6), synthetic=True)
    assert all(yes["candidate_passed"].values())
    no = acceptance.assess_acceptance(gate_inputs(candidate_errors=[1e-6] * 6, baseline_errors=[0.] * 6), synthetic=True)
    assert not any(no["candidate_passed"].values())


def test_synthetic_cannot_be_promoted_by_relabeling_flags_and_epoch():
    arms = gate_inputs()
    for name, arm in arms.items():
        arm["synthetic"] = False
        arm["training_exposure"]["epochs"] = 120
        objective = arm["condition"]["mixed_objective"]
        objective["alpha"] = 0. if name == acceptance.CASES[0] else 1.769152228020146
        objective["calibration_sha256"] = acceptance.CALIBRATION_SHA256
    with pytest.raises(ValueError, match="production scope"):
        acceptance.assess_acceptance(arms)


@pytest.mark.parametrize("mutation", ["order", "runtime", "steps", "state", "selectors", "heldout", "sse", "nonfinite", "common"])
def test_identity_and_aggregation_corruption_fail_closed(mutation):
    arms = gate_inputs()
    arm = arms[acceptance.CASES[2]]
    if mutation == "order": arm["primary"]["ordered_target_sha256"] = "0" * 64
    elif mutation == "runtime": arm["execution_identity"] = {"runtime": "different"}
    elif mutation == "steps": arm["last120"]["global_step"] = 3
    elif mutation == "state": arm["primary"]["state_sha256"] = "e" * 64
    elif mutation == "selectors": del arm["selectors"]["legacy_time_loss"]
    elif mutation == "heldout": arm["held_out_evaluated"] = True
    elif mutation == "sse": arm["primary"]["partitions"]["quantity_cells"][0]["raw_squared_error_sum"] += 1
    elif mutation == "nonfinite": arm["primary"]["partitions"]["overall"]["raw_squared_error_sum"] = float("nan")
    elif mutation == "common": arm["training_contract_common_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        acceptance.assess_acceptance(arms, synthetic=True)


def test_partition_boundaries_and_paired_signed_additivity():
    values = rows(truth=[2, 31, 46, 187, 188, 100])
    summary = cells.summarize_rows(values, [2, 31, 46, 187], [64, 128])
    assert [x["count"] for x in summary["quantity_cells"]] == [1, 1, 1, 2, 1]
    assert [x["count"] for x in summary["history_cells"]] == [2, 2, 2]
    result = acceptance.assess_acceptance(gate_inputs(candidate_errors=[-2, 0, 0, 1, 1, 1]), synthetic=True)
    for pair in result["paired_comparisons"]:
        for label in acceptance.PARTITIONS:
            assert sum(x["delta_SSE"] for x in pair[label]) == pytest.approx(pair["overall"]["delta_SSE"])


@pytest.mark.parametrize("corruption", ["predicted_sum", "negative_under", "negative_over"])
def test_consistent_partition_tampering_cannot_bypass_local_error_accounting(corruption):
    arms = gate_inputs()
    partitions = arms[acceptance.CASES[2]]["primary"]["partitions"]
    all_cells = [partitions["overall"]] + [row for label in acceptance.PARTITIONS for row in partitions[label]]
    for row in all_cells:
        n = row["count"]
        if corruption == "predicted_sum":
            row["predicted_quantity_sum"] += n
        else:
            # Preserve all existing partition-sum and sign/absolute identities
            # while forging a negative directional absolute-error sum.
            row["under_absolute_error_sum"] = -n if corruption == "negative_under" else 2 * n
            row["over_absolute_error_sum"] = 2 * n if corruption == "negative_under" else -n
            row["signed_error_sum"] = row["over_absolute_error_sum"] - row["under_absolute_error_sum"]
            row["predicted_quantity_sum"] = row["true_quantity_sum"] + row["signed_error_sum"]
            row["mean_bias"] = row["signed_error_sum"] / n if n else None
    # The original shared conservation audit still passes this consistent corruption.
    cells._conservation(partitions["overall"], {label: partitions[label] for label in acceptance.PARTITIONS})
    with pytest.raises(ValueError, match="accounting mismatch|Negative directional"):
        acceptance.assess_acceptance(arms, synthetic=True)


def _run_synthetic(path, name):
    dataset = TensorDataset(*tensors())
    train = DataLoader(dataset, batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(41))
    validation = DataLoader(dataset, batch_size=2, shuffle=False, generator=torch.Generator().manual_seed(42))
    objective = MixedQuantityObjective("B_log_original" if name == acceptance.CASES[0] else "mixed_original",
                                       0. if name == acceptance.CASES[0] else .2, 1., "c" * 64)
    model = make_model()
    run_case(model=model, train_loader=train, validation_loader=validation, case=QuantityCase.B_LOG_ORIGINAL,
             statistics=statistics(), output_dir=path, epochs=2, seed=42,
             identity={"source": {"revision": "synthetic"}, "data": {"sha256": "synthetic"},
                       "runtime": {"name": "synthetic-cpu"}, "execution_contract_sha256": "a" * 64},
             mixed_objective=objective, raw_aux_control=RawAuxGradientControl() if name == acceptance.CASES[2] else None)
    return model, validation, objective


def test_actual_cpu_titan_six_state_replay_no_validation_gradients_and_restore(tmp_path, monkeypatch):
    # Only generated <=4-row TensorDataset checkpoints; never load research PT/data.
    prepared = {name: _run_synthetic(tmp_path / name, name) for name in acceptance.CASES}
    def forbid(*args, **kwargs):
        raise AssertionError("Validation must not query gradients or update optimizer")
    monkeypatch.setattr(torch.autograd, "grad", forbid)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbid)
    results = {}
    for name, (model, loader, objective) in prepared.items():
        model.train()
        before = canonical_state_dict_sha256(model.state_dict())
        rng = torch.get_rng_state().clone()
        generator = loader.generator.get_state().clone()
        results[name] = acceptance.evaluate_checkpoints(model=model, validation_loader=loader, statistics=statistics(),
                                                        objective=objective, arm_dir=tmp_path / name, arm_id=name, synthetic=True)
        assert model.training and canonical_state_dict_sha256(model.state_dict()) == before
        assert torch.equal(rng, torch.get_rng_state()) and torch.equal(generator, loader.generator.get_state())
        for scope in acceptance.SCOPES:
            assert results[name][scope]["count"] == 4
            assert results[name][scope]["partitions"]["conservation_audit"]["passed"]
    result = acceptance.assess_acceptance(results, synthetic=True)
    assert result["status"] == "synthetic_only"
    assert not any(result["candidate_passed"].values())  # absent train-frozen bins fail closed


def test_replay_refuses_training_split_or_changed_controller_before_load(tmp_path, monkeypatch):
    model, loader, objective = _run_synthetic(tmp_path / "capped", acceptance.CASES[2])
    def forbid(*args, **kwargs): raise AssertionError("Unexpected checkpoint load")
    monkeypatch.setattr(acceptance, "torch_load_checkpoint", forbid)
    with pytest.raises(ValueError, match="Validation-only"):
        acceptance.evaluate_checkpoints(model=model, validation_loader=loader, statistics=statistics(), objective=objective,
                                        arm_dir=tmp_path / "capped", arm_id=acceptance.CASES[2])
    path = tmp_path / "capped/contract.json"
    contract = json.loads(path.read_text())
    contract["condition"]["raw_aux_control"]["cap_ratio"] = .5
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="raw_aux_control"):
        acceptance.evaluate_checkpoints(model=model, validation_loader=loader, statistics=statistics(), objective=objective,
                                        arm_dir=tmp_path / "capped", arm_id=acceptance.CASES[2], synthetic=True)
