import copy
import json
from pathlib import Path
import pickle

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.TPPs.CountAwareFactory import build_count_aware_model
from paper.scripts import quantity_objective_comparison as objective
from paper.scripts import run_quantity_checkpoint_diagnosis as diagnosis
from paper.scripts import time_quantity_diagnostic as engine
from simple_lab_test.search.common.runner import canonical_state_dict_sha256


POLICY = {"absolute_tolerance": 1e-5, "relative_tolerance": 1e-5,
          "paired_delta_absolute_tolerance": .001, "paired_delta_relative_tolerance": .01}


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def row_set(prediction):
    truth = np.array([0., 2., 3., 31., 46., 187., 188.], dtype=np.float32)
    return {"true_qty": truth, "pred_qty": np.asarray(prediction, dtype=np.float32),
            "history_length": np.array([1, 64, 65, 128, 129, 64, 129]),
            "series_ids": np.array(["a", "a", "b", "b", "c", "c", "d"]),
            "target_position": np.arange(1, 8), "target_seq": np.arange(11, 18),
            "quantity_train_loss": np.arange(7, dtype=np.float32),
            "log_qty_loss": np.arange(7, dtype=np.float32),
            "time_loss": np.arange(7, dtype=np.float32) - 3}


def test_fixed_bins_equality_body_tail_and_sse_additivity():
    base = row_set([1, 2, 3, 30, 40, 180, 180])
    candidate = row_set([2, 1, 4, 32, 42, 185, 186])
    result = diagnosis.paired_diagnosis(base, candidate, [2, 31, 46, 187], [64, 128])
    assert [c["n"] for c in result["quantity_cells"]] == [2, 2, 1, 1, 1]
    assert [c["n"] for c in result["history_cells"]] == [3, 2, 2]
    assert result["body"]["n"] == 5 and result["tail"]["n"] == 1
    expected = np.square(candidate["pred_qty"].astype(float) - base["true_qty"]) - np.square(base["pred_qty"].astype(float) - base["true_qty"])
    assert result["overall"]["delta_raw_sse"] == pytest.approx(expected.sum())
    assert sum(c["delta_raw_mse_contribution"] for c in result["cross_cells"]) == pytest.approx(expected.mean())
    assert result["additivity_audit"]["passed"]
    concentration = result["series_concentration"]
    assert concentration["positive_delta_sse"] - concentration["negative_delta_sse_magnitude"] == pytest.approx(expected.sum())


def test_paired_identity_rejects_reordered_targets_even_when_values_match():
    base, candidate = row_set([1] * 7), row_set([1] * 7)
    candidate["target_position"] = candidate["target_position"][::-1]
    with pytest.raises(ValueError, match="identity mismatch"):
        diagnosis.paired_diagnosis(base, candidate, [2, 31, 46, 187], [64, 128])


@pytest.mark.parametrize("field", diagnosis.METRICS)
def test_each_replay_metric_must_match(field):
    expected = dict.fromkeys(diagnosis.METRICS, 2.0)
    actual = dict(expected)
    assert diagnosis.verify_replay(actual, expected, POLICY)["passed"]
    actual[field] += .01
    with pytest.raises(ValueError, match="metric replay"):
        diagnosis.verify_replay(actual, expected, POLICY)


def test_pair_delta_gate_catches_small_effect_flip_despite_per_arm_tolerance():
    base = dict.fromkeys(diagnosis.METRICS, 100.)
    candidate = {**base, "quantity_mae": 100.0001}
    replay = {**base, "quantity_mae": 99.9999}
    diagnosis.verify_replay(replay, candidate, POLICY)
    with pytest.raises(ValueError, match="Paired effect"):
        diagnosis.verify_pair_replay(base, replay, base, candidate, POLICY)
    distorted = {**candidate, "quantity_mae": 100.000102}
    with pytest.raises(ValueError, match="Paired effect"):
        diagnosis.verify_pair_replay(base, distorted, base, candidate, POLICY)


def checkpoint_fixture(last):
    weights = {"weight": torch.arange(3, dtype=torch.float32)}
    tensor_hash = canonical_state_dict_sha256(weights)
    arm = {"condition": {"case": "raw_softplus", "mu": 1., "raw_scale": 3.}}
    expected = {"raw_quantity_rmse": 2., "epoch": 120, "global_step": 36000}
    state = {"arm_contract_sha256": diagnosis.sha_json(arm), "arm_contract": arm,
             "selector": "last" if last else "raw_quantity_rmse", "epoch": 120,
             "global_step": 36000, "expected": expected, "state_sha256": tensor_hash}
    payload = {"schema_version": "quantity_comparison_engine_v1", "contract_sha256": diagnosis.sha_json(arm),
               "model_state_dict": weights, "global_step": 36000}
    if last:
        payload.update(contract=arm, epoch=120, history=[expected], model_state_sha256=tensor_hash, rng_state={"numpy": np.random.get_state()})
    else:
        payload.update(condition=arm["condition"], selector="raw_quantity_rmse", applicable=True,
                       best_epoch=120, best_value=2., state_sha256=tensor_hash)
    return state, payload


@pytest.mark.parametrize("last", [False, True])
def test_weights_only_roundtrip_and_identity_rejection(tmp_path, last):
    state, payload = checkpoint_fixture(last)
    path = tmp_path / "state.pt"
    torch.save(payload, path)
    loaded = diagnosis.load_checkpoint(path, last=last)
    diagnosis.verify_checkpoint_payload(loaded, state, canonical_state_dict_sha256)
    loaded["model_state_dict"]["weight"].add_(1)
    with pytest.raises(ValueError, match="tensor-state"):
        diagnosis.verify_checkpoint_payload(loaded, state, canonical_state_dict_sha256)
    loaded = diagnosis.load_checkpoint(path, last=last)
    loaded["global_step"] += 1
    with pytest.raises(ValueError, match="global step"):
        diagnosis.verify_checkpoint_payload(loaded, state, canonical_state_dict_sha256)


class UnapprovedObject:
    pass


def test_checkpoint_loader_never_falls_back_to_unsafe_pickle(tmp_path):
    path = tmp_path / "unsafe.pt"
    torch.save({"unapproved": UnapprovedObject()}, path)
    for last in (False, True):
        with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
            diagnosis.load_checkpoint(path, last=last)


@pytest.mark.parametrize("case", diagnosis.CASES)
def test_real_functional_case_routing_is_causal_and_does_not_update_weights(case, monkeypatch):
    stats = objective.QuantityStatistics(mu=1.2, raw_scale=5.)
    torch.manual_seed(42)
    model, _ = build_count_aware_model("titantpp", hidden_dim=8, train_log_mean=stats.mu,
                                     max_seq_len=8, time_intercept_limit=300.)
    # Move away from the calibration point so an incorrect original link is
    # observably different for the two softplus cases.
    with torch.no_grad():
        model.quantity_head.bias.add_(1.)
    model.eval()
    original_hash = canonical_state_dict_sha256(model.state_dict())
    dts = torch.tensor([[0., 0., 1., 2.], [0., 1., 1., 3.]])
    mask = torch.tensor([[False, False, True, True], [False, True, True, True]])
    quantities = torch.tensor([[0., 0., 2., 4.], [0., 1., 2., 3.]])

    def loader_for(values, times):
        ds = TensorDataset(times, mask, values)
        ds.parts, ds.index = ["a", "b"], [(0, 0), (1, 0)]
        ds.seq_lists = [[1, 2], [1, 2]]
        ds.val_lists = [[2., float(values[0, -1])], [2., float(values[1, -1])]]
        return DataLoader(ds, batch_size=2, shuffle=False)

    def forbidden_optimizer(*args, **kwargs):
        raise AssertionError("Inference may not construct an optimizer")
    monkeypatch.setattr(torch.optim, "AdamW", forbidden_optimizer)
    rows = diagnosis.collect_rows(model, loader_for(quantities, dts), objective, engine, stats, case, "cpu", lambda: None)
    with torch.no_grad():
        expected = objective.joint_causal_batch_objective(model, dts, mask, quantities, statistics=stats, case=objective.QuantityCase(case))
        original = objective.joint_causal_batch_objective(model, dts, mask, quantities, statistics=stats, case=objective.QuantityCase.B_LOG_ORIGINAL)
    assert np.array_equal(rows["pred_qty"], expected["pred_qty"].numpy())
    if "softplus" in case:
        assert not np.allclose(rows["pred_qty"], original["pred_qty"].numpy())
    changed = quantities.clone()
    changed[:, -1] += 100
    changed[~mask] = 999.
    changed_time = dts.clone()
    changed_time[~mask] = 999.
    replay = diagnosis.collect_rows(model, loader_for(changed, changed_time), objective, engine, stats, case, "cpu", lambda: None)
    assert np.array_equal(rows["pred_qty"], replay["pred_qty"])
    assert canonical_state_dict_sha256(model.state_dict()) == original_hash
    assert all(parameter.grad is None for parameter in model.parameters())


def test_actual_contract_prepare_only_and_scope_tampering():
    path = Path(__file__).resolve().parents[3] / "paper/contracts/quantity_checkpoint_diagnosis_v1.json"
    contract = json.loads(path.read_text())
    diagnosis.validate_contract(contract)
    assert diagnosis.sha_json(contract) == "0b1ca25d348d90c05a959330da52b87fdd43d8bb633bd70dc5d27f6b09848a85"
    changed = copy.deepcopy(contract)
    changed["datasets"][0]["states"][0]["scope"] = "test"
    with pytest.raises(ValueError, match="eight fixed states"):
        diagnosis.validate_contract(changed)


def synthetic_instacart_contract():
    """Adapt metadata only to exercise the schema guard; never load its data."""
    path = Path(__file__).resolve().parents[3] / "paper/contracts/quantity_checkpoint_diagnosis_v1.json"
    contract = json.loads(path.read_text())
    contract["schema"] = diagnosis.INSTACART_SCHEMA
    contract["datasets"] = [contract["datasets"][0]]
    dataset = contract["datasets"][0]
    dataset["dataset_id"] = dataset["data"]["dataset_id"] = "insta_market_basket"
    for state in dataset["states"]:
        state["arm_contract"]["identity"]["dataset_id"] = "insta_market_basket"
        state["arm_contract_sha256"] = diagnosis.sha_json(state["arm_contract"])
    return contract


def test_v2_admits_only_one_instacart_dataset_and_v1_scope_stays_fixed():
    contract = synthetic_instacart_contract()
    diagnosis.validate_contract(contract)
    changed = copy.deepcopy(contract)
    changed["schema"] = diagnosis.SCHEMA
    with pytest.raises(ValueError, match="Dataset scope"):
        diagnosis.validate_contract(changed)
    changed = copy.deepcopy(contract)
    changed["datasets"].append(copy.deepcopy(changed["datasets"][0]))
    with pytest.raises(ValueError, match="Dataset scope"):
        diagnosis.validate_contract(changed)
    changed = copy.deepcopy(contract)
    changed["datasets"][0]["dataset_id"] = "yellow_trip_hourly"
    with pytest.raises(ValueError, match="Dataset scope"):
        diagnosis.validate_contract(changed)


@pytest.mark.parametrize("target_splits", [["test"], ["validation", "test"], ["train"]])
def test_v2_refuses_wrong_loader_split_before_checkpoint_loading(target_splits):
    contract = synthetic_instacart_contract()
    for state in contract["datasets"][0]["states"]:
        state["arm_contract"]["validation_loader"]["target_splits"] = target_splits
        state["arm_contract_sha256"] = diagnosis.sha_json(state["arm_contract"])
    with pytest.raises(ValueError, match="Only validation loader targets"):
        diagnosis.validate_contract(contract)


def test_v2_refuses_held_out_and_incomplete_state_scope():
    contract = synthetic_instacart_contract()
    changed = copy.deepcopy(contract)
    changed["held_out_evaluated"] = True
    with pytest.raises(ValueError, match="Inference-only scope"):
        diagnosis.validate_contract(changed)
    contract["datasets"][0]["states"].pop()
    with pytest.raises(ValueError, match="eight fixed states"):
        diagnosis.validate_contract(contract)


def test_new_json_only_and_byte_budget(tmp_path):
    path = tmp_path / "result.json"
    diagnosis.write_json(path, {"n": 1}, 100)
    with pytest.raises(FileExistsError):
        diagnosis.write_json(path, {"n": 2}, 100)
    with pytest.raises(ValueError, match="byte budget"):
        diagnosis.write_json(tmp_path / "large.json", {"text": "x" * 100}, 100)
