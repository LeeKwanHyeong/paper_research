"""Synthetic CPU checks through the real shared training and checkpoint path."""
from __future__ import annotations

import copy
import json

import pytest
import torch

from models.TPPs.CountAwareFactory import validate_checkpoint_route
from paper.scripts.count_aware_tpp_backbone import observed_time as obs, training
from simple_lab_test.search.tests import test_observed_time_joint_training as h


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def pair_args(monkeypatch, tmp_path, backbone, name="run"):
    args = h.args_for(monkeypatch, tmp_path, backbone, name)
    args.model_role = obs.PAIR_ROLE
    return args


def test_three_arms_same_batches_and_selected_checkpoint_cache(monkeypatch, tmp_path):
    traces_by_arm = []
    for backbone in obs.PAIR_BACKBONES:
        traces = {"train": [], "validation": []}
        h.install_loader(monkeypatch, traces)
        args = pair_args(monkeypatch, tmp_path, backbone)
        summary, _, _ = h.run(args, backbone)
        last = h.torch_load_checkpoint(h.directory(args, backbone) / "last_epoch_state.pt", map_location="cpu")
        assert summary["best_epoch"] == min(last["history"], key=lambda x: x["val_qty_rmse"])["epoch"]
        assert summary["completed_epochs"] == 3 and not summary["stopped_early"]
        assert all(x["train_all_finite"] for x in last["history"])
        saved_traces = copy.deepcopy(traces)
        cached, _, _ = h.run(args, backbone)
        assert cached["checkpoint_state_sha256"] == summary["checkpoint_state_sha256"]
        assert traces == saved_traces
        traces_by_arm.append(traces)
    assert all(x == traces_by_arm[0] for x in traces_by_arm)


@pytest.mark.parametrize("backbone", obs.PAIR_BACKBONES[1:])
def test_exact_interrupted_resume_and_foreign_route_rejection(monkeypatch, tmp_path, backbone):
    h.install_loader(monkeypatch, {"train": [], "validation": []})
    full_args = pair_args(monkeypatch, tmp_path, backbone, "full")
    expected, _, _ = h.run(full_args, backbone)
    args = pair_args(monkeypatch, tmp_path, backbone, "resumed")
    actual_save = training.atomic_torch_save
    def save_and_stop(payload, path):
        actual_save(payload, path)
        if path.name == "last_epoch_state.pt" and payload["epoch"] == 1:
            raise RuntimeError("synthetic interruption")
    monkeypatch.setattr(training, "atomic_torch_save", save_and_stop)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        h.run(args, backbone)
    monkeypatch.setattr(training, "atomic_torch_save", actual_save)
    actual, _, _ = h.run(args, backbone)
    assert actual["checkpoint_state_sha256"] == expected["checkpoint_state_sha256"]
    a = h.torch_load_checkpoint(h.directory(full_args, backbone)/"last_epoch_state.pt", map_location="cpu")
    b = h.torch_load_checkpoint(h.directory(args, backbone)/"last_epoch_state.pt", map_location="cpu")
    for key in ("model_state_dict", "best_state_dict", "optimizer_state_dict", "rng_state", "train_loader_generator_state", "history"):
        h.same(a[key], b[key])
    for foreign in set(obs.PAIR_BACKBONES) - {backbone}:
        with pytest.raises(ValueError):
            validate_checkpoint_route(b, foreign)


@pytest.mark.parametrize("backbone", obs.PAIR_BACKBONES[1:])
def test_strict_earliest_tie(monkeypatch, tmp_path, backbone):
    h.install_loader(monkeypatch, {"train": [], "validation": []})
    actual = training.evaluate
    def tied(**kwargs):
        return {**actual(**kwargs), "qty_rmse": 42.0}
    monkeypatch.setattr(training, "evaluate", tied)
    summary, _, _ = h.run(pair_args(monkeypatch, tmp_path, backbone), backbone)
    assert summary["best_epoch"] == 1


@pytest.mark.parametrize("dataset_id", ["intermittent_frozen_5000", "yellow_trip_hourly"])
def test_three_arm_production_cli_synthetic_parquet(monkeypatch, tmp_path, dataset_id):
    original = h.args_for
    def arguments(*args, **kwargs):
        result = original(*args, **kwargs)
        result.model_role = obs.PAIR_ROLE
        result.backbones = ",".join(obs.PAIR_BACKBONES)
        return result
    monkeypatch.setattr(h, "args_for", arguments)
    h.test_production_cli_on_synthetic_parquet_uses_one_observation_law(monkeypatch, tmp_path, dataset_id)
    for backbone in obs.PAIR_BACKBONES:
        summary = json.loads((tmp_path/"run"/"runs"/backbone/h.VARIANT/"seed_42"/"summary.json").read_text())
        assert summary["status"] == "success" and summary["completed_epochs"] == 1
        assert summary["encoder_config"]["time_head"]["observation_likelihood"] == obs.observation_contract(dataset_id)


def test_foreign_model_role_rejects_pair_candidate(monkeypatch, tmp_path):
    args = h.args_for(monkeypatch, tmp_path, obs.PAIR_BACKBONES[1])
    with pytest.raises(ValueError, match="unsupported"):
        h.run(args, obs.PAIR_BACKBONES[1])
