"""Pilot isolation, position substitution and observation-invariance checks.

All numerical fits here use tiny fabricated Train/Validation populations on CPU;
they do not open project datasets and do not authorize native CUDA training.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import random
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from paper.scripts import run_phen_fixed_position_pilot as r


@pytest.fixture
def contract(tmp_path):
    files = {"runner.py": "a" * 64}
    inputs = {"inputs/data.parquet": "b" * 64, "inputs/splits.json": "c" * 64}
    return {"schema": r.SCHEMA, "seeds": [42], "model_role": "observed_time_joint_v1",
        "evaluation_scope": "validation_only", "position_encoding": dict(r.POSITION_POLICY),
        "jobs": [{"id": r.job_id(d), "dataset": d, "seed": 42, "host": "5080"} for d in r.DATASETS],
        "source": {"files": files, "files_sha256": r.canonical(files), "base_git_revision": "d" * 40},
        "input_files": inputs, "hosts": {"5080": {"root": str(tmp_path / "native"),
            "python": "/usr/bin/python3", "boot_id": "boot", "gpu_uuid": "GPU-test", "paused_processes": []}},
        "training": {"maximum_epochs": 300, "minimum_epochs": 40, "patience": 40,
            "batch_size": 128, "monitor": r.MONITOR},
        "datasets": [{"dataset_id": d, "loader": {"batch_size": 128, "max_seq_len": 84, "lookback_weeks": 84},
            "model": {"hidden_dim": 64, "quantity_variant": r.VARIANT, "lambda_log_qty": 1.,
                "lambda_tail": 0., "time_head_mode": "heteroscedastic_lognormal_duration"},
            "inherited_data_identity": {"data": {"path": "inputs/data.parquet", "sha256": "b" * 64},
                "split_manifest": {"path": "inputs/splits.json", "sha256": "c" * 64},
                "populations": {"train": {}, "validation": {}}}} for d in r.DATASETS]}


def test_exact_two_dataset_single_seed_contract(contract):
    r.validate(contract)
    assert [j["dataset"] for j in contract["jobs"]] == list(r.DATASETS)


@pytest.mark.parametrize("fault", ["seed", "host", "order", "third_dataset", "test", "test_population",
    "warmstart", "position_trainable", "position_base", "epochs", "minimum", "patience", "monitor",
    "batch", "source_empty", "source_sha", "input_sha", "path", "loss", "hidden", "head"])
def test_scope_change_fails_closed(contract, fault):
    if fault == "seed": contract["seeds"] = [42, 52]
    if fault == "host": contract["hosts"]["5090"] = contract["hosts"]["5080"]
    if fault == "order": contract["jobs"].reverse()
    if fault == "third_dataset": contract["datasets"].append(deepcopy(contract["datasets"][0]))
    if fault == "test": contract["evaluation_scope"] = "test"
    if fault == "test_population": contract["datasets"][0]["inherited_data_identity"]["populations"]["test"] = {}
    if fault == "warmstart": contract["training"]["warm_start"] = True
    if fault == "position_trainable": contract["position_encoding"]["trainable"] = True
    if fault == "position_base": contract["position_encoding"]["base"] = 100.
    if fault == "epochs": contract["training"]["maximum_epochs"] = 500
    if fault == "minimum": contract["training"]["minimum_epochs"] = 20
    if fault == "patience": contract["training"]["patience"] = 20
    if fault == "monitor": contract["training"]["monitor"] = "validation_joint_objective"
    if fault == "batch": contract["datasets"][0]["loader"]["batch_size"] = 256
    if fault == "source_empty": contract["source"]["files"] = {}
    if fault == "source_sha": contract["source"]["files_sha256"] = "e" * 64
    if fault == "input_sha": contract["input_files"]["inputs/data.parquet"] = "e" * 64
    if fault == "path": contract["input_files"]["../escape"] = "e" * 64
    if fault == "loss": contract["datasets"][0]["model"]["lambda_log_qty"] = 0.
    if fault == "hidden": contract["datasets"][0]["model"]["hidden_dim"] = 128
    if fault == "head": contract["datasets"][0]["model"]["time_head_mode"] = "legacy_clamped_rmtpp"
    with pytest.raises(ValueError):
        r.validate(contract)


def test_sinusoidal_values_no_rng_draws_and_long_positions():
    before = torch.get_rng_state().clone()
    actual = r.sinusoidal_table(257, 64, dtype=torch.float64)
    assert torch.equal(torch.get_rng_state(), before)
    assert actual.shape == (1, 257, 64)
    assert torch.equal(actual[0, 0, 0::2], torch.zeros(32, dtype=torch.float64))
    assert torch.equal(actual[0, 0, 1::2], torch.ones(32, dtype=torch.float64))
    for position in (1, 83, 128, 256):
        for pair in (0, 1, 15, 31):
            phase = position / 10000 ** (2 * pair / 64)
            assert actual[0, position, 2 * pair].item() == pytest.approx(math.sin(phase), abs=1e-12)
            assert actual[0, position, 2 * pair + 1].item() == pytest.approx(math.cos(phase), abs=1e-12)


@pytest.mark.parametrize("length,width", [(0, 64), (8, 0), (8, 63)])
def test_invalid_sinusoidal_shapes(length, width):
    with pytest.raises(ValueError): r.sinusoidal_table(length, width)


def test_full_phen_substitution_preserves_all_other_initialization_and_rng():
    from paper.scripts import run_shared_component_ablation as shared
    data = shared.raw9.prepared.fabricated_data()
    torch.manual_seed(42)
    original, _ = shared._build_model(data, "full")
    state = {k: v.clone() for k, v in original.state_dict().items()}
    before_rng = torch.get_rng_state().clone()
    old_parameters = sum(p.numel() for p in original.parameters())
    proof = r.apply_fixed_positions(original)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert "encoder.pos_emb" not in dict(original.named_parameters())
    assert "encoder.pos_emb" in dict(original.named_buffers())
    assert not original.encoder.pos_emb.requires_grad
    assert old_parameters - sum(p.numel() for p in original.parameters()) == state["encoder.pos_emb"].numel()
    assert proof["position_shape"] == list(state["encoder.pos_emb"].shape)
    for name, tensor in state.items():
        if name != "encoder.pos_emb": assert torch.equal(tensor, original.state_dict()[name]), name
    rebuilt, _, _ = r.build_fixed_model(data)
    rebuilt.load_state_dict(original.state_dict(), strict=True)
    assert r.tensor_mapping_sha(rebuilt.state_dict()) == r.tensor_mapping_sha(original.state_dict())
    old_consumer, _ = shared._build_model(data, "full")
    with pytest.raises(RuntimeError): old_consumer.load_state_dict(original.state_dict(), strict=True)
    with pytest.raises(ValueError): r.apply_fixed_positions(original)


@pytest.mark.parametrize("raise_inside", [False, True])
def test_monitor_restores_global_private_rng_and_mixed_module_modes(raise_inside):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.5))
    model.train()
    model[1].eval()
    private = torch.Generator().manual_seed(22)
    random.seed(11)
    np.random.seed(22)
    torch.manual_seed(33)
    expected = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(), private.get_state().clone())
    try:
        with r.preserve_evaluation_state(model, generators=(private,)):
            assert all(not m.training for m in model.modules())
            random.random()
            np.random.random()
            torch.rand(3)
            torch.rand(3, generator=private)
            if raise_inside: raise RuntimeError("synthetic observation error")
    except RuntimeError:
        assert raise_inside
    assert random.getstate() == expected[0]
    assert np.array_equal(np.random.get_state()[1], expected[1][1])
    assert np.random.get_state()[2:] == expected[1][2:]
    assert torch.equal(torch.get_rng_state(), expected[2])
    assert torch.equal(private.get_state(), expected[3])
    assert model.training and model[0].training and not model[1].training


def test_monitor_rejects_model_mutation():
    model = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="modified model state"):
        with r.preserve_evaluation_state(model):
            with torch.no_grad(): model.weight.add_(1)


def engine_fixture(folder):
    from paper.scripts import run_shared_component_ablation as shared
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    folder.mkdir()
    args, frame, data, interface, quantity = shared.raw9.prepared._engine_fixture(folder, shared.raw9.ARMS[0])
    data["dataset_id"] = "raf_spare_parts"
    data["loader"].update(batch_size=128, lookback_weeks=84, max_seq_len=84)
    data["quantity_boundaries_all_train_rows"] = quantity["boundaries"]
    data["inherited_data_identity"] = {"populations": {}}
    for split in ("train", "validation"):
        loader = make_loader(frame, target_split=split, shuffle=False, generator=None, **data["loader"])
        _, identity = r.common.population_identity(loader.dataset, split)
        data["inherited_data_identity"]["populations"][split] = identity
    args.model_role, args.time_alpha = "observed_time_joint_v1", 1.
    for key, value in {**r.common.INACTIVE_TAIL, **shared.raw9.input_identity()}.items(): setattr(args, key, value)
    interface.update(shared.raw9.input_identity())
    interface["source_files_sha256"] = "a" * 64
    return args, frame, data, interface, quantity


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor): assert torch.equal(left, right)
    elif isinstance(left, np.ndarray): assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left: assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right): assert_nested_equal(a, b)
    else: assert left == right


def test_actual_engine_monitor_does_not_change_trajectory_selection_or_fit_proof(tmp_path):
    """Two real PHEN optimizer runs on fabricated data, with/without monitoring."""
    from paper.scripts.count_aware_tpp_backbone import training
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    configure_runtime("cpu", 4)
    runs = []
    for monitor in (False, True):
        folder = tmp_path / ("observed" if monitor else "baseline")
        args, frame, data, interface, quantity = engine_fixture(folder)
        with r.fixed_training_hooks(data, frame, folder, monitor=monitor) as (tracked, state):
            summary, _, _ = training.train_one(args=args, frame=frame, quantity_contract=quantity,
                interface_meta=interface, backbone="full", quantity_variant=r.VARIANT, seed=42)
            proof = tracked["tracker"].proof()
        checkpoint = Path(summary["checkpoint_path"])
        history = r.read(checkpoint.parent / "history.json")["history"]
        last = torch_load_checkpoint(checkpoint.parent / "last_epoch_state.pt", map_location="cpu")
        runs.append((summary, last, history, proof))
        if monitor:
            rows = [json.loads(line) for line in (folder / "learning_curves.jsonl").read_text().splitlines()]
            assert len(rows) == len(list((folder / "curve_epochs").glob("*.json"))) == summary["completed_epochs"] == 2
            for index, row in enumerate(rows, 1):
                assert row == r.read(folder / "curve_epochs" / f"epoch_{index:04d}.json")
                assert row["train_eval"]["count"] == 28 and row["validation"]["count"] == 8
                assert row["evaluation_mode"] == "eval" and row["monitoring_rng_preserved"]
                assert row["monitoring_joint_loss_observations"] == 1
                assert row["validation"]["qty_rmse"] == history[index-1]["val_qty_rmse"]
                assert row["train_eval"]["joint_objective"] == pytest.approx(
                    row["train_eval"]["log_qty_mse"] + row["train_eval"]["time_nll"], abs=1e-5)
            model, path, digest = r.load_selected({"source": {"files_sha256": "a" * 64}}, data,
                                                folder, summary, proof, "cpu")
            first = r.evaluate_validation(data, frame, model, folder / "validation.jsonl.gz", "cpu")
            second = r.evaluate_validation(data, frame, model, folder / "replay.jsonl.gz", "cpu")
            assert r.common.compare_prediction_files(folder / "validation.jsonl.gz", folder / "replay.jsonl.gz") == 8
            assert first["state_sha256"] == second["state_sha256"] == digest
            assert first["metrics"]["overall"]["qty_rmse"] == pytest.approx(summary["best_val_qty_rmse"], abs=1e-5)
    left, right = runs
    assert left[0]["best_epoch"] == right[0]["best_epoch"]
    for key in ("model_state_dict", "optimizer_state_dict", "rng_state", "train_loader_generator_state"):
        assert_nested_equal(left[1][key], right[1][key])
    # Timing telemetry differs, but every numerical loss/metric and proof is exact.
    for a, b in zip(left[2], right[2]):
        for key in a:
            if "seconds" not in key and "throughput" not in key:
                assert_nested_equal(a[key], b[key])
    assert left[3] == right[3]


def test_immutable_receipt_and_path_confinement(tmp_path):
    r.write(tmp_path / "receipt.json", {"original": True})
    with pytest.raises(FileExistsError): r.write(tmp_path / "receipt.json", {"original": False})
    assert r.read(tmp_path / "receipt.json") == {"original": True}
    with pytest.raises(ValueError): r.below(tmp_path, "../old.pt")
    with pytest.raises(ValueError): r.below(tmp_path, "/outside.pt")
    (tmp_path / "link").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError): r.below(tmp_path, "link/checkpoint.pt")


def write_qualifications(contract, device="cuda:0"):
    root = Path(contract["hosts"]["5080"]["root"])
    for dataset in r.DATASETS:
        for mode, shape in (("synthetic", [3, 8]), ("capacity", [128, 84])):
            r.write(root / "qualification" / dataset / (mode + ".json"), {"status": "passed", "host": "5080",
                "dataset_id": dataset, "contract_sha256": r.canonical(contract),
                "source_files_sha256": contract["source"]["files_sha256"], "shape": shape, "runtime": {"device": device}})
    return root


def test_cpu_qualification_cannot_start_gpu_fit(contract):
    root = write_qualifications(contract, "cpu")
    with pytest.raises(ValueError, match="native qualification"):
        r.verify_qualification(contract, r.DATASETS[0], root)


def test_failed_raf_child_stops_before_taxi_and_records_actual_exit(contract, tmp_path, monkeypatch):
    root = write_qualifications(contract)
    (root / "source").mkdir()
    monkeypatch.setattr(r, "verify_files", lambda *_: (contract["datasets"][0], root))
    monkeypatch.setattr(r, "gpu_guard", lambda *_: {"verified": True})
    child = Mock(pid=123)
    child.wait.return_value = 7
    popen = Mock(return_value=child)
    monkeypatch.setattr(r.subprocess, "Popen", popen)
    assert r.dispatch(contract, tmp_path / "contract.json", "e" * 64, "cuda:0") == 7
    assert popen.call_count == 1
    first = r.job_id(r.DATASETS[0])
    assert r.read(root / (first + ".exit.json"))["actual_os_returncode"] == 7
    assert not (root / (r.job_id(r.DATASETS[1]) + ".launch.json")).exists()
    with pytest.raises(FileExistsError): r.dispatch(contract, tmp_path / "contract.json", "e" * 64, "cuda:0")
    assert popen.call_count == 1


def test_real_fixed_phen_synthetic_qualification(tmp_path, monkeypatch):
    from paper.scripts import run_shared_component_ablation as shared
    data = shared.raw9.prepared.fabricated_data()
    monkeypatch.setattr(r, "verify_files", lambda *_: (data, tmp_path))
    monkeypatch.setattr(r, "verify_internal_modules", lambda *_: None)
    result = r.qualify({"source": {"files_sha256": "a" * 64}}, r.DATASETS[0], "synthetic", "cpu")
    assert result["status"] == "passed" and result["shape"] == [3, 8]
    assert not result["real_data_loaded"] and not result["held_out_test_evaluated"]
    assert result["time_head_training_verification"]["nonzero_time_head_gradient_observed"]
    assert result["position_initialization"]["position_encoding"] == r.POSITION_POLICY
