"""Scope, immutable evidence, admission and PHEN round-trip regression checks."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from paper.scripts import run_phen_instacart_3seed as r


@pytest.fixture
def contract(tmp_path):
    files = {"runner.py": "a" * 64}
    inputs = {"inputs/data.parquet": "b" * 64, "inputs/splits.json": "c" * 64}
    return {"schema": r.SCHEMA, "campaign": "fixture", "seeds": [42, 52, 62],
        "host_by_seed": dict(r.HOST_BY_SEED), "model_role": "observed_time_joint_v1",
        "evaluation_scope": "validation_only", "source": {"files": files,
            "files_sha256": r.canonical(files), "base_git_revision": "d" * 40},
        "input_files": inputs, "hosts": {h: {"root": str(tmp_path / h), "python": "/usr/bin/python3",
            "boot_id": "boot", "gpu_uuid": "GPU-test", "paused_processes": []} for h in ("5080", "5090")},
        "training": {"maximum_epochs": 300, "minimum_epochs": 40, "patience": 40,
            "batch_size": 128, "monitor": r.MONITOR},
        "datasets": [{"dataset_id": "insta_market_basket", "loader": {"batch_size": 128,
            "max_seq_len": 64, "lookback_weeks": 52}, "model": {"hidden_dim": 64,
            "quantity_variant": r.VARIANT, "lambda_log_qty": 1., "lambda_tail": 0.,
            "time_head_mode": "heteroscedastic_lognormal_duration"}, "inherited_data_identity": {
                "data": {"path": "inputs/data.parquet", "sha256": "b" * 64},
                "split_manifest": {"path": "inputs/splits.json", "sha256": "c" * 64},
                "populations": {"train": {}, "validation": {}}}}]}


def test_three_fresh_phen_fits_only(contract):
    assert r.validate(contract)["dataset_id"] == "insta_market_basket"
    assert [s for s in r.SEEDS if r.HOST_BY_SEED[str(s)] == "5090"] == [52, 62]


@pytest.mark.parametrize("fault", ["seed", "placement", "dataset", "test_scope", "test_population",
    "monitor", "epochs", "batch", "source_empty", "source_digest", "input_digest", "path_escape",
    "source_revision", "warmstart", "quantity_loss", "hidden", "time_head", "max_seq", "lookback"])
def test_contract_scope_changes_fail_closed(contract, fault):
    data = contract["datasets"][0]
    if fault == "seed": contract["seeds"][-1] = 72
    if fault == "placement": contract["host_by_seed"]["42"] = "5090"
    if fault == "dataset": data["dataset_id"] = "yellow_trip_hourly"
    if fault == "test_scope": contract["evaluation_scope"] = "test"
    if fault == "test_population": data["inherited_data_identity"]["populations"]["test"] = {}
    if fault == "monitor": contract["training"]["monitor"] = "validation_joint_objective"
    if fault == "epochs": contract["training"]["maximum_epochs"] = 40
    if fault == "batch": contract["training"]["batch_size"] = 256
    if fault == "source_empty": contract["source"]["files"] = {}
    if fault == "source_digest": contract["source"]["files_sha256"] = "bad"
    if fault == "input_digest": contract["input_files"]["inputs/data.parquet"] = "e" * 64
    if fault == "path_escape": contract["input_files"]["../other"] = "e" * 64
    if fault == "source_revision": contract["source"]["base_git_revision"] = "bad"
    if fault == "warmstart": contract["training"]["warm_start"] = True
    if fault == "quantity_loss": data["model"]["lambda_log_qty"] = 2.
    if fault == "hidden": data["model"]["hidden_dim"] = 128
    if fault == "time_head": data["model"]["time_head_mode"] = "legacy_clamped_rmtpp"
    if fault == "max_seq": data["loader"]["max_seq_len"] = 128
    if fault == "lookback": data["loader"]["lookback_weeks"] = 104
    with pytest.raises(ValueError): r.validate(contract)


def test_exclusive_receipt_preserves_first_attempt(tmp_path):
    path = tmp_path / "receipt.json"
    r.write(path, {"first": True})
    with pytest.raises(FileExistsError): r.write(path, {"first": False})
    assert r.read(path) == {"first": True}


def test_artifact_path_cannot_escape_or_follow_symlink(tmp_path):
    with pytest.raises(ValueError): r.below(tmp_path, "../original.pt")
    with pytest.raises(ValueError): r.below(tmp_path, "/tmp/original.pt")
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    (root / "redirect").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"): r.below(root, "redirect/checkpoint.pt")


def qualification(contract, host, device="cuda:0"):
    root = Path(contract["hosts"][host]["root"])
    for mode, shape in (("synthetic", [3, 8]), ("capacity", [128, 64])):
        r.write(root / "qualification" / (mode + ".json"), {"status": "passed", "host": host,
            "contract_sha256": r.canonical(contract), "source_files_sha256": contract["source"]["files_sha256"],
            "shape": shape, "runtime": {"device": device}})
    return root


def test_cpu_smoke_does_not_admit_cuda_training(contract):
    root = qualification(contract, "5080", device="cpu")
    with pytest.raises(ValueError, match="native qualification"):
        r.verify_qualification(contract, "5080", root)


def test_wrong_contract_qualification_rejected(contract):
    root = qualification(contract, "5080")
    contract["campaign"] = "another_attempt"
    with pytest.raises(ValueError, match="native qualification"):
        r.verify_qualification(contract, "5080", root)


def test_guard_rejects_gpu_overlap_and_preserves_stopped_state(contract, monkeypatch):
    expected = {"pid": 10, "pgid": 10, "start_ticks": 123, "argv": ["original"], "cwd": "/old"}
    contract["hosts"]["5080"]["paused_processes"] = [expected]
    original_read = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda p, *a, **kw: "boot" if str(p) == "/proc/sys/kernel/random/boot_id" else original_read(p, *a, **kw))
    actual = {**expected, "state": "T"}
    monkeypatch.setattr(r, "process_identity", lambda _pid: actual)
    outputs = iter(["GPU-test\n", "10\n"])
    monkeypatch.setattr(r.subprocess, "check_output", lambda *a, **kw: next(outputs))
    assert r.gpu_guard(contract, "5080")["gpu_processes"] == [10]
    outputs = iter(["GPU-test\n", "10\n99\n"])
    with pytest.raises(ValueError, match="Unexpected GPU process"): r.gpu_guard(contract, "5080")
    outputs = iter(["GPU-test\n"])
    actual["state"] = "S"
    with pytest.raises(ValueError, match="not stopped"): r.gpu_guard(contract, "5080")


def test_dispatch_stops_after_failed_first_seed_without_retry(contract, tmp_path, monkeypatch):
    root = qualification(contract, "5090")
    (root / "source").mkdir()
    monkeypatch.setattr(r, "verify_files", lambda *_: (contract["datasets"][0], root))
    monkeypatch.setattr(r, "gpu_guard", lambda *_: {"verified": True})
    child = Mock(pid=123)
    child.wait.return_value = 7
    popen = Mock(return_value=child)
    monkeypatch.setattr(r.subprocess, "Popen", popen)
    assert r.dispatch(contract, "5090", tmp_path / "contract.json", "e" * 64, "cuda:0") == 7
    assert popen.call_count == 1
    assert r.read(root / "seed52.exit.json")["actual_os_returncode"] == 7
    assert not (root / "seed62.launch.json").exists()
    with pytest.raises(FileExistsError):
        r.dispatch(contract, "5090", tmp_path / "contract.json", "e" * 64, "cuda:0")
    assert popen.call_count == 1


def test_real_phen_cpu_synthetic_roundtrip(tmp_path, monkeypatch):
    from paper.scripts import run_titantpp_shared_history_raw9 as raw9
    data = raw9.prepared.fabricated_data()
    data["loader"]["max_seq_len"] = 64
    data["model"]["time_observation_contract"] = {"mode": "positive_integer_round_clamp_v1", "unit": "day", "top_code": 30}
    monkeypatch.setattr(r, "verify_files", lambda *_: (data, tmp_path))
    result = r.qualify({"source": {"files_sha256": "a" * 64}}, "5080", "synthetic", "cpu")
    assert result["status"] == "passed" and result["parameter_count"] > 10000
    assert result["real_data_loaded"] is False
    assert result["time_head_training_verification"]["nonzero_time_head_gradient_observed"] is True
    assert result["shape"] == [3, 8]


@pytest.mark.parametrize("key", ["qty_rmse", "qty_mae", "time_nll"])
def test_self_agreeing_replay_cannot_replace_training_validation(key):
    values = {"qty_rmse": 2., "qty_mae": 1., "time_nll": 3.}
    evaluation = {"metrics": {"overall": values}}
    summary = {"best_val_qty_rmse": 2., "best_val_qty_mae": 1., "best_val_time_nll": 3.}
    r.verify_selected_metrics(evaluation, summary)
    values[key] += 1.
    with pytest.raises(ValueError, match="did not reproduce training"):
        r.verify_selected_metrics(evaluation, summary)


def test_actual_two_epoch_training_checkpoint_and_individual_validation(tmp_path):
    """Exercise the exact engine hooks and new adapter's load/evaluation paths."""
    import torch
    from paper.scripts import run_shared_component_ablation as shared
    from paper.scripts.count_aware_tpp_backbone import training
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    r.runtime("cpu")
    args, frame, data, interface, quantity = shared.raw9.prepared._engine_fixture(tmp_path, shared.raw9.ARMS[0])
    data["loader"].update(batch_size=128, lookback_weeks=52, max_seq_len=64)
    data["model"]["time_observation_contract"] = {"mode": "positive_integer_round_clamp_v1", "unit": "day", "top_code": 30}
    data["quantity_boundaries_all_train_rows"] = quantity["boundaries"]
    data["inherited_data_identity"] = {"populations": {}}
    for split in ("train", "validation"):
        loader = make_loader(frame, target_split=split, batch_size=128, lookback_weeks=52,
                             max_seq_len=64, shuffle=False, generator=None)
        _, identity = r.common.population_identity(loader.dataset, split)
        data["inherited_data_identity"]["populations"][split] = identity
    args.dataset_contract = "insta_market_basket"
    args.model_role = "observed_time_joint_v1"
    args.max_seq_len, args.lookback_weeks = 64, 52
    args.time_alpha = 1.
    for key, value in {**r.common.INACTIVE_TAIL, **shared.raw9.input_identity()}.items():
        setattr(args, key, value)
    interface.update(shared.raw9.input_identity())
    interface["time_head"]["observation_likelihood"] = data["model"]["time_observation_contract"]
    contract = {"source": {"files_sha256": "a" * 64}}
    interface["source_files_sha256"] = contract["source"]["files_sha256"]
    with shared._engine_hooks(data, "full") as tracked:
        summary, _, _ = training.train_one(args=args, frame=frame, quantity_contract=quantity,
            interface_meta=interface, backbone="full", quantity_variant=r.VARIANT, seed=42)
        proof = tracked["tracker"].proof()
    assert summary["completed_epochs"] == 2
    model, checkpoint, state = r.load_selected(contract, data, 42, tmp_path, summary, proof, "cpu")
    binary_sha = r.sha(checkpoint)
    first = r.evaluate_validation(data, frame, model, tmp_path / "reference.jsonl.gz", "cpu")
    r.verify_selected_metrics(first, summary)
    del model
    model, _, second_state = r.load_selected(contract, data, 42, tmp_path, summary, proof, "cpu")
    second = r.evaluate_validation(data, frame, model, tmp_path / "replay.jsonl.gz", "cpu")
    r.verify_selected_metrics(second, summary)
    assert r.common.compare_prediction_files(tmp_path / "reference.jsonl.gz", tmp_path / "replay.jsonl.gz") == 8
    assert state == second_state and binary_sha == r.sha(checkpoint)
