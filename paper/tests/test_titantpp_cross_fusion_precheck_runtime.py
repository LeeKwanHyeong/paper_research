"""Focused admission, duplicate-run, and owned-child checks; no GPUs or data."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock, patch

import pytest

from paper.scripts import titantpp_cross_fusion_precheck_runtime as runtime

pytest_plugins = ("simple_lab_test.search.tests.test_multilag_detail_execution",)


def contract(tmp_path):
    files = {"model.py": "a" * 64}
    datasets = ["taxi", "raf", "intermittent"]
    jobs = [dict(id=f"{data}_{arm}_42", host="5080" if i % 2 == 0 else "5090",
                 dataset=data, arm=arm, seed=42)
            for i, (data, arm) in enumerate((d, a) for d in datasets for a in ("control", "cross"))]
    return dict(schema=runtime.SCHEMA, evaluation_scope="validation_only", held_out_test_evaluated=False,
        datasets=[{"dataset_id": d} for d in datasets], jobs=jobs,
        hosts={h: {"root": str(tmp_path / h), "source_root": str(tmp_path),
                   "python": sys.executable, "gpu_uuid": "GPU-fixed", "runtime_expected": {}} for h in ("5080", "5090")},
        source={"files": files, "files_sha256": runtime.sha_json(files)}, input_files={"data.parquet": "b" * 64},
        training=dict(maximum_epochs=300, minimum_epochs=40, patience=40, batch_size=128,
                      monitor="validation_raw_quantity_rmse"), limits={"fit_wall_seconds": 129600})


def test_contract_accepts_only_six_paired_validation_jobs(tmp_path):
    c = contract(tmp_path)
    assert len(runtime.validate_contract(c, "5080")) == 3
    mutations = [lambda x: x.pop("held_out_test_evaluated"),
        lambda x: x.update(held_out_test_evaluated=True),
        lambda x: x.update(evaluation_scope="test"),
        lambda x: x["jobs"][0].update(seed=52),
        lambda x: x["jobs"][0].update(id=x["jobs"][1]["id"]),
        lambda x: x["jobs"][0].update(id="../escape"),
        lambda x: x["jobs"][0].update(dataset="unknown"),
        lambda x: x["jobs"][0].update(host="5090"),
        lambda x: x["training"].update(maximum_epochs=10),
        lambda x: x["limits"].update(fit_wall_seconds=999999),
        lambda x: x["source"].update(files={})]
    for mutate in mutations:
        changed = deepcopy(c)
        mutate(changed)
        with pytest.raises(ValueError):
            runtime.validate_contract(changed, "5080")


def test_sha_and_exclusive_files(tmp_path):
    c = contract(tmp_path)
    (tmp_path / "model.py").write_text("x = 1\n")
    (tmp_path / "data.parquet").write_bytes(b"fabricated")
    c["source"]["files"]["model.py"] = runtime.sha_file(tmp_path / "model.py")
    c["input_files"]["data.parquet"] = runtime.sha_file(tmp_path / "data.parquet")
    runtime.verify_files(c, tmp_path)
    (tmp_path / "model.py").write_text("x = 2\n")
    with pytest.raises(ValueError, match="SHA mismatch"):
        runtime.verify_files(c, tmp_path)
    for name in ("../bad", "/absolute"):
        with pytest.raises(ValueError):
            runtime.local_path(tmp_path, name)
    runtime.write(tmp_path / "receipt.json", {"status": "first"}, exclusive=True)
    with pytest.raises(FileExistsError):
        runtime.write(tmp_path / "receipt.json", {"status": "second"}, exclusive=True)
    assert runtime.read(tmp_path / "receipt.json")["status"] == "first"


def test_duplicate_dispatch_rejected_before_gpu_call(tmp_path, monkeypatch):
    c = contract(tmp_path)
    root = Path(c["hosts"]["5080"]["root"])
    runtime.write(root / "qualification.json", {"status": "passed", "contract_sha256": runtime.sha_json(c)})
    runtime.write(root / "start.json", {"original": True})
    monkeypatch.chdir(tmp_path)
    with patch.object(runtime, "gpu_snapshot") as gpu:
        with pytest.raises(ValueError, match="no restart"):
            runtime.dispatch(c, "5080", tmp_path / "contract.json")
        gpu.assert_not_called()


def test_runtime_mismatch_stops_before_new_jobs(tmp_path, monkeypatch):
    c = contract(tmp_path)
    root = Path(c["hosts"]["5080"]["root"])
    runtime.write(root / "qualification.json", {"status": "passed", "contract_sha256": "wrong"})
    monkeypatch.chdir(tmp_path)
    with patch.object(runtime.subprocess, "Popen") as spawn:
        with pytest.raises(ValueError, match="Qualification"):
            runtime.dispatch(c, "5080", tmp_path / "contract.json")
        spawn.assert_not_called()


def test_gpu_identity_and_busy_list():
    with patch.object(runtime.subprocess, "run", side_effect=[
            subprocess.CompletedProcess([], 0, "GPU-fixed, RTX, 32768\n", ""),
            subprocess.CompletedProcess([], 0, "GPU-fixed, 123\n", "")]):
        assert runtime.gpu_snapshot("GPU-fixed")["compute_pids"] == [123]
    with patch.object(runtime.subprocess, "run", return_value=
            subprocess.CompletedProcess([], 0, "GPU-wrong, RTX, 32768\n", "")):
        with pytest.raises(ValueError, match="UUID"):
            runtime.gpu_snapshot("GPU-fixed")


def test_timeout_cleanup_never_targets_an_unowned_group():
    child = Mock(pid=123)
    child.poll.return_value = None
    with patch.object(runtime.os, "getpgid", return_value=456), patch.object(runtime.os, "killpg") as kill:
        with pytest.raises(ValueError, match="identity"):
            runtime.terminate_owned_child(child)
        kill.assert_not_called()
    child.poll.return_value = 0
    with patch.object(runtime.os, "killpg") as kill:
        runtime.terminate_owned_child(child)
        kill.assert_not_called()


@pytest.mark.parametrize("peak_bytes, passes", [(100, True), (400, False)])
def test_full_batch_parallel_memory_admission(tmp_path, monkeypatch, peak_bytes, passes):
    from paper.scripts import run_titantpp_cross_fusion as runner
    c = contract(tmp_path)
    for item in c["datasets"]:
        item["loader"] = {"max_seq_len": 256}
    for job in c["jobs"]:
        job["arm"] = runner.ARMS[job["arm"] == "cross"]
    measured = {"gpu": {"total_memory_bytes": 1000}}
    synthetic = {"cuda_memory_peaks": {arm: {"max_memory_reserved_bytes": peak_bytes}
                                       for arm in runner.ARMS}}
    monkeypatch.chdir(tmp_path)
    with patch.object(runtime, "verify_files"), \
            patch.object(runtime, "gpu_snapshot", return_value={"compute_pids": []}), \
            patch.object(runtime, "runtime_check", return_value=measured), \
            patch.object(runner, "synthetic_check", return_value=synthetic) as check:
        if passes:
            receipt = runtime.qualify(c, "5080")
            assert receipt["parallel_memory_admission"]["passed"]
        else:
            with pytest.raises(ValueError, match="memory admission"):
                runtime.qualify(c, "5080")
        check.assert_called_once_with(device="cuda:0", batch_size=128, length=256)


def test_dispatch_records_actual_child_exit_and_never_retries(tmp_path, monkeypatch):
    c = contract(tmp_path)
    root = Path(c["hosts"]["5080"]["root"])
    runtime.write(root / "qualification.json", {"status": "passed", "contract_sha256": runtime.sha_json(c)})
    monkeypatch.chdir(tmp_path)
    processes = []
    for index in range(3):
        child = Mock(pid=123 + index)
        child.poll.return_value = 7 if index == 1 else 0
        processes.append(child)
    with patch.object(runtime, "gpu_snapshot", return_value={"compute_pids": []}), \
            patch.object(runtime.subprocess, "Popen", side_effect=processes) as spawn:
        assert runtime.dispatch(c, "5080", tmp_path / "contract.json") == 1
        assert spawn.call_count == 3
    receipt = runtime.read(root / "final_receipt.json")
    assert receipt["status"] == "failed"
    assert sorted(v["actual_os_returncode"] for v in receipt["actual_worker_exits"].values()) == [0, 0, 7]
    assert receipt["held_out_test_evaluated"] is False
    assert receipt["dispatcher_own_exit_directly_observed"] is False


@pytest.mark.parametrize("dataset_id", ["yellow_trip_hourly", "raf_spare_parts", "intermittent_frozen_5000"])
def test_fabricated_full_training_route_and_selected_checkpoint(cpu_training, admitted_contract, tmp_path, dataset_id):
    """Two one-epoch fake-data fits exercise the actual loader/head/selector/save path."""
    import torch
    from paper.scripts import run_titantpp_cross_fusion as runner
    from paper.scripts.count_aware_tpp_backbone import observed_time, training
    from models.TPPs import CountAwareFactory

    c = deepcopy(admitted_contract)
    c["training"] = {"maximum_epochs": 300, "minimum_epochs": 40, "patience": 40}
    data = deepcopy(c["datasets"][0])
    data["dataset_id"] = dataset_id
    data["model"]["time_observation_contract"] = observed_time.observation_contract(dataset_id)
    frame, inputs = runner.base.prepare_admitted_data(data)
    assert inputs["held_out_materialized"] is False
    quantity = {"boundaries": data["quantity_boundaries_all_train_rows"],
        "strata": [{"label": f"bin_{i}"} for i in range(5)]}
    for arm in runner.ARMS:
        args = runner.training_args(c, data, tmp_path / dataset_id, arm)
        assert (args.epochs, args.min_epochs, args.early_stopping_patience) == (300, 40, 40)
        args.device = "cpu"
        args.epochs = args.min_epochs = args.early_stopping_patience = 1
        interface = runner.time_interface(data, frame, c, arm=arm)
        summary, _, _ = training.train_one(args=args, frame=frame,
            quantity_contract=quantity, interface_meta=interface, backbone=arm,
            quantity_variant=runner.VARIANT, seed=42)
        assert summary["status"] == "success" and summary["completed_epochs"] == 1
        assert summary["held_out_test_evaluated"] is False
        payload = torch.load(summary["checkpoint_path"], map_location="cpu", weights_only=False)
        CountAwareFactory.validate_checkpoint_route(payload, arm)
        assert payload["best_epoch"] == 1
        assert summary["best_val_qty_rmse"] == pytest.approx(payload["selected_metric_value"], rel=1e-10)
