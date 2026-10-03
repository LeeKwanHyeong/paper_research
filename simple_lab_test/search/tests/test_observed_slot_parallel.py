"""Synthetic CPU checks for the bounded observed-slot parallel execution path."""

from __future__ import annotations

import copy
import ast
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from paper.scripts import run_observed_slot_partition as partition
from paper.scripts import observed_slot_parallel_common as common


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def execution_contract():
    """Use only frozen configuration metadata; never load the referenced rows."""
    inherited = json.loads((common.ROOT / common.INHERITED).read_text())
    datasets = copy.deepcopy(inherited["datasets"])
    for data in datasets:
        data["epochs"] = 120
    hosts = {}
    for alias, assignments in common.ASSIGNMENTS.items():
        root = "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/observed_slot_seed42_parallel_synthetic_contract_test"
        hosts[alias] = {
            "root": root, "source_root": root + "/source", "output_dir": root + "/run",
            "assigned_datasets": list(assignments), "environment": copy.deepcopy(common.ENVIRONMENT),
            "python": (
                "/opt/miniconda3/envs/ai_env/bin/python3.12" if alias == "5090"
                else "/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python3.12"
            ),
            "gpu_uuid": (
                "GPU-c9adc246-ef66-7906-bc07-6e152fa9952f" if alias == "5090"
                else "GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6"
            ),
        }
    return {
        "schema": common.SCHEMA, "limits": copy.deepcopy(common.LIMITS),
        "policy": copy.deepcopy(common.POLICY), "acceptance": copy.deepcopy(common.ACCEPTANCE),
        "cost_gates": copy.deepcopy(common.COST_GATES), "seed": 42, "epochs": 120,
        "model": {"baseline": "titantpp", "candidate": "titantpp_observed_slot_memory"},
        "hosts": hosts, "datasets": datasets, "total_optimizer_steps": 4544160,
        "design_sha256": common.sha_file(common.ROOT / common.DESIGN),
        "inherited_data_contract_sha256": common.sha_file(common.ROOT / common.INHERITED),
        "source": {"base_git_revision": "1" * 40, "files": {}, "files_sha256": common.sha_json({})},
        "authorization": {
            "scope": "fresh_six_arm_same_host_paired_training_after_two_native_cuda_qualifications",
            "user_instruction": "이 작업 관련해서 5080/5090 병렬 실행하자",
        },
    }


def test_frozen_host_assignments_and_training_budget_are_explicit(tmp_path):
    contract = execution_contract()
    path = tmp_path / "contract.json"
    common.write_json(path, contract)
    assert common.read_contract(path, "5090", check_source=False) == contract
    assert partition.validate_assignment(contract, "5090")["assigned_datasets"] == ["insta_market_basket"]
    assert partition.validate_assignment(contract, "5080")["assigned_datasets"] == [
        "intermittent_frozen_5000", "yellow_trip_hourly",
    ]
    assert contract["model"] == {"baseline": "titantpp", "candidate": "titantpp_observed_slot_memory"}
    for data in contract["datasets"]:
        args = partition.training_args(contract, data, tmp_path / data["dataset_id"])
        assert args.epochs == args.min_epochs == 120
        assert args.batch_size == 128
        assert args.device == "cuda:0"
        assert args.checkpoint_monitor == "validation_raw_quantity_rmse"
        assert args.time_intercept_limit == 300.0
        assert args.lambda_tail == 0.0


@pytest.mark.parametrize("mutation", [
    lambda c: c["limits"].update(total_seconds=86401),
    lambda c: c["hosts"]["5080"]["assigned_datasets"].reverse(),
    lambda c: c["hosts"]["5090"]["assigned_datasets"].append("yellow_trip_hourly"),
    lambda c: c["policy"].update(resume=True),
    lambda c: c["datasets"][0]["loader"].update(batch_size=256),
    lambda c: c.update(seed=43),
])
def test_contract_drift_is_rejected_before_accessing_inputs(tmp_path, mutation):
    contract = execution_contract()
    mutation(contract)
    path = tmp_path / "changed_contract.json"
    common.write_json(path, contract)
    with pytest.raises(ValueError):
        common.read_contract(path, "5090", check_source=False)


def test_source_binding_covers_imported_dependency_bytes(tmp_path, monkeypatch):
    for relative in (*common.ENTRYPOINTS, common.DESIGN, common.INHERITED):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}" if path.suffix == ".json" else "")
    entrypoint = tmp_path / common.ENTRYPOINTS[0]
    entrypoint.write_text("from models.synthetic_dependency import value\n")
    dependency = tmp_path / "models/synthetic_dependency.py"
    dependency.parent.mkdir()
    dependency.write_text("value = 1\n")
    scan = common.source_manifest
    files = scan(tmp_path)
    assert "models/synthetic_dependency.py" in files
    contract = {"source": {"files": files, "files_sha256": common.sha_json(files)}}
    monkeypatch.setattr(common, "source_manifest", lambda: scan(tmp_path))
    common.verify_source(contract)
    dependency.write_text("value = 2\n")
    with pytest.raises(ValueError, match="source changed"):
        common.verify_source(contract)


def permit_with_local_receipts(tmp_path):
    contract = execution_contract()
    for alias in common.ASSIGNMENTS:
        contract["hosts"][alias]["root"] = str(tmp_path / alias)
        contract["hosts"][alias]["runtime_expected"] = {"device": "cuda:0"}
    permit = {
        "schema": "observed_slot_parallel_start_v1", "contract_sha256": common.sha_json(contract),
        "started_at_unix": 1000.0, "deadline_unix": 87400.0, "qualifications": {},
    }
    for alias, host in contract["hosts"].items():
        receipt = {
            "status": "passed", "host": alias, "contract_sha256": common.sha_json(contract),
            "source_files_sha256": contract["source"]["files_sha256"],
            "runtime": {"device": "cuda:0", "gpu": {"uuid": host["gpu_uuid"]}},
        }
        path = Path(host["root"]) / "qualification/receipt.json"
        common.write_json(path, receipt)
        permit["qualifications"][alias] = {
            "path": str(path), "sha256": common.sha_file(path),
            "receipt": receipt, "receipt_sha256": common.sha_json(receipt),
        }
    return contract, permit


@pytest.mark.parametrize("corruption", ["status", "gpu", "source"])
def test_own_host_requires_the_other_hosts_bound_native_qualification(tmp_path, monkeypatch, corruption):
    contract, permit = permit_with_local_receipts(tmp_path)
    monkeypatch.setattr(common.time, "time", lambda: 1100.0)
    common.verify_permit(contract, permit, "5090")
    other = permit["qualifications"]["5080"]
    if corruption == "status":
        other["receipt"]["status"] = "failed"
    elif corruption == "gpu":
        other["receipt"]["runtime"]["gpu"]["uuid"] = "GPU-unqualified"
    else:
        other["receipt"]["source_files_sha256"] = "0" * 64
    other["receipt_sha256"] = common.sha_json(other["receipt"])
    with pytest.raises(ValueError):
        common.verify_permit(contract, permit, "5090")


def test_local_qualification_file_must_match_the_permit_bytes(tmp_path, monkeypatch):
    contract, permit = permit_with_local_receipts(tmp_path)
    monkeypatch.setattr(common.time, "time", lambda: 1100.0)
    path = Path(permit["qualifications"]["5090"]["path"])
    changed = copy.deepcopy(permit["qualifications"]["5090"]["receipt"])
    changed["status"] = "failed"
    common.write_json(path, changed)
    with pytest.raises(ValueError, match="binding mismatch"):
        common.verify_permit(contract, permit, "5090")


def test_launch_uses_one_fixed_deadline_instead_of_resetting_budget_per_host():
    permit = {"started_at_unix": 1000.0, "deadline_unix": 87400.0}
    assert partition.fixed_start(permit, wall=lambda: 1100.0, monotonic=lambda: 17.0) == 86317.0
    assert partition.fixed_start(permit, wall=lambda: 47000.0, monotonic=lambda: 117.0) == 40517.0


@pytest.mark.parametrize("permit, now", [
    ({"started_at_unix": 1000.0, "deadline_unix": 87400.0}, 999.0),
    ({"started_at_unix": 1000.0, "deadline_unix": 87400.0}, 87400.0),
    ({"started_at_unix": 1000.0, "deadline_unix": 87401.0}, 1001.0),
    ({"started_at_unix": float("nan"), "deadline_unix": 87400.0}, 1001.0),
])
def test_expired_future_or_extended_launch_budget_is_rejected(permit, now):
    with pytest.raises(ValueError):
        partition.fixed_start(permit, wall=lambda: now, monotonic=lambda: 17.0)


def synthetic_loader(*, seed=42, change=None):
    row = torch.arange(9, dtype=torch.float32)[:, None]
    position = torch.arange(5, dtype=torch.float32)[None]
    dts = 0.2 + row / 10 + position
    quantities = 1.0 + row + 2 * position
    mask = torch.arange(5)[None] < (torch.arange(9) % 4 + 2)[:, None]
    dts[~mask] = 0.0
    quantities[~mask] = 0.0
    if change is not None:
        quantities = change(quantities)
    dataset = TensorDataset(torch.zeros_like(dts), dts, mask, torch.zeros_like(dts), quantities)
    return DataLoader(dataset, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(seed))


def test_exposure_hash_covers_actual_batches_and_is_paired_across_epochs():
    baseline, candidate = [], []
    checks = []
    left = partition.ExposureLoader(synthetic_loader(), "train", baseline, lambda: checks.append(True))
    right = partition.ExposureLoader(synthetic_loader(), "train", candidate, lambda: checks.append(True))
    for _ in range(2):
        assert len(list(left)) == len(list(right)) == 3
    assert baseline == candidate
    assert [row["epoch"] for row in baseline] == [1, 2]
    assert all(row["count"] == 9 and row["batches"] == 3 for row in baseline)
    assert all(row["split"] == "train" for row in baseline)
    assert baseline[0]["batch_order_sha256"] != baseline[1]["batch_order_sha256"]
    assert len(checks) >= 12


@pytest.mark.parametrize("change", [
    lambda quantity: quantity + 1.0,
    lambda quantity: quantity.double(),
])
def test_exposure_hash_detects_actual_value_and_dtype_drift(change):
    expected, changed = [], []
    list(partition.ExposureLoader(synthetic_loader(), "train", expected, lambda: None))
    list(partition.ExposureLoader(synthetic_loader(change=change), "train", changed, lambda: None))
    assert expected[0]["count"] == changed[0]["count"]
    assert expected[0]["batch_order_sha256"] != changed[0]["batch_order_sha256"]


def test_exposure_does_not_record_an_incomplete_epoch():
    records = []
    checks = 0

    def timeout():
        nonlocal checks
        checks += 1
        if checks == 2:
            raise TimeoutError("synthetic fixed-deadline exhaustion")

    audited = partition.ExposureLoader(synthetic_loader(), "train", records, timeout)
    with pytest.raises(TimeoutError, match="fixed-deadline"):
        list(audited)
    assert records == []


def test_paired_initialization_matches_actual_common_tensors_and_preserves_rng():
    from models.TPPs.CountAwareFactory import build_count_aware_model
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256

    data = execution_contract()["datasets"][0]
    torch.manual_seed(199)
    caller_rng = torch.get_rng_state().clone()
    identity = partition.paired_initialization(data)
    assert torch.equal(torch.get_rng_state(), caller_rng)
    kwargs = {key: value for key, value in data["model"].items() if key != "backbone"}
    kwargs.update(
        train_log_mean=data["statistics"]["train_log_mean"],
        train_log_std=data["statistics"]["train_log_std"],
        max_seq_len=data["loader"]["max_seq_len"],
    )
    torch.manual_seed(42)
    baseline, _ = build_count_aware_model("titantpp", **kwargs)
    torch.manual_seed(42)
    candidate, _ = build_count_aware_model("titantpp_observed_slot_memory", **kwargs)
    common_state = {key: candidate.state_dict()[key] for key in baseline.state_dict()}
    assert identity["shared_initial_state_sha256"] == canonical_state_dict_sha256(common_state)
    assert identity["baseline_initial_state_sha256"] == canonical_state_dict_sha256(baseline.state_dict())
    assert identity["candidate_initial_state_sha256"] == canonical_state_dict_sha256(candidate.state_dict())
    assert set(identity["shared_state_keys"]) == set(baseline.state_dict())


def test_audited_wrapper_runs_real_paired_training_and_records_actual_exposure(monkeypatch, tmp_path):
    from paper.scripts.count_aware_tpp_backbone import training

    contract = execution_contract()
    data = copy.deepcopy(contract["datasets"][0])
    data["model"]["hidden_dim"] = 16
    data["loader"].update(batch_size=4, max_seq_len=8)
    data["expected_global_steps"] = 6
    for population in data["inherited_data_identity"]["populations"].values():
        population["target_count"] = 9

    def make_loader(_frame, *, target_split, batch_size, shuffle, generator, **_):
        assert target_split in {"train", "validation"}
        assert batch_size == 4
        reference = synthetic_loader(seed=44 if target_split == "train" else 55)
        return DataLoader(reference.dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)

    monkeypatch.setattr(training, "make_loader", make_loader)
    original_loader, original_save = training.make_loader, training.atomic_torch_save
    initialization = partition.paired_initialization(data)
    completed = {}
    statuses = []
    for backbone in (partition.BASELINE, partition.CANDIDATE):
        args = partition.training_args(contract, data, tmp_path)
        # This fixture changes only local synthetic runtime/budget. The actual
        # production constructor is separately checked to require CUDA/120.
        args.device = "cpu"
        args.epochs = args.min_epochs = args.early_stopping_patience = 2
        args.model_role = "hard_lmm_observed_slot_memory_candidate" if backbone == partition.CANDIDATE else "existing_backbone"
        with partition.audited_training(training, data, lambda: None, lambda epoch, _: statuses.append((backbone, epoch))) as exposures:
            summary, _, _ = training.train_one(
                args=args, frame=None,
                quantity_contract={"boundaries": [], "strata": [{"label": "synthetic_all"}]},
                interface_meta={
                    "train_target_mean": data["statistics"]["train_log_mean"],
                    "train_target_std": data["statistics"]["train_log_std"],
                    "time_head": {"time_initial_intercept": 0.0},
                    "data_scope": "synthetic_cpu_only",
                },
                backbone=backbone, quantity_variant=partition.VARIANT, seed=42,
            )
        assert training.make_loader is original_loader
        assert training.atomic_torch_save is original_save
        assert summary["completed_epochs"] == 2
        assert partition.audit_exposure(exposures, data, epochs=2) == 6
        role = "candidate" if backbone == partition.CANDIDATE else "baseline"
        assert summary["initial_state_sha256"] == initialization[role + "_initial_state_sha256"]
        completed[backbone] = copy.deepcopy(exposures)
    assert completed[partition.BASELINE] == completed[partition.CANDIDATE]
    assert statuses == [(partition.BASELINE, 1), (partition.BASELINE, 2), (partition.CANDIDATE, 1), (partition.CANDIDATE, 2)]


def test_audited_wrapper_restores_training_hooks_after_failure():
    from paper.scripts.count_aware_tpp_backbone import training

    original_loader, original_save = training.make_loader, training.atomic_torch_save
    with pytest.raises(RuntimeError, match="synthetic interrupted run"):
        with partition.audited_training(training, {}, lambda: None, lambda *_: None):
            assert training.make_loader is not original_loader
            raise RuntimeError("synthetic interrupted run")
    assert training.make_loader is original_loader
    assert training.atomic_torch_save is original_save


def test_checkpoint_replay_uses_p95_body_and_strictly_above_p99_tail(monkeypatch, tmp_path):
    from paper.scripts import run_taxi_quantity_interface_ablation as interface
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256

    data = copy.deepcopy(execution_contract()["datasets"][0])
    data["model"]["hidden_dim"] = 16
    data["loader"].update(batch_size=2, max_seq_len=8)
    data["statistics"].update(train_log_mean=math.log(2.0), train_log_std=1.0)
    data["quantity_boundaries_all_train_rows"] = [2.0, 5.0, 10.0, 20.0]
    data["inherited_data_identity"]["populations"]["validation"]["target_count"] = 6
    model, metadata = partition._build_model(data, partition.BASELINE)
    checkpoint = tmp_path / "synthetic_selected.pt"
    torch.save({
        "backbone": partition.BASELINE, "variant": partition.VARIANT,
        "encoder_config": metadata, "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False, "model_state_dict": model.state_dict(),
        "model_state_sha256": canonical_state_dict_sha256(model.state_dict()),
    }, checkpoint)
    # Exact threshold values and the gap between p95/p99 distinguish all four
    # possible boundary mistakes. The genuine zero-weight quantity head emits 1.
    targets = torch.tensor([2.0, 5.0, 10.0, 11.0, 20.0, 21.0])
    dts = torch.ones(6, 2)
    quantities = torch.stack((torch.ones(6), targets), dim=1)
    mask = torch.ones(6, 2, dtype=torch.bool)
    rows = TensorDataset(torch.zeros_like(dts), dts, mask, torch.zeros_like(dts), quantities)
    requests = []

    def validation_loader(frame, **kwargs):
        assert frame is None
        assert kwargs["target_split"] == "validation"
        assert kwargs["shuffle"] is False
        requests.append(kwargs)
        return DataLoader(rows, batch_size=kwargs["batch_size"], shuffle=False)

    monkeypatch.setattr(interface, "make_loader", validation_loader)
    replay = partition.replay_checkpoint(checkpoint, data, None, lambda: None, device="cpu")
    assert len(requests) == 1
    assert replay["count"] == 6
    assert replay["body"]["count"] == 3
    assert replay["body"]["qty_mae"] == pytest.approx(14.0 / 3.0, abs=1e-6)
    assert replay["tail"]["count"] == 1
    assert replay["tail"]["qty_mae"] == pytest.approx(20.0, abs=1e-6)
    assert [cell["count"] for cell in replay["quantity_cells"]] == [1, 1, 1, 2, 1]


def test_last30_summary_uses_sample_sd_and_only_the_last_30_epochs():
    history = [
        {"epoch": epoch, "val_qty_rmse": 10000.0 + epoch if epoch <= 90 else float(epoch - 90)}
        for epoch in range(1, 121)
    ]
    summary = partition.last30_summary(history)
    assert summary["mean"] == pytest.approx(15.5)
    # Integers 1..30 have population variance 899/12; ddof=1 gives 77.5.
    assert summary["sd"] == pytest.approx(math.sqrt(77.5))
    assert summary["count"] == 30
    assert summary["standard_deviation"] == "sample_ddof_1"
    with pytest.raises(ValueError):
        partition.last30_summary(history[:29])


def test_slot_cli_branch_pushes_down_splits_and_binds_validation_population(tmp_path):
    from models.TPPs.CountAwareTitanSlotMemory import SLOT_MEMORY_ROLE
    from paper.scripts import run_count_aware_tpp_backbone_control as cli

    tree = ast.parse(Path(cli.__file__).read_text())

    def branch_for(call_name):
        branches = [
            node for node in ast.walk(tree) if isinstance(node, ast.If)
            and any(
                isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Name)
                and statement.value.func.id == call_name
                for statement in node.body
            )
        ]
        assert len(branches) == 1, call_name
        return ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(branches[0])], type_ignores=[]))

    sentinel_frame, sentinel_population, calls = object(), object(), []
    args = SimpleNamespace(
        model_role=SLOT_MEMORY_ROLE, data=tmp_path / "never_materialized.parquet",
        lookback_weeks=520, max_seq_len=256,
    )

    def pushed_down(path):
        calls.append(("train_validation_pushdown", path))
        return sentinel_frame

    def unrestricted_read(*_args, **_kwargs):
        raise AssertionError("Slot role must not materialize unrestricted parquet rows")

    def fixed_population(frame, **kwargs):
        assert frame is sentinel_frame
        assert kwargs == {"target_split": "validation", "lookback_weeks": 520, "max_seq_len": 256}
        calls.append(("fixed_validation_population", kwargs["target_split"]))
        return None, sentinel_population

    # Execute the actual CLI branch predicates/bodies with local stand-ins; no
    # full command, file loader, training run, or held-out rows are accessed.
    namespace = dict(vars(cli), args=args, frame=sentinel_frame, validation_target_population=None)
    namespace.update(load_train_validation_frame=pushed_down,
                     exact_target_population=fixed_population,
                     pl=SimpleNamespace(read_parquet=unrestricted_read))
    exec(compile(branch_for("load_train_validation_frame"), str(cli.__file__), "exec"), namespace)
    exec(compile(branch_for("exact_target_population"), str(cli.__file__), "exec"), namespace)
    assert namespace["raw_frame"] is sentinel_frame
    assert namespace["validation_target_population"] is sentinel_population
    assert calls == [
        ("train_validation_pushdown", args.data), ("fixed_validation_population", "validation"),
    ]


def local_supervisor_inputs(tmp_path, monkeypatch):
    contract = execution_contract()
    root = tmp_path / "owned_run"
    (root / "source").mkdir(parents=True)
    contract["hosts"]["5090"].update(
        root=str(root), source_root=str(root / "source"), output_dir=str(root / "run"), python=sys.executable,
    )
    now = time.time()
    permit = {"started_at_unix": now - 1.0, "deadline_unix": now + 86399.0}
    permit_path = tmp_path / "permit.json"
    common.write_json(permit_path, permit)
    monkeypatch.setattr(common, "read_contract", lambda *_args, **_kwargs: contract)
    monkeypatch.setattr(common, "verify_permit", lambda *_: None)
    monkeypatch.setattr(common, "gpu_pids", lambda _: set())
    monkeypatch.setattr(common, "apply_environment", lambda _: None)
    monkeypatch.setattr(partition, "verify_host_paths", lambda _: None)
    return contract, tmp_path / "contract.json", permit_path


def test_existing_output_refuses_resume_before_any_child_is_created(monkeypatch, tmp_path):
    contract, contract_path, permit_path = local_supervisor_inputs(tmp_path, monkeypatch)
    output = Path(contract["hosts"]["5090"]["output_dir"])
    output.mkdir()
    original = output / "existing_checkpoint.pt"
    original.write_bytes(b"existing run must remain untouched")
    launched = []
    monkeypatch.setattr(partition.subprocess, "Popen", lambda *args, **kwargs: launched.append(args))
    with pytest.raises(ValueError, match="Fresh output"):
        partition.supervisor(contract_path, permit_path, "5090")
    assert launched == []
    assert original.read_bytes() == b"existing run must remain untouched"


def test_supervisor_deadline_stops_only_its_owned_local_process(monkeypatch, tmp_path):
    contract, contract_path, permit_path = local_supervisor_inputs(tmp_path, monkeypatch)
    real_popen = subprocess.Popen
    unrelated = real_popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    children = []

    def local_child(_args, **kwargs):
        # Preserve the supervisor's actual owned process group, inherited pipe,
        # log file and cwd while substituting a harmless CPU sleeper for CUDA.
        child = real_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(partition.subprocess, "Popen", local_child)
    monkeypatch.setattr(partition, "fixed_start", lambda _: time.monotonic() + 0.05)
    monkeypatch.setattr(partition, "check_storage", lambda *_: 0)
    try:
        with pytest.raises(ValueError, match="deadline reached"):
            partition.supervisor(contract_path, permit_path, "5090")
        assert len(children) == 1
        assert children[0].poll() is not None
        assert unrelated.poll() is None
        status = json.loads((Path(contract["hosts"]["5090"]["output_dir"]) / "status.json").read_text())
        assert status["status"] == "stopped"
        assert status["automatic_retry"] is False
    finally:
        for child in [*children, unrelated]:
            if child.poll() is None:
                os.killpg(child.pid, 15)
            child.wait(timeout=5)


def metrics():
    return {
        "selected": {
            "qty_rmse": 10.0, "qty_mae": 5.0, "time_nll": 2.0,
            "body": {"qty_mae": 3.0}, "tail": {"qty_mae": 20.0},
        },
        "last30": {"mean": 10.0, "sd": 1.0},
    }


def test_acceptance_distinguishes_success_from_completed_quality_failure():
    baseline = metrics()
    candidate = copy.deepcopy(baseline)
    candidate["selected"]["qty_rmse"] = 8.0
    candidate["selected"]["qty_mae"] = 4.0
    candidate["selected"]["body"]["qty_mae"] = 2.0
    candidate["selected"]["tail"]["qty_mae"] = 15.0
    candidate["last30"] = {"mean": 8.0, "sd": 0.5}
    accepted = partition.paired_acceptance(baseline, candidate)
    assert accepted["accepted"] is True
    assert all(accepted["checks"].values())
    candidate["selected"]["qty_rmse"] = 11.0
    rejected = partition.paired_acceptance(baseline, candidate)
    assert rejected["accepted"] is False
    assert not all(rejected["checks"].values())


def test_missing_acceptance_measurements_are_an_error_not_a_quality_rejection():
    incomplete = metrics()
    del incomplete["selected"]["tail"]
    with pytest.raises((KeyError, ValueError)):
        partition.paired_acceptance(metrics(), incomplete)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), None])
def test_nonfinite_acceptance_measurements_are_a_run_error(invalid):
    invalid_metrics = metrics()
    invalid_metrics["selected"]["qty_rmse"] = invalid
    with pytest.raises(ValueError, match="finite"):
        partition.paired_acceptance(metrics(), invalid_metrics)
