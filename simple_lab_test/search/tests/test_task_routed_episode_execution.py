"""Synthetic CPU checks for the task-routed three-arm/two-host execution adapter."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from paper.scripts import task_routed_episode_parallel_common as common
from paper.scripts import run_task_routed_episode_comparison as comparison
from paper.scripts import run_task_routed_episode_parallel as runner
from paper.scripts import run_observed_slot_partition as shared


@pytest.fixture(autouse=True)
def one_cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(before)


def contract_fixture():
    # Read configuration provenance only; referenced data paths are never opened.
    reference = json.loads((common.ROOT / common.RUNTIME_REFERENCE).read_text())
    datasets = copy.deepcopy(json.loads((common.ROOT / common.INHERITED).read_text())["datasets"])
    for data in datasets:
        data["epochs"] = 120
    root = "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/task_routed_episode_seed42_parallel_20260916_v1"
    hosts = {}
    for alias, assigned in common.ASSIGNMENTS.items():
        hosts[alias] = copy.deepcopy(reference["hosts"][alias])
        hosts[alias].update(root=root, source_root=root + "/source", output_dir=root + "/run",
            assigned_datasets=list(assigned), tmux=f"task_routed_episode_seed42_parallel_20260916_{alias}")
    return {"schema": common.SCHEMA, "datasets": datasets, "hosts": hosts,
        "arms": list(common.ARMS), "limits": dict(common.LIMITS), "policy": copy.deepcopy(common.POLICY),
        "acceptance": copy.deepcopy(common.ACCEPTANCE), "cost_gates": copy.deepcopy(common.COST_GATES),
        "seed": 42, "epochs": 120, "approval_required": True,
        "total_optimizer_steps": 6816240, "endpoint_replays": 18,
        "design_sha256": common.sha_file(common.ROOT / common.DESIGN),
        "inherited_data_contract_sha256": common.sha_file(common.ROOT / common.INHERITED),
        "cpu_receipt_sha256": common.sha_file(common.ROOT / common.CPU_RECEIPT),
        "implementation_receipt_sha256": common.sha_file(common.ROOT / common.IMPLEMENTATION_RECEIPT),
        "source": {"base_git_revision": "1" * 40, "files": {}, "files_sha256": common.sha_json({})}}


def test_contract_all_nine_arms_budgets_roles_and_selector(tmp_path):
    contract = contract_fixture()
    path = tmp_path / "contract.json"
    common.write_json(path, contract)
    for host in ("5080", "5090"):
        assert common.read_contract(path, host, check_source=False) == contract
    assert comparison.partition_totals(contract, "5080") == {"arms": 6, "steps": 1215720, "replays": 12}
    assert comparison.partition_totals(contract, "5090") == {"arms": 3, "steps": 5600520, "replays": 6}
    for data in contract["datasets"]:
        for name, role in zip(common.ARMS, common.ROLES, strict=True):
            args = comparison.training_args(contract, data, tmp_path / "unused", name)
            assert args.model_role == role
            assert args.execution_role == "fresh_task_routed_episode_parallel_validation"
            assert args.epochs == args.min_epochs == args.early_stopping_patience == 120
            assert args.device == "cuda:0" and args.batch_size == 128
            assert args.lambda_tail == 0. and args.time_intercept_limit == 300.
            assert args.checkpoint_monitor == "validation_raw_quantity_rmse"
    with pytest.raises(ValueError, match="arm"):
        comparison.training_args(contract, contract["datasets"][0], tmp_path, "titantpp_successor_episode_memory")
    with pytest.raises(ValueError, match="role"):
        comparison.training_args(contract, contract["datasets"][0], tmp_path, common.ARMS[0], execution_role="old_run")


@pytest.mark.parametrize("mutate", [
    lambda c: c["hosts"]["5090"]["assigned_datasets"].append("insta_market_basket"),
    lambda c: c["hosts"]["5080"]["assigned_datasets"].remove("yellow_trip_hourly"),
    lambda c: c["hosts"]["5080"]["assigned_datasets"].reverse(),
    lambda c: c["hosts"]["5090"]["assigned_datasets"].append("yellow_trip_hourly"),
    lambda c: c["hosts"].pop("5080"),
    lambda c: c["arms"].reverse(),
    lambda c: c["limits"].update(total_seconds=172801),
    lambda c: c["datasets"][0]["loader"].update(batch_size=64),
    lambda c: c["policy"].update(resume=True),
    lambda c: c.update(approval_required=False),
])
def test_contract_drift_rejected_before_data(tmp_path, mutate):
    contract = contract_fixture()
    mutate(contract)
    path = tmp_path / "changed.json"
    common.write_json(path, contract)
    with pytest.raises(ValueError):
        common.read_contract(path, "5090", check_source=False)


def metric(rmse, mae=1.):
    return {"selected": {"qty_rmse": rmse, "qty_mae": mae, "time_nll": 2.,
                         "body": {"qty_mae": mae}, "tail": {"qty_mae": mae}},
            "last30": {"mean": 10., "sd": 1.}}


def test_all_seven_gates_against_B_and_shared_mixture_control():
    result = comparison.compare_arms(dict(zip(common.ARMS, (metric(10.), metric(8.), metric(9.)), strict=True)))
    assert result["comparisons"]["candidate_vs_B"]["accepted"]
    assert not result["candidate_engineering_gates_passed"]
    result = comparison.compare_arms(dict(zip(common.ARMS, (metric(10.), metric(9.), metric(8.)), strict=True)))
    assert result["candidate_engineering_gates_passed"]
    assert all(len(row["checks"]) == 7 for row in result["comparisons"].values())
    with pytest.raises(ValueError):
        comparison.compare_arms({common.ARMS[0]: metric(10.)})
    # Candidate time guard still rejects a quantity-only gain.
    arms = dict(zip(common.ARMS, (metric(10.), metric(9.), metric(8.)), strict=True))
    arms[common.ARMS[2]]["selected"]["time_nll"] = 2.02
    assert not comparison.compare_arms(arms)["candidate_engineering_gates_passed"]


@pytest.mark.parametrize("host", ["5080", "5090"])
@pytest.mark.parametrize("command", ["execute", "_worker"])
def test_cli_passes_new_common_and_authority_to_shared_supervision(monkeypatch, tmp_path, host, command):
    calls = []
    argv = ["runner", command, "--contract", str(tmp_path / "contract.json"),
            "--permit", str(tmp_path / "permit.json"), "--host", host]
    if command == "_worker":
        argv += ["--owner-pid", "123", "--authority-fd", "456"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(shared, "supervisor" if command == "execute" else "worker",
                        lambda *args, **kwargs: calls.append((args, kwargs)))
    runner.main()
    args, kwargs = calls[0]
    assert args[2] == host and kwargs["common"] is common
    assert kwargs["assignment_validator"] is common.validate_assignment
    assert kwargs["entrypoint"] == runner.__file__
    if command == "_worker":
        assert args[3:] == (123, 456) and kwargs["production"] is runner.production


@pytest.mark.parametrize("command,extra", [("execute", ["--owner-pid", "123"]), ("_worker", [])])
def test_cli_refuses_unauthorized_worker_entry(monkeypatch, tmp_path, command, extra):
    monkeypatch.setattr(sys, "argv", ["runner", command, "--contract", str(tmp_path / "contract.json"),
        "--permit", str(tmp_path / "permit.json"), "--host", "5090", *extra])
    with pytest.raises(ValueError, match="authority"):
        runner.main()


def test_parallel_dispatch_and_every_helper_resolves_new_common(monkeypatch):
    captured = []
    monkeypatch.setattr(comparison, "production", lambda *args: captured.append(args) or "delegated")
    assert runner.production({}, {}, "5080", {}, lambda: None) == "delegated"
    assert captured[0][2] == "5080"
    assert runner.common is comparison.common is common
    for helper in (comparison.paired_initialization, comparison.training_args,
                   comparison.compare_arms, comparison.partition_totals):
        assert helper.__globals__["common"] is common


@pytest.mark.parametrize("drift", ["extra_key", "dtype", "shape", "tensor", "routing_code"])
def test_real_paired_initialization_rejects_extra_state_drift(monkeypatch, drift):
    data = copy.deepcopy(contract_fixture()["datasets"][0])
    original = shared._build_model
    def changed(data, name):
        model, meta = original(data, name)
        if name == common.ARMS[2]:
            state_dict = model.state_dict
            def state(*args, **kwargs):
                result = state_dict(*args, **kwargs)
                key = "task_routed_episode_memory.value_projection.weight"
                if drift == "extra_key":
                    result["task_routed_episode_memory.unexpected"] = torch.tensor(0.)
                elif drift == "dtype":
                    result[key] = result[key].double()
                elif drift == "shape":
                    result[key] = result[key].flatten()
                elif drift == "tensor":
                    result[key] = result[key] + .001
                else:
                    result["task_routed_episode_memory.routing_code"] = torch.tensor(0, dtype=torch.int64)
                return result
            model.state_dict = state
        return model, meta
    monkeypatch.setattr(shared, "_build_model", changed)
    with pytest.raises(ValueError, match="(state keys|initialization|identity)"):
        comparison.paired_initialization(data)


def test_real_paired_initialization_rejects_symmetric_task_queries(monkeypatch):
    """Matching parameters across arms do not justify identical queries within one arm."""
    data = copy.deepcopy(contract_fixture()["datasets"][0])
    original = shared._build_model
    def symmetric(data, name):
        model, metadata = original(data, name)
        if name in common.ARMS[1:]:
            memory = model.task_routed_episode_memory
            with torch.no_grad():
                memory.quantity_query_projection.weight.copy_(memory.time_query_projection.weight)
        return model, metadata
    monkeypatch.setattr(shared, "_build_model", symmetric)
    with pytest.raises(ValueError, match="independently initialized"):
        comparison.paired_initialization(data)


def test_real_adapter_initialization_exposure_and_both_endpoint_replays(tmp_path, monkeypatch):
    from paper.scripts.count_aware_tpp_backbone import training
    from paper.scripts import run_taxi_quantity_interface_ablation as replay_loader
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    contract = contract_fixture()
    data = copy.deepcopy(contract["datasets"][0])
    data["loader"].update(batch_size=4, max_seq_len=8)
    data["quantity_boundaries_all_train_rows"] = [2., 4., 8., 16.]
    data["history_boundaries"] = [2., 4.]
    data["statistics"].update(train_log_mean=1.5, train_log_std=1.)
    data["expected_global_steps"] = 9
    for split, count in (("train", 9), ("validation", 7)):
        data["inherited_data_identity"]["populations"][split]["target_count"] = count
    def make_loader(_frame, *, target_split, batch_size, shuffle, generator, **kwargs):
        count = 9 if target_split == "train" else 7
        i, j = torch.arange(count)[:, None], torch.arange(6)[None]
        dts = .1 + i.float() * .3 + j * .5
        quantities = 1. + i.float() * 4 + j
        mask = j < (i % 5 + 2)
        dataset = TensorDataset(torch.zeros_like(dts), dts * mask, mask, torch.zeros_like(dts), quantities * mask)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)
    monkeypatch.setattr(training, "make_loader", make_loader)
    monkeypatch.setattr(replay_loader, "make_loader", make_loader)
    before_rng = torch.get_rng_state().clone()
    identity = comparison.paired_initialization(data)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert identity["within_model_task_queries_distinct"] == {name: True for name in common.ARMS[1:]}
    exposures = []
    bounds = data["quantity_boundaries_all_train_rows"]
    quantity_contract = {"boundaries": bounds, "strata": [{"label": str(i)} for i in range(5)]}
    interface_meta = {"train_target_mean": 1.5, "train_target_std": 1.,
        "time_head": {"time_initial_intercept": 0.}, "data_scope": "synthetic_cpu_only",
        "source_files_sha256": contract["source"]["files_sha256"],
        "execution_contract_sha256": common.sha_json(contract)}
    for name in common.ARMS:
        args = comparison.training_args(contract, data, tmp_path / "training", name)
        args.device = "cpu"
        args.epochs = args.min_epochs = args.early_stopping_patience = 3
        with shared.audited_training(training, data, lambda: None, lambda *_: None) as records:
            summary, _, _ = training.train_one(args=args, frame=None,
                quantity_contract=quantity_contract, interface_meta=interface_meta,
                backbone=name, quantity_variant=shared.VARIANT, seed=42)
        assert summary["initial_state_sha256"] == identity["initial_state_sha256"][name]
        assert shared.audit_exposure(records, data, epochs=3) == 9
        exposures.append(records)
        directory = args.output_dir / "runs" / name / shared.VARIANT / "seed_42"
        history = shared.read_json(directory / "history.json")["history"]
        audit_identity = {"backbone": name, "initial_state_sha256": identity["initial_state_sha256"][name],
            "source_revision": args.source_revision,
            "resume_identity": training._resume_identity(args=args, backbone=name,
                quantity_variant=shared.VARIANT, seed=42, monitor=args.checkpoint_monitor,
                quantity_contract=quantity_contract, interface_meta=interface_meta)}
        comparison.audit_artifact_identity(summary, **audit_identity)
        for path, row in ((Path(summary["checkpoint_path"]), history[summary["best_epoch"] - 1]),
                          (directory / "last_epoch_state.pt", history[-1])):
            payload = torch_load_checkpoint(path, map_location="cpu")
            comparison.audit_artifact_identity(payload, **audit_identity)
            for field, value in (("source_revision", "2" * 40), ("seed", 43),
                                 ("source_revision_history", [args.source_revision, "2" * 40]),
                                 ("checkpoint_selection", "best_validation_joint_objective"),
                                 ("held_out_test_evaluated", True)):
                corrupt = dict(payload)
                corrupt[field] = value
                with pytest.raises(ValueError):
                    comparison.audit_artifact_identity(corrupt, **audit_identity)
            corrupt = dict(payload)
            corrupt["resume_identity"] = copy.deepcopy(payload["resume_identity"])
            corrupt["resume_identity"]["arguments"]["data_sha256"] = "0" * 64
            with pytest.raises(ValueError, match="identity"):
                comparison.audit_artifact_identity(corrupt, **audit_identity)
            if name == common.ARMS[2]:
                # Labels/configuration cannot convert a task-split state into the control.
                from models.TPPs.CountAwareTitanTaskRoutedMemory import task_routed_memory_metadata
                corrupt = dict(payload)
                corrupt["backbone"] = common.ARMS[1]
                corrupt["encoder_config"] = {**payload["encoder_config"],
                    **task_routed_memory_metadata(data["model"]["hidden_dim"], "shared")}
                with pytest.raises(ValueError):
                    comparison.audit_artifact_identity(corrupt, **{**audit_identity, "backbone": common.ARMS[1]})
            replay = shared.replay_checkpoint(path, data, None, lambda: None, device="cpu", allowed_backbones=common.ARMS)
            assert replay["qty_rmse"] == pytest.approx(row["val_qty_rmse"], rel=1e-10, abs=1e-8)
            assert replay["qty_mae"] == pytest.approx(row["val_qty_mae"], rel=1e-10, abs=1e-8)
            assert replay["time_nll"] == pytest.approx(row["val_time_nll"], rel=1e-10, abs=1e-8)
            assert sum(cell["count"] for cell in replay["quantity_cells"]) == 7
    assert exposures[0] == exposures[1] == exposures[2]


@pytest.mark.parametrize("host", ["5080", "5090"])
def test_mocked_production_visits_only_assigned_task_routed_arms_and_audits_both_endpoints(tmp_path, monkeypatch, host):
    from paper.scripts.count_aware_tpp_backbone import training
    from paper.scripts import quantity_comparison_data, run_quantity_comparison
    from simple_lab_test.search.common import runner as checkpoint_loader

    contract = contract_fixture()
    output = tmp_path / host / "run"
    output.mkdir(parents=True)
    contract["hosts"][host]["output_dir"] = str(output)
    runtime = {"synthetic": host}
    calls, expected_identities, saved_payloads = [], {}, {}
    original_initialization = comparison.paired_initialization
    initialized = {}
    def initialize(data):
        result = original_initialization(data)
        initialized[data["dataset_id"]] = result
        calls.append(("initialization", data["dataset_id"]))
        return result
    monkeypatch.setattr(comparison, "paired_initialization", initialize)
    monkeypatch.setattr(common, "runtime_check", lambda c, h: calls.append(("runtime", h)) or runtime)
    monkeypatch.setattr(common, "verify_source", lambda c: calls.append(("source", c["schema"])))
    def prepare(data):
        calls.append(("prepare", data["dataset_id"]))
        return object(), {"held_out_materialized": False,
            "populations": data["inherited_data_identity"]["populations"],
            "train_log_mean": data["statistics"]["train_log_mean"],
            "train_log_std": data["statistics"]["train_log_std"]}
    monkeypatch.setattr(quantity_comparison_data, "prepare_quantity_comparison_data", prepare)
    monkeypatch.setattr(run_quantity_comparison, "bind_frozen_statistics", lambda d, m: m)
    live_exposure = {}
    @contextmanager
    def audited(_training, data, budget, status):
        records = {split: [{"epoch": i, "split": split,
            "count": data["inherited_data_identity"]["populations"][split]["target_count"],
            "batches": math.ceil(data["inherited_data_identity"]["populations"][split]["target_count"] / 128),
            "batch_order_sha256": "synthetic_same_order"} for i in range(1, count + 1)]
            for split, count in (("train", 120), ("validation", 121))}
        live_exposure.update(records=records, status=status)
        yield records
    monkeypatch.setattr(shared, "audited_training", audited)
    def train_one(*, args, frame, quantity_contract, interface_meta, backbone, quantity_variant, seed):
        dataset = args.dataset_contract
        calls.append(("train", dataset, backbone, args.model_role, args.execution_role))
        expected_identities[(dataset, backbone)] = training._resume_identity(args=args,
            backbone=backbone, quantity_variant=quantity_variant, seed=seed,
            monitor=args.checkpoint_monitor, quantity_contract=quantity_contract, interface_meta=interface_meta)
        assert args.epochs == args.min_epochs == 120 and args.batch_size == 128
        assert seed == 42 and quantity_variant == shared.VARIANT
        assert interface_meta["execution_contract_sha256"] == common.sha_json(contract)
        assert interface_meta["source_files_sha256"] == contract["source"]["files_sha256"]
        directory = args.output_dir / "runs" / backbone / shared.VARIANT / "seed_42"
        directory.mkdir(parents=True, exist_ok=True)
        base = 10. - common.ARMS.index(backbone)
        history = [{"epoch": epoch, "val_qty_rmse": base + (120 - epoch) / 100.,
                    "val_qty_mae": 1., "val_time_nll": 2.} for epoch in range(1, 121)]
        common.write_json(directory / "history.json", {"history": history})
        selected = directory / "best_val_qty_rmse_model.pt"
        last = directory / "last_epoch_state.pt"
        digest = "1" * 64
        summary = {"status": "success", "backbone": backbone, "completed_epochs": 120,
            "stopped_early": False, "best_epoch": 120, "checkpoint_path": str(selected),
            "checkpoint_state_sha256": digest,
            "initial_state_sha256": initialized[dataset]["initial_state_sha256"][backbone]}
        for path, values in ((selected, {"model_state_sha256": digest}),
                             (last, {"epoch": 120, "history": history})):
            saved_payloads[str(path)] = {**summary, **values}
        live_exposure["status"](120, live_exposure["records"])
        return summary, [], []
    monkeypatch.setattr(training, "train_one", train_one)
    def audit_identity(payload, **identity):
        expected = expected_identities[(identity["resume_identity"]["arguments"]["dataset_contract"], identity["backbone"])]
        assert identity["resume_identity"] == expected
        assert identity["source_revision"] == contract["source"]["base_git_revision"]
        assert payload["initial_state_sha256"] == identity["initial_state_sha256"]
        calls.append(("artifact_identity", identity["backbone"]))
    monkeypatch.setattr(comparison, "audit_artifact_identity", audit_identity)
    monkeypatch.setattr(checkpoint_loader, "torch_load_checkpoint", lambda p, **_: saved_payloads[str(p)])
    def replay(path, data, frame, budget, *, allowed_backbones):
        budget()
        assert allowed_backbones == common.ARMS
        name = Path(path).parents[2].name
        calls.append(("replay", data["dataset_id"], name, Path(path).name))
        return metric(10. - common.ARMS.index(name))["selected"]
    monkeypatch.setattr(shared, "replay_checkpoint", replay)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(("synthetic_cleanup", host)))
    result = runner.production(contract, {"deadline_unix": 9999999999.}, host,
        {"runtime": runtime}, lambda: calls.append(("budget", host)))
    totals = comparison.partition_totals(contract, host)
    assert result["status"] == "complete" and result["completed_arms"] == totals["arms"]
    assert result["total_optimizer_steps"] == totals["steps"]
    assert result["endpoint_replays"] == totals["replays"]
    assert list(result["datasets"]) == contract["hosts"][host]["assigned_datasets"]
    assert [row[1:3] for row in calls if row[0] == "train"] == [
        (dataset, arm) for dataset in common.ASSIGNMENTS[host] for arm in common.ARMS]
    assert len([row for row in calls if row[0] == "artifact_identity"]) == totals["arms"] * 3
    assert len([row for row in calls if row[0] == "replay"]) == totals["replays"]
    for row in calls:
        if row[0] == "train":
            assert row[3] == common.ROLES[common.ARMS.index(row[2])]
            assert row[4] == common.EXECUTION_ROLE
    assert result["evaluation_scope"] == "validation_only" and not result["held_out_test_evaluated"]
    for dataset, pair in result["datasets"].items():
        assert pair["exposure_equal"] is True and set(pair["arms"]) == set(common.ARMS)
        assert shared.read_json(output / dataset / "paired_comparison.json") == pair


def test_production_rejects_unqualified_runtime_before_materializing_data(monkeypatch, tmp_path):
    from paper.scripts import quantity_comparison_data
    contract = contract_fixture()
    monkeypatch.setattr(common, "runtime_check", lambda *_: {"torch": "changed"})
    monkeypatch.setattr(quantity_comparison_data, "prepare_quantity_comparison_data",
                        lambda *_: pytest.fail("Runtime drift reached data preparation"))
    with pytest.raises(ValueError, match="Runtime"):
        runner.production(contract, {}, "5090", {"runtime": {"torch": "qualified"}}, lambda: None)
