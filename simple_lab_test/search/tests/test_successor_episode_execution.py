"""Contract enforcement and actual shared-trainer adapter checks, synthetic CPU only."""
from __future__ import annotations
import copy
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from paper.scripts import successor_episode_common as common
from paper.scripts import run_successor_episode_comparison as runner
from paper.scripts import run_observed_slot_partition as shared


@pytest.fixture(autouse=True)
def cpu_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def contract_fixture():
    old = json.loads((common.ROOT / common.RUNTIME_REFERENCE).read_text())
    value = {key: copy.deepcopy(old[key]) for key in ("datasets", "policy", "acceptance", "cost_gates", "source", "seed", "epochs")}
    host = copy.deepcopy(old["hosts"]["5090"])
    root = "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/successor_episode_seed42_5090_synthetic_test"
    host.update(root=root, source_root=root + "/source", output_dir=root + "/run", assigned_datasets=list(common.DATASETS))
    value.update(schema=common.SCHEMA, hosts={"5090": host}, arms=list(common.ARMS), limits=dict(common.LIMITS),
        approval_required=True, total_optimizer_steps=6816240, endpoint_replays=18,
        design_sha256=common.sha_file(common.ROOT / common.DESIGN), inherited_data_contract_sha256=common.sha_file(common.ROOT / common.INHERITED))
    return value


def test_three_arm_contract_budget_and_sources(tmp_path):
    contract = contract_fixture()
    common.write_json(tmp_path / "contract.json", contract)
    assert common.read_contract(tmp_path / "contract.json", check_source=False) == contract
    assert 3 * sum(data["expected_global_steps"] for data in contract["datasets"]) == 6816240
    for data in contract["datasets"]:
        for name, role in zip(common.ARMS, common.ROLES):
            args = runner.training_args(contract, data, tmp_path / "run", name)
            assert args.model_role == role
            assert args.epochs == args.min_epochs == 120
            assert args.device == "cuda:0" and args.batch_size == 128
            assert args.lambda_tail == 0. and args.time_intercept_limit == 300.
            assert args.checkpoint_monitor == "validation_raw_quantity_rmse"
    files = common.source_manifest()
    for name in (common.RUNTIME_REFERENCE, common.DESIGN, "models/TPPs/CountAwareTitanSuccessorMemory.py",
                 "paper/scripts/count_aware_tpp_backbone/training.py", "paper/scripts/run_observed_slot_partition.py"):
        assert name in files


@pytest.mark.parametrize("mutate", [
    lambda c: c["hosts"].update({"5080": c["hosts"]["5090"]}),
    lambda c: c["hosts"]["5090"]["assigned_datasets"].reverse(),
    lambda c: c["arms"].reverse(),
    lambda c: c["limits"].update(total_seconds=172801),
    lambda c: c["policy"].update(resume=True),
    lambda c: c.update(approval_required=False),
    lambda c: c.update(total_optimizer_steps=4544160),
    lambda c: c["datasets"][0]["loader"].update(batch_size=64),
    lambda c: c["hosts"]["5090"]["runtime_expected"].update(torch="changed"),
])
def test_contract_drift_fails_before_data(tmp_path, mutate):
    contract = contract_fixture()
    mutate(contract)
    common.write_json(tmp_path / "contract.json", contract)
    with pytest.raises(ValueError):
        common.read_contract(tmp_path / "contract.json", check_source=False)


def test_approval_and_native_qualification_bind_deadline_from_qualification_start(tmp_path, monkeypatch):
    contract = contract_fixture()
    contract["hosts"]["5090"]["root"] = str(tmp_path)
    runtime = {**contract["hosts"]["5090"]["runtime_expected"], "gpu": {"uuid": contract["hosts"]["5090"]["gpu_uuid"]}}
    receipt = {"status": "passed", "host": "5090", "contract_sha256": common.sha_json(contract),
        "source_files_sha256": contract["source"]["files_sha256"], "runtime": runtime, "started_at_unix": 1000.}
    common.write_json(tmp_path / "qualification/receipt.json", receipt)
    approval = {"schema": "successor_episode_approval_v1", "contract_sha256": common.sha_json(contract),
                "host": "5090", "approved": True, "user_instruction": "synthetic test approval only",
                "scope": "synthetic_cuda_qualification_and_fresh_nine_arm_validation"}
    monkeypatch.setattr(common.time, "time", lambda: 1100.)
    permit = common.create_permit(contract, approval)
    assert permit["deadline_unix"] == 173800.
    assert shared.fixed_start(permit, wall=lambda: 1100., monotonic=lambda: 20., total_seconds=172800) == 172720.
    for field, changed in (("host", "5080"), ("approved", False), ("contract_sha256", "0" * 64), ("user_instruction", "")):
        bad = copy.deepcopy(permit)
        bad["approval"][field] = changed
        with pytest.raises(ValueError):
            common.verify_permit(contract, bad, "5090")
    changed = copy.deepcopy(permit)
    changed["deadline_unix"] += 1
    with pytest.raises(ValueError):
        common.verify_permit(contract, changed, "5090")
    receipt["status"] = "cost_gate_failed"
    common.write_json(tmp_path / "qualification/receipt.json", receipt)
    with pytest.raises(ValueError):
        common.verify_permit(contract, permit, "5090")


def metric(rmse, mae=1.):
    return {"selected": {"qty_rmse": rmse, "qty_mae": mae, "time_nll": 2., "body": {"qty_mae": mae}, "tail": {"qty_mae": mae}},
            "last30": {"mean": 10., "sd": 1.}}


def test_gain_over_B_alone_cannot_pass_association_control():
    result = runner.compare_arms(dict(zip(common.ARMS, (metric(10.), metric(8.), metric(9.)))))
    assert result["comparisons"]["candidate_vs_B"]["accepted"]
    assert not result["candidate_engineering_gates_passed"]
    result = runner.compare_arms(dict(zip(common.ARMS, (metric(10.), metric(9.), metric(8.)))))
    assert result["candidate_engineering_gates_passed"]
    with pytest.raises(ValueError):
        runner.compare_arms({common.ARMS[0]: metric(10.)})


def test_real_adapter_initialization_exposure_and_both_endpoint_replays(tmp_path, monkeypatch):
    from paper.scripts.count_aware_tpp_backbone import training
    from paper.scripts import run_taxi_quantity_interface_ablation as replay_loader
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
    identity = runner.paired_initialization(data)
    assert torch.equal(before_rng, torch.get_rng_state())
    exposures = []
    bounds = data["quantity_boundaries_all_train_rows"]
    for name in common.ARMS:
        args = runner.training_args(contract, data, tmp_path / "training", name)
        args.device = "cpu"
        args.epochs = args.min_epochs = args.early_stopping_patience = 3
        with shared.audited_training(training, data, lambda: None, lambda *_: None) as records:
            summary, _, _ = training.train_one(args=args, frame=None,
                quantity_contract={"boundaries": bounds, "strata": [{"label": str(i)} for i in range(5)]},
                interface_meta={"train_target_mean": 1.5, "train_target_std": 1., "time_head": {"time_initial_intercept": 0.}},
                backbone=name, quantity_variant=shared.VARIANT, seed=42)
        assert summary["initial_state_sha256"] == identity["initial_state_sha256"][name]
        assert shared.audit_exposure(records, data, epochs=3) == 9
        exposures.append(records)
        directory = args.output_dir / "runs" / name / shared.VARIANT / "seed_42"
        history = shared.read_json(directory / "history.json")["history"]
        for path, row in ((Path(summary["checkpoint_path"]), history[summary["best_epoch"] - 1]),
                          (directory / "last_epoch_state.pt", history[-1])):
            replay = shared.replay_checkpoint(path, data, None, lambda: None, device="cpu", allowed_backbones=common.ARMS)
            assert replay["qty_rmse"] == pytest.approx(row["val_qty_rmse"], rel=1e-10, abs=1e-8)
            assert replay["qty_mae"] == pytest.approx(row["val_qty_mae"], rel=1e-10, abs=1e-8)
            assert replay["time_nll"] == pytest.approx(row["val_time_nll"], rel=1e-10, abs=1e-8)
            assert sum(cell["count"] for cell in replay["quantity_cells"]) == 7
    assert exposures[0] == exposures[1] == exposures[2]
