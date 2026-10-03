"""CPU/metadata checks for the explicit two-host mixed-objective handoff."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from paper.scripts import run_mixed_quantity_partition as wrapper
from paper.scripts import quantity_comparison_engine as engine
from paper.scripts import mixed_quantity_objective as primitives
from paper.scripts.quantity_objective_comparison import QuantityCase
from simple_lab_test.search.tests.test_mixed_quantity_engine import make_model, statistics, tensors, objectives

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def contract():
    parent = wrapper.read_json(ROOT / "paper/contracts/mixed_quantity_execution_5090_v1.json")
    runtime5080 = wrapper.read_json(ROOT / "search_artifacts/quantity_partition_preparation_v1/5080_readiness.json")["runtime"]
    started = 1000.
    deadline = started + parent["limits"]["max_wall_seconds"]
    qualification = {"passed": True, "qualifies_cuda": True,
                     "execution_contract_sha256": wrapper.sha_json(parent), "runtime": parent["runtime_expected"]}
    hosts = {}
    for role, alias, runtime, python, tmux in (
        ("continuation_5090", "5090", parent["runtime_expected"], "/opt/miniconda3/envs/ai_env/bin/python3.12", "/opt/miniconda3/envs/ai_env/bin/tmux"),
        ("instacart_5080", "5080", runtime5080, "/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python3.12", "/usr/bin/tmux"),
    ):
        base = "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/mixed_partition_test_" + alias
        hosts[role] = {"server_alias": alias, "python": python, "tmux_binary": tmux,
            "tmux_session": "mixed_partition_" + alias, "source_root": base + "/source",
            "output_dir": base + "/run", "wrapper_path": base + "/wrapper.py", "runtime_expected": runtime,
            "qualification": "reused_parent_receipt" if alias == "5090" else "new_cuda_required"}
    return {"schema": wrapper.SCHEMA, "status": "frozen_pending_explicit_approval",
        "parent": {"contract": parent, "contract_sha256": wrapper.sha_json(parent),
                   "source_files_sha256": parent["source"]["files_sha256"], "started_at_unix": started,
                   "deadline_unix": deadline, "execution_receipt": {"contract_sha256": wrapper.sha_json(parent),
                       "started_at_unix": started, "deadline_unix": deadline},
                   "qualification_receipt": qualification, "qualification_receipt_sha256": wrapper.sha_json(qualification)},
        "hosts": hosts, "arms": [{"dataset_id": dataset, "case": case,
             "host": "instacart_5080" if dataset == "insta_market_basket" else "continuation_5090",
             "mode_policy": "fresh" if dataset == "insta_market_basket" else "inherit_or_fresh"}
             for dataset in wrapper.DATASETS for case in wrapper.CASES],
        "wrapper": {"sha256": wrapper.sha_file(wrapper.__file__)}, "deadline_unix": deadline,
        "limits": deepcopy(wrapper.LIMITS), "policy": {"automatic_retry": False, "automatic_resume": False,
            "held_out": False, "budget_extension": False, "cpu_fallback": False}}


def approval(contract):
    return {"status": "approved_by_user", "partition_contract_sha256": wrapper.sha_json(contract),
            "scope": wrapper.APPROVAL_SCOPE, "user_instruction": "Synthetic explicit handoff test only."}


def handoff(contract):
    return {"schema": "mixed_quantity_handoff_v1", "partition_contract_sha256": wrapper.sha_json(contract),
            "approved_transfer": True, "observed_old_process_dead": True, "original_process_ids": [12345],
            "persisted_file_hashes": {}, "dataset_calibrations": {d: {"mode": "fresh"} for d in wrapper.DATASETS},
            "arms": [{"dataset_id": a["dataset_id"], "case": a["case"], "mode": "fresh"} for a in contract["arms"]]}


def test_actual_parent_nine_arm_split_and_old_absolute_deadline(contract):
    assert wrapper.validate_contract(contract, wrapper_path=wrapper.__file__, now=1001.)["arms"] == 9
    assert len(wrapper.assigned_arms(contract, "continuation_5090")) == 6
    assert {a["dataset_id"] for a in wrapper.assigned_arms(contract, "instacart_5080")} == {"insta_market_basket"}
    wrapper.validate_approval(contract, approval(contract))
    wrapper.validate_handoff(contract, handoff(contract))
    with pytest.raises(ValueError, match="deadline"):
        wrapper.validate_contract(contract, now=contract["deadline_unix"])


@pytest.mark.parametrize("change", ["assignment", "duplicate", "deadline", "old_output", "wrapper_inside_source", "runtime", "qualification", "limits"])
def test_partition_scope_and_original_evidence_cannot_drift(contract, change):
    if change == "assignment":
        contract["arms"][0]["host"] = "instacart_5080"
    elif change == "duplicate":
        contract["arms"][-1] = deepcopy(contract["arms"][0])
    elif change == "deadline":
        contract["deadline_unix"] += 60
    elif change == "old_output":
        contract["hosts"]["continuation_5090"]["output_dir"] = contract["parent"]["contract"]["execution"]["output_dir"]
    elif change == "wrapper_inside_source":
        host = contract["hosts"]["instacart_5080"]
        host["wrapper_path"] = host["source_root"] + "/wrapper.py"
    elif change == "runtime":
        contract["hosts"]["instacart_5080"]["runtime_expected"] = contract["hosts"]["continuation_5090"]["runtime_expected"]
    elif change == "qualification":
        contract["hosts"]["instacart_5080"]["qualification"] = "reused_parent_receipt"
    else:
        contract["limits"]["output_bytes_per_host"] *= 2
    with pytest.raises(ValueError):
        wrapper.validate_contract(contract)


def test_old_approval_or_live_process_handoff_rejected(contract):
    with pytest.raises(ValueError, match="approval SHA"):
        wrapper.validate_approval(contract, {**approval(contract), "partition_contract_sha256": "0" * 64})
    record = handoff(contract)
    record["observed_old_process_dead"] = False
    with pytest.raises(ValueError, match="ownership"):
        wrapper.validate_handoff(contract, record)
    record = handoff(contract)
    record["arms"][-1]["mode"] = "resume"
    with pytest.raises(ValueError, match="fresh"):
        wrapper.validate_handoff(contract, record)


def test_inherited_arm_cannot_replace_calibration_or_hide_extra_files(contract):
    record = handoff(contract)
    arm = record["arms"][0]
    base = Path(contract["parent"]["contract"]["execution"]["output_dir"]) / arm["dataset_id"]
    source = base / arm["case"]
    arm.update(mode="resume", source_dir=str(source))
    record["persisted_file_hashes"] = {str(source / name): "a" * 64 for name in ("contract.json", "last_epoch_state.pt")}
    with pytest.raises(ValueError, match="original calibration"):
        wrapper.validate_handoff(contract, record)
    calibration = str(base / "calibration.json")
    record["dataset_calibrations"][arm["dataset_id"]] = {"mode": "reuse", "source_path": calibration, "sha256": "b" * 64}
    record["persisted_file_hashes"][calibration] = "b" * 64
    wrapper.validate_handoff(contract, record)
    record["persisted_file_hashes"][str(base / "unassigned.json")] = "c" * 64
    with pytest.raises(ValueError, match="Unassigned"):
        wrapper.validate_handoff(contract, record)


def test_effective_identity_preserves_5090_and_changes_only_5080_execution_runtime(contract):
    parent = contract["parent"]["contract"]
    assert wrapper.effective_parent_contract(parent, contract, "continuation_5090") == parent
    effective = wrapper.effective_parent_contract(parent, contract, "instacart_5080")
    assert {k for k in parent if parent[k] != effective[k]} == {"runtime_expected", "execution"}
    assert wrapper.sha_json(effective) != wrapper.sha_json(parent)
    data = parent["datasets"][0]
    old = wrapper.training_identity(parent, contract, "continuation_5090", data, parent["runtime_expected"])
    assert old == {"execution_contract_sha256": wrapper.sha_json(parent), "runtime": parent["runtime_expected"],
                   "source": parent["source"], "data": data["inherited_data_identity"], "dataset_id": data["dataset_id"]}
    new = wrapper.training_identity(parent, contract, "instacart_5080", parent["datasets"][2], effective["runtime_expected"])
    assert new["execution_contract_sha256"] == wrapper.sha_json(effective)


def test_source_verification_uses_unchanged_seventy_file_snapshot(contract, tmp_path):
    parent = contract["parent"]["contract"]
    wrapper.verify_source(parent["source"]["local_snapshot"], parent)
    with pytest.raises(ValueError, match="70 frozen"):
        wrapper.verify_source(tmp_path, {"source": {"files": {}, "files_sha256": wrapper.sha_json({})}})


def synthetic_identity():
    return {"source": {"revision": "synthetic"}, "data": {"kind": "synthetic"},
            "runtime": {"device": "cpu"}, "execution_contract_sha256": "a" * 64}


def synthetic_run(path, *, stop=2, resume=False):
    dataset = TensorDataset(*tensors())
    train = DataLoader(dataset, batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(41))
    validation = DataLoader(dataset, batch_size=2, generator=torch.Generator().manual_seed(42))
    return engine.run_case(model=make_model(), train_loader=train, validation_loader=validation,
        case=QuantityCase.B_LOG_ORIGINAL, mixed_objective=objectives()[2], statistics=statistics(), output_dir=path,
        epochs=120, seed=42, identity=synthetic_identity(), stop_after_epochs=stop, resume=resume)


def files_under(path):
    return {str(p): wrapper.sha_file(p) for p in path.rglob("*") if p.is_file()}


def test_manual_resume_repairs_lagging_json_from_committed_checkpoint_and_preserves_original(tmp_path):
    source, target, full = (tmp_path / name for name in ("source", "target", "full"))
    synthetic_run(source)
    wrapper.write_json(source / "summary.json", {"stale": True})
    wrapper.write_json(source / "history.json", {"history": []})
    (source / "best_raw_quantity_rmse_model.pt").unlink()
    timing = wrapper.read_json(source / "timing.json")
    timing["epochs"] = timing["epochs"][:1] + [{"epoch": 3, "epoch_wall_seconds": 999.}]
    wrapper.write_json(source / "timing.json", timing)
    original_hashes = files_under(source)
    arm = {"mode": "resume", "source_dir": str(source)}
    repaired = wrapper.copy_and_repair_arm(arm, target, {"persisted_file_hashes": original_hashes},
        identity=synthetic_identity(), objective=objectives()[2], engine=engine)
    assert repaired["epochs_completed"] == 2
    assert files_under(source) == original_hashes
    repair = wrapper.read_json(target / "handoff_repair_receipt.json")
    assert repair["timing_epochs_missing"] == [2]
    resumed = synthetic_run(target, stop=3, resume=True)
    uninterrupted = synthetic_run(full, stop=3)
    assert resumed == uninterrupted
    a, b = [torch.load(p / "last_epoch_state.pt", weights_only=False) for p in (target, full)]
    assert all(a[key] == b[key] for key in ("model_state_sha256", "optimizer_state_sha256", "rng_state_sha256"))


@pytest.mark.parametrize("failure", ["source_changed", "forged_state_hash", "wrong_mode"])
def test_inherited_checkpoint_rejects_changed_files_corruption_and_mode(tmp_path, failure):
    source = tmp_path / "source"
    synthetic_run(source, stop=1)
    record = {"persisted_file_hashes": files_under(source)}
    arm = {"mode": "resume", "source_dir": str(source)}
    if failure == "source_changed":
        (source / "unexpected.txt").write_text("changed after handoff")
    elif failure == "forged_state_hash":
        path = source / "last_epoch_state.pt"
        payload = torch.load(path, weights_only=False)
        payload["model_state_dict"][next(iter(payload["model_state_dict"]))].add_(1.)
        torch.save(payload, path)
        record["persisted_file_hashes"] = files_under(source)
    else:
        arm["mode"] = "reuse"
    with pytest.raises(ValueError, match="hash|mode"):
        wrapper.copy_and_repair_arm(arm, tmp_path / "target", record,
            identity=synthetic_identity(), objective=objectives()[2], engine=engine)


def test_original_calibration_is_copied_exactly_without_recalibration(tmp_path, monkeypatch):
    identity, model = synthetic_identity(), make_model()
    receipt = primitives.calibrate_mixed_objective(model, TensorDataset(*tensors()), statistics=statistics(), identity=identity, synthetic=True)
    source = tmp_path / "original.json"
    wrapper.write_json(source, receipt)
    digest = wrapper.sha_file(source)
    def forbidden(*args, **kwargs):
        pytest.fail("Existing calibration must never be recomputed")
    monkeypatch.setattr(primitives, "calibrate_mixed_objective", forbidden)
    restored = wrapper.load_calibration({"source_path": str(source), "sha256": digest}, target=tmp_path / "new.json",
        identity=identity, statistics=statistics(), initial_state_sha256=receipt["initial_model_state_sha256"], primitives=primitives)
    assert restored == receipt and wrapper.sha_file(tmp_path / "new.json") == digest == wrapper.sha_file(source)
    with pytest.raises(ValueError, match="identity"):
        wrapper.load_calibration({"source_path": str(source), "sha256": digest}, target=tmp_path / "wrong.json",
            identity={**identity, "dataset_id": "other"}, statistics=statistics(),
            initial_state_sha256=receipt["initial_model_state_sha256"], primitives=primitives)


def test_budget_preserves_original_deadline_with_clock_rollback_and_per_file_cap(contract, tmp_path):
    ticks = {"wall": 1001., "mono": 10.}
    budget = wrapper.Budget(contract, tmp_path, clock=lambda: ticks["wall"], monotonic=lambda: ticks["mono"])
    with (tmp_path / "oversized.pt").open("wb") as handle:
        handle.truncate(contract["limits"]["per_file_bytes"] + 1)
    with pytest.raises(ValueError, match="Individual artifact"):
        budget("before_epoch_commit")
    (tmp_path / "oversized.pt").unlink()
    ticks.update(wall=-1000., mono=budget.deadline_mono)
    with pytest.raises(TimeoutError, match="Inherited suite deadline"):
        budget()


def test_calibration_watchdog_remains_bounded_during_a_hung_phase_and_resets_dataset():
    ticks = {"wall": 1000., "mono": 10.}
    deadline = wrapper.CalibrationDeadline(clock=lambda: ticks["wall"], monotonic=lambda: ticks["mono"])
    status = {"status": "calibrating", "dataset_id": "yellow_trip_hourly", "phase_started_at_unix": 990.}
    assert deadline.remaining(status) == 590.
    ticks.update(wall=0., mono=601.)
    assert deadline.remaining(status) < 0.
    status.update(dataset_id="insta_market_basket", phase_started_at_unix=0.)
    assert deadline.remaining(status) == 600.


def test_execute_rejects_live_old_process_before_loading_frozen_runtime(contract, tmp_path, monkeypatch):
    monkeypatch.setattr(wrapper, "_authority_checks", lambda _: None)
    monkeypatch.setattr(wrapper.os, "kill", lambda *args: None)
    monkeypatch.setattr(wrapper, "_loaded_frozen", lambda *_: pytest.fail("Runtime reached while old process exists"))
    with pytest.raises(ValueError, match="still exists"):
        wrapper.execute(contract, approval(contract), source_root=contract["hosts"]["continuation_5090"]["source_root"],
            output=tmp_path / "new", role="continuation_5090", handoff=handoff(contract))
    assert not (tmp_path / "new").exists()


def test_execute_requires_fresh_output_and_exact_tmux_before_gpu_query(contract, tmp_path, monkeypatch):
    monkeypatch.setattr(wrapper, "_authority_checks", lambda _: None)
    monkeypatch.setattr(wrapper, "_loaded_frozen", lambda *_: None)
    monkeypatch.setattr(wrapper, "_gpu_pids", lambda *_: pytest.fail("GPU query reached before launch guards"))
    output = tmp_path / "new"
    output.mkdir()
    kwargs = dict(source_root=contract["hosts"]["instacart_5080"]["source_root"], output=output,
                  role="instacart_5080", handoff=handoff(contract))
    with pytest.raises(ValueError, match="Fresh partition output"):
        wrapper.execute(contract, approval(contract), **kwargs)
    output.rmdir()
    monkeypatch.delenv("TMUX", raising=False)
    with pytest.raises(ValueError, match="tmux"):
        wrapper.execute(contract, approval(contract), **kwargs)
    monkeypatch.setenv("TMUX", "synthetic")
    monkeypatch.setattr(wrapper.subprocess, "check_output", lambda *args, **kwargs: "wrong-session\n")
    with pytest.raises(ValueError, match="tmux session"):
        wrapper.execute(contract, approval(contract), **kwargs)
    assert not output.exists()


def test_child_requires_owned_supervisor_ancestry_before_any_runtime(contract, tmp_path, monkeypatch):
    role = "instacart_5080"
    authority = {"contract": contract, "parent": contract["parent"]["contract"], "role": role,
                 "output": str(tmp_path), "supervisor_pid": 12345}
    path = tmp_path / "partition_authority.json"
    wrapper.write_json(path, authority)
    monkeypatch.setattr(wrapper, "_authority_checks", lambda a: (contract, contract["parent"]["contract"], role))
    monkeypatch.setattr(wrapper.os, "kill", lambda *args: None)
    monkeypatch.setattr(wrapper.os, "getppid", lambda: 54321)
    monkeypatch.setattr(wrapper, "_loaded_frozen", lambda *_: pytest.fail("Unowned child reached runtime"))
    with pytest.raises(ValueError, match="descend from supervisor"):
        wrapper.child_main("_worker", path)
    assert not (tmp_path / "worker_owner.json").exists()


def test_5090_qualification_reuses_exact_parent_receipt_without_any_probe(contract, tmp_path, monkeypatch):
    parent = contract["parent"]["contract"]
    calls = []
    frozen = SimpleNamespace(configured_runtime=lambda value, device: (calls.append((value, device)) or parent["runtime_expected"]))
    monkeypatch.setattr(wrapper.subprocess, "run", lambda *args, **kwargs: pytest.fail("5090 must not qualify again"))
    receipt = wrapper._qualification(contract, parent, "continuation_5090", frozen, parent, tmp_path / "authority.json")
    assert receipt == contract["parent"]["qualification_receipt"]
    assert calls == [(parent, "cuda:0")]


def test_actual_authority_rejects_unapproved_receipt_and_wrong_local_python_before_runtime(contract, tmp_path):
    role = "instacart_5080"
    host = contract["hosts"][role]
    authority = {"contract": contract, "parent": contract["parent"]["contract"], "role": role,
                 "approval": {}, "handoff": handoff(contract), "source_root": host["source_root"],
                 "output": host["output_dir"], "deadline_unix": contract["deadline_unix"]}
    with pytest.raises(ValueError, match="Explicit partition approval"):
        wrapper._authority_checks(authority, now=1001.)
    authority["approval"] = approval(contract)
    assert str(Path(sys.executable).resolve()) != host["python"]
    with pytest.raises(ValueError, match="Pinned host Python"):
        wrapper._authority_checks(authority, now=1001.)


def test_5080_qualification_uses_owned_wrapper_children_and_effective_identity(contract, tmp_path, monkeypatch):
    import time
    from simple_lab_test.search.common import runner
    role, parent = "instacart_5080", contract["parent"]["contract"]
    contract["hosts"][role]["output_dir"] = str(tmp_path)
    contract["deadline_unix"] = time.time() + 1000.
    effective = wrapper.effective_parent_contract(parent, contract, role)
    calls, parent_audits = [], []
    authority = tmp_path / "partition_authority.json"
    def child(command, **kwargs):
        calls.append(command)
        assert command[:2] == [sys.executable, str(Path(wrapper.__file__).resolve())]
        assert command[3:5] == ["--authority", str(authority)]
        assert 0 < kwargs["timeout"] <= 900.
        if command[2] == "_probe":
            case, stage = command[command.index("--case") + 1], command[command.index("--stage") + 1]
            folder = "full" if stage == "full" else "split"
            wrapper.write_json(tmp_path / "qualification" / case / folder / "summary.json", {"case": case, "complete": stage != "first"})
    monkeypatch.setattr(wrapper.subprocess, "run", child)
    monkeypatch.setattr(runner, "torch_load_checkpoint", lambda *args, **kwargs: {
        "optimizer_state_sha256": "same", "rng_state_sha256": "same", "model_state_sha256": "same"})
    def compare(first, second):
        assert first == second and first["complete"]
    frozen = SimpleNamespace(
        configured_runtime=lambda *_: effective["runtime_expected"],
        base=SimpleNamespace(compare_replay=compare),
        audit_pairs=lambda summaries: pytest.fail("Missing qualification arm") if len(summaries) != 3 else None,
        validate_contract=lambda value: parent_audits.append(value))
    receipt = wrapper._qualification(contract, parent, role, frozen, effective, authority)
    assert [(cmd[cmd.index("--case") + 1], cmd[cmd.index("--stage") + 1]) for cmd in calls[:-1]] == [
        (case, stage) for case in wrapper.CASES for stage in ("full", "first", "resume")]
    assert calls[-1][2] == "_shapes" and len(calls) == 10
    assert parent_audits == [parent]
    assert receipt["execution_contract_sha256"] == wrapper.sha_json(effective)
    assert receipt["runtime"] == effective["runtime_expected"] and receipt["qualifies_cuda"] is True
