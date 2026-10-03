"""Stdlib fake-process checks; no research model/data, GPU, or remote commands."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from paper.scripts import run_raw_aux_gradient_partition as partition


ROOT = Path(__file__).resolve().parents[3]


def contract(tmp_path):
    hosts = {}
    for host, arms in partition.ASSIGNMENT.items():
        root = tmp_path / host
        root.mkdir(exist_ok=True)
        hosts[host] = {"root": str(root), "source_root": str(root / "source"),
            "output_dir": str(root / "run"), "assigned_arms": arms[:],
            "python": sys.executable, "gpu_uuid": "GPU-" + host,
            "tmux_binary": "/fake/tmux", "tmux": "synthetic_" + host}
    return {"schema": "raw_aux_gradient_parallel_execution_v1", "hosts": hosts,
        "policy": {"resume": False, "retry": False, "held_out": False},
        "source": {"files": {}, "files_sha256": partition.sha_json({})},
        "dataset": {"dataset_id": "intermittent_frozen_5000", "expected_global_steps": 369240,
            "loader": {"batch_size": 128, "max_seq_len": 256, "num_workers": 0, "drop_last": False}},
        "limits": {"training_seconds": 43200, "total_seconds": 46800,
            "preflight_seconds": 1800, "checkpoint_seconds": 1800,
            "per_host_output_bytes": 2147483648, "max_file_bytes": 67108864, "min_free_bytes": 1}}


def child(code, *, authenticate=True):
    program = "import os, sys, time, signal\n"
    program += f"sys.path.insert(0, {str(ROOT)!r})\n"
    program += "from paper.scripts.run_raw_aux_gradient_partition import WorkerAuthority\nassert 'torch' not in sys.modules\n"
    if authenticate:
        program += "authority = WorkerAuthority.from_environment(started=float(os.environ['INTERMITTENT_CELL_PARENT_START']))\n"
    return [sys.executable, "-c", program + code]


def tiny_limits():
    return {"max_wall_seconds": 2., "stage_seconds": {
        "preflight_and_finalize": 1.5, "training": .3, "checkpoint_replay": .3}}


def test_module_and_direct_worker_denial_do_not_import_torch():
    code = "from paper.scripts.run_raw_aux_gradient_partition import worker\n"
    code += "try:\n worker('missing','missing','5090',time.monotonic())\nexcept ValueError as e:\n assert 'inherited supervisor' in str(e)\nelse:\n raise AssertionError('direct worker accepted')\nassert 'torch' not in sys.modules\n"
    result = subprocess.run(child(code, authenticate=False), capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("change", ["assignment", "resume", "retry", "held_out", "batch", "steps"])
def test_fixed_assignment_and_training_policy_reject_mutation(tmp_path, change):
    value = contract(tmp_path)
    assert partition.validate_assignment(value, "5090")["assigned_arms"] == partition.ASSIGNMENT["5090"]
    if change == "assignment":
        value["hosts"]["5090"]["assigned_arms"].reverse()
    elif change in value["policy"]:
        value["policy"][change] = True
    elif change == "batch":
        value["dataset"]["loader"]["batch_size"] = 64
    else:
        value["dataset"]["expected_global_steps"] = 120
    with pytest.raises(ValueError):
        partition.validate_assignment(value, "5090")


def test_stage_mapping_and_common_deadline_cannot_restart(tmp_path):
    limits = partition.supervision_limits(contract(tmp_path))
    assert limits["stage_seconds"] == {"preflight_and_finalize": 1800, "training": 43200, "checkpoint_replay": 1800}
    permit = {"started_at_unix": 100., "deadline_unix": 46900.}
    assert partition.fixed_start(permit, wall=lambda: 200., monotonic=lambda: 1000.) == 900.
    for now in (99., 46900.):
        with pytest.raises(ValueError, match="deadline"):
            partition.fixed_start(permit, wall=lambda: now, monotonic=lambda: 1000.)
    with pytest.raises(ValueError, match="deadline"):
        partition.fixed_start({**permit, "deadline_unix": 46901.}, wall=lambda: 200., monotonic=lambda: 1000.)


def test_inherited_context_and_normal_fake_child_completion():
    context = "a" * 64
    program = "authority.check_owner(started=authority.started,limits=" + repr(tiny_limits()) + ",context_sha256=" + repr(context) + ")\n"
    program += "authority.set_stage('training')\nauthority.set_stage('checkpoint_replay')\nauthority.set_stage('preflight_and_finalize')\nauthority.finish()\nauthority.close()\n"
    result = partition.supervise_child(child(program), tiny_limits(), context_sha256=context,
                                      poll_interval=.01, kill_grace=.02)
    assert result["authenticated"] and result["exit_code"] == 0 and result["execution_context_sha256"] == context
    assert result["stage_transitions"] == 3


@pytest.mark.parametrize("stage", ["training", "checkpoint_replay"])
def test_parent_times_out_native_hang_in_own_group_only(stage):
    bystander = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(10)"], start_new_session=True)
    try:
        code = f"authority.set_stage({stage!r})\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\ntime.sleep(10)\n"
        with pytest.raises(TimeoutError, match=stage + " cumulative") as caught:
            partition.supervise_child(child(code), tiny_limits(), poll_interval=.01, kill_grace=.02)
        with pytest.raises(ProcessLookupError):
            os.getpgid(caught.value.supervision_pid)
        assert bystander.poll() is None
    finally:
        bystander.terminate()
        bystander.wait(timeout=3)


def test_parent_cumulative_training_stage_and_total_deadline():
    code = "authority.set_stage('training')\ntime.sleep(.18)\nauthority.set_stage('checkpoint_replay')\ntime.sleep(.03)\nauthority.set_stage('training')\ntime.sleep(10)\n"
    with pytest.raises(TimeoutError, match="training cumulative"):
        partition.supervise_child(child(code), tiny_limits(), poll_interval=.01, kill_grace=.02)
    bounds = tiny_limits()
    bounds["max_wall_seconds"] = .25
    bounds["stage_seconds"]["training"] = 1.5
    with pytest.raises(TimeoutError, match="Total diagnostic"):
        partition.supervise_child(child("authority.set_stage('training')\ntime.sleep(10)\n"), bounds,
                                  poll_interval=.01, kill_grace=.02)


def prepare_execute(tmp_path, monkeypatch):
    value = contract(tmp_path)
    now = time.time()
    permit = {"started_at_unix": now, "deadline_unix": now + 46800,
              "qualifications": {"5080": {}, "5090": {}}}
    contract_path, permit_path = tmp_path / "contract.json", tmp_path / "permit.json"
    contract_path.write_text(json.dumps(value)); permit_path.write_text(json.dumps(permit))
    validations = []
    def check_permit(c, p, h):
        validations.append((c, p, h))
        if set(p["qualifications"]) != {"5080", "5090"}:
            raise ValueError("Both fresh qualifications required")
        return {"passed": True}
    common = SimpleNamespace(read_contract=lambda path: value, validate_permit=check_permit,
                             verify_native_proofs=lambda *a: None, apply_environment=lambda h: None)
    monkeypatch.setitem(sys.modules, "paper.scripts.raw_aux_parallel_common", common)
    import paper.scripts
    monkeypatch.setattr(paper.scripts, "raw_aux_parallel_common", common, raising=False)
    monkeypatch.setattr(partition, "verify_host_paths", lambda h: None)
    monkeypatch.setattr(partition, "gpu_pids", lambda h: set())
    monkeypatch.setattr(partition.subprocess, "check_output", lambda *a, **k: "synthetic_5090\n")
    monkeypatch.setenv("TMUX", "synthetic")
    return value, permit, contract_path, permit_path, validations


def test_start_requires_both_receipts_before_output_or_child(tmp_path, monkeypatch):
    value, permit, contract_path, permit_path, _ = prepare_execute(tmp_path, monkeypatch)
    permit["qualifications"].pop("5080")
    permit_path.write_text(json.dumps(permit))
    monkeypatch.setattr(partition, "supervise_child", lambda *a, **k: pytest.fail("launched without both qualifications"))
    with pytest.raises(ValueError, match="Both fresh"):
        partition.execute(contract_path, permit_path, "5090")
    assert not Path(value["hosts"]["5090"]["output_dir"]).exists()


@pytest.mark.parametrize("fail_after_worker_complete", [False, True])
def test_only_parent_writes_terminal_and_preserves_failed_supervision(tmp_path, monkeypatch, fail_after_worker_complete):
    value, permit, contract_path, permit_path, validations = prepare_execute(tmp_path, monkeypatch)
    output = Path(value["hosts"]["5090"]["output_dir"])
    def fake_supervise(command, bounds, **kwargs):
        result = {"status": "host_complete", "context_sha256": kwargs["context_sha256"],
                  "completed_arms": partition.ASSIGNMENT["5090"], "total_host_optimizer_steps": 738480}
        partition.write_exclusive(output / "worker_terminal_status.json", result)
        if fail_after_worker_complete:
            raise TimeoutError("Finalization exceeded deadline")
        return {"authenticated": True, "exit_code": 0, "budget": {"elapsed_seconds": 1.}}
    monkeypatch.setattr(partition, "supervise_child", fake_supervise)
    if fail_after_worker_complete:
        with pytest.raises(TimeoutError):
            partition.execute(contract_path, permit_path, "5090")
    else:
        result = partition.execute(contract_path, permit_path, "5090")
        assert result["total_host_optimizer_steps"] == 738480
    terminal = json.loads((output / "terminal_status.json").read_text())
    assert terminal["status"] == ("timeout" if fail_after_worker_complete else "host_complete")
    assert terminal["parent_authoritative"] and validations
    before = (output / "terminal_status.json").read_bytes()
    with pytest.raises(ValueError, match="Fresh output"):
        partition.execute(contract_path, permit_path, "5090")
    assert (output / "terminal_status.json").read_bytes() == before


def test_busy_gpu_refused_without_kill_or_claim(tmp_path, monkeypatch):
    value, _, contract_path, permit_path, _ = prepare_execute(tmp_path, monkeypatch)
    monkeypatch.setattr(partition, "gpu_pids", lambda h: {123456})
    monkeypatch.setattr(partition, "supervise_child", lambda *a, **k: pytest.fail("busy GPU launch"))
    with pytest.raises(ValueError, match="GPU is busy"):
        partition.execute(contract_path, permit_path, "5090")
    assert not Path(value["hosts"]["5090"]["output_dir"]).exists()


def test_exclusive_worker_claim_and_file_budget(tmp_path, monkeypatch):
    path = tmp_path / "worker_owner.json"
    partition.write_exclusive(path, {"pid": 1})
    with pytest.raises(FileExistsError):
        partition.write_exclusive(path, {"pid": 2})
    value = contract(tmp_path)
    output = tmp_path / "artifacts"; output.mkdir()
    budget = partition.StorageBudget(SimpleNamespace(check=lambda: None), value, output)
    assert budget.check_storage() == 0
    value["limits"]["max_file_bytes"] = 4
    (output / "large").write_bytes(b"12345")
    with pytest.raises(ValueError, match="per-file"):
        budget.check_storage()


def test_missing_native_proof_rejected_before_output(tmp_path, monkeypatch):
    value, _, contract_path, permit_path, _ = prepare_execute(tmp_path, monkeypatch)
    common = sys.modules["paper.scripts.raw_aux_parallel_common"]
    def missing(*args):
        raise ValueError("Native proof digest differs")
    common.verify_native_proofs = missing
    monkeypatch.setattr(partition, "supervise_child", lambda *a, **k: pytest.fail("launched without native proofs"))
    with pytest.raises(ValueError, match="Native proof"):
        partition.execute(contract_path, permit_path, "5090")
    assert not Path(value["hosts"]["5090"]["output_dir"]).exists()


@pytest.mark.parametrize("host_name", ["5090", "5080"])
def test_production_adapter_fixed_calls_with_fake_modules_only(tmp_path, monkeypatch, host_name):
    value = contract(tmp_path)
    data = value["dataset"]
    populations = {"train": {"target_count": 393824}, "validation": {"target_count": 86285}}
    data["inherited_data_identity"] = {"populations": populations}
    output = Path(value["hosts"][host_name]["output_dir"]); output.mkdir()
    initial, runtime = "f" * 64, {"device": "cuda:0", "synthetic_fake_module": True}
    metadata = {"held_out_materialized": False, "populations": populations, "train_log_mean": 1., "raw_scale": 5.}
    model = SimpleNamespace(state_dict=lambda: {"synthetic_not_a_tensor": 1})
    common = SimpleNamespace(runtime_check=lambda *a: runtime, source_manifest=lambda: value["source"],
        prepare_inputs=lambda c: (object(), metadata, data),
        build_inputs=lambda *a: (model, object(), object(), {}),
        make_identity=lambda *a: {"runtime": runtime, "host_alias": host_name})
    import paper.scripts
    monkeypatch.setitem(sys.modules, "paper.scripts.raw_aux_parallel_common", common)
    monkeypatch.setattr(paper.scripts, "raw_aux_parallel_common", common, raising=False)
    calls = []
    def fake_run_case(**kwargs):
        calls.append(kwargs)
        arm = Path(kwargs["output_dir"]).name
        condition = {"mixed_objective": {"name": kwargs["mixed_objective"].name}}
        if kwargs["raw_aux_control"] is not None:
            condition["raw_aux_control"] = {"synthetic": True}
        folder = Path(kwargs["output_dir"]); folder.mkdir()
        partition.write_exclusive(folder / "contract.json", {"identity": kwargs["identity"], "condition": condition})
        return {"status": "complete", "epochs_completed": 120, "epochs_budget": 120,
            "global_step": 369240, "initial_state_sha256": initial, "condition": condition,
            "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
            "history": [{"epoch": epoch, "global_step": epoch * 3077, "train_batches": 3077,
                         "validation_batches": 675, "train_count": 393824, "validation_count": 86285,
                         "train_batch_order_sha256": "same_order"} for epoch in range(1, 121)]}
    fake_modules = {
        "paper.scripts.quantity_comparison_engine": SimpleNamespace(run_case=fake_run_case),
        "paper.scripts.quantity_objective_comparison": SimpleNamespace(
            QuantityCase=SimpleNamespace(B_LOG_ORIGINAL="B_log_original"),
            QuantityStatistics=lambda mu, scale: SimpleNamespace(mu=mu, raw_scale=scale)),
        "paper.scripts.raw_aux_gradient_acceptance": SimpleNamespace(evaluate_checkpoints=lambda **kw: {"arm_id": kw["arm_id"], "synthetic": kw["synthetic"]}),
        "paper.scripts.run_raw_aux_gradient_comparison": SimpleNamespace(
            load_design=lambda: {"initialization_and_exposure": {"initial_state_sha256": initial}},
            objective_for_arm=lambda arm, design: (SimpleNamespace(name="B_log_original" if arm == "B_log_original" else "mixed_original"),
                                                   object() if arm == "mixed_raw_capped_original" else None)),
        "paper.scripts.run_time_quantity_diagnostic": SimpleNamespace(validate_import_origins=lambda: None),
        "simple_lab_test.search.common.runner": SimpleNamespace(canonical_state_dict_sha256=lambda state: initial),
    }
    for name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(partition, "gpu_pids", lambda host: set())
    stages = []
    budget = SimpleNamespace(started=0., limits=partition.supervision_limits(value),
        set_stage=stages.append, check=lambda: None, snapshot=lambda: {"synthetic": True})
    authority = SimpleNamespace(check_owner=lambda **kwargs: None)
    result = partition._production_arms(value, {"deadline_unix": 46800}, host_name,
        {"runtime": runtime, "initial_state_sha256": initial}, authority, budget)
    assert [Path(call["output_dir"]).name for call in calls] == partition.ASSIGNMENT[host_name]
    for call in calls:
        assert call["epochs"] == 120 and call["seed"] == 42 and call["device"] == "cuda:0"
        assert call["lr"] == .001 and call["weight_decay"] == .01 and call["grad_clip"] == 1.
        assert call["resume"] is False and call["stop_after_epochs"] is None
    assert result["total_host_optimizer_steps"] == 369240 * len(calls)
    assert result["validation_replays"] == 2 * len(calls)
    for arm in partition.ASSIGNMENT[host_name]:
        evaluation = read_json_for_test(output / arm / "validation_diagnosis.json")
        assert evaluation["training_contract"]["identity"]["host_alias"] == host_name
        assert evaluation["synthetic"] is False  # Production arguments, mocked evaluator.


def read_json_for_test(path):
    return json.loads(path.read_text())
