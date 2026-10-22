"""Independent local continuation QA: real pack format, injected OS boundaries.

Fixture construction is reused, but production guard/pack/gate/archive verification
executes. No network, GPU, inference, or real experiment artifacts are accessed.
"""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from paper.scripts import advance_shared_component_ablation as a
from paper.tests.test_advance_shared_component_ablation import controls, dump, host_evidence, identity


@pytest.fixture(autouse=True)
def no_external_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("An independent QA test attempted an unmocked external command")
    monkeypatch.setattr(a.subprocess, "run", forbidden)
    monkeypatch.setattr(a.subprocess, "Popen", forbidden)


def native_setup(controls, tmp_path, monkeypatch):
    c = deepcopy(controls["c"])
    for host in a.HOSTS:
        root = tmp_path / ("native_" + host)
        c["hosts"][host].update(root=str(root), source_root=str(root / "source"),
            boot_id="qa-boot", absolute_host_deadline_unix=10**12,
            paused_processes=[{**identity(900), "state": "T"}],
            environment={"PYTHONDONTWRITEBYTECODE": "1"}, python="/pinned/python", tmux="qa")
    roots = {h: Path(c["hosts"][h]["root"]) for h in a.HOSTS}
    launches = {}
    for host, root in roots.items():
        host_evidence(root, c, host)
        for name in c["source"]["files"]:
            target = root / "source" / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(controls["root"] / name, target)
        for name in c["input_files"]:
            target = root / "source" / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(controls["bundle"] / "input_snapshot" / name, target)
        dump(root / "execution_contract.json", c)
        launch = {"phase": "fit_validation", "identity": identity(700), "supervisor_identity": identity(701)}
        launches[host] = launch
        dump(root / "fit_validation.claim.json", {"identity": launch["identity"]})
        dump(root / "fit_validation.supervisor.intent.json", {"identity": launch["supervisor_identity"]})
        for stream in ("stdout", "stderr"):
            (root / ("fit_validation." + stream + ".log")).write_bytes(b"")
        end = a.read(root / "fit_validation.supervisor_exit.json")
        end.update(command=launch["identity"]["argv"], parent_own_os_exit_directly_observed=False,
            stdout_sha256=a.sha(root / "fit_validation.stdout.log"), stderr_sha256=a.sha(root / "fit_validation.stderr.log"))
        dump(root / "fit_validation.supervisor_exit.json", end)
    state = {"boot": "qa-boot", "gpu": [900], "identities": {900: {**identity(900), "state": "T"}}, "commands": []}
    read_text = Path.read_text
    def read_os(path, *args, **kwargs):
        if str(path) == "/proc/sys/kernel/random/boot_id": return state["boot"]
        return read_text(path, *args, **kwargs)
    def gpu(command, **kwargs):
        assert command == ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]
        assert kwargs["timeout"] == 15
        state["commands"].append(command)
        return subprocess.CompletedProcess(command, 0, "\n".join(map(str, state["gpu"])), "")
    monkeypatch.setattr(Path, "read_text", read_os)
    monkeypatch.setattr(a.manager, "proc_identity", lambda pid: state["identities"].get(pid))
    monkeypatch.setattr(a.subprocess, "run", gpu)
    return c, roots, launches, state


def execute_probe(c, host, launch):
    out = io.StringIO()
    with redirect_stdout(out):
        exec(compile(a.terminal_program(c, host, "fit_validation", launch), "<independent native probe>", "exec"), {})
    return a.json_response(out.getvalue().encode())


def wrap_actual_pack(c, host, phase, root, destination):
    """Invoke unchanged manager pack and copy its actual format, not a replica."""
    receipt = a.manager.pack(c, host, phase)
    destination.mkdir(parents=True)
    shutil.copyfile(receipt["archive"], destination / ("collection_" + phase + ".tar.gz"))
    dump(destination / "collection_receipt.json", receipt)
    (destination / "pack.stdout").write_bytes((json.dumps(receipt) + "\n").encode())
    (destination / "pack.stderr").write_bytes(b"")
    dump(destination / "transport.json", {"actual_os_returncode": 0, "timeout": False, "transport_error": None,
        "stdout_sha256": a.sha(destination / "pack.stdout"), "stderr_sha256": a.sha(destination / "pack.stderr")})
    for stream in ("stdout", "stderr"): (destination / ("transfer." + stream)).write_bytes(b"")
    dump(destination / "transfer.json", {"actual_os_returncode": 0, "timeout": False, "transport_error": None,
        "stdout_sha256": a.sha(destination / "transfer.stdout"), "stderr_sha256": a.sha(destination / "transfer.stderr")})
    result = a.validate_collection(c, host, phase, destination, extract=True)
    dump(destination / "VERIFIED.json", {"status": "verified"})
    return result


@pytest.mark.parametrize("host", a.HOSTS)
def test_real_manager_pack_format_is_accepted_and_preserves_legacy(controls, tmp_path, monkeypatch, host):
    c, roots, launches, state = native_setup(controls, tmp_path, monkeypatch)
    value = execute_probe(c, host, launches[host])
    assert value["status"] == "terminal"
    assert value["dispatcher_exit"]["parent_own_os_exit_directly_observed"] is False
    folder = controls["bundle"] / "retrieval/fit_validation" / host
    originals = wrap_actual_pack(c, host, "fit_validation", roots[host], folder)
    assert a.validate_collection(c, host, "fit_validation", folder) == originals
    assert a.sha(originals / "source/legacy/original.pt") == c["legacy_files"]["legacy/original.pt"]
    assert not (originals / "source/dummy.py").exists()
    assert state["identities"][900]["state"] == "T"


def test_parent_observed_dispatcher_stream_hash_must_match_collected_stream(controls, tmp_path, monkeypatch):
    c, roots, _, _ = native_setup(controls, tmp_path, monkeypatch)
    # Parent recorded SHA(empty); pack's fresh inventory alone cannot replace that
    # earlier exit observation after its referenced output bytes change.
    (roots["5080"] / "fit_validation.stdout.log").write_bytes(b"changed after exit observation")
    with pytest.raises(RuntimeError):
        wrap_actual_pack(c, "5080", "fit_validation", roots["5080"], controls["bundle"] / "retrieval/fit_validation/5080")


@pytest.mark.parametrize("fault", ["boot", "paused_state", "paused_ticks", "foreign_gpu", "source", "input", "legacy", "dispatcher_alive", "child_alive", "dispatcher_exit", "child_exit", "dispatcher_stdout", "dispatcher_stderr"])
def test_actual_generated_probe_refuses_identity_and_evidence_faults(controls, tmp_path, monkeypatch, fault):
    c, roots, launches, state = native_setup(controls, tmp_path, monkeypatch)
    root = roots["5080"]
    if fault == "boot": state["boot"] = "different-boot"
    elif fault == "paused_state": state["identities"][900]["state"] = "R"
    elif fault == "paused_ticks": state["identities"][900]["start_ticks"] += 1
    elif fault == "foreign_gpu": state["gpu"].append(12345)
    elif fault in ("source", "input", "legacy"):
        relative = {"source": "dummy.py", "input": "data/0.bin", "legacy": "legacy/original.pt"}[fault]
        (root / "source" / relative).write_bytes(b"modified")
    elif fault == "dispatcher_alive": state["identities"][700] = launches["5080"]["identity"]
    elif fault == "child_alive": state["identities"][100] = identity()
    elif fault == "dispatcher_exit":
        end = a.read(root / "fit_validation.supervisor_exit.json"); end["actual_os_returncode"] = 1
        dump(root / "fit_validation.supervisor_exit.json", end)
    elif fault == "child_exit":
        complete = a.read(root / "fit_validation.complete.json"); complete["actual_child_exits"][0]["actual_os_returncode"] = 1
        dump(root / "fit_validation.complete.json", complete)
    elif fault in ("dispatcher_stdout", "dispatcher_stderr"):
        (root / ("fit_validation." + fault.removeprefix("dispatcher_") + ".log")).write_bytes(b"changed after observed exit")
    before = {str(p): a.sha(p) for p in root.rglob("*") if p.is_file()}
    with pytest.raises(RuntimeError): execute_probe(c, "5080", launches["5080"])
    assert {str(p): a.sha(p) for p in root.rglob("*") if p.is_file()} == before
    assert all(command[0] == "nvidia-smi" for command in state["commands"])


def test_complete_file_is_not_parent_observed_dispatcher_exit(controls, tmp_path, monkeypatch):
    c, roots, launches, state = native_setup(controls, tmp_path, monkeypatch)
    (roots["5080"] / "fit_validation.supervisor_exit.json").unlink()
    # A complete file alone must not admit collection when the dispatcher/parent vanished.
    with pytest.raises(RuntimeError, match="dispatcher/parent identity differs"):
        execute_probe(c, "5080", launches["5080"])
    state["identities"].update({i["pid"]: i for i in (launches["5080"]["identity"], launches["5080"]["supervisor_identity"])})
    assert execute_probe(c, "5080", launches["5080"])["status"] == "waiting"


@pytest.mark.parametrize("fault", ["missing_raw", "wrong_host", "failed_reuse", "aggregate_only"])
def test_global_gate_faults_prevent_any_test_launch(controls, tmp_path, monkeypatch, fault):
    c, roots, launches, _ = native_setup(controls, tmp_path, monkeypatch)
    gate = a.build_global_gate(c, roots, tmp_path / "gate")
    job = next(j for j in c["jobs"] if j.get("reuse"))
    key = "train_exit" if fault == "failed_reuse" else "validation_receipt"
    item = a.read(gate / "global_validation_gate.json")["jobs"][job["id"]][key]
    path = gate / item["path"]
    if fault == "missing_raw": path.unlink()
    else:
        value = a.read(path)
        if fault == "wrong_host": value["host"] = "5080"
        elif fault == "failed_reuse": value["actual_os_returncode"] = 9
        else: value["individual_predictions_match"] = False
        dump(path, value)
        data = a.read(gate / "global_validation_gate.json"); data["jobs"][job["id"]][key]["sha256"] = a.sha(path)
        dump(gate / "global_validation_gate.json", data)
    payload = {str(p.relative_to(gate)): {"text": p.read_text(), "sha256": a.sha(p)} for p in gate.rglob("*") if p.is_file()}
    calls = []
    monkeypatch.setattr(a.manager, "verify_native_qualification", lambda *args: None)
    monkeypatch.setattr(a.manager, "guard", lambda *args, **kwargs: {})
    monkeypatch.setattr(a.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(a.sys, "stdin", io.StringIO(json.dumps(payload)))
    with pytest.raises((RuntimeError, FileNotFoundError)):
        exec(compile(a.test_launch_program(c, "5080", gate), "<independent gate launch>", "exec"), {})
    assert calls == []


def test_real_pack_and_scp_use_900_seconds_and_are_one_shot(controls, tmp_path, monkeypatch):
    c, roots, _, _ = native_setup(controls, tmp_path, monkeypatch)
    packed = a.manager.pack(c, "5080", "fit_validation")
    calls = []
    def shell(command, **kwargs):
        calls.append((command, kwargs["timeout"]))
        if command[0] == "ssh":
            assert "--mode pack" in command[-1]
            return subprocess.CompletedProcess(command, 0, (json.dumps(packed) + "\n").encode(), b"")
        assert command[0] == "scp"
        shutil.copyfile(packed["archive"], command[-1])
        return subprocess.CompletedProcess(command, 0, b"", b"")
    monkeypatch.setattr(a.subprocess, "run", shell)
    a.collect(c, "5080", "fit_validation")
    assert [timeout for _, timeout in calls] == [900, 900]
    with pytest.raises(RuntimeError, match="partial collection"): a.collect(c, "5080", "fit_validation")
    assert len(calls) == 2


@pytest.mark.parametrize("exit_kind", ["ssh255", "timeout", "oserror"])
def test_probe_failure_persists_actual_bytes_and_blocks_future_rounds(controls, monkeypatch, exit_kind):
    c = controls["c"]
    for h in c["hosts"].values(): h.update(environment={}, python="/pinned/python")
    monkeypatch.setattr(a, "verify", lambda *args: (controls["binding"], c, controls["launch"]))
    calls = []
    def fail(command, **kwargs):
        calls.append(command); assert kwargs["timeout"] == 45
        if exit_kind == "timeout": raise subprocess.TimeoutExpired(command, 45, output=b"raw\x00", stderr=b"bad\xff")
        if exit_kind == "oserror": raise OSError("synthetic transport unavailable")
        return subprocess.CompletedProcess(command, 255, b"raw\x00", b"bad\xff")
    monkeypatch.setattr(a.subprocess, "run", fail)
    first = a.advance(*controls["args"]); second = a.advance(*controls["args"])
    assert first["status"] == "failed" and second["cached_failure"] and len(calls) == 1
    assert first["training_failure_inferred_from_transport"] is False
    receipt = a.read(next(controls["report"].glob("rounds/*/5080/check.json")))
    assert receipt["actual_os_returncode"] == (255 if exit_kind == "ssh255" else None)
    if exit_kind != "oserror":
        assert next(controls["report"].glob("rounds/*/5080/check.stdout")).read_bytes() == b"raw\x00"
        assert next(controls["report"].glob("rounds/*/5080/check.stderr")).read_bytes() == b"bad\xff"


@pytest.mark.parametrize("fault", ["empty", "missing", "symlink"])
def test_completion_inventory_rejects_empty_missing_and_symlink_roots(controls, tmp_path, fault):
    for root in (controls["bundle"] / "retrieval", controls["bundle"] / "global_validation_v1", controls["bundle"] / "comparison_v1"):
        root.mkdir(); (root / "evidence.json").write_text("{}")
    for host in a.HOSTS: dump(controls["report"] / ("TEST_LAUNCHED_" + host + ".json"), {"host": host})
    assert a.completion_inventory()
    root = controls["bundle"] / "comparison_v1"
    if fault == "empty": (root / "evidence.json").unlink()
    elif fault == "missing": shutil.rmtree(root)
    else:
        shutil.rmtree(root); target = tmp_path / "outside"; target.mkdir(); (target / "data").write_text("data")
        root.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError): a.completion_inventory()


def add_test_cost_exit_evidence(c, root, host):
    jobs = [j for j in c["jobs"] if j["host"] == host]
    stages = [(j, "test") for j in jobs] + [(jobs[0], "prepare_batches")] + [
        (j, f"cost_{mode}_{rep}") for j in jobs for mode in ("train", "eval") for rep in (1, 2, 3)]
    entries = []
    for index, (job, stage) in enumerate(stages):
        stem = root / "control" / (job["id"] + "." + stage)
        for stream in ("stdout", "stderr"): Path(str(stem) + "." + stream + ".log").write_bytes(b"")
        value = {"job_id": job["id"], "stage": stage, "host": host, "contract_sha256": a.canonical(c),
            "actual_os_returncode": 0, "timeout": False, "supervisor_error": None, "resource_incident": None,
            "identity": identity(2000 + index), "stdout_sha256": a.sha(str(stem) + ".stdout.log"),
            "stderr_sha256": a.sha(str(stem) + ".stderr.log")}
        dump(Path(str(stem) + ".exit.json"), value); entries.append(value)
    dump(root / "test_cost.claim.json", {"identity": identity(800)})
    dump(root / "test_cost.complete.json", {"status": "complete", "host": host, "phase": "test_cost",
        "contract_sha256": a.canonical(c), "actual_child_exits": entries,
        "supervisor_own_os_exit_directly_observed": False})
    for stream in ("stdout", "stderr"): (root / ("test_cost." + stream + ".log")).write_bytes(b"")
    dump(root / "test_cost.supervisor_exit.json", {"host": host, "phase": "test_cost", "command": identity(800)["argv"],
        "contract_sha256": a.canonical(c), "actual_os_returncode": 0, "parent_own_os_exit_directly_observed": False,
        "stdout_sha256": a.sha(root / "test_cost.stdout.log"), "stderr_sha256": a.sha(root / "test_cost.stderr.log")})


def test_complete_cache_revalidates_four_real_archives_gate_and_local_comparison(controls, tmp_path, monkeypatch):
    c, roots, _, state = native_setup(controls, tmp_path, monkeypatch)
    fit_roots = {host: wrap_actual_pack(c, host, "fit_validation", roots[host],
        controls["bundle"] / "retrieval/fit_validation" / host) for host in a.HOSTS}
    gate = a.build_global_gate(c, fit_roots, controls["bundle"] / "global_validation_v1")
    for host, root in roots.items():
        shutil.copytree(gate, root, dirs_exist_ok=True)
        add_test_cost_exit_evidence(c, root, host)
        wrap_actual_pack(c, host, "test_cost", root, controls["bundle"] / "retrieval/test_cost" / host)
        dump(controls["report"] / ("TEST_LAUNCHED_" + host + ".json"), {"host": host, "phase": "test_cost"})
    dump(controls["bundle"] / "comparison_v1/summary.json", {"status": "complete"})
    dump(controls["report"] / "COMPLETE.json", {"status": "complete", "binding_sha256": "qa-binding",
        "contract_sha256": a.canonical(c), "files": a.completion_inventory()})
    before_gpu_calls = len(state["commands"])
    calls = []
    def compare(binding, contract, folder, *, cached=False):
        calls.append(cached); assert cached is True and contract == c
        return {"status": "complete"}
    monkeypatch.setattr(a, "compare", compare)
    value = a.verify_cache({}, c, "qa-binding", tmp_path / "round")
    assert value["cache_verified"] and value["remote_calls"] == 0 and calls == [True]
    assert len(state["commands"]) == before_gpu_calls
    # Even if a top-level COMPLETE file is refreshed, internal archive and gate
    # proofs must still reject missing scientific exits instead of trusting it.
    victim = next((controls["bundle"] / "retrieval/test_cost/5090/originals/control").glob("*.exit.json"))
    victim.unlink()
    complete = a.read(controls["report"] / "COMPLETE.json"); complete["files"] = a.completion_inventory()
    dump(controls["report"] / "COMPLETE.json", complete)
    with pytest.raises(RuntimeError): a.verify_cache({}, c, "qa-binding", tmp_path / "other_round")
    assert calls == [True]


def test_local_comparison_order_and_complete_cache_only_recheck(controls, tmp_path, monkeypatch):
    monkeypatch.setattr(a, "comparator_qualified", lambda binding: True)
    calls = []
    def run(command, **kwargs):
        assert command[:3] == [a.sys.executable, "-m", "paper.scripts.compare_shared_component_ablation"]
        assert kwargs["timeout"] == 900
        calls.append(command)
        if "--check-only" not in command: (controls["bundle"] / "comparison_v1").mkdir()
        return subprocess.CompletedProcess(command, 0, b'{"status":"pass"}\n', b"")
    monkeypatch.setattr(a.subprocess, "run", run)
    assert a.compare({}, controls["c"], tmp_path / "first")["status"] == "complete"
    assert ["--check-only" in c for c in calls] == [True, False, True]
    assert a.compare({}, controls["c"], tmp_path / "cache", cached=True)["status"] == "complete"
    assert len(calls) == 4 and "--check-only" in calls[-1]
    assert all(command[0] not in ("ssh", "scp") for command in calls)


def test_both_hosts_receive_identical_144_raw_evidence_files_and_only_test_cost(controls, tmp_path, monkeypatch):
    c, roots, _, state = native_setup(controls, tmp_path, monkeypatch)
    gate = a.build_global_gate(c, roots, tmp_path / "gate")
    payload = {str(p.relative_to(gate)): {"text": p.read_text(), "sha256": a.sha(p)} for p in gate.rglob("*") if p.is_file()}
    assert len(payload) == 145
    monkeypatch.setattr(a.manager, "verify_native_qualification", lambda *args: None)
    old_run = a.subprocess.run
    commands = []
    active_host = []
    def run(command, **kwargs):
        if command[0] == "nvidia-smi": return old_run(command, **kwargs)
        host = active_host[-1]; commands.append(command)
        assert command[0] == ("/usr/bin/tmux" if host == "5080" else "/opt/miniconda3/envs/ai_env/bin/tmux")
        assert "--phase test_cost" in command[-1] and "--phase fit_validation" not in command[-1]
        assert "--mode supervise" in command[-1] and "resume" not in command[-1]
        for suffix, pid in (("claim", 800), ("supervisor.intent", 801)):
            state["identities"][pid] = identity(pid)
            dump(roots[host] / ("test_cost." + suffix + ".json"), {"identity": identity(pid)})
        return subprocess.CompletedProcess(command, 0, b"", b"")
    monkeypatch.setattr(a.subprocess, "run", run)
    for host in a.HOSTS:
        active_host.append(host)
        monkeypatch.setattr(a.sys, "stdin", io.StringIO(json.dumps(payload)))
        with redirect_stdout(io.StringIO()):
            exec(compile(a.test_launch_program(c, host, gate), "<independent fixed Test start>", "exec"), {})
        for name, item in payload.items():
            assert (roots[host] / name).read_bytes() == (gate / name).read_bytes()
            assert a.sha(roots[host] / name) == item["sha256"]
    assert len(commands) == 2

