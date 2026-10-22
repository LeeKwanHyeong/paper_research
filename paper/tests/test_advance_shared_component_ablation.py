"""Synthetic local evidence tests; no SSH, GPU or scientific inference."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from paper.scripts import advance_shared_component_ablation as a


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def identity(pid=100):
    return {"pid": pid, "start_ticks": 1000 + pid, "pgid": pid,
        "argv": ["/python", "-m", "manager", "--mode", "dispatch"], "cwd": "/isolated/source"}


@pytest.fixture
def controls(tmp_path, monkeypatch):
    root = tmp_path / "repo"; bundle = root / "bundle"; report_parent = root / "report"
    report = report_parent / "continuation_v1"
    for path in (root, bundle, report): path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(a, "ROOT", root); monkeypatch.setattr(a, "BUNDLE", bundle)
    monkeypatch.setattr(a, "REPORT", report); monkeypatch.setattr(a.manager, "REPORT", report_parent)
    spec = dump(report_parent / "SCIENCE_SPEC.json", {"status": "locked"})
    for name in (a.CONTROLLER, a.COMPARATOR, "dummy.py"):
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name)
    inputs = {}
    for index in range(9):
        name = f"data/{index}.bin"; path = bundle / "input_snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(str(index).encode()); inputs[name] = a.sha(path)
    legacy_path = bundle / "input_snapshot/legacy/original.pt"
    legacy_path.parent.mkdir(parents=True); legacy_path.write_bytes(b"original checkpoint")
    jobs = [{"id": f"{dataset}__{arm}__seed{seed}", "dataset": dataset, "arm": arm,
        "seed": seed, "host": "5080" if seed == 42 else "5090",
        **({"reuse": {"checkpoint": "preserved"}} if arm == "full" and seed != 42 else {})}
        for dataset in a.manager.DATASETS for seed in (42, 52, 62) for arm in a.manager.ARMS]
    sources = {"dummy.py": a.sha(root / "dummy.py")}
    c = {"schema": a.manager.SCHEMA, "campaign": a.manager.NAME,
        "hosts": {host: {"source_root": "/isolated/source", "root": "/isolated"} for host in a.HOSTS},
        "jobs": jobs, "source": {"files": sources, "files_sha256": a.canonical(sources)},
        "input_files": inputs, "legacy_files": {"legacy/original.pt": a.sha(legacy_path)},
        "science_spec_sha256": a.sha(spec),
        "common_condition": {"time_input": "log1p", "loss_mode": "log_mse", "time_alpha": 1.0},
        "training": {"maximum_epochs": 300, "minimum_epochs": 40, "patience": 40,
            "batch_size": 128, "monitor": "validation_raw_quantity_rmse"}}
    contract = dump(bundle / "execution_contract.json", c)
    (bundle / "deployment.tar.gz").write_bytes(b"immutable deployment")
    integrity = dump(bundle / "integrity.json", {"status": "sealed", "contract_sha256": a.sha(contract),
        "canonical_contract_sha256": a.canonical(c), "archive_sha256": a.sha(bundle / "deployment.tar.gz"),
        "source_files_sha256": c["source"]["files_sha256"]})
    launch = {"contract_sha256": a.canonical(c), "hosts": {host: {"phase": "fit_validation",
        "identity": identity(), "supervisor_identity": identity(101)} for host in a.HOSTS}}
    launched = dump(report / "LAUNCHED.json", launch)
    auth = {"status": "approved", "campaign": a.manager.NAME, "maximum_check_seconds": 45,
        "minimum_check_interval_seconds": 900, "transfer_timeout_seconds": 900,
        **{key: True for key in ("matched_component_3seed", "initial_fit_adoption_only", "short_terminal_checks",
            "collect_once", "global_validation_then_test_cost", "local_comparison", "preserve_existing",
            "mac_deferred", "no_scientific_retry")}}
    authorization = dump(report / "AUTHORIZATION.json", auth)
    binding = {"campaign": a.manager.NAME, "report_root": str(report), "bundle_root": str(bundle),
        "authorization_sha256": a.sha(authorization), "contract_sha256": a.sha(contract),
        "integrity_sha256": a.sha(integrity), "science_spec_sha256": a.sha(spec),
        "initial_launch_sha256": a.sha(launched), "files": {name: a.sha(root / name)
            for name in (a.CONTROLLER, a.COMPARATOR)}}
    bound = dump(report / "CONTROL_BINDINGS.json", binding)
    return {"c": c, "launch": launch, "binding": binding, "bound": bound, "authorization": authorization,
        "root": root, "bundle": bundle, "report": report,
        "args": (a.sha(bound), a.sha(authorization))}


def host_evidence(root, c, host):
    entries = []
    for job in c["jobs"]:
        if job["host"] != host: continue
        for stage in ("train", "validation"):
            exit_stage = "reuse" if stage == "train" and job.get("reuse") else stage
            base = {"job_id": job["id"], "stage": stage, "host": host,
                "contract_sha256": a.canonical(c), "status": "complete"}
            if stage == "validation": base.update(validation_replay_passed=True, individual_predictions_match=True)
            dump(root / "run" / job["id"] / (stage + "_receipt.json"), base)
            prefix = root / "control" / (job["id"] + "." + exit_stage)
            prefix.parent.mkdir(parents=True, exist_ok=True)
            Path(str(prefix) + ".stdout.log").write_bytes(b"{}\n")
            Path(str(prefix) + ".stderr.log").write_bytes(b"")
            item = {**base, "stage": exit_stage, "actual_os_returncode": 0, "timeout": False,
                "supervisor_error": None, "resource_incident": None,
                "stdout_sha256": a.sha(str(prefix) + ".stdout.log"),
                "stderr_sha256": a.sha(str(prefix) + ".stderr.log"), "identity": identity()}
            dump(Path(str(prefix) + ".exit.json"), item); entries.append(item)
    (root / "fit_validation.stdout.log").write_bytes(b'{"status":"complete"}\n')
    (root / "fit_validation.stderr.log").write_bytes(b"")
    for suffix in ("complete", "supervisor_exit"):
        dump(root / ("fit_validation." + suffix + ".json"), {"status": "complete", "host": host,
            "phase": "fit_validation", "contract_sha256": a.canonical(c), "actual_os_returncode": 0,
            "actual_child_exits": entries,
            "stdout_sha256": a.sha(root / "fit_validation.stdout.log"),
            "stderr_sha256": a.sha(root / "fit_validation.stderr.log")})
    for name in c["legacy_files"]:
        path = root / "source" / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"original checkpoint")


def collection_fixture(folder, c, host, source):
    folder.mkdir(parents=True)
    inventory = {str(path.relative_to(source)): {"sha256": a.sha(path), "bytes": path.stat().st_size}
        for path in source.rglob("*") if path.is_file()}
    manifest = {"host": host, "phase": "fit_validation", "contract_sha256": a.canonical(c),
        "files": inventory, "source_files": c["source"]["files"], "input_files": c["input_files"],
        "legacy_files": c["legacy_files"], "all_source_input_legacy_actual_sha_verified": True}
    manifest_name = "collection_fit_validation.manifest.json"
    manifest_path = dump(source / manifest_name, manifest)
    archive = folder / "collection_fit_validation.tar.gz"
    with tarfile.open(archive, "w:gz") as packed:
        packed.add(manifest_path, arcname=manifest_name)
        for name in inventory: packed.add(source / name, arcname=name)
    receipt = {"status": "complete", "host": host, "phase": "fit_validation", "contract_sha256": a.canonical(c),
        "archive_sha256": a.sha(archive), "archive_bytes": archive.stat().st_size,
        "manifest_sha256": a.sha(manifest_path), "files": len(inventory)}
    dump(folder / "collection_receipt.json", receipt)
    dump(folder / "pack.stdout", receipt); (folder / "pack.stderr").write_bytes(b"")
    dump(folder / "transport.json", {"actual_os_returncode": 0, "timeout": False, "transport_error": None,
        "stdout_sha256": a.sha(folder / "pack.stdout"), "stderr_sha256": a.sha(folder / "pack.stderr")})
    for stream in ("stdout", "stderr"): (folder / ("transfer." + stream)).write_bytes(b"")
    dump(folder / "transfer.json", {"actual_os_returncode": 0, "timeout": False, "transport_error": None,
        "stdout_sha256": a.sha(folder / "transfer.stdout"), "stderr_sha256": a.sha(folder / "transfer.stderr")})


def test_verify_every_pinned_file_before_any_transport(controls, monkeypatch):
    calls = []; monkeypatch.setattr(a.subprocess, "run", lambda *args, **kw: calls.append(args))
    assert a.verify(*controls["args"])[1] == controls["c"]
    (controls["root"] / "dummy.py").write_text("changed")
    with pytest.raises(RuntimeError, match="source file changed"): a.verify(*controls["args"])
    assert not calls


@pytest.mark.parametrize("target", ["controller", "input", "legacy", "launch", "archive", "authorization"])
def test_binding_rejects_local_changes(controls, target):
    path = {"controller": controls["root"] / a.CONTROLLER,
        "input": controls["bundle"] / "input_snapshot/data/0.bin",
        "legacy": controls["bundle"] / "input_snapshot/legacy/original.pt",
        "launch": controls["report"] / "LAUNCHED.json", "archive": controls["bundle"] / "deployment.tar.gz",
        "authorization": controls["authorization"]}[target]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(RuntimeError): a.verify(*controls["args"])


def test_busy_and_cooldown_never_connect(controls, monkeypatch):
    from contextlib import contextmanager
    @contextmanager
    def busy(): yield False
    monkeypatch.setattr(a, "advance_lock", busy)
    monkeypatch.setattr(a, "short_check", lambda *x: pytest.fail("SSH during busy"))
    assert a.advance(*controls["args"])["status"] == "busy"
    @contextmanager
    def ready(): yield True
    monkeypatch.setattr(a, "advance_lock", ready)
    round_file = dump(controls["report"] / "rounds/prior/ROUND.json", {"status": "waiting"})
    dump(controls["report"] / "LATEST.json", {"round_path": str(round_file), "round_sha256": a.sha(round_file),
        "status": "waiting", "check_started_unix": a.time.time()})
    assert a.advance(*controls["args"])["status"] == "waiting_next_check"


def test_first_transport_failure_blocks_future_calls(controls, monkeypatch):
    calls = []
    def fail(*args): calls.append(args); raise RuntimeError("SSH255 evidence saved")
    monkeypatch.setattr(a, "short_check", fail)
    first = a.advance(*controls["args"])
    assert first["status"] == "failed" and len(calls) == 1
    second = a.advance(*controls["args"])
    assert second["cached_failure"] is True and len(calls) == 1
    assert second["training_failure_inferred_from_transport"] is False


def test_transport_preserves_timeout_bytes_and_no_retry(tmp_path, monkeypatch):
    calls = []
    def fail(command, **kwargs):
        calls.append(command); raise subprocess.TimeoutExpired(command, 45, output=b"partial\x00stdout", stderr=b"error\xff")
    monkeypatch.setattr(a.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="no retry"):
        a.transport(["ssh", "example"], tmp_path, "check", timeout=45)
    receipt = a.read(tmp_path / "check.json")
    assert receipt["actual_os_returncode"] is None and receipt["timeout"] is True
    assert (tmp_path / "check.stdout").read_bytes() == b"partial\x00stdout"
    assert (tmp_path / "check.stderr").read_bytes() == b"error\xff"
    assert receipt["stdout_sha256"] == a.sha(tmp_path / "check.stdout") and len(calls) == 1
    with pytest.raises(RuntimeError, match="Existing transport"):
        a.transport(["ssh", "example"], tmp_path, "check", timeout=45)
    assert len(calls) == 1


@pytest.mark.parametrize("name", ["/absolute", "../escape", "a/../../escape", "./value", "a//value", ""])
def test_archive_path_rejected(name):
    with pytest.raises(RuntimeError): a.safe_member(name)


def test_global_gate_copies_all_36_raw_jobs_and_reuse_exit_stages(controls, tmp_path):
    c = controls["c"]; roots = {host: tmp_path / host for host in a.HOSTS}
    for host in a.HOSTS: host_evidence(roots[host], c, host)
    target = a.build_global_gate(c, roots, tmp_path / "gate")
    gate = a.manager.verify_global_gate(c, target)
    assert len(gate["jobs"]) == 36
    reused = next(job for job in c["jobs"] if job.get("reuse"))
    item = gate["jobs"][reused["id"]]["train_exit"]
    assert a.read(target / item["path"])["stage"] == "reuse"
    assert len(list((target / "global_validation_sources").rglob("*.json"))) == 144
    with pytest.raises(FileExistsError): a.build_global_gate(c, roots, target)


@pytest.mark.parametrize("bad", ["host", "exit", "replay"])
def test_global_gate_refuses_wrong_host_failed_exit_and_missing_individual_replay(controls, tmp_path, bad):
    c = controls["c"]; roots = {host: tmp_path / host for host in a.HOSTS}
    for host in a.HOSTS: host_evidence(roots[host], c, host)
    job = c["jobs"][0]
    if bad == "exit":
        path = roots[job["host"]] / "control" / (job["id"] + ".validation.exit.json")
        value = a.read(path); value["actual_os_returncode"] = 1
    else:
        path = roots[job["host"]] / "run" / job["id"] / "validation_receipt.json"
        value = a.read(path)
        if bad == "host": value["host"] = "5090"
        else: value["individual_predictions_match"] = False
    dump(path, value)
    with pytest.raises(RuntimeError): a.build_global_gate(c, roots, tmp_path / "gate")


def test_actual_archive_extract_rehash_and_phase_exit_validation(controls, tmp_path):
    c = controls["c"]; source, folder = tmp_path / "source", tmp_path / "collected"
    host_evidence(source, c, "5080"); collection_fixture(folder, c, "5080", source)
    originals = a.validate_collection(c, "5080", "fit_validation", folder, extract=True)
    assert len(a.manager.verify_phase_receipt(c, "5080", "fit_validation", originals)[0]["actual_child_exits"]) == 24
    assert a.validate_collection(c, "5080", "fit_validation", folder) == originals
    target = next((originals / "run").rglob("train_receipt.json")); target.write_text("tampered")
    with pytest.raises(RuntimeError, match="SHA differ"):
        a.validate_collection(c, "5080", "fit_validation", folder)


@pytest.mark.parametrize("bad", ["receipt", "pack_stdout", "archive", "transfer_exit", "manifest_host"])
def test_collection_refuses_broken_receipt_byte_and_transport_chain(controls, tmp_path, bad):
    c = controls["c"]; source, folder = tmp_path / "source", tmp_path / "collected"
    host_evidence(source, c, "5080"); collection_fixture(folder, c, "5080", source)
    if bad in ("receipt", "pack_stdout"):
        path = folder / ("collection_receipt.json" if bad == "receipt" else "pack.stdout")
        value = a.read(path); value["host"] = "5090"; dump(path, value)
    elif bad == "archive":
        with (folder / "collection_fit_validation.tar.gz").open("ab") as stream: stream.write(b"tamper")
    elif bad == "transfer_exit":
        path = folder / "transfer.json"; value = a.read(path); value["actual_os_returncode"] = 255; dump(path, value)
    else:
        path = folder / "collection_receipt.json"; value = a.read(path); value["manifest_sha256"] = "0" * 64
        dump(path, value); dump(folder / "pack.stdout", value)
        transport = a.read(folder / "transport.json"); transport["stdout_sha256"] = a.sha(folder / "pack.stdout")
        dump(folder / "transport.json", transport)
    with pytest.raises(RuntimeError): a.validate_collection(c, "5080", "fit_validation", folder, extract=True)


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE])
def test_collection_rejects_nonregular_tar_members(controls, tmp_path, kind):
    c = controls["c"]; source, folder = tmp_path / "source", tmp_path / "collected"
    host_evidence(source, c, "5080"); collection_fixture(folder, c, "5080", source)
    archive = folder / "collection_fit_validation.tar.gz"
    with tarfile.open(archive, "w:gz") as packed:
        member = tarfile.TarInfo("unsafe"); member.type = kind; member.linkname = "/etc/passwd"; packed.addfile(member)
    receipt = a.read(folder / "collection_receipt.json")
    receipt.update(archive_sha256=a.sha(archive), archive_bytes=archive.stat().st_size)
    dump(folder / "collection_receipt.json", receipt); dump(folder / "pack.stdout", receipt)
    transport = a.read(folder / "transport.json"); transport["stdout_sha256"] = a.sha(folder / "pack.stdout")
    dump(folder / "transport.json", transport)
    with pytest.raises(RuntimeError, match="nonregular"):
        a.validate_collection(c, "5080", "fit_validation", folder, extract=True)


def test_partial_collection_blocks_retrieval_retry(controls):
    folder = controls["bundle"] / "retrieval/fit_validation/5080"; folder.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Partial collection"):
        a.collected(controls["c"], "5080", "fit_validation")


def test_cached_archive_rehashes_payload_even_when_originals_are_untouched(controls, tmp_path):
    c = controls["c"]; source, folder = tmp_path / "source", tmp_path / "collected"
    host_evidence(source, c, "5080"); collection_fixture(folder, c, "5080", source)
    a.validate_collection(c, "5080", "fit_validation", folder, extract=True)
    archive = folder / "collection_fit_validation.tar.gz"
    with tarfile.open(archive, "r:gz") as packed:
        rows = [(member, packed.extractfile(member).read()) for member in packed.getmembers()]
    original_name = "source/legacy/original.pt"
    with tarfile.open(archive, "w:gz") as packed:
        for member, payload in rows:
            if member.name == original_name: payload = b"X" * len(payload)
            packed.addfile(member, io.BytesIO(payload))
    receipt = a.read(folder / "collection_receipt.json")
    receipt.update(archive_sha256=a.sha(archive), archive_bytes=archive.stat().st_size)
    dump(folder / "collection_receipt.json", receipt); dump(folder / "pack.stdout", receipt)
    transport = a.read(folder / "transport.json"); transport["stdout_sha256"] = a.sha(folder / "pack.stdout")
    dump(folder / "transport.json", transport)
    assert (folder / "originals" / original_name).read_bytes() == b"original checkpoint"
    with pytest.raises(RuntimeError, match="Packed file SHA"):
        a.validate_collection(c, "5080", "fit_validation", folder)


def test_test_program_has_no_training_or_checkpoint_write_and_uses_stdin(controls, tmp_path):
    c = controls["c"]
    for host in a.HOSTS:
        program = a.test_launch_program(c, host, tmp_path)
        compile(program, "<test launch>", "exec")
        assert "json.load(sys.stdin)" in program and "'--phase','test_cost'" in program
        assert "verify_global_gate" in program and "fit_validation" in program
        assert "'--stage','train'" not in program and "torch.save" not in program
        probe = a.terminal_program(c, host, "fit_validation", controls["launch"]["hosts"][host])
        compile(probe, "<read-only probe>", "exec")
        assert "subprocess.Popen" not in probe and "kill" not in probe and "write(" not in probe


def test_read_only_probe_executes_actual_terminal_and_waiting_identity_paths(controls, tmp_path, monkeypatch):
    from contextlib import redirect_stdout
    c = deepcopy(controls["c"]); native = tmp_path / "native"
    c["hosts"]["5080"]["root"] = str(native)
    host_evidence(native, c, "5080"); dump(native / "execution_contract.json", c)
    launch = controls["launch"]["hosts"]["5080"]
    dump(native / "fit_validation.claim.json", {"identity": launch["identity"]})
    dump(native / "fit_validation.supervisor.intent.json", {"identity": launch["supervisor_identity"]})
    end = a.read(native / "fit_validation.supervisor_exit.json")
    end["command"] = launch["identity"]["argv"]; dump(native / "fit_validation.supervisor_exit.json", end)
    monkeypatch.setattr(a.manager, "verify_files", lambda *x: None)
    monkeypatch.setattr(a.manager, "guard", lambda *x, **kw: {"compute_pids": [], "paused_preserved": 0})
    monkeypatch.setattr(a.manager, "proc_identity", lambda pid: None)
    program = a.terminal_program(c, "5080", "fit_validation", launch)
    out = io.StringIO()
    with redirect_stdout(out): exec(compile(program, "<terminal synthetic>", "exec"), {})
    assert a.json_response(out.getvalue().encode())["status"] == "terminal"
    (native / "fit_validation.supervisor_exit.json").unlink()
    history = native / "run/job/runs/model/history.json"
    dump(history, {"history": [{"epoch": epoch, "elapsed_seconds": 2.0 * epoch} for epoch in range(300)]})
    identities = {item["pid"]: item for item in (launch["identity"], launch["supervisor_identity"])}
    monkeypatch.setattr(a.manager, "proc_identity", lambda pid: identities.get(pid))
    out = io.StringIO()
    with redirect_stdout(out): exec(compile(program, "<waiting synthetic>", "exec"), {})
    waiting = a.json_response(out.getvalue().encode())
    assert waiting["status"] == "waiting"
    assert waiting["progress"][0]["rows"] == 300 and len(waiting["progress"][0]["tail"]) == 5
    assert waiting["progress"][0]["tail"][0]["epoch"] == 295
    identities[launch["identity"]["pid"]] = {**launch["identity"], "start_ticks": 999999}
    with pytest.raises(RuntimeError, match="identity differs"):
        exec(compile(program, "<changed synthetic>", "exec"), {})


def test_native_test_launch_executes_global_gate_and_pinned_tmux_only(controls, tmp_path, monkeypatch):
    from contextlib import redirect_stdout
    c = deepcopy(controls["c"]); roots = {host: tmp_path / ("native_" + host) for host in a.HOSTS}
    for host in a.HOSTS:
        c["hosts"][host].update(root=str(roots[host]), source_root=str(roots[host] / "source"),
            python="/pinned/python", environment={"SOURCE_REVISION": "pinned", "PYTHONPATH": str(roots[host] / "source")},
            tmux="isolated_" + host)
    for host in a.HOSTS:
        host_evidence(roots[host], c, host); dump(roots[host] / "execution_contract.json", c)
    gate = a.build_global_gate(c, roots, tmp_path / "global_gate")
    payload = {str(path.relative_to(gate)): {"text": path.read_text(), "sha256": a.sha(path)}
        for path in gate.rglob("*") if path.is_file()}
    monkeypatch.setattr(a.manager, "verify_files", lambda *x: None)
    monkeypatch.setattr(a.manager, "verify_native_qualification", lambda *x: None)
    monkeypatch.setattr(a.manager, "guard", lambda *x, **kw: {"compute_pids": [], "paused_preserved": 0})
    launch = {"identity": identity(), "supervisor_identity": identity(101)}; commands = []
    def tmux(command, **kwargs):
        commands.append(command)
        dump(roots["5080"] / "test_cost.claim.json", {"identity": launch["identity"]})
        dump(roots["5080"] / "test_cost.supervisor.intent.json", {"identity": launch["supervisor_identity"]})
        return subprocess.CompletedProcess(command, 0, b"", b"")
    monkeypatch.setattr(a.subprocess, "run", tmux)
    monkeypatch.setattr(a.manager, "proc_identity", lambda pid: next((v for v in launch.values() if v["pid"] == pid), None))
    monkeypatch.setattr(a.sys, "stdin", io.StringIO(json.dumps(payload)))
    out = io.StringIO()
    program = a.test_launch_program(c, "5080", gate)
    with redirect_stdout(out): exec(compile(program, "<Test launch synthetic>", "exec"), {})
    value = a.json_response(out.getvalue().encode())
    assert value["status"] == "launched" and value["phase"] == "test_cost"
    assert a.manager.verify_global_gate(c, roots["5080"])["status"] == "passed"
    assert len(commands) == 1 and commands[0][0] == "/usr/bin/tmux"
    shell_command = commands[0][-1]
    assert "SOURCE_REVISION=pinned" in shell_command and "--phase test_cost" in shell_command
    monkeypatch.setattr(a.sys, "stdin", io.StringIO(json.dumps(payload)))
    with pytest.raises(RuntimeError, match="Existing Test/gate"):
        exec(compile(program, "<Duplicate Test launch synthetic>", "exec"), {})
    assert len(commands) == 1


def test_comparison_requires_independent_source_bound_qa(controls):
    assert not a.comparator_qualified(controls["binding"])
    qa = dump(controls["report"].parent / "COMPARATOR_QA.json", {"status": "PASS",
        "files_reviewed_sha256": {a.COMPARATOR: a.sha(controls["root"] / a.COMPARATOR)}})
    binding = {**controls["binding"], "comparator_qa_sha256": a.sha(qa)}
    assert a.comparator_qualified(binding)
    value = a.read(qa); value["status"] = "FAIL"; dump(qa, value)
    with pytest.raises(RuntimeError): a.comparator_qualified(binding)


def test_round_receipts_are_preserved_and_pointer_bound(controls):
    folder = controls["report"] / "rounds/one"
    dump(folder / "INTENT.json", {"check_started_unix": 123})
    value = a.publish_round(folder, {"status": "waiting"})
    pointer = a.read(controls["report"] / "LATEST.json")
    assert pointer["round_sha256"] == a.sha(folder / "ROUND.json") and pointer["check_started_unix"] == 123
    assert value["status"] == "waiting"
    with pytest.raises(FileExistsError): a.publish_round(folder, {"status": "different"})
