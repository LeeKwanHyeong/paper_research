"""Independent recovery QA using synthetic local files and mocked OS boundaries.

No real SSH, GPU process, model inference, or experiment metrics are used.
The original archive validator and 36-job gate execute against real pack format.
"""
from copy import deepcopy
import ast
import fcntl
from pathlib import Path
import subprocess

import pytest

from paper.scripts import advance_shared_component_ablation_recovery as r
from paper.tests.test_advance_shared_component_ablation import controls as _controls, dump
from paper.tests.test_shared_component_continuation_independent import native_setup, wrap_actual_pack
from paper.tests.test_advance_shared_component_ablation_recovery import (
    recovery_controls as _recovery_controls, reseal_preservation, rebind_auth,
    create_collection, fake_launch,
)

# Expose shared data builders as fixtures; all assertions below are independent.
controls = _controls
original_controls = _controls
recovery_controls = _recovery_controls


@pytest.fixture(autouse=True)
def forbid_external_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Independent recovery QA attempted an unmocked external command")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)


@pytest.fixture
def packed(controls, tmp_path, monkeypatch):
    c, roots, _, _ = native_setup(controls, tmp_path, monkeypatch)
    monkeypatch.setattr(r, "ROOT", controls["root"])
    monkeypatch.setattr(r, "BUNDLE", controls["bundle"])
    folder = controls["bundle"] / "retrieval/fit_validation/5090"
    original = wrap_actual_pack(c, "5090", "fit_validation", roots["5090"], folder)
    receipt = r.read(folder / "collection_receipt.json")
    marker = {"status": "verified", "host": "5090", "phase": "fit_validation",
        "contract_sha256": r.canonical(c), "manifest_sha256": receipt["manifest_sha256"],
        "archive_sha256": receipt["archive_sha256"]}
    dump(folder / "VERIFIED.json", marker)
    return c, folder, original, marker


@pytest.mark.parametrize("field", ["status", "host", "phase", "contract_sha256", "manifest_sha256", "archive_sha256"])
@pytest.mark.parametrize("mutation", ["missing", "wrong"])
def test_verified_marker_contents_are_required(packed, field, mutation):
    c, folder, original, marker = packed
    assert r.verified_collection(c, "5090", "fit_validation") == original
    if mutation == "missing":
        marker.pop(field)
    else:
        marker[field] = "incorrect"
    dump(folder / "VERIFIED.json", marker)
    with pytest.raises(RuntimeError, match="verification marker"):
        r.verified_collection(c, "5090", "fit_validation")


@pytest.mark.parametrize("target", ["archive", "extracted", "transport", "marker_symlink"])
def test_verified_marker_does_not_replace_original_archive_revalidation(packed, target, tmp_path):
    c, folder, original, _ = packed
    if target == "archive":
        path = folder / "collection_fit_validation.tar.gz"
        path.write_bytes(path.read_bytes() + b"altered")
    elif target == "extracted":
        (original / "source/legacy/original.pt").write_bytes(b"changed checkpoint")
    elif target == "transport":
        value = r.read(folder / "transport.json")
        value["actual_os_returncode"] = 255
        dump(folder / "transport.json", value)
    else:
        source = folder / "VERIFIED.json"
        outside = tmp_path / "external-marker.json"
        outside.write_bytes(source.read_bytes()); source.unlink(); source.symlink_to(outside)
    with pytest.raises(RuntimeError):
        r.verified_collection(c, "5090", "fit_validation")


def test_recovery_source_does_not_reassign_frozen_module_paths_or_start_fit():
    source = Path(r.__file__).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        for target in targets:
            assert not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                and target.value.id in {"upstream", "manager"} and target.attr in {"REPORT", "BUNDLE", "ROOT"})
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"Popen", "train", "fit", "dispatch", "supervise"}
    assert "upstream.advance_lock()" in source
    assert "upstream.launch_test(" in source


@pytest.mark.parametrize("name", ["REPORT.json", "REPORT.md", "TABLE.csv"])
@pytest.mark.parametrize("fault", ["missing", "empty", "symlink"])
def test_expected_comparison_files_cannot_be_omitted_or_empty(tmp_path, monkeypatch, name, fault):
    monkeypatch.setattr(r, "BUNDLE", tmp_path)
    output = tmp_path / "comparison_v1"; output.mkdir()
    for item in ("REPORT.json", "REPORT.md", "TABLE.csv"):
        (output / item).write_text("synthetic content")
    r.comparison_outputs_exist()
    target = output / name
    if fault == "missing": target.unlink()
    elif fault == "empty": target.write_bytes(b"")
    else:
        external = tmp_path / "outside"; external.write_text("synthetic content")
        target.unlink(); target.symlink_to(external)
    with pytest.raises(RuntimeError, match="outputs missing or empty"):
        r.comparison_outputs_exist()


def test_original_advisory_lock_excludes_recovery_without_remote_calls(controls, monkeypatch):
    monkeypatch.setattr(r, "verify", lambda *_: (controls["binding"], controls["c"], controls["launch"]))
    with (controls["report"] / "advance.lock").open("a+") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for method in (r.check_only, r.advance):
            assert method(*controls["args"]) == {"status": "busy", "remote_calls": 0}
        fcntl.flock(held, fcntl.LOCK_UN)


def test_comparison_executes_precheck_create_postcheck_then_read_only_cache(controls, monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(r.upstream, "comparator_qualified", lambda *_: True)
    def transport(command, folder, stem, **kwargs):
        commands.append((command, stem, kwargs))
        assert command[0] != "ssh" and command[0] != "scp"
        assert command[1:3] == ["-m", "paper.scripts.compare_shared_component_ablation"]
        assert command[command.index("--retrieval") + 1] == str(controls["bundle"] / "retrieval")
        assert kwargs["timeout"] == 900
        return {}, b"{}"
    monkeypatch.setattr(r.upstream, "transport", transport)
    assert r.upstream.compare(controls["binding"], controls["c"], tmp_path)["status"] == "complete"
    assert [item[1] for item in commands] == ["comparison_precheck", "comparison_create", "comparison_postcheck"]
    assert ["--check-only" in item[0] for item in commands] == [True, False, True]
    r.upstream.compare(controls["binding"], controls["c"], tmp_path, cached=True)
    assert commands[-1][1] == "comparison_cache" and "--check-only" in commands[-1][0]


@pytest.mark.parametrize("fault", ["no_job", "missing_raw", "raw_bytes", "failed_exit", "aggregate_only"])
def test_all_36_jobs_and_144_original_raw_files_required_for_launch(controls, tmp_path, monkeypatch, fault):
    c, roots, _, _ = native_setup(controls, tmp_path, monkeypatch)
    gate_root = r.upstream.build_global_gate(c, roots, tmp_path / "gate")
    gate = r.manager.verify_global_gate(c, gate_root)
    assert len(gate["jobs"]) == 36
    raw = [item for evidence in gate["jobs"].values() for key, item in evidence.items() if key != "host"]
    assert len(raw) == len({item["path"] for item in raw}) == 144
    job = next(iter(gate["jobs"]))
    item = gate["jobs"][job]["train_exit" if fault == "failed_exit" else "validation_receipt"]
    path = gate_root / item["path"]
    if fault == "no_job": gate["jobs"].pop(job)
    elif fault == "missing_raw": path.unlink()
    elif fault == "raw_bytes": path.write_bytes(path.read_bytes() + b"changed")
    else:
        value = r.read(path)
        value["actual_os_returncode" if fault == "failed_exit" else "individual_predictions_match"] = 1 if fault == "failed_exit" else False
        dump(path, value); item["sha256"] = r.sha(path)
    dump(gate_root / "global_validation_gate.json", gate)
    with pytest.raises((RuntimeError, FileNotFoundError)):
        r.manager.verify_global_gate(c, gate_root)


@pytest.mark.parametrize("field,value", [("actual_os_returncode", 1), ("actual_os_returncode", None),
    ("timeout", True), ("transport_error", "OSError"), ("timeout_seconds", 900),
    ("command", ["scp", "5080", "anything"]), ("stderr_sha256", "0" * 64), ("stderr_bytes", 0)])
def test_fresh_admission_rejects_nonapproved_historical_failures(recovery_controls, field, value):
    state = recovery_controls
    path = state["old_report"] / "rounds/historical/5080/check.json"
    check = r.read(path); check[field] = value; dump(path, check)
    reseal_preservation(state)
    with pytest.raises(RuntimeError): r.advance(*state["args"])
    assert not (r.REPORT / "rounds").exists()


@pytest.mark.parametrize("residual", ["partial_fit", "partial_test", "attempt", "interrupted",
    "old_interrupted", "one_launch", "comparison"])
def test_residual_artifacts_fail_before_any_remote_action(recovery_controls, residual):
    state = recovery_controls
    if residual == "partial_fit":
        dump(r.BUNDLE / "retrieval/fit_validation/5080/COLLECTION_ATTEMPT.json", {})
    elif residual == "partial_test":
        dump(r.BUNDLE / "retrieval/test_cost/5090/COLLECTION_ATTEMPT.json", {})
    elif residual == "attempt":
        dump(r.REPORT / "rounds/incomplete/TEST_LAUNCH_ATTEMPT_5080.json", {})
    elif residual == "interrupted":
        dump(r.REPORT / "rounds/incomplete/INTENT.json", {})
    elif residual == "old_interrupted":
        dump(state["old_report"] / "rounds/incomplete/INTENT.json", {})
        reseal_preservation(state)
    elif residual == "one_launch":
        dump(state["old_report"] / "TEST_LAUNCHED_5080.json", {})
    else:
        (r.BUNDLE / "comparison_v1").mkdir()
    first = r.advance(*state["args"])
    assert first["status"] == "failed"
    assert r.advance(*state["args"])["cached_failure"] is True


@pytest.mark.parametrize("kind", ["ssh255", "timeout", "oserror"])
def test_first_actual_transport_failure_is_byte_preserved_and_cached(recovery_controls, monkeypatch, kind):
    state = recovery_controls; calls = []
    monkeypatch.setattr(r.upstream, "ssh_command", lambda *_: ["ssh", "5080", "synthetic-read-only"])
    def fail(command, **kwargs):
        calls.append(command); assert kwargs["timeout"] == 45
        if kind == "timeout": raise subprocess.TimeoutExpired(command, 45, output=b"stdout\x00", stderr=b"stderr\xff")
        if kind == "oserror": raise OSError("synthetic unavailable transport")
        return subprocess.CompletedProcess(command, 255, b"stdout\x00", b"stderr\xff")
    monkeypatch.setattr(subprocess, "run", fail)
    preserved = {name: r.sha(name) for name in r.regular_files(state["old_report"])}
    first = r.advance(*state["args"])
    assert first["status"] == "failed" and not first["training_failure_inferred_from_transport"]
    second = r.advance(*state["args"])
    assert second["cached_failure"] and second["remote_calls"] == 0 and len(calls) == 1
    assert preserved == {name: r.sha(name) for name in r.regular_files(state["old_report"])}
    folder = next((r.REPORT / "rounds").glob("*/5080"))
    receipt = r.read(folder / "check.json")
    assert receipt["actual_os_returncode"] == (255 if kind == "ssh255" else None)
    for stream in ("stdout", "stderr"):
        assert r.sha(folder / ("check." + stream)) == receipt[stream + "_sha256"]
    if kind != "oserror":
        assert (folder / "check.stdout").read_bytes() == b"stdout\x00"
        assert (folder / "check.stderr").read_bytes() == b"stderr\xff"


@pytest.mark.parametrize("fault", ["expired", "round_limit"])
def test_expiry_and_round_limit_stop_without_transport(recovery_controls, fault):
    state = recovery_controls
    if fault == "expired": rebind_auth(state, absolute_recovery_deadline_unix=1)
    else:
        rebind_auth(state, maximum_recovery_rounds=1)
        folder = r.REPORT / "rounds/prior"
        dump(folder / "INTENT.json", {"binding_sha256": state["args"][0], "check_started_unix": 0})
        r.publish_round(folder, {"status": "waiting"})
    assert r.advance(*state["args"])["status"] == "failed"
    assert r.advance(*state["args"])["cached_failure"] is True


@pytest.mark.parametrize("mutation", ["failure", "seal"])
def test_cached_failure_bytes_are_not_silently_accepted(recovery_controls, monkeypatch, mutation):
    state = recovery_controls
    monkeypatch.setattr(r.upstream, "short_check", lambda *_: (_ for _ in ()).throw(RuntimeError("fixture failure")))
    assert r.advance(*state["args"])["status"] == "failed"
    path = r.REPORT / ("FAILURE.json" if mutation == "failure" else "FAILURE_SEAL.json")
    value = r.read(path)
    value["status" if mutation == "failure" else "failure_sha256"] = "changed"
    dump(path, value)
    with pytest.raises(RuntimeError, match="Cached recovery failure changed"):
        r.advance(*state["args"])


def test_paired_launch_receipts_without_completed_transition_are_rejected(recovery_controls):
    state = recovery_controls
    create_collection(state, "5080")
    roots = {host: r.verified_collection(state["c"], host, "fit_validation") for host in r.HOSTS}
    gate = r.upstream.build_global_gate(state["c"], roots, r.BUNDLE / "global_validation_v1")
    folder = r.REPORT / "rounds/orphan"
    for host in r.HOSTS: fake_launch(state["c"], host, gate, folder)
    result = r.advance(*state["args"])
    assert result["status"] == "failed"
    assert "launch" in result["error"].lower() or "transition" in result["error"].lower()
    assert not (folder / "INTENT.json").exists()
    assert r.advance(*state["args"])["cached_failure"] is True


@pytest.fixture
def completion_envelope(recovery_controls, monkeypatch):
    """Isolate cache-envelope validation; deep scientific verification is injected.

    Real archive validation has separate pack-format tests above. This fixture
    deliberately does not claim an end-to-end 36-Test/216-cost scientific run.
    """
    from paper.scripts import compare_shared_component_ablation as comparator
    state = recovery_controls
    for path in (r.BUNDLE / "global_validation_v1/evidence.json",): dump(path, {})
    output = r.BUNDLE / "comparison_v1"; output.mkdir()
    pure = {"status": "complete", "table": [], "fixture": "synthetic aggregate seam"}
    dump(output / "REPORT.json", pure)
    (output / "REPORT.md").write_text(comparator.render_markdown(pure))
    (output / "TABLE.csv").write_text(comparator.render_csv(pure))
    for host in r.HOSTS:
        for phase in r.PHASES:
            folder = r.BUNDLE / "retrieval" / phase / host
            for name in ("collection_receipt.json", "VERIFIED.json"):
                if not (folder / name).exists(): dump(folder / name, {"synthetic_envelope": True})
    for host in r.HOSTS: dump(state["old_report"] / ("TEST_LAUNCHED_" + host + ".json"), {"host": host})
    monkeypatch.setattr(r, "all_collections", lambda _: {(h, p): Path("synthetic-root") for h in r.HOSTS for p in r.PHASES})
    monkeypatch.setattr(r, "validate_rounds_and_launches", lambda *_, **__: {})
    monkeypatch.setattr(r.manager, "verify_global_gate", lambda *_: {})
    calls = []
    def aggregate(c, retrieval):
        assert c == state["c"] and retrieval == r.BUNDLE / "retrieval"
        calls.append("pure aggregate")
        return deepcopy(pure)
    monkeypatch.setattr(comparator, "aggregate", aggregate)
    complete = {"status": "complete", "binding_sha256": state["args"][0],
        "contract_sha256": r.canonical(state["c"]), "files": r.upstream.completion_inventory(),
        "collection_complete_sha256": r.collection_marker(state["c"], state["args"][0], create=True)}
    dump(r.REPORT / "COMPLETE.json", complete)
    folder = r.REPORT / "rounds/completed"
    dump(folder / "INTENT.json", {"check_started_unix": 0, "binding_sha256": state["args"][0]})
    r.publish_round(folder, {"status": "complete"})
    return state, pure, calls


def test_cache_verifies_all_bytes_and_invokes_pure_aggregate_without_remote(completion_envelope):
    state, _, calls = completion_envelope
    result = r.verify_completion(state["binding"], state["c"], state["args"][0])
    assert result["cache_verified"] and result["remote_calls"] == 0
    assert calls == ["pure aggregate"]


@pytest.mark.parametrize("fault", ["none", "missing_test", "missing_cost", "nonzero_cost", "duplicate_cost"])
def test_fixed_test36_cost216_exit_coverage_is_mandatory(controls, tmp_path, fault):
    c = controls["c"]; counts = {"test": 0, "cost": 0}
    for host in r.HOSTS:
        root = tmp_path / ("test_cost_" + host)
        jobs = [job for job in c["jobs"] if job["host"] == host]
        stages = [(job["id"], "test") for job in jobs]
        stages += [(jobs[0]["id"], "prepare_batches")]
        stages += [(job["id"], f"cost_{mode}_{rep}") for job in jobs for mode in ("train", "eval") for rep in (1, 2, 3)]
        entries = []
        for job, stage in stages:
            stem = root / "control" / (job + "." + stage)
            stem.parent.mkdir(parents=True, exist_ok=True)
            for stream in ("stdout", "stderr"): Path(str(stem) + "." + stream + ".log").write_bytes(b"")
            entry = {"job_id": job, "stage": stage, "host": host, "contract_sha256": r.canonical(c),
                "actual_os_returncode": 0, "timeout": False, "supervisor_error": None, "resource_incident": None,
                **{stream + "_sha256": r.sha(str(stem) + "." + stream + ".log") for stream in ("stdout", "stderr")}}
            entries.append(entry); dump(Path(str(stem) + ".exit.json"), entry)
            if stage == "test": counts["test"] += 1
            if stage.startswith("cost_"): counts["cost"] += 1
        if host == "5080":
            if fault.startswith("missing_"):
                entries.remove(next(e for e in entries if e["stage"].startswith(fault.removeprefix("missing_"))))
            elif fault == "nonzero_cost": next(e for e in entries if e["stage"].startswith("cost_"))["actual_os_returncode"] = 1
            elif fault == "duplicate_cost": entries.append(deepcopy(next(e for e in entries if e["stage"].startswith("cost_"))))
        base = {"status": "complete", "host": host, "phase": "test_cost", "contract_sha256": r.canonical(c), "actual_os_returncode": 0}
        dump(root / "test_cost.complete.json", {**base, "actual_child_exits": entries})
        dump(root / "test_cost.supervisor_exit.json", base)
        if host == "5080" and fault != "none":
            with pytest.raises(RuntimeError): r.manager.verify_phase_receipt(c, host, "test_cost", root)
        else: r.manager.verify_phase_receipt(c, host, "test_cost", root)
    assert counts == {"test": 36, "cost": 216}


@pytest.mark.parametrize("fault", ["empty_inventory", "partial_inventory", "sha", "bytes", "new_file", "removed_file", "complete_receipt"])
def test_complete_cache_rejects_inventory_and_receipt_changes(completion_envelope, fault):
    state, _, calls = completion_envelope
    path = r.REPORT / "COMPLETE.json"; complete = r.read(path)
    name = next(iter(complete["files"]))
    if fault == "empty_inventory": complete["files"] = {}
    elif fault == "partial_inventory": complete["files"].pop(name)
    elif fault == "sha": complete["files"][name]["sha256"] = "0" * 64
    elif fault == "bytes": complete["files"][name]["bytes"] += 1
    elif fault == "new_file": (r.BUNDLE / "comparison_v1/unexpected.json").write_text("{}")
    elif fault == "removed_file": Path(name).unlink()
    else: complete["unbound_change"] = True
    dump(path, complete)
    with pytest.raises((RuntimeError, FileNotFoundError)):
        r.verify_completion(state["binding"], state["c"], state["args"][0])
    assert not calls


@pytest.mark.parametrize("file", ["REPORT.json", "REPORT.md", "TABLE.csv"])
def test_resealed_cache_still_recomputes_report_content(completion_envelope, file):
    state, _, calls = completion_envelope
    target = r.BUNDLE / "comparison_v1" / file
    if file == "REPORT.json":
        value = r.read(target); value["fixture"] = "forged aggregate"; dump(target, value)
    else: target.write_text(target.read_text() + "forged\n")
    path = r.REPORT / "COMPLETE.json"; complete = r.read(path)
    complete["files"] = r.upstream.completion_inventory(); dump(path, complete)
    round_path = r.REPORT / "rounds/completed/ROUND.json"
    round_value = r.read(round_path); round_value["complete_sha256"] = r.sha(path); dump(round_path, round_value)
    latest_path = r.REPORT / "LATEST.json"
    latest = r.read(latest_path); latest["round_sha256"] = r.sha(round_path); dump(latest_path, latest)
    with pytest.raises(RuntimeError, match="pure metric recomputation"):
        r.verify_completion(state["binding"], state["c"], state["args"][0])
    assert calls == ["pure aggregate"]
