"""Synthetic recovery controls; no remote execution, GPU, or original writes."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from paper.scripts import advance_shared_component_ablation as upstream
from paper.scripts import advance_shared_component_ablation_recovery as r
from paper.tests.test_advance_shared_component_ablation import (
    controls as original_controls, collection_fixture, dump, host_evidence, identity,
)


def refresh_binding(state):
    """Rebind only this test's newly authored recovery envelope."""
    state["binding"]["authorization_sha256"] = r.sha(state["authorization"])
    state["binding"]["preservation_sha256"] = r.sha(r.REPORT / "PRESERVATION.json")
    dump(state["bound"], state["binding"])
    state["args"] = (r.sha(state["bound"]), r.sha(state["authorization"]))
    return state["args"]


def rebind_auth(state, **updates):
    value = r.read(state["authorization"]); value.update(updates)
    dump(state["authorization"], value)
    return refresh_binding(state)


def reseal_preservation(state):
    files = {name: {"sha256": r.sha(name), "bytes": Path(name).stat().st_size}
        for root in (upstream.REPORT, r.BUNDLE / "retrieval/fit_validation/5090")
        for name in r.regular_files(root)}
    dump(r.REPORT / "PRESERVATION.json", {"files": files})
    return refresh_binding(state)


def create_collection(state, host, phase="fit_validation"):
    assert phase == "fit_validation", "This local fixture contains fit/Validation evidence only"
    source = state["root"] / ("native_" + host)
    host_evidence(source, state["c"], host)
    folder = state["bundle"] / "retrieval" / phase / host
    collection_fixture(folder, state["c"], host, source)
    root = upstream.validate_collection(state["c"], host, phase, folder, extract=True)
    receipt = r.read(folder / "collection_receipt.json")
    dump(folder / "VERIFIED.json", {"status": "verified", "host": host, "phase": phase,
        "contract_sha256": r.canonical(state["c"]), "manifest_sha256": receipt["manifest_sha256"],
        "archive_sha256": receipt["archive_sha256"]})
    return root


@pytest.fixture
def recovery_controls(original_controls, monkeypatch):
    state = original_controls
    monkeypatch.setattr(r, "ROOT", state["root"]); monkeypatch.setattr(r, "BUNDLE", state["bundle"])
    monkeypatch.setattr(r, "REPORT", state["report"].parent / "recovery_v1")
    r.REPORT.mkdir()
    c = state["c"]
    for host in r.HOSTS:
        c["hosts"][host]["absolute_host_deadline_unix"] = 10**12
    contract = dump(state["bundle"] / "execution_contract.json", c)
    integrity_path = state["bundle"] / "integrity.json"
    integrity = r.read(integrity_path)
    integrity.update(contract_sha256=r.sha(contract), canonical_contract_sha256=r.canonical(c))
    dump(integrity_path, integrity)
    state["launch"]["contract_sha256"] = r.canonical(c)
    dump(upstream.REPORT / "LAUNCHED.json", state["launch"])
    old_binding = deepcopy(state["binding"])
    old_binding.update(contract_sha256=r.sha(contract), integrity_sha256=r.sha(integrity_path),
        initial_launch_sha256=r.sha(upstream.REPORT / "LAUNCHED.json"))
    dump(upstream.REPORT / "CONTROL_BINDINGS.json", old_binding)
    monkeypatch.setattr(r, "UPSTREAM_CONTROLLER_SHA256", old_binding["files"][upstream.CONTROLLER])
    historical = upstream.REPORT / "rounds/historical"
    dump(historical / "INTENT.json", {"check_started_unix": 0})
    for stream, data in (("stdout", b""), ("stderr", b"ssh failure bytes\xff")):
        path = historical / "5080" / ("check." + stream)
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    dump(historical / "5080/check.json", {"command": ["ssh", "5080", "read-only probe"],
        "actual_os_returncode": 255, "timeout": False, "transport_error": None, "timeout_seconds": 45,
        **{stream + "_sha256": r.sha(historical / "5080" / ("check." + stream)) for stream in ("stdout", "stderr")},
        **{stream + "_bytes": (historical / "5080" / ("check." + stream)).stat().st_size for stream in ("stdout", "stderr")}})
    failure = {"status": "failed", "error": r.HISTORICAL_ERROR, "round_path": str(historical),
        "automatic_retry": False, "training_failure_inferred_from_transport": False}
    dump(upstream.REPORT / "FAILURE.json", failure)
    dump(historical / "ROUND.json", failure)
    dump(upstream.REPORT / "LATEST.json", {"status": "failed", "round_path": str(historical / "ROUND.json"),
        "round_sha256": r.sha(historical / "ROUND.json"), "check_started_unix": 0})
    (upstream.REPORT / "advance.lock").write_bytes(b"preserved advisory lock bytes\n")
    create_collection(state, "5090")
    for name in (r.CONTROLLER, r.TEST):
        path = state["root"] / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name)
    auth = {"status": "approved", "campaign": r.manager.NAME, "user_instruction": "Preserve and recover; no new training or MAC.",
        "upstream_binding_sha256": r.sha(upstream.REPORT / "CONTROL_BINDINGS.json"),
        "upstream_authorization_sha256": r.sha(upstream.REPORT / "AUTHORIZATION.json"),
        "historical_failure_sha256": r.sha(upstream.REPORT / "FAILURE.json"),
        "maximum_check_seconds": 45, "minimum_check_interval_seconds": 900, "transfer_timeout_seconds": 900,
        "maximum_recovery_rounds": 96, "absolute_recovery_deadline_unix": 10**12,
        **{key: True for key in ("recovery_followup_only", "initial_fit_adoption_only", "short_terminal_checks", "collect_once",
            "global_validation_then_test_cost", "local_comparison", "preserve_existing", "mac_deferred", "no_scientific_retry",
            "allow_new_test_launch_receipts_in_original_continuation", "no_new_training", "no_mac_execution")}}
    state["authorization"] = dump(r.REPORT / "AUTHORIZATION.json", auth)
    files = {name: r.sha(state["root"] / name) for name in (r.CONTROLLER, r.TEST)}
    qa = dump(r.REPORT / "INDEPENDENT_QA.json", {"status": "PASS", "independent_reviewer": True,
        "reviewer": "synthetic independent reviewer", "files_reviewed_sha256": files})
    state["binding"] = {"campaign": r.manager.NAME, "report_root": str(r.REPORT), "bundle_root": str(r.BUNDLE),
        "authorization_sha256": r.sha(state["authorization"]), "files": files, "recovery_qa_sha256": r.sha(qa),
        "user_instruction_sha256": hashlib.sha256(auth["user_instruction"].encode()).hexdigest(),
        **{key: auth[key] for key in ("upstream_binding_sha256", "upstream_authorization_sha256", "historical_failure_sha256")}}
    state["bound"] = r.REPORT / "CONTROL_BINDINGS.json"
    state["old_report"], state["report"] = upstream.REPORT, r.REPORT
    reseal_preservation(state)
    return state


@pytest.fixture(autouse=True)
def no_external_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Recovery test attempted an unmocked external command")
    monkeypatch.setattr(upstream.subprocess, "run", forbidden)
    monkeypatch.setattr(upstream.subprocess, "Popen", forbidden)


def test_local_check_keeps_original_failure_and_latest_bytes(recovery_controls):
    state = recovery_controls
    before = {name: r.sha(name) for name in r.regular_files(upstream.REPORT)}
    assert r.check_only(*state["args"]) == {"status": "ready_to_advance", "remote_calls": 0,
        "collected_fit_hosts": ["5090"], "collected_test_cost_hosts": []}
    assert {name: r.sha(name) for name in r.regular_files(upstream.REPORT)} == before
    assert not (r.REPORT / "LATEST.json").exists()


@pytest.mark.parametrize("target", ["source", "test", "qa", "old_latest", "old_failure", "5090_archive", "input", "legacy"])
def test_pins_and_preservation_block_before_any_action(recovery_controls, monkeypatch, target):
    state = recovery_controls
    path = {"source": r.ROOT / r.CONTROLLER, "test": r.ROOT / r.TEST,
        "qa": r.REPORT / "INDEPENDENT_QA.json", "old_latest": upstream.REPORT / "LATEST.json",
        "old_failure": upstream.REPORT / "FAILURE.json",
        "5090_archive": r.BUNDLE / "retrieval/fit_validation/5090/collection_fit_validation.tar.gz",
        "input": r.BUNDLE / "input_snapshot/data/0.bin", "legacy": r.BUNDLE / "input_snapshot/legacy/original.pt"}[target]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(RuntimeError):
        r.advance(*state["args"])
    assert not (r.REPORT / "LATEST.json").exists()


def test_marker_contents_must_match_actual_validated_collection(recovery_controls):
    state = recovery_controls
    marker = r.BUNDLE / "retrieval/fit_validation/5090/VERIFIED.json"
    value = r.read(marker); value["host"] = "5080"; dump(marker, value)
    reseal_preservation(state)
    with pytest.raises(RuntimeError, match="marker differs"):
        r.check_only(*state["args"])


def test_busy_uses_original_lock_without_touching_original_bytes(recovery_controls):
    state = recovery_controls
    with upstream.advance_lock() as acquired:
        assert acquired
        assert r.advance(*state["args"])["status"] == "busy"
    assert (upstream.REPORT / "advance.lock").read_bytes() == b"preserved advisory lock bytes\n"


def test_first_transport_failure_is_recovery_cached_without_changing_historical_failure(recovery_controls, monkeypatch):
    state = recovery_controls; calls = []
    old = {name: r.sha(name) for name in r.regular_files(upstream.REPORT)}
    def fail(*args):
        calls.append(args); raise RuntimeError("new SSH255; original bytes preserved")
    monkeypatch.setattr(upstream, "short_check", fail)
    first = r.advance(*state["args"]); second = r.advance(*state["args"])
    assert first["status"] == "failed" and second["cached_failure"] is True
    assert second["remote_calls"] == 0 and len(calls) == 1
    assert second["training_failure_inferred_from_transport"] is False
    assert {name: r.sha(name) for name in r.regular_files(upstream.REPORT)} == old
    assert r.check_only(*state["args"])["cached_failure"] is True


def test_waiting_status_only_once_and_next_call_obeys_cooldown(recovery_controls, monkeypatch):
    state = recovery_controls; calls = []
    def waiting(c, host, phase, launch, folder):
        calls.append((host, phase)); return {"status": "waiting", "host": host, "phase": phase}
    monkeypatch.setattr(upstream, "short_check", waiting)
    result = r.advance(*state["args"])
    assert result["status"] == "waiting" and calls == [("5080", "fit_validation")]
    assert r.advance(*state["args"])["status"] == "waiting_next_check"
    assert len(calls) == 1


def test_interrupted_round_is_preserved_and_failure_is_cached(recovery_controls, monkeypatch):
    state = recovery_controls
    dump(r.REPORT / "rounds/interrupted/INTENT.json", {"binding_sha256": state["args"][0], "check_started_unix": 0})
    result = r.advance(*state["args"])
    assert result["status"] == "failed" and "Interrupted" in result["error"]
    assert not (r.REPORT / "rounds/interrupted/ROUND.json").exists()
    assert r.advance(*state["args"])["cached_failure"] is True


def test_partial_retrieval_blocks_before_status_call(recovery_controls):
    state = recovery_controls
    dump(r.BUNDLE / "retrieval/fit_validation/5080/COLLECTION_ATTEMPT.json", {"status": "attempted"})
    result = r.advance(*state["args"])
    assert result["status"] == "failed" and "Partial collection" in result["error"]
    assert r.advance(*state["args"])["cached_failure"] is True


def fake_launch(c, host, gate, folder):
    gate_sha = r.sha(gate / "global_validation_gate.json")
    value = {"status": "launched", "host": host, "phase": "test_cost", "contract_sha256": r.canonical(c),
        "identity": identity(), "supervisor_identity": identity(101), "global_validation_gate_sha256": gate_sha}
    dump(folder / ("TEST_LAUNCH_ATTEMPT_" + host + ".json"), {"host": host, "contract_sha256": r.canonical(c), "global_gate_sha256": gate_sha})
    stem = "test_launch_" + host
    dump(folder / (stem + ".stdout"), value); (folder / (stem + ".stderr")).write_bytes(b"")
    dump(folder / (stem + ".json"), {"actual_os_returncode": 0, "timeout": False, "transport_error": None,
        "stdout_sha256": r.sha(folder / (stem + ".stdout")), "stderr_sha256": r.sha(folder / (stem + ".stderr"))})
    dump(upstream.REPORT / ("TEST_LAUNCHED_" + host + ".json"), value)
    return value


def test_real_gate_admits_both_fixed_test_launches_after_5080_collection(recovery_controls, monkeypatch):
    state = recovery_controls; actions = []
    def terminal(c, host, phase, launch, folder):
        actions.append(("check", host, phase)); return {"status": "terminal"}
    def collect(c, host, phase):
        actions.append(("collect", host, phase)); return create_collection(state, host, phase)
    def launch(c, host, gate, folder):
        assert len(r.manager.verify_global_gate(c, gate)["jobs"]) == 36
        actions.append(("test", host)); return fake_launch(c, host, gate, folder)
    monkeypatch.setattr(upstream, "short_check", terminal); monkeypatch.setattr(upstream, "collect", collect)
    monkeypatch.setattr(upstream, "launch_test", launch)
    old = r.sha(upstream.REPORT / "FAILURE.json"), r.sha(upstream.REPORT / "LATEST.json")
    result = r.advance(*state["args"])
    assert result["status"] == "waiting" and result["phase"] == "test_cost"
    assert actions == [("check", "5080", "fit_validation"), ("collect", "5080", "fit_validation"),
        ("test", "5080"), ("test", "5090")]
    assert len(list((r.BUNDLE / "global_validation_v1/global_validation_sources").rglob("*.json"))) == 144
    assert r.check_only(*state["args"])["status"] == "ready_to_advance"
    assert r.advance(*state["args"])["status"] == "waiting_next_check"
    assert old == (r.sha(upstream.REPORT / "FAILURE.json"), r.sha(upstream.REPORT / "LATEST.json"))


def test_partial_test_launch_failure_never_launches_again(recovery_controls, monkeypatch):
    state = recovery_controls; create_collection(state, "5080"); calls = []
    def launch(c, host, gate, folder):
        calls.append(host)
        if host == "5090":
            raise RuntimeError("launch failed")
        return fake_launch(c, host, gate, folder)
    monkeypatch.setattr(upstream, "launch_test", launch)
    first = r.advance(*state["args"])
    assert first["status"] == "failed" and calls == ["5080", "5090"]
    assert r.advance(*state["args"])["cached_failure"] is True
    assert calls == ["5080", "5090"]
    assert (upstream.REPORT / "TEST_LAUNCHED_5080.json").exists()
    assert not (upstream.REPORT / "TEST_LAUNCHED_5090.json").exists()


def test_orphan_paired_test_launch_attempts_block_before_any_remote_call(recovery_controls, monkeypatch):
    state = recovery_controls
    roots = {"5080": create_collection(state, "5080"),
        "5090": r.verified_collection(state["c"], "5090", "fit_validation")}
    gate = upstream.build_global_gate(state["c"], roots, r.BUNDLE / "global_validation_v1")
    for host in r.HOSTS:
        fake_launch(state["c"], host, gate, r.REPORT / "rounds/orphan")
    calls = []
    def checked(*args):
        calls.append(args); return {"status": "waiting"}
    monkeypatch.setattr(upstream, "short_check", checked)
    result = r.advance(*state["args"])
    assert result["status"] == "failed" and "Orphan Test launch attempt" in result["error"]
    assert calls == [] and r.advance(*state["args"])["cached_failure"] is True
    assert all((upstream.REPORT / ("TEST_LAUNCHED_" + host + ".json")).exists() for host in r.HOSTS)


@pytest.mark.parametrize("key,value", [("absolute_recovery_deadline_unix", 1), ("maximum_recovery_rounds", 1)])
def test_deadline_and_round_count_prevent_next_remote_action(recovery_controls, monkeypatch, key, value):
    state = recovery_controls
    rebind_auth(state, **{key: value})
    if key == "maximum_recovery_rounds":
        folder = r.REPORT / "rounds/prior"
        dump(folder / "INTENT.json", {"binding_sha256": state["args"][0], "check_started_unix": 0})
        r.publish_round(folder, {"status": "waiting"})
    result = r.advance(*state["args"])
    assert result["status"] == "failed" and "Approved recovery" in result["error"]
    assert r.advance(*state["args"])["cached_failure"] is True


def test_comparison_precheck_create_postcheck_is_the_original_qualified_helper(recovery_controls, monkeypatch, tmp_path):
    state = recovery_controls; commands = []
    monkeypatch.setattr(upstream, "comparator_qualified", lambda binding: True)
    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, b'{"status":"verified"}\n', b"")
    monkeypatch.setattr(upstream.subprocess, "run", run)
    result = upstream.compare({}, state["c"], tmp_path / "comparison_round")
    assert result["status"] == "complete" and len(commands) == 3
    assert ["--check-only" in command for command in commands] == [True, False, True]
    assert all(command[1:3] == ["-m", "paper.scripts.compare_shared_component_ablation"] for command in commands)
