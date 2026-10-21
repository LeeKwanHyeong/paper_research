"""Synthetic terminal archives and transports. No SSH, model or project artifact reads."""
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time
from types import SimpleNamespace

import pytest

from paper.scripts import collect_titans_mac_accuracy9_terminal_finish as m

_spec = importlib.util.spec_from_file_location("_mac9_synthetic_prior", Path(__file__).resolve().parents[2] / "tests/test_titans_mac_accuracy9_recovery_collection.py")
prior = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prior)
put, hashes = prior.put, prior.hashes


def replace_json(root, replacements):
    def transform(value):
        if isinstance(value, dict):
            return {k: transform(v) for k, v in value.items()}
        if isinstance(value, list):
            return [transform(v) for v in value]
        return replacements.get(value, value) if isinstance(value, str) else value
    for path in root.rglob("*.json"):
        if "continuation_v3" in path.parts or "original" in path.parts: continue
        put(root, str(path.relative_to(root)), transform(m.read(path)))


@pytest.fixture
def terminal(tmp_path, monkeypatch):
    old = prior.terminal.__wrapped__(tmp_path)
    c, evidence = prior.bridged.__wrapped__(old, tmp_path, monkeypatch)
    root, finish = evidence / "recovery", evidence / "finish"
    finish.mkdir()
    old_sha = m.canonical(c)
    c.update(gpu_uuid="synthetic-gpu", limits={"queue_wall_seconds": 86400}, absolute_host_deadline_unix=1890000000.)
    new_sha = m.canonical(c)
    replace_json(evidence, {old_sha: new_sha, "start": "2026-10-21T12:00:00+00:00",
        "exit": "2026-10-21T12:01:00+00:00", "bridge-start": "2026-10-21T12:00:00+00:00",
        "bridge-exit": "2026-10-21T13:01:00+00:00"})
    put(root, "recovery_contract.json", c)
    stages = m.read(root / "final_receipt.json")["stages"]
    test_bytes, test_logs = {}, {}
    for path in list(root.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root))
        if rel == "validation_gate.json" or (rel.startswith("run/") and path.name.startswith("test")):
            test_bytes[rel] = path.read_bytes(); path.unlink()
        elif rel.startswith("logs/") and ".test." in rel:
            test_logs[rel] = path.read_bytes(); path.unlink()
    failed = deepcopy(stages["validation"])
    failure = {"stage": "validation", "at": "2026-10-21T12:59:00+00:00",
        "traceback": " in initial_identity\n in _identity\nValueError: Child identity disappeared without observable OS exit\n"}
    children = {job["id"]: failed["exits"][job["id"]]["pid"] for job in c["jobs"]
                if job["recovery_mode"] == "resume" and job["seed"] in (42, 62)}
    failed.update(status="failed", failure=failure, pending_job_ids=[])
    for stage in ("reference", "train"):
        stages[stage].update(pending_job_ids=[])
    put(root, "dispatch_failure.json", {**failure, "children_left_running": children})
    put(root, "validation_stage_receipt.json", failed)
    original_final = m.read(root / "final_receipt.json")
    original_final.update(status="failed", candidate_test_count=0, held_out_test_evaluated=False,
        stages={"reference": stages["reference"], "train": stages["train"], "validation": failed})
    put(root, "final_receipt.json", original_final)
    ex = m.read(root / "dispatcher_exit.json"); ex["actual_os_returncode"] = 1; ex["cleanup_error"] = None
    put(root, "dispatcher_exit.json", ex)
    imported = m.read(root / "import_receipt.json")
    raw = root / "continuation_v3"
    # Restore byte-identical imported fit files; only current Validation/Test and
    # the new controller use the synthetic ISO timestamps.
    for job in c["jobs"]:
        item = imported["jobs"][job["id"]]
        for name in item["files"]:
            put(root, name, (raw / name).read_bytes())
        for key in ("exit", "launch"):
            entry = item[key]
            relocated = Path(entry["relocated_path"])
            if not relocated.is_absolute(): relocated = root / relocated
            current = m.read(relocated)
            original_name = entry.get("original_path")
            if original_name is None:
                raw_value = m.read(raw / "run" / job["id"] / "train_exit.json")
                value = {k: raw_value[k] for k in ("job_id", "stage", "pid", "command", "started_at", "boot_id")}
                value["start_ticks"] = current["start_ticks"]
            else:
                raw_path = Path(original_name)
                if raw_path.is_absolute(): raw_path = raw / raw_path.relative_to(c["continuation_from"]["root"])
                else: raw_path = root / raw_path if str(raw_path).startswith("continuation_v3/") else raw / raw_path
                value = m.read(raw_path)
                if entry.get("identity_supplemented"): value["start_ticks"] = current["start_ticks"]
            put(relocated.parent, relocated.name, value)
        for key in ("scientific_receipt", "recovery_receipt", "exit", "launch"):
            entry = item[key]; relocated = Path(entry["relocated_path"])
            if not relocated.is_absolute(): relocated = root / relocated
            entry["relocated_sha256"] = m.sha(relocated)
    put(root, "import_receipt.json", imported)
    for stage in ("reference", "train"):
        for key in stages[stage]["exits"]:
            stages[stage]["exits"][key] = m.read(root / "run" / key / (stage + "_exit.json"))
    for stage in ("reference", "train", "validation"):
        for key in stages[stage]["receipts"]:
            folder = root / "run" / key
            for suffix in ("_receipt.json", "_recovery_receipt.json"):
                path = folder / (stage + suffix)
                if path.exists():
                    value = m.read(path)
                    if "files" in value:
                        value["files"] = {n: m.sha(root / n) for n in value["files"]}
                        put(path.parent, path.name, value)
            binding = stages[stage]["receipts"][key]
            binding["sha256"] = m.sha(root / binding["path"])
    for item in imported["jobs"].values():
        for key in ("scientific_receipt", "recovery_receipt", "exit", "launch"):
            entry = item[key]; relocated = Path(entry["relocated_path"])
            if not relocated.is_absolute(): relocated = root / relocated
            entry["relocated_sha256"] = m.sha(relocated)
    put(root, "import_receipt.json", imported)
    failed["receipts"] = stages["validation"]["receipts"]
    put(root, "validation_stage_receipt.json", failed)
    original_final["stages"].update({k: stages[k] for k in ("reference", "train")})
    put(root, "final_receipt.json", original_final)
    before = hashes(root)
    for name, data in test_bytes.items():
        put(root, name, data)
    for key in stages["test"]["receipts"]:
        folder = root / "run" / key
        for suffix in ("_receipt.json", "_recovery_receipt.json"):
            path = folder / ("test" + suffix); value = m.read(path)
            value["files"] = {n: m.sha(root / n) for n in value["files"]}
            put(path.parent, path.name, value)
        stages["test"]["receipts"][key]["sha256"] = m.sha(folder / "test_recovery_receipt.json")
    gate = m.read(root / "validation_gate.json")
    for job in c["jobs"]:
        folder = root / "run" / job["id"]
        gate["jobs"][job["id"]] = {k: m.sha(folder / n) for k, n in (("train_exit_sha256", "train_exit.json"),
            ("validation_exit_sha256", "validation_exit.json"), ("validation_receipt_sha256", "validation_receipt.json"))}
    put(root, "validation_gate.json", gate)
    for job in c["jobs"]:
        for name in ("test_claim.json", "test_lease.json", "test.intent.json"):
            put(root, "run/" + job["id"] + "/" + name, {"synthetic": True})
    for name, data in test_logs.items():
        put(finish, name, data)
    auth = put(finish, "AUTHORIZATION.json", {"instruction": "finish fixed Test9 and collection"})
    operations = {name: m.sha(put(finish, name, b"synthetic operation")) for name in m.OPERATION_NAMES}
    f = {"schema": m.FINISH_SCHEMA, "root": str(finish), "upstream_root": c["root"],
        "upstream_contract_path": c["root"] + "/recovery_contract.json", "upstream_contract_sha256": m.canonical(c),
        "python": c["python"], "gpu_uuid": c["gpu_uuid"], "boot_id": c["boot_id"], "limits": c["limits"],
        "absolute_host_deadline_unix": c["absolute_host_deadline_unix"], "queue_started_at": ex["started_at"],
        "upstream_contract_raw_sha256": m.sha(root / "recovery_contract.json"),
        "upstream_evidence_files": {name: m.sha(root / name) for name in ("final_receipt.json", "dispatch_failure.json",
            "dispatcher_exit.json", "dispatcher_launch.json", "validation_stage_receipt.json")},
        "operation_files": operations, "authorization": {"path": str(auth), "sha256": m.sha(auth), "instruction": "finish"}}
    put(finish, "finish_contract.json", f)
    monkeypatch.setattr(m, "UPSTREAM_SHA", m.canonical(c))
    admitted = {**failed, "status": "complete", "failure": None, "imported_without_execution": True,
        "control_failure_preserved": {"path": c["root"] + "/validation_stage_receipt.json", "sha256": m.sha(root / "validation_stage_receipt.json")}}
    stages["validation"] = admitted
    admission = {"schema": m.FINISH_SCHEMA, "status": "passed", "contract_sha256": m.canonical(f),
        "upstream_contract_sha256": m.canonical(c), "preexisting_v4_files": before,
        "preexisting_v4_files_sha256": m.canonical(before), "validated_stages": {k: stages[k] for k in ("reference", "train", "validation")},
        "validation_failure_preserved": {"path": c["root"] + "/dispatch_failure.json", "sha256": m.sha(root / "dispatch_failure.json")},
        "upstream_dispatcher_exit": {"path": c["root"] + "/dispatcher_exit.json", "sha256": m.sha(root / "dispatcher_exit.json")}}
    put(finish, "admission.json", admission)
    put(finish, "test_stage_receipt.json", stages["test"])
    command = [f["python"], "-u", f["root"] + "/source/titans_mac_accuracy9_terminal_runtime.py", "--contract",
        f["root"] + "/finish_contract.json", "--contract-sha", m.canonical(f), "--mode", "dispatch"]
    identity = {"pid": 5000, "start_ticks": 5001, "argv": command, "cwd": c["source_root"], "pgid": 5000}
    put(finish, "dispatcher_launch.json", {"identity": identity, "command": command,
        "at": "2026-10-21T14:00:00+00:00", "contract_sha256": m.canonical(f)})
    put(finish, "start.json", {"identity": identity, "contract_sha256": m.canonical(f)})
    exited = {"pid": 5000, "start_ticks": 5001, "command": command, "started_at": "2026-10-21T14:00:00+00:00",
        "observed_exit_at": "2026-10-21T14:01:00+00:00", "boot_id": c["boot_id"], "actual_os_returncode": 0,
        "timeout": False, "cleanup_error": None}
    for stream in ("stdout", "stderr"):
        p = put(finish, "dispatcher." + stream + ".log", b"")
        exited[stream + "_sha256"], exited[stream + "_bytes"] = m.sha(p), 0
    put(finish, "dispatcher_exit.json", exited)
    files = {n: h for n, h in hashes(finish).items() if n not in m.LIFECYCLE_FILES}
    manifest = put(finish, "artifact_manifest.json", {"files": files, "excluded_lifecycle_files": sorted(m.LIFECYCLE_FILES)})
    original_proof = {"file_count": 1141, "files_sha256": c["immutable_originals"]["files_sha256"], "original_files_unchanged": True}
    final = {"schema": m.FINISH_SCHEMA, "status": "complete", "contract_sha256": m.canonical(f),
        "upstream_contract_sha256": m.canonical(c), "admission": {"path": "admission.json", "sha256": m.sha(finish / "admission.json")},
        "test_stage_receipt": {"path": "test_stage_receipt.json", "sha256": m.sha(finish / "test_stage_receipt.json")},
        "stages": stages, "new_training_worker_count": 0, "new_validation_worker_count": 0, "candidate_test_count": 9,
        "actual_scientific_worker_exit0_count": 27, "owned_worker_pids_absent": True, "preexisting_v4_files_unchanged": True,
        "added_v4_files": {n: h for n, h in hashes(root).items() if n not in before},
        "immutable_originals_verification": original_proof, "dispatcher_own_exit_directly_observed": False,
        "artifact_manifest": {"path": "artifact_manifest.json", "sha256": m.sha(manifest), "file_count": len(files)}}
    put(finish, "final_receipt.json", final)
    observation = m.read(evidence / "collection_observation.json")
    observation.update(schema=m.SCHEMA, contract_sha256=m.canonical(f), upstream_contract_sha256=m.canonical(c),
        gpu={"uuid": c["gpu_uuid"], "compute_pids": []})
    observation["owned_pids"] = sorted((set(observation["owned_pids"]) - {4000}) | {5000})
    observation["inherited_process_identities"] += [m.read(p / "dispatcher_launch.json")["identity"]
        for p in (root, root / "continuation_v3")]
    put(evidence, "collection_observation.json", observation)
    return f, c, evidence


def test_full_terminal_preserves_failed_history_and_all_27_scientific_exits(terminal):
    f, c, evidence = terminal
    result = m.verify_terminal_evidence(f, c, evidence)
    assert result["actual_scientific_stage_exit0_count"] == 27
    assert result["new_training_worker_count"] == result["new_validation_worker_count"] == 0
    assert result["dispatcher_exit0_count"] == 1 and result["historical_failed_dispatcher_exit1_count"] == 2
    assert m.read(evidence / "recovery/final_receipt.json")["status"] == "failed"
    assert len(m.comparison_records(evidence, c)) == 18


@pytest.mark.parametrize("target", ["recovery/source/frozen/0.py", "original/source/frozen/0.py",
    "recovery/dispatcher_exit.json", "finish/dispatcher_exit.json", "finish/source/titans_mac_accuracy9_terminal_identity.py"])
def test_raw_scientific_and_control_tampering_rejected(terminal, target):
    f, c, evidence = terminal
    (evidence / target).write_bytes(b"changed")
    with pytest.raises((ValueError, KeyError, json.JSONDecodeError)):
        m.verify_terminal_evidence(f, c, evidence)


@pytest.mark.parametrize("which", ["empty", "unapproved", "overlap"])
def test_v4_exact_added_inventory_rejects_missing_extra_or_existing_test(terminal, which):
    f, c, evidence = terminal
    p = evidence / "finish/final_receipt.json"; final = m.read(p)
    if which == "empty": final["added_v4_files"] = {}
    if which == "unapproved": final["added_v4_files"]["unapproved.py"] = "1" * 64
    if which == "overlap": final["added_v4_files"]["final_receipt.json"] = m.sha(evidence / "recovery/final_receipt.json")
    put(p.parent, p.name, final)
    with pytest.raises(ValueError): m.verify_terminal_evidence(f, c, evidence)


def test_fake_dispatcher_success_or_nonzero_scientific_exit_is_rejected(terminal):
    f, c, evidence = terminal
    final = m.read(evidence / "finish/final_receipt.json")
    key = c["test_jobs"][0]
    final["stages"]["test"]["exits"][key]["actual_os_returncode"] = 1
    put(evidence / "finish", "final_receipt.json", final)
    with pytest.raises(ValueError): m.verify_terminal_evidence(f, c, evidence)


def configure(tmp_path, monkeypatch, f, c):
    report, bundle = tmp_path / "report", tmp_path / "bundle"
    monkeypatch.setattr(m, "REPORT", report); monkeypatch.setattr(m, "BUNDLE", bundle)
    monkeypatch.setattr(m, "verify_local", lambda expected: (f, c))
    put(report, "LAUNCHED.json", {"contract_sha256": m.canonical(f)})
    return report, bundle


def test_cacheless_check_and_preserved_attempts_never_open_ssh(terminal, tmp_path, monkeypatch):
    f, c, _ = terminal; report, _ = configure(tmp_path, monkeypatch, f, c)
    monkeypatch.setattr(m, "transport", lambda *a, **k: pytest.fail("No remote call allowed"))
    assert m.advance(m.canonical(f), check_local=True)["status"] == "not_collected"
    put(report / "collection_v1", "COLLECTION_ATTEMPT.json", {"preserved": True})
    assert m.advance(m.canonical(f))["remote_calls"] == 0


def archive_bytes(evidence):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for path in sorted(evidence.rglob("*")):
            archive.add(path, arcname=str(path.relative_to(evidence)), recursive=False)
    return output.getvalue()


def fake_transport(f, c, evidence, *, terminal_status="terminal", bad_transfer=False):
    calls = []
    def transport(_f, _c, program, folder, *, label, timeout, output=None):
        folder.mkdir(parents=True); calls.append((label, timeout)); path = Path(output) if output else folder / "stdout"
        if output:
            path.write_bytes(b"partial" if bad_transfer else archive_bytes(evidence))
        else:
            put(path.parent, path.name, {"schema": m.SCHEMA, "status": terminal_status,
                "contract_sha256": m.canonical(f), "upstream_contract_sha256": m.canonical(c),
                "boot_id": c["boot_id"], "read_only_check": True, "observed_at": "observed"})
        err = put(folder, "stderr", b"")
        value = {"actual_transport_exit": 0, "transport_timeout": False, "transport_error": None,
            "stdout_path": str(path), "stdout_sha256": m.sha(path), "stdout_bytes": path.stat().st_size,
            "stderr_path": str(err), "stderr_sha256": m.sha(err), "stderr_bytes": 0}
        put(folder, "TRANSPORT.json", value)
        return value
    return calls, transport


def test_single_first_retrieval_then_ssh_free_full_cache(terminal, tmp_path, monkeypatch):
    f, c, evidence = terminal; configure(tmp_path, monkeypatch, f, c)
    calls, transfer = fake_transport(f, c, evidence)
    monkeypatch.setattr(m, "transport", transfer)
    result = m.advance(m.canonical(f))
    assert calls == [("read_only_terminal", 45), ("first_collection", 900)]
    assert result["status"] == "collection_complete"
    monkeypatch.setattr(m, "transport", lambda *a, **k: pytest.fail("Cache must be SSH0"))
    assert m.advance(m.canonical(f), check_local=True)["cache_verified"] is True
    assert m.advance(m.canonical(f))["remote_calls"] == 0
    (m.retrieval_dir() / "evidence/recovery/source/frozen/0.py").write_bytes(b"tampered")
    with pytest.raises(ValueError): m.advance(m.canonical(f), check_local=True)


def test_waiting_guard_and_terminal_failure_never_retry(terminal, tmp_path, monkeypatch):
    f, c, evidence = terminal; report, _ = configure(tmp_path, monkeypatch, f, c)
    calls, transfer = fake_transport(f, c, evidence, terminal_status="waiting")
    monkeypatch.setattr(m, "transport", transfer)
    assert m.advance(m.canonical(f))["status"] == "waiting"
    assert m.advance(m.canonical(f))["status"] == "waiting_next_check"
    assert len(calls) == 1
    (m.report_dir() / "LATEST.json").unlink()
    calls, transfer = fake_transport(f, c, evidence, terminal_status="terminal_failure")
    monkeypatch.setattr(m, "transport", transfer)
    with pytest.raises(ValueError): m.advance(m.canonical(f))
    assert (m.report_dir() / "FAILURE.json").exists()
    assert m.advance(m.canonical(f))["status"] == "blocked_preserved_failure_or_attempt"
    assert len(calls) == 1


def test_partial_archive_is_preserved_and_blocks_every_later_remote_call(terminal, tmp_path, monkeypatch):
    f, c, evidence = terminal; configure(tmp_path, monkeypatch, f, c)
    calls, transfer = fake_transport(f, c, evidence, bad_transfer=True)
    monkeypatch.setattr(m, "transport", transfer)
    with pytest.raises(tarfile.ReadError): m.advance(m.canonical(f))
    assert (m.retrieval_dir() / "archive.tar.gz").read_bytes() == b"partial"
    assert m.advance(m.canonical(f))["remote_calls"] == 0 and len(calls) == 2


@pytest.mark.parametrize("name,kind", [("../escape", "file"), ("finish/link", "symlink"), ("unexpected/file", "file")])
def test_archive_path_namespace_and_links_rejected(tmp_path, name, kind):
    p = tmp_path / "bad.tar.gz"
    with tarfile.open(p, "w:gz") as stream:
        info = tarfile.TarInfo(name)
        if kind == "symlink": info.type = tarfile.SYMTYPE; info.linkname = "elsewhere"
        else: info.size = 1
        stream.addfile(info, None if kind == "symlink" else io.BytesIO(b"x"))
    with pytest.raises(ValueError): m.safe_extract(p, tmp_path / "bad")


def test_generated_programs_have_no_fit_or_evaluation_entrypoint(terminal):
    f, c, _ = terminal
    for source in (m.terminal_program(f, c), m.collection_program(f, c)):
        compile(source, "synthetic-probe", "exec")
        assert "subprocess.Popen" not in source and "run_stage(" not in source and "terminate_owned" not in source
    assert m.CHECK_SECONDS == 45 and m.TRANSFER_SECONDS == 900


def test_pure_three_seed_comparison_full_cycle_recomputes_all_metrics_and_rejects_tamper(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("_mac9_synthetic_comparison", Path(__file__).resolve().parents[2] / "tests/test_titans_mac_accuracy9_recovery_comparison.py")
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    fixture = helper.fixture.__wrapped__(tmp_path, monkeypatch)
    f, c = {"schema": m.FINISH_SCHEMA, "synthetic": "finish"}, fixture["contract"]
    report, bundle = fixture["report"], fixture["bundle"]
    monkeypatch.setattr(m, "REPORT", report); monkeypatch.setattr(m, "BUNDLE", bundle)
    monkeypatch.setattr(m, "UPSTREAM_BUNDLE", bundle)
    monkeypatch.setattr(m, "verify_local", lambda expected: (f, c))
    monkeypatch.setattr(m, "_manager", lambda: SimpleNamespace(__file__=__file__))
    put(bundle, "finish_contract.json", f)
    collection = {"status": "collection_complete", "cache_verified": True, "remote_calls": 0,
        "archive_sha256": m.sha(fixture["retrieval"] / "archive.tar.gz"),
        "manifest_sha256": m.sha(fixture["retrieval"] / "manifest.json"),
        "receipt_sha256": m.sha(fixture["retrieval"] / "receipt.json")}
    monkeypatch.setattr(m, "advance", lambda expected, check_local=False: collection if check_local else pytest.fail("Comparison must only verify local cache"))
    monkeypatch.setattr(m, "comparison_records", helper.wrapper.collector.comparison_records)
    monkeypatch.setattr(m, "transport", lambda *a, **k: pytest.fail("Comparison must be SSH0"))
    expected = m.canonical(f)
    assert m.compare(expected, check_only=True)["status"] == "comparison_ready_outputs_not_written"
    assert m.compare(expected)["outputs_created"] is True
    folder = report / "comparison_v1"
    result = m.read(folder / m.JSON_NAME)
    assert result["seed_metric_comparison_count"] == 108 and result["aggregate_comparison_count"] == 36
    assert result["aggregates"][0]["shared_worse_seeds"] == [52]
    assert result["sample_sd_ddof"] == 1 and result["runtime_context"]["all_accuracy_endpoints_same_gpu_runtime"] is False
    assert "5080/cuDNN91900" in (folder / m.MARKDOWN_NAME).read_text()
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    assert m.compare(expected, check_only=True)["cache_verified"] is True
    assert before == {p.name: p.read_bytes() for p in folder.iterdir()}
    result["seed_differences"][0]["shared"] = 999.
    put(folder, m.JSON_NAME, result)
    manifest = m.read(folder / "MANIFEST.json"); manifest["files"][m.JSON_NAME] = m.sha(folder / m.JSON_NAME)
    put(folder, "MANIFEST.json", manifest)
    with pytest.raises(ValueError, match="recomputation"):
        m.compare(expected, check_only=True)
