"""Local manager contracts; mocked transport only, no SSH/model/data execution."""
from copy import deepcopy
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
from types import ModuleType, SimpleNamespace

import pytest

from paper.scripts import manage_titantpp_external_cap27 as m
from paper.tests.test_titantpp_external_cap27_runtime import contract as runtime_fixture, receipt, fake_queue


@pytest.fixture
def setup(tmp_path, monkeypatch):
    c = runtime_fixture.__wrapped__(tmp_path)
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "BUNDLE", tmp_path / "bundle")
    monkeypatch.setattr(m, "REPORT", tmp_path / "report")
    monkeypatch.setattr(m, "PARENT", tmp_path / "parent.json")
    monkeypatch.setattr(m, "REFERENCE_BUNDLE", tmp_path / "v1")
    monkeypatch.setattr(m, "REFERENCE_REPORT", tmp_path / "v1_report")
    for h in c["hosts"].values():
        h.update(tmux="new-session", tmux_binary="/bin/tmux")
    m.write(m.PARENT, {"hosts": c["hosts"]})
    return c


def test_host_assignment_keeps_each_dataset_seed_together():
    jobs = m.build_jobs()
    assert len(jobs) == len({j["id"] for j in jobs}) == 27
    assert sum(j["host"] == "5080" for j in jobs) == 12
    assert sum(j["host"] == "5090" for j in jobs) == 15
    for d in {j["dataset"] for j in jobs}:
        for s in (42, 52, 62):
            assert len({j["host"] for j in jobs if j["dataset"] == d and j["seed"] == s}) == 1


def test_snapshot_allows_untracked_reports_but_not_modified_python(setup, monkeypatch):
    for name in m.REQUIRED_NEW_SOURCE:
        path = m.ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# committed source\n")
    changes = [""]

    def git(command, **kwargs):
        if "diff" in command:
            return changes[0]
        if "ls-files" in command:
            return "\n".join(m.REQUIRED_NEW_SOURCE)
        return "committed-revision\n"

    monkeypatch.setattr(m.subprocess, "check_output", git)
    assert m.source_snapshot()["base_git_revision"] == "committed-revision"
    changes[0] = "paper/scripts/new.py\n"
    with pytest.raises(ValueError, match="Commit"):
        m.source_snapshot()


def reference_fixture(c):
    """Fabricated committed Cap36 metadata; no real rows are loaded."""
    files = {}
    for name in m.REQUIRED_NEW_SOURCE:
        path = m.ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# scientific source\n")
        files[name] = m.sha(path)
    previous = deepcopy(c)
    previous.update(schema="titantpp_mlp_cap36_v1",
        datasets=[{"dataset_id": d, "model": {"quantity_variant": "count_only_log_regression",
                     "lambda_log_qty": 1., "lambda_tail": 0.}, "loader": {"max_seq_len": 256},
                     "cap_q3": cap} for d, cap in
                  (("yellow_trip_hourly", 65), ("raf_spare_parts", 10), ("intermittent_frozen_5000", 20))],
        input_files={}, data_receipt={"status": "prepared", "original_data_preserved": True},
        parent_runtime_contract={"path": str(m.PARENT), "sha256": m.sha(m.PARENT)},
        source={"base_git_revision": "old-committed", "files": files, "files_sha256": m.canonical(files)})
    for dataset in previous["datasets"]:
        for filename in ("train_validation.parquet", "full_capped.parquet", "split_manifest.json"):
            name = "inputs/data/" + dataset["dataset_id"] + "/" + filename
            path = m.REFERENCE_BUNDLE / "input_snapshot" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((dataset["dataset_id"] + " Cap synthetic " + filename).encode())
            previous["input_files"][name] = m.sha(path)
    previous["training"]["monitor"] = "validation_raw_quantity_rmse"
    # Old failed-bootstrap logic may exist in the reference; it is never inherited.
    previous["hosts"]["5080"]["wait_predecessor"] = {"root": "/preserved/old", "required_outcome": "historical"}
    m.write(m.REFERENCE_BUNDLE / "execution_contract.json", previous)
    m.write(m.REFERENCE_REPORT / "PREPARED.json", {
        "contract_sha256": m.sha(m.REFERENCE_BUNDLE / "execution_contract.json"),
        "canonical_contract_sha256": m.canonical(previous)})
    reference_result = m.REFERENCE_REPORT / "THREE_SEED_TEST.json"
    m.write(reference_result, {"synthetic_completed_reference": True})
    m.write(m.REFERENCE_BUNDLE / "retrieval/final_receipt.json", {"status": "complete",
        "candidate_test_count": 36, "seed_count": 3, "held_out_test_evaluated": True,
        "report_files": {str(reference_result): m.sha(reference_result)}})
    return previous, {"base_git_revision": "new-committed", "files": files,
                      "files_sha256": m.canonical(files)}


def test_prepare_reuses_exact_cap36_nine_inputs_budget_and_is_exclusive(setup, monkeypatch):
    previous, source = reference_fixture(setup)
    monkeypatch.setattr(m, "source_snapshot", lambda: source)
    preserved = {str(p): p.read_bytes() for directory in (m.REFERENCE_BUNDLE, m.REFERENCE_REPORT)
                 for p in directory.rglob("*") if p.is_file()}
    result = m.prepare()
    assert result["planned_fit_count"] == 27
    assert result["host_fit_counts"] == {"5080": 12, "5090": 15}
    c = m.verify_local(archive=True)
    for key in ("datasets", "input_files", "data_receipt", "training", "limits"):
        assert c[key] == previous[key]
    assert len(c["input_files"]) == 9
    assert set(c["architecture"]["arms"]) == {"rmtpp", "thp", "s2p2_matched_head"}
    assert c["architecture"]["model_architecture_changed"] is False
    assert c["architecture"]["native_paper_heads_used"] is False
    assert "same count-only log regression" in c["architecture"]["shared_loss_head"]
    assert c["stored_cap_mlp_reference"]["retrained"] is False
    assert c["stored_cap_mlp_reference"]["reevaluated"] is False
    assert c["execution"]["recurring_automation"] is False
    assert c["historical_cnn_retrained"] is c["historical_cnn_reevaluated"] is False
    assert all("wait_predecessor" not in h for h in c["hosts"].values())
    assert all("external_cap27" in h["root"] for h in c["hosts"].values())
    for name, digest in c["input_files"].items():
        assert m.sha(m.BUNDLE / "input_snapshot" / name) == digest
    assert all(Path(path).read_bytes() == raw for path, raw in preserved.items())
    with tarfile.open(m.BUNDLE / "deployment.tar.gz") as archive:
        assert set(archive.getnames()) == {"execution_contract.json", "source/sample_data",
            *("source/" + name for name in source["files"]), *("source/" + name for name in c["input_files"])}
        assert archive.getmember("source/sample_data").isdir()
    with pytest.raises(ValueError, match="Existing"):
        m.prepare()
    (m.BUNDLE / "deployment.tar.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="archive"):
        m.verify_local(archive=True)


def test_reference_contract_rejects_incomplete_or_changed_cap_result(setup):
    reference_fixture(setup)
    previous, binding = m.reference_contract()
    assert binding["fit_count"] == 36 and previous["schema"] == "titantpp_mlp_cap36_v1"
    result = m.REFERENCE_REPORT / "THREE_SEED_TEST.json"
    result.write_text("changed")
    with pytest.raises(ValueError, match="report changed"):
        m.reference_contract()


def test_input_corruption_preserves_old_data_and_partial_attempt_without_retry(setup, monkeypatch):
    previous, source = reference_fixture(setup)
    monkeypatch.setattr(m, "source_snapshot", lambda: source)
    name = next(iter(previous["input_files"]))
    original = m.REFERENCE_BUNDLE / "input_snapshot" / name
    original.write_bytes(b"corrupted source preserved")
    with pytest.raises(ValueError, match="Cap input SHA"):
        m.prepare()
    assert original.read_bytes() == b"corrupted source preserved"
    with pytest.raises(ValueError, match="Existing"):
        m.prepare()


@pytest.mark.parametrize("host,count,mode", [("5080", 3, "dispatch"), ("5090", 3, "dispatch")])
def test_deploy_one_install_and_launch_with_correct_wait_policy(setup, monkeypatch, host, count, mode):
    m.BUNDLE.mkdir()
    (m.BUNDLE / "deployment.tar.gz").write_bytes(b"snapshot")
    m.write(m.BUNDLE / "execution_contract.json", setup)
    monkeypatch.setattr(m, "verify_local", lambda **kwargs: setup)
    calls = []

    def ssh(host, command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=b"{}", stderr=b"")

    monkeypatch.setattr(m, "ssh", ssh)
    result = m.deploy(host)
    assert result["mode"] == mode and len(calls) == count
    shell = shlex.split(calls[-1])[-1]
    words = shlex.split(shell)
    supervisor = words[words.index("-c") + 1]
    compile(supervisor, "supervisor", "exec")
    assert repr(mode) in supervisor and "actual_os_returncode" in supervisor
    # Run the generated supervisor around a mock child; inspect actual file bytes.
    monkeypatch.setattr(m.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(pid=123, wait=lambda: 1))
    with pytest.raises(SystemExit) as exit_info:
        exec(compile(supervisor, "supervisor", "exec"), {})
    assert exit_info.value.code == 1
    output = Path(setup["hosts"][host]["root"]) / "dispatcher_exit.json"
    raw = output.read_bytes()
    assert raw.endswith(b"\n") and not raw.endswith(b"\\n")
    assert json.loads(raw)["actual_os_returncode"] == 1
    assert "--mode qualify" in calls[1]
    assert sum("--mode qualify" in command for command in calls) == 1
    assert not any("wait_dispatch" in command for command in calls)
    with pytest.raises(FileExistsError):
        m.deploy(host)
    assert len(calls) == count


def test_extracted_package_resolves_root_without_repository_ancestor(setup):
    actual_pathing = Path(__file__).resolve().parents[2] / "simple_lab_test/common/pathing.py"
    names = ("models/example.py", "utils/example.py", "simple_lab_test/common/pathing.py")
    files = {}
    for name in names:
        path = m.ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(actual_pathing.read_bytes() if name.endswith("pathing.py") else b"# synthetic source\n")
        files[name] = m.sha(path)
    m.BUNDLE.mkdir()
    m.write(m.BUNDLE / "execution_contract.json", {"synthetic_fixture": True})
    archive = m.package_archive({"input_files": {}}, {"files": files})
    extracted = m.ROOT / "isolated_extract"
    m.safe_extract(archive, extracted)
    source = extracted / "source"
    assert list((source / "sample_data").iterdir()) == []
    code = ("import importlib.util,pathlib; "
        f"p=pathlib.Path({str(source / 'simple_lab_test/common/pathing.py')!r}); "
        "s=importlib.util.spec_from_file_location('isolated_pathing',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        f"assert m.resolve_project_root(p)==pathlib.Path({str(source)!r}); print('isolated_root_ok')")
    good = subprocess.run([sys.executable, "-I", "-c", code], cwd=m.ROOT, capture_output=True, text=True)
    assert good.returncode == 0 and good.stdout.strip() == "isolated_root_ok", good.stderr
    (source / "sample_data").rmdir()
    missing = subprocess.run([sys.executable, "-I", "-c", code], cwd=m.ROOT, capture_output=True, text=True)
    assert missing.returncode != 0 and "Could not locate" in missing.stderr


def test_failed_transport_keeps_original_bytes(setup):
    result = SimpleNamespace(returncode=3, stdout=b"partial output", stderr=b"original failure")
    with pytest.raises(ValueError, match="preserved"):
        m.save_transport("5080", "install", result)
    records = list((m.BUNDLE / "transport/5080").glob("*.json"))
    value = m.read(records[0])
    assert value["actual_transport_exit"] == 3
    assert records[0].with_suffix(".stderr").read_bytes() == b"original failure"


def test_timeout_preserves_partial_bytes_without_fabricating_remote_exit(setup, monkeypatch):
    def timeout(*args, **kwargs):
        raise m.subprocess.TimeoutExpired(args[0], 1, output=b"partial", stderr=b"diagnostic")

    monkeypatch.setattr(m.subprocess, "run", timeout)
    result = m.ssh("5090", "read only", timeout=1)
    assert result.returncode is None and result.timed_out is True
    with pytest.raises(ValueError):
        m.save_transport("5090", "status", result)
    path = next((m.BUNDLE / "transport/5090").glob("*.json"))
    assert m.read(path)["actual_transport_exit"] is None and m.read(path)["transport_timeout"] is True
    assert path.with_suffix(".stdout").read_bytes() == b"partial"


@pytest.mark.parametrize("kind", ["escape", "absolute", "link", "duplicate"])
def test_archive_rejects_unsafe_or_duplicate_members(tmp_path, kind):
    path = tmp_path / "archive.tar"
    with tarfile.open(path, "w") as archive:
        name = "../escape" if kind == "escape" else "/absolute" if kind == "absolute" else "safe"
        item = tarfile.TarInfo(name)
        if kind == "link":
            item.type, item.linkname = tarfile.SYMTYPE, "/elsewhere"
            archive.addfile(item)
        else:
            item.size = 1
            archive.addfile(item, io.BytesIO(b"x"))
            if kind == "duplicate":
                archive.addfile(item, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="archive"):
        m.safe_extract(path, tmp_path / "originals")


def test_collection_program_syntax_and_single_terminal_guard(setup):
    program = m.collection_program(setup, "5080")
    compile(program, "remote_collection", "exec")
    assert "gpu_snapshot" in program and "actual_os_returncode" in program
    assert "proc_identity" in program and "mode='w|gz'" in program


def test_collected_result_rechecks_all_scientific_sha(setup, monkeypatch):
    c = setup
    host = "5080"
    target, root = c["hosts"][host], Path(c["hosts"][host]["root"])
    m.BUNDLE.mkdir()
    m.write(m.BUNDLE / "execution_contract.json", c)
    m.write(root / "execution_contract.json", c)
    m.write(root / "qualification.json", {"status": "passed", "host": host, "contract_sha256": m.canonical(c)})
    monkeypatch.chdir(target["source_root"])
    fake_queue(monkeypatch, c)
    (root / "logs").rmdir()
    assert m.runtime.dispatch(c, host, root / "execution_contract.json") == 0
    m.write(root / "dispatcher_exit.json", {"actual_os_returncode": 0})
    m.write(root / "collection_observation.json", {"owned_pids_absent": True, "gpu": {"compute_pids": []}})
    assert m.verify_collected(c, host, root)["test_count"] == 12
    job = m.runtime.validate_contract(c, host)[0]
    (root / "run" / job["id"] / "test.txt").write_text("changed after collection")
    with pytest.raises(ValueError, match="SHA"):
        m.verify_collected(c, host, root)


def test_status_uses_one_connection_and_records_raw_payload(setup, monkeypatch):
    monkeypatch.setattr(m, "verify_local", lambda **kwargs: setup)
    calls = []
    payload = {"host": "5090", "contract_sha256": m.canonical(setup), "jobs": []}

    def ssh(host, command, **kwargs):
        calls.append((host, command))
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload).encode(), stderr=b"")

    monkeypatch.setattr(m, "ssh", ssh)
    observed = m.status("5090")
    assert len(calls) == 1 and m.read(observed["observation_path"]) == payload
    assert m.sha(observed["observation_path"]) == observed["observation_sha256"]


def test_source_snapshot_excludes_untracked_reports_and_untracked_python(setup, monkeypatch):
    for name in m.REQUIRED_NEW_SOURCE:
        path = m.ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# tracked source\n")
    unrelated = m.ROOT / "reports/other_session/private_analysis.py"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("# preserve unrelated draft\n")
    untracked = m.ROOT / "paper/scripts/untracked_other_session.py"
    untracked.write_text("# untracked must not enter science snapshot\n")
    before = unrelated.read_bytes(), untracked.read_bytes()
    calls = []

    def git(command, **kwargs):
        calls.append(command)
        if "diff" in command:
            return ""
        if "ls-files" in command:
            return "\n".join(m.REQUIRED_NEW_SOURCE)
        return "committed-revision\n"

    monkeypatch.setattr(m.subprocess, "check_output", git)
    source = m.source_snapshot()
    assert set(source["files"]) == set(m.REQUIRED_NEW_SOURCE)
    assert (unrelated.read_bytes(), untracked.read_bytes()) == before
    assert all("reports" not in command for command in calls)


@pytest.mark.parametrize("field,value", [("status", "partial"), ("candidate_test_count", 35),
                                         ("seed_count", 1), ("held_out_test_evaluated", False)])
def test_incomplete_cap_reference_is_not_a_launch_basis(setup, field, value):
    reference_fixture(setup)
    path = m.REFERENCE_BUNDLE / "retrieval/final_receipt.json"
    original = m.read(path)
    path.write_text(json.dumps({**original, field: value}))
    with pytest.raises(ValueError, match="reference is incomplete"):
        m.reference_contract()
    assert not m.BUNDLE.exists()


@pytest.mark.parametrize("failure_phase,index", [("install", 0), ("qualify", 1), ("launch", 2)])
def test_deploy_failure_keeps_transport_and_never_retries_or_advances(setup, monkeypatch, failure_phase, index):
    m.BUNDLE.mkdir()
    (m.BUNDLE / "deployment.tar.gz").write_bytes(b"snapshot")
    m.write(m.BUNDLE / "execution_contract.json", setup)
    monkeypatch.setattr(m, "verify_local", lambda **kwargs: setup)
    calls = []

    def ssh(host, command, **kwargs):
        number = len(calls)
        calls.append(command)
        return SimpleNamespace(returncode=2 if number == index else 0,
                               stdout=b"raw output", stderr=b"phase failure" if number == index else b"")

    monkeypatch.setattr(m, "ssh", ssh)
    with pytest.raises(ValueError, match="raw transport preserved"):
        m.deploy("5090")
    assert len(calls) == index + 1
    failed = next((m.BUNDLE / "transport/5090").glob("*" + failure_phase + ".json"))
    assert m.read(failed)["actual_transport_exit"] == 2
    assert failed.with_suffix(".stdout").read_bytes() == b"raw output"
    assert failed.with_suffix(".stderr").read_bytes() == b"phase failure"
    assert not (m.REPORT / "LAUNCHED_5090.json").exists()
    with pytest.raises(FileExistsError):
        m.deploy("5090")
    assert len(calls) == index + 1
