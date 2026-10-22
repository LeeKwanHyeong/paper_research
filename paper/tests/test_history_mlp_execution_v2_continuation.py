"""Local synthetic admission, archive and interruption tests; no SSH or GPU."""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from importlib.util import module_from_spec, spec_from_file_location
from types import SimpleNamespace


def load_versioned(name):
    path = (
        Path(__file__).resolve().parents[2]
        / "reports/shared_history_mlp_control_20261022_v1/execution_v2"
        / (name + ".py")
    )
    spec = spec_from_file_location("test_history_mlp_execution_v2_" + name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


a = load_versioned("advance")
comparison = load_versioned("compare")


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def identity(command, pid=100):
    return {
        "pid": pid,
        "start_ticks": 1000 + pid,
        "pgid": pid,
        "argv": command,
        "cwd": "/isolated/source",
    }


@pytest.fixture
def controls(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    bundle = root / "bundle"
    report_parent = root / "report"
    report = report_parent / "continuation_v1"
    for path in (root, bundle, report):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(a, "ROOT", root)
    monkeypatch.setattr(a, "BUNDLE", bundle)
    monkeypatch.setattr(a, "REPORT", report)
    fake_manager = SimpleNamespace(
        **{
            key: getattr(a.manager, key)
            for key in ("NAME", "below", "verify_phase_receipt", "verify_global_gate")
        }
    )
    fake_manager.REPORT = report_parent
    fake_manager.validate = lambda _c: None
    monkeypatch.setattr(a, "manager", fake_manager)
    monkeypatch.setattr(a, "EXEC_REPORT", report_parent / "execution_v2")
    a.EXEC_REPORT.mkdir(parents=True, exist_ok=True)
    spec = dump(report_parent / "SCIENCE_SPEC.json", {"status": "locked"})
    source = root / "source.py"
    source.write_text("# frozen scientific source\n")
    files = {}
    for name in (a.CONTROLLER, a.COMPARATOR, "paper/tests/new_test.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        files[name] = a.sha(path)
    input_path = bundle / "input_snapshot/input.bin"
    input_path.parent.mkdir(parents=True)
    input_path.write_bytes(b"frozen input")
    legacy = root / "parent/checkpoint.pt"
    legacy.parent.mkdir()
    legacy.write_bytes(b"original")
    preservation = dump(
        report_parent / "PRESERVATION.json",
        {
            "status": "frozen",
            "files": {
                str(legacy): {"sha256": a.sha(legacy), "bytes": legacy.stat().st_size}
            },
        },
    )
    qa = dump(
        a.EXEC_REPORT / "INDEPENDENT_QA.json",
        {
            "status": "PASS",
            "independent_reviewer": True,
            "files_reviewed_sha256": files,
        },
    )
    hosts = {
        host: {
            "root": "/isolated",
            "source_root": "/isolated/source",
            "python": "/python",
            "absolute_host_deadline_unix": 4000000000.0,
            "environment": {},
            "tmux_binary": "/tmux",
            "tmux": "new",
        }
        for host in a.HOSTS
    }
    c = {
        "execution_version": 2,
        "design_sha256": a.sha(spec),
        "hosts": hosts,
        "source": {
            "files": {"source.py": a.sha(source)},
            "files_sha256": a.canonical({"source.py": a.sha(source)}),
        },
        "input_files": {"input.bin": a.sha(input_path)},
        "legacy_files": {},
        "science_spec_sha256": a.sha(spec),
        "jobs": [],
    }
    contract = dump(bundle / "execution_contract.json", c)
    archive = bundle / "deployment.tar.gz"
    archive.write_bytes(b"frozen deployment")
    integrity = dump(
        bundle / "integrity.json",
        {
            "status": "sealed",
            "contract_sha256": a.sha(contract),
            "canonical_contract_sha256": a.canonical(c),
            "archive_sha256": a.sha(archive),
            "source_files_sha256": c["source"]["files_sha256"],
        },
    )
    launch = {"contract_sha256": a.canonical(c), "hosts": {}}
    for host in a.HOSTS:
        command = a.expected_command(c, host, "fit_validation")
        launch["hosts"][host] = {
            "phase": "fit_validation",
            "identity": identity(command),
            "supervisor_identity": identity(
                ["supervise" if x == "dispatch" else x for x in command], 101
            ),
        }
    launched = dump(report / "LAUNCHED.json", launch)
    instruction = "검증 진행하고 실험이 필요하면 동일하게 5080/5090서버에서 진행하자"
    auth = dump(
        a.EXEC_REPORT / "AUTHORIZATION.json",
        {
            "status": "approved",
            "campaign": a.manager.NAME,
            "remote_hosts": list(a.HOSTS),
            "held_out_test_evaluation": False,
            "scientific_retry": False,
            "old_files_overwrite": False,
            "runtime_credentials_network_changes": False,
            "new_fit_count": 9,
            "existing_checkpoint_reuse": 18,
            "train_only_cost_processes": 162,
            "user_instruction": instruction,
        },
    )
    admission = dump(a.EXEC_REPORT / "ADMISSION.json", {"execution_version": 2})
    binding = {
        "campaign": a.manager.NAME,
        "execution_admission_sha256": a.sha(admission),
        "report_root": str(report),
        "bundle_root": str(bundle),
        "authorization_sha256": a.sha(auth),
        "user_instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
        "max_rounds": 96,
        "min_interval_seconds": 900,
        "files": files,
        "contract_file_sha256": a.sha(contract),
        "canonical_contract_sha256": a.canonical(c),
        "integrity_sha256": a.sha(integrity),
        "archive_sha256": a.sha(archive),
        "science_spec_sha256": a.sha(spec),
        "preservation_sha256": a.sha(preservation),
        "preservation_roots": [str(legacy.parent)],
        "independent_qa_sha256": a.sha(qa),
        "initial_launch_sha256": a.sha(launched),
    }
    bound = dump(report / "CONTROL_BINDINGS.json", binding)
    return {
        "c": c,
        "launch": launch,
        "binding": binding,
        "bound": bound,
        "authorization": auth,
        "root": root,
        "bundle": bundle,
        "report": report,
        "legacy": legacy,
        "args": (a.sha(bound), a.sha(auth)),
    }


def test_bound_control_accepts_exact_sources_and_authorization(controls):
    binding, c, launch = a.verify(*controls["args"])
    assert (
        binding == controls["binding"]
        and c == controls["c"]
        and launch == controls["launch"]
    )


@pytest.mark.parametrize(
    "target", ["source.py", a.CONTROLLER, a.COMPARATOR, "paper/tests/new_test.py"]
)
def test_bound_source_change_blocks_before_remote(controls, target):
    (controls["root"] / target).write_text("changed")
    with pytest.raises((ValueError, RuntimeError), match="Bound file changed"):
        a.verify(*controls["args"])


def test_checkpoint_preservation_is_actual_sha_and_bytes(controls):
    controls["legacy"].write_bytes(b"changed")
    with pytest.raises((ValueError, RuntimeError), match="Bound file changed"):
        a.verify(*controls["args"])


def test_new_unbound_parent_file_is_rejected(controls):
    (controls["legacy"].parent / "extra").write_text("untouched but not inventoried")
    with pytest.raises((ValueError, RuntimeError), match="Unbound original evidence"):
        a.verify(*controls["args"])


def test_launch_requires_exact_native_module_and_parent(controls):
    value = deepcopy(controls["launch"]["hosts"]["5080"])
    value["identity"]["argv"][3] = "paper.scripts.old_manager"
    with pytest.raises((ValueError, RuntimeError), match="Dispatcher command"):
        a.validate_launch(controls["c"], "5080", "fit_validation", value)


def test_budget_allows_96th_current_round_but_never_97th(controls):
    for index in range(96):
        dump(controls["report"] / "rounds" / str(index) / "INTENT.json", {})
    a.check_budget(controls["c"], controls["binding"], current_round=True)
    with pytest.raises((ValueError, RuntimeError), match="round budget"):
        a.check_budget(controls["c"], controls["binding"])
    dump(controls["report"] / "rounds/97/INTENT.json", {})
    with pytest.raises((ValueError, RuntimeError), match="round budget"):
        a.check_budget(controls["c"], controls["binding"], current_round=True)


def test_expired_absolute_deadline_blocks(controls):
    controls["c"]["hosts"]["5080"]["absolute_host_deadline_unix"] = 1
    with pytest.raises((ValueError, RuntimeError), match="deadline"):
        a.check_budget(controls["c"], controls["binding"])


def empty_collections():
    return {(host, phase): None for host in a.HOSTS for phase in a.PHASES}


def test_orphan_intent_cannot_be_retried(controls):
    dump(controls["report"] / "rounds/old/INTENT.json", {})
    with pytest.raises((ValueError, RuntimeError), match="Interrupted prior round"):
        a.validate_state(controls["binding"], controls["c"], empty_collections())


def test_historical_failed_round_cannot_be_erased_by_removing_failure(controls):
    folder = controls["report"] / "rounds/old"
    dump(
        folder / "INTENT.json",
        {
            "binding_sha256": a.sha(controls["bound"]),
            "contract_sha256": a.canonical(controls["c"]),
        },
    )
    dump(folder / "ROUND.json", {"status": "failed", "round_path": str(folder)})
    with pytest.raises((ValueError, RuntimeError), match="Prior round failed"):
        a.validate_state(controls["binding"], controls["c"], empty_collections())


def test_partial_or_single_host_cost_launch_never_retries(controls):
    dump(controls["report"] / "COST_LAUNCHED_5080.json", {"phase": "cost"})
    with pytest.raises((ValueError, RuntimeError), match="Single-host"):
        a.validate_state(controls["binding"], controls["c"], empty_collections())


def test_existing_global_gate_without_launch_blocks(controls):
    (controls["bundle"] / "global_validation_v1").mkdir()
    with pytest.raises((ValueError, RuntimeError), match="Existing gate"):
        a.validate_state(controls["binding"], controls["c"], empty_collections())


def test_partial_comparison_blocks(controls):
    (controls["bundle"] / "comparison_v1").mkdir()
    with pytest.raises((ValueError, RuntimeError), match="partial comparison"):
        a.validate_state(controls["binding"], controls["c"], empty_collections())


def test_failure_cache_is_sealed_and_uses_zero_remote(controls, monkeypatch):
    value = a.preserve_failure(
        RuntimeError("transport refused"), context={"failure_phase": "terminal_check"}
    )
    monkeypatch.setattr(
        a, "transport", lambda *_args, **_kwargs: pytest.fail("remote called")
    )
    cached = a.advance(*controls["args"])
    assert cached["cached_failure"] is True and cached["remote_calls"] == 0
    assert (
        cached["error"] == value["error"]
        and cached["training_failure_inferred_from_transport"] is False
    )


def test_deleted_failure_cannot_reactivate_remote(controls):
    a.preserve_failure(ValueError("saved failure"))
    (controls["report"] / "FAILURE.json").unlink()
    with pytest.raises((ValueError, RuntimeError), match="failure was removed"):
        a.cached_failure()


def test_busy_and_cooldown_do_not_create_round_or_remote(controls, monkeypatch):
    @contextmanager
    def busy():
        yield False

    original = a.advance_lock
    monkeypatch.setattr(a, "advance_lock", busy)
    assert a.advance(*controls["args"])["status"] == "busy"
    monkeypatch.setattr(a, "advance_lock", original)
    folder = controls["report"] / "rounds/previous"
    dump(
        folder / "INTENT.json",
        {
            "check_started_unix": a.time.time(),
            "binding_sha256": controls["args"][0],
            "contract_sha256": a.canonical(controls["c"]),
        },
    )
    a.publish_round(folder, {"status": "waiting"})
    monkeypatch.setattr(
        a, "transport", lambda *_args, **_kwargs: pytest.fail("remote called")
    )
    assert a.advance(*controls["args"])["status"] == "waiting_next_check"
    assert len(list((controls["report"] / "rounds").iterdir())) == 1


def test_waiting_checks_each_host_once_with_no_launch_or_collection(
    controls, monkeypatch
):
    calls = []
    monkeypatch.setattr(a, "boundary", lambda *_args: None)
    monkeypatch.setattr(
        a,
        "short_check",
        lambda _c, host, phase, _launch, _folder: (
            calls.append((host, phase)) or {"status": "waiting"}
        ),
    )
    monkeypatch.setattr(
        a,
        "collect",
        lambda *_args, **_kwargs: pytest.fail("collection called while waiting"),
    )
    monkeypatch.setattr(
        a, "launch_cost", lambda *_args: pytest.fail("cost launched before global gate")
    )
    result = a.advance(*controls["args"])
    assert result["status"] == "waiting" and calls == [
        ("5080", "fit_validation"),
        ("5090", "fit_validation"),
    ]


def test_first_transport_failure_preserves_error_and_prevents_second_host(
    controls, monkeypatch
):
    calls = []
    monkeypatch.setattr(a, "boundary", lambda *_args: None)

    def fail(_c, host, *_args):
        calls.append(host)
        raise TimeoutError("actual 45-second check timeout")

    monkeypatch.setattr(a, "short_check", fail)
    result = a.advance(*controls["args"])
    assert result["status"] == "failed" and calls == ["5080"]
    assert (
        result["failure_phase"] == "terminal_check" and result["failure_host"] == "5080"
    )
    assert a.advance(*controls["args"])["cached_failure"] is True and calls == ["5080"]


def test_complete_cache_never_creates_round_or_remote(controls, monkeypatch):
    dump(controls["report"] / "COMPLETE.json", {"status": "complete"})
    monkeypatch.setattr(a, "validate_state", lambda *_args: {})
    monkeypatch.setattr(
        a,
        "verify_completion",
        lambda *_args: {
            "status": "complete",
            "cache_verified": True,
            "remote_calls": 0,
        },
    )
    monkeypatch.setattr(
        a, "transport", lambda *_args, **_kwargs: pytest.fail("remote called")
    )
    result = a.advance(*controls["args"])
    assert result["cache_verified"] and not (controls["report"] / "rounds").exists()


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "a//b", "a/./b", "", "a/../b"]
)
def test_archive_noncanonical_paths_rejected(name):
    with pytest.raises((ValueError, RuntimeError), match="Unsafe"):
        a.safe_member(name)


def archive_fixture(tmp_path, monkeypatch):
    folder = tmp_path / "collection"
    folder.mkdir()
    original = folder / "originals"
    original.mkdir()
    c = {
        "source": {"files": {"source.py": "pinned"}},
        "input_files": {"data": "pinned"},
        "legacy_files": {},
    }
    phase = "fit_validation"
    terminal = {"actual_os_returncode": 0}
    for stream in ("stdout", "stderr"):
        p = original / (phase + "." + stream + ".log")
        p.write_bytes(b"{}\n" if stream == "stdout" else b"")
        terminal[stream + "_sha256"] = a.sha(p)
    dump(original / (phase + ".complete.json"), {"status": "complete"})
    dump(original / (phase + ".supervisor_exit.json"), terminal)
    inventory = {
        str(p.relative_to(original)): {"bytes": p.stat().st_size, "sha256": a.sha(p)}
        for p in original.iterdir()
    }
    manifest = {
        "host": "5080",
        "phase": phase,
        "contract_sha256": a.canonical(c),
        "files": inventory,
        "source_files": c["source"]["files"],
        "input_files": c["input_files"],
        "legacy_files": {},
        "all_source_input_legacy_actual_sha_verified": True,
    }
    manifest_path = dump(
        original / ("collection_" + phase + ".manifest.json"), manifest
    )
    archive = folder / ("collection_" + phase + ".tar.gz")
    with tarfile.open(archive, "w:gz") as packed:
        for path in original.iterdir():
            packed.add(path, arcname=path.name)
    receipt = {
        "status": "complete",
        "host": "5080",
        "phase": phase,
        "contract_sha256": a.canonical(c),
        "archive_sha256": a.sha(archive),
        "archive_bytes": archive.stat().st_size,
        "manifest_sha256": a.sha(manifest_path),
        "files": len(inventory),
    }
    dump(folder / "collection_receipt.json", receipt)
    for stem in ("pack", "transfer"):
        (folder / (stem + ".stdout")).write_bytes(
            (json.dumps(receipt) + "\n").encode() if stem == "pack" else b""
        )
        (folder / (stem + ".stderr")).write_bytes(b"")
        record = {"actual_os_returncode": 0, "timeout": False, "transport_error": None}
        for stream in ("stdout", "stderr"):
            p = folder / (stem + "." + stream)
            record.update(
                {stream + "_sha256": a.sha(p), stream + "_bytes": p.stat().st_size}
            )
        dump(folder / (stem + ".json"), record)
        if stem == "pack":
            dump(folder / "transport.json", record)
    monkeypatch.setattr(
        a.manager, "verify_phase_receipt", lambda *_args: ({}, terminal)
    )
    return folder, c


def test_archive_validates_originals_bytes_member_sha_and_actual_transport(
    tmp_path, monkeypatch
):
    folder, c = archive_fixture(tmp_path, monkeypatch)
    assert (
        a.validate_collection(c, "5080", "fit_validation", folder)
        == folder / "originals"
    )
    (folder / "originals/fit_validation.complete.json").write_text("changed")
    with pytest.raises((ValueError, RuntimeError), match="Extracted file"):
        a.validate_collection(c, "5080", "fit_validation", folder)


def test_archive_transport_byte_mutation_is_rejected(tmp_path, monkeypatch):
    folder, c = archive_fixture(tmp_path, monkeypatch)
    record = a.read(folder / "transfer.json")
    record["stdout_bytes"] = 100
    dump(folder / "transfer.json", record)
    with pytest.raises((ValueError, RuntimeError), match="stream bytes"):
        a.validate_collection(c, "5080", "fit_validation", folder)


def test_archive_actual_pack_transport_copy_is_bound(tmp_path, monkeypatch):
    folder, c = archive_fixture(tmp_path, monkeypatch)
    record = a.read(folder / "transport.json")
    record["unexpected"] = True
    dump(folder / "transport.json", record)
    with pytest.raises((ValueError, RuntimeError), match="transport copy"):
        a.validate_collection(c, "5080", "fit_validation", folder)


def test_symlinked_or_partial_collection_never_retries(controls):
    path = controls["bundle"] / "retrieval/fit_validation/5080"
    path.mkdir(parents=True)
    with pytest.raises((ValueError, RuntimeError), match="Partial collection"):
        a.collected(controls["c"], "5080", "fit_validation")


def test_cli_has_no_training_or_heldout_mode():
    with pytest.raises(SystemExit):
        a.main(["--mode", "test", "--binding-sha", "x", "--authorization-sha", "y"])


def test_versioned_paths_never_change_frozen_module_globals():
    from paper.scripts import advance_shared_history_mlp_control as original
    from paper.scripts import compare_shared_history_mlp_control as original_compare
    from paper.scripts import manage_shared_history_mlp_control as original_manager

    before = (original.ROOT, original.BUNDLE, original.REPORT, original_manager.BUNDLE)
    assert a.BUNDLE == original.BUNDLE / "execution_v2"
    assert comparison.BUNDLE == original_manager.BUNDLE / "execution_v2"
    assert a.REPORT == original_manager.REPORT / "execution_v2/continuation_v1"
    assert a.CONTROLLER.endswith("execution_v2/advance.py")
    assert a.COMPARATOR.endswith("execution_v2/compare.py")
    assert comparison.frozen_aggregate is original_compare.aggregate
    assert before == (
        original.ROOT,
        original.BUNDLE,
        original.REPORT,
        original_manager.BUNDLE,
    )


def test_v2_admission_pin_is_checked_at_each_boundary(controls):
    (a.EXEC_REPORT / "ADMISSION.json").write_text("changed")
    with pytest.raises((RuntimeError, ValueError), match="Bound file changed"):
        a.verify(*controls["args"])


def test_new_controller_calls_own_comparator_file_without_original_cli(
    controls, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        a,
        "comparator_module",
        lambda: SimpleNamespace(SCHEMA="shared_history_mlp_control_comparison_v1"),
    )
    monkeypatch.setattr(
        a, "transport", lambda command, *_args, **_kwargs: calls.append(command)
    )
    monkeypatch.setattr(a, "require_outputs", lambda: None)
    a.compare(controls["binding"], controls["c"], controls["report"] / "round")
    assert len(calls) == 3
    assert all(command[1] == str(controls["root"] / a.COMPARATOR) for command in calls)
    assert all("-m" not in command for command in calls)
    assert [command[-1] == "--check-only" for command in calls] == [True, False, True]


@pytest.mark.parametrize(
    "changes",
    [{"execution_version": 1}, {"design_sha256": "0" * 64}, {"design_sha256": None}],
)
def test_comparator_rejects_wrong_execution_or_design(changes):
    value = {
        "execution_version": 2,
        "design_sha256": "a" * 64,
        "science_spec_sha256": "a" * 64,
        **changes,
    }
    with pytest.raises((RuntimeError, ValueError), match="design metadata"):
        comparison.validate_version(value)


def test_adapter_pure_aggregate_uses_explicit_new_root_and_keeps_frozen_result_unmodified(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(comparison, "BUNDLE", tmp_path)
    contract = {
        "execution_version": 2,
        "design_sha256": "a" * 64,
        "science_spec_sha256": "a" * 64,
    }
    calls = []
    frozen_result = {"status": "complete", "records": ["original marker"]}
    monkeypatch.setattr(
        comparison,
        "frozen_aggregate",
        lambda c, root: calls.append((c, root)) or frozen_result,
    )
    result = comparison.aggregate(contract, tmp_path / "retrieval")
    assert result["execution_version"] == 2 and result["design_sha256"] == "a" * 64
    assert calls == [(contract, tmp_path / "retrieval")]
    assert "execution_version" not in frozen_result
    with pytest.raises((RuntimeError, ValueError), match="version 2 retrieval"):
        comparison.aggregate(contract, tmp_path.parent / "old-retrieval")


def test_adapter_cli_is_fixed_to_new_paths_and_exclusive_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, "BUNDLE", tmp_path)
    contract = {
        "execution_version": 2,
        "design_sha256": "a" * 64,
        "science_spec_sha256": "a" * 64,
    }
    path = dump(tmp_path / "execution_contract.json", contract)
    result = {
        "mean_definition": "synthetic fixture",
        "architecture_caveat": "input and total parameter matched",
        "cost_definition": "fresh Train costs",
        "table": [],
        "adverse_differences": [],
        "cost_summaries": [],
        "fit_records": [],
    }
    calls = []
    monkeypatch.setattr(
        comparison,
        "frozen_aggregate",
        lambda c, root: calls.append((c, root)) or result,
    )
    args = [
        "--contract",
        str(path),
        "--contract-sha",
        comparison.sha(path),
        "--retrieval",
        str(tmp_path / "retrieval"),
        "--output",
        str(tmp_path / "comparison_v1"),
    ]
    assert comparison.main(args + ["--check-only"]) == 0
    assert not (tmp_path / "comparison_v1").exists()
    assert comparison.main(args) == 0
    assert comparison.main(args + ["--check-only"]) == 0 and len(calls) == 3
    with pytest.raises(FileExistsError):
        comparison.main(args)
    old_args = list(args)
    old_args[-1] = str(tmp_path.parent / "comparison_v1")
    with pytest.raises((RuntimeError, ValueError), match="version 2 fixed"):
        comparison.main(old_args + ["--check-only"])
