"""One finite fixed-input backbone follow-up directly after loss27 completes.

The two superseded/deferred local waits stay SIGSTOP'ed. Current loss training
and its finite collector are read-only dependencies, never restarted or killed.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

_PREVIOUS = Path(__file__).with_name("follow_titantpp_late_path18.py")
_PREVIOUS_SHA = "44cf935074c21951fc7e123aef39124aa4d49458e85e94a8693efccfd968a5cb"
if hashlib.sha256(_PREVIOUS.read_bytes()).hexdigest() != _PREVIOUS_SHA:
    raise ValueError("Preserved late-path follow-up changed")
_spec = importlib.util.spec_from_file_location("paper.scripts._late_path18_v2_follow_private", _PREVIOUS)
_adapter = importlib.util.module_from_spec(_spec)
_source = _PREVIOUS.read_text()
_REPLACEMENTS = {
    'NAME = "titantpp_mlp16_late_path18_20261019_v1"': 'NAME = "titantpp_mlp16_late_path18_20261019_v2"',
    'PARENT_REPORT = ROOT / "reports/titantpp_raw_time_mlp27_20261019_v1"':
        'PARENT_REPORT = ROOT / "reports/titantpp_quantity_loss_mlp18_20261019_v2"',
    'from paper.scripts import manage_titantpp_late_path18 as manager':
        'from paper.scripts import manage_titantpp_late_path18_v2 as manager',
    'from paper.scripts.select_compare_titantpp_late_path18 import main as compare':
        'from paper.scripts.select_compare_titantpp_late_path18_v2 import main as compare',
    '"from paper.scripts import titantpp_late_path18_runtime as r"':
        '"from paper.scripts import titantpp_late_path18_v2_runtime as r"',
}
_EXPECTED_COUNTS = [1, 1, 2, 1, 2]
for (_before, _after), _count in zip(_REPLACEMENTS.items(), _EXPECTED_COUNTS):
    if _source.count(_before) != _count:
        raise ValueError("Unexpected preserved follow-up adaptation: " + _before)
    _source = _source.replace(_before, _after)
exec(compile(_source, str(_PREVIOUS), "exec"), _adapter.__dict__)
_adapter.__file__ = __file__
ROOT, REPORT, READY, PARENT_REPORT = _adapter.ROOT, _adapter.REPORT, _adapter.READY, _adapter.PARENT_REPORT
require, read, sha, write, now = _adapter.require, _adapter.read, _adapter.sha, _adapter.write, _adapter.now
PARENT_CONTRACT = ROOT / "search_artifacts/titantpp_quantity_loss_mlp18_20261019_v2/execution_contract.json"
PARENT_CONTRACT_SHA = "bca53c976981c168bd0c7850c89a178e154e8e851498d1a2646756febd36441b"
HOLD_RECORD = REPORT / "DEFERRED_FOLLOWUPS.json"


def verify_ready(path=READY):
    ready = read(path)
    require(ready["status"] == "locally_verified_waiting_for_loss_comparison"
        and ready["new_fit_count"] == 18 and ready["baseline_reference_count"] == 9
        and ready["held_out_test_authorized"] is False
        and ready["time_input"] == ready["quantity_input"] == "log1p"
        and ready["loss_candidate_count"] == 3 and ready["raw_time_experiment_required"] is False,
        "Fixed-input backbone scope changed")
    require(ready["files"], "Empty new preparation bindings")
    for name, digest in ready["files"].items():
        target = (ROOT / name).resolve()
        require(target.is_relative_to(ROOT) and sha(target) == digest,
                "Prepared evidence/source changed: " + name)
    require(sha(PARENT_CONTRACT) == PARENT_CONTRACT_SHA == ready["parent_contract_sha256"],
            "Running loss comparison contract changed")
    require(sha(PARENT_REPORT / "COMPLETION_STARTED.json") == ready["parent_completion_start_sha256"],
            "Existing finite loss completion identity changed")
    require(sha(HOLD_RECORD) == ready["deferred_followups_sha256"], "Deferred wait record changed")
    return ready


def verify_deferred_waits():
    """Read two known local wait identities and stopped states; send no signal."""
    receipt = read(HOLD_RECORD)
    expected = {"titantpp_raw_time_mlp27_20261019_v1", "titantpp_mlp16_late_path18_20261019_v1"}
    require(receipt["status"] == "verified_local_waits_paused"
        and receipt["signal"] == "SIGSTOP" and receipt["remote_processes_signalled"] == 0
        and len(receipt["targets"]) == 2 and {t["name"] for t in receipt["targets"]} == expected,
        "Deferred wait scope differs")
    for target in receipt["targets"]:
        old_report = ROOT / "reports" / target["name"]
        start_path = old_report / "FOLLOWUP_STARTED.json"
        require(sha(start_path) == target["start_record_sha256"], "Deferred identity record changed")
        start = read(start_path)
        require(start["pid"] == target["pid"] and target["explicit_stopped_state_verified"] is True,
                "Deferred PID changed")
        result = _adapter.subprocess.run(["ps", "-p", str(target["pid"]), "-o", "stat=",
                                         "-o", "lstart=", "-o", "command="], capture_output=True, text=True)
        fields = result.stdout.strip().split(maxsplit=1)
        require(result.returncode == 0 and len(fields) == 2 and fields[0].startswith("T")
            and _adapter.same_local_process_identity(start, fields[1]), "Deferred local wait no longer stopped")
        require(not (ROOT / "search_artifacts" / target["name"]).exists(),
                "A deferred experiment has prepared or started; inspect without retry")


def wait_for_parent(ready, output, *, sleep=_adapter.time.sleep):
    """Wait only on the current loss completion and then verify27 real endpoints."""
    verify_deferred_waits()
    start = read(PARENT_REPORT / "COMPLETION_STARTED.json")
    require(start["command"] == ["/usr/local/bin/python3", "-u", "-m",
        "paper.scripts.finish_titantpp_quantity_loss_mlp18_v2"], "Unexpected loss completion command")
    deadline = min(ready["absolute_host_deadlines"].values())
    completion = PARENT_REPORT / "completion_v1/complete.json"
    write(output / "waiting_parent.json", {"status": "waiting_for_existing_loss_comparison",
        "at": now(), "parent_pid": start["pid"],
        "parent_start_sha256": sha(PARENT_REPORT / "COMPLETION_STARTED.json"),
        "raw_time27_deferred": True, "new_remote_calls": 0,
        "new_training_started": 0, "recurring_automation": False})
    while not completion.exists():
        require(_adapter.time.time() < deadline, "Inherited deadline expired; no new work")
        require(not list((PARENT_REPORT / "completion_v1").glob("*failure.json")),
                "Existing loss completion failed; no automatic retry")
        if not _adapter.parent_process_alive(start):
            sleep(2.)
            require(completion.exists(), "Parent exited without successful completion; no retry")
            break
        sleep(min(30., max(.01, deadline - _adapter.time.time())))
    require(_adapter.time.time() < deadline, "Inherited deadline expired; no new training")
    verify_deferred_waits()
    from paper.scripts import manage_titantpp_late_path18_v2 as manager
    verified = manager.verify_parent_completion()
    require(len(verified[2]) == 27 and verified[3]["selected_condition"]["time_input"] == "log1p",
            "Only completed loss27 and fixed logged inputs are admitted")
    write(output / "parent_gate.json", {"status": "passed", "at": now(),
        "parent_completion_sha256": sha(completion), "verified_parent_endpoint_count": 27,
        "selected_condition": verified[3]["selected_condition"], "parent_reexecuted": False,
        "parent_recollected": False, "raw_time27_executed": False, "new_remote_calls": 0})


_adapter.verify_ready = verify_ready
_adapter.wait_for_parent = wait_for_parent
for _name, _value in tuple(_adapter.__dict__.items()):
    if not _name.startswith("_"):
        globals()[_name] = _value

if __name__ == "__main__":
    main()
