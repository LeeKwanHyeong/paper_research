"""Fabricated fixed-input continuation gates; no SSH, signals or training."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import paper.scripts as package
from paper.scripts import follow_titantpp_late_path18_v2 as f
from paper.scripts import manage_titantpp_late_path18_v2 as manager
from paper.scripts import compare_titantpp_quantity_loss_mlp18 as loss_comparison
from paper.scripts import select_compare_titantpp_late_path18_v2 as comparison


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def completion_start():
    return {"pid": 69408, "command": ["/usr/local/bin/python3", "-u", "-m",
        "paper.scripts.finish_titantpp_quantity_loss_mlp18_v2"], "local_process_identity":
        "Mon Oct 19 08:45:23 2026 /usr/local/bin/python3 -u -m paper.scripts.finish_titantpp_quantity_loss_mlp18_v2"}


def deferred_start(name, pid):
    module = "paper.scripts.follow_titantpp_raw_time_mlp27" if "raw_time" in name else "paper.scripts.follow_titantpp_late_path18"
    return {"pid": pid, "command": ["/usr/local/bin/python3", "-u", "-m", module, "--mode", "advance"],
        "local_process_identity": f"Mon Oct 19 09:21:28 2026 /usr/local/bin/python3 -u -m {module} --mode advance"}


def loss_records():
    rows = []
    for dataset, scale in zip(comparison.DATASETS, (100., 10., 1.)):
        for index, seed in enumerate(comparison.SEEDS, 1):
            for loss in comparison.LOSSES:
                value = scale * index * (.5 if loss == "raw_mae" else 1.)
                cohort = {"count": 100, "qty_rmse": value, "qty_mae": value / 2., "time_nll": float(index)}
                rows.append({"dataset": dataset, "seed": seed, "host": comparison._impl._expected_host(dataset, seed),
                    "loss_mode": loss, "time_alpha": 0., "time_head_trained": False, "selected_epoch": index,
                    "target_identity": {"dataset": dataset, "seed": seed},
                    "metrics": {"overall": cohort, "cap_hit": {**cohort, "count": 10},
                        "tail": {"count": 0, "qty_rmse": None, "qty_mae": None, "time_nll": None},
                        **comparison.METRIC_METADATA, "cap_value": comparison.CAP_VALUES[dataset]}})
    return rows


@contextmanager
def deferred_fixture(root):
    names = ("titantpp_raw_time_mlp27_20261019_v1", "titantpp_mlp16_late_path18_20261019_v1")
    targets, starts = [], {}
    for index, name in enumerate(names):
        start = deferred_start(name, (78887, 5949)[index])
        path = root / "reports" / name / "FOLLOWUP_STARTED.json"
        dump(path, start)
        starts[start["pid"]] = start
        targets.append({"name": name, "pid": start["pid"], "start_record_sha256": f.sha(path),
                        "explicit_stopped_state_verified": True})
    hold = root / "v2/DEFERRED_FOLLOWUPS.json"
    dump(hold, {"status": "verified_local_waits_paused", "signal": "SIGSTOP",
                "remote_processes_signalled": 0, "targets": targets})
    def stopped(command, **kwargs):
        start = starts[int(command[2])]
        return SimpleNamespace(returncode=0, stdout="T " + start["local_process_identity"] + "\n")
    with patch.object(f, "ROOT", root), patch.object(f, "HOLD_RECORD", hold), \
            patch.object(f._adapter.subprocess, "run", side_effect=stopped) as ps:
        yield hold, targets, starts, ps


@contextmanager
def parent_fixture(root, *, complete=False):
    parent, output = root / "parent_loss", root / "new_followup"
    parent.mkdir(); output.mkdir()
    dump(parent / "COMPLETION_STARTED.json", completion_start())
    if complete:
        dump(parent / "completion_v1/complete.json", {"status": "complete"})
    refs = [{"endpoint": index} for index in range(27)]
    verified = ({}, {}, refs, {"selected_condition": {"time_input": "log1p", "loss_mode": "log_mse", "time_alpha": 0.}}, {})
    fake = SimpleNamespace(verify_parent_completion=lambda: verified)
    with patch.object(f, "PARENT_REPORT", parent), patch.object(f._adapter.time, "time", return_value=10.), \
            patch.object(f, "verify_deferred_waits") as deferred, \
            patch.object(package, "manage_titantpp_late_path18_v2", fake, create=True):
        yield parent, output, {"absolute_host_deadlines": {"5080": 100., "5090": 101.}}, fake, deferred


class ReadyTests(unittest.TestCase):
    def test_ready_static_hashes_fixed_scope_parent_and_hold(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            parent, hold, contract = root / "parent", root / "hold.json", root / "contract.json"
            dump(parent / "COMPLETION_STARTED.json", completion_start())
            dump(hold, {"synthetic_hold": True})
            dump(contract, {"synthetic_contract": True})
            (root / "new.py").write_text("new code\n")
            ready = {"status": "locally_verified_waiting_for_loss_comparison", "new_fit_count": 18,
                "baseline_reference_count": 9, "held_out_test_authorized": False, "time_input": "log1p",
                "quantity_input": "log1p", "loss_candidate_count": 3, "raw_time_experiment_required": False,
                "files": {"new.py": f.sha(root / "new.py")}, "parent_contract_sha256": f.sha(contract),
                "parent_completion_start_sha256": f.sha(parent / "COMPLETION_STARTED.json"),
                "deferred_followups_sha256": f.sha(hold)}
            path = root / "ready.json"; dump(path, ready)
            with patch.object(f, "ROOT", root), patch.object(f, "PARENT_REPORT", parent), \
                    patch.object(f, "PARENT_CONTRACT", contract), patch.object(f, "PARENT_CONTRACT_SHA", f.sha(contract)), \
                    patch.object(f, "HOLD_RECORD", hold), patch.object(f._adapter.subprocess, "run", side_effect=AssertionError("no ps")):
                self.assertEqual(f.verify_ready(path), ready)
                for key, value in (("held_out_test_authorized", True), ("time_input", "raw"),
                        ("quantity_input", "raw"), ("loss_candidate_count", 6), ("raw_time_experiment_required", True),
                        ("new_fit_count", 27), ("files", {})):
                    mutated = {**ready, key: value}; dump(path, mutated)
                    with self.subTest(key=key):
                        with self.assertRaises(ValueError):
                            f.verify_ready(path)
                dump(path, ready)
                (root / "new.py").write_text("changed")
                with self.assertRaisesRegex(ValueError, "changed"):
                    f.verify_ready(path)

    def test_ready_rejects_path_escape_and_changed_parent_or_hold(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            parent, hold, contract = root / "parent", root / "hold.json", root / "contract.json"
            dump(parent / "COMPLETION_STARTED.json", completion_start())
            dump(hold, {}); dump(contract, {})
            ready = {"status": "locally_verified_waiting_for_loss_comparison", "new_fit_count": 18,
                "baseline_reference_count": 9, "held_out_test_authorized": False, "time_input": "log1p",
                "quantity_input": "log1p", "loss_candidate_count": 3, "raw_time_experiment_required": False,
                "files": {"../outside.py": "fake"}, "parent_contract_sha256": f.sha(contract),
                "parent_completion_start_sha256": f.sha(parent / "COMPLETION_STARTED.json"), "deferred_followups_sha256": f.sha(hold)}
            path = root / "ready.json"; dump(path, ready)
            with patch.object(f, "ROOT", root), patch.object(f, "PARENT_REPORT", parent), \
                    patch.object(f, "PARENT_CONTRACT", contract), patch.object(f, "PARENT_CONTRACT_SHA", f.sha(contract)), \
                    patch.object(f, "HOLD_RECORD", hold):
                with self.assertRaisesRegex(ValueError, "changed"):
                    f.verify_ready(path)
                (root / "new.py").write_text("new")
                ready["files"] = {"new.py": f.sha(root / "new.py")}
                for key in ("parent_contract_sha256", "parent_completion_start_sha256", "deferred_followups_sha256"):
                    dump(path, {**ready, key: "wrong"})
                    with self.subTest(key=key):
                        with self.assertRaises(ValueError):
                            f.verify_ready(path)

    def test_check_mode_never_reads_processes_waits_or_remote(self):
        with patch.object(f._adapter, "verify_ready", return_value={}), \
                patch.object(f._adapter, "wait_for_parent", side_effect=AssertionError("no wait")), \
                patch.object(f._adapter.subprocess, "run", side_effect=AssertionError("no ps or remote")):
            f.main(["--mode", "check"])


class DeferredTests(unittest.TestCase):
    def test_only_two_exact_stopped_waits_no_signal(self):
        with TemporaryDirectory() as directory, deferred_fixture(Path(directory)) as (_, _, _, ps):
            f.verify_deferred_waits()
            self.assertEqual(ps.call_count, 2)
            self.assertTrue(all(call.args[0][0] == "ps" for call in ps.call_args_list))

    def test_mac_runtime_transition_retains_pid_start_module_arguments(self):
        with TemporaryDirectory() as directory, deferred_fixture(Path(directory)) as (_, _, starts, ps):
            def resolved(command, **kwargs):
                identity = starts[int(command[2])]["local_process_identity"].replace("/usr/local/bin/python3",
                    "/Library/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python")
                return SimpleNamespace(returncode=0, stdout="Ts " + identity)
            ps.side_effect = resolved
            f.verify_deferred_waits()

    def test_running_missing_changed_start_module_or_argument_rejected(self):
        with TemporaryDirectory() as directory, deferred_fixture(Path(directory)) as (_, _, starts, ps):
            identity = starts[78887]["local_process_identity"]
            for response in (SimpleNamespace(returncode=0, stdout="S " + identity),
                    SimpleNamespace(returncode=1, stdout=""), SimpleNamespace(returncode=0, stdout="T " + identity.replace("09:21:28", "09:21:29")),
                    SimpleNamespace(returncode=0, stdout="T " + identity.replace("raw_time_mlp27", "other")),
                    SimpleNamespace(returncode=0, stdout="T " + identity.replace("advance", "check"))):
                ps.side_effect = None; ps.return_value = response
                with self.subTest(response=response):
                    with self.assertRaisesRegex(ValueError, "no longer stopped"):
                        f.verify_deferred_waits()

    def test_deferred_sha_pid_bundle_and_scope_guards(self):
        with TemporaryDirectory() as directory, deferred_fixture(Path(directory)) as (hold, targets, _, _):
            original = json.loads(hold.read_text())
            for field, value in (("start_record_sha256", "bad"), ("pid", 1), ("explicit_stopped_state_verified", False)):
                changed = deepcopy(original); changed["targets"][0][field] = value; dump(hold, changed)
                with self.subTest(field=field):
                    with self.assertRaises(ValueError):
                        f.verify_deferred_waits()
            dump(hold, original)
            bundle = Path(directory) / "search_artifacts" / targets[0]["name"]
            bundle.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "prepared or started"):
                f.verify_deferred_waits()
            bundle.rmdir()
            for field, value in (("status", "running"), ("signal", "SIGCONT"), ("remote_processes_signalled", 1)):
                dump(hold, {**original, field: value})
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ValueError, "scope"):
                        f.verify_deferred_waits()


class WaitTests(unittest.TestCase):
    def test_complete_loss27_gate_never_reexecutes_or_recollects(self):
        with TemporaryDirectory() as directory, parent_fixture(Path(directory), complete=True) as (_, out, ready, _, stopped):
            with patch.object(f._adapter, "parent_process_alive", side_effect=AssertionError("already complete")):
                f.wait_for_parent(ready, out)
            gate = json.loads((out / "parent_gate.json").read_text())
            self.assertEqual(gate["verified_parent_endpoint_count"], 27)
            self.assertFalse(gate["parent_reexecuted"] or gate["parent_recollected"] or gate["raw_time27_executed"])
            self.assertEqual(gate["new_remote_calls"], 0)
            self.assertEqual(stopped.call_count, 2)

    def test_wait_only_loss_completion_then_gate_and_no_raw_time_dependency(self):
        with TemporaryDirectory() as directory, parent_fixture(Path(directory)) as (parent, out, ready, _, _):
            def finish(_):
                dump(parent / "completion_v1/complete.json", {"status": "complete"})
            with patch.object(f._adapter, "parent_process_alive", return_value=True):
                f.wait_for_parent(ready, out, sleep=finish)
            waiting = json.loads((out / "waiting_parent.json").read_text())
            self.assertEqual(waiting["status"], "waiting_for_existing_loss_comparison")
            self.assertTrue(waiting["raw_time27_deferred"])
            self.assertTrue((out / "parent_gate.json").exists())

    def test_parent_failure_dead_missing_deadline_and_verification_errors_no_retry(self):
        for case in ("failure", "dead", "missing_start", "deadline", "bad_archive"):
            with TemporaryDirectory() as directory, parent_fixture(Path(directory), complete=case in ("deadline", "bad_archive")) as (parent, out, ready, fake, _):
                if case == "failure": dump(parent / "completion_v1/5080.failure.json", {"status": "action_required"})
                if case == "missing_start": (parent / "COMPLETION_STARTED.json").unlink()
                if case == "deadline": ready["absolute_host_deadlines"]["5080"] = 5.
                if case == "bad_archive": fake.verify_parent_completion = lambda: (_ for _ in ()).throw(ValueError("bad archive"))
                with self.subTest(case=case), patch.object(f._adapter, "parent_process_alive", return_value=False):
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        f.wait_for_parent(ready, out, sleep=lambda _: None)
                self.assertFalse((out / "parent_gate.json").exists())

    def test_wrong_parent_command_wrong27_or_raw_selection_rejected(self):
        for case in ("command", "count", "raw"):
            with TemporaryDirectory() as directory, parent_fixture(Path(directory), complete=True) as (parent, out, ready, fake, _):
                if case == "command":
                    record = completion_start(); record["command"][-1] = "paper.scripts.follow_titantpp_raw_time_mlp27"
                    dump(parent / "COMPLETION_STARTED.json", record)
                else:
                    returned = list(fake.verify_parent_completion())
                    if case == "count": returned[2] = list(range(54))
                    else: returned[3] = {"selected_condition": {"time_input": "raw"}}
                    fake.verify_parent_completion = lambda: returned
                with self.subTest(case=case):
                    with self.assertRaises(ValueError):
                        f.wait_for_parent(ready, out)
                self.assertFalse((out / "parent_gate.json").exists())

    def test_duplicate_attempt_and_gate_failure_written_once_no_prepare(self):
        with TemporaryDirectory() as directory:
            report = Path(directory); ready = report / "ready.json"; dump(ready, {})
            with patch.object(f._adapter, "REPORT", report), patch.object(f._adapter, "READY", ready), \
                    patch.object(f._adapter, "verify_ready", return_value={}), \
                    patch.object(f._adapter, "wait_for_parent", side_effect=ValueError("loss not verified")) as wait:
                with self.assertRaisesRegex(ValueError, "loss not verified"):
                    f.main(["--mode", "advance"])
                failure = json.loads((report / "followup_v1/failure.json").read_text())
                self.assertFalse(failure["automatic_retry"])
                self.assertEqual(wait.call_count, 1)
                original = (report / "followup_v1/failure.json").read_bytes()
                with self.assertRaises(FileExistsError):
                    f.main(["--mode", "advance"])
                self.assertEqual(wait.call_count, 1)
                self.assertEqual((report / "followup_v1/failure.json").read_bytes(), original)

    def test_emitted_start_completion_collection_only_new_v2_runtime_compile(self):
        contract = {"hosts": {"5080": {"root": "/fake/root", "source_root": "/fake/root/source"}}}
        source_paths = (f.ROOT / "paper/scripts/follow_titantpp_late_path18.py",
                        f.ROOT / "paper/scripts/follow_titantpp_raw_time_mlp27.py")
        before = [f.sha(path) for path in source_paths]
        programs = (f.start_adapter().start_wait_program(contract, "5080", "digest"),
                    f.completion_adapter().wait_program(contract, "5080", "digest", {"pid": 123}),
                    manager.collection_program(contract, "5080"))
        for program in programs:
            self.assertIn("from paper.scripts import titantpp_late_path18_v2_runtime as r", program)
            self.assertNotIn("import titantpp_raw_time_mlp27_runtime as r", program)
            self.assertNotIn("import titantpp_late_path18_runtime as r", program)
            compile(program, "synthetic remote program not executed", "exec")
        self.assertEqual(before, [f.sha(path) for path in source_paths])


class ManagerTests(unittest.TestCase):
    def test_only_loss27_verified_reaggregated_and_one_common_log_loss_selected(self):
        rows = loss_records()
        stored_comparison = {"records": deepcopy(rows), **loss_comparison.aggregate(rows)}
        verified = ({"synthetic_contract": True}, {"synthetic_final": True}, stored_comparison, {"synthetic_binding": True})
        verifier = SimpleNamespace(verify_parent_completion=lambda: verified,
                                   baseline_validation_references=lambda **kwargs: deepcopy(rows))
        with patch.object(package, "manage_titantpp_raw_time_mlp27", verifier, create=True):
            result = manager.verify_parent_completion()
            self.assertEqual(len(result[2]), 27)
            self.assertEqual(result[3]["selected_condition"], {"time_input": "log1p", "loss_mode": "raw_mae", "time_alpha": 0.})
            self.assertEqual(len(result[3]["condition_scores"]), 3)
            selected = manager.baseline_validation_references(verified=result)
            self.assertEqual(len(selected), 9)
            self.assertTrue(all(row["loss_mode"] == "raw_mae" and row["time_input"] == "log1p" for row in selected))
            stored_comparison["summary"][0]["mean"] += .1
            with self.assertRaisesRegex(ValueError, "aggregation differs"):
                manager.verify_parent_completion()

    def test_job18_fixed_condition_host8_10_and_parent_failure_prevents_prepare(self):
        condition = {"time_input": "log1p", "loss_mode": "raw_mae", "time_alpha": 0.}
        jobs = manager.build_jobs(condition)
        self.assertEqual((len(jobs), len({job["id"] for job in jobs})), (18, 18))
        self.assertEqual((sum(job["host"] == "5080" for job in jobs), sum(job["host"] == "5090" for job in jobs)), (8, 10))
        self.assertTrue(all(all(job[key] == value for key, value in condition.items()) for job in jobs))
        with TemporaryDirectory() as directory:
            bundle = Path(directory) / "new_bundle"
            with patch.object(manager._impl, "BUNDLE", bundle), patch.object(manager._adapter, "verify_parent_completion", side_effect=ValueError("loss incomplete")):
                with self.assertRaisesRegex(ValueError, "loss incomplete"):
                    manager.prepare()
            self.assertFalse(bundle.exists())


if __name__ == "__main__":
    unittest.main()
