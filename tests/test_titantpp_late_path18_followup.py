"""Fabricated local continuation gates; no SSH, datasets or experiment execution."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from paper.scripts import follow_titantpp_late_path18 as f


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def start_record():
    return {"pid": 12345, "command": ["/usr/local/bin/python3", "-u", "-m",
        "paper.scripts.follow_titantpp_raw_time_mlp27", "--mode", "advance"],
        "local_process_identity": "Mon Oct 19 09:21:28 2026 /usr/local/bin/python3 -u -m paper.scripts.follow_titantpp_raw_time_mlp27 --mode advance"}


@contextmanager
def parent_fixture(root, *, complete=False):
    import paper.scripts as package
    parent, output = root / "parent", root / "new_followup"
    parent.mkdir(); output.mkdir()
    dump(parent / "FOLLOWUP_STARTED.json", start_record())
    if complete:
        dump(parent / "followup_v1/complete.json", {"status": "complete"})
    verified = ({}, {}, list(range(54)), {"selected_condition": {
        "time_input": "log1p", "loss_mode": "log_mse", "time_alpha": 0.}}, {})
    fake = SimpleNamespace(verify_parent_completion=lambda: verified)
    with patch.object(f, "PARENT_REPORT", parent), patch.object(f.time, "time", return_value=10.), \
            patch.object(package, "manage_titantpp_late_path18", fake, create=True):
        yield parent, output, {"absolute_host_deadlines": {"5080": 100., "5090": 101.}}, fake


class FollowupTests(unittest.TestCase):
    def test_mac_launcher_transition_same_start_and_args_only(self):
        start = start_record()
        for executable in f.PYTHON_EXECUTABLES:
            line = start["local_process_identity"].replace("/usr/local/bin/python3", executable)
            self.assertTrue(f.same_local_process_identity(start, line))

    def test_changed_start_executable_module_or_arguments_rejected(self):
        start = start_record(); line = start["local_process_identity"]
        for wrong in (line.replace("09:21:28", "09:21:29"), line.replace("/usr/local/bin/python3", "/tmp/python3"),
                      line.replace("raw_time_mlp27", "different_module"), line.replace("advance", "check"),
                      line + " --extra", ""):
            self.assertFalse(f.same_local_process_identity(start, wrong))

    def test_ps_pid_bound_no_process_is_not_success(self):
        result = SimpleNamespace(returncode=1, stdout="")
        with patch.object(f.subprocess, "run", return_value=result) as run:
            self.assertFalse(f.parent_process_alive(start_record()))
            self.assertIn("12345", run.call_args.args[0])

    def test_complete_parent_local_gate_does_not_reexecute_or_recollect(self):
        with TemporaryDirectory() as directory, parent_fixture(Path(directory), complete=True) as (_, out, ready, _):
            with patch.object(f, "parent_process_alive", side_effect=AssertionError("not needed")):
                f.wait_for_parent(ready, out)
            gate = json.loads((out / "parent_gate.json").read_text())
            self.assertEqual(gate["verified_parent_endpoint_count"], 54)
            self.assertFalse(gate["parent_reexecuted"])
            self.assertFalse(gate["parent_recollected"])
            self.assertEqual(gate["new_remote_calls"], 0)

    def test_wait_reads_local_evidence_then_publishes_gate_once(self):
        with TemporaryDirectory() as directory, parent_fixture(Path(directory)) as (parent, out, ready, _):
            def finish(_):
                dump(parent / "followup_v1/complete.json", {"status": "complete"})
            with patch.object(f, "parent_process_alive", return_value=True):
                f.wait_for_parent(ready, out, sleep=finish)
            self.assertTrue((out / "parent_gate.json").exists())

    def test_parent_failure_blocks_gate_and_never_retries(self):
        with TemporaryDirectory() as directory, parent_fixture(Path(directory)) as (parent, out, ready, _):
            dump(parent / "followup_v1/failure.json", {"status": "action_required"})
            with self.assertRaisesRegex(ValueError, "failed"):
                f.wait_for_parent(ready, out, sleep=lambda _: self.fail("no retry"))
            self.assertFalse((out / "parent_gate.json").exists())

    def test_parent_missing_without_success_is_failure(self):
        with TemporaryDirectory() as directory, parent_fixture(Path(directory)) as (_, out, ready, _):
            with patch.object(f, "parent_process_alive", return_value=False):
                with self.assertRaisesRegex(ValueError, "without successful"):
                    f.wait_for_parent(ready, out, sleep=lambda _: None)
            self.assertFalse((out / "parent_gate.json").exists())

    def test_deadline_expired_blocks_even_already_complete_parent(self):
        with TemporaryDirectory() as directory, parent_fixture(Path(directory), complete=True) as (_, out, ready, _):
            ready["absolute_host_deadlines"]["5080"] = 5.
            with self.assertRaisesRegex(ValueError, "deadline expired"):
                f.wait_for_parent(ready, out)
            self.assertFalse((out / "parent_gate.json").exists())

    def test_parent_verification_error_blocks_new_remote_work(self):
        with TemporaryDirectory() as directory, parent_fixture(Path(directory), complete=True) as (_, out, ready, fake):
            fake.verify_parent_completion = lambda: (_ for _ in ()).throw(ValueError("bad archive"))
            with self.assertRaisesRegex(ValueError, "bad archive"):
                f.wait_for_parent(ready, out)
            self.assertFalse((out / "parent_gate.json").exists())

    def test_ready_hashes_scope_and_parent_binding(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve(); parent = root / "parent"
            dump(parent / "LOCAL_READY.json", {"status": "parent frozen"})
            dump(parent / "FOLLOWUP_STARTED.json", start_record())
            (root / "new.py").write_text("new code\n")
            ready = {"status": "locally_verified_waiting_for_raw_time27", "new_fit_count": 18,
                "baseline_reference_count": 9, "held_out_test_authorized": False,
                "files": {"new.py": f.sha(root / "new.py")},
                "parent_followup_start_sha256": f.sha(parent / "FOLLOWUP_STARTED.json")}
            path = root / "ready.json"; dump(path, ready)
            with patch.object(f, "ROOT", root), patch.object(f, "PARENT_REPORT", parent), \
                    patch.object(f, "PARENT_READY_SHA", f.sha(parent / "LOCAL_READY.json")):
                self.assertEqual(f.verify_ready(path), ready)
                (root / "new.py").write_text("changed")
                with self.assertRaisesRegex(ValueError, "changed"):
                    f.verify_ready(path)

    def test_ready_empty_inventory_or_test_scope_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ready.json"
            value = {"status": "locally_verified_waiting_for_raw_time27", "new_fit_count": 18,
                "baseline_reference_count": 9, "held_out_test_authorized": False, "files": {}}
            dump(path, value)
            with self.assertRaisesRegex(ValueError, "Empty"):
                f.verify_ready(path)
            value["held_out_test_authorized"] = True; dump(path, value)
            with self.assertRaisesRegex(ValueError, "scope"):
                f.verify_ready(path)

    def test_check_mode_has_no_wait_or_new_experiment_work(self):
        with patch.object(f, "verify_ready", return_value={}), patch.object(f, "wait_for_parent") as wait:
            f.main(["--mode", "check"])
            wait.assert_not_called()

    def test_duplicate_followup_directory_rejected(self):
        with TemporaryDirectory() as directory:
            report = Path(directory); (report / "followup_v1").mkdir()
            with patch.object(f, "REPORT", report), patch.object(f, "verify_ready", return_value={}), \
                    patch.object(f, "wait_for_parent") as wait:
                with self.assertRaises(FileExistsError):
                    f.main(["--mode", "advance"])
                wait.assert_not_called()

    def test_gate_failure_recorded_once_without_retry_or_prepare(self):
        with TemporaryDirectory() as directory:
            report = Path(directory)
            ready_path = report / "LOCAL_READY.json"
            dump(ready_path, {})
            with patch.object(f, "REPORT", report), patch.object(f, "READY", ready_path), \
                    patch.object(f, "verify_ready", return_value={}), \
                    patch.object(f, "wait_for_parent", side_effect=ValueError("parent not verified")) as wait:
                with self.assertRaisesRegex(ValueError, "parent not verified"):
                    f.main(["--mode", "advance"])
                self.assertEqual(wait.call_count, 1)
                self.assertEqual(json.loads((report / "followup_v1/failure.json").read_text())["automatic_retry"], False)

    def test_private_start_and_completion_program_use_new_runtime_only(self):
        c = {"hosts": {"5080": {"root": "/fake/root", "source_root": "/fake/root/source"}}}
        before = f.sha(f.ROOT / "paper/scripts/follow_titantpp_raw_time_mlp27.py")
        start = f.start_adapter().start_wait_program(c, "5080", "digest")
        complete = f.completion_adapter().wait_program(c, "5080", "digest", {"pid": 123})
        for program in (start, complete):
            self.assertIn("from paper.scripts import titantpp_late_path18_runtime as r", program)
            self.assertNotIn("import titantpp_raw_time_mlp27_runtime as r", program)
            compile(program, "fabricated remote program not executed", "exec")
        self.assertEqual(before, f.sha(f.ROOT / "paper/scripts/follow_titantpp_raw_time_mlp27.py"))

    def test_manager_job_matrix_same_condition_host_counts_and_no_parent_retry(self):
        from paper.scripts import manage_titantpp_late_path18 as m
        condition = {"time_input": "raw", "loss_mode": "raw_mae", "time_alpha": 0.}
        jobs = m.build_jobs(condition)
        self.assertEqual(len(jobs), 18); self.assertEqual(len({j["id"] for j in jobs}), 18)
        self.assertEqual(sum(j["host"] == "5080" for j in jobs), 8)
        self.assertEqual(sum(j["host"] == "5090" for j in jobs), 10)
        self.assertTrue(all(all(j[k] == v for k, v in condition.items()) for j in jobs))
        with TemporaryDirectory() as directory:
            bundle = Path(directory) / "new_bundle"
            with patch.object(m._impl, "BUNDLE", bundle), patch.object(m, "verify_parent_completion", side_effect=ValueError("not complete")):
                with self.assertRaisesRegex(ValueError, "not complete"):
                    m.prepare()
            self.assertFalse(bundle.exists())


if __name__ == "__main__":
    unittest.main()
