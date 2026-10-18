"""Finite completion preserves failure evidence and never repeats retrieval."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from paper.scripts import finish_titantpp_quantity_loss_mlp18 as finish


def require(ok, message):
    if not ok:
        raise ValueError(message)


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.manager = SimpleNamespace(BUNDLE=self.root, require=require,
            canonical=lambda value: "canonical", sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(),
            collect=Mock(return_value={"status": "complete", "host": "5080"}))
        self.contract = {"hosts": {"5080": {"root": "/isolated/run", "source_root": "/isolated/run/source",
                            "python": "/python", "absolute_host_deadline_unix": time.time() + 60.}}}
        self.identity = {"pid": 123, "argv": ["python", "dispatcher"], "cwd": "/isolated/run/source", "pgid": 123, "start_ticks": 77}

    @staticmethod
    def successful_transport(*args, **kwargs):
        kwargs["stdout"].write(json.dumps({"status": "terminal_ready_for_collection", "host": "5080"}).encode())
        return SimpleNamespace(returncode=0)

    def test_success_collects_once_after_terminal_transport(self):
        with patch.object(finish.subprocess, "run", side_effect=self.successful_transport) as remote:
            result = finish.finish_host(self.manager, self.contract, "5080", self.output, self.identity)
        self.assertEqual(result["status"], "complete")
        remote.assert_called_once()
        self.manager.collect.assert_called_once_with("5080")
        record = json.loads((self.output / "5080.wait.transport.json").read_text())
        self.assertEqual(record["actual_transport_exit"], 0)
        self.assertFalse(record["automatic_retry"])

    def test_transport_failure_keeps_stderr_and_does_not_collect(self):
        def failed_transport(*args, **kwargs):
            kwargs["stderr"].write(b"original connection failure\n")
            return SimpleNamespace(returncode=255)
        with patch.object(finish.subprocess, "run", side_effect=failed_transport) as remote:
            with self.assertRaisesRegex(ValueError, "no automatic retry"):
                finish.finish_host(self.manager, self.contract, "5080", self.output, self.identity)
        remote.assert_called_once()
        self.manager.collect.assert_not_called()
        self.assertEqual((self.output / "5080.wait.stderr").read_bytes(), b"original connection failure\n")
        self.assertEqual(json.loads((self.output / "5080.wait.transport.json").read_text())["actual_transport_exit"], 255)

    def test_existing_attempt_prevents_any_new_remote_operation(self):
        (self.root / "retrieval/5080").mkdir(parents=True)
        with patch.object(finish.subprocess, "run") as remote:
            with self.assertRaisesRegex(ValueError, "Existing retrieval"):
                finish.finish_host(self.manager, self.contract, "5080", self.output, self.identity)
        remote.assert_not_called()
        self.manager.collect.assert_not_called()

    def test_timeout_has_no_collection_or_experiment_signals(self):
        with patch.object(finish.subprocess, "run", side_effect=subprocess.TimeoutExpired("read-only wait", 60)) as remote:
            with self.assertRaisesRegex(ValueError, "no automatic retry"):
                finish.finish_host(self.manager, self.contract, "5080", self.output, self.identity)
        remote.assert_called_once()
        self.manager.collect.assert_not_called()
        record = json.loads((self.output / "5080.wait.transport.json").read_text())
        self.assertTrue(record["timeout"])
        self.assertEqual(record["signals_to_experiment"], 0)

    def test_remote_wait_is_valid_python_and_binds_observed_pid_identity(self):
        code = finish.wait_program(self.contract, "5080", "canonical", self.identity)
        compile(code, "remote_read_only_wait", "exec")
        self.assertIn("'start_ticks','argv','cwd','pgid'", code)
        self.assertNotIn(".kill(", code)
        self.assertNotIn("r.dispatch(", code)


if __name__ == "__main__":
    unittest.main()
