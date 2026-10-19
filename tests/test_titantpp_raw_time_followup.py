"""The finite follow-up waits locally and cannot restart a failed predecessor."""
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from paper.scripts import follow_titantpp_raw_time_mlp27 as follow


class FollowupTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.parent = self.root / "parent"
        (self.parent / "completion_v1").mkdir(parents=True)
        self.output = self.root / "output"
        self.output.mkdir()
        follow.write(self.parent / "COMPLETION_STARTED.json",
                     {"pid": 123, "local_process_identity": "known identity"})
        self.ready = {"absolute_host_deadlines": {"5080": time.time() + 100., "5090": time.time() + 100.}}

    def test_failed_parent_is_not_recollected_or_restarted(self):
        follow.write(self.parent / "completion_v1/failure.json", {"status": "action_required"})
        with patch.object(follow, "PARENT_REPORT", self.parent), patch.object(follow.subprocess, "run") as remote:
            with self.assertRaisesRegex(ValueError, "Parent completion failed"):
                follow.wait_for_parent(self.ready, self.output, sleep=lambda value: None)
        remote.assert_not_called()

    def test_dead_parent_stops_before_new_manager_import_or_ssh(self):
        with patch.object(follow, "PARENT_REPORT", self.parent), \
             patch.object(follow, "parent_process_alive", return_value=False), \
             patch.object(follow.subprocess, "run") as remote:
            with self.assertRaisesRegex(ValueError, "exited without success"):
                follow.wait_for_parent(self.ready, self.output, sleep=lambda value: None)
        remote.assert_not_called()

    def test_start_wait_program_only_observes_the_new_root_once(self):
        contract = {"hosts": {"5080": {"root": "/isolated/new", "source_root": "/isolated/new/source"}}}
        program = follow.start_wait_program(contract, "5080", "hash")
        compile(program, "start_observation", "exec")
        self.assertEqual(program.count("r.status("), 1)
        self.assertNotIn(".kill(", program)
        self.assertNotIn("r.dispatch(", program)
        self.assertNotIn("SIGCONT", program)
        self.assertIn("training_worker_gpu_start_not_asserted", program)

    def test_completion_runtime_is_private_and_original_module_unchanged(self):
        from paper.scripts import finish_titantpp_quantity_loss_mlp18 as parent
        original = parent.wait_program
        adapter = follow.completion_adapter()
        self.assertIs(parent.wait_program, original)
        self.assertIsNot(adapter.wait_program, original)
        contract = {"hosts": {"5080": {"root": "/new", "source_root": "/new/source", "python": "/python"}}}
        code = adapter.wait_program(contract, "5080", "hash", {"pid": 123})
        compile(code, "completion_wait", "exec")
        self.assertIn("titantpp_raw_time_mlp27_runtime as r", code)
        self.assertNotIn("titantpp_quantity_loss_mlp18_runtime as r", code)

    def test_local_parent_identity_is_exact_and_cannot_accept_other_process(self):
        with patch.object(follow.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="same   identity\n")):
            self.assertTrue(follow.parent_process_alive({"pid": 1, "local_process_identity": "same identity"}))
            self.assertFalse(follow.parent_process_alive({"pid": 1, "local_process_identity": "different identity"}))


if __name__ == "__main__":
    unittest.main()
