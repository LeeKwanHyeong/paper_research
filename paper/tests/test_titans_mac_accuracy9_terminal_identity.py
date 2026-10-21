"""Regression coverage for MAC9's terminal identity observation race."""
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

from paper.scripts import titans_mac_accuracy9_completion_bridge as bridge
from paper.scripts import titantpp_late_path18_5090_recovery_runtime as original
from paper.scripts.titans_mac_accuracy9_terminal_identity import make_terminal_identity


class TerminalIdentityTests(unittest.TestCase):
    def setUp(self):
        self.child = Mock(pid=123, returncode=None)
        self.child.poll.return_value = None
        self.command = ["python", "worker"]
        self.source = Path("/source")
        self.live = dict(pid=123, pgid=123, start_ticks=456,
                         argv=self.command, cwd=str(self.source))
        self.lookup = Mock(return_value=None)
        self.initial = Mock(return_value=self.live)
        self.identity = make_terminal_identity(
            proc_identity=self.lookup, initial_identity=self.initial)

    def terminal_wait(self, code):
        def wait(*, timeout):
            self.child.returncode = code
            return code
        self.child.wait.side_effect = wait

    def test_original_reproduces_false_failure_before_exit_is_observed(self):
        self.terminal_wait(0)
        with patch.object(original, "proc_identity", self.lookup):
            with self.assertRaisesRegex(ValueError, "without observable OS exit"):
                original._identity(self.child, self.command, self.source, 456)
        self.child.wait.assert_not_called()

    def test_exit_race_waits_once_on_owned_child(self):
        self.terminal_wait(0)
        self.assertIsNone(self.identity(self.child, self.command, self.source, 456))
        self.child.wait.assert_called_once_with(timeout=0.5)
        self.lookup.assert_called_once_with(123)
        self.assertEqual(self.child.returncode, 0)
        self.initial.assert_not_called()

    def test_nonzero_exit_is_not_changed_to_success(self):
        for code in (1, -9):
            with self.subTest(code=code):
                self.terminal_wait(code)
                self.assertIsNone(self.identity(self.child, self.command, self.source, 456))
                self.assertEqual(self.child.returncode, code)

    def test_observed_exit_needs_no_wait(self):
        self.child.poll.return_value = self.child.returncode = 1
        self.assertIsNone(self.identity(self.child, self.command, self.source, 456))
        self.child.wait.assert_not_called()

    def test_timeout_fails_closed_without_retry(self):
        self.child.wait.side_effect = subprocess.TimeoutExpired(self.command, 0.5)
        with self.assertRaisesRegex(ValueError, "bounded terminal wait"):
            self.identity(self.child, self.command, self.source, 456)
        self.child.wait.assert_called_once_with(timeout=0.5)
        self.lookup.assert_called_once_with(123)
        self.assertIsNone(self.child.returncode)

    def test_live_identity_mismatches_fail_immediately_without_wait(self):
        for key, wrong in (("pid", 999), ("pgid", 999), ("start_ticks", 789),
                           ("argv", ["other"]), ("cwd", "/other")):
            with self.subTest(key=key):
                self.lookup.return_value = {**self.live, key: wrong}
                with self.assertRaisesRegex(ValueError, "Owned child identity changed"):
                    self.identity(self.child, self.command, self.source, 456)
        self.child.poll.assert_not_called()
        self.child.wait.assert_not_called()

    def test_matching_live_identity_is_preserved(self):
        self.lookup.return_value = self.live
        self.assertIs(self.identity(self.child, self.command, self.source, 456), self.live)
        self.child.poll.assert_not_called()
        self.child.wait.assert_not_called()

    def test_unregistered_child_uses_existing_initial_callback(self):
        self.assertIs(self.identity(self.child, self.command, self.source), self.live)
        self.initial.assert_called_once_with(self.child, self.command, self.source, None)
        self.lookup.assert_not_called()
        self.child.wait.assert_not_called()

    def test_existing_initial_exec_handoff_remains_bounded(self):
        identity = make_terminal_identity(proc_identity=self.lookup,
                                          initial_identity=bridge.initial_identity)
        with patch.object(bridge.r, "proc_identity", side_effect=[None, self.live]), \
             patch.object(bridge, "_parent_pid", return_value=os.getpid()), \
             patch.object(bridge, "_birth_ticks", return_value=456), \
             patch.object(bridge.time, "sleep") as sleep:
            self.assertEqual(identity(self.child, self.command, self.source), self.live)
        sleep.assert_called_once_with(.01)
        self.child.wait.assert_not_called()

    def test_existing_initial_exec_timeout_is_not_swallowed(self):
        identity = make_terminal_identity(proc_identity=self.lookup,
                                          initial_identity=bridge.initial_identity)
        with patch.object(bridge.r, "proc_identity", return_value=None), \
             patch.object(bridge, "_parent_pid", return_value=os.getpid()), \
             patch.object(bridge, "_birth_ticks", return_value=456), \
             patch.object(bridge.time, "monotonic", side_effect=[0.0, 2.1]):
            with self.assertRaisesRegex(ValueError, "Initial child exec identity handoff timed out"):
                identity(self.child, self.command, self.source)
        self.child.wait.assert_not_called()

    def test_unrecorded_wait_result_is_rejected(self):
        self.child.wait.return_value = 0
        with self.assertRaisesRegex(ValueError, "not recorded"):
            self.identity(self.child, self.command, self.source, 456)

    def test_wait_errors_do_not_become_exit_success(self):
        self.child.wait.side_effect = ChildProcessError("not this parent's child")
        with self.assertRaises(ChildProcessError):
            self.identity(self.child, self.command, self.source, 456)

    def test_wait_bounds_are_fixed_before_any_process_observation(self):
        for invalid in (0, .249, 1.001, float("nan"), float("inf"), True, "0.5"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                make_terminal_identity(proc_identity=self.lookup,
                                       initial_identity=self.initial,
                                       exit_wait_seconds=invalid)
        self.lookup.assert_not_called()

    def test_real_owned_popen_retains_actual_nonzero_exit(self):
        # A local CPU-only child, no scientific worker or remote action.
        with subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"]
                              ) as child:
            self.assertIsNone(self.identity(child, [], self.source, 456))
            self.assertEqual(child.returncode, 7)


if __name__ == "__main__":
    unittest.main()
