"""Contract checks for independent, bounded incident evidence capture."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from paper.scripts.collect_5090_network_incident import AlreadyRunning, CollectorLock, JsonlWriter, ProbeRunner, collect_snapshot, journal_event, redact, save_cursor, summarize_tailscale


class NetworkCaptureTests(unittest.TestCase):
    def test_rotation_stays_bounded_and_keeps_latest(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            writer = JsonlWriter(directory, max_bytes=1024, retained=3)
            for number in range(40):
                writer.append({"n": number, "text": "x" * 180})
            files = list(directory.glob("network.jsonl*"))
            self.assertEqual(len(files), 3)
            self.assertTrue(all(path.stat().st_size <= 1024 for path in files))
            self.assertEqual(json.loads(writer.path.read_text().splitlines()[-1])["n"], 39)

    def test_large_record_does_not_exceed_file_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = JsonlWriter(Path(temporary), max_bytes=1024)
            writer.append({"text": "x" * 10000})
            self.assertEqual(json.loads(writer.path.read_text())["kind"], "record_oversize")

    def test_power_loss_partial_record_is_preserved_and_next_is_readable(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = JsonlWriter(Path(temporary), max_bytes=1024)
            writer.path.write_bytes(b'{"incomplete":')
            writer.append({"kind": "restart"})
            lines = writer.path.read_bytes().splitlines()
            self.assertEqual(lines[0], b'{"incomplete":')
            self.assertEqual(json.loads(lines[1])["kind"], "restart")

    def test_repeated_collector_refused_without_deleting_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock"
            first = CollectorLock(path, os.getuid())
            try:
                with self.assertRaises(AlreadyRunning):
                    CollectorLock(path, os.getuid())
                self.assertTrue(path.exists())
            finally:
                first.close()
            again = CollectorLock(path, os.getuid())
            again.close()

    def test_symlink_lock_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "real").touch()
            (root / "link").symlink_to(root / "real")
            with self.assertRaises(OSError):
                CollectorLock(root / "link", os.getuid())

    def test_existing_lock_is_opened_without_create_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock"
            path.touch()
            real_open = os.open
            flags_seen = []
            def tracked_open(name, flags, *args):
                flags_seen.append(flags)
                return real_open(name, flags, *args)
            with patch("paper.scripts.collect_5090_network_incident.os.open", tracked_open):
                lock = CollectorLock(path, os.getuid())
                lock.close()
            self.assertFalse(flags_seen[0] & os.O_CREAT)

    def test_create_race_reopens_existing_lock_without_create(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock"
            real_open = os.open
            calls = []
            def racing_open(name, flags, *args):
                calls.append(flags)
                if len(calls) == 2:
                    other_fd = real_open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    os.close(other_fd)
                    raise FileExistsError(str(path))
                return real_open(name, flags, *args)
            with patch("paper.scripts.collect_5090_network_incident.os.open", racing_open):
                lock = CollectorLock(path, os.getuid())
                lock.close()
            self.assertEqual(len(calls), 3)
            self.assertTrue(calls[1] & os.O_EXCL)
            self.assertFalse(calls[2] & os.O_CREAT)

    def test_common_secret_forms_removed(self):
        result = redact({"text": "authkey=tskey-auth-123456789 password=hello Bearer abc.def", "PrivateKey": "private", "nested": ["token=topsecret", '{"password": "quotedpass"}']})
        serialized = json.dumps(result)
        for forbidden in ("tskey-auth", "hello", "abc.def", "topsecret", "quotedpass", '"private"'):
            self.assertNotIn(forbidden, serialized)

    def test_tailscale_omits_other_peers_and_keys(self):
        result = summarize_tailscale({"returncode": 0, "stdout": json.dumps({"PrivateKey": "hidden", "Self": {"Online": True, "PublicKey": "selfkey"}, "Peer": {"private1": {"TailscaleIPs": ["100.74.94.125"], "PublicKey": "mac_key", "Online": False}, "private2": {"TailscaleIPs": ["100.0.0.9"], "HostName": "unrelated"}}})})
        value = json.dumps(result)
        for forbidden in ("hidden", "selfkey", "mac_key", "100.0.0.9", "unrelated", "private1"):
            self.assertNotIn(forbidden, value)
        self.assertFalse(result["peers"][0]["online"])

    def test_probe_failure_does_not_erase_other_evidence(self):
        class FakeRunner:
            def run(self, argv, timeout=3.0):
                if argv[0] == "ss":
                    raise RuntimeError("probe unavailable")
                if argv[0] == "tailscale":
                    return {"returncode": 1, "stderr": "private diagnostics"}
                return {"returncode": 0, "stdout": "observed"}
        result = collect_snapshot("lo", "127.0.0.1", runner=FakeRunner())
        self.assertEqual(result["probes"]["ssh_listen"]["error"], "RuntimeError")
        self.assertEqual(result["probes"]["route"]["stdout"], "observed")
        self.assertNotIn("private diagnostics", json.dumps(result))

    def test_timeout_and_global_deadline_are_explicit(self):
        runner = ProbeRunner(1.0)
        result = runner.run(["/usr/bin/python3", "-c", "import time; time.sleep(5)"], timeout=0.02)
        self.assertEqual(result["error"], "timeout")
        runner.deadline = 0
        self.assertEqual(runner.run(["false"])["error"], "snapshot_deadline_exceeded")

    def test_journal_failure_does_not_advance_cursor(self):
        class FakeRunner:
            def run(self, argv, timeout):
                return {"returncode": 1, "stderr": "permission denied", "stdout": ""}
        event, cursor = journal_event(FakeRunner(), "prior")
        self.assertEqual(cursor, "prior")
        self.assertEqual(event["returncode"], 1)
        self.assertIn("permission denied", event["stderr"])

    def test_journal_persists_only_selected_fields(self):
        class FakeRunner:
            def run(self, argv, timeout):
                return {"returncode": 0, "stdout": json.dumps({"__CURSOR": "next", "MESSAGE": "link down", "_BOOT_ID": "boot", "EXTRA_SECRET": "do not persist"})}
        event, cursor = journal_event(FakeRunner(), "prior")
        self.assertEqual(cursor, "next")
        self.assertEqual(event["entries"][0]["MESSAGE"], "link down")
        self.assertNotIn("EXTRA_SECRET", json.dumps(event))

    def test_truncated_journal_advances_complete_prefix_not_partial_last_line(self):
        first = json.dumps({"__CURSOR": "saved-first", "MESSAGE": "first"}) + "\n"
        class FakeRunner:
            def run(self, argv, timeout):
                self.argv = argv
                return {"returncode": 0, "stdout": first + '{"__CURSOR":"partial",', "stdout_truncated": True}
        runner = FakeRunner()
        event, cursor = journal_event(runner, "prior")
        self.assertEqual(cursor, "saved-first")
        self.assertEqual(len(event["entries"]), 1)
        self.assertTrue(event["incomplete_last_line_deferred"])
        self.assertIn("--no-tail", runner.argv)
        self.assertTrue(any(arg.startswith("--output-fields=") for arg in runner.argv))
        self.assertNotIn("-n", runner.argv)

    def test_invalid_cursor_records_gap_and_resets_only_for_next_check(self):
        class FakeRunner:
            def run(self, argv, timeout):
                return {"returncode": 1, "stdout": "", "stderr": "Failed to seek to cursor: Invalid argument"}
        event, cursor = journal_event(FakeRunner(), "expired")
        self.assertIsNone(cursor)
        self.assertEqual(event["gap"]["reason"], "journal_cursor_unavailable")
        self.assertEqual(event["gap"]["next_regular_check_fallback_seconds"], 120)

    def test_cursor_write_failure_preserves_last_durable_cursor(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cursor.json"
            save_cursor(path, "prior")
            with patch("paper.scripts.collect_5090_network_incident.os.fsync", side_effect=OSError("simulated storage failure")):
                with self.assertRaises(OSError):
                    save_cursor(path, "next")
            self.assertEqual(json.loads(path.read_text())["cursor"], "prior")
            self.assertEqual(list(path.parent.glob(".journal-cursor-*")), [])

    def test_bad_middle_journal_line_never_advances_past_it(self):
        class FakeRunner:
            def run(self, argv, timeout):
                return {"returncode": 0, "stdout": json.dumps({"__CURSOR": "first", "MESSAGE": "first"}) + "\nnot-json\n" + json.dumps({"__CURSOR": "third", "MESSAGE": "third"}) + "\n"}
        event, cursor = journal_event(FakeRunner(), "prior")
        self.assertEqual(cursor, "first")
        self.assertTrue(event["parse_error"])
        self.assertEqual(len(event["entries"]), 1)

    def test_output_limit_stops_burst_and_preserves_explicit_bound(self):
        result = ProbeRunner(2).run(["/usr/bin/python3", "-c", "import os,time; os.write(1,b'x'*1000000); time.sleep(5)"], timeout=1)
        self.assertEqual(result["error"], "output_limit_reached")
        self.assertTrue(result["stdout_truncated"])
        self.assertEqual(len(result["stdout"]), 256 * 1024)
        self.assertLess(result["elapsed_seconds"], 1)


if __name__ == "__main__":
    unittest.main()
