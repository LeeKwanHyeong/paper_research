"""Synthetic administrative-boundary tests; no inference, training or SSH."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from paper.scripts import titantpp_components_parallel_monitor as monitor

PROJECT = Path(__file__).resolve().parents[2]


class ParallelMonitorTests(unittest.TestCase):
    def setUp(self):
        self.c = json.loads((PROJECT / "search_artifacts/titantpp_cnn_gru_components_20261012_v5/execution_contract.json").read_bytes())
        self.moved = [j for j in self.c["jobs"] if j["host"] == "5090" and j["seed"] == 62]
        self.retained = [j for j in self.c["jobs"] if j["host"] == "5090" and j["seed"] != 62]
        self.destination = {"jobs": [{**j, "host": "5080"} for j in self.moved]}

    def groups(self):
        return {name: {"status": status, "jobs": [{"job": j, "status": "complete"} for j in jobs]}
                for name, status, jobs in (
                    ("original_5080", monitor.TERMINAL, [j for j in self.c["jobs"] if j["host"] == "5080"]),
                    ("parent_5090", monitor.ADMIN_TERMINAL, self.retained),
                    ("handoff_5080", monitor.TERMINAL, self.destination["jobs"]))}

    def test_migrated_ids_are_counted_once(self):
        result = monitor.unique36_counts(self.c, self.destination, self.groups())
        self.assertEqual(result["completed"], 36)
        self.assertTrue(result["all_training_terminal_verified"])

    def test_duplicate_parent_transfer_cannot_inflate_completed(self):
        groups = self.groups()
        groups["parent_5090"]["jobs"] += [{"job": j, "status": "complete"} for j in self.moved]
        with self.assertRaises(ValueError): monitor.unique36_counts(self.c, self.destination, groups)

    def test_wrong_physical_owner_is_rejected(self):
        groups = self.groups()
        groups["handoff_5080"]["jobs"][0]["job"] = self.moved[0]
        with self.assertRaises(ValueError): monitor.unique36_counts(self.c, self.destination, groups)

    def test_query_failure_stays_unconfirmed(self):
        groups = self.groups()
        groups["parent_5090"] = {"status": "query_failed", "jobs": [], "user_action_required": True}
        result = monitor.unique36_counts(self.c, self.destination, groups)
        self.assertEqual((result["completed"], result["unconfirmed"]), (28, 8))
        self.assertFalse(result["all_training_terminal_verified"])

    def test_current_original_completion_is_not_enough_to_delete_monitor(self):
        groups = self.groups()
        groups["handoff_5080"]["status"] = "training_running"
        for row in groups["handoff_5080"]["jobs"]: row["status"] = "waiting"
        result = monitor.unique36_counts(self.c, self.destination, groups)
        self.assertEqual((result["completed"], result["waiting"]), (32, 4))
        self.assertFalse(result["all_training_terminal_verified"])

    def test_unconfirmed_identity_keeps_completed_but_never_estimates_live_fit(self):
        from paper.scripts.titantpp_components_terminal_verifier import training_status
        path = PROJECT / "search_artifacts/titantpp_cnn_gru_components_20261012_v5/control/status/5090/f3d579fe9c9e48838beeea4bbac51d0e.json"
        snapshot = json.loads(path.read_bytes())
        report = training_status(snapshot, self.c, "5090")
        report.update(status="training_identity_unconfirmed", identity_verified=False, user_action_required=True)
        report["jobs"] = [r for r in report["jobs"] if r["job"]["seed"] != 62]
        result = monitor._enrich(snapshot, report, self.c, "5090", path, 1, "synthetic_negative_case")
        active = next(r for r in result["jobs"] if r["job"]["arm"] == "titantpp_cnn_gru54" and r["job"]["seed"] == 42)
        self.assertEqual((active["status"], active["eta"]["status"]), ("unconfirmed", "unconfirmed"))
        self.assertTrue(result["user_action_required"])
        groups = self.groups()
        groups["parent_5090"] = result
        for row in groups["handoff_5080"]["jobs"]: row["status"] = "waiting"
        counts = monitor.unique36_counts(self.c, self.destination, groups)
        self.assertEqual((counts["completed"], counts["running"], counts["waiting"], counts["unconfirmed"]), (27, 0, 8, 1))

    def test_transport_failure_on_parent_does_not_skip_other_live_root(self):
        from paper.scripts import titantpp_components_seed62_handoff as handoff
        from paper.scripts import titantpp_components_terminal_verifier as verifier
        old = PROJECT / "search_artifacts/titantpp_cnn_gru_components_20261012_v5"
        path = old / "control/status/5080/1ecf99983b614d9b8f3984cad020c2fe.json"
        snapshot = json.loads(path.read_bytes())
        report = verifier.training_status(snapshot, self.c, "5080")
        target = {**self.c, "jobs": self.destination["jobs"]}
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "target"
            (bundle / "control/intents").mkdir(parents=True)
            (bundle / "control/intents/train_5080.json").write_text("{}")
            proof = bundle / "ownership/queue_exclusion_source.json"
            real_read = monitor.original.read
            with patch.object(monitor.original, "check_bundle", return_value=(self.c, {})), \
                 patch.object(handoff, "check_bundle", return_value=(target, {})), \
                 patch.object(handoff, "launcher_for"), \
                 patch.object(monitor.original, "read", side_effect=lambda p: {} if Path(p) == proof else real_read(p)), \
                 patch.object(verifier, "verified_terminal", side_effect=[(path, snapshot, report), None]), \
                 patch.object(monitor, "_acquire", side_effect=OSError("synthetic transport unavailable")) as acquire:
                _, _, groups = monitor.observe_groups(old, bundle, Path(tmp) / "reports")
            self.assertEqual(acquire.call_count, 2)
            for key in ("parent_5090", "handoff_5080"):
                self.assertEqual(groups[key]["status"], "query_or_evidence_unconfirmed")
                self.assertEqual(groups[key]["remote_calls"], 1)
                self.assertFalse(groups[key]["training_failure_inferred"])
            self.assertEqual(groups["original_5080"]["completed"], 24)

    def fixture(self):
        c = self.c
        digest = monitor.original.canonical(c)
        root = c["hosts"]["5090"]["root"]
        relative = "claims/" + self.moved[0]["id"] + ".json"
        argv = [c["hosts"]["5090"]["python"], root + "/operation/campaign.py", "--contract",
                root + "/execution_contract.json", "--host", "5090", "--mode", "dispatch"]
        state = {"status": "failed", "type": "FileExistsError", "message": "File exists: " + root + "/" + relative,
                 "active_batch": self.moved, "active_jobs": [], "owned_children_stopped": [],
                 "automatic_retry": False, "completed": {j["id"]: {} for j in self.retained}}
        phase = {"phase": "train", "pid": 101, "ppid": 100, "argv": argv, "contract_sha256": digest,
                 "os_exit_code": 1, "completed_unix": 1791849000.0}
        records = {relative: {"scientific_worker_created": False}, "status.json": state, "failure.json": state,
                   "phases/train_launch.json": phase, "phases/train_exit.json": phase,
                   "supervisor.claim": phase, "phases/train_failure.json": {"contract_sha256": digest,
                    "phase": "train", "child_pid": 101, "retry": False}}
        snapshot = {"host": "5090", "root": root, "contract_sha256": digest,
                    "gpu_uuid": c["hosts"]["5090"]["gpu_uuid"], "source_files_sha256": c["source"]["files_sha256"],
                    "gpu_pids": [], "processes": {}, "observed_unix": 1791849001.0, "files": {}}
        self.encode(snapshot, records)
        proof = {"parent_contract_sha256": digest, "transfer_jobs": self.moved,
                 "barrier_path": root + "/" + relative,
                 "barrier_raw_sha256": snapshot["files"][relative]["raw_sha256"]}
        return snapshot, records, proof

    @staticmethod
    def encode(snapshot, records):
        for name, value in records.items():
            raw = json.dumps(value, sort_keys=True).encode()
            snapshot["files"][name] = {"raw_sha256": hashlib.sha256(raw).hexdigest(),
                                        "base64": base64.b64encode(raw).decode()}

    def test_expected_admin_exit_is_preserved_with_scientific_successes(self):
        snapshot, _, proof = self.fixture()
        with patch("paper.scripts.titantpp_components_terminal_verifier.verify_completed_job",
                   side_effect=lambda s, c, h, j: {"job": j, "status": "complete"}) as single:
            result = monitor.planned_parent_terminal(snapshot, self.c, proof)
        self.assertEqual(single.call_count, 8)
        self.assertEqual(result["administrative_dispatcher_actual_exit"], 1)
        self.assertFalse(result["parent_original12_complete"])

    def test_wrong_failure_is_not_relabelled_admin(self):
        snapshot, records, proof = self.fixture()
        records["status.json"]["type"] = "RuntimeError"
        self.encode(snapshot, records)
        with self.assertRaises(ValueError): monitor.planned_parent_terminal(snapshot, self.c, proof)

    def test_active_children_prevent_admin_terminal(self):
        snapshot, records, proof = self.fixture()
        records["status.json"]["active_jobs"] = [self.retained[-1]["id"]]
        self.encode(snapshot, records)
        with self.assertRaises(ValueError): monitor.planned_parent_terminal(snapshot, self.c, proof)

    def test_changed_barrier_is_rejected(self):
        snapshot, _, proof = self.fixture()
        proof["barrier_raw_sha256"] = "0" * 64
        with self.assertRaises(ValueError): monitor.planned_parent_terminal(snapshot, self.c, proof)

    def test_admin_exit0_cannot_be_fabricated(self):
        snapshot, records, proof = self.fixture()
        records["phases/train_exit.json"]["os_exit_code"] = 0
        self.encode(snapshot, records)
        with self.assertRaises(ValueError): monitor.planned_parent_terminal(snapshot, self.c, proof)

    def test_transferred_worker_execution_is_rejected(self):
        snapshot, records, proof = self.fixture()
        records["leases/" + self.moved[-1]["id"] + ".json"] = {"unexpected": True}
        self.encode(snapshot, records)
        with self.assertRaises(ValueError): monitor.planned_parent_terminal(snapshot, self.c, proof)


if __name__ == "__main__": unittest.main()
