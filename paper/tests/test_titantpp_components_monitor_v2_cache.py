"""Synthetic cache integration regressions: no server, model or data access.

Bundle admission, qualification, native launcher authorization, legacy training
classification and completed binary verification are mocked explicitly. Acquisition returns local synthetic
snapshots. Cache path/SHA/time checks and planned owner-probe verification run
through the actual observe_groups implementation; the new live-probe gate is
also called directly without a function mock.
"""
from __future__ import annotations

import base64
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from paper.scripts import titantpp_components_parallel_monitor_v2 as monitor
from paper.scripts import titantpp_components_seed62_handoff as handoff
from paper.scripts import titantpp_components_terminal_verifier as verifier


ARMS = (monitor.NONE, "titantpp_cnn_gru54_cnn_only", "titantpp_cnn_gru54_gru_only",
        "titantpp_cnn_gru54")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def encode(records):
    files = {}
    for name, value in records.items():
        raw = json.dumps(value, sort_keys=True).encode()
        files[name] = {"base64": base64.b64encode(raw).decode(),
                       "raw_sha256": hashlib.sha256(raw).hexdigest()}
    return files


class MonitorV2CacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="monitor_v2_cache_synthetic_", dir="/private/tmp")
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.parent_bundle, self.destination_bundle, self.output = root / "parent", root / "destination", root / "reports"
        self.parent = {"source": {"files_sha256": "a" * 64}, "hosts": {
            "5080": {"root": "/synthetic/original_5080", "python": "/synthetic/python5080", "gpu_uuid": "GPU-synthetic5080"},
            "5090": {"root": "/synthetic/original_5090", "python": "/synthetic/python5090", "gpu_uuid": "GPU-synthetic5090"}}, "jobs": []}
        for dataset in ("yellow_trip_hourly", "raf_spare_parts", "intermittent_frozen_5000"):
            host = "5090" if dataset == "intermittent_frozen_5000" else "5080"
            for seed in (42, 52, 62):
                for arm in ARMS:
                    self.parent["jobs"].append({"id": f"{dataset}__{seed}__{arm}", "dataset": dataset,
                                                "seed": seed, "arm": arm, "host": host})
        self.moved = [job for job in self.parent["jobs"] if job["host"] == "5090" and job["seed"] == 62]
        self.retained = [job for job in self.parent["jobs"] if job["host"] == "5090" and job["seed"] != 62]
        self.destination = deepcopy(self.parent)
        self.destination["jobs"] = [{**job, "host": "5080"} for job in self.moved]
        self.destination["hosts"]["5080"]["root"] = "/synthetic/new_seed62_5080"
        self.parent_snapshot, self.proof = self.planned_parent_snapshot()
        self.proof_path = write_json(self.destination_bundle / "ownership/queue_exclusion_source.json", self.proof)
        self.parent_path = write_json(self.parent_bundle / "control/status/5090/synthetic.json", self.parent_snapshot)
        self.old_snapshot = self.snapshot(self.parent, "5080", 1000.0, {})
        self.old_path = write_json(self.parent_bundle / "control/status/5080/old.json", self.old_snapshot)
        self.old_report = {"status": monitor.TERMINAL, "host": "5080", "total": 24, "completed": 24,
                           "identity_verified": True, "jobs": [
                               {"job": job, "status": "complete"} for job in self.parent["jobs"] if job["host"] == "5080"]}
        # Original terminal cache has no owner probes: the approved old-root exception.
        active_job = self.destination["jobs"][0]
        target_state = {"status": "running", "supervisor_pid": 401,
                        "active_jobs": {active_job["id"]: {"worker_group_pid": 501, "job": active_job}}}
        self.target_snapshot = self.snapshot(self.destination, "5080", 3000.0, {
            "status.json": target_state,
            "phases/train_launch.json": {"pid": 401, "ppid": 400}})
        processes = {}
        for pid, ppid, role in ((400, 1, "pipeline"), (401, 400, "dispatch"), (501, 401, "worker")):
            processes[str(pid)] = {"pid": pid, "ppid": ppid, "pgrp": pid,
                "argv": ["/synthetic/python5080", "/synthetic/new_seed62_5080/operation/campaign.py",
                         "--mode", role], "start_ticks": 10000 + pid,
                "observation_state": "live", "identity_stable_during_query": True}
        self.target_snapshot.update(gpu_inventory_uuids=[self.destination["hosts"]["5080"]["gpu_uuid"]],
            gpu_pids=[501], processes=processes, observed_owner_pids=[400, 401, 501],
            process_observations={pid: {"pid": int(pid), "state": "live", "actual": deepcopy(process)}
                                  for pid, process in processes.items()})
        self.target_path = write_json(self.destination_bundle / "control/status/5080/new.json", self.target_snapshot)
        self.target_report = {"status": "training_running", "host": "5080", "total": 4, "completed": 0,
                              "identity_verified": True, "jobs": [
                                  {"job": job, "status": "running_before_first_checkpoint" if job == active_job else "waiting"}
                                  for job in self.destination["jobs"]]}
        write_json(self.destination_bundle / "control/intents/train_5080.json", {"synthetic": True})
        self.bridge = {"synthetic_qualification_bridge": True}
        self.cache_path = self.output / "terminal_cache/parent_5090.json"

    @staticmethod
    def snapshot(contract, host, observed_unix, records):
        return {"host": host, "root": contract["hosts"][host]["root"],
                "contract_sha256": monitor.original.canonical(contract),
                "gpu_uuid": contract["hosts"][host]["gpu_uuid"],
                "source_files_sha256": contract["source"]["files_sha256"],
                "gpu_pids": [], "processes": {}, "observed_unix": observed_unix,
                "files": encode(records)}

    def planned_parent_snapshot(self):
        digest = monitor.original.canonical(self.parent)
        remote_root = self.parent["hosts"]["5090"]["root"]
        barrier = "claims/" + self.moved[0]["id"] + ".json"
        argv = [self.parent["hosts"]["5090"]["python"], remote_root + "/operation/campaign.py",
                "--contract", remote_root + "/execution_contract.json", "--host", "5090", "--mode", "dispatch"]
        state = {"status": "failed", "type": "FileExistsError", "message": "File exists: " + remote_root + "/" + barrier,
                 "active_batch": self.moved, "active_jobs": [], "owned_children_stopped": [],
                 "automatic_retry": False, "completed": {job["id"]: {} for job in self.retained}}
        phase = {"phase": "train", "pid": 101, "ppid": 100, "argv": argv, "contract_sha256": digest,
                 "os_exit_code": 1, "completed_unix": 1999.0}
        records = {barrier: {"scientific_worker_created": False}, "status.json": state, "failure.json": state,
                   "phases/train_launch.json": phase, "phases/train_exit.json": phase, "supervisor.claim": phase,
                   "phases/train_failure.json": {"contract_sha256": digest, "phase": "train", "child_pid": 101, "retry": False}}
        sources = {100: set(), 101: set()}
        for name in ("supervisor.claim", "phases/train_launch.json", "phases/train_exit.json"):
            sources[100].add(name)
            sources[101].add(name)
        for index, job in enumerate(self.retained):
            name = "worker_exits/" + job["id"] + ".json"
            pid = 201 + index
            records[name] = {"worker_group_pid": pid, "supervisor_pid": 101,
                             "contract_sha256": digest, "job": job}
            sources[pid] = {name}
            sources[101].add(name)
        snapshot = self.snapshot(self.parent, "5090", 2000.0, records)
        snapshot.update(gpu_inventory_uuids=[self.parent["hosts"]["5090"]["gpu_uuid"]],
                        observed_owner_pids=sorted(sources), process_observations={
                            str(pid): {"pid": pid, "state": "absent", "evidence_sources": sorted(names)}
                            for pid, names in sources.items()})
        proof = {"parent_contract_sha256": digest, "transfer_jobs": self.moved,
                 "barrier_path": remote_root + "/" + barrier,
                 "barrier_raw_sha256": snapshot["files"][barrier]["raw_sha256"]}
        return snapshot, proof

    def seed_parent_cache(self):
        write_json(self.cache_path, {"snapshot_path": str(self.parent_path),
                    "snapshot_raw_sha256": monitor.original.sha(self.parent_path),
                    "original_observed_unix": self.parent_snapshot["observed_unix"],
                    "queue_proof_raw_sha256": monitor.original.sha(self.proof_path),
                    "qualification_bridge": self.bridge,
                    "scope": "synthetic planned administrative exit1"})

    def mocks(self, acquire):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(monitor.original, "check_bundle", return_value=(self.parent, {})))
        stack.enter_context(patch.object(handoff, "check_bundle", return_value=(self.destination, {})))
        stack.enter_context(patch.object(handoff, "launcher_for", return_value=Mock()))
        stack.enter_context(patch.object(verifier, "qualification_bridge", return_value=self.bridge))
        stack.enter_context(patch.object(verifier, "verify_completed_job",
                                        side_effect=lambda s, c, h, j: {"job": j, "status": "complete"}))
        stack.enter_context(patch.object(verifier, "verified_terminal", side_effect=lambda b, o, c, h:
            (self.old_path, self.old_snapshot, self.old_report) if Path(b) == self.parent_bundle and h == "5080" else None))
        stack.enter_context(patch.object(verifier, "training_status", return_value=self.target_report))
        return stack.enter_context(patch.object(monitor, "_acquire", side_effect=acquire))

    def acquire(self, bundle, contract, host, launcher):
        if Path(bundle) == self.parent_bundle and host == "5090":
            return self.parent_path, self.parent_snapshot
        if Path(bundle) == self.destination_bundle and host == "5080":
            return self.target_path, self.target_snapshot
        self.fail("Unexpected synthetic acquisition scope")

    def test_parent_cache_sha_or_owner_probe_failure_blocks_before_any_acquisition(self):
        for corruption in ("snapshot_sha", "owner_probe"):
            with self.subTest(corruption=corruption):
                snapshot = deepcopy(self.parent_snapshot)
                if corruption == "owner_probe":
                    snapshot.pop("process_observations")
                write_json(self.parent_path, snapshot)
                self.seed_parent_cache()
                if corruption == "snapshot_sha":
                    write_json(self.parent_path, {**snapshot, "synthetic_tampering": True})
                cache_bytes = self.cache_path.read_bytes()
                acquire = self.mocks(self.acquire)
                with self.assertRaisesRegex(ValueError, "cache pin changed|probes missing"):
                    monitor.observe_groups(self.parent_bundle, self.destination_bundle, self.output)
                acquire.assert_not_called()
                self.assertEqual(self.cache_path.read_bytes(), cache_bytes)

    def test_live_parent_missing_owner_probes_cannot_seed_cache_and_other_root_is_observed_once(self):
        self.parent_snapshot.pop("process_observations")
        write_json(self.parent_path, self.parent_snapshot)
        acquire = self.mocks(self.acquire)
        _, _, groups = monitor.observe_groups(self.parent_bundle, self.destination_bundle, self.output)
        self.assertFalse(self.cache_path.exists())
        self.assertEqual(acquire.call_count, 2)
        self.assertEqual([call.args[2] for call in acquire.call_args_list], ["5090", "5080"])
        self.assertEqual(groups["parent_5090"]["status"], "query_or_evidence_unconfirmed")
        self.assertIn("probes missing", groups["parent_5090"]["reason"])
        self.assertFalse(groups["parent_5090"]["identity_verified"])
        self.assertFalse(groups["parent_5090"]["training_failure_inferred"])
        self.assertEqual(groups["parent_5090"]["remote_calls"], 1)
        self.assertEqual(groups["original_5080"]["completed"], 24)
        self.assertEqual(groups["original_5080"]["remote_calls"], 0)
        self.assertEqual(groups["original_5080"]["observed_unix"], 1000.0)
        self.assertEqual(groups["handoff_5080"]["status"], "training_running")
        self.assertTrue(groups["handoff_5080"]["live_process_probes_verified"])
        self.assertEqual(groups["handoff_5080"]["remote_calls"], 1)
        counts = monitor.unique36_counts(self.parent, self.destination, groups)
        self.assertEqual((counts["completed"], counts["unconfirmed"], counts["waiting"], counts["running"]), (24, 8, 3, 1))
        self.assertFalse(counts["all_training_terminal_verified"])

    def test_valid_parent_cache_reuse_has_zero5090_acquisitions_and_preserves_original_time(self):
        self.seed_parent_cache()
        cache_bytes = self.cache_path.read_bytes()
        acquire = self.mocks(self.acquire)
        _, _, groups = monitor.observe_groups(self.parent_bundle, self.destination_bundle, self.output)
        acquire.assert_called_once()
        self.assertEqual(acquire.call_args.args[2], "5080")
        self.assertEqual(Path(acquire.call_args.args[0]), self.destination_bundle)
        report = groups["parent_5090"]
        self.assertEqual(report["status"], monitor.ADMIN_TERMINAL)
        self.assertEqual(report["remote_calls"], 0)
        self.assertEqual(report["observed_unix"], 2000.0)
        self.assertEqual(report["observation_source"], "verified_administrative_terminal_cache_original_observation")
        self.assertTrue(report["owner_process_absence_verified"])
        self.assertEqual(report["owner_process_probe_count"], 10)
        self.assertEqual(report["administrative_dispatcher_actual_exit"], 1)
        self.assertFalse(report["parent_original12_complete"])
        self.assertFalse(report["pipeline_wrapper_OS_exit_directly_observed"])
        self.assertEqual(groups["original_5080"]["completed"], 24)
        self.assertEqual(groups["original_5080"]["remote_calls"], 0)
        self.assertEqual(groups["original_5080"]["observed_unix"], 1000.0)
        self.assertEqual(groups["handoff_5080"]["observed_unix"], 3000.0)
        self.assertEqual(groups["handoff_5080"]["remote_calls"], 1)
        self.assertTrue(groups["handoff_5080"]["live_process_probes_verified"])
        self.assertEqual(self.cache_path.read_bytes(), cache_bytes)


if __name__ == "__main__":
    unittest.main()
