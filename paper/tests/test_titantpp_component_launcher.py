"""Standard-library fake-remote lifecycle tests; SSH, GPU and real training0."""
from __future__ import annotations

from copy import deepcopy
import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch

from paper.scripts import launch_titantpp_cnn_gru_components as launcher


def raw(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


class FakeRemote:
    def __init__(self, c, *, busy=None, status=None):
        self.c, self.busy, self.status = c, busy, status or {}
        self.calls, self.uploads = [], []

    def call(self, host, action, payload):
        self.calls.append((host, action, payload))
        c, spec = self.c, self.c["hosts"][host]
        result = {"host": host, "root": spec["root"], "contract_sha256": launcher.canonical(c),
            "transport_os_exit_code": 0}
        if action == "health":
            result.update(status="waiting" if host == self.busy else "ready", gpu_uuid=spec["gpu_uuid"],
                gpu_pids=[999] if host == self.busy else [], reboot_required=False, update_processes=[], package_locks=[],
                root_exists=not payload["require_root_absent"], runtime={"python": "3.12.13",
                    **spec["runtime_expected"], "gpu": {"uuid": spec["gpu_uuid"]}},
                native_probe={"actual_child_exit_observed": True, "returncode": 0, "gpu_inventory_after_probe_exit": True})
        elif action == "deploy":
            now = time.time()
            result.update(status="deployed", start_permit={"host": host, "contract_sha256": launcher.canonical(c),
                "approval_sha256": launcher.canonical(payload["approval"]), "started_at_unix": now,
                "deadline_unix": now + c["limits"]["total_wall_seconds"]})
        elif action == "status":
            return {**self.status[host], "transport_os_exit_code": 0}
        else:
            result.update(status="reserved" if action == "reserve" else "started")
        return result

    def upload(self, host, local, remote_path):
        self.uploads.append((host, local, remote_path))
        return 0


class ComponentLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.project = (Path(self.tmp.name) / "project").resolve()
        self.bundle = self.project / "search_artifacts" / launcher.SCHEMA
        self.bundle.mkdir(parents=True)
        self.addCleanup(patch.stopall)
        patch.object(launcher, "PROJECT", self.project).start()
        old = {"old/module_" + str(index) + ".py": hashlib.sha256(("old" + str(index)).encode()).hexdigest()
            for index in range(129)}
        patch.object(launcher, "PARENT_SOURCE", launcher.canonical(old)).start()
        extension = {name: hashlib.sha256(name.encode()).hexdigest() for name in (
            "models/TPPs/CountAwareTitanCNNGRUComponents.py", "paper/scripts/run_titantpp_cnn_gru_components.py")}
        files = {**old, **extension}
        inputs = {"inputs/opaque_" + str(i): hashlib.sha256(("input" + str(i)).encode()).hexdigest() for i in range(7)}
        operations = {"operation/" + name: hashlib.sha256(name.encode()).hexdigest()
            for name in ("campaign.py", "frozen_runtime.py", "qualification.py", "pipeline.py")}
        assignments = {"5080": [[d, s] for d in ("yellow_trip_hourly", "raf_spare_parts") for s in (42, 52, 62)],
            "5090": [["intermittent_frozen_5000", s] for s in (42, 52, 62)]}
        datasets = [{"dataset_id": d} for d in ("yellow_trip_hourly", "raf_spare_parts", "intermittent_frozen_5000")]
        self.c = {"schema": launcher.SCHEMA, "stage": "components", "arms": list(launcher.ARMS), "seeds": [42, 52, 62],
            "assignments": assignments, "jobs": [{"id": f"{d}__{s}__{arm}", "host": host, "dataset": d,
                "seed": s, "arm": arm} for host, rows in assignments.items() for d, s in rows for arm in launcher.ARMS],
            "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
            "training": {"batch_size": 128, "maximum_epochs": 300, "minimum_epochs": 40, "patience": 40,
                "monitor": "validation_raw_quantity_rmse", "tie": "strict_earliest_finite_minimum", "warm_start": False},
            "parallelism": {"maximum_workers": 3, "grouping": "same_dataset_seed_four_arms_waves_3_plus_1",
                "per_process_cuda_fraction": .30, "aggregate_gpu_fraction": .8},
            "source": {"frozen_science_files": old, "frozen_science_files_sha256": launcher.PARENT_SOURCE,
                "files": files, "files_sha256": launcher.canonical(files), "git_revision": launcher.REVISION,
                "frozen_science_git_revision": launcher.REVISION,
                "extension_provenance": "uncommitted component extension pinned by SHA256"},
            "input_files": inputs, "operation": {"files": operations, "files_sha256": launcher.canonical(operations)},
            "datasets": datasets, "dataset_sha256": {d["dataset_id"]: launcher.canonical(d) for d in datasets},
            "limits": {"automatic_retry": False, "workers_per_host": 3, "total_wall_seconds": 168 * 3600,
                "per_condition_seconds": 36 * 3600}, "hosts": {},
            "launch_policy": {"all_hosts_qualification_before_any_fit": True, "fresh_user_start_instruction_required": True}}
        for host in launcher.HOSTS:
            root = launcher.ROOT_PARENT / (launcher.SCHEMA + "_" + host)
            self.c["hosts"][host] = {"root": str(root), "source_root": str(root / "source"),
                "operation_root": str(root / "operation"), "output_dir": str(root / "run"),
                "python": launcher.PYTHONS[host], "gpu_uuid": launcher.GPU_UUIDS[host],
                "runtime_expected": {"cuda": "13.0", "cudnn": 91900 if host == "5080" else 92000,
                    "numpy": "2.4.4", "polars": "1.39.3", "torch": "2.11.0+cu130"},
                "environment": {"PYTHONPATH": str(root / "source"), "SOURCE_REVISION": launcher.REVISION}}
        for i in range(129):
            self.bytes("source/old/module_" + str(i) + ".py", ("old" + str(i)).encode())
        for name in extension:
            self.bytes("source/" + name, name.encode())
        for i in range(7):
            self.bytes("source/inputs/opaque_" + str(i), ("input" + str(i)).encode())
        for name in operations:
            self.bytes(name, Path(name).name.encode())
        self.json("execution_contract.json", self.c)
        self.json("current.json", {"schema": launcher.SCHEMA, "contract_sha256": launcher.canonical(self.c),
            "bundle": str(self.bundle)})
        self.json("launch_hold.json", {"hold": True, "contract_sha256": launcher.canonical(self.c)})
        tools_path = self.project / "paper/scripts/launch_titantpp_cnn_gru_components.py"
        tools_path.parent.mkdir(parents=True)
        tools_path.write_text("sealed_fake_tool")
        tools = {"paper/scripts/launch_titantpp_cnn_gru_components.py": launcher.sha(tools_path)}
        self.json("local_tools_integrity.json", {"files": tools, "files_sha256": launcher.canonical(tools)})
        members = launcher.expected_members(self.c, launcher.sha(self.bundle / "execution_contract.json"))
        with tarfile.open(self.bundle / "deployment.tar.gz", "w:gz") as archive:
            for name in members:
                archive.add(self.bundle / name, arcname=name, recursive=False)
        self.json("preparation_receipt.json", {"contract_sha256": launcher.canonical(self.c),
            "contract_raw_sha256": launcher.sha(self.bundle / "execution_contract.json"),
            "local_tools_integrity_raw_sha256": launcher.sha(self.bundle / "local_tools_integrity.json"),
            "payload_sha256": launcher.sha(self.bundle / "deployment.tar.gz")})

    def bytes(self, name, content):
        path = self.bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def json(self, name, value):
        self.bytes(name, raw(value))

    def release(self):
        approval = {"approved": True, "explicit_new_user_instruction": True, "user_instruction": "이제 양 서버에서 진행하자",
            "hosts": list(launcher.HOSTS), "contract_sha256": launcher.canonical(self.c)}
        self.json("launch_authorization.json", approval)
        self.json("launch_release.json", {"released": True, "contract_sha256": launcher.canonical(self.c),
            "original_hold_raw_sha256": launcher.sha(self.bundle / "launch_hold.json"),
            "authorization_raw_sha256": launcher.sha(self.bundle / "launch_authorization.json")})
        return approval

    def deployment(self, host, approval):
        now = time.time()
        return {"host": host, "root": self.c["hosts"][host]["root"], "status": "deployed",
            "contract_sha256": launcher.canonical(self.c), "start_permit": {"host": host,
                "contract_sha256": launcher.canonical(self.c), "approval_sha256": launcher.canonical(approval),
                "started_at_unix": now, "deadline_unix": now + self.c["limits"]["total_wall_seconds"]}}

    def snapshot(self, host):
        c, spec = self.c, self.c["hosts"][host]
        q = {"status": "passed", "host": host, "contract_sha256": launcher.canonical(c),
            "source_files_sha256": c["source"]["files_sha256"], "held_out_test_evaluated": False,
            "runtime": {"python": "3.12.13", **spec["runtime_expected"], "gpu": {"uuid": spec["gpu_uuid"]}},
            "native_qualification_exit": {"actual_child_exit_observed": True, "returncode": 0},
            "concurrency_admission": {"status": "passed", "workers": 3, "arms": [launcher.FULL, launcher.CNN, launcher.GRU],
                "shape": [128, 256], "contract_sha256": launcher.canonical(c), "simultaneous_ready_verified": True,
                "scientific_optimizer_updates": 0, "timing_is_scientific_efficiency_evidence": False,
                "GPU_witness": {"gpu_pids": [10, 20, 30]}, "final_GPU_observation": {"gpu_pids": []},
                "actual_child_exits": {arm: {"returncode": 0, "actual_child_exit_observed": True, "worker_group_pid": i}
                    for i, arm in enumerate((launcher.FULL, launcher.CNN, launcher.GRU), 1)}}, "qualification_files": {}}
        process = {"phase": "qualify", "contract_sha256": launcher.canonical(c), "pid": 12, "ppid": 11,
            "argv": [spec["python"], "campaign.py", "--mode", "qualify"]}
        contents = {"qualification/receipt.json": q, "phases/qualify_exit.json": {**process, "os_exit_code": 0},
            "phases/qualify_launch.json": process, "phases/qualify_complete.json": {"host": host, "phase": "qualify",
                "contract_sha256": launcher.canonical(c), "subprocess_os_exit_code": 0, "gpu_compute_pids": [],
                "pipeline_os_exit_directly_observed": False}}
        files = {name: {"base64": base64.b64encode(raw(value)).decode(),
            "raw_sha256": hashlib.sha256(raw(value)).hexdigest()} for name, value in contents.items()}
        return {"host": host, "root": spec["root"], "contract_sha256": launcher.canonical(c),
            "gpu_uuid": spec["gpu_uuid"], "gpu_pids": [], "source_files_sha256": c["source"]["files_sha256"], "files": files}

    def set_cache(self, host, snapshot):
        self.json("control/qualification_cache/" + host + "/snapshot.json", snapshot)

    def add_record(self, snapshot, name, value):
        content = raw(value)
        snapshot["files"][name] = {"base64": base64.b64encode(content).decode(),
            "raw_sha256": hashlib.sha256(content).hexdigest()}

    def live_snapshot(self, host):
        snapshot = self.snapshot(host)
        now = time.time()
        spec = self.c["hosts"][host]
        job = next(j for j in self.c["jobs"] if j["host"] == host)
        parent_argv = [spec["python"], spec["operation_root"] + "/campaign.py", "--contract",
            spec["root"] + "/execution_contract.json", "--host", host, "--mode", "dispatch"]
        worker_argv = ["timeout", "--signal=TERM", "--kill-after=15s", "1000", spec["python"],
            spec["operation_root"] + "/campaign.py", "--contract", spec["root"] + "/execution_contract.json",
            "--host", host, "--mode", "fit", "--job", job["id"], "--deadline", str(now + 1000)]
        state = {"status": "running", "supervisor_pid": 100, "supervisor_argv": parent_argv, "deadline_unix": now + 2000,
            "active_jobs": {job["id"]: {"job": job, "worker_group_pid": 200, "worker_guard_argv": worker_argv,
                "condition_deadline_unix": now + 1000}}}
        snapshot.update(observed_unix=now, gpu_pids=[201], processes={
            "100": {"pid": 100, "ppid": 90, "pgrp": 100, "argv": parent_argv},
            "200": {"pid": 200, "ppid": 100, "pgrp": 200, "argv": worker_argv},
            "201": {"pid": 201, "ppid": 200, "pgrp": 200, "argv": worker_argv[4:]}}, checkpoints={})
        self.add_record(snapshot, "status.json", state)
        self.add_record(snapshot, "leases/" + job["id"] + ".json", {"job": job["id"], "contract_sha256": launcher.canonical(self.c),
            "worker_group_pid": 200, "supervisor_pid": 100, "expires_unix": now + 90})
        return snapshot, job

    def completed_snapshot(self, host):
        snapshot = self.snapshot(host)
        spec, digest = self.c["hosts"][host], launcher.canonical(self.c)
        argv = [spec["python"], spec["operation_root"] + "/campaign.py", "--contract",
            spec["root"] + "/execution_contract.json", "--host", host, "--mode", "dispatch"]
        phase = {"phase": "train", "pid": 100, "ppid": 90, "argv": argv, "contract_sha256": digest}
        self.add_record(snapshot, "phases/train_launch.json", phase)
        self.add_record(snapshot, "phases/train_exit.json", {**phase, "os_exit_code": 0})
        self.add_record(snapshot, "phases/train_complete.json", {"phase": "train", "host": host,
            "contract_sha256": digest, "subprocess_os_exit_code": 0, "gpu_compute_pids": []})
        self.add_record(snapshot, "supervisor.claim", {"pid": 100, "ppid": 90, "argv": argv, "contract_sha256": digest})
        completed = {}
        snapshot["terminal_hashes"] = {}
        for job in (j for j in self.c["jobs"] if j["host"] == host):
            name, deadline = job["id"], time.time() + 1000
            terminal_name = "run/" + name + "/terminal_manifest.json"
            self.add_record(snapshot, terminal_name, {"job": job, "contract_sha256": digest, "status": "complete",
                "scientific_success": True, "evaluation_scope": "validation_only", "held_out_test_evaluated": False})
            worker_name = "worker_exits/" + name + ".json"
            worker_argv = ["timeout", "--signal=TERM", "--kill-after=15s", "1000", spec["python"],
                spec["operation_root"] + "/campaign.py", "--contract", spec["root"] + "/execution_contract.json",
                "--host", host, "--mode", "fit", "--job", name, "--deadline", str(deadline)]
            self.add_record(snapshot, worker_name, {"actual_child_exit_observed": True, "returncode": 0,
                "job": job, "contract_sha256": digest, "supervisor_pid": 100, "worker_group_pid": 200,
                "argv": worker_argv})
            self.add_record(snapshot, "claims/" + name + ".json", {"job": job, "contract_sha256": digest, "deadline_unix": deadline})
            completed[name] = {"worker_exit_sha256": snapshot["files"][worker_name]["raw_sha256"],
                "terminal_manifest_sha256": snapshot["files"][terminal_name]["raw_sha256"]}
            snapshot["terminal_hashes"][name] = {"all_files_verified": True}
        self.add_record(snapshot, "status.json", {"status": "complete", "completed": completed,
            "completed_conditions": len(completed), "held_out_test_evaluated": False})
        return snapshot

    def test_check_complete_stdlib_seals_no_remote(self):
        remote = FakeRemote(self.c)
        result = launcher.run_mode(self.bundle, "check", remote=remote)
        self.assertEqual(result["remote_calls"], 0)
        self.assertEqual(result["status"], "prepared_on_hold")
        self.assertEqual(remote.calls, [])

    def test_hold_all_remote_modes_failclosed_before_any_ssh(self):
        for mode in ("deploy", "qualify", "train", "status"):
            remote = FakeRemote(self.c)
            with self.assertRaises(ValueError):
                launcher.run_mode(self.bundle, mode, remote=remote)
            self.assertEqual(remote.calls, [])
            self.assertEqual(remote.uploads, [])

    def test_bad_approval_and_release_binding_failclosed(self):
        self.release()
        approval = launcher.read(self.bundle / "launch_authorization.json")
        approval["explicit_new_user_instruction"] = False
        self.json("launch_authorization.json", approval)
        remote = FakeRemote(self.c)
        with self.assertRaises(ValueError):
            launcher.run_mode(self.bundle, "status", remote=remote)
        self.assertEqual(remote.calls, [])
        self.release()
        release = launcher.read(self.bundle / "launch_release.json")
        release["original_hold_raw_sha256"] = "0" * 64
        self.json("launch_release.json", release)
        with self.assertRaises(ValueError):
            launcher.run_mode(self.bundle, "deploy", remote=remote)
        self.assertEqual(remote.calls, [])

    def test_pointer_source_input_operation_localtool_tamper_rejected_before_ssh(self):
        self.release()
        for name in ("source/old/module_3.py", "source/inputs/opaque_2", "operation/campaign.py"):
            path = self.bundle / name
            original = path.read_bytes()
            path.write_bytes(b"corrupt")
            remote = FakeRemote(self.c)
            with self.assertRaises(ValueError):
                launcher.run_mode(self.bundle, "deploy", remote=remote)
            self.assertEqual(remote.calls, [])
            path.write_bytes(original)
        current = launcher.read(self.bundle / "current.json")
        current["bundle"] = "/foreign"
        self.json("current.json", current)
        with self.assertRaises(ValueError):
            launcher.run_mode(self.bundle, "check")

    def test_root_mismatch_rejected_before_ssh(self):
        self.release()
        bad = deepcopy(self.c)
        bad["hosts"]["5090"]["root"] = "/foreign/root"
        with self.assertRaises(ValueError):
            launcher.validate_contract(bad)
        remote = FakeRemote(self.c)
        self.json("execution_contract.json", bad)
        with self.assertRaises(ValueError):
            launcher.run_mode(self.bundle, "deploy", remote=remote)
        self.assertEqual(remote.calls, [])

    def test_deploy_both_health_before_mutation_and_busy_host_starts_nothing(self):
        self.release()
        remote = FakeRemote(self.c, busy="5090")
        result = launcher.run_mode(self.bundle, "deploy", remote=remote)
        self.assertEqual(result["status"], "waiting")
        self.assertEqual([(h, action) for h, action, _ in remote.calls], [("5080", "health"), ("5090", "health")])
        self.assertFalse(remote.uploads)
        self.assertEqual(result["phases_started"], 0)

    def test_busy_read_only_health_can_be_observed_again_without_retrying_mutation(self):
        self.release()
        remote = FakeRemote(self.c, busy="5090")
        launcher.run_mode(self.bundle, "deploy", remote=remote)
        first = set((self.bundle / "control/intents").glob("*.json"))
        launcher.run_mode(self.bundle, "deploy", remote=remote)
        second = set((self.bundle / "control/intents").glob("*.json"))
        self.assertEqual(len(remote.calls), 4)
        self.assertTrue(first < second)
        self.assertFalse(remote.uploads)
        self.assertTrue(all(action == "health" for host, action, payload in remote.calls))

    def test_deploy_exclusive_intents_no_automatic_second_attempt(self):
        self.release()
        remote = FakeRemote(self.c)
        launcher.run_mode(self.bundle, "deploy", remote=remote)
        self.assertEqual([action for host, action, payload in remote.calls[:2]], ["health", "health"])
        self.assertEqual(len(remote.uploads), 2)
        count = len(remote.calls)
        with self.assertRaises((FileExistsError, ValueError)):
            launcher.run_mode(self.bundle, "deploy", remote=remote)
        self.assertEqual(len(remote.calls), count)
        self.assertTrue((self.bundle / "control/intents/deploy_5080.json").exists())
        self.assertTrue((self.bundle / "control/intents/deploy_5080_exit.json").exists())

    def test_dual_global_training_gate_missing_or_nonzero_host_ssh0(self):
        self.release()
        self.set_cache("5080", self.snapshot("5080"))
        remote = FakeRemote(self.c)
        with self.assertRaises(ValueError):
            launcher.run_mode(self.bundle, "train", remote=remote)
        self.assertEqual(remote.calls, [])
        snapshot = self.snapshot("5090")
        record = json.loads(base64.b64decode(snapshot["files"]["phases/qualify_exit.json"]["base64"]))
        record["os_exit_code"] = 1
        content = raw(record)
        snapshot["files"]["phases/qualify_exit.json"] = {"base64": base64.b64encode(content).decode(), "raw_sha256": hashlib.sha256(content).hexdigest()}
        self.set_cache("5090", snapshot)
        with self.assertRaises(ValueError):
            launcher.run_mode(self.bundle, "train", remote=remote)
        self.assertEqual(remote.calls, [])

    def test_global_training_gate_checks_runtime_source_gpu_raw_sha_and_exit_identity(self):
        snapshot = self.snapshot("5090")
        for key, wrong in (("root", "/foreign"), ("gpu_uuid", "GPU-other"), ("source_files_sha256", "0" * 64)):
            bad = deepcopy(snapshot)
            bad[key] = wrong
            with self.assertRaises(ValueError):
                launcher.qualification_binding(bad, self.c, "5090")
        bad = deepcopy(snapshot)
        bad["files"]["qualification/receipt.json"]["raw_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            launcher.qualification_binding(bad, self.c, "5090")
        self.assertEqual(launcher.qualification_binding(snapshot, self.c, "5090")["os_exit_code"], 0)

    def test_train_both_permits_before_either_phase_and_raw_bridge_bound(self):
        approval = self.release()
        for host in launcher.HOSTS:
            self.set_cache(host, self.snapshot(host))
            self.json("control/deployment/" + host + ".json", self.deployment(host, approval))
        remote = FakeRemote(self.c)
        result = launcher.run_mode(self.bundle, "train", remote=remote)
        self.assertEqual(result["mode"], "train")
        self.assertEqual([action for host, action, payload in remote.calls], ["health", "health", "phase", "phase"])
        for host, action, payload in remote.calls:
            if action != "phase":
                continue
            self.assertEqual(set(payload["training_permit"]["qualifications"]), set(launcher.HOSTS))
            for other in launcher.HOSTS:
                content = base64.b64decode(payload["qualification_raw"][other])
                self.assertEqual(hashlib.sha256(content).hexdigest(), payload["training_permit"]["qualifications"][other]["raw_sha256"])
                self.assertTrue((self.bundle / "control/training_permits" / (other + ".json")).exists())

    def test_missing_deployment_and_expired_lease_ssh0(self):
        approval = self.release()
        remote = FakeRemote(self.c)
        with self.assertRaises(ValueError):
            launcher.run_mode(self.bundle, "qualify", remote=remote)
        self.assertFalse(remote.calls)
        for host in launcher.HOSTS:
            deployment = self.deployment(host, approval)
            deployment["start_permit"]["started_at_unix"] -= 200 * 3600
            deployment["start_permit"]["deadline_unix"] -= 200 * 3600
            self.json("control/deployment/" + host + ".json", deployment)
        with self.assertRaises(ValueError):
            launcher.run_mode(self.bundle, "qualify", remote=remote)
        self.assertFalse(remote.calls)

    def test_status_one_requested_snapshot_then_qualification_cache_ssh0(self):
        self.release()
        snapshots = {host: self.snapshot(host) for host in launcher.HOSTS}
        remote = FakeRemote(self.c, status=snapshots)
        first = launcher.run_mode(self.bundle, "status", remote=remote)
        self.assertEqual(len(remote.calls), 2)
        self.assertTrue(all(v["status"] == "qualification_passed" for v in first["hosts"].values()))
        second = launcher.run_mode(self.bundle, "status", remote=remote)
        self.assertEqual(len(remote.calls), 2)
        self.assertTrue(all(v["remote_calls"] == 0 for v in second["hosts"].values()))

    def test_live_training_status_with_owned_gpu_before_first_checkpoint(self):
        snapshot, job = self.live_snapshot("5090")
        report = launcher.training_status(snapshot, self.c, "5090")
        self.assertEqual(report["status"], "training_running")
        self.assertEqual(report["active_jobs"], 1)
        active = next(x for x in report["jobs"] if x["job"] == job)
        self.assertEqual(active["status"], "running_before_first_checkpoint")
        self.assertIsNone(active["saved_epoch"])
        self.assertIsNone(active["best_epoch"])
        self.assertEqual(report["whole_queue_eta"], "unconfirmed")

    def test_saved_history_best_epoch_is_original_strict_earliest_selector(self):
        snapshot, job = self.live_snapshot("5090")
        snapshot["checkpoints"][job["id"]] = {"saved_epoch": 3, "snapshot_complete": True, "best_selected_file_present": False}
        name = "run/" + job["id"] + "/runs/" + job["arm"] + "/count_only_log_regression/seed_" + str(job["seed"]) + "/history.json"
        self.add_record(snapshot, name, {"history": [
            {"epoch": 1, "val_qty_rmse": 3.}, {"epoch": 2, "val_qty_rmse": 2.}, {"epoch": 3, "val_qty_rmse": 2.}]})
        report = launcher.training_status(snapshot, self.c, "5090")
        active = next(x for x in report["jobs"] if x["job"] == job)
        self.assertEqual((active["saved_epoch"], active["best_epoch"]), (3, 2))
        self.assertFalse(active["selected_checkpoint_file_present"])

    def test_foreign_gpu_or_disappeared_supervisor_is_unconfirmed_not_fit_failure(self):
        snapshot, job = self.live_snapshot("5080")
        snapshot["processes"]["201"]["argv"] = ["other_task.py"]
        report = launcher.training_status(snapshot, self.c, "5080")
        self.assertEqual(report["status"], "training_identity_unconfirmed")
        self.assertTrue(report["user_action_required"])
        snapshot, job = self.live_snapshot("5080")
        snapshot["processes"]["100"] = {"pid": 100, "disappeared_during_observation": True}
        self.assertEqual(launcher.training_status(snapshot, self.c, "5080")["status"], "training_identity_unconfirmed")

    def test_status_distinguishes_live_training_failures_and_query_failures(self):
        self.release()
        snapshots = {host: self.live_snapshot(host)[0] for host in launcher.HOSTS}
        for host in launcher.HOSTS:
            self.json("control/intents/train_" + host + ".json", {"phase": "train"})
        self.add_record(snapshots["5080"], "phases/train_failure.json", {"phase": "train", "message": "failed"})
        remote = FakeRemote(self.c, status=snapshots)
        original_call = remote.call
        def call(host, action, payload):
            if host == "5090":
                remote.calls.append((host, action, payload))
                raise OSError("read-only SSH query unavailable")
            return original_call(host, action, payload)
        remote.call = call
        report = launcher.run_mode(self.bundle, "status", remote=remote)["hosts"]
        self.assertEqual(report["5080"]["status"], "training_failed")
        self.assertEqual(report["5090"]["status"], "query_failed_or_identity_unconfirmed")
        self.assertFalse(report["5090"]["training_failure_inferred"])
        self.assertEqual(len(remote.calls), 2)

    def test_remote_helper_source_compiles_without_executing(self):
        compile(launcher.REMOTE_SCRIPT, "sealed_remote_helper", "exec")

    def test_completion_requires_actual_phase_and_every_worker_raw_exit_binding(self):
        snapshot = self.completed_snapshot("5090")
        good = launcher.training_status(snapshot, self.c, "5090")
        self.assertEqual(good["status"], "training_complete_remote_verified")
        self.assertEqual(good["completed"], 12)
        self.assertEqual(good["immutable_local_binary_retrieval"], "not_performed")
        for field, value in (("pid", 999), ("ppid", 999), ("argv", ["foreign.py"]), ("phase", "qualify")):
            bad = deepcopy(snapshot)
            name = "phases/train_exit.json"
            data = json.loads(base64.b64decode(bad["files"][name]["base64"]))
            data[field] = value
            self.add_record(bad, name, data)
            with self.assertRaises(ValueError):
                launcher.training_status(bad, self.c, "5090")
        first_job = next(j for j in self.c["jobs"] if j["host"] == "5090")
        name = "worker_exits/" + first_job["id"] + ".json"
        for field, value in (("returncode", 1), ("supervisor_pid", 999), ("actual_child_exit_observed", False), ("argv", ["foreign.py"])):
            bad = deepcopy(snapshot)
            data = json.loads(base64.b64decode(bad["files"][name]["base64"]))
            data[field] = value
            self.add_record(bad, name, data)
            state = json.loads(base64.b64decode(bad["files"]["status.json"]["base64"]))
            state["completed"][first_job["id"]]["worker_exit_sha256"] = bad["files"][name]["raw_sha256"]
            self.add_record(bad, "status.json", state)
            with self.assertRaises(ValueError):
                launcher.training_status(bad, self.c, "5090")
        bad = deepcopy(snapshot)
        state = json.loads(base64.b64decode(bad["files"]["status.json"]["base64"]))
        state["completed"][first_job["id"]]["worker_exit_sha256"] = "0" * 64
        self.add_record(bad, "status.json", state)
        with self.assertRaises(ValueError):
            launcher.training_status(bad, self.c, "5090")

    def test_archive_reject_traversal_links_duplicates_unknown_and_altered_bytes(self):
        for kind in ("traversal", "symlink", "duplicate", "unknown", "altered"):
            target = Path(self.tmp.name) / (kind + ".tar.gz")
            with tarfile.open(target, "w:gz") as archive:
                entry = tarfile.TarInfo("../escape" if kind == "traversal" else "unknown" if kind == "unknown" else "safe")
                data = b"changed" if kind == "altered" else b"safe"
                entry.size = len(data)
                if kind == "symlink":
                    entry.type, entry.linkname, entry.size = tarfile.SYMTYPE, "/foreign", 0
                    archive.addfile(entry)
                else:
                    archive.addfile(entry, io.BytesIO(data))
                    if kind == "duplicate":
                        archive.addfile(entry, io.BytesIO(data))
            with self.assertRaises(ValueError):
                launcher.inspect_archive(target, {"safe": hashlib.sha256(b"safe").hexdigest()})

    def test_ssh_shell_escaping_never_interpolates_payload_as_commands(self):
        c = deepcopy(self.c)
        approval = {"user_instruction": "literal $(touch /foreign); `whoami`; ' quoted"}
        response = type("Response", (), {"returncode": 0, "stdout": json.dumps({"status": "fake"})})()
        payload = {"contract": c, "host": "5080", "approval": approval}
        with patch.object(launcher.subprocess, "run", return_value=response) as process:
            launcher.SSHRemote().call("5080", "status", payload)
        command = process.call_args.args[0]
        self.assertEqual(command[:7], ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "5080", command[6]])
        parts = __import__("shlex").split(command[-1])
        self.assertEqual(json.loads(parts[-1])["approval"], approval)


if __name__ == "__main__":
    unittest.main()
