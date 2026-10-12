"""Stdlib-only operation checks: no SSH, CUDA, real dataset or scientific fits."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

from paper.scripts import run_titantpp_cnn_gru_component_campaign as campaign

PROJECT = Path(__file__).resolve().parents[2]


def fake_contract():
    # The historical contract contributes opaque source/input identity only;
    # no result, prediction, checkpoint or held-out metric file is opened.
    old = json.loads((PROJECT / "search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1/execution_contract.json").read_text())
    c = deepcopy(old)
    c.update(schema=campaign.SCHEMA, stage="components", arms=list(campaign.ARMS), seeds=list(campaign.SEEDS),
             training=campaign.TRAINING.copy(), parallelism=campaign.PARALLELISM.copy())
    c["assignments"] = {"5080": [[d, s] for d in campaign.DATASETS[:2] for s in campaign.SEEDS],
                         "5090": [[campaign.DATASETS[2], s] for s in campaign.SEEDS]}
    c["jobs"] = campaign.expected_jobs(c)
    c["runner_modules"] = {arm: "paper.scripts.run_titantpp_cnn_gru_components" for arm in campaign.ARMS}
    files = deepcopy(old["source"]["files"])
    c["source"]["frozen_science_files"] = files.copy()
    c["source"]["frozen_science_files_sha256"] = campaign.FROZEN_CLOSURE
    files["models/TPPs/CountAwareTitanCNNGRUComponents.py"] = "1" * 64
    files["paper/scripts/run_titantpp_cnn_gru_components.py"] = "2" * 64
    c["source"].update(files=files, files_sha256=campaign.sha_json(files),
        frozen_science_git_revision=campaign.FROZEN_REVISION,
        extension_provenance="uncommitted component extension pinned by SHA256")
    ops = {"operation/campaign.py": "3" * 64, "operation/qualification.py": "4" * 64,
           "operation/frozen_runtime.py": campaign.FROZEN_RUNTIME_SHA}
    c["operation"] = {"files": ops, "files_sha256": campaign.sha_json(ops)}
    for host, spec in c["hosts"].items():
        root = "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/" + campaign.SCHEMA + "_" + host
        spec.update(root=root, source_root=root + "/source", operation_root=root + "/operation", output_dir=root + "/run",
                    python=campaign.PYTHONS[host], gpu_uuid=campaign.GPU_UUIDS[host])
        spec["environment"]["PYTHONPATH"] = spec["source_root"]
    return c


def fake_qualification(c, host):
    digest = campaign.sha_json(c)
    inputs = {}
    for data in c["datasets"]:
        receipt = {"held_out_materialized": False, "populations": data["inherited_data_identity"]["populations"]}
        inputs[data["dataset_id"]] = {"input_receipt": receipt, "input_receipt_sha256": campaign.sha_json(receipt),
                                     "interface_sha256": {arm: "a" * 64 for arm in campaign.ARMS}}
    actual = {"actual_child_exit_observed": True, "returncode": 0, "contract_sha256": digest}
    admission = {"status": "passed", "host": host, "workers": 3, "arms": [campaign.FULL, campaign.CNN, campaign.GRU],
                 "shape": [128, 256], "scientific_optimizer_updates": 0, "timing_is_scientific_efficiency_evidence": False,
                 "simultaneous_ready_verified": True, "contract_sha256": digest,
                 "aggregate_peak_fraction_observed": .2, "GPU_witness": {"gpu_pids": [11, 12, 13]},
                 "final_GPU_observation": {"gpu_pids": []}, "actual_child_exits": {
                     arm: {**actual, "worker_group_pid": 21 + i} for i, arm in enumerate((campaign.FULL, campaign.CNN, campaign.GRU))}}
    return {"status": "passed", "host": host, "contract_sha256": digest,
            "source_files_sha256": c["source"]["files_sha256"],
            "runtime": {**c["hosts"][host]["runtime_expected"], "python": "3.12.13", "gpu": {"uuid": campaign.GPU_UUIDS[host]}},
            "inputs": inputs, "initialization": {data: {str(seed): {arm: "b" * 64 for arm in campaign.ARMS}
                for seed in campaign.SEEDS} for data in campaign.DATASETS},
            "synthetic": {str(seed): {"status": "passed", "seed": seed} for seed in campaign.SEEDS},
            "causality_checks": {"status": "passed"}, "held_out_test_evaluated": False,
            "scientific_optimizer_updates": 0, "native_qualification_exit": actual,
            "concurrency_admission": admission, "qualification_files": {}}


class ContractTests(unittest.TestCase):
    def test_two_host_36_fixed_four_arm_groups(self):
        c = fake_contract()
        campaign.validate(c, verify_files=False)
        self.assertEqual(len(campaign.batch_groups(c, "5080")), 6)
        self.assertEqual(len(campaign.batch_groups(c, "5090")), 3)
        for host in c["hosts"]:
            for batch in campaign.batch_groups(c, host):
                self.assertEqual([len(wave) for wave in campaign.group_waves(batch)], [3, 1])
                self.assertEqual([j["arm"] for j in batch], list(campaign.ARMS))

    def test_seed_gpu_split_rejected_before_execution(self):
        c = fake_contract()
        c["assignments"]["5090"].append(c["assignments"]["5080"].pop(0))
        c["jobs"] = campaign.expected_jobs(c)
        with self.assertRaisesRegex(ValueError, "cross GPU"):
            campaign.validate(c, verify_files=False)

    def test_runtime_core_and_science_mismatch_fail_closed(self):
        for section, name in (("operation", "operation/frozen_runtime.py"),
                              ("source", next(iter(fake_contract()["source"]["frozen_science_files"])))):
            c = fake_contract()
            c[section]["files"][name] = "0" * 64
            c[section]["files_sha256"] = campaign.sha_json(c[section]["files"])
            with self.assertRaises(ValueError):
                campaign.validate(c, verify_files=False)

    def test_test_scope_and_retry_rejected(self):
        for key, value in (("evaluation_scope", "test"), ("held_out_test_evaluated", True)):
            c = fake_contract()
            c[key] = value
            with self.assertRaisesRegex(ValueError, "Held-out"):
                campaign.validate(c, verify_files=False)
        c = fake_contract()
        c["limits"]["automatic_retry"] = True
        with self.assertRaisesRegex(ValueError, "retry"):
            campaign.validate(c, verify_files=False)

    def test_mixed_or_reordered_four_arm_wave_rejected(self):
        batch = campaign.batch_groups(fake_contract(), "5080")[0]
        with self.assertRaises(ValueError):
            campaign.group_waves([batch[1], batch[0], *batch[2:]])
        batch[-1] = {**batch[-1], "seed": 52}
        with self.assertRaises(ValueError):
            campaign.group_waves(batch)

    def test_native_population_seed_and_capacity_exit_negatives(self):
        c = fake_contract()
        q = fake_qualification(c, "5090")
        self.assertTrue(campaign.verify_qualification_summary(c, "5090", q))
        for mutation in (lambda value: value["synthetic"].pop("62"),
                         lambda value: value["concurrency_admission"]["actual_child_exits"][campaign.FULL].update(returncode=7),
                         lambda value: value["concurrency_admission"].update(aggregate_peak_fraction_observed=float("nan")),
                         lambda value: value["runtime"].update(cudnn=91900),
                         lambda value: value["inputs"][campaign.DATASETS[0]]["input_receipt"].update(held_out_materialized=True)):
            changed = deepcopy(q)
            mutation(changed)
            with self.assertRaises(ValueError):
                campaign.verify_qualification_summary(c, "5090", changed)


class PermitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.c = fake_contract()
        self.start = {"host": "5080", "contract_sha256": campaign.sha_json(self.c)}
        campaign.write_json(self.root / "start_permit.json", self.start, exclusive=True)
        q = fake_qualification(self.c, "5080")
        raw_exit_path = self.root / "qualification/native_exit.json"
        campaign.write_json(raw_exit_path, q["native_qualification_exit"], exclusive=True)
        q["qualification_files"] = {"qualification/native_exit.json": campaign.sha_file(raw_exit_path)}
        campaign.write_json(self.root / "qualification/receipt.json", q, exclusive=True)
        bridge = {}
        for host in self.c["hosts"]:
            value = q if host == "5080" else fake_qualification(self.c, host)
            path = self.root / "qualification" / ("host_" + host + "_receipt.json")
            campaign.write_json(path, value, exclusive=True)
            bridge[host] = {"receipt": value, "raw_sha256": campaign.sha_file(path), "os_exit_code": 0}
        self.permit = {"contract_sha256": campaign.sha_json(self.c), "start_permit_sha256": campaign.sha_json(self.start),
                       "qualification_sha256": campaign.sha_file(self.root / "qualification/receipt.json"),
                       "qualification": q, "qualifications": bridge}
        campaign.write_json(self.root / "training_permit.json", self.permit, exclusive=True)

    def test_valid_pair_requires_both_raw_native_receipts(self):
        self.assertEqual(campaign.verify_training_permit(self.c, self.root), self.permit)
        changed = deepcopy(self.permit)
        changed["qualifications"].pop("5090")
        campaign.write_json(self.root / "training_permit.json", changed)
        with self.assertRaisesRegex(ValueError, "Both native"):
            campaign.verify_training_permit(self.c, self.root)

    def test_bridge_nonzero_exit_or_raw_drift_rejected(self):
        changed = deepcopy(self.permit)
        changed["qualifications"]["5090"]["os_exit_code"] = 1
        campaign.write_json(self.root / "training_permit.json", changed)
        with self.assertRaises(ValueError):
            campaign.verify_training_permit(self.c, self.root)
        campaign.write_json(self.root / "training_permit.json", self.permit)
        (self.root / "qualification/host_5090_receipt.json").write_text("{}\n")
        with self.assertRaises(ValueError):
            campaign.verify_training_permit(self.c, self.root)

    def test_raw_admission_file_tamper_rejected(self):
        (self.root / "qualification/native_exit.json").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "file drift"):
            campaign.verify_training_permit(self.c, self.root)

    def test_no_approval_and_existing_failure_block_native_entry(self):
        c = deepcopy(self.c)
        c["hosts"]["5080"]["root"] = str(self.root)
        campaign.write_json(self.root / "approval.json", {"approved": False, "contract_sha256": campaign.sha_json(c),
                              "hosts": ["5080", "5090"], "user_instruction": "prepared only"})
        with patch.object(campaign, "validate", return_value=c):
            with self.assertRaisesRegex(ValueError, "approval"):
                campaign.authorization(c, self.root)
            campaign.write_json(self.root / "failure.json", {"automatic_retry": False}, exclusive=True)
            with self.assertRaisesRegex(ValueError, "Preserved failure"):
                campaign.authorization(c, self.root)


class NativeProcessIdentityTests(unittest.TestCase):
    def test_actual_subprocess_exit_zero_and_nonzero(self):
        for code in (0, 7):
            command = [sys.executable, "-c", "import sys; sys.exit(" + str(code) + ")"]
            child = subprocess.Popen(command, start_new_session=True)
            child.wait(timeout=5)
            receipt = campaign.observe_exit(child, command, "a" * 64)
            self.assertEqual(receipt["returncode"], code)
            self.assertEqual(receipt["worker_group_pid"], child.pid)
            self.assertTrue(receipt["actual_child_exit_observed"])

    def test_owned_timeout_cleanup_only_stops_owned_group(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(15)"], start_new_session=True)
        try:
            campaign.stop_owned(child)
            self.assertIsNotNone(child.poll())
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)

    def test_sigterm_controller_cleans_separate_session_child_before_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "exit.json"
            program = "\n".join([
                "import subprocess,sys,time",
                "from pathlib import Path",
                "from paper.scripts import run_titantpp_cnn_gru_component_campaign as c",
                "with c.owned_signal_guard():",
                "    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)",
                "    print(child.pid,flush=True)",
                "    try: time.sleep(30)",
                "    except BaseException:",
                "        c.stop_owned(child)",
                "        c.write_json(Path(sys.argv[1]),c.observe_exit(child,[], 'a'*64),exclusive=True)",
                "        raise",
            ])
            controller = subprocess.Popen([sys.executable, "-c", program, str(output)], cwd=PROJECT,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
            descendant_pid = int(controller.stdout.readline().strip())
            try:
                os.kill(controller.pid, signal.SIGTERM)
                controller.communicate(timeout=5)
                self.assertNotEqual(controller.returncode, 0)
                receipt = campaign.read(output)
                self.assertEqual(receipt["worker_group_pid"], descendant_pid)
                self.assertTrue(receipt["actual_child_exit_observed"])
                with self.assertRaises(ProcessLookupError):
                    os.kill(descendant_pid, 0)
            finally:
                if controller.poll() is None:
                    controller.kill()
                    controller.wait(timeout=5)
                try:
                    os.killpg(descendant_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_gpu_membership_exact_process_argv_required(self):
        script = Path("/owned/operation/campaign.py")
        child = types.SimpleNamespace(pid=400)
        command = ["timeout", "--signal=TERM", "--kill-after=15s", "60", "/python", str(script), "--job", "job"]
        active = {"job": {"child": child, "command": command}}
        processes = {401: {"ppid": 400, "pgrp": 400, "argv": command[4:]}}
        self.assertTrue(campaign.verify_gpu_membership({401}, processes, active, script=script, used_bytes=20, total_bytes=100))
        processes[401]["argv"][-1] = "foreign_job"
        with self.assertRaisesRegex(ValueError, "command"):
            campaign.verify_gpu_membership({401}, processes, active, script=script, used_bytes=20, total_bytes=100)


class DispatcherTests(unittest.TestCase):
    def exercise_dispatch(self, failure=False):
        c = fake_contract()
        events, children = [], []
        original_read = campaign.read
        class FakeChild:
            def __init__(self, command):
                self.pid = 2000 + len(children)
                self.command = command
                self.job = command[command.index("--job") + 1]
                self.returncode = 7 if failure and not children else 0
                children.append(self)
                events.append(("start", self.job))
            def poll(self):
                return self.returncode
        def exit_record(child, command, digest, **fields):
            events.append(("exit", child.job))
            return {**fields, "returncode": child.returncode, "worker_group_pid": child.pid,
                    "contract_sha256": digest, "actual_child_exit_observed": True}
        def fake_read(path):
            return {} if Path(path).name == "exposure.json" else original_read(path)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            start = {"deadline_unix": time.time() + 100}
            with patch.object(campaign, "authorization", return_value=start), \
                 patch.object(campaign, "verify_training_permit", return_value={}), \
                 patch.object(campaign, "validate", return_value=c), \
                 patch.object(campaign, "common", types.SimpleNamespace(gpu_pids=lambda _: set())), \
                 patch.object(campaign, "frozen", types.SimpleNamespace(VARIANT="count_only_log_regression", audit_host_batch_prefix=lambda *args: True)), \
                 patch.object(campaign, "ROOT", root), \
                 patch.object(campaign, "gpu_observation", return_value={"gpu_pids": []}), \
                 patch.object(campaign, "validate_terminal", return_value={"terminal_manifest_sha256": "a" * 64}), \
                 patch.object(campaign, "observe_exit", side_effect=exit_record), \
                 patch.object(campaign, "read", side_effect=fake_read), \
                 patch.object(campaign.subprocess, "Popen", side_effect=lambda command, **kwargs: FakeChild(command)), \
                 patch.object(campaign.shutil, "which", return_value="/usr/bin/timeout"):
                if failure:
                    with self.assertRaisesRegex(ValueError, "no automatic retry"):
                        campaign.dispatch(c, "5080", root)
                    saved = original_read(root / "failure.json")
                    self.assertFalse(saved["automatic_retry"])
                    self.assertEqual(len(children), 3)
                    self.assertTrue((root / "claims" / (children[0].job + ".json")).exists())
                else:
                    campaign.dispatch(c, "5080", root)
                    status = original_read(root / "status.json")
                    self.assertEqual(status["completed_conditions"], 24)
                    self.assertEqual(len(list((root / "worker_exits").glob("*.json"))), 24)
            return c, events

    def test_four_arm_barrier_prevents_next_seed_and_enforces_three_plus_one(self):
        c, events = self.exercise_dispatch()
        starts = [job for kind, job in events if kind == "start"]
        self.assertEqual(starts, [j["id"] for j in c["jobs"] if j["host"] == "5080"])
        for batch in campaign.batch_groups(c, "5080"):
            full_start = events.index(("start", batch[-1]["id"]))
            self.assertTrue(all(events.index(("exit", job["id"])) < full_start for job in batch[:3]))
        batches = campaign.batch_groups(c, "5080")
        for prior, following in zip(batches, batches[1:]):
            self.assertLess(max(events.index(("exit", job["id"])) for job in prior),
                            events.index(("start", following[0]["id"])))

    def test_worker_failure_preserves_claim_and_stops_without_new_wave(self):
        self.exercise_dispatch(failure=True)


if __name__ == "__main__":
    unittest.main()
