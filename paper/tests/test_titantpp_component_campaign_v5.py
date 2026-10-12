"""Stdlib-only operation checks: no SSH, CUDA, real dataset or scientific fits."""
from __future__ import annotations

from copy import deepcopy
import ast
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

from paper.scripts import run_titantpp_cnn_gru_component_campaign_v5 as campaign

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
    scientific = json.loads((PROJECT / "search_artifacts/titantpp_cnn_gru_components_20261012_v1/execution_contract.json").read_text())
    files = deepcopy(scientific["source"]["files"])
    c["source"]["frozen_science_files"] = deepcopy(old["source"]["files"])
    c["source"]["frozen_science_files_sha256"] = campaign.FROZEN_CLOSURE
    c["source"].update(files=files, files_sha256=campaign.sha_json(files),
        frozen_science_git_revision=campaign.FROZEN_REVISION,
        extension_provenance="uncommitted component extension pinned by SHA256")
    c["launch_policy"] = campaign.LAUNCH_POLICY.copy()
    c["design_sha256"] = campaign.FROZEN_DESIGN_SHA
    ops = {"operation/campaign.py": "3" * 64, "operation/qualification.py": campaign.NATIVE_QUALIFIER_SHA,
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
    def test_missing_design_metadata_is_rejected_before_native_imports(self):
        c = fake_contract()
        c.pop("design_sha256")
        with self.assertRaisesRegex(ValueError, "Frozen design metadata missing"):
            campaign.validate(c, verify_files=False)
        c["design_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Frozen design metadata missing"):
            campaign.validate(c, verify_files=False)

    def test_v5_keeps_science_and_original_positive_guard_identical_to_v4(self):
        old = ast.parse((PROJECT / "paper/scripts/run_titantpp_cnn_gru_component_campaign_v4.py").read_text())
        new = ast.parse((PROJECT / "paper/scripts/run_titantpp_cnn_gru_component_campaign_v5.py").read_text())
        changed = {"gpu_observation", "_fresh_gpu_inventory"}
        old_functions = {node.name: ast.dump(node, include_attributes=False) for node in old.body
                         if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name not in changed}
        new_functions = {node.name: ast.dump(node, include_attributes=False) for node in new.body
                         if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name not in changed}
        self.assertEqual(old_functions, new_functions)

    def test_independent_launch_policy_and_science131_native_qualifier_are_fixed(self):
        c = fake_contract()
        c["launch_policy"]["all_hosts_qualification_before_any_fit"] = True
        with self.assertRaisesRegex(ValueError, "launch policy"):
            campaign.validate(c, verify_files=False)
        c = fake_contract()
        c["source"]["files"]["paper/scripts/run_titantpp_cnn_gru_components.py"] = "f" * 64
        c["source"]["files_sha256"] = campaign.sha_json(c["source"]["files"])
        with self.assertRaisesRegex(ValueError, "science131"):
            campaign.validate(c, verify_files=False)
        c = fake_contract()
        c["operation"]["files"]["operation/qualification.py"] = "f" * 64
        c["operation"]["files_sha256"] = campaign.sha_json(c["operation"]["files"])
        with self.assertRaisesRegex(ValueError, "native qualifier replaced"):
            campaign.validate(c, verify_files=False)

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
        for host in (self.start["host"],):
            value = q
            path = self.root / "qualification" / ("host_" + host + "_receipt.json")
            campaign.write_json(path, value, exclusive=True)
            bridge[host] = {"receipt": value, "raw_sha256": campaign.sha_file(path), "os_exit_code": 0}
        self.permit = {"contract_sha256": campaign.sha_json(self.c), "start_permit_sha256": campaign.sha_json(self.start),
                       "qualification_sha256": campaign.sha_file(self.root / "qualification/receipt.json"),
                       "qualification": q, "qualifications": bridge}
        campaign.write_json(self.root / "training_permit.json", self.permit, exclusive=True)

    def test_5080_own_qualification_passes_without_5090_access_or_receipt(self):
        self.assertEqual(campaign.verify_training_permit(self.c, self.root), self.permit)
        changed = deepcopy(self.permit)
        self.assertFalse((self.root / "qualification/host_5090_receipt.json").exists())
        changed["qualifications"].pop("5080")
        campaign.write_json(self.root / "training_permit.json", changed)
        with self.assertRaisesRegex(ValueError, "Only this explicitly approved host"):
            campaign.verify_training_permit(self.c, self.root)

    def test_other_host_qualification_cannot_be_added_to_current_host_permit(self):
        changed = deepcopy(self.permit)
        changed["qualifications"]["5090"] = {"receipt": fake_qualification(self.c, "5090"),
                                               "raw_sha256": "0" * 64, "os_exit_code": 0}
        campaign.write_json(self.root / "training_permit.json", changed)
        with self.assertRaisesRegex(ValueError, "Only this explicitly approved host"):
            campaign.verify_training_permit(self.c, self.root)

    def test_5080_approval_authorizes_only_5080_and_broad_approval_is_rejected(self):
        c = deepcopy(self.c)
        approval = {"approved": True, "contract_sha256": campaign.sha_json(c),
                    "hosts": ["5080"], "user_instruction": "우선 5080 먼저 진행하자. 5090은 별도 지시 후."}
        for host in c["hosts"]:
            c["hosts"][host]["root"] = str(self.root / host)
        approval["contract_sha256"] = campaign.sha_json(c)
        now = time.time()
        for host in c["hosts"]:
            root = Path(c["hosts"][host]["root"])
            campaign.write_json(root / "approval.json", approval, exclusive=True)
            campaign.write_json(root / "start_permit.json", {"host": host,
                "contract_sha256": campaign.sha_json(c), "approval_sha256": campaign.sha_json(approval),
                "started_at_unix": now - 1, "deadline_unix": now - 1 + c["limits"]["total_wall_seconds"]}, exclusive=True)
        with patch.object(campaign, "validate", return_value=c):
            self.assertEqual(campaign.authorization(c, self.root / "5080")["host"], "5080")
            with self.assertRaisesRegex(ValueError, "Host-specific user launch approval"):
                campaign.authorization(c, self.root / "5090")
            broad = {**approval, "hosts": ["5080", "5090"]}
            campaign.write_json(self.root / "5080/approval.json", broad)
            with self.assertRaisesRegex(ValueError, "Host-specific user launch approval"):
                campaign.authorization(c, self.root / "5080")

    def test_bridge_nonzero_exit_or_raw_drift_rejected(self):
        changed = deepcopy(self.permit)
        changed["qualifications"]["5080"]["os_exit_code"] = 1
        campaign.write_json(self.root / "training_permit.json", changed)
        with self.assertRaises(ValueError):
            campaign.verify_training_permit(self.c, self.root)
        campaign.write_json(self.root / "training_permit.json", self.permit)
        (self.root / "qualification/host_5080_receipt.json").write_text("{}\n")
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
                "from paper.scripts import run_titantpp_cnn_gru_component_campaign_v5 as c",
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


class GPUExitBoundaryTests(unittest.TestCase):
    def setUp(self):
        campaign._GPU_IDENTITY_WITNESSES.clear()
        self.c = fake_contract()
        self.host = "5080"
        self.script = Path(self.c["hosts"][self.host]["operation_root"]) / "qualification.py"
        self.command = ["timeout", "--signal=TERM", "--kill-after=15s", "120", self.c["hosts"][self.host]["python"],
                        str(self.script), "--mode", "capacity-worker", "--arm", campaign.CNN]
        self.active = {campaign.CNN: {"child": types.SimpleNamespace(pid=100), "command": self.command}}
        self.proc = {"ppid": 100, "pgrp": 100, "argv": self.command[4:], "start_ticks": 900,
                     "state": "S", "before": {"ppid": 100, "pgrp": 100, "start_ticks": 900, "state": "S"},
                     "consistent": True}

    def observe(self, inventories, process_results, *, active=None):
        inventory = types.SimpleNamespace(gpu_pids=unittest.mock.Mock(side_effect=inventories))
        memory = types.SimpleNamespace(stdout=self.c["hosts"][self.host]["gpu_uuid"] + ", 24000, 1000\n")
        clock = {"now": 0.}
        sleeps = []
        timeouts = []
        def sleep(delay):
            sleeps.append(delay)
            clock["now"] += delay
        def fresh(spec, timeout):
            timeouts.append(timeout)
            return inventory.gpu_pids(spec)
        with patch.object(campaign, "common", inventory), \
             patch.object(campaign.subprocess, "run", return_value=memory), \
             patch.object(campaign, "_fresh_gpu_inventory", side_effect=fresh), \
             patch.object(campaign.time, "monotonic", side_effect=lambda: clock["now"]), \
             patch.object(campaign.time, "sleep", side_effect=sleep), \
             patch.object(campaign, "_gpu_process_snapshot", side_effect=process_results):
            try:
                result = campaign.gpu_observation(self.c, self.host, self.active if active is None else active, self.script)
            finally:
                self.last_settling_trace = {"elapsed": clock["now"], "sleeps": sleeps,
                                            "timeouts": timeouts, "inventory_calls": inventory.gpu_pids.call_count}
        return result, inventory.gpu_pids.call_count

    def witnessed(self):
        result, count = self.observe([{200}], [deepcopy(self.proc)])
        self.assertEqual(result["gpu_pids"], [200])
        self.assertEqual(count, 1)

    def zombie(self):
        return {**deepcopy(self.proc), "state": "Z", "argv": [""]}

    def test_exact_live_owner_keeps_original_positive_guard_and_records_stable_identity(self):
        self.witnessed()
        key = (campaign.sha_json(self.c), self.host, str(self.script))
        witness = campaign._GPU_IDENTITY_WITNESSES[key][200]
        self.assertEqual(witness["guard_pid"], 100)
        self.assertEqual(witness["guard_command"], self.command)
        self.assertEqual(witness["argv"], self.command[4:])

    def test_owned_empty_zombie_disappears_in_one_fresh_inventory(self):
        self.witnessed()
        result, count = self.observe([{200}, set()], [self.zombie()])
        self.assertEqual(result["gpu_pids"], [])
        self.assertEqual(count, 2)
        self.assertEqual(len(result["exit_boundary_reconciliation"]), 1)
        self.assertEqual(result["exit_boundary_reconciliation"][0]["pid"], 200)

    def test_owned_proc_disappearance_is_reconciled_only_after_nvml_absence(self):
        self.witnessed()
        result, count = self.observe([{200}, set()], [FileNotFoundError("owned PID exited")])
        self.assertEqual(result["gpu_pids"], [])
        self.assertEqual(count, 2)

    def test_lingering_driver_pid_beyond_two_reads_can_settle_and_persistent_pid_rejected(self):
        self.witnessed()
        result, count = self.observe([{200}, {200}, {200}, {200}, set()], [self.zombie()] * 4)
        self.assertEqual(result["gpu_pids"], [])
        self.assertEqual(count, 5)
        self.assertEqual(result["maximum_fresh_inventory_rechecks"], 25)
        self.assertLess(result["settling_elapsed_seconds"], 5.)
        with self.assertRaisesRegex(ValueError, "bounded5s settling"):
            self.observe([{200}] * 27, [self.zombie()] * 27)
        self.assertLessEqual(self.last_settling_trace["elapsed"], 5.)
        self.assertLessEqual(len(self.last_settling_trace["timeouts"]), 25)
        self.assertTrue(all(0 < value <= .2 for value in self.last_settling_trace["sleeps"]))
        self.assertTrue(all(0 < value < 5. for value in self.last_settling_trace["timeouts"]))

    def test_fresh_nvml_subprocess_timeout_uses_remaining_budget_not_old15_seconds(self):
        spec = self.c["hosts"][self.host]
        output = types.SimpleNamespace(stdout=spec["gpu_uuid"] + ", 200\nGPU-foreign, 999\n")
        with patch.object(campaign.subprocess, "run", return_value=output) as process:
            pids = campaign._fresh_gpu_inventory(spec, .35)
        self.assertEqual(pids, {200})
        self.assertEqual(process.call_args.kwargs["timeout"], .35)
        for budget in (0., -1., 5.1, float("nan")):
            with self.assertRaises(ValueError):
                campaign._fresh_gpu_inventory(spec, budget)

    def test_hard_monotonic_deadline_is_checked_after_fresh_query(self):
        self.witnessed()
        inventory = types.SimpleNamespace(gpu_pids=lambda _: {200})
        memory = types.SimpleNamespace(stdout=self.c["hosts"][self.host]["gpu_uuid"] + ", 24000, 1000\n")
        # Started0; before sleep0; after sleep.2; then delayedquery returns5.1.
        with patch.object(campaign, "common", inventory), \
             patch.object(campaign.subprocess, "run", return_value=memory), \
             patch.object(campaign, "_fresh_gpu_inventory", return_value=set()), \
             patch.object(campaign, "_gpu_process_snapshot", return_value=self.zombie()), \
             patch.object(campaign.time, "sleep"), \
             patch.object(campaign.time, "monotonic", side_effect=[0., 0., .2, 5.1]):
            with self.assertRaisesRegex(ValueError, "hard5s monotonic deadline"):
                campaign.gpu_observation(self.c, self.host, self.active, self.script)

    def test_stable_foreign_live_argv_rejected_without_inventory_retry(self):
        foreign = {**deepcopy(self.proc), "argv": ["/python", "foreign_job.py"]}
        inventory = types.SimpleNamespace(gpu_pids=unittest.mock.Mock(side_effect=[{200}, set()]))
        memory = types.SimpleNamespace(stdout=self.c["hosts"][self.host]["gpu_uuid"] + ", 24000, 1000\n")
        with patch.object(campaign, "common", inventory), \
             patch.object(campaign.subprocess, "run", return_value=memory), \
             patch.object(campaign, "_gpu_process_snapshot", return_value=foreign):
            with self.assertRaisesRegex(ValueError, "Foreign GPU worker command"):
                campaign.gpu_observation(self.c, self.host, self.active, self.script)
        self.assertEqual(inventory.gpu_pids.call_count, 1)

    def test_unknown_disappeared_or_still_present_gpu_is_never_waived(self):
        with self.assertRaisesRegex(ValueError, "Unwitnessed GPU PID"):
            self.observe([{200}, set()], [FileNotFoundError("unknown PID")])
        with self.assertRaisesRegex(ValueError, "bounded5s settling"):
            self.observe([{200}] * 27, [self.zombie()] * 27)

    def test_stale_no_active_or_wrong_guard_cache_cannot_authorize_disappearance(self):
        self.witnessed()
        key = (campaign.sha_json(self.c), self.host, str(self.script))
        campaign._GPU_IDENTITY_WITNESSES[key][200]["observed_unix"] = 1.
        with self.assertRaisesRegex(ValueError, "recent exact current active-owner"):
            self.observe([{200}, set()], [FileNotFoundError("exited")])
        self.witnessed()
        with self.assertRaisesRegex(ValueError, "recent exact current active-owner"):
            self.observe([{200}, set()], [FileNotFoundError("exited")], active={})
        self.witnessed()
        other = {campaign.CNN: {"child": types.SimpleNamespace(pid=100), "command": self.command + ["foreign"]}}
        with self.assertRaisesRegex(ValueError, "recent exact current active-owner"):
            self.observe([{200}, set()], [FileNotFoundError("exited")], active=other)

    def test_pid_reuse_read_permission_failure_and_new_foreign_gpu_rejected(self):
        self.witnessed()
        reused = deepcopy(self.proc)
        reused["start_ticks"] = reused["before"]["start_ticks"] = 901
        with self.assertRaisesRegex(ValueError, "PID reused"):
            self.observe([{200}], [reused])
        with self.assertRaises(PermissionError):
            self.observe([{200}], [PermissionError("proc permission changed")])
        foreign = {**deepcopy(self.proc), "ppid": 999, "pgrp": 999, "argv": ["foreign.py"]}
        with self.assertRaisesRegex(ValueError, "Foreign GPU process group"):
            self.observe([{200}, {201}], [self.zombie(), foreign])

    def test_unsteady_owner_exit_can_reconcile_but_kernel_pid_reuse_cannot(self):
        self.witnessed()
        reparented = {**deepcopy(self.proc), "ppid": 1, "state": "Z", "argv": [""], "consistent": False}
        result, count = self.observe([{200}, set()], [reparented])
        self.assertEqual(result["gpu_pids"], [])
        reused = {**deepcopy(reparented), "start_ticks": 901}
        with self.assertRaisesRegex(ValueError, "PID reused during identity capture"):
            self.observe([{200}], [reused])

    def test_final_empty_active_idle_is_confirmed_without_borrowing_cache(self):
        self.witnessed()
        result, count = self.observe([set()], [], active={})
        self.assertEqual(result["gpu_pids"], [])
        self.assertEqual(result["exit_boundary_reconciliation"], [])
        self.assertEqual(count, 1)

    def test_consistent_proc_reader_checks_identity_around_cmdline(self):
        rows = ["200 (worker) R 100 100 " + "0 " * 16 + "900",
                "200 (worker) S 100 100 " + "0 " * 16 + "900"]
        class FakeProc:
            def __truediv__(self, name):
                if name in ("200",):
                    return self
                if name == "stat":
                    return types.SimpleNamespace(read_text=lambda: rows.pop(0))
                return types.SimpleNamespace(read_bytes=lambda: ("\x00".join(self_command) + "\x00").encode())
        self_command = self.command[4:]
        with patch.object(campaign, "Path", return_value=FakeProc()):
            proc = campaign._gpu_process_snapshot(200)
        self.assertTrue(proc["consistent"])
        self.assertEqual(proc["start_ticks"], 900)
        self.assertEqual(proc["argv"], self.command[4:])


class FrozenContractConsumerTests(unittest.TestCase):
    def test_actual_train_validation_loader_four_arm_interfaces_and_training_args_on_cpu(self):
        """Consume real admitted data through exact source131; no model fit/eval."""
        source = PROJECT / "search_artifacts/titantpp_cnn_gru_components_20261012_v1/source"
        contract = PROJECT / "search_artifacts/titantpp_cnn_gru_components_20261012_v2/execution_contract.json"
        program = r'''
import gc,hashlib,json,platform,sys,tempfile
from pathlib import Path
source=Path(sys.argv[1]).resolve()
c=json.loads(Path(sys.argv[2]).read_text())
c['design_sha256']='d3a579750f60c037e45b881ee03ad27a211bdbcc61f2677a4fa6ca128b0336a5'
assert 'base_git_revision' in c['source'] and c['architecture']
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as handle:
        for block in iter(lambda:handle.read(1024**2),b''):h.update(block)
    return h.hexdigest()
for name,digest in c['source']['files'].items():assert sha(source/name)==digest,name
sys.path.insert(0,str(source))
import torch
from paper.scripts import run_titantpp_cnn_gru_components as runner
from paper.scripts import run_multilag_detail_execution as base
assert Path(runner.__file__).resolve().is_relative_to(source)
assert Path(base.__file__).resolve().is_relative_to(source)
assert not torch.cuda.is_initialized()
rows=[]
with tempfile.TemporaryDirectory() as temporary:
    for data in c['datasets']:
        frame,receipt=base.prepare_admitted_data(data,data_root=source)
        assert receipt['held_out_materialized'] is False
        assert set(frame['chronological_split'].unique().to_list())<={'train','validation'}
        arms=[]
        for arm in c['arms']:
            interface=runner.time_interface(data,frame,c,arm=arm)
            assert interface['data_scope']=='verified_train_validation_only'
            assert interface['source_files_sha256']==c['source']['files_sha256']
            assert interface['time_head']['statistics_source_split']=='train'
            args=runner.training_args(c,data,Path(temporary)/data['dataset_id']/arm,arm)
            assert (args.epochs,args.min_epochs,args.early_stopping_patience,args.batch_size)==(300,40,40,128)
            assert (args.lr,args.weight_decay,args.grad_clip,args.lambda_log_qty,args.lambda_tail)==(.001,.01,1.,1.,0.)
            assert args.force_rerun is False and args.checkpoint_monitor=='validation_raw_quantity_rmse'
            assert args.source_revision==c['source']['base_git_revision']
            arms.append(arm)
        rows.append({'dataset':data['dataset_id'],'arms':arms,'interface_count':len(arms),'training_args_count':len(arms),
                     'populations':{s:receipt['populations'][s]['target_count'] for s in ('train','validation')}})
        del frame
        gc.collect()
assert not torch.cuda.is_initialized()
print(json.dumps({'status':'passed','rows':rows,'interface_count':sum(r['interface_count'] for r in rows),
                 'training_args_count':sum(r['training_args_count'] for r in rows),
                 'scientific_fits':0,'Test_materialized':False,'model_predictions':0,
                 'local_python':platform.python_version(),'local_torch':torch.__version__,
                 'qualification_kind':'CPU_contract_consumer_preflight_not_native_GPU_qualification'}))
'''
        with tempfile.TemporaryDirectory() as cache:
            environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
                           "MPLCONFIGDIR": str(Path(cache) / "matplotlib"), "XDG_CACHE_HOME": cache,
                           "MPLBACKEND": "Agg"}
            completed = subprocess.run([sys.executable, "-c", program, str(source), str(contract)],
                                       cwd=source, env=environment, text=True, capture_output=True, timeout=180)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["interface_count"], 12)
        self.assertEqual(result["training_args_count"], 12)
        self.assertEqual(result["scientific_fits"], 0)
        self.assertFalse(result["Test_materialized"])
        print("CPU contract consumer preflight: " + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
