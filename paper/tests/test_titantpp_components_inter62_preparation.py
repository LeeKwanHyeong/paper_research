"""Exact4 subset binding tests; opaque synthetic bytes and stdlib checks only."""
from __future__ import annotations

import base64
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from paper.scripts import prepare_titantpp_components_inter62_frozen_evaluation as prepare
from paper.tests.test_titantpp_components_evaluation_preparation import BindingFixture


class Inter62Fixture(BindingFixture):
    def __init__(self, root):
        super().__init__(root)
        self.bindings["schema"] = "titantpp_components_inter62_checkpoint_bindings_v1"
        original_jobs = self.bindings["jobs"]
        self.bindings["jobs"] = [job for job in original_jobs
                                 if job["dataset"] == prepare.DATASET and job["seed"] == 62]
        for job in self.bindings["jobs"]:
            job["cohort"] = prepare.COHORT
        self.bindings["datasets"] = {prepare.DATASET: self.bindings["datasets"][prepare.DATASET]}
        self.runtimes = {"5080": self.runtimes["5080"]}
        self.design_closure = prepare.canonical(self.design)
        training = prepare.read(self.root / self.bindings["training_contract_path"])
        training.update(source={"files_sha256": self.source_closure},
                        hosts={"5080": {"root": "/synthetic/original_5080",
                                           "python": "/synthetic/bin/python",
                                           "operation_root": "/synthetic/original_5080/operation",
                                           "gpu_uuid": prepare.base.GPU_UUIDS["5080"]}},
                        jobs=[{"id": job["id"], "host": job["training_host"], "dataset": job["dataset"],
                               "seed": job["seed"], "arm": job["arm"]} for job in self.bindings["jobs"]])
        self.parent = prepare.canonical(training)
        self.bindings["training_contract_sha256"] = self.write_json(
            self.bindings["training_contract_path"], training)
        records = {}
        monitor_rows = []
        pid_sources = {100: set(), 101: set()}
        for index, job in enumerate(self.bindings["jobs"]):
            self.refresh_json(job, "terminal_manifest_path", "terminal_manifest_sha256",
                              lambda value: value.update(contract_sha256=self.parent))
            self.refresh_json(job, "worker_exit_path", "worker_exit_sha256",
                              lambda value, index=index: value.update(
                                  contract_sha256=self.parent, worker_group_pid=201 + index, supervisor_pid=101))
            name = "worker_exits/" + job["id"] + ".json"
            records[name] = prepare.read(self.root / job["worker_exit_path"])
            records["run/" + job["id"] + "/terminal_manifest.json"] = prepare.read(
                self.root / job["terminal_manifest_path"])
            pid_sources[201 + index] = {name}
            pid_sources[101].add(name)
            monitor_rows.append({"job": records[name]["job"], "status": "complete",
                "selected_checkpoint_file_present": True, "remote_all_terminal_file_hashes_verified": True,
                "terminal_manifest_raw_sha256": job["terminal_manifest_sha256"],
                "worker_exit_raw_sha256": job["worker_exit_sha256"]})
        phase = {"phase": "train", "contract_sha256": self.parent, "pid": 101, "ppid": 100,
                 "argv": ["/synthetic/bin/python", "/synthetic/original_5080/operation/campaign.py",
                          "--contract", "/synthetic/original_5080/execution_contract.json",
                          "--host", "5080", "--mode", "dispatch"], "os_exit_code": 0}
        for name in ("supervisor.claim", "phases/train_launch.json", "phases/train_exit.json"):
            records[name] = phase
            pid_sources[100].add(name)
            pid_sources[101].add(name)
        records["phases/train_complete.json"] = {"phase": "train", "host": "5080",
            "contract_sha256": self.parent, "subprocess_os_exit_code": 0, "gpu_compute_pids": []}
        records["status.json"] = {"status": "complete", "completed_conditions": 4, "held_out_test_evaluated": False,
            "completed": {job["id"]: {"worker_exit_sha256": job["worker_exit_sha256"],
                                      "terminal_manifest_sha256": job["terminal_manifest_sha256"]}
                          for job in self.bindings["jobs"]}}
        process_observations = {str(pid): {"pid": pid, "state": "absent", "evidence_sources": sorted(names)}
                                for pid, names in pid_sources.items()}
        snapshot = {"host": "5080", "root": "/synthetic/original_5080", "contract_sha256": self.parent,
                    "source_files_sha256": self.source_closure, "gpu_uuid": prepare.base.GPU_UUIDS["5080"],
                    "gpu_inventory_uuids": [prepare.base.GPU_UUIDS["5080"]], "gpu_pids": [], "processes": {},
                    "process_observations": process_observations, "observed_owner_pids": sorted(pid_sources),
                    "observed_unix": 2000.0,
                    "terminal_hashes": {job["id"]: {"all_files_verified": True,
                        "files": {name: True for name in prepare.read(self.root / job["terminal_manifest_path"])["files"]}}
                        for job in self.bindings["jobs"]},
                    "files": {}}
        for name, value in records.items():
            raw = json.dumps(value, sort_keys=True).encode()
            snapshot["files"][name] = {"base64": base64.b64encode(raw).decode(),
                                       "raw_sha256": hashlib.sha256(raw).hexdigest()}
        self.bindings["handoffterminal_evidence_path"] = "evidence/parent5080_snapshot.json"
        self.bindings["handoffterminal_evidence_sha256"] = self.write_json(
            self.bindings["handoffterminal_evidence_path"], snapshot)
        monitor = {"parent_contract_sha256": prepare.base.PARENT, "destination_contract_sha256": self.parent, "all_training_terminal_verified": True,
                   "held_out_test_evaluated": False,
                   "totals": {"total": 36, "completed": 36, "running": 0, "waiting": 0, "unconfirmed": 0,
                              "all_training_terminal_verified": True},
                   "groups": {"handoff_5080": {"status": "training_complete_remote_verified",
                       "host": "5080", "contract_sha256": self.parent, "completed": 4, "total": 4,
                       "root": snapshot["root"], "observed_unix": snapshot["observed_unix"],
                       "snapshot_raw_sha256": self.bindings["handoffterminal_evidence_sha256"],
                       "identity_verified": True, "owner_process_absence_verified": True,
                       "pipeline_os_exit_directly_observed": False, "parent_original12_complete": False,
                       "pipeline_wrapper_OS_exit_directly_observed": False,
                       "original_failure_and_claim_preserved": True, "scientific_failure_inferred": False,
                       "owner_process_probe_count": 6,
                       "owner_process_observations_sha256": prepare.canonical(process_observations),
                       "jobs": monitor_rows}}}
        self.bindings["training_monitor_sha256"] = self.write_json(self.bindings["training_monitor_path"], monitor)
        instruction = "Intermittent seed62 selected checkpoint8을 5080에서 우선 Test 평가한다."
        self.local_root = str(self.root.resolve())
        self.remote_root = str((self.root / "remote_approval_scope").resolve())
        approval = {"schema": "titantpp_components_inter62_partial_approval_v1",
                    "partial_cohort_explicitly_authorized": True, "evaluation_scope": "test_seed62_followup",
                    "cohort": prepare.COHORT, "host": "5080", "expected_condition_count": 4,
                    "job_ids": sorted(prepare.expected_ids()), "parent_contract_sha256": self.parent,
                    "frozen_source_closure": self.source_closure, "new_training": False,
                    "original36_global_gate_modified": False, "user_instruction": instruction,
                    "user_instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
                    "authorization_root": self.remote_root, "allowed_local_binding_root": self.local_root}
        self.bindings["partial_approval_path"] = "evidence/partial_approval.json"
        self.bindings["partial_approval_sha256"] = self.write_json(self.bindings["partial_approval_path"], approval)

    def pins(self):
        stack = ExitStack()
        stack.enter_context(patch.multiple(prepare.base, SOURCE=self.source_closure, HANDOFF=self.parent,
                                           TEST_COUNTS={dataset: 5 for dataset in prepare.base.DATASETS}))
        stack.enter_context(patch.multiple(prepare, SOURCE=self.source_closure, PARENT=self.parent,
                                           ORIGINAL_DESIGN_CLOSURE=self.design_closure,
                                           AUTHORIZED_LOCAL_ROOT=self.local_root, AUTHORIZED_REMOTE_ROOT=self.remote_root))
        return stack

    def build(self):
        with self.pins():
            return prepare.build_contract(self.root, self.design, self.bindings, self.runtimes)

    def reseal_snapshot_record(self, name, mutate):
        snapshot = prepare.read(self.root / self.bindings["handoffterminal_evidence_path"])
        value = json.loads(base64.b64decode(snapshot["files"][name]["base64"]))
        mutate(value)
        raw = json.dumps(value, sort_keys=True).encode()
        snapshot["files"][name] = {"base64": base64.b64encode(raw).decode(),
                                    "raw_sha256": hashlib.sha256(raw).hexdigest()}
        self.bindings["handoffterminal_evidence_sha256"] = self.write_json(
            self.bindings["handoffterminal_evidence_path"], snapshot)
        self.refresh_json(self.bindings, "training_monitor_path", "training_monitor_sha256",
            lambda value: value["groups"]["handoff_5080"].update(
                snapshot_raw_sha256=self.bindings["handoffterminal_evidence_sha256"]))


class Inter62PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="inter62_preparation_synthetic_", dir="/private/tmp")
        self.addCleanup(self.temporary.cleanup)
        self.fixture = Inter62Fixture(self.temporary.name)

    @property
    def job(self):
        return self.fixture.bindings["jobs"][0]

    def reject(self, message):
        with self.assertRaisesRegex(ValueError, message):
            self.fixture.build()

    def test_exact4_handoff_approval_preserves_separate_original36_evaluation_gate(self):
        contract = self.fixture.build()
        self.assertEqual(contract["schema"], prepare.KIND)
        self.assertEqual(contract["cohort"], prepare.COHORT)
        self.assertEqual(contract["expected_seeds"], [62])
        self.assertEqual(len(contract["jobs"]), 4)
        self.assertEqual(set(contract["runtimes"]), {"5080"})
        self.assertTrue(contract["original36_all_training_terminal_verified"])
        self.assertFalse(contract["original36_global_gate_modified"])
        self.assertFalse(contract["authorization"]["Test_authorized_by_binding"])
        self.assertTrue(contract["partial_cohort_explicitly_authorized"])

    def test_generated_contract_matches_real_subset_evaluator_static_guard(self):
        from paper.scripts import evaluate_titantpp_components_inter62_frozen as evaluator
        files = {"operational/inter62_evaluator.py": self.fixture.write_bytes(
            "operational/inter62_evaluator.py", Path(evaluator.__file__).read_bytes())}
        self.fixture.bindings["evaluator_integrity"] = {"files": files, "closure": prepare.canonical(files)}
        contract = self.fixture.build()
        self.fixture.write_json("contract.json", contract)
        with patch.multiple(evaluator, SOURCE_CLOSURE=self.fixture.source_closure, TRAINING_CONTRACT=self.fixture.parent):
            actual = evaluator.verify_contract(self.fixture.root, require_bound=True)
        self.assertEqual(actual["contract_sha256"], contract["contract_sha256"])
        self.assertEqual(actual["expected_jobs"], 4)

    def test_seed42_and_other_dataset_cannot_enter_seed62_registry(self):
        original = self.job["id"]
        for replacement in (original.replace("__62__", "__42__"), original.replace(prepare.DATASET, "raf_spare_parts")):
            with self.subTest(replacement=replacement):
                self.job["id"] = replacement
                self.reject("exactly4")

    def test_missing_or_duplicate_job_rejected(self):
        jobs = self.fixture.bindings["jobs"]
        self.fixture.bindings["jobs"] = jobs[:-1]
        self.reject("exactly4")
        self.fixture.bindings["jobs"] = jobs[:-1] + [deepcopy(jobs[0])]
        self.reject("exactly4")

    def test_test_cannot_select_checkpoint(self):
        self.fixture.refresh_json(self.job, "checkpoint_metadata_path", "checkpoint_metadata_sha256",
                                  lambda value: value.update(checkpoint_monitor="test_raw_quantity_rmse"))
        self.reject("Selected metadata mismatch")

    def test_original36_design_is_preserved(self):
        self.fixture.design["fresh_Test_condition_count"] = 8
        self.reject("Original36 frozen selection/metric design changed")

    def test_incomplete_global_training_cannot_bind_seed62(self):
        self.fixture.refresh_json(self.fixture.bindings, "training_monitor_path", "training_monitor_sha256",
            lambda value: value.update(all_training_terminal_verified=False,
                totals={"total": 36, "completed": 35, "running": 1, "waiting": 0, "unconfirmed": 0,
                        "all_training_terminal_verified": False}))
        self.reject("terminal observation is incomplete")

    def test_missing_or_changed_selected_binary_rejected(self):
        (self.fixture.root / self.job["checkpoint"]["relative_path"]).write_bytes(b"changed selected synthetic bytes")
        self.reject("Artifact SHA mismatch")

    def test_scientific_worker_exit_nonzero_rejected(self):
        self.fixture.refresh_json(self.job, "worker_exit_path", "worker_exit_sha256",
                                  lambda value: value.update(returncode=1))
        self.reject("Actual scientific worker exit0")

    def test_handoff_terminal_flag_cannot_replace_native_dispatcher_exit0(self):
        self.fixture.reseal_snapshot_record("phases/train_exit.json", lambda value: value.update(os_exit_code=1))
        self.reject("dispatcher actual exit0/phase complete")

    def test_phase_complete_must_have_actual_zero_exit(self):
        self.fixture.reseal_snapshot_record("phases/train_complete.json",
                                           lambda value: value.update(subprocess_os_exit_code=1))
        self.reject("dispatcher actual exit0/phase complete")

    def test_dispatcher_completed_registry_cannot_skip_a_native_worker(self):
        self.fixture.reseal_snapshot_record("status.json",
                                           lambda value: value["completed"].pop(self.job["id"]))
        self.reject("dispatcher actual exit0/phase complete")

    def test_recovered_exit_and_native_snapshot_must_have_same_original_bytes(self):
        self.fixture.reseal_snapshot_record("worker_exits/" + self.job["id"] + ".json",
                                           lambda value: value.update(synthetic_tampering=True))
        self.reject("Recovered worker/terminal bytes differ")

    def test_endpoint_self_sha_cannot_replace_original_terminal_binding(self):
        self.fixture.refresh_json(self.job, "validation_endpoint_path", "validation_endpoint_sha256",
                                  lambda value: value["selected"]["tail"].update(qty_rmse=100.0))
        self.reject("Validation endpoint not bound by original terminal")

    def test_reference_cannot_replace_original_endpoint_metrics(self):
        self.job["validation_reference"]["tail"]["qty_rmse"] = 100.0
        self.reject("Validation reference differs")

    def test_partial_approval_false_or_foreign_roots_rejected(self):
        self.fixture.refresh_json(self.fixture.bindings, "partial_approval_path", "partial_approval_sha256",
                                  lambda value: value.update(partial_cohort_explicitly_authorized=False))
        self.reject("explicit partial approval")
        self.fixture.refresh_json(self.fixture.bindings, "partial_approval_path", "partial_approval_sha256",
                                  lambda value: value.update(partial_cohort_explicitly_authorized=True,
                                                            authorization_root="/synthetic/foreign_remote"))
        self.reject("outside the authorized local/remote pair")

    def test_partial_instruction_byte_sha_required(self):
        self.fixture.refresh_json(self.fixture.bindings, "partial_approval_path", "partial_approval_sha256",
                                  lambda value: value.update(user_instruction="different instruction"))
        self.reject("Original user instruction SHA differs")

    def test_completed_row_worker_sha_must_equal_recovered_original(self):
        self.fixture.refresh_json(self.fixture.bindings, "training_monitor_path", "training_monitor_sha256",
            lambda value: value["groups"]["handoff_5080"]["jobs"][0].update(worker_exit_raw_sha256="f" * 64))
        self.reject("worker/binary proof differs")

    def test_parent_owner_absence_cannot_be_inferred_from_empty_processes(self):
        self.fixture.refresh_json(self.fixture.bindings, "handoffterminal_evidence_path", "handoffterminal_evidence_sha256",
                                  lambda value: value.pop("process_observations"))
        changed = self.fixture.bindings["handoffterminal_evidence_sha256"]
        self.fixture.refresh_json(self.fixture.bindings, "training_monitor_path", "training_monitor_sha256",
            lambda value: value["groups"]["handoff_5080"].update(snapshot_raw_sha256=changed))
        self.reject("probes missing")

    def test_dataset_statistics_cannot_change(self):
        self.fixture.bindings["datasets"][prepare.DATASET]["definition"]["model"]["train_mean"] += 1
        self.reject("statistics/model/preprocessing differ")

    def test_additional_runtime_not_approved(self):
        self.fixture.runtimes["5090"] = deepcopy(self.fixture.runtimes["5080"])
        self.reject("Only native5080")

    def test_cli_writes_only_immutable_contract_without_permit(self):
        values = {"design": self.fixture.design, "bindings": self.fixture.bindings, "runtimes": self.fixture.runtimes}
        argv = ["--root", str(self.fixture.root)]
        for name, value in values.items():
            self.fixture.write_json(name + ".json", value)
            argv += ["--" + name, str(self.fixture.root / (name + ".json"))]
        before = {path.relative_to(self.fixture.root) for path in self.fixture.root.rglob("*")}
        stdout = io.StringIO()
        with self.fixture.pins(), redirect_stdout(stdout):
            prepare.main(argv)
        after = {path.relative_to(self.fixture.root) for path in self.fixture.root.rglob("*")}
        self.assertEqual(after - before, {Path("contract.json")})
        self.assertFalse(json.loads(stdout.getvalue())["Test_permit_created"])
        self.assertTrue(json.loads(stdout.getvalue())["original36_all_training_terminal_verified"])
        with self.fixture.pins(), self.assertRaises(FileExistsError), redirect_stdout(io.StringIO()):
            prepare.main(argv)


if __name__ == "__main__":
    unittest.main()
