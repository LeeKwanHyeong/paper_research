"""Operational regression checks using fabricated metadata and mocked network calls.

No research rows, historical results, checkpoints, SSH, or GPU are accessed.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from paper.scripts import titantpp_contribution_runtime as runtime
from paper.scripts import deploy_titantpp_contribution_campaigns as deploy
from paper.scripts import finalize_titantpp_contribution_campaigns as finalize
from paper.scripts import observe_titantpp_contribution_campaigns as observe
from paper.scripts import prepare_titantpp_contribution_campaigns as prepare


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def small_contract(stage="instacart"):
    """A fabricated scope, with hashes that cannot describe a research source."""
    dataset_names = ("insta_market_basket",) if stage == "instacart" else (
        "yellow_trip_hourly", "intermittent_frozen_5000", "raf_spare_parts")
    groups = {"5080": [[d, s] for d in dataset_names for s in runtime.SEEDS], "5090": []}
    parent = {f"fake_source_{i}.py": f"fake_digest_{i}" for i in range(126)}
    datasets = [{"dataset_id": d} for d in dataset_names]
    c = {"schema": "titantpp_contribution_" + stage + "_20261008_v1",
         "stage": stage, "arms": list(runtime.STAGES[stage]),
         "training": deepcopy(runtime.TRAINING), "assignments": groups,
         "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
         "limits": {"total_wall_seconds": 168 * 3600, "per_condition_seconds": 36 * 3600,
                    "workers_per_host": 1, "automatic_retry": False},
         "source": {"parent_files": parent, "files": deepcopy(parent),
                    "files_sha256": runtime.sha_json(parent)},
         "operation": {"files": {}, "files_sha256": runtime.sha_json({})},
         "datasets": datasets, "dataset_sha256": {d["dataset_id"]: runtime.sha_json(d) for d in datasets},
         "hosts": {}, "prior_stage": None}
    for host in groups:
        root = "/owned/" + c["schema"] + "_" + host
        c["hosts"][host] = {"root": root, "source_root": root + "/source",
                             "operation_root": root + "/operation", "python": "/existing/python",
                             "gpu_uuid": "GPU-fake-" + host}
    c["jobs"] = runtime.expected_jobs(c)
    return c


def fabricated_owner():
    c = small_contract()
    job = c["jobs"][0]
    c["jobs"] = [job]
    root = c["hosts"]["5080"]["root"]
    common = ["/existing/python", root + "/operation/campaign.py", "--contract",
              root + "/execution_contract.json", "--host", "5080"]
    worker_args = common + ["--mode", "fit", "--job", job["id"], "--deadline", "1000.0"]
    processes = [{"pid": 100, "ppid": 99, "argv": common + ["--mode", "dispatch"]},
                 {"pid": 101, "ppid": 100, "argv": ["timeout", "--signal=TERM", "--kill-after=15s", "900"] + worker_args},
                 {"pid": 102, "ppid": 101, "argv": worker_args}]
    raw = {"stage": "instacart", "host": "5080", "contract_sha256": runtime.sha_json(c),
           "observed_unix": 100., "GPU": "fabricated", "gpu_processes": "GPU-fake-5080, 102\n",
           "processes": processes, "status": {"active_job": job, "supervisor_pid": 100, "worker_group_pid": 101,
               "supervisor_argv": deepcopy(processes[0]["argv"]), "worker_guard_argv": deepcopy(processes[1]["argv"]),
               "condition_deadline_unix": 1000., "deadline_unix": 10000.}, "failure": None,
           "supervisor_exit": None, "lease": {"issued_unix": 95., "expires_unix": 185., "worker_group_pid": 101,
               "supervisor_pid": 100, "job": job["id"], "contract_sha256": runtime.sha_json(c)}, "jobs": {job["id"]: {
               "job": job, "status": {"status": "running"}, "claim": {"job": job, "deadline_unix": 1000.,
               "contract_sha256": runtime.sha_json(c)}, "terminal": None, "checkpoint_receipt": None}}, "rawSHA_verified": True}
    return c, job, raw


class CampaignContractTests(unittest.TestCase):
    def validate(self, c):
        with patch.object(runtime, "PARENT_CLOSURE", runtime.sha_json(c["source"]["parent_files"])):
            return runtime.validate(c, verify_files=False)

    def test_three_stages_hold_63_fresh_conditions_with_no_mean_retraining(self):
        counts = []
        for stage in runtime.STAGES:
            c = small_contract(stage)
            self.validate(c)
            counts.append(len(c["jobs"]))
        self.assertEqual(counts, [9, 27, 27])
        self.assertEqual(sum(counts), 63)
        self.assertNotIn("titantpp_cnn_gru54", runtime.STAGES["retrieval"])

    def test_duplicate_owner_is_rejected_even_after_rebuilding_job_list(self):
        c = small_contract()
        c["assignments"]["5090"] = [c["assignments"]["5080"][0]]
        c["jobs"] = runtime.expected_jobs(c)
        with self.assertRaisesRegex(ValueError, "Job ownership drift"):
            self.validate(c)

    def test_changed_source_parent_is_rejected_even_when_new_closure_is_consistent(self):
        c = small_contract()
        c["source"]["files"]["fake_source_0.py"] = "changed"
        c["source"]["files_sha256"] = runtime.sha_json(c["source"]["files"])
        with self.assertRaisesRegex(ValueError, "Frozen scientific source overwritten"):
            self.validate(c)

    def test_budget_or_test_access_change_is_not_silent(self):
        for key in ("budget", "test"):
            c = small_contract()
            if key == "budget":
                c["limits"]["automatic_retry"] = True
            else:
                c["held_out_test_evaluated"] = True
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(c)

    def test_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("../outside", "/absolute"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    runtime.safe_relative(directory, name)

    def test_width8_and_width12_interface_reports_actual_requested_width(self):
        from paper.scripts import run_titantpp_history_width as width
        for n in (8, 12):
            arm = "titantpp_history_mlp_width" + str(n)
            with self.subTest(width=n), patch.dict(runtime.engines, {arm: width}, clear=True), \
                    patch.object(width.base, "time_interface", return_value={"fabricated": True}):
                result = runtime.time_interface({}, None, {"training": runtime.TRAINING}, arm)
            metadata = result["backbone_design"]["model_metadata"]
            self.assertIn(str(n), result["backbone_design"]["intervention"])
            self.assertEqual(metadata["history_mlp_width"], n)


class ActivationBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.c = small_contract()
        self.q = {"status": "passed", "contract_sha256": runtime.sha_json(self.c)}
        self.q_text = json.dumps(self.q)
        self.q_sha = __import__("hashlib").sha256(self.q_text.encode()).hexdigest()
        save(self.root / "start_permit_5080.json", {"fabricated_start": True})

    def observation(self, skew=2., age=0.):
        return {"host": "5080", "observed_unix": 100. + skew - age,
                "local_received_unix": 100. - age, "receipt_text": self.q_text,
                "receipt_sha256": self.q_sha, "exit": {"returncode": 0},
                "failure": None, "owned_processes": [], "gpu_processes": ""}

    def launch_with_mocked_remote(self, out):
        with patch.object(deploy, "bundle", return_value=self.root), \
                patch.object(deploy.time, "time", return_value=100.), \
                patch.object(deploy, "command", return_value=["fabricated_launch"]), \
                patch.object(deploy, "prefix", return_value=""), \
                patch.object(deploy, "remote", return_value={"status": "training_launched"}) as remote:
            result = deploy.launch(self.c, "5080", out)
            remote.assert_called_once()
            return result

    def test_local_received_clock_allows_server_two_seconds_ahead(self):
        self.assertEqual(self.launch_with_mocked_remote(self.observation())["status"], "training_launched")
        self.assertTrue((self.root / "deployment/launch_intent_5080.json").exists())

    def test_skew_over_ten_seconds_blocks_before_intent_and_remote_call(self):
        with patch.object(deploy, "remote") as remote, self.assertRaises(AssertionError):
            self.launch_with_mocked_remote(self.observation(skew=10.001))
        remote.assert_not_called()
        self.assertFalse((self.root / "training_permit_5080.json").exists())
        self.assertFalse((self.root / "deployment/launch_intent_5080.json").exists())

    def test_stale_local_receipt_blocks_before_intent(self):
        with self.assertRaises(AssertionError):
            self.launch_with_mocked_remote(self.observation(age=600.))
        self.assertFalse((self.root / "deployment/launch_intent_5080.json").exists())

    def test_collect_qualification_checks_raw_binding_and_local_clock(self):
        with patch.object(deploy, "bundle", return_value=self.root), \
                patch.object(deploy, "prefix", return_value=""), \
                patch.object(deploy, "remote", return_value=self.observation()) as remote:
            result = deploy.collect_qualification(self.c, "5080")
        remote.assert_called_once()
        self.assertEqual(runtime.sha_file(self.root / "qualification/5080/receipt.json"), self.q_sha)
        self.assertEqual(result["local_received_unix"], 100.)

    def test_collect_rejects_raw_receipt_hash_drift(self):
        out = self.observation()
        out["receipt_sha256"] = "changed"
        with patch.object(deploy, "bundle", return_value=self.root), \
                patch.object(deploy, "prefix", return_value=""), \
                patch.object(deploy, "remote", return_value=out), self.assertRaises(AssertionError):
            deploy.collect_qualification(self.c, "5080")
        self.assertFalse((self.root / "qualification/5080/receipt.json").exists())

    def test_dependency_without_completed_originals_has_zero_remote_calls(self):
        c = {"prior_stage": {"bundle": "missing_previous"}}
        with patch.object(deploy, "PROJECT", self.root), patch.object(deploy, "remote") as remote:
            self.assertFalse(deploy.dependency_ready(c))
        remote.assert_not_called()

    def test_completed_dependency_detects_original_bytes_changed(self):
        prior = self.root / "previous"
        (prior / "analysis").mkdir(parents=True)
        (prior / "archive").write_bytes(b"original")
        save(prior / "analysis/completion_receipt.json", {
            "contract_sha256": "previous_sha", "completed_conditions": 9,
            "actual_owner_shutdown_verified": True, "originals_rawSHA_verified": True,
            "final_Validation_report_verified": True,
            "local_files": {"archive": runtime.sha_file(prior / "archive")}})
        c = {"prior_stage": {"bundle": "previous", "contract_sha256": "previous_sha", "jobs": 9}}
        with patch.object(deploy, "PROJECT", self.root):
            self.assertTrue(deploy.dependency_ready(c))
            (prior / "archive").write_bytes(b"changed")
            with self.assertRaisesRegex(AssertionError, "Prior originals/report drift"):
                deploy.dependency_ready(c)

    def activate(self, **patches):
        stack = __import__("contextlib").ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(deploy, "local_contract", return_value=(self.root, self.c)))
        for name, value in patches.items():
            stack.enter_context(patch.object(deploy, name, value))
        return deploy.activate_host("instacart", "5080")

    def test_blocked_stage_returns_pending_before_any_remote_action(self):
        actions = {name: Mock() for name in ("deploy", "start_qualification", "collect_qualification", "launch")}
        result = self.activate(dependency_ready=Mock(return_value=False), **actions)
        self.assertEqual(result["remote_calls"], 0)
        for action in actions.values():
            action.assert_not_called()

    def test_already_launched_is_local_only(self):
        save(self.root / "deployment/launch_receipt_5080.json", {"preserved": True})
        remote = Mock()
        result = self.activate(dependency_ready=Mock(return_value=True), remote=remote)
        self.assertEqual(result["status"], "already_launched")
        remote.assert_not_called()

    def test_incomplete_transfer_qualification_or_launch_intent_never_retries(self):
        for phase in ("transfer", "qualification", "launch"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                if phase != "transfer":
                    save(root / "deployment/deployment_receipt_5080.json", {})
                if phase == "launch":
                    save(root / "deployment/qualification_start_5080.json", {})
                save(root / f"deployment/{phase}_intent_5080.json", {"preserved": True})
                with patch.object(deploy, "local_contract", return_value=(root, self.c)), \
                        patch.object(deploy, "dependency_ready", return_value=True), \
                        patch.object(deploy, "remote") as remote, self.assertRaisesRegex(AssertionError, "Preserved incomplete"):
                    deploy.activate_host("instacart", "5080")
                remote.assert_not_called()


class ObservationTests(unittest.TestCase):
    def test_eta_uses_latest_ten_completed_epochs_from_this_fit(self):
        timing = {"epochs": [{"epoch": n, "elapsed_seconds": float(n)} for n in range(1, 21)]}
        timing["epochs"] += [{"epoch": 21, "elapsed_seconds": 99999.},
                             {"epoch": 20, "elapsed_seconds": -1.}]
        result = observe.eta(timing, 20, 13)
        self.assertEqual(result["sample_epochs"], 10)
        self.assertEqual(result["median_epoch_seconds"], 15.5)
        self.assertEqual(result["no_new_best_end_epoch"], 53)
        self.assertEqual(result["no_new_best_remaining_seconds"], 33 * 15.5)
        self.assertEqual(result["max300_remaining_seconds"], 280 * 15.5)

    def test_eta_has_no_cross_arm_fallback_when_own_timing_absent(self):
        self.assertEqual(observe.eta({}, 20, 13)["status"], "unconfirmed_no_own_completed_epoch_timing")

    def test_eta_filters_nonfinite_completed_times(self):
        timing = {"epochs": [{"epoch": 1, "elapsed_seconds": 2.},
                             {"epoch": 2, "elapsed_seconds": float("inf")},
                             {"epoch": 3, "elapsed_seconds": float("nan")}]}
        result = observe.eta(timing, 3, 1)
        self.assertEqual(result["sample_epochs"], 1)
        self.assertEqual(result["median_epoch_seconds"], 2.)

    def test_verified_terminal_cache_retains_original_time_and_uses_no_ssh(self):
        c = small_contract()
        with tempfile.TemporaryDirectory() as directory:
            b = Path(directory)
            cached = {"contract_sha256": runtime.sha_json(c), "verified_terminal": True,
                      "observed_unix": 42., "observed_kst": "original observation"}
            save(b / "hourly_monitor/terminal_cache_5080.json", cached)
            save(b / "hourly_monitor/terminal_pointer_5080.json", {
                "sha256": runtime.sha_file(b / "hourly_monitor/terminal_cache_5080.json")})
            with patch.object(observe, "local_contract", return_value=(b, c)), \
                    patch.object(observe, "remote") as remote:
                result = observe.observe_host("instacart", "5080")
            self.assertEqual(result["observed_unix"], 42.)
            self.assertEqual(result["observed_kst"], "original observation")
            self.assertEqual(result["remote_calls"], 0)
            remote.assert_not_called()

    def test_worker_gpu_pid_is_child_of_timeout_not_timeout_pid(self):
        c, job, raw = fabricated_owner()
        with tempfile.TemporaryDirectory() as directory:
            b = Path(directory)
            save(b / "deployment/launch_receipt_5080.json", {})
            with patch.object(observe, "local_contract", return_value=(b, c)), \
                    patch.object(observe, "prefix", return_value=""), \
                    patch.object(observe, "remote", return_value=raw):
                result = observe.observe_host("instacart", "5080")
            self.assertEqual(result["counts"]["running"], 1)
            self.assertEqual(result["counts"]["unconfirmed"], 0)
            self.assertEqual(result["active"]["worker_PID"], 102)
            self.assertTrue(result["active"]["first_checkpoint_not_saved"])

    def test_exact_worker_argv_and_owner_lease_reject_unrelated_processes(self):
        def check(c, job, raw):
            supervisor, timeout, worker = raw["processes"]
            return observe.owned_processes_match(c, "5080", job,
                raw["jobs"][job["id"]]["claim"], raw, raw["status"], timeout, supervisor, worker)

        c, job, raw = fabricated_owner()
        self.assertTrue(check(c, job, raw))
        for kind in ("python", "host", "contract", "claim", "deadline", "expired_lease", "foreign_lease", "wrong_parent"):
            c, job, raw = fabricated_owner()
            worker = raw["processes"][2]
            if kind == "python":
                worker["argv"][0] = "/other/python"
            elif kind in ("host", "contract"):
                flag = "--host" if kind == "host" else "--contract"
                worker["argv"][worker["argv"].index(flag) + 1] = "foreign"
            elif kind == "claim":
                raw["jobs"][job["id"]]["claim"]["contract_sha256"] = "foreign"
            elif kind == "deadline":
                raw["status"]["condition_deadline_unix"] = 1001.
            elif kind == "expired_lease":
                raw["lease"]["expires_unix"] = raw["observed_unix"]
            elif kind == "foreign_lease":
                raw["lease"]["job"] = "foreign"
            elif kind == "wrong_parent":
                worker["ppid"] = 103
            with self.subTest(kind=kind):
                self.assertFalse(check(c, job, raw))


class FinalizationTests(unittest.TestCase):
    @staticmethod
    def paired(host="5080"):
        return {"host": host, "dataset": "fabricated", "seed": 42,
                "train_exposure": {"train": [{"epoch": 1, "batch_order_sha256": "train1"},
                                             {"epoch": 2, "batch_order_sha256": "train2"}],
                                   "validation": [{"batch_order_sha256": "val1"}]}}

    def test_pairing_accepts_different_fit_lengths_with_same_completed_prefix(self):
        a, b = self.paired(), self.paired()
        b["train_exposure"]["train"] = b["train_exposure"]["train"][:1]
        finalize.validate_pair_exposure(a, b)

    def test_pairing_rejects_training_validation_or_owner_drift(self):
        for kind in ("train", "validation", "owner"):
            a, b = self.paired(), self.paired()
            if kind == "owner":
                b["host"] = "5090"
            else:
                b["train_exposure"][kind][0]["batch_order_sha256"] = "changed"
            with self.subTest(kind=kind), self.assertRaises(AssertionError):
                finalize.validate_pair_exposure(a, b)

    def test_pending_finalization_is_local_and_does_not_claim_or_retrieve(self):
        c = small_contract()
        with tempfile.TemporaryDirectory() as directory:
            b = Path(directory)
            with patch.object(finalize, "local_contract", return_value=(b, c)), \
                    patch.object(finalize, "terminal_cache", return_value=None), \
                    patch.object(finalize, "retrieve_host") as retrieve, patch.object(finalize, "remote") as remote:
                result = finalize.finalize("instacart")
            self.assertEqual(result["status"], "pending")
            self.assertEqual(result["remote_calls"], 0)
            self.assertFalse((b / "retrieval/finalize.claim").exists())
            retrieve.assert_not_called()
            remote.assert_not_called()

    def test_preserved_finalization_claim_blocks_retry(self):
        c = small_contract()
        with tempfile.TemporaryDirectory() as directory:
            b = Path(directory)
            save(b / "retrieval/finalize.claim", {"preserved": True})
            with patch.object(finalize, "local_contract", return_value=(b, c)), \
                    patch.object(finalize, "terminal_cache", return_value={"verified": True}), \
                    patch.object(finalize, "retrieve_host") as retrieve:
                result = finalize.finalize("instacart")
            self.assertEqual(result["status"], "claimed_or_failed_preserved")
            self.assertTrue(result["user_action_required"])
            retrieve.assert_not_called()

    def test_completed_finalization_only_revalidates_local_original_hashes(self):
        c = small_contract()
        with tempfile.TemporaryDirectory() as directory:
            b = Path(directory)
            (b / "archive").write_bytes(b"fabricated immutable original")
            save(b / "analysis/completion_receipt.json", {"contract_sha256": runtime.sha_json(c),
                "local_files": {"archive": runtime.sha_file(b / "archive")}, "decision": {"development_support": False}})
            with patch.object(finalize, "local_contract", return_value=(b, c)), \
                    patch.object(finalize, "remote") as remote, patch.object(finalize, "retrieve_host") as retrieve:
                result = finalize.finalize("instacart")
                self.assertEqual(result["remote_calls"], 0)
                remote.assert_not_called()
                retrieve.assert_not_called()
                (b / "archive").write_bytes(b"changed")
                with self.assertRaisesRegex(AssertionError, "Finalized local drift"):
                    finalize.finalize("instacart")

    def test_writer_preserves_existing_intent_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "claim.json"
            prepare.write(p, {"first_attempt": True})
            original = p.read_bytes()
            with self.assertRaises(FileExistsError):
                prepare.write(p, {"retry": True})
            self.assertEqual(p.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
