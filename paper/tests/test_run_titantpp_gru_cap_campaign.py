"""Local scope guards for the isolated cap9 operational campaign."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from paper.scripts import run_titantpp_gru_cap_campaign as campaign


def contract(tmp):
    datasets = [{"dataset_id": "yellow_trip_hourly"}, {"dataset_id": "raf_spare_parts"},
                {"dataset_id": "intermittent_frozen_5000"}]
    jobs = []
    for dataset in datasets[:2]:
        for seed in (42, 52, 62):
            jobs.append({"id": f"{dataset['dataset_id']}__{seed}__{campaign.CAPPED}", "host": "5080",
                         "dataset": dataset["dataset_id"], "seed": seed, "arm": campaign.CAPPED})
    for seed in (42, 52):
        jobs.append({"id": f"intermittent_frozen_5000__{seed}__{campaign.CAPPED}", "host": "5090",
                     "dataset": "intermittent_frozen_5000", "seed": seed, "arm": campaign.CAPPED})
    jobs.append({"id": f"intermittent_frozen_5000__62__{campaign.CAPPED}", "host": "5080",
                 "dataset": "intermittent_frozen_5000", "seed": 62, "arm": campaign.CAPPED})
    hosts = {}
    for host in ("5080", "5090"):
        root = Path(tmp) / host
        hosts[host] = {"root": str(root), "source_root": str(root / "source"),
            "operation_root": str(root / "operation"), "python": "/python", "gpu_uuid": "gpu-" + host,
            "runtime_expected": {}, "environment": {}}
    source = {"files": {}}
    source["files_sha256"] = campaign.sha_json(source["files"])
    operation = {"files": {}}
    operation["files_sha256"] = campaign.sha_json(operation["files"])
    return {"schema": "titantpp_gru_residual_cap_validation_20261014_v1", "candidate": {"arm": campaign.CAPPED},
        "science_preparation": {"path": "preparation/execution_contract.json", "raw_sha256": campaign.PREPARATION_RAW,
            "canonical_sha256": campaign.PREPARATION_CANONICAL},
        "arms": [campaign.CAPPED], "seeds": [42, 52, 62], "training": deepcopy(campaign.TRAINING),
        "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
        "execution_authorization": {"approved": True, "fresh_fit_count": 9, "Test": False,
            "automatic_retry": False, "preserve_original_CNN_FULL": True},
        "runner_modules": {campaign.CAPPED: "paper.scripts.run_titantpp_gru_residual_cap"},
        "parallelism": {"mode": "sequential_per_host", "workers_per_host": 1},
        "limits": {"per_condition_seconds": 36 * 3600, "total_wall_seconds": 168 * 3600,
            "automatic_retry": False}, "datasets": datasets, "jobs": jobs, "source": source,
        "input_files": {}, "operation": operation, "hosts": hosts}


class CapCampaignGuardsTests(unittest.TestCase):
    def test_valid_contract_is_cap_only_and_has_exact_host_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            current = contract(tmp)
            self.assertIs(campaign.validate(current, verify_files=False), current)

    def test_rejects_test_scope_retries_or_old_reference_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            current = contract(tmp)
            for mutate in (
                lambda c: c.__setitem__("evaluation_scope", "test"),
                lambda c: c["execution_authorization"].__setitem__("Test", True),
                lambda c: c["limits"].__setitem__("automatic_retry", True),
                lambda c: c["jobs"][0].__setitem__("arm", "titantpp_cnn_gru54"),
            ):
                bad = deepcopy(current)
                mutate(bad)
                with self.assertRaises(ValueError):
                    campaign.validate(bad, verify_files=False)

    def test_job_lookup_rejects_foreign_host_and_duplicate_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            current = contract(tmp)
            job = current["jobs"][0]
            self.assertEqual(campaign._host_job(current, "5080", job["id"]), job)
            with self.assertRaises(ValueError):
                campaign._host_job(current, "5090", job["id"])
            current["jobs"].append(deepcopy(job))
            with self.assertRaises(ValueError):
                campaign._host_job(current, "5080", job["id"])

    def test_worker_exit_uses_recorded_scientific_pid_not_timeout_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            current = contract(tmp)
            job = current["jobs"][0]
            identity = Path(current["hosts"]["5080"]["root"]) / "worker_identities" / (job["id"] + ".json")
            campaign.write_json(identity, {"pid": 1234, "ppid": 5678}, exclusive=True)
            class Child:
                pid, returncode = 9999, 0
            exit_record = campaign._observe_exit(Child(), ["timeout"], current, job, 111)
            self.assertEqual(exit_record["worker_group_pid"], 9999)
            self.assertEqual((exit_record["scientific_worker_pid"], exit_record["scientific_worker_ppid"]), (1234, 5678))

    def test_missing_worker_identity_preserves_guard_exit_without_false_scientific_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            current = contract(tmp)
            job = current["jobs"][0]
            class Child:
                pid, returncode = 9999, 1
            exit_record = campaign._observe_exit(Child(), ["timeout"], current, job, 111)
            self.assertTrue(exit_record["actual_guard_exit_only"])
            self.assertNotIn("pid", exit_record)
            self.assertEqual(exit_record["returncode"], 1)

    def test_qualification_interrupt_cleanup_records_real_guard_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            current = contract(tmp)
            root = Path(current["hosts"]["5080"]["root"])
            class Child:
                pid, returncode = 4321, -15
                def poll(self):
                    return self.returncode
            with patch.object(campaign, "stop_owned") as stop:
                receipt = campaign._cleanup_qualification_child(root, current, Child(), ["timeout"])
            stop.assert_called_once()
            self.assertEqual(receipt["status"], "interrupted")
            self.assertTrue(receipt["actual_child_exit_observed"])
            self.assertEqual(receipt["returncode"], -15)
            self.assertEqual(campaign.read(root / "qualification" / "native_exit.json"), receipt)


if __name__ == "__main__":
    unittest.main()
