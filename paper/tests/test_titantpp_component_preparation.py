"""Local preparation and one-attempt phase boundaries; no SSH or real fits."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from paper.scripts import prepare_titantpp_cnn_gru_components as preparation
from paper.scripts import run_titantpp_cnn_gru_component_phase as phase


class PreparationBoundaries(unittest.TestCase):
    def test_36_fresh_ids_preserve_dataset_owners_and_old_science(self):
        parent = preparation.read(preparation.PARENT / "execution_contract.json")
        c = preparation.contract(parent)
        self.assertEqual(len(c["jobs"]), 36)
        self.assertEqual(len({j["id"] for j in c["jobs"]}), 36)
        self.assertEqual([sum(j["host"] == h for j in c["jobs"]) for h in ("5080", "5090")], [24, 12])
        for dataset in (d["dataset_id"] for d in c["datasets"]):
            self.assertEqual(len({j["host"] for j in c["jobs"] if j["dataset"] == dataset}), 1)
            self.assertEqual(sum(j["dataset"] == dataset for j in c["jobs"]), 12)
        for name, digest in parent["source"]["files"].items():
            self.assertEqual(c["source"]["files"][name], digest)
        for field in ("datasets", "dataset_sha256", "training", "input_files", "diagnostic", "policy"):
            self.assertEqual(c[field], parent[field])
        self.assertTrue(c["launch_policy"]["fresh_user_start_instruction_required"])
        self.assertTrue(c["launch_policy"]["all_hosts_qualification_before_any_fit"])

    def test_parent_contract_drift_rejects_before_destination_write(self):
        parent = preparation.read(preparation.PARENT / "execution_contract.json")
        bad = deepcopy(parent)
        bad["training"]["warm_start"] = True
        with tempfile.TemporaryDirectory() as directory, patch.object(preparation, "read", return_value=bad):
            target = Path(directory) / "bundle"
            with self.assertRaisesRegex(ValueError, "Original contract"):
                preparation.prepare(target)
            self.assertFalse(target.exists())

    def test_path_escape_and_external_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            root.mkdir()
            outside = Path(directory) / "outside"
            outside.write_text("original")
            (root / "alias").symlink_to(outside)
            for name in ("../outside", str(outside), "alias"):
                with self.assertRaises(ValueError):
                    preparation.safe(root, name)
            self.assertEqual(outside.read_text(), "original")


class PhaseBoundaries(unittest.TestCase):
    def test_claim_and_exit_receipts_are_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "intent.json"
            phase.write(target, {"attempt": 1})
            with self.assertRaises(FileExistsError):
                phase.write(target, {"attempt": 2})
            self.assertEqual(json.loads(target.read_text()), {"attempt": 1})

    def test_old_approval_cannot_launch_new_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "source").mkdir()
            (root / "operation").mkdir()
            spec = {"root": str(root), "python": phase.sys.executable,
                    "environment": {}, "gpu_uuid": "fixed-test-gpu"}
            c = {"hosts": {"5080": spec, "5090": dict(spec)}}
            approval = {"approved": True, "contract_sha256": phase.canonical(c),
                        "hosts": ["5080", "5090"]}
            phase.write(root / "approval.json", approval)
            with patch.object(phase, "__file__", str(root / "operation/pipeline.py")), \
                    patch.object(phase.Path, "cwd", return_value=root / "source"), \
                    patch.object(phase.subprocess, "Popen") as child, \
                    patch.object(phase, "gpu_pids") as gpu:
                with self.assertRaisesRegex(ValueError, "Explicit dual-host approval"):
                    phase.run_phase(c, "5080", "qualify")
                child.assert_not_called()
                gpu.assert_not_called()


if __name__ == "__main__":
    unittest.main()
