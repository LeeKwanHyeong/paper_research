"""Independent-host operations preserve the completed v1 preparation."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from paper.scripts import prepare_titantpp_cnn_gru_components_v2 as preparation
from paper.scripts import run_titantpp_cnn_gru_component_phase_v2 as phase


class IndependentPreparation(unittest.TestCase):
    def test_all_science_and36ids_preserved_host_gate_only_changes(self):
        parent = preparation.read(preparation.PARENT / "execution_contract.json")
        c = preparation.contract(parent)
        for field in ("source", "input_files", "datasets", "dataset_sha256", "training", "diagnostic",
            "architecture", "analysis", "parallelism", "jobs", "assignments", "arms", "seeds", "policy"):
            self.assertEqual(c[field], parent[field])
        self.assertEqual(c["launch_policy"], preparation.POLICY)
        self.assertFalse(c["launch_policy"]["all_hosts_qualification_before_any_fit"])
        self.assertTrue(c["launch_policy"]["host_independent_start"])
        for host, spec in c["hosts"].items():
            self.assertTrue(spec["root"].endswith(preparation.SCHEMA + "_" + host))
            self.assertEqual(spec["python"], parent["hosts"][host]["python"])
            self.assertEqual(spec["runtime_expected"], parent["hosts"][host]["runtime_expected"])
        self.assertEqual(c["operation"]["files"]["operation/qualification.py"],
            parent["operation"]["files"]["operation/qualification.py"])
        self.assertEqual(c["operation"]["files"]["operation/frozen_runtime.py"],
            parent["operation"]["files"]["operation/frozen_runtime.py"])

    def test_parent_mutation_is_rejected(self):
        parent = preparation.read(preparation.PARENT / "execution_contract.json")
        parent["jobs"][0]["seed"] = 1
        with self.assertRaisesRegex(ValueError, "v1 contract changed"):
            preparation.contract(parent)

    def test_broad_two_host_approval_rejected_before_native_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "source").mkdir()
            (root / "operation").mkdir()
            spec = {"root": str(root), "python": phase.sys.executable, "environment": {}, "gpu_uuid": "fixed"}
            c = {"hosts": {"5080": spec, "5090": dict(spec)}}
            phase.write(root / "approval.json", {"approved": True, "contract_sha256": phase.canonical(c),
                "hosts": ["5080", "5090"], "explicit_new_user_instruction": True})
            with patch.object(phase, "__file__", str(root / "operation/pipeline.py")), \
                patch.object(phase.Path, "cwd", return_value=root / "source"), \
                patch.object(phase, "gpu_pids") as gpu, patch.object(phase.subprocess, "Popen") as child:
                with self.assertRaisesRegex(ValueError, "selected-host approval"):
                    phase.run_phase(c, "5080", "qualify")
                gpu.assert_not_called()
                child.assert_not_called()


if __name__ == "__main__":
    unittest.main()
