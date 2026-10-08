"""Fail-closed checks before any remote call or GPU update."""
import json
from pathlib import Path
import tempfile
import unittest
from paper.scripts import deploy_titantpp_baseline_parity as deploy


class DeploymentBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bundle = Path(self.temp.name)
        self.contract = {
            "approval": {key: False for key in ["full_fit", "validation_inference", "test_inference", "expansion_training", "shared_runtime_change", "automatic_retry"]},
            "runtime": {"host": "5080"}, "real_train": {"max_updates": 3},
            "synthetic": {"lengths": [64, 256], "updates_per_length": 1},
            "resources": {"process_timeout_seconds": 900},
        }
        (self.bundle / "operation.py").write_text("# sealed operation\n")
        (self.bundle / "source/sample_data").mkdir(parents=True)
        self.seal()

    def seal(self):
        (self.bundle / "execution_contract.json").write_text(json.dumps(self.contract))
        self.manifest = {"contract_raw_sha256": deploy.sha(self.bundle / "execution_contract.json"), "files": {"operation.py": deploy.sha(self.bundle / "operation.py")}, "required_empty_directories": ["source/sample_data"]}
        (self.bundle / "deployment_manifest.json").write_text(json.dumps(self.manifest))

    def test_sealed_short_scope_passes(self):
        contract, _ = deploy.verify_bundle(self.bundle)
        self.assertEqual(contract["real_train"]["max_updates"], 3)

    def test_changed_operation_rejected(self):
        (self.bundle / "operation.py").write_text("# changed\n")
        with self.assertRaisesRegex(ValueError, "Sealed file changed"):
            deploy.verify_bundle(self.bundle)

    def test_scope_expansion_rejected_even_when_resealed(self):
        self.contract["approval"]["test_inference"] = True
        self.seal()
        with self.assertRaisesRegex(ValueError, "exceeds"):
            deploy.verify_bundle(self.bundle)

    def test_update_expansion_rejected(self):
        self.contract["real_train"]["max_updates"] = 4
        self.seal()
        with self.assertRaisesRegex(ValueError, "update budget"):
            deploy.verify_bundle(self.bundle)

    def test_path_traversal_rejected(self):
        self.manifest["files"] = {"../other": "invalid"}
        (self.bundle / "deployment_manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            deploy.verify_bundle(self.bundle)

    def test_existing_intent_cannot_be_overwritten(self):
        path = self.bundle / "intent.json"
        deploy.write_once(path, {"attempt": 1})
        with self.assertRaises(FileExistsError):
            deploy.write_once(path, {"attempt": 2})
        self.assertEqual(deploy.read(path), {"attempt": 1})

    def test_unsealed_or_nonempty_sentinel_rejected(self):
        (self.bundle / "source/sample_data/unexpected").write_text("not empty")
        with self.assertRaisesRegex(ValueError, "empty directory"):
            deploy.verify_bundle(self.bundle)


if __name__ == "__main__":
    unittest.main()
