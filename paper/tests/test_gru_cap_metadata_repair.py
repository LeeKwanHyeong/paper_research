"""Only the two legacy metadata fields may change in the local proposal."""
from copy import deepcopy
import unittest
from paper.scripts import prepare_titantpp_gru_cap_metadata_repair as repair


class MetadataRepairTests(unittest.TestCase):
    def setUp(self):
        self.original = {"candidate": {"rho": .25}, "training": {"maximum_epochs": 300},
            "limits": {"automatic_retry": False}, "hosts": {"5080": {"root": "/failed/root"}}}
        self.parent = {"design_sha256": "a" * 64, "architecture": {"original": [1, 2]},
            "operational_replacement": {"must_not_be_copied": True}, "assignments": {"old": True}}

    def test_adds_only_legacy_metadata_without_changing_inputs(self):
        before = deepcopy(self.original)
        proposed = repair.propose_contract(self.original, self.parent)
        self.assertEqual(self.original, before)
        self.assertEqual(set(proposed) - set(before), set(repair.KEYS))
        self.assertNotIn("operational_replacement", proposed)
        self.assertNotIn("assignments", proposed)
        proposed["architecture"]["original"].append(3)
        self.assertEqual(self.parent["architecture"]["original"], [1, 2])

    def test_rejects_science_retry_or_root_drift(self):
        for key, nested, value in (("candidate", "rho", .5), ("training", "maximum_epochs", 301),
                                   ("limits", "automatic_retry", True), ("hosts", "5080", {"root": "/new"})):
            proposed = repair.propose_contract(self.original, self.parent)
            proposed[key][nested] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                repair.validate_delta(self.original, proposed, self.parent)

    def test_rejects_duplicate_repair_and_unbound_metadata(self):
        proposed = repair.propose_contract(self.original, self.parent)
        with self.assertRaises(ValueError):
            repair.propose_contract(proposed, self.parent)
        proposed["design_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            repair.validate_delta(self.original, proposed, self.parent)


if __name__ == "__main__":
    unittest.main()
