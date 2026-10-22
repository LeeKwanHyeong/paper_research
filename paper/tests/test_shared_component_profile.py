"""CPU-only contract tests for the Shared component cost profiler."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import torch

from paper.scripts import profile_shared_component_ablation as p


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(2.0))


class Hooks:
    def __init__(self): self.tick = 0; self.resets = 0; self.events = []
    def sync(self): self.events.append("sync")
    def clock(self): self.tick += 1; return float(self.tick)
    def transfer(self, batch): self.events.append("transfer"); return batch
    def reset_peak(self): self.resets += 1; self.events.append("reset")
    def memory(self): return {"allocated": 100, "reserved": 200}
    def peak(self): return {"allocated": 130 if self.resets else 150, "reserved": 250}


def batch(size=128):
    return {"dts": torch.ones(size, 2), "quantities": torch.ones(size, 2), "mask": torch.ones(size, 2, dtype=torch.bool)}


def output(model, value):
    return {"joint_loss": (model.weight * value["quantities"][:, -1]).square()}


class ProfileTests(unittest.TestCase):
    def test_actual_component_factory_keeps_frozen_joint_target_path(self):
        from models.TPPs.CountAwareComponentAblation import ARMS, build_model
        from paper.scripts import run_titantpp_shared_history_cap as fixture
        data = fixture.fabricated_data(); dts, quantities, mask = fixture.fabricated_batch()
        for arm in ARMS:
            with self.subTest(arm=arm):
                model, metadata = build_model(data, arm)
                value = p._target_outputs(model, {"dts": dts, "quantities": quantities, "mask": mask})
                self.assertIsInstance(metadata, dict)
                self.assertTrue(bool(torch.isfinite(value["joint_loss"]).all()))

    def test_raw_contract_sha_blocks_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"; path.write_text('{"schema":"fixture"}\n'); digest = p.sha_file(path)
            self.assertEqual(p.read_sealed_contract(path, digest), {"schema": "fixture"})
            path.write_text('{ "schema":"fixture"}\n')
            with self.assertRaisesRegex(ValueError, "raw SHA"):
                p.read_sealed_contract(path, digest)

    def test_train_measurement_resets_after_warmup_and_only_changes_ram_model(self):
        model = Tiny(); hooks = Hooks(); optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        result = p.measure_batches(model, [batch(2)] * 5, "train", optimizer, hooks, output, grad_clip=1., warmup=2, measured=3)
        self.assertEqual(hooks.resets, 1); self.assertEqual(hooks.events[:hooks.events.index("reset")].count("transfer"), 2)
        self.assertEqual(result["optimizer_steps"], 5); self.assertEqual(result["throughput_targets_per_second"], 2.)
        self.assertEqual(result["measured_incremental_peak_bytes"], {"allocated": 30, "reserved": 50})
        self.assertLess(model.weight.item(), 2.)

    def test_eval_is_grad_free_and_does_not_mutate_model(self):
        model = Tiny(); before = p.tensor_sha(model.state_dict()); seen = []
        def checked(m, b): seen.append((m.training, torch.is_grad_enabled())); return output(m, b)
        result = p.measure_batches(model, [batch(2)] * 3, "eval", None, Hooks(), checked, grad_clip=1., warmup=1, measured=2)
        self.assertEqual(seen, [(False, False)] * 3); self.assertEqual(p.tensor_sha(model.state_dict()), before); self.assertEqual(result["optimizer_steps"], 0)

    def test_train_rejects_nonfinite_parameter_created_by_step(self):
        model = Tiny()
        class BadOptimizer:
            def zero_grad(self, **_): model.weight.grad = None
            def step(self): model.weight.data.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "post-step parameter"):
            p.measure_batches(model, [batch(2)] * 2, "train", BadOptimizer(), Hooks(), output, grad_clip=1., warmup=1, measured=1)

    def test_cache_rejects_test_or_duplicate_targets(self):
        data = {"dataset_id": "fixture"}; batches = [batch() for _ in range(37)]; rows = []
        targets = []
        for index, value in enumerate(batches):
            chosen = [["p", index * 128 + item] for item in range(128)]; targets.extend(chosen)
            rows.append({"batch": index, "phase": "warmup" if index < 5 else "measured", "target_split": "train", "targets": chosen, "tensor_sha256": p.tensor_sha(value)})
        manifest = {"dataset": "fixture", "split": "train", "synthetic": False, "preparation_seed": 42, "batches": rows, "target_ids_sha256": p.canonical_sha(targets)}
        p.validate_batch_cache(manifest, batches, data)
        changed = deepcopy(manifest); changed["split"] = "test"
        with self.assertRaisesRegex(ValueError, "real Train"):
            p.validate_batch_cache(changed, batches, data)
        changed = deepcopy(manifest); changed["batches"][1]["targets"][0] = changed["batches"][0]["targets"][0]
        with self.assertRaisesRegex(ValueError, "Duplicated"):
            p.validate_batch_cache(changed, batches, data)

    def test_profile_does_not_create_checkpoint_or_use_non_train_cache(self):
        # This tests the pre-GPU cache gate, so no CUDA or model implementation is involved.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / "cost/batches/fixture.pt"; path.parent.mkdir(parents=True)
            torch.save([batch() for _ in range(37)], path)
            contract = {"hosts": {"5080": {"root": str(root)}}, "source": {"files_sha256": "sealed"}}
            manifest = {"dataset": "fixture", "split": "validation", "synthetic": False, "preparation_seed": 42, "file_sha256": p.sha_file(path), "batches": [], "target_ids_sha256": "x", "contract_sha256": p.canonical_sha(contract), "contract_file_sha256": None}
            path.with_suffix(".json").write_text(json.dumps(manifest))
            job = {"id": "job", "host": "5080", "arm": "full", "dataset": "fixture", "seed": 42}
            selected_value = (Tiny(), {"checkpoint_path": str(root / "selected.pt")}, {"binary_sha256": "x", "tensor_sha256": "x"}, {"dataset_id": "fixture"})
            with patch.object(p, "select_job", return_value=(job, {"dataset_id": "fixture"}, root, root / "run/job")), patch.object(p, "_load_selected", return_value=selected_value) as selected:
                with self.assertRaisesRegex(ValueError, "real Train"):
                    p.profile(contract, "5080", "job", "eval", 1)
            selected.assert_called_once()


if __name__ == "__main__":
    unittest.main()
