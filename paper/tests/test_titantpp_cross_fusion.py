"""Fabricated-input checks; never read research datasets or old checkpoints."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch

from models.TPPs.CountAwareTitanCrossFusion import (
    ARMS, POINTWISE, CROSS, TimeQuantityFusion, FUSION_PARAMETERS,
    metadata, role_for_arm, validate_checkpoint,
)
from models.TPPs.CountAwareTitanHistoryWidth import CountAwareTitanHistoryWidth
from paper.scripts import run_titantpp_cross_fusion as runner
from paper.scripts.count_aware_tpp_backbone.core import target_outputs


class FusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        runner.install_hooks()

    def model(self, arm=CROSS):
        torch.manual_seed(42)
        return runner.build_model(runner.fabricated_data(), arm)[0]

    def nonzero_branch(self, arm=CROSS):
        torch.manual_seed(17)
        branch = TimeQuantityFusion(arm)
        with torch.no_grad():
            branch.output.weight.normal_(0., .1)
        return branch

    def test_parameters_and_shared_initialization(self):
        data = runner.fabricated_data()
        torch.manual_seed(42)
        original = CountAwareTitanHistoryWidth(64, 1.2, 256, history_mlp_width=16,
            quantity_variant=runner.VARIANT, time_head_mode=data["model"]["time_head_mode"],
            train_log_std=.8, time_scale=1., time_initial_location=.2, time_initial_scale=.8,
            time_observation_contract=data["model"]["time_observation_contract"])
        original_rng = torch.get_rng_state().clone()
        counts = []
        for arm in ARMS:
            model = self.model(arm)
            self.assertEqual(sum(p.numel() for p in model.input_fusion.parameters()), FUSION_PARAMETERS)
            self.assertTrue(torch.equal(torch.get_rng_state(), original_rng))
            for name, value in original.state_dict().items():
                self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
            counts.append(sum(p.numel() for p in model.parameters()))
        self.assertEqual(counts[0], counts[1])

    def test_cpu_synthetic_training_and_roundtrip(self):
        result = runner.synthetic_check()
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["real_data_loaded"])

    def test_future_exclusion_after_nonzero_output(self):
        branch = self.nonzero_branch()
        features = torch.rand(2, 7, 2)
        valid = torch.ones(2, 7, dtype=torch.bool)
        original = branch(features, valid)
        changed = features.clone()
        changed[:, 4:] += 50
        self.assertTrue(torch.equal(original[:, :4], branch(changed, valid)[:, :4]))

    def test_cross_uses_previous_other_modality_control_does_not(self):
        features = torch.rand(2, 7, 2)
        valid = torch.ones(2, 7, dtype=torch.bool)
        changed = features.clone()
        changed[:, :4, 1] += 5
        for arm in ARMS:
            branch = self.nonzero_branch(arm)
            equal = torch.equal(branch(features, valid)[:, -1], branch(changed, valid)[:, -1])
            self.assertEqual(equal, arm == POINTWISE)

    def test_padding_skips_and_masks_nonfinite(self):
        for arm in ARMS:
            branch = self.nonzero_branch(arm)
            compact = torch.tensor([[[.3, 1.], [.8, 2.], [.2, 1.3]]])
            padded = torch.tensor([[[float("nan"), float("inf")], [.3, 1.],
                                    [float("nan"), float("inf")], [.8, 2.], [.2, 1.3]]])
            valid = torch.tensor([[False, True, False, True, True]])
            output = branch(padded, valid)
            torch.testing.assert_close(output[:, valid[0]], branch(compact, torch.ones(1, 3, dtype=torch.bool)),
                                       rtol=1e-6, atol=1e-7)
            self.assertTrue((output[:, ~valid[0]] == 0).all())

    def test_withheld_resets_branch_and_only_branch(self):
        branch = self.nonzero_branch()
        features = torch.rand(1, 6, 2)
        valid = torch.ones(1, 6, dtype=torch.bool)
        observed = valid.clone()
        observed[:, 2] = False
        original = branch(features, valid, memory_write_mask=observed)
        changed = features.clone()
        changed[:, :2] += 10
        changed[:, 2] = float("nan")
        result = branch(changed, valid, memory_write_mask=observed)
        self.assertTrue(torch.equal(original[:, 3:], result[:, 3:]))
        torch.testing.assert_close(original[:, 3:], branch(features[:, 3:], torch.ones(1, 3, dtype=torch.bool)),
                                   rtol=1e-6, atol=1e-7)
        self.assertTrue((result[:, 2] == 0).all())

    def test_all_masked_finite_zero_and_gradient(self):
        for arm in ARMS:
            branch = self.nonzero_branch(arm)
            features = torch.full((2, 4, 2), float("nan"))
            output = branch(features, torch.zeros(2, 4, dtype=torch.bool))
            self.assertTrue((output == 0).all())
            output.sum().backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in branch.parameters()))

    def test_gradient_reaches_qk_after_zero_output_update(self):
        torch.manual_seed(19)
        features = torch.rand(3, 9, 2)
        target = torch.randn(3, 9, 64)
        valid = torch.ones(3, 9, dtype=torch.bool)
        for arm in ARMS:
            branch = TimeQuantityFusion(arm)
            optimizer = torch.optim.SGD(branch.parameters(), lr=.1)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                (branch(features, valid) - target).square().mean().backward()
                optimizer.step()
            for name, value in branch.named_parameters():
                self.assertIsNotNone(value.grad, name)
                self.assertTrue(torch.isfinite(value.grad).all(), name)
                self.assertGreater(value.grad.abs().sum().item(), 0., name)

    def test_strict_checkpoint_identity_rejects_other_arm_and_incomplete(self):
        model = self.model()
        state = copy.deepcopy(model.state_dict())
        model.load_state_dict(state)
        with self.assertRaises(ValueError):
            self.model(POINTWISE).load_state_dict(state)
        state.pop("input_fusion.output.weight")
        with self.assertRaises(RuntimeError):
            model.load_state_dict(state, strict=False)

    def test_new_arm_route_and_checkpoint_metadata(self):
        from models.TPPs.CountAwareFactory import validate_checkpoint_route
        for arm in ARMS:
            model, meta = runner.build_model(runner.fabricated_data(), arm)
            interface = {"time_head": meta["time_head"]}
            payload = {"backbone": arm, "encoder_config": meta, "model_state_dict": model.state_dict(),
                "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
                "variant": runner.VARIANT, "interface_meta": interface,
                "resume_identity": {"backbone": arm, "interface_meta": interface,
                    "checkpoint_monitor": "validation_raw_quantity_rmse", "arguments": {"model_role": role_for_arm(arm)}}}
            self.assertTrue(validate_checkpoint(payload, arm))
            validate_checkpoint_route(payload, arm)
            with self.assertRaises(ValueError):
                validate_checkpoint_route(payload, "titantpp_history_mlp_width16")
            changed = copy.deepcopy(payload)
            changed["held_out_test_evaluated"] = True
            with self.assertRaises(ValueError):
                validate_checkpoint(changed, arm)

    def test_model_future_and_target_leak_exclusion(self):
        dt = torch.tensor([[1., 2., 3., 4., 5., 6.]])
        qty = torch.tensor([[5., 4., 3., 6., 7., 8.]])
        valid = torch.ones_like(dt, dtype=torch.bool)
        observed = valid.clone()
        observed[:, -1] = False
        for arm in ARMS:
            model = self.model(arm).eval()
            with torch.no_grad():
                model.input_fusion.output.weight.normal_(0., .1)
                original = model.encode_task_states(dt, qty, valid, memory_write_mask=observed)[0]
                altered_dt, altered_qty = dt.clone(), qty.clone()
                altered_dt[:, -1] = float("nan")
                altered_qty[:, -1] = float("inf")
                masked = model.encode_task_states(altered_dt, altered_qty, valid, memory_write_mask=observed)[0]
                self.assertTrue(torch.equal(original, masked))
                altered_dt[:, -1], altered_qty[:, -1] = 100., 1000.
                first = target_outputs(model, dt, valid, qty, lambda_log_qty=1.)
                second = target_outputs(model, altered_dt, valid, altered_qty, lambda_log_qty=1.)
                self.assertTrue(torch.equal(first["pred_qty"], second["pred_qty"]))
                altered_dt[:, 3], altered_qty[:, 3] = 55., 77.
                future = model.encode_task_states(altered_dt, altered_qty, valid, memory_write_mask=observed)[0]
                self.assertTrue(torch.equal(original[:, :3], future[:, :3]))

    def test_finite_nonnegative_observations_required(self):
        for arm in ARMS:
            model = self.model(arm)
            valid = torch.ones(1, 4, dtype=torch.bool)
            for bad in (-1., float("nan"), float("inf")):
                dt = torch.ones(1, 4)
                dt[:, 1] = bad
                with self.assertRaises(ValueError):
                    model.encode_task_states(dt, torch.ones_like(dt), valid)

    def test_real_trainer_one_epoch_on_fabricated_frame(self):
        """Exercise loader, role checks, loss, optimizer and checkpoint route unmocked."""
        import numpy as np
        import polars as pl
        from data_loader.event_seq_data_module import RMTPPWeekLookbackDataset
        from paper.scripts.count_aware_tpp_backbone import training
        from paper.scripts.run_count_aware_tpp_backbone_control import derive_train_time_contract
        from paper.scripts.run_hard_lmm_frozen_lognormal_duration import target_dt_sha256
        from simple_lab_test.search.common.runner import torch_load_checkpoint
        frame = pl.DataFrame({"oper_part_no": ["a"] * 6 + ["b"] * 6,
            "seq": [1, 2, 4, 7, 8, 10] * 2, "delta_t": [1., 1., 2., 3., 1., 2.] * 2,
            "mark": [0] * 12, "scale_residual": [2., 3., 5., 4., 6., 3., 1., 2., 4., 3., 7., 2.],
            "chronological_split": (["train"] * 4 + ["validation"] * 2) * 2})
        stats = derive_train_time_contract(frame, lookback_weeks=52, max_seq_len=16)
        data = runner.fabricated_data()
        data["dataset_id"] = "raf_spare_parts"
        data["model"].update(time_scale=stats["time_scale"],
            time_initial_location=stats["target_log_scaled_mean"],
            time_initial_scale=stats["target_log_scaled_std"], time_sigma_floor=.05,
            time_observation_contract={"mode": "positive_integer_round_clamp_v1", "unit": "month", "top_code": None})
        data["loader"] = {"batch_size": 128, "num_workers": 0, "drop_last": False,
            "train_shuffle": True, "validation_shuffle": False, "max_seq_len": 16, "lookback_weeks": 52}
        data["optimizer"] = {"name": "AdamW", "lr": .001, "weight_decay": .01, "grad_clip": 1.}
        data["quantity_boundaries_all_train_rows"] = [2., 3., 4., 5.]
        data["inherited_data_identity"] = {"data": {"path": "synthetic-unused.parquet", "sha256": "a" * 64},
            "split_manifest": {"path": "synthetic-unused.json", "sha256": "b" * 64}}
        data["time_statistics"] = {"train_time_scale": stats["time_scale"],
            "train_log_scaled_mean": stats["target_log_scaled_mean"],
            "train_log_scaled_std": stats["target_log_scaled_std"]}
        for split in ("train", "validation"):
            ds = RMTPPWeekLookbackDataset(frame, lookback_weeks=52, max_seq_len=16,
                mode="all", target_splits={split})
            values = torch.tensor(np.asarray([max(1., float(ds.dt_lists[p][i + 1])) for p, i in ds.index], dtype=np.float64))
            data["time_statistics"]["expected_" + split + "_targets"] = len(values)
            data["time_statistics"]["expected_" + split + "_target_dt_sha256"] = target_dt_sha256(values)
        contract = {"source": {"base_git_revision": "synthetic_local_only", "files_sha256": "c" * 64},
            "training": {"maximum_epochs": 1, "minimum_epochs": 1, "patience": 1},
            "design_sha256": "d" * 64, "architecture": "fabricated_integration"}
        quantity_contract = {"boundaries": [2., 3., 4., 5.],
            "strata": [{"stratum": str(i)} for i in range(5)]}
        with tempfile.TemporaryDirectory(prefix="fusion-fabricated-") as folder:
            for arm in ARMS:
                output = Path(folder) / arm
                args = runner.training_args(contract, data, output, arm)
                args.device = "cpu"
                interface = runner.time_interface(data, frame, contract, arm=arm)
                summary, _, _ = training.train_one(args=args, frame=frame, quantity_contract=quantity_contract,
                    interface_meta=interface, backbone=arm, quantity_variant=runner.VARIANT, seed=42)
                self.assertEqual(summary["completed_epochs"], 1)
                self.assertEqual(summary["best_epoch"], 1)
                self.assertFalse(summary["held_out_test_evaluated"])
                validate_checkpoint(summary, arm)
                selected = output / "runs" / arm / runner.VARIANT / "seed_42" / "best_val_qty_rmse_model.pt"
                payload = torch_load_checkpoint(selected, map_location="cpu")
                validate_checkpoint(payload, arm)
                model, _ = runner.build_model(data, arm)
                model.load_state_dict(payload["model_state_dict"], strict=True)


if __name__ == "__main__":
    unittest.main()
