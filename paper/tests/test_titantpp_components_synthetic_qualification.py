"""Run sealed source131 models on fabricated CPU inputs, never research rows.

This test intentionally uses a fresh child process so process-local factory
hooks cannot leak into other tests. It is engineering qualification on the
installed CPU runtime, not checkpoint or Validation/Test metric reproduction.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[2]
BUNDLE = PROJECT / "search_artifacts/titantpp_cnn_gru_components_20261012_v5"
SOURCE_CLOSURE = "15f6d2b482c5f76c0488d017f5c923c76bc54fd75543aab838c477266a608b53"

CHILD = r'''
import hashlib, importlib, json, os, sys
from pathlib import Path
from unittest.mock import patch

bundle = Path(sys.argv[1]).resolve()
expected_closure = sys.argv[2]
contract = json.loads((bundle / "execution_contract.json").read_text())
source = contract["source"]
files = source["files"]
canonical = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False,
    allow_nan=False, separators=(",", ":")).encode()).hexdigest()
assert len(files) == 131
assert canonical(files) == expected_closure == source["files_sha256"]
source_root = (bundle / "source").resolve()
for name, expected in files.items():
    path = (source_root / name).resolve()
    assert path.is_relative_to(source_root)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name

# No current checkout fallback or previously imported scientific modules.
prefixes = {"models", "paper", "data_loader", "simple_lab_test", "utils"}
assert not any(name.split(".")[0] in prefixes for name in sys.modules)
sys.path[:] = [str(source_root)] + [p for p in sys.path if p
    and not (Path(p).resolve() / "paper").is_dir()]
sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import polars as pl
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

forbidden_calls = []
def forbidden(*args, **kwargs):
    forbidden_calls.append("data/checkpoint/training")
    raise AssertionError("Synthetic qualification must not read data/checkpoints or fit")

with patch.object(torch, "load", forbidden), patch.object(pl, "read_parquet", forbidden), \
        patch.object(pl, "scan_parquet", forbidden), \
        patch.object(torch.optim, "AdamW", forbidden), patch.object(torch.optim, "Adam", forbidden), \
        patch.object(torch.optim, "SGD", forbidden):
    route = importlib.import_module("paper.scripts.run_titantpp_cnn_gru_components")
    route.install_hooks()
    factory = importlib.import_module("models.TPPs.CountAwareFactory")
    core = importlib.import_module("paper.scripts.count_aware_tpp_backbone.core")
    tensor_tools = importlib.import_module("simple_lab_test.search.common.runner")
    components = importlib.import_module("models.TPPs.CountAwareTitanCNNGRUComponents")
    parent = importlib.import_module("models.TPPs.CountAwareTitanCNNGRU")
    controls = importlib.import_module("models.TPPs.CountAwareTitanCNNHistoryControls")
    arms = list(route.ALL_ARMS)
    assert arms == contract["arms"]

    # Different left-padded histories retain a target in the final position.
    dts = torch.tensor([[0., 1., 3., 2., 4., 2., 3., 5.],
                        [0., 0., 0., 2., 5., 1., 4., 2.],
                        [0., 0., 0., 0., 0., 3., 1., 4.]])
    qty = torch.tensor([[1., 0., 5., 2., 7., 4., 3., 6.],
                        [0., 0., 0., 2., 4., 8., 1., 5.],
                        [0., 0., 0., 0., 0., 3., 2., 7.]])
    mask = torch.tensor([[True] * 8,
                         [False] * 3 + [True] * 5,
                         [False] * 5 + [True] * 3])
    results = []
    for data in contract["datasets"]:
        for arm in arms:
            torch.manual_seed(321)
            model, meta = route.build_model(data, arm)
            active = sum(p.numel() for p in model.parameters() if p.requires_grad)
            # Legacy architecture metadata declares max_len=256 budgets for
            # every dataset. RAF uses 84 learned positions and has 11008 fewer
            # registered/common active parameters. Preserve, expose and check
            # that discrepancy; never alter the frozen model or metadata.
            position_adjustment = (data["loader"]["max_seq_len"] - 256) * 64
            declared_active = contract["architecture"]["active_parameters"][arm]
            registered = sum(p.numel() for p in model.parameters())
            assert tuple(model.encoder.pos_emb.shape) == (1, data["loader"]["max_seq_len"], 64)
            assert model.encoder.pos_emb.requires_grad
            assert model.encoder.pos_emb.numel() == 64 * data["loader"]["max_seq_len"]
            assert active == declared_active + position_adjustment, (data["dataset_id"], arm, active)
            assert registered == 115167 + position_adjustment, (data["dataset_id"], arm, registered)
            assert meta["candidate_name"] == arm and meta["model_role"] == route.role_for_arm(arm)
            assert meta["cnn_enabled"] == (arm in (components.CNN, components.FULL))
            state = model.state_dict()
            assert torch.equal(state["cnn_gru_identity"].cpu(), parent.identity(arm=components.FULL))
            if arm in components.ARMS:
                assert torch.equal(state["cnn_gru_component_identity"].cpu(), components.identity(arm=arm))
            elif arm == components.NONE:
                assert torch.equal(state["cnn_history_control_identity"].cpu(), controls.identity(arm=arm))

            # Fresh heads/output corrections start at zero. Fabricated weights
            # exercise those paths without optimizer steps or research fitting.
            with torch.no_grad():
                for parameter in route.correction_parameters(model).values():
                    parameter.copy_(torch.randn_like(parameter) * .05)
                model.quantity_head.weight.copy_(torch.randn_like(model.quantity_head.weight) * .025)
                model.time_scale_weight.weight.copy_(torch.randn_like(model.time_scale_weight.weight) * .01)
            model.eval().requires_grad_(False)
            before = tensor_tools.canonical_state_dict_sha256(model.state_dict())
            assert all(p.grad is None and not p.requires_grad for p in model.parameters())
            gru_calls = []
            handle = model.multilag_detail.gru.register_forward_hook(
                lambda *args: gru_calls.append(1))

            original_train = torch.nn.Module.train
            def eval_only(module, mode=True):
                if mode:
                    return forbidden()
                return original_train(module, mode)

            with patch.object(torch.nn.Module, "train", eval_only), torch.no_grad():
                value = core.target_outputs(model, dts, mask, qty, lambda_log_qty=1.)
                q = qty.clone()
                q[:, -1] += 12345.
                q_value = core.target_outputs(model, dts, mask, q, lambda_log_qty=1.)
                gap = dts.clone()
                gap[:, -1] += 123.
                gap_value = core.target_outputs(model, gap, mask, qty, lambda_log_qty=1.)
                for key in ("pred_qty", "time_loss"):
                    assert torch.equal(value[key], q_value[key]), (arm, "target quantity leak", key)
                assert torch.equal(value["pred_qty"], gap_value["pred_qty"]), (arm, "target gap leak")
                assert torch.equal(value["true_qty"] + 12345., q_value["true_qty"])
                assert not torch.equal(value["time_loss"], gap_value["time_loss"])
                # Avoid a vacuous constant-head pass: observed history must matter.
                history_qty = qty.clone()
                history_qty[:, -2] += 9.
                history = core.target_outputs(model, dts, mask, history_qty, lambda_log_qty=1.)
                assert not torch.equal(value["pred_qty"], history["pred_qty"]), (arm, "constant prediction")
                residuals = []
                for i in range(len(qty)):
                    single = core.target_outputs(model, dts[i:i+1], mask[i:i+1], qty[i:i+1], lambda_log_qty=1.)
                    for key in ("pred_qty", "time_loss"):
                        assert torch.allclose(value[key][i:i+1], single[key], rtol=1e-5, atol=1e-5), (arm, key)
                        residuals.append(float((value[key][i:i+1] - single[key]).abs().max()))
                for key in ("true_qty", "pred_qty", "time_loss"):
                    assert torch.isfinite(value[key]).all(), (arm, key)
                assert (value["pred_qty"] >= 0).all()
                assert value["history_length"].tolist() == [7, 4, 2]
            handle.remove()
            assert bool(gru_calls) == (arm in (components.GRU, components.FULL)), (arm, "GRU execution route")
            after = tensor_tools.canonical_state_dict_sha256(model.state_dict())
            assert before == after, (arm, "mutable state")
            assert all(p.grad is None for p in model.parameters())
            results.append({"dataset_metadata": data["dataset_id"], "arm": arm,
                "active_parameters": active, "registered_parameters": registered,
                "legacy_declared_active_parameters": declared_active,
                "legacy_declared_registered_parameters": 115167,
                "legacy_parameter_metadata_matches_actual": position_adjustment == 0,
                "position_parameter_adjustment": position_adjustment,
                "actual_position_embedding_shape": list(model.encoder.pos_emb.shape),
                "actual_position_embedding_parameters": model.encoder.pos_emb.numel(),
                "structure_identity_verified": True, "single_batch_verified": True,
                "target_quantity_mask": True, "target_gap_mask": True,
                "observed_history_sensitive": True, "single_batch_max_absolute_residual": max(residuals),
                "state_sha256_before": before, "state_sha256_after": after,
                "gru_forward_calls": len(gru_calls), "parameters_unchanged": True})
            del model

    imports = {}
    for name, module in sys.modules.items():
        if name.split(".")[0] not in prefixes or not getattr(module, "__file__", None):
            continue
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(source_root), (name, str(path))
        relative = path.relative_to(source_root).as_posix()
        assert relative in files, (name, relative)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == files[relative]
        imports[name] = relative
    assert not forbidden_calls
print(json.dumps({"status": "PASS", "source_closure": expected_closure, "sealed_files": len(files),
    "model_cases": results, "imported_frozen_modules": imports,
    "runtime": {"python": sys.version.split()[0], "torch": torch.__version__,
        "numpy": np.__version__, "polars": pl.__version__, "device": "cpu"},
    "fabricated_only": True, "real_data_loaded": False, "checkpoint_loaded": False,
    "optimizer_steps": 0, "held_out_test_evaluated": False}))
'''


class SealedComponentsSyntheticQualification(unittest.TestCase):
    def test_real_source131_four_arm_inference_on_fabricated_cpu_inputs(self):
        with tempfile.TemporaryDirectory(prefix="components-synthetic-") as scratch:
            env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
                   "MPLCONFIGDIR": scratch, "XDG_CACHE_HOME": scratch}
            result = subprocess.run([sys.executable, "-B", "-c", CHILD, str(BUNDLE), SOURCE_CLOSURE],
                                    cwd=scratch, env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        receipt = json.loads(result.stdout.splitlines()[-1])
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(len(receipt["model_cases"]), 12)
        self.assertEqual({r["arm"] for r in receipt["model_cases"]}, {
            "titantpp_cnn_gru54_correction_disabled", "titantpp_cnn_gru54_cnn_only",
            "titantpp_cnn_gru54_gru_only", "titantpp_cnn_gru54"})
        self.assertEqual(receipt["optimizer_steps"], 0)
        self.assertFalse(receipt["real_data_loaded"])
        self.assertFalse(receipt["checkpoint_loaded"])
        self.assertFalse(receipt["held_out_test_evaluated"])
        self.assertTrue(all(r["state_sha256_before"] == r["state_sha256_after"]
                            for r in receipt["model_cases"]))
        # Compact inspectable evidence accompanies unittest/pytest output.
        print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
