"""Synthetic tests; never open research datasets/checkpoints or contact hosts."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl
import torch

from paper.scripts import evaluate_titantpp_cnn_memory_sensitivity as evaluator


PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / "search_artifacts/titantpp_components_taxi_raf24_test_20261014_v1/source"


class ObservedFrameBoundaryTests(unittest.TestCase):
    def test_lazy_predicate_precedes_materialization_and_preserves_allowed_rows(self):
        fabricated = pl.DataFrame({
            "oper_part_no": ["b", "a", "a", "b", "a", "b", "a"],
            "seq": [3, 3, 2, 1, 4, 2, 1],
            "chronological_split": ["test", "validation", "train", "train", "test", "validation", "train"],
            "demand_qty": [987654321., 3., 2., 4., 987654322., 5., 1.],
        })
        with tempfile.TemporaryDirectory(prefix="cnn-memory-split-", dir="/private/tmp") as tmp:
            path = Path(tmp) / "synthetic_mixed_splits.parquet"
            fabricated.write_parquet(path)
            original_scan = pl.scan_parquet
            for split, allowed, expected_values in (
                ("train", {"train"}, [1., 2., 4.]),
                ("validation", {"train", "validation"}, [1., 2., 3., 4., 5.]),
            ):
                events = []
                test_case = self

                class GuardedLazy:
                    def __init__(self, query, filtered=False):
                        self.query, self.filtered = query, filtered

                    def filter(self, predicate):
                        events.append("filter")
                        return GuardedLazy(self.query.filter(predicate), filtered=True)

                    def collect(self):
                        # Check the query before executing it, not after loading a
                        # complete frame and dropping held-out rows in Python.
                        test_case.assertTrue(self.filtered)
                        plan = self.query.explain(optimized=False)
                        test_case.assertIn("FILTER", plan)
                        test_case.assertIn("chronological_split", plan)
                        test_case.assertEqual(events, ["scan", "filter"])
                        events.append("collect")
                        result = self.query.collect()
                        test_case.assertLessEqual(set(result["chronological_split"]), allowed)
                        test_case.assertFalse((result["demand_qty"] >= 987654321.).any())
                        return result

                def guarded_scan(filename):
                    self.assertEqual(Path(filename), path)
                    events.append("scan")
                    return GuardedLazy(original_scan(filename))

                with self.subTest(split=split), patch.object(pl, "scan_parquet", side_effect=guarded_scan), \
                     patch.object(pl, "read_parquet", side_effect=AssertionError("eager materialization forbidden")):
                    actual = evaluator.load_observed_frame(path, split, pl)
                self.assertEqual(events, ["scan", "filter", "collect"])
                self.assertEqual(actual["demand_qty"].to_list(), expected_values)
                expected = fabricated.filter(pl.col("chronological_split").is_in(sorted(allowed))).sort(["oper_part_no", "seq"])
                self.assertTrue(actual.equals(expected))

    def test_held_out_request_is_rejected_before_io(self):
        with patch.object(pl, "scan_parquet") as scan, patch.object(pl, "read_parquet") as eager:
            with self.assertRaisesRegex(ValueError, "Only observed diagnostic splits"):
                evaluator.load_observed_frame(Path("/does/not/exist.parquet"), "test", pl)
        scan.assert_not_called()
        eager.assert_not_called()


class ObservedHistoryAndTelemetryTests(unittest.TestCase):
    def test_features_ignore_targets_and_left_padding_values(self):
        quantities = torch.tensor([[9., 9., 2., 4., 100.], [9., 9., 9., 3., 200.]])
        mask = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 0, 1, 1]], dtype=torch.bool)
        actual = evaluator.observed_features(quantities, mask, np)
        changed = quantities.clone()
        changed[:, -1] = torch.tensor([10000., 1.])
        changed[~mask] = 70000.
        altered = evaluator.observed_features(changed, mask, np)
        for key in actual:
            np.testing.assert_array_equal(actual[key], altered[key], err_msg=key)
        np.testing.assert_array_equal(actual["history_length"], [2, 1])
        np.testing.assert_array_equal(actual["last_quantity"], [4., 3.])
        np.testing.assert_array_equal(actual["mean_quantity"], [3., 3.])
        np.testing.assert_allclose(actual["history_log_mean"], [(np.log(3) + np.log(5))/2, np.log(4)])
        np.testing.assert_allclose(actual["history_log_std"], [(np.log(5) - np.log(3))/2, 0.])
        np.testing.assert_allclose(actual["last_observed_log_change"], [np.log(5)-np.log(3), 0.])
        np.testing.assert_array_equal(actual["previous_observation_available"], [True, False])

    def test_empty_observed_context_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Empty observed context"):
            evaluator.observed_features(torch.ones(1, 3), torch.tensor([[0, 0, 1]], dtype=torch.bool), np)

    def test_minimum_one_history_and_one_target(self):
        result = evaluator.observed_features(torch.tensor([[2., 90.]]), torch.ones(1, 2, dtype=torch.bool), np)
        np.testing.assert_array_equal(result["last_observed_log_change"], [0.])
        np.testing.assert_array_equal(result["history_length"], [1])

    def test_selected_telemetry_uses_last_observed_not_target_or_padding(self):
        record = {
            "encoded_norm": torch.tensor([[10., 20., 30., 999.], [11., 22., 888., 777.]]),
            "memory_norm": torch.tensor([[1., 2., 3., 99.], [1., 2., 88., 77.]]),
            "cosine_h_memory": torch.tensor([[.1, .2, .3, .9], [.1, .2, .8, .7]]),
            "prototype_indices": torch.tensor([[[0, 1], [1, 2], [2, 3], [8, 9]],
                                                 [[0, 1], [4, 5], [8, 9], [7, 8]]]),
            "topk_similarity": torch.tensor([[[.1, .2], [.3, .4], [.5, .6], [.8, .9]],
                                              [[.2, .4], [.5, .6], [.8, .9], [.7, .8]]]),
        }
        out = evaluator.selected_telemetry(record, np.asarray([3, 1]), np)
        np.testing.assert_array_equal(out["encoded_norm"], [30., 11.])
        np.testing.assert_array_equal(out["memory_norm"], [3., 1.])
        np.testing.assert_array_equal(out["prototype_0"], [2, 0])
        np.testing.assert_array_equal(out["prototype_1"], [3, 1])
        np.testing.assert_array_equal(out["previous_topk_overlap"], [.5, 0.])
        np.testing.assert_allclose(out["topk_similarity_mean"], [.55, .3])
        np.testing.assert_allclose(out["memory_to_encoded_norm"], [.1, 1/11])

    def test_zero_encoded_norm_has_explicit_flag(self):
        record = {key: torch.zeros(1, 1, dtype=torch.float64)
                  for key in ("encoded_norm", "memory_norm", "cosine_h_memory")}
        record.update(prototype_indices=torch.tensor([[[1, 2]]]),
                      topk_similarity=torch.zeros(1, 1, 2, dtype=torch.float64))
        out = evaluator.selected_telemetry(record, np.asarray([1]), np)
        np.testing.assert_array_equal(out["encoded_norm_zero"], [True])
        self.assertTrue(np.isfinite(out["memory_to_encoded_norm"]).all())


class DiagnosticContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cnn-memory-contract-", dir="/private/tmp")
        self.root = Path(self.temp.name).resolve()
        (self.root / "operation").mkdir()
        (self.root / "operation/evaluate.py").write_bytes(Path(evaluator.__file__).read_bytes())
        (self.root / "operation/control.py").write_text("# synthetic operation identity\n")
        jobs = [dict(id=f"{dataset}__{seed}__{evaluator.ARM}", dataset=dataset, seed=seed, arm=evaluator.ARM,
                     host="5090" if dataset == "intermittent_frozen_5000" and seed in (42, 52) else "5080")
                for dataset in ("yellow_trip_hourly", "raf_spare_parts", "intermittent_frozen_5000")
                for seed in (42, 52, 62)]
        self.contract = dict(schema=evaluator.SCHEMA, allowed_splits=["train", "validation"],
            new_training=False, frozen_selection=True, max_parallel_per_host=3,
            alphas=list(evaluator.ALPHAS), jobs=jobs, deadline_unix=time.time()+600,
            operation_files={f"operation/{name}": evaluator.sha(self.root / "operation" / name)
                             for name in ("evaluate.py", "control.py")}, input_files={})

    def tearDown(self):
        self.temp.cleanup()

    def seal(self, contract=None):
        (self.root / "contract.json").write_text(json.dumps(contract or self.contract))
        return evaluator.sha(self.root / "contract.json")

    def test_train_and_validation_contract_paths_are_accepted(self):
        digest = self.seal()
        for split in ("train", "validation"):
            contract, job = evaluator.verify(self.root, digest, self.contract["jobs"][0]["id"], split)
            self.assertEqual(contract, self.contract)
            self.assertEqual(job["arm"], evaluator.ARM)

    def test_test_is_rejected_before_any_file_read(self):
        with patch.object(evaluator, "sha") as checksum, self.assertRaisesRegex(ValueError, "Only Train/Validation"):
            evaluator.verify(self.root, "unused", "unused", "test")
        checksum.assert_not_called()

    def test_duplicate_job_ids_are_rejected(self):
        c = deepcopy(self.contract)
        c["jobs"][-1]["id"] = c["jobs"][0]["id"]
        digest = self.seal(c)
        with self.assertRaisesRegex(ValueError, "nine unique CNN models"):
            evaluator.verify(self.root, digest, c["jobs"][0]["id"], "train")

    def test_mutated_operation_bytes_are_rejected(self):
        digest = self.seal()
        (self.root / "operation/control.py").write_text("# altered operation\n")
        with self.assertRaisesRegex(ValueError, "Frozen bytes changed"):
            evaluator.verify(self.root, digest, self.contract["jobs"][0]["id"], "validation")

    def test_contract_sha_grid_and_route_are_checked(self):
        digest = self.seal()
        with self.assertRaisesRegex(ValueError, "Contract SHA"):
            evaluator.verify(self.root, "0"*64, self.contract["jobs"][0]["id"], "validation")
        for field, changed, message in (("alphas", [1., 0.], "attenuation grid"),):
            c = deepcopy(self.contract)
            c[field] = changed
            with self.assertRaisesRegex(ValueError, message):
                evaluator.verify(self.root, self.seal(c), c["jobs"][0]["id"], "validation")
        c = deepcopy(self.contract)
        c["jobs"][0]["arm"] = "titantpp_cnn_gru54"
        with self.assertRaisesRegex(ValueError, "identity or native host"):
            evaluator.verify(self.root, self.seal(c), c["jobs"][0]["id"], "validation")

    def test_nonselected_job_identity_and_cohort_are_checked(self):
        c = deepcopy(self.contract)
        c["jobs"][-1]["host"] = "5090"
        with self.assertRaisesRegex(ValueError, "identity or native host"):
            evaluator.verify(self.root, self.seal(c), c["jobs"][0]["id"], "train")
        c = deepcopy(self.contract)
        c["jobs"][-1]["seed"] = 52
        with self.assertRaisesRegex(ValueError, "Dataset/seed cohort"):
            evaluator.verify(self.root, self.seal(c), c["jobs"][0]["id"], "train")

    def test_operation_input_path_overlap_is_rejected(self):
        c = deepcopy(self.contract)
        c["input_files"]["operation/control.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Operation/input paths overlap"):
            evaluator.verify(self.root, self.seal(c), c["jobs"][0]["id"], "train")


class FrozenScienceIntegrationTests(unittest.TestCase):
    def test_source131_only_package_requires_preserved_operational_root_sentinel(self):
        # Copy only the sealed scientific manifest, not the existing directory:
        # the latter already contains operational scaffolding and hides omissions.
        parent = SOURCE.parent
        contract = json.loads((parent / "contract.json").read_text())
        source = contract["sources"]["components"]
        self.assertEqual(len(source["files"]), 131)
        sentinel_name = "source/sample_data/.frozen_root_sentinel"
        sentinel_sha = contract["evaluator_integrity"]["files"][sentinel_name]
        self.assertEqual(evaluator.sha(parent / sentinel_name), sentinel_sha)
        code = r'''
import json, pathlib, sys
expected = pathlib.Path(sys.argv[1]).resolve()
from paper.scripts import run_titantpp_cnn_gru_components as engine
engine.install_hooks()
from paper.scripts.count_aware_tpp_backbone import core
from models.TPPs import CountAwareFactory as factory
from simple_lab_test.common.pathing import resolve_project_root
resolved = resolve_project_root(pathlib.Path(core.__file__))
assert resolved == expected, (resolved, expected)
prefixes = {"models", "paper", "utils", "simple_lab_test", "data_loader"}
origins = {}
for name, obj in tuple(sys.modules.items()):
    path = getattr(obj, "__file__", None)
    if name.split(".")[0] in prefixes and path:
        actual = pathlib.Path(path).resolve()
        assert actual.is_relative_to(expected), (name, str(actual))
        origins[name] = str(actual.relative_to(expected))
assert "paper.scripts.count_aware_tpp_backbone.core" in origins
assert "models.TPPs.CountAwareFactory" in origins
print(json.dumps({"resolved_root": str(resolved), "verified_import_count": len(origins)}))
'''
        with tempfile.TemporaryDirectory(prefix="cnn-memory-deployment-", dir="/private/tmp") as cwd:
            root = (Path(cwd) / "source").resolve()
            root.mkdir()
            for name, digest in source["files"].items():
                original = SOURCE / name
                self.assertEqual(evaluator.sha(original), digest)
                destination = (root / name).resolve()
                self.assertTrue(destination.is_relative_to(root))
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, destination)
            self.assertFalse((root / "sample_data").exists())
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(root),
                       CUDA_VISIBLE_DEVICES="", MPLCONFIGDIR=str(Path(cwd)/"matplotlib"),
                       XDG_CACHE_HOME=str(Path(cwd)/"cache"))
            args = [sys.executable, "-B", "-c", code, str(root)]
            missing = subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True, timeout=60)
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("Could not locate the paper_research project root", missing.stderr)
            sentinel = root / "sample_data/.frozen_root_sentinel"
            sentinel.parent.mkdir()
            shutil.copyfile(parent / sentinel_name, sentinel)
            self.assertEqual(evaluator.sha(sentinel), sentinel_sha)
            self.assertEqual(sentinel.stat().st_size, 0)
            restored = subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True, timeout=60)
            self.assertEqual(restored.returncode, 0, restored.stdout + restored.stderr)
            proof = json.loads(restored.stdout.strip().splitlines()[-1])
            self.assertEqual(proof["resolved_root"], str(root))
            self.assertGreater(proof["verified_import_count"], 10)
            for name, digest in source["files"].items():
                self.assertEqual(evaluator.sha(root / name), digest)

    def test_real_source131_cnn_and_full_on_fabricated_inputs(self):
        # Fresh interpreter prevents installed factory adapters or current source
        # imports elsewhere in the test suite from contaminating source131.
        code = r'''
import importlib.util, json, pathlib, sys
import numpy as np
import torch
torch.set_num_threads(1)
source, control_path, evaluator_path = map(pathlib.Path, sys.argv[1:])
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
control = load(control_path, "diagnostic_control")
evaluation = load(evaluator_path, "diagnostic_evaluation")
from paper.scripts import run_titantpp_cnn_gru_components as engine
engine.install_hooks()
from models.TPPs import CountAwareFactory as factory
from paper.scripts.count_aware_tpp_backbone import core
for obj in (engine, factory, core):
    assert pathlib.Path(obj.__file__).resolve().is_relative_to(source)
dts = torch.tensor([[0., 0., 1., 2., 1., 3., 1., 2.],
                    [1., 2., 1., 3., 1., 2., 1., 3.],
                    [0., 0., 0., 0., 0., 0., 1., 2.]])
quantities = torch.tensor([[0., 0., 2., 4., 1., 3., 2., 5.],
                           [1., 2., 3., 4., 3., 2., 1., 6.],
                           [0., 0., 0., 0., 0., 0., 3., 20.]])
mask = torch.ones_like(dts, dtype=torch.bool)
mask[0, :2] = False
mask[2, :6] = False
summary = {}
for arm in (engine.CNN, engine.FULL):
    torch.manual_seed(42)
    definition = engine.fabricated_data()
    config = {k: v for k, v in definition["model"].items()
              if k not in ("backbone", "lambda_log_qty", "lambda_tail", "time_head_lr_multiplier")}
    model, meta = factory.build_count_aware_model(arm, **config,
        train_log_mean=definition["statistics"]["train_log_mean"],
        train_log_std=definition["statistics"]["train_log_std"],
        max_seq_len=definition["loader"]["max_seq_len"])
    model.eval().requires_grad_(False)
    # Activate otherwise zero-initialized synthetic correction paths; no fitting.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "multilag_detail" in name or any(k in name for k in
                ("causal_q_kernel", "causal_k_kernel", "causal_v_kernel")):
                parameter.copy_(torch.randn_like(parameter) * .03)
    before = control.memory_state_sha256(model)
    traces_by_alpha = {}
    with torch.inference_mode():
        native = core.target_outputs(model, dts, mask, quantities, lambda_log_qty=1.)
        for alpha in (1., .5, .25, 0.):
            traces = []
            with control.frozen_memory_attenuation(model.lmm, alpha, traces.append):
                actual = core.target_outputs(model, dts, mask, quantities, lambda_log_qty=1.)
            assert len(traces) == 1
            if alpha == 1:
                for key in native:
                    assert torch.equal(actual[key], native[key]), (arm, key, "native changed")
            observed = evaluation.observed_features(quantities, mask, np)
            lengths = actual["history_length"].numpy()
            np.testing.assert_array_equal(lengths, observed["history_length"])
            selected = evaluation.selected_telemetry(traces[0], lengths, np)
            traces_by_alpha[alpha] = selected
            altered_q = quantities.clone(); altered_q[:, -1] += 12345.
            altered_dt = dts.clone(); altered_dt[:, -1] += 123.
            for alternate_q, alternate_dt, mode in (
                (altered_q, dts, "quantity"), (quantities, altered_dt, "gap")):
                alternative_traces = []
                with control.frozen_memory_attenuation(model.lmm, alpha, alternative_traces.append):
                    altered = core.target_outputs(model, alternate_dt, mask, alternate_q, lambda_log_qty=1.)
                torch.testing.assert_close(actual["pred_qty"], altered["pred_qty"], rtol=0, atol=0)
                if mode == "quantity":
                    torch.testing.assert_close(actual["time_loss"], altered["time_loss"], rtol=0, atol=0)
                alternative_selected = evaluation.selected_telemetry(alternative_traces[0], lengths, np)
                for key in selected:
                    np.testing.assert_array_equal(selected[key], alternative_selected[key], err_msg=key)
            outputs = []
            for i in range(len(dts)):
                with control.frozen_memory_attenuation(model.lmm, alpha, lambda _: None):
                    outputs.append(core.target_outputs(model, dts[i:i+1], mask[i:i+1],
                        quantities[i:i+1], lambda_log_qty=1.))
            for key in ("pred_qty", "time_loss"):
                torch.testing.assert_close(actual[key], torch.cat([v[key] for v in outputs]),
                                           rtol=1e-5, atol=1e-5)
            assert control.memory_state_sha256(model) == before
        for selected in traces_by_alpha.values():
            for key in selected:
                np.testing.assert_array_equal(selected[key], traces_by_alpha[1.][key], err_msg=key)
    assert "forward" not in model.lmm.__dict__ and "retrieve" not in model.lmm.__dict__
    summary[arm] = {"alpha1_exact_native": True, "all_alpha_target_blocked": True,
                    "batch_consistent": True, "retrieval_unchanged": True,
                    "parameters_unchanged": True}
print(json.dumps(summary, sort_keys=True))
'''
        with tempfile.TemporaryDirectory(prefix="cnn-memory-source131-", dir="/private/tmp") as cwd:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(SOURCE),
                       CUDA_VISIBLE_DEVICES="", MPLCONFIGDIR=str(Path(cwd)/"matplotlib"),
                       XDG_CACHE_HOME=str(Path(cwd)/"cache"))
            completed = subprocess.run([sys.executable, "-B", "-c", code, str(SOURCE),
                str(PROJECT/"paper/scripts/titantpp_cnn_memory_control.py"), evaluator.__file__],
                cwd=cwd, env=env, text=True, capture_output=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(set(result), {evaluator.ARM, "titantpp_cnn_gru54"})
        self.assertTrue(all(all(checks.values()) for checks in result.values()))


if __name__ == "__main__":
    unittest.main()
