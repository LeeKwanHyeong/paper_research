"""Focused synthetic invariant checks; no selected checkpoint or real data inference."""
from pathlib import Path
import importlib.util
import json
import math
import sys
import unittest
from unittest import mock
from types import SimpleNamespace
import contextlib
import hashlib
import io
import tempfile

ROOT = Path(__file__).resolve().parents[2]

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

telemetry = load(ROOT / "paper/scripts/titantpp_history_diagnostic_telemetry.py", "fixture_telemetry")
evaluator = load(ROOT / "paper/scripts/evaluate_titantpp_history_diagnostics.py", "fixture_evaluator")


class AggregateInvariantTests(unittest.TestCase):
    def fixture(self):
        a = telemetry.AggregateDiagnostics([2., 4.], [2, 4], [1, 3], [2, 5], 3)
        for i, (q, p, loss, history, truncated) in enumerate([(1.,2.,.5,2,False),(5.,3.,1.5,4,True),(6.,9.,2.5,5,False)]):
            a.add(entity="a" if i < 2 else "b", calendar="2001-01", last_input_calendar="2000-12",
                  history_length=history, unclipped_history_length=history+int(truncated), truncated=truncated,
                  series_in_validation=i < 2, true_quantity=q, predicted_quantity=p, time_nll=loss,
                  recorded_gap=2+i, last_input_gap=1+i, last_quantity=2., mean_quantity=3.,
                  features={"duration_location": float(i), "duration_scale": 1.})
        return a.finish()

    def test_additive_error_accounting_and_tail(self):
        result = self.fixture()
        whole = result["strata"]["all"][0]
        self.assertEqual(whole["count"], 3)
        self.assertEqual(whole["sums"]["squared_error"], 14.)
        self.assertEqual(whole["sums"]["absolute_error"], 6.)
        self.assertEqual(whole["sums"]["signed_error"], 2.)
        self.assertEqual(whole["sums"]["squared_time_nll"], .25+2.25+6.25)
        tail = next(x for x in result["strata"]["quantity_bin"] if x["key"] == "2")
        expected = {"count":3,"qty_rmse":math.sqrt(14/3),"qty_mae":2.,"time_nll":1.5,
                    "tail":{"count":2,"qty_rmse":math.sqrt(13/2),"qty_mae":2.5,"time_nll":2.}}
        self.assertTrue(evaluator.verify_aggregate_metrics(result, expected))
        self.assertEqual(tail["count"],2)
        self.assertEqual(sum(x["count"] for x in result["strata"]["tail_by_series"]), 2)
        self.assertEqual(sum(x["count"] for x in result["strata"]["history_vs_train_max"] if x["key"] == "above_train_max"),2)
        expected["tail"]["time_nll"] += 1
        with self.assertRaises(ValueError): evaluator.verify_aggregate_metrics(result, expected)

    def test_joint_moments_not_causality(self):
        result = self.fixture()
        whole = result["strata"]["all"][0]
        feature = whole["features"]["duration_location"]
        self.assertAlmostEqual(feature["mean"],1.)
        self.assertAlmostEqual(feature["population_variance"],2/3)
        self.assertAlmostEqual(feature["sum_product_signed_error"],4.)
        self.assertAlmostEqual(feature["covariance_with_signed_error"],2/3)
        self.assertFalse(result["association_is_causal_effect"])
        self.assertFalse(result["raw_prediction_rows_written"])
        self.assertNotIn("targets",result)

    def test_nonfinite_and_feature_drift_rejected(self):
        with self.assertRaises(ValueError): telemetry.AggregateDiagnostics([1,1], [], [], [], 3)
        with self.assertRaises(ValueError): telemetry.AggregateDiagnostics([1], [], [], [], 0)


class FrozenSyntheticHookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Explicit local fabricated CPU fixture; its Runtime is not native5080 admission.
        sys.path.insert(0,str(ROOT / "search_artifacts/titantpp_history_instacart_test_20261012_v1/sources/history"))
        import torch
        from paper.scripts import run_titantpp_cnn_history_controls as engine
        engine.install_hooks()
        from models.TPPs import CountAwareFactory as factory
        from models.TPPs import CountAwareTitanCNNGRU as adapter
        from paper.scripts.count_aware_tpp_backbone import core
        from simple_lab_test.search.common import runner
        cls.torch,cls.factory,cls.adapter,cls.core,cls.runner = torch,factory,adapter,core,runner
        cls.contract = json.loads((ROOT / "search_artifacts/titantpp_history_instacart_test_20261012_v1/contract.json").read_text())

    def test_each_route_noninterference_with_nonzero_fixture(self):
        torch = self.torch
        for arm in evaluator.HISTORY_ARMS:
            with self.subTest(arm=arm), torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                definition = self.contract["datasets"]["raf_spare_parts"]["definition"]
                config = {k:v for k,v in definition["model"].items() if k not in ("backbone","lambda_log_qty","lambda_tail","time_head_lr_multiplier")}
                model,_ = self.factory.build_count_aware_model(arm,**config,train_log_mean=1.2,train_log_std=.8,max_seq_len=84)
                model.eval().requires_grad_(False)
                with torch.no_grad():
                    for name,p in model.named_parameters():
                        if "multilag_detail" in name or "causal_" in name and name.endswith("_kernel"):
                            p.copy_(torch.randn_like(p)*.03)
                    before = self.runner.canonical_state_dict_sha256(model.state_dict())
                    dts=torch.tensor([[0.,0.,1.,2.,1.,3.,1.,2.],[1.,2.,1.,3.,1.,2.,1.,3.]])
                    qty=torch.tensor([[0.,0.,2.,4.,1.,3.,2.,5.],[1.,2.,3.,4.,3.,2.,1.,6.]])
                    mask=torch.ones_like(dts,dtype=torch.bool);mask[0,:2]=False
                    plain=self.core.target_outputs(model,dts,mask,qty,lambda_log_qty=1.)
                    observer=telemetry.ReadOnlyTelemetry(model,arm,torch,self.adapter.causal_depthwise_residual)
                    observer.begin()
                    captured=self.core.target_outputs(model,dts,mask,qty,lambda_log_qty=1.)
                    features=observer.snapshot(captured["history_length"],dts[:,-1])
                    for key in plain:self.assertTrue(torch.equal(plain[key],captured[key]),key)
                    self.assertEqual(features["quantity_head_logit"],model.quantity_head(observer.values["shared_head_hidden"]).squeeze(-1).tolist())
                    if arm==telemetry.FULL:
                        self.assertTrue(any(x>0 for x in features["gru_correction_norm"]))
                        self.assertTrue(any(x>0 for x in features["cnn_q_residual_norm"]))
                        self.assertEqual(len(features["cnn_q_residual_norm"]),2)
                    else:
                        self.assertFalse(any(name.startswith("cnn_") or name.startswith("gru_") for name in features))
                    saved={k:v.clone() for k,v in observer.values.items()}
                    with observer.suspended():
                        self.core.target_outputs(model,dts[:1],mask[:1],qty[:1],lambda_log_qty=1.)
                    self.assertTrue(all(torch.equal(v,observer.values[k]) for k,v in saved.items()))
                    observer.close()
                    self.assertFalse(observer.handles)
                    self.assertEqual(before,self.runner.canonical_state_dict_sha256(model.state_dict()))
                    self.assertTrue(all(not p.requires_grad for p in model.parameters()))


class FrozenPopulationPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FrozenSyntheticHookTests.setUpClass()
        cls.torch, cls.core, cls.runner = FrozenSyntheticHookTests.torch, FrozenSyntheticHookTests.core, FrozenSyntheticHookTests.runner
        import numpy as np
        import polars as pl
        import importlib
        cls.np, cls.pl = np, pl
        cls.loader_module = importlib.import_module("paper.scripts.run_taxi_quantity_interface_ablation")
        cls.factory = importlib.import_module("models.TPPs.CountAwareFactory")
        cls.numerical = importlib.import_module("paper.scripts.quantity_comparison_runtime")

    def frame(self, split):
        rows = [("A",0,"train",1.),("A",1,"train",2.),("A",2,"validation",5.),("A",3,"test",8.),
                ("B",0,"train",3.),("B",1,"test",9.),
                ("C",0,"validation",4.),("C",1,"test",10.),
                ("D",0,"train",4.),("D",1,"validation",6.),("D",2,"validation",7.),("D",3,"test",11.),
                ("E",0,"validation",4.)]
        data = self.pl.DataFrame({"oper_part_no":[r[0] for r in rows], "seq":[r[1] for r in rows],
            "chronological_split":[r[2] for r in rows], "demand_qty":[r[3] for r in rows],
            "delta_t":[1]*len(rows), "demand_dt":[20010101+r[1] for r in rows]}).sort(["oper_part_no","seq"])
        return data if split == "test" else data.filter(self.pl.col("chronological_split").is_in(["train","validation"]))

    def prepare(self, frame, split):
        return evaluator.prepare_diagnostic_population(frame, core=self.core, loader_module=self.loader_module,
            definition={"loader":{"lookback_weeks":168,"max_seq_len":3}},split=split,np=self.np)

    def test_real_frozen_loader_membership_for_both_splits(self):
        for split, count, target_entities in (("validation",3,["A","D","D"]),("test",4,["A","B","C","D"])):
            with self.subTest(split=split):
                loader,ds,identity,metadata,offsets,members = self.prepare(self.frame(split),split)
                self.assertIsInstance(members,set)
                self.assertEqual(members,{"A","D"})
                self.assertEqual(identity["target_count"],count)
                self.assertEqual(len(ds),count)
                self.assertEqual([str(ds.parts[p]) for p,_ in ds.index],target_entities)
                self.assertNotIn("C",members) # first-record-only Validation is not eligible
                self.assertNotIn("E",members) # no next-event target
                memberships=[]
                for p,end in ds.index:
                    raw=metadata.row(offsets[p]+end+1,named=True)
                    self.assertEqual(raw["chronological_split"],split)
                    self.assertEqual(str(raw["oper_part_no"]),str(ds.parts[p]))
                    memberships.append(str(ds.parts[p]) in members)
                self.assertEqual(memberships,[True,True,True] if split=="validation" else [True,False,False,True])
                for _,_,mask,_,_ in loader:
                    self.assertTrue(bool((mask.sum(1)>=2).all()))

    def test_worker_consumes_prepared_membership_without_overwrite(self):
        # Exercise the real worker's loading/mapping and membership expression.
        # Model/runtime are fabricated stubs; a sentinel stops before collector
        # insertion, so this is metadata regression coverage, not real inference.
        torch=self.torch
        class MembershipReached(Exception): pass
        class TrackedMembers(set):
            def __contains__(self, entity):
                raise MembershipReached("worker preserved eligible membership for " + entity)
        class FixtureModel(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer("fixture_identity",torch.tensor([1]))
        class FixtureObserver:
            def __init__(self,*args):pass
            def begin(self):pass
            def snapshot(self,lengths,gaps):return {}
            def suspended(self):return contextlib.nullcontext()
        def fixture_outputs(model,dts,mask,qty,**kwargs):
            return {"true_qty":qty[:,-1].float(),"pred_qty":qty[:,-1].float()+1.,
                    "time_loss":torch.ones(qty.shape[0]),"history_length":mask.sum(1)-1}
        original_prepare=evaluator.prepare_diagnostic_population
        def tracked_prepare(*args,**kwargs):
            values=list(original_prepare(*args,**kwargs));values[-1]=TrackedMembers(values[-1]);return tuple(values)
        for split in ("validation","test"):
            with self.subTest(split=split), tempfile.TemporaryDirectory(prefix="diagnostic_metadata_fixture_") as directory:
                root=Path(directory);frame=self.frame(split);frame.write_parquet(root/"fixture.parquet")
                frame.filter(self.pl.col("chronological_split").is_in(["train","validation"])).write_parquet(root/"fixture_training.parquet")
                (root/"fixture.pt").write_bytes(b"fabricated checkpoint placeholder, not selected data")
                (root/"metadata.json").write_text("{}")
                (root/"operation_integrity.json").write_text("{}")
                (root/"validation_gate.json").write_text("{}")
                model=FixtureModel();state=model.state_dict();state_sha=self.runner.canonical_state_dict_sha256(state)
                train=frame.filter(self.pl.col("chronological_split")=="train")["demand_qty"].to_numpy().astype(self.np.float64)
                stats={"all_train_quantity_sha256":hashlib.sha256(b"quantity_comparison_all_train_row_quantity_v1\0"+train.astype("<f8").tobytes()).hexdigest(),
                    "train_log_mean":float(self.np.log1p(train).mean()),"train_log_std":float(self.np.log1p(train).std())}
                loader,ds,identity,_,_,_=self.prepare(frame,split)
                definition={"loader":{"lookback_weeks":168,"max_seq_len":3},"statistics":stats,
                    "model":{"time_observation_contract":{"top_code":None}},"quantity_boundaries_all_train_rows":[2.,4.],
                    "inherited_data_identity":{"populations":{"validation":identity}}}
                job={"id":"fixture_membership_job","study":"history","dataset":"raf_spare_parts",
                    "arm":evaluator.HISTORY_ARMS[1],"seed":42,"checkpoint":{"relative_path":"fixture.pt","sha256":evaluator.sha(root/"fixture.pt")},
                    "checkpoint_metadata_path":"metadata.json","checkpoint_metadata_sha256":evaluator.sha(root/"metadata.json"),
                    "checkpoint_state_sha256":state_sha,"validation_reference":{"count":len(ds)}}
                c={"contract_sha256":"fabricated metadata regression only","jobs":[job],"sources":{"history":{}},
                    "runtime":{"torch_num_threads":4},"datasets":{"raf_spare_parts":{"definition":definition,"data_path":"fixture.parquet",
                    "data_sha256":evaluator.sha(root/"fixture.parquet"),"expected_test_count":len(ds),
                    "training_data_path":"fixture_training.parquet","training_data_sha256":evaluator.sha(root/"fixture_training.parquet"),
                    "diagnostics":{"input_history_boundaries":[1,2],"input_last_gap_boundaries":[1],"target_gap_boundaries":[1],"train_history_max":2}}}}
                telemetry_stub=SimpleNamespace(ReadOnlyTelemetry=FixtureObserver,AggregateDiagnostics=telemetry.AggregateDiagnostics)
                with contextlib.ExitStack() as stack:
                    for name,value in {"verify_contract":lambda *a:c,"reference_receipt":lambda *a:{},"verify_operations":lambda *a:{},
                        "verify_source":lambda *a:ROOT/"search_artifacts/titantpp_history_instacart_test_20261012_v1/sources/history",
                        "verify_imports":lambda *a:{},"verify_global_validation_gate":lambda *a:{},"check_runtime":lambda *a:None,
                        "validate_payload":lambda *a:None,"qualify":lambda *a:{},"load_telemetry":lambda:telemetry_stub,
                        "prepare_diagnostic_population":tracked_prepare,"PROJECT_PREFIXES":set()}.items():
                        stack.enter_context(mock.patch.object(evaluator,name,value))
                    stack.enter_context(mock.patch.object(self.factory,"build_count_aware_model",lambda *a,**k:(model,{})))
                    stack.enter_context(mock.patch.object(self.factory,"validate_checkpoint_route",lambda *a:None))
                    stack.enter_context(mock.patch.object(self.runner,"torch_load_checkpoint",lambda *a,**k:{"model_state_dict":state}))
                    stack.enter_context(mock.patch.object(self.numerical,"configure_runtime",lambda *a:None))
                    stack.enter_context(mock.patch.object(self.numerical,"runtime_identity",lambda *a:{}))
                    stack.enter_context(mock.patch.object(self.core,"target_outputs",fixture_outputs))
                    with self.assertRaisesRegex(MembershipReached,"worker preserved eligible membership"):
                        evaluator.worker(root,job["id"],split,"cpu")
                failure=json.loads((root/"results"/job["id"]/(("validation_replay" if split=="validation" else "test_metrics")+"_failure.json")).read_text())
                self.assertEqual(failure["error"],"MembershipReached")
                self.assertFalse(failure["receipt_written"])


if __name__ == "__main__":
    unittest.main()
