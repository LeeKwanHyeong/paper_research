"""Focused fabricated CPU regressions; selected checkpoints and remote GPU are not used."""
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import contextlib
import copy
import hashlib
import importlib
import importlib.util
import json
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "search_artifacts/titantpp_history_diagnostics_20261012_v2/sources/history"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluator = load(ROOT / "paper/scripts/evaluate_titantpp_raf_log_raw.py", "fixture_raf_evaluator")
telemetry = load(ROOT / "paper/scripts/titantpp_history_diagnostic_telemetry.py", "fixture_train_telemetry")


class ContractTests(unittest.TestCase):
    def test_split_cli_lock(self):
        with self.assertRaises(SystemExit):
            evaluator.main(["--root", "unused", "--split", "test", "--job-id", "unused", "--device", "cpu"])

    def test_empty_tail_is_not_zero_performance(self):
        aggregate = telemetry.AggregateDiagnostics([5.], [], [], [], 3)
        aggregate.add(entity="fixture", calendar="2001-01", last_input_calendar="2001-01",
            history_length=1, unclipped_history_length=1, truncated=False, series_in_validation=False,
            true_quantity=1., predicted_quantity=2., time_nll=3., recorded_gap=1., last_input_gap=1.,
            last_quantity=1., mean_quantity=1., features={})
        native = {"count":1, "qty_rmse":1., "qty_mae":1., "time_nll":3.,
                  "tail":{"count":0, "qty_rmse":None, "qty_mae":None, "time_nll":None}}
        self.assertTrue(evaluator.verify_aggregate_metrics(aggregate.finish(), native))
        native["tail"]["qty_rmse"] = 0.
        with self.assertRaisesRegex(ValueError, "empty Train tail"):
            evaluator.verify_aggregate_metrics(aggregate.finish(), native)


class FrozenTrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(FROZEN))
        import torch
        import polars as pl
        import numpy as np
        from paper.scripts import run_titantpp_cnn_history_controls as engine
        engine.install_hooks()
        from models.TPPs import CountAwareFactory as factory
        from models.TPPs import CountAwareTitanCNNGRU as adapter
        from paper.scripts.count_aware_tpp_backbone import core
        from simple_lab_test.search.common import runner
        from paper.scripts import quantity_comparison_runtime as numerical
        cls.torch,cls.pl,cls.np = torch,pl,np
        cls.factory,cls.adapter,cls.core,cls.runner,cls.numerical = factory,adapter,core,runner,numerical
        cls.loader = importlib.import_module("paper.scripts.run_taxi_quantity_interface_ablation")
        cls.definition = json.loads((ROOT/"search_artifacts/titantpp_history_diagnostics_20261012_v2/contract.json").read_text())["datasets"]["raf_spare_parts"]["definition"]

    def frame(self):
        rows = [("A",0,"train",1.),("A",1,"train",7.),("A",2,"train",10.),
                ("A",3,"validation",12.),("A",4,"test",20.),
                ("B",0,"validation",4.),("B",1,"test",22.),
                ("C",0,"train",5.),("C",1,"train",20.),("C",2,"validation",4.)]
        return self.pl.DataFrame({"oper_part_no":[r[0] for r in rows],"seq":[r[1] for r in rows],
            "chronological_split":[r[2] for r in rows],"demand_qty":[r[3] for r in rows],
            "delta_t":[1]*len(rows),"demand_dt":[20010101+r[1] for r in rows]}).sort(["oper_part_no","seq"])

    def prepare(self, frame):
        return evaluator.prepare_population(frame,core=self.core,loader_module=self.loader,
            definition={"loader":{"lookback_weeks":168,"max_seq_len":3}},split="train",np=self.np)

    def test_real_frozen_loader_train_membership_and_context(self):
        frame = self.frame().filter(self.pl.col("chronological_split").is_in(["train","validation"]))
        loader, dataset, identity, metadata, offsets, members = self.prepare(frame)
        self.assertEqual(identity["target_count"],3)
        self.assertEqual(set(map(str,dataset.parts)),{"A","B","C"})
        self.assertEqual(members,{"A","C"}) # first-record-only Validation is not an eligible target
        self.assertIsInstance(members,set)
        self.assertEqual([str(dataset.parts[p]) for p,_ in dataset.index],["A","A","C"])
        for p,end in dataset.index:
            raw=metadata.row(offsets[p]+end+1,named=True)
            self.assertEqual(raw["chronological_split"],"train")
            self.assertEqual(str(raw["oper_part_no"]),str(dataset.parts[p]))
            self.assertTrue(all(v=="train" for v in dataset.split_lists[p][:end+2]))
        _,_,mask,_,quantities=next(iter(loader))
        self.assertEqual(quantities[:,-1].tolist(),[7.,10.,20.])
        self.assertTrue(bool((mask.sum(1)>=2).all()))
        train_only=frame.filter(self.pl.col("chronological_split")=="train")
        self.assertNotEqual(self.prepare(train_only)[2]["target_identity_sha256"],identity["target_identity_sha256"])

    def fixture_worker(self, root, *, split="train", changed_bytes=False, population_drift=False):
        torch=self.torch
        frame=self.frame();frame.write_parquet(root/"fixture.parquet")
        (root/"fixture.pt").write_bytes(b"fabricated placeholder; never a selected model")
        (root/"metadata.json").write_text("{}")
        (root/"operation_integrity.json").write_text("{}")
        class FixtureModel(torch.nn.Module):
            def __init__(self):
                super().__init__();self.register_buffer("fixture_identity",torch.tensor([1]))
        class FixtureObserver:
            def __init__(self,*args): self.values={}
            def begin(self): pass
            def snapshot(self,*args):
                self.values["quantity_head_logit"]=model.fixture_logit
                return {}
            def suspended(self): return contextlib.nullcontext()
            def close(self): pass
            def interpretation(self): return {"fabricated_test_fixture":True}
        model=FixtureModel();state=model.state_dict();state_sha=self.runner.canonical_state_dict_sha256(state)
        prepared_frame=frame.filter(self.pl.col("chronological_split").is_in(["train","validation"]))
        loader,dataset,identity,metadata,offsets,_=evaluator.prepare_population(prepared_frame, core=self.core,
            loader_module=self.loader, definition={"loader":{"lookback_weeks":168,"max_seq_len":3}},split=split,np=self.np)
        train=prepared_frame.filter(self.pl.col("chronological_split")=="train")["demand_qty"].to_numpy().astype(self.np.float64)
        stats={"all_train_quantity_sha256":hashlib.sha256(b"quantity_comparison_all_train_row_quantity_v1\0"+train.astype("<f8").tobytes()).hexdigest(),
            "train_log_mean":float(self.np.log1p(train).mean()),"train_log_std":float(self.np.log1p(train).std())}
        definition={"loader":{"batch_size":128,"num_workers":0,"drop_last":False,"lookback_weeks":168,"max_seq_len":3},
            "statistics":stats,"model":{"time_observation_contract":{"top_code":None}},
            "quantity_boundaries_all_train_rows":[2.,8.]}
        source={"closure":"fabricated source lineage only","revision":"fixture"}
        job={"id":"fixture_train_job","study":"history","dataset":"raf_spare_parts","arm":evaluator.HISTORY_ARMS[1],
            "seed":42,"selected_epoch":3,"initial_state_sha256":"fixture initial state",
            "frozen_source_closure":source["closure"],"source_revision_history":[],
            "checkpoint":{"relative_path":"fixture.pt","sha256":evaluator.sha(root/"fixture.pt")},
            "checkpoint_metadata_path":"metadata.json","checkpoint_metadata_sha256":evaluator.sha(root/"metadata.json"),
            "checkpoint_state_sha256":state_sha}
        spec={"definition":definition,"data_sha256":evaluator.sha(root/"fixture.parquet"),
            "training_data_path":"fixture.parquet","training_data_sha256":evaluator.sha(root/"fixture.parquet"),
            "diagnostics":{"input_history_boundaries":[1,2],"input_last_gap_boundaries":[1],"target_gap_boundaries":[1],"train_history_max":2}}
        c={"schema":evaluator.KIND,"contract_sha256":"fabricated fixture only","jobs":[job],"sources":{"history":source},
            "runtime":{"torch_num_threads":4},"datasets":{"raf_spare_parts":spec},
            "h1":{"expected_populations":{"raf_spare_parts":{split:identity}},"reference_receipts":{"fixture_train_job":{split:{"sha256":"fixture"}}}}}
        if changed_bytes:
            spec["training_data_sha256"]="0"*64
        if population_drift:
            c["h1"]["expected_populations"]["raf_spare_parts"][split]={**identity,"target_quantity_sha256":"0"*64}
        invocation={"forward":0,"checkpoint":0}
        def fixture_outputs(model,dts,mask,qty,**kwargs):
            self.assertFalse(model.training)
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(torch.is_inference_mode_enabled())
            invocation["forward"]+=1
            truth=qty[:,-1].float()
            model.fixture_logit=torch.log(torch.expm1(torch.log1p(truth+1.)))
            location=torch.nn.functional.softplus(model.fixture_logit)
            return {"true_qty":truth,"pred_qty":torch.expm1(location),
                    "log_qty_loss":torch.nn.functional.mse_loss(location,torch.log1p(truth),reduction="none"),
                    "time_loss":torch.ones(qty.shape[0]),"history_length":mask.sum(1)-1}
        def checkpoint(*args,**kwargs):
            invocation["checkpoint"]+=1
            return {"model_state_dict":state}
        qualified={key:True for key in evaluator.original.QUALIFICATION_KEYS if key not in {"parameters_unchanged","finite_outputs"}}
        modules=SimpleNamespace(ReadOnlyTelemetry=FixtureObserver,AggregateDiagnostics=telemetry.AggregateDiagnostics,
                                __file__=telemetry.__file__)
        # Reference digest construction uses the real frozen loader's chronological
        # target and causal history metadata; no selected model/data is loaded.
        target_digest,truth_digest=hashlib.sha256(),hashlib.sha256()
        metric_module=importlib.import_module("paper.scripts.evaluate_titantpp_history_width")
        raw_metrics=metric_module.QuantityMetrics([2.,8.])
        processed=0
        for _,dts,mask,parts,qty in loader:
            q=qty[:,-1].float();location=torch.nn.functional.softplus(torch.log(torch.expm1(torch.log1p(q+1.))))
            raw_metrics.add(q.double().numpy(),torch.expm1(location).double().numpy(),self.np.ones(len(q)))
            hm=mask.clone();hm[:,-1]=False
            lengths=hm.sum(1);last=qty[:,-2].double().numpy()
            mean=((qty.double()*hm).sum(1)/lengths).numpy()
            for offset in range(len(q)):
                part,end=dataset.index[processed+offset]
                raw=metadata.row(offsets[part]+end+1,named=True)
                entity,seq=str(dataset.parts[part]),int(dataset.seq_lists[part][end+1]);date=str(raw["demand_dt"])
                target=hashlib.sha256(evaluator.json_bytes([job["dataset"],spec["data_sha256"],entity,seq,split,date])).hexdigest()
                row={"target_id":target,"entity_id":entity,"seq":seq,"split":split,"recorded_date":date,
                     "raw_quantity":float(q[offset]),"recorded_gap":float(dts[offset,-1]),
                     "history_length":int(lengths[offset]),"last_quantity":float(last[offset]),"mean_quantity":float(mean[offset])}
                target_digest.update((target+"\n").encode());truth_digest.update(evaluator.json_bytes([row[k] for k in evaluator.TRUTH_FIELDS])+b"\n")
            processed+=len(q)
        reference={"metrics":{"split":split,**raw_metrics.finish()},"original_population_identity":identity,
                   "target_identity_sha256":target_digest.hexdigest(),"truth_sha256":truth_digest.hexdigest()}
        return c,job,model,modules,fixture_outputs,checkpoint,qualified,invocation,reference

    def run_fixture(self, root, *, split="train", changed_bytes=False, population_drift=False):
        c,job,model,modules,outputs,checkpoint,qualified,invocation,reference=self.fixture_worker(root,split=split if split in ("train","validation") else "train",changed_bytes=changed_bytes,population_drift=population_drift)
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(sys,"path",sys.path.copy()))
            for name,value in {"verify_contract":lambda *a:c,"verify_operations":lambda *a:{},
                "verify_source":lambda *a:FROZEN,"verify_imports":lambda *a:{},"check_runtime":lambda *a:None,
                "validate_payload":lambda *a:None,"qualify":lambda *a:qualified,"load_telemetry":lambda:modules,
                "reference_receipt":lambda *a:reference,"PROJECT_PREFIXES":set()}.items():
                stack.enter_context(mock.patch.object(evaluator,name,value))
            stack.enter_context(mock.patch.object(self.factory,"build_count_aware_model",lambda *a,**k:(model,{})))
            stack.enter_context(mock.patch.object(self.factory,"validate_checkpoint_route",lambda *a:None))
            stack.enter_context(mock.patch.object(self.runner,"torch_load_checkpoint",checkpoint))
            stack.enter_context(mock.patch.object(self.numerical,"configure_runtime",lambda *a:None))
            stack.enter_context(mock.patch.object(self.numerical,"runtime_identity",lambda *a:{}))
            stack.enter_context(mock.patch.object(self.core,"target_outputs",outputs))
            result=evaluator.worker(root,job["id"],split,"cpu")
        return result,invocation,c,job

    def test_real_worker_finishes_train_only_eval_and_preserves_membership(self):
        with tempfile.TemporaryDirectory(prefix="train_worker_fixture_") as directory:
            root=Path(directory)
            result,invocation,c,job=self.run_fixture(root)
            self.assertEqual(result,0);self.assertEqual(invocation,{"forward":1,"checkpoint":1})
            receipt=json.loads((root/"results/fixture_train_job/train_metrics.json").read_text())
            self.assertEqual(receipt["prediction_rows"],3)
            self.assertAlmostEqual(receipt["metrics"]["qty_rmse"],1.,places=5)
            self.assertAlmostEqual(receipt["metrics"]["qty_bias"],1.,places=5)
            self.assertEqual(receipt["metrics"]["tail"]["count"],2)
            self.assertTrue(receipt["log_native_per_target_verified"])
            self.assertEqual(receipt["native_reference_match"],"PASS")
            self.assertTrue(receipt["inference_mode_enabled"])
            aggregate=json.loads((root/receipt["telemetry_path"]).read_text())
            self.assertEqual(sum(v["count"] for v in aggregate["strata"]["series_validation_membership"]),3)
            self.assertEqual(aggregate["strata"]["series_validation_membership"][0]["key"],"shared")
            altered=copy.deepcopy(receipt);altered["optimizer_updates"]=1
            with self.assertRaisesRegex(ValueError,"Unapproved operation"):
                evaluator.verify_evaluation_receipt(altered,c,job,{"metrics":receipt["metrics"],"original_population_identity":receipt["original_population_identity"],"target_identity_sha256":receipt["target_identity_sha256"],"truth_sha256":receipt["truth_sha256"]})

    def test_real_worker_validation_population_and_native_log_aggregates(self):
        with tempfile.TemporaryDirectory(prefix="raf_val_worker_fixture_") as directory:
            root=Path(directory)
            result,invocation,c,job=self.run_fixture(root,split="validation")
            self.assertEqual(result,0);self.assertEqual(invocation,{"forward":1,"checkpoint":1})
            receipt=json.loads((root/"results/fixture_train_job/validation_metrics.json").read_text())
            self.assertEqual(receipt["prediction_rows"],2)
            aggregate=json.loads((root/receipt["telemetry_path"]).read_text())["log_raw"]
            self.assertEqual(aggregate["count"],2)
            self.assertEqual(aggregate["cells"]["0"]["model"]["count"],0)
            self.assertIsNone(aggregate["cells"]["0"]["model"]["log_mse_native"])
            self.assertTrue(evaluator.verify_log_raw_aggregates(aggregate,receipt["metrics"],receipt["simple_quantity_baselines"]))
            altered=copy.deepcopy(aggregate)
            altered["cells"]["all"]["model"]["count"]+=1
            with self.assertRaisesRegex(ValueError,"counts differ"):
                evaluator.verify_log_raw_aggregates(altered,receipt["metrics"],receipt["simple_quantity_baselines"])

    def test_log_observation_rejects_scalar_and_wrong_loss(self):
        torch=self.torch
        with torch.inference_mode():
            logits=torch.tensor([0.,1.,2.]);truth=torch.tensor([1.,5.,200.])
            location=torch.nn.functional.softplus(logits)
            value={"true_qty":truth,"pred_qty":torch.expm1(location),
                   "log_qty_loss":torch.nn.functional.mse_loss(location,torch.log1p(truth),reduction="none")}
            observed=evaluator.native_log_observations(value,logits,torch)
            self.assertTrue(all(x<=0 for x in observed["log_error"]))
            self.assertTrue(any(x<0 for x in observed["log_error"]))
            wrong={**value,"log_qty_loss":value["log_qty_loss"].mean()}
            with self.assertRaisesRegex(ValueError,"per-target"):
                evaluator.native_log_observations(wrong,logits,torch)
            wrong={**value,"log_qty_loss":value["log_qty_loss"]+1.}
            with self.assertRaisesRegex(ValueError,"does not match"):
                evaluator.native_log_observations(wrong,logits,torch)

    def test_worker_rejects_other_splits_and_byte_or_population_drift(self):
        for kwargs,reason in (({"split":"test"},"Only Train/Validation"),
                              ({"changed_bytes":True},"Frozen input byte"),({"population_drift":True},"eligible Train/Validation population")):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory(prefix="train_negative_fixture_") as directory:
                root=Path(directory)
                with self.assertRaisesRegex(ValueError,reason):
                    self.run_fixture(root,**kwargs)
                self.assertFalse((root/"results/fixture_train_job/train_metrics.json").exists())
                if "split" not in kwargs:
                    failure=json.loads((root/"results/fixture_train_job/train_metrics_failure.json").read_text())
                    self.assertFalse(failure["receipt_written"])
                    self.assertEqual(failure["error"],"ValueError")

    def test_three_real_frozen_routes_hook_noninterference_in_inference_mode(self):
        torch=self.torch;definition=self.definition
        config={k:v for k,v in definition["model"].items() if k not in ("backbone","lambda_log_qty","lambda_tail","time_head_lr_multiplier")}
        for arm in evaluator.HISTORY_ARMS:
            with self.subTest(arm=arm),torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                model,_=self.factory.build_count_aware_model(arm,**config,train_log_mean=1.2,train_log_std=.8,max_seq_len=84)
                model.eval().requires_grad_(False)
                with torch.no_grad():
                    for name,p in model.named_parameters():
                        if "multilag_detail" in name or "causal_" in name and name.endswith("_kernel"):
                            p.copy_(torch.randn_like(p)*.03)
                before=self.runner.canonical_state_dict_sha256(model.state_dict())
                dts=torch.tensor([[0.,0.,1.,2.,1.,3.,1.,2.],[1.,2.,1.,3.,1.,2.,1.,3.]])
                qty=torch.tensor([[0.,0.,2.,4.,1.,3.,2.,5.],[1.,2.,3.,4.,3.,2.,1.,6.]])
                mask=torch.ones_like(dts,dtype=torch.bool);mask[0,:2]=False
                with torch.no_grad():plain=self.core.target_outputs(model,dts,mask,qty,lambda_log_qty=1.)
                observer=telemetry.ReadOnlyTelemetry(model,arm,torch,self.adapter.causal_depthwise_residual)
                with torch.inference_mode():
                    observer.begin();observed=self.core.target_outputs(model,dts,mask,qty,lambda_log_qty=1.)
                    features=observer.snapshot(observed["history_length"],dts[:,-1])
                    log_values=evaluator.native_log_observations(observed,observer.values["quantity_head_logit"],torch)
                    self.assertEqual(len(log_values["log_native_loss"]),2)
                    self.assertTrue(all(torch.equal(plain[k],observed[k]) for k in plain))
                    self.assertTrue(all(all(self.np.isfinite(values)) for values in features.values()))
                    with observer.suspended():
                        qualification=evaluator.qualify(torch,model,self.core.target_outputs,dts,mask,qty,observed,
                            definition["model"]["time_observation_contract"]["top_code"])
                    self.assertTrue(all(qualification.values()))
                observer.close()
                gradient_observer=telemetry.ReadOnlyTelemetry(model,arm,torch,self.adapter.causal_depthwise_residual)
                model.requires_grad_(True)
                cudnn_enabled=torch.backends.cudnn.enabled
                with torch.backends.cudnn.flags(enabled=False),torch.enable_grad():
                    with gradient_observer.suspended():
                        plain=self.core.target_outputs(model,dts,mask,qty,lambda_log_qty=1.)
                        plain_grad=torch.autograd.grad(plain["joint_loss"].sum(),tuple(model.parameters()),allow_unused=True)
                    gradient_observer.begin()
                    hooked=self.core.target_outputs(model,dts,mask,qty,lambda_log_qty=1.)
                    hooked_grad=torch.autograd.grad(hooked["joint_loss"].sum(),tuple(model.parameters()),allow_unused=True)
                self.assertTrue(all((a is None and b is None) or (a is not None and b is not None and torch.equal(a,b)) for a,b in zip(plain_grad,hooked_grad)))
                self.assertTrue(all(torch.equal(plain[k],hooked[k]) for k in plain))
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                self.assertEqual(torch.backends.cudnn.enabled,cudnn_enabled)
                gradient_observer.close();model.requires_grad_(False)
                self.assertEqual(self.runner.canonical_state_dict_sha256(model.state_dict()),before)


if __name__=="__main__":unittest.main()
