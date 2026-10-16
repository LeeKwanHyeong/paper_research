#!/usr/bin/env python3
"""Read-only Intermittent Train/Validation location/scale cross scoring.

Six physical joint workers each execute three immutable model-split passes.
Only paired sufficient statistics are persisted: no arrays, fitting, new Test
inference, forward replacement or encoder intervention is performed.
"""
from __future__ import annotations
import argparse
from bisect import bisect_left
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback

KIND = "titantpp_intermittent_time_cross_20261012_v1"
PARENT = "76248e2174d7bd651d821f649ad860030da985d59da88a8ce50725b476f55ccf"
VALIDATION_PARENT = "70e8e22dbe85aa93cb5dc555b06303511b0e5e7f511771b455b4b243ddb3ad1d"
SOURCE = "c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a"
DATASET = "intermittent_frozen_5000"
FULL = "titantpp_cnn_gru54"
HISTORY_ARMS = (FULL, FULL + "_correction_disabled", FULL + "_current_only")
SEEDS = (42, 52, 62)
SPLITS = ("train", "validation")
PROJECT_PREFIXES = {"models", "paper", "data_loader", "simple_lab_test"}
TRUTH_FIELDS = ("target_id", "entity_id", "seq", "split", "recorded_date", "raw_quantity", "recorded_gap", "history_length", "last_quantity", "mean_quantity")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_here = Path(__file__).resolve().parent
_original_path = _here / "original_evaluate.py"
if not _original_path.exists():
    _original_path = _here / "evaluate_titantpp_history_instacart_frozen_test.py"
original = load_module(_original_path, "_original_time_cross_primitives")
require, read, sha, json_bytes, canonical = original.require, original.read, original.sha, original.json_bytes, original.canonical
confined, write_exclusive = original.confined, original.write_exclusive


def verify_contract(root):
    c = read(Path(root) / "contract.json")
    require(c.get("contract_sha256") == canonical({k:v for k,v in c.items() if k != "contract_sha256"}), "Cross contract SHA mismatch")
    require(c.get("schema") == KIND and c.get("parent_contract_sha256") == PARENT, "Foreign time cross contract")
    require(c.get("expected_seeds") == list(SEEDS), "Cross seed matrix changed")
    require(set(c["sources"]) == {"history"} and c["sources"]["history"]["closure"] == SOURCE
            and set(c["datasets"]) == {DATASET}, "Cross science source or dataset changed")
    expected = {(DATASET, arm, seed) for arm in HISTORY_ARMS for seed in SEEDS}
    require(len(c["jobs"]) == 9 and len({j["id"] for j in c["jobs"]}) == 9
            and {(j["dataset"], j["arm"], j["seed"]) for j in c["jobs"]} == expected
            and all(j["study"] == "history" for j in c["jobs"]), "Cross must bind exactly Intermittent9")
    scope = c["time_cross"]
    require(scope["evaluation_splits"] == list(SPLITS) and scope["new_fits"] == scope["test_inference"] == 0
            and scope["model_split_passes"] == 18 and scope["joint_worker_count"] == 6
            and scope["concurrent_joint_workers"] == 1 and scope["resident_models_max"] == 3,
            "Cross Train/Validation-only execution authorization differs")
    spec = c["datasets"][DATASET]
    definition = spec["definition"]
    require(spec.get("training_data_path") and spec.get("training_data_sha256"), "Missing original Train/Validation export")
    require(definition["loader"]["batch_size"] == 128 and definition["loader"]["num_workers"] == 0
            and definition["loader"]["drop_last"] is False, "Cross original loader differs")
    model = definition["model"]
    require(model["time_head_mode"] == "heteroscedastic_lognormal_duration" and model["time_scale"] == 3.
            and model["time_sigma_floor"] == .001 and model["time_observation_contract"]["mode"] == "positive_integer_round_clamp_v1"
            and model["time_observation_contract"]["top_code"] is None, "Original integer duration law changed")
    require(spec["diagnostics"]["target_gap_boundaries"] == [1.,3.,5.,7.], "Frozen Train duration boundaries changed")
    for job in c["jobs"]:
        require(job["id"] and all(x not in job["id"] for x in ("/", "\\", "..")), "Unsafe cross job ID")
        require(job["selected_epoch"] >= 1 and job["frozen_source_closure"] == SOURCE, "Cross checkpoint lineage changed")
        require(job["validation_reference"]["split"] == "validation", "Original selector binding differs")
        require(set(scope["reference_receipts"][job["id"]]) == set(SPLITS), "Cross reference scope differs")
    return c


def verify_operations(root, c):
    seal = read(Path(root) / "operation_integrity.json")
    require(seal.get("contract_sha256") == c["contract_sha256"], "Foreign cross operation seal")
    require({"control/evaluate.py", "control/original_evaluate.py"} <= set(seal["files"]), "Missing cross operation primitives")
    for relative, digest in seal["files"].items():
        require(sha(confined(root, relative)) == digest, "Cross operation file changed: " + relative)
    return seal


def reference_receipt(root, c, job, split):
    ref = c["time_cross"]["reference_receipts"][job["id"]][split]
    path = confined(root, ref["relative_path"])
    require(sha(path) == ref["sha256"], "Cross native reference bytes changed")
    r = read(path)
    require(r["status"] == "complete" and r["split"] == split, "Foreign native reference")
    expected_parent = PARENT if split == "train" else VALIDATION_PARENT
    require(ref["contract_sha256"] == r.get("contract_sha256") == expected_parent,
            "Native reference parent contract differs")
    for key in ("id", "study", "dataset", "arm", "seed", "selected_epoch", "initial_state_sha256"):
        require(r[key] == job[key], "Native model identity differs: " + key)
    require(r["checkpoint_sha256"] == job["checkpoint"]["sha256"] and r["state_sha256"] == job["checkpoint_state_sha256"]
            and r["state_after_sha256"] == r["state_sha256"] and r["source_closure"] == SOURCE,
            "Native checkpoint/source reference differs")
    require(r.get("retrained") is False and r.get("reselected") is False and r.get("raw_predictions_written") is False,
            "Native reference is not immutable selected inference")
    return r


class TimeObserver:
    """Read actual shared quantity/time hidden; never replace a forward method."""
    def __init__(self, model, torch):
        require(model.memory_mode == "static_hard_lmm" and model.time_head_mode == "heteroscedastic_lognormal_duration", "Time shared-state route differs")
        require(not model.training and all(not p.requires_grad for p in model.parameters()), "Observer requires frozen eval model")
        self.model, self.torch, self.hidden, self.enabled = model, torch, None, True
        self.handle = model.quantity_head.register_forward_pre_hook(self.capture)

    def capture(self, module, args):
        if self.enabled:
            self.hidden = args[0].detach()

    def begin(self):
        self.hidden = None

    def parameters(self, gap):
        require(not self.torch.is_grad_enabled() and self.hidden is not None, "Missing actual shared hidden")
        location, scale, _ = self.model._lognormal_time_terms(self.hidden, gap)
        require(location.shape == scale.shape == gap.shape and bool(self.torch.isfinite(location).all())
                and bool(self.torch.isfinite(scale).all()) and bool((scale > 0).all()), "Malformed actual time parameters")
        return location.detach(), scale.detach()

    @contextmanager
    def suspended(self):
        before = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = before

    def close(self):
        self.handle.remove()
        self.hidden = None


def paired_scores(torch, mass, gap, full, control, own_full, own_control):
    """Four eventwise original integer masses; exact original diagonal guard."""
    require(torch.equal(full[2], control[2]) and torch.equal(full[3], control[3]), "Paired target/history alignment differs")
    result = {}
    for name, location, scale in (("ff", full[0], full[1]), ("fc", full[0], control[1]),
                                  ("cf", control[0], full[1]), ("cc", control[0], control[1])):
        result[name] = -mass(gap, location, scale, time_scale=3., top_code=None)
        require(result[name].dtype == torch.float64 and bool(torch.isfinite(result[name]).all()), "Nonfinite float64 cross score")
    require(torch.equal(result["ff"], own_full) and torch.equal(result["cc"], own_control), "Original paired NLL cells were not exactly reproduced")
    return result


class CrossAccumulator:
    """Paired score moments including difference covariance from actual events."""
    def __init__(self, boundaries):
        self.boundaries = boundaries
        self.cells = {}

    def add(self, gaps, scores):
        names = ("ff", "fc", "cf", "cc")
        require(set(scores) == set(names) and all(len(v) == len(gaps) for v in scores.values()), "Cross arrays are not aligned")
        for index, gap in enumerate(gaps):
            require(math.isfinite(gap) and gap >= 1 and gap == int(gap), "Recorded target gap differs")
            x = {name:float(scores[name][index]) for name in names}
            require(all(math.isfinite(v) for v in x.values()), "Nonfinite score event")
            x.update(scale_effect=x["fc"]-x["ff"], location_effect=x["cf"]-x["ff"],
                     interaction=x["cc"]-x["fc"]-x["cf"]+x["ff"], total_effect=x["cc"]-x["ff"])
            for group, key in (("all", "all"), ("target_gap_bin", str(bisect_left(self.boundaries,gap)))):
                cell = self.cells.setdefault((group,key), {"group":group,"key":key,"count":0,"sums":{k:0. for k in x},"squares":{k:0. for k in x}})
                cell["count"] += 1
                for name, value in x.items():
                    cell["sums"][name] += value
                    cell["squares"][name] += value * value

    def finish(self):
        cells = []
        for cell in self.cells.values():
            require(all(math.isfinite(v) for v in [*cell["sums"].values(),*cell["squares"].values()]), "Cross sufficient-statistic overflow")
            cells.append({**cell, "means":{name:value/cell["count"] for name,value in cell["sums"].items()}})
        return sorted(cells,key=lambda cell:(cell["group"],cell["key"]))


def prepare_population(frame, core, loader_module, definition, split, np):
    frame = core.prepare_count_frame(frame)
    loader = loader_module.make_loader(frame, target_split=split, batch_size=128, shuffle=False, generator=None,
        lookback_weeks=definition["loader"]["lookback_weeks"], max_seq_len=definition["loader"]["max_seq_len"])
    dataset = loader.dataset
    identity = original.population_identity(dataset, split, np)
    metadata = frame.select(["oper_part_no","seq","chronological_split","demand_dt"])
    offsets, total = [], 0
    for sequences in dataset.seq_lists:
        offsets.append(total)
        total += len(sequences)
    require(total == frame.height, "Frozen loader metadata mapping differs")
    return loader, dataset, identity, metadata, offsets


def verify_joint_receipt(root, c, seed, split):
    """Validate six physical joint receipts and their honest shared-PID passes."""
    root = Path(root).resolve()
    require(seed in SEEDS and split in SPLITS, "Foreign joint receipt identity")
    output = root / "results" / ("seed"+str(seed)) / (split+"_joint")
    require(not (output/"failure.json").exists(), "Preserved joint failure prevents completion")
    joint = read(output/"receipt.json")
    require(joint.get("status") == "complete" and joint.get("schema") == KIND
            and joint.get("contract_sha256") == c["contract_sha256"]
            and joint.get("seed") == seed and joint.get("split") == split
            and joint.get("actual_model_split_passes") == joint.get("resident_models_max") == 3,
            "Foreign or incomplete joint receipt")
    require(joint["pid"] > 0 and joint["ppid"] > 0 and isinstance(joint["argv"], list)
            and joint["started_unix"] <= joint["completed_unix"], "Joint actual process/timing invalid")
    for key in ("new_fits", "test_inference", "calibration_fits"):
        require(type(joint.get(key)) is int and joint[key] == 0, "Unapproved joint operation: "+key)
    paired_path = confined(root, joint["paired_scores_path"])
    require(sha(paired_path) == joint["paired_scores_sha256"], "Joint paired score bytes changed")
    paired = read(paired_path)
    require(paired["schema"] == KIND+"_paired_scores" and paired["contract_sha256"] == c["contract_sha256"]
            and paired["seed"] == seed and paired["split"] == split and paired["count"] == joint["count"]
            and paired["paired_score_cells"] == 8 and paired["new_cross_cells"] == 4
            and {p["control_arm"] for p in paired["pairs"]} == set(HISTORY_ARMS[1:]), "Paired score matrix differs")
    jobs = [j for j in c["jobs"] if j["seed"] == seed]
    require(set(joint["model_receipts"]) == {j["id"] for j in jobs}, "Joint omits model receipts")
    for job in jobs:
        path = root/"results"/job["id"]/(split+"_metrics.json")
        require(sha(path) == joint["model_receipts"][job["id"]], "Model pass bytes changed")
        r = read(path)
        require(r.get("status") == "complete" and r.get("schema") == KIND and r.get("contract_sha256") == c["contract_sha256"]
                and r.get("split") == split and r.get("physical_joint_worker") is True
                and r.get("shared_pid_model_split_pass") is True and r.get("joint_worker_seed") == seed,
                "Physical/model-pass accounting differs")
        for key in ("id", "study", "dataset", "arm", "seed", "selected_epoch", "initial_state_sha256"):
            require(r[key] == job[key], "Model pass identity differs: "+key)
        require(r["checkpoint_sha256"] == job["checkpoint"]["sha256"]
                and r["state_sha256"] == r["state_after_sha256"] == job["checkpoint_state_sha256"]
                and r["source_closure"] == r["frozen_source_closure"] == SOURCE, "Model pass science identity differs")
        for key in ("pid", "ppid", "argv"):
            require(r[key] == joint[key], "Model pass must share actual joint process: "+key)
        require(joint["started_unix"] <= r["joint_started_unix"] <= r["started_unix"] <= r["completed_unix"] <= joint["completed_unix"],
                "Model pass timing outside joint execution")
        require(r["prediction_rows"] == r["expected_target_count"] == joint["count"]
                and r["target_identity_sha256"] == paired["target_identity_sha256"]
                and r["truth_sha256"] == paired["truth_sha256"], "Paired target history differs")
        for key in ("native_six_metrics_match", "native_target_history_match", "original_diagonal_eventwise_bitwise_match"):
            require(r[key] == "PASS", "Native cross integrity did not pass: "+key)
        require(r["gradients_enabled"] is False and r["raw_predictions_written"] is False
                and r["retrained"] is False and r["reselected"] is False and r["held_out_test_evaluated"] is False
                and r["no_forward_replacement"] is True and r["inference_mode_enabled"] is True,
                "Unauthorized scientific intervention")
        for key in ("new_fits", "optimizer_updates", "test_inference", "calibration_fits", "encoder_path_interventions"):
            require(type(r.get(key)) is int and r[key] == 0, "Unapproved model-pass operation: "+key)
        require(set(r["qualification"]) == original.QUALIFICATION_KEYS and all(v is True for v in r["qualification"].values()),
                "Cross causality qualification is incomplete")
        ref = reference_receipt(root,c,job,split)
        original.compare_validation_metrics(r["metrics"],ref["metrics"])
        require(r["original_population_identity"] == ref["original_population_identity"]
                and r["target_identity_sha256"] == ref["target_identity_sha256"] and r["truth_sha256"] == ref["truth_sha256"],
                "Original native target population differs")
    return joint


def worker(root, seed, split, device):
    root = Path(root).resolve()
    c = verify_contract(root)
    require(seed in SEEDS and split in SPLITS and device in ("cpu","cuda"), "Only authorized seed/Train/Validation inference")
    jobs = [next(j for j in c["jobs"] if j["seed"] == seed and j["arm"] == arm) for arm in HISTORY_ARMS]
    started_unix, started = time.time(), time.monotonic()
    output = root / "results" / ("seed"+str(seed)) / (split+"_joint")
    require(not (output/"receipt.json").exists() and not (output/"failure.json").exists(), "Existing joint receipt/failure prohibits repeat")
    write_exclusive(output/"claim.json", {"seed":seed,"split":split,"pid":os.getpid(),"ppid":os.getppid(),"argv":sys.argv,"contract_sha256":c["contract_sha256"]})
    phase = "integrity"
    try:
        verify_operations(root,c)
        operation_sha = sha(root/"operation_integrity.json")
        script_sha = sha(__file__)
        source = c["sources"]["history"]
        source_root = original.verify_source(root,source)
        references = {j["arm"]:reference_receipt(root,c,j,split) for j in jobs}
        spec = c["datasets"][DATASET]
        definition = spec["definition"]
        data_path = confined(root,spec["training_data_path"])
        require(sha(data_path) == spec["training_data_sha256"], "Frozen Train/Validation input changed")
        require(not any(n.split(".")[0] in PROJECT_PREFIXES for n in sys.modules), "Fresh joint worker required")
        sys.dont_write_bytecode = True
        sys.path[:] = [str(source_root)] + [p for p in sys.path if p and not Path(p).resolve().is_relative_to(root) and not (Path(p).resolve()/"paper").is_dir()]
        os.environ["MPLCONFIGDIR"] = str(output/"runtime_cache/matplotlib")
        os.environ["XDG_CACHE_HOME"] = str(output/"runtime_cache")
        if device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import numpy as np
        import polars as pl
        import torch
        numerical = importlib.import_module("paper.scripts.quantity_comparison_runtime")
        execution_device = "cuda:0" if device == "cuda" else "cpu"
        numerical.configure_runtime(execution_device,c["runtime"]["torch_num_threads"])
        runtime = numerical.runtime_identity(execution_device)
        original.check_runtime(runtime,c["runtime"],device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        engine = importlib.import_module(original.route_for(jobs[0]));engine.install_hooks()
        factory = importlib.import_module("models.TPPs.CountAwareFactory")
        core = importlib.import_module("paper.scripts.count_aware_tpp_backbone.core")
        loader_module = importlib.import_module("paper.scripts.run_taxi_quantity_interface_ablation")
        runner = importlib.import_module("simple_lab_test.search.common.runner")
        metrics_module = importlib.import_module("paper.scripts.evaluate_titantpp_history_width")
        mass = importlib.import_module("models.TPPs.positive_integer_time").positive_integer_log_mass
        original.verify_imports(source_root,source)
        phase = "population"
        frame = pl.scan_parquet(data_path).filter(pl.col("chronological_split").is_in(list(SPLITS))).collect().sort(["oper_part_no","seq"])
        original.validate_frame(frame,definition,"validation",np,pl)
        require(set(frame["chronological_split"].unique().to_list()) <= set(SPLITS), "Test materialization is forbidden")
        train_q = frame.filter(pl.col("chronological_split") == "train")["demand_qty"].to_numpy().astype(np.float64)
        stats = definition["statistics"]
        require(hashlib.sha256(b"quantity_comparison_all_train_row_quantity_v1\0"+train_q.astype("<f8").tobytes()).hexdigest() == stats["all_train_quantity_sha256"], "Frozen Train quantity identity changed")
        for key,value in (("train_log_mean",float(np.log1p(train_q).mean())),("train_log_std",float(np.log1p(train_q).std()))):
            require(math.isclose(value,stats[key],rel_tol=0,abs_tol=1e-12), "Train-only initial scalar changed")
        loader,dataset,identity,metadata,offsets = prepare_population(frame,core,loader_module,definition,split,np)
        expected = definition["inherited_data_identity"]["populations"][split]
        require(identity == expected and len(dataset) == expected["target_count"] > 0, "Frozen eligible population differs")
        del frame,train_q
        models,observers,accumulators,qualifications,inference_seconds,before = {},{},{},{},{},{}
        modelpass_started,modelpass_ended = {},{}
        config = {k:v for k,v in definition["model"].items() if k not in ("backbone","lambda_log_qty","lambda_tail","time_head_lr_multiplier")}
        for job in jobs:
            checkpoint = confined(root,job["checkpoint"]["relative_path"])
            metadata_path = confined(root,job["checkpoint_metadata_path"])
            require(sha(checkpoint) == job["checkpoint"]["sha256"] and sha(metadata_path) == job["checkpoint_metadata_sha256"], "Checkpoint/metadata bytes changed")
            payload = runner.torch_load_checkpoint(checkpoint,map_location="cpu")
            original.validate_payload(payload,job,source,definition,read(metadata_path));factory.validate_checkpoint_route(payload,job["arm"])
            require(runner.canonical_state_dict_sha256(payload["model_state_dict"]) == job["checkpoint_state_sha256"], "Checkpoint tensor identity changed")
            model,_ = factory.build_count_aware_model(job["arm"],**config,train_log_mean=stats["train_log_mean"],train_log_std=stats["train_log_std"],max_seq_len=definition["loader"]["max_seq_len"])
            model.load_state_dict(payload["model_state_dict"],strict=True);model.to(execution_device).eval().requires_grad_(False)
            before[job["arm"]] = runner.canonical_state_dict_sha256(model.state_dict())
            require(before[job["arm"]] == job["checkpoint_state_sha256"], "Loaded tensor identity changed")
            models[job["arm"]],observers[job["arm"]] = model,TimeObserver(model,torch)
            accumulators[job["arm"]] = metrics_module.QuantityMetrics(definition["quantity_boundaries_all_train_rows"])
            inference_seconds[job["arm"]] = 0.
            del payload
        cross = {arm:CrossAccumulator(spec["diagnostics"]["target_gap_boundaries"]) for arm in HISTORY_ARMS[1:]}
        target_digest,truth_digest = hashlib.sha256(),hashlib.sha256()
        processed = 0
        phase = "inference"
        with torch.inference_mode():
            for batch_index,(_,dts,mask,part_indices,quantities) in enumerate(loader):
                dts,mask,quantities = (v.to(execution_device) for v in (dts,mask,quantities))
                outputs,parameters = {},{}
                for arm in HISTORY_ARMS:
                    if arm not in modelpass_started:modelpass_started[arm] = time.time()
                    if device == "cuda":torch.cuda.synchronize()
                    batch_start = time.monotonic();observers[arm].begin()
                    value = core.target_outputs(models[arm],dts,mask,quantities,lambda_log_qty=1.)
                    mu,sigma = observers[arm].parameters(dts[:,-1])
                    if device == "cuda":torch.cuda.synchronize()
                    inference_seconds[arm] += time.monotonic()-batch_start
                    require(all(bool(torch.isfinite(value[k]).all()) for k in ("true_qty","pred_qty","time_loss")), "Nonfinite original output")
                    if batch_index == 0:
                        with observers[arm].suspended():
                            qualifications[arm] = original.qualify(torch,models[arm],core.target_outputs,dts,mask,quantities,value,None)
                    outputs[arm],parameters[arm] = value,(mu,sigma,value["true_qty"],value["history_length"])
                    accumulators[arm].add(*(value[k].cpu().numpy().astype(np.float64) for k in ("true_qty","pred_qty","time_loss")))
                    modelpass_ended[arm] = time.time()
                for arm in HISTORY_ARMS[1:]:
                    scores = paired_scores(torch,mass,dts[:,-1],parameters[FULL],parameters[arm],outputs[FULL]["time_loss"],outputs[arm]["time_loss"])
                    cross[arm].add(dts[:,-1].cpu().tolist(),{k:v.cpu().tolist() for k,v in scores.items()})
                history_mask = mask.cpu().clone();history_mask[:,-1] = False
                history_length = history_mask.sum(1)
                quantity_cpu = quantities.cpu().double()
                last = quantity_cpu[:,-2].numpy();mean = ((quantity_cpu*history_mask).sum(1)/history_length).numpy()
                q = outputs[FULL]["true_qty"].cpu().numpy().astype(np.float64)
                gaps = dts[:,-1].cpu().tolist();lengths = outputs[FULL]["history_length"].cpu().tolist()
                for offset in range(len(q)):
                    part,end = dataset.index[processed+offset]
                    require(int(part_indices[offset]) == part and int(history_length[offset]) == lengths[offset], "Loader order/history differs")
                    raw = metadata.row(offsets[part]+end+1,named=True)
                    entity,sequence = str(dataset.parts[part]),int(dataset.seq_lists[part][end+1])
                    require(raw["chronological_split"] == split and str(raw["oper_part_no"]) == entity and int(raw["seq"]) == sequence and raw["demand_dt"] is not None,"Target metadata alignment differs")
                    date = str(raw["demand_dt"])
                    target_id = hashlib.sha256(json_bytes([DATASET,spec["data_sha256"],entity,sequence,split,date])).hexdigest()
                    row = {"target_id":target_id,"entity_id":entity,"seq":sequence,"split":split,"recorded_date":date,"raw_quantity":float(q[offset]),"recorded_gap":float(gaps[offset]),"history_length":int(lengths[offset]),"last_quantity":float(last[offset]),"mean_quantity":float(mean[offset])}
                    target_digest.update((target_id+"\n").encode());truth_digest.update(json_bytes([row[k] for k in TRUTH_FIELDS])+b"\n")
                processed += len(q)
        require(processed == expected["target_count"],"Incomplete joint target population")
        all_metrics = {arm:{"split":split,**a.finish()} for arm,a in accumulators.items()}
        for job in jobs:
            arm = job["arm"];ref = references[arm]
            original.compare_validation_metrics(all_metrics[arm],ref["metrics"])
            require(identity == ref["original_population_identity"] and target_digest.hexdigest() == ref["target_identity_sha256"] and truth_digest.hexdigest() == ref["truth_sha256"],"Native target/full truth-history SHA differs")
            if split == "validation":original.compare_validation_metrics(all_metrics[arm],job["validation_reference"])
        pairs = []
        for arm,a in cross.items():
            cells = a.finish();cell = next(x for x in cells if x["group"] == "all")
            require(cell["count"] == processed,"Cross paired coverage differs")
            for name,native_arm in (("ff",FULL),("cc",arm)):
                require(math.isclose(cell["means"][name],all_metrics[native_arm]["time_nll"],rel_tol=1e-5,abs_tol=1e-5),"Cross diagonal/native mean differs")
            pairs.append({"full_arm":FULL,"control_arm":arm,"seed":seed,"split":split,"cells":cells})
        phase = "final_integrity"
        for job in jobs:
            arm = job["arm"]
            require(runner.canonical_state_dict_sha256(models[arm].state_dict()) == before[arm] and sha(confined(root,job["checkpoint"]["relative_path"])) == job["checkpoint"]["sha256"] and sha(confined(root,job["checkpoint_metadata_path"])) == job["checkpoint_metadata_sha256"],"Model/checkpoint/metadata changed")
            observers[arm].close()
        require(sha(data_path) == spec["training_data_sha256"] and sha(__file__) == script_sha and verify_contract(root) == c,"Input/adapter/contract changed")
        original.verify_source(root,source);verify_operations(root,c)
        require(sha(root/"operation_integrity.json") == operation_sha,"Cross operation seal changed")
        imports = original.verify_imports(source_root,source)
        payload = {"schema":KIND+"_paired_scores","contract_sha256":c["contract_sha256"],"seed":seed,"split":split,"count":processed,"target_identity_sha256":target_digest.hexdigest(),"truth_sha256":truth_digest.hexdigest(),"target_gap_boundaries":spec["diagnostics"]["target_gap_boundaries"],"pairs":pairs,"paired_score_cells":8,"new_cross_cells":4,"target_gap_is_descriptive_outcome":True,"raw_predictions_written":False,"counterfactual_head_scores_only":True,"calibration_fits":0,"new_fits":0,"test_inference":0}
        paired_path = output/"time_cross.json";write_exclusive(paired_path,payload)
        model_receipts = {}
        for job in jobs:
            arm = job["arm"]
            r = {"status":"complete","schema":KIND,"contract_sha256":c["contract_sha256"],**{k:job[k] for k in ("id","study","dataset","arm","seed","selected_epoch","initial_state_sha256")},"split":split,"evaluation_scope":"intermittent_train_validation_readonly_time_cross","checkpoint_sha256":job["checkpoint"]["sha256"],"state_sha256":before[arm],"state_after_sha256":before[arm],"source_closure":SOURCE,"frozen_source_closure":SOURCE,"source_revision":source["revision"],"source_revision_history":job["source_revision_history"],"metrics":all_metrics[arm],"prediction_rows":processed,"expected_target_count":processed,"original_population_identity":identity,"target_identity_sha256":target_digest.hexdigest(),"truth_sha256":truth_digest.hexdigest(),"data_sha256":spec["training_data_sha256"],"loader":definition["loader"],"qualification":{**qualifications[arm],"parameters_unchanged":True,"finite_outputs":True},"native_six_metrics_match":"PASS","native_target_history_match":"PASS","original_diagonal_eventwise_bitwise_match":"PASS","inference_mode_enabled":True,"gradients_enabled":False,"raw_predictions_written":False,"no_forward_replacement":True,"retrained":False,"reselected":False,"held_out_test_evaluated":False,"new_fits":0,"optimizer_updates":0,"test_inference":0,"calibration_fits":0,"encoder_path_interventions":0,"runtime":runtime,"pid":os.getpid(),"ppid":os.getppid(),"argv":sys.argv,"physical_joint_worker":True,"shared_pid_model_split_pass":True,"joint_worker_seed":seed,"started_unix":modelpass_started[arm],"completed_unix":modelpass_ended[arm],"joint_started_unix":started_unix,"inference_seconds":inference_seconds[arm],"operation_integrity_sha256":operation_sha,"evaluator_sha256":script_sha,"imported_frozen_modules":imports,"reference":c["time_cross"]["reference_receipts"][job["id"]][split],"paired_scores_path":paired_path.relative_to(root).as_posix(),"paired_scores_sha256":sha(paired_path)}
            path = root/"results"/job["id"]/(split+"_metrics.json");write_exclusive(path,r);model_receipts[job["id"]] = sha(path)
        joint = {"status":"complete","schema":KIND,"contract_sha256":c["contract_sha256"],"seed":seed,"split":split,"actual_model_split_passes":3,"model_receipts":model_receipts,"paired_scores_path":paired_path.relative_to(root).as_posix(),"paired_scores_sha256":sha(paired_path),"count":processed,"pid":os.getpid(),"ppid":os.getppid(),"argv":sys.argv,"started_unix":started_unix,"completed_unix":time.time(),"elapsed_seconds":time.monotonic()-started,"inference_seconds":inference_seconds,"new_fits":0,"test_inference":0,"calibration_fits":0,"resident_models_max":3,"cuda_peak_allocated_bytes":torch.cuda.max_memory_allocated() if device == "cuda" else 0,"cuda_peak_reserved_bytes":torch.cuda.max_memory_reserved() if device == "cuda" else 0,"cpu_peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == "darwin" else 1024)}
        write_exclusive(output/"receipt.json",joint)
        verify_joint_receipt(root,c,seed,split)
        print(json.dumps({k:joint[k] for k in ("status","seed","split","actual_model_split_passes","count","elapsed_seconds")}),flush=True)
        return 0
    except BaseException as error:
        write_exclusive(output/"failure.json",{"status":"failed","phase":phase,"seed":seed,"split":split,"contract_sha256":c["contract_sha256"],"error":type(error).__name__,"message":str(error),"traceback":traceback.format_exc(),"elapsed_seconds":time.monotonic()-started,"automatic_retry":False})
        raise


def native_probe(root, study, arm, device):
    root = Path(root).resolve();c = verify_contract(root)
    require(study == "history" and arm in HISTORY_ARMS,"Foreign cross native route")
    job = next(j for j in c["jobs"] if j["arm"] == arm);source = c["sources"]["history"]
    verify_operations(root,c);operation_sha = sha(root/"operation_integrity.json")
    source_root = original.verify_source(root,source)
    require(not any(n.split(".")[0] in PROJECT_PREFIXES for n in sys.modules),"Fresh cross native process required")
    sys.dont_write_bytecode = True
    sys.path[:] = [str(source_root)]+[p for p in sys.path if p and not Path(p).resolve().is_relative_to(root) and not (Path(p).resolve()/"paper").is_dir()]
    cache = root/"control/native_probe_runtime_cache"/arm
    os.environ["MPLCONFIGDIR"] = str(cache/"matplotlib");os.environ["XDG_CACHE_HOME"] = str(cache)
    if device == "cpu":os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    numerical = importlib.import_module("paper.scripts.quantity_comparison_runtime")
    execution_device = "cuda:0" if device == "cuda" else "cpu"
    numerical.configure_runtime(execution_device,c["runtime"]["torch_num_threads"])
    runtime = numerical.runtime_identity(execution_device)
    if device == "cuda":original.check_runtime(runtime,c["runtime"],device)
    engine = importlib.import_module(original.route_for(job));engine.install_hooks()
    factory = importlib.import_module("models.TPPs.CountAwareFactory")
    core = importlib.import_module("paper.scripts.count_aware_tpp_backbone.core")
    runner = importlib.import_module("simple_lab_test.search.common.runner")
    mass = importlib.import_module("models.TPPs.positive_integer_time").positive_integer_log_mass
    definition = c["datasets"][DATASET]["definition"]
    config = {k:v for k,v in definition["model"].items() if k not in ("backbone","lambda_log_qty","lambda_tail","time_head_lr_multiplier")}
    with torch.random.fork_rng(devices=[] if device == "cpu" else [0]):
        torch.manual_seed(42)
        model,_ = factory.build_count_aware_model(arm,**config,train_log_mean=1.2,train_log_std=.8,max_seq_len=definition["loader"]["max_seq_len"])
        model.to(execution_device).eval().requires_grad_(False)
        with torch.inference_mode():
            for name,parameter in model.named_parameters():
                if "multilag_detail" in name or "causal_q_kernel" in name or "causal_k_kernel" in name or "causal_v_kernel" in name:parameter.copy_(torch.randn_like(parameter)*.03)
            before = runner.canonical_state_dict_sha256(model.state_dict())
            dts = torch.tensor([[0.,0.,1.,2.,1.,3.,1.,2.],[1.,2.,1.,3.,1.,2.,1.,3.]],device=execution_device)
            quantities = torch.tensor([[0.,0.,2.,4.,1.,3.,2.,5.],[1.,2.,3.,4.,3.,2.,1.,6.]],device=execution_device)
            mask = torch.ones_like(dts,dtype=torch.bool);mask[0,:2] = False
            baseline = core.target_outputs(model,dts,mask,quantities,lambda_log_qty=1.)
            observer = TimeObserver(model,torch);observer.begin()
            value = core.target_outputs(model,dts,mask,quantities,lambda_log_qty=1.)
            mu,sigma = observer.parameters(dts[:,-1])
            require(all(torch.equal(baseline[k],value[k]) for k in baseline),"Read-only time hook altered original synthetic output")
            require(torch.equal(-mass(dts[:,-1],mu,sigma,time_scale=3.,top_code=None),value["time_loss"]),"Synthetic actual time terms not exact")
            with observer.suspended():checks = original.qualify(torch,model,core.target_outputs,dts,mask,quantities,value,None)
            observer.close()
            require(runner.canonical_state_dict_sha256(model.state_dict()) == before,"Native observer changed tensors")
    original.verify_source(root,source);verify_operations(root,c)
    require(sha(root/"operation_integrity.json") == operation_sha,"Native operation seal changed")
    imports = original.verify_imports(source_root,source)
    print(json.dumps({"status":"PASS","study":study,"arm":arm,"device":device,"runtime":runtime,"native_pinned_contract_checked":device == "cuda","contract_sha256":c["contract_sha256"],"source_closure":SOURCE,"qualification":{**checks,"parameters_unchanged":True,"finite_outputs":True},"hooked_and_unhooked_all_original_outputs_bitwise_equal":True,"actual_integer_diagonal_eventwise_bitwise_equal":True,"telemetry_finite":True,"optimizer_updates":0,"checkpoint_reads":0,"real_data_loaded":False,"held_out_test_evaluated":False,"imported_frozen_modules":imports,"pid":os.getpid(),"ppid":os.getppid(),"argv":sys.argv},allow_nan=False),flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True,type=Path)
    parser.add_argument("--seed",type=int,choices=SEEDS)
    parser.add_argument("--split",choices=SPLITS)
    parser.add_argument("--native-probe",action="store_true")
    parser.add_argument("--study",choices=("history",))
    parser.add_argument("--arm",choices=HISTORY_ARMS)
    parser.add_argument("--device",required=True,choices=("cpu","cuda"))
    args = parser.parse_args(argv)
    if args.native_probe:
        require(args.study is not None and args.arm is not None and args.seed is None and args.split is None,"Native cross probe takes --study/--arm only")
        return native_probe(args.root,args.study,args.arm,args.device)
    require(args.seed is not None and args.split is not None and args.study is None and args.arm is None,"Joint cross worker takes --seed/--split only")
    return worker(args.root,args.seed,args.split,args.device)


if __name__ == "__main__":
    raise SystemExit(main())
