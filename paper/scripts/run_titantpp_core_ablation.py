"""Frozen 36-fit Titan ablation campaign, fresh owned process for every fit."""
from __future__ import annotations
import argparse
from copy import deepcopy
import faulthandler
import gc, io, json, math, os, signal, statistics, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from models.TPPs.CountAwareTitanCoreAblation import ARMS, ROLE, MODES
from paper.scripts import run_local_detail_benchmark as original
from paper.scripts.run_local_detail_benchmark import (
    common, shared, base, require, read, VARIANT, TIME_METRIC, audit_arm,
    audit_replay_accounting, verify_start_memory, verify_process_environment,
    apply_process_environment, storage_limits, ArmBudget, Budget)
ALL_ARMS=('titantpp','titantpp_local_detail',*ARMS)
SCHEMA='titantpp_core_ablation_dual_host_v1'


def scheduled_jobs(c,host):
    return [j for j in c['jobs'] if j['host']==host]


def validate_execution_contract(c,verify_source=True):
    require(c['schema']==SCHEMA and c['execution_arms']==list(ARMS),'Wrong campaign')
    expected=[{'host':h,'dataset':d,'seed':s,'arm':a,'id':f'{d}__{s}__{a}'}
        for h,ds in [('5080',['yellow_trip_hourly','intermittent_frozen_5000']),('5090',['insta_market_basket'])]
        for d in ds for s in (42,52,62) for a in ARMS]
    require(c['jobs']==expected and c['new_fits']==36 and c['endpoint_replays']==72,'Scope changed')
    require(c['training']=={'batch_size':128,'maximum_epochs':300,'minimum_epochs':40,'patience':40,
        'monitor':'validation_raw_quantity_rmse','tie':'strict_earliest_finite_minimum','warm_start':False},'Training changed')
    require(c['limits']['total_wall_seconds']==120*3600 and c['limits']['max_gpu_hours_aggregate']==240,'Budget changed')
    require(c['source']['files_sha256']==common.sha_json(c['source']['files']),'Source identity malformed')
    if verify_source:
        for path,digest in c['source']['files'].items():
            require(common.sha_file(ROOT/path)==digest,'Source drift: '+path)
    return c


def verify_authorization(c,approval,permit,host,training=False):
    validate_execution_contract(c)
    require(host in ('5080','5090') and approval.get('approved') is True
        and approval.get('hosts')==['5080','5090'] and approval.get('user_instruction')
        and approval['contract_sha256']==common.sha_json(c),'Approval mismatch')
    require(permit['contract_sha256']==common.sha_json(c) and permit['approval_sha256']==common.sha_json(approval)
        and permit['deadline_unix']-permit['started_at_unix']==c['limits']['total_wall_seconds']
        and permit['started_at_unix']<=time.time()<permit['deadline_unix'],'Invalid or expired permit')
    if training:
        root=Path(c['hosts'][host]['root']);q=read(root/'qualification/receipt.json')
        require(q['status']=='passed' and q['contract_sha256']==common.sha_json(c)
            and q['host']==host and q['approval_sha256']==common.sha_json(approval)
            and q['synthetic_optimizer_updates']==60 and q['source_files_sha256']==c['source']['files_sha256']
            and q['deadline_unix']==permit['deadline_unix'],'Native qualification mismatch')
        validate_qualification_measurements(c,q)
    return c['hosts'][host]


def build_model(data,backbone):
    from models.TPPs.CountAwareFactory import build_count_aware_model
    require(backbone in ALL_ARMS,'Unknown model')
    config={k:v for k,v in data['model'].items() if k not in ('backbone','lambda_log_qty','lambda_tail','time_head_lr_multiplier')}
    return build_count_aware_model(backbone,**config,train_log_mean=data['statistics']['train_log_mean'],
        train_log_std=data['statistics']['train_log_std'],max_seq_len=data['loader']['max_seq_len'])


def initialization(data,seed):
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    states={};rngs={}
    with torch.random.fork_rng(devices=[]):
        for arm in ALL_ARMS:
            torch.manual_seed(seed);model,_=build_model(data,arm)
            states[arm]=model.state_dict();rngs[arm]=torch.get_rng_state().clone()
        for arm in ALL_ARMS:
            require(torch.equal(rngs[arm],rngs['titantpp']),'Initialization consumed extra RNG')
            for k,v in states['titantpp'].items():
                if arm=='titantpp_no_static_lmm' and k.startswith('lmm.'):continue
                require(torch.equal(v,states[arm][k]),'Common initial tensor changed: '+arm+'/'+k)
    return {a:canonical_state_dict_sha256(s) for a,s in states.items()}


def training_args(c,data,output,arm):
    require(arm in ARMS,'Only four new fits authorized')
    a=base.training_args({**c,'model_role':ROLE},data,output)
    a.epochs=300;a.min_epochs=40;a.early_stopping_patience=40
    a.execution_role='fresh_core_ablation_validation';return a


def time_interface(data,frame,c):
    value=base.time_interface(data,frame,c)
    value['backbone_design']={'schema':SCHEMA,'variants':c.get('variants',{}),'training':c['training'],'comparison':c['comparison']}
    return value


def check_baseline_prefix(c,host,dataset,seed,exposure):
    for row in c['reuse']:
        if row['host']!=host or row['dataset']!=dataset or row['seed']!=seed:continue
        prior=read(Path(row['remote_run'])/'exposure.json')
        n=min(len(prior['train']),len(exposure['train']))
        require(prior['train'][:n]==exposure['train'][:n],'Reused baseline train prefix changed')
        hashes={x['batch_order_sha256'] for e in (prior,exposure) for x in e['validation']}
        require(len(hashes)==1,'Reused validation order/population differs')
    return True


def verify_baselines(c,host):
    for row in c['reuse']:
        if row['host']!=host:continue
        for name,digest in row['file_sha256'].items():
            require(common.sha_file(Path(row['remote_run'])/name)==digest,'Reused baseline file changed')


def replay_checkpoint(path, data, frame, budget, *, device="cuda:0", expected_arm=None, expected_identity=None, expected_initial=None, expected_epoch=None, expected_seed=42):
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    payload = torch_load_checkpoint(Path(path), map_location="cpu")
    if expected_arm is not None:
        from paper.scripts.count_aware_tpp_backbone.training import checkpoint_monitor_spec
        selector = checkpoint_monitor_spec("validation_raw_quantity_rmse")
        require(payload.get("backbone") == expected_arm and payload.get("seed") == expected_seed
                and payload.get("variant") == VARIANT, "Replay arm/seed/objective mismatch")
        require(payload.get("resume_identity") == expected_identity
                and payload.get("interface_meta") == expected_identity["interface_meta"]
                and payload.get("initial_state_sha256") == expected_initial,
                "Replay source/initialization/interface identity mismatch")
        revision = expected_identity["arguments"]["source_revision"] if "source_revision" in expected_identity.get("arguments", {}) else None
        if revision is not None:
            require(payload.get("source_revision") == revision and payload.get("source_revision_history") == [revision], "Replay source revision mismatch")
        require(payload.get("checkpoint_monitor") == "validation_raw_quantity_rmse"
                and payload.get("checkpoint_monitor_history_key") == selector["history_key"]
                and payload.get("checkpoint_selection") == selector["selection"], "Replay selector mismatch")
        require(payload.get("epoch", payload.get("best_epoch")) == expected_epoch, "Replay epoch mismatch")
    head = payload.get("interface", {}).get("time_head", {})
    # The shared trainer writes its interface under interface_meta.
    if not head:
        head = payload.get("interface_meta", {}).get("time_head", {})
    require(head.get("observation_likelihood") == data["model"]["time_observation_contract"],
            "Replay observation likelihood metadata mismatch")
    result = _replay_validation(payload, path, data, frame, budget, device=device)
    result["time_metric"] = TIME_METRIC
    return result

def _replay_validation(payload, path, data, frame, budget_check, *, device="cuda:0"):
    """One streaming validation pass with fixed train-derived strata; no records."""
    import numpy as np
    import torch
    from models.TPPs.CountAwareFactory import validate_checkpoint_route
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    require(payload.get("backbone") in ALL_ARMS, "Unexpected endpoint backbone")
    validate_checkpoint_route(payload, payload["backbone"])
    require(payload["evaluation_scope"] == "validation_only"
            and payload["held_out_test_evaluated"] is False, "Replay split changed")
    state_digest = canonical_state_dict_sha256(payload["model_state_dict"])
    require(payload.get("model_state_sha256") == state_digest, "Checkpoint state digest differs")
    model, _ = build_model(data, payload["backbone"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    loader = make_loader(frame, target_split="validation", **{
        key: data["loader"][key] for key in ("batch_size", "lookback_weeks", "max_seq_len")},
        shuffle=False, generator=None)
    q_bounds, h_bounds = data["quantity_boundaries_all_train_rows"], data["history_boundaries"]
    partitions = [("quantity", q_bounds), ("history", h_bounds)]
    if "additional_history_boundaries" in data:
        partitions.append(("additional_history", data["additional_history_boundaries"]))
    accumulator = {"overall": [0, 0., 0., 0.], "body": [0, 0., 0., 0.], "tail": [0, 0., 0., 0.]}
    for kind, bounds in partitions:
        accumulator.update({f"{kind}_{i}": [0, 0., 0., 0.] for i in range(len(bounds) + 1)})
    with torch.no_grad():
        for _, dts, mask, _, quantities in loader:
            budget_check()
            result = target_outputs(model, dts.to(device), mask.to(device), quantities.to(device), lambda_log_qty=1.)
            q = result["true_qty"].cpu().numpy().astype(np.float64)
            pred = result["pred_qty"].cpu().numpy().astype(np.float64)
            t = result["time_loss"].cpu().numpy().astype(np.float64)
            h = result["history_length"].cpu().numpy()
            require(np.isfinite(q).all() and np.isfinite(pred).all() and np.isfinite(t).all(),
                    "Nonfinite checkpoint replay")
            masks = {"overall": np.ones(q.shape, dtype=bool), "body": q <= q_bounds[2], "tail": q > q_bounds[3]}
            for kind, bounds in partitions:
                values = q if kind == "quantity" else h
                ids = np.searchsorted(bounds, values, side="left")
                masks.update({f"{kind}_{i}": ids == i for i in range(len(bounds) + 1)})
            error = pred - q
            for key, selected in masks.items():
                acc = accumulator[key]
                acc[0] += int(selected.sum())
                acc[1] += float(np.abs(error[selected]).sum())
                acc[2] += float(np.square(error[selected]).sum())
                acc[3] += float(t[selected].sum())
    budget_check()
    def finish(acc):
        n, absolute, squared, temporal = acc
        return {"count": n, "qty_mae": absolute / n if n else None, "qty_sse": squared,
            "qty_rmse": math.sqrt(squared / n) if n else None,
            "time_nll": temporal / n if n else None}
    result = {**finish(accumulator["overall"]), "body": finish(accumulator["body"]),
        "tail": finish(accumulator["tail"]), "quantity_boundaries": q_bounds,
        "history_boundaries": h_bounds, "quantity_cells": [], "history_cells": [],
        "checkpoint_path": str(path), "state_sha256": state_digest,
        "evaluation_scope": "validation_only", "held_out_test_evaluated": False}
    for kind, bounds in partitions:
        result[kind + "_boundaries"] = bounds
        result[kind + "_cells"] = [{"bin": i, **finish(accumulator[f"{kind}_{i}"])}
                                    for i in range(len(bounds) + 1)]
    require(result["count"] == data["inherited_data_identity"]["populations"]["validation"]["target_count"],
            "Replay validation target count changed")
    audit_replay_accounting(result)
    return result

def validate_qualification_measurements(contract, receipt):
    if True:
        require(receipt.get("seed") == contract["qualification_seed"], "Qualification seed differs")
    require(receipt.get("real_data_loaded") is False and receipt.get("held_out_evaluated") is False,
            "Qualification split changed")
    rows = receipt.get("costs", [])
    require(len(rows)==2 and {r["length"] for r in rows}=={64,256}, "Qualification lengths differ")
    require(receipt.get("checks") == {"finite_joint_updates":True,"B_local_initial_output_equal":True,
        "shared_head_contract":True,"causality":True,"state_restore":True}, "Qualification checks missing")
    memory = receipt["runtime"]["gpu"]["total_memory_bytes"]
    for row in rows:
        require(row["batch_size"]==128 and set(row["measurements"])==set(ALL_ARMS), "Qualification workload differs")
        for value in row["measurements"].values():
            times = value["step_seconds"]
            require(len(times)==3 and all(math.isfinite(t) and t>0 for t in times)
                    and value["median_step_seconds"]==statistics.median(times)
                    and 0 < value["peak_allocated_bytes"] <= .8*memory
                    and value["parameters"] > 0, "Invalid native cost measurement")

def qualify(contract, *, device="cpu", batch_size=4, lengths=(8,20), budget=lambda:None, seed=42):
    import torch
    from paper.scripts import run_titantpp_core_ablation as runner
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from paper.scripts.count_aware_tpp_backbone.training import build_optimizer
    native = device == "cuda:0"
    require(device in ("cpu","cuda:0"), "Explicit CPU or qualified CUDA device required")
    require((not native) or (batch_size==128 and tuple(lengths)==(64,256)), "Native workload changed")
    previous_threads=torch.get_num_threads()
    if not native: torch.set_num_threads(1)
    rows=[]; total_updates=0; configs={}; parameters={}
    try:
        for length in lengths:
            data=deepcopy(contract["datasets"][2 if length<=64 else 1])
            data["loader"]["max_seq_len"]=length
            count=batch_size*length
            dts=(torch.arange(count,device=device).reshape(batch_size,length)%7+1).float()
            qty=(torch.arange(count,device=device).reshape(batch_size,length)%19+1).float()
            mask=torch.ones_like(dts,dtype=torch.bool)
            mask[::2,-2:]=False;dts[~mask]=0;qty[~mask]=0
            measurements={}; original_output=None
            for arm in ALL_ARMS:
                budget();torch.manual_seed(seed)
                model,meta=runner.build_model(data,arm);model.to(device).eval()
                configs[arm]=meta["time_head"]
                parameters[arm]=sum(p.numel() for p in model.parameters() if p.requires_grad)
                captured=[]; original_log=model.log_observation_dt
                def capture(hidden,target):
                    captured.append(hidden.detach().clone());return original_log(hidden,target)
                model.log_observation_dt=capture
                with torch.no_grad():
                    out=target_outputs(model,dts,mask,qty,lambda_log_qty=1.)
                    changed_dt=dts.clone();changed_qty=qty.clone()
                    target=mask.sum(1)-1;idx=torch.arange(batch_size,device=device)
                    changed_dt[idx,target]=2;changed_qty[idx,target]+=100
                    changed_dt[~mask]=999;changed_qty[~mask]=999
                    changed=target_outputs(model,changed_dt,mask,changed_qty,lambda_log_qty=1.)
                    require(torch.equal(captured[0],captured[1]) and torch.equal(out["pred_qty"],changed["pred_qty"]),
                            "Target or padding leaked into prediction")
                model.log_observation_dt=original_log
                if arm=="titantpp":
                    original_output={k:out[k].detach().clone() for k in ("pred_qty","time_loss","joint_loss")}
                if arm=="titantpp_local_detail":
                    require(all(torch.equal(out[k],v) for k,v in original_output.items()),"B/local initial outputs differ")
                require(meta["time_head"]["observation_likelihood"]==data["model"]["time_observation_contract"],
                        "Shared time head observation changed")
                optimizer=build_optimizer(model,lr=.001,time_head_lr_multiplier=1.)
                model.train();times=[]
                if native:torch.cuda.reset_peak_memory_stats()
                for step in range(5):
                    budget()
                    if native:torch.cuda.synchronize()
                    start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                    value=target_outputs(model,dts,mask,qty,lambda_log_qty=1.)["joint_loss"].mean()
                    require(torch.isfinite(value).item(),"Nonfinite synthetic objective")
                    value.backward()
                    require(all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None),
                            "Nonfinite synthetic gradient")
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                    if native:torch.cuda.synchronize()
                    elapsed=time.perf_counter()-start
                    if step>=2:times.append(elapsed)
                    total_updates+=1
                model.eval()
                with torch.no_grad():before_restore=target_outputs(model,dts,mask,qty,lambda_log_qty=1.)
                saved=io.BytesIO();torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict()},saved);saved.seek(0)
                payload=torch.load(saved,map_location=device,weights_only=False)
                with torch.no_grad():
                    next(model.parameters()).add_(1.)
                optimizer.param_groups[0]['lr']=.123
                model.load_state_dict(payload['model'],strict=True);optimizer.load_state_dict(payload['optimizer'])
                model.eval()
                with torch.no_grad():restored=target_outputs(model,dts,mask,qty,lambda_log_qty=1.)
                require(all(torch.equal(before_restore[k],restored[k]) for k in ('pred_qty','time_loss','joint_loss')),
                        "Restored model predictions changed")
                require(optimizer.param_groups[0]['lr']==payload['optimizer']['param_groups'][0]['lr'],
                        "Optimizer state restore failed")
                measurements[arm]={"parameters":parameters[arm],"step_seconds":times,
                    "median_step_seconds":statistics.median(times),
                    "peak_allocated_bytes":torch.cuda.max_memory_allocated() if native else None}
                del optimizer,model,payload,saved,out,changed,restored,before_restore,value;gc.collect()
                if native:torch.cuda.empty_cache()
            rows.append({"length":length,"batch_size":batch_size,"measurements":measurements})
    finally:
        if not native:torch.set_num_threads(previous_threads)
    return {"status":"passed" if native else "cpu_observation_complete","device":device,"seed":seed,
        "synthetic_optimizer_updates":total_updates,"real_data_loaded":False,"held_out_evaluated":False,
        "checks":{"finite_joint_updates":True,"B_local_initial_output_equal":True,"shared_head_contract":True,
                  "causality":True,"state_restore":True},"costs":rows,
        "interpretation":"CPU step times are not GPU epoch estimates" if not native else "native synthetic cost, not measured real-data epoch time"}

def run_arm(contract, permit, host, qualification, budget, job):
    import torch
    from paper.scripts.count_aware_tpp_backbone import training
    verify_process_environment(contract, host, active=True)
    runtime = common.runtime_check(contract, host)
    require(runtime == qualification["runtime"], "Qualified runtime changed")
    output = Path(contract["hosts"][host]["output_dir"])
    requested_arm=job["arm"]
    require(job in scheduled_jobs(contract,host), "Unapproved job")
    for dataset_id, seed in [(job["dataset"],job["seed"])]:
        data = next(d for d in contract["datasets"] if d["dataset_id"] == dataset_id)
        frame, metadata = base.prepare_admitted_data(data)
        interface, initial = time_interface(data, frame, contract), initialization(data, seed)
        if recovery := contract.get("recovery"):
            require(host == recovery["host"] and dataset_id == recovery["dataset"]
                    and initial == recovery["initial_state_sha256"], "Recovery initialization/scope changed")
        job_id = f"{dataset_id}/seed_{seed}"
        dest = output / job_id; dest.mkdir(parents=True, exist_ok=True)
        require(initial == qualification['initialization'][dataset_id][str(seed)], 'Frozen initialization changed')
        for name, value in (('input_receipt.json', metadata), ('initialization.json', initial)):
            path = dest/name
            if path.exists():
                require(read(path) == value, 'Shared input or initialization changed')
            else:
                common.write_json(path, value, exclusive=True)
        quantity = {"boundaries": data["quantity_boundaries_all_train_rows"],
                    "strata": [{"label": f"frozen_quantity_bin_{i}"} for i in range(5)]}
        arms, exposures = {}, {}
        for arm in (requested_arm,):
            budget(); validate_execution_contract(contract)
            require(common.runtime_check(contract, host) == runtime, "Runtime drift")
            arm_budget = ArmBudget(contract, host, dataset_id, arm, budget, seed=seed)
            args = training_args(contract, data, dest, arm)
            args.seeds = str(seed)
            run = dest / "runs" / arm / VARIANT / f"seed_{seed}"
            require(not run.exists(), "Fresh arm required; no implicit resume or retry")
            def status(epoch, records):
                common.write_json(output / "status.json", {"status":"training", "host":host,
                    "dataset":dataset_id, "backbone":arm, "seed":seed, "epoch":epoch,
                    "global_steps":sum(r["batches"] for r in records["train"]),
                    "deadline_unix":permit["deadline_unix"]})
                common.write_json(run / "exposure.json", records)
            with shared.audited_training(training, data, arm_budget, status) as exposure:
                summary, _, _ = training.train_one(args=args, frame=frame, quantity_contract=quantity,
                    interface_meta=interface, backbone=arm, quantity_variant=VARIANT, seed=seed)
            history = read(run / "history.json")["history"]
            steps = audit_arm(history, summary, exposure, data, contract["training"])
            require(summary["initial_state_sha256"] == initial[arm], "Initial model state mismatch")
            identity = training._resume_identity(args=args, backbone=arm, quantity_variant=VARIANT, seed=seed,
                monitor="validation_raw_quantity_rmse", interface_meta=interface, quantity_contract=quantity)
            endpoints = {}
            for label, path in (("selected",Path(summary["checkpoint_path"])), ("last",run/"last_epoch_state.pt")):
                epoch = summary["best_epoch"] if label == "selected" else len(history)
                endpoints[label] = replay_checkpoint(path, data, frame, arm_budget, expected_arm=arm,
                    expected_identity=identity, expected_initial=initial[arm], expected_epoch=epoch, expected_seed=seed)
                audit_replay_accounting(endpoints[label])
                require(all(math.isclose(endpoints[label][m], history[epoch-1][f],rel_tol=1e-10,abs_tol=1e-8)
                    for m,f in (("qty_rmse","val_qty_rmse"),("qty_mae","val_qty_mae"),("time_nll","val_time_nll"))),
                    "Checkpoint evaluation differs from recorded history")
            first40 = [r["val_qty_rmse"] for r in history[:40]]
            arms[arm] = {**endpoints, "last30":shared.last30_summary(history),
                "first40":{"mean":statistics.mean(first40), "sd":statistics.stdev(first40), "count":40},
                "best_epoch":summary["best_epoch"], "completed_epochs":len(history), "global_steps":steps,
                "stopped_early":summary["stopped_early"], "initial_state_sha256":initial[arm],
                "parameter_count":summary["parameter_count"], "elapsed_seconds":summary["elapsed_seconds"]}
            exposures[arm] = deepcopy(exposure)
            common.write_json(run/"exposure.json",exposure)
            common.write_json(run/"endpoint_replays.json",arms[arm],exclusive=True)
            require(check_baseline_prefix(contract,host,dataset_id,seed,exposure),'Baseline prefix mismatch')
            arm_budget.finish(); gc.collect(); torch.cuda.empty_cache()
        return arms[requested_arm]

class ProgressBudget:
    """A pulse is emitted by genuine batch/phase progress, never by a timer thread."""
    def __init__(self, outer, path, arm, contract, *, clock=time.time, mono=time.monotonic):
        self.outer, self.path, self.arm, self.contract = outer, path, arm, contract
        self.clock, self.mono, self.last = clock, mono, -math.inf
        self.calls = 0

    def __call__(self):
        self.outer(); self.calls += 1
        if self.mono() - self.last >= self.contract['execution_isolation']['progress_write_interval_seconds']:
            common.write_json(self.path, {'at_unix':self.clock(), 'monotonic':self.mono(),
                'pid':os.getpid(), 'arm':self.arm, 'calls':self.calls,
                'contract_sha256':common.sha_json(self.contract)})
            self.last = self.mono()
            faulthandler.cancel_dump_traceback_later()
            faulthandler.dump_traceback_later(
                self.contract['execution_isolation']['stack_dump_after_stall_seconds'], repeat=True)

def check_progress(record, *, child_pid, arm, digest, started, now, timeout):
    require(record.get('pid') == child_pid and record.get('arm') == arm
            and record.get('contract_sha256') == digest, 'Worker progress identity changed')
    pulse = record.get('monotonic')
    require(type(pulse) in (int, float) and math.isfinite(pulse)
            and started <= pulse <= now, 'Invalid progress clock')
    require(now - pulse < timeout, 'Owned worker made no progress; no automatic retry')

def worker(a):
    c,approval,permit=read(a.contract),read(a.approval),read(a.permit)
    host=a.host; spec=verify_authorization(c,approval,permit,host,training=a.stage=='_fit')
    shared.verify_host_paths(spec,__file__)
    require(os.getppid()==a.owner_pid and os.getsid(0)==os.getpid(),'Owned isolated worker required')
    expected={'contract_sha256':common.sha_json(c),'approval_sha256':common.sha_json(approval),
              'permit_sha256':common.sha_json(permit),'owner_pid':a.owner_pid,'job':a.job,'stage':a.stage}
    with os.fdopen(a.authority_fd,'rb') as pipe:authority=json.loads(pipe.read(4096))
    require(authority==expected,'Inherited authority mismatch')
    apply_process_environment(c,host);verify_start_memory(c,host)
    import resource
    resource.setrlimit(resource.RLIMIT_FSIZE,(c['limits']['max_file_bytes'],)*2)
    faulthandler.enable();faulthandler.register(signal.SIGUSR1,all_threads=True)
    root=Path(spec['root']);dest=root/('qualification' if a.stage=='_qualify' else 'run')
    outer=Budget(c,permit,host,a.owner_pid,dest)
    budget=ProgressBudget(outer,dest/'progress'/f'{a.job}.json',a.job,c)
    budget()
    runtime=common.runtime_check(c,host)
    if a.stage=='_qualify':
        result=qualify(c,device='cuda:0',batch_size=128,lengths=(64,256),budget=budget,seed=42)
        result.update(host=host,runtime=runtime,contract_sha256=common.sha_json(c),
            approval_sha256=common.sha_json(approval),source_files_sha256=c['source']['files_sha256'],
            started_at_unix=permit['started_at_unix'],deadline_unix=permit['deadline_unix'],
            process_environment=c['process_environment'][host],library_sha256=c['library_sha256'][host])
        result['initialization']={d['dataset_id']:{str(seed):initialization(d,seed) for seed in (42,52,62)}
            for d in c['datasets'] if d['dataset_id'] in spec['assigned_datasets']}
        for row in c['reuse']:
            if row['host']==host:
                require(result['initialization'][row['dataset']][str(row['seed'])][row['arm']]==row['initial_state_sha256'],
                    'Native reused baseline initialization changed')
        validate_qualification_measurements(c,result)
        common.write_json(dest/'receipt.json',result,exclusive=True)
    else:
        jobs=[j for j in scheduled_jobs(c,host) if j['id']==a.job];require(len(jobs)==1,'Unknown fit')
        job=jobs[0];q=read(root/'qualification/receipt.json')
        require(runtime==q['runtime'],'Qualified runtime changed')
        common.write_json(dest/'status.json',{'status':'training','phase':'initializing','host':host,
            'dataset':job['dataset'],'seed':job['seed'],'backbone':job['arm'],'epoch':0,'global_steps':0,
            'deadline_unix':permit['deadline_unix']})
        result=run_arm(c,permit,host,q,budget,job)
        common.write_json(dest/'arm_receipts'/f'{a.job}.json',{'status':'complete','job':job,'pid':os.getpid(),
            'contract_sha256':common.sha_json(c),'result':result},exclusive=True)
    budget();faulthandler.cancel_dump_traceback_later()


def owned_call(c,approval,permit,host,stage,job):
    spec=c['hosts'][host];root=Path(spec['root']);dest=root/('qualification' if stage=='_qualify' else 'run')
    authority={'contract_sha256':common.sha_json(c),'approval_sha256':common.sha_json(approval),
        'permit_sha256':common.sha_json(permit),'owner_pid':os.getpid(),'job':job,'stage':stage}
    rd,wr=os.pipe();child=None;started=time.monotonic();last_scan=-math.inf
    common.write_json(dest/'workers'/f'{job}.json',{'authority':authority},exclusive=True)
    try:
        with (dest/'workers'/f'{job}.log').open('xb') as stream:
            child=subprocess.Popen([spec['python'],str(Path(__file__).resolve()),stage,
                '--host',host,'--contract',str(root/'frozen_execution/execution_contract.json'),
                '--approval',str(root/'approval.json'),'--permit',str(root/'start_permit.json'),
                '--job',job,'--owner-pid',str(os.getpid()),'--authority-fd',str(rd)],
                cwd=spec['source_root'],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(rd,))
        os.close(rd);rd=None
        with os.fdopen(wr,'wb') as pipe:wr=None;pipe.write(json.dumps(authority).encode())
        common.write_json(dest/'workers'/f'{job}.json',{'authority':authority,'pid':child.pid,
            'started_at_unix':time.time(),'started_monotonic':started})
        deadline=shared.fixed_start(permit,total_seconds=c['limits']['total_wall_seconds'])
        while child.poll() is None:
            now=time.monotonic()
            require(time.time()<permit['deadline_unix'] and now<deadline,'Common deadline reached')
            if stage=='_qualify':require(now-started<c['limits']['qualification_seconds_per_host'],'Qualification timeout')
            else:
                active=dest/'active_arm.json'
                if active.exists():
                    record=read(active)
                    require(record['contract_sha256']==common.sha_json(c) and record['host']==host,'Active arm identity')
                    if record['status']=='active':
                        require(now<record['deadline_monotonic'] and time.time()<record['deadline_unix'],'Per-arm deadline')
            progress=dest/'progress'/f'{job}.json'
            if progress.exists():check_progress(read(progress),child_pid=child.pid,arm=job,digest=common.sha_json(c),
                started=started,now=time.monotonic(),timeout=c['execution_isolation']['no_progress_timeout_seconds'])
            else:require(now-started<c['execution_isolation']['no_progress_timeout_seconds'],'Initialization stalled')
            if now-last_scan>=10:shared.check_storage(root,storage_limits(c));last_scan=now
            time.sleep(1)
        require(child.returncode==0,f'Owned worker {job} exited {child.returncode}; no retry')
    except BaseException:
        if child is not None:shared.kill_owned_process_group(child)
        raise
    finally:
        for fd in (rd,wr):
            if fd is not None:os.close(fd)


def dispatch(a):
    c,approval,permit=read(a.contract),read(a.approval),read(a.permit);host=a.host
    spec=verify_authorization(c,approval,permit,host);shared.verify_host_paths(spec,__file__)
    root=Path(spec['root']);verify_baselines(c,host);verify_start_memory(c,host);apply_process_environment(c,host)
    require(not (root/'qualification').exists() and not (root/'run').exists(),'Previous attempt exists; no retry')
    common.write_json(root/'dispatch_claim.json',{'host':host,'pid':os.getpid(),'claimed_at_unix':time.time(),
        'contract_sha256':common.sha_json(c)},exclusive=True)
    old_signal=signal.signal(signal.SIGTERM,shared._supervisor_terminated)
    try:
        (root/'qualification').mkdir(exist_ok=False)
        common.write_json(root/'dispatch_status.json',{'status':'qualifying','host':host})
        owned_call(c,approval,permit,host,'_qualify','qualification')
        verify_authorization(c,approval,permit,host,training=True)
        output=root/'run';output.mkdir(exist_ok=False)
        common.write_json(output/'wrapper_manifest.json',{'contract_sha256':common.sha_json(c),
            'isolation':'fresh_exec_process_per_fit','jobs':scheduled_jobs(c,host),
            'deadline_unix':permit['deadline_unix'],'automatic_retry':False},exclusive=True)
        common.write_json(root/'dispatch_status.json',{'status':'training','host':host})
        results=[]
        for job in scheduled_jobs(c,host):
            verify_authorization(c,approval,permit,host,training=True)
            verify_start_memory(c,host)
            owned_call(c,approval,permit,host,'_fit',job['id'])
            result=read(output/'arm_receipts'/f"{job['id']}.json")
            require(result['status']=='complete' and result['job']==job and result['contract_sha256']==common.sha_json(c),'Fit receipt mismatch')
            results.append(result)
        verify_baselines(c,host)
        common.write_json(output/'partition_summary.json',{'status':'complete','host':host,
            'new_fits':len(results),'endpoint_replays':2*len(results),'jobs':results,
            'contract_sha256':common.sha_json(c),'evaluation_scope':'validation_only',
            'held_out_test_evaluated':False,'performance_acceptance':'pending_combined_analysis'},exclusive=True)
        common.write_json(output/'status.json',{'status':'complete','host':host,'completed_at_unix':time.time()})
        common.write_json(root/'dispatch_status.json',{'status':'complete','host':host,'completed_at_unix':time.time()})
    except BaseException as error:
        common.write_json(root/'dispatch_status.json',{'status':'failed','host':host,'error':str(error),'automatic_retry':False})
        if (root/'run').exists():common.write_json(root/'run/status.json',{'status':'stopped','host':host,'error':str(error),'automatic_retry':False})
        raise
    finally:signal.signal(signal.SIGTERM,old_signal)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('dispatch','_qualify','_fit'))
    for k in ('contract','approval','permit','host'):p.add_argument('--'+k,required=True)
    p.add_argument('--job');p.add_argument('--owner-pid',type=int);p.add_argument('--authority-fd',type=int)
    a=p.parse_args()
    if a.stage=='dispatch':dispatch(a)
    else:worker(a)


if __name__=='__main__':main()
