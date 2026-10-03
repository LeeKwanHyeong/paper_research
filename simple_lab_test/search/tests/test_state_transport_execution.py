"""Authorization/deadline gates use synthetic receipts and never CUDA/SSH."""
from __future__ import annotations

import copy
import json
import signal
import subprocess
from types import SimpleNamespace

import pytest

from paper.scripts import run_state_transport_execution as runner


@pytest.fixture
def authority(monkeypatch, tmp_path):
    # Scientific contract validation is covered by the proposal tests. Here the
    # authority layer uses a minimal synthetic contract with no production files.
    monkeypatch.setattr(runner.proposal, "validate_contract", lambda *a, **k: None)
    c = {"limits": {"total_wall_seconds":86400,"qualification_seconds_per_host":900,
            "per_host_output_bytes":4*1024**3,"min_free_bytes":5*1024**3},
        "cost_gates": {"median_cuda_step_ratio_max":2.5,"peak_cuda_allocated_ratio_max":2.5,
            "device_memory_fraction_max":.8,"parameter_ratio_max":1.05},
        "source": {"files_sha256":"synthetic_source"}, "hosts": {}}
    for alias in ("5080","5090"):
        c["hosts"][alias] = {"gpu_uuid":"synthetic_"+alias,"runtime_expected":{"torch":"synthetic"},
            "root":str(tmp_path/alias),"source_root":str(tmp_path/alias/"source"),
            "python":"synthetic_python"}
    a = {"approved":True,"contract_sha256":runner.common.sha_json(c),"hosts":["5080","5090"],
         "user_instruction":"synthetic approval fixture only"}
    monkeypatch.setattr(runner.time,"time",lambda:1000.)
    p = runner.make_start_permit(c,a,started_at_unix=1000.)
    return c,a,p


def receipt(contract, approval, permit, host):
    costs=[]
    for length in (64,256):
        measurements={arm:{"step_seconds":[1.,1.1,1.2,1.3,1.4],"median_step_seconds":1.2,
            "peak_allocated_bytes":100,"parameters":1000 if arm==runner.ARMS[0] else 1040} for arm in runner.ARMS}
        checks={arm+":"+key:True for arm in runner.ARMS[1:] for key in ("step","memory","device","parameters")}
        costs.append({"length":length,"batch_size":128,"measurements":measurements,"checks":checks})
    return {"status":"passed","host":host,"device":"cuda:0",
        "contract_sha256":runner.common.sha_json(contract),"approval_sha256":runner.common.sha_json(approval),
        "source_files_sha256":contract["source"]["files_sha256"],"started_at_unix":permit["started_at_unix"],
        "deadline_unix":permit["deadline_unix"],"synthetic_optimizer_updates":48,
        "runtime":{"torch":"synthetic","gpu":{"uuid":"synthetic_"+host,"total_memory_bytes":1000}},
        "checks":dict.fromkeys(runner.QUALIFICATION_CHECKS,True),
        "real_data_loaded":False,"held_out_evaluated":False,"costs":costs}


def qualified(authority):
    c,a,p=authority
    receipts={h:receipt(c,a,p,h) for h in c["hosts"]}
    return runner.make_training_permit(c,a,p,receipts)


def test_two_receipts_preserve_common_origin(authority):
    c,a,p=authority
    train=qualified(authority)
    assert train["started_at_unix"]==p["started_at_unix"]
    assert train["deadline_unix"]==p["deadline_unix"]
    runner.verify_authorization(c,a,train,"5090",training=True)


@pytest.mark.parametrize("change",["unapproved","empty_instruction","foreign_contract","missing_host"])
def test_approval_rejection_precedes_host_gpu_and_spawn(monkeypatch, authority, change):
    c,a,p=authority
    if change=="unapproved":a["approved"]=False
    elif change=="empty_instruction":a["user_instruction"]=" "
    elif change=="foreign_contract":a["contract_sha256"]="foreign"
    else:a["hosts"]=["5080"]
    monkeypatch.setattr(runner,"read",lambda path:{"c":c,"a":a,"p":p}[path])
    def forbidden(*a,**k):pytest.fail("Invalid approval reached runtime/GPU/spawn")
    monkeypatch.setattr(runner,"source_and_host",forbidden)
    monkeypatch.setattr(runner.common,"gpu_pids",forbidden)
    monkeypatch.setattr(runner.subprocess,"Popen",forbidden)
    with pytest.raises(ValueError,match="Explicit approval"):
        runner.supervisor("qualify","c","a","p","5080")


@pytest.mark.parametrize("change",["future","expired","extended","approval_sha","contract_sha"])
def test_common_deadline_and_permit_identity_fail_closed(authority,change):
    c,a,p=authority
    if change=="future":p["started_at_unix"]+=1;p["deadline_unix"]+=1
    elif change=="expired":p["started_at_unix"]-=86400;p["deadline_unix"]-=86400
    elif change=="extended":p["deadline_unix"]+=1
    elif change=="approval_sha":p["approval_sha256"]="foreign"
    else:p["contract_sha256"]="foreign"
    with pytest.raises(ValueError):runner.verify_authorization(c,a,p,"5080")


def test_training_cannot_skip_either_native_qualification(authority):
    c,a,p=authority
    with pytest.raises(ValueError,match="Both native"):
        runner.verify_authorization(c,a,p,"5080",training=True)
    q=qualified(authority);q["qualifications"].pop("5090")
    with pytest.raises(ValueError,match="Both native"):
        runner.verify_authorization(c,a,q,"5080",training=True)


@pytest.mark.parametrize("change",["cpu","updates","runtime","source","origin","empty_checks","wrong_checks","held_out"])
def test_native_receipt_identity_and_scope(authority,change):
    c,a,p=authority;q=qualified(authority)
    r=q["qualifications"]["5090"]["receipt"]
    if change=="cpu":r["device"]="cpu"
    elif change=="updates":r["synthetic_optimizer_updates"]=47
    elif change=="runtime":r["runtime"]["torch"]="other"
    elif change=="source":r["source_files_sha256"]="other"
    elif change=="origin":r["started_at_unix"]+=1
    elif change=="empty_checks":r["checks"]={}
    elif change=="wrong_checks":r["checks"]={"anything":True}
    else:r["held_out_evaluated"]=True
    q["qualifications"]["5090"]["receipt_sha256"]=runner.common.sha_json(r)
    with pytest.raises(ValueError):runner.verify_authorization(c,a,q,"5080",training=True)


@pytest.mark.parametrize("change",["false_fast_median","slow_steps","memory","parameters","empty_gates","duplicate_length","missing_arm","four_samples","zero_step","wrong_batch","capacity"])
def test_native_cost_raw_values_recomputed(authority,change):
    c,a,p=authority;q=qualified(authority)
    r=q["qualifications"]["5080"]["receipt"];row=r["costs"][0]
    m=row["measurements"][runner.ARMS[2]]
    if change=="false_fast_median":m["median_step_seconds"]=.1
    elif change=="slow_steps":m["step_seconds"]=[10.]*5;m["median_step_seconds"]=10.
    elif change=="memory":m["peak_allocated_bytes"]=801
    elif change=="parameters":m["parameters"]=1051
    elif change=="empty_gates":row["checks"]={}
    elif change=="duplicate_length":r["costs"][1]["length"]=64
    elif change=="missing_arm":row["measurements"].pop(runner.ARMS[1])
    elif change=="four_samples":m["step_seconds"].pop()
    elif change=="zero_step":m["step_seconds"][0]=0.
    elif change=="wrong_batch":row["batch_size"]=127
    else:m["parameters"]=1041
    q["qualifications"]["5080"]["receipt_sha256"]=runner.common.sha_json(r)
    with pytest.raises(ValueError):runner.verify_authorization(c,a,q,"5090",training=True)


def test_cpu_cannot_qualify_native_and_other_host_cannot_substitute(authority):
    c,a,p=authority;q=qualified(authority)
    q["qualifications"]["5090"]=copy.deepcopy(q["qualifications"]["5080"])
    with pytest.raises(ValueError,match="identity"):
        runner.verify_authorization(c,a,q,"5090",training=True)


def test_storage_or_supervisor_error_stops_only_owned_group(monkeypatch,authority,tmp_path):
    c,a,p=authority
    monkeypatch.setattr(runner,"read",lambda path:{"c":c,"a":a,"p":p}[path])
    monkeypatch.setattr(runner,"source_and_host",lambda *a:None)
    monkeypatch.setattr(runner.common,"apply_environment",lambda *a:None)
    monkeypatch.setattr(runner.common,"gpu_pids",lambda *a:set())
    child=SimpleNamespace(pid=123456,poll=lambda:None)
    monkeypatch.setattr(runner.subprocess,"Popen",lambda *a,**k:child)
    # Do not write to a pipe with no child reader in this mocked process test.
    class FakeWriter(__import__("io").BytesIO):
        def __init__(self,fd):super().__init__();self.fd=fd
        def close(self):
            if not self.closed:runner.os.close(self.fd)
            super().close()
    monkeypatch.setattr(runner.os,"fdopen",lambda fd,*a,**k:FakeWriter(fd))
    monkeypatch.setattr(runner.shared,"check_storage",lambda *a:(_ for _ in ()).throw(ValueError("synthetic storage cap")))
    killed=[]
    monkeypatch.setattr(runner.shared,"kill_owned_process_group",lambda c:killed.append(c.pid))
    with pytest.raises(ValueError,match="storage cap"):
        runner.supervisor("qualify","c","a","p","5080")
    assert killed==[123456]
    assert json.loads((tmp_path/"5080"/"qualification"/"status.json").read_text())["automatic_retry"] is False


def test_busy_gpu_rejected_without_spawning_or_stopping(monkeypatch,authority,tmp_path):
    c,a,p=authority
    monkeypatch.setattr(runner,"read",lambda path:{"c":c,"a":a,"p":p}[path])
    monkeypatch.setattr(runner,"source_and_host",lambda *a:None)
    monkeypatch.setattr(runner.common,"gpu_pids",lambda *a:{999})
    def forbidden(*a,**k):pytest.fail("Foreign process touched")
    monkeypatch.setattr(runner.subprocess,"Popen",forbidden)
    monkeypatch.setattr(runner.shared,"kill_owned_process_group",forbidden)
    with pytest.raises(ValueError,match="GPU busy"):
        runner.supervisor("qualify","c","a","p","5080")
    assert not (tmp_path/"5080"/"qualification").exists()


def test_termination_signals_only_child_owned_process_group(monkeypatch):
    child=SimpleNamespace(pid=123,poll=lambda:None)
    calls=[];waits=[]
    def wait(timeout):
        waits.append(timeout)
        if len(waits)==1:raise subprocess.TimeoutExpired("synthetic",timeout)
    child.wait=wait
    monkeypatch.setattr(runner.os,"killpg",lambda pid,sig:calls.append((pid,sig)))
    runner.shared.kill_owned_process_group(child,grace_seconds=1)
    assert calls==[(123,signal.SIGTERM),(123,signal.SIGKILL)]


def test_local_qualification_must_equal_embedded_receipt(authority,tmp_path):
    c,a,p=authority;q=qualified(authority)
    path=tmp_path/"5080"/"qualification"/"receipt.json";path.parent.mkdir(parents=True)
    path.write_text(json.dumps(q["qualifications"]["5080"]["receipt"]))
    assert runner.verify_local_qualification(c,q,"5080")["status"]=="passed"
    path.write_text('{}')
    with pytest.raises(ValueError,match="Local/embedded"):
        runner.verify_local_qualification(c,q,"5080")


def test_direct_native_qualification_rejects_unapproved_before_runtime(monkeypatch,authority):
    c,a,p=authority;a["approved"]=False
    monkeypatch.setattr(runner,"source_and_host",lambda *a:pytest.fail("unapproved reached runtime"))
    with pytest.raises(ValueError,match="Explicit approval"):
        runner.native_qualification(c,a,p,"5080",lambda:None)


def test_worker_budget_foreign_process_and_both_clocks(monkeypatch,authority,tmp_path):
    c,a,p=authority
    monkeypatch.setattr(runner.os,"getppid",lambda:42)
    monkeypatch.setattr(runner.time,"monotonic",lambda:100.)
    monkeypatch.setattr(runner.shared,"check_storage",lambda *a:0)
    monkeypatch.setattr(runner.common,"gpu_pids",lambda *a:{runner.os.getpid()})
    budget=runner.Budget(c,p,"5080",42,tmp_path);budget()
    monkeypatch.setattr(runner.common,"gpu_pids",lambda *a:{runner.os.getpid(),555})
    monkeypatch.setattr(runner.time,"monotonic",lambda:111.)
    with pytest.raises(ValueError,match="Other GPU"):
        budget()
    monkeypatch.setattr(runner.common,"gpu_pids",lambda *a:{runner.os.getpid()})
    monkeypatch.setattr(runner.time,"monotonic",lambda:86501.)
    with pytest.raises(ValueError,match="deadline"):
        budget()
    monkeypatch.setattr(runner.time,"monotonic",lambda:112.)
    monkeypatch.setattr(runner.time,"time",lambda:p["deadline_unix"])
    with pytest.raises(ValueError,match="deadline"):
        budget()


def test_orphaned_worker_rejected(monkeypatch,authority,tmp_path):
    c,a,p=authority
    monkeypatch.setattr(runner.time,"monotonic",lambda:100.)
    budget=runner.Budget(c,p,"5080",42,tmp_path)
    monkeypatch.setattr(runner.os,"getppid",lambda:1)
    with pytest.raises(ValueError,match="supervisor exited"):
        budget()

# The integration cases below use actual models, objectives, optimizer and
# checkpoint code on tiny synthetic CPU tensors. No real dataset is loaded.
@pytest.fixture
def cpu_training():
    import torch
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def synthetic_args(monkeypatch, tmp_path, arm, name="run"):
    from simple_lab_test.search.tests import test_observed_time_joint_training as h
    args = h.args_for(monkeypatch, tmp_path, arm, name)
    args.model_role = "observed_time_state_transport_v1"
    args.dataset_contract = "intermittent_frozen_5000"
    return args


def synthetic_interface():
    from paper.scripts.count_aware_tpp_backbone import observed_time
    from simple_lab_test.search.tests import test_observed_time_joint_training as h
    meta = h.interface()
    meta["time_head"]["observation_likelihood"] = observed_time.observation_contract("intermittent_frozen_5000")
    meta.update(source_files_sha256="synthetic_source_A", execution_contract_sha256="synthetic_contract_A",
        backbone_design={"design_sha256":"synthetic_design_A", "architecture":{"rank":8,"scan_mode":"parallel"},
            "clock_by_arm":{runner.ARMS[0]:None,runner.ARMS[1]:"event",runner.ARMS[2]:"elapsed"},
            "train_time_scale":3.0})
    return meta


def test_actual_training_same_batches_checkpoint_cache_and_fresh_B(cpu_training, monkeypatch, tmp_path):
    from paper.scripts.count_aware_tpp_backbone import training
    from simple_lab_test.search.tests import test_observed_time_joint_training as h
    traces_by_arm = []
    for arm in runner.ARMS:
        traces = {"train":[],"validation":[]}
        h.install_loader(monkeypatch,traces)
        args = synthetic_args(monkeypatch,tmp_path,arm)
        summary,_,_ = h.run(args,arm,meta=synthetic_interface())
        payload = h.torch_load_checkpoint(h.directory(args,arm)/"last_epoch_state.pt",map_location="cpu")
        earliest = training.earliest_strict_minimum(payload["history"],metric_key="val_qty_rmse")
        assert summary["completed_epochs"]==3 and not summary["stopped_early"]
        assert summary["best_epoch"]==earliest["epoch"]
        assert summary["checkpoint_monitor"]=="validation_raw_quantity_rmse"
        assert all(x["train_all_finite"] for x in payload["history"])
        assert payload["resume_identity"]["interface_meta"]["backbone_design"]==synthetic_interface()["backbone_design"]
        saved=copy.deepcopy(traces)
        cached,_,_=h.run(args,arm,meta=synthetic_interface())
        assert cached["checkpoint_state_sha256"]==summary["checkpoint_state_sha256"]
        assert traces==saved
        traces_by_arm.append(traces)
    assert all(x==traces_by_arm[0] for x in traces_by_arm)


@pytest.mark.parametrize("arm",runner.ARMS[1:])
def test_actual_interrupted_resume_exact_model_optimizer_rng_selector(cpu_training,monkeypatch,tmp_path,arm):
    from models.TPPs.CountAwareFactory import validate_checkpoint_route
    from paper.scripts.count_aware_tpp_backbone import training
    from simple_lab_test.search.tests import test_observed_time_joint_training as h
    h.install_loader(monkeypatch,{"train":[],"validation":[]})
    full_args=synthetic_args(monkeypatch,tmp_path,arm,"full")
    expected,_,_=h.run(full_args,arm,meta=synthetic_interface())
    args=synthetic_args(monkeypatch,tmp_path,arm,"resumed")
    actual_save=training.atomic_torch_save
    def interrupt(payload,path):
        actual_save(payload,path)
        if path.name=="last_epoch_state.pt" and payload["epoch"]==1:
            raise RuntimeError("synthetic interruption")
    monkeypatch.setattr(training,"atomic_torch_save",interrupt)
    with pytest.raises(RuntimeError,match="synthetic interruption"):
        h.run(args,arm,meta=synthetic_interface())
    monkeypatch.setattr(training,"atomic_torch_save",actual_save)
    actual,_,_=h.run(args,arm,meta=synthetic_interface())
    assert actual["checkpoint_state_sha256"]==expected["checkpoint_state_sha256"]
    a=h.torch_load_checkpoint(h.directory(full_args,arm)/"last_epoch_state.pt",map_location="cpu")
    b=h.torch_load_checkpoint(h.directory(args,arm)/"last_epoch_state.pt",map_location="cpu")
    for key in ("model_state_dict","best_state_dict","optimizer_state_dict","rng_state","train_loader_generator_state","history"):
        h.same(a[key],b[key])
    for foreign in set(runner.ARMS)-{arm}:
        with pytest.raises(ValueError):validate_checkpoint_route(b,foreign)


@pytest.mark.parametrize("arm",runner.ARMS)
def test_actual_strict_earliest_tie(cpu_training,monkeypatch,tmp_path,arm):
    from paper.scripts.count_aware_tpp_backbone import training
    from simple_lab_test.search.tests import test_observed_time_joint_training as h
    h.install_loader(monkeypatch,{"train":[],"validation":[]})
    original=training.evaluate
    def tied(**kwargs):return {**original(**kwargs),"qty_rmse":42.0}
    monkeypatch.setattr(training,"evaluate",tied)
    summary,_,_=h.run(synthetic_args(monkeypatch,tmp_path,arm),arm,meta=synthetic_interface())
    assert summary["best_epoch"]==1


@pytest.mark.parametrize("field",["source","design","clock","scale","architecture"])
def test_actual_resume_rejects_changed_identity_before_training_update(cpu_training,monkeypatch,tmp_path,field):
    from paper.scripts.count_aware_tpp_backbone import training
    from simple_lab_test.search.tests import test_observed_time_joint_training as h
    arm=runner.ARMS[2]
    h.install_loader(monkeypatch,{"train":[],"validation":[]})
    args=synthetic_args(monkeypatch,tmp_path,arm)
    original=training.atomic_torch_save
    def interrupt(payload,path):
        original(payload,path)
        if path.name=="last_epoch_state.pt" and payload["epoch"]==1:raise RuntimeError("stop")
    monkeypatch.setattr(training,"atomic_torch_save",interrupt)
    with pytest.raises(RuntimeError,match="stop"):h.run(args,arm,meta=synthetic_interface())
    monkeypatch.setattr(training,"atomic_torch_save",original)
    meta=synthetic_interface()
    if field=="source":meta["source_files_sha256"]="different"
    elif field=="design":meta["backbone_design"]["design_sha256"]="different"
    elif field=="clock":meta["backbone_design"]["clock_by_arm"][arm]="event"
    elif field=="scale":meta["backbone_design"]["train_time_scale"]=1.0
    else:meta["backbone_design"]["architecture"]["rank"]=4
    def forbidden(*a,**k):pytest.fail("Changed identity reached a training update")
    monkeypatch.setattr(training,"train_epoch_with_telemetry",forbidden)
    with pytest.raises(ValueError):h.run(args,arm,meta=meta)


def test_separate_runner_keeps_prior_replay_factory_unchanged():
    from paper.scripts import run_pair_message_execution as prior
    assert prior.ARMS==("titantpp","titantpp_pair_message_post_pool","titantpp_pair_message_pre_pool")
    assert "_build_model =" not in __import__("inspect").getsource(runner.replay_checkpoint)


def test_frozen_scale_initialization_matches_all_common_B_tensors(cpu_training):
    data=runner.read(runner.ROOT/runner.proposal.PRIOR)["datasets"]
    for entry in data:
        hashes=runner.initialization(entry)
        assert set(hashes)==set(runner.ARMS)
        assert len(set(hashes.values()))==3  # Clock identity is intentionally distinct.


@pytest.mark.parametrize("arm",runner.ARMS)
def test_owned_endpoint_replay_preserves_selected_metrics_and_split(cpu_training,monkeypatch,tmp_path,arm):
    import math
    import torch
    from torch.utils.data import DataLoader
    from paper.scripts import run_taxi_quantity_interface_ablation as loaders
    from simple_lab_test.search.tests import test_observed_time_joint_training as h
    h.install_loader(monkeypatch,{"train":[],"validation":[]})
    args=synthetic_args(monkeypatch,tmp_path,arm)
    meta=synthetic_interface()
    summary,_,_=h.run(args,arm,meta=meta)
    def loader(_frame,*,target_split,**kwargs):
        assert target_split=="validation"
        return DataLoader(h.dataset("validation"),batch_size=4,shuffle=False)
    monkeypatch.setattr(loaders,"make_loader",loader)
    data={"model":{"hidden_dim":16,"quantity_variant":runner.VARIANT,
        "time_head_mode":meta["time_head"]["mode"],"time_scale":3.0,"time_initial_location":.1,
        "time_initial_scale":.7,"time_sigma_floor":args.time_sigma_floor,
        "time_observation_contract":meta["time_head"]["observation_likelihood"]},
        "statistics":{"train_log_mean":1.5,"train_log_std":1.0},
        "loader":{"max_seq_len":8,"batch_size":4,"lookback_weeks":520},
        "quantity_boundaries_all_train_rows":[2.,4.,8.,16.],"history_boundaries":[2,4],
        "inherited_data_identity":{"populations":{"validation":{"target_count":7}}}}
    checkpoint=h.torch_load_checkpoint(summary["checkpoint_path"],map_location="cpu")
    result=runner.replay_checkpoint(summary["checkpoint_path"],data,h.grid_frame(),lambda:None,
        device="cpu",expected_arm=arm,expected_identity=checkpoint["resume_identity"],
        expected_initial=summary["initial_state_sha256"],expected_epoch=summary["best_epoch"])
    history=runner.read(h.directory(args,arm)/"history.json")["history"]
    selected=history[summary["best_epoch"]-1]
    assert result["count"]==7 and result["evaluation_scope"]=="validation_only"
    assert result["held_out_test_evaluated"] is False and result["time_metric"]==runner.TIME_METRIC
    for metric,key in (("qty_rmse","val_qty_rmse"),("qty_mae","val_qty_mae"),("time_nll","val_time_nll")):
        assert math.isclose(result[metric],selected[key],rel_tol=1e-10,abs_tol=1e-8)
    assert sum(cell["count"] for cell in result["quantity_cells"])==7
    assert sum(cell["count"] for cell in result["history_cells"])==7
    foreign=copy.deepcopy(checkpoint["resume_identity"])
    foreign["interface_meta"]["source_files_sha256"]="wrong"
    with pytest.raises(ValueError,match="identity"):
        runner.replay_checkpoint(summary["checkpoint_path"],data,h.grid_frame(),lambda:None,
            device="cpu",expected_arm=arm,expected_identity=foreign,
            expected_initial=summary["initial_state_sha256"],expected_epoch=summary["best_epoch"])
