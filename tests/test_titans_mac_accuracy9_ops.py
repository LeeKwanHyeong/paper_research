"""Fabricated operation tests; no SSH, real-data training or held-out inference."""
from copy import deepcopy
import json
from pathlib import Path
import shlex
import sys

import pytest

from paper.scripts import prepare_titans_mac_accuracy9 as p
from paper.scripts import manage_titans_mac_accuracy9 as m
from paper.scripts import titans_mac_accuracy9_runtime as r


@pytest.fixture
def contract(tmp_path):
    c = deepcopy(p.read(p.PARENT / "execution_contract.json"))
    h = deepcopy(c["hosts"]["5090"])
    h.update(gpu_uuid=r.GPU_UUID,absolute_host_deadline_unix=r.DEADLINE,paused_processes=[],
        runtime_expected={"python":"3.12.13","torch":"2.11.0+cu130","cuda":"13.0","cudnn":92000,"threads":4,"interop_threads":1})
    h.update(root=str(tmp_path),source_root=str(tmp_path / "source")); (tmp_path / "source").mkdir()
    c.update(schema=r.SCHEMA,hosts={"5090":h},jobs=p.build_jobs(),maximum_parallel_workers=3,
        historical_models_retrained=False,checkpoint_warmstart=False,held_out_test_authorized=True,
        execution_authorization={"real_data_training":True,"real_data_inference":True,"remote_execution":True,"test_evaluation":True},
        prior_owned_pids=[123456],prior_completion_bindings={})
    for d in c["datasets"]:
        name=f"inputs/full/{d['dataset_id']}.parquet"; c["input_files"][name]="f"*64
        d["full_data_identity"]={"data":{"path":name,"sha256":"f"*64},"populations":{"test":{"target_count":1,
            "target_identity_sha256":"a"*64,"target_quantity_sha256":"b"*64}}}
    return c


def test_exact_mac_matrix_order_and_joint_condition(contract):
    jobs=r.validate_contract(contract,"5090")
    assert len(jobs)==9 and {j["seed"] for j in jobs}=={42,52,62}
    assert jobs[0]["dataset"]=="raf_spare_parts" and jobs[-1]["dataset"]=="intermittent_frozen_5000"
    assert {j["arm"] for j in jobs}==set(r.ARMS) and "cnn" not in r.ARMS


@pytest.mark.parametrize("mutation",("old_seed","host","alpha","cap","runtime","deadline","job_id","duplicate","test_lock"))
def test_contract_rejects_scope_mutations_before_real_data(contract,mutation):
    c=deepcopy(contract)
    if mutation=="old_seed":c["jobs"][0]["seed"]=72
    if mutation=="host":c["jobs"][0]["host"]="5080"
    if mutation=="alpha":c["common_condition"]["time_alpha"]=0.
    if mutation=="cap":c["datasets"][0]["cap_value"]=10.
    if mutation=="runtime":c["hosts"]["5090"]["runtime_expected"]["cudnn"]=91900
    if mutation=="deadline":c["hosts"]["5090"]["absolute_host_deadline_unix"]+=1
    if mutation=="job_id":c["jobs"][0]["id"]="../old"
    if mutation=="duplicate":c["jobs"][-1]=c["jobs"][0]
    if mutation=="test_lock":c["held_out_test_authorized"]=False
    with pytest.raises(ValueError):r.validate_contract(c,"5090")


def test_population_keeps_only_exact_target_keys():
    x={"target_count":2,"target_identity_sha256":"a","target_quantity_sha256":"b","recorded_time_sha256":"c"}
    assert p.population(x)=={k:x[k] for k in ("target_count","target_identity_sha256","target_quantity_sha256")}


def test_bounded_preserved_queue_9_fresh_children_never_exceeds_three(contract,monkeypatch):
    c=contract; root=Path(c["hosts"]["5090"]["root"]); (root/"logs").mkdir()
    children=[];maximum=[0]
    class Child:
        def __init__(self,command,**kwargs):
            self.pid=9000+len(children);self.command=command;self.returncode=None;self.ticks=0
            assert kwargs["start_new_session"] is True
            children.append(self);maximum[0]=max(maximum[0],sum(x.returncode is None for x in children))
        def poll(self):
            if self.returncode is None:
                self.ticks+=1
                if self.ticks>=4:self.returncode=0
            return self.returncode
    def identity(pid):
        child=next((x for x in children if x.pid==pid),None)
        return None if child is None or child.returncode is not None else {"pid":pid,"start_ticks":pid+1,"argv":child.command,
            "cwd":str(root/"source"),"pgid":pid}
    monkeypatch.setattr(r._impl.subprocess,"Popen",Child);monkeypatch.setattr(r._impl,"proc_identity",identity)
    monkeypatch.setattr(r._impl,"gpu_state",lambda *a:( {"compute_pids":[]+[x.pid for x in children if x.returncode is None]},c["hosts"]["5090"]["paused_processes"]))
    monkeypatch.setattr(r._impl,"verify_receipt",lambda c,h,j,s:{"path":"fabricated/"+j["id"],"sha256":"f"*64})
    monkeypatch.setattr(r._impl.time,"sleep",lambda _:None)
    monkeypatch.setattr(r,"_QUEUE_OWNED_CHILDREN",[]);r._impl._EXTRA42_OWNED=r._QUEUE_OWNED_CHILDREN
    result=r.run_stage(c,"5090",root/"execution_contract.json","train",r.time.monotonic())
    assert result["status"]=="complete" and len(result["exits"])==len(result["receipts"])==9 and maximum[0]==3
    assert all(x["actual_os_returncode"]==0 and not x["timeout"] for x in result["exits"].values())


@pytest.mark.parametrize("failure",(None,"train","validation"))
def test_dispatch_stages_gate_then_test_and_never_after_failure(contract,monkeypatch,failure):
    c=contract;root=Path(c["hosts"]["5090"]["root"]);r.write(root/"execution_contract.json",c)
    r.write(root/"qualification.json",{"status":"passed","contract_sha256":r.sha_json(c),"maximum_simultaneous_workers":3,"native_worker_exit0_count":4})
    monkeypatch.chdir(root/"source");monkeypatch.setattr(r,"verify_files",lambda *a:None)
    monkeypatch.setattr(r,"predecessor_gate",lambda *a,**kw:{"status":"passed"})
    monkeypatch.setattr(r,"gpu_state",lambda *a:({"compute_pids":[]},c["hosts"]["5090"]["paused_processes"]))
    monkeypatch.setattr(r,"proc_identity",lambda pid:{"pid":pid,"start_ticks":10,"argv":["newdispatcher"],"cwd":str(root/"source"),"pgid":pid} if pid==r.os.getpid() else None)
    calls=[]
    def stage(c,h,cp,s,started):
        if s=="test":assert calls[-1]=="gate"
        calls.append(s);ids={j["id"]:{"pid":1000+i,"actual_os_returncode":0,"timeout":False} for i,j in enumerate(c["jobs"])}
        return {"status":"failed" if s==failure else "complete","exits":ids,"receipts":{k:{} for k in ids}}
    monkeypatch.setattr(r,"run_stage",stage);monkeypatch.setattr(r,"validation_gate",lambda *a:calls.append("gate"))
    result=r.dispatch(c,"5090",root/"execution_contract.json")
    assert calls==(["train"] if failure=="train" else ["train","validation"] if failure=="validation" else ["train","validation","gate","test"])
    final=r.read(root/"final_receipt.json");assert (result==0)==(failure is None)
    assert final["owned_worker_pids_absent"] is True and final["prior_campaign_resumed"] is False


def test_native_synthetic_actual_three_barrier_and_elapsed_timeout(contract,monkeypatch):
    c=contract;root=Path(c["hosts"]["5090"]["root"]);children=[];waits=[]
    class Child:
        def __init__(self,command,**kw):
            self.pid=8000+len(children);self.command=command;self.returncode=None
            children.append(self);r.write(Path(command[-3]),{"pid":self.pid},exclusive=True)
            value={"status":"passed","real_data_loaded":False,"arm":command[-4],"cuda_memory_peak":{"max_memory_reserved_bytes":100}}
            kw["stdout"].write(json.dumps(value).encode());kw["stdout"].flush()
        def poll(self):return self.returncode
        def wait(self,timeout):
            assert (root/"synthetic.go").exists() and len(children)==3
            overlap=r.read(root/"synthetic.overlap.json");assert overlap["simultaneous_live_count"]==3
            waits.append(timeout);self.returncode=0;return 0
    def identity(pid):
        child=next(x for x in children if x.pid==pid)
        return None if child.returncode is not None else {"pid":pid,"start_ticks":pid+10,"argv":child.command,"cwd":str(root/"source"),"pgid":pid}
    monkeypatch.setattr(r.subprocess,"Popen",Child);monkeypatch.setattr(r,"proc_identity",identity)
    clock=iter((100.,101.,102.,110.,120.,130.));monkeypatch.setattr(r.time,"monotonic",lambda:next(clock))
    rows=r._synthetic_group(c,"5090",[r.ARMS[0]]*3,"synthetic")
    assert len(rows)==3 and all(x["start_ticks"]==x["pid"]+10 for x in rows)
    assert waits==[7190.,7181.,7172.]


def test_synthetic_cleanup_rejects_foreign_identity_without_signal(contract,monkeypatch):
    child=type("Child",(),{"pid":999,"poll":lambda self:None})()
    monkeypatch.setattr(r,"proc_identity",lambda _: {"pid":999,"argv":["foreign"],"cwd":"/other","pgid":999,"start_ticks":1})
    monkeypatch.setattr(r,"terminate_owned_child",lambda _:pytest.fail("Must not signal unknown owner"))
    with pytest.raises(ValueError,match="no signal"):r.cleanup_owned_synthetic(child,["new"],Path("/new"),{"start_ticks":2})


def test_qualification_cleanup_records_exit_after_unexpected_wait_error(contract,monkeypatch):
    c=contract;root=Path(c["hosts"]["5090"]["root"]);children=[]
    class Child:
        def __init__(self,command,**kw):self.command=command;self.pid=999;self.returncode=None;children.append(self)
        def poll(self):return self.returncode
        def wait(self,timeout):raise RuntimeError("fabricated wait failure")
    monkeypatch.setattr(r.subprocess,"Popen",Child)
    monkeypatch.setattr(r,"proc_identity",lambda pid:{"pid":pid,"argv":children[0].command,"cwd":str(root/"source"),"pgid":pid,"start_ticks":10})
    def cleanup(child,*args,**kw):assert child is children[0];assert kw["grace"]==90;child.returncode=-15
    monkeypatch.setattr(r,"cleanup_owned_synthetic",cleanup)
    with pytest.raises(RuntimeError,match="fabricated"):r.launch_dispatch(c,"5090",root/"execution_contract.json")
    assert children[0].returncode==-15 and r.read(root/"qualification_exit.json")["actual_os_returncode"]==-15


def test_qualifier_sigterm_drains_only_its_three_synthetic_groups(contract,monkeypatch):
    c=contract;root=Path(c["hosts"]["5090"]["root"]);r.write(root/"execution_contract.json",c)
    handlers={};children=[];stopped=[]
    monkeypatch.setattr(r.signal,"signal",lambda sig,fn:handlers.setdefault(sig,fn))
    class Child:
        def __init__(self,command,**kw):
            self.pid=7000+len(children);self.command=command;self.returncode=None;children.append(self)
            r.write(Path(command[-3]),{"pid":self.pid},exclusive=True)
        def poll(self):return self.returncode
        def wait(self,timeout):handlers[r.signal.SIGTERM](r.signal.SIGTERM,None)
    def identity(pid):
        child=next(x for x in children if x.pid==pid)
        return None if child.returncode is not None else {"pid":pid,"argv":child.command,"cwd":str(root/"source"),"pgid":pid,"start_ticks":pid+1}
    def stop(child):
        assert child in children and child.pid not in {351034,351048,351049,353905}
        stopped.append(child.pid);child.returncode=-15
    monkeypatch.setattr(r.subprocess,"Popen",Child);monkeypatch.setattr(r,"proc_identity",identity)
    monkeypatch.setattr(r,"terminate_owned_child",stop)
    monkeypatch.setattr(r,"qualify",lambda c,h:r._synthetic_group(c,h,[r.ARMS[0]]*3,"signal_case"))
    with pytest.raises(KeyboardInterrupt):r.main(["--contract",str(root/"execution_contract.json"),"--host","5090","--mode","qualify"])
    assert len(stopped)==3 and all(x.returncode==-15 for x in children)
    assert all(r.read(root/f"signal_case_{i}.json")["actual_os_returncode"]==-15 for i in range(3))


def test_queue_registers_new_child_before_proc_read_interrupt(contract,monkeypatch):
    c=contract;root=Path(c["hosts"]["5090"]["root"]);(root/"logs").mkdir();children=[]
    class Child:
        def __init__(self,command,**kw):self.pid=8888;self.command=command;self.returncode=None;children.append(self)
        def poll(self):return self.returncode
    monkeypatch.setattr(r._impl.subprocess,"Popen",Child)
    monkeypatch.setattr(r._impl,"gpu_state",lambda *a:({"compute_pids":[]},c["hosts"]["5090"]["paused_processes"]))
    monkeypatch.setattr(r._impl,"proc_identity",lambda pid:(_ for _ in ()).throw(KeyboardInterrupt("fabricated proc read interrupt")))
    monkeypatch.setattr(r,"proc_identity",lambda pid:{"pid":pid,"argv":children[0].command,"cwd":str(root/"source"),"pgid":pid,"start_ticks":42})
    monkeypatch.setattr(r,"_QUEUE_OWNED_CHILDREN",[]);r._impl._EXTRA42_OWNED=r._QUEUE_OWNED_CHILDREN
    def stop(child):assert child is children[0];child.returncode=-15
    monkeypatch.setattr(r,"terminate_owned_child",stop)
    with pytest.raises(KeyboardInterrupt):r.run_stage(c,"5090",root/"execution_contract.json","train",r.time.monotonic())
    assert len(r._QUEUE_OWNED_CHILDREN)==1 and children[0].returncode==-15


def test_finish_existing_failure_or_retrieval_uses_no_ssh(tmp_path,monkeypatch):
    monkeypatch.setattr(m,"BUNDLE",tmp_path/"b");monkeypatch.setattr(m,"REPORT",tmp_path/"r")
    m.write(m.REPORT/"QUEUE_FAILURE.json",{"status":"failed"})
    monkeypatch.setattr(m._transport,"ssh",lambda *a,**kw:pytest.fail("No repeated remote work"))
    with pytest.raises(ValueError,match="no repeated remote"):m.finish("fake")


def test_emitted_deploy_supervisor_is_bounded_and_identity_bound(contract,tmp_path,monkeypatch):
    c=contract;bundle=tmp_path/"bundle";bundle.mkdir();report=tmp_path/"report"
    m.write(bundle/"execution_contract.json",c);(bundle/"deployment.tar.gz").write_bytes(b"fabricated")
    monkeypatch.setattr(m,"BUNDLE",bundle);monkeypatch.setattr(m,"REPORT",report)
    monkeypatch.setattr(m,"verify_local",lambda *a,**kw:c);monkeypatch.setattr(m,"fresh_preflight",lambda *a:None)
    commands=[];monkeypatch.setattr(m._transport,"ssh",lambda host,command,**kw:commands.append(command) or object())
    monkeypatch.setattr(m._transport,"save_transport",lambda *a:None)
    assert m.deploy("fake")["new_fit_count"]==9 and len(commands)==2
    shell=shlex.split(commands[-1])[-1];words=shlex.split(shell);program=words[words.index("-c")+1]
    compile(program,"<bounded-dispatcher-supervisor>","exec")
    assert "deadline-time.time()" in program and "dispatcher_launch.json" in program and "start_ticks" in program and "start_new_session=True" in program
    with pytest.raises(ValueError,match="no retry"):m.deploy("fake")
    assert len(commands)==2


def test_collection_program_compiles_and_requires_all27_exits(contract):
    program=m.collection_program(contract);compile(program,"<immutable-mac9-collection>","exec")
    assert "f['test_count']==9" in program and "r.verify_receipt" in program and "Dispatcher identity differs" in program
    assert "gpu,paused=r.gpu_state" in program and "pids" in program and ".write(" not in program


def test_preserved_queue_and_transport_unchanged():
    assert r.sha_file(r._BASE)==r._BASE_SHA and m.sha(m._BASE)==m._BASE_SHA


def test_empty_paused_processes_are_supported(contract,monkeypatch):
    assert r.paused_processes(contract,"5090")==[]
    monkeypatch.setattr(r,"gpu_snapshot",lambda uuid:{"compute_pids":[],"uuid":uuid})
    assert r.gpu_state(contract,"5090")[1]==[]
    monkeypatch.setattr(r,"gpu_snapshot",lambda uuid:{"compute_pids":[100],"uuid":uuid})
    with pytest.raises(ValueError,match="Unidentified GPU"):r.gpu_state(contract,"5090")
    assert r.gpu_state(contract,"5090",[100])[0]["compute_pids"]==[100]


def test_unexpected_paused_process_or_arm_rejected(contract):
    c=deepcopy(contract);c["hosts"]["5090"]["paused_processes"]=[{"pid":1}]
    with pytest.raises(ValueError):r.validate_contract(c,"5090")
    c=deepcopy(contract);c["jobs"][0]["arm"]="titantpp_history_mlp_shared15_v1"
    with pytest.raises(ValueError):r.validate_contract(c,"5090")


def test_fresh_preflight_is_read_only_and_rejects_compute_process(contract,monkeypatch,tmp_path):
    captured=[];monkeypatch.setattr(m,"REPORT",tmp_path)
    monkeypatch.setattr(m._transport,"ssh",lambda host,command:captured.append(command) or object())
    monkeypatch.setattr(m._transport,"save_transport",lambda *a:json.dumps({"status":"passed","prior_owned_pids_absent":True}))
    m.fresh_preflight(contract)
    words=shlex.split(captured[0]);program=words[words.index("-c")+1]
    compile(program,"<fresh-5090-preflight>","exec")
    assert "not gpu['compute_pids']" in program and "r.proc_identity(pid) is None" in program
    assert "gpu_state(" not in program and "os.kill" not in program


def _fabricated_complete_collection(c, bundle):
    root=Path(c["hosts"]["5090"]["root"]);m.write(root/"execution_contract.json",c)
    m.write(bundle/"execution_contract.json",c)
    stages={stage:{"status":"complete","exits":{},"receipts":{}} for stage in r.STAGES}
    for i,job in enumerate(c["jobs"]):
        selected=root/"run"/job["id"]/"selected.pt";selected.parent.mkdir(parents=True)
        selected.write_bytes(b"fabricated-model-"+str(i).encode());rel=str(selected.relative_to(root));digest=m.sha(selected)
        for stage in r.STAGES:
            receipt={"schema":"titans_mac_accuracy9_stage_v1","status":"complete","contract_sha256":m.canonical(c),
                "job_id":job["id"],"stage":stage,**{k:job[k] for k in ("dataset","arm","seed","time_input","loss_mode","time_alpha")},
                "time_head_trained":True,"quantity_only":False,"held_out_test_evaluated":stage=="test",
                "files":{rel:digest},"selected_checkpoint":str(selected),"checkpoint_sha256":digest,
                "selected_epoch":i+1,"state_sha256":"b"*64,"validation_replay_passed":stage=="validation"}
            path=selected.parent/f"{stage}_receipt.json";m.write(path,receipt)
            stages[stage]["receipts"][job["id"]]={"path":str(path.relative_to(root)),"sha256":m.sha(path)}
            stages[stage]["exits"][job["id"]]={"pid":2000+i,"actual_os_returncode":0,"timeout":False}
    identity={"pid":501,"start_ticks":8,"argv":["fabricated-dispatcher"],"cwd":str(root/"source"),"pgid":501}
    m.write(root/"dispatcher_launch.json",{"pid":501,"command":identity["argv"],"identity":identity})
    m.write(root/"start.json",{"pid":501,"identity":identity})
    m.write(root/"dispatcher_exit.json",{"pid":501,"start_ticks":8,"command":identity["argv"],
        "launch_sha256":m.sha(root/"dispatcher_launch.json"),"timeout":False,"actual_child_exit_observed":True,"actual_os_returncode":0})
    m.write(root/"collection_observation.json",{"owned_pids_absent":True,"historical_campaigns_changed":False,"paused_processes":[]})
    files={str(p.relative_to(root)):m.sha(p) for p in root.rglob("*") if p.is_file()}
    m.write(root/"artifact_manifest.json",{"files":files})
    final={"status":"complete","contract_sha256":m.canonical(c),"completed_fit_count":9,"validation_replay_count":9,
        "test_count":9,"owned_worker_pids_absent":True,"held_out_test_evaluated":True,"stages":stages,
        "artifact_manifest":{"path":"artifact_manifest.json","sha256":m.sha(root/"artifact_manifest.json"),"file_count":len(files)}}
    m.write(root/"final_receipt.json",final);return root,final


def test_local_collected_all27_exits_and_selected_binaries_are_bound(contract,monkeypatch,tmp_path):
    bundle=tmp_path/"bundle";monkeypatch.setattr(m,"BUNDLE",bundle)
    root,final=_fabricated_complete_collection(contract,bundle)
    assert m.verify_collected(contract,root)["test_count"]==9
    final["stages"]["test"]["exits"][contract["jobs"][0]["id"]]["actual_os_returncode"]=1
    (root/"final_receipt.json").write_text(json.dumps(final))
    with pytest.raises(ValueError,match="Actual stage exit"):m.verify_collected(contract,root)


def test_empty_terminal_inventory_rejected_even_with_rehashed_manifest(contract,monkeypatch,tmp_path):
    bundle=tmp_path/"bundle";monkeypatch.setattr(m,"BUNDLE",bundle)
    root,final=_fabricated_complete_collection(contract,bundle)
    (root/"artifact_manifest.json").write_text(json.dumps({"files":{}}))
    final["artifact_manifest"].update(sha256=m.sha(root/"artifact_manifest.json"),file_count=0)
    (root/"final_receipt.json").write_text(json.dumps(final))
    with pytest.raises(ValueError,match="Empty terminal inventory"):m.verify_collected(contract,root)
