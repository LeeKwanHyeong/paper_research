"""Fabricated archives and transports; no project data, SSH or model loading."""
from copy import deepcopy
import gzip
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time
from unittest.mock import patch

import pytest
from paper.scripts import collect_titans_mac_accuracy9_recovery as m


def put(root, name, value):
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(value if isinstance(value, bytes) else (json.dumps(value, sort_keys=True) + "\n").encode())
    return p


def hashes(root):
    return {str(p.relative_to(root)): m.sha(p) for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def terminal(tmp_path):
    evidence = tmp_path / "evidence"
    new, old = evidence / "recovery", evidence / "original"
    new.mkdir(parents=True); old.mkdir()
    remote, prior = str(new), str(old)
    for i in range(989):
        for root in (new, old):
            put(root / "source", f"frozen/{i}.py", b"fabricated source")
    source = hashes(new / "source")
    inp = "inputs/fabricated.bin"
    for root in (new, old):
        put(root / "source", inp, b"fabricated input")
    inputs = {inp: m.sha(new / "source" / inp)}
    operations = {}
    for name in ("run_titans_mac_accuracy9_recovery.py", "titans_mac_accuracy9_recovery_runtime.py", "bounded.py"):
        rel = "paper/scripts/" + name
        operations[rel] = m.sha(put(new / "source", rel, b"fabricated operation"))
    directories = ["models", "utils", "sample_data"]
    for name in directories:
        (new / "source" / name).mkdir()
    datasets = ("raf_spare_parts", "yellow_trip_hourly", "intermittent_frozen_5000")
    jobs = [dict(id=f"{d}__titantpp_titans_mac__seed{s}", dataset=d, seed=s, arm="titantpp_titans_mac",
                 host="5090", time_alpha=1., time_input="log1p", loss_mode="log_mse")
            for d in datasets for s in (42, 52, 62)]
    population = dict(target_count=1, target_identity_sha256="1"*64, target_quantity_sha256="2"*64)
    original = dict(hosts={"5090": {"root": prior}}, design_sha256="e"*64, source=dict(files=source, files_sha256=m.canonical(source),
                    base_git_revision="fabricated"), jobs=jobs, datasets=[
                    dict(dataset_id=d, inherited_data_identity={"populations": {"validation": population}},
                         full_data_identity={"populations": {"test": population}}) for d in datasets])
    original_file = put(old, "execution_contract.json", original)
    effective = deepcopy(original); effective["hosts"]["5090"]["root"] = remote
    proof = dict(schema="time_head_joint_training_verification_v1", time_alpha=1.,
        initial_time_head_sha256="a"*64, final_time_head_sha256="b"*64, checkpoint_time_head_sha256="c"*64,
        optimizer_steps_checked=2, joint_loss_observations=4,
        **{k: True for k in ("finite_time_head_gradient_observed", "nonzero_time_head_gradient_observed",
           "time_head_parameters_changed", "optimizer_state_present", "joint_loss_includes_time",
           "all_trainable_parameters_in_optimizer")})
    metrics = dict(overall=dict(count=1, qty_rmse=1., qty_mae=1., time_nll=1.),
                   tail=dict(count=0, qty_rmse=None, qty_mae=None, time_nll=None))
    scientific, enriched = {}, []
    for job in jobs:
        jid = job["id"]; folder = "run/" + jid + "/"; checkpoint = folder + "runs/selected.pt"
        put(new, checkpoint, b"fabricated checkpoint")
        put(new, folder + "runs/history.json", {"history": [dict(epoch=1, val_qty_rmse=1.), dict(epoch=2, val_qty_rmse=2.)]})
        prediction = folder + "training_validation_predictions.jsonl.gz"
        row = dict(target_index=0, entity_id="fabricated", target_seq=3, raw_quantity=1.,
                   history_length=2, split="validation", predicted_raw_quantity=2., time_nll=1.)
        put(new, prediction, gzip.compress((json.dumps(row)+"\n").encode(), mtime=0))
        fit = dict(**job, job_id=jid, schema="titans_mac_accuracy9_stage_v1", status="complete", stage="train",
            source_files_sha256=m.canonical(source), source_revision="fabricated", contract_sha256=m.canonical(effective),
            selected_checkpoint=remote+"/"+checkpoint, selected_epoch=1, checkpoint_sha256=m.sha(new/checkpoint),
            state_sha256="d"*64, time_head_training_verification=proof, target_identity=population,
            summary=dict(best_epoch=1, checkpoint_path=remote+"/"+checkpoint,
                         checkpoint_state_sha256="d"*64, time_head_training_verification=proof),
            validation_reference=dict(prediction_path=remote+"/"+prediction, metrics=metrics), held_out_test_evaluated=False)
        fit["files"] = {str(p.relative_to(new)): m.sha(p) for p in (new/folder).rglob("*") if p.is_file()}
        put(new, folder+"train_receipt.json", fit)
        scientific[jid] = fit
        item = dict(**job, recovery_mode="resume" if job["dataset"] == datasets[-1] else "reference")
        if item["recovery_mode"] == "reference":
            origin = deepcopy(fit); origin.update(contract_sha256=m.canonical(original), selected_checkpoint=prior+"/"+checkpoint)
            origin["validation_reference"]["prediction_path"] = prior+"/"+prediction
            for name in fit["files"]:
                put(old, name, (new/name).read_bytes())
            p = put(old, folder+"train_receipt.json", origin)
            item["origin_train_receipt"] = dict(path=prior+"/"+folder+"train_receipt.json", sha256=m.sha(p))
            exited = dict(job_id=jid, stage="train", actual_os_returncode=0, timeout=False, pid=500+job["seed"],
                          start_ticks=5000+job["seed"], started_at="original-start", observed_exit_at="original-exit")
            p = put(old, folder+"train_exit.json", exited)
            item["origin_train_exit"] = dict(path=prior+"/"+folder+"train_exit.json", sha256=m.sha(p))
            put(old, folder+"train_launch.json", exited); put(new, folder+"train_exit.json", exited)
        else:
            p = put(old, folder+"last_epoch_state.pt", b"fabricated interrupted state")
            put(new, folder+"origin/last_epoch_state.pt", p.read_bytes())
            item.update(resume_epoch=2, origin_resume_checkpoint=dict(path=prior+"/"+folder+"last_epoch_state.pt", sha256=m.sha(p)))
        enriched.append(item)
    for i in range(1141-len(hashes(old))):
        put(old, f"padding/{i}.txt", b"immutable historical record")
    immutable = hashes(old)
    c = dict(schema=m.r.SCHEMA, host="5090", root=remote, source_root=remote+"/source", python="/native/python",
        boot_id="fabricated-boot", source=original["source"], input_files=inputs, operation_files=operations,
        operation_files_sha256=m.canonical(operations), required_source_directories=directories, jobs=enriched,
        scientific_contract=dict(path=prior+"/execution_contract.json", raw_sha256=m.sha(original_file),
                                 canonical_sha256=m.canonical(original)),
        immutable_originals=dict(root=prior, files=immutable, files_sha256=m.canonical(immutable)))
    for key, mode in (("training_jobs","resume"), ("reference_jobs","reference")):
        c[key] = [j["id"] for j in enriched if j["recovery_mode"] == mode]
    c["validation_jobs"] = c["test_jobs"] = [j["id"] for j in enriched]
    put(new, "recovery_contract.json", c)
    stages, owned, pid = {}, {2000}, 1000
    for stage, keys in (("reference", c["reference_jobs"]), ("train", c["training_jobs"]),
                        ("validation", c["validation_jobs"]), ("test", c["test_jobs"])):
        stages[stage] = dict(status="complete", exits={}, receipts={})
        for job in [j for j in enriched if j["id"] in keys]:
            pid += 1; owned.add(pid); folder = "run/"+job["id"]+"/"; fit = scientific[job["id"]]
            command = m.r.worker_command(c, Path(remote)/"recovery_contract.json", job, stage)
            exited = dict(pid=pid, start_ticks=pid+100, command=command, started_at="start", observed_exit_at="exit",
                          boot_id=c["boot_id"], actual_os_returncode=0, timeout=False, job_id=job["id"], stage=stage)
            for stream in ("stdout","stderr"):
                p = put(new, "logs/"+job["id"]+"."+stage+"."+stream+".log", b"")
                exited[stream+"_sha256"], exited[stream+"_bytes"] = m.sha(p), 0
            put(new, folder+stage+"_exit.json", exited); put(new, folder+stage+"_launch.json", exited)
            value = deepcopy(fit); value.update(schema=m.r.STAGE_SCHEMA, stage=stage, contract_sha256=m.canonical(c),
                scientific_contract_sha256=m.canonical(original), scientific_contract_raw_sha256=c["scientific_contract"]["raw_sha256"],
                effective_scientific_contract_sha256=m.canonical(effective), recovery_mode=job["recovery_mode"])
            if stage in ("reference","train"):
                value["metrics"] = metrics
                value["lineage"] = {"reference_retrained": False} if stage == "reference" else dict(
                    resumed_from_epoch=2, next_epoch=3, resume_validation={"validated":True}, resume_checkpoint=dict(
                    original_path=job["origin_resume_checkpoint"]["path"], original_sha256=job["origin_resume_checkpoint"]["sha256"],
                    copy_sha256=job["origin_resume_checkpoint"]["sha256"], copy_path=remote+"/"+folder+"origin/last_epoch_state.pt"))
            else:
                p = put(new, folder+stage+"_predictions.jsonl.gz", (new/(folder+"training_validation_predictions.jsonl.gz")).read_bytes())
                value.update(metrics=metrics, target_identity=population, parameters_unchanged=True, context_unchanged=True,
                    native_runtime_checked=True, held_out_test_evaluated=stage=="test", validation_replay_passed=stage=="validation",
                    individual_predictions_match=stage=="validation", prediction_path=remote+"/"+str(p.relative_to(new)))
                internal = dict(value, contract_sha256=m.canonical(effective))
                put(new, folder+stage+"_receipt.json", internal)
            value["files"] = {str(p.relative_to(new)):m.sha(p) for p in (new/folder).rglob("*") if p.is_file()}
            p = put(new, folder+stage+"_recovery_receipt.json", value)
            stages[stage]["exits"][job["id"]] = exited
            stages[stage]["receipts"][job["id"]] = dict(path=str(p.relative_to(new)), sha256=m.sha(p))
    command = [c["python"],"-u","-m",m.r.MODULE,"--contract",remote+"/recovery_contract.json","--contract-sha",m.canonical(c),"--mode","dispatch"]
    identity = dict(pid=2000,start_ticks=2001,argv=command,cwd=c["source_root"],pgid=2000)
    put(new,"dispatcher_launch.json",dict(identity=identity,command=command,at="start",contract_sha256=m.canonical(c)))
    put(new,"start.json",dict(pid=2000,identity=identity,contract_sha256=m.canonical(c)))
    exited = dict(pid=2000,start_ticks=2001,command=command,started_at="start",observed_exit_at="exit",
                  actual_os_returncode=0,timeout=False,boot_id=c["boot_id"])
    for stream in ("stdout","stderr"):
        p=put(new,"dispatcher."+stream+".log",b"");exited[stream+"_sha256"],exited[stream+"_bytes"]=m.sha(p),0
    put(new,"dispatcher_exit.json",exited)
    gate=dict(status="passed",host="5090",contract_sha256=m.canonical(effective),jobs={})
    for job in enriched:
        f="run/"+job["id"]+"/"
        gate["jobs"][job["id"]]={key:m.sha(new/(f+name)) for key,name in (
            ("train_exit_sha256","train_exit.json"),("validation_exit_sha256","validation_exit.json"),
            ("validation_receipt_sha256","validation_receipt.json"))}
    put(new,"validation_gate.json",gate)
    excluded={"artifact_manifest.json","final_receipt.json","dispatcher.stdout.log","dispatcher.stderr.log","dispatcher_exit.json"}
    files={n:h for n,h in hashes(new).items() if not n.startswith("source/") and n not in excluded}
    p=put(new,"artifact_manifest.json",dict(files=files,excluded_lifecycle_files=sorted(excluded)))
    put(new,"final_receipt.json",dict(schema=m.r.SCHEMA,status="complete",contract_sha256=m.canonical(c),
        resumed_fit_count=3,preserved_completed_fit_count=6,validation_replay_count=9,candidate_test_count=9,
        held_out_test_evaluated=True,owned_worker_pids_absent=True,stages=stages,
        immutable_originals_verification=dict(file_count=1141,files_sha256=m.canonical(immutable),original_files_unchanged=True),
        artifact_manifest=dict(path="artifact_manifest.json",sha256=m.sha(p),file_count=len(files))))
    put(evidence,"collection_observation.json",dict(schema=m.SCHEMA,contract_sha256=m.canonical(c),boot_id=c["boot_id"],
        owned_pids_absent=True,owned_pids=sorted(owned),gpu={"compute_pids":[]},original_files_unchanged=True,
        supervisor_own_os_exit_directly_observed=False))
    return c,evidence


def test_full_terminal_and_correct_exit_count_interpretation(terminal):
    c,e=terminal; result=m.verify_terminal_evidence(c,e)
    assert result["historical_fit_exit0_count"]==6 and result["resumed_fit_exit0_count"]==3
    assert result["actual_scientific_stage_exit0_count"]==27 and result["current_scientific_child_exit0_count"]==21
    assert result["reference_copy_child_exit0_count"]==6 and len(m.comparison_records(e,c))==18


@pytest.mark.parametrize("target",["original","source","checkpoint","log","observation"])
def test_collected_tampering_rejected(terminal,target):
    c,e=terminal
    paths={"original":e/"original/padding/0.txt","source":e/"recovery/source/frozen/0.py",
           "checkpoint":next((e/"recovery/run").rglob("selected.pt")),
           "log":next((e/"recovery/logs").iterdir()),"observation":e/"collection_observation.json"}
    paths[target].write_bytes(b"tampered")
    with pytest.raises((ValueError,json.JSONDecodeError)):m.verify_terminal_evidence(c,e)


def test_empty_manifest_and_unbound_result_rejected(terminal):
    c,e=terminal;root=e/"recovery";put(root,"unbound.json",{})
    with pytest.raises(ValueError,match="omitted"):m.verify_terminal_evidence(c,e)
    (root/"unbound.json").unlink()
    manifest=m.read(root/"artifact_manifest.json");manifest["files"]={};put(root,"artifact_manifest.json",manifest)
    final=m.read(root/"final_receipt.json");final["artifact_manifest"].update(sha256=m.sha(root/"artifact_manifest.json"),file_count=0)
    put(root,"final_receipt.json",final)
    with pytest.raises(ValueError,match="Empty"):m.verify_terminal_evidence(c,e)


@pytest.mark.parametrize("name,kind",[("../escape","file"),("/absolute","file"),("recovery/link","symlink"),
    ("original/hard","hardlink"),("unknown.txt","file"),("recovery/duplicate","duplicate")])
def test_unsafe_tar_rejected(tmp_path,name,kind):
    archive=tmp_path/"bad.tar.gz"
    with tarfile.open(archive,"w:gz") as t:
        info=tarfile.TarInfo(name);info.size=1
        if kind=="symlink":info.type,info.linkname,info.size=tarfile.SYMTYPE,"/outside",0
        if kind=="hardlink":info.type,info.linkname,info.size=tarfile.LNKTYPE,"../outside",0
        t.addfile(info,io.BytesIO(b"x") if info.isfile() else None)
        if kind=="duplicate":t.addfile(info,io.BytesIO(b"x"))
    with pytest.raises(ValueError):m.safe_extract(archive,tmp_path/"extracted")


def test_empty_required_directory_survives_archive(terminal,tmp_path):
    c,e=terminal;archive=tmp_path/"valid.tar.gz"
    with tarfile.open(archive,"w:gz") as t:
        for name in c["required_source_directories"]:
            info=tarfile.TarInfo("recovery/source/"+name);info.type=tarfile.DIRTYPE;t.addfile(info)
        for p in e.rglob("*"):
            if p.is_file():t.add(p,arcname=str(p.relative_to(e)),recursive=False)
    m.safe_extract(archive,tmp_path/"extracted")
    assert (tmp_path/"extracted/recovery/source/sample_data").is_dir()
    assert m.verify_terminal_evidence(c,tmp_path/"extracted")["status"]=="verified"


def configure(tmp_path,monkeypatch,c):
    monkeypatch.setattr(m.manager,"REPORT",tmp_path/"reports/execution_v3")
    monkeypatch.setattr(m.manager,"BUNDLE",tmp_path/"bundle_v3")
    monkeypatch.setattr(m.manager,"verify_local",lambda _:c)
    m.manager.REPORT.mkdir(parents=True)
    put(m.manager.REPORT,"LAUNCHED.json",{"contract_sha256":m.canonical(c)})


def test_local_waiting_and_existing_attempt_never_open_ssh(terminal,tmp_path,monkeypatch):
    c,_=terminal;configure(tmp_path,monkeypatch,c)
    with patch.object(m,"transport",side_effect=AssertionError("Unexpected SSH")):
        assert m.advance(m.canonical(c),check_local=True)["remote_calls"]==0
        m.report_dir().mkdir(exist_ok=True)
        put(m.report_dir(),"LATEST.json",{"checked_unix":time.time(),"status":"waiting"})
        assert m.advance(m.canonical(c))["status"]=="waiting_next_check"
        put(m.report_dir(),"COLLECTION_ATTEMPT.json",{"prior_attempt":True})
        assert m.advance(m.canonical(c))["status"]=="blocked_preserved_failure_or_attempt"


def test_timeout_retains_actual_exit_partial_output_and_never_signals_remote(tmp_path,monkeypatch):
    class Child:
        returncode=None
        waits=0
        def __init__(self,command,*,stdout,stderr,**kwargs):
            stdout.write(b"partial archive");stderr.write(b"fabricated timeout")
        def wait(self,timeout=None):
            self.waits+=1
            if self.waits==1:raise subprocess.TimeoutExpired("fabricated SSH",timeout)
            return self.returncode
        def kill(self):self.returncode=-9
    monkeypatch.setattr(m.subprocess,"Popen",Child)
    with pytest.raises(ValueError,match="preserved"):
        m.transport(dict(root="/native",source_root="/native/source",python="/native/python"),"pass",tmp_path/"transport",label="fabricated",timeout=45)
    record=m.read(tmp_path/"transport/TRANSPORT.json")
    assert record["actual_transport_exit"]==-9 and record["transport_timeout"]
    assert record["native_scientific_process_signaled"] is False and record["automatic_retry"] is False
    assert (tmp_path/"transport/stdout").read_bytes()==b"partial archive"


def test_generated_probes_are_bounded_and_reuse_native_verifiers(terminal):
    import ast
    c,_=terminal;short=m.terminal_program(c);archive=m.collection_program(c)
    compile(short,"short","exec");compile(archive,"archive","exec")
    assert "for attempt in range(51)" in short and "deadline=began+1." in short
    assert "time.sleep(.02)" in short and "while " not in short and "r.verify_files(c)" not in short
    assert all(x in archive for x in ("r.verify_files(c)","r.verify_receipt(c,j,stage)","r.gpu_state(c)"))
    for code in (short,archive):
        assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
            and isinstance(n.func.value,ast.Name) and n.func.value.id=="r" and n.func.attr=="write"
            for n in ast.walk(ast.parse(code)))
    assert "terminate" not in short+archive


def archive_bytes(c, evidence):
    blob=io.BytesIO()
    with tarfile.open(fileobj=blob,mode="w:gz") as t:
        for name in c["required_source_directories"]:
            info=tarfile.TarInfo("recovery/source/"+name);info.type=tarfile.DIRTYPE;t.addfile(info)
        for p in evidence.rglob("*"):
            if p.is_file():t.add(p,arcname=str(p.relative_to(evidence)),recursive=False)
    return blob.getvalue()


def test_first_collection_then_cache_and_check_local_are_remote_free(terminal,tmp_path,monkeypatch):
    c,e=terminal;configure(tmp_path,monkeypatch,c);calls=[]
    data=archive_bytes(c,e)
    def fake_transport(contract,program,folder,*,label,timeout,output=None):
        calls.append((label,timeout));folder.mkdir(parents=True)
        out=Path(output) if output is not None else folder/"stdout"
        if label=="read_only_terminal":
            payload=dict(schema=m.SCHEMA,contract_sha256=m.canonical(c),boot_id=c["boot_id"],
                         observed_at="fabricated-observation",read_only_check=True,status="terminal")
            put(out.parent,out.name,payload)
        else:
            out.write_bytes(data)
        err=put(folder,"stderr",b"")
        record=dict(actual_transport_exit=0,transport_timeout=False,transport_error=None,
            stdout_path=str(out),stdout_sha256=m.sha(out),stderr_path=str(err),stderr_sha256=m.sha(err))
        put(folder,"TRANSPORT.json",record)
        return record
    monkeypatch.setattr(m,"transport",fake_transport)
    result=m.advance(m.canonical(c))
    assert result["status"]=="collection_complete" and result["remote_calls"]==2
    assert calls==[("read_only_terminal",45),("first_collection",900)]
    cached=m.advance(m.canonical(c))
    checked=m.advance(m.canonical(c),check_local=True)
    assert cached["cache_verified"] and checked["cache_verified"]
    assert cached["remote_calls"]==checked["remote_calls"]==0 and len(calls)==2
    receipt=m.retrieval_dir()/"receipt.json"
    receipt.write_bytes(b"tampered final receipt")
    with pytest.raises(ValueError,match="cache changed"):
        m.advance(m.canonical(c))
    assert len(calls)==2


def test_incomplete_first_transfer_is_preserved_and_blocks_every_later_remote_call(terminal,tmp_path,monkeypatch):
    c,_=terminal;configure(tmp_path,monkeypatch,c);calls=[]
    def fake_transport(contract,program,folder,*,label,timeout,output=None):
        calls.append(label);folder.mkdir(parents=True)
        out=Path(output) if output is not None else folder/"stdout"
        if label=="read_only_terminal":
            put(out.parent,out.name,dict(schema=m.SCHEMA,contract_sha256=m.canonical(c),boot_id=c["boot_id"],
                observed_at="fabricated",read_only_check=True,status="terminal"))
        else:
            out.write_bytes(b"incomplete tar transport")
        err=put(folder,"stderr",b"")
        record=dict(stdout_path=str(out),stdout_sha256=m.sha(out),stderr_path=str(err),
                    stderr_sha256=m.sha(err),actual_transport_exit=0,transport_timeout=False,transport_error=None)
        put(folder,"TRANSPORT.json",record)
        return record
    monkeypatch.setattr(m,"transport",fake_transport)
    with pytest.raises(tarfile.ReadError):
        m.advance(m.canonical(c))
    assert (m.report_dir()/"FAILURE.json").exists()
    assert (m.report_dir()/"COLLECTION_ATTEMPT.json").exists()
    assert (m.retrieval_dir()/"archive.tar.gz").read_bytes()==b"incomplete tar transport"
    assert m.advance(m.canonical(c))["remote_calls"]==0 and len(calls)==2


def test_terminal_failure_is_not_collection_or_training_retry(terminal,tmp_path,monkeypatch):
    c,_=terminal;configure(tmp_path,monkeypatch,c)
    def fake_transport(contract,program,folder,**kwargs):
        folder.mkdir(parents=True);out=put(folder,"stdout",dict(schema=m.SCHEMA,contract_sha256=m.canonical(c),
            boot_id=c["boot_id"],observed_at="fabricated",read_only_check=True,status="terminal_failure"))
        record=dict(stdout_path=str(out));put(folder,"TRANSPORT.json",record);return record
    monkeypatch.setattr(m,"transport",fake_transport)
    with pytest.raises(ValueError,match="Native dispatcher exited unsuccessfully"):
        m.advance(m.canonical(c))
    assert not (m.report_dir()/"COLLECTION_ATTEMPT.json").exists()
    assert m.advance(m.canonical(c))["remote_calls"]==0


@pytest.fixture
def bridged(terminal,tmp_path,monkeypatch):
    import shutil
    from paper.scripts import titans_mac_accuracy9_completion_bridge as bridge
    from paper.scripts import run_titans_mac_accuracy9_recovery as worker
    c3,e3=terminal; old=Path(c3["root"])
    # All changes below belong solely to fabricated evidence.  The bridge's
    # specific admitted control failure has successful scientific child exits.
    for p in list((old/"run").rglob("*"))+list((old/"logs").rglob("*")):
        if p.is_file() and (p.name.startswith(("validation","test")) or ".validation." in p.name or ".test." in p.name):
            p.unlink()
    final=m.read(old/"final_receipt.json")
    final["stages"]={s:final["stages"][s] for s in ("reference","train")}
    final.update(status="failed",held_out_test_evaluated=False,validation_replay_count=0,candidate_test_count=0)
    failure=dict(stage="train",traceback="  in _identity\nValueError: Owned child identity changed")
    final["stages"]["train"].update(status="failed",failure=failure,pending_job_ids=[])
    put(old,"dispatch_failure.json",failure)
    x=m.read(old/"dispatcher_exit.json");x["actual_os_returncode"]=1;put(old,"dispatcher_exit.json",x)
    identities={}
    for job in [j for j in c3["jobs"] if j["recovery_mode"]=="resume"]:
        folder=old/"run"/job["id"];x=m.read(folder/"train_exit.json")
        identity=dict(pid=x["pid"],start_ticks=x["start_ticks"],argv=x["command"],cwd=c3["source_root"],pgid=x["pid"])
        identities[job["id"]]=identity
        if job["seed"]==62:
            x["start_ticks"]=None
            put(folder,"train_exit.json",x)
            (folder/"train_launch.json").unlink()
            final["stages"]["train"]["exits"][job["id"]]=x
    put(old,"final_receipt.json",final)
    e4=tmp_path/"bridge_evidence";new=e4/"recovery";new.mkdir(parents=True)
    shutil.copytree(e3/"original",e4/"original")
    shutil.copytree(old/"source",new/"source")
    observation=put(new,"proofs/initial_live.json",{"actual_identities":list(identities.values())})
    c4=deepcopy(c3);c4.update(root=str(new),source_root=str(new/"source"))
    c4["continuation_from"]=dict(root=c3["root"],contract_sha256=m.canonical(c3),
        contract_raw_sha256=m.sha(old/"recovery_contract.json"),
        initial_live_observation=dict(path=str(observation),sha256=m.sha(observation)))
    c4["bridge_files"]={"paper/scripts/bridge.py":m.sha(put(new/"source","paper/scripts/bridge.py",b"fabricated bridge"))}
    put(new,"recovery_contract.json",c4)
    monkeypatch.setattr(bridge,"terminal_ready",lambda *_:True)
    monkeypatch.setattr(worker.science,"verify_train_receipt",lambda contract,host,job,root,value:m.inventory(root,value["files"]))
    monkeypatch.setattr(bridge.r,"ORIGINAL_CANONICAL",c3["scientific_contract"]["canonical_sha256"])
    stages=bridge.import_completed(c4,c3,identities)
    current=set();next_pid=3000
    effective=deepcopy(m.read(e4/"original/execution_contract.json"));effective["hosts"]["5090"]["root"]=str(new)
    for stage in ("validation","test"):
        stages[stage]=dict(status="complete",exits={},receipts={})
        for job in c4["jobs"]:
            jid=job["id"];folder=new/"run"/jid;train=m.read(folder/"train_receipt.json")
            next_pid+=1;current.add(next_pid)
            command=bridge.worker_command(c4,new/"recovery_contract.json",job,stage)
            exited=dict(pid=next_pid,start_ticks=next_pid+100,command=command,started_at="bridge-start",
                observed_exit_at="bridge-exit",boot_id=c4["boot_id"],actual_os_returncode=0,timeout=False,job_id=jid,stage=stage)
            for stream in ("stdout","stderr"):
                p=put(new,"logs/"+jid+"."+stage+"."+stream+".log",b"")
                exited[stream+"_sha256"],exited[stream+"_bytes"]=m.sha(p),0
            put(folder,stage+"_exit.json",exited);put(folder,stage+"_launch.json",exited)
            p=put(folder,stage+"_predictions.jsonl.gz",Path(train["validation_reference"]["prediction_path"]).read_bytes())
            value=dict(train,stage=stage,metrics=train["validation_reference"]["metrics"],parameters_unchanged=True,
                context_unchanged=True,native_runtime_checked=True,held_out_test_evaluated=stage=="test",
                validation_replay_passed=stage=="validation",individual_predictions_match=stage=="validation",
                prediction_path=str(p))
            value["files"]={str(p.relative_to(new)):m.sha(p) for p in folder.rglob("*") if p.is_file()}
            put(folder,stage+"_receipt.json",value)
            wrapper=dict(value,schema=bridge.STAGE_SCHEMA,contract_sha256=m.canonical(c4),
                scientific_contract_sha256=c4["scientific_contract"]["canonical_sha256"],
                scientific_contract_raw_sha256=c4["scientific_contract"]["raw_sha256"],
                effective_scientific_contract_sha256=m.canonical(effective))
            wrapper["files"]={str(p.relative_to(new)):m.sha(p) for p in folder.rglob("*") if p.is_file()}
            p=put(folder,stage+"_recovery_receipt.json",wrapper)
            stages[stage]["exits"][jid]=exited
            stages[stage]["receipts"][jid]=dict(path=str(p.relative_to(new)),sha256=m.sha(p))
    pid=4000;current.add(pid)
    command=[c4["python"],"-u","-m",bridge.MODULE,"--contract",str(new/"recovery_contract.json"),
             "--contract-sha",m.canonical(c4),"--mode","dispatch"]
    identity=dict(pid=pid,start_ticks=pid+1,argv=command,cwd=c4["source_root"],pgid=pid)
    put(new,"dispatcher_launch.json",dict(identity=identity,command=command,at="bridge-start",contract_sha256=m.canonical(c4)))
    put(new,"start.json",dict(pid=pid,identity=identity,contract_sha256=m.canonical(c4)))
    x=dict(pid=pid,start_ticks=pid+1,command=command,started_at="bridge-start",observed_exit_at="bridge-exit",
           actual_os_returncode=0,timeout=False,boot_id=c4["boot_id"])
    for stream in ("stdout","stderr"):
        p=put(new,"dispatcher."+stream+".log",b"");x[stream+"_sha256"],x[stream+"_bytes"]=m.sha(p),0
    put(new,"dispatcher_exit.json",x)
    gate=dict(status="passed",host="5090",contract_sha256=m.canonical(effective),jobs={})
    inherited=[]
    for job in c4["jobs"]:
        f=new/"run"/job["id"];stage="train" if job["recovery_mode"]=="resume" else "reference"
        exited=m.read(f/(stage+"_exit.json"))
        inherited.append(dict(pid=exited["pid"],start_ticks=exited["start_ticks"],argv=exited["command"],cwd=c3["source_root"],pgid=exited["pid"]))
        gate["jobs"][job["id"]]={key:m.sha(f/name) for key,name in (("train_exit_sha256","train_exit.json"),
            ("validation_exit_sha256","validation_exit.json"),("validation_receipt_sha256","validation_receipt.json"))}
    put(new,"validation_gate.json",gate)
    excluded={"artifact_manifest.json","final_receipt.json","dispatcher.stdout.log","dispatcher.stderr.log","dispatcher_exit.json"}
    files={n:h for n,h in hashes(new).items() if not n.startswith("source/") and n not in excluded}
    p=put(new,"artifact_manifest.json",dict(files=files,excluded_lifecycle_files=sorted(excluded)))
    put(new,"final_receipt.json",dict(schema=bridge.SCHEMA,status="complete",contract_sha256=m.canonical(c4),stages=stages,
        resumed_fit_count=3,preserved_completed_fit_count=6,validation_replay_count=9,candidate_test_count=9,
        held_out_test_evaluated=True,owned_worker_pids_absent=True,current_new_training_count=0,
        imported_completed_fit_count=9,imported_resumed_fit_count=3,imported_reference_copy_exit0_count=6,
        immutable_originals_verification=dict(file_count=1141,files_sha256=c4["immutable_originals"]["files_sha256"],original_files_unchanged=True),
        artifact_manifest=dict(path="artifact_manifest.json",sha256=m.sha(p),file_count=len(files))))
    put(e4,"collection_observation.json",dict(schema=m.SCHEMA,contract_sha256=m.canonical(c4),boot_id=c4["boot_id"],
        owned_pids_absent=True,owned_pids=sorted(current),gpu={"compute_pids":[]},original_files_unchanged=True,
        supervisor_own_os_exit_directly_observed=False,inherited_process_identities=inherited,inherited_process_identities_absent=True))
    monkeypatch.setattr(m,"r",bridge)
    return c4,e4


def test_completion_bridge_full_archive_imports_nine_and_launches_no_train(bridged):
    c,e=bridged;result=m.verify_terminal_evidence(c,e)
    assert result["current_new_training_count"]==0 and result["current_scientific_child_exit0_count"]==18
    assert result["imported_completed_fit_count"]==9 and result["actual_scientific_stage_exit0_count"]==27
    assert len(m.comparison_records(e,c))==18


def test_completion_bridge_rejects_invented_tick_and_changed_raw_copy(bridged):
    c,e=bridged
    imported=m.read(e/"recovery/import_receipt.json")
    key=next(j["id"] for j in c["jobs"] if j["seed"]==62 and j["recovery_mode"]=="resume")
    entry=imported["jobs"][key]
    assert entry["launch"]["original_path"] is None
    observation=e/"recovery"/entry["start_ticks_proof"]["observation_path"]
    observation.write_bytes(b'{"invented":true}')
    with pytest.raises(ValueError):m.verify_continuation_import(c,e/"recovery")


def test_completion_controller_context_is_separate_and_restored(tmp_path,monkeypatch):
    from paper.scripts import manage_titans_mac_accuracy9_completion_bridge as actual_manager
    from paper.scripts import titans_mac_accuracy9_completion_bridge as actual_runtime
    original_manager,original_runtime=m.manager,m.r
    with m.controller_context(True):
        assert m.manager is actual_manager and m.r is actual_runtime
        assert m.report_dir()==actual_manager.REPORT/"collection_v1"
    assert m.manager is original_manager and m.r is original_runtime


def test_actual_bridge_manager_cli_check_local_never_uses_remote(tmp_path,monkeypatch,capsys):
    from paper.scripts import manage_titans_mac_accuracy9_completion_bridge as actual_manager
    monkeypatch.setattr(actual_manager,"REPORT",tmp_path/"real_module_report")
    monkeypatch.setattr(actual_manager,"BUNDLE",tmp_path/"real_module_bundle")
    monkeypatch.setattr(actual_manager,"verify_local",lambda expected:{"fabricated_contract":expected})
    with patch.object(m,"transport",side_effect=AssertionError("Unexpected SSH")):
        m.main(("--completion-bridge","--contract-sha","fabricated","--mode","check-local"))
    value=json.loads(capsys.readouterr().out)
    assert value["status"]=="not_collected" and value["remote_calls"]==0


def test_generated_native_program_names_exist_on_actual_bridge_runtime():
    import ast
    from paper.scripts import titans_mac_accuracy9_completion_bridge as actual_runtime
    from paper.scripts import titans_mac_accuracy9_recovery_runtime as preserved_runtime
    with m.controller_context(True):
        for program in (m.terminal_program({"root":"/native"}),m.collection_program({"root":"/native"})):
            module=ast.parse(program)
            attributes={n.attr for n in ast.walk(module) if isinstance(n,ast.Attribute)
                        and isinstance(n.value,ast.Name) and n.value.id=="r"}
            assert attributes and all(hasattr(actual_runtime,name) for name in attributes)
            assert "from paper.scripts.titans_mac_accuracy9_recovery_runtime import boot_id as native_boot_id" in program
            assert "r.boot_id(" not in program
    assert callable(preserved_runtime.boot_id)


def test_dispatcher_cleanup_error_is_rejected_even_outside_native_manifest(terminal):
    c,e=terminal
    path=e/"recovery/dispatcher_exit.json"
    value=m.read(path);value["cleanup_error"]="cleanup failed";put(path.parent,path.name,value)
    with pytest.raises(ValueError,match="Dispatcher cleanup failed"):
        m.verify_terminal_evidence(c,e)



@pytest.mark.parametrize("kind",["eventually_complete","stays_partial","settled_invalid","complete_failure"])
def test_native_lifecycle_reader_waits_only_for_partial_new_records(kind):
    import ast
    import hashlib
    from types import SimpleNamespace
    code=m.terminal_program(dict(root="/native/recovery"))
    function=next(node for node in ast.parse(code).body if isinstance(node,ast.FunctionDef) and node.name=="complete_record")
    clock=[0.];sleeps=[];errors=io.BytesIO();reads=[]
    class Raw:
        index=0
        def __str__(self):return "/native/recovery/dispatcher_exit.json"
        def stat(self):return SimpleNamespace(st_mtime=90. if kind=="settled_invalid" else 100.)
        def read_bytes(self):
            self.index+=1
            if kind=="complete_failure":return b'{"actual_os_returncode":1}'
            if kind in ("stays_partial","settled_invalid"):return b'{"actual_os_returncode":'
            return (b'',b'{"actual_os_returncode":',b'{"actual_os_returncode":0}')[min(self.index-1,2)]
    def pause(seconds):
        sleeps.append(seconds);clock[0]+=seconds
    env=dict(json=json,hashlib=hashlib,time=SimpleNamespace(time=lambda:100.1,monotonic=lambda:clock[0],sleep=pause),
             r=SimpleNamespace(require=m.require),sys=SimpleNamespace(stderr=SimpleNamespace(buffer=errors)),reads=reads)
    exec(compile(ast.Module(body=[function],type_ignores=[]),"bounded-record-reader","exec"),env)
    raw=Raw()
    if kind in ("stays_partial","settled_invalid"):
        with pytest.raises(ValueError,match="one-second"):
            env["complete_record"](raw)
        assert raw.index<=51 and sum(sleeps)<=1.000001
        if kind=="settled_invalid":assert raw.index==1 and not sleeps
        assert b'PRESERVED_INCOMPLETE_NATIVE_RECORD_BEGIN' in errors.getvalue()
        assert b'{"actual_os_returncode":' in errors.getvalue()
        assert b'"sha256":' in errors.getvalue()
    else:
        result=env["complete_record"](raw)
        assert result["actual_os_returncode"]==(1 if kind=="complete_failure" else 0)
        assert reads[0]["incomplete_read_count"]==(0 if kind=="complete_failure" else 2)
        assert len(sleeps)==(0 if kind=="complete_failure" else 2)
    assert "c=r.read(p/'recovery_contract.json')" in code




