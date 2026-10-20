"""One immutable ongoing Intermittent best-state capture, then bounded5080 evaluation.

The original5090 training files are read only. Capture is a single attempt;
preparation and native evaluation reuse those exact bytes without reselection.
"""
from pathlib import Path as _Path
import hashlib as _hashlib

_FROZEN_PATH = _Path(__file__).with_name("manage_titantpp_shared_raw_raf3_test.py")
_FROZEN_SHA = "c516112d606732a3093953c387b14c29ab347b565c531fd8c7a629932b33397a"
_original = _FROZEN_PATH.read_text()
if _hashlib.sha256(_original.encode()).hexdigest() != _FROZEN_SHA:
    raise ValueError("Preserved RAF3 manager changed")
_adapted = _original.replace("shared_raw_raf3", "shared_inter3_interim")
_adapted = _adapted.replace('if __name__ == "__main__": raise SystemExit(main())', '')
exec(compile(_adapted, str(_FROZEN_PATH), "exec"), globals())

DATASET = "intermittent_frozen_5000"
SCHEMA = "titantpp_shared_inter3_interim_frozen_v1"
GPU_UUID = "GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6"
REMOTE = "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/" + NAME + "_5080"
PRESERVED_OPERATIONS = {
    "paper/scripts/manage_titantpp_shared_raw_raf3_test.py": _FROZEN_SHA,
    "paper/scripts/evaluate_titantpp_shared_raw_raf3_frozen.py": "2fdd0f56757c4330e3e140ca800bbfbb5d13060f8b663d7156a1e40a7e3225c5",
    "paper/scripts/compare_titantpp_shared_raw_raf3_test.py": "5469ceb02be20d2559c51232a7b63dd5917540d5bc317e2903d0ef29141664dd",
    "paper/scripts/manage_titantpp_shared_raf_taxi6_test.py": "a95e2ea6aa6843db3f1ca9866b31754bad9c1dd7f04fd9777daa96051b0c6b6c",
    "paper/scripts/evaluate_titantpp_shared_raf_taxi6_frozen.py": "3d796732555f2d97b0a8e74f1b30cbe22310a49e0b380ad061d3f92e983f70a9",
    "paper/scripts/compare_titantpp_shared_raf_taxi6_test.py": "6254eb9304ff6e13768b6882bc299eaf2826f40cd969b6d467f26fed9ecb9b7a",
}
REVIEW_FILES = (*NEW_SOURCE, "tests/test_titantpp_shared_inter3_interim_test_ops.py",
    "tests/test_titantpp_shared_inter3_interim_frozen.py", "tests/test_titantpp_shared_inter3_interim_test_compare.py",
    "reports/" + NAME + "/SCIENCE_SPEC.json", "reports/" + NAME + "/REFERENCES.json")
_transport.BUNDLE, _transport.REPORT = BUNDLE, REPORT
_plain_gpu_snapshot = gpu_snapshot


def _training():
    from paper.scripts import manage_titantpp_shared_history_raw9_5090 as parent
    require(sha(TRAIN_BUNDLE / "execution_contract.json") == TRAIN_SHA, "Original raw9 source contract changed")
    c = parent.verify_local()
    require(len(c["source"]["files"]) == 983, "Original source983 closure changed")
    return c


def _jobs(training):
    jobs = [j for j in training["jobs"] if j["dataset"] == DATASET]
    require(len(jobs) == 3 and {j["seed"] for j in jobs} == {42, 52, 62}, "Intermittent three-seed scope changed")
    return jobs


def snapshot_collection_program(training, jobs):
    """Emit no remote writes and load only captured bytes on CPU, exactly once."""
    h = training["hosts"]["5090"]
    return f'''import io,json,pathlib,sys,tarfile,hashlib,math,datetime
from paper.scripts import titantpp_shared_history_raw9_5090_runtime as r
from simple_lab_test.search.common.runner import torch_load_checkpoint,canonical_state_dict_sha256
from paper.scripts import run_titantpp_shared_history_raw9_5090 as scientific
root=pathlib.Path({h['root']!r});cp=root/'execution_contract.json';source=root/'source'
r.require(r.sha_file(cp)=={TRAIN_SHA!r},'Raw9 contract changed');c=r.read(cp)
r.require(len(c['source']['files'])==983,'Source983 closure changed')
for name,digest in {{**c['source']['files'],**c['input_files']}}.items():r.require(r.sha_file(r.local_path(source,name))==digest,'Frozen source/input changed')
blobs={{'execution_contract.json':cp.read_bytes()}};captures=[]
digest=lambda blob:hashlib.sha256(blob).hexdigest()
for job in {jobs!r}:
 folder=root/'run'/job['id'];paths=list((folder/'runs').rglob('last_epoch_state.pt'))
 r.require(len(paths)==1,'Expected exactly one ongoing raw resume checkpoint')
 selected=paths[0];history_path=selected.parent/'history.json'
 before=r.sha_file(selected);blob=selected.read_bytes();r.require(digest(blob)==before,'Checkpoint changed during byte copy')
 payload=torch_load_checkpoint(io.BytesIO(blob),map_location='cpu')
 r.require(payload['checkpoint_type']=='epoch_resume' and payload['seed']==job['seed'] and payload['backbone']==job['arm'],'Snapshot checkpoint identity differs')
 r.require(payload['checkpoint_monitor']=='validation_raw_quantity_rmse' and payload['evaluation_scope']=='validation_only' and payload['held_out_test_evaluated'] is False,'Snapshot monitor/split differs')
 history=payload['history'];r.require(history and [x['epoch'] for x in history]==list(range(1,payload['epoch']+1)),'Snapshot history is not consecutive')
 selected_epoch=scientific.prepared.select_best_epoch(history)
 r.require(selected_epoch==payload['best_epoch'],'Snapshot best epoch is not earliest finite RMSE minimum')
 row=next(x for x in history if x['epoch']==selected_epoch)
 r.require(all(type(row[k]) in (int,float) and math.isfinite(row[k]) for k in ('val_qty_rmse','val_qty_mae','val_time_nll')),'Selected history metrics nonfinite')
 r.require(float(payload['best_selection_value'])==float(row['val_qty_rmse']),'Snapshot best metric differs')
 r.require(canonical_state_dict_sha256(payload['model_state_dict'])==payload['model_state_sha256'],'Current tensor SHA differs')
 r.require(canonical_state_dict_sha256(payload['best_state_dict'])==payload['best_state_sha256'],'Selected tensor SHA differs')
 scientific.verify_time_head_proof(payload['time_head_training_verification'],payload['model_state_dict'])
 external_blob=history_path.read_bytes();external=json.loads(external_blob)['history']
 r.require(external[:len(history)]==history,'External history does not preserve embedded history prefix')
 selected_name=str(selected.relative_to(root));history_name=str(history_path.relative_to(root))
 blobs[selected_name]=blob;blobs[history_name]=external_blob
 extra={{}}
 summary=selected.parent/'summary.json'
 if summary.exists():
  extra['summary_path']=str(summary.relative_to(root));blobs[extra['summary_path']]=summary.read_bytes()
 # Completed-training predictions are optional originals, never invented.
 receipt_path=folder/'train_receipt.json'
 if receipt_path.exists():
  original_receipt_blob=receipt_path.read_bytes();receipt=json.loads(original_receipt_blob)
  if receipt.get('selected_epoch')==selected_epoch and receipt.get('state_sha256')==payload['best_state_sha256']:
   prediction=pathlib.Path(receipt['validation_reference']['prediction_path'])
   r.require(prediction.resolve().is_relative_to(root.resolve()),'Prediction escapes original root')
   extra['validation_predictions_path']=str(prediction.relative_to(root));blobs[extra['validation_predictions_path']]=prediction.read_bytes()
   extra['original_training_reference_path']=str(receipt_path.relative_to(root));blobs[extra['original_training_reference_path']]=original_receipt_blob
 after=r.sha_file(selected);after_history_blob=history_path.read_bytes();after_history=json.loads(after_history_blob)['history']
 r.require(before==after,'Checkpoint advanced during snapshot; single attempt rejected')
 r.require(after_history[:len(external)]==external and after_history[:len(history)]==history,'External history prefix changed during snapshot')
 r.require(r.sha_file(selected)==after,'Checkpoint advanced before snapshot binding')
 definition=next(d for d in c['datasets'] if d['dataset_id']==job['dataset'])
 captures.append({{'job_id':job['id'],'dataset':job['dataset'],'arm':job['arm'],'seed':job['seed'],
 'checkpoint_path':selected_name,'checkpoint_sha256':before,'state_sha256':payload['best_state_sha256'],
 'history_path':history_name,'history_sha256':digest(external_blob),'external_history_after_sha256':digest(after_history_blob),
 'checkpoint_before_sha256':before,'checkpoint_after_sha256':after,'selected_epoch':selected_epoch,'snapshot_epoch':payload['epoch'],
 'selected_history_row':row,'source_files_sha256':c['source']['files_sha256'],'target_identity':definition['inherited_data_identity']['populations']['validation'],
 'source_root':str(source),'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra}})
for name,digest_value in {{**c['source']['files'],**c['input_files']}}.items():r.require(r.sha_file(r.local_path(source,name))==digest_value,'Frozen source/input changed after capture')
r.require(r.sha_file(cp)=={TRAIN_SHA!r},'Raw9 contract changed after capture')
blobs['snapshot_capture.json']=(json.dumps({{'schema':'titantpp_shared_inter3_interim_capture_v1','status':'frozen_interim_captured','jobs':captures,'training_terminal_required':False,'whole_raw9_complete':False,'held_out_test_evaluated':False}},ensure_ascii=False,indent=2,allow_nan=False)+chr(10)).encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for name,blob in sorted(blobs.items()):
  info=tarfile.TarInfo(name);info.size=len(blob);archive.addfile(info,io.BytesIO(blob))
'''


def _verify_capture():
    receipt = read(BUNDLE / "selected/receipt.json")
    require(receipt.get("schema") == "titantpp_shared_inter3_interim_snapshot_v1" and receipt.get("status") == "frozen_interim_captured"
        and receipt.get("selected_fit_count") == 3 and receipt.get("training_terminal_required") is False
        and receipt.get("whole_raw9_complete") is False and receipt.get("held_out_test_evaluated") is False, "Interim snapshot receipt invalid")
    require(sha(BUNDLE / "selected/original.tar.gz") == receipt["archive_sha256"]
        and sha(BUNDLE / "selected/manifest.json") == receipt["manifest_sha256"], "Interim snapshot archive/manifest changed")
    require(read(BUNDLE / "selected/manifest.json")["files"] == receipt["files"], "Interim snapshot inventory differs")
    for name, digest in receipt["files"].items(): require(sha(local_path(BUNDLE, name)) == digest, "Interim snapshot original changed")
    originals = BUNDLE / "selected/originals"
    require(sha(originals / "execution_contract.json") == TRAIN_SHA, "Snapshot training contract changed")
    capture = read(originals / "snapshot_capture.json")
    require(capture["schema"] == "titantpp_shared_inter3_interim_capture_v1" and len(capture["jobs"]) == 3
        and {(j["dataset"], j["seed"]) for j in capture["jobs"]} == {(DATASET, s) for s in (42, 52, 62)}, "Captured scope differs")
    for job in capture["jobs"]:
        require(job["checkpoint_before_sha256"] == job["checkpoint_after_sha256"] == job["checkpoint_sha256"]
            == sha(local_path(originals, job["checkpoint_path"])), "Captured checkpoint binding differs")
        require(sha(local_path(originals, job["history_path"])) == job["history_sha256"], "Captured history binding differs")
    return capture


def capture():
    """Freeze source bytes now without waiting for original training completion."""
    require(not BUNDLE.exists(), "Existing interim snapshot attempt; no overwrite/retry")
    training = _training(); jobs = _jobs(training); spec = read(SPEC)
    require(spec["schema"] == "shared_inter3_interim_test_science_v1"
        and spec["training_contract_raw_sha256"] == TRAIN_SHA, "Interim capture authorization differs")
    BUNDLE.mkdir(parents=True)
    write(BUNDLE / "snapshot_collection.intent.json", {"at": now(), "host": "5090", "selected_count": 3,
        "single_attempt": True, "training_terminal_required": False, "whole_raw9_completion_claimed": False,
        "science_spec_sha256": sha(SPEC)})
    target = BUNDLE / "selected"; target.mkdir()
    h = training["hosts"]["5090"]
    command = "cd " + shlex.quote(h["source_root"]) + " && " + shlex.join(["env", *[f"{k}={v}" for k, v in h["environment"].items()], "PYTHONDONTWRITEBYTECODE=1", h["python"], "-c", snapshot_collection_program(training, jobs)])
    result = _transport.ssh("5090", command, timeout=900)
    archive = target / "original.tar.gz"; archive.write_bytes(result.stdout)
    _transport.save_transport("5090", "interim_readonly_capture", result)
    _transport.safe_extract(archive, target / "originals")
    inventory = {str(p.relative_to(BUNDLE)): sha(p) for p in (target / "originals").rglob("*") if p.is_file()}
    write(target / "manifest.json", {"files": inventory})
    receipt = {"schema": "titantpp_shared_inter3_interim_snapshot_v1", "status": "frozen_interim_captured", "selected_fit_count": 3,
        "training_terminal_required": False, "whole_raw9_complete": False, "held_out_test_evaluated": False,
        "archive_sha256": sha(archive), "manifest_sha256": sha(target / "manifest.json"), "files": inventory}
    write(target / "receipt.json", receipt)
    captured = _verify_capture()
    return {"status": "frozen_interim_captured", "selected_epochs": {j["seed"]: j["selected_epoch"] for j in captured["jobs"]},
        "snapshot_epochs": {j["seed"]: j["snapshot_epoch"] for j in captured["jobs"]}, "archive_sha256": sha(archive)}


def _paused(contract):
    from paper.scripts import titantpp_quantity_loss_mlp18_runtime as preserved
    require(len(contract["native"]["paused_processes"]) == 4, "Expected four exact paused external identities")
    return preserved.paused_processes({"hosts": {"5080": contract["native"]}}, "5080")


def _paused_gpu_pids(contract):
    return {p["pid"] for p in _paused(contract)[1:]}


def _gpu_allowed(contract, active=()):
    allowed = _paused_gpu_pids(contract) | set(active)
    observed = _plain_gpu_snapshot(contract["native"]["gpu_uuid"])
    require(set(observed["compute_pids"]) <= allowed, "Unknown active5080 GPU owner")
    return observed


def _completed_predecessor(spec):
    if "completed_raf_taxi6" in spec: return dict(spec["completed_raf_taxi6"])
    previous = ROOT / "search_artifacts/titantpp_shared_raf_taxi6_test_20261020_v1"
    old_contract = read(previous / "evaluation_contract.json")
    original = previous / "retrieval/originals"
    receipt, exited = read(original / "pipeline_receipt.json"), read(original / "pipeline_exit.json")
    require(receipt["status"] == "complete" and receipt["actual_native_worker_exit0_count"] == 18
        and exited["actual_os_returncode"] == 0 and exited["timeout"] is False, "RAF/Taxi6 predecessor is not complete")
    return {"root": old_contract["remote_root"], "pipeline_receipt_sha256": sha(original / "pipeline_receipt.json"),
        "pipeline_exit_sha256": sha(original / "pipeline_exit.json")}


def _population_bindings(populations):
    keys = ("target_count", "target_identity_sha256", "target_quantity_sha256")
    return {split: {key: population[key] for key in keys} for split, population in populations.items()}


def prepare():
    """Prepare exactly the already captured binary snapshot, never a newer best."""
    verify_reviewed_implementation()
    require(not (BUNDLE / "evaluation_contract.json").exists() and not (BUNDLE / "preparation.intent.json").exists(), "Existing interim preparation; no retry")
    training = _training(); jobs = _jobs(training); spec = read(SPEC)
    require(spec["schema"] == "shared_inter3_interim_test_science_v1" and spec["training_contract_raw_sha256"] == TRAIN_SHA,
        "Interim science authorization changed")
    captured = _verify_capture()
    stop = spec["cap_stop"]; stop_receipt = Path(stop.get("receipt_path", stop["path"]))
    require(sha(stop_receipt) == stop.get("receipt_sha256", stop["sha256"]), "Cap5080 stop receipt changed")
    stopped = read(stop_receipt)
    require(stopped["status"] == "user_stop_verified" and stopped["actual_transport_exit"] == 0, "Cap5080 cleanup is not verified")
    for name, digest in stopped["files"].items(): require(sha(local_path(stop_receipt.parent, name)) == digest, "Cap stop original changed")
    predecessor = _completed_predecessor(spec)
    write(BUNDLE / "preparation.intent.json", {"at": now(), "captured_snapshot_reused": True, "single_attempt": True})
    operation_files = {}
    for name, digest in training["source"]["files"].items(): _copy(ROOT / name, "source/" + name, digest)
    for name in NEW_SOURCE:
        operation_files[name] = sha(ROOT / name); _copy(ROOT / name, "source/" + name, operation_files[name])
    for name, digest in PRESERVED_OPERATIONS.items():
        operation_files[name] = digest; _copy(ROOT / name, "source/" + name, digest)
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    worker = importlib.import_module(WORKER)
    bound_jobs = []; originals = BUNDLE / "selected/originals"
    for job in jobs:
        snapshot = next(x for x in captured["jobs"] if x["job_id"] == job["id"])
        entry = {**job, "selected_epoch": snapshot["selected_epoch"], "snapshot_epoch": snapshot["snapshot_epoch"], "state_sha256": snapshot["state_sha256"]}
        checkpoint, history_path = local_path(originals, snapshot["checkpoint_path"]), local_path(originals, snapshot["history_path"])
        payload = torch_load_checkpoint(checkpoint, map_location="cpu")
        definition = next(x for x in training["datasets"] if x["dataset_id"] == DATASET)
        row, proof = worker.verify_snapshot_payload(payload, entry, definition, training["source"]["files_sha256"])
        require(row == snapshot["selected_history_row"] and payload["best_state_sha256"] == snapshot["state_sha256"], "Captured selected metadata differs")
        derived = BUNDLE / "selected/derived" / job["id"]
        metadata_path, receipt_path = derived / "checkpoint_metadata.json", derived / "snapshot_receipt.json"
        write(metadata_path, worker.checkpoint_snapshot_metadata(payload))
        historical = {"split": "validation", "selected_epoch": entry["selected_epoch"], "source": "embedded_checkpoint_history",
            "metrics": {"overall": {"qty_rmse": row["val_qty_rmse"], "qty_mae": row["val_qty_mae"], "time_nll": row["val_time_nll"]}},
            "predictions_available": False}
        receipt = {"schema": "titantpp_shared_inter3_interim_snapshot_job_v1", "status": "frozen_interim_selected",
            "job_id": job["id"], "dataset": job["dataset"], "arm": job["arm"], "seed": job["seed"],
            "selected_epoch": entry["selected_epoch"], "snapshot_epoch": entry["snapshot_epoch"], "checkpoint_sha256": sha(checkpoint),
            "state_sha256": entry["state_sha256"], "source_files_sha256": training["source"]["files_sha256"],
            "target_identity": definition["inherited_data_identity"]["populations"]["validation"], "history_reference": historical,
            "original_training_terminal_required": False, "held_out_test_evaluated": False, "historical_tail_metrics_available": False,
            "historical_individual_predictions_available": False, "predictions_available": False,
            "time_head_snapshot_verification": proof, "capture_record": snapshot}
        write(receipt_path, receipt)
        for label, path in {"checkpoint": checkpoint, "selection_history": history_path,
                "checkpoint_metadata": metadata_path, "snapshot_receipt": receipt_path, "training_validation_reference": receipt_path}.items():
            entry[label] = {"path": str(path.relative_to(BUNDLE)), "sha256": sha(path)}
        bound_jobs.append(entry)
    data = {}
    relative = "inputs/data/" + DATASET + "/train_validation.parquet"
    training_input = _copy(TRAIN_BUNDLE / "input_snapshot" / relative, "source/" + relative, training["input_files"][relative])
    manifest = "inputs/data/" + DATASET + "/split_manifest.json"
    _copy(TRAIN_BUNDLE / "input_snapshot" / manifest, "source/" + manifest, training["input_files"][manifest])
    definition = spec["datasets"][DATASET]; full = definition["full_data"]; original = Path(full["path"])
    full_input = _copy(original if original.is_absolute() else ROOT / original, "inputs/" + DATASET + "_full.parquet", full["sha256"])
    data[DATASET] = {"training": training_input, "full": full_input, "populations": _population_bindings(definition["populations"])}
    native = dict(spec["native"]); native.update(root=REMOTE, source_root=REMOTE + "/source")
    native["environment"] = {**native["environment"], "PYTHONPATH": REMOTE + "/source", "MPLCONFIGDIR": REMOTE + "/cache/matplotlib", "XDG_CACHE_HOME": REMOTE + "/cache"}
    stop_binding = {**stop, **_copy(Path(stop["path"]), "references/cap_stop.json", stop["sha256"])}
    if "receipt_path" in stop:
        copied = _copy(stop_receipt, "references/cap_stop_receipt.json", stop["receipt_sha256"])
        stop_binding.update(receipt_path=copied["path"], receipt_sha256=copied["sha256"])
    c = {"schema": SCHEMA, "host": "5080", "evaluation_authorized": True, "source_root": "source", "remote_root": REMOTE,
        "training_contract": {"path": "selected/originals/execution_contract.json", "sha256": TRAIN_SHA},
        "interim_snapshot_receipt": {"path": "selected/receipt.json", "sha256": sha(BUNDLE / "selected/receipt.json")},
        "jobs": bound_jobs, "data": data, "native": native, "cap_stop": stop_binding, "completed_raf_taxi6": predecessor,
        "operation_files": operation_files, "references": _copy(REFERENCES, "references/REFERENCES.json", sha(REFERENCES)),
        "science_spec_sha256": sha(SPEC), "references_sha256": sha(REFERENCES), "child_timeout_seconds": 7200,
        "absolute_host_deadline_unix": native["absolute_host_deadline_unix"], "new_training_count": 0, "checkpoint_reselection": False,
        "reference_reexecution_count": 0, "whole_raw9_complete": False, "interim_snapshot": True,
        "original_training_terminal_required": False, "evaluation_plan": ["qualification3", "validation3", "gate3", "test3"]}
    write(BUNDLE / "evaluation_contract.json", c)
    with (BUNDLE / "deployment.tar.gz").open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as archive_out:
        for path in sorted(BUNDLE.rglob("*")):
            if path.is_file() and path.name != "deployment.tar.gz": archive_out.add(path, arcname=str(path.relative_to(BUNDLE)))
        marker = tarfile.TarInfo("source/sample_data"); marker.type, marker.mode = tarfile.DIRTYPE, 0o755; archive_out.addfile(marker)
    result = {"status": "prepared", "contract_sha256": sha(BUNDLE / "evaluation_contract.json"), "canonical_contract_sha256": canonical(c),
        "archive_sha256": sha(BUNDLE / "deployment.tar.gz"), "candidate_count": 3, "baseline_reference_count": 42,
        "actual_native_evaluation_count": 0, "interim_snapshot": True}
    write(REPORT / "PREPARED.json", result)
    return result


_prepared_verify_local = verify_local

def verify_local(*, archive=False):
    c = _prepared_verify_local(archive=archive)
    require(c["schema"] == SCHEMA and c["host"] == "5080" and c["remote_root"] == REMOTE
        and c["interim_snapshot"] is True and c["original_training_terminal_required"] is False
        and c["checkpoint_reselection"] is False and c["new_training_count"] == 0, "Interim frozen evaluation scope changed")
    _verify_capture()
    for job in c["jobs"]:
        for label in ("checkpoint", "selection_history", "checkpoint_metadata", "snapshot_receipt", "training_validation_reference"):
            require(sha(local_path(BUNDLE, job[label]["path"])) == job[label]["sha256"], "Frozen job artifact changed")
    return c


# The preserved queue owns only its new children and retains their actual exits.
import inspect as _inspect
_stage_source = _inspect.getsource(_run_remote_stage)
_stage_source = _stage_source.replace('gpu = gpu_snapshot(native["gpu_uuid"])', 'gpu = _gpu_allowed(contract, (r["process"].pid for r in active.values()))')
_stage_source = _stage_source.replace('set(gpu["compute_pids"]) <= {r["process"].pid for r in active.values()}',
    'set(gpu["compute_pids"]) <= _paused_gpu_pids(contract) | {r["process"].pid for r in active.values()}')
exec(compile(_stage_source, str(_FROZEN_PATH), "exec"), globals())


def _verify_completed_remote(c):
    previous = c["completed_raf_taxi6"]; root = Path(previous["root"])
    require(sha(root / "pipeline_receipt.json") == previous["pipeline_receipt_sha256"]
        and sha(root / "pipeline_exit.json") == previous["pipeline_exit_sha256"], "Completed RAF/Taxi6 terminal originals changed")
    receipt, exited = read(root / "pipeline_receipt.json"), read(root / "pipeline_exit.json")
    require(receipt["status"] == "complete" and receipt["actual_native_worker_exit0_count"] == 18
        and exited["actual_os_returncode"] == 0 and exited["timeout"] is False, "RAF/Taxi6 predecessor terminal state differs")
    pids = [exited["pid"]] + [e["pid"] for stage in ("qualification", "validation", "test") for e in receipt[stage].values()]
    require(all(_owned_pid_absent(pid) for pid in pids), "Completed RAF/Taxi6 worker remains")
    return pids


def remote_run(root):
    root = Path(root).resolve(); c = read(root / "evaluation_contract.json")
    require(c["schema"] == SCHEMA and c["remote_root"] == str(root), "Evaluation scope/root differs")
    require(not any((root / name).exists() for name in ("pipeline.intent.json", "pipeline_receipt.json", "pipeline_failure.json", "validation_gate.json", "results", "qualification")), "Existing evaluation attempt; no retry")
    write(root / "pipeline.intent.json", {"pid": os.getpid(), "at": now(), "contract_sha256": canonical(c)})
    def interrupted(signum, frame): raise KeyboardInterrupt("Interim evaluation supervisor interrupted")
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    _gpu_allowed(c); _verify_completed_remote(c)
    require(all(_owned_pid_absent(pid) for pid in c["cap_stop"].get("owned_pids", [])), "A stopped Cap-owned process remains")
    qualification = run_remote_stage(root, c, "qualification")
    validation = run_remote_stage(root, c, "validation")
    gate = {"schema": "titantpp_shared_inter3_interim_validation_gate_v1", "status": "PASS", "contract_sha256": canonical(c), "job_count": 3,
        "validation_receipts": {j["id"]: sha(_receipt_path(root, j, "validation")) for j in c["jobs"]},
        "validation_worker_exits": {j["id"]: sha(root / "results" / j["id"] / "validation_exit.json") for j in c["jobs"]}}
    write(root / "validation_gate.json", gate)
    importlib.import_module(WORKER).verify_validation_gate(root, c)
    tested = run_remote_stage(root, c, "test")
    pids = [e["pid"] for stage in (qualification, validation, tested) for e in stage.values()]
    gpu = _gpu_allowed(c)
    require(all(_owned_pid_absent(pid) for pid in pids) and not set(pids).intersection(gpu["compute_pids"]), "Evaluation owner remains")
    files = {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()
        and p.relative_to(root).parts[0] not in ("source", "cache") and p.name != "pipeline.log"}
    result = {"status": "complete", "contract_sha256": canonical(c), "qualification": qualification, "validation": validation, "test": tested,
        "qualification_exit0_count": 3, "actual_data_worker_exit0_count": 6, "actual_native_worker_exit0_count": 9,
        "owned_pids_absent": True, "owned_gpu_pids_absent": True, "gpu_compute_pids": gpu["compute_pids"],
        "paused_external_preserved": True, "whole_raw9_complete": False, "interim_snapshot": True, "files": files}
    write(root / "remote_final_receipt.json", result); write(root / "pipeline_receipt.json", result)
    return result


def fresh_preflight_program(c):
    h = c["native"]
    return ("import pathlib,json,hashlib,subprocess\n"
        f"assert not pathlib.Path({REMOTE!r}).exists(),'Existing new evaluation root'\n"
        f"expected={h['paused_processes']!r}\n"
        "def ident(pid):\n p=pathlib.Path('/proc')/str(pid);s=(p/'stat').read_text();tail=s[s.rfind(')')+2:].split();return {'pid':pid,'state':tail[0],'start_ticks':int(tail[19]),'pgid':int(tail[2]),'argv':(p/'cmdline').read_bytes().decode().rstrip(chr(0)).split(chr(0)),'cwd':str((p/'cwd').resolve())}\n"
        "for p in expected:\n live=ident(p['pid']);assert live['state'] in ('T','t') and all(live[k]==p[k] for k in ('pid','start_ticks','pgid','argv','cwd')),'Paused identity changed'\n"
        f"previous={c['completed_raf_taxi6']!r};r=pathlib.Path(previous['root'])\n"
        "for name,key in [('pipeline_receipt.json','pipeline_receipt_sha256'),('pipeline_exit.json','pipeline_exit_sha256')]:assert hashlib.sha256((r/name).read_bytes()).hexdigest()==previous[key],'RAF/Taxi6 predecessor original changed'\n"
        "f=json.loads((r/'pipeline_receipt.json').read_text());x=json.loads((r/'pipeline_exit.json').read_text())\n"
        "assert f['status']=='complete' and f['actual_native_worker_exit0_count']==18 and x['actual_os_returncode']==0 and not x['timeout']\n"
        "pids=[x['pid']]+[e['pid'] for stage in ('qualification','validation','test') for e in f[stage].values()]\n"
        "assert all(not (pathlib.Path('/proc')/str(pid)).exists() for pid in pids),'RAF/Taxi6 worker remains'\n"
        f"assert all(not (pathlib.Path('/proc')/str(pid)).exists() for pid in {c['cap_stop'].get('owned_pids',[])!r}),'Cap owner remains'\n"
        "x=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=20)\n"
        f"assert x.stdout.strip()=={h['gpu_uuid']!r},'GPU UUID changed'\n"
        "x=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=20)\n"
        f"allowed={set(p['pid'] for p in h['paused_processes'][1:])!r}\n"
        "actual={int(v.strip()) for v in x.stdout.splitlines() if v.strip()};assert actual<=allowed,'Unknown GPU owner'\n"
        "print(json.dumps({'status':'passed','compute_pids':sorted(actual),'paused_external_preserved':True,'predecessor_workers_absent':True}))\n")


def advance():
    c = verify_local(archive=True)
    require(time.time() < c["absolute_host_deadline_unix"], "Inherited evaluation deadline expired")
    write(BUNDLE / "remote.intent.json", {"at": now(), "automatic_retry": False})
    h = c["native"]
    _transport.save_transport("5080", "fresh_idle", _transport.ssh("5080", shlex.join([h["python"], "-c", fresh_preflight_program(c)])))
    _transport.save_transport("5080", "install", _transport.ssh("5080", "mkdir " + shlex.quote(REMOTE) + " && tar -xzf - -C " + shlex.quote(REMOTE), stdin=(BUNDLE / "deployment.tar.gz").read_bytes(), timeout=300))
    command = [h["python"], "-u", "-m", MODULE, "--mode", "remote-run", "--root", REMOTE]
    supervisor = ("import os,subprocess,pathlib,json,sys,time,signal\n" + f"p=subprocess.Popen({command!r},cwd={REMOTE + '/source'!r},env={{**os.environ,**{h['environment']!r}}},start_new_session=True)\n"
        f"limit=max(.01,min(21600.,{c['absolute_host_deadline_unix']!r}-time.time()));timed_out=False\n"
        "try:code=p.wait(timeout=limit)\nexcept subprocess.TimeoutExpired:\n timed_out=True;os.killpg(p.pid,signal.SIGTERM)\n try:code=p.wait(timeout=60)\n except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()\n"
        f"path=pathlib.Path({REMOTE!r})/'pipeline_exit.json'\n"
        "path.open('x').write(json.dumps({'pid':p.pid,'actual_os_returncode':code,'timeout':timed_out,'actual_child_exit_observed':True,'supervisor_own_exit_directly_observed':False})+chr(10))\nsys.exit(code)\n")
    _transport.save_transport("5080", "evaluation", _transport.ssh("5080", shlex.join([h["python"], "-u", "-c", supervisor]),
        timeout=max(1., min(21660., c["absolute_host_deadline_unix"] - time.time() + 60))))
    return collect_and_compare(c)


def collect_and_compare(c):
    destination = BUNDLE / "retrieval"; destination.mkdir(exist_ok=False)
    program = ("import pathlib,json,tarfile,sys\nfrom paper.scripts import manage_titantpp_shared_inter3_interim_test as m\n"
        f"root=pathlib.Path({REMOTE!r});c=m.read(root/'evaluation_contract.json')\n"
        "f=m.read(root/'pipeline_receipt.json');x=m.read(root/'pipeline_exit.json')\n"
        "assert f['status']=='complete' and f['actual_native_worker_exit0_count']==9 and f['actual_data_worker_exit0_count']==6 and x['actual_os_returncode']==0 and not x['timeout']\n"
        "pids=[x['pid']]+[e['pid'] for stage in ('qualification','validation','test') for e in f[stage].values()]\n"
        "assert all(m._owned_pid_absent(pid) for pid in pids);gpu=m._gpu_allowed(c);assert not set(pids).intersection(gpu['compute_pids'])\n"
        "with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:\n"
        " for path in sorted(root.rglob('*')):\n  rel=path.relative_to(root)\n  if rel.parts[0] in ('source','cache'):continue\n  assert not path.is_symlink()\n  if path.is_file():archive.add(path,arcname=str(rel),recursive=False)\n")
    command = "cd " + shlex.quote(REMOTE + "/source") + " && " + shlex.join(["env", *[f"{k}={v}" for k,v in c["native"]["environment"].items()], c["native"]["python"], "-c", program])
    result = _transport.ssh("5080", command, timeout=900)
    archive = destination / "original.tar.gz"; archive.write_bytes(result.stdout)
    _transport.save_transport("5080", "collection", result); _transport.safe_extract(archive, destination / "originals")
    originals = destination / "originals"; final, exited = read(originals / "pipeline_receipt.json"), read(originals / "pipeline_exit.json")
    require(final["status"] == "complete" and final["contract_sha256"] == canonical(c) and final["actual_native_worker_exit0_count"] == 9
        and final["actual_data_worker_exit0_count"] == 6 and final["qualification_exit0_count"] == 3 and final["owned_pids_absent"] is True
        and final["owned_gpu_pids_absent"] is True and exited["actual_os_returncode"] == 0 and exited["timeout"] is False and final["files"], "Incomplete interim-model evaluation")
    for name, digest in final["files"].items(): require(sha(local_path(originals, name)) == digest, "Evaluation original changed")
    manifest = {str(p.relative_to(originals)): sha(p) for p in originals.rglob("*") if p.is_file()}
    write(destination / "manifest.json", {"files": manifest})
    write(destination / "receipt.json", {"status": "verified", "archive_sha256": sha(archive), "manifest_sha256": sha(destination / "manifest.json"),
        "file_count": len(manifest), "candidate_test_count": 3, "validation_replay_count": 3, "qualification_exit0_count": 3,
        "actual_data_worker_exit0_count": 6, "actual_native_worker_exit0_count": 9, "actual_pipeline_exit0_count": 1,
        "owner_absence_verified": True, "paused_external_preserved": True, "whole_raw9_complete": False, "interim_snapshot": True})
    importlib.import_module("paper.scripts.compare_titantpp_shared_inter3_interim_test").main(["--bundle", str(BUNDLE)])
    return {"status": "complete", "final_receipt": str(BUNDLE / "final_receipt.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("capture", "prepare", "verify", "advance", "remote-run"), required=True)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.mode == "capture": result = capture()
        else:
            methods = {"prepare": prepare, "verify": lambda: verify_local(archive=True), "advance": advance,
                "remote-run": lambda: remote_run(args.root)}
            result = methods[args.mode]()
    except Exception as error:
        target = args.root if args.mode == "remote-run" else REPORT
        if target is not None and not (Path(target) / "FAILURE.json").exists():
            write(Path(target) / "FAILURE.json", {"status": "failed", "at": now(), "error": str(error), "automatic_retry": False})
        raise
    print(json.dumps(result, ensure_ascii=False, allow_nan=False)); return 0

if __name__ == "__main__": raise SystemExit(main())
