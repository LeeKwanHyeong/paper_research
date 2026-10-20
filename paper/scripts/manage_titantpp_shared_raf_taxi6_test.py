"""Read-only completed-fit subset, then frozen RAF/Taxi Test6 on idle5080.

The original raw9 Intermittent fits and their finisher remain untouched. Only
six completed train children are collected, never represented as raw9 complete.
"""
from pathlib import Path as _Path
import hashlib as _hashlib

_FROZEN_PATH = _Path(__file__).with_name("manage_titantpp_shared_raw_raf3_test.py")
_FROZEN_SHA = "c516112d606732a3093953c387b14c29ab347b565c531fd8c7a629932b33397a"
_original = _FROZEN_PATH.read_text()
if _hashlib.sha256(_original.encode()).hexdigest() != _FROZEN_SHA:
    raise ValueError("Preserved RAF3 manager changed")
_adapted = _original.replace("shared_raw_raf3", "shared_raf_taxi6")
_adapted = _adapted.replace('"5090"', '"5080"')
_adapted = _adapted.replace('len(exits) == 3', 'len(exits) == 6')
_adapted = _adapted.replace('"job_count": 3', '"job_count": 6')
_adapted = _adapted.replace('"qualification_exit0_count": 3', '"qualification_exit0_count": 6')
_adapted = _adapted.replace('"actual_data_worker_exit0_count": 6', '"actual_data_worker_exit0_count": 12')
_adapted = _adapted.replace('"actual_native_worker_exit0_count": 9', '"actual_native_worker_exit0_count": 18')
_adapted = _adapted.replace('if __name__ == "__main__": raise SystemExit(main())', '')
exec(compile(_adapted, str(_FROZEN_PATH), "exec"), globals())
__doc__ = "Read-only collection of six completed RAF/Taxi fits, then frozen5080 Validation6/Test6; no raw9-wide completion wait."

GPU_UUID = "GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6"
REMOTE = "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/" + NAME + "_5080"
PRESERVED_OPERATIONS = {
    "paper/scripts/manage_titantpp_shared_raw_raf3_test.py": _FROZEN_SHA,
    "paper/scripts/evaluate_titantpp_shared_raw_raf3_frozen.py": "2fdd0f56757c4330e3e140ca800bbfbb5d13060f8b663d7156a1e40a7e3225c5",
    "paper/scripts/compare_titantpp_shared_raw_raf3_test.py": "5469ceb02be20d2559c51232a7b63dd5917540d5bc317e2903d0ef29141664dd",
}
_transport.BUNDLE, _transport.REPORT = BUNDLE, REPORT
_plain_gpu_snapshot = gpu_snapshot


def _paused(contract):
    from paper.scripts import titantpp_quantity_loss_mlp18_runtime as preserved
    expected = contract["native"]["paused_processes"]
    require(len(expected) == 4, "Expected four unchanged paused external-model identities")
    return preserved.paused_processes({"hosts": {"5080": contract["native"]}}, "5080")


def _paused_gpu_pids(contract):
    return {record["pid"] for record in _paused(contract)[1:]}


def _gpu_allowed(contract, active=()):
    paused = _paused_gpu_pids(contract)
    observed = _plain_gpu_snapshot(contract["native"]["gpu_uuid"])
    require(set(observed["compute_pids"]) <= paused | set(active), "Unknown active5080 GPU owner")
    return observed


def verify_selected_subset():
    require(sha(TRAIN_BUNDLE / "execution_contract.json") == TRAIN_SHA, "Original raw9 source contract changed")
    c = verify_local(archive=False)
    receipt_path = local_path(BUNDLE, c["selected_training_receipt"]["path"])
    require(sha(receipt_path) == c["selected_training_receipt"]["sha256"], "Selected subset receipt changed")
    receipt = read(receipt_path)
    require(receipt["schema"] == "titantpp_shared_raf_taxi6_selected_training_v1" and receipt["status"] == "complete_selected_subset"
        and receipt["selected_fit_count"] == receipt["training_validation_reference_count"] == 6
        and receipt["actual_scientific_worker_exit0_count"] == 6 and receipt["all_selected_training_terminal_verified"] is True
        and receipt["whole_raw9_complete"] is False and receipt["held_out_test_evaluated"] is False,
        "Completed six-fit subset evidence missing")
    require(receipt["files"] and sha(BUNDLE / "selected/original.tar.gz") == receipt["archive_sha256"]
        and sha(BUNDLE / "selected/manifest.json") == receipt["manifest_sha256"], "Selected subset archive/manifest changed")
    require(read(BUNDLE / "selected/manifest.json")["files"] == receipt["files"], "Selected inventory differs")
    for name, digest in receipt["files"].items(): require(sha(local_path(BUNDLE, name)) == digest, "Selected subset original changed")
    return c


def subset_collection_program(training, jobs):
    h = training["hosts"]["5090"]
    # Tensor hashes are computed on CPU; no model inference, optimizer or CUDA
    # context is created on the concurrently training5090 host.
    return f'''import io,json,pathlib,sys,tarfile,hashlib
from paper.scripts import titantpp_shared_history_raw9_5090_runtime as r
from simple_lab_test.search.common.runner import torch_load_checkpoint,canonical_state_dict_sha256
from paper.scripts import run_titantpp_shared_history_raw9_5090 as scientific
root=pathlib.Path({h['root']!r});cp=root/'execution_contract.json'
r.require(r.sha_file(cp)=={TRAIN_SHA!r},'Raw9 contract changed');c=r.read(cp)
jobs={jobs!r};files={{'execution_contract.json':r.sha_file(cp)}};derived={{}}
for job in jobs:
 folder=root/'run'/job['id'];receipt=r.read(folder/'train_receipt.json')
 r.verify_receipt(c,'5090',job,'train')
 launch_path=folder/'train_launch.json';exit_path=folder/'train_exit.json'
 launch=r.read(launch_path);actual=r.read(exit_path)
 r.require(actual['actual_os_returncode']==0 and actual['timeout'] is False,'Selected worker not successful')
 r.require(all(actual[k]==launch[k] for k in ('job_id','stage','pid','start_ticks')) and actual['job_id']==job['id'] and actual['stage']=='train','Selected launch/exit identity differs')
 r.require(r.proc_identity(actual['pid']) is None,'Selected worker still exists')
 command=launch['command'];r.require('--job' in command and command[command.index('--job')+1]==job['id'] and command[-2:]==['--stage','train'],'Selected launch differs')
 selected=pathlib.Path(receipt['selected_checkpoint']);payload=torch_load_checkpoint(selected,map_location='cpu')
 r.require(canonical_state_dict_sha256(payload['model_state_dict'])==payload['model_state_sha256']==receipt['state_sha256'],'Selected tensor SHA differs')
 r.require(scientific.prepared.select_best_epoch(r.read(selected.parent/'history.json')['history'])==receipt['selected_epoch'],'Selection history differs')
 r.require(payload['best_epoch']==receipt['selected_epoch'] and payload['seed']==job['seed'],'Selected epoch/seed differs')
 for name,digest in receipt['files'].items():files[name]=digest
 for path in (folder/'train_receipt.json',launch_path,exit_path):files[str(path.relative_to(root))]=r.sha_file(path)
 binding={{'job_id':job['id'],'stage':'train','command':command,'pid':actual['pid'],'actual_os_returncode':0,'timeout':False,'actual_child_exit_observed':True,
 'original_exit':{{'path':str(exit_path.relative_to(root)),'sha256':r.sha_file(exit_path)}},'original_launch':{{'path':str(launch_path.relative_to(root)),'sha256':r.sha_file(launch_path)}}}}
 derived['verified_exits/'+job['id']+'.json']=(json.dumps(binding,ensure_ascii=False,indent=2)+chr(10)).encode()
for name,digest in files.items():r.require(r.sha_file(r.local_path(root,name))==digest,'Subset original changed')
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for name in sorted(files):archive.add(r.local_path(root,name),arcname=name,recursive=False)
 for name,blob in derived.items():
  info=tarfile.TarInfo(name);info.size=len(blob);archive.addfile(info,io.BytesIO(blob))
'''


def prepare():
    verify_reviewed_implementation()
    require(not BUNDLE.exists(), "Existing six-fit evaluation bundle; no retry")
    from paper.scripts import manage_titantpp_shared_history_raw9_5090 as parent
    training = parent.verify_local()
    require(sha(TRAIN_BUNDLE / "execution_contract.json") == TRAIN_SHA, "Frozen raw9 contract changed")
    spec, refs = read(SPEC), read(REFERENCES)
    require(len(refs["references"]) == 42, "Expected42 stored dataset/model/seed Test references")
    stop = spec["cap_stop"]
    require(sha(stop["path"]) == stop["sha256"], "5080 stop/cleanup evidence changed")
    stop_receipt = Path(stop.get("receipt_path", stop["path"]))
    require(sha(stop_receipt) == stop.get("receipt_sha256", stop["sha256"]), "Cap5080 stop receipt changed")
    stopped = read(stop_receipt)
    require(stopped["status"] == "user_stop_verified" and stopped["actual_transport_exit"] == 0,
        "Cap5080 stop/cleanup is not verified")
    for name, digest in stopped["files"].items():
        require(sha(local_path(stop_receipt.parent, name)) == digest, "Cap5080 stop original changed")
    tracked = set(subprocess.check_output(["git", "ls-files", "--", *NEW_SOURCE], cwd=ROOT, text=True).splitlines())
    require(tracked == set(NEW_SOURCE) and not subprocess.check_output(["git", "diff", "--name-only", "HEAD", "--", *NEW_SOURCE], cwd=ROOT, text=True).strip(), "Commit new isolated operations before prepare")
    jobs = [job for job in training["jobs"] if job["dataset"] in ("raf_spare_parts", "yellow_trip_hourly")]
    require(len(jobs) == 6 and {(j["dataset"], j["seed"]) for j in jobs} == {(d, s) for d in ("raf_spare_parts", "yellow_trip_hourly") for s in (42, 52, 62)}, "Selected subset matrix differs")
    BUNDLE.mkdir(parents=True)
    write(BUNDLE / "selected_collection.intent.json", {"at": now(), "selected_count": 6, "whole_raw9_completion_claimed": False})
    collection_root = BUNDLE / "selected"; collection_root.mkdir()
    h = training["hosts"]["5090"]
    command = "cd " + shlex.quote(h["source_root"]) + " && " + shlex.join([h["python"], "-c", subset_collection_program(training, jobs)])
    result = _transport.ssh("5090", command, timeout=900)
    archive = collection_root / "original.tar.gz"; archive.write_bytes(result.stdout)
    _transport.save_transport("5090", "selected_readonly_collection", result)
    originals = collection_root / "originals"; _transport.safe_extract(archive, originals)
    inventory = {str(path.relative_to(BUNDLE)): sha(path) for path in originals.rglob("*") if path.is_file()}
    write(collection_root / "manifest.json", {"files": inventory})
    subset = {"schema": "titantpp_shared_raf_taxi6_selected_training_v1", "status": "complete_selected_subset", "selected_fit_count": 6,
        "training_validation_reference_count": 6, "actual_scientific_worker_exit0_count": 6, "all_selected_training_terminal_verified": True,
        "whole_raw9_complete": False, "held_out_test_evaluated": False, "archive_sha256": sha(archive), "manifest_sha256": sha(collection_root / "manifest.json"), "files": inventory}
    write(collection_root / "receipt.json", subset)
    operation_files = {}
    for name, digest in training["source"]["files"].items(): _copy(ROOT / name, "source/" + name, digest)
    for name in NEW_SOURCE:
        operation_files[name] = sha(ROOT / name); _copy(ROOT / name, "source/" + name, operation_files[name])
    for name, digest in PRESERVED_OPERATIONS.items():
        operation_files[name] = digest; _copy(ROOT / name, "source/" + name, digest)
    bound_jobs = []
    for job in jobs:
        folder = originals / "run" / job["id"]; receipt = read(folder / "train_receipt.json")
        expected = next(item for item in spec["selected_checkpoints"] if item["id"] == job["id"])
        require(receipt["selected_epoch"] == expected["selected_epoch"] and receipt["checkpoint_sha256"] == expected["checkpoint_sha256"]
            and receipt["state_sha256"] == expected["state_sha256"], "Approved selected checkpoint changed")
        selected = local_path(originals, str(Path(receipt["selected_checkpoint"]).relative_to(h["root"])))
        prediction = local_path(originals, str(Path(receipt["validation_reference"]["prediction_path"]).relative_to(h["root"])))
        entry = {**job, "selected_epoch": expected["selected_epoch"]}
        bindings = {"train_receipt": folder / "train_receipt.json", "training_validation_reference": folder / "train_receipt.json",
            "train_exit": originals / "verified_exits" / (job["id"] + ".json"), "checkpoint": selected,
            "selection_history": selected.parent / "history.json", "checkpoint_metadata": selected.parent / "summary.json", "validation_predictions": prediction}
        for label, path in bindings.items(): entry[label] = {"path": str(path.relative_to(BUNDLE)), "sha256": sha(path)}
        bound_jobs.append(entry)
    data = {}
    for dataset in ("raf_spare_parts", "yellow_trip_hourly"):
        relative = "inputs/data/" + dataset + "/train_validation.parquet"
        training_input = _copy(TRAIN_BUNDLE / "input_snapshot" / relative, "source/" + relative, training["input_files"][relative])
        manifest = "inputs/data/" + dataset + "/split_manifest.json"
        _copy(TRAIN_BUNDLE / "input_snapshot" / manifest, "source/" + manifest, training["input_files"][manifest])
        definition = spec["datasets"][dataset]
        full = definition["full_data"]; original = Path(full["path"])
        full_input = _copy(original if original.is_absolute() else ROOT / original, "inputs/" + dataset + "_full.parquet", full["sha256"])
        data[dataset] = {"training": training_input, "full": full_input, "populations": definition["populations"]}
    native = dict(spec["native"])
    native.update(root=REMOTE, source_root=REMOTE + "/source")
    native["environment"] = {**native["environment"], "PYTHONPATH": REMOTE + "/source", "MPLCONFIGDIR": REMOTE + "/cache/matplotlib", "XDG_CACHE_HOME": REMOTE + "/cache"}
    stop_binding = {**stop, **_copy(Path(stop["path"]), "references/cap_stop.json", stop["sha256"])}
    if "receipt_path" in stop:
        copied_stop_receipt = _copy(stop_receipt, "references/cap_stop_receipt.json", stop["receipt_sha256"])
        stop_binding.update(receipt_path=copied_stop_receipt["path"], receipt_sha256=copied_stop_receipt["sha256"])
    c = {"schema": "titantpp_shared_raf_taxi6_frozen_v1", "host": "5080", "evaluation_authorized": True, "source_root": "source",
        "remote_root": REMOTE, "training_contract": {"path": "selected/originals/execution_contract.json", "sha256": TRAIN_SHA},
        "selected_training_receipt": {"path": "selected/receipt.json", "sha256": sha(collection_root / "receipt.json")},
        "jobs": bound_jobs, "data": data, "native": native, "cap_stop": stop_binding,
        "operation_files": operation_files, "references": _copy(REFERENCES, "references/REFERENCES.json", sha(REFERENCES)),
        "science_spec_sha256": sha(SPEC), "references_sha256": sha(REFERENCES), "child_timeout_seconds": 7200,
        "absolute_host_deadline_unix": native["absolute_host_deadline_unix"], "new_training_count": 0,
        "checkpoint_reselection": False, "reference_reexecution_count": 0, "whole_raw9_complete": False}
    write(BUNDLE / "evaluation_contract.json", c)
    with (BUNDLE / "deployment.tar.gz").open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as archive_out:
        for path in sorted(BUNDLE.rglob("*")):
            if path.is_file() and path.name != "deployment.tar.gz": archive_out.add(path, arcname=str(path.relative_to(BUNDLE)))
        marker = tarfile.TarInfo("source/sample_data"); marker.type, marker.mode = tarfile.DIRTYPE, 0o755; archive_out.addfile(marker)
    result = {"status": "prepared", "contract_sha256": sha(BUNDLE / "evaluation_contract.json"), "canonical_contract_sha256": canonical(c),
        "archive_sha256": sha(BUNDLE / "deployment.tar.gz"), "candidate_count": 6, "baseline_reference_count": 42, "actual_native_evaluation_count": 0}
    write(REPORT / "PREPARED.json", result)
    return result


# Reuse the bounded child queue unchanged except for the explicit allowlist of
# still-stopped external-model CUDA contexts. Raw GPU observations are retained.
import inspect as _inspect
_stage_source = _inspect.getsource(_run_remote_stage)
require(_stage_source.count('len(exits) == 3') == 1, "Preserved stage terminal count changed")
_stage_source = _stage_source.replace('len(exits) == 3', 'len(exits) == 6')
require(_stage_source.count('set(gpu["compute_pids"]) <= {r["process"].pid for r in active.values()}') == 1,
    "Preserved stage GPU admission changed")
_stage_source = _stage_source.replace('set(gpu["compute_pids"]) <= {r["process"].pid for r in active.values()}',
    'set(gpu["compute_pids"]) <= _paused_gpu_pids(contract) | {r["process"].pid for r in active.values()}')
exec(compile(_stage_source, str(_FROZEN_PATH), "exec"), globals())


def remote_run(root):
    root = Path(root).resolve(); c = read(root / "evaluation_contract.json")
    require(c["schema"] == "titantpp_shared_raf_taxi6_frozen_v1" and c["remote_root"] == str(root), "Evaluation scope/root differs")
    require(not any((root / name).exists() for name in ("pipeline.intent.json", "pipeline_receipt.json", "pipeline_failure.json", "validation_gate.json", "results", "qualification")), "Existing evaluation attempt; no retry")
    write(root / "pipeline.intent.json", {"pid": os.getpid(), "at": now(), "contract_sha256": canonical(c)})
    def interrupted(signum, frame): raise KeyboardInterrupt("New evaluation supervisor interrupted")
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    _gpu_allowed(c)
    stopped_cap_pids = c["cap_stop"].get("owned_pids", [])
    require(all(_owned_pid_absent(pid) for pid in stopped_cap_pids), "A stopped Cap-owned process remains")
    qualification = run_remote_stage(root, c, "qualification")
    validation = run_remote_stage(root, c, "validation")
    gate = {"schema": "titantpp_shared_raf_taxi6_validation_gate_v1", "status": "PASS", "contract_sha256": canonical(c), "job_count": 6,
        "validation_receipts": {job["id"]: sha(_receipt_path(root, job, "validation")) for job in c["jobs"]},
        "validation_worker_exits": {job["id"]: sha(root / "results" / job["id"] / "validation_exit.json") for job in c["jobs"]}}
    write(root / "validation_gate.json", gate)
    importlib.import_module(WORKER).verify_validation_gate(root, c)
    tested = run_remote_stage(root, c, "test")
    pids = [record["pid"] for stage in (qualification, validation, tested) for record in stage.values()]
    gpu = _gpu_allowed(c)
    require(all(_owned_pid_absent(pid) for pid in pids) and not set(pids).intersection(gpu["compute_pids"]), "Evaluation owner remains")
    files = {str(path.relative_to(root)): sha(path) for path in root.rglob("*") if path.is_file()
        and path.relative_to(root).parts[0] not in ("source", "cache") and path.name != "pipeline.log"}
    result = {"status": "complete", "contract_sha256": canonical(c), "qualification": qualification, "validation": validation, "test": tested,
        "qualification_exit0_count": 6, "actual_data_worker_exit0_count": 12, "actual_native_worker_exit0_count": 18,
        "owned_pids_absent": True, "owned_gpu_pids_absent": True, "gpu_compute_pids": gpu["compute_pids"],
        "paused_external_preserved": True, "whole_raw9_complete": False, "files": files}
    write(root / "remote_final_receipt.json", result); write(root / "pipeline_receipt.json", result)
    return result


def advance():
    c = verify_selected_subset(); verify_local(archive=True)
    write(BUNDLE / "remote.intent.json", {"at": now(), "automatic_retry": False})
    h = c["native"]
    # The new remote supervisor verifies all four exact stopped identities and
    # their descendant inventory before starting any qualification child.
    fresh = ("import pathlib,subprocess,json\n" + f"assert not pathlib.Path({REMOTE!r}).exists()\n"
        "x=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=20)\n"
        f"allowed={set(p['pid'] for p in h['paused_processes'][1:])!r}\n"
        "assert {int(v.strip()) for v in x.stdout.splitlines() if v.strip()}<=allowed,'Unknown GPU owner'\n")
    _transport.save_transport("5080", "fresh_idle", _transport.ssh("5080", shlex.join([h["python"], "-c", fresh])))
    _transport.save_transport("5080", "install", _transport.ssh("5080", "mkdir " + shlex.quote(REMOTE) + " && tar -xzf - -C " + shlex.quote(REMOTE), stdin=(BUNDLE / "deployment.tar.gz").read_bytes(), timeout=300))
    command = [h["python"], "-u", "-m", MODULE, "--mode", "remote-run", "--root", REMOTE]
    supervisor = ("import os,subprocess,pathlib,json,sys,time,signal\n" + f"p=subprocess.Popen({command!r},cwd={REMOTE + '/source'!r},env={{**os.environ,**{h['environment']!r}}},start_new_session=True)\n"
        f"limit=max(.01,min(21600.,{c['absolute_host_deadline_unix']!r}-time.time()));timed_out=False\n"
        "try:code=p.wait(timeout=limit)\nexcept subprocess.TimeoutExpired:\n timed_out=True;os.killpg(p.pid,signal.SIGTERM)\n try:code=p.wait(timeout=60)\n except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()\n"
        f"path=pathlib.Path({REMOTE!r})/'pipeline_exit.json'\n"
        "path.open('x').write(json.dumps({'pid':p.pid,'actual_os_returncode':code,'timeout':timed_out,'actual_child_exit_observed':True,'supervisor_own_exit_directly_observed':False})+chr(10))\nsys.exit(code)\n")
    _transport.save_transport("5080", "evaluation", _transport.ssh("5080", shlex.join([h["python"], "-u", "-c", supervisor]), timeout=max(1., min(21660., c["absolute_host_deadline_unix"] - time.time() + 60))))
    return collect_and_compare(c)


def collect_and_compare(c):
    destination = BUNDLE / "retrieval"; destination.mkdir(exist_ok=False)
    program = ("import pathlib,json,tarfile,sys\nfrom paper.scripts import manage_titantpp_shared_raf_taxi6_test as m\n"
        f"root=pathlib.Path({REMOTE!r});c=m.read(root/'evaluation_contract.json')\n"
        "f=m.read(root/'pipeline_receipt.json');x=m.read(root/'pipeline_exit.json')\n"
        "assert f['status']=='complete' and f['actual_native_worker_exit0_count']==18 and f['actual_data_worker_exit0_count']==12 and x['actual_os_returncode']==0 and not x['timeout']\n"
        "pids=[x['pid']]+[e['pid'] for stage in ('qualification','validation','test') for e in f[stage].values()]\n"
        "assert all(m._owned_pid_absent(pid) for pid in pids);gpu=m._gpu_allowed(c);assert not set(pids).intersection(gpu['compute_pids'])\n"
        "with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:\n"
        " for path in sorted(root.rglob('*')):\n  rel=path.relative_to(root)\n  if rel.parts[0] in ('source','cache'):continue\n  assert not path.is_symlink()\n  if path.is_file():archive.add(path,arcname=str(rel),recursive=False)\n")
    env_command = "cd " + shlex.quote(REMOTE + "/source") + " && " + shlex.join([c["native"]["python"], "-c", program])
    result = _transport.ssh("5080", env_command, timeout=900)
    archive = destination / "original.tar.gz"; archive.write_bytes(result.stdout)
    _transport.save_transport("5080", "collection", result); _transport.safe_extract(archive, destination / "originals")
    originals = destination / "originals"; final, exited = read(originals / "pipeline_receipt.json"), read(originals / "pipeline_exit.json")
    require(final["status"] == "complete" and final["contract_sha256"] == canonical(c) and final["actual_native_worker_exit0_count"] == 18
        and final["actual_data_worker_exit0_count"] == 12 and final["qualification_exit0_count"] == 6 and final["owned_pids_absent"] is True
        and final["owned_gpu_pids_absent"] is True and exited["actual_os_returncode"] == 0 and exited["timeout"] is False and final["files"], "Incomplete selected-model evaluation")
    for name, digest in final["files"].items(): require(sha(local_path(originals, name)) == digest, "Evaluation original changed")
    manifest = {str(path.relative_to(originals)): sha(path) for path in originals.rglob("*") if path.is_file()}
    write(destination / "manifest.json", {"files": manifest})
    write(destination / "receipt.json", {"status": "verified", "archive_sha256": sha(archive), "manifest_sha256": sha(destination / "manifest.json"),
        "file_count": len(manifest), "candidate_test_count": 6, "validation_replay_count": 6,
        "qualification_exit0_count": 6, "actual_data_worker_exit0_count": 12, "actual_native_worker_exit0_count": 18, "actual_pipeline_exit0_count": 1,
        "owner_absence_verified": True, "paused_external_preserved": True, "whole_raw9_complete": False})
    importlib.import_module("paper.scripts.compare_titantpp_shared_raf_taxi6_test").main(["--bundle", str(BUNDLE)])
    return {"status": "complete", "final_receipt": str(BUNDLE / "final_receipt.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "verify", "advance", "remote-run"), required=True)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args(argv)
    functions = {"prepare": prepare, "verify": lambda: verify_local(archive=True), "advance": advance, "remote-run": lambda: remote_run(args.root)}
    try: result = functions[args.mode]()
    except Exception as error:
        target = args.root if args.mode == "remote-run" else REPORT
        if target is not None and not (Path(target) / "FAILURE.json").exists(): write(Path(target) / "FAILURE.json", {"status": "failed", "at": now(), "error": str(error), "automatic_retry": False})
        raise
    print(json.dumps(result, ensure_ascii=False, allow_nan=False)); return 0

if __name__ == "__main__": raise SystemExit(main())
