"""Recover the ten original retrieval executions after the approved repair.

Pending and already-finalized calls are local only.  A claimed or failed
retrieval is never retried.  No checkpoint is deserialized and no evaluation,
selection change, optimizer update, or held-out result read occurs here.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import tarfile
import time

SCHEMA = "titantpp_contextual_retrieval_pilot_repair_20261007_v1"
PARENT_SHA = "5372f2ec93c98e6b45ecf37d2b19ae1375a838fdfd9eb8a9c4f3a908456e7953"
FIXED = "5090__yellow_trip_hourly__42__titantpp_retrieval_fixed"
ARCHIVE_SCHEMA = "titantpp_retrieval_repair_original_archive_v1"
TRANSFER = "transfer_retrieval_repair_20261007_v1"
MAX_MANIFEST = 8 * 1024 ** 2
VARIANT = "count_only_log_regression"
BINARIES = ("best_val_qty_rmse_model.pt", "last_epoch_state.pt")
REQUIRED = (*BINARIES, "endpoint_replays.json", "history.json", "exposure.json",
            "epoch_timing.json", "summary.json", "server_checkpoint_receipt.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b""):
            result.update(chunk)
    return result.hexdigest()


def safe(root, name):
    require(isinstance(name, str) and name and "\\" not in name and "\x00" not in name,
            "Invalid original path")
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and ".." not in relative.parts
            and str(relative) == name, "Unsafe or noncanonical original path")
    root = Path(root).resolve()
    result = root.joinpath(*relative.parts)
    require(result.resolve().is_relative_to(root)
            and not any(p.is_symlink() for p in (result, *result.parents) if p.is_relative_to(root)),
            "Original path escapes root or uses a symlink")
    return result


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def load_bundle(bundle):
    bundle = Path(bundle).resolve()
    c = read(bundle / "execution_contract.json")
    current = read(bundle / "current.json")
    require(c["schema"] == SCHEMA and current.get("canonical_sha256") == canonical(c),
            "Repair current/contract binding changed")
    runtime = module(bundle / "operation/repair_runtime.py", "_sealed_retrieval_repair_runtime")
    runtime.validate(c, bundle=bundle)
    require(c["evaluation_scope"] == "validation_only" and c["held_out_test_evaluated"] is False,
            "Repair result scope changed")
    control = bundle / "control/integrity_receipt.json"
    integrity = read(control)
    require(current["control_integrity_receipt_sha256"] == sha(control)
            and integrity["contract_sha256"] == canonical(c)
            and integrity["automatic_retry"] is False, "Control integrity binding changed")
    files = integrity["files"]
    require(files and integrity.get("files_sha256", canonical(files)) == canonical(files),
            "Control integrity closure changed")
    for name, digest in files.items():
        require(sha(safe(bundle, name)) == digest, "Control file SHA changed: " + name)
    require(any("finaliz" in name for name in files) and any("observ" in name for name in files),
            "Finalizer/observer control source is not sealed")
    observer_paths = [safe(bundle, name) for name in files if "observ" in name and name.endswith(".py")]
    require(len(observer_paths) == 1, "Ambiguous sealed repair observer")
    observer = module(observer_paths[0], "_sealed_retrieval_repair_observer")
    qualification = module(bundle / "parent_operation/operation/qualify.py", "_sealed_retrieval_parent_qualification")
    observer.profile_eligibility = qualification.profile_eligibility
    return c, observer, {"current_sha256": sha(bundle / "current.json"),
        "control_integrity_receipt_sha256": sha(control), "control_files_sha256": canonical(files),
        "source_files_sha256": c["parent_contract"]["source"]["files_sha256"],
        "parent_operation_files_sha256": c["parent_contract"]["operation"]["files_sha256"],
        "repair_operation_files_sha256": c["operation"]["files_sha256"]}


def run_prefix(job):
    return "runs/" + job["arm"] + "/" + VARIANT + "/seed_42/"


def endpoint_gate(c, owner, row, job, digest):
    require(row["job"] == job and row.get("SHA_verified") is True,
            "Terminal job/file SHA verification missing")
    terminal = row["terminal"]
    require(terminal["status"] == "complete" and terminal["scientific_success"] is True
            and terminal["job"] == job and terminal["contract_sha256"] == digest
            and terminal["evaluation_scope"] == "validation_only"
            and terminal["held_out_test_evaluated"] is False, "Scientific terminal changed")
    files = terminal["files"]
    prefix = run_prefix(job)
    require({prefix + name for name in REQUIRED} <= set(files), "Required original binary/record missing")
    for name, value in files.items():
        safe(Path("/tmp"), name)
        require(isinstance(value, str) and len(value) == 64
                and not any(token in part.lower() for part in PurePosixPath(name).parts
                            for token in ("test", "heldout", "held_out")), "Invalid/held-out result original")
    endpoint, history = row["endpoint"], row["history"]["history"]
    require(endpoint["status"] == "complete" and endpoint["job"] == job
            and endpoint["evaluation_scope"] == "validation_only"
            and endpoint["held_out_test_evaluated"] is False and history,
            "Endpoint history/scope missing")
    require([r["epoch"] for r in history] == list(range(1, len(history) + 1))
            and all(r["train_all_finite"] and math.isfinite(r["val_qty_rmse"]) for r in history),
            "History is not consecutive finite scientific training")
    chosen = min(history, key=lambda r: r["val_qty_rmse"])
    require(endpoint["best_epoch"] == chosen["epoch"] and endpoint["completed_epochs"] == len(history)
            and 40 <= len(history) <= 300, "Strict earliest RMSE selection changed")
    stop = next((i for i in range(40, len(history) + 1)
        if i - min(history[:i], key=lambda r: r["val_qty_rmse"])["epoch"] >= 40), None)
    require(len(history) == (stop if stop is not None else 300), "Prescribed first stopping epoch changed")
    population = c["parent_contract"]["datasets"][0]
    for label, epoch, binary in (("selected", chosen["epoch"], BINARIES[0]),
                                 ("last", len(history), BINARIES[1])):
        cell = endpoint[label]
        require(cell["evaluation_scope"] == "validation_only" and cell["held_out_test_evaluated"] is False
                and cell.get("count", cell.get("target_count")) == 8268
                and cell["tail"]["count"] == 79
                and cell["quantity_boundaries"] == population["quantity_boundaries_all_train_rows"],
                "Full Validation/tail population changed")
        expected_path = owner["root"] + "/run/" + job["id"] + "/" + prefix + binary
        require(cell["checkpoint_path"] == expected_path, "Endpoint binary owner/path changed")
        for field, key in (("qty_rmse", "val_qty_rmse"), ("qty_mae", "val_qty_mae"), ("time_nll", "val_time_nll")):
            require(type(cell[field]) in (int, float) and math.isfinite(cell[field])
                    and math.isclose(cell[field], history[epoch - 1][key], rel_tol=1e-10, abs_tol=1e-8)
                    and type(cell["tail"][field]) in (int, float) and math.isfinite(cell["tail"][field]),
                    "Selected/last same-epoch mandatory metric mismatch")
    return True


def snapshot_path(bundle, reference):
    path = Path(reference)
    result = path if path.is_absolute() else safe(bundle, reference)
    require(result.resolve().is_relative_to(Path(bundle).resolve()) and not result.is_symlink(),
            "Evidence snapshot outside owned bundle")
    return result


def terminal_gate(c, bundle, observer):
    evidence = {}
    for host in ("5090", "5080"):
        path = Path(bundle) / "hourly_monitor" / ("terminal_cache_" + host + ".json")
        if not path.exists():
            return None
        cache = read(path)
        sp = snapshot_path(bundle, cache["snapshot"])
        require(sha(sp) == cache["snapshot_sha256"], "Terminal snapshot raw SHA changed")
        snap = read(sp)
        analysis = observer.analyze(c, snap)
        ap = snapshot_path(bundle, cache["analysis_path"])
        require(sha(ap) == cache["analysis_sha256"], "Terminal analysis raw SHA changed")
        require(analysis == read(ap) and analysis["terminal_verified"] is True,
                "Independent repair terminal analysis changed")
        require(snap["host"] == host and snap["processes"] == [] and not snap["gpu_pids"]
                and snap["gpu_uuid_verified"] is True and snap["source_SHA_verified"] is True
                and snap["operation_SHA_verified"] is True and not snap.get("failure")
                and snap["server_status"]["status"] == "complete"
                and snap["supervisor_exit"]["returncode"] == 0, "Actual repair owner shutdown incomplete")
        q = snap["qualification"]
        require(q["status"] == "passed" and q["host"] == host and q["contract_sha256"] == PARENT_SHA
                and q["source_files_sha256"] == c["parent_contract"]["source"]["files_sha256"]
                and q["operation_files_sha256"] == c["parent_contract"]["operation"]["files_sha256"]
                and q["runtime"]["gpu"]["uuid"] == c["hosts"][host]["gpu_uuid"]
                and q["cost_candidate_eligibility"] == observer.profile_eligibility(q["profiles"], c["parent_contract"]["arms"]),
                "Own inherited native qualification/cost proof changed")
        jobs = [j for j in c["jobs"] if j["host"] == host]
        require([r["job"] for r in snap["rows"]] == jobs
                and set(snap["server_status"]["completed"]) == {j["id"] for j in jobs},
                "Repair terminal coverage/order changed")
        owner = {"root": c["hosts"][host]["root"], "host": host, "python": c["hosts"][host]["python"],
                 "gpu_uuid": c["hosts"][host]["gpu_uuid"], "kind": "repair", "contract": c,
                 "contract_sha256": canonical(c), "snapshot": snap,
                 "snapshot_sha256": sha(sp), "cache_sha256": sha(path), "jobs": jobs}
        for row, job in zip(snap["rows"], jobs):
            endpoint_gate(c, owner, row, job, canonical(c))
            require(row["manifest_sha256"] == snap["server_status"]["completed"][job["id"]]["terminal_manifest_sha256"],
                    "Repair terminal manifest/supervisor binding changed")
            if job["id"] == FIXED:
                gate = row.get("resume_startup_gate")
                copy = row.get("resume_copy_receipt")
                require(gate and copy and gate["status"] == "passed" and gate["epoch"] == 148
                        and gate["best_epoch"] == 120 and gate["native_identity_validated"] is True
                        and gate["optimizer_rng_loader_and_selected_state_validated"] is True
                        and gate["optimizer_updates"] == 0
                        and gate["parent_checkpoint_sha256"] == c["resume"][FIXED]["files"]["last_epoch_state.pt"]
                        and row["endpoint"]["resume_parent_epoch"] == 148,
                        "Native E148 resume startup proof incomplete")
                require(copy["copied_raw_sha256"] == c["resume"][FIXED]["files"]
                        and copy["last_epoch"] == 148 and copy["best_epoch"] == 120
                        and copy["original_files_modified"] is False and copy["optimizer_updates"] == 0,
                        "Immutable E148 source-copy proof changed")
                old_history = read(snapshot_path(bundle, c["parent_terminal_evidence"]["snapshot"]))
                prefix = next(r for r in old_history["rows"] if r["job"]["id"] == FIXED)["history"]["history"]
                require(row["history"]["history"][:148] == prefix and len(prefix) == 148,
                        "Original fixed history prefix changed")
        evidence["repair_" + host] = owner
    binding = c["parent_terminal_evidence"]
    sp = snapshot_path(bundle, binding["snapshot"])
    require(sha(sp) == binding["snapshot_sha256"], "Parent timeout snapshot changed")
    snap, parent = read(sp), c["parent_contract"]
    require(snap["contract_sha256"] == PARENT_SHA and snap["host"] == "5090"
            and snap["processes"] == [] and not snap["gpu_pids"]
            and snap["gpu_uuid_verified"] is True and snap["source_SHA_verified"] is True
            and snap["operation_SHA_verified"] is True and snap["supervisor_exit"]["returncode"] == 1,
            "Parent failed-owner shutdown/source proof incomplete")
    failure = snap["failure"]
    require(failure == snap["server_status"] and failure["type"] == "ValueError"
            and failure["message"] == "Worker exited 124; no automatic retry"
            and failure["active_job"]["id"] == FIXED, "Parent is not the exact approved fixed timeout")
    fixed = next(r for r in snap["rows"] if r["job"]["id"] == FIXED)
    require(fixed["actual_saved_epoch"] == 148 and fixed["actual_best_epoch"] == 120
            and fixed["checkpoint_receipt"]["files"] == c["resume"][FIXED]["files"]
            and fixed["last_checkpoint_file_sha256"] == c["resume"][FIXED]["files"]["last_epoch_state.pt"],
            "Parent fixed timeout checkpoint binding changed")
    jobs = c["reused_jobs"]
    rows = [r for r in snap["rows"] if r["job"] in jobs]
    require([r["job"] for r in rows] == jobs and len(rows) == 2
            and set(failure["completed"]) == {j["id"] for j in jobs}, "Exactly two original completed jobs required")
    owner = {"root": parent["hosts"]["5090"]["root"], "host": "5090",
             "python": parent["hosts"]["5090"]["python"], "gpu_uuid": parent["hosts"]["5090"]["gpu_uuid"],
             "kind": "parent_timeout", "contract": parent, "contract_sha256": PARENT_SHA,
             "snapshot": snap, "snapshot_sha256": sha(sp), "cache_sha256": binding["snapshot_sha256"], "jobs": jobs}
    for row, job in zip(rows, jobs):
        endpoint_gate(c, owner, row, job, PARENT_SHA)
        require(row["manifest_sha256"] == failure["completed"][job["id"]]["terminal_manifest_sha256"],
                "Preserved terminal manifest binding changed")
    owner["snapshot_rows"] = rows
    evidence["parent_5090"] = owner
    ids = [j["id"] for value in evidence.values() for j in value["jobs"]]
    require(len(ids) == len(set(ids)) == 10 and set(ids) == {j["id"] for j in parent["jobs"]},
            "Ten original executions are not mapped exactly once")
    return evidence


def inventory(c, owner):
    """Create the exact SHA-admitted archive inventory without reading binaries."""
    contract, snap = owner["contract"], owner["snapshot"]
    files, records = {}, {}
    def add(name, digest=None, value=None):
        safe(Path("/tmp"), name)
        files[name] = digest
        if value is not None:
            records[name] = value
    add("execution_contract.json", value=contract)
    add("status.json", value=snap["server_status"])
    add("supervisor_process_exit.json", value=snap["supervisor_exit"])
    for name in ("approval.json", "start_permit.json", "training_permit.json", "supervisor.claim"):
        add(name)
    add("qualification/receipt.json", c["qualification_bindings"][owner["host"]]["receipt_sha256"], snap["qualification"])
    for name, digest in c["parent_contract"]["source"]["files"].items():
        add("source/" + name, digest)
    if owner["kind"] == "repair":
        for name, digest in c["operation"]["files"].items():
            add(name, digest)
        for name, digest in c["parent_contract"]["operation"]["files"].items():
            add("parent_operation/" + name, digest)
    else:
        for name, digest in contract["operation"]["files"].items():
            add(name, digest)
        add("failure.json", value=snap["failure"])
        fixed = next(r for r in snap["rows"] if r["job"]["id"] == FIXED)
        add("claims/" + FIXED + ".json", value=fixed["claim"])
        add("run/" + FIXED + "/status.json", value=fixed["status"])
        for name, digest in c["resume"][FIXED]["files"].items():
            add("run/" + FIXED + "/" + run_prefix(fixed["job"]) + name, digest)
    for row in owner.get("snapshot_rows", snap["rows"]):
        job = row["job"]
        base = "run/" + job["id"] + "/"
        add("claims/" + job["id"] + ".json", value=row["claim"])
        add(base + "terminal_manifest.json", row["manifest_sha256"], row["terminal"])
        for name, digest in row["terminal"]["files"].items():
            add(base + name, digest)
        add(base + run_prefix(job) + "endpoint_replays.json",
            row["terminal"]["files"][run_prefix(job) + "endpoint_replays.json"], row["endpoint"])
        add(base + run_prefix(job) + "history.json",
            row["terminal"]["files"][run_prefix(job) + "history.json"], row["history"])
    return files, records


REMOTE = r'''
import hashlib,json,subprocess,tarfile,time
from pathlib import Path,PurePosixPath
root=Path(ROOT).resolve()
def require(v,m):
 if not v:raise ValueError(m)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def safe(n):
 p=PurePosixPath(n);require(not p.is_absolute() and '..' not in p.parts and str(p)==n and '\\' not in n,'Unsafe original')
 r=root.joinpath(*p.parts);require(r.resolve().is_relative_to(root) and r.is_file() and not any(x.is_symlink() for x in (r,*r.parents) if x.is_relative_to(root)),'Unsafe or absent original');return r
def absent():
 ps=subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True)
 entries=(str(root/'operation/repair_runtime.py'),str(root/'operation/campaign.py'))
 require(not [l for l in ps.splitlines() if any(e in l for e in entries) and ('--mode fit' in l or '--mode worker' in l or '--mode dispatch' in l)],'Owned process appeared')
absent()
uuids=[r.split(',')[0].strip() for r in subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True).splitlines() if r.strip()]
require(uuids==[GPU],'Native GPU UUID changed')
files={}
for name,digest in FILES.items():
 p=safe(name);actual=sha(p);require(digest is None or actual==digest,'Native original raw SHA changed: '+name)
 if name in RECORDS:require(read(p)==RECORDS[name],'Native original record changed: '+name)
 files[name]={'sha256':actual,'size_bytes':p.stat().st_size}
for name,digest in INPUTS.items():require(sha(safe('source/'+name))==digest,'Native input bytes changed')
approval=read(safe('approval.json'));start=read(safe('start_permit.json'));permit=read(safe('training_permit.json'))
require(approval['approved'] is True and approval['contract_sha256']==EXPECTED and HOST in approval['hosts'],'Native approval changed')
require(start['host']==HOST and start['contract_sha256']==EXPECTED and permit['host']==HOST and permit['contract_sha256']==EXPECTED,'Native permit identity changed')
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
require(start['approval_sha256']==canonical(approval) and permit['start_permit_sha256']==canonical(start),'Native permit binding changed')
require(permit['qualification_sha256']==sha(safe('qualification/receipt.json')),'Native qualification raw SHA binding changed')
require(sum(v['size_bytes'] for v in files.values())<=MAX_BYTES,'Original archive too large')
directory=root/TRANSFER
require(not directory.is_symlink() and directory.resolve().is_relative_to(root),'Unsafe archive output')
directory.mkdir(exist_ok=True)
archive=directory/'training_originals.tar.gz';manifest_path=directory/'archive_manifest.json';receipt_path=directory/'receipt.json'
require(not any(p.exists() for p in (directory/'retrieval.claim',archive,manifest_path,receipt_path)),'Native retrieval already claimed; no retry')
with (directory/'retrieval.claim').open('x') as f:json.dump({'owner':OWNER,'repair_contract_sha256':REPAIR_SHA,'automatic_retry':False,'started_unix':time.time()},f)
manifest={'schema':ARCHIVE_SCHEMA,'owner':OWNER,'contract_sha256':EXPECTED,'repair_contract_sha256':REPAIR_SHA,'terminal_snapshot_sha256':SNAPSHOT_SHA,'files':files,'held_out_test_evaluated':False}
with manifest_path.open('x') as f:json.dump(manifest,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
with tarfile.open(archive,'x:gz') as tf:
 for name in sorted(files):tf.add(safe(name),arcname=name,recursive=False)
 tf.add(manifest_path,arcname='archive_manifest.json',recursive=False)
for name,value in files.items():require(sha(safe(name))==value['sha256'],'Original changed while archiving')
for name,value in RECORDS.items():require(read(safe(name))==value,'Record changed while archiving')
absent()
receipt={'status':'sealed','owner':OWNER,'host':HOST,'contract_sha256':EXPECTED,'repair_contract_sha256':REPAIR_SHA,'archive_path':str(archive),'archive_sha256':sha(archive),'manifest_sha256':sha(manifest_path),'terminal_snapshot_sha256':SNAPSHOT_SHA,'actual_owner_processes_absent':True,'gpu_uuid':GPU,'supervisor_exit_returncode':EXIT_CODE,'original_bytes':sum(v['size_bytes'] for v in files.values()),'held_out_test_evaluated':False,'automatic_retry':False,'created_unix':time.time()}
with receipt_path.open('x') as f:json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(receipt,allow_nan=False))
'''


def remote_script(c, key, owner):
    files, records = inventory(c, owner)
    values = {"ROOT": owner["root"], "HOST": owner["host"], "OWNER": key,
        "EXPECTED": owner["contract_sha256"], "REPAIR_SHA": canonical(c),
        "FILES": files, "RECORDS": records, "GPU": owner["gpu_uuid"],
        "INPUTS": c["parent_contract"]["input_files"], "MAX_BYTES": c["limits"]["root_bytes"],
        "TRANSFER": TRANSFER, "ARCHIVE_SCHEMA": ARCHIVE_SCHEMA,
        "SNAPSHOT_SHA": owner["snapshot_sha256"],
        "EXIT_CODE": 1 if owner["kind"] == "parent_timeout" else 0}
    return "\n".join(key + "=" + repr(value) for key, value in values.items()) + "\n" + REMOTE


def extract_verified(archive, destination, receipt, maximum):
    require(sha(archive) == receipt["archive_sha256"] and Path(archive).stat().st_size <= maximum + MAX_MANIFEST,
            "Archive raw SHA/size changed")
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        require(len(members) < 10000 and len({m.name for m in members}) == len(members)
                and sum(m.size for m in members) <= maximum + MAX_MANIFEST, "Archive member/expansion bound changed")
        for member in members:
            safe(destination, member.name)
            require(member.isfile() and not member.issym() and not member.islnk() and not member.sparse
                    and member.size >= 0, "Unsafe archive member")
        manifest_member = stream.getmember("archive_manifest.json")
        require(manifest_member.size <= MAX_MANIFEST, "Manifest exceeds bound")
        raw = stream.extractfile(manifest_member).read()
        require(hashlib.sha256(raw).hexdigest() == receipt["manifest_sha256"], "Manifest raw SHA changed")
        manifest = json.loads(raw)
        require(manifest["schema"] == ARCHIVE_SCHEMA and manifest["owner"] == receipt["owner"]
                and manifest["contract_sha256"] == receipt["contract_sha256"]
                and manifest["repair_contract_sha256"] == receipt["repair_contract_sha256"]
                and manifest["terminal_snapshot_sha256"] == receipt["terminal_snapshot_sha256"]
                and manifest["held_out_test_evaluated"] is False
                and {m.name for m in members} == set(manifest["files"]) | {"archive_manifest.json"},
                "Archive manifest coverage/identity changed")
        Path(destination).mkdir(parents=True, exist_ok=False)
        for member in members:
            target = safe(destination, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as source, target.open("xb") as output:
                for block in iter(lambda: source.read(1024 ** 2), b""):
                    output.write(block)
    return manifest


def verify_originals(c, key, owner, destination, receipt):
    original = Path(destination) / "original"
    manifest = read(original / "archive_manifest.json")
    require(sha(Path(destination) / "training_originals.tar.gz") == receipt["archive_sha256"]
            and sha(original / "archive_manifest.json") == receipt["manifest_sha256"]
            and manifest["owner"] == key and manifest["repair_contract_sha256"] == canonical(c)
            and manifest["contract_sha256"] == owner["contract_sha256"]
            and manifest["terminal_snapshot_sha256"] == owner["snapshot_sha256"], "Recovered archive binding changed")
    expected, records = inventory(c, owner)
    require(set(manifest["files"]) == set(expected)
            and {str(p.relative_to(original)) for p in original.rglob("*") if p.is_file()}
                == set(expected) | {"archive_manifest.json"}, "Recovered original closure changed")
    for name, value in manifest["files"].items():
        path = safe(original, name)
        require(type(value["size_bytes"]) is int and value["size_bytes"] >= 0
                and path.stat().st_size == value["size_bytes"] and sha(path) == value["sha256"]
                and (expected[name] is None or value["sha256"] == expected[name]), "Recovered original raw SHA changed")
        if name in records:
            require(read(path) == records[name], "Recovered record differs from terminal snapshot")
    binaries = []
    for row in owner.get("snapshot_rows", owner["snapshot"]["rows"]):
        for label, binary in (("selected", BINARIES[0]), ("last", BINARIES[1])):
            name = "run/" + row["job"]["id"] + "/" + run_prefix(row["job"]) + binary
            binaries.append({"id": row["job"]["id"], "owner": key, "role": label,
                "epoch": row["endpoint"]["best_epoch"] if label == "selected" else row["endpoint"]["completed_epochs"],
                "path": name, **manifest["files"][name], "CPU_binary_read_or_inference": False})
    return binaries


def command(argv, output, runner, *, timeout, input=None):
    try:
        result = runner(argv, input=input, text=True, capture_output=True, timeout=timeout)
    except BaseException as error:
        write(output, {"status": "transport_failed", "type": type(error).__name__, "automatic_retry": False})
        raise
    write(output, {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
                   "automatic_retry": False, "observed_unix": time.time()})
    require(result.returncode == 0, "Original recovery transport failed; no retry")
    return result


def retrieve(c, key, owner, bundle, runner=subprocess.run, *, allow_remote=True):
    destination = safe(bundle, "retrieved/" + key)
    receipt_path = destination / "local_original_SHA_receipt.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        require(receipt["status"] == "passed" and receipt["owner"] == key
                and receipt["local_originals_SHA_verified"] is True
                and receipt["actual_owner_processes_absent"] is True
                and receipt["supervisor_exit_returncode"] == (1 if owner["kind"] == "parent_timeout" else 0)
                and receipt["held_out_test_evaluated"] is False and receipt["automatic_retry"] is False,
                "Completed original receipt changed")
        require(verify_originals(c, key, owner, destination, receipt) == receipt["binary_originals"],
                "Completed binary original proof changed")
        return receipt
    require(allow_remote and not destination.exists(), "Missing or partial originals preserved; no automatic retry")
    write(destination / "retrieval.claim", {"owner": key, "repair_contract_sha256": canonical(c),
                                         "automatic_retry": False, "started_unix": time.time()})
    result = command(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", owner["host"],
        shlex.join([owner["python"], "-"])], destination / "archive_readback.json", runner,
        input=remote_script(c, key, owner), timeout=240)
    native = json.loads(result.stdout.strip().splitlines()[-1])
    archive_path = owner["root"] + "/" + TRANSFER + "/training_originals.tar.gz"
    require(native["status"] == "sealed" and native["owner"] == key and native["host"] == owner["host"]
            and native["contract_sha256"] == owner["contract_sha256"]
            and native["repair_contract_sha256"] == canonical(c) and native["archive_path"] == archive_path
            and native["terminal_snapshot_sha256"] == owner["snapshot_sha256"]
            and native["actual_owner_processes_absent"] is True and native["gpu_uuid"] == owner["gpu_uuid"]
            and native["supervisor_exit_returncode"] == (1 if owner["kind"] == "parent_timeout" else 0)
            and native["held_out_test_evaluated"] is False and native["automatic_retry"] is False,
            "Foreign native original seal")
    write(destination / "remote_archive_receipt.json", native)
    archive = destination / "training_originals.tar.gz"
    command(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        owner["host"] + ":" + archive_path, str(archive)], destination / "transfer_readback.json", runner, timeout=600)
    extract_verified(archive, destination / "original", native, c["limits"]["root_bytes"])
    receipt = {**native, "status": "passed", "local_originals_SHA_verified": True,
        "binary_originals": verify_originals(c, key, owner, destination, native),
        "actual_terminal_observed_utc": owner["snapshot"]["actual_observed_utc"],
        "CPU_binary_reinference_audit": "not_performed", "verified_unix": time.time()}
    write(receipt_path, receipt)
    return receipt


def summarize(c, evidence, originals):
    rows = []
    for key, owner in evidence.items():
        costs = owner["snapshot"]["qualification"]["cost_candidate_eligibility"]
        for row in owner.get("snapshot_rows", owner["snapshot"]["rows"]):
            job, endpoint = row["job"], row["endpoint"]
            selected = endpoint["selected"]
            rows.append({"id": job["id"], "configuration_id": job["configuration_id"], "host": job["host"],
                "owner": key, "arm": job["arm"], "seed": 42, "selected_epoch": endpoint["best_epoch"],
                "completed_epochs": endpoint["completed_epochs"],
                **{scope + "_" + metric: cell[field] for scope, cell in (("full", selected), ("tail", selected["tail"]))
                    for metric, field in (("RMSE", "qty_rmse"), ("MAE", "qty_mae"), ("TimeNLL", "time_nll"))},
                "cost_eligible_all_lengths": costs[job["arm"]]["eligible"],
                "resume_parent_epoch": 148 if job["id"] == FIXED else None,
                "terminal_manifest_sha256": row["manifest_sha256"]})
    rows.sort(key=lambda r: next(i for i, j in enumerate(c["parent_contract"]["jobs"]) if j["id"] == r["id"]))
    verdicts = {}
    parent = c["parent_contract"]
    for host in ("5090", "5080"):
        models = {r["arm"]: r for r in rows if r["host"] == host}
        adaptive = models[parent["pilot_success"]["candidate"]]
        comparisons = {arm: {metric: adaptive[metric] < models[arm][metric]
            for metric in ("full_RMSE", "tail_RMSE")} for arm in parent["pilot_success"]["comparators"]}
        verdicts[host] = {"strict_RMSE_comparisons": comparisons,
            "adaptive_cost_eligible_all_lengths": adaptive["cost_eligible_all_lengths"],
            "passed_prespecified_rule": adaptive["cost_eligible_all_lengths"]
                and all(all(values.values()) for values in comparisons.values())}
        mean = models[parent["arms"][0]]
        for row in models.values():
            row["full_TimeNLL_delta_vs_mean"] = row["full_TimeNLL"] - mean["full_TimeNLL"]
            row["tail_TimeNLL_delta_vs_mean"] = row["tail_TimeNLL"] - mean["tail_TimeNLL"]
    return {"schema": "titantpp_retrieval_repair_Validation_summary_v1", "contract_sha256": canonical(c),
        "parent_contract_sha256": PARENT_SHA, "configuration_count": 5, "execution_count": 10,
        "scientific_seed_count": 1, "seed": 42, "selected_metrics": rows, "per_host_decision": verdicts,
        "primary_success": verdicts["5090"]["passed_prespecified_rule"],
        "hardware_repetition_success": verdicts["5080"]["passed_prespecified_rule"],
        "binary_originals": [b for receipt in originals.values() for b in receipt["binary_originals"]],
        "native_GPU_cost_profiles": {host: evidence["repair_" + host]["snapshot"]["qualification"]["profiles"]
                                     for host in ("5090", "5080")},
        "cost_limits": parent["cost_limits"],
        "owner_mapping": {j["id"]: key for key, owner in evidence.items() for j in owner["jobs"]},
        "old_fixed_timeout_preserved": True, "selection_changed": False,
        "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
        "independent_evaluation": False, "hardware_timings_pooled": False,
        "CPU_binary_reinference_audit": "not_performed", "automatic_multidataset_expansion": False}


def write_summary(bundle, result):
    out = Path(bundle) / "analysis"
    write(out / "Validation_summary.json", result)
    write(out / "binary_originals_manifest.json", {"contract_sha256": result["contract_sha256"],
        "binary_originals": result["binary_originals"], "all20_selected_last_binaries_raw_SHA_verified": True,
        "CPU_binary_reinference_audit": "not_performed", "held_out_test_evaluated": False})
    with (out / "selected_Validation_metrics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["selected_metrics"][0]))
        writer.writeheader()
        writer.writerows(result["selected_metrics"])
    lines = ["# 검색부 파일럿 수정 실행 최종 Validation 결과", "",
        "5개 설정·동일 seed42의 두 GPU 반복입니다. 원래 mean/weighted 결과와 fixed E148 부분 원본을 보존했습니다.",
        "fixed는 같은 5090에서 E149부터 재개했고 나머지 미시작 실행은 fresh fit입니다.", "",
        "| GPU | 방식 | 선택/종료 epoch | 전체 RMSE | 전체 MAE | 전체 TimeNLL | 큰 수량 RMSE | 큰 수량 MAE | 큰 수량 TimeNLL |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in result["selected_metrics"]:
        lines.append("| " + " | ".join([row["host"], row["arm"].removeprefix("titantpp_retrieval_"),
            str(row["selected_epoch"]) + "/" + str(row["completed_epochs"]),
            *(format(row[name], ".8g") for name in ("full_RMSE", "full_MAE", "full_TimeNLL", "tail_RMSE", "tail_MAE", "tail_TimeNLL"))]) + " |")
    lines += ["", "5090 사전 판정: " + ("통과" if result["primary_success"] else "미통과"),
        "5080 같은 seed 하드웨어 반복 판정: " + ("통과" if result["hardware_repetition_success"] else "미통과"), "",
        "Validation 개발 증거이며 독립 Test·통계적 유의성·외부 우월성의 확정이 아닙니다.",
        "CPU binary 재추론 감사·Test·새 독립 평가는 미완료입니다. 새 GPU 평가나 선택 변경은 하지 않았습니다.", ""]
    with (out / "report.md").open("x") as stream:
        stream.write("\n".join(lines))
    return {name: sha(out / name) for name in ("Validation_summary.json", "binary_originals_manifest.json",
                                               "selected_Validation_metrics.csv", "report.md")}


def run(bundle, runner=subprocess.run):
    bundle = Path(bundle).resolve()
    c, observer, integrity = load_bundle(bundle)
    out = bundle / "analysis"
    completion, claim, failure = (out / name for name in ("completion_receipt.json", "finalize.claim", "failure.json"))
    if not completion.exists() and (claim.exists() or failure.exists()):
        return {"status": "manual_action_required_preserved_partial", "remote_calls": 0, "automatic_retry": False}
    evidence = terminal_gate(c, bundle, observer)
    if evidence is None:
        require(not completion.exists(), "Completed terminal evidence removed")
        pending = {"status": "pending_verified_all10", "contract_sha256": canonical(c),
            "integrity": integrity, "remote_calls": 0, "retrieval_started": False,
            "held_out_test_evaluated": False, "automatic_retry": False, "observed_unix": time.time()}
        write(bundle / "hourly_monitor/finalizer_pending" / (str(time.time_ns()) + ".json"), pending)
        return pending
    if completion.exists():
        receipt = read(completion)
        require(receipt["contract_sha256"] == canonical(c) and receipt["integrity"] == integrity
                and receipt["local_originals_SHA_verified"] is True and receipt["execution_conditions"] == 10
                and receipt["canonical_configurations"] == 5 and set(receipt["originals"]) == set(evidence),
                "Completed repair receipt changed")
        require(set(receipt["outputs"]) == {"Validation_summary.json", "binary_originals_manifest.json",
                "selected_Validation_metrics.csv", "report.md"}, "Completed output coverage changed")
        originals = {key: retrieve(c, key, owner, bundle, runner, allow_remote=False) for key, owner in evidence.items()}
        require(originals == receipt["originals"], "Completed original receipts changed")
        for name, digest in receipt["outputs"].items():
            require(sha(safe(out, name)) == digest, "Completed output raw SHA changed")
        require(summarize(c, evidence, originals) == read(out / "Validation_summary.json"),
                "Completed Validation selection/decision changed")
        return {**receipt, "remote_calls": 0, "already_finalized_local_revalidated": True}
    out.mkdir(exist_ok=True)
    with (out / "finalize.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "manual_action_required_lock_held", "remote_calls": 0, "automatic_retry": False}
        require(not claim.exists(), "Concurrent finalizer claim preserved")
        write(claim, {"contract_sha256": canonical(c), "integrity": integrity,
                      "automatic_retry": False, "started_unix": time.time()})
        try:
            originals = {key: retrieve(c, key, owner, bundle, runner) for key, owner in evidence.items()}
            result = summarize(c, evidence, originals)
            outputs = write_summary(bundle, result)
            require(sha(bundle / "current.json") == integrity["current_sha256"]
                    and sha(bundle / "control/integrity_receipt.json") == integrity["control_integrity_receipt_sha256"],
                    "Current/control changed during finalization")
            receipt = {"status": "all10_Validation_originals_complete", "contract_sha256": canonical(c),
                "canonical_configurations": 5, "execution_conditions": 10, "scientific_seed_count": 1,
                "integrity": integrity, "originals": originals, "outputs": outputs,
                "local_originals_SHA_verified": True, "three_owner_roots_terminated": True,
                "old_fixed_timeout_preserved": True, "primary_success": result["primary_success"],
                "hardware_repetition_success": result["hardware_repetition_success"],
                "selection_changed": False, "held_out_test_evaluated": False,
                "independent_evaluation": False, "CPU_binary_reinference_audit": "not_performed",
                "automatic_retry": False, "completed_unix": time.time()}
            write(completion, receipt)
            return receipt
        except BaseException as error:
            if not failure.exists():
                write(failure, {"status": "manual_action_required", "type": type(error).__name__,
                    "message": str(error), "automatic_retry": False, "held_out_test_evaluated": False,
                    "observed_unix": time.time()})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parents[2] / "search_artifacts" / SCHEMA)
    args = parser.parse_args()
    print(json.dumps(run(args.bundle), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
