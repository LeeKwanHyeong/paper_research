"""Prepare, launch and retrieve the separately approved 27 Cap-data external-backbone fits.

The two host pipelines finish training, replay selected Validation checkpoints,
then evaluate Test automatically. This manager creates no recurring automation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
from types import SimpleNamespace

from paper.scripts import titantpp_external_cap27_runtime as runtime

ROOT = Path(__file__).resolve().parents[2]
NAME = "titantpp_external_cap27_20261018_v1"
REFERENCE_NAME = "titantpp_mlp_cap36_20261017_v2"
BUNDLE = ROOT / "search_artifacts" / NAME
REPORT = ROOT / "reports" / NAME
REFERENCE_BUNDLE = ROOT / "search_artifacts" / REFERENCE_NAME
REFERENCE_REPORT = ROOT / "reports" / REFERENCE_NAME
PARENT = ROOT / "search_artifacts/titantpp_time_quantity_cross_precheck6_20261016_v1/execution_contract.json"
MODULE = "paper.scripts.titantpp_external_cap27_runtime"
SOURCE_DIRECTORIES = ("models", "data_loader", "utils", "paper/scripts", "simple_lab_test")
REQUIRED_NEW_SOURCE = ("paper/scripts/manage_titantpp_external_cap27.py", "paper/scripts/titantpp_external_cap27_runtime.py",
                       "paper/scripts/run_titantpp_external_cap27.py", "paper/scripts/compare_titantpp_external_cap27.py")
read, sha, canonical, require = runtime.read, runtime.sha_file, runtime.sha_json, runtime.require


def write(path, value):
    runtime.write(path, value, exclusive=True)


def now():
    return datetime.now(timezone.utc).isoformat()


def source_snapshot():
    """Untracked reports do not block a committed Python source snapshot."""
    changed = subprocess.check_output(["git", "diff", "--name-only", "HEAD", "--", *SOURCE_DIRECTORIES],
                                      cwd=ROOT, text=True).splitlines()
    require(not any(p.endswith(".py") for p in changed), "Commit changed tracked Python source before prepare")
    names = subprocess.check_output(["git", "ls-files", "--", *SOURCE_DIRECTORIES], cwd=ROOT, text=True).splitlines()
    files = {p: sha(ROOT / p) for p in names if p.endswith(".py")}
    require(all(p in files for p in REQUIRED_NEW_SOURCE), "Commit the new Cap27 operating files before prepare")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return {"base_git_revision": revision, "files": files, "files_sha256": canonical(files)}


def build_jobs():
    jobs = []
    for dataset in ("yellow_trip_hourly", "raf_spare_parts", "intermittent_frozen_5000"):
        for seed in (42, 52, 62):
            host = "5080" if dataset == "yellow_trip_hourly" or (dataset == "intermittent_frozen_5000" and seed == 42) else "5090"
            for arm in runtime.ARMS:
                jobs.append({"id": f"{dataset}__{arm}__seed{seed}", "dataset": dataset,
                             "arm": arm, "seed": seed, "host": host})
    return jobs


def reference_contract():
    """Reuse completed Cap inputs/conditions, never retrain or re-evaluate MLPs."""
    path = REFERENCE_BUNDLE / "execution_contract.json"
    prepared = read(REFERENCE_REPORT / "PREPARED.json")
    require(sha(path) == prepared["contract_sha256"], "Cap reference contract SHA changed")
    previous = read(path)
    require(canonical(previous) == prepared["canonical_contract_sha256"], "Cap reference canonical SHA changed")
    receipt_path = REFERENCE_BUNDLE / "retrieval/final_receipt.json"
    receipt = read(receipt_path)
    require(receipt["status"] == "complete" and receipt["candidate_test_count"] == 36
            and receipt["seed_count"] == 3 and receipt["held_out_test_evaluated"] is True,
            "Cap36 reference is incomplete")
    for name, digest in receipt["report_files"].items():
        require(sha(name) == digest, "Stored Cap MLP report changed")
    return previous, {"name": REFERENCE_NAME, "contract_path": str(path),
        "contract_raw_sha256": sha(path), "final_receipt_path": str(receipt_path),
        "final_receipt_sha256": sha(receipt_path), "report_files": receipt["report_files"],
        "fit_count": 36, "reevaluated": False, "retrained": False}


def package_archive(contract, source):
    archive_path = BUNDLE / "deployment.tar.gz"
    with archive_path.open("xb") as raw, tarfile.open(fileobj=raw, mode="w:gz") as archive:
        archive.add(BUNDLE / "execution_contract.json", arcname="execution_contract.json")
        for name in source["files"]:
            archive.add(ROOT / name, arcname="source/" + name)
        for name in contract["input_files"]:
            archive.add(BUNDLE / "input_snapshot" / name, arcname="source/" + name)
        # Root discovery needs the directory, not the repository's sample data.
        marker = tarfile.TarInfo("source/sample_data")
        marker.type, marker.mode = tarfile.DIRTYPE, 0o755
        archive.addfile(marker)
    return archive_path


def prepare():
    require(not BUNDLE.exists(), "Existing Cap27 bundle; no overwrite or retry")
    source = source_snapshot()
    previous, reference = reference_contract()
    require(sha(PARENT) == previous["parent_runtime_contract"]["sha256"], "Original Runtime parent SHA changed")
    BUNDLE.mkdir(parents=True, exist_ok=False)
    for name, digest in previous["input_files"].items():
        origin = runtime.local_path(REFERENCE_BUNDLE / "input_snapshot", name)
        require(sha(origin) == digest, "Reference Cap input SHA changed")
        destination = runtime.local_path(BUNDLE / "input_snapshot", name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with origin.open("rb") as src, destination.open("xb") as dst:
            shutil.copyfileobj(src, dst)
    prepared_data = {key: deepcopy(previous[key]) for key in ("datasets", "input_files", "data_receipt")}
    hosts = deepcopy(previous["hosts"])
    for host, h in hosts.items():
        root = f"/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/{NAME}_{host}"
        h.pop("wait_predecessor", None)
        h.update(root=root, source_root=root + "/source", tmux=f"external_cap27_{host}_20261018_v1")
        h["environment"].update(SOURCE_REVISION=source["base_git_revision"], PYTHONPATH=h["source_root"],
            MPLCONFIGDIR=root + "/cache/matplotlib", XDG_CACHE_HOME=root + "/cache")
    architecture = {"arms": list(runtime.ARMS), "type": "three_existing_matched_head_external_backbones_on_cap",
        "model_architecture_changed": False,
        "shared_input": "Train-Q3-capped observed and target quantities; log1p unchanged",
        "shared_loss_head": "same count-only log regression and recorded positive-integer lognormal time head as Cap36",
        "native_paper_heads_used": False, "checkpoint_warmstart": False}
    contract = {"schema": runtime.SCHEMA, "created_at": now(), "source": source, "hosts": hosts,
        "jobs": build_jobs(), "datasets": prepared_data["datasets"], "input_files": prepared_data["input_files"],
        "data_receipt": prepared_data["data_receipt"], "architecture": architecture,
        "design_sha256": canonical(architecture), "evaluation_scope": "validation_selection_then_frozen_test",
        "authorization": "2026-10-18 user requested S2P2, RMTPP and THP on Cap for comparison; same three datasets and seeds42/52/62,27 fresh fits, maximum3 per dedicated5080/5090 GPU, selected-checkpoint Validation/Test comparison",
        "held_out_test_authorized": True, "historical_cnn_retrained": False, "historical_cnn_reevaluated": False,
        "training": deepcopy(previous["training"]), "limits": deepcopy(previous["limits"]),
        "execution": {"maximum_concurrent_per_host": 3, "validation_gate_scope": "all assigned jobs on each host",
            "5080_launch": "fresh GPU idle and native qualification before dispatch",
            "5090_launch": "fresh GPU idle and native qualification before dispatch", "recurring_automation": False},
        "comparison": {"scope": "same Cap only; stored four MLPs and three new external backbones",
                       "seeds": [42,52,62], "sample_sd_ddof": 1},
        "stored_cap_mlp_reference": reference,
        "parent_runtime_contract": {"path": str(PARENT), "sha256": sha(PARENT)},
        "packaging": {"required_empty_source_directories": ["sample_data"]}}
    for key in ("datasets", "input_files", "training", "limits"):
        require(contract[key] == previous[key], "Cap reference scientific condition changed: " + key)
    for host in hosts:
        runtime.validate_contract(contract, host)
    write(BUNDLE / "execution_contract.json", contract)
    archive_path = package_archive(contract, source)
    receipt = {"status": "prepared", "created_at": now(), "source_revision": source["base_git_revision"],
        "source_file_count": len(source["files"]), "contract_sha256": sha(BUNDLE / "execution_contract.json"),
        "canonical_contract_sha256": canonical(contract), "archive_sha256": sha(archive_path),
        "planned_fit_count": 27, "host_fit_counts": {h: sum(j["host"] == h for j in contract["jobs"]) for h in hosts},
        "held_out_test_evaluated": False, "historical_cnn_retrained": False}
    write(REPORT / "PREPARED.json", receipt)
    return receipt


def verify_local(*, archive=False):
    receipt = read(REPORT / "PREPARED.json")
    require(sha(BUNDLE / "execution_contract.json") == receipt["contract_sha256"], "Prepared contract SHA changed")
    contract = read(BUNDLE / "execution_contract.json")
    require(canonical(contract) == receipt["canonical_contract_sha256"], "Canonical contract changed")
    for host in contract["hosts"]:
        runtime.validate_contract(contract, host)
    if archive:
        require(sha(BUNDLE / "deployment.tar.gz") == receipt["archive_sha256"], "Deployment archive SHA changed")
    return contract


def ssh(host, command, *, stdin=None, timeout=180):
    try:
        return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, command],
                              input=stdin, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        # subprocess has no directly observed remote child exit on a transport timeout.
        return SimpleNamespace(returncode=None, stdout=error.stdout or b"", stderr=error.stderr or b"", timed_out=True)


def save_transport(host, phase, result):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    prefix = BUNDLE / "transport" / host / f"{stamp}_{phase}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    out, err = prefix.with_suffix(".stdout"), prefix.with_suffix(".stderr")
    out.write_bytes(result.stdout)
    err.write_bytes(result.stderr)
    write(prefix.with_suffix(".json"), {"observed_at": now(), "actual_transport_exit": result.returncode,
        "transport_timeout": getattr(result, "timed_out", False), "stdout_bytes": len(result.stdout),
        "stdout_sha256": sha(out), "stderr_bytes": len(result.stderr), "stderr_sha256": sha(err)})
    require(result.returncode == 0, f"{host} {phase} failed with exit {result.returncode}; raw transport preserved")
    return result.stdout


def invocation(c, host, mode):
    h = c["hosts"][host]
    words = ["env", *[f"{k}={v}" for k, v in h["environment"].items()], h["python"], "-u", "-m", MODULE,
             "--contract", h["root"] + "/execution_contract.json", "--host", host, "--mode", mode]
    return "cd " + shlex.quote(h["source_root"]) + " && " + shlex.join(words)


def deploy(host):
    c = verify_local(archive=True)
    h = c["hosts"][host]
    write(BUNDLE / "attempts" / f"{host}.json", {"at": now(), "host": host, "automatic_retry": False,
          "contract_sha256": sha(BUNDLE / "execution_contract.json")})
    install = "mkdir " + shlex.quote(h["root"]) + " && tar -xzf - -C " + shlex.quote(h["root"])
    save_transport(host, "install", ssh(host, install, stdin=(BUNDLE / "deployment.tar.gz").read_bytes(), timeout=300))
    save_transport(host, "qualify", ssh(host, invocation(c, host, "qualify"), timeout=c["limits"]["evaluation_wall_seconds"]))
    mode = "dispatch"
    command = [h["python"], "-u", "-m", MODULE, "--contract", h["root"] + "/execution_contract.json",
               "--host", host, "--mode", mode]
    supervisor = (
        "import os,subprocess,json,pathlib,datetime,sys; "
        f"p=subprocess.Popen({command!r},cwd={h['source_root']!r},env={{**os.environ,**{h['environment']!r}}}); "
        "code=p.wait(); "
        f"path=pathlib.Path({h['root']!r})/'dispatcher_exit.json'; "
        "path.open('x').write(json.dumps({'pid':p.pid,'actual_os_returncode':code,"
        "'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),"
        "'supervisor_own_exit_directly_observed':False})+'\\n'); sys.exit(code)"
    )
    shell = shlex.join([h["python"], "-u", "-c", supervisor]) + " > " + shlex.quote(h["root"] + "/dispatcher.log") + " 2>&1"
    save_transport(host, "launch", ssh(host, shlex.join([h["tmux_binary"], "new-session", "-d", "-s", h["tmux"], shell])))
    receipt = {"status": "launched_not_yet_verified", "host": host, "launched_at": now(), "root": h["root"],
               "mode": mode, "planned_fit_count": sum(j["host"] == host for j in c["jobs"])}
    write(REPORT / f"LAUNCHED_{host}.json", receipt)
    return receipt


def status(host):
    c = verify_local()
    payload = json.loads(save_transport(host, "status", ssh(host, invocation(c, host, "status"), timeout=180)))
    require(payload["host"] == host and payload["contract_sha256"] == canonical(c), "Status identity mismatch")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = REPORT / "observations" / f"{stamp}_{host}.json"
    write(path, payload)
    return {"observation_path": str(path), "observation_sha256": sha(path), "snapshot": payload}


def safe_extract(archive_path, destination):
    destination = Path(destination)
    destination.mkdir(exist_ok=False)
    with tarfile.open(archive_path) as archive:
        members, seen = archive.getmembers(), set()
        for member in members:
            path = Path(member.name)
            require(member.name not in seen and not path.is_absolute() and ".." not in path.parts
                    and (member.isfile() or member.isdir()), "Unsafe/duplicate archive member")
            seen.add(member.name)
        archive.extractall(destination, members=members, filter="data")


def verify_collected(c, host, originals):
    originals = Path(originals)
    require(sha(originals / "execution_contract.json") == sha(BUNDLE / "execution_contract.json"), "Collected contract changed")
    final = read(originals / "final_receipt.json")
    jobs = runtime.validate_contract(c, host)
    ids = {j["id"] for j in jobs}
    require(final["status"] == "complete" and final["host"] == host and final["contract_sha256"] == canonical(c)
            and final["owned_worker_pids_absent"] is True, "Incomplete host receipt")
    require(final["completed_fit_count"] == final["validation_replay_count"] == final["test_count"] == len(jobs), "Stage counts differ")
    require(read(originals / "dispatcher_exit.json")["actual_os_returncode"] == 0, "Dispatcher exit was not successful")
    observation = read(originals / "collection_observation.json")
    require(observation["owned_pids_absent"] is True and observation["gpu"]["compute_pids"] == [], "Collection terminal state unknown")
    manifest_binding = final["artifact_manifest"]
    manifest_path = runtime.local_path(originals, manifest_binding["path"])
    require(sha(manifest_path) == manifest_binding["sha256"], "Terminal artifact manifest changed")
    inventory = read(manifest_path)["files"]
    require(inventory and len(inventory) == manifest_binding["file_count"], "Empty/incomplete terminal inventory")
    for path, digest in inventory.items():
        require(sha(runtime.local_path(originals, path)) == digest, "Collected artifact SHA differs: " + path)
    for stage in runtime.STAGES:
        result = final["stages"][stage]
        require(result["status"] == "complete" and set(result["exits"]) == ids and set(result["receipts"]) == ids,
                "Incomplete terminal stage")
        for job_id in ids:
            e, binding = result["exits"][job_id], result["receipts"][job_id]
            require(e["actual_os_returncode"] == 0 and not e["timeout"], "Worker did not exit successfully")
            path = runtime.local_path(originals, binding["path"])
            require(sha(path) == binding["sha256"], "Stage receipt SHA differs")
            receipt = read(path)
            require(receipt["status"] == "complete" and receipt["job_id"] == job_id
                    and receipt["stage"] == stage and receipt["contract_sha256"] == canonical(c)
                    and receipt["files"], "Invalid scientific receipt")
            if stage == "validation":
                require(receipt["validation_replay_passed"] is True, "Validation did not replay")
            for name, digest in receipt["files"].items():
                require(sha(runtime.local_path(originals, name)) == digest, "Scientific artifact SHA differs")
    return final


def collection_program(c, host):
    h = c["hosts"][host]
    # One connection checks current terminal state, then streams originals.
    return f'''import io,json,pathlib,sys,tarfile
from paper.scripts import titantpp_external_cap27_runtime as r
root=pathlib.Path({h["root"]!r})
c=r.read(root/'execution_contract.json')
r.require(r.sha_json(c)=={canonical(c)!r},'Contract mismatch')
jobs=r.validate_contract(c,{host!r}); ids={{j['id'] for j in jobs}}
f=r.read(root/'final_receipt.json'); x=r.read(root/'dispatcher_exit.json')
r.require(f['status']=='complete' and f['host']=={host!r} and f['contract_sha256']==r.sha_json(c),'Incomplete host')
r.require(f['completed_fit_count']==f['validation_replay_count']==f['test_count']==len(jobs),'Counts differ')
r.require(x['actual_os_returncode']==0,'Dispatcher failed')
pids=[x['pid']]
for stage in r.STAGES:
 s=f['stages'][stage]
 r.require(s['status']=='complete' and set(s['exits'])==ids and set(s['receipts'])==ids,'Stage incomplete')
 for j in jobs:
  e=s['exits'][j['id']]; r.require(e['actual_os_returncode']==0 and not e['timeout'],'Worker failed')
  pids.append(e['pid']); r.verify_receipt(c,{host!r},j,stage)
gpu=r.gpu_snapshot({h["gpu_uuid"]!r})
r.require(not gpu['compute_pids'] and all(r.proc_identity(p) is None for p in pids),'Owned process or GPU still present')
observation={{'host':{host!r},'observed_at':r.now(),'owned_pids':pids,'owned_pids_absent':True,'gpu':gpu,'supervisor_own_exit_directly_observed':False}}
blob=(json.dumps(observation,ensure_ascii=False,indent=2)+'\\n').encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for path in sorted(root.rglob('*')):
  rel=path.relative_to(root)
  if rel.parts[0] in ('source','cache'): continue
  r.require(not path.is_symlink(),'Unexpected symlink')
  if path.is_file(): archive.add(path,arcname=str(rel),recursive=False)
 info=tarfile.TarInfo('collection_observation.json'); info.size=len(blob); archive.addfile(info,io.BytesIO(blob))
'''


def collect(host):
    c = verify_local()
    h = c["hosts"][host]
    dest = BUNDLE / "retrieval" / host
    dest.mkdir(parents=True, exist_ok=False)
    write(dest / "attempt.json", {"host": host, "started_at": now(), "automatic_retry": False})
    archive_path, stderr_path = dest / "original.tar.gz", dest / "transport.stderr"
    command = "cd " + shlex.quote(h["source_root"]) + " && " + shlex.join([h["python"], "-c", collection_program(c, host)])
    timed_out = False
    with archive_path.open("xb") as out, stderr_path.open("xb") as err:
        try:
            completed = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, command],
                                       stdout=out, stderr=err, timeout=900)
        except subprocess.TimeoutExpired:
            timed_out = True
            completed = SimpleNamespace(returncode=None)
    write(dest / "transport.json", {"actual_transport_exit": completed.returncode, "observed_at": now(),
        "transport_timeout": timed_out,
        "stdout_bytes": archive_path.stat().st_size, "stdout_sha256": sha(archive_path),
        "stderr_bytes": stderr_path.stat().st_size, "stderr_sha256": sha(stderr_path)})
    require(completed.returncode == 0, "Collection failed; partial original archive/stderr preserved; no automatic retry")
    originals = dest / "originals"
    safe_extract(archive_path, originals)
    final = verify_collected(c, host, originals)
    files = {str(p.relative_to(originals)): sha(p) for p in sorted(originals.rglob("*")) if p.is_file()}
    write(dest / "manifest.json", {"files": files})
    receipt = {"status": "complete", "host": host, "collected_at": now(), "new_fit_count": final["completed_fit_count"],
        "validation_replay_count": final["validation_replay_count"], "test_count": final["test_count"],
        "archive_sha256": sha(archive_path), "manifest_sha256": sha(dest / "manifest.json"), "file_count": len(files),
        "held_out_test_evaluated": True, "existing_models_modified": False}
    write(dest / "receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("prepare", "deploy", "status", "collect"))
    parser.add_argument("--host", choices=("5080", "5090"))
    args = parser.parse_args(argv)
    require((args.mode == "prepare") == (args.host is None), "Only prepare omits --host")
    result = prepare() if args.mode == "prepare" else globals()[args.mode](args.host)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
