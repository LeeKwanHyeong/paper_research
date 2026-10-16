"""Add seeds 52/62 in a separate bounded queue; preserve the seed42 experiment."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import shlex
import subprocess
import tarfile

from paper.scripts.manage_titantpp_cross_fusion_precheck import (
    ROOT, ARMS, canonical, read, sha, ssh, write,
)

NAME = "titantpp_time_quantity_cross_seed52_62_20261016_v1"
PARENT_NAME = "titantpp_time_quantity_cross_precheck6_20261016_v1"
BUNDLE = ROOT / "search_artifacts" / NAME
REPORT = ROOT / "reports" / NAME
PARENT = ROOT / "search_artifacts" / PARENT_NAME
PARENT_REPORT = ROOT / "reports" / PARENT_NAME
RUNTIME = "paper.scripts.titantpp_cross_fusion_seed_extension_runtime"
NEW_FILES = (
    "paper/scripts/titantpp_cross_fusion_seed_extension_runtime.py",
    "paper/scripts/manage_titantpp_cross_fusion_seed_extension.py",
)
PARENT_SHA = "0d64592e1458dc8434d3791e6c9ec0fc18edd6d041e27f6fde72f7e02437116d"
MLP_REFS_SHA = "4cbb11fac1c1f8a26d7dcbfa9496a4fc3a4a4bf7c87c52ff418c71020d7ec4fe"


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare():
    if BUNDLE.exists():
        raise FileExistsError("Extension bundle already exists; no overwrite")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise ValueError("Commit the extension implementation and tests before prepare")
    if sha(PARENT / "execution_contract.json") != PARENT_SHA:
        raise ValueError("Seed42 contract changed")
    parent = read(PARENT / "execution_contract.json")
    prepared = read(PARENT_REPORT / "PREPARED.json")
    if sha(PARENT / "deployment.tar.gz") != prepared["archive_sha256"]:
        raise ValueError("Seed42 deployment archive changed")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source = {**parent["source"]["files"], **{p: sha(ROOT / p) for p in NEW_FILES}}
    c = deepcopy(parent)
    c.update(schema="titantpp_cross_fusion_seed_extension12_v1", created_utc=now(),
        authorization="2026-10-16 user explicitly approved adding seeds52 and62 to both existing arms on RAF, Intermittent and Taxi; maximum3 concurrent fits per host INCLUDING seed42",
        source={"base_git_revision": revision, "scientific_source_revision": parent["source"]["base_git_revision"],
                "files": source, "files_sha256": canonical(source)},
        limits={"fit_wall_seconds": 129600, "queue_wall_seconds": 259200},
        parent={"contract_sha256": PARENT_SHA, "canonical_sha256": canonical(parent),
                "source_files_sha256": parent["source"]["files_sha256"], "qualification_sha256": {}},
        comparison={"primary": "Cross minus pointwise by dataset and seed42/52/62",
            "baseline": "Stored History MLP16 Validation seed42/52/62 only; no baseline execution",
            "intermittent_cross_host_limit": "Pointwise5080/cuDNN91900; Cross5090/cuDNN92000 for all three seeds",
            "single_seed": False, "test": False, "original_seed42_reused": True,
            "new_fit_count": 12, "combined_candidate_fit_count": 18, "cap_data_used": False})
    for host, h in c["hosts"].items():
        observed = read(sorted((PARENT_REPORT / "observations").glob(f"*_{host}.json"))[-1])
        q = observed["qualification.json"]
        start = observed["start.json"]
        if q["status"] != "passed" or q["contract_sha256"] != canonical(parent):
            raise ValueError("Parent qualification identity differs")
        c["parent"]["qualification_sha256"][host] = start["qualification_sha256"]
        h["parent_root"] = h["root"]
        h["root"] = f"/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/{NAME}_{host}"
        h["source_root"] = h["root"] + "/source"
        h["tmux"] = f"cross_seeds52_62_{host}_20261016_v1"
        h["environment"].update(SOURCE_REVISION=revision, PYTHONPATH=h["source_root"],
            MPLCONFIGDIR=h["root"] + "/cache/matplotlib", XDG_CACHE_HOME=h["root"] + "/cache")
    c["jobs"] = []
    # Within each host, shorter Taxi/RAF pairs precede the two Intermittent fits.
    for dataset in ("yellow_trip_hourly", "raf_spare_parts", "intermittent_frozen_5000"):
        for seed in (52, 62):
            for arm in ARMS:
                old = next(j for j in parent["jobs"] if j["dataset"] == dataset and j["arm"] == arm)
                c["jobs"].append({**old, "seed": seed, "id": f"{dataset}__{arm}__seed{seed}"})
    from paper.scripts.titantpp_cross_fusion_seed_extension_runtime import validate_contract
    for host in c["hosts"]:
        validate_contract(c, host)
    refs_path = ROOT / "search_artifacts/titantpp_cnn_gru54_3seed_dual_20261005_v1/width16_validation_refs.json"
    if sha(refs_path) != MLP_REFS_SHA:
        raise ValueError("Stored MLP16 reference inventory changed")
    refs = read(refs_path)["rows"]
    if len(refs) != 9 or {(r["dataset"], r["seed"]) for r in refs} != {
            (d["dataset_id"], s) for d in c["datasets"] for s in (42, 52, 62)}:
        raise ValueError("Expected nine stored MLP16 Validation references")
    for r in refs:
        if sha(r["endpoint_path"]) != r["endpoint_sha256"]:
            raise ValueError("Stored MLP16 reference changed")
    BUNDLE.mkdir(parents=True)
    write(BUNDLE / "execution_contract.json", c)
    write(BUNDLE / "stored_mlp16_validation_references.json", refs)
    with tarfile.open(PARENT / "deployment.tar.gz") as original, tarfile.open(BUNDLE / "deployment.tar.gz", "w:gz") as output:
        expected = {**parent["source"]["files"], **parent["input_files"]}
        seen = set()
        for member in original:
            if member.name == "execution_contract.json":
                continue
            if not member.isfile() or not member.name.startswith("source/"):
                raise ValueError("Unexpected parent archive entry")
            name = member.name[len("source/"):]
            if name not in expected or name in seen:
                raise ValueError("Duplicate/unexpected inherited source or input")
            with original.extractfile(member) as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != expected[name]:
                raise ValueError("Inherited source/input changed: " + name)
            with original.extractfile(member) as handle:
                output.addfile(member, handle)
            seen.add(name)
        if seen != set(expected):
            raise ValueError("Incomplete inherited archive")
        output.add(BUNDLE / "execution_contract.json", arcname="execution_contract.json")
        for name in NEW_FILES:
            output.add(ROOT / name, arcname="source/" + name)
    receipt = {"status": "prepared", "source_revision": revision, "scientific_source_revision": parent["source"]["base_git_revision"],
        "contract_sha256": sha(BUNDLE / "execution_contract.json"), "canonical_sha256": canonical(c),
        "archive_sha256": sha(BUNDLE / "deployment.tar.gz"), "source_file_count": len(source),
        "byte_identical_parent_source_count": len(parent["source"]["files"]),
        "planned_fit_count": 12, "seeds": [52, 62], "held_out_test_evaluated": False,
        "cap_data_used": False, "historical_baseline_retrained": False}
    write(REPORT / "PREPARED.json", receipt)
    return receipt


def verify_local():
    p = read(REPORT / "PREPARED.json")
    if sha(BUNDLE / "execution_contract.json") != p["contract_sha256"]:
        raise ValueError("Extension contract changed")
    c = read(BUNDLE / "execution_contract.json")
    if canonical(c) != p["canonical_sha256"]:
        raise ValueError("Extension canonical contract changed")
    for name in NEW_FILES:
        if sha(ROOT / name) != c["source"]["files"][name]:
            raise ValueError("Extension operation changed after prepare")
    if sha(PARENT / "execution_contract.json") != PARENT_SHA:
        raise ValueError("Parent contract changed")
    return c


def save_transport(host, phase, result):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    prefix = BUNDLE / "transport" / host / f"{stamp}_{phase}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, value in ((".stdout", result.stdout), (".stderr", result.stderr)):
        with prefix.with_suffix(suffix).open("xb") as handle:
            handle.write(value)
    write(prefix.with_suffix(".json"), {"actual_transport_exit": result.returncode,
        "stdout_bytes": len(result.stdout), "stderr_bytes": len(result.stderr),
        "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr).hexdigest(), "observed_utc": now()})
    if result.returncode:
        raise RuntimeError(f"{host} {phase} failed with exit{result.returncode}: " + result.stderr.decode(errors="replace")[-2000:])
    return result.stdout


def invocation(c, host, mode):
    h = c["hosts"][host]
    return "cd " + shlex.quote(h["source_root"]) + " && " + shlex.join([
        "env", *[f"{k}={v}" for k, v in h["environment"].items()], h["python"], "-u", "-m", RUNTIME,
        "--contract", h["root"] + "/execution_contract.json", "--host", host, "--mode", mode])


def deploy_start(host):
    c = verify_local()
    h = c["hosts"][host]
    if sha(BUNDLE / "deployment.tar.gz") != read(REPORT / "PREPARED.json")["archive_sha256"]:
        raise ValueError("Deployment archive changed")
    write(BUNDLE / "attempts" / f"{host}.json", {"created_utc": now(), "host": host,
        "contract_sha256": canonical(c), "automatic_retry": False})
    install = "mkdir " + shlex.quote(h["root"]) + " && tar -xzf - -C " + shlex.quote(h["root"])
    save_transport(host, "install", ssh(host, install, stdin=(BUNDLE / "deployment.tar.gz").read_bytes()))
    admission = json.loads(save_transport(host, "admit", ssh(host, invocation(c, host, "admit"), timeout=180)))
    command = [h["python"], "-u", "-m", RUNTIME, "--contract", h["root"] + "/execution_contract.json",
               "--host", host, "--mode", "dispatch"]
    supervisor = (
        "import os,subprocess,json,pathlib,sys,datetime; "
        f"p=subprocess.Popen({command!r},cwd={h['source_root']!r},env={{**os.environ,**{h['environment']!r}}}); "
        "code=p.wait(); "
        f"path=pathlib.Path({h['root']!r})/'dispatcher_exit.json'; "
        "path.write_text(json.dumps({'pid':p.pid,'actual_os_returncode':code,"
        "'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),"
        "'supervisor_own_exit_directly_observed':False})); sys.exit(code)"
    )
    launch = shlex.join([h["python"], "-c", supervisor]) + " > " + shlex.quote(h["root"] + "/dispatcher.log") + " 2>&1"
    save_transport(host, "launch", ssh(host, shlex.join([h["tmux_binary"], "new-session", "-d", "-s", h["tmux"], launch])))
    receipt = {"status": "queue_dispatched_not_yet_verified", "host": host, "utc": now(),
        "admission": admission, "root": h["root"], "maximum_combined_workers": 3}
    write(REPORT / f"DISPATCHED_{host}.json", receipt)
    return receipt


def status(host):
    c = verify_local()
    h = c["hosts"][host]
    program = '''import json,pathlib,subprocess,datetime,math,os,hashlib
root=pathlib.Path(__ROOT__); parent=pathlib.Path(__PARENT__)
c=json.loads((root/'execution_contract.json').read_text())
result={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':__HOST__,'jobs':[],'parent_jobs':[]}
for name in ('qualification.json','start.json','status.json','dispatcher_exit.json','final_receipt.json','dispatch_failure.json'):
 p=root/name
 if p.exists():result[name]=json.loads(p.read_text())
for j in c['jobs']:
 if j['host']!=__HOST__:continue
 d=root/'run'/j['id']; r={'job':j}
 for name in ('worker.json','worker_exit.json','fit_receipt.json'):
  p=d/name
  if p.exists():r[name]=json.loads(p.read_text())
 for p in d.glob('runs/*/*/seed_'+str(j['seed'])+'/history.json'):
  rows=json.loads(p.read_text())['history'];r['saved_epochs']=len(rows)
  eligible=[x for x in rows if math.isfinite(x['val_qty_rmse'])]
  if eligible:r['best']=min(eligible,key=lambda x:x['val_qty_rmse'])
 result['jobs'].append(r)
pc=json.loads((parent/'execution_contract.json').read_text())
for j in pc['jobs']:
 if j['host']!=__HOST__:continue
 r={'job':j}; d=parent/'run'/j['id']
 for name in ('worker.json','worker_exit.json'):
  p=d/name
  if p.exists():r[name]=json.loads(p.read_text())
 result['parent_jobs'].append(r)
result['gpu']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
result['processes']=[]
for path in pathlib.Path('/proc').iterdir():
 if not path.name.isdigit():continue
 try:
  cmd=(path/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='replace')
  if int(path.name)!=os.getpid() and (str(root) in cmd or str(parent) in cmd):
   stat=(path/'stat').read_text().rsplit(') ',1)[1].split()
   result['processes'].append({'pid':int(path.name),'start_ticks':int(stat[19]),'cmd':cmd})
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
print(json.dumps(result,allow_nan=False))
'''.replace("__ROOT__", repr(h["root"])).replace("__PARENT__", repr(h["parent_root"])).replace("__HOST__", repr(host))
    payload = json.loads(save_transport(host, "status", ssh(host, shlex.join([h["python"], "-c", program]), timeout=90)))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    write(REPORT / "observations" / f"{stamp}_{host}.json", payload)
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("prepare", "deploy-start", "status", "verify-local"), required=True)
    p.add_argument("--host", choices=("5080", "5090"))
    a = p.parse_args()
    if a.mode in ("deploy-start", "status") and not a.host:
        p.error("--host required")
    result = {"prepare": prepare, "deploy-start": lambda: deploy_start(a.host),
        "status": lambda: status(a.host), "verify-local": lambda: {"status": "verified", "contract_sha256": canonical(verify_local())}}[a.mode]()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
