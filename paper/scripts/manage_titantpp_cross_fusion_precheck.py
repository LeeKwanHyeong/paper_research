"""Small, explicit launcher for the approved six-fit Validation pre-check.

No scheduler, baseline inference, Test evaluation, resume, or runtime installation.
Remote mutations are restricted to one new experiment directory on each host.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
NAME = "titantpp_time_quantity_cross_precheck6_20261016_v1"
BUNDLE = ROOT / "search_artifacts" / NAME
REPORT = ROOT / "reports" / NAME
PARENT = ROOT / "search_artifacts/titantpp_gru_residual_cap_validation_20261014_v1/execution_manual2/execution_contract.json"
ARMS = ("titantpp_history16_pointwise_fusion_v1", "titantpp_history16_cross_fusion_v1")
RUNTIME = "paper.scripts.titantpp_cross_fusion_precheck_runtime"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare():
    if BUNDLE.exists():
        raise FileExistsError(f"Existing pre-check bundle: {BUNDLE}")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise ValueError("Commit local implementation and tests before preparing the snapshot")
    parent = read(PARENT)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    files = subprocess.check_output(["git", "ls-files", "models", "data_loader", "utils",
        "paper/scripts", "simple_lab_test"], cwd=ROOT, text=True).splitlines()
    source = {p: sha(ROOT / p) for p in files if p.endswith(".py")}
    inputs = parent["input_files"]
    input_root = Path(parent["inherited_input_root"])
    for path, digest in inputs.items():
        if sha(input_root / path) != digest:
            raise ValueError("Admitted input changed: " + path)
    hosts = {}
    for host in ("5080", "5090"):
        old = parent["hosts"][host]
        root = f"/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/{NAME}_{host}"
        env = {**old["environment"], "SOURCE_REVISION": revision, "PYTHONPATH": root + "/source",
            "MPLCONFIGDIR": root + "/cache/matplotlib", "XDG_CACHE_HOME": root + "/cache"}
        hosts[host] = {"root": root, "source_root": root + "/source", "python": old["python"],
            "gpu_uuid": old["gpu_uuid"], "environment": env, "tmux_binary": old["tmux_binary"],
            "tmux": f"cross_precheck_{host}_20261016_v1",
            "runtime_expected": {"python": "3.12.13", "torch": "2.11.0+cu130", "cuda": "13.0",
                "cudnn": 91900 if host == "5080" else 92000, "threads": 4, "interop_threads": 1}}
    jobs = []
    for dataset, label, assignment in (("yellow_trip_hourly", "Taxi", ("5080", "5080")),
            ("raf_spare_parts", "RAF", ("5090", "5090")),
            ("intermittent_frozen_5000", "Intermittent", ("5080", "5090"))):
        for arm, host in zip(ARMS, assignment):
            jobs.append({"id": dataset + "__" + arm + "__seed42", "dataset": dataset,
                "label": label, "arm": arm, "host": host, "seed": 42})
    design = {"base": "History MLP width16", "location": "before first encoder layer",
        "arms": list(ARMS), "additional_parameters_each": 6176,
        "cross": "bidirectional causal observed-prefix cross attention, 16 dimensions each direction",
        "control": "two pointwise 16-32-16 GELU MLPs",
        "shared": "scalar GELU embeddings, observed-position encoding, zero-output residual, original memory and heads",
        "early_memory_relocation": False}
    contract = {"schema": "titantpp_cross_fusion_precheck6_v1", "created_utc": now(),
        "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
        "authorization": "User approved RAF/Taxi/Intermittent one-seed pre-check on both idle dedicated GPUs, maximum three concurrent per host",
        "datasets": deepcopy(parent["datasets"]), "jobs": jobs, "hosts": hosts,
        "input_files": inputs, "source": {"base_git_revision": revision,
            "files": source, "files_sha256": canonical(source)},
        "architecture": design, "design_sha256": canonical(design),
        "training": {"maximum_epochs": 300, "minimum_epochs": 40, "patience": 40,
            "batch_size": 128, "monitor": "validation_raw_quantity_rmse"},
        "limits": {"fit_wall_seconds": 36 * 3600},
        "comparison": {"primary": "Cross minus pointwise by dataset, seed42",
            "historical_baselines": "Stored Validation only; no model inference",
            "intermittent_cross_host_limit": "Control5080/cuDNN91900; cross5090/cuDNN92000",
            "single_seed": True, "test": False},
        "parent_data_contract": {"path": str(PARENT), "sha256": sha(PARENT)}}
    # Baselines are saved Validation references, never loaded as executable models.
    references = [r for r in parent["baseline_references"]
        if r["seed"] == 42 and r["arm"] == "titantpp_cnn_gru54_cnn_only"]
    if len(references) != 3:
        raise ValueError("Expected three stored CNN seed42 Validation references")
    for reference in references:
        if sha(reference["validation_receipt_path"]) != reference["validation_receipt_sha256"]:
            raise ValueError("Stored CNN Validation receipt changed")
    width_path = ROOT / "search_artifacts/titantpp_cnn_gru54_3seed_dual_20261005_v1/width16_validation_refs.json"
    if sha(width_path) != "4cbb11fac1c1f8a26d7dcbfa9496a4fc3a4a4bf7c87c52ff418c71020d7ec4fe":
        raise ValueError("Stored History MLP16 references changed")
    width_references = [r for r in read(width_path)["rows"] if r["seed"] == 42]
    for reference in width_references:
        if sha(reference["endpoint_path"]) != reference["endpoint_sha256"]:
            raise ValueError("Stored History MLP16 endpoint changed")
    BUNDLE.mkdir(parents=True)
    write(BUNDLE / "execution_contract.json", contract)
    write(BUNDLE / "stored_cnn_validation_references.json", references)
    write(BUNDLE / "stored_mlp16_validation_references.json", width_references)
    with tarfile.open(BUNDLE / "deployment.tar.gz", "w:gz") as archive:
        archive.add(BUNDLE / "execution_contract.json", arcname="execution_contract.json")
        for relative in source:
            archive.add(ROOT / relative, arcname="source/" + relative)
        for relative in inputs:
            archive.add(input_root / relative, arcname="source/" + relative)
    receipt = {"status": "prepared", "source_revision": revision,
        "contract_sha256": sha(BUNDLE / "execution_contract.json"),
        "archive_sha256": sha(BUNDLE / "deployment.tar.gz"), "source_file_count": len(source),
        "planned_fit_count": 6, "seed": 42, "held_out_test_evaluated": False}
    write(REPORT / "PREPARED.json", receipt)
    return receipt


def ssh(host, command, *, stdin=None, timeout=180):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host,
        command], input=stdin, capture_output=True, timeout=timeout)


def save_transport(host, phase, result):
    dest = BUNDLE / "transport" / host
    dest.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    prefix = dest / f"{stamp}_{phase}"
    prefix.with_suffix(".stdout").write_bytes(result.stdout)
    prefix.with_suffix(".stderr").write_bytes(result.stderr)
    write(prefix.with_suffix(".json"), {"actual_transport_exit": result.returncode,
        "stdout_bytes": len(result.stdout), "stderr_bytes": len(result.stderr), "observed_utc": now()})
    if result.returncode:
        raise RuntimeError(f"{host} {phase} exit{result.returncode}: {result.stderr.decode(errors='replace')[-3000:]}")
    return result.stdout


def remote_invocation(c, host, mode):
    h = c["hosts"][host]
    words = ["env", *[f"{k}={v}" for k, v in h["environment"].items()], h["python"], "-u", "-m", RUNTIME,
        "--contract", h["root"] + "/execution_contract.json", "--host", host, "--mode", mode]
    return "cd " + shlex.quote(h["source_root"]) + " && " + shlex.join(words)


def deploy_start(host):
    c = read(BUNDLE / "execution_contract.json")
    h = c["hosts"][host]
    write(BUNDLE / "attempts" / (host + ".json"), {"created_utc": now(), "host": host,
        "contract_sha256": sha(BUNDLE / "execution_contract.json"), "automatic_retry": False})
    install = "mkdir " + shlex.quote(h["root"]) + " && tar -xzf - -C " + shlex.quote(h["root"])
    save_transport(host, "install", ssh(host, install, stdin=(BUNDLE / "deployment.tar.gz").read_bytes()))
    qualification = save_transport(host, "qualify", ssh(host, remote_invocation(c, host, "qualify"), timeout=300))
    dispatch_words = [h["python"], "-u", "-m", RUNTIME, "--contract", h["root"] + "/execution_contract.json",
        "--host", host, "--mode", "dispatch"]
    supervisor = (
        "import os,subprocess,json,pathlib,sys,datetime; "
        f"p=subprocess.Popen({dispatch_words!r},cwd={h['source_root']!r},env={{**os.environ,**{h['environment']!r}}}); "
        "code=p.wait(); "
        f"path=pathlib.Path({h['root']!r})/'dispatcher_exit.json'; "
        "path.write_text(json.dumps({'pid':p.pid,'actual_os_returncode':code,"
        "'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),"
        "'supervisor_own_exit_directly_observed':False})+'\\n'); sys.exit(code)"
    )
    launch = shlex.join([h["python"], "-c", supervisor]) + " > " + shlex.quote(h["root"] + "/dispatcher.log") + " 2>&1"
    command = shlex.join([h["tmux_binary"], "new-session", "-d", "-s", h["tmux"], launch])
    save_transport(host, "launch", ssh(host, command))
    receipt = {"status": "dispatched_not_yet_verified", "host": host, "utc": now(),
        "qualification_stdout": qualification.decode(errors="replace"), "root": h["root"]}
    write(REPORT / ("DISPATCHED_" + host + ".json"), receipt)
    return receipt


def status(host):
    c = read(BUNDLE / "execution_contract.json")
    h = c["hosts"][host]
    # One read-only host snapshot; reading history does not evaluate a model.
    program = '''import json,pathlib,subprocess,datetime,math,os
root=pathlib.Path(__ROOT__)
c=json.loads((root/'execution_contract.json').read_text())
result={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':__HOST__,'jobs':[]}
for name in ('qualification.json','start.json','status.json','dispatcher_exit.json','final_receipt.json'):
 p=root/name
 if p.exists():result[name]=json.loads(p.read_text())
for j in c['jobs']:
 if j['host']!=__HOST__:continue
 d=root/'run'/j['id']; r={'job':j}
 for name in ('worker.json','worker_exit.json','fit_receipt.json'):
  p=d/name
  if p.exists():r[name]=json.loads(p.read_text())
 for p in d.glob('runs/*/*/seed_42/history.json'):
  rows=json.loads(p.read_text())['history'];r['saved_epochs']=len(rows)
  eligible=[x for x in rows if math.isfinite(x['val_qty_rmse'])]
  if eligible:r['best']=min(eligible,key=lambda x:x['val_qty_rmse'])
  r['recent_elapsed_seconds']=[x.get('elapsed_seconds') for x in rows[-10:]]
 result['jobs'].append(r)
result['gpu']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
result['processes']=[]
for path in pathlib.Path('/proc').iterdir():
 if not path.name.isdigit():continue
 try:
  cmd=(path/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='replace')
  if str(root) in cmd and int(path.name)!=os.getpid():result['processes'].append({'pid':int(path.name),'cmd':cmd})
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
print(json.dumps(result,allow_nan=False))
'''.replace("__ROOT__", repr(h["root"])).replace("__HOST__", repr(host))
    result = ssh(host, shlex.join([h["python"], "-c", program]), timeout=90)
    payload = json.loads(save_transport(host, "status", result))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    write(REPORT / "observations" / f"{stamp}_{host}.json", payload)
    return payload


def collect(host):
    """Retrieve completed host outputs once; never loads model checkpoints."""
    c = read(BUNDLE / "execution_contract.json")
    h = c["hosts"][host]
    destination = BUNDLE / "retrieval" / host
    if destination.exists():
        raise FileExistsError("Existing retrieval; inspect it instead of repeating the remote read")
    program = '''import pathlib,json,hashlib,tarfile,sys
root=pathlib.Path(__ROOT__)
final=json.loads((root/'final_receipt.json').read_text())
assert final['status']=='complete' and final['completed_fit_count']==3
assert len(final['actual_worker_exits'])==3
assert all(v['actual_os_returncode']==0 and not v['timeout'] for v in final['actual_worker_exits'].values())
assert json.loads((root/'dispatcher_exit.json').read_text())['actual_os_returncode']==0
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for name in ('run','logs','qualification.json','start.json','status.json','final_receipt.json','dispatcher_exit.json','dispatcher.log','execution_contract.json'):
  archive.add(root/name,arcname=name)
'''.replace("__ROOT__", repr(h["root"]))
    destination.mkdir(parents=True)
    result = ssh(host, shlex.join([h["python"], "-c", program]), timeout=300)
    # Preserve failed collection stdout/stderr too; no implicit second attempt.
    data = save_transport(host, "collect", result)
    archive_path = destination / "original.tar.gz"
    with archive_path.open("xb") as handle:
        handle.write(data)
    originals = destination / "originals"
    originals.mkdir()
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            if not (member.isfile() or member.isdir()) or not (originals / member.name).resolve().is_relative_to(originals.resolve()):
                raise ValueError("Invalid archive member")
        archive.extractall(originals, filter="data")
    final = read(originals / "final_receipt.json")
    if sha(originals / "execution_contract.json") != sha(BUNDLE / "execution_contract.json"):
        raise ValueError("Retrieved contract differs")
    for entry in final["fit_receipts"].values():
        receipt_path = originals / entry["path"]
        if sha(receipt_path) != entry["sha256"]:
            raise ValueError("Retrieved fit receipt differs")
        for path, digest in read(receipt_path)["files"].items():
            if sha(originals / path) != digest:
                raise ValueError("Retrieved scientific artifact differs: " + path)
    manifest = {str(p.relative_to(originals)): sha(p) for p in originals.rglob("*") if p.is_file()}
    write(destination / "manifest.json", manifest)
    receipt = {"status": "complete", "host": host, "utc": now(),
        "archive_sha256": sha(archive_path), "manifest_sha256": sha(destination / "manifest.json"),
        "file_count": len(manifest), "new_fit_count": 3, "held_out_test_evaluated": False}
    write(destination / "receipt.json", receipt)
    return receipt


def compare():
    """Pure aggregation of completed Validation summaries, not inference."""
    c = read(BUNDLE / "execution_contract.json")
    rows = []
    references = {r["dataset"]: r for r in read(BUNDLE / "stored_cnn_validation_references.json")}
    width_references = {r["dataset"]: r for r in read(BUNDLE / "stored_mlp16_validation_references.json")}
    for host in ("5080", "5090"):
        retrieval = BUNDLE / "retrieval" / host
        receipt = read(retrieval / "receipt.json")
        if receipt["status"] != "complete" or sha(retrieval / "manifest.json") != receipt["manifest_sha256"]:
            raise ValueError("Unverified retrieval")
        for path, digest in read(retrieval / "manifest.json").items():
            if sha(retrieval / "originals" / path) != digest:
                raise ValueError("Retrieved file changed: " + path)
    for job in c["jobs"]:
        output = BUNDLE / "retrieval" / job["host"] / "originals" / "run" / job["id"]
        summary = read(output / "fit_receipt.json")["summary"]
        baseline = references[job["dataset"]]
        width = width_references[job["dataset"]]
        if sha(baseline["validation_receipt_path"]) != baseline["validation_receipt_sha256"]:
            raise ValueError("Stored CNN Validation evidence changed")
        if sha(width["endpoint_path"]) != width["endpoint_sha256"]:
            raise ValueError("Stored MLP16 Validation evidence changed")
        rows.append({"dataset": job["dataset"], "arm": job["arm"], "seed": 42,
            "host": job["host"], "best_epoch": summary["best_epoch"],
            "rmse": summary["best_val_qty_rmse"], "mae": summary["best_val_qty_mae"],
            "minus_cnn_rmse": summary["best_val_qty_rmse"] - baseline["metrics"]["whole_qty_rmse"],
            "minus_cnn_mae": summary["best_val_qty_mae"] - baseline["metrics"]["whole_qty_mae"],
            "minus_mlp16_rmse": summary["best_val_qty_rmse"] - width["validation"]["qty_rmse"],
            "minus_mlp16_mae": summary["best_val_qty_mae"] - width["validation"]["qty_mae"],
            "time_nll": summary["best_val_time_nll"], "elapsed_seconds": summary["elapsed_seconds"]})
    differences = []
    for dataset in sorted(references):
        control = next(r for r in rows if r["dataset"] == dataset and r["arm"] == ARMS[0])
        cross = next(r for r in rows if r["dataset"] == dataset and r["arm"] == ARMS[1])
        differences.append({"dataset": dataset, "cross_minus_control_rmse": cross["rmse"] - control["rmse"],
            "cross_minus_control_mae": cross["mae"] - control["mae"], "same_host": cross["host"] == control["host"]})
    result = {"status": "complete", "evaluation_scope": "validation_only", "seed_count": 1,
        "new_fit_count": 6, "held_out_test_evaluated": False, "rows": rows, "differences": differences,
        "limitations": c["comparison"], "stored_cnn_references": references,
        "stored_mlp16_references": width_references}
    write(REPORT / "VALIDATION_PRECHECK.json", result)
    lines = ["# 시간·수량 결합 1-seed Validation pre-check", "", "seed42 결과이며 낮을수록 좋다. 기존 CNN은 저장 결과만 재사용했다.", "",
        "| 데이터 | 구조 | GPU | best epoch | RMSE | MAE | ΔCNN RMSE | ΔCNN MAE |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['dataset']} | {'Cross' if r['arm']==ARMS[1] else 'Pointwise'} | {r['host']} | {r['best_epoch']} | {r['rmse']:.6f} | {r['mae']:.6f} | {r['minus_cnn_rmse']:+.6f} | {r['minus_cnn_mae']:+.6f} |")
    lines += ["", "Intermittent는 두 후보의 GPU·cuDNN이 다르다. 시간은 host별로 해석한다. 1-seed 예비 결과이며 Test·seed 안정성·전단 메모리 효과를 검증한 결과가 아니다."]
    with (REPORT / "VALIDATION_PRECHECK.md").open("x") as handle:
        handle.write("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "deploy-start", "status", "collect", "compare"), required=True)
    parser.add_argument("--host", choices=("5080", "5090"))
    a = parser.parse_args()
    if a.mode in ("deploy-start", "status", "collect") and a.host is None:
        parser.error("--host is required")
    actions = {"prepare": prepare, "compare": compare, "deploy-start": lambda: deploy_start(a.host),
        "status": lambda: status(a.host), "collect": lambda: collect(a.host)}
    result = actions[a.mode]()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
