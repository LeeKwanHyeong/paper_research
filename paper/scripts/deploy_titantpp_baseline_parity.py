"""Deploy and execute one sealed, bounded Train-only implementation check."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "search_artifacts/titantpp_baseline_parity_20261008_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def verify_bundle(bundle=BUNDLE):
    manifest = read(bundle / "deployment_manifest.json")
    contract = read(bundle / "execution_contract.json")
    if sha(bundle / "execution_contract.json") != manifest["contract_raw_sha256"]:
        raise ValueError("Contract changed after sealing")
    for rel, expected in manifest["files"].items():
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise ValueError("Unsafe manifest path")
        if sha(bundle / rel) != expected:
            raise ValueError("Sealed file changed: " + rel)
    for rel, expected in manifest.get("project_files", {}).items():
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise ValueError("Unsafe project file path")
        if sha(ROOT / rel) != expected:
            raise ValueError("Sealed local execution file changed: " + rel)
    if manifest.get("required_empty_directories") != ["source/sample_data"]:
        raise ValueError("Unexpected source sentinel directory contract")
    for rel in manifest["required_empty_directories"]:
        directory = bundle / rel
        if not directory.is_dir() or any(directory.iterdir()):
            raise ValueError("Source sentinel must be an empty directory")
    approval = contract["approval"]
    if any(approval[key] for key in ["full_fit", "validation_inference", "test_inference", "expansion_training", "shared_runtime_change", "automatic_retry"]):
        raise ValueError("Execution exceeds the approved short check")
    if contract["runtime"]["host"] != "5080" or contract["real_train"]["max_updates"] != 3:
        raise ValueError("Unexpected host or update budget")
    if contract["synthetic"]["lengths"] != [64, 256] or contract["synthetic"]["updates_per_length"] != 1:
        raise ValueError("Unexpected synthetic update budget")
    if contract["resources"]["process_timeout_seconds"] != 900:
        raise ValueError("Unexpected execution time limit")
    return contract, manifest


DEPLOY = r'''
import datetime,hashlib,io,json,pathlib,subprocess,sys,tarfile
root=pathlib.Path(sys.argv[1]); expected=sys.argv[2]
blob=sys.stdin.buffer.read()
assert hashlib.sha256(blob).hexdigest()==expected,"Package SHA differs"
assert not root.exists(),"Existing remote root preserved"
assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"],text=True).strip(),"GPU is busy"
with tarfile.open(fileobj=io.BytesIO(blob),mode="r:gz") as archive:
    members=archive.getmembers()
    assert all(m.isfile() and not pathlib.PurePosixPath(m.name).is_absolute() and ".." not in pathlib.PurePosixPath(m.name).parts for m in members)
    root.mkdir(parents=True,exist_ok=False)
    archive.extractall(root,filter="data")
manifest=json.loads((root/"deployment_manifest.json").read_text())
assert manifest.get("required_empty_directories")==["source/sample_data"]
for rel in manifest["required_empty_directories"]: (root/rel).mkdir(exist_ok=False)
for rel,digest in manifest["files"].items():
    assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest,rel
assert hashlib.sha256((root/"execution_contract.json").read_bytes()).hexdigest()==manifest["contract_raw_sha256"]
print(json.dumps({"status":"deployed","root":str(root),"contract_raw_sha256":manifest["contract_raw_sha256"],"package_sha256":expected,"observed_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}))
'''

RUN = r'''
import base64,datetime,hashlib,json,os,pathlib,subprocess,sys,time
root=pathlib.Path(sys.argv[1]); expected=sys.argv[2]
assert hashlib.sha256((root/"execution_contract.json").read_bytes()).hexdigest()==expected
c=json.loads((root/"execution_contract.json").read_text())
assert str(root)==c["runtime"]["root"] and sys.executable==c["runtime"]["python"]
m=json.loads((root/"deployment_manifest.json").read_text())
assert m.get("required_empty_directories")==["source/sample_data"]
assert (root/"source/sample_data").is_dir() and not any((root/"source/sample_data").iterdir())
for rel,digest in m["files"].items(): assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest,rel
assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"],text=True).strip(),"GPU became busy"
gpu=subprocess.check_output(["nvidia-smi","--query-gpu=uuid","--format=csv,noheader"],text=True).strip()
assert gpu==c["runtime"]["gpu_uuid"],"GPU UUID differs"
claims=root/"claims"; claims.mkdir(exist_ok=True)
argv=["/usr/bin/timeout","--signal=TERM","--kill-after=15s","900s",sys.executable,str(root/"operation/check.py"),"--source",str(root/"source"),"--contract",str(root/"execution_contract.json"),"--output",str(root/"result.json"),"--device","cuda"]
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (claims/"native.json").open("x") as stream: json.dump({"argv":argv,"started_at_utc":started,"wrapper_pid":os.getpid(),"wrapper_ppid":os.getppid(),"retry":False},stream)
clock=time.monotonic(); result=subprocess.run(argv,capture_output=True,text=True,timeout=940)
after=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"],text=True).strip()
log={"argv":argv,"returncode":result.returncode,"stdout":result.stdout,"stderr":result.stderr,"elapsed_seconds":time.monotonic()-clock,"started_at_utc":started,"finished_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"compute_processes_after":after}
with (root/"execution.json").open("x") as stream: json.dump(log,stream,indent=2,allow_nan=False)
raw=(root/"result.json").read_bytes() if (root/"result.json").exists() else b""
unchanged=all(hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest for rel,digest in m["files"].items())
reply={"status":"complete" if result.returncode==0 and not after and unchanged else "failed","returncode":result.returncode,"compute_processes_after":after,"sealed_inputs_unchanged":unchanged,"result_raw_sha256":hashlib.sha256(raw).hexdigest(),"result_base64":base64.b64encode(raw).decode(),"execution":log}
print(json.dumps(reply,allow_nan=False)); sys.exit(0 if reply["status"]=="complete" else 1)
'''


def remote(contract, code, args, input_bytes, timeout):
    runtime = contract["runtime"]
    env = runtime["environment"]
    argv = ["env"] + [key + "=" + value for key, value in sorted(env.items())]
    argv += [runtime["python"], "-c", code] + args
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "5080", shlex.join(argv)],
        input=input_bytes, capture_output=True, timeout=timeout,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["deploy", "run"], required=True)
    args = parser.parse_args()
    contract, manifest = verify_bundle()
    runtime = contract["runtime"]
    evidence = BUNDLE / "deployment"
    evidence.mkdir(exist_ok=True)
    if args.mode == "deploy":
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            for rel in sorted(list(manifest["files"]) + ["deployment_manifest.json"]):
                archive.add(BUNDLE / rel, arcname=rel, recursive=False)
        blob = stream.getvalue()
        write_once(evidence / "deploy_intent.json", {"package_sha256": hashlib.sha256(blob).hexdigest(), "root": runtime["root"]})
        result = remote(contract, DEPLOY, [runtime["root"], hashlib.sha256(blob).hexdigest()], blob, 180)
        write_once(evidence / "deploy_transport.json", {"returncode": result.returncode, "stdout": result.stdout.decode(), "stderr": result.stderr.decode()})
        if result.returncode:
            raise RuntimeError("Deployment failed; evidence retained, no automatic retry")
        receipt = json.loads(result.stdout)
        if receipt["contract_raw_sha256"] != manifest["contract_raw_sha256"]:
            raise ValueError("Deployment contract receipt differs")
        write_once(evidence / "deploy_receipt.json", receipt)
        print(json.dumps({"status": "deployed", "host": "5080", "root": runtime["root"]}))
        return
    if not (evidence / "deploy_receipt.json").is_file():
        raise ValueError("Deployment has not been verified")
    write_once(evidence / "run_intent.json", {"root": runtime["root"], "contract_raw_sha256": manifest["contract_raw_sha256"], "max_real_updates_per_model": 3})
    result = remote(contract, RUN, [runtime["root"], manifest["contract_raw_sha256"]], b"", 970)
    write_once(evidence / "run_transport.json", {"returncode": result.returncode, "stdout": result.stdout.decode(), "stderr": result.stderr.decode()})
    if not result.stdout.strip():
        raise RuntimeError("No native result returned; evidence retained, no retry")
    reply = json.loads(result.stdout)
    raw = base64.b64decode(reply.pop("result_base64"))
    if hashlib.sha256(raw).hexdigest() != reply["result_raw_sha256"]:
        raise ValueError("Returned result SHA differs")
    if raw:
        with (BUNDLE / "result_5080.json").open("xb") as stream:
            stream.write(raw)
    write_once(evidence / "execution_receipt.json", reply)
    if result.returncode or reply["status"] != "complete":
        raise RuntimeError("Native check failed; failure and claim retained")
    print(json.dumps({"status": "complete", "host": "5080", "result_raw_sha256": reply["result_raw_sha256"], "elapsed_seconds": reply["execution"]["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
