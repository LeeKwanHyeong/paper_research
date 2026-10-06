"""Explicit one-attempt deployment/qualification/launch of the approved 54 fits.

Importing this module performs no network or GPU operation. Failed intents and
readbacks are preserved; this helper never retries or modifies a shared runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

PROJECT = Path(__file__).resolve().parents[2]
B = PROJECT / "search_artifacts/titantpp_architecture_contribution_dual_20261006_v1"
HOSTS = ("5080", "5090")
SCHEMA = "titantpp_architecture_contribution_training_permit_v1"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def contract(host):
    if host not in HOSTS:
        raise ValueError("Foreign deployment host")
    c = read(B / "execution_contract.json")
    spec = c["hosts"][host]
    attempt = c.get("execution_attempt", 1)
    if type(attempt) is not int or attempt not in (1, 2):
        raise ValueError("Only the original or explicitly prepared manual attempt2 is admitted")
    expected = "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/" + B.name + "_" + host
    if attempt == 2:
        expected += "_attempt2"
    if spec["root"] != expected or spec["source_root"] != expected + "/source" or spec["operation_root"] != expected + "/operation":
        raise ValueError("Deployment root differs from the approved isolated campaign")
    if spec["environment"].get("SOURCE_REVISION") != c["source"]["git_revision"]:
        raise ValueError("Pinned SOURCE_REVISION missing")
    if not Path(spec["python"]).is_absolute() or not Path(spec["tmux_binary"]).is_absolute():
        raise ValueError("Native runtime paths must be absolute")
    approval, start = read(B / "approval.json"), read(B / "start_permit.json")
    if (approval.get("approved") is not True or approval.get("hosts") != list(HOSTS)
            or approval.get("contract_sha256") != canonical(c)
            or start.get("contract_sha256") != canonical(c)
            or start.get("approval_sha256") != canonical(approval)
            or not start["started_at_unix"] <= time.time() < start["deadline_unix"]):
        raise ValueError("Approval/start lease is missing, changed or expired")
    return c


def ssh(host, code, *, action, timeout=120):
    """Preserve every result, including transport or remote failures."""
    log = B / "deployment" / "readbacks" / (action + "_" + host + "_" + str(time.time_ns()) + ".json")
    try:
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, "python3 -"],
            input=code, text=True, capture_output=True, timeout=timeout)
    except BaseException as exc:
        write(log, {"host": host, "action": action, "observed_unix": time.time(),
                    "status": "transport_failed", "type": type(exc).__name__, "message": str(exc),
                    "automatic_retry": False})
        raise
    write(log, {"host": host, "action": action, "observed_unix": time.time(), "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr, "automatic_retry": False})
    if result.returncode:
        raise RuntimeError("Owned remote operation failed; readback preserved: " + str(log) + " " + result.stderr[-2000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def remote_prefix(c, host, *, idle=True, deployed=True):
    """Validate exact owner, runtime, GPU and package before any mutation."""
    return '''import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
c=%r;host=%r;s=c['hosts'][host];b=Path(s['root'])
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(1024**2),b''):h.update(v)
 return h.hexdigest()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True).strip().splitlines()
assert len(gpu)==1 and gpu[0].split(',')[0].strip()==s['gpu_uuid'],'GPU UUID mismatch'
assert s.get('gpu_name_contains',host) in gpu[0],'GPU name mismatch'
compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True).strip()
assert not %r or not compute,'GPU occupied; no process will be stopped'
ps=subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True)
owned=[line for line in ps.splitlines() if str(b) in line]
assert not %r or not owned,'Owned processes already present'
assert Path(s['python']).is_file() and os.access(s['python'],os.X_OK),'Native Python unavailable'
assert Path(s['tmux_binary']).is_file() and os.access(s['tmux_binary'],os.X_OK),'Native tmux unavailable'
assert shutil.which('timeout'),'Native timeout unavailable'
env={**os.environ,**s['environment']}
probe="import sys,torch,numpy,polars,json;print(json.dumps(dict(python=sys.executable,torch=str(torch.__version__),numpy=numpy.__version__,polars=polars.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())))"
runtime=json.loads(subprocess.check_output([s['python'],'-c',probe],env=env,text=True))
assert Path(runtime['python']).resolve()==Path(s['python']).resolve(),'Foreign Python'
assert all(runtime.get(k)==v for k,v in s['runtime_expected'].items()),'Native runtime changed: '+str(runtime)
if %r:
 assert json.loads((b/'execution_contract.json').read_text())==c,'Deployed contract changed'
 receipt=json.loads((b/'deployment_receipt.json').read_text());assert receipt['contract_sha256']==canonical(c),'Deployment receipt changed'
 manifest=json.loads((b/'package_manifest.json').read_text())
 for name,digest in manifest['files'].items():assert sha(b/name)==digest,'Package changed: '+name
 assert not (b/'failure.json').exists() and not (b/'qualification/failure.json').exists(),'Failure evidence exists; no automatic retry'
''' % (c, host, idle, idle, deployed)


def command(c, host, mode):
    spec = c["hosts"][host]
    return [spec["python"], spec["root"] + "/operation/campaign.py", "--contract",
            spec["root"] + "/execution_contract.json", "--host", host, "--mode", mode]


def shell_command(c, host, mode):
    spec = c["hosts"][host]
    root = spec["root"]
    cmd = command(c, host, mode)
    if mode == "qualify":
        cmd = ["timeout", "--signal=TERM", "--kill-after=15s", "5400", *cmd]
        exit_path, log = root + "/qualification_shell_exit.json", root + "/qualification.log"
    else:
        exit_path, log = root + "/supervisor_process_exit.json", root + "/supervisor.log"
    environment = " ".join(shlex.quote(str(k) + "=" + str(v)) for k, v in sorted(spec["environment"].items()))
    exit_code = "import json,sys,time;from pathlib import Path;p=Path(sys.argv[1]);f=p.open('x');json.dump({'returncode':int(sys.argv[2]),'finished_unix':time.time(),'automatic_retry':False},f);f.close()"
    trap_body = "rc=$?; trap - EXIT; " + shlex.join([spec["python"], "-c", exit_code, exit_path]) + ' "$rc"; exit "$rc"'
    # Keep a shell parent to preserve the native timeout/dispatch return code.
    return "trap " + shlex.quote(trap_body) + " EXIT; cd " + shlex.quote(spec["source_root"]) + " || exit $?; env " + environment + " " + shlex.join(cmd) + " > " + shlex.quote(log) + " 2>&1"


def verify_archive(archive):
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        names = [m.name for m in members]
        if len(set(names)) != len(names) or any(not m.isfile() or Path(m.name).is_absolute() or ".." in Path(m.name).parts for m in members):
            raise ValueError("Unsafe package member")
        if "package_manifest.json" not in names:
            raise ValueError("Package manifest missing")
        raw_manifest = stream.extractfile("package_manifest.json").read()
        manifest = json.loads(raw_manifest)
        if set(names) != set(manifest["files"]) | {"package_manifest.json"}:
            raise ValueError("Package member list differs from manifest")
        for name, digest in manifest["files"].items():
            if hashlib.sha256(stream.extractfile(name).read()).hexdigest() != digest:
                raise ValueError("Package file SHA mismatch: " + name)
        return hashlib.sha256(raw_manifest).hexdigest(), manifest


def deploy(host):
    c = contract(host)
    spec, archive = c["hosts"][host], B / "deployment" / ("package_" + host + ".tar.gz")
    manifest_sha, manifest = verify_archive(archive)
    root, remote_archive = spec["root"], spec["root"] + ".tar.gz"
    write(B / "deployment" / ("transfer_intent_" + host + ".json"),
          {"host": host, "root": root, "archive_sha256": sha(archive), "package_manifest_sha256": manifest_sha,
           "created_unix": time.time(), "automatic_retry": False})
    pre = ssh(host, remote_prefix(c, host, deployed=False) +
        "assert not b.exists() and not Path(%r).exists(),'Root/archive already exists'\nassert b.parent.is_dir(),'Artifact parent absent'\nprint(json.dumps({'GPU_UUID':s['gpu_uuid'],'gpu_idle':True,'root_absent':True,'runtime':runtime}))\n" % remote_archive,
        action="deploy_preflight")
    try:
        transfer = subprocess.run(["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", str(archive), host + ":" + remote_archive],
                                  text=True, capture_output=True, timeout=240)
        write(B / "deployment" / ("transfer_result_" + host + ".json"),
              {"host": host, "returncode": transfer.returncode, "stdout": transfer.stdout, "stderr": transfer.stderr,
               "finished_unix": time.time(), "automatic_retry": False})
        if transfer.returncode:
            raise RuntimeError("Owned package transfer failed; staging and intent preserved")
    except BaseException as exc:
        write(B / "deployment" / ("transfer_failure_" + host + ".json"),
              {"host": host, "type": type(exc).__name__, "message": str(exc), "automatic_retry": False})
        raise
    extraction = remote_prefix(c, host, deployed=False) + '''import tarfile
a=Path(%r);assert sha(a)==%r,'Transferred archive SHA mismatch'
assert not b.exists(),'Fresh campaign root required'
with tarfile.open(a,'r:gz') as t:
 members=t.getmembers();names=[m.name for m in members]
 assert len(set(names))==len(names) and all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members),'Unsafe archive'
 raw=t.extractfile('package_manifest.json').read();assert hashlib.sha256(raw).hexdigest()==%r,'Manifest SHA mismatch'
 manifest=json.loads(raw);assert set(names)==set(manifest['files'])|{'package_manifest.json'},'Unexpected package files'
 b.mkdir(mode=0o700)
 for member in members:
  target=b/member.name;target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as out:shutil.copyfileobj(t.extractfile(member),out)
for name,digest in manifest['files'].items():assert sha(b/name)==digest,'Extracted SHA mismatch: '+name
assert json.loads((b/'execution_contract.json').read_text())==c,'Foreign package contract'
receipt={'status':'SHA_verified','host':host,'archive_sha256':%r,'package_manifest_sha256':%r,'contract_sha256':canonical(c),'files':len(manifest['files']),'runtime':runtime,'gpu_uuid':s['gpu_uuid'],'gpu_idle':True,'completed_unix':time.time()}
with (b/'deployment_receipt.json').open('x') as f:json.dump(receipt,f)
print(json.dumps(receipt))
''' % (remote_archive, sha(archive), manifest_sha, sha(archive), manifest_sha)
    result = ssh(host, extraction, action="deploy_extract", timeout=240)
    write(B / "deployment" / ("deployment_receipt_" + host + ".json"), {**result, "preflight": pre})
    print(json.dumps(result))


def qualify(host):
    c = contract(host)
    spec = c["hosts"][host]
    session = spec["tmux"] + "_qualification"
    shell = shell_command(c, host, "qualify")
    write(B / "deployment" / ("qualification_intent_" + host + ".json"),
          {"host": host, "command": command(c, host, "qualify"), "created_unix": time.time(),
           "native_timeout_seconds": 5400, "scientific_training_launched": False, "automatic_retry": False})
    code = remote_prefix(c, host) + '''assert not (b/'qualification_start.json').exists() and not (b/'qualification/receipt.json').exists(),'Qualification attempt exists'
session=%r
p=subprocess.run([s['tmux_binary'],'has-session','-t',session],capture_output=True);assert p.returncode!=0,'Session already exists'
with (b/'qualification_start.json').open('x') as f:json.dump({'host':host,'session':session,'command':%r,'started_unix':time.time(),'automatic_retry':False,'scientific_training_launched':False},f)
subprocess.run([s['tmux_binary'],'new-session','-d','-s',session,'bash','-lc',%r],check=True)
print(json.dumps({'status':'qualification_started','host':host,'session':session,'gpu_uuid':s['gpu_uuid'],'started_unix':time.time(),'scientific_training_launched':False}))
''' % (session, command(c, host, "qualify"), shell)
    result = ssh(host, code, action="qualify_start")
    write(B / "deployment" / ("qualification_start_" + host + ".json"), result)
    print(json.dumps(result))


def collect(host):
    c = contract(host)
    root = c["hosts"][host]["root"]
    code = '''import hashlib,json,subprocess,time
from pathlib import Path
b=Path(%r);host=%r
def read_if(name):
 p=b/name;return json.loads(p.read_text()) if p.exists() else None
q=b/'qualification/receipt.json';log=b/'qualification.log'
gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True)
compute=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True)
ps=subprocess.run(['ps','-eo','pid,ppid,args'],capture_output=True,text=True)
out={'host':host,'observed_unix':time.time(),'receipt':read_if('qualification/receipt.json'),'receipt_json_text':q.read_text() if q.exists() else None,'receipt_sha256':hashlib.sha256(q.read_bytes()).hexdigest() if q.exists() else None,'qualification_failure':read_if('qualification/failure.json'),'qualification_shell_exit':read_if('qualification_shell_exit.json'),'supervisor_exit':read_if('supervisor_process_exit.json'),'status':read_if('status.json'),'failure':read_if('failure.json'),'GPU':gpu.stdout,'GPU_query_returncode':gpu.returncode,'gpu_processes':compute.stdout,'GPU_process_query_returncode':compute.returncode,'owned_processes':[line for line in ps.stdout.splitlines() if str(b) in line],'ps_returncode':ps.returncode,'log_tail':log.read_text()[-12000:] if log.exists() else ''}
print(json.dumps(out))
''' % (root, host)
    result = ssh(host, code, action="collect")
    dest = B / "qualification" / host
    write(dest / (str(time.time_ns()) + ".json"), result)
    if result["receipt"] is not None:
        receipt_path = dest / "receipt.json"
        raw = result["receipt_json_text"]
        if (json.loads(raw) != result["receipt"] or
                hashlib.sha256(raw.encode()).hexdigest() != result["receipt_sha256"]):
            raise ValueError("Qualification original receipt SHA/content mismatch")
        if receipt_path.exists():
            if receipt_path.read_text() != raw:
                raise ValueError("Remote qualification receipt changed; preserved snapshot requires review")
        else:
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            with receipt_path.open("x") as stream:
                stream.write(raw)
        if sha(receipt_path) != result["receipt_sha256"]:
            raise ValueError("Recovered qualification original receipt SHA mismatch")
    print(json.dumps({"host": host, "observed_unix": result["observed_unix"],
                      "qualification_status": result["receipt"].get("status") if result["receipt"] else "pending",
                      "qualification_shell_exit": result["qualification_shell_exit"],
                      "owned_processes": result["owned_processes"], "gpu_processes": result["gpu_processes"],
                      "failure": result["qualification_failure"] or result["failure"]}))


def qualified_receipts(c):
    operation = B / "operation" / "campaign.py"
    spec = importlib.util.spec_from_file_location("architecture_deploy_qualification_validator", operation)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate(c, verify_files=False)
    result = {}
    for host in HOSTS:
        receipt = read(B / "qualification" / host / "receipt.json")
        module.validate_qualification(c, host, receipt)
        result[host] = receipt
    return result


def launch(host):
    c = contract(host)
    spec = c["hosts"][host]
    receipts = qualified_receipts(c)
    start = read(B / "start_permit.json")
    permit = {"schema": SCHEMA, "host": host, "contract_sha256": canonical(c),
              "start_permit_sha256": canonical(start), "qualifications": receipts}
    write(B / ("training_permit_" + host + ".json"), permit)
    write(B / "deployment" / ("launch_intent_" + host + ".json"),
          {"host": host, "session": spec["tmux"], "created_unix": time.time(), "automatic_retry": False,
           "command": command(c, host, "dispatch"), "contract_sha256": canonical(c)})
    code = remote_prefix(c, host) + '''q=json.loads((b/'qualification/receipt.json').read_text());assert q==%r,'Own native qualification changed'
exit_receipt=json.loads((b/'qualification_shell_exit.json').read_text());assert exit_receipt['returncode']==0,'Qualification shell did not exit successfully'
assert not any((b/name).exists() for name in ('launch_intent.json','supervisor.claim','training_permit.json')),'Launch already attempted'
assert not (b/'claims').exists() or not any((b/'claims').iterdir()),'Fit claim already exists'
p=subprocess.run([s['tmux_binary'],'has-session','-t',s['tmux']],capture_output=True);assert p.returncode!=0,'Dispatch session already exists'
permit=%r
assert permit['contract_sha256']==canonical(c) and permit['start_permit_sha256']==canonical(json.loads((b/'start_permit.json').read_text())),'Permit binding changed'
with (b/'training_permit.json').open('x') as f:json.dump(permit,f)
with (b/'launch_intent.json').open('x') as f:json.dump({'host':host,'session':s['tmux'],'started_unix':time.time(),'automatic_retry':False},f)
subprocess.run([s['tmux_binary'],'new-session','-d','-s',s['tmux'],'bash','-lc',%r],check=True)
print(json.dumps({'status':'dispatcher_launched','host':host,'session':s['tmux'],'gpu_uuid':s['gpu_uuid'],'started_unix':time.time(),'training_entry_confirmed':False,'automatic_retry':False}))
''' % (receipts[host], permit, shell_command(c, host, "dispatch"))
    result = ssh(host, code, action="launch")
    write(B / "deployment" / ("launch_receipt_" + host + ".json"), result)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=HOSTS, required=True)
    parser.add_argument("--mode", choices=("deploy", "qualify", "collect", "launch"), required=True)
    args = parser.parse_args()
    {"deploy": deploy, "qualify": qualify, "collect": collect, "launch": launch}[args.mode](args.host)


if __name__ == "__main__":
    main()
