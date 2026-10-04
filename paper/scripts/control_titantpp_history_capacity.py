"""Explicit operations for the approved, isolated dual-server capacity study.

No import-time network/launch. Each action creates an exclusive intent and
does not automatically retry an ambiguous deployment, qualification or start.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.scripts import observed_slot_parallel_common as common
from paper.scripts.prepare_titantpp_history_capacity import NAME

B = ROOT / "search_artifacts" / NAME


def read(path):
    return json.loads(Path(path).read_text())


def load():
    c = read(B / "execution_contract.json")
    a, s, pointer = (read(B / n) for n in ("approval.json", "start_permit.json", "current.json"))
    digest = common.sha_json(c)
    common.require(c["schema"] == NAME and a["approved"] is True and a["user_instruction"], "Missing scoped authority")
    common.require(a["contract_sha256"] == s["contract_sha256"] == pointer["contract_sha256"] == digest, "Changed current contract")
    common.require(s["approval_sha256"] == common.sha_json(a) and s["started_at_unix"] <= time.time() < s["deadline_unix"], "Expired or foreign authority")
    common.require(c["source"]["files_sha256"] == common.sha_json(c["source"]["files"]), "Bad source closure")
    for rel, digest in c["source"]["files"].items():
        common.require(common.sha_file(B / "frozen_source" / rel) == digest, "Source changed: " + rel)
    common.require(pointer["hosts"] == {h: v["root"] for h, v in c["hosts"].items()}, "Changed target")
    return c


def ssh(host, script, timeout=60):
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, "python3 -"], input=script, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr + result.stdout)
    return result.stdout


def deploy(c, host):
    spec = c["hosts"][host]
    common.write_json(B / "deployment" / (host + "_intent.json"), {"host": host, "contract_sha256": common.sha_json(c), "requested_unix": time.time()}, exclusive=True)
    files = {"source/" + rel: B / "frozen_source" / rel for rel in c["source"]["files"]}
    files["source/sample_data/.keep"] = B / "frozen_source/sample_data/.keep"
    for name in ("execution_contract.json", "approval.json", "start_permit.json", "design.json", "README.md"):
        files[name] = B / name
    for subdir in (("data",) if host == "5080" else ("references",)):
        for p in (B / subdir).rglob("*"):
            if p.is_file():
                files[str(p.relative_to(B))] = p
    manifest = {rel: common.sha_file(p) for rel, p in files.items()}
    manifest_path = B / "deployment" / (host + "_manifest.json")
    common.write_json(manifest_path, manifest, exclusive=True)
    files["deployment_manifest.json"] = manifest_path
    archive = B / "deployment" / (host + ".tar.gz")
    with tarfile.open(archive, "w:gz") as stream:
        for rel, path in sorted(files.items()):
            stream.add(path, arcname=rel)
    ssh(host, "from pathlib import Path\np=Path(" + repr(spec["root"]) + ")\np.mkdir(exist_ok=False)\nprint(p)\n")
    subprocess.run(["scp", "-q", str(archive), host + ":" + spec["root"] + "/deployment.tar.gz"], check=True)
    script = """import json,tarfile,hashlib
from pathlib import Path
p=Path(ROOT)
with tarfile.open(p/'deployment.tar.gz') as stream:
 for m in stream.getmembers():
  assert not m.issym() and not m.islnk() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
 stream.extractall(p,filter='data')
manifest=json.loads((p/'deployment_manifest.json').read_text())
for rel,digest in manifest.items():assert hashlib.sha256((p/rel).read_bytes()).hexdigest()==digest,rel
print(json.dumps({'status':'passed','files':len(manifest),'root':str(p)}))
""".replace("ROOT", repr(spec["root"]))
    receipt = json.loads(ssh(host, script))
    common.write_json(B / "deployment" / (host + "_receipt.json"), {**receipt, "archive_sha256": common.sha_file(archive), "contract_sha256": common.sha_json(c)}, exclusive=True)
    print(json.dumps({"host": host, **receipt}), flush=True)


def command(c, host, mode):
    spec = c["hosts"][host]
    seconds = 5400 if mode == "qualify" else max(1, int(read(B / "start_permit.json")["deadline_unix"] - time.time()))
    args = ["timeout", "--signal=TERM", "--kill-after=15s", str(seconds), "env",
            *(k + "=" + v for k, v in spec["environment"].items()),
            "MPLCONFIGDIR=/tmp/titantpp-capacity-mpl", spec["python"],
            "paper/scripts/run_titantpp_history_capacity_campaign.py", "--contract", spec["root"] + "/execution_contract.json",
            "--host", host, "--mode", mode]
    return "cd " + shlex.quote(spec["source_root"]) + " && exec " + shlex.join(args)


def qualify(c, host):
    dest = B / "native" / host
    common.write_json(dest / "request.json", {"host": host, "contract_sha256": common.sha_json(c), "requested_unix": time.time()}, exclusive=True)
    with (dest / "qualification.log").open("xb") as stream:
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, command(c, host, "qualify")], stdout=stream, stderr=subprocess.STDOUT)
    common.write_json(dest / "process_exit.json", {"returncode": result.returncode, "finished_unix": time.time()}, exclusive=True)
    if result.returncode:
        print((dest / "qualification.log").read_text()[-6500:])
        raise SystemExit(result.returncode)
    subprocess.run(["scp", "-q", host + ":" + c["hosts"][host]["root"] + "/qualification/receipt.json", str(dest / "receipt.json")], check=True)
    print(json.dumps({"host": host, "status": "passed", "receipt": str(dest / "receipt.json")}), flush=True)


def launch(c, host):
    spec = c["hosts"][host]
    q, start = read(B / "native" / host / "receipt.json"), read(B / "start_permit.json")
    common.require(q["status"] == "passed" and q["contract_sha256"] == common.sha_json(c) and q["host"] == host, "Native qualification missing")
    permit = {"schema": "titantpp_history_capacity_training_permit_v1", "host": host,
        "contract_sha256": common.sha_json(c), "start_permit_sha256": common.sha_json(start), "qualifications": {host: q}}
    permit_path = B / "launch" / (host + "_training_permit.json")
    common.write_json(permit_path, permit, exclusive=True)
    common.write_json(B / "launch" / (host + "_intent.json"), {"host": host, "contract_sha256": common.sha_json(c), "requested_unix": time.time()}, exclusive=True)
    subprocess.run(["scp", "-q", str(permit_path), host + ":" + spec["root"] + "/training_permit.json"], check=True)
    launch_command = command(c, host, "dispatch") + " > " + shlex.quote(spec["root"] + "/supervisor.log") + " 2>&1"
    script = """import json,subprocess
from pathlib import Path
root=Path(ROOT)
assert not (root/'supervisor.claim').exists() and not (root/'run').exists()
assert subprocess.run([TMUX,'has-session','-t',SESSION],capture_output=True).returncode!=0
subprocess.run([TMUX,'new-session','-d','-s',SESSION,COMMAND],check=True)
print(json.dumps({'status':'launch_requested','root':str(root),'session':SESSION}))
""".replace("ROOT", repr(spec["root"])).replace("TMUX", repr(spec["tmux_binary"])).replace("SESSION", repr(spec["tmux"])).replace("COMMAND", repr(launch_command))
    receipt = json.loads(ssh(host, script))
    common.write_json(B / "launch" / (host + "_receipt.json"), {"host": host, "contract_sha256": common.sha_json(c), "response": receipt, "confirmed_training_started": False, "time": time.time()}, exclusive=True)
    print(json.dumps({"host": host, **receipt}), flush=True)


def observe(c, host):
    script = """import json,time,subprocess
from pathlib import Path
p=Path(ROOT)
result={'observed_unix':time.time(),'files':{}}
for name in ['status.json','progress.json','failure.json','server_lease.json','supervisor.claim','qualification/receipt.json']:
 f=p/name
 if f.exists():result['files'][name]=json.loads(f.read_text())
for f in sorted((p/'run').glob('*/status.json')):
 result['files'][str(f.relative_to(p))]=json.loads(f.read_text())
for f in sorted((p/'run').glob('*/runs/*/*/seed_*/history.json')):
 data=json.loads(f.read_text());rows=data.get('history',[])
 result['files'][str(f.relative_to(p))]={'count':len(rows),'last':rows[-1:]}
result['gpu']=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout
result['processes']=[x for x in subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True).stdout.splitlines() if str(p) in x and 'python3 -' not in x]
print(json.dumps(result))
""".replace("ROOT", repr(c["hosts"][host]["root"]))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = B / "observations" / stamp / (host + ".json")
    result = json.loads(ssh(host, script))
    common.write_json(path, result, exclusive=True)
    print(json.dumps({"host": host, "path": str(path), "status": result["files"].get("status.json"), "gpu": result["gpu"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("deploy", "qualify", "launch", "observe"))
    parser.add_argument("--host", choices=("5080", "5090"), required=True)
    args = parser.parse_args()
    globals()[args.action](load(), args.host)
