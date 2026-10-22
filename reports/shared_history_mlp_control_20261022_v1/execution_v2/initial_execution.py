"""Bounded initial deployment/qualification for the metadata-only execution v2.

Uses explicit versioned paths and a new raw contract SHA. Frozen module globals,
scientific sources, original attempts and host deadlines are never rewritten.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import shlex
import sys
import time
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper.scripts import manage_shared_history_mlp_control as manager  # noqa: E402
from paper.scripts import advance_shared_component_ablation as boundary  # noqa: E402

ROOT = manager.ROOT
REPORT = manager.REPORT / "execution_v2"
BUNDLE = manager.BUNDLE / "execution_v2"
STATE = REPORT / "initial_execution_v1"
SELF = (
    "reports/shared_history_mlp_control_20261022_v1/execution_v2/initial_execution.py"
)
KEYS = ("pid", "start_ticks", "pgid", "argv", "cwd")
HOSTS = ("5080", "5090")
MODULE = "paper.scripts.manage_shared_history_mlp_control"
require, read, sha, write, canonical = (
    manager.require,
    manager.read,
    manager.sha,
    manager.write,
    manager.canonical,
)


def preparation():
    spec = importlib.util.spec_from_file_location(
        "metadata_execution_v2_preparation", REPORT / "prepare_execution.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_local(admission_sha, integrity_sha, qa_sha, self_sha):
    qa_path = REPORT / "INDEPENDENT_QA.json"
    require(
        qa_path.is_file() and not qa_path.is_symlink() and sha(qa_path) == qa_sha,
        "Execution-v2 QA raw SHA differs",
    )
    reviewed = read(qa_path).get("files_reviewed_sha256", {})
    prepare_name = "reports/shared_history_mlp_control_20261022_v1/execution_v2/prepare_execution.py"
    require(
        sha(ROOT / SELF) == self_sha == reviewed.get(SELF)
        and sha(ROOT / prepare_name) == reviewed.get(prepare_name),
        "Execution-v2 reviewed local control changed",
    )
    p = preparation()
    c, integrity = p.verify_sealed(admission_sha, integrity_sha, qa_sha)
    require(
        sha(ROOT / SELF) == self_sha
        and read(REPORT / "INDEPENDENT_QA.json")["files_reviewed_sha256"].get(SELF)
        == self_sha,
        "Initial execution-v2 source changed",
    )
    return c, integrity, p.original_helper()


def expected_command(c, host, phase, contract_file_sha256):
    require(phase in ("qualification", "fit_validation", "cost"), "Test is forbidden")
    h = c["hosts"][host]
    return [
        h["python"],
        "-u",
        "-m",
        MODULE,
        "--mode",
        "dispatch",
        "--contract",
        h["root"] + "/execution_contract.json",
        "--contract-sha",
        contract_file_sha256,
        "--host",
        host,
        "--phase",
        phase,
    ]


def validate_launch(c, host, phase, value, contract_file_sha256):
    command = expected_command(c, host, phase, contract_file_sha256)
    require(
        value.get("host") == host
        and value.get("phase") == phase
        and value.get("contract_sha256") == canonical(c)
        and boundary.valid_identity(value.get("identity"))
        and boundary.valid_identity(value.get("supervisor_identity")),
        "Actual launch identities missing",
    )
    require(
        value["identity"]["argv"] == command
        and value["supervisor_identity"]["argv"]
        == [("supervise" if x == "dispatch" else x) for x in command]
        and all(
            i["cwd"] == c["hosts"][host]["source_root"]
            for i in (value["identity"], value["supervisor_identity"])
        ),
        "Execution-v2 command/source identity differs",
    )


def deployment_program(c, host, integrity, initial):
    h = c["hosts"][host]
    size = (BUNDLE / "deployment.tar.gz").stat().st_size
    return (
        "import hashlib,importlib,io,json,pathlib,sys,tarfile\n"
        + inspect.getsource(initial.check_archive)
        + f"""
root=pathlib.Path({h["root"]!r});payload=sys.stdin.buffer.read({size + 1!r})
if len(payload)!={size!r} or hashlib.sha256(payload).hexdigest()!={integrity["archive_sha256"]!r}:raise RuntimeError('Deployment bytes/SHA differ')
c=check_archive(payload,{integrity["contract_sha256"]!r},{canonical(c)!r})
root.mkdir(parents=False,exist_ok=False)
with (root/'deployment.tar.gz').open('xb') as stream:stream.write(payload)
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:
 for member in archive.getmembers():
  path=root/member.name
  if member.isdir():path.mkdir(parents=True,exist_ok=True)
  else:
   path.parent.mkdir(parents=True,exist_ok=True)
   with path.open('xb') as stream:stream.write(archive.extractfile(member).read())
sys.path.insert(0,str(root/'source'));importlib.invalidate_caches()
from paper.scripts import manage_shared_history_mlp_control as m
from paper.scripts import run_shared_history_mlp_control as r
m.validate(c);m.verify_files(c,root/'source');m.guard(c,{host!r})
runtime=r._native_runtime(c,{host!r},'cuda:0')
print(json.dumps({{'status':'deployed','host':{host!r},'execution_version':2,'contract_sha256':m.canonical(c),'contract_file_sha256':{integrity["contract_sha256"]!r},'archive_sha256':{integrity["archive_sha256"]!r},'runtime':runtime,'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()}}),flush=True)
"""
    )


def launch_program(c, host, phase, contract_file_sha256):
    command = expected_command(c, host, phase, contract_file_sha256)
    h = c["hosts"][host]
    return f"""import json,pathlib,shlex,subprocess,time
from paper.scripts import manage_shared_history_mlp_control as m
root=pathlib.Path({h["root"]!r});host={host!r};phase={phase!r};expected={command!r};keys={KEYS!r}
c=m.read(root/'execution_contract.json');m.require(m.canonical(c)=={canonical(c)!r} and m.sha(root/'execution_contract.json')=={contract_file_sha256!r},'Contract identity changed')
m.verify_files(c,root/'source');m.guard(c,host)
m.require(not any(root.glob('*.failure.json')) and not any(root.glob(phase+'.*')),'Existing native attempt/failure; preserve')
if phase=='fit_validation':m.verify_phase_receipt(c,host,'qualification',root)
h=c['hosts'][host]
args=['env',*[key+'='+str(value) for key,value in h['environment'].items()],*[('supervise' if x=='dispatch' else x) for x in expected]]
result=subprocess.run([h['tmux_binary'],'new-session','-d','-s',h['tmux']+'_'+phase,'-c',h['source_root'],shlex.join(args)],capture_output=True,timeout=15)
m.require(result.returncode==0,'Tmux launch failed: '+repr(result.stderr))
deadline=time.monotonic()+10
while not ((root/(phase+'.claim.json')).exists() and (root/(phase+'.supervisor.intent.json')).exists()) and time.monotonic()<deadline:time.sleep(.05)
claim=m.read(root/(phase+'.claim.json'));parent=m.read(root/(phase+'.supervisor.intent.json'))
m.require(claim.get('contract_sha256')==m.canonical(c) and parent.get('contract_sha256')==m.canonical(c),'Claim contract differs')
m.require(claim['identity']['argv']==expected and parent['identity']['argv']==[('supervise' if x=='dispatch' else x) for x in expected],'Actual command differs')
for identity in (claim['identity'],parent['identity']):
 m.require(identity['cwd']==h['source_root'],'Native cwd differs')
 live=m.proc_identity(identity['pid'])
 if live is not None:m.require(all(live[k]==identity[k] for k in keys),'Native identity race')
 else:m.require((root/(phase+'.supervisor_exit.json')).exists(),'New process vanished without exit proof')
print(json.dumps({{'status':'launched','host':host,'phase':phase,'execution_version':2,'contract_sha256':m.canonical(c),'contract_file_sha256':{contract_file_sha256!r},'identity':claim['identity'],'supervisor_identity':parent['identity'],'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'observed_at':time.time()}}),flush=True)
"""


def verify_pass(c, host):
    value = read(STATE / host / "QUALIFICATION_PASS.json")
    require(
        value.get("status") == "PASS"
        and value.get("host") == host
        and value.get("contract_sha256") == canonical(c),
        "Qualification identity differs",
    )
    snapshot = manager.below(STATE / host, value["snapshot"])
    files = value.get("files", {})
    paths = list(snapshot.rglob("*"))
    require(
        bool(files)
        and not snapshot.is_symlink()
        and not any(p.is_symlink() for p in paths)
        and {str(p.relative_to(snapshot)) for p in paths if p.is_file()} == set(files),
        "Qualification inventory differs",
    )
    for name, digest in files.items():
        require(
            sha(manager.below(snapshot, name)) == digest, "Qualification bytes changed"
        )
    complete, _ = manager.verify_phase_receipt(c, host, "qualification", snapshot)
    require(
        set(complete.get("capacity_receipts", {})) == set(manager.ARMS),
        "All three capacity runtime proofs required",
    )
    for item in complete["capacity_receipts"].values():
        runtime = read(manager.below(snapshot, item["path"])).get("runtime", {})
        require(
            all(
                runtime.get(key) == value
                for key, value in c["hosts"][host]["runtime_expected"].items()
            )
            and runtime.get("gpu", {}).get("uuid") == c["hosts"][host]["gpu_uuid"],
            "Full native runtime/GPU proof differs",
        )
    return value


def _action(name, host, *, admission_sha, integrity_sha, qa_sha, self_sha):
    require(
        name in ("deploy", "launch-qualification", "check-qualification", "launch-fit")
        and host in HOSTS,
        "Unsupported initial v2 action; Test forbidden",
    )
    c, integrity, initial = verify_local(admission_sha, integrity_sha, qa_sha, self_sha)
    require(
        not any(STATE.rglob("FAILURE.json")),
        "Preserved failure blocks additional remote calls",
    )
    require(
        all((p.parent / "RESULT.json").is_file() for p in STATE.rglob("ATTEMPT.json")),
        "Partial attempt blocks remote calls",
    )
    host_root = STATE / host
    host_root.mkdir(parents=True, exist_ok=True)
    if name == "check-qualification":
        previous = sorted((host_root / name).glob("*/RESULT.json"))
        if previous:
            latest = read(previous[-1])
            require(latest.get("status") == "waiting", "Qualification already terminal")
            if time.time() - latest["checked_unix"] < 900:
                return {"status": "waiting_next_check", "ssh_calls": 0}
        folder = host_root / name / (str(time.time_ns()) + "_" + uuid.uuid4().hex[:8])
    else:
        folder = host_root / name
    folder.mkdir(parents=True, exist_ok=False)
    write(
        folder / "ATTEMPT.json",
        {
            "action": name,
            "host": host,
            "contract_sha256": canonical(c),
            "execution_version": 2,
            "admission_sha256": admission_sha,
            "integrity_sha256": integrity_sha,
        },
    )
    try:
        h = c["hosts"][host]
        raw_sha = integrity["contract_sha256"]
        phase = "fit_validation" if name == "launch-fit" else "qualification"
        if name == "deploy":
            program = deployment_program(c, host, integrity, initial)
            timeout, payload = 900, (BUNDLE / "deployment.tar.gz").read_bytes()
        else:
            for peer in HOSTS:
                deployed = read(STATE / peer / "deploy/RESULT.json")
                require(
                    deployed.get("status") == "deployed"
                    and deployed.get("contract_sha256") == canonical(c),
                    "Both new deployments required",
                )
            if name == "launch-fit":
                for peer in HOSTS:
                    verify_pass(c, peer)
            if name == "check-qualification":
                launch = read(host_root / "launch-qualification/RESULT.json")
                validate_launch(c, host, phase, launch, raw_sha)
                program = initial.qualification_program(c, host, launch)
            else:
                program = launch_program(c, host, phase, raw_sha)
            timeout, payload = 45, None
        compile(program, "<native-execution-v2>", "exec")
        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            host,
            shlex.join(
                [
                    "env",
                    *[
                        key + "=" + str(value)
                        for key, value in h["environment"].items()
                    ],
                    h["python"],
                    "-u",
                    "-c",
                    program,
                ]
            ),
        ]
        _, raw = boundary.transport(
            command, folder, "transport", timeout=timeout, input_bytes=payload
        )
        value = boundary.json_response(raw)
        require(
            value.get("host") == host
            and value.get("contract_sha256") == canonical(c)
            and value.get("boot_id", h["boot_id"]) == h["boot_id"],
            "Native v2 response identity differs",
        )
        if name == "deploy":
            require(
                value.get("status") == "deployed"
                and value.get("archive_sha256") == integrity["archive_sha256"],
                "Deployment not verified",
            )
        elif name.startswith("launch-"):
            require(value.get("status") == "launched", "Launch not verified")
            validate_launch(c, host, phase, value, raw_sha)
        else:
            require(
                value.get("status") in ("waiting", "terminal")
                and value.get("read_only_check") is True,
                "Unknown qualification status",
            )
            value["checked_unix"] = time.time()
            if value["status"] == "terminal":
                evidence = value.get("evidence", {})
                require(bool(evidence), "Native proof is empty")
                snapshot = folder / "snapshot"
                snapshot.mkdir()
                for entry, item in evidence.items():
                    path = manager.below(snapshot, entry)
                    raw_bytes = base64.b64decode(item["base64"], validate=True)
                    require(
                        len(raw_bytes) == item["bytes"]
                        and hashlib.sha256(raw_bytes).hexdigest() == item["sha256"],
                        "Raw proof changed",
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("xb") as stream:
                        stream.write(raw_bytes)
                manager.verify_phase_receipt(c, host, "qualification", snapshot)
                write(
                    host_root / "QUALIFICATION_PASS.json",
                    {
                        "status": "PASS",
                        "host": host,
                        "contract_sha256": canonical(c),
                        "snapshot": str(snapshot.relative_to(host_root)),
                        "files": {
                            entry: item["sha256"] for entry, item in evidence.items()
                        },
                    },
                )
                verify_pass(c, host)
        write(folder / "RESULT.json", value)
        return value
    except Exception as error:
        write(
            folder / "FAILURE.json",
            {
                "error_type": type(error).__name__,
                "message": str(error),
                "retry_authorized": False,
            },
        )
        raise


def action(name, host, **pins):
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "initial.lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "busy", "ssh_calls": 0}
        try:
            return _action(name, host, **pins)
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=("deploy", "launch-qualification", "check-qualification", "launch-fit"),
        required=True,
    )
    parser.add_argument("--host", choices=HOSTS, required=True)
    for field in ("admission", "integrity", "qa", "self"):
        parser.add_argument("--" + field + "-sha", required=True)
    args = vars(parser.parse_args(argv))
    name, host = args.pop("action"), args.pop("host")
    print(json.dumps(action(name, host, **args), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
