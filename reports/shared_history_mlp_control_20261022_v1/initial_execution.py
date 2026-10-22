"""Bounded, exclusive deployment and initial native qualification/fit controls.

No command runs on import. Every CLI action performs one bounded SSH call at
most. An unsuccessful attempt is retained and blocks subsequent remote work.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import inspect
import json
from pathlib import Path
import shlex
import time
import uuid

from paper.scripts import manage_shared_history_mlp_control as manager
from paper.scripts import advance_shared_component_ablation as boundary
from paper.scripts import advance_shared_history_mlp_control as controller

ROOT, BUNDLE, REPORT = manager.ROOT, manager.BUNDLE, manager.REPORT
STATE = REPORT / "initial_execution_v1"
SELF = "reports/shared_history_mlp_control_20261022_v1/initial_execution.py"
MODULE = "paper.scripts.manage_shared_history_mlp_control"
HOSTS = ("5080", "5090")
KEYS = ("pid", "start_ticks", "pgid", "argv", "cwd")
require, read, sha, write, canonical = (
    manager.require,
    manager.read,
    manager.sha,
    manager.write,
    manager.canonical,
)
transport, ssh_command, json_response = (
    boundary.transport,
    boundary.ssh_command,
    boundary.json_response,
)


def check_archive(archive_bytes, raw_sha, canonical_sha):
    """Validate the entire tar inventory before extracting any member."""
    import hashlib
    import io
    import json
    import tarfile
    from pathlib import PurePosixPath

    def demand(value, reason):
        if not value:
            raise RuntimeError(reason)

    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        demand(len(names) == len(set(names)), "Duplicate deployment member")
        for member in members:
            path = PurePosixPath(member.name)
            demand(
                not path.is_absolute()
                and ".." not in path.parts
                and str(path) == member.name
                and not member.issym()
                and not member.islnk(),
                "Unsafe deployment member",
            )
            demand(
                member.isfile()
                or (member.isdir() and member.name == "source/sample_data"),
                "Unsupported deployment member type",
            )
        demand("execution_contract.json" in names, "Missing deployment contract")
        raw = archive.extractfile("execution_contract.json").read()
        demand(hashlib.sha256(raw).hexdigest() == raw_sha, "Contract raw SHA changed")
        contract = json.loads(raw)
        digest = hashlib.sha256(
            json.dumps(
                contract,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        demand(digest == canonical_sha, "Canonical contract SHA changed")
        inventory = {
            **contract["source"]["files"],
            **contract["input_files"],
            **contract["legacy_files"],
        }
        expected = {"execution_contract.json", "source/sample_data"} | {
            "source/" + name for name in inventory
        }
        demand(set(names) == expected, "Deployment inventory differs")
        for name, digest in inventory.items():
            demand(
                hashlib.sha256(archive.extractfile("source/" + name).read()).hexdigest()
                == digest,
                "Deployment member SHA changed: " + name,
            )
        return contract


def pinned(path, digest):
    path = Path(path)
    require(
        path.is_file()
        and not path.is_symlink()
        and path.resolve().is_relative_to(ROOT.resolve())
        and sha(path) == digest,
        "Pinned local file changed: " + str(path),
    )


def verify_local(integrity_sha, authorization_sha, qa_sha, driver_sha):
    pinned(ROOT / SELF, driver_sha)
    pinned(BUNDLE / "integrity.json", integrity_sha)
    pinned(REPORT / "AUTHORIZATION.json", authorization_sha)
    pinned(REPORT / "INDEPENDENT_QA.json", qa_sha)
    integrity, auth, qa = (
        read(BUNDLE / "integrity.json"),
        read(REPORT / "AUTHORIZATION.json"),
        read(REPORT / "INDEPENDENT_QA.json"),
    )
    c = read(BUNDLE / "execution_contract.json")
    manager.validate(c)
    require(
        c.get("authorization_sha256") == authorization_sha,
        "Sealed authorization binding differs",
    )
    pinned(REPORT / "PRESERVATION.json", c.get("preservation_sha256"))
    require(
        c.get("implementation_qualification", {}).get("independent_qa_sha256") == qa_sha
        and c["implementation_qualification"].get("tests_receipt_sha256")
        == integrity.get("tests_receipt_sha256"),
        "Sealed implementation gate binding differs",
    )
    require(
        integrity.get("status") == "sealed"
        and integrity.get("contract_sha256") == sha(BUNDLE / "execution_contract.json")
        and integrity.get("canonical_contract_sha256") == canonical(c)
        and integrity.get("source_files_sha256") == c["source"]["files_sha256"]
        and integrity.get("archive_sha256") == sha(BUNDLE / "deployment.tar.gz")
        and integrity.get("independent_qa_sha256") == qa_sha,
        "Sealed integrity changed",
    )
    require(
        auth.get("status") == "approved"
        and auth.get("campaign") == manager.NAME
        and auth.get("remote_hosts") == list(HOSTS)
        and auth.get("new_fit_count") == 9
        and auth.get("existing_checkpoint_reuse") == 18
        and auth.get("held_out_test_evaluation") is False
        and auth.get("scientific_retry") is False
        and auth.get("old_files_overwrite") is False
        and auth.get("runtime_credentials_network_changes") is False,
        "Approved scope changed",
    )
    require(
        qa.get("status") == "PASS"
        and qa.get("independent_reviewer") is True
        and qa.get("files_reviewed_sha256", {}).get(SELF) == driver_sha,
        "Independent driver QA is missing",
    )
    for name, digest in qa["files_reviewed_sha256"].items():
        pinned(ROOT / name, digest)
    tests_path = Path(
        c["implementation_qualification"].get(
            "tests_receipt_path", REPORT / "LOCAL_TESTS.json"
        )
    )
    tests_sha = integrity.get("tests_receipt_sha256")
    if not tests_path.exists():
        candidates = [p for p in REPORT.rglob("*.json") if sha(p) == tests_sha]
        require(
            len(candidates) == 1, "Source-bound local test receipt missing or ambiguous"
        )
        tests_path = candidates[0]
    pinned(tests_path, tests_sha)
    tests = read(tests_path)
    require(
        tests.get("actual_os_returncode") == 0
        and tests.get("source_unchanged_during_tests") is True,
        "Local tests have not passed",
    )
    require(
        sha(REPORT / "SCIENCE_SPEC.json")
        == c["science_spec_sha256"]
        == auth["science_spec_sha256"],
        "Science changed",
    )
    for name, digest in c["source"]["files"].items():
        pinned(manager.below(ROOT, name), digest)
        require(
            qa["files_reviewed_sha256"].get(name, digest) == digest,
            "Scientific QA binding changed",
        )
    for name, digest in {**c["input_files"], **c["legacy_files"]}.items():
        pinned(manager.below(BUNDLE / "input_snapshot", name), digest)
    preservation = manager.verify_parent_preservation()
    require(bool(preservation["files"]), "Empty original evidence inventory")
    check_archive(
        (BUNDLE / "deployment.tar.gz").read_bytes(),
        integrity["contract_sha256"],
        canonical(c),
    )
    return c, integrity


def deployment_program(c, host, integrity):
    h = c["hosts"][host]
    return (
        "import hashlib,io,json,pathlib,sys,tarfile\n"
        + inspect.getsource(check_archive)
        + f"""
root=pathlib.Path({h["root"]!r})
payload=sys.stdin.buffer.read({(BUNDLE / "deployment.tar.gz").stat().st_size + 1!r})
if len(payload)!={(BUNDLE / "deployment.tar.gz").stat().st_size!r}:raise RuntimeError('Deployment byte size differs')
if hashlib.sha256(payload).hexdigest()!={integrity["archive_sha256"]!r}:raise RuntimeError('Deployment archive SHA differs')
c=check_archive(payload,{integrity["contract_sha256"]!r},{canonical(c)!r})
root.mkdir(parents=False,exist_ok=False)
with (root/'deployment.tar.gz').open('xb') as out:out.write(payload)
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:
 for member in archive.getmembers():
  target=root/member.name
  if member.isdir():target.mkdir(parents=True,exist_ok=True)
  else:
   target.parent.mkdir(parents=True,exist_ok=True)
   with target.open('xb') as out:out.write(archive.extractfile(member).read())
sys.path.insert(0,str(root/'source'))
from paper.scripts import manage_shared_history_mlp_control as m
from paper.scripts import run_shared_history_mlp_control as r
m.validate(c);m.verify_files(c,root/'source');m.guard(c,{host!r})
runtime=r._native_runtime(c,{host!r},'cuda:0')
print(json.dumps({{'status':'deployed','host':{host!r},'contract_sha256':m.canonical(c),'archive_sha256':{integrity["archive_sha256"]!r},'runtime':runtime,'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()}}),flush=True)
"""
    )


def launch_program(c, host, phase):
    require(phase in ("qualification", "fit_validation"), "Unsupported initial phase")
    h = c["hosts"][host]
    expected = controller.expected_command(c, host, phase)
    return f"""import json,pathlib,shlex,subprocess,time
from paper.scripts import manage_shared_history_mlp_control as m
root=pathlib.Path({h["root"]!r});host={host!r};phase={phase!r};expected={expected!r};keys={KEYS!r}
c=m.read(root/'execution_contract.json');m.require(m.canonical(c)=={canonical(c)!r},'Contract changed')
m.verify_files(c,root/'source');m.guard(c,host)
m.require(not any(root.glob('*.failure.json')),'Preserved native failure')
m.require(not any(root.glob(phase+'.*')),'Existing native attempt; preserve')
if phase=='fit_validation':m.verify_phase_receipt(c,host,'qualification',root)
h=c['hosts'][host]
command=['env',*[k+'='+str(v) for k,v in h['environment'].items()],*[('supervise' if item=='dispatch' else item) for item in expected]]
result=subprocess.run([h['tmux_binary'],'new-session','-d','-s',h['tmux']+'_'+phase,'-c',h['source_root'],shlex.join(command)],capture_output=True,timeout=15)
m.require(result.returncode==0,'Tmux failed: '+repr(result.stderr))
deadline=time.monotonic()+10
while not ((root/(phase+'.claim.json')).exists() and (root/(phase+'.supervisor.intent.json')).exists()) and time.monotonic()<deadline:time.sleep(.05)
claim=m.read(root/(phase+'.claim.json'));parent=m.read(root/(phase+'.supervisor.intent.json'))
m.require(claim.get('contract_sha256')==m.canonical(c) and parent.get('contract_sha256')==m.canonical(c),'Launch contract differs')
m.require(claim['identity']['argv']==expected and parent['identity']['argv']==[('supervise' if x=='dispatch' else x) for x in expected],'Launch argv differs')
for identity in (claim['identity'],parent['identity']):
 m.require(identity['cwd']==h['source_root'],'Launch cwd differs')
 live=m.proc_identity(identity['pid'])
 if live is not None:m.require(all(live[k]==identity[k] for k in keys),'Launch identity race')
 else:m.require((root/(phase+'.supervisor_exit.json')).exists(),'Launch process disappeared without exit proof')
print(json.dumps({{'status':'launched','host':host,'phase':phase,'contract_sha256':m.canonical(c),'identity':claim['identity'],'supervisor_identity':parent['identity'],'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'observed_at':time.time()}}),flush=True)
"""


def qualification_program(c, host, launch):
    program = controller.terminal_program(c, host, "qualification", launch)
    program = program.replace(
        "print(json.dumps(result,allow_nan=False),flush=True)",
        """
if result['status']=='terminal':
 import base64
 paths={'execution_contract.json','qualification.complete.json','qualification.claim.json','qualification.supervisor.intent.json','qualification.supervisor_exit.json','qualification.stdout.log','qualification.stderr.log'}
 for entry in complete['actual_child_exits']:
  for suffix in ('exit.json','stdout.log','stderr.log'):
   paths.add('control/'+entry['job_id']+'.'+entry['stage']+'.'+suffix)
 paths.update(item['path'] for item in complete['capacity_receipts'].values())
 evidence={};total=0
 for name in sorted(paths):
  p=m.below(root,name);m.require(p.is_file() and not p.is_symlink(),'Invalid native evidence')
  raw=p.read_bytes();total+=len(raw);m.require(total<=50*1024*1024,'Qualification proof exceeds bound')
  evidence[name]={'base64':base64.b64encode(raw).decode(),'sha256':m.sha(p),'bytes':len(raw)}
 result['evidence']=evidence
print(json.dumps(result,allow_nan=False),flush=True)
""",
    )
    return program


def verify_pass(c, host):
    path = STATE / host / "QUALIFICATION_PASS.json"
    value = read(path)
    require(
        value.get("status") == "PASS"
        and value.get("contract_sha256") == canonical(c)
        and value.get("host") == host,
        "Local qualification binding differs",
    )
    snapshot = manager.below(STATE / host, value["snapshot"])
    require(bool(value.get("files")), "Empty local qualification inventory")
    actual = {str(p.relative_to(snapshot)) for p in snapshot.rglob("*") if p.is_file()}
    require(actual == set(value["files"]), "Qualification inventory changed")
    for name, digest in value["files"].items():
        target = manager.below(snapshot, name)
        require(
            not target.is_symlink() and sha(target) == digest,
            "Qualification raw proof changed",
        )
    manager.verify_phase_receipt(c, host, "qualification", snapshot)
    return value


def _action(name, host, *, integrity_sha, authorization_sha, qa_sha, driver_sha):
    require(
        name in ("deploy", "launch-qualification", "check-qualification", "launch-fit")
        and host in HOSTS,
        "Unsupported action or host",
    )
    c, integrity = verify_local(integrity_sha, authorization_sha, qa_sha, driver_sha)
    STATE.mkdir(parents=True, exist_ok=True)
    require(
        not any(STATE.rglob("FAILURE.json")),
        "Preserved failure blocks all additional remote work",
    )
    require(
        all(
            (path.parent / "RESULT.json").is_file()
            for path in STATE.rglob("ATTEMPT.json")
        ),
        "Incomplete earlier attempt is preserved; no additional remote work",
    )
    host_root = STATE / host
    host_root.mkdir(exist_ok=True)
    if name == "check-qualification":
        rounds = sorted((host_root / name).glob("*/RESULT.json"))
        if rounds:
            latest = read(rounds[-1])
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
            "started_unix": time.time(),
        },
    )
    try:
        if name == "deploy":
            program = deployment_program(c, host, integrity)
            h = c["hosts"][host]
            remote = shlex.join(
                [
                    "env",
                    *[k + "=" + str(v) for k, v in h["environment"].items()],
                    h["python"],
                    "-u",
                    "-c",
                    program,
                ]
            )
            command = [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=15",
                host,
                remote,
            ]
            _, raw = transport(
                command,
                folder,
                "transport",
                timeout=900,
                input_bytes=(BUNDLE / "deployment.tar.gz").read_bytes(),
            )
        else:
            require(
                read(host_root / "deploy/RESULT.json").get("status") == "deployed",
                "Native deployment not verified",
            )
            if name == "launch-fit":
                for peer in HOSTS:
                    verify_pass(c, peer)
                phase = "fit_validation"
            else:
                phase = "qualification"
            if name == "check-qualification":
                launch = read(host_root / "launch-qualification/RESULT.json")
                controller.validate_launch(c, host, phase, launch)
                program = qualification_program(c, host, launch)
            else:
                program = launch_program(c, host, phase)
            _, raw = transport(
                ssh_command(c, host, program), folder, "transport", timeout=45
            )
        value = json_response(raw)
        require(
            value.get("host") == host
            and value.get("contract_sha256") == canonical(c)
            and value.get("boot_id", c["hosts"][host]["boot_id"])
            == c["hosts"][host]["boot_id"],
            "Remote response identity differs",
        )
        if name.startswith("launch-"):
            require(value.get("status") == "launched", "Launch not confirmed")
            controller.validate_launch(c, host, phase, value)
        elif name == "deploy":
            require(
                value.get("status") == "deployed"
                and value.get("archive_sha256") == integrity["archive_sha256"],
                "Deployment not confirmed",
            )
        else:
            require(
                value.get("status") in ("waiting", "terminal")
                and value.get("read_only_check") is True,
                "Unknown qualification status",
            )
            value["checked_unix"] = time.time()
            if value["status"] == "terminal":
                evidence = value.get("evidence", {})
                require(bool(evidence), "No native raw proof")
                snapshot = folder / "snapshot"
                snapshot.mkdir()
                for entry, item in evidence.items():
                    path = manager.below(snapshot, entry)
                    raw_bytes = base64.b64decode(item["base64"], validate=True)
                    require(
                        len(raw_bytes) == item["bytes"]
                        and hashlib.sha256(raw_bytes).hexdigest() == item["sha256"],
                        "Raw native proof differs",
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
                "remote_retry_authorized": False,
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
        required=True,
        choices=("deploy", "launch-qualification", "check-qualification", "launch-fit"),
    )
    parser.add_argument("--host", required=True, choices=HOSTS)
    for field in ("integrity", "authorization", "qa", "driver"):
        parser.add_argument("--" + field + "-sha", required=True)
    args = vars(parser.parse_args(argv))
    name, host = args.pop("action"), args.pop("host")
    print(json.dumps(action(name, host, **args), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
