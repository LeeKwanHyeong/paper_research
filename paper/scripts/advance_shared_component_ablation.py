"""Bounded local continuation of the approved, already-launched A--D study.

This controller never starts training.  It adopts the two recorded fit/Validation
dispatchers, collects immutable terminal evidence, then admits the fixed Test/cost
phase only after the global 36-job Validation gate.  Any failed transport or
partial action is preserved and blocks further remote calls.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import uuid

from paper.scripts import manage_shared_component_ablation as manager

ROOT, BUNDLE = manager.ROOT, manager.BUNDLE
REPORT = manager.REPORT / "continuation_v1"
HOSTS = ("5080", "5090")
PHASES = ("fit_validation", "test_cost")
IDENTITY_KEYS = ("pid", "start_ticks", "pgid", "argv", "cwd")
CHECK_SECONDS, INTERVAL_SECONDS, TRANSFER_SECONDS = 45, 900, 900
CONTROLLER = "paper/scripts/advance_shared_component_ablation.py"
COMPARATOR = "paper/scripts/compare_shared_component_ablation.py"
read, sha, write, canonical, require = manager.read, manager.sha, manager.write, manager.canonical, manager.require


def now():
    return datetime.now(timezone.utc).isoformat()


def identity_equal(first, second):
    return isinstance(first, dict) and isinstance(second, dict) and all(
        first.get(key) == second.get(key) for key in IDENTITY_KEYS)


def valid_identity(value):
    return (isinstance(value, dict) and all(key in value for key in IDENTITY_KEYS)
        and type(value["pid"]) is int and value["pid"] > 0
        and type(value["start_ticks"]) is int and value["start_ticks"] > 0
        and type(value["pgid"]) is int and isinstance(value["argv"], list)
        and bool(value["argv"]) and isinstance(value["cwd"], str))


def verify(binding_sha, authorization_sha):
    binding_path, auth_path = REPORT / "CONTROL_BINDINGS.json", REPORT / "AUTHORIZATION.json"
    require(sha(binding_path) == binding_sha, "Continuation binding SHA differs")
    binding = read(binding_path)
    require(binding.get("campaign") == manager.NAME and binding.get("report_root") == str(REPORT)
        and binding.get("bundle_root") == str(BUNDLE), "Continuation path identity differs")
    require(binding.get("authorization_sha256") == authorization_sha and sha(auth_path) == authorization_sha,
        "Continuation authorization SHA differs")
    auth = read(auth_path)
    required = ("matched_component_3seed", "initial_fit_adoption_only", "short_terminal_checks",
        "collect_once", "global_validation_then_test_cost", "local_comparison", "preserve_existing",
        "mac_deferred", "no_scientific_retry")
    require(auth.get("status") == "approved" and auth.get("campaign") == manager.NAME
        and all(auth.get(key) is True for key in required), "Continuation scope is not approved")
    require(auth.get("maximum_check_seconds") == CHECK_SECONDS
        and auth.get("minimum_check_interval_seconds") == INTERVAL_SECONDS
        and auth.get("transfer_timeout_seconds") == TRANSFER_SECONDS, "Bounded transport policy differs")
    files = binding.get("files")
    require(isinstance(files, dict) and {CONTROLLER, COMPARATOR} <= set(files), "Local controller bindings missing")
    for name, digest in files.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        require(path.resolve().is_relative_to(ROOT.resolve()) and not path.is_symlink()
            and sha(path) == digest, "Bound local file changed: " + name)
    contract_path, integrity_path = BUNDLE / "execution_contract.json", BUNDLE / "integrity.json"
    require(binding.get("contract_sha256") == sha(contract_path)
        and binding.get("integrity_sha256") == sha(integrity_path), "Sealed contract/integrity changed")
    c, integrity = read(contract_path), read(integrity_path)
    manager.validate(c)
    require(integrity.get("status") == "sealed" and integrity.get("contract_sha256") == sha(contract_path)
        and integrity.get("canonical_contract_sha256") == canonical(c)
        and integrity.get("archive_sha256") == sha(BUNDLE / "deployment.tar.gz")
        and integrity.get("source_files_sha256") == c["source"]["files_sha256"], "Sealed inventory differs")
    require(binding.get("science_spec_sha256") == c["science_spec_sha256"]
        == sha(manager.REPORT / "SCIENCE_SPEC.json"), "Scientific contract changed")
    for name, digest in c["source"]["files"].items():
        require(sha(manager.below(ROOT, name)) == digest, "Original/source file changed: " + name)
    for name, digest in {**c["input_files"], **c.get("legacy_files", {})}.items():
        require(sha(manager.below(BUNDLE / "input_snapshot", name)) == digest,
            "Prepared input or original A evidence changed: " + name)
    launch_path = REPORT / "LAUNCHED.json"
    require(sha(launch_path) == binding.get("initial_launch_sha256"), "Initial fit launch evidence changed")
    launch = read(launch_path)
    require(launch.get("contract_sha256") == canonical(c) and set(launch.get("hosts", {})) == set(HOSTS),
        "Both original fit launches are required")
    for host, item in launch["hosts"].items():
        require(item.get("phase") == "fit_validation" and valid_identity(item.get("identity"))
            and valid_identity(item.get("supervisor_identity")), "Actual fit dispatcher/parent identity missing")
        require(item["identity"]["cwd"] == c["hosts"][host]["source_root"]
            and item["supervisor_identity"]["cwd"] == c["hosts"][host]["source_root"], "Fit launch root differs")
    return binding, c, launch


@contextmanager
def advance_lock():
    with (REPORT / "advance.lock").open("a+") as stream:
        acquired = False
        try:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB); acquired = True
            except BlockingIOError:
                pass
            yield acquired
        finally:
            if acquired: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def transport(command, directory, stem, *, timeout, input_bytes=None):
    """Record original bytes and the actually observed OS result of one call."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    paths = {kind: directory / (stem + "." + kind) for kind in ("stdout", "stderr")}
    require(not any(path.exists() for path in paths.values()) and not (directory / (stem + ".json")).exists(),
        "Existing transport attempt must be preserved")
    started = time.time(); out = err = b""; code = None; expired = False; fault = None
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=timeout, input=input_bytes)
        out, err, code = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as error:
        out, err, expired = error.stdout or b"", error.stderr or b"", True
        fault = str(error)
    except OSError as error:
        fault = type(error).__name__ + ": " + str(error)
    for kind, value in (("stdout", out), ("stderr", err)):
        with paths[kind].open("xb") as stream: stream.write(value)
    receipt = {"command": command, "started_at_unix": started, "observed_at": now(),
        "elapsed_seconds": time.time() - started, "timeout_seconds": timeout,
        "actual_os_returncode": code, "timeout": expired, "transport_error": fault,
        **{kind + "_sha256": sha(path) for kind, path in paths.items()},
        "stdout_bytes": len(out), "stderr_bytes": len(err)}
    write(directory / (stem + ".json"), receipt)
    require(code == 0 and not expired and fault is None, "Transport failed; original bytes preserved; no retry")
    return receipt, out


def ssh_command(c, host, program):
    h = c["hosts"][host]
    remote = "cd " + shlex.quote(h["source_root"]) + " && " + shlex.join(
        ["env", *(key + "=" + str(value) for key, value in h["environment"].items()),
            h["python"], "-u", "-c", program])
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, remote]


def json_response(raw):
    lines = [line for line in raw.decode("utf-8").splitlines() if line.strip()]
    require(len(lines) == 1, "Expected exactly one JSON response")
    result = json.loads(lines[0]); require(isinstance(result, dict), "JSON response must be an object")
    return result


def terminal_program(c, host, phase, launch):
    """Generated remote code is read-only; no worker is signalled or launched."""
    return f'''import json,pathlib,time
from paper.scripts import manage_shared_component_ablation as m
root=pathlib.Path({c['hosts'][host]['root']!r});host={host!r};phase={phase!r};expected={launch!r};keys={IDENTITY_KEYS!r}
c=m.read(root/'execution_contract.json')
m.require(m.canonical(c)=={canonical(c)!r},'Native contract changed')
m.verify_files(c,root/'source')
claim=m.read(root/(phase+'.claim.json'));parent=m.read(root/(phase+'.supervisor.intent.json'))
same=lambda a,b:isinstance(a,dict) and all(a.get(k)==b.get(k) for k in keys)
m.require(same(claim['identity'],expected['identity']) and same(parent['identity'],expected['supervisor_identity']),'Launch identity changed')
failures=sorted(str(p.relative_to(root)) for p in root.glob('*.failure.json'))
m.require(not failures,'Preserved native phase failure: '+repr(failures))
active=[]
for p in (root/'control').glob('*.launch.json'):
 i=m.read(p);live=m.proc_identity(i['pid'])
 if same(live,i):active.append(i['pid'])
gpu=m.guard(c,host,active=active)
result={{'host':host,'phase':phase,'contract_sha256':m.canonical(c),'observed_at':time.time(),'read_only_check':True,'gpu':gpu}}
terminal=root/(phase+'.supervisor_exit.json')
if terminal.exists():
 complete,end=m.verify_phase_receipt(c,host,phase,root)
 for stream in ('stdout','stderr'):
  m.require(m.sha(root/(phase+'.'+stream+'.log'))==end.get(stream+'_sha256'),'Actual dispatcher output changed')
 m.require(end['command']==expected['identity']['argv'],'Actual dispatcher command differs')
 for i in [expected['identity'],*[x.get('identity') for x in complete['actual_child_exits']]]:
  m.require(isinstance(i,dict),'Scientific identity missing')
  m.require(not same(m.proc_identity(i['pid']),i),'Terminal scientific process still alive')
 result.update(status='terminal',dispatcher_exit=end,complete_sha256=m.sha(root/(phase+'.complete.json')))
else:
 for i in (expected['identity'],expected['supervisor_identity']):
  m.require(same(m.proc_identity(i['pid']),i),'Running dispatcher/parent identity differs')
 progress=[]
 for p in sorted((root/'run').glob('*/runs/**/history.json'))[:36]:
  if p.stat().st_size<=1048576:
   try:
    v=json.loads(p.read_text());rows=v.get('history',[]) if isinstance(v,dict) else v
    if isinstance(rows,list):progress.append({{'path':str(p.relative_to(root)),'mtime':p.stat().st_mtime,'rows':len(rows),'tail':rows[-5:]}})
   except (ValueError,OSError):pass
 result.update(status='waiting',dispatcher_current_identity=m.proc_identity(expected['identity']['pid']),progress=progress)
print(json.dumps(result,allow_nan=False),flush=True)
'''


def short_check(c, host, phase, launch, folder):
    receipt, raw = transport(ssh_command(c, host, terminal_program(c, host, phase, launch)),
        folder, "check", timeout=CHECK_SECONDS)
    value = json_response(raw)
    require(value.get("host") == host and value.get("phase") == phase
        and value.get("contract_sha256") == canonical(c) and value.get("read_only_check") is True,
        "Terminal response identity differs")
    require(value.get("status") in ("waiting", "terminal"), "Unknown native status")
    if value["status"] == "waiting":
        require(identity_equal(value.get("dispatcher_current_identity"), launch["identity"]),
            "Native running identity differs")
    else:
        exit_receipt = value.get("dispatcher_exit", {})
        require(exit_receipt.get("actual_os_returncode") == 0
            and exit_receipt.get("command") == launch["identity"]["argv"], "Actual successful dispatcher exit missing")
    return value


def safe_member(name):
    path = PurePosixPath(name)
    require(bool(name) and not path.is_absolute() and ".." not in path.parts
        and "." not in path.parts and str(path) == name, "Unsafe or noncanonical archive path")
    return path


def validate_collection(c, host, phase, folder, *, extract=False):
    """Bind actual pack transport, archive, manifest and every extracted byte."""
    folder = Path(folder); receipt = read(folder / "collection_receipt.json")
    require(receipt == json_response((folder / "pack.stdout").read_bytes()), "Pack stdout differs from parsed receipt")
    pack_transport = read(folder / "transport.json")
    require(pack_transport.get("actual_os_returncode") == 0 and pack_transport.get("timeout") is False
        and pack_transport.get("transport_error") is None
        and pack_transport.get("stdout_sha256") == sha(folder / "pack.stdout")
        and pack_transport.get("stderr_sha256") == sha(folder / "pack.stderr"), "Pack actual transport evidence differs")
    require(receipt.get("status") == "complete" and receipt.get("host") == host and receipt.get("phase") == phase
        and receipt.get("contract_sha256") == canonical(c), "Collection receipt identity differs")
    archive_path = folder / ("collection_" + phase + ".tar.gz")
    require(archive_path.stat().st_size == receipt.get("archive_bytes") and sha(archive_path) == receipt.get("archive_sha256"),
        "Collection archive bytes or SHA differ")
    transfer = read(folder / "transfer.json")
    require(transfer.get("actual_os_returncode") == 0 and transfer.get("timeout") is False
        and transfer.get("transport_error") is None
        and transfer.get("stdout_sha256") == sha(folder / "transfer.stdout")
        and transfer.get("stderr_sha256") == sha(folder / "transfer.stderr"), "Actual archive transfer evidence differs")
    originals, manifest_name = folder / "originals", "collection_" + phase + ".manifest.json"
    require(not originals.is_symlink(), "Symbolic extracted evidence root")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers(); names = [member.name for member in members]
        require(len(names) == len(set(names)) and all(member.isfile() for member in members),
            "Duplicate or nonregular archive member")
        for name in names: safe_member(name)
        require(manifest_name in names, "Archive manifest missing")
        manifest_bytes = archive.extractfile(manifest_name).read()
        require(hashlib.sha256(manifest_bytes).hexdigest() == receipt.get("manifest_sha256"), "Manifest SHA differs")
        manifest = json.loads(manifest_bytes)
        require(manifest.get("host") == host and manifest.get("phase") == phase
            and manifest.get("contract_sha256") == canonical(c)
            and manifest.get("all_source_input_legacy_actual_sha_verified") is True, "Manifest identity or verification missing")
        for key, expected in (("source_files", c["source"]["files"]), ("input_files", c["input_files"]),
                ("legacy_files", c.get("legacy_files", {}))):
            require(manifest.get(key) == expected, "Manifest inventory differs: " + key)
        inventory = manifest.get("files")
        require(isinstance(inventory, dict) and bool(inventory) and receipt.get("files") == len(inventory),
            "Collection inventory is missing or empty")
        require(set(names) == set(inventory) | {manifest_name}, "Archive and manifest inventory differ")
        if extract: originals.mkdir(exist_ok=False)
        for member in members:
            target = manager.below(originals, member.name)
            expected = {"bytes": len(manifest_bytes), "sha256": receipt["manifest_sha256"]} if member.name == manifest_name else inventory[member.name]
            require(type(expected.get("bytes")) is int and member.size == expected["bytes"], "Member size differs")
            require(member.size > 0 or member.name.endswith(".log"), "Empty scientific evidence file")
            packed_digest = hashlib.sha256()
            with archive.extractfile(member) as packed_stream:
                for block in iter(lambda: packed_stream.read(1048576), b""): packed_digest.update(block)
            require(packed_digest.hexdigest() == expected["sha256"], "Packed file SHA differs")
            if extract:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output, archive.extractfile(member) as source:
                    shutil.copyfileobj(source, output)
            require(target.is_file() and not target.is_symlink() and target.stat().st_size == member.size
                and sha(target) == expected["sha256"], "Extracted file bytes or SHA differ")
        extracted_paths = list(originals.rglob("*"))
        require(not any(path.is_symlink() for path in extracted_paths), "Symbolic extracted evidence")
        actual = {str(path.relative_to(originals)) for path in extracted_paths if path.is_file()}
        require(actual == set(names), "Unexpected or missing extracted evidence")
    for relative, digest in c.get("legacy_files", {}).items():
        require(sha(manager.below(originals / "source", relative)) == digest, "Original A evidence copy changed")
    _, terminal = manager.verify_phase_receipt(c, host, phase, originals)
    for stream in ("stdout", "stderr"):
        require(sha(originals / (phase + "." + stream + ".log")) == terminal.get(stream + "_sha256"),
            "Actual dispatcher output changed")
    if phase == "test_cost": manager.verify_global_gate(c, originals)
    return originals


def collect(c, host, phase):
    folder = BUNDLE / "retrieval" / phase / host
    require(not folder.exists(), "Existing or partial collection must be preserved; no retry")
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / "COLLECTION_ATTEMPT.json", {"host": host, "phase": phase,
        "contract_sha256": canonical(c), "at": now(), "automatic_retry": False})
    h = c["hosts"][host]
    command = [h["python"], "-u", "-m", "paper.scripts.manage_shared_component_ablation", "--mode", "pack",
        "--contract", h["root"] + "/execution_contract.json", "--contract-sha", sha(BUNDLE / "execution_contract.json"),
        "--host", host, "--phase", phase]
    native = "cd " + shlex.quote(h["source_root"]) + " && " + shlex.join(
        ["env", *(key + "=" + str(value) for key, value in h["environment"].items()), *command])
    record, raw = transport(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, native],
        folder, "pack", timeout=TRANSFER_SECONDS)
    write(folder / "transport.json", record)
    receipt = json_response(raw)
    require(receipt.get("host") == host and receipt.get("phase") == phase
        and receipt.get("contract_sha256") == canonical(c), "Pack response identity differs")
    remote_archive = h["root"] + "/collection_" + phase + ".tar.gz"
    require(receipt.get("archive") == remote_archive, "Pack archive path escaped the approved root")
    write(folder / "collection_receipt.json", receipt)
    transport(["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host + ":" + remote_archive,
        str(folder / ("collection_" + phase + ".tar.gz"))], folder, "transfer", timeout=TRANSFER_SECONDS)
    originals = validate_collection(c, host, phase, folder, extract=True)
    write(folder / "VERIFIED.json", {"status": "verified", "contract_sha256": canonical(c), "host": host,
        "phase": phase, "manifest_sha256": receipt["manifest_sha256"], "archive_sha256": receipt["archive_sha256"], "at": now()})
    return originals


def collected(c, host, phase):
    folder = BUNDLE / "retrieval" / phase / host
    if not folder.exists(): return None
    require((folder / "VERIFIED.json").is_file(), "Partial collection preserved; no automatic retry")
    return validate_collection(c, host, phase, folder)


def build_global_gate(c, roots, destination):
    destination = Path(destination)
    require(set(roots) == set(HOSTS), "Both fit/Validation collections are required")
    destination.mkdir(parents=True, exist_ok=False)
    jobs = {}
    for job in c["jobs"]:
        source = Path(roots[job["host"]]); evidence = {"host": job["host"]}
        for stage in ("train", "validation"):
            for kind in ("receipt", "exit"):
                if kind == "receipt": relative = "run/" + job["id"] + "/" + stage + "_receipt.json"
                else:
                    exit_stage = "reuse" if stage == "train" and job.get("reuse") else stage
                    relative = "control/" + job["id"] + "." + exit_stage + ".exit.json"
                original = manager.below(source, relative)
                target_relative = "global_validation_sources/" + job["host"] + "/" + relative
                target = manager.below(destination, target_relative); target.parent.mkdir(parents=True, exist_ok=True)
                with original.open("rb") as inp, target.open("xb") as out: shutil.copyfileobj(inp, out)
                require(sha(original) == sha(target), "Raw Validation evidence copy changed")
                evidence[stage + "_" + kind] = {"path": target_relative, "sha256": sha(target)}
        jobs[job["id"]] = evidence
    write(destination / "global_validation_gate.json", {"status": "passed", "contract_sha256": canonical(c), "jobs": jobs})
    manager.verify_global_gate(c, destination)
    return destination


def test_launch_program(c, host, gate_folder):
    tmux = "/usr/bin/tmux" if host == "5080" else "/opt/miniconda3/envs/ai_env/bin/tmux"
    return f'''import hashlib,json,pathlib,shlex,subprocess,sys,time
from paper.scripts import manage_shared_component_ablation as m
root=pathlib.Path({c['hosts'][host]['root']!r});host={host!r};files=json.load(sys.stdin)
c=m.read(root/'execution_contract.json');m.require(m.canonical(c)=={canonical(c)!r},'Native contract differs')
m.verify_files(c,root/'source');m.guard(c,host);m.verify_native_qualification(c,host,root);m.verify_phase_receipt(c,host,'fit_validation',root)
m.require(not any(root.glob('*.failure.json')),'Preserved native failure')
m.require(not any(root.glob('test_cost.*')) and not (root/'global_validation_gate.json').exists() and not (root/'global_validation_sources').exists(),'Existing Test/gate attempt; preserve')
for name,item in files.items():
 p=m.below(root,name);m.require(name=='global_validation_gate.json' or name.startswith('global_validation_sources/'),'Unexpected gate path')
 text=item['text'];m.require(hashlib.sha256(text.encode()).hexdigest()==item['sha256'],'Gate transfer bytes changed')
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as out:out.write(text)
m.verify_global_gate(c,root)
h=c['hosts'][host]
command=['env',*[k+'='+str(v) for k,v in h['environment'].items()],h['python'],'-u','-m','paper.scripts.manage_shared_component_ablation','--mode','supervise','--contract',str(root/'execution_contract.json'),'--contract-sha',m.sha(root/'execution_contract.json'),'--host',host,'--phase','test_cost']
result=subprocess.run([{tmux!r},'new-session','-d','-s',h['tmux']+'_test_cost','-c',h['source_root'],shlex.join(command)],capture_output=True,timeout=15)
m.require(result.returncode==0,'tmux launch failed: '+repr(result.stderr))
deadline=time.monotonic()+10
while not ((root/'test_cost.claim.json').exists() and (root/'test_cost.supervisor.intent.json').exists()) and time.monotonic()<deadline:time.sleep(.05)
claim=m.read(root/'test_cost.claim.json');parent=m.read(root/'test_cost.supervisor.intent.json')
keys={IDENTITY_KEYS!r}
for expected in (claim['identity'],parent['identity']):
 live=m.proc_identity(expected['pid']);m.require(live is not None and all(live[k]==expected[k] for k in keys),'New Test dispatcher identity differs')
print(json.dumps({{'status':'launched','host':host,'phase':'test_cost','contract_sha256':m.canonical(c),'identity':claim['identity'],'supervisor_identity':parent['identity'],'global_validation_gate_sha256':m.sha(root/'global_validation_gate.json'),'observed_at':time.time()}}),flush=True)
'''


def launch_test(c, host, gate_folder, round_folder):
    path = REPORT / ("TEST_LAUNCHED_" + host + ".json")
    require(not path.exists(), "Test phase already launched; no retry")
    write(round_folder / ("TEST_LAUNCH_ATTEMPT_" + host + ".json"), {"host": host, "at": now(),
        "contract_sha256": canonical(c), "global_gate_sha256": sha(gate_folder / "global_validation_gate.json")})
    payload = {str(path.relative_to(gate_folder)): {"text": path.read_text(), "sha256": sha(path)}
        for path in sorted(Path(gate_folder).rglob("*")) if path.is_file()}
    _, raw = transport(ssh_command(c, host, test_launch_program(c, host, gate_folder)),
        round_folder, "test_launch_" + host, timeout=CHECK_SECONDS,
        input_bytes=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode())
    value = json_response(raw)
    require(value.get("status") == "launched" and value.get("host") == host and value.get("phase") == "test_cost"
        and value.get("contract_sha256") == canonical(c) and valid_identity(value.get("identity"))
        and valid_identity(value.get("supervisor_identity"))
        and value.get("global_validation_gate_sha256") == sha(gate_folder / "global_validation_gate.json"),
        "New Test phase launch identity differs")
    write(path, value)
    return value


def comparator_qualified(binding):
    path = manager.REPORT / "COMPARATOR_QA.json"
    if not path.is_file() or not binding.get("comparator_qa_sha256"): return False
    require(sha(path) == binding["comparator_qa_sha256"], "Independent comparator review changed")
    qa = read(path)
    require(qa.get("status") == "PASS" and qa.get("files_reviewed_sha256", {}).get(COMPARATOR) == sha(ROOT / COMPARATOR),
        "Comparator has not passed a source-bound independent review")
    return True


def compare(binding, c, folder, *, cached=False):
    if not comparator_qualified(binding): return {"status": "collection_complete", "comparison": "independent_review_pending"}
    output = BUNDLE / "comparison_v1"
    base = [sys.executable, "-m", "paper.scripts.compare_shared_component_ablation", "--contract", str(BUNDLE / "execution_contract.json"),
        "--contract-sha", sha(BUNDLE / "execution_contract.json"), "--retrieval", str(BUNDLE / "retrieval"), "--output", str(output)]
    if cached:
        transport(base + ["--check-only"], folder, "comparison_cache", timeout=TRANSFER_SECONDS)
    else:
        require(not output.exists(), "Existing or partial comparison must be preserved")
        for label, command in (("comparison_precheck", base + ["--check-only"]), ("comparison_create", base),
                ("comparison_postcheck", base + ["--check-only"])):
            transport(command, folder, label, timeout=TRANSFER_SECONDS)
    return {"status": "complete", "output": str(output)}


def completion_inventory():
    inventory = {}
    for root in (BUNDLE / "retrieval", BUNDLE / "global_validation_v1", BUNDLE / "comparison_v1"):
        require(root.is_dir(), "Completion evidence directory missing")
        require(not root.is_symlink() and not any(path.is_symlink() for path in root.rglob("*")),
            "Symbolic completion evidence directory")
        paths = [path for path in root.rglob("*") if path.is_file()]
        require(bool(paths), "Empty completion evidence inventory")
        for path in paths:
            require(not path.is_symlink(), "Symbolic completion evidence")
            inventory[str(path)] = {"sha256": sha(path), "bytes": path.stat().st_size}
    for host in HOSTS:
        path = REPORT / ("TEST_LAUNCHED_" + host + ".json")
        inventory[str(path)] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return inventory


def verify_cache(binding, c, binding_sha, folder):
    complete = read(REPORT / "COMPLETE.json")
    require(complete.get("status") == "complete" and complete.get("binding_sha256") == binding_sha
        and complete.get("contract_sha256") == canonical(c) and complete.get("files") == completion_inventory(),
        "Cached completion evidence changed or is incomplete")
    for host in HOSTS:
        validate_collection(c, host, "fit_validation", BUNDLE / "retrieval/fit_validation" / host)
        validate_collection(c, host, "test_cost", BUNDLE / "retrieval/test_cost" / host)
    manager.verify_global_gate(c, BUNDLE / "global_validation_v1")
    require(compare(binding, c, folder, cached=True)["status"] == "complete", "Cached comparison review pending")
    return {"status": "complete", "cache_verified": True, "remote_calls": 0, "output": str(BUNDLE / "comparison_v1")}


def publish_round(folder, value):
    value = {**value, "round_path": str(folder), "observed_at": now()}
    write(folder / "ROUND.json", value)
    pointer = REPORT / (".LATEST." + uuid.uuid4().hex + ".json")
    write(pointer, {"round_path": str(folder / "ROUND.json"), "round_sha256": sha(folder / "ROUND.json"),
        "status": value["status"], "check_started_unix": read(folder / "INTENT.json")["check_started_unix"]})
    os.replace(pointer, REPORT / "LATEST.json")
    return value


def preserve_failure(error, folder=None):
    value = {"status": "failed", "error": type(error).__name__ + ": " + str(error), "at": now(),
        "round_path": str(folder) if folder else None, "automatic_retry": False,
        "further_remote_calls": 0, "training_failure_inferred_from_transport": False}
    if not (REPORT / "FAILURE.json").exists(): write(REPORT / "FAILURE.json", value)
    return value


def advance(binding_sha, authorization_sha):
    binding, c, initial = verify(binding_sha, authorization_sha)
    with advance_lock() as acquired:
        if not acquired: return {"status": "busy", "remote_calls": 0}
        if (REPORT / "FAILURE.json").exists(): return {**read(REPORT / "FAILURE.json"), "remote_calls": 0, "cached_failure": True}
        latest = read(REPORT / "LATEST.json") if (REPORT / "LATEST.json").exists() else None
        if latest:
            require(sha(latest["round_path"]) == latest["round_sha256"], "Latest round evidence changed")
            if latest.get("status") != "complete" and time.time() - latest["check_started_unix"] < INTERVAL_SECONDS:
                return {"status": "waiting_next_check", "remote_calls": 0,
                    "next_check_unix": latest["check_started_unix"] + INTERVAL_SECONDS}
        for path in (REPORT / "rounds").glob("*/INTENT.json"):
            require((path.parent / "ROUND.json").exists(), "Interrupted prior round preserved; manual reconciliation required")
        folder = REPORT / "rounds" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "_" + uuid.uuid4().hex[:12])
        write(folder / "INTENT.json", {"check_started_unix": time.time(), "binding_sha256": binding_sha,
            "contract_sha256": canonical(c), "at": now()})
        try:
            if (REPORT / "COMPLETE.json").exists(): return publish_round(folder, verify_cache(binding, c, binding_sha, folder))
            statuses, roots = {}, {}
            for host in HOSTS:
                test_path = REPORT / ("TEST_LAUNCHED_" + host + ".json")
                launch = read(test_path) if test_path.exists() else initial["hosts"][host]
                phase = launch["phase"]
                root = collected(c, host, phase)
                if root is None:
                    host_folder = folder / host
                    value = short_check(c, host, phase, launch, host_folder)
                    statuses[host] = value
                    if value["status"] == "terminal": root = collect(c, host, phase)
                if root is not None: roots[host] = root
            fit_roots = {host: collected(c, host, "fit_validation") for host in HOSTS}
            if all(fit_roots.values()) and not any((REPORT / ("TEST_LAUNCHED_" + host + ".json")).exists() for host in HOSTS):
                gate = build_global_gate(c, fit_roots, BUNDLE / "global_validation_v1")
                launches = {host: launch_test(c, host, gate, folder) for host in HOSTS}
                return publish_round(folder, {"status": "waiting", "phase": "test_cost", "transition": "global_Validation36_passed_Test_cost_launched",
                    "hosts": launches})
            test_roots = {host: collected(c, host, "test_cost") for host in HOSTS}
            if all(test_roots.values()):
                marker = REPORT / "COLLECTION_COMPLETE.json"
                if not marker.exists(): write(marker, {"status": "collection_complete", "contract_sha256": canonical(c), "at": now()})
                result = compare(binding, c, folder)
                if result["status"] == "complete":
                    write(REPORT / "COMPLETE.json", {**result, "binding_sha256": binding_sha,
                        "contract_sha256": canonical(c), "files": completion_inventory(), "at": now()})
                return publish_round(folder, result)
            return publish_round(folder, {"status": "waiting", "hosts": statuses,
                "collected_fit_hosts": [host for host, root in fit_roots.items() if root],
                "collected_test_cost_hosts": [host for host, root in test_roots.items() if root]})
        except BaseException as error:
            return publish_round(folder, preserve_failure(error, folder))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("advance",), required=True)
    parser.add_argument("--binding-sha", required=True); parser.add_argument("--authorization-sha", required=True)
    args = parser.parse_args(argv)
    try: result = advance(args.binding_sha, args.authorization_sha)
    except BaseException as error:
        result = preserve_failure(error)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 2 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
