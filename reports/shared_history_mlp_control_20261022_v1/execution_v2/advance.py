"""Version 2 bounded continuation of the fixed Validation-only MLP control.

This module adopts already launched fit/Validation dispatchers. It never starts,
resumes or retries a fit, and has no Test launch or held-out evaluation path.
Archive validation is adapted from the frozen A--D controller without changing
its globals, modules, paths or command-line identity.
"""

from __future__ import annotations

import argparse
from importlib import import_module
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import sys
import tarfile
import time
import uuid

# A direct file invocation must resolve the project without relying on an old CLI.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

manager = import_module("paper.scripts.manage_shared_history_mlp_control")
boundary_tools = import_module("paper.scripts.advance_shared_component_ablation")

ROOT, BUNDLE = manager.ROOT, manager.BUNDLE / "execution_v2"
EXEC_REPORT = manager.REPORT / "execution_v2"
REPORT = EXEC_REPORT / "continuation_v1"
HOSTS, PHASES = ("5080", "5090"), ("fit_validation", "cost")
IDENTITY_KEYS = ("pid", "start_ticks", "pgid", "argv", "cwd")
CHECK_SECONDS, INTERVAL_SECONDS, TRANSFER_SECONDS = 45, 900, 900
CONTROLLER = "reports/shared_history_mlp_control_20261022_v1/execution_v2/advance.py"
COMPARATOR = "reports/shared_history_mlp_control_20261022_v1/execution_v2/compare.py"
MANAGER_MODULE = "paper.scripts.manage_shared_history_mlp_control"
read, sha, write, canonical, require = (
    manager.read,
    manager.sha,
    manager.write,
    manager.canonical,
    manager.require,
)
transport, ssh_command, json_response = (
    boundary_tools.transport,
    boundary_tools.ssh_command,
    boundary_tools.json_response,
)
identity_equal, valid_identity = (
    boundary_tools.identity_equal,
    boundary_tools.valid_identity,
)


def now():
    return datetime.now(timezone.utc).isoformat()


def regular_files(root):
    root = Path(root)
    require(
        root.is_dir() and not root.is_symlink(),
        "Evidence directory missing or symbolic",
    )
    paths = list(root.rglob("*"))
    require(not any(path.is_symlink() for path in paths), "Symbolic evidence path")
    return {str(path) for path in paths if path.is_file()}


def pinned(path, expected):
    path = Path(path)
    require(
        path.is_file()
        and not path.is_symlink()
        and path.resolve().is_relative_to(ROOT.resolve())
        and isinstance(expected, str)
        and sha(path) == expected,
        "Bound file changed: " + str(path),
    )


def verify_preservation(binding):
    path = manager.REPORT / "PRESERVATION.json"
    pinned(path, binding.get("preservation_sha256"))
    evidence = read(path)
    require(evidence.get("status") == "frozen", "Original preservation must be frozen")
    files = evidence.get("files")
    require(
        isinstance(files, dict) and bool(files),
        "Preservation inventory missing or empty",
    )
    for name, item in files.items():
        target = Path(name)
        require(
            target.is_absolute()
            and isinstance(item, dict)
            and type(item.get("bytes")) is int
            and item["bytes"] >= 0,
            "Invalid preservation inventory entry",
        )
        pinned(target, item.get("sha256"))
        require(
            target.stat().st_size == item["bytes"],
            "Preserved byte size changed: " + name,
        )
    for name in binding.get("preservation_roots", []):
        require(regular_files(name) <= set(files), "Unbound original evidence: " + name)
    return files


def verify(binding_sha, authorization_sha):
    pinned(REPORT / "CONTROL_BINDINGS.json", binding_sha)
    binding = read(REPORT / "CONTROL_BINDINGS.json")
    require(
        binding.get("campaign") == manager.NAME
        and binding.get("report_root") == str(REPORT)
        and binding.get("bundle_root") == str(BUNDLE),
        "Continuation path identity differs",
    )
    auth_path = EXEC_REPORT / "AUTHORIZATION.json"
    pinned(auth_path, authorization_sha)
    require(
        binding.get("authorization_sha256") == authorization_sha,
        "Authorization binding differs",
    )
    auth = read(auth_path)
    require(
        auth.get("status") == "approved"
        and auth.get("campaign") == manager.NAME
        and auth.get("remote_hosts") == list(HOSTS)
        and auth.get("held_out_test_evaluation") is False
        and auth.get("scientific_retry") is False
        and auth.get("old_files_overwrite") is False
        and auth.get("runtime_credentials_network_changes") is False
        and auth.get("new_fit_count") == 9
        and auth.get("existing_checkpoint_reuse") == 18
        and auth.get("train_only_cost_processes") == 162,
        "Approved Validation-only scope differs",
    )
    instruction = auth.get("user_instruction")
    require(
        isinstance(instruction, str)
        and bool(instruction.strip())
        and hashlib.sha256(instruction.encode()).hexdigest()
        == binding.get("user_instruction_sha256"),
        "Exact human instruction is not bound",
    )
    require(
        type(binding.get("max_rounds")) is int
        and 0 < binding["max_rounds"] <= 96
        and binding.get("min_interval_seconds") == INTERVAL_SECONDS
        and binding.get("maximum_check_seconds", CHECK_SECONDS) == CHECK_SECONDS
        and binding.get("transfer_timeout_seconds", TRANSFER_SECONDS)
        == TRANSFER_SECONDS,
        "Bounded continuation policy differs",
    )
    files = binding.get("files")
    require(
        isinstance(files, dict) and {CONTROLLER, COMPARATOR} <= set(files),
        "Local code bindings missing",
    )
    for name, digest in files.items():
        pinned(Path(name) if Path(name).is_absolute() else ROOT / name, digest)
    contract_path, integrity_path = (
        BUNDLE / "execution_contract.json",
        BUNDLE / "integrity.json",
    )
    pinned(contract_path, binding.get("contract_file_sha256"))
    pinned(integrity_path, binding.get("integrity_sha256"))
    c, integrity = read(contract_path), read(integrity_path)
    manager.validate(c)
    require(
        c.get("execution_version") == 2
        and c.get("design_sha256") == c.get("science_spec_sha256"),
        "Version 2 design metadata is missing or differs",
    )
    pinned(EXEC_REPORT / "ADMISSION.json", binding.get("execution_admission_sha256"))
    pinned(BUNDLE / "deployment.tar.gz", binding.get("archive_sha256"))
    require(
        binding.get("canonical_contract_sha256") == canonical(c),
        "Canonical contract binding differs",
    )
    require(
        integrity.get("status") == "sealed"
        and integrity.get("contract_sha256") == sha(contract_path)
        and integrity.get("canonical_contract_sha256") == canonical(c)
        and integrity.get("archive_sha256")
        == binding.get("archive_sha256")
        == sha(BUNDLE / "deployment.tar.gz")
        and integrity.get("source_files_sha256") == c["source"]["files_sha256"],
        "Sealed inventory differs",
    )
    pinned(manager.REPORT / "SCIENCE_SPEC.json", binding.get("science_spec_sha256"))
    require(
        binding["science_spec_sha256"] == c["science_spec_sha256"],
        "Scientific contract changed",
    )
    for name, digest in c["source"]["files"].items():
        pinned(manager.below(ROOT, name), digest)
    for name, digest in {**c["input_files"], **c.get("legacy_files", {})}.items():
        pinned(manager.below(BUNDLE / "input_snapshot", name), digest)
    qa_path = EXEC_REPORT / "INDEPENDENT_QA.json"
    pinned(qa_path, binding.get("independent_qa_sha256"))
    qa = read(qa_path)
    require(
        qa.get("status") == "PASS"
        and qa.get("independent_reviewer") is True
        and all(
            qa.get("files_reviewed_sha256", {}).get(name) == digest
            for name, digest in files.items()
            if name.endswith(".py")
        ),
        "Source-bound independent review is missing",
    )
    verify_preservation(binding)
    pinned(REPORT / "LAUNCHED.json", binding.get("initial_launch_sha256"))
    launch = read(REPORT / "LAUNCHED.json")
    require(
        launch.get("contract_sha256") == canonical(c)
        and set(launch.get("hosts", {})) == set(HOSTS),
        "Both fixed fit launches are required",
    )
    for host, value in launch["hosts"].items():
        validate_launch(c, host, "fit_validation", value)
    return binding, c, launch


def expected_command(c, host, phase):
    h = c["hosts"][host]
    return [
        h["python"],
        "-u",
        "-m",
        MANAGER_MODULE,
        "--mode",
        "dispatch",
        "--contract",
        h["root"] + "/execution_contract.json",
        "--contract-sha",
        sha(BUNDLE / "execution_contract.json"),
        "--host",
        host,
        "--phase",
        phase,
    ]


def validate_launch(c, host, phase, value):
    require(
        value.get("phase") == phase
        and value.get("host", host) == host
        and value.get("contract_sha256", canonical(c)) == canonical(c)
        and valid_identity(value.get("identity"))
        and valid_identity(value.get("supervisor_identity")),
        "Actual dispatcher and parent launch identities required",
    )
    for identity in (value["identity"], value["supervisor_identity"]):
        require(
            identity["cwd"] == c["hosts"][host]["source_root"],
            "Launch source root differs",
        )
    require(
        value["identity"]["argv"] == expected_command(c, host, phase),
        "Dispatcher command identity differs",
    )
    require(
        value["supervisor_identity"]["argv"]
        == [
            "supervise" if item == "dispatch" else item
            for item in value["identity"]["argv"]
        ],
        "Parent command identity differs",
    )


@contextmanager
def advance_lock():
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / "advance.lock").open("a+") as stream:
        acquired = False
        try:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                pass
            yield acquired
        finally:
            if acquired:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def terminal_program(c, host, phase, launch):
    """Generated remote code is read-only; no worker is signalled or launched."""
    return f"""import json,pathlib,time
from paper.scripts import manage_shared_history_mlp_control as m
root=pathlib.Path({c["hosts"][host]["root"]!r});host={host!r};phase={phase!r};expected={launch!r};keys={IDENTITY_KEYS!r}
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
 for p in sorted((root/'run').glob('*/runs/**/history.json'))[:27]:
  if p.stat().st_size<=1048576:
   try:
    v=json.loads(p.read_text());rows=v.get('history',[]) if isinstance(v,dict) else v
    if isinstance(rows,list):progress.append({{'path':str(p.relative_to(root)),'mtime':p.stat().st_mtime,'rows':len(rows),'tail':rows[-5:]}})
   except (ValueError,OSError):pass
 result.update(status='waiting',dispatcher_current_identity=m.proc_identity(expected['identity']['pid']),progress=progress)
print(json.dumps(result,allow_nan=False),flush=True)
"""


def short_check(c, host, phase, launch, folder):
    receipt, raw = transport(
        ssh_command(c, host, terminal_program(c, host, phase, launch)),
        folder,
        "check",
        timeout=CHECK_SECONDS,
    )
    value = json_response(raw)
    require(
        value.get("host") == host
        and value.get("phase") == phase
        and value.get("contract_sha256") == canonical(c)
        and value.get("read_only_check") is True,
        "Terminal response identity differs",
    )
    require(value.get("status") in ("waiting", "terminal"), "Unknown native status")
    if value["status"] == "waiting":
        require(
            identity_equal(
                value.get("dispatcher_current_identity"), launch["identity"]
            ),
            "Native running identity differs",
        )
    else:
        exit_receipt = value.get("dispatcher_exit", {})
        require(
            exit_receipt.get("actual_os_returncode") == 0
            and exit_receipt.get("command") == launch["identity"]["argv"],
            "Actual successful dispatcher exit missing",
        )
    return value


def safe_member(name):
    path = PurePosixPath(name)
    require(
        bool(name)
        and not path.is_absolute()
        and ".." not in path.parts
        and "." not in path.parts
        and str(path) == name,
        "Unsafe or noncanonical archive path",
    )
    return path


def validate_collection(c, host, phase, folder, *, extract=False):
    """Bind actual pack transport, archive, manifest and every extracted byte."""
    folder = Path(folder)
    if not extract:
        regular_files(folder)
    receipt = read(folder / "collection_receipt.json")
    require(
        receipt == json_response((folder / "pack.stdout").read_bytes()),
        "Pack stdout differs from parsed receipt",
    )
    pack_transport = read(folder / "transport.json")
    require(
        pack_transport == verify_streams(folder, "pack"), "Pack transport copy differs"
    )
    require(
        pack_transport.get("actual_os_returncode") == 0
        and pack_transport.get("timeout") is False
        and pack_transport.get("transport_error") is None
        and pack_transport.get("stdout_sha256") == sha(folder / "pack.stdout")
        and pack_transport.get("stderr_sha256") == sha(folder / "pack.stderr"),
        "Pack actual transport evidence differs",
    )
    require(
        receipt.get("status") == "complete"
        and receipt.get("host") == host
        and receipt.get("phase") == phase
        and receipt.get("contract_sha256") == canonical(c),
        "Collection receipt identity differs",
    )
    archive_path = folder / ("collection_" + phase + ".tar.gz")
    require(
        archive_path.stat().st_size == receipt.get("archive_bytes")
        and sha(archive_path) == receipt.get("archive_sha256"),
        "Collection archive bytes or SHA differ",
    )
    transfer = read(folder / "transfer.json")
    verify_streams(folder, "transfer")
    require(
        transfer.get("actual_os_returncode") == 0
        and transfer.get("timeout") is False
        and transfer.get("transport_error") is None
        and transfer.get("stdout_sha256") == sha(folder / "transfer.stdout")
        and transfer.get("stderr_sha256") == sha(folder / "transfer.stderr"),
        "Actual archive transfer evidence differs",
    )
    originals, manifest_name = (
        folder / "originals",
        "collection_" + phase + ".manifest.json",
    )
    require(not originals.is_symlink(), "Symbolic extracted evidence root")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        require(
            len(names) == len(set(names))
            and all(member.isfile() for member in members),
            "Duplicate or nonregular archive member",
        )
        for name in names:
            safe_member(name)
        require(manifest_name in names, "Archive manifest missing")
        manifest_bytes = archive.extractfile(manifest_name).read()
        require(
            hashlib.sha256(manifest_bytes).hexdigest()
            == receipt.get("manifest_sha256"),
            "Manifest SHA differs",
        )
        manifest = json.loads(manifest_bytes)
        require(
            manifest.get("host") == host
            and manifest.get("phase") == phase
            and manifest.get("contract_sha256") == canonical(c)
            and manifest.get("all_source_input_legacy_actual_sha_verified") is True,
            "Manifest identity or verification missing",
        )
        for key, expected in (
            ("source_files", c["source"]["files"]),
            ("input_files", c["input_files"]),
            ("legacy_files", c.get("legacy_files", {})),
        ):
            require(manifest.get(key) == expected, "Manifest inventory differs: " + key)
        inventory = manifest.get("files")
        require(
            isinstance(inventory, dict)
            and bool(inventory)
            and receipt.get("files") == len(inventory),
            "Collection inventory is missing or empty",
        )
        require(
            set(names) == set(inventory) | {manifest_name},
            "Archive and manifest inventory differ",
        )
        if extract:
            originals.mkdir(exist_ok=False)
        for member in members:
            target = manager.below(originals, member.name)
            expected = (
                {"bytes": len(manifest_bytes), "sha256": receipt["manifest_sha256"]}
                if member.name == manifest_name
                else inventory[member.name]
            )
            require(
                type(expected.get("bytes")) is int and member.size == expected["bytes"],
                "Member size differs",
            )
            require(
                member.size > 0 or member.name.endswith(".log"),
                "Empty scientific evidence file",
            )
            packed_digest = hashlib.sha256()
            with archive.extractfile(member) as packed_stream:
                for block in iter(lambda: packed_stream.read(1048576), b""):
                    packed_digest.update(block)
            require(
                packed_digest.hexdigest() == expected["sha256"],
                "Packed file SHA differs",
            )
            if extract:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output, archive.extractfile(member) as source:
                    shutil.copyfileobj(source, output)
            require(
                target.is_file()
                and not target.is_symlink()
                and target.stat().st_size == member.size
                and sha(target) == expected["sha256"],
                "Extracted file bytes or SHA differ",
            )
        extracted_paths = list(originals.rglob("*"))
        require(
            not any(path.is_symlink() for path in extracted_paths),
            "Symbolic extracted evidence",
        )
        actual = {
            str(path.relative_to(originals))
            for path in extracted_paths
            if path.is_file()
        }
        require(actual == set(names), "Unexpected or missing extracted evidence")
    for relative, digest in c.get("legacy_files", {}).items():
        require(
            sha(manager.below(originals / "source", relative)) == digest,
            "Original B/D evidence copy changed",
        )
    _, terminal = manager.verify_phase_receipt(c, host, phase, originals)
    for stream in ("stdout", "stderr"):
        require(
            sha(originals / (phase + "." + stream + ".log"))
            == terminal.get(stream + "_sha256"),
            "Actual dispatcher output changed",
        )
    if phase == "cost":
        manager.verify_global_gate(c, originals)
    return originals


def collect(c, host, phase, *, admission=None):
    folder = BUNDLE / "retrieval" / phase / host
    require(
        not folder.exists(),
        "Existing or partial collection must be preserved; no retry",
    )
    folder.mkdir(parents=True, exist_ok=False)
    write(
        folder / "COLLECTION_ATTEMPT.json",
        {
            "host": host,
            "phase": phase,
            "contract_sha256": canonical(c),
            "at": now(),
            "automatic_retry": False,
        },
    )
    h = c["hosts"][host]
    command = [
        h["python"],
        "-u",
        "-m",
        "paper.scripts.manage_shared_history_mlp_control",
        "--mode",
        "pack",
        "--contract",
        h["root"] + "/execution_contract.json",
        "--contract-sha",
        sha(BUNDLE / "execution_contract.json"),
        "--host",
        host,
        "--phase",
        phase,
    ]
    native = (
        "cd "
        + shlex.quote(h["source_root"])
        + " && "
        + shlex.join(
            [
                "env",
                *(key + "=" + str(value) for key, value in h["environment"].items()),
                *command,
            ]
        )
    )
    if admission is not None:
        admission()
    record, raw = transport(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, native],
        folder,
        "pack",
        timeout=TRANSFER_SECONDS,
    )
    write(folder / "transport.json", record)
    receipt = json_response(raw)
    require(
        receipt.get("host") == host
        and receipt.get("phase") == phase
        and receipt.get("contract_sha256") == canonical(c),
        "Pack response identity differs",
    )
    remote_archive = h["root"] + "/collection_" + phase + ".tar.gz"
    require(
        receipt.get("archive") == remote_archive,
        "Pack archive path escaped the approved root",
    )
    write(folder / "collection_receipt.json", receipt)
    if admission is not None:
        admission()
    transport(
        [
            "scp",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            host + ":" + remote_archive,
            str(folder / ("collection_" + phase + ".tar.gz")),
        ],
        folder,
        "transfer",
        timeout=TRANSFER_SECONDS,
    )
    originals = validate_collection(c, host, phase, folder, extract=True)
    write(
        folder / "VERIFIED.json",
        {
            "status": "verified",
            "contract_sha256": canonical(c),
            "host": host,
            "phase": phase,
            "manifest_sha256": receipt["manifest_sha256"],
            "archive_sha256": receipt["archive_sha256"],
            "at": now(),
        },
    )
    return originals


def build_global_gate(c, roots, destination):
    destination = Path(destination)
    require(set(roots) == set(HOSTS), "Both fit/Validation collections are required")
    destination.mkdir(parents=True, exist_ok=False)
    jobs = {}
    for job in c["jobs"]:
        source = Path(roots[job["host"]])
        evidence = {"host": job["host"]}
        for stage in ("train", "validation"):
            for kind in ("receipt", "exit"):
                if kind == "receipt":
                    relative = "run/" + job["id"] + "/" + stage + "_receipt.json"
                else:
                    exit_stage = (
                        "reuse" if stage == "train" and job.get("reuse") else stage
                    )
                    relative = "control/" + job["id"] + "." + exit_stage + ".exit.json"
                original = manager.below(source, relative)
                target_relative = (
                    "global_validation_sources/" + job["host"] + "/" + relative
                )
                target = manager.below(destination, target_relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                with original.open("rb") as inp, target.open("xb") as out:
                    shutil.copyfileobj(inp, out)
                require(
                    sha(original) == sha(target), "Raw Validation evidence copy changed"
                )
                evidence[stage + "_" + kind] = {
                    "path": target_relative,
                    "sha256": sha(target),
                }
        jobs[job["id"]] = evidence
    write(
        destination / "global_validation_gate.json",
        {"status": "passed", "contract_sha256": canonical(c), "jobs": jobs},
    )
    manager.verify_global_gate(c, destination)
    return destination


def collected(c, host, phase):
    folder = BUNDLE / "retrieval" / phase / host
    if not folder.exists() and not folder.is_symlink():
        return None
    regular_files(folder)
    marker_path = folder / "VERIFIED.json"
    require(marker_path.is_file(), "Partial collection preserved; no automatic retry")
    root = validate_collection(c, host, phase, folder)
    marker, receipt = read(marker_path), read(folder / "collection_receipt.json")
    require(
        marker.get("status") == "verified"
        and marker.get("host") == host
        and marker.get("phase") == phase
        and marker.get("contract_sha256") == canonical(c)
        and marker.get("manifest_sha256") == receipt.get("manifest_sha256")
        and marker.get("archive_sha256") == receipt.get("archive_sha256"),
        "Collection verification marker differs",
    )
    return root


def all_collections(c):
    return {
        (host, phase): collected(c, host, phase) for host in HOSTS for phase in PHASES
    }


def verify_streams(folder, stem, *, expected_code=0):
    value = read(Path(folder) / (stem + ".json"))
    require(
        value.get("actual_os_returncode") == expected_code
        and value.get("timeout") is False
        and value.get("transport_error") is None,
        "Actual successful transport required",
    )
    for stream in ("stdout", "stderr"):
        p = Path(folder) / (stem + "." + stream)
        require(
            p.is_file()
            and not p.is_symlink()
            and p.stat().st_size == value.get(stream + "_bytes")
            and sha(p) == value.get(stream + "_sha256"),
            "Actual transport stream bytes differ",
        )
    return value


def latest_round():
    path = REPORT / "LATEST.json"
    if not path.exists():
        return None, None
    latest = read(path)
    target = Path(latest.get("round_path", ""))
    require(
        target.name == "ROUND.json"
        and target.parent.parent == REPORT / "rounds"
        and not target.is_symlink()
        and target.is_file()
        and sha(target) == latest.get("round_sha256"),
        "Latest round evidence changed",
    )
    result, intent = read(target), read(target.parent / "INTENT.json")
    require(
        latest.get("status") == result.get("status")
        and latest.get("check_started_unix") == intent.get("check_started_unix"),
        "Latest identity differs",
    )
    return latest, result


def validate_state(binding, c, collections):
    intents = sorted((REPORT / "rounds").glob("*/INTENT.json"))
    for path in intents:
        require(
            not path.is_symlink() and (path.parent / "ROUND.json").is_file(),
            "Interrupted prior round must be preserved; manual reconciliation required",
        )
        intent = read(path)
        result = read(path.parent / "ROUND.json")
        require(
            intent.get("binding_sha256") == sha(REPORT / "CONTROL_BINDINGS.json")
            and intent.get("contract_sha256") == canonical(c)
            and result.get("round_path") == str(path.parent)
            and result.get("status") in ("waiting", "complete"),
            "Prior round failed or binding differs",
        )
        for host in HOSTS:
            if (path.parent / host / "check.json").exists():
                verify_streams(path.parent / host, "check")
    latest_round()
    launches = {
        host: read(REPORT / ("COST_LAUNCHED_" + host + ".json"))
        for host in HOSTS
        if (REPORT / ("COST_LAUNCHED_" + host + ".json")).exists()
    }
    attempts = list((REPORT / "rounds").glob("*/COST_LAUNCH_ATTEMPT_*.json"))
    require(
        not launches or set(launches) == set(HOSTS),
        "Single-host cost launch preserved; no retry",
    )
    gate = BUNDLE / "global_validation_v1"
    if not launches:
        require(
            not attempts and not gate.exists(),
            "Existing gate or cost attempt preserved; no retry",
        )
        require(
            not any(collections[(host, "cost")] for host in HOSTS),
            "Cost collection has no bound launch",
        )
    else:
        require(
            all(collections[(host, "fit_validation")] for host in HOSTS),
            "Cost launch lacks both fit collections",
        )
        manager.verify_global_gate(c, gate)
        require(
            len(attempts) == 2 and len({p.parent for p in attempts}) == 1,
            "Exactly one paired cost launch attempt is required",
        )
        round_folder = attempts[0].parent
        result = read(round_folder / "ROUND.json")
        require(
            result.get("status") == "waiting"
            and result.get("phase") == "cost"
            and result.get("transition") == "global_Validation27_passed_cost_launched"
            and set(result.get("hosts", {})) == set(HOSTS),
            "Paired cost launches lack a completed transition",
        )
        for host, value in launches.items():
            validate_launch(c, host, "cost", value)
            require(
                result["hosts"][host] == value,
                "Cost launch differs from completed transition",
            )
            attempt = round_folder / ("COST_LAUNCH_ATTEMPT_" + host + ".json")
            item = read(attempt)
            require(
                item.get("host") == host
                and item.get("contract_sha256") == canonical(c)
                and item.get("global_gate_sha256")
                == sha(gate / "global_validation_gate.json"),
                "Cost attempt binding differs",
            )
            require(
                value.get("global_validation_gate_sha256")
                == sha(gate / "global_validation_gate.json"),
                "Cost launch gate digest differs",
            )
            verify_streams(round_folder, "cost_launch_" + host)
            require(
                json_response(
                    (round_folder / ("cost_launch_" + host + ".stdout")).read_bytes()
                )
                == value,
                "Cost launch raw output differs",
            )
    marker = REPORT / "COLLECTION_COMPLETE.json"
    if marker.exists():
        require(
            all(collections.values()),
            "Collection marker lacks all four verified collections",
        )
        require(
            read(marker).get("status") == "collection_complete"
            and read(marker).get("contract_sha256") == canonical(c)
            and read(marker).get("collections")
            == {
                host + ":" + phase: sha(
                    BUNDLE / "retrieval" / phase / host / "VERIFIED.json"
                )
                for host in HOSTS
                for phase in PHASES
            },
            "Collection marker identity differs",
        )
    if (BUNDLE / "comparison_v1").exists():
        require(
            (REPORT / "COMPLETE.json").exists(),
            "Existing or partial comparison preserved; no automatic overwrite",
        )
    return launches


def check_budget(c, binding, *, current_round=False):
    require(
        all(
            type(c["hosts"][host].get("absolute_host_deadline_unix")) in (int, float)
            and math.isfinite(c["hosts"][host]["absolute_host_deadline_unix"])
            and time.time() < c["hosts"][host]["absolute_host_deadline_unix"]
            for host in HOSTS
        ),
        "Approved absolute host deadline reached",
    )
    count = len(list((REPORT / "rounds").glob("*/INTENT.json")))
    require(
        count <= binding["max_rounds"]
        if current_round
        else count < binding["max_rounds"],
        "Approved continuation round budget reached",
    )


def boundary(binding_sha, authorization_sha):
    binding, c, initial = verify(binding_sha, authorization_sha)
    check_budget(c, binding, current_round=True)
    return binding, c, initial


def cost_launch_program(c, host):
    tmux = c["hosts"][host]["tmux_binary"]
    return f"""import hashlib,json,pathlib,shlex,subprocess,sys,time
from paper.scripts import manage_shared_history_mlp_control as m
root=pathlib.Path({c["hosts"][host]["root"]!r});host={host!r};files=json.load(sys.stdin)
c=m.read(root/'execution_contract.json');m.require(m.canonical(c)=={canonical(c)!r},'Native contract differs')
m.verify_files(c,root/'source');m.guard(c,host);m.verify_native_qualification(c,host,root);m.verify_phase_receipt(c,host,'fit_validation',root)
m.require(not any(root.glob('*.failure.json')),'Preserved native failure')
m.require(not any(root.glob('cost.*')) and not (root/'global_validation_gate.json').exists() and not (root/'global_validation_sources').exists(),'Existing cost or gate attempt; preserve')
for name,item in files.items():
 p=m.below(root,name);m.require(name=='global_validation_gate.json' or name.startswith('global_validation_sources/'),'Unexpected gate path')
 text=item['text'];m.require(hashlib.sha256(text.encode()).hexdigest()==item['sha256'],'Gate bytes changed')
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as out:out.write(text)
m.verify_global_gate(c,root)
h=c['hosts'][host]
command=['env',*[k+'='+str(v) for k,v in h['environment'].items()],h['python'],'-u','-m',{"paper.scripts.manage_shared_history_mlp_control"!r},'--mode','supervise','--contract',str(root/'execution_contract.json'),'--contract-sha',m.sha(root/'execution_contract.json'),'--host',host,'--phase','cost']
result=subprocess.run([{tmux!r},'new-session','-d','-s',h['tmux']+'_cost','-c',h['source_root'],shlex.join(command)],capture_output=True,timeout=15)
m.require(result.returncode==0,'Cost tmux launch failed: '+repr(result.stderr))
deadline=time.monotonic()+10
while not ((root/'cost.claim.json').exists() and (root/'cost.supervisor.intent.json').exists()) and time.monotonic()<deadline:time.sleep(.05)
claim=m.read(root/'cost.claim.json');parent=m.read(root/'cost.supervisor.intent.json')
keys={IDENTITY_KEYS!r}
for expected in (claim['identity'],parent['identity']):
 live=m.proc_identity(expected['pid']);m.require(live is not None and all(live[k]==expected[k] for k in keys),'New cost dispatcher identity differs')
print(json.dumps({{'status':'launched','host':host,'phase':'cost','contract_sha256':m.canonical(c),'identity':claim['identity'],'supervisor_identity':parent['identity'],'global_validation_gate_sha256':m.sha(root/'global_validation_gate.json'),'observed_at':time.time()}}),flush=True)
"""


def launch_cost(c, host, gate_folder, round_folder):
    target = REPORT / ("COST_LAUNCHED_" + host + ".json")
    require(not target.exists(), "Cost phase already launched; no retry")
    regular_files(gate_folder)
    write(
        round_folder / ("COST_LAUNCH_ATTEMPT_" + host + ".json"),
        {
            "host": host,
            "at": now(),
            "contract_sha256": canonical(c),
            "global_gate_sha256": sha(gate_folder / "global_validation_gate.json"),
        },
    )
    payload = {
        str(path.relative_to(gate_folder)): {
            "text": path.read_text(),
            "sha256": sha(path),
        }
        for path in sorted(Path(gate_folder).rglob("*"))
        if path.is_file()
    }
    _, raw = transport(
        ssh_command(c, host, cost_launch_program(c, host)),
        round_folder,
        "cost_launch_" + host,
        timeout=CHECK_SECONDS,
        input_bytes=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
    )
    value = json_response(raw)
    validate_launch(c, host, "cost", value)
    require(
        value.get("status") == "launched"
        and value.get("global_validation_gate_sha256")
        == sha(gate_folder / "global_validation_gate.json"),
        "New cost launch gate differs",
    )
    write(target, value)
    return value


def comparator_module():
    """Load this version's adapter without changing a frozen module's globals."""
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location(
        "history_mlp_execution_v2_compare", ROOT / COMPARATOR
    )
    require(
        spec is not None and spec.loader is not None, "Version 2 comparator is missing"
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(binding, c, folder):
    del binding
    comparator = comparator_module()

    require(
        not (BUNDLE / "comparison_v1").exists(),
        "Existing or partial comparison must be preserved",
    )
    base = [
        sys.executable,
        str(ROOT / COMPARATOR),
        "--contract",
        str(BUNDLE / "execution_contract.json"),
        "--contract-sha",
        sha(BUNDLE / "execution_contract.json"),
        "--retrieval",
        str(BUNDLE / "retrieval"),
        "--output",
        str(BUNDLE / "comparison_v1"),
    ]
    for label, command in (
        ("comparison_precheck", base + ["--check-only"]),
        ("comparison_create", base),
        ("comparison_postcheck", base + ["--check-only"]),
    ):
        transport(command, folder, label, timeout=TRANSFER_SECONDS)
    require_outputs()
    require(
        comparator.SCHEMA == "shared_history_mlp_control_comparison_v1",
        "Comparator identity differs",
    )
    return {"status": "complete", "output": str(BUNDLE / "comparison_v1")}


def require_outputs():
    for name in ("REPORT.json", "REPORT.md", "TABLE.csv"):
        path = BUNDLE / "comparison_v1" / name
        require(
            path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
            "Missing or empty comparison output",
        )


def completion_inventory():
    inventory = {}
    for root in (
        BUNDLE / "retrieval",
        BUNDLE / "global_validation_v1",
        BUNDLE / "comparison_v1",
        REPORT / "rounds",
    ):
        names = regular_files(root)
        require(bool(names), "Empty completion evidence inventory: " + str(root))
        for name in names:
            path = Path(name)
            inventory[name] = {"sha256": sha(path), "bytes": path.stat().st_size}
    for host in HOSTS:
        path = REPORT / ("COST_LAUNCHED_" + host + ".json")
        require(path.is_file() and not path.is_symlink(), "Cost launch receipt missing")
        inventory[str(path)] = {"sha256": sha(path), "bytes": path.stat().st_size}
    require(bool(inventory), "Empty completion inventory")
    return inventory


def verify_completion(binding, c, binding_sha):
    comparator = comparator_module()

    complete = read(REPORT / "COMPLETE.json")
    require(
        complete.get("status") == "complete"
        and complete.get("binding_sha256") == binding_sha
        and complete.get("contract_sha256") == canonical(c)
        and complete.get("files") == completion_inventory(),
        "Cached completion evidence changed or missing",
    )
    marker = REPORT / "COLLECTION_COMPLETE.json"
    require(
        complete.get("collection_complete_sha256") == sha(marker),
        "Collection completion marker changed",
    )
    collections = all_collections(c)
    require(
        all(collections.values()),
        "Cached completion requires all four verified collections",
    )
    validate_state(binding, c, collections)
    manager.verify_global_gate(c, BUNDLE / "global_validation_v1")
    require_outputs()
    result = comparator.aggregate(c, BUNDLE / "retrieval")
    output = BUNDLE / "comparison_v1"
    require(
        read(output / "REPORT.json") == result
        and (output / "REPORT.md").read_text() == comparator.render_markdown(result)
        and (output / "TABLE.csv").read_text() == comparator.render_csv(result),
        "Cached comparison differs from pure recomputation",
    )
    return {
        "status": "complete",
        "cache_verified": True,
        "remote_calls": 0,
        "output": str(output),
    }


def publish_round(folder, value):
    value = {**value, "round_path": str(folder), "observed_at": now()}
    write(folder / "ROUND.json", value)
    pointer = REPORT / (".LATEST." + uuid.uuid4().hex + ".json")
    write(
        pointer,
        {
            "round_path": str(folder / "ROUND.json"),
            "round_sha256": sha(folder / "ROUND.json"),
            "status": value["status"],
            "check_started_unix": read(folder / "INTENT.json")["check_started_unix"],
        },
    )
    os.replace(pointer, REPORT / "LATEST.json")
    return value


def cached_failure():
    path = REPORT / "FAILURE.json"
    if not path.exists():
        require(
            not (REPORT / "FAILURE_SEAL.json").exists(), "Preserved failure was removed"
        )
        return None
    value = read(path)
    seal = read(REPORT / "FAILURE_SEAL.json")
    require(
        seal.get("failure_sha256") == sha(path)
        and value.get("status") == "failed"
        and value.get("automatic_retry") is False
        and value.get("training_failure_inferred_from_transport") is False,
        "Cached failure changed",
    )
    return {**value, "cached_failure": True, "remote_calls": 0}


def preserve_failure(error, folder=None, *, context=None):
    path = REPORT / "FAILURE.json"
    if path.exists():
        return {**read(path), "cached_failure": True, "remote_calls": 0}
    value = {
        "status": "failed",
        "error": type(error).__name__ + ": " + str(error),
        "at": now(),
        "round_path": str(folder) if folder else None,
        "automatic_retry": False,
        "further_remote_calls": 0,
        "training_failure_inferred_from_transport": False,
        **(context or {}),
    }
    write(path, value)
    write(REPORT / "FAILURE_SEAL.json", {"failure_sha256": sha(path)})
    return value


def check_only(binding_sha, authorization_sha):
    binding, c, _initial = verify(binding_sha, authorization_sha)
    with advance_lock() as acquired:
        if not acquired:
            return {"status": "busy", "remote_calls": 0}
        failure = cached_failure()
        if failure:
            return failure
        collections = all_collections(c)
        validate_state(binding, c, collections)
        if (REPORT / "COMPLETE.json").exists():
            return verify_completion(binding, c, binding_sha)
        check_budget(c, binding)
        return {
            "status": "ready_to_advance",
            "remote_calls": 0,
            "collected_fit_hosts": [
                host for host in HOSTS if collections[(host, "fit_validation")]
            ],
            "collected_cost_hosts": [
                host for host in HOSTS if collections[(host, "cost")]
            ],
        }


def advance(binding_sha, authorization_sha):
    binding, c, initial = verify(binding_sha, authorization_sha)
    with advance_lock() as acquired:
        if not acquired:
            return {"status": "busy", "remote_calls": 0}
        failure = cached_failure()
        if failure:
            return failure
        folder = None
        context = {"failure_phase": "local_admission", "collection_complete": False}
        try:
            collections = all_collections(c)
            launches = validate_state(binding, c, collections)
            if (REPORT / "COMPLETE.json").exists():
                return verify_completion(binding, c, binding_sha)
            check_budget(c, binding)
            latest, _ = latest_round()
            if latest and time.time() - latest["check_started_unix"] < INTERVAL_SECONDS:
                return {
                    "status": "waiting_next_check",
                    "remote_calls": 0,
                    "next_check_unix": latest["check_started_unix"] + INTERVAL_SECONDS,
                }
            folder = (
                REPORT
                / "rounds"
                / (
                    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
                    + "_"
                    + uuid.uuid4().hex[:12]
                )
            )
            write(
                folder / "INTENT.json",
                {
                    "check_started_unix": time.time(),
                    "binding_sha256": binding_sha,
                    "contract_sha256": canonical(c),
                    "at": now(),
                },
            )
            statuses = {}
            for host in HOSTS:
                launch = launches.get(host, initial["hosts"][host])
                phase = launch["phase"]
                if collections[(host, phase)] is not None:
                    continue
                context = {
                    "failure_phase": "terminal_check",
                    "failure_host": host,
                    "native_phase": phase,
                    "collection_complete": False,
                }
                boundary(binding_sha, authorization_sha)
                value = short_check(c, host, phase, launch, folder / host)
                statuses[host] = value
                boundary(binding_sha, authorization_sha)
                if value["status"] == "terminal":
                    context["failure_phase"] = "collection"
                    collect(
                        c,
                        host,
                        phase,
                        admission=lambda: boundary(binding_sha, authorization_sha),
                    )
                    boundary(binding_sha, authorization_sha)
                    collections[(host, phase)] = collected(c, host, phase)
            if (
                all(collections[(host, "fit_validation")] for host in HOSTS)
                and not launches
            ):
                context = {
                    "failure_phase": "global_validation_gate",
                    "collection_complete": False,
                }
                boundary(binding_sha, authorization_sha)
                gate = build_global_gate(
                    c,
                    {host: collections[(host, "fit_validation")] for host in HOSTS},
                    BUNDLE / "global_validation_v1",
                )
                for host in HOSTS:
                    context = {
                        "failure_phase": "cost_launch",
                        "failure_host": host,
                        "collection_complete": False,
                    }
                    boundary(binding_sha, authorization_sha)
                    launches[host] = launch_cost(c, host, gate, folder)
                boundary(binding_sha, authorization_sha)
                return publish_round(
                    folder,
                    {
                        "status": "waiting",
                        "phase": "cost",
                        "transition": "global_Validation27_passed_cost_launched",
                        "hosts": launches,
                    },
                )
            if all(collections[(host, "cost")] for host in HOSTS):
                context = {
                    "failure_phase": "local_comparison",
                    "collection_complete": True,
                }
                marker = REPORT / "COLLECTION_COMPLETE.json"
                require(
                    not marker.exists(),
                    "Existing collection completion attempt preserved",
                )
                write(
                    marker,
                    {
                        "status": "collection_complete",
                        "contract_sha256": canonical(c),
                        "at": now(),
                        "collections": {
                            host + ":" + phase: sha(
                                BUNDLE / "retrieval" / phase / host / "VERIFIED.json"
                            )
                            for host in HOSTS
                            for phase in PHASES
                        },
                    },
                )
                result = compare(binding, c, folder)
                verify(binding_sha, authorization_sha)
                result = publish_round(folder, result)
                context["failure_phase"] = "completion_seal"
                write(
                    REPORT / "COMPLETE.json",
                    {
                        "status": "complete",
                        "output": result["output"],
                        "binding_sha256": binding_sha,
                        "contract_sha256": canonical(c),
                        "collection_complete_sha256": sha(marker),
                        "files": completion_inventory(),
                        "at": now(),
                    },
                )
                return {**result, "complete_sha256": sha(REPORT / "COMPLETE.json")}
            return publish_round(
                folder,
                {
                    "status": "waiting",
                    "hosts": statuses,
                    "collected_fit_hosts": [
                        host for host in HOSTS if collections[(host, "fit_validation")]
                    ],
                    "collected_cost_hosts": [
                        host for host in HOSTS if collections[(host, "cost")]
                    ],
                },
            )
        except BaseException as error:
            result = preserve_failure(error, folder, context=context)
            if folder is not None and not (folder / "ROUND.json").exists():
                return publish_round(folder, result)
            return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("advance", "check-only"), required=True)
    parser.add_argument("--binding-sha", required=True)
    parser.add_argument("--authorization-sha", required=True)
    args = parser.parse_args(argv)
    try:
        result = (
            check_only(args.binding_sha, args.authorization_sha)
            if args.mode == "check-only"
            else advance(args.binding_sha, args.authorization_sha)
        )
    except BaseException as error:
        result = preserve_failure(error)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 2 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
