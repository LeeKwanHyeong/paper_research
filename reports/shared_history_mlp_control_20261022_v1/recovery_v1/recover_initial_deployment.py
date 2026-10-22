"""Separately admitted setup recovery under the original scientific contract.

Existing 5080 files are only read. 5090 is first deployed once with importer
cache invalidation. Qualification and initial fits use a new local control state.
Original failure records and all scientific files remain immutable.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import importlib.util
import inspect
import json
from pathlib import Path
import shlex
import sys
import time
import uuid

from paper.scripts import manage_shared_history_mlp_control as manager
from paper.scripts import advance_shared_component_ablation as boundary

ROOT, REPORT, BUNDLE = manager.ROOT, manager.REPORT, manager.BUNDLE
SELF = "reports/shared_history_mlp_control_20261022_v1/recovery_v1/recover_initial_deployment.py"
ORIGINAL = REPORT / "initial_execution_v1/5080/deploy"
STATE = REPORT / "recovery_v1/initial_execution"
SCHEMA = "shared_history_mlp_control_deployment_readonly_recovery_v1"
HOST = "5080"
require, read, sha, write, canonical = (
    manager.require,
    manager.read,
    manager.sha,
    manager.write,
    manager.canonical,
)


def initial_module():
    path = REPORT / "initial_execution.py"
    spec = importlib.util.spec_from_file_location(
        "frozen_history_mlp_initial_execution", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_admission(path, admission_sha, self_sha, recovery_qa_sha):
    path = Path(path)
    require(
        path == REPORT / "recovery_v1/ADMISSION.json"
        and path.is_file()
        and not path.is_symlink()
        and sha(path) == admission_sha,
        "Explicit recovery admission SHA differs",
    )
    admission = read(path)
    require(
        admission.get("status") == "approved_within_existing_setup_scope"
        and admission.get("campaign") == manager.NAME
        and admission.get("user_instruction")
        == "검증 진행하고 실험이 필요하면 동일하게 5080/5090서버에서 진행하자"
        and admission.get("observed_failure_kind")
        == "remote_deployment_bootstrap_command_error"
        and admission.get("observed_ssh_returncode") == 1
        and admission.get("observed_transport_error") is None
        and admission.get("new_fit_started_before_recovery") == 0,
        "Existing setup-scope recovery admission differs",
    )
    required_false = (
        "scientific_retry",
        "additional_network_retry_after_recovery_failure",
        "old_failure_delete",
        "old_controller_state_overwrite",
        "runtime_network_credentials_changes",
        "held_out_test_evaluation",
    )
    require(
        all(admission.get(key) is False for key in required_false)
        and admission.get("MAC_deferred") is True,
        "Recovery scope broadened",
    )
    allowed = admission.get("allowed_initial_actions", {})
    require(
        allowed.get("5080_existing_deployment_admission")
        == {
            "read_only_ssh_calls": 1,
            "timeout_seconds": 45,
            "require_all_actual_sha_runtime_boot_gpu_guard": True,
            "reupload": False,
        }
        and allowed.get("5090_first_deployment")
        == {
            "transfers": 1,
            "timeout_seconds": 900,
            "per_process_import_cache_invalidation": True,
        },
        "Recovery transport bounds differ",
    )
    pins = admission.get("pins", {})
    require(
        set(pins)
        == {
            "original_authorization",
            "original_independent_qa",
            "science",
            "integrity",
            "original_failure",
            "original_transport",
        },
        "Original pin inventory differs",
    )
    for item in pins.values():
        target = Path(item["path"])
        require(
            target.is_file()
            and not target.is_symlink()
            and target.resolve().is_relative_to(ROOT.resolve())
            and sha(target) == item["sha256"],
            "Original pinned evidence changed",
        )
    require(sha(ROOT / SELF) == self_sha, "Recovery driver changed")
    qa_path = REPORT / "recovery_v1/INDEPENDENT_QA.json"
    require(
        qa_path.is_file()
        and not qa_path.is_symlink()
        and sha(qa_path) == recovery_qa_sha,
        "Independent recovery QA pin differs",
    )
    qa = read(qa_path)
    require(
        qa.get("status") == "PASS"
        and qa.get("independent_reviewer") is True
        and qa.get("files_reviewed_sha256", {}).get(SELF) == self_sha,
        "Independent recovery QA is missing",
    )
    for name, digest in qa["files_reviewed_sha256"].items():
        target = ROOT / name
        require(
            target.is_file()
            and not target.is_symlink()
            and target.resolve().is_relative_to(ROOT.resolve())
            and sha(target) == digest,
            "Recovery reviewed source changed",
        )
    old_qa = read(pins["original_independent_qa"]["path"])
    old_driver = "reports/shared_history_mlp_control_20261022_v1/initial_execution.py"
    require(
        sha(ROOT / old_driver) == old_qa["files_reviewed_sha256"][old_driver],
        "Frozen initial driver changed",
    )
    initial = initial_module()
    c, integrity = initial.verify_local(
        pins["integrity"]["sha256"],
        pins["original_authorization"]["sha256"],
        pins["original_independent_qa"]["sha256"],
        old_qa["files_reviewed_sha256"][old_driver],
    )
    actual = read(ORIGINAL / "transport.json")
    require(
        actual.get("actual_os_returncode") == 1
        and actual.get("timeout") is False
        and actual.get("transport_error") is None,
        "Preserved failure is not the admitted post-extraction import error",
    )
    for stream in ("stdout", "stderr"):
        target = ORIGINAL / ("transport." + stream)
        require(
            sha(target) == actual[stream + "_sha256"]
            and target.stat().st_size == actual[stream + "_bytes"],
            "Original transport stream changed",
        )
    require(
        b"ModuleNotFoundError: No module named 'paper'"
        in (ORIGINAL / "transport.stderr").read_bytes(),
        "Different native deployment error",
    )
    require(
        len(c["source"]["files"]) == 991
        and len(c["input_files"]) == 9
        and len(c["legacy_files"]) == 199,
        "Exact uploaded scientific/input/legacy inventory differs",
    )
    return admission, c, integrity, initial


def remote_program(c, integrity, initial):
    h = c["hosts"][HOST]
    helper = inspect.getsource(initial.check_archive)
    return (
        "import hashlib,importlib,json,os,pathlib,sys\n"
        + helper
        + f"""
root=pathlib.Path({h["root"]!r});source=root/'source';host='5080'
def demand(value,reason):
 if not value:raise RuntimeError(reason)
demand(root.is_dir() and not root.is_symlink(),'Existing isolated root missing or symbolic')
demand(sys.dont_write_bytecode,'Recovery must not write Python bytecode')
raw=(root/'deployment.tar.gz').read_bytes()
demand(hashlib.sha256(raw).hexdigest()=={integrity["archive_sha256"]!r},'Original uploaded archive changed')
c=check_archive(raw,{integrity["contract_sha256"]!r},{canonical(c)!r})
demand(hashlib.sha256((root/'execution_contract.json').read_bytes()).hexdigest()=={integrity["contract_sha256"]!r},'Extracted contract changed')
expected={{'execution_contract.json','deployment.tar.gz'}}|{{'source/'+name for name in {{**c['source']['files'],**c['input_files'],**c['legacy_files']}}}}
paths=list(root.rglob('*'))
demand(not any(p.is_symlink() for p in paths),'Symbolic uploaded member')
demand({{str(p.relative_to(root)) for p in paths if p.is_file()}}==expected,'Existing upload file inventory differs')
for name,digest in {{**c['source']['files'],**c['input_files'],**c['legacy_files']}}.items():
 p=source/name;demand(hashlib.sha256(p.read_bytes()).hexdigest()==digest,'Extracted member SHA changed: '+name)
demand(not any(root.glob('qualification.*')) and not any(root.glob('fit_validation.*')) and not any(root.glob('cost.*')),'Native phase already attempted; no recovery launch')
# Reject any attempted filesystem mutation inside the exact deployed root.
def audit(event,args):
 if event=='open':
  path,mode,flags=args
  if isinstance(path,(str,bytes)):
   resolved=pathlib.Path(os.fsdecode(path)).resolve()
   if resolved.is_relative_to(root):
    demand(not (isinstance(mode,str) and any(ch in mode for ch in 'wax+')) and not (flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)),'Recovery attempted a write')
 elif event in ('os.mkdir','os.remove','os.rmdir','os.rename','os.link','os.symlink','os.chmod','os.chown','os.utime'):
  for path in args[:2]:
   if isinstance(path,(str,bytes)):
    demand(not pathlib.Path(os.fsdecode(path)).resolve().is_relative_to(root),'Recovery attempted deployed filesystem mutation')
sys.addaudithook(audit)
sys.path.insert(0,str(source));importlib.invalidate_caches()
from paper.scripts import manage_shared_history_mlp_control as m
m.validate(c);m.verify_files(c,source);gpu=m.guard(c,host)
h=c['hosts'][host]
demand(pathlib.Path(sys.executable).resolve()==pathlib.Path(h['python']).resolve(),'Python interpreter changed')
for name,value in h['environment'].items():demand(os.environ.get(name)==str(value),'Native environment changed: '+name)
from paper.scripts.quantity_comparison_runtime import configure_runtime,runtime_identity
configure_runtime('cuda:0',threads=h['runtime_expected']['threads'])
runtime=runtime_identity('cuda:0')
demand(all(runtime.get(key)==value for key,value in h['runtime_expected'].items()),'Native runtime changed')
demand(runtime['gpu']['uuid']==h['gpu_uuid'],'Native GPU UUID changed')
m.guard(c,host,active=(os.getpid(),))
print(json.dumps({{'status':'existing_deployment_verified','host':host,'read_only_verification':True,'contract_sha256':m.canonical(c),'archive_sha256':{integrity["archive_sha256"]!r},'source_files_sha256':c['source']['files_sha256'],'source_count':991,'input_count':9,'legacy_count':199,'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'gpu_before':gpu,'runtime':runtime,'import_caches_invalidated':True,'scientific_processes_launched':0,'files_reuploaded':0,'files_removed_or_overwritten':0}}),flush=True)
"""
    )


def verify_pass(c, host):
    """Verify newly stored native raw qualification evidence without old globals."""
    folder = STATE / host
    value = read(folder / "QUALIFICATION_PASS.json")
    require(
        value.get("status") == "PASS"
        and value.get("host") == host
        and value.get("contract_sha256") == canonical(c),
        "Qualification binding differs",
    )
    snapshot = manager.below(folder, value["snapshot"])
    files = value.get("files")
    require(
        isinstance(files, dict) and bool(files) and not snapshot.is_symlink(),
        "Missing qualification files",
    )
    paths = list(snapshot.rglob("*"))
    require(
        not any(p.is_symlink() for p in paths)
        and {str(p.relative_to(snapshot)) for p in paths if p.is_file()} == set(files),
        "Qualification inventory differs",
    )
    for name, digest in files.items():
        require(
            sha(manager.below(snapshot, name)) == digest,
            "Qualification raw evidence changed",
        )
    manager.verify_phase_receipt(c, host, "qualification", snapshot)
    return value


def deployment_5090_program(c, integrity, initial):
    program = initial.deployment_program(c, "5090", integrity)
    marker = "sys.path.insert(0,str(root/'source'))\n"
    require(program.count(marker) == 1, "Frozen bootstrap marker differs")
    return program.replace(
        marker, marker + "import importlib;importlib.invalidate_caches()\n"
    )


def _action(name, host, admission_path, admission_sha, self_sha, recovery_qa_sha):
    require(
        name
        in (
            "verify-existing",
            "deploy",
            "launch-qualification",
            "check-qualification",
            "launch-fit",
        )
        and host in ("5080", "5090"),
        "Unsupported recovery action or host",
    )
    _, c, integrity, initial = verify_admission(
        admission_path, admission_sha, self_sha, recovery_qa_sha
    )
    require(
        not any(STATE.rglob("FAILURE.json")),
        "Preserved recovery failure; no additional remote call",
    )
    require(
        all((p.parent / "RESULT.json").is_file() for p in STATE.rglob("ATTEMPT.json")),
        "Partial recovery attempt; no additional remote call",
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
            "admission_sha256": admission_sha,
            "recovery_driver_sha256": self_sha,
            "contract_sha256": canonical(c),
        },
    )
    try:
        h = c["hosts"][host]
        phase = "fit_validation" if name == "launch-fit" else "qualification"
        if name == "verify-existing":
            require(host == "5080", "Only existing 5080 deployment may be adopted")
            program, timeout, payload = remote_program(c, integrity, initial), 45, None
        elif name == "deploy":
            require(
                host == "5090", "5080 deployment must never be repeated or reuploaded"
            )
            require(
                read(STATE / "5080/verify-existing/RESULT.json").get("status")
                == "existing_deployment_verified",
                "Existing 5080 deployment must be admitted first",
            )
            program, timeout = deployment_5090_program(c, integrity, initial), 900
            payload = (BUNDLE / "deployment.tar.gz").read_bytes()
        else:
            for peer, step, status in (
                ("5080", "verify-existing", "existing_deployment_verified"),
                ("5090", "deploy", "deployed"),
            ):
                receipt = read(STATE / peer / step / "RESULT.json")
                require(
                    receipt.get("status") == status
                    and receipt.get("contract_sha256") == canonical(c),
                    "Both deployment admissions are required",
                )
            if name == "launch-fit":
                for peer in ("5080", "5090"):
                    verify_pass(c, peer)
            if name == "check-qualification":
                launch = read(host_root / "launch-qualification/RESULT.json")
                initial.controller.validate_launch(c, host, phase, launch)
                program = initial.qualification_program(c, host, launch)
            else:
                program = initial.launch_program(c, host, phase)
            timeout, payload = 45, None
        compile(program, "<recovery-native-program>", "exec")
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
            "Recovery response identity differs",
        )
        if name == "verify-existing":
            require(
                value.get("status") == "existing_deployment_verified"
                and value.get("read_only_verification") is True
                and value.get("archive_sha256") == integrity["archive_sha256"]
                and value.get("source_files_sha256") == c["source"]["files_sha256"]
                and value.get("scientific_processes_launched") == 0
                and value.get("files_reuploaded") == 0
                and value.get("files_removed_or_overwritten") == 0,
                "Readonly admission differs",
            )
        elif name == "deploy":
            require(
                value.get("status") == "deployed"
                and value.get("archive_sha256") == integrity["archive_sha256"],
                "First 5090 deployment not confirmed",
            )
        elif name.startswith("launch-"):
            require(value.get("status") == "launched", "Native launch not confirmed")
            initial.controller.validate_launch(c, host, phase, value)
        else:
            require(
                value.get("status") in ("waiting", "terminal")
                and value.get("read_only_check") is True,
                "Unknown native qualification status",
            )
            value["checked_unix"] = time.time()
            if value["status"] == "terminal":
                evidence = value.get("evidence", {})
                require(bool(evidence), "Missing actual native proof")
                snapshot = folder / "snapshot"
                snapshot.mkdir()
                import hashlib

                for entry, item in evidence.items():
                    target = manager.below(snapshot, entry)
                    raw_bytes = base64.b64decode(item["base64"], validate=True)
                    require(
                        len(raw_bytes) == item["bytes"]
                        and hashlib.sha256(raw_bytes).hexdigest() == item["sha256"],
                        "Native proof bytes changed",
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("xb") as stream:
                        stream.write(raw_bytes)
                manager.verify_phase_receipt(c, host, phase, snapshot)
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
    with (STATE / "recovery.lock").open("a+") as stream:
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
        choices=(
            "verify-existing",
            "deploy",
            "launch-qualification",
            "check-qualification",
            "launch-fit",
        ),
    )
    parser.add_argument("--host", required=True, choices=("5080", "5090"))
    parser.add_argument("--admission", dest="admission_path", type=Path, required=True)
    parser.add_argument("--admission-sha", required=True)
    parser.add_argument("--self-sha", required=True)
    parser.add_argument("--recovery-qa-sha", required=True)
    args = vars(parser.parse_args(argv))
    name, host = args.pop("action"), args.pop("host")
    print(json.dumps(action(name, host, **args), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
