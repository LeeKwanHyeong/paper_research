"""One-attempt operational overlay for the explicitly approved fixed cap9 study.

The immutable preparation contract and source134 remain intact. All deployment
and launch actions are gated locally before constructing the remote transport.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
import types

PROJECT = Path(__file__).resolve().parents[2]
PREPARED = PROJECT / "search_artifacts/titantpp_gru_residual_cap_validation_20261014_v1"
BUNDLE = PREPARED / "execution"
PARENT = PROJECT / "search_artifacts/titantpp_cnn_gru_components_20261012_v5"
SCHEMA = PREPARED.name
PREPARED_RAW = "fae92b1695483ee4c978a6f72ddff7628881fc9bb2af42b246508835bddf06c9"
PREPARED_CANONICAL = "6b5564173eb6f9feed35bc89142bfdde50944c285c4a402d2a4f5b311e2983f0"
PREPARATION_RECEIPT_RAW = "709dc99c777d5b199820fc1d40baf65a4a534d7e26808a018c64ab678004cb68"
CAPPED = "titantpp_cnn_gru54_residual_cap025"
USER_APPROVAL = "ㅇㅇ 실행 승인할게. 고정 계약과 기존 CNN·FULL 결과는 유지하고, GPU·Runtime·입력 SHA 검증을 통과하면 GRU 보정량 제한 후보 9개를 학습하자. 끝나면 3seed Validation 결과를 기존 CNN·FULL과 비교해서 정리해줘.\n그리고 지금 이 작업을 왜하는지 명확하게 기억해둬. 그래야 방향성이 흔들리지 않지"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe(root, name):
    root = Path(root).resolve()
    name = Path(name)
    require(not name.is_absolute() and ".." not in name.parts, "Unsafe relative path")
    target = root / name
    require(not target.is_symlink() and target.resolve().is_relative_to(root), "Symlink/path escape")
    return target


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def frozen_preparation():
    require(sha(PREPARED / "execution_contract.json") == PREPARED_RAW, "Frozen science contract changed; remote0")
    require(sha(PREPARED / "preparation_receipt.json") == PREPARATION_RECEIPT_RAW, "Preparation evidence changed; remote0")
    prepared = read(PREPARED / "execution_contract.json")
    require(canonical(prepared) == PREPARED_CANONICAL, "Science canonical mismatch")
    receipt = read(PREPARED / "preparation_receipt.json")
    for name, expected in receipt["verified_reference_files"].items():
        require(sha(name) == expected, "Frozen reference changed: " + name)
    for name, expected in prepared["source"]["files"].items():
        require(sha(safe(PREPARED / "source", name)) == expected, "Frozen source changed: " + name)
    parent = read(PARENT / "execution_contract.json")
    require(sha(PARENT / "execution_contract.json") == prepared["parent"]["contract_raw_sha256"], "Parent changed")
    return prepared, parent


def validate_contract(c, prepared=None, parent=None):
    if prepared is None:
        prepared, parent = frozen_preparation()
    require(c["schema"] == SCHEMA and c["science_preparation"] == {
        "raw_sha256": PREPARED_RAW, "canonical_sha256": PREPARED_CANONICAL,
        "path": "preparation/execution_contract.json"}, "Preparation bridge changed")
    for key in ("candidate", "arms", "seeds", "datasets", "input_files", "training", "objective_weights",
                "limits", "jobs", "source", "analysis", "baseline_references", "baseline_references_sha256"):
        require(c[key] == prepared[key], "Fixed science changed: " + key)
    require(c["evaluation_scope"] == "validation_only" and c["held_out_test_evaluated"] is False,
            "Only Train/Validation authorized")
    require(c["execution_authorization"] == {"approved": True, "user_instruction": USER_APPROVAL,
        "fresh_fit_count": 9, "Test": False, "automatic_retry": False,
        "preserve_original_CNN_FULL": True}, "Execution approval scope changed")
    require(c["runner_modules"] == {CAPPED: "paper.scripts.run_titantpp_gru_residual_cap"}, "Foreign candidate runner")
    require(c["parallelism"] == {"mode": "sequential_per_host", "workers_per_host": 1}, "Concurrency changed")
    require(set(c["hosts"]) == {"5080", "5090"}, "Host set changed")
    for host, spec in c["hosts"].items():
        old, planned = parent["hosts"][host], prepared["hosts"][host]
        root = planned["planned_root"]
        for key in ("python", "gpu_uuid", "runtime_expected"):
            require(spec[key] == planned[key], "Native identity changed")
        require(spec["root"] == root and spec["source_root"] == root + "/source"
            and spec["operation_root"] == root + "/operation" and spec["output_dir"] == root + "/run"
            and spec["tmux"] == SCHEMA + "_" + host
            and spec["tmux_binary"] == old["tmux_binary"], "Foreign root/tmux")
        environment = {k: v.replace(old["root"], root) for k, v in old["environment"].items()}
        require(spec["environment"] == environment, "Native environment changed")
    require(canonical(c["operation"]["files"]) == c["operation"]["files_sha256"], "Operation closure mismatch")
    return c


def prepare_execution(bundle=BUNDLE):
    """Create only this approved overlay, never replace the preparation evidence."""
    from paper.scripts import launch_titantpp_cnn_gru_components_v5 as parent_launcher
    bundle = Path(bundle).resolve()
    require(bundle == BUNDLE and not bundle.exists(), "Execution overlay already exists or foreign path")
    prepared, parent = frozen_preparation()
    c = deepcopy(prepared)
    c.update(status="authorized_pending_native_qualification", evaluation_scope="validation_only",
        science_preparation={"raw_sha256": PREPARED_RAW, "canonical_sha256": PREPARED_CANONICAL,
            "path": "preparation/execution_contract.json"},
        execution_authorization={"approved": True, "user_instruction": USER_APPROVAL, "fresh_fit_count": 9,
            "Test": False, "automatic_retry": False, "preserve_original_CNN_FULL": True},
        runner_modules={CAPPED: "paper.scripts.run_titantpp_gru_residual_cap"},
        parallelism={"mode": "sequential_per_host", "workers_per_host": 1})
    c["hosts"] = {}
    for host, planned in prepared["hosts"].items():
        spec = deepcopy(parent["hosts"][host]); oldroot = spec["root"]; root = planned["planned_root"]
        spec.update(root=root, source_root=root + "/source", operation_root=root + "/operation",
            output_dir=root + "/run", tmux=SCHEMA + "_" + host, old_owner_roots=[],
            assigned_datasets=list(dict.fromkeys(j["dataset"] for j in c["jobs"] if j["host"] == host)))
        spec["environment"] = {k: v.replace(oldroot, root) for k, v in spec["environment"].items()}
        c["hosts"][host] = spec
    operations = {
        "operation/campaign.py": PROJECT / "paper/scripts/run_titantpp_gru_cap_campaign.py",
        "operation/frozen_runtime.py": PARENT / "operation/frozen_runtime.py",
        "operation/pipeline.py": PARENT / "operation/pipeline.py",
        "operation/native_checks.py": PROJECT / "paper/scripts/qualify_titantpp_gru_cap_native.py",
    }
    c["operation"] = {"files": {k: sha(v) for k, v in operations.items()}}
    c["operation"]["files_sha256"] = canonical(c["operation"]["files"])
    validate_contract(c, prepared, parent)
    members = {**{f"source/{name}": safe(PREPARED / "source", name) for name in c["source"]["files"]},
        **{f"source/{name}": safe(Path(prepared["inherited_input_root"]), name) for name in c["input_files"]},
        **operations, "preparation/execution_contract.json": PREPARED / "execution_contract.json"}
    for name, expected in c["input_files"].items():
        require(sha(members["source/" + name]) == expected, "Input bytes mismatch before deployment")
    bundle.mkdir()
    for name, source in members.items():
        target = safe(bundle, name); target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)
    write(bundle / "execution_contract.json", c)
    members["execution_contract.json"] = bundle / "execution_contract.json"
    expected = {name: sha(safe(bundle, name)) for name in members}
    archive = bundle / "deployment.tar.gz"
    with tarfile.open(archive, "x:gz") as output:
        for name in sorted(expected):
            output.add(safe(bundle, name), arcname=name, recursive=False)
    parent_launcher.inspect_archive(archive, expected)
    receipt = {"contract_raw_sha256": sha(bundle / "execution_contract.json"),
        "contract_sha256": canonical(c), "payload_sha256": sha(archive), "members": expected,
        "preparation_raw_sha256": PREPARED_RAW, "created_unix": time.time(),
        "status": "authorized_not_launched", "started_fits": 0}
    write(bundle / "preparation_receipt.json", receipt)
    write(bundle / "launch_hold.json", {"contract_sha256": canonical(c), "native_qualification_required": True})
    for host in c["hosts"]:
        approval = {"approved": True, "explicit_new_user_instruction": True, "user_instruction": USER_APPROVAL,
            "contract_sha256": canonical(c), "hosts": [host], "new_training_authorized": True,
            "scope": "fixed_cap9_train_validation_native_qualification_and_completion_analysis",
            "created_unix": time.time()}
        path = bundle / ("launch_authorization_" + host + ".json"); write(path, approval)
        write(bundle / ("launch_release_" + host + ".json"), {"released": True,
            "contract_sha256": canonical(c), "original_hold_raw_sha256": sha(bundle / "launch_hold.json"),
            "authorization_raw_sha256": sha(path)})
    local_tools = [Path(parent_launcher.__file__), Path(__file__),
        PROJECT / "paper/scripts/titantpp_components_status_transport_v2.py"]
    write(bundle / "execution_integrity.json", {"files": {
        str(p.relative_to(bundle)): sha(p) for p in sorted(bundle.rglob("*")) if p.is_file()},
        "local_tools": {str(p.relative_to(PROJECT)): sha(p) for p in local_tools},
        "created_unix": time.time()})
    return {**receipt, "execution_integrity_raw_sha256": sha(bundle / "execution_integrity.json")}


def check_bundle(bundle=BUNDLE):
    bundle = Path(bundle).resolve(); require(bundle == BUNDLE, "Foreign execution bundle; remote0")
    integrity = read(bundle / "execution_integrity.json")
    for name, expected in integrity["files"].items():
        require(sha(safe(bundle, name)) == expected, "Execution evidence drift: " + name)
    for name, expected in integrity["local_tools"].items():
        require(sha(safe(PROJECT, name)) == expected, "Local transport changed")
    c = validate_contract(read(bundle / "execution_contract.json"))
    require(sha(bundle / "preparation/execution_contract.json") == PREPARED_RAW, "Prepared bridge changed")
    receipt = read(bundle / "preparation_receipt.json")
    require(receipt["contract_sha256"] == canonical(c)
        and receipt["contract_raw_sha256"] == sha(bundle / "execution_contract.json")
        and receipt["payload_sha256"] == sha(bundle / "deployment.tar.gz"), "Payload/contract mismatch")
    return c, {"status": "verified", "contract_sha256": canonical(c), "jobs": 9, "remote_calls": 0}


def qualification_binding(snapshot, c, host, launcher):
    files = launcher.decode_snapshot(snapshot, c, host)
    require(not snapshot["gpu_pids"], "Qualification GPU still occupied")
    require(not any(name.endswith("failure.json") for name in files), "Preserved failure; no retry")
    def value(name):
        require(name in files, "Qualification pending: " + name)
        return json.loads(files[name])
    q = value("qualification/receipt.json")
    exit_record = value("phases/qualify_exit.json")
    launch = value("phases/qualify_launch.json")
    complete = value("phases/qualify_complete.json")
    require(q["status"] == "passed" and q["host"] == host and q["contract_sha256"] == canonical(c)
        and q["source_files_sha256"] == c["source"]["files_sha256"]
        and q["held_out_test_evaluated"] is False, "Qualification identity mismatch")
    require(exit_record["os_exit_code"] == 0 and complete["subprocess_os_exit_code"] == 0
        and complete["phase"] == "qualify" and complete["host"] == host and not complete["gpu_compute_pids"]
        and exit_record["contract_sha256"] == canonical(c) and complete["contract_sha256"] == canonical(c),
        "Actual qualification exit incomplete")
    require(all(launch[k] == exit_record[k] for k in ("pid", "ppid", "argv", "contract_sha256")), "Qualification process identity mismatch")
    native = q["native_qualification_exit"]
    require(native == value("qualification/native_exit.json")
        and native["actual_child_exit_observed"] is True and native["returncode"] == 0
        and native["contract_sha256"] == canonical(c) and native["ppid"] == launch["pid"]
        and type(native["pid"]) is int and native["pid"] != native["ppid"]
        and native["worker_group_pid"] == native["pid"], "Native qualification guard exit mismatch")
    argv = native["argv"]; spec = c["hosts"][host]
    require(len(argv) == 12 and argv[:3] == ["timeout", "--signal=TERM", "--kill-after=15s"]
        and argv[3].isdigit() and 0 < int(argv[3]) <= c["limits"]["qualification_seconds"]
        and argv[4:] == [spec["python"], spec["operation_root"] + "/campaign.py", "--contract",
            spec["root"] + "/execution_contract.json", "--host", host, "--mode", "qualify-worker"],
        "Native qualification timeout/command mismatch")
    runtime = q["runtime"]
    require(runtime["python"] == "3.12.13" and runtime["gpu"]["uuid"] == c["hosts"][host]["gpu_uuid"]
        and all(runtime[k] == v for k, v in c["hosts"][host]["runtime_expected"].items()), "Qualification Runtime mismatch")
    verify_qualification_coverage(q, c, host)
    for name, expected in q["qualification_files"].items():
        require(name in files and hashlib.sha256(files[name]).hexdigest() == expected, "Qualification evidence mismatch")
    return {"receipt": q, "raw_sha256": hashlib.sha256(files["qualification/receipt.json"]).hexdigest(), "os_exit_code": 0}


def verify_qualification_coverage(q, c, host):
    """Reject a passed label that lacks the owned input/seed/native checks."""
    jobs = [j for j in c["jobs"] if j["host"] == host]
    datasets = {j["dataset"] for j in jobs}
    checks = {j["dataset"] + "__" + str(j["seed"]): j for j in jobs}
    require(q["scientific_optimizer_updates"] == 0 and q["held_out_test_evaluated"] is False
        and q["real_Validation_predictions"] is False, "Qualification exceeded synthetic scope")
    require(set(q["inputs"]) == datasets and set(q["initialization"]) == datasets
        and set(q["synthetic"]) == set(checks), "Qualification dataset/seed coverage incomplete")
    for dataset in datasets:
        data = next(d for d in c["datasets"] if d["dataset_id"] == dataset)
        item = q["inputs"][dataset]; receipt = item["input_receipt"]
        require(canonical(receipt) == item["input_receipt_sha256"]
            and receipt["held_out_materialized"] is False
            and receipt["populations"] == data["inherited_data_identity"]["populations"],
            "Qualification Train/Validation identity mismatch")
        require(set(item["interface_sha256"]) == {CAPPED}
            and valid_sha(item["interface_sha256"][CAPPED]), "Qualified interface binding missing")
        initial = q["initialization"][dataset]
        require(set(initial) == {str(s) for s in c["seeds"]}, "Initialization seed coverage incomplete")
        require(all(set(v) == {CAPPED} and valid_sha(v[CAPPED]) for v in initial.values()),
            "Candidate initialization SHA missing")
    for name, job in checks.items():
        result = q["synthetic"][name]
        data = next(d for d in c["datasets"] if d["dataset_id"] == job["dataset"])
        require(result["status"] == "passed" and result["device"] == "cuda:0"
            and result["seed"] == job["seed"] and result["shape"] == [128, data["loader"]["max_seq_len"]]
            and result["synthetic_optimizer_updates"] == 4 and result["scientific_optimizer_updates"] == 0
            and result["held_out_test_evaluated"] is False and result["real_data_loaded"] is False,
            "Native synthetic execution coverage mismatch")
        require(result["initial_FULL_candidate_tensors_outputs"] == "exact"
            and result["registered_parameters"] == job["expected_registered_parameters"]
            and result["active_parameters"] == job["expected_active_parameters"]
            and result["initial_state_sha256"][CAPPED] == q["initialization"][job["dataset"]][str(job["seed"])][CAPPED],
            "Native initial tensor/parameter identity mismatch")
        require(result["cap_bound_zero_unsaturated_saturated"] == "passed"
            and result["all_gradients_finite"] is True and result["correction_parameter_names_with_gradient"]
            and result["withheld_target"] == "invariant" and result["future_prefix"] == "invariant"
            and type(result["peak_allocated_bytes"]) is int and result["peak_allocated_bytes"] > 0,
            "Native cap/gradient/causal/memory checks incomplete")


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def launcher_for(bundle=BUNDLE):
    from paper.scripts import launch_titantpp_cnn_gru_components_v5 as original
    namespace = types.ModuleType("isolated_cap9_launcher")
    namespace.__file__ = original.__file__
    exec(compile(Path(original.__file__).read_text(), original.__file__, "exec"), namespace.__dict__)
    namespace.SCHEMA = SCHEMA
    namespace.check_bundle = check_bundle
    namespace.validate_contract = validate_contract
    namespace.expected_members = lambda c, raw: read(Path(bundle) / "preparation_receipt.json")["members"]
    namespace.qualification_binding = lambda snapshot, c, host: qualification_binding(snapshot, c, host, namespace)
    require(namespace.REMOTE_SCRIPT == original.REMOTE_SCRIPT, "Frozen transport changed")
    return namespace


def run_mode(bundle, mode, host):
    launcher = launcher_for(bundle)
    if mode != "status":
        return launcher.run_mode(bundle, mode, host)
    from paper.scripts.titantpp_components_status_transport_v2 import read_status_once
    c, _ = check_bundle(bundle)
    path, snapshot = read_status_once(bundle, launcher, c, host, Path(bundle) / "control/diagnostics")
    binding = qualification_binding(snapshot, c, host, launcher)
    cache = Path(bundle) / "control/qualification_cache" / host / "snapshot.json"
    if not cache.exists():
        write(cache, snapshot)
    return {"host": host, "status": "qualification_passed", "snapshot_path": str(path),
        "snapshot_raw_sha256": sha(path), "qualification_sha256": binding["raw_sha256"]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=BUNDLE)
    p.add_argument("--mode", choices=("prepare", "check", "deploy", "qualify", "status", "train"), required=True)
    p.add_argument("--host", choices=("5080", "5090"), default="5080")
    p.add_argument("--integrity-sha")
    a = p.parse_args(argv)
    if a.mode != "prepare":
        require(a.integrity_sha and sha(a.bundle / "execution_integrity.json") == a.integrity_sha,
                "An externally pinned execution-integrity SHA is required; remote0")
    result = prepare_execution(a.bundle) if a.mode == "prepare" else run_mode(a.bundle, a.mode, a.host)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
