"""Prepare an immutable dual-host CNN/GRU study, without remote execution.

The original 129 scientific files and seven opaque input files are SHA checked.
New component routes are explicitly recorded as extensions, not old-commit code.
Preparation creates no launch approval, permit, SSH connection or training job.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

PROJECT = Path(__file__).resolve().parents[2]
SCHEMA = "titantpp_cnn_gru_components_20261012_v1"
PARENT = PROJECT / "search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1"
PARENT_CONTRACT = "518467643d248ba7384482cdb9c8806877392be0eaa81b961992cfa1a0381d6b"
PARENT_SOURCE = "c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a"
BASE_REVISION = "57da14a2a7a5687e1bdef7d3164f3e8c0051da59"
ARMS = ["titantpp_cnn_gru54_correction_disabled", "titantpp_cnn_gru54_cnn_only",
        "titantpp_cnn_gru54_gru_only", "titantpp_cnn_gru54"]
ASSIGNMENTS = {
    "5080": [[d, s] for d in ("yellow_trip_hourly", "raf_spare_parts") for s in (42, 52, 62)],
    "5090": [["intermittent_frozen_5000", s] for s in (42, 52, 62)],
}
EXTENSIONS = ("models/TPPs/CountAwareTitanCNNGRUComponents.py",
              "paper/scripts/run_titantpp_cnn_gru_components.py")
OPERATIONS = {
    "operation/frozen_runtime.py": "paper/scripts/titantpp_contribution_runtime.py",
    "operation/campaign.py": "paper/scripts/run_titantpp_cnn_gru_component_campaign.py",
    "operation/qualification.py": "paper/scripts/qualify_titantpp_cnn_gru_components.py",
    "operation/pipeline.py": "paper/scripts/run_titantpp_cnn_gru_component_phase.py",
}
LOCAL_TOOLS = (*EXTENSIONS, *OPERATIONS.values(),
    "paper/scripts/prepare_titantpp_cnn_gru_components.py",
    "paper/scripts/launch_titantpp_cnn_gru_components.py",
    "paper/tests/test_titantpp_cnn_gru_components.py",
    "paper/tests/test_titantpp_component_campaign.py",
    "paper/tests/test_titantpp_component_preparation.py",
    "paper/tests/test_titantpp_component_launcher.py")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe path")
    result = (Path(root) / relative).resolve()
    require(result.is_relative_to(Path(root).resolve()), "Path escape")
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def contract(parent):
    c = {k: deepcopy(parent[k]) for k in ("datasets", "dataset_sha256", "input_files",
        "training", "diagnostic", "limits", "policy")}
    c.update(schema=SCHEMA, stage="components", arms=ARMS, seeds=[42, 52, 62],
        assignments=deepcopy(ASSIGNMENTS), evaluation_scope="validation_only",
        held_out_test_evaluated=False,
        runner_modules={arm: "paper.scripts.run_titantpp_cnn_gru_components" for arm in ARMS},
        parallelism={"maximum_workers": 3,
            "grouping": "same_dataset_seed_four_arms_waves_3_plus_1",
            "per_process_cuda_fraction": .30, "aggregate_gpu_fraction": .8})
    c["hosts"] = {}
    for host, groups in ASSIGNMENTS.items():
        spec = deepcopy(parent["hosts"][host])
        root = str(Path(spec["root"]).parent / (SCHEMA + "_" + host))
        spec.update(root=root, source_root=root + "/source", operation_root=root + "/operation",
            output_dir=root + "/run", tmux=SCHEMA + "_" + host,
            assigned_datasets=list(dict.fromkeys(d for d, s in groups)), old_owner_roots=[])
        spec["environment"].update(PYTHONPATH=root + "/source", MPLCONFIGDIR=root + "/cache/matplotlib",
            XDG_CACHE_HOME=root + "/cache")
        c["hosts"][host] = spec
    original = deepcopy(parent["source"]["files"])
    extensions = {name: sha(PROJECT / name) for name in EXTENSIONS}
    files = {**original, **extensions}
    require(len(files) == 131, "Expected parent129 plus two routes")
    c["source"] = {"git_revision": BASE_REVISION, "base_git_revision": BASE_REVISION,
        "frozen_science_git_revision": BASE_REVISION,
        "frozen_science_files": original, "frozen_science_files_sha256": PARENT_SOURCE,
        "extension_files": extensions, "extension_provenance": "uncommitted component extension pinned by SHA256",
        "files": files, "files_sha256": canonical(files)}
    op = {name: sha(PROJECT / origin) for name, origin in OPERATIONS.items()}
    c["operation"] = {"files": op, "files_sha256": canonical(op)}
    c["architecture"] = {"question": "CNN by GRU path-removal factorial",
        "active_parameters": dict(zip(ARMS, (89859, 90435, 114591, 115167))),
        "registered_parameters": 115167,
        "initialization": "fresh original FULL construction; common tensors and RNG identical",
        "limitations": ["Active parameter capacities differ", "Current encoder retains prefix history",
            "GPU and Runtime differ between datasets", "Three-plus-one concurrency confounds wall time"]}
    c["analysis"] = {"selection": "first strict finite minimum full Validation raw quantity RMSE",
        "metrics": ["whole_qty_rmse", "whole_qty_mae", "whole_time_nll",
                    "tail_qty_rmse", "tail_qty_mae", "tail_time_nll"],
        "tail": "strictly above original Train upper boundary",
        "reporting": "all seeds, paired contrasts, mean and sample SD; retain adverse results",
        "interaction": "M11-M10-M01+M00; metric-scale dependent, not mechanistic synergy",
        "joint_path_support": "FULL strictly improves whole raw quantity RMSE over NONE, CNN-only and GRU-only for every seed in at least two datasets",
        "stronger_claims": "No statistical significance, novelty, universal necessity or independent generalization proof"}
    c["launch_policy"] = {"prepared_only": True, "fresh_user_start_instruction_required": True,
        "all_hosts_qualification_before_any_fit": True, "automatic_reassignment": False,
        "continuous_remote_polling": False}
    c["parent"] = {"contract_sha256": PARENT_CONTRACT, "scientific_source_sha256": PARENT_SOURCE,
        "completed_results_reused_as_matched_fits": False}
    c["jobs"] = [{"id": f"{d}__{s}__{arm}", "host": h, "dataset": d, "seed": s, "arm": arm}
        for h, groups in ASSIGNMENTS.items() for d, s in groups for arm in ARMS]
    require(len(c["jobs"]) == 36 and len({j["id"] for j in c["jobs"]}) == 36, "36 unique jobs")
    return c


def prepare(destination):
    parent = read(PARENT / "execution_contract.json")
    require(canonical(parent) == PARENT_CONTRACT, "Original contract changed")
    require(len(parent["source"]["files"]) == 129
        and canonical(parent["source"]["files"]) == PARENT_SOURCE, "Original source seal changed")
    for name, digest in {**parent["source"]["files"], **parent["input_files"]}.items():
        require(sha(safe(PARENT / "source", name)) == digest, "Original file mismatch: " + name)
    require(not destination.exists(), "Existing bundle preserved; no automatic rebuild")
    c = contract(parent)
    local_files = {name: sha(PROJECT / name) for name in LOCAL_TOOLS}
    destination.mkdir(parents=True, exist_ok=False)
    for name, digest in {**c["source"]["files"], **c["input_files"]}.items():
        origin = PROJECT / name if name in EXTENSIONS else safe(PARENT / "source", name)
        target = safe(destination / "source", name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
        require(sha(target) == digest, "Copied source/input changed: " + name)
    for name, origin in OPERATIONS.items():
        target = safe(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / origin, target)
        require(sha(target) == c["operation"]["files"][name], "Copied operation changed")
    write_json(destination / "execution_contract.json", c)
    write_json(destination / "current.json", {"schema": SCHEMA, "contract_sha256": canonical(c),
        "bundle": str(destination.resolve()), "status": "prepared_not_launched"})
    write_json(destination / "launch_hold.json", {"contract_sha256": canonical(c), "hold": True,
        "reason": "User requires preparation only while 5090 has another workload",
        "release": "A new explicit user start instruction and current native checks are required"})
    members = ["execution_contract.json"] + list(c["operation"]["files"]) + [
        "source/" + p for p in {**c["source"]["files"], **c["input_files"]}]
    with tarfile.open(destination / "deployment.tar.gz", "w:gz") as archive:
        for name in members:
            archive.add(safe(destination, name), arcname=name, recursive=False)
    write_json(destination / "local_tools_integrity.json", {"files": local_files,
        "files_sha256": canonical(local_files)})
    write_json(destination / "preparation_receipt.json", {"status": "prepared_not_launched",
        "contract_sha256": canonical(c), "contract_raw_sha256": sha(destination / "execution_contract.json"),
        "source_files": 131, "input_files": 7, "operation_files": len(c["operation"]["files"]),
        "payload_sha256": sha(destination / "deployment.tar.gz"),
        "local_tools_integrity_raw_sha256": sha(destination / "local_tools_integrity.json"),
        "remote_calls": 0,
        "native_qualification": "not_run", "fresh_training_fits": 0,
        "launch_approval": "absent; requires new user start instruction"})
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--destination", type=Path, default=PROJECT / "search_artifacts" / SCHEMA)
    a = p.parse_args()
    c = prepare(a.destination)
    print(json.dumps({"status": "prepared_not_launched", "contract_sha256": canonical(c),
        "jobs": len(c["jobs"]), "remote_calls": 0}))


if __name__ == "__main__":
    main()
