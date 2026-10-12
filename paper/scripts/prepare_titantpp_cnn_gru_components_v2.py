"""Preserve v1 science and issue independent-host operational v2, locally.

No SSH, native CUDA, fit, approval or start permit is created by preparation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tarfile

from paper.scripts.prepare_titantpp_cnn_gru_components import canonical, sha, read, safe, require, write_json
from paper.scripts.launch_titantpp_cnn_gru_components import check_bundle

PROJECT = Path(__file__).resolve().parents[2]
SCHEMA = "titantpp_cnn_gru_components_20261012_v2"
PARENT = PROJECT / "search_artifacts/titantpp_cnn_gru_components_20261012_v1"
PARENT_SHA = "593501da72c8628a50c5158835e0f877a5423b01bb5911872f3058a0f02e1ed3"
POLICY = {"host_independent_start": True, "all_hosts_qualification_before_any_fit": False,
    "fresh_user_start_instruction_required": True, "automatic_reassignment": False,
    "continuous_remote_polling": False}
OPERATIONS = {
    "operation/campaign.py": PROJECT / "paper/scripts/run_titantpp_cnn_gru_component_campaign_v2.py",
    "operation/pipeline.py": PROJECT / "paper/scripts/run_titantpp_cnn_gru_component_phase_v2.py",
    "operation/frozen_runtime.py": PARENT / "operation/frozen_runtime.py",
    "operation/qualification.py": PARENT / "operation/qualification.py",
}
NEW_TOOLS = ("paper/scripts/prepare_titantpp_cnn_gru_components_v2.py",
    "paper/scripts/run_titantpp_cnn_gru_component_phase_v2.py",
    "paper/scripts/run_titantpp_cnn_gru_component_campaign_v2.py",
    "paper/scripts/launch_titantpp_cnn_gru_components_v2.py",
    "paper/tests/test_titantpp_component_preparation_v2.py",
    "paper/tests/test_titantpp_component_campaign_v2.py",
    "paper/tests/test_titantpp_component_launcher_v2.py")


def contract(parent):
    require(canonical(parent) == PARENT_SHA, "Prepared v1 contract changed")
    c = deepcopy(parent)
    c["schema"] = SCHEMA
    c["launch_policy"] = deepcopy(POLICY)
    c["parent"] = {"contract_sha256": PARENT_SHA,
        "scientific_source_sha256": parent["source"]["files_sha256"],
        "scientific_fields_unchanged": True, "v1_launched_fits": 0,
        "reason": "User starts 5080 first; 5090 requires later separate instruction"}
    for host, spec in c["hosts"].items():
        root = str(Path(spec["root"]).parent / (SCHEMA + "_" + host))
        spec.update(root=root, source_root=root + "/source", operation_root=root + "/operation",
            output_dir=root + "/run", tmux=SCHEMA + "_" + host)
        spec["environment"].update(PYTHONPATH=root + "/source", MPLCONFIGDIR=root + "/cache/matplotlib",
            XDG_CACHE_HOME=root + "/cache")
    op = {name: sha(origin) for name, origin in OPERATIONS.items()}
    c["operation"] = {"files": op, "files_sha256": canonical(op)}
    for field in ("source", "input_files", "datasets", "dataset_sha256", "training", "diagnostic",
                  "architecture", "analysis", "parallelism", "jobs", "assignments", "arms", "seeds", "policy"):
        require(c[field] == parent[field], "Scientific or assignment drift: " + field)
    return c


def prepare(destination):
    parent, proof = check_bundle(PARENT)
    require(proof["contract_sha256"] == PARENT_SHA, "Parent pointer drift")
    require(not (PARENT / "control").exists()
        and not any((PARENT / name).exists() for name in ("launch_authorization.json", "launch_release.json")),
        "v1 execution may exist; no duplicated IDs")
    require(not destination.exists(), "Existing v2 bundle preserved")
    c = contract(parent)
    tools = deepcopy(read(PARENT / "local_tools_integrity.json")["files"])
    tools.update({name: sha(PROJECT / name) for name in NEW_TOOLS})
    destination.mkdir(parents=True, exist_ok=False)
    for name, digest in {**c["source"]["files"], **c["input_files"]}.items():
        target = safe(destination / "source", name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe(PARENT / "source", name), target)
        require(sha(target) == digest, "Scientific/input copy changed")
    for name, origin in OPERATIONS.items():
        target = safe(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
        require(sha(target) == c["operation"]["files"][name], "Operational copy changed")
    write_json(destination / "execution_contract.json", c)
    write_json(destination / "current.json", {"schema": SCHEMA, "contract_sha256": canonical(c),
        "bundle": str(destination.resolve()), "status": "prepared_independent_hosts"})
    write_json(destination / "launch_hold.json", {"hold": True, "contract_sha256": canonical(c),
        "reason": "Each host requires its own explicit user instruction and immutable release",
        "unreleased_host_actions": "SSH0 deployment0 qualification0 fit0"})
    members = ["execution_contract.json"] + list(op for op in c["operation"]["files"]) + [
        "source/" + name for name in {**c["source"]["files"], **c["input_files"]}]
    with tarfile.open(destination / "deployment.tar.gz", "w:gz") as archive:
        for name in members:
            archive.add(safe(destination, name), arcname=name, recursive=False)
    write_json(destination / "local_tools_integrity.json", {"files": tools, "files_sha256": canonical(tools)})
    write_json(destination / "preparation_receipt.json", {"status": "prepared_independent_hosts",
        "contract_sha256": canonical(c), "contract_raw_sha256": sha(destination / "execution_contract.json"),
        "payload_sha256": sha(destination / "deployment.tar.gz"),
        "local_tools_integrity_raw_sha256": sha(destination / "local_tools_integrity.json"),
        "parent_contract_sha256": PARENT_SHA, "source_files": 131, "input_files": 7,
        "operation_files": 4, "remote_calls": 0, "native_qualification": "not_run", "fresh_training_fits": 0})
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--destination", type=Path, default=PROJECT / "search_artifacts" / SCHEMA)
    a = p.parse_args()
    c = prepare(a.destination)
    print(json.dumps({"status": "prepared_independent_hosts", "contract_sha256": canonical(c),
        "jobs": len(c["jobs"]), "remote_calls": 0}))


if __name__ == "__main__":
    main()
