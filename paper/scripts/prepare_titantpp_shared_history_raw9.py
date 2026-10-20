"""Prepare only the authorized uncapped shared-history nine-fit successor.

Frozen Cap sources, original MLP16 endpoints and running jobs are never edited.
Preparation may finish while Cap18 is running; deployment requires its verified
final collection and a separate fresh native admission.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
NAME = "titantpp_history_mlp_shared_raw9_20261020_v1"
REPORT = ROOT / "reports" / NAME
BUNDLE = ROOT / "search_artifacts" / NAME
PARENT = ROOT / "search_artifacts/titantpp_history_mlp_shared_cap_20261019_v1/execution_v3"
PARENT_REPORT = ROOT / "reports/titantpp_history_mlp_shared_cap_20261019_v1/execution_v3"
PARENT_SHA = "a9278125b8a8e01e56298d13ee74c0c628f779020c26950923998e04f05d5226"
RAW_CONTRACT = ROOT / "search_artifacts/titantpp_history_width16_dual_20261003_v1/execution_contract.json"
BASELINES = REPORT / "BASELINE_REFERENCES.json"
SPEC = REPORT / "SCIENCE_SPEC.json"
ARM = "titantpp_history_mlp_shared15_v1"
ARMS = (ARM,)
DATASETS = ("raf_spare_parts", "yellow_trip_hourly", "intermittent_frozen_5000")
CONDITION = {"time_input": "log1p", "loss_mode": "log_mse", "time_alpha": 1.0}
NEW_SOURCE = (
    "paper/scripts/prepare_titantpp_shared_history_raw9.py",
    "paper/scripts/manage_titantpp_shared_history_raw9.py",
    "paper/scripts/titantpp_shared_history_raw9_runtime.py",
    "paper/scripts/run_titantpp_shared_history_raw9.py",
    "paper/scripts/compare_titantpp_shared_history_raw9.py",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def baseline_document():
    document = read(BASELINES)
    refs = deepcopy(document["references"])
    require(len(refs) == 9 and {(r["dataset"], r["seed"]) for r in refs}
            == {(d, s) for d in DATASETS for s in (42, 52, 62)}, "Original baseline9 matrix differs")
    for ref in refs:
        require(ref.get("time_alpha") == 1 and ref.get("loss_mode") == "log_mse"
                and ref.get("time_input") == "log1p", "Original baseline objective differs")
        for path, digest in ref["source_bindings"].items():
            require(sha(path) == digest, "Original baseline evidence changed: " + path)
        ref["time_head_trained"] = True
        ref["time_head_training_evidence"] = "frozen_original_joint_objective_alpha1_contract_and_training_history"
    return document, refs


def _raw_inputs(data):
    """Materialize only Train/Validation rows; never decode held-out rows."""
    import polars as pl
    dataset = data["dataset_id"]
    identity = data["inherited_data_identity"]
    if dataset == "raf_spare_parts":
        original = ROOT / "search_artifacts/titantpp_history_width16_dual_20261003_v1/data/raf_spare_parts"
        paths = {"data": original / "train_validation.parquet", "split_manifest": original / "split_manifest.json"}
    else:
        paths = {k: ROOT / "sample_data" / Path(v["path"]).relative_to(
            "/home/leekwanhyeong/workspace/paper_research/sample_data") for k, v in identity.items()
                 if k in ("data", "split_manifest")}
    result, lineage = {}, {}
    for kind, path in paths.items():
        expected = identity[kind]["sha256"]
        require(sha(path) == expected, "Original uncapped input changed: " + str(path))
        relative = f"inputs/data/{dataset}/" + ("train_validation.parquet" if kind == "data" else "split_manifest.json")
        destination = BUNDLE / "input_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        require(not destination.exists(), "Preserve original snapshot")
        if kind == "data" and dataset != "raf_spare_parts":
            frame = pl.scan_parquet(path).filter(pl.col("chronological_split").is_in(["train", "validation"])).collect()
            require(set(frame["chronological_split"].unique()) == {"train", "validation"}, "Unexpected admitted split")
            frame.write_parquet(destination)
        else:
            shutil.copyfile(path, destination)
        result[kind] = {"path": relative, "sha256": sha(destination)}
        lineage[str(path)] = expected
    return result, lineage


def prepare():
    require(not BUNDLE.exists(), "Existing raw9 bundle; no overwrite/retry")
    require(sha(PARENT / "execution_contract.json") == PARENT_SHA, "Frozen Cap contract changed")
    parent = read(PARENT / "execution_contract.json")
    tracked = set(subprocess.check_output(["git", "ls-files", "--", *NEW_SOURCE], cwd=ROOT, text=True).splitlines())
    require(tracked == set(NEW_SOURCE) and not subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD", "--", *NEW_SOURCE], cwd=ROOT, text=True).strip(),
        "Commit the new isolated operation files before sealing")
    document, refs = baseline_document()
    raw = read(RAW_CONTRACT)
    source = {**parent["source"]["files"], **{n: sha(ROOT / n) for n in NEW_SOURCE}}
    for name, digest in source.items():
        require(sha(ROOT / name) == digest, "Preserved source changed: " + name)
    BUNDLE.mkdir(parents=True)
    datasets, input_files, original_inputs = [], {}, {}
    for dataset in DATASETS:
        data = deepcopy(next(d for d in raw["datasets"] if d["dataset_id"] == dataset))
        admitted, lineage = _raw_inputs(data)
        data["inherited_data_identity"] = {**admitted,
            "populations": {k: v for k, v in data["inherited_data_identity"]["populations"].items()
                            if k in ("train", "validation")}}
        data.pop("parent_data_identity", None)
        data.update(data_variant="original_uncapped", quantity_capping_applied=False,
            prepared_train_validation_path=admitted["data"]["path"], data_sha256=admitted["data"]["sha256"],
            mu_all_train_rows=data["statistics"]["train_log_mean"])
        for binding in admitted.values():
            input_files[binding["path"]] = binding["sha256"]
        original_inputs.update(lineage)
        datasets.append(data)
    h = deepcopy(parent["hosts"]["5080"])
    predecessor = {"contract_path": str(PARENT / "execution_contract.json"),
        "contract_raw_sha256": PARENT_SHA, "contract_canonical_sha256": canonical(parent),
        "source_files_sha256": parent["source"]["files_sha256"],
        "remote_root": h["root"], "source_root": h["source_root"],
        "receipt_path": str(PARENT / "retrieval/final_receipt.json"),
        "finisher_receipt_path": str(PARENT_REPORT / "FINISHER_STARTED.json"),
        "failure_path": str(PARENT_REPORT / "FINISH_FAILURE.json"), "expected_count": 18}
    remote = f"/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/{NAME}_5080"
    h.update(root=remote, source_root=remote + "/source", tmux="shared_history_raw9_5080_20261020_v1",
        completed_predecessor={"root": predecessor["remote_root"],
            "contract_raw_sha256": PARENT_SHA, "contract_canonical_sha256": canonical(parent),
            "source_files_sha256": parent["source"]["files_sha256"]})
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    h["environment"].update(SOURCE_REVISION=revision, PYTHONPATH=h["source_root"],
        MPLCONFIGDIR=remote + "/cache/matplotlib", XDG_CACHE_HOME=remote + "/cache")
    architecture = deepcopy(parent["architecture"])
    architecture.pop("independent", None)
    architecture.update(arms=[ARM], time_alpha=1.0, quantity_loss_mode="log_mse",
        data_variant="original_uncapped", quantity_capping_applied=False)
    contract = {"schema": "titantpp_shared_history_raw9_v1", "scope": "execution", "campaign": NAME,
        "created_at": datetime.now(timezone.utc).isoformat(), "data_variant": "original_uncapped",
        "quantity_capping_applied": False, "common_condition": CONDITION,
        "architecture": architecture, "design_sha256": canonical(architecture),
        "science_spec_sha256": sha(SPEC), "datasets": datasets, "training": deepcopy(parent["training"]),
        "hosts": {"5080": h}, "source": {"files": source, "files_sha256": canonical(source),
            "base_git_revision": revision, "preserved_parent_file_count": len(parent["source"]["files"])},
        "jobs": [{"id": f"{d}__{ARM}__seed{s}__losslog_mse", "dataset": d, "arm": ARM,
                  "seed": s, "host": "5080", **CONDITION} for d in DATASETS for s in (42, 52, 62)],
        "input_files": input_files, "original_input_bindings": original_inputs,
        "baseline_validation_references": refs, "baseline_document_sha256": sha(BASELINES),
        "raw_parent_contract": {"path": str(RAW_CONTRACT), "sha256": sha(RAW_CONTRACT)},
        "predecessor": predecessor, "limits": deepcopy(parent["limits"]), "maximum_parallel_workers": 3,
        "execution_authorization": {"real_data_training": True, "real_data_inference": True,
            "remote_execution": True, "test_evaluation": False}, "evaluation_scope": "validation_only",
        "held_out_test_authorized": False, "recurring_automation_created": False,
        "historical_baselines_retrained_or_reevaluated": False,
        "approval": "2026-10-20 user approved shared MLP original data 3seed after current Intermittent; explicitly selected time weight1 to match original MLP16/external baselines; use5080 maximum3 after verified completion"}
    from paper.scripts import titantpp_shared_history_raw9_runtime as runtime
    runtime.validate_contract(contract, "5080")
    from paper.scripts import run_multilag_detail_execution as data_checks
    checks = []
    for data in datasets:
        local = deepcopy(data)
        for kind in ("data", "split_manifest"):
            binding = local["inherited_data_identity"][kind]
            binding["path"] = str(BUNDLE / "input_snapshot" / binding["path"])
        frame, checked = data_checks.prepare_admitted_data(local)
        data_checks.time_interface(local, frame, contract)
        checks.append({"dataset": data["dataset_id"], "status": "passed",
            "populations": checked["populations"], "frozen_quantity_and_time_statistics_reproduced": True,
            "held_out_materialized": checked["held_out_materialized"], "model_inference_calls": 0})
    write(REPORT / "LOCAL_DATA_CHECKS.json", {"status": "passed", "datasets": checks,
        "actual_training_calls": 0, "held_out_materialized": False})
    contract["local_data_checks_sha256"] = sha(REPORT / "LOCAL_DATA_CHECKS.json")
    write(BUNDLE / "execution_contract.json", contract)
    with (BUNDLE / "deployment.tar.gz").open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as archive:
        archive.add(BUNDLE / "execution_contract.json", arcname="execution_contract.json")
        for name in source:
            archive.add(ROOT / name, arcname="source/" + name)
        for name in input_files:
            archive.add(BUNDLE / "input_snapshot" / name, arcname="source/" + name)
        marker = tarfile.TarInfo("source/sample_data")
        marker.type, marker.mode = tarfile.DIRTYPE, 0o755
        archive.addfile(marker)
    receipt = {"status": "prepared_waiting_for_predecessor", "contract_sha256": sha(BUNDLE / "execution_contract.json"),
        "canonical_contract_sha256": canonical(contract), "archive_sha256": sha(BUNDLE / "deployment.tar.gz"),
        "new_fit_count": 9, "baseline_reference_count": 9, "source_file_count": len(source),
        "host": "5080", "max_parallel": 3, "held_out_test_evaluated": False}
    write(REPORT / "PREPARED.json", receipt)
    return receipt


def verify_predecessor_complete():
    """Recheck terminal evidence and original archive locally, without SSH."""
    from paper.scripts import manage_titantpp_shared_history_cap_execution_v3 as old
    contract = old.verify_local()
    require(sha(old.BUNDLE / "execution_contract.json") == PARENT_SHA, "Wrong predecessor")
    final = read(old.BUNDLE / "retrieval/final_receipt.json")
    require(final["status"] == "complete" and final["new_fit_count"] == 18
        and final["validation_replay_count"] == 18 and final["baseline_reference_count"] == 9
        and final["actual_scientific_worker_exit0_count"] == 36
        and final["actual_dispatcher_exit0_count"] == 1
        and final["held_out_test_evaluated"] is False, "Cap18 final receipt incomplete")
    retrieval = old.BUNDLE / "retrieval/5080"
    originals = retrieval / "originals"
    old.verify_collected(contract, "5080", originals)
    collection = read(retrieval / "receipt.json")
    inventory = read(retrieval / "manifest.json")["files"]
    require(collection["status"] == "complete" and inventory and len(inventory) == collection["file_count"], "Missing predecessor originals")
    require(sha(retrieval / "original.tar.gz") == collection["archive_sha256"]
            and sha(retrieval / "manifest.json") == collection["manifest_sha256"], "Cap archive changed")
    for name, digest in inventory.items():
        require(sha(old.runtime.local_path(originals, name)) == digest, "Predecessor original changed")
    for field in ("source_bindings", "archive_bindings", "report_files"):
        require(final.get(field), "Empty predecessor binding: " + field)
        for name, digest in final[field].items():
            require(sha(name) == digest, "Predecessor completion changed: " + name)
    return {"status": "complete_verified_local", "final_receipt_sha256": sha(old.BUNDLE / "retrieval/final_receipt.json"),
            "file_count": len(inventory)}


def verify(*, archive=False, require_predecessor_complete=False):
    receipt = read(REPORT / "PREPARED.json")
    path = BUNDLE / "execution_contract.json"
    require(sha(path) == receipt["contract_sha256"], "Prepared raw9 contract changed")
    contract = read(path)
    require(canonical(contract) == receipt["canonical_contract_sha256"], "Raw9 canonical SHA changed")
    require(sha(SPEC) == contract["science_spec_sha256"], "Raw9 science specification changed")
    require(sha(BASELINES) == contract["baseline_document_sha256"], "Baseline document changed")
    require(sha(REPORT / "LOCAL_DATA_CHECKS.json") == contract["local_data_checks_sha256"], "Data preparation check changed")
    baseline_document()
    for name, digest in contract["source"]["files"].items():
        require(sha(ROOT / name) == digest, "Frozen scientific/operation source changed: " + name)
    for name, digest in contract["input_files"].items():
        require(sha(BUNDLE / "input_snapshot" / name) == digest, "Raw9 input changed")
    for name, digest in contract["original_input_bindings"].items():
        require(sha(name) == digest, "Original uncapped input changed")
    require(sha(RAW_CONTRACT) == contract["raw_parent_contract"]["sha256"]
        and sha(PARENT / "execution_contract.json") == PARENT_SHA, "Frozen parent contract changed")
    if archive:
        require(sha(BUNDLE / "deployment.tar.gz") == receipt["archive_sha256"], "Raw9 deployment archive changed")
    from paper.scripts import titantpp_shared_history_raw9_runtime as runtime
    runtime.validate_contract(contract, "5080")
    if require_predecessor_complete:
        verify_predecessor_complete()
    return contract


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "verify"), required=True)
    args = parser.parse_args(argv)
    result = prepare() if args.mode == "prepare" else verify(archive=True)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
