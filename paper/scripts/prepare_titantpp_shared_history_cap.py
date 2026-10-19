"""Local preparation only; this module has no deployment or training command.

The running late-path study and its frozen sources remain immutable. A future
training approval needs its own execution contract, admission and native checks.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
NAME = "titantpp_history_mlp_shared_cap_20261019_v1"
REPORT = ROOT / "reports" / NAME
BUNDLE = ROOT / "search_artifacts" / NAME
PARENT = ROOT / "search_artifacts/titantpp_mlp16_late_path18_20261019_v2"
PRESERVED_INPUT_ROOT = ROOT / "search_artifacts/titantpp_mlp16_late_path18_5090_recovery_20261019_v1/origin_archive/originals/source"
PARENT_SHA = "67bfcad872b68886bf1de09879e031e055c70063bafbf4aa0db42b4ae5ca6fc7"
SPEC_SHA = "dac86c71954b5ea35b769e0d56caf036ce53e60225ebdb3796ce097b4f0dc486"
SPEC = REPORT / "SCIENCE_SPEC.json"
CONDITION = {"time_input": "log1p", "loss_mode": "log_mse", "time_alpha": 0.0}
ARMS = ("titantpp_history_mlp_shared15_v1", "titantpp_history_mlp_prefix_capacity_v1")
DATASETS = ("yellow_trip_hourly", "raf_spare_parts", "intermittent_frozen_5000")
AUTHORIZATION = {"real_data_training": False, "real_data_inference": False,
                 "remote_execution": False, "test_evaluation": False}
NEW_SOURCE = ("models/TPPs/CountAwareTitanSharedHistoryMLP.py",
              "paper/scripts/run_titantpp_shared_history_cap.py",
              "paper/scripts/prepare_titantpp_shared_history_cap.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def verify_parent():
    path = PARENT / "execution_contract.json"
    require(sha(path) == PARENT_SHA, "Preserved parent contract changed")
    parent = read(path)
    require(parent["common_condition"] == CONDITION, "Common input/loss condition changed")
    require(len(parent["source"]["files"]) == 963, "Expected preserved source963")
    for name, digest in parent["source"]["files"].items():
        require(sha(ROOT / name) == digest, "Preserved scientific source changed: " + name)
    require(canonical(parent["source"]["files"]) == parent["source"]["files_sha256"],
            "Preserved source inventory changed")
    bindings = {}
    for name, digest in parent["input_files"].items():
        path = PARENT / name
        if not path.exists():
            path = PRESERVED_INPUT_ROOT / name
        require(sha(path) == digest, "Preserved input changed: " + name)
        bindings[name] = {"path": str(path), "sha256": digest}
    require(len(bindings) == 6, "Only three Train/Validation input pairs may be bound")
    refs = deepcopy(parent["baseline_validation_references"])
    require(len(refs) == 9 and {(r["dataset"], r["seed"]) for r in refs}
            == {(d, s) for d in DATASETS for s in (42, 52, 62)}, "Expected stored MLP16 references9")
    for ref in refs:
        require(ref["arm"] == "titantpp_history_mlp_width16"
                and ref["time_input"] == "log1p" and ref["loss_mode"] == "log_mse"
                and ref["time_alpha"] == 0 and ref["held_out_test_evaluated"] is False,
                "Baseline common condition differs")
        for field, digest_field in (("path", "sha256"), ("manifest_path", "manifest_sha256"),
                                    ("completion_receipt_path", "completion_receipt_sha256"),
                                    ("selected_checkpoint", "checkpoint_sha256")):
            require(sha(ref[field]) == ref[digest_field], "Stored reference changed: " + field)
        endpoint = read(ref["path"])
        require(endpoint["status"] == "complete" and endpoint["validation_replay_passed"] is True
                and endpoint["selected_epoch"] == ref["selected_epoch"]
                and endpoint["target_identity"] == ref["target_identity"]
                and endpoint["metrics"] == ref["metrics"]
                and endpoint["checkpoint_sha256"] == ref["checkpoint_sha256"],
                "Stored reference endpoint differs")
        require(all(math.isfinite(ref["metrics"]["overall"][k]) for k in ("qty_rmse", "qty_mae")),
                "Nonfinite baseline")
        require(read(ref["completion_receipt_path"])["status"] == "complete", "Reference collection incomplete")
    return parent, bindings, refs


def validate_contract(contract):
    require(contract.get("schema") == "titantpp_shared_history_cap_preparation_v1"
            and contract.get("scope") == "preparation_only", "Not a preparation contract")
    require(contract.get("execution_authorization") == AUTHORIZATION,
            "Real data/server execution is not authorized by this preparation")
    require(contract.get("common_condition") == CONDITION
            and contract.get("arms") == list(ARMS)
            and contract.get("seeds") == [42, 52, 62], "Shared condition or matrix changed")
    jobs = contract.get("planned_jobs", [])
    expected = {(d, a, s) for d in DATASETS for a in ARMS for s in (42, 52, 62)}
    require(len(jobs) == 18 and {(j["dataset"], j["arm"], j["seed"]) for j in jobs} == expected
            and all(j["loss_mode"] == "log_mse" and j["time_input"] == "log1p"
                    and j["time_alpha"] == 0 for j in jobs), "Planned18 matrix changed")
    require(contract.get("correction_parameter_count") == 9600
            and contract.get("shared_width") == 15
            and contract.get("independent_widths") == [15, 5, 5, 5, 5, 5, 5, 5]
            and contract.get("active_prefix_parameter_counts") == [0] + [960 * (k + 2) for k in range(1, 9)],
            "Prefix parameter matching changed")
    require(contract.get("held_out_test_evaluated") is False
            and contract.get("native_qualification_performed") is False
            and contract.get("real_data_training_started") is False
            and contract.get("current_training_modified") is False, "Preparation boundary changed")
    return contract


def prepare():
    require(sha(SPEC) == SPEC_SHA, "Frozen design changed")
    parent, bindings, references = verify_parent()
    tracked = set(subprocess.check_output(["git", "ls-files", "--", *NEW_SOURCE], cwd=ROOT, text=True).splitlines())
    require(tracked == set(NEW_SOURCE) and not subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD", "--", *NEW_SOURCE], cwd=ROOT, text=True).strip(),
        "Commit new implementation before sealing preparation")
    source = {**parent["source"]["files"], **{name: sha(ROOT / name) for name in NEW_SOURCE}}
    contract = {
        "schema": "titantpp_shared_history_cap_preparation_v1", "scope": "preparation_only",
        "campaign": NAME, "created_at": datetime.now(timezone.utc).isoformat(),
        "science_spec": {"path": str(SPEC), "sha256": SPEC_SHA},
        "parent_contract": {"path": str(PARENT / "execution_contract.json"), "sha256": PARENT_SHA},
        "common_condition": CONDITION, "arms": list(ARMS), "seeds": [42, 52, 62],
        "shared_width": 15, "independent_widths": [15, 5, 5, 5, 5, 5, 5, 5],
        "correction_parameter_count": 9600,
        "active_prefix_parameter_counts": [0] + [960 * (k + 2) for k in range(1, 9)],
        "datasets": deepcopy(parent["datasets"]), "training": deepcopy(parent["training"]),
        "input_files": bindings, "baseline_validation_references": references,
        "source": {"files": source, "files_sha256": canonical(source),
                   "preserved_parent_file_count": 963, "new_source_count": 3,
                   "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()},
        "planned_jobs": [{"id": f"{d}__{a}__seed{s}", "dataset": d, "arm": a, "seed": s,
                          **CONDITION} for d in DATASETS for a in ARMS for s in (42, 52, 62)],
        "execution_authorization": AUTHORIZATION, "native_qualification_performed": False,
        "real_data_training_started": False, "current_training_modified": False,
        "held_out_test_evaluated": False, "stored_baselines_retrained_or_reevaluated": False,
        "future_execution": "Requires separate user training approval and isolated execution/native admission; this file is not an execution permit",
    }
    validate_contract(contract)
    path = BUNDLE / "preparation_contract.json"
    write(path, contract)
    receipt = {"status": "prepared_local_only", "contract_path": str(path),
               "contract_raw_sha256": sha(path), "contract_canonical_sha256": canonical(contract),
               "source_files_sha256": contract["source"]["files_sha256"], "source_file_count": len(source),
               "input_file_count": len(bindings), "baseline_reference_count": len(references),
               "planned_new_fit_count": 18, "actual_real_data_fit_count": 0,
               "native_gpu_checks_performed": False, "held_out_test_evaluated": False}
    write(REPORT / "PREPARED.json", receipt)
    return receipt


def verify():
    receipt = read(REPORT / "PREPARED.json")
    path = Path(receipt["contract_path"])
    require(sha(path) == receipt["contract_raw_sha256"], "Preparation contract bytes changed")
    contract = validate_contract(read(path))
    require(canonical(contract) == receipt["contract_canonical_sha256"], "Canonical contract changed")
    parent, bindings, refs = verify_parent()
    require(contract["input_files"] == bindings and contract["baseline_validation_references"] == refs
            and contract["datasets"] == parent["datasets"] and contract["training"] == parent["training"],
            "Parent inputs/statistics/head/budget/references changed")
    require(contract["source"]["preserved_parent_file_count"] == 963
            and contract["source"]["new_source_count"] == len(NEW_SOURCE)
            and contract["source"]["files"] == {**parent["source"]["files"], **{n: sha(ROOT / n) for n in NEW_SOURCE}},
            "Source closure differs from preserved parent plus3")
    require(sha(SPEC) == contract["science_spec"]["sha256"] == SPEC_SHA, "Science spec changed")
    require(canonical(contract["source"]["files"]) == contract["source"]["files_sha256"]
            == receipt["source_files_sha256"], "Source closure SHA changed")
    return {"status": "verified_local_only", "contract_raw_sha256": sha(path),
            "source_file_count": len(contract["source"]["files"]), "planned_fit_count": 18,
            "actual_real_data_fit_count": 0, "held_out_test_evaluated": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "verify"), required=True)
    args = parser.parse_args(argv)
    print(json.dumps((prepare if args.mode == "prepare" else verify)(), ensure_ascii=False))


if __name__ == "__main__":
    main()
