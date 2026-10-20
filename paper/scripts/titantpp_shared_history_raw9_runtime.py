"""Original-data shared-history9 queue, after immutable Cap18 completion.

The preserved bounded queue owns only this new campaign's child groups. No
previous source, checkpoint, stopped process or running campaign is modified.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import traceback

from paper.scripts.titantpp_cross_fusion_precheck_runtime import require, sha_file

_PATH = Path(__file__).with_name("titantpp_shared_history_cap_execution_runtime.py")
_SHA = "0a3cd1f203d4f41b9b483dbfc2b1b94623e69543fb21ca26fd7daceac98918ae"
require(sha_file(_PATH) == _SHA, "Preserved Cap18 runtime changed")
_spec = importlib.util.spec_from_file_location(__name__ + "_private_cap", _PATH)
_adapter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_adapter)
_impl = _adapter._impl
_impl.SCHEMA = "titantpp_shared_history_raw9_v1"
_impl.MODULE = "paper.scripts.titantpp_shared_history_raw9_runtime"
_impl.WORKER_MODULE = "paper.scripts.run_titantpp_shared_history_raw9"
_impl.ARMS = ("titantpp_history_mlp_shared15_v1",)
_CAP_RAW_SHA = "a9278125b8a8e01e56298d13ee74c0c628f779020c26950923998e04f05d5226"
_CAP_CANONICAL_SHA = "041a8e7583f2d4b9177473e50373dde53518a2c8ee998cd710ce8dcc66700713"
_CAP_ROOT = "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_mlp_shared_cap_20261019_v1_execution_v3_5080"
_HOST_DEADLINE = 1792888885.911408
_AUTHORIZATION = {"real_data_training": True, "real_data_inference": True, "remote_execution": True, "test_evaluation": False}
_CONDITION = {"time_input": "log1p", "loss_mode": "log_mse", "time_alpha": 1.0}
_PREFIX_COUNTS = [0, 2880, 3840, 4800, 5760, 6720, 7680, 8640, 9600]


def alpha_label(alpha):
    require(type(alpha) in (int, float) and alpha == 1., "Only approved joint alpha1 is admitted")
    return "1"


def validate_contract(contract, host):
    r = _impl
    require(contract.get("schema") == r.SCHEMA and contract.get("scope") == "execution", "Wrong raw9 execution contract")
    require(contract.get("execution_authorization") == _AUTHORIZATION, "Raw9 execution authorization differs")
    require(host == "5080" and set(contract["hosts"]) == {"5080"}, "Only approved 5080 is assigned")
    require(contract.get("common_condition") == _CONDITION, "Joint alpha1/log input/log loss changed")
    require(len(contract["datasets"]) == 3 and {d["dataset_id"] for d in contract["datasets"]} == set(r.DATASETS), "Original dataset3 matrix changed")
    jobs = contract["jobs"]
    require(len(jobs) == len({j["id"] for j in jobs}) == 9
        and {(j["dataset"], j["arm"], j["seed"]) for j in jobs}
        == {(d, r.ARMS[0], s) for d in r.DATASETS for s in (42, 52, 62)}, "Expected9 unique shared-only fits")
    for job in jobs:
        alpha_label(job["time_alpha"])
        require(type(job["seed"]) is int and job["host"] == host and job["loss_mode"] == "log_mse"
            and job["time_input"] == "log1p" and job["id"] == f"{job['dataset']}__{job['arm']}__seed{job['seed']}__losslog_mse", "Job identity/condition differs")
    require(contract["training"] == {"maximum_epochs": 300, "minimum_epochs": 40, "patience": 40,
        "batch_size": 128, "monitor": "validation_raw_quantity_rmse"}, "Original budget/selector changed")
    require(contract["limits"] == {"fit_wall_seconds": 129600, "queue_wall_seconds": 604800, "evaluation_wall_seconds": 7200}, "Fixed limits changed")
    require(contract["evaluation_scope"] == "validation_only" and contract["held_out_test_authorized"] is False, "Test is not authorized")
    require(contract.get("data_variant") == "original_uncapped" and contract.get("quantity_capping_applied") is False, "Only upper-cap-free original data are admitted")
    require(all("cap_value" not in d and d["model"]["hidden_dim"] == 64 and d["loader"]["batch_size"] == 128
        and d["optimizer"] == {"name": "AdamW", "lr": .001, "weight_decay": .01, "grad_clip": 1.}
        for d in contract["datasets"]), "Original model/loader/optimizer differs")
    source = contract["source"]
    require(source["files"] and r.sha_json(source["files"]) == source["files_sha256"], "Source closure differs")
    expected_inputs = {f"inputs/data/{dataset}/{name}" for dataset in r.DATASETS for name in ("train_validation.parquet", "split_manifest.json")}
    require(set(contract["input_files"]) == expected_inputs and all(isinstance(name, str) and not Path(name).is_absolute()
        and ".." not in Path(name).parts and isinstance(digest, str) and len(digest) == 64
        for name, digest in contract["input_files"].items()), "Original input6 binding invalid")
    architecture = contract["architecture"]
    expected = {"arms": list(r.ARMS), "hidden_dim": 64, "correction_input_dim": 128, "branch_count": 8,
        "branch_thresholds": [1, 2, 4, 8, 16, 32, 64, 128], "correction_trainable_parameter_count": 9600,
        "residual_divisor": 8, "bias": False, "activation": "exact GELU", "encoder_blocks": 2,
        "time_alpha": 1.0, "quantity_loss_mode": "log_mse", "quantity_loss_modes": ["log_mse"],
        "quantity_loss_normalization": "none", "model_architecture_changed": True, "checkpoint_warmstart": False,
        "time_input_transform": "log1p_nonnegative_float32", "quantity_input_transform": "log1p_nonnegative_float32"}
    require(all(architecture.get(k) == value for k, value in expected.items()), "Shared-only architecture/head/input changed")
    require(architecture["shared"]["arm"] == r.ARMS[0] and architecture["shared"]["input_width"] == 15
        and architecture["shared"]["input_transform_count"] == 1 and architecture["shared"]["output_widths"] == [15] * 8,
        "Shared basis design changed")
    refs = contract["baseline_validation_references"]
    require(len(refs) == 9 and {(ref["dataset"], ref["seed"]) for ref in refs}
        == {(d, s) for d in r.DATASETS for s in (42, 52, 62)}, "Stored original MLP16 reference9 differs")
    require(all(ref["time_alpha"] == 1. and ref["time_input"] == "log1p" and ref["loss_mode"] == "log_mse"
        and ref["time_head_trained"] is True and ref["retrained"] is False and ref["reevaluated"] is False
        and ref["sha256"] for ref in refs), "Stored original joint MLP16 conditions differ")
    h = contract["hosts"][host]
    require(all(Path(h[k]).is_absolute() for k in ("root", "source_root", "python"))
        and Path(h["source_root"]) == Path(h["root"]) / "source", "Isolated host paths invalid")
    deadline = h["absolute_host_deadline_unix"]
    require(type(deadline) in (int, float) and math.isfinite(deadline) and deadline == _HOST_DEADLINE, "Original host deadline changed")
    stopped = h["paused_processes"]
    require(len(stopped) == len({p["pid"] for p in stopped}) == 4 and all(p["state"] in ("T", "t") for p in stopped), "Stopped process4 identities missing")
    predecessor = contract["predecessor"]
    require(predecessor["contract_raw_sha256"] == _CAP_RAW_SHA and predecessor["expected_count"] == 18
        and Path(predecessor["contract_path"]).is_absolute() and Path(predecessor["receipt_path"]).is_absolute(), "Cap18 predecessor binding differs")
    return jobs


def verify_time_head_training(proof):
    require(isinstance(proof, dict) and proof.get("schema") == "time_head_joint_training_verification_v1", "Actual joint time-head proof missing")
    keys = ("initial_time_head_sha256", "final_time_head_sha256", "checkpoint_time_head_sha256")
    require(all(isinstance(proof.get(k), str) and len(proof[k]) == 64 and all(c in "0123456789abcdef" for c in proof[k]) for k in keys)
        and proof[keys[0]] != proof[keys[1]] and proof[keys[0]] != proof[keys[2]], "Joint time-head tensors did not change")
    require(all(proof.get(k) is True for k in ("finite_time_head_gradient_observed", "nonzero_time_head_gradient_observed", "time_head_parameters_changed",
        "optimizer_state_present", "joint_loss_includes_time")) and type(proof.get("optimizer_steps_checked")) is int
        and proof["optimizer_steps_checked"] > 0 and type(proof.get("joint_loss_observations")) is int
        and proof["joint_loss_observations"] > 0 and proof.get("time_alpha") == 1.
        and isinstance(proof.get("time_head_parameter_names"), list) and proof["time_head_parameter_names"], "Joint time-head training evidence incomplete")
    return proof


def verify_receipt(contract, host, job, stage, *, originals=None):
    """Validate remote receipts or their unchanged collected originals."""
    r = _impl
    require(stage in r.STAGES, "Test stage forbidden")
    remote_root = Path(contract["hosts"][host]["root"])
    root = remote_root if originals is None else Path(originals)
    path = root / "run" / job["id"] / f"{stage}_receipt.json"
    receipt = r.read(path)
    require(receipt.get("schema") == "titantpp_shared_history_raw9_stage_v1" and receipt.get("status") == "complete"
        and receipt.get("job_id") == job["id"] and receipt.get("stage") == stage
        and receipt.get("contract_sha256") == r.sha_json(contract)
        and all(receipt.get(k) == job[k] for k in ("dataset", "arm", "seed", "time_alpha", "loss_mode"))
        and receipt.get("held_out_test_evaluated") is False, "Raw9 worker identity/completion differs")
    require(receipt.get("quantity_only") is False and receipt.get("time_head_trained") is True
        and receipt.get("quantity_loss_normalization") == "none", "Joint alpha1 training identity differs")
    verify_time_head_training(receipt.get("time_head_training_verification"))
    require(receipt.get("candidate_name") == job["arm"] and receipt.get("history_relation_sharing") == "shared"
        and receipt.get("history_relation_parameter_count") == 9600
        and receipt.get("architecture_metadata", {}).get("active_prefix_parameter_counts") == _PREFIX_COUNTS
        and receipt.get("time_input") == "log1p" and receipt.get("time_input_transform")
        == receipt.get("quantity_input_transform") == "log1p_nonnegative_float32", "Shared original-data candidate/input proof missing")
    files = receipt.get("files")
    require(isinstance(files, dict) and files, "Empty raw9 artifact inventory")
    job_root = (root / "run" / job["id"]).resolve()
    for name, digest in files.items():
        artifact = r.local_path(root, name)
        require(artifact.is_relative_to(job_root) and artifact != path.resolve(), "Raw9 artifact escapes assigned job")
        require(r.sha_file(artifact) == digest, "Raw9 artifact SHA differs: " + name)
    selected = Path(receipt["selected_checkpoint"])
    require(selected.is_absolute() and selected.is_relative_to(remote_root / "run" / job["id"]), "Selected checkpoint escapes assigned remote job")
    relative = str(selected.relative_to(remote_root))
    require(files.get(relative) == receipt["checkpoint_sha256"] == r.sha_file(r.local_path(root, relative)), "Selected binary binding differs")
    require(type(receipt.get("selected_epoch")) is int and 0 < receipt["selected_epoch"] <= 300
        and isinstance(receipt.get("state_sha256"), str) and len(receipt["state_sha256"]) == 64, "Missing selected epoch/state")
    if stage == "validation":
        train = r.read(path.with_name("train_receipt.json"))
        fields = ("selected_checkpoint", "selected_epoch", "checkpoint_sha256", "state_sha256", "target_identity",
            "loss_mode", "architecture_metadata", "time_head_training_verification")
        require(all(train.get(k) == receipt.get(k) for k in fields) and receipt.get("validation_replay_passed") is True,
            "Selected Validation replay differs")
    return {"path": str(path.relative_to(root)), "sha256": r.sha_file(path)}


def predecessor_gate(contract, host, *, require_complete=False):
    """Require every Cap18 worker and selected Validation before new GPU work."""
    from paper.scripts import titantpp_shared_history_cap_execution_v3_runtime as parent
    r = _impl
    validate_contract(contract, host)
    root = Path(_CAP_ROOT)
    cp = root / "execution_contract.json"
    require(r.sha_file(cp) == _CAP_RAW_SHA, "Cap18 predecessor contract bytes differ")
    old = r.read(cp)
    require(r.sha_json(old) == _CAP_CANONICAL_SHA, "Cap18 canonical contract differs")
    jobs = parent.validate_contract(old, host)
    ids = {job["id"] for job in jobs}
    require(len(ids) == 18, "Cap18 job count differs")
    require(all(contract["hosts"][host][k] == old["hosts"][host][k] for k in
        ("python", "gpu_uuid", "runtime_expected", "paused_processes", "absolute_host_deadline_unix")), "Host/Runtime/stopped campaign inheritance differs")
    require(not (root / "dispatch_failure.json").exists(), "Cap18 failure remains unresolved")
    final, exit_record = r.read(root / "final_receipt.json"), r.read(root / "dispatcher_exit.json")
    require(final["status"] == "complete" and final["host"] == host and final["contract_sha256"] == r.sha_json(old)
        and final["completed_fit_count"] == final["validation_replay_count"] == 18 and final["test_count"] == 0
        and final["held_out_test_evaluated"] is False and final["owned_worker_pids_absent"] is True, "Cap18 is not fully complete")
    require(exit_record["actual_os_returncode"] == 0, "Cap18 dispatcher actual exit0 missing")
    pids = [exit_record["pid"]]
    require(set(final["stages"]) == set(parent.STAGES), "Cap18 stage inventory differs")
    for stage in parent.STAGES:
        result = final["stages"][stage]
        require(result["status"] == "complete" and set(result["exits"]) == set(result["receipts"]) == ids, "Cap18 terminal stage incomplete")
        for job in jobs:
            child, binding = result["exits"][job["id"]], result["receipts"][job["id"]]
            require(child["actual_os_returncode"] == 0 and child["timeout"] is False, "A Cap18 scientific worker failed")
            pids.append(child["pid"])
            path = r.local_path(root, binding["path"])
            require(r.sha_file(path) == binding["sha256"], "Cap18 selected stage receipt changed")
            parent.verify_receipt(old, host, job, stage)
    manifest = final["artifact_manifest"]
    mp = r.local_path(root, manifest["path"])
    require(r.sha_file(mp) == manifest["sha256"], "Cap18 artifact manifest changed")
    inventory = r.read(mp)["files"]
    require(inventory and len(inventory) == manifest["file_count"], "Cap18 original inventory empty/incomplete")
    for name, digest in inventory.items():
        require(r.sha_file(r.local_path(root, name)) == digest, "Cap18 original artifact changed")
    require(all(r.proc_identity(pid) is None for pid in pids), "Cap18 owned PID remains")
    for pid in parent.predecessor_process_candidates(root):
        identity = r.proc_identity(pid)
        require(identity is None or not identity.get("argv") or "python" not in Path(identity["argv"][0]).name,
            "An unrecorded Cap18 Python owner remains")
    gpu, paused = r.gpu_state(contract, host)
    require(not set(pids).intersection(gpu["compute_pids"]), "Cap18 CUDA worker remains")
    return {"status": "complete", "parent_root": str(root), "parent_final_receipt_sha256": r.sha_file(root / "final_receipt.json"),
        "parent_dispatcher_exit_sha256": r.sha_file(root / "dispatcher_exit.json"), "parent_manifest_sha256": r.sha_file(mp),
        "parent_completed_fit_count": 18, "parent_validation_replay_count": 18,
        "parent_owned_pids_absent": True, "paused_processes": paused, "gpu": gpu}


_original_three = _adapter.qualify_three_concurrent


def qualify_three_concurrent(contract, host):
    result = _original_three(contract, host)
    require(set(result["per_arm"]) == set(_impl.ARMS) and len(result["per_arm"][_impl.ARMS[0]]) == 3,
        "Raw9 single-arm native admission differs")
    result["native_worker_exit0_count"] = 3
    return result


# The preserved qualification invokes the rebound validator and the one-arm
# synthetic worker, retaining its exact current-owner CUDA-context allowance.
_adapter.validate_contract = validate_contract
_impl.alpha_label = alpha_label
_impl.validate_contract = validate_contract
_impl.verify_receipt = verify_receipt
_impl.predecessor_gate = predecessor_gate
_impl.qualify_three_concurrent = qualify_three_concurrent
_impl.qualify = _adapter.qualify
for _name, _value in tuple(_impl.__dict__.items()):
    if not _name.startswith("_"):
        globals()[_name] = _value

if __name__ == "__main__":
    try:
        raise SystemExit(_impl.main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
