"""Continue the three interrupted MAC fits without altering their frozen source.

The native engine restores its own model, AdamW, best state, history and RNG.
Six completed fits are byte-for-byte references.  Scientific receipts remain in
the original format; separate recovery receipts describe the new attempt.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import sys
import time

from paper.scripts import run_titans_mac_accuracy9 as science

SCHEMA = "titans_mac_accuracy9_recovery_stage_v1"
CONTRACT_SCHEMA = "titans_mac_accuracy9_recovery_v1"
SCIENCE_WORKER_SHA256 = "8fda9deb7e4252cd8f8f03a95249878adc6c94fa7e6703e57f9d49e8ca0db4a3"
require, read, sha_file, sha_json = science.require, science.read, science.sha_file, science.sha_json
write = science.write_exclusive


def load_scientific(contract):
    require(contract.get("schema") == CONTRACT_SCHEMA and contract.get("host") == "5090",
            "Wrong MAC recovery scope")
    require(sha_file(science.__file__) == SCIENCE_WORKER_SHA256,
            "Preserved MAC scientific worker changed")
    binding = contract["scientific_contract"]
    require(sha_file(binding["path"]) == binding["raw_sha256"], "Original contract bytes changed")
    original = read(binding["path"])
    require(sha_json(original) == binding["canonical_sha256"], "Original contract identity changed")
    root = Path(contract["root"]).resolve()
    old_root = Path(contract["immutable_originals"]["root"]).resolve()
    require(Path(contract["root"]).is_absolute()
            and Path(contract["source_root"]).resolve() == root / "source"
            and old_root == Path(original["hosts"]["5090"]["root"]).resolve()
            and not root.is_relative_to(old_root) and not old_root.is_relative_to(root),
            "Recovery requires a separate root")
    require(contract["source"]["files"] == original["source"]["files"]
            and contract["source"]["files_sha256"] == original["source"]["files_sha256"],
            "Scientific source inventory changed")
    require(contract["source"].get("source_revision", contract["source"].get("base_git_revision"))
            == original["source"]["base_git_revision"], "Scientific source revision changed")
    if "absolute_host_deadline_unix" in original["hosts"]["5090"]:
        require(contract["absolute_host_deadline_unix"]
                == original["hosts"]["5090"]["absolute_host_deadline_unix"],
                "Original MAC host deadline changed")
    return original


def effective_scientific(contract):
    """Relocate artifacts only; native training identity uses the original contract."""
    original = load_scientific(contract)
    effective = deepcopy(original)
    effective["hosts"]["5090"]["root"] = contract["root"]
    return effective


def select_job(contract, job_id):
    original = load_scientific(contract)
    jobs = contract.get("jobs", [])
    require(len(jobs) == 9 and {j.get("id") for j in jobs} == {j["id"] for j in original["jobs"]},
            "Recovery must preserve exactly the nine original jobs")
    for item in jobs:
        reference, _, _, _ = science.select_job(original, "5090", item["id"])
        require(all(item.get(k) == v for k, v in reference.items()), "Recovery scientific job changed")
        expected_mode = "resume" if item["dataset"] == "intermittent_frozen_5000" else "reference"
        require(item.get("recovery_mode") == expected_mode, "Only the interrupted Intermittent fits may resume")
    expected_train = {j["id"] for j in jobs if j["recovery_mode"] == "resume"}
    require(len(contract.get("training_jobs", [])) == 3 and set(contract["training_jobs"]) == expected_train,
            "Only three interrupted fits may train")
    for phase in ("validation", "test"):
        admitted = contract.get(phase + "_jobs", [])
        require(len(admitted) == 9 and set(admitted) == {j["id"] for j in jobs},
                "The all-nine " + phase + " matrix changed")
    selected = [j for j in jobs if j["id"] == job_id]
    require(len(selected) == 1, "Recovery job must be uniquely bound")
    reference, data, old_root, _ = science.select_job(original, "5090", job_id)
    return selected[0], reference, data, original, Path(contract["root"]), old_root


def verify_origin(contract, binding):
    old_root = Path(contract["immutable_originals"]["root"]).resolve()
    path = Path(binding["path"]).resolve()
    require(path.is_relative_to(old_root), "Original binding escapes its preserved root")
    relative = str(path.relative_to(old_root))
    require(contract["immutable_originals"]["files"].get(relative) == binding["sha256"]
            and sha_file(path) == binding["sha256"], "Frozen original binding changed")
    return path


def copy_origin(contract, binding, destination):
    path = verify_origin(contract, binding)
    destination = science.resolve_under(contract["root"], destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with path.open("rb") as source, destination.open("xb") as target:
        shutil.copyfileobj(source, target)
    require(sha_file(destination) == binding["sha256"], "Original copy SHA changed")
    return {"original_path": str(path), "original_sha256": binding["sha256"],
            "copy_path": str(destination), "copy_sha256": sha_file(destination)}


def quantity_contract(data):
    return {"boundaries": data["quantity_boundaries_all_train_rows"],
            "strata": [{"stratum": f"quantity_{i}", "stratum_order": i,
                        "stratum_label": f"Raw Train bin {i}"} for i in range(5)]}


@contextmanager
def resume_guard(payload, tracked):
    """Validate the engine's exact prefix before restoring cumulative joint proof."""
    from paper.scripts.count_aware_tpp_backbone import training
    original = training._validate_resume_payload
    observed = {"validated": False}

    def validate(value, **kwargs):
        require(value["model_state_sha256"] == payload["model_state_sha256"]
                and value["best_state_sha256"] == payload["best_state_sha256"]
                and value["epoch"] == payload["epoch"], "Resume payload changed before restore")
        result = original(value, **kwargs)
        proof = science.verify_time_head_proof(value.get("time_head_training_verification"),
                                               value["model_state_dict"])
        tracker = tracked["tracker"]
        require(tracker.steps == 0 and tracker.joint_observations == 0
                and tracker.initial_sha == proof["initial_time_head_sha256"],
                "Resume joint tracker initialization differs")
        require(tuple(proof["trainable_parameter_names"]) == tracker.trainable_names,
                "Resume optimizer parameter inventory changed")
        tracker.steps = proof["optimizer_steps_checked"]
        tracker.joint_observations = proof["joint_loss_observations"]
        tracker.finite_gradient = proof["finite_time_head_gradient_observed"]
        tracker.nonzero_gradient = proof["nonzero_time_head_gradient_observed"]
        # The engine restores model/AdamW after this validator.  Calling proof()
        # here would inspect the un-restored optimizer and is deliberately avoided.
        observed.update(validated=True, prefix_optimizer_steps=tracker.steps,
                        prefix_joint_loss_observations=tracker.joint_observations,
                        prefix_current_state_sha256=value["model_state_sha256"],
                        prefix_best_state_sha256=value["best_state_sha256"],
                        prefix_epoch=value["epoch"])
        return result

    training._validate_resume_payload = validate
    try:
        yield observed
    finally:
        training._validate_resume_payload = original


def _outer_receipt(contract, job, original, phase, scientific, lineage):
    root = Path(contract["root"])
    folder = root / "run" / job["id"]
    result = {**scientific, "schema": SCHEMA, "stage": phase,
              "contract_sha256": sha_json(contract),
              "scientific_contract_sha256": sha_json(original),
              "scientific_contract_raw_sha256": contract["scientific_contract"]["raw_sha256"],
              "effective_scientific_contract_sha256": sha_json(effective_scientific(contract)),
              "recovery_mode": job["recovery_mode"], "lineage": lineage,
              "original_files_modified": False, "scientific_worker_source_unchanged": True}
    if phase in ("train", "reference"):
        result["metrics"] = scientific["validation_reference"]["metrics"]
    receipt_path = folder / (phase + "_recovery_receipt.json")
    result["files"] = science.file_inventory(root, [p for p in folder.rglob("*")
                                                   if p.is_file() and p != receipt_path])
    write(receipt_path, result)
    return result


def reference(contract, job_id):
    job, reference_job, _, original, root, old_root = select_job(contract, job_id)
    require(job["recovery_mode"] == "reference", "Interrupted fits cannot be referenced as complete")
    folder = root / "run" / job_id
    write(folder / "reference.intent.json", {"contract_sha256": sha_json(contract),
                                            "job_id": job_id, "stage": "reference"})
    origin_path = verify_origin(contract, job["origin_train_receipt"])
    origin = read(origin_path)
    science.verify_train_receipt(original, "5090", reference_job, old_root, origin)
    lineage = {"reference_retrained": False,
               "train_receipt": copy_origin(contract, job["origin_train_receipt"],
                                             folder / "origin" / "train_receipt.json")}
    for relative, digest in origin["files"].items():
        path = science.resolve_under(old_root, relative)
        require(path.is_relative_to(old_root / "run" / job_id), "Reference file belongs to another fit")
        copy_origin(contract, {"path": str(path), "sha256": digest}, root / relative)
    exit_path = verify_origin(contract, job["origin_train_exit"])
    exited = read(exit_path)
    require(exited.get("job_id") == job_id and exited.get("stage") == "train"
            and type(exited.get("actual_os_returncode")) is int and exited["actual_os_returncode"] == 0
            and exited.get("timeout") is False and type(exited.get("pid")) is int and exited["pid"] > 0
            and exited.get("start_ticks") is not None and exited.get("observed_exit_at") is not None,
            "Completed original fit has no actual child exit0")
    lineage["original_train_exit"] = copy_origin(contract, job["origin_train_exit"], folder / "train_exit.json")
    selected = science.resolve_under(root, Path(origin["selected_checkpoint"]).relative_to(old_root))
    prediction = science.resolve_under(root, Path(origin["validation_reference"]["prediction_path"]).relative_to(old_root))
    scientific = {**origin, **science.base_receipt(effective_scientific(contract), "5090", reference_job, "train"),
                  "selected_checkpoint": str(selected),
                  "summary": {**origin["summary"], "checkpoint_path": str(selected)},
                  "validation_reference": {**origin["validation_reference"], "prediction_path": str(prediction)}}
    science.verify_train_receipt(effective_scientific(contract), "5090", reference_job, root, scientific)
    write(folder / "train_receipt.json", scientific)
    verify_origin(contract, job["origin_train_receipt"])
    return _outer_receipt(contract, job, original, "reference", scientific, lineage)


def train(contract, job_id, *, device="cuda:0"):
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    from paper.scripts.count_aware_tpp_backbone import training
    from paper.scripts import run_multilag_detail_execution as base
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    job, reference_job, data, original, root, _ = select_job(contract, job_id)
    require(job["recovery_mode"] == "resume" and job_id in contract["training_jobs"],
            "Only the interrupted Intermittent fit may train")
    folder = root / "run" / job_id
    write(folder / "train.intent.json", {"contract_sha256": sha_json(contract), "job_id": job_id,
                                        "stage": "train", "started_unix": time.time()})
    run_dir = folder / "runs" / reference_job["arm"] / science.VARIANT / f"seed_{reference_job['seed']}"
    copied = copy_origin(contract, job["origin_resume_checkpoint"], folder / "origin" / "last_epoch_state.pt")
    payload = torch_load_checkpoint(Path(copied["copy_path"]), map_location="cpu")
    require(payload["checkpoint_type"] == "epoch_resume" and type(payload["epoch"]) is int
            and payload["epoch"] == job["resume_epoch"]
            and payload["seed"] == reference_job["seed"]
            and payload["source_revision"] == original["source"]["base_git_revision"],
            "Approved resume snapshot identity changed")
    copy_origin(contract, job["origin_resume_checkpoint"], run_dir / "last_epoch_state.pt")
    # A crash may happen after the final durable epoch but before the selected
    # export.  The engine then performs no new epoch and does not rewrite its
    # history file; materialize only the checkpoint's authoritative prefix in
    # this new attempt, never the possibly newer external original history.
    write(run_dir / "history.json", {"history": payload["history"]})
    configure_runtime(device, 4)
    runtime_before = science.check_native_runtime(original)
    science.mac_runtime.configure_dynamo()
    frame, admitted = base.prepare_admitted_data(data)
    write(folder / "input_receipt.json", admitted)
    with science._engine_hooks(reference_job["arm"], contract=original, data=data) as tracked:
        args = science.training_args(original, data, folder, reference_job["arm"])
        args.seeds, args.device = str(reference_job["seed"]), device
        interface = science.training_interface(original, data, frame, reference_job["arm"])
        with resume_guard(payload, tracked) as continuity:
            summary, _, _ = training.train_one(args=args, frame=frame, quantity_contract=quantity_contract(data),
                interface_meta=interface, backbone=reference_job["arm"],
                quantity_variant=science.VARIANT, seed=reference_job["seed"])
        require(continuity["validated"], "The native engine did not restore the approved checkpoint")
        proof = tracked["tracker"].proof()
    summary = {**summary, **science.input_identity(), "time_head_training_verification": proof}
    checkpoint = science.resolve_under(root, summary["checkpoint_path"])
    scientific = {**science.base_receipt(effective_scientific(contract), "5090", reference_job, "train"),
                  "selected_checkpoint": str(checkpoint), "selected_epoch": summary["best_epoch"],
                  "checkpoint_sha256": sha_file(checkpoint), "state_sha256": summary["checkpoint_state_sha256"],
                  "summary": summary, "time_head_training_verification": proof}
    prediction = folder / "training_validation_predictions.jsonl.gz"
    with science.runtime_hooks(reference_job["arm"]):
        evaluated = science.evaluate_selected(data, reference_job, scientific, root, frame,
                                              "validation", prediction, device=device)
    for key in ("qty_rmse", "qty_mae", "time_nll"):
        require(math.isclose(evaluated["metrics"]["overall"][key], summary["best_val_" + key],
                            rel_tol=science.REPLAY_RTOL, abs_tol=science.REPLAY_ATOL),
                "Resumed selected Validation differs from the training summary")
    scientific.update(validation_reference=evaluated, target_identity=evaluated["target_identity"],
                      native_runtime_checked=True, runtime=runtime_before,
                      runtime_after=science.check_native_runtime(original))
    require(scientific["runtime"] == scientific["runtime_after"], "Native runtime changed during resume")
    scientific["files"] = science.file_inventory(root, [*[p for p in run_dir.rglob("*") if p.is_file()],
                                                       prediction, folder / "input_receipt.json"])
    science.verify_train_receipt(effective_scientific(contract), "5090", reference_job, root, scientific)
    write(folder / "train_receipt.json", scientific)
    verify_origin(contract, job["origin_resume_checkpoint"])
    return _outer_receipt(contract, job, original, "train", scientific,
        {"resume_checkpoint": copied, "resumed_from_epoch": payload["epoch"],
         "next_epoch": payload["epoch"] + 1, "resume_validation": continuity,
         "new_optimizer_steps": proof["optimizer_steps_checked"] - continuity["prefix_optimizer_steps"],
         "elapsed_seconds_scope": "this_recovery_segment_only",
         "embedded_checkpoint_history_authoritative": True})


def run_stage(contract, job_id, phase, *, device="cuda:0"):
    require(phase in ("reference", "train", "validation", "test"), "Unknown recovery stage")
    science.verify_frozen_dependencies()
    if phase == "reference":
        return reference(contract, job_id)
    if phase == "train":
        return train(contract, job_id, device=device)
    require(device == "cuda:0", "Scientific replay requires the sealed native CUDA runtime")
    job, _, _, original, _, _ = select_job(contract, job_id)
    require(job_id in contract[phase + "_jobs"], "Replay is outside its approved job matrix")
    scientific = science.run_stage(effective_scientific(contract), "5090", job_id, phase)
    return _outer_receipt(contract, job, original, phase, scientific,
                          {"scientific_replay_worker": "preserved_original"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", "--recovery-contract", required=True, type=Path)
    parser.add_argument("--contract-sha", required=True)
    parser.add_argument("--phase", "--stage", required=True, choices=("reference", "train", "validation", "test"))
    parser.add_argument("--job-id", "--job", required=True)
    args = parser.parse_args(argv)
    contract = read(args.contract)
    require(sha_json(contract) == args.contract_sha, "Recovery contract SHA changed")
    with redirect_stdout(sys.stderr):
        result = run_stage(contract, args.job_id, args.phase)
    print(json.dumps({key: result[key] for key in ("schema", "status", "stage", "job_id", "host", "contract_sha256")},
                     sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
