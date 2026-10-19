"""Resume only approved interrupted late-path fits in a new immutable-lineage root.

The sealed scientific worker, model, original interface and optimizer are reused.
RAF references are only replayed, and original files are never modified.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil

from paper.scripts import run_titantpp_late_path18 as science

SCHEMA = "titantpp_late_path18_recovery_stage_v1"
CONTRACT_SCHEMA = "titantpp_late_path18_5090_recovery_v1"
SCIENCE_WORKER_SHA256 = "ece9e4856695962cf157de9c7c40d97fa2cce5ad3712f920e62174c06b7d4ab4"
require, read, sha_file, sha_json = science.require, science.read, science.sha_file, science.sha_json
write = science.write_exclusive


def load_scientific(contract):
    require(sha_file(science.__file__) == SCIENCE_WORKER_SHA256, "Preserved scientific worker changed")
    require(contract.get("schema") == CONTRACT_SCHEMA and contract.get("host") == "5090", "Wrong recovery scope")
    root, source = Path(contract["root"]), Path(contract["source_root"])
    require(root.is_absolute() and source == root / "source", "Wrong isolated recovery root")
    binding = contract["scientific_contract"]
    require(sha_file(binding["path"]) == binding["raw_sha256"], "Original scientific contract bytes changed")
    original = read(binding["path"])
    require(sha_json(original) == binding["canonical_sha256"] and original["schema"] == science.CONTRACT_SCHEMA,
        "Original scientific contract identity changed")
    require(original["common_condition"] == {"time_input": "log1p", "loss_mode": "log_mse", "time_alpha": 0.},
        "Original common condition changed")
    require(contract["source"]["files_sha256"] == original["source"]["files_sha256"]
        and contract["source"]["files"] == original["source"]["files"]
        and contract["source"]["source_revision"] == original["source"]["base_git_revision"],
        "Scientific source or source revision changed")
    require(Path(contract["immutable_originals"]["root"]) == Path(original["hosts"]["5090"]["root"])
        and root != Path(contract["immutable_originals"]["root"]), "Recovery must not write into the old root")
    require(contract["absolute_host_deadline_unix"] == original["hosts"]["5090"]["absolute_host_deadline_unix"],
        "Original host deadline changed")
    return original


def select_job(contract, job_id):
    original = load_scientific(contract)
    jobs = [j for j in contract["jobs"] if j["id"] == job_id]
    require(len(jobs) == 1, "Recovery job is not uniquely bound")
    job = jobs[0]
    reference, data, old_root, _ = science.select_job(original, "5090", job_id)
    require(all(job[k] == reference[k] for k in ("id", "dataset", "arm", "seed", "host", "time_alpha", "loss_mode", "time_input")),
        "Recovery scientific job changed")
    require(job.get("recovery_mode") in ("reference", "resume", "fresh"), "Unknown recovery mode")
    require((job["recovery_mode"] == "reference" and job["dataset"] == "raf_spare_parts")
        or (job["recovery_mode"] != "reference" and job["dataset"] == "intermittent_frozen_5000"
            and job["seed"] in (52, 62)), "Only RAF references and remaining Intermittent fits are approved")
    if job["recovery_mode"] == "fresh":
        require(job["seed"] == 62 and job["arm"] == science.ARMS[1],
            "Only never-started Intermittent seed62 hidden path may start fresh")
    if job["recovery_mode"] == "resume":
        require(job["seed"] == 52 or (job["seed"] == 62 and job["arm"] == science.ARMS[0]),
            "Only the three interrupted fits may resume")
    return job, reference, data, original, Path(contract["root"]), old_root


def verify_origin(contract, binding):
    root = Path(contract["immutable_originals"]["root"]).resolve()
    path = Path(binding["path"]).resolve()
    require(path.is_relative_to(root), "Original binding escapes the frozen old root")
    relative = str(path.relative_to(root))
    require(contract["immutable_originals"]["files"].get(relative) == binding["sha256"]
        and sha_file(path) == binding["sha256"], "Frozen original binding changed")
    return path


def copy_origin(contract, binding, destination):
    origin = verify_origin(contract, binding)
    destination = Path(destination)
    require(destination.resolve().is_relative_to(Path(contract["root"]).resolve()), "Copy outside new recovery root")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with origin.open("rb") as source, destination.open("xb") as target:
        shutil.copyfileobj(source, target)
    require(sha_file(destination) == binding["sha256"], "Original copy SHA differs")
    return {"original_path": str(origin), "original_sha256": binding["sha256"],
        "copy_path": str(destination), "copy_sha256": sha_file(destination)}


def quantity_contract(data):
    return {"boundaries": data["quantity_boundaries_all_train_rows"],
        "strata": [{"stratum": f"quantity_{i}", "stratum_order": i, "stratum_label": f"Cap Train bin {i}"}
                   for i in range(5)]}


def interface_for(original, data, frame, arm):
    from paper.scripts import run_multilag_detail_execution as base
    interface = base.time_interface(data, frame, original)
    interface["backbone_design"] = {"schema": science.SCHEMA, "arm": arm,
        "cap_value": data["cap_value"], "training": original["training"]}
    return interface


@contextmanager
def resume_guard(payload):
    """Keep validated prefix optimizer-step proof while the sealed engine restores RNG/state."""
    from paper.scripts.count_aware_tpp_backbone import training
    original = training._validate_resume_payload
    observed = {"validated": False}
    def validate(value, **kwargs):
        require(value["model_state_sha256"] == payload["model_state_sha256"]
            and value["epoch"] == payload["epoch"], "Resume payload changed before engine restore")
        result = original(value, **kwargs)
        proof = science.verify_time_head_proof(value.get("time_head_no_training_verification"), value["model_state_dict"])
        science.verify_time_head_proof(proof, value["best_state_dict"])
        tracker = science._parent._zero._TRACKER
        require(tracker is not None and tracker.steps == 0 and tracker.initial_sha == proof["initial_time_head_sha256"],
            "Resume time-head tracker initialization differs")
        tracker.steps = proof["optimizer_steps_checked"]
        tracker.check()
        observed.update(validated=True, prefix_optimizer_steps=tracker.steps)
        return result
    training._validate_resume_payload = validate
    try:
        yield observed
    finally:
        training._validate_resume_payload = original


def stage_base(contract, job, original, stage):
    return {"schema": SCHEMA, "status": "complete", "host": "5090", "job_id": job["id"],
        "stage": stage, "contract_sha256": sha_json(contract),
        "scientific_contract_sha256": sha_json(original),
        "scientific_contract_raw_sha256": contract["scientific_contract"]["raw_sha256"],
        **{k: job[k] for k in ("dataset", "arm", "seed", "time_alpha", "loss_mode", "time_input")},
        **science.candidate_labels(job["arm"]), "recovery_mode": job["recovery_mode"],
        "held_out_test_evaluated": False, "quantity_only": True, "time_head_trained": False,
        "quantity_loss_normalization": "none", "real_data_loaded": True,
        "original_files_modified": False, "scientific_worker_source_unchanged": True}


def _folder_inventory(root, folder, exclude):
    return science.file_inventory(root, [p for p in folder.rglob("*") if p.is_file() and p != exclude])


def train(contract, job_id, *, device="cuda:0"):
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    from paper.scripts.count_aware_tpp_backbone import training
    from paper.scripts import run_multilag_detail_execution as base
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    job, reference, data, original, root, old_root = select_job(contract, job_id)
    require(job["recovery_mode"] != "reference" and job_id in contract["training_jobs"], "RAF references cannot be retrained")
    folder = root / "run" / job_id
    write(folder / "train.intent.json", {"job_id": job_id, "contract_sha256": sha_json(contract), "stage": "train"})
    configure_runtime(device, 4)
    lineage = {}
    payload = None
    run_dir = folder / "runs" / reference["arm"] / science.VARIANT / f"seed_{reference['seed']}"
    if job["recovery_mode"] == "resume":
        saved = copy_origin(contract, job["origin_resume_checkpoint"], folder / "origin" / "last_epoch_state.pt")
        payload = torch_load_checkpoint(Path(saved["copy_path"]), map_location="cpu")
        require(payload["epoch"] == job["resume_epoch"] and payload["checkpoint_type"] == "epoch_resume",
            "Resume epoch/type differs from the approved snapshot")
        require(payload["source_revision"] == original["source"]["base_git_revision"], "Resume scientific source changed")
        copy_origin(contract, job["origin_resume_checkpoint"], run_dir / "last_epoch_state.pt")
        lineage.update(resume_checkpoint=saved, resumed_from_epoch=payload["epoch"],
            next_epoch=payload["epoch"] + 1, original_started_at=job["original_started_at"])
    else:
        require(not run_dir.exists(), "Fresh recovery condition already exists")
        lineage.update(fresh_from_original_seed=True, resumed_from_epoch=None)
    with science.runtime_hooks(reference["loss_mode"], time_input=reference["time_input"], arm=reference["arm"]):
        frame, admitted = base.prepare_admitted_data(data)
        write(folder / "input_receipt.json", admitted)
        args = science.training_args(original, data, folder, reference["arm"], time_alpha=0.)
        args.seeds, args.device = str(reference["seed"]), device
        interface = interface_for(original, data, frame, reference["arm"])
        if payload is not None:
            with resume_guard(payload) as resume_check:
                summary, _, _ = training.train_one(args=args, frame=frame, quantity_contract=quantity_contract(data),
                    interface_meta=interface, backbone=reference["arm"], quantity_variant=science.VARIANT, seed=reference["seed"])
            require(resume_check["validated"], "Sealed engine did not restore the approved resume state")
            lineage.update(resume_validation=resume_check)
        else:
            summary, _, _ = training.train_one(args=args, frame=frame, quantity_contract=quantity_contract(data),
                interface_meta=interface, backbone=reference["arm"], quantity_variant=science.VARIANT, seed=reference["seed"])
        checkpoint = science.resolve_under(root, summary["checkpoint_path"])
        scientific = {**science.base_receipt(original, "5090", reference, "train"),
            "selected_checkpoint": str(checkpoint), "selected_epoch": summary["best_epoch"],
            "checkpoint_sha256": sha_file(checkpoint), "state_sha256": summary["checkpoint_state_sha256"], "summary": summary}
        prediction = folder / "training_validation_predictions.jsonl.gz"
        evaluated = science.evaluate_selected(data, reference, scientific, root, frame, "validation", prediction, device=device)
        for key, summary_key in (("qty_rmse", "best_val_qty_rmse"), ("qty_mae", "best_val_qty_mae"), ("time_nll", "best_val_time_nll")):
            import math
            require(math.isclose(evaluated["metrics"]["overall"][key], summary[summary_key], rel_tol=1e-5, abs_tol=1e-5),
                "Recovered selected Validation differs from summary")
        scientific.update(validation_reference=evaluated, target_identity=evaluated["target_identity"])
        scientific["files"] = science.file_inventory(root, [*[p for p in run_dir.rglob("*") if p.is_file()],
            prediction, folder / "input_receipt.json"])
        scientific_path = folder / "train_scientific_receipt.json"
        write(scientific_path, scientific)
        result = {**stage_base(contract, job, original, "train"),
            **{k: scientific[k] for k in ("selected_checkpoint", "selected_epoch", "checkpoint_sha256", "state_sha256", "target_identity", "time_head_no_training_verification")},
            "metrics": evaluated["metrics"], "summary": summary, "lineage": lineage,
            "scientific_receipt": {"path": str(scientific_path), "sha256": sha_file(scientific_path)}}
        receipt_path = folder / "train_receipt.json"
        result["files"] = _folder_inventory(root, folder, receipt_path)
        write(receipt_path, result)
    return result


def validation(contract, job_id, *, device="cuda:0"):
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    from paper.scripts import run_multilag_detail_execution as base
    job, reference, data, original, root, old_root = select_job(contract, job_id)
    require(job_id in contract["validation_jobs"], "Validation job not approved")
    folder = root / "run" / job_id
    write(folder / "validation.intent.json", {"job_id": job_id, "contract_sha256": sha_json(contract), "stage": "validation"})
    configure_runtime(device, 4)
    with science.runtime_hooks(reference["loss_mode"], time_input=reference["time_input"], arm=reference["arm"]):
        lineage = {}
        if job["recovery_mode"] == "reference":
            origin_path = verify_origin(contract, job["origin_train_receipt"])
            origin = read(origin_path)
            science.verify_train_receipt(original, "5090", reference, old_root, origin)
            require(origin["checkpoint_sha256"] == job["origin_selected_checkpoint"]["sha256"], "RAF selected checkpoint changed")
            receipt_copy = copy_origin(contract, job["origin_train_receipt"], folder / "origin" / "train_receipt.json")
            checkpoint_copy = copy_origin(contract, job["origin_selected_checkpoint"], folder / "origin" / "selected.pt")
            selected = {**origin, "selected_checkpoint": checkpoint_copy["copy_path"]}
            lineage.update(reference_train_receipt=receipt_copy, reference_checkpoint=checkpoint_copy,
                reference_retrained=False, original_selected_checkpoint=origin["selected_checkpoint"])
        else:
            outer_path = folder / "train_receipt.json"
            outer = read(outer_path)
            require(outer["schema"] == SCHEMA and outer["status"] == "complete"
                and outer["contract_sha256"] == sha_json(contract), "Recovery training receipt mismatch")
            binding = outer["scientific_receipt"]
            require(sha_file(binding["path"]) == binding["sha256"], "Recovered scientific receipt changed")
            selected = read(binding["path"])
            science.verify_train_receipt(original, "5090", reference, root, selected)
            lineage.update(recovery_train_receipt={"path": str(outer_path), "sha256": sha_file(outer_path)},
                recovery_scientific_receipt=binding)
        frame, _ = base.prepare_admitted_data(data)
        prediction = folder / "validation_predictions.jsonl.gz"
        evaluated = science.evaluate_selected(data, reference, selected, root, frame, "validation", prediction, device=device)
        science.compare_metrics(evaluated["metrics"], selected["validation_reference"]["metrics"])
        require(evaluated["target_identity"] == selected["target_identity"], "Recovered Validation targets differ")
        science.compare_prediction_files(prediction, selected["validation_reference"]["prediction_path"])
        result = {**stage_base(contract, job, original, "validation"), **evaluated,
            "selected_checkpoint": selected["selected_checkpoint"], "selected_epoch": selected["selected_epoch"],
            "checkpoint_sha256": selected["checkpoint_sha256"], "validation_replay_passed": True,
            "time_head_no_training_verification": science.verify_time_head_proof(selected["time_head_no_training_verification"]),
            "lineage": lineage}
        receipt_path = folder / "validation_receipt.json"
        result["files"] = _folder_inventory(root, folder, receipt_path)
        write(receipt_path, result)
    return result


def qualify(contract, job_id=None, *, device="cuda:0"):
    original = load_scientific(contract)
    if job_id is None:
        return science.synthetic_check(original, "5090", device=device)
    job, reference, _, _, _, _ = select_job(contract, job_id)
    return science.synthetic_condition(original, "5090", reference["loss_mode"], arm=reference["arm"], device=device)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-contract", required=True, type=Path)
    parser.add_argument("--job")
    parser.add_argument("--stage", required=True, choices=("train", "validation", "qualify"))
    args = parser.parse_args(argv)
    contract = read(args.recovery_contract)
    require(args.stage == "qualify" or args.job is not None, "Scientific stage requires its exact job")
    result = qualify(contract, args.job) if args.stage == "qualify" else (train if args.stage == "train" else validation)(contract, args.job)
    public = result if args.stage == "qualify" else {k: result[k] for k in ("status", "job_id", "stage", "host", "real_data_loaded", "candidate_name") if k in result}
    print(json.dumps(public, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
