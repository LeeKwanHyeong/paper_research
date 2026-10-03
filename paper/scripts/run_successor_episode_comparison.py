#!/usr/bin/env python3
"""5090-only fresh B/same-event/successor comparison through the shared trainer."""
from __future__ import annotations

import argparse
import gc
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paper.scripts import successor_episode_common as common
from paper.scripts import run_observed_slot_partition as shared


def paired_initialization(data):
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    hashes, states = {}, {}
    with torch.random.fork_rng(devices=[]):
        expected_rng = None
        for name in common.ARMS:
            torch.manual_seed(42)
            model, _ = shared._build_model(data, name)
            states[name] = model.state_dict()
            hashes[name] = canonical_state_dict_sha256(states[name])
            if expected_rng is None:
                expected_rng = torch.get_rng_state()
            common.require(torch.equal(expected_rng, torch.get_rng_state()), "Three-arm initialization RNG differs")
        baseline = states[common.ARMS[0]]
        common.require(all(torch.equal(value, states[name][key]) for name in common.ARMS[1:]
                           for key, value in baseline.items()), "Shared B initial tensors differ")
        # The association identity is a buffer; trainable initial tensors match.
        extra_keys = set(states[common.ARMS[1]]) - set(baseline)
        common.require(all(torch.equal(states[common.ARMS[1]][key], states[common.ARMS[2]][key])
                           for key in extra_keys if not key.endswith("association_code")), "New module initialization differs")
    return {"initial_state_sha256": hashes, "shared_state_sha256": hashes[common.ARMS[0]],
            "shared_state_equal": True, "new_parameter_state_equal": True, "caller_cpu_rng_equal": True}


def training_args(contract, data, output_dir, backbone, *, execution_role="fresh_successor_episode_5090_validation"):
    common.require(backbone in common.ARMS, "Unknown successor comparison arm")
    args = shared.training_args(contract, data, output_dir)
    args.execution_role = execution_role
    args.model_role = common.ROLES[common.ARMS.index(backbone)]
    return args


def compare_arms(arms):
    common.require(set(arms) == set(common.ARMS), "All three completed arms are required")
    baseline, control, candidate = (arms[name] for name in common.ARMS)
    comparisons = {
        "control_vs_B": shared.paired_acceptance(baseline, control),
        "candidate_vs_B": shared.paired_acceptance(baseline, candidate),
        "candidate_vs_control": shared.paired_acceptance(control, candidate),
    }
    return {"comparisons": comparisons,
            "candidate_engineering_gates_passed": comparisons["candidate_vs_B"]["accepted"] and comparisons["candidate_vs_control"]["accepted"],
            "scope": "exploratory_single_seed_validation; not_benchmark_or_universal_superiority"}


def partition_totals(contract, host_name):
    """All three arms of each assigned dataset stay on exactly one host."""
    assigned = contract["hosts"][host_name]["assigned_datasets"]
    datasets = [data for data in contract["datasets"] if data["dataset_id"] in assigned]
    common.require(len(datasets) == len(assigned), "Assigned dataset is missing")
    return {"arms": len(assigned) * len(common.ARMS),
            "steps": sum(data["expected_global_steps"] for data in datasets) * len(common.ARMS),
            "replays": len(assigned) * len(common.ARMS) * 2}


def production(contract, permit, host_name, qualification, budget, *, execution_common=common):
    common = execution_common
    import torch
    from paper.scripts.count_aware_tpp_backbone import training
    from paper.scripts.quantity_comparison_data import prepare_quantity_comparison_data
    from paper.scripts.run_quantity_comparison import bind_frozen_statistics
    host = common.validate_assignment(contract, host_name)
    output, runtime = Path(host["output_dir"]), common.runtime_check(contract, host_name)
    common.require(runtime == qualification["runtime"], "Runtime differs from CUDA qualification")
    comparisons, completed, total_steps, replay_count = {}, 0, 0, 0
    for dataset_id in host["assigned_datasets"]:
        budget()
        data = next(item for item in contract["datasets"] if item["dataset_id"] == dataset_id)
        common.verify_source(contract)
        frame, metadata = prepare_quantity_comparison_data(data)
        metadata = bind_frozen_statistics(data, metadata)
        common.require(metadata["held_out_materialized"] is False
                       and metadata["populations"] == data["inherited_data_identity"]["populations"], "Split/populations changed")
        initialization = paired_initialization(data)
        dataset_output = output / dataset_id
        dataset_output.mkdir(exist_ok=False)
        common.write_json(dataset_output / "initialization.json", initialization, exclusive=True)
        common.write_json(dataset_output / "input_receipt.json", metadata, exclusive=True)
        bounds = data["quantity_boundaries_all_train_rows"]
        quantity_contract = {"boundaries": bounds, "strata": [{"label": f"frozen_quantity_bin_{i}"} for i in range(len(bounds) + 1)]}
        interface = {"train_target_mean": metadata["train_log_mean"], "train_target_std": metadata["train_log_std"],
            "time_head": {"time_initial_intercept": 0.}, "data_scope": "verified_train_validation_only",
            "statistics": data["statistics"], "execution_contract_sha256": common.sha_json(contract),
            "source_files_sha256": contract["source"]["files_sha256"]}
        arms, exposures = {}, {}
        for backbone in common.ARMS:
            budget()
            common.verify_source(contract)
            common.require(common.runtime_check(contract, host_name) == runtime, "Runtime changed before arm")
            args = training_args(contract, data, dataset_output, backbone,
                execution_role=getattr(common, "EXECUTION_ROLE", "fresh_successor_episode_5090_validation"))
            run_dir = dataset_output / "runs" / backbone / shared.VARIANT / "seed_42"
            common.require(not run_dir.exists(), "Fresh arm required; no retry/resume/overwrite")
            def status(epoch, records):
                common.write_json(output / "status.json", {"status": "training", "host": host_name,
                    "dataset": dataset_id, "backbone": backbone, "epoch": epoch,
                    "global_steps": sum(row["batches"] for row in records["train"]),
                    "completed_arms": completed, "completed_arm_steps": total_steps,
                    "updated_at_unix": time.time(), "deadline_unix": permit["deadline_unix"]})
                common.write_json(run_dir / "exposure.json", records)
            status(0, {"train": [], "validation": []})
            with shared.audited_training(training, data, budget, status) as exposure:
                summary, _, _ = training.train_one(args=args, frame=frame, quantity_contract=quantity_contract,
                    interface_meta=interface, backbone=backbone, quantity_variant=shared.VARIANT, seed=42)
            common.require(summary["status"] == "success" and summary["completed_epochs"] == 120
                           and summary["stopped_early"] is False, "120-epoch arm incomplete")
            common.require(summary["initial_state_sha256"] == initialization["initial_state_sha256"][backbone], "Trainer initialization differs")
            steps = shared.audit_exposure(exposure, data)
            exposures[backbone] = exposure
            common.write_json(run_dir / "exposure.json", exposure)
            history = shared.read_json(run_dir / "history.json")["history"]
            common.require(len(history) == 120, "History incomplete")
            replay = {}
            for label, path in (("selected", Path(summary["checkpoint_path"])), ("last", run_dir / "last_epoch_state.pt")):
                common.write_json(output / "status.json", {"status": "checkpoint_replay", "host": host_name,
                    "dataset": dataset_id, "backbone": backbone, "checkpoint": label, "completed_arms": completed,
                    "updated_at_unix": time.time(), "deadline_unix": permit["deadline_unix"]})
                replay[label] = shared.replay_checkpoint(path, data, frame, budget, allowed_backbones=common.ARMS)
                row = history[summary["best_epoch"] - 1] if label == "selected" else history[-1]
                common.require(all(math.isclose(replay[label][metric], row["val_" + metric], rel_tol=1e-10, abs_tol=1e-8)
                                   for metric in ("qty_rmse", "qty_mae", "time_nll")), "Endpoint replay differs from history")
                replay_count += 1
            arms[backbone] = {**replay, "last30": shared.last30_summary(history), "global_steps": steps,
                             "best_epoch": summary["best_epoch"], "initial_state_sha256": summary["initial_state_sha256"]}
            common.write_json(run_dir / "endpoint_replays.json", arms[backbone], exclusive=True)
            completed += 1
            total_steps += steps
            gc.collect()
            torch.cuda.empty_cache()
        common.require(all(exposures[name] == exposures[common.ARMS[0]] for name in common.ARMS[1:]), "Actual three-arm batch exposure differs")
        result = {"status": "complete", "dataset": dataset_id, "host": host_name, "arms": arms,
            "initialization": initialization, "exposure_equal": True, "epochs": 120,
            "acceptance": compare_arms(arms), "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
            "baseline_provenance": "fresh_same_host_shared_trainer_not_historical_checkpoint_reproduction"}
        common.write_json(dataset_output / "paired_comparison.json", result, exclusive=True)
        comparisons[dataset_id] = result
        del frame
        gc.collect()
    expected = partition_totals(contract, host_name)
    common.require(completed == expected["arms"] and total_steps == expected["steps"]
                   and replay_count == expected["replays"], "Terminal assigned-arm exposure audit incomplete")
    return {"status": "complete", "host": host_name, "datasets": comparisons,
            "completed_arms": completed, "total_optimizer_steps": total_steps, "endpoint_replays": replay_count,
            "contract_sha256": common.sha_json(contract), "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False, "deadline_unix": permit["deadline_unix"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("execute", "_worker", "prepare-start"))
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--permit", type=Path, required=True)
    parser.add_argument("--host", choices=("5090",), default="5090")
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--owner-pid", type=int)
    parser.add_argument("--authority-fd", type=int)
    args = parser.parse_args()
    if args.command == "prepare-start":
        common.require(args.approval is not None and args.owner_pid is None and args.authority_fd is None, "Explicit approval file required")
        contract = common.read_contract(args.contract)
        permit = common.create_permit(contract, shared.read_json(args.approval))
        common.write_json(args.permit, permit, exclusive=True)
    elif args.command == "execute":
        common.require(args.owner_pid is None and args.authority_fd is None, "Supervisor cannot inherit worker arguments")
        shared.supervisor(args.contract, args.permit, args.host, common=common, entrypoint=__file__, assignment_validator=common.validate_assignment)
    else:
        common.require(args.owner_pid is not None and args.authority_fd is not None, "Internal worker authority required")
        shared.worker(args.contract, args.permit, args.host, args.owner_pid, args.authority_fd,
                      common=common, entrypoint=__file__, production=production, assignment_validator=common.validate_assignment)


if __name__ == "__main__":
    main()
