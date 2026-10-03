#!/usr/bin/env python3
"""Fresh B/shared-mixture/task-specific retrieval comparison through the unchanged shared trainer."""
from __future__ import annotations

import gc
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paper.scripts import task_routed_episode_parallel_common as common
from paper.scripts import run_observed_slot_partition as shared


def paired_initialization(data):
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    from models.TPPs.CountAwareTitanTaskRoutedMemory import TASK_ROUTED_MEMORY_PARAMETER_KEYS
    hashes, states, distinct_queries = {}, {}, {}
    routing_key = "task_routed_episode_memory.routing_code"
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
        baseline_keys = set(baseline)
        for name in common.ARMS[1:]:
            common.require(baseline_keys <= set(states[name]), "Shared B state keys differ")
            common.require(all(value.dtype == states[name][key].dtype
                               and value.shape == states[name][key].shape
                               and torch.equal(value, states[name][key])
                               for key, value in baseline.items()), "Shared B initial tensors differ")
        extra_keys = set(states[common.ARMS[1]]) - baseline_keys
        common.require(extra_keys == set(states[common.ARMS[2]]) - baseline_keys
                       and extra_keys == {*TASK_ROUTED_MEMORY_PARAMETER_KEYS, routing_key},
                       "New module state keys differ")
        common.require(all(states[common.ARMS[1]][key].dtype == states[common.ARMS[2]][key].dtype
                           and states[common.ARMS[1]][key].shape == states[common.ARMS[2]][key].shape
                           and torch.equal(states[common.ARMS[1]][key], states[common.ARMS[2]][key])
                           for key in extra_keys if key != routing_key), "New module initialization differs")
        for name, code in zip(common.ARMS[1:], (0, 1), strict=True):
            value = states[name][routing_key]
            common.require(value.dtype == torch.int64 and value.shape == torch.Size([])
                           and value.item() == code, "Task route identity differs")
            time_query = states[name]["task_routed_episode_memory.time_query_projection.weight"]
            quantity_query = states[name]["task_routed_episode_memory.quantity_query_projection.weight"]
            distinct_queries[name] = not torch.equal(time_query, quantity_query)
            common.require(distinct_queries[name], "Within-model task queries must be independently initialized")
    return {"initial_state_sha256": hashes, "shared_state_sha256": hashes[common.ARMS[0]],
            "shared_state_equal": True, "new_parameter_state_equal": True, "caller_cpu_rng_equal": True,
            "within_model_task_queries_distinct": distinct_queries}


def training_args(contract, data, output_dir, backbone, *, execution_role=common.EXECUTION_ROLE):
    common.require(backbone in common.ARMS, "Unknown task-routed comparison arm")
    common.require(execution_role == common.EXECUTION_ROLE, "Frozen task-routed execution role changed")
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
    common.validate_assignment(contract, host_name)
    assigned = contract["hosts"][host_name]["assigned_datasets"]
    datasets = [data for data in contract["datasets"] if data["dataset_id"] in assigned]
    common.require(len(datasets) == len(assigned), "Assigned dataset is missing")
    return {"arms": len(assigned) * len(common.ARMS),
            "steps": sum(data["expected_global_steps"] for data in datasets) * len(common.ARMS),
            "replays": len(assigned) * len(common.ARMS) * 2}


def audit_artifact_identity(payload, *, backbone, resume_identity, initial_state_sha256,
                            source_revision):
    """Bind a selected, last, or summary artifact to the actual run before replay."""
    from models.TPPs.CountAwareFactory import validate_checkpoint_route
    from paper.scripts.count_aware_tpp_backbone.training import checkpoint_monitor_spec

    common.require(backbone in common.ARMS and payload.get("backbone") == backbone,
                   "Artifact arm changed")
    validate_checkpoint_route(payload, backbone)
    monitor = "validation_raw_quantity_rmse"
    selector = checkpoint_monitor_spec(monitor)
    common.require(payload.get("checkpoint_monitor") == monitor
                   and payload.get("checkpoint_monitor_history_key") == selector["history_key"]
                   and payload.get("checkpoint_selection") == selector["selection"],
                   "Artifact checkpoint selector changed")
    common.require(payload.get("resume_identity") == resume_identity
                   and payload.get("interface_meta") == resume_identity["interface_meta"],
                   "Artifact run/source/split/role identity changed")
    common.require(payload.get("source_revision") == source_revision
                   and payload.get("source_revision_history") == [source_revision]
                   and payload.get("initial_state_sha256") == initial_state_sha256,
                   "Artifact source or fresh initialization changed")
    common.require(payload.get("seed") == 42 and payload.get("variant") == shared.VARIANT
                   and payload.get("evaluation_scope") == "validation_only"
                   and payload.get("held_out_test_evaluated") is False,
                   "Artifact seed/objective/evaluation scope changed")


def production(contract, permit, host_name, qualification, budget):
    import torch
    from paper.scripts.count_aware_tpp_backbone import training
    from paper.scripts.quantity_comparison_data import prepare_quantity_comparison_data
    from paper.scripts.run_quantity_comparison import bind_frozen_statistics
    from simple_lab_test.search.common.runner import torch_load_checkpoint
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
                execution_role=common.EXECUTION_ROLE)
            identity = training._resume_identity(args=args, backbone=backbone,
                quantity_variant=shared.VARIANT, seed=42,
                monitor="validation_raw_quantity_rmse", quantity_contract=quantity_contract,
                interface_meta=interface)
            artifact_identity = {"backbone": backbone, "resume_identity": identity,
                "initial_state_sha256": initialization["initial_state_sha256"][backbone],
                "source_revision": contract["source"]["base_git_revision"]}
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
            audit_artifact_identity(summary, **artifact_identity)
            steps = shared.audit_exposure(exposure, data)
            exposures[backbone] = exposure
            common.write_json(run_dir / "exposure.json", exposure)
            history = shared.read_json(run_dir / "history.json")["history"]
            common.require(len(history) == 120
                           and [row["epoch"] for row in history] == list(range(1, 121)),
                           "History incomplete")
            earliest = training.earliest_strict_minimum(history, metric_key="val_qty_rmse")
            common.require(earliest is not None and summary["best_epoch"] == earliest["epoch"],
                           "Selected checkpoint is not the earliest strict RMSE minimum")
            replay = {}
            for label, path in (("selected", Path(summary["checkpoint_path"])), ("last", run_dir / "last_epoch_state.pt")):
                common.write_json(output / "status.json", {"status": "checkpoint_replay", "host": host_name,
                    "dataset": dataset_id, "backbone": backbone, "checkpoint": label, "completed_arms": completed,
                    "updated_at_unix": time.time(), "deadline_unix": permit["deadline_unix"]})
                budget()
                payload = torch_load_checkpoint(path, map_location="cpu")
                audit_artifact_identity(payload, **artifact_identity)
                common.require(payload.get("best_epoch") == summary["best_epoch"],
                               "Endpoint selector differs from summary")
                if label == "selected":
                    common.require(payload.get("model_state_sha256") == summary["checkpoint_state_sha256"],
                                   "Selected checkpoint digest differs from summary")
                else:
                    common.require(payload.get("epoch") == 120 and payload.get("history") == history,
                                   "Final endpoint epoch/history differs")
                del payload
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

