"""Summarize the completed, fixed-weight Cap MLP16 Train gradient probe.

Reads only the named CPU-probe outputs and their frozen input/source bindings.
This script does not import Torch, construct a model, infer, train or contact hosts.
The first three shuffled batches are descriptive probes, not independent samples.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
NAME = "titantpp_time_quantity_balance_20261018_v1"
REPORT = ROOT / "reports" / NAME
BUNDLE = ROOT / "search_artifacts" / NAME
LOSSES = ("existing_log_mse", "cap_normalized_raw_mse", "cap_normalized_raw_mae")
CAPS = {"yellow_trip_hourly": 65., "raf_spare_parts": 10., "intermittent_frozen_5000": 20.}
SEEDS = (42, 52, 62)
INITIALIZATIONS = ("fresh", "selected")
EXPECTED_HOST = {(ds, seed): ("5080" if ds == "yellow_trip_hourly" or
    (ds == "intermittent_frozen_5000" and seed == 42) else "5090")
    for ds in CAPS for seed in SEEDS}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def close(actual, expected, label):
    if expected is None:
        require(actual is None, label + " must be null")
    else:
        require(type(actual) in (int, float) and math.isfinite(actual), label + " is not finite")
        require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-11), label + " differs")


def ratio(a, b):
    return a / b if b > 0 else None


def describe(values):
    finite = [float(v) for v in values if v is not None]
    require(all(math.isfinite(v) for v in finite), "Nonfinite summary input")
    return {"available_count": len(finite), "null_count": len(values) - len(finite),
        "median": statistics.median(finite) if finite else None,
        "minimum": min(finite) if finite else None, "maximum": max(finite) if finite else None}


def write_csv(path, rows):
    require(bool(rows), "Empty output table")
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_comparison(comparison, losses, loss_name):
    """Check recorded vector-norm algebra; no gradients are recomputed here."""
    parameter_rows = comparison["parameter_norms"]
    require(len({r["name"] for r in parameter_rows}) == len(parameter_rows), "Duplicate parameter")
    for p in parameter_rows:
        for key in ("time_gradient_norm", "quantity_gradient_norm"):
            require(type(p[key]) in (int, float) and math.isfinite(p[key]) and p[key] >= 0,
                "Invalid per-parameter norm")
        require(p["time_graph_present"] or p["time_gradient_norm"] == 0, "Absent time graph has norm")
        require(p["quantity_graph_present"] or p["quantity_gradient_norm"] == 0, "Absent quantity graph has norm")
    by_name = {p["name"]: p for p in parameter_rows}
    groups = comparison["groups"]
    expected = {
        "shared_graph": [p for p in parameter_rows if p["time_graph_present"] and p["quantity_graph_present"]],
        "time_only_graph": [p for p in parameter_rows if p["time_graph_present"] and not p["quantity_graph_present"]],
        "quantity_only_graph": [p for p in parameter_rows if not p["time_graph_present"] and p["quantity_graph_present"]],
        "inactive_none": [p for p in parameter_rows if not p["time_graph_present"] and not p["quantity_graph_present"]]}
    shared = expected["shared_graph"]
    expected.update(
        shared_nonzero_both=[p for p in shared if p["time_gradient_norm"] > 0 and p["quantity_gradient_norm"] > 0],
        shared_time_nonzero_quantity_zero=[p for p in shared if p["time_gradient_norm"] > 0 and p["quantity_gradient_norm"] == 0],
        shared_time_zero_quantity_nonzero=[p for p in shared if p["time_gradient_norm"] == 0 and p["quantity_gradient_norm"] > 0],
        shared_zero_both=[p for p in shared if p["time_gradient_norm"] == p["quantity_gradient_norm"] == 0])
    require(set(groups) == set(expected), "Unknown or missing parameter group")
    require(set(groups["quantity_only_graph"]["parameter_names"]) == {"quantity_head.weight", "quantity_head.bias"},
        "Quantity-exclusive graph is not the expected head")
    require(set(groups["time_only_graph"]["parameter_names"]) ==
        {"b_t", "w_raw", "v_t.weight", "time_scale_weight.weight"}, "Time-exclusive graph changed")
    for name, group in [*groups.items(), ("global", comparison["global"])]:
        ps = parameter_rows if name == "global" else expected[name]
        require(group["parameter_names"] == [p["name"] for p in ps], "Group inventory differs: " + name)
        require(group["parameter_tensor_count"] == len(ps), "Tensor count differs")
        require(group["parameter_element_count"] == sum(p["elements"] for p in ps), "Element count differs")
        t = math.sqrt(math.fsum(p["time_gradient_norm"] ** 2 for p in ps))
        q = math.sqrt(math.fsum(p["quantity_gradient_norm"] ** 2 for p in ps))
        close(group["time_gradient_norm"], t, name + " time norm")
        close(group["quantity_gradient_norm"], q, name + " quantity norm")
        dot = group["gradient_dot_product"]
        require(type(dot) in (int, float) and math.isfinite(dot), "Invalid gradient dot product")
        require(abs(dot) <= t * q + 1e-10 * max(1., t * q), "Cauchy-Schwarz bound failed")
        close(group["time_to_quantity_gradient_norm_ratio"], ratio(t, q), "time/quantity ratio")
        close(group["quantity_to_time_gradient_norm_ratio"], ratio(q, t), "quantity/time ratio")
        cosine = max(-1., min(1., dot / (t * q))) if t > 0 and q > 0 else None
        close(group["gradient_cosine"], cosine, "cosine")
        require(group["cosine_status"] == ("available" if cosine is not None else "undefined_zero_norm"), "Cosine status differs")
        close(group["lambda_quantity_for_norm_equality"], t / q if t > 0 and q > 0 else None, "Norm-equality lambda")
        close(group["lambda1_joint_gradient_norm"], math.sqrt(max(0., t*t + q*q + 2*dot)), "Joint norm")
    global_group = comparison["global"]
    close(global_group["clip_threshold"], 1., "Clip threshold")
    require(global_group["actual_clipping_applied"] is False, "Unexpected gradient clipping")
    joint = global_group["lambda1_joint_gradient_norm"]
    require(global_group["lambda1_joint_norm_exceeds_clip1"] == (joint > 1.), "Clip indicator differs")
    close(global_group["descriptive_clip1_scale"], min(1., 1. / (joint + 1e-6)), "Clip scale")
    scalar = comparison["loss_scalar"]
    close(scalar["time"], losses["time"], "Time loss scalar")
    close(scalar["quantity"], losses[loss_name], "Quantity loss scalar")
    close(scalar["time_to_quantity_scalar_ratio"], ratio(losses["time"], losses[loss_name]), "Loss scalar ratio")
    return by_name


def analyze():
    outputs = [REPORT / name for name in ("BALANCE_PER_BATCH.csv", "BALANCE_PER_SEED.csv", "BALANCE_SUMMARY.json")]
    require(not any(p.exists() for p in outputs), "Existing summary output; refusing overwrite")
    initial_script_sha = sha(__file__)
    inputs_path = REPORT / "PROBE_INPUTS.json"
    prepared_path = REPORT / "PROBE_PREPARED.json"
    operations_path = BUNDLE / "package/operations.json"
    inputs, prepared, operations = read(inputs_path), read(prepared_path), read(operations_path)
    require(sha(operations_path) == prepared["operations_sha256"], "Prepared operations SHA changed")
    require(operations["files"]["PROBE_INPUTS.json"] == sha(inputs_path), "Input specification changed")
    require(operations["reference_contract_sha256"] == inputs["cap36_lineage"]["contract_raw_sha256"], "Reference contract changed")
    require(operations["maximum_cpu_children"] == 3 and operations["optimizer_steps"] == 0
        and operations["new_training"] is False and operations["native_gpu_used"] is False
        and operations["existing_campaign_signals_sent"] is False, "Operations scope changed")
    verified_files = {}
    def verify(path, expected):
        p = Path(path)
        actual = sha(p)
        require(actual == expected, "SHA mismatch: " + str(p))
        verified_files[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)] = actual
    for name, digest in operations["files"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Invalid package inventory path")
        verify(BUNDLE / "package" / name, digest)
    for path, digest in inputs["input_files_sha256"].items():
        verify(path, digest)
    for name, digest in inputs["cap36_lineage"]["source_inventory"].items():
        verify(ROOT / name, digest)
    verify(ROOT / "paper/scripts/probe_titantpp_time_quantity_balance.py",
        operations["files"]["source/paper/scripts/probe_titantpp_time_quantity_balance.py"])
    cp_by_condition = {(p["dataset"], p["seed"]): p for p in inputs["checkpoints"]}
    data_by_name = {p["dataset"]: p for p in inputs["datasets"]}
    require(set(cp_by_condition) == set(EXPECTED_HOST) and len(inputs["checkpoints"]) == 9, "Checkpoint inventory not exactly nine")
    require(set(data_by_name) == set(CAPS), "Dataset inventory differs")
    expected_conditions = {(ds, seed, mode) for ds, seed in EXPECTED_HOST for mode in INITIALIZATIONS}
    all_jobs = []
    results = {}
    observed_times = {}
    for host in ("5080", "5090"):
        h = operations["hosts"][host]
        expected_here = {(ds, seed, mode) for ds, seed, mode in expected_conditions if EXPECTED_HOST[ds, seed] == host}
        require({(j["dataset"], j["seed"], j["initialization"]) for j in h["jobs"]} == expected_here
            and len(h["jobs"]) == len(expected_here), "Host assignment differs")
        original = BUNDLE / "retrieval" / host / "originals"
        receipt_path = original.parent / "receipt.json"
        receipt = read(receipt_path)
        require(receipt["status"] == "complete" and receipt["host"] == host, "Collection not complete")
        verify(original.parent / "original.tar.gz", receipt["archive_sha256"])
        final_path = original / "final_receipt.json"
        verify(final_path, receipt["final_receipt_sha256"])
        final = read(final_path)
        require(final["status"] == "complete" and final["host"] == host
            and final["condition_count"] == final["actual_worker_exit0_count"] == len(expected_here)
            and final["optimizer_steps"] == 0 and final["held_out_test_evaluated"] is False
            and final["diagnostic_child_pids_absent"] is True and final["prior_campaign_resumed"] is False,
            "Native terminal evidence incomplete")
        observed_times[host] = final["completed_at"]
        require({r["id"] for r in final["results"]} == {j["id"] for j in h["jobs"]}
            and len(final["results"]) == len(h["jobs"]), "Native exit inventory differs")
        require(all(r["actual_os_returncode"] == 0 and r["timeout"] is False and r["status"] == "complete"
            for r in final["results"]), "Worker exit failure")
        require(receipt["raw_inventory"] == final["files"], "Collection/native inventory mismatch")
        require(sorted(p.name for p in (original / "results").iterdir()) == sorted(j["id"] + ".json" for j in h["jobs"]),
            "Unexpected or missing result file")
        for name, digest in final["files"].items():
            require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Invalid native inventory path")
            verify(original / name, digest)
        for j in h["jobs"]:
            key = (j["dataset"], j["seed"], j["initialization"])
            p = original / "results" / (j["id"] + ".json")
            require("results/" + p.name in final["files"], "Result missing from sealed inventory")
            r = read(p)
            require(key not in results, "Duplicate condition")
            require(r["schema"] == "titantpp_train_gradient_balance_v1" and r["status"] == "complete"
                and (r["dataset"], r["seed"], r["initialization"]) == key
                and r["arm"] == "titantpp_history_mlp_width16", "Result identity differs")
            require(r["source_sha256"] == operations["files"]["source/paper/scripts/probe_titantpp_time_quantity_balance.py"], "Worker source changed")
            require(r["contract_raw_sha256"] == operations["reference_contract_sha256"]
                and r["contract_canonical_sha256"] == inputs["cap36_lineage"]["contract_canonical_sha256"], "Worker reference contract differs")
            ds, seed, mode = key
            d, cp = data_by_name[ds], cp_by_condition[ds, seed]
            require(cp["training_host"] == EXPECTED_HOST[ds, seed] == host, "Original host assignment changed")
            require(r["train_input_sha256"] == d["train_validation_input_sha256"], "Train input differs")
            require(r["train_target_identity"] == d["training_target_population"], "Original Train population differs")
            require(r["cap_value"] == d["cap_value"] == CAPS[ds], "Cap value differs")
            require(r["batch_count"] == 3 and r["batch_size"] == 128 and r["training_mode"] is True,
                "Train batch count/mode differs")
            require(r["new_training"] is False and r["optimizer_steps"] == 0
                and r["actual_gradient_clipping"] is False and r["network_calls"] == 0
                and r["validation_loader_created"] is False and r["validation_records_selected"] == 0
                and r["test_records_opened"] == 0 and r["historical_cnn_accessed"] is False,
                "Worker scope violation")
            require(r["process_rng_restored"] is True and r["rng_before_probe_sha256"] == r["rng_after_probe_sha256"]
                and r["model_parameters_buffers_context_unchanged"] is True, "Worker state/RNG changed")
            runtime = r["runtime"]
            require(runtime["device_type"] == runtime["device"] == "cpu" and runtime["gpu"] is None
                and runtime["torch"] == "2.11.0+cu130" and runtime["python"] == "3.12.13"
                and runtime["threads"] == 4 and runtime["environment"]["CUDA_VISIBLE_DEVICES"] == ""
                and runtime["environment"]["PYTHONHASHSEED"] == str(seed), "Native CPU runtime differs")
            if mode == "selected":
                c = r["checkpoint"]
                require(c["sha256"] == cp["checkpoint_sha256"] and c["state_sha256"] == cp["checkpoint_state_sha256"]
                    and c["selected_epoch"] == cp["selected_epoch"] and c["reselected"] is False
                    and c["selection"] == "validation_raw_quantity_rmse" and r["probed_state_sha256"] == c["state_sha256"], "Selected checkpoint binding differs")
            else:
                require(r["checkpoint"] is None and r["probed_state_sha256"] == r["fresh_initial_state_sha256"]
                    and r["quantity_head_weight_all_zero"] is True, "Fresh control differs")
            results[key] = (r, p)
            all_jobs.append({"host": host, "id": j["id"], "result_path": str(p.relative_to(ROOT)),
                "result_sha256": final["files"]["results/" + p.name], "runtime": runtime})
    require(set(results) == expected_conditions and len(results) == 18, "Expected exactly eighteen conditions")
    for ds, seed in EXPECTED_HOST:
        a, b = (results[ds, seed, mode][0] for mode in INITIALIZATIONS)
        require(a["fresh_initial_state_sha256"] == b["fresh_initial_state_sha256"], "Initial model RNG/state differs")
        require(a["train_target_identity"] == b["train_target_identity"], "Train population differs")
        require(a["train_loader_generator_start_sha256"] == b["train_loader_generator_start_sha256"]
            and a["train_loader_generator_end_sha256"] == b["train_loader_generator_end_sha256"], "Loader RNG differs")
        for x, y in zip(a["batches"], b["batches"]):
            require(all(x[k] == y[k] for k in ("batch_number_zero_based", "target_ids_sha256", "batch_tensors_sha256", "targets", "padded_shape")),
                "Fresh/selected Train batch mismatch")
    rows = []
    for key, (r, p) in sorted(results.items()):
        ds, seed, mode = key
        require([b["batch_number_zero_based"] for b in r["batches"]] == [0, 1, 2], "Batch indices differ")
        target_indices = []
        for b in r["batches"]:
            require(b["batch_size"] == len(b["targets"]) == b["padded_shape"][0] == 128, "Batch size differs")
            require(canonical_sha(b["targets"]) == b["target_ids_sha256"], "Batch target IDs changed")
            target_indices.extend(t["target_index"] for t in b["targets"])
            require(b["state_sha256_before"] == b["state_sha256_after"] == r["probed_state_sha256"], "Batch changed state")
            require(all(b[flag] is True for flag in ("training_mode", "parameter_buffer_values_unchanged",
                "parameter_grad_fields_unchanged", "episodic_context_unchanged", "rng_unchanged_by_gradient_computation")), "Batch state invariant failed")
            require(b["forward_count"] == 1 and b["autograd_grad_calls"] == 4 and set(b["comparisons"]) == set(LOSSES), "Forward/gradient or loss inventory differs")
            for loss_name in LOSSES:
                c = b["comparisons"][loss_name]
                check_comparison(c, b["loss_means"], loss_name)
                g, s = c["global"], c["groups"]["shared_graph"]
                row = {"dataset": ds, "host": EXPECTED_HOST[ds, seed], "seed": seed,
                    "initialization": mode, "selected_epoch": r["checkpoint"]["selected_epoch"] if r["checkpoint"] else None,
                    "quantity_loss": loss_name, "batch_number_zero_based": b["batch_number_zero_based"],
                    "batch_size": 128, "cap_value": r["cap_value"],
                    "time_loss_scalar": c["loss_scalar"]["time"], "quantity_loss_scalar": c["loss_scalar"]["quantity"],
                    "scalar_time_to_quantity_ratio": c["loss_scalar"]["time_to_quantity_scalar_ratio"],
                    "shared_time_gradient_norm": s["time_gradient_norm"], "shared_quantity_gradient_norm": s["quantity_gradient_norm"],
                    "shared_quantity_to_time_norm_ratio": s["quantity_to_time_gradient_norm_ratio"],
                    "shared_time_to_quantity_norm_ratio": s["time_to_quantity_gradient_norm_ratio"],
                    "shared_gradient_cosine": s["gradient_cosine"], "shared_lambda_quantity_for_norm_equality": s["lambda_quantity_for_norm_equality"],
                    "shared_lambda1_joint_gradient_norm": s["lambda1_joint_gradient_norm"],
                    "shared_parameter_tensor_count": s["parameter_tensor_count"],
                    "shared_parameter_element_count": s["parameter_element_count"],
                    "shared_nonzero_both_tensor_count": c["groups"]["shared_nonzero_both"]["parameter_tensor_count"],
                    "shared_time_nonzero_quantity_zero_tensor_count": c["groups"]["shared_time_nonzero_quantity_zero"]["parameter_tensor_count"],
                    "global_time_gradient_norm": g["time_gradient_norm"], "global_quantity_gradient_norm": g["quantity_gradient_norm"],
                    "global_quantity_to_time_norm_ratio": g["quantity_to_time_gradient_norm_ratio"],
                    "global_gradient_cosine": g["gradient_cosine"], "global_lambda1_joint_gradient_norm": g["lambda1_joint_gradient_norm"],
                    "global_lambda1_joint_norm_exceeds_clip1": g["lambda1_joint_norm_exceeds_clip1"],
                    "descriptive_global_clip1_scale": g["descriptive_clip1_scale"],
                    "time_exclusive_head_gradient_norm": c["groups"]["time_only_graph"]["time_gradient_norm"],
                    "quantity_exclusive_head_gradient_norm": c["groups"]["quantity_only_graph"]["quantity_gradient_norm"],
                    "quantity_head_weight_all_zero": r["quantity_head_weight_all_zero"],
                    "actual_gradient_clipping_applied": False, "target_ids_sha256": b["target_ids_sha256"],
                    "batch_tensors_sha256": b["batch_tensors_sha256"], "result_sha256": sha(p)}
                if mode == "fresh":
                    require(s["quantity_gradient_norm"] == 0 and s["gradient_cosine"] is None
                        and s["lambda_quantity_for_norm_equality"] is None, "Fresh zero-head control failed")
                rows.append(row)
        require(len(target_indices) == len(set(target_indices)) == 384, "Repeated target within the three shuffled batches")
    require(len(rows) == 162, "Expected 162 batch/loss rows")
    metric_keys = ("time_loss_scalar", "quantity_loss_scalar", "scalar_time_to_quantity_ratio",
        "shared_time_gradient_norm", "shared_quantity_gradient_norm", "shared_quantity_to_time_norm_ratio",
        "shared_time_to_quantity_norm_ratio", "shared_gradient_cosine", "shared_lambda_quantity_for_norm_equality",
        "shared_lambda1_joint_gradient_norm", "global_time_gradient_norm", "global_quantity_gradient_norm",
        "global_quantity_to_time_norm_ratio", "global_gradient_cosine", "global_lambda1_joint_gradient_norm",
        "descriptive_global_clip1_scale", "time_exclusive_head_gradient_norm", "quantity_exclusive_head_gradient_norm")
    seed_rows = []
    for ds, seed, mode in sorted(expected_conditions):
        for loss_name in LOSSES:
            cohort = [x for x in rows if (x["dataset"], x["seed"], x["initialization"], x["quantity_loss"]) == (ds, seed, mode, loss_name)]
            require(len(cohort) == 3, "Seed cohort not three batches")
            s = {"dataset": ds, "host": EXPECTED_HOST[ds, seed], "seed": seed, "initialization": mode,
                "selected_epoch": cohort[0]["selected_epoch"], "quantity_loss": loss_name, "batch_count": 3,
                "shared_negative_cosine_count": sum(x["shared_gradient_cosine"] is not None and x["shared_gradient_cosine"] < 0 for x in cohort),
                "shared_undefined_cosine_count": sum(x["shared_gradient_cosine"] is None for x in cohort),
                "shared_zero_quantity_gradient_count": sum(x["shared_quantity_gradient_norm"] == 0 for x in cohort),
                "global_clip1_exceedance_count": sum(x["global_lambda1_joint_norm_exceeds_clip1"] for x in cohort)}
            for metric in metric_keys:
                s.update({metric + "_" + stat: val for stat, val in describe([x[metric] for x in cohort]).items()})
            seed_rows.append(s)
    require(len(seed_rows) == 54, "Expected 54 seed/loss summaries")
    dataset_rows = []
    for ds in CAPS:
        for mode in INITIALIZATIONS:
            for loss_name in LOSSES:
                cohort = [x for x in rows if (x["dataset"], x["initialization"], x["quantity_loss"]) == (ds, mode, loss_name)]
                require(len(cohort) == 9, "Dataset cohort not nine batches")
                dataset_rows.append({"dataset": ds, "initialization": mode, "quantity_loss": loss_name,
                    "seed_count": 3, "batch_count": 9, "targets_per_seed": 384,
                    "shared_negative_cosine_count": sum(x["shared_gradient_cosine"] is not None and x["shared_gradient_cosine"] < 0 for x in cohort),
                    "shared_undefined_cosine_count": sum(x["shared_gradient_cosine"] is None for x in cohort),
                    "shared_zero_quantity_gradient_count": sum(x["shared_quantity_gradient_norm"] == 0 for x in cohort),
                    "global_clip1_exceedance_count": sum(x["global_lambda1_joint_norm_exceeds_clip1"] for x in cohort),
                    "metrics": {m: describe([x[m] for x in cohort]) for m in metric_keys}})
    # Recheck source/input/output evidence after all arithmetic, before creating reports.
    for relative, digest in verified_files.items():
        require(sha(ROOT / relative if not Path(relative).is_absolute() else relative) == digest, "Analysis changed bound evidence")
    require(sha(__file__) == initial_script_sha, "Summary source changed while running")
    write_csv(outputs[0], rows)
    write_csv(outputs[1], seed_rows)
    summary = {"schema": "titantpp_time_quantity_balance_summary_v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": initial_script_sha,
        "probe_inputs_sha256": sha(inputs_path), "prepared_receipt_sha256": sha(prepared_path),
        "operations_sha256": sha(operations_path), "native_completed_at_unix_by_host": observed_times,
        "counts": {"conditions": 18, "fresh": 9, "selected": 9, "datasets": 3, "seeds": 3,
            "batches_per_condition": 3, "targets_per_batch": 128, "unique_condition_batches": 54,
            "quantity_losses_per_batch": 3, "per_batch_csv_rows": 162, "per_seed_csv_rows": 54,
            "dataset_initialization_loss_summaries": 18, "fresh_selected_matched_condition_pairs": 9,
            "fresh_selected_matched_batch_pairs": 27, "actual_native_worker_exit0": 18},
        "checks": {"exact_conditions_and_original_host_assignment": True,
            "all_batch_ids_and_tensors_identical_fresh_selected": True,
            "all_original_and_packaged_source_input_checkpoint_sha_unchanged": True,
            "all_parameter_buffer_context_and_rng_unchanged": True,
            "fresh_zero_quantity_to_shared_gradient_control_passed": True,
            "recorded_parameter_group_norm_and_ratio_algebra_recomputed": True,
            "all_selected_checkpoint_metadata_and_byte_state_bindings_matched": True,
            "all_native_data_process_exit0_and_local_collection_sha_verified": True},
        "scope": {"dataset_reference": "Cap official reference data", "model": "titantpp_history_mlp_width16",
            "train_batches_only": True, "optimizer_steps": 0, "new_training": False,
            "validation_records_selected": 0, "test_records_opened": 0, "historical_cnn_accessed": False,
            "network_calls_by_summary": 0, "new_model_forward_calls_by_summary": 0,
            "actual_gradient_clipping": False, "prior_cap27_campaign_resumed": False},
        "dataset_summaries": dataset_rows, "native_result_inventory": all_jobs,
        "verified_input_and_native_evidence_files_sha256": verified_files,
        "outputs": {p.name: sha(p) for p in outputs[:2]},
        "interpretation_limits": [
            "The three first shuffled Train batches per seed are a fixed local diagnostic, not independent draws, a learning trajectory, full-Train gradients or Validation/Test performance.",
            "Medians and ranges pool the nine probed batch gradients per dataset/initialization/loss; they are not a training mean, sample SD, confidence interval or significance estimate. Seeds can sample overlapping Train targets.",
            "The three quantity losses share the same predictions and time-loss gradients from one training-mode forward per batch. Raw losses are normalized by the fixed dataset Cap (65/10/20); scalar loss scale and gradient scale are different.",
            "Fresh zero quantity-head weights numerically block quantity gradients to shared parameters despite graph connectivity. Fresh zero norms and undefined cosine/norm-equality lambda are preserved; selected zero-norm exceptions would also remain null.",
            "Norm-equality lambda values and clip-1 exceedances are descriptive diagnostics at lambda=1, not recommended weights or actual optimizer/clipping operations.",
            "Shared versus exclusive head inventories follow autograd graph connectivity. Global gradients include exclusive heads and can conceal shared-gradient conflict.",
            "Native CPU uses original host Python3.12.13/Torch2.11.0+cu130 with four threads and no GPU. Host CUDA/cuDNN installation metadata is recorded, but these gradients are CPU computations and are not asserted identical to GPU training gradients.",
            "Fixed selected Validation checkpoints use the same fresh seeded training dropout/batch RNG as controls, not the historical RNG at selection. No checkpoint reselection or cause/performance claim is supported.",
            "This summarizer checks saved scalar/vector-norm algebra and frozen evidence; it does not independently regenerate gradient vectors or infer models. Controller wrapper OS exit and current global host idleness are not established by this report."]}
    with outputs[2].open("x") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return {"status": "complete", "rows": len(rows), "seed_rows": len(seed_rows),
        "summary_path": str(outputs[2]), "summary_sha256": sha(outputs[2])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(analyze(), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
