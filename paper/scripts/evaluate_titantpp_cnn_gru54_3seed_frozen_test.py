#!/usr/bin/env python3
"""Once-only aggregate evaluation of nine immutable CNN/GRU checkpoints.

The sealed capacity evaluator supplies its existing streaming inference and
population digests. This operational adapter changes only its process-local
model routing to the frozen CNN/GRU routes. Every full Validation receipt is
verified before any Test population can be loaded. Importing this file performs
no model, data, CUDA, network, training, or inference work.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

KIND = "titantpp_cnn_gru54_frozen_test_3seed_v1"
PARENT_SHA = "46a0eb48d5207df800553f5d3508016ce9063d1dec9be2c5e84bfa87a1b1fdae"
PARENT_SOURCE_SHA = "40184c7ff418f35b191ebfa309e462b9e88889443f36e4722cc5d06988d182cd"
SOURCE_SHA = "4e94229fae002dc678025ecf056862f832c64459969f403b3cb5ac8cea8ba7b0"
THREESEED_SHA = "845a891db284f2cf88aa2ab17bee47716ea8dc11e6268646a71c772073ab7130"
PRIOR_SHA = "c6a41926b01e40b8c8b0c699f16e151a05467235275e0719885e9f972ea50667"
ARMS = ("titantpp_cnn_gru54",)
SEEDS = (42, 52, 62)
DATASETS = ("yellow_trip_hourly", "intermittent_frozen_5000", "raf_spare_parts")
TAILS = dict(zip(DATASETS, (3449., 187., 200.)))
EXPECTED = {(d, a, s) for d in DATASETS for a in ARMS for s in SEEDS}
METRICS = ("qty_rmse", "qty_mae", "time_nll")
BUNDLE_CLOSURES = {"cnn_gru_117": PARENT_SOURCE_SHA, "cnn_gru_123": SOURCE_SHA}
QUALIFICATION_KEYS = {
    "target_quantity_perturbation_preserves_prediction",
    "target_quantity_perturbation_preserves_time_loss",
    "target_gap_perturbation_preserves_quantity_prediction",
    "single_vs_batch_prediction_matches", "single_vs_batch_time_loss_matches",
    "parameters_unchanged", "finite_outputs",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def confined(root, path):
    root = Path(root).resolve()
    found = (root / path).resolve()
    require(found.is_relative_to(root), "Path escapes evaluation root: " + str(path))
    return found


def write(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    temporary.replace(path)


def identity(row):
    return f"{row['dataset']}__{row['model']}__seed{row['seed']}"


def row_source_closure(row):
    name = row["evaluator_source_bundle"]
    require(name in BUNDLE_CLOSURES, "Foreign source bundle")
    require(name == ("cnn_gru_117" if row["seed"] == 42 else "cnn_gru_123"),
        "Checkpoint seed/source lineage differs")
    return BUNDLE_CLOSURES[name]


def first_minimum(history):
    rows = history["history"]
    require(rows and [r["epoch"] for r in rows] == list(range(1, len(rows) + 1)),
        "Original history must be contiguous from epoch one")
    require(all(r.get("train_all_finite") is True and math.isfinite(r["val_qty_rmse"])
        and all(not isinstance(v, float) or math.isfinite(v) for v in r.values()) for r in rows),
        "Original history is not finite")
    return min(rows, key=lambda r: r["val_qty_rmse"])


def dataset_science(specs):
    result = deepcopy(specs)
    for d in result:
        d.pop("prepared_train_validation_path", None)
        d.pop("data_sha256", None)
        for key in ("inherited_data_identity", "parent_data_identity"):
            for field in ("data", "split_manifest"):
                value = d.get(key, {}).get(field, {})
                if "path" in value:
                    value["path"] = "<mapped-operational-path>"
    return sorted(result, key=lambda d: d["dataset_id"])


def verify_comparison_gate(root, c, rows):
    """The prior final Validation comparison authorizes evidence, not adoption."""
    binding = c["final_validation_comparison_gate"]
    path = confined(root, binding["path"])
    require(sha(path) == binding["sha256"], "Final Validation comparison gate SHA changed")
    gate = read(path)
    require(gate.get("passed") is True and gate.get("conditions") == 9
        and gate.get("evaluation_scope") == "validation_only"
        and gate.get("held_out_test_evaluated") is False,
        "Final Validation comparison evidence is incomplete or foreign")
    receipt_path = confined(root, gate["comparison_receipt_path"])
    require(sha(receipt_path) == gate["comparison_receipt_sha256"],
        "Final Validation comparison receipt SHA changed")
    receipt = read(receipt_path)
    require(receipt.get("status") == "complete" and receipt.get("candidate_conditions") == 9
        and receipt.get("scope") == "Validation_only_original_quantity_selected_endpoints"
        and receipt.get("frozen_contract_sha256") == THREESEED_SHA
        and all(receipt.get(k) is False for k in ("held_out_performance_read", "mixed_performance_file_read", "raw_prediction_read")),
        "Final Validation comparison receipt scope/source/conditions differ")
    entries = gate["rows"]
    expected = {identity(r): r for r in rows}
    require(len(entries) == 9 and {identity(r) for r in entries} == set(expected),
        "Final Validation comparison must bind nine unique conditions")
    for entry in entries:
        row = expected[identity(entry)]
        for key in ("dataset", "model", "seed", "checkpoint_file_sha256", "state_tensor_sha256", "selected_epoch"):
            require(entry[key] == row[key], "Final Validation comparison checkpoint changed: " + key)
    return gate


def verify_exposure_lineage(root, c):
    path = confined(root, "evaluation_lineage.json")
    require(sha(path) == c["evaluation_lineage_sha256"], "Evaluation exposure lineage SHA changed")
    lineage = read(path)
    require(lineage == c["test_exposure_lineage"], "Evaluation exposure lineage differs from contract")
    require(lineage.get("Test_previously_accessed") is True
        and lineage.get("independent_untouched_Test") is False
        and lineage.get("seed42_checkpoint_Test_previously_evaluated") is True
        and lineage.get("seed52_62_new_checkpoint_Test_inference") is True
        and lineage.get("head_refits_Test_evaluated") is False
        and lineage.get("fresh_replay_splits") == ["validation", "test"],
        "Prior Test exposure/replay interpretation changed")
    return lineage


def verify_original(root, row, accepted):
    """Validate a selected endpoint from SHA-bound training evidence, never select anew."""
    binding = row["original_binding"]
    terminal_path = confined(root, binding["terminal_manifest_path"])
    require(sha(terminal_path) == binding["terminal_manifest_sha256"], "Original terminal SHA changed")
    terminal = read(terminal_path)
    require(terminal["status"] == "complete" and terminal["scientific_success"] is True,
        "Original fit is not scientifically complete")
    require(terminal["contract_sha256"] == binding["training_contract_sha256"]
        and terminal["contract_sha256"] in accepted, "Unbound original training contract")
    job = terminal["job"]
    require(job == row["job"] and (job["dataset"], job["arm"], job["seed"])
        == (row["dataset"], row["model"], row["seed"]), "Original scientific job differs")
    for name, pathkey, shakey in (
        ("best_val_qty_rmse_model.pt", "checkpoint_path", "checkpoint_file_sha256"),
        ("history.json", "history_path", "history_sha256"),
        ("endpoint_replays.json", "endpoint_path", "endpoint_sha256"),
    ):
        spec = row if pathkey == "checkpoint_path" else binding
        path = confined(root, spec[pathkey])
        require(path.is_relative_to(terminal_path.parent), "Original binding escapes its job folder")
        relative = path.relative_to(terminal_path.parent).as_posix()
        require(path.name == name and terminal["files"].get(relative) == spec[shakey]
            and sha(path) == spec[shakey], "Original binding SHA changed: " + name)
    history = read(confined(root, binding["history_path"]))
    endpoint = read(confined(root, binding["endpoint_path"]))
    best = first_minimum(history)
    require(endpoint["status"] == "complete" and endpoint["evaluation_scope"] == "validation_only"
        and endpoint["held_out_test_evaluated"] is False, "Original endpoint scope/completion differs")
    require(endpoint["completed_epochs"] == len(history["history"])
        and endpoint["best_epoch"] == best["epoch"] == row["selected_epoch"], "Earliest minimum epoch changed")
    require(best["val_qty_rmse"] == row["selected_metric_value"], "Original selected objective changed")
    for point in ("selected", "last"):
        p = endpoint[point]
        require(p["evaluation_scope"] == "validation_only" and p["held_out_test_evaluated"] is False,
            "Original replay contains a foreign split")
        require(p["count"] == row["validation_count"] and all(math.isfinite(p[k]) for k in METRICS),
            "Original full Validation population/metrics differ")
    require(endpoint["selected"]["state_sha256"] == row["state_tensor_sha256"], "Original selected tensor SHA changed")
    # Resumed endpoints may replay an earlier A100-selected state on another GPU;
    # preserve its history value, and separately bound the cross-GPU endpoint.
    require(math.isclose(best["val_qty_rmse"], endpoint["selected"]["qty_rmse"], rel_tol=1e-5, abs_tol=1e-5),
        "Original selected endpoint is not the history-selected state")
    return endpoint["selected"]


def load_campaign(campaign, *, verify_bytes=True):
    campaign = Path(campaign).resolve()
    c = read(campaign / "execution_contract.json")
    root = Path(c["resources"]["root"]).resolve()
    require(campaign.is_relative_to(root), "Campaign must be inside sealed evaluation root")
    require(c.get("schema_version") == 1 and c.get("campaign_kind") == KIND, "Foreign Test campaign")
    require(c["parent_training_contract_sha256"] == PARENT_SHA and c["source_closure_sha256"] == SOURCE_SHA,
        "Foreign parent/source closure")
    require(c["original_3seed_training_contract_sha256"] == THREESEED_SHA,
        "Foreign original three-seed training contract")
    require(all(c.get(k) is False for k in ("automatic_retry", "raw_predictions_written", "selection_changed", "new_training")),
        "Retry, prediction export, retraining, or reselection is forbidden")
    a = c["approval"]
    require(a.get("test_inference_authorized") is True and a.get("retraining_authorized") is False
        and set(a["allowed_splits"]) == {"validation", "test"} and set(a["datasets"]) == set(DATASETS)
        and set(a["models"]) == set(ARMS) and a["seeds"] == list(SEEDS), "Approval scope differs from nine CNNGRU three-seed conditions")
    require(c["validation_gate"] == {"conditions": 9, "all_before_test": True, "rel_tol": 1e-5, "abs_tol": 1e-5},
        "Global Validation9 replay gate changed")
    limits = c["resources"]
    require(limits["concurrent_workers"] == 1 and limits["threads"] == 4
        and 0 < limits["condition_timeout_seconds"] <= limits["campaign_timeout_seconds"]
        and 0 < limits["max_device_memory_fraction"] <= .8
        and 0 < limits["rss_limit_gib"] <= 16 and 0 < limits["gpu_memory_limit_gib"] <= 16,
        "Invalid or expanded resource ceilings")
    for name, key in (("evaluation_registry.json", "registry_sha256"), ("dataset_manifest.json", "dataset_manifest_sha256")):
        require(sha(campaign / name) == c[key], "Campaign input SHA changed: " + name)
    require(c["prior_evaluator"]["sha256"] == c["evaluator_sha256"] == PRIOR_SHA, "Unsealed prior evaluator")
    if verify_bytes:
        require(sha(__file__) == c["operation_adapter_sha256"], "Operational evaluator SHA changed")
        require(sha(confined(root, c["prior_evaluator"]["path"])) == PRIOR_SHA, "Prior evaluator bytes changed")
        approval_path = confined(root, a["source_approval_path"])
        require(sha(approval_path) == a["source_approval_sha256"], "Human authorization file changed")
        approval = read(approval_path)
        require(approval.get("approved") is True and approval.get("test_inference_authorized") is True,
            "Human authorization is not explicit")
    accepted = {}
    for entry in c["accepted_training_contracts"]:
        training = read(confined(root, entry["path"]))
        digest = canonical(training)
        require(digest == entry["canonical_sha256"] and digest not in accepted, "Training contract SHA/uniqueness changed")
        accepted[digest] = training
    require(PARENT_SHA in accepted and THREESEED_SHA in accepted, "Original scientific contracts are missing")
    parent = accepted[PARENT_SHA]
    current = accepted[THREESEED_SHA]
    require(len(parent["source"]["files"]) == 117
        and canonical(parent["source"]["files"]) == parent["source"]["files_sha256"] == PARENT_SOURCE_SHA,
        "Original source117 closure changed")
    require(len(current["source"]["files"]) == 123
        and canonical(current["source"]["files"]) == current["source"]["files_sha256"] == SOURCE_SHA
        and current["source"]["parent_canonical_sha256"] == PARENT_SHA
        and current["source"]["parent_source_closure"] == PARENT_SOURCE_SHA
        and all(current["source"]["files"].get(p) == digest for p, digest in parent["source"]["files"].items()),
        "Current source123 is not the unchanged parent117 plus sealed additions")
    require(dataset_science(current["datasets"]) == dataset_science(parent["datasets"]),
        "Current original scientific data/model/objective contract differs")
    for digest, training in accepted.items():
        require(training["training"] == parent["training"],
            "Training scientific source/objective/selector differs")
        if digest == THREESEED_SHA:
            require(training["evaluation_scope"] == "validation_only" and training["held_out_test_evaluated"] is False
                and training["arms"] == list(ARMS) and len(training["jobs"]) == 6
                and {(j["dataset"], j["arm"], j["seed"]) for j in training["jobs"]}
                == {(d, ARMS[0], s) for d in DATASETS for s in (52, 62)},
                "New original training conditions/scope differ")
        else:
            require(training["source"] == parent["source"]
                and (digest == PARENT_SHA or training.get("parent", {}).get("contract_sha256") == PARENT_SHA),
                "Derived original source117 training contract is not bound to parent")
    registry = read(campaign / "evaluation_registry.json")
    manifest = read(campaign / "dataset_manifest.json")
    require(len(manifest["datasets"]) == 3 and {d["dataset"] for d in manifest["datasets"]} == set(DATASETS),
        "Dataset manifest must contain each of the three datasets exactly once")
    rows = registry["rows"]
    require(len(rows) == 9 and {(r["dataset"], r["model"], r["seed"]) for r in rows} == EXPECTED
        and len({r["job"]["id"] for r in rows}) == 9, "Nine canonical scientific conditions required")
    require(set(c["validation_references"]) == {identity(r) for r in rows}, "Validation reference coverage differs")
    require(set(c["population_references"]) == set(DATASETS), "Split population coverage differs")
    require(set(registry["bundles"]) == set(BUNDLE_CLOSURES), "Exactly original117/current123 source bundles required")
    require(len({confined(root, b["source_root"]) for b in registry["bundles"].values()}) == 2,
        "Scientific source117/source123 roots must remain separate")
    for name, bundle in registry["bundles"].items():
        original = parent if name == "cnn_gru_117" else current
        require(bundle["source_files"] == original["source"]["files"]
            and canonical(bundle["source_files"]) == bundle["source_closure_sha256"] == BUNDLE_CLOSURES[name],
            "Scientific source closure changed: " + name)
        require(dataset_science(bundle["datasets"]) == dataset_science(original["datasets"]), "Scientific data/model contract changed")
        if verify_bytes:
            source = confined(root, bundle["source_root"])
            require((source / "sample_data/.keep").is_file(), "Frozen project-root marker missing")
            for relative, digest in bundle["source_files"].items():
                require(sha(confined(source, relative)) == digest, "Scientific source changed: " + relative)
    for row in rows:
        row_source_closure(row)
        digest = row["original_binding"]["training_contract_sha256"]
        require(digest in accepted, "Row has an unaccepted original training contract")
        training = accepted[digest]
        require((digest == THREESEED_SHA) is (row["seed"] in (52, 62)),
            "Row checkpoint belongs to the wrong original training lineage")
        require(row["endpoint"] == "selected" and row["source_revision"] == training["source"]["base_git_revision"],
            "Checkpoint endpoint/source revision changed")
        require(row["job"] in training["jobs"],
            "Original job outside bound training contract")
        bundle = registry["bundles"][row["evaluator_source_bundle"]]
        spec = next(d for d in bundle["datasets"] if d["dataset_id"] == row["dataset"])
        data = next(d for d in manifest["datasets"] if d["dataset"] == row["dataset"])
        origins = [spec.get(key, {}) for key in ("inherited_data_identity", "parent_data_identity")]
        matching = [origin for origin in origins if origin.get("data", {}).get("sha256") == data["sha256"]]
        require(matching and data["sha256"] == c["dataset_sha256"][row["dataset"]]
            and any(origin["split_manifest"]["sha256"] == data["split_manifest"]["sha256"] for origin in matching),
            "Data/split manifest differs from corresponding frozen identity")
        for split in ("validation", "test"):
            p = c["population_references"][row["dataset"]][split]
            require(p["loader"] == spec["loader"] == data["loader"]
                and p["expected_target_count"] == data["populations"][split]["target_count"]
                and p["data_file_sha256"] == data["sha256"], "Frozen split count/loader/data differs")
        require(spec["quantity_boundaries_all_train_rows"][-1] == TAILS[row["dataset"]]
            and spec["loader"]["batch_size"] == 128, "TRAIN tail boundary/batch size changed")
        if verify_bytes:
            # These operations hash bytes only. They never materialize a Test row.
            require(sha((root / data["path"]).resolve()) == data["sha256"], "Dataset bytes changed")
            require(sha((root / data["split_manifest"]["path"]).resolve()) == data["split_manifest"]["sha256"],
                "Split manifest bytes changed")
            require(verify_original(root, row, accepted) == c["validation_references"][identity(row)],
                "Frozen Validation reference changed")
    verify_comparison_gate(root, c, rows)
    verify_exposure_lineage(root, c)
    return root, c, registry


def verify_sufficient_statistics(metrics):
    n = metrics["count"]
    require(type(n) is int and n >= 0 and metrics["represented_n"] == n,
        "Full population aggregate count/represented mass differs")
    sums = ("qty_sse", "qty_absolute_error_sum", "qty_signed_error_sum", "time_nll_sum")
    require(all(isinstance(metrics[k], (int, float)) and not isinstance(metrics[k], bool)
        and math.isfinite(metrics[k]) for k in sums), "Nonfinite aggregate sufficient statistics")
    require(all(metrics[k] >= 0 for k in ("qty_sse", "qty_absolute_error_sum", "time_nll_sum")),
        "Negative aggregate sufficient statistics")
    derived = {"qty_rmse": math.sqrt(metrics["qty_sse"] / n) if n else None,
        "qty_mae": metrics["qty_absolute_error_sum"] / n if n else None,
        "qty_bias": metrics["qty_signed_error_sum"] / n if n else None,
        "time_nll": metrics["time_nll_sum"] / n if n else None}
    for key, expected in derived.items():
        actual = metrics[key]
        require((actual is None and expected is None) or (expected is not None
            and isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8)),
            "Aggregate metric does not derive from sufficient statistics: " + key)
    if not n:
        require(all(metrics[k] == 0 for k in sums), "Empty aggregate has nonzero sufficient statistics")


def check_receipt(path, row, split, c, campaign):
    r = read(path)
    require(r["status"] == "complete" and r["full_population"] and r["max_batches"] is None,
        "Incomplete full population receipt")
    require(r["retrained"] is False and r["raw_predictions_written"] is False and r["parts"] == []
        and r["persisted_row_fields"] == [] and not (path.parent / "predictions").exists(),
        "Training or raw prediction output is forbidden")
    require(r["split"] == split and r["heldout_read"] is (split == "test")
        and r["contract_sha256"] == sha(campaign / "execution_contract.json"), "Split/contract receipt differs")
    require(r["source_closure_sha256"] == row_source_closure(row)
        and r.get("evaluator_source_bundle") == row["evaluator_source_bundle"]
        and set(r["qualification"]) == QUALIFICATION_KEYS
        and all(v is True for v in r["qualification"].values()), "Source/causal qualification differs")
    for key in ("dataset", "model", "seed", "selected_epoch", "checkpoint_file_sha256", "state_tensor_sha256"):
        require(r[key] == row[key], "Receipt checkpoint identity differs: " + key)
    for key, ckey in (("runner_sha256", "evaluator_sha256"), ("registry_sha256", "registry_sha256"), ("dataset_manifest_sha256", "dataset_manifest_sha256")):
        require(r[key] == c[ckey], "Receipt input/source identity differs: " + key)
    p = c["population_references"][row["dataset"]][split]
    for key in ("target_identity_sha256", "truth_sha256", "loader", "data_file_sha256", "expected_target_count"):
        require(r[key] == p[key], "Frozen population/history identity differs: " + key)
    m = r["metrics"]
    registry_path = campaign / "evaluation_registry.json"
    require(sha(registry_path) == c["registry_sha256"], "Receipt scientific registry SHA changed")
    bundle = read(registry_path)["bundles"][row["evaluator_source_bundle"]]
    spec = next(d for d in bundle["datasets"] if d["dataset_id"] == row["dataset"])
    verify_sufficient_statistics(m)
    require(m["quantity_boundaries"] == spec["quantity_boundaries_all_train_rows"]
        and m["tail_definition"] == "raw_quantity > quantity_boundaries[-1]"
        and [x["bin"] for x in m["quantity_cells"]] == list(range(5)), "Full TRAIN quantity partition/order changed")
    for cell in m["quantity_cells"]:
        verify_sufficient_statistics(cell)
    require(m["count"] == r["prediction_rows"] == p["expected_target_count"]
        and all(math.isfinite(m[k]) for k in METRICS), "Count/nonfinite aggregate differs")
    require(m["quantity_boundaries"][-1] == TAILS[row["dataset"]]
        and sum(x["count"] for x in m["quantity_cells"]) == m["count"]
        and m["tail"] == {k: v for k, v in m["quantity_cells"][-1].items() if k != "bin"},
        "TRAIN tail partition changed")
    for key in ("qty_sse", "qty_absolute_error_sum", "qty_signed_error_sum", "time_nll_sum", "represented_n"):
        require(math.isclose(sum(x[key] for x in m["quantity_cells"]), m[key], rel_tol=1e-10, abs_tol=1e-8),
            "Aggregate strata do not reconstruct sufficient statistics: " + key)
    if split == "validation":
        reference = c["validation_references"][identity(row)]
        for current, prior in ((m, reference), (m["tail"], reference["tail"])):
            require(current["count"] == prior["count"], "Validation/tail count changed")
            for key in METRICS:
                require((current[key] is None and prior[key] is None) or (current[key] is not None
                    and prior[key] is not None and math.isclose(current[key], prior[key], rel_tol=1e-5, abs_tol=1e-5)),
                    "Validation endpoint replay differs: " + identity(row) + " " + key)
    runtime = r["runtime"]
    require(runtime["cpu_peak_rss_bytes"] <= c["resources"]["rss_limit_gib"] * 1024 ** 3
        and runtime["cuda_peak_reserved_bytes"] <= c["resources"]["gpu_memory_limit_gib"] * 1024 ** 3,
        "Native evaluation memory ceiling exceeded")
    return {"condition": identity(row), "split": split,
        "receipt_path": path.relative_to(campaign).as_posix(), "receipt_sha256": sha(path), "metrics": m,
        "target_identity_sha256": r["target_identity_sha256"], "truth_sha256": r["truth_sha256"]}


def verify_validation_gate(campaign, c, rows):
    gate = read(campaign / "validation_gate.json")
    require(gate.get("passed") is True and gate.get("conditions") == 9
        and gate.get("contract_sha256") == sha(campaign / "execution_contract.json"), "All Validation9 gates must pass before Test")
    entries = gate["rows"]
    require(len(entries) == 9 and {r["condition"] for r in entries} == {identity(r) for r in rows},
        "Global Validation gate does not cover exactly nine conditions")
    by_id = {identity(r): r for r in rows}
    for entry in entries:
        path = confined(campaign, entry["receipt_path"])
        require(entry["split"] == "validation" and sha(path) == entry["receipt_sha256"], "Validation gate receipt SHA changed")
        require(check_receipt(path, by_id[entry["condition"]], "validation", c, campaign) == entry,
            "Validation gate receipt content differs")
    return gate


class FrozenImportProxy:
    """Limit the prior evaluator's route substitution to one explicit request."""
    def __init__(self, delegate, c, root, row):
        self.delegate, self.c, self.root, self.row = delegate, c, root, row
        self.route_requests = 0

    def import_module(self, name, package=None):
        if name == "paper.scripts.run_titantpp_history_width":
            require(self.route_requests == 0, "Repeated/ambiguous model routing request")
            self.route_requests += 1
            engine = self.delegate.import_module("paper.scripts.run_titantpp_cnn_gru")
            runner = self.delegate.import_module("simple_lab_test.search.common.runner")
            payload = runner.torch_load_checkpoint(confined(self.root, self.row["checkpoint_path"]), map_location="cpu")
            require(payload["selected_metric_value"] == self.row["selected_metric_value"]
                and payload["model_state_sha256"] == self.row["state_tensor_sha256"], "Selected payload objective/state seal differs")
            return engine
        module = self.delegate.import_module(name, package)
        if name == "paper.scripts.quantity_comparison_runtime":
            def configure(device, threads):
                require(threads == self.c["resources"]["threads"], "Evaluator thread setting differs")
                module.configure_runtime(device, threads)
                actual = module.runtime_identity(device)
                for field, value in self.c["runtime_expected"].items():
                    require(actual[field] == value, "Native numerical runtime differs: " + field)
                if device.startswith("cuda"):
                    require(actual["gpu"]["uuid"] == self.c["resources"]["gpu_uuid"], "Wrong native GPU UUID")
                    limit = self.c["resources"]["gpu_memory_limit_gib"] * 1024 ** 3
                    fraction = min(self.c["resources"]["max_device_memory_fraction"], limit / actual["gpu"]["total_memory_bytes"])
                    module.torch.cuda.set_per_process_memory_fraction(fraction, 0)
            return SimpleNamespace(configure_runtime=configure, runtime_identity=module.runtime_identity)
        return module


def prior_evaluator(root, c):
    path = confined(root, c["prior_evaluator"]["path"])
    require(sha(path) == PRIOR_SHA, "Prior evaluator changed")
    spec = importlib.util.spec_from_file_location("_frozen_cnn_gru_aggregate_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def process_command(pid):
    return Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").decode().split("\0")


def process_parent_pid(pid):
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])


def verify_worker_launch(campaign, condition, split, c, *, token=None, process_reader=process_command,
                         process_parent_reader=process_parent_pid):
    """A public worker invocation cannot bypass the owning bounded pipeline."""
    token = os.environ.get("TITANTPP_EVAL_LAUNCH_TOKEN") if token is None else token
    require(isinstance(token, str) and len(token) == 64, "Worker requires an ephemeral pipeline launch token")
    permit = read(campaign / "launches" / f"{split}__{condition}.json")
    require(permit["condition"] == condition and permit["split"] == split
        and permit["contract_sha256"] == sha(campaign / "execution_contract.json")
        and secrets.compare_digest(permit["token_sha256"], hashlib.sha256(token.encode()).hexdigest()),
        "Worker launch permit/token binding differs")
    require(0 < permit["native_timeout_seconds"] <= c["resources"]["condition_timeout_seconds"]
        and time.monotonic() < permit["deadline_monotonic"], "Worker native launch deadline expired")
    # GNU timeout is the immediate parent; the live owning pipeline is one
    # level above it. Both exact command vectors are read from the kernel.
    native = process_reader(os.getppid())
    require(native == permit["native_command"], "Worker is not owned by the admitted native timeout")
    require(process_parent_reader(os.getppid()) == permit["pipeline_pid"],
        "Native timeout is not a child of the admitted owning pipeline")
    owner = process_reader(permit["pipeline_pid"])
    require(str(Path(__file__).resolve()) in owner and "--mode" in owner
        and owner[owner.index("--mode") + 1] == "pipeline" and "--campaign" in owner
        and Path(owner[owner.index("--campaign") + 1]).resolve() == campaign.resolve(),
        "Worker owning pipeline command differs")
    return permit


def worker(campaign, condition, split, device):
    root, c, registry = load_campaign(campaign)
    matches = [r for r in registry["rows"] if identity(r) == condition]
    require(len(matches) == 1, "Unapproved evaluation condition")
    row = matches[0]
    permit = verify_worker_launch(campaign, condition, split, c)
    if split == "test":
        verify_validation_gate(campaign, c, registry["rows"])
    os.environ.update({**c["environment"], "SOURCE_REVISION": row["source_revision"]})
    with (campaign / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        claim = campaign / "claims" / f"{split}__{condition}.json"
        write(claim, {"condition": condition, "split": split, "contract_sha256": sha(campaign / "execution_contract.json"),
            "pid": os.getpid(), "created_unix": time.time(), "automatic_retry": False,
            "pipeline_pid": permit["pipeline_pid"]}, exclusive=True)
        output = campaign / "runs" / split / condition
        prior = prior_evaluator(root, c)
        proxy = FrozenImportProxy(importlib, c, root, row)
        prior.importlib = proxy  # Only this loaded evaluator's namespace changes.
        args = SimpleNamespace(root=root, registry=campaign / "evaluation_registry.json",
            dataset_manifest=campaign / "dataset_manifest.json", dataset=row["dataset"], model=row["model"],
            seed=row["seed"], split=split, device=device, output=output,
            contract=campaign / "execution_contract.json", inventory_only=False, max_batches=None,
            deadline_seconds=min(permit["native_timeout_seconds"], permit["deadline_monotonic"] - time.monotonic()))
        require(args.deadline_seconds > 0, "Worker launch budget exhausted before inference")
        result = prior.run_worker(args)
        require(result == 0 and proxy.route_requests == 1, "Frozen evaluator did not install exactly one CNN/GRU route")
        check_receipt(output / "receipt.json", row, split, c, campaign)
        write(output / "operation_receipt.json", {"status": "complete", "condition": condition, "split": split,
            "adapter_sha256": sha(__file__), "prior_evaluator_sha256": PRIOR_SHA, "source_closure_sha256": row_source_closure(row),
            "evaluator_source_bundle": row["evaluator_source_bundle"],
            "model_route_module": "paper.scripts.run_titantpp_cnn_gru", "model_routing_requests": 1,
            "new_training": False, "selection_changed": False, "raw_predictions_written": False,
            "receipt_sha256": sha(output / "receipt.json"), "CPU_binary_audit_performed": False}, exclusive=True)


def stop_child(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def process_group_rss_bytes(pgid, *, proc_root=Path("/proc")):
    """Aggregate resident bytes for this owned child's complete process group."""
    require(proc_root.is_dir(), "Native /proc process-group RSS monitoring is required")
    total = 0
    for folder in proc_root.iterdir():
        if not folder.name.isdigit():
            continue
        try:
            fields = (folder / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) != pgid:
                continue
            rss = [line for line in (folder / "status").read_text().splitlines() if line.startswith("VmRSS:")]
            if rss:
                total += int(rss[0].split()[1]) * 1024
        except (FileNotFoundError, ProcessLookupError):
            continue  # Process exited between the two kernel reads.
    return total


def run_condition(campaign, c, row, split, device, deadline):
    timeout = shutil.which("timeout")
    require(timeout is not None, "Native GNU timeout is required")
    seconds = min(c["resources"]["condition_timeout_seconds"], deadline - time.monotonic())
    require(seconds > 0, "Campaign deadline exceeded")
    log = campaign / "logs" / f"{split}__{identity(row)}.log"
    log.parent.mkdir(exist_ok=True)
    command = [timeout, "--signal=TERM", "--kill-after=10s", str(seconds),
        c["resources"]["python"], str(Path(__file__).resolve()), "--campaign", str(campaign),
        "--mode", "worker", "--condition", identity(row), "--split", split, "--device", device]
    token = secrets.token_hex(32)
    launch_path = campaign / "launches" / f"{split}__{identity(row)}.json"
    write(launch_path, {"condition": identity(row), "split": split, "pipeline_pid": os.getpid(),
        "contract_sha256": sha(campaign / "execution_contract.json"),
        "token_sha256": hashlib.sha256(token.encode()).hexdigest(), "native_command": command,
        "native_timeout_seconds": seconds, "deadline_monotonic": time.monotonic() + seconds,
        "automatic_retry": False}, exclusive=True)
    # Native library loading observes the environment inherited at Python exec;
    # changing LD_LIBRARY_PATH inside the already-running worker is too late.
    environment = {**os.environ, **c["environment"], "SOURCE_REVISION": row["source_revision"],
        "TITANTPP_EVAL_LAUNCH_TOKEN": token}
    process, max_group_rss = None, 0
    try:
        with log.open("x") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, env=environment)
            while process.poll() is None:
                require(time.monotonic() < deadline, "Campaign deadline exceeded")
                max_group_rss = max(max_group_rss, process_group_rss_bytes(process.pid))
                require(max_group_rss <= c["resources"]["rss_limit_gib"] * 1024 ** 3, "Native child process-group RSS ceiling exceeded")
                time.sleep(.25)
            returncode = process.returncode
        require(returncode == 0, "Evaluation failed; preserve and inspect " + str(log))
    except BaseException as error:
        write(campaign / "failures" / f"{split}__{identity(row)}.json", {"status": "failed",
            "condition": identity(row), "split": split, "exception": type(error).__name__, "message": str(error),
            "owned_process_group": process.pid if process is not None else None,
            "peak_observed_process_group_rss_bytes": max_group_rss, "automatic_retry": False,
            "raw_predictions_written": False}, exclusive=True)
        raise
    finally:
        if process is not None:
            stop_child(process)
    return check_receipt(campaign / "runs" / split / identity(row) / "receipt.json", row, split, c, campaign)


def records(rows, completed, c, manifest):
    by_id = {(r["condition"], r["split"]): r for r in completed}
    return [{"job": row["job"], "validation": {"selected": by_id[(identity(row), "validation")]["metrics"]},
        "test": by_id[(identity(row), "test")]["metrics"],
        "checkpoint_file_sha256": row["checkpoint_file_sha256"], "state_tensor_sha256": row["state_tensor_sha256"],
        "selected_epoch": row["selected_epoch"], "selected_metric_value": row["selected_metric_value"],
        "source_closure_sha256": row_source_closure(row), "source_revision": row["source_revision"],
        "evaluator_source_bundle": row["evaluator_source_bundle"],
        "training_contract_sha256": row["original_binding"]["training_contract_sha256"],
        "terminal_manifest_sha256": row["original_binding"]["terminal_manifest_sha256"],
        "data_file_sha256": c["dataset_sha256"][row["dataset"]],
        "split_manifest_sha256": next(d["split_manifest"]["sha256"] for d in manifest["datasets"] if d["dataset"] == row["dataset"]),
        "split_population": {split: {key: by_id[(identity(row), split)][key]
            for key in ("receipt_path", "receipt_sha256", "target_identity_sha256", "truth_sha256")}
            for split in ("validation", "test")}} for row in rows]


def pipeline(campaign, device, *, runner=run_condition):
    root, c, registry = load_campaign(campaign)
    require(not (campaign / "pipeline_status.json").exists(), "Existing attempt preserved; automatic retry forbidden")
    completed, started = [], time.monotonic()
    with (campaign / "campaign.lock").open("x") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for split in ("validation", "test"):
                if split == "test":
                    verify_validation_gate(campaign, c, registry["rows"])
                for row in registry["rows"]:
                    write(campaign / "pipeline_status.json", {"status": "running", "split": split,
                        "condition": identity(row), "completed": completed, "pid": os.getpid(), "automatic_retry": False})
                    completed.append(runner(campaign, c, row, split, device, started + c["resources"]["campaign_timeout_seconds"]))
                    print(json.dumps({"completed": len(completed), "split": split, "condition": identity(row)}), flush=True)
                if split == "validation":
                    write(campaign / "validation_gate.json", {"passed": True, "conditions": 9,
                        "contract_sha256": sha(campaign / "execution_contract.json"), "rows": completed.copy(),
                        "completed_before_test_unix": time.time()}, exclusive=True)
            result = {"status": "complete", "records": records(registry["rows"], completed, c, read(campaign / "dataset_manifest.json")), "completed": completed,
                "conditions": 9, "full_population_splits": 18, "contract_sha256": sha(campaign / "execution_contract.json"),
                "source_closure_sha256": SOURCE_SHA, "elapsed_seconds": time.monotonic() - started,
                "new_training": False, "selection_changed": False, "raw_predictions_written": False,
                "automatic_retry": False, "CPU_binary_audit_performed": False,
                "test_exposure_lineage": deepcopy(c["test_exposure_lineage"]),
                "final_validation_comparison_gate": deepcopy(c["final_validation_comparison_gate"]),
                "test_interpretation": "Previously accessed frozen Test splits; fixed original quantity-selected CNNGRU three-seed follow-up"}
            write(campaign / "inference_completion.json", result, exclusive=True)
            write(campaign / "pipeline_status.json", result)
            return result
        except BaseException as error:
            write(campaign / "pipeline_status.json", {"status": "failed", "completed": completed,
                "exception": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(),
                "automatic_retry": False, "elapsed_seconds": time.monotonic() - started})
            raise


def interrupted(signum, frame):
    raise InterruptedError("Evaluation received termination signal " + str(signum))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--mode", choices=("inspect", "pipeline", "worker"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--condition")
    parser.add_argument("--split", choices=("validation", "test"))
    args = parser.parse_args(argv)
    args.campaign = args.campaign.resolve()
    signal.signal(signal.SIGTERM, interrupted)
    if args.mode == "inspect":
        root, c, registry = load_campaign(args.campaign)
        print(json.dumps({"status": "ready", "conditions": len(registry["rows"]), "root": str(root), "inference_calls": 0}))
    elif args.mode == "worker":
        require(args.condition is not None and args.split is not None, "Worker requires condition and split")
        worker(args.campaign, args.condition, args.split, args.device)
    else:
        require(args.condition is None and args.split is None, "Pipeline is the complete nine-condition batch")
        pipeline(args.campaign, args.device)


if __name__ == "__main__":
    main()
