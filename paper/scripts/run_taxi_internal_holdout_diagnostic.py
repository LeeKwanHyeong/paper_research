"""One fit-only Taxi single-MLP diagnostic; no Test route or automatic resume.

The established trainer/model/target function are reused unchanged.  Instrumentation
adds same-state eval-mode fit metrics and retains a predeclared epoch grid.  The
original Validation is reached only after internal selection has been sealed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Subset

SCHEMA = "taxi_internal_holdout_diagnostic_v1"
GRID = (1, 5, 10, 20, *range(40, 301, 20))
COUNTS = {"fit": 23108, "dev": 7536, "audit": 7487, "purged": 262, "validation": 8268}
META_FIELDS = ("split", "target_index", "entity_id", "target_seq", "raw_quantity",
               "target_gap", "history_length", "origin_time", "label_time")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def audit_epochs(completed_epoch, best_epoch):
    require(type(completed_epoch) is int and 1 <= completed_epoch <= 300, "Invalid completed epoch")
    require(type(best_epoch) is int and 1 <= best_epoch <= completed_epoch, "Invalid selected epoch")
    return sorted({e for e in GRID if e <= completed_epoch} | {best_epoch, completed_epoch})


def authorized_evaluation_pairs(completed_epoch, best_epoch):
    endpoints = sorted({best_epoch, completed_epoch})
    return {"fit": endpoints, "dev": endpoints,
            "audit": audit_epochs(completed_epoch, best_epoch), "validation": endpoints}


def partition_rows(rows):
    require(bool(rows), "Empty Train metadata")
    output = {key: [] for key in ("fit", "dev", "audit", "purged")}
    seen, indices = set(), set()
    for row in rows:
        require(all(key in row for key in META_FIELDS), "Missing metadata field")
        require(row["split"] == "train", "Only Train metadata can be partitioned; Test is locked")
        key = (str(row["entity_id"]), int(row["target_seq"]))
        require(key not in seen and row["target_index"] not in indices, "Duplicate target identity")
        seen.add(key)
        indices.add(row["target_index"])
        origin, label = row["origin_time"], row["label_time"]
        require(all(isinstance(v, (int, float)) and math.isfinite(v) and v == int(v)
                    for v in (origin, label, row["target_gap"])), "Noninteger recorded time")
        require(origin < label and row["target_gap"] >= 1 and label-origin == row["target_gap"],
                "Invalid target chronology")
        if label <= 310:
            stage = "fit"
        elif label <= 411:
            stage = "dev" if origin > 310 else "purged"
        else:
            stage = "audit" if origin > 411 else "purged"
        output[stage].append(dict(row))
    require(indices == set(range(len(rows))), "Missing or noncanonical target index")
    return output


def _context_indices(dataset, index):
    part, end = dataset.index[index]
    seq = np.asarray(dataset.seq_lists[part])
    start = int(np.searchsorted(seq, seq[end] - (dataset.W - 1), side="left"))
    start = max(start, end + 2 - dataset.max_len)
    return part, end, range(start, end + 1)


def validate_metadata(dataset, rows, split):
    require(split in ("train", "validation"), "Test is locked")
    require(len(rows) == len(dataset), "Metadata/dataset size mismatch")
    for number, row in enumerate(rows):
        require(row["split"] == split and row["target_index"] == number, "Metadata order/split changed")
        part, end, context = _context_indices(dataset, number)
        seq = dataset.seq_lists[part]
        require(str(dataset.parts[part]) == str(row["entity_id"]) and seq[end+1] == row["target_seq"],
                "Target identity mismatch")
        require(seq[end] == row["origin_time"] and seq[end+1] == row["label_time"],
                "Recorded chronology differs from loader sequence coordinate")
        require(float(dataset.val_lists[part][end+1]) == row["raw_quantity"]
                and max(1., float(dataset.dt_lists[part][end+1])) == row["target_gap"],
                "Target value mismatch")
        require(len(context) == row["history_length"], "Observed history length mismatch")
    return {"split": split, "targets": len(rows), "all_target_fields_equal": True}


def leakage_audit(dataset, fit_indices, heldout_keys):
    heldout_keys = {(str(part), int(seq)) for part, seq in heldout_keys}
    exposed, target_keys = set(), set()
    for index in fit_indices:
        part, end, context = _context_indices(dataset, int(index))
        entity, sequences = str(dataset.parts[part]), dataset.seq_lists[part]
        target = (entity, int(sequences[end+1]))
        require(target not in heldout_keys, "Heldout target used for fitting")
        target_keys.add(target)
        for position in context:
            key = (entity, int(sequences[position]))
            require(key not in heldout_keys, "Heldout target entered fitting context")
            exposed.add(key)
    require(bool(target_keys), "No fit targets")
    return {"fit_target_count": len(target_keys), "fit_context_unique_events": len(exposed),
            "heldout_target_overlap": 0, "heldout_context_overlap": 0,
            "fit_context_identity_sha256": hashlib.sha256(json.dumps(sorted(exposed)).encode()).hexdigest()}


@contextmanager
def preserved_rng(model=None, generators=()):
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    generator_states = [(g, g.get_state()) for g in generators if g is not None]
    modes = [(module, module.training) for module in model.modules()] if model is not None else []
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        for generator, state in generator_states:
            generator.set_state(state)
        for module, mode in modes:
            module.training = mode


def evaluate_preserving_rng(evaluate, *, generators=(), **kwargs):
    """Keep both added fit evaluation and native dev evaluation observational."""
    loader_generator = getattr(kwargs["loader"], "generator", None)
    with preserved_rng(kwargs["model"], (*generators, loader_generator)):
        result = evaluate(**kwargs)
        for key in ("val_joint_objective", "val_time_nll", "val_log_qty_mse", "qty_rmse", "qty_mae"):
            require(key in result and math.isfinite(float(result[key])), "Nonfinite/missing evaluation metric: " + key)
        require(result.get("evaluated_count", 0) > 0, "Empty evaluation population")
        return result


def _subset_loader(dataset, indices, *, shuffle=False, generator=None):
    from data_loader.event_seq_data_module import collate_week_lookback
    return DataLoader(Subset(dataset, list(indices)), batch_size=128, shuffle=shuffle,
                      generator=generator, num_workers=0, collate_fn=collate_week_lookback)


def _loader(frame, split):
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    return make_loader(frame, target_split=split, batch_size=128, lookback_weeks=168,
                       max_seq_len=256, shuffle=False, generator=None)


def safe_load_input(args):
    from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
    frame = pl.read_parquet(args.input_frame)
    require(set(frame["chronological_split"].unique().to_list()) == {"train", "validation"},
            "Input must be Train/Validation only; Test is locked")
    return prepare_count_frame(frame.sort(["oper_part_no", "seq"]))


def validate_context_only_grid(frame, dataset, expected_context_keys, native_validator):
    """Validate purged rows as observed context without making them targets."""
    require(dataset == "yellow_trip_hourly", "Internal holdout is Taxi only")
    labels = set(frame["chronological_split"].unique().to_list())
    require(labels <= {"train", "validation", "context_only"}, "Test/foreign split is locked")
    require(frame.height > 0 and frame["seq"].max() <= 411, "Audit/future rows entered the fit/dev frame")
    require(frame.filter((pl.col("chronological_split") == "train") & (pl.col("seq") > 310)).height == 0,
            "Heldout label entered fit rows")
    require(frame.filter((pl.col("chronological_split") == "validation") & (pl.col("seq") <= 310)).height == 0,
            "Fit label entered internal dev rows")
    contexts = frame.filter(pl.col("chronological_split") == "context_only")
    keys = {(str(row["oper_part_no"]), int(row["seq"])) for row in contexts.select("oper_part_no", "seq").to_dicts()}
    expected = {(str(entity), int(seq)) for entity, seq in expected_context_keys}
    require(keys == expected and len(keys) == contexts.height, "Context-only rows are not the frozen purged targets")
    require(contexts.filter((pl.col("seq") <= 310) | ((pl.col("seq") - pl.col("delta_t")) > 310)).height == 0,
            "Context-only row does not cross the frozen fit/dev boundary")
    # The shadow exists only for the existing integer-grid validator. No loader,
    # statistics function, model or metric consumes its relabeled split column.
    shadow = frame.with_columns(pl.when(pl.col("chronological_split") == "context_only")
                               .then(pl.lit("validation")).otherwise(pl.col("chronological_split"))
                               .alias("chronological_split"))
    return native_validator(shadow, dataset)


@contextmanager
def internal_grid_guard(expected_context_keys):
    from paper.scripts.count_aware_tpp_backbone import observed_time, training
    original_grid, original_training_grid = observed_time.validate_recorded_grid, training.validate_recorded_grid

    def guarded(frame, dataset):
        return validate_context_only_grid(frame, dataset, expected_context_keys, original_grid)

    observed_time.validate_recorded_grid = training.validate_recorded_grid = guarded
    try:
        yield
    finally:
        observed_time.validate_recorded_grid, training.validate_recorded_grid = original_grid, original_training_grid


def _check_spec(args):
    spec = read_json(args.science_spec)
    expected = {"schema": SCHEMA, "dataset": "yellow_trip_hourly", "seed": 42,
                "test_locked": True, "fit_boundary": 310, "dev_boundary": 411}
    for key, value in expected.items():
        require(spec.get(key) == value, "Science specification mismatch: " + key)
    require(spec.get("expected_counts") == COUNTS, "Expected population sizes changed")
    require(spec.get("audit_epoch_grid") == list(GRID), "Audit epoch grid changed")
    require(all(spec.get("training", {}).get(key) == value for key, value in
                {"max_epochs": 300, "min_epochs": 40, "patience": 40, "batch_size": 128,
                 "lr": .001, "weight_decay": .01, "grad_clip": 1., "optimizer": "AdamW",
                 "lookback_hours": 168, "max_seq_len": 256, "quantity_weight": 1.,
                 "time_weight": 1., "fixed_time_scale": 1., "time_sigma_floor": .001,
                 "fit_only_initialization_statistics": True,
                 "monitor": "internal_dev_raw_quantity_rmse", "tie_rule": "earliest_strict_minimum"}.items()),
            "Training budget changed")
    group_spec = spec.get("groups", {})
    require(group_spec.get("history_edges") == [1, 2, 4, 8, 16, 32, 64, 128, 256]
            and group_spec.get("minimum_fit_events_per_cell") == 30
            and group_spec.get("minimum_fit_entities_per_cell") == 10
            and group_spec.get("minimum_report_events") == 30
            and group_spec.get("minimum_report_entities") == 10
            and group_spec.get("bootstrap") is False, "Support/reporting criteria changed")
    return spec


def _inputs(args):
    return {key: {"sha256": sha(getattr(args, key)), "bytes": Path(getattr(args, key)).stat().st_size}
            for key in ("source_contract", "input_frame", "train_metadata", "validation_metadata", "science_spec")}


def _features(dataset, rows):
    output = []
    for row in rows:
        part, end, context = _context_indices(dataset, row["target_index"])
        q = dataset.val_lists[part]
        previous, older = float(q[end]), float(q[end-1]) if len(context) >= 2 else None
        change = previous-older if older is not None else None
        output.append({**row, "previous_quantity": previous,
                       "previous_gap": max(1., float(dataset.dt_lists[part][end])),
                       "previous_change": change,
                       "log_previous_quantity": math.log1p(previous),
                       "log_previous_gap": math.log1p(max(1., float(dataset.dt_lists[part][end]))),
                       "log_quantity_change": math.log1p(previous)-math.log1p(older) if older is not None else None})
    return output


def _support_contract(fit_rows):
    boundaries = {}
    for key in ("log_previous_quantity", "log_previous_gap", "log_quantity_change"):
        values = np.asarray([r[key] for r in fit_rows if r[key] is not None], dtype=np.float64)
        require(len(values) > 0 and np.isfinite(values).all(), "Invalid fit feature")
        boundaries[key] = sorted(set(np.quantile(values, [.25, .5, .75], method="linear").tolist()))
    boundaries["history_length"] = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    contract = {"boundaries": boundaries, "bin_side": "left", "minimum_events": 30,
                "minimum_entities": 10, "bootstrap": False,
                "fit_max_history_length": max(r["history_length"] for r in fit_rows),
                "missing_previous_change_bin": -1}
    contract["fit_observed_position_event_counts"] = [
        sum(row["history_length"] > position for row in fit_rows) for position in range(256)]
    contract["position_count_scope"] = "zero-based observed positions only; target and padding excluded"
    counts = {}
    for row in fit_rows:
        key = _cell(row, contract)
        counts.setdefault(key, []).append(str(row["entity_id"]))
    contract["fit_cells"] = {key: {"events": len(values), "entities": len(set(values)),
                                   "supported": len(values) >= 30 and len(set(values)) >= 10}
                              for key, values in sorted(counts.items())}
    return contract


def _cell(row, contract):
    ids = []
    for key, boundaries in contract["boundaries"].items():
        ids.append(-1 if row[key] is None else int(np.searchsorted(boundaries, row[key], side="left")))
    return ":".join(map(str, ids))


def prepare(args):
    from paper.scripts import run_shared_history_mlp_control as old
    from paper.scripts.run_count_aware_tpp_backbone_control import derive_train_time_contract
    from paper.scripts.run_taxi_quantity_interface_ablation import train_quantile_contract
    from paper.scripts.run_hard_lmm_frozen_lognormal_duration import target_dt_sha256
    _check_spec(args)
    out = args.root / "prepared"
    require(not out.exists(), "Prepared evidence already exists; no overwrite")
    source = read_json(args.source_contract)
    data = deepcopy(next(d for d in source["datasets"] if d["dataset_id"] == "yellow_trip_hourly"))
    frame = safe_load_input(args)
    require(sha(args.input_frame) == data["inherited_data_identity"]["data"]["sha256"], "Frozen input SHA changed")
    train_rows = pl.read_parquet(args.train_metadata).select(META_FIELDS).to_dicts()
    val_rows = pl.read_parquet(args.validation_metadata).select(META_FIELDS).to_dicts()
    train_dataset, val_dataset = _loader(frame, "train").dataset, _loader(frame, "validation").dataset
    checks = [validate_metadata(train_dataset, train_rows, "train"), validate_metadata(val_dataset, val_rows, "validation")]
    groups = partition_rows(train_rows)
    groups["validation"] = val_rows
    require({key: len(value) for key, value in groups.items()} == COUNTS, "Frozen cohort size mismatch")
    heldout = {(str(r["entity_id"]), int(r["target_seq"]))
               for name in ("dev", "audit", "purged", "validation") for r in groups[name]}
    leak = leakage_audit(train_dataset, [r["target_index"] for r in groups["fit"]], heldout)
    target_stage = {(r["entity_id"], r["target_seq"]): name for name in ("fit", "dev") for r in groups[name]}
    rows = []
    for row in frame.filter((pl.col("chronological_split") == "train") & (pl.col("seq") <= 411)).to_dicts():
        stage = target_stage.get((row["oper_part_no"], row["seq"]))
        # First events have no next-event target row but remain observed fit context.
        row["chronological_split"] = "train" if row["seq"] <= 310 else ("validation" if stage == "dev" else "context_only")
        rows.append(row)
    fitdev = pl.DataFrame(rows, schema=frame.schema)
    fit_native, dev_native = _loader(fitdev, "train").dataset, _loader(fitdev, "validation").dataset
    for name, dataset in (("fit", fit_native), ("dev", dev_native)):
        identities = [(str(dataset.parts[p]), dataset.seq_lists[p][i+1]) for p, i in dataset.index]
        require(identities == [(str(r["entity_id"]), r["target_seq"]) for r in groups[name]], "Relabeling altered cohort order")
        for offset, row in enumerate(groups[name]):
            original = train_dataset[row["target_index"]]
            relabeled = dataset[offset]
            require(original.keys() == relabeled.keys() and
                    all(torch.equal(original[key], relabeled[key]) for key in original),
                    "Relabeling altered model input")
    fit_raw = fitdev.filter(pl.col("chronological_split") == "train")
    quantities = fit_raw["demand_qty"].to_numpy().astype(np.float64)
    logs = np.log1p(quantities)
    times = derive_train_time_contract(fitdev, lookback_weeks=168, max_seq_len=256)
    require(times["time_scale"] == 1., "Fit median duration changed fixed time scale")
    data["statistics"] = {"train_log_mean": float(logs.mean()), "train_log_std": float(logs.std()),
                          "raw_scale": float(quantities.mean()), "initialization_policy": "fit_only_raw_rows",
                          "raw_row_count": len(quantities)}
    data["mu_all_train_rows"] = data["statistics"]["train_log_mean"]
    data["steps_per_epoch"] = math.ceil(len(fit_native) / 128)
    data["expected_global_steps"] = data["steps_per_epoch"] * 300
    data["model"]["time_initial_location"] = times["target_log_scaled_mean"]
    data["model"]["time_initial_scale"] = times["target_log_scaled_std"]
    quantity_contract = train_quantile_contract(fitdev)
    data["quantity_boundaries_all_train_rows"] = quantity_contract["boundaries"]
    data["time_statistics"] = {"train_time_scale": 1., "train_log_scaled_mean": times["target_log_scaled_mean"],
                               "train_log_scaled_std": times["target_log_scaled_std"]}
    for name, dataset in (("train", fit_native), ("validation", dev_native)):
        values = torch.tensor([max(1., dataset.dt_lists[p][i+1]) for p, i in dataset.index], dtype=torch.float64)
        data["time_statistics"][f"expected_{name}_targets"] = len(dataset)
        data["time_statistics"][f"expected_{name}_target_dt_sha256"] = target_dt_sha256(values)
    data.pop("full_data_identity", None)
    data["statistics_source_scope"] = "fit_only"
    data["parent_execution_contract_sha256"] = sha(args.source_contract)
    for name in groups:
        groups[name] = _features(val_dataset if name == "validation" else train_dataset, groups[name])
    support = _support_contract(groups["fit"])
    for rows in groups.values():
        for row in rows:
            row["condition_cell"] = _cell(row, support)
            row["fit_supported_cell"] = support["fit_cells"].get(row["condition_cell"], {}).get("supported", False)
            row["within_fit_position_range"] = row["history_length"] <= support["fit_max_history_length"]
    out.mkdir(parents=True, exist_ok=False)
    fitdev.write_parquet(out / "fit_dev_frame.parquet")
    for name, rows in groups.items():
        pl.DataFrame(rows).write_parquet(out / f"{name}_metadata.parquet")
    data["inherited_data_identity"]["data"] = {"path": str(out / "fit_dev_frame.parquet"), "sha256": sha(out / "fit_dev_frame.parquet")}
    data["data_sha256"] = sha(out / "fit_dev_frame.parquet")
    data["prepared_train_validation_path"] = str(out / "fit_dev_frame.parquet")
    data["inherited_data_identity"]["populations"] = {
        split: old.common.population_identity(dataset, split)[1]
        for split, dataset in (("train", fit_native), ("validation", dev_native))}
    data["inherited_data_identity"]["split_manifest"] = {"path": str(out / "SPLIT.json"), "sha256": "pending"}
    write_json(out / "SPLIT.json", {"scope": "internal_train_holdout", "counts": COUNTS, "fit_boundary": 310,
                                   "dev_boundary": 411, "test_locked": True, "checks": checks, "leakage": leak})
    data["inherited_data_identity"]["split_manifest"]["sha256"] = sha(out / "SPLIT.json")
    write_json(out / "DATA.json", data)
    write_json(out / "QUANTITY_CONTRACT.json", quantity_contract)
    write_json(out / "SUPPORT.json", support)
    context_keys = {(str(row["entity_id"]), int(row["target_seq"]))
                    for row in groups["purged"] if row["label_time"] <= 411}
    with internal_grid_guard(context_keys):
        interface = old._training_interface(source, data, fitdev, "single_mlp")
    interface.update(diagnostic_schema=SCHEMA, statistics_source_scope="fit_only", selection_scope="internal_dev")
    write_json(out / "INTERFACE.json", interface)
    write_json(out / "INPUTS.json", _inputs(args))
    write_json(out / "INVENTORY.json", {path.name: {"sha256": sha(path), "bytes": path.stat().st_size}
                                        for path in sorted(out.iterdir()) if path.is_file()})
    return {"prepared": str(out), "counts": COUNTS, "leakage": leak, "fit_statistics": data["statistics"]}


def _prepared(args):
    _check_spec(args)
    folder = args.root / "prepared"
    require(read_json(folder / "INPUTS.json") == _inputs(args), "Prepared input identity changed")
    inventory = read_json(folder / "INVENTORY.json")
    expected = {"fit_dev_frame.parquet", "DATA.json", "SPLIT.json", "QUANTITY_CONTRACT.json",
                "SUPPORT.json", "INTERFACE.json", "INPUTS.json"} | {f"{group}_metadata.parquet" for group in COUNTS}
    require(set(inventory) == expected, "Prepared inventory incomplete or expanded")
    for name, identity in inventory.items():
        path = folder / name
        require(sha(path) == identity["sha256"] and path.stat().st_size == identity["bytes"], "Prepared artifact changed: " + name)
    return folder


def aggregate_events(rows):
    require(bool(rows), "Empty evaluation")
    for key in ("true_qty", "pred_qty", "log_qty_mse", "time_nll", "joint_objective"):
        require(all(math.isfinite(float(r[key])) for r in rows), "Nonfinite evaluation: " + key)
    errors = np.asarray([r["pred_qty"]-r["true_qty"] for r in rows], dtype=np.float64)
    return {"count": len(rows), "qty_rmse": float(np.sqrt(np.mean(errors**2))),
            "qty_mae": float(np.abs(errors).mean()),
            **{key: float(np.mean([r[key] for r in rows])) for key in ("log_qty_mse", "time_nll", "joint_objective")}}


def group_event_metrics(rows):
    """Descriptive, frozen-support summaries; empty or small groups stay explicit."""
    definitions = {
        "all": rows,
        "fit_cell_supported": [r for r in rows if r["fit_supported_cell"]],
        "fit_cell_unsupported": [r for r in rows if not r["fit_supported_cell"]],
        "position_supported": [r for r in rows if r["within_fit_position_range"]],
        "position_unsupported": [r for r in rows if not r["within_fit_position_range"]],
        "common_supported": [r for r in rows if r["fit_supported_cell"] and r["within_fit_position_range"]],
        "outside_common_support": [r for r in rows if not (r["fit_supported_cell"] and r["within_fit_position_range"])],
    }
    for row in rows:
        definitions.setdefault("cell:" + row["condition_cell"], []).append(row)
    summaries = []
    for group, members in definitions.items():
        entities = len({str(row["entity_id"]) for row in members})
        eligible = len(members) >= 30 and entities >= 10
        metrics = aggregate_events(members) if eligible else {
            "count": len(members), "qty_rmse": None, "qty_mae": None,
            "log_qty_mse": None, "time_nll": None, "joint_objective": None}
        summaries.append({"group": group, "entities": entities, "report_eligible": eligible,
                          "reason": "descriptive" if eligible else "below_30_events_or_10_entities",
                          **metrics})
    return summaries


@torch.no_grad()
def _evaluate_events(model, loader, metadata, device):
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    result, offset = [], 0
    with preserved_rng(model, (getattr(loader, "generator", None),)):
        model.eval()
        for _, dts, mask, _, quantities in loader:
            output = target_outputs(model, dts.to(device), mask.to(device), quantities.to(device), lambda_log_qty=1.)
            arrays = {key: value.detach().cpu().double().numpy() for key, value in output.items()}
            count = len(arrays["true_qty"])
            for index in range(count):
                row = dict(metadata[offset+index])
                require(float(arrays["true_qty"][index]) == row["raw_quantity"], "Evaluation target order changed")
                require(int(arrays["history_length"][index]) == row["history_length"], "Evaluation history changed")
                row.update(true_qty=float(arrays["true_qty"][index]), pred_qty=float(arrays["pred_qty"][index]),
                           log_qty_mse=float(arrays["log_qty_loss"][index]), time_nll=float(arrays["time_loss"][index]),
                           joint_objective=float(arrays["joint_loss"][index]))
                result.append(row)
            offset += count
    require(offset == len(metadata), "Incomplete event evaluation")
    return result, aggregate_events(result)


def _plot_curves(path, rows, best, last):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, metric, label in zip(axes.flat, ("qty_rmse", "qty_mae", "log_qty_mse", "time_nll"),
                                  ("Quantity RMSE", "Quantity MAE", "Log-quantity MSE", "Recorded-time NLL")):
        for split, color in (("fit", "#2864a3"), ("dev", "#de8f05"), ("audit", "#3b8c63"), ("validation", "#a64f83")):
            values = sorted((r for r in rows if r["split"] == split), key=lambda r: r["epoch"])
            ax.plot([r["epoch"] for r in values], [r[metric] for r in values],
                    label=split, color=color, marker="o" if split in ("audit", "validation") else None,
                    linestyle="none" if split == "validation" else "-")
        ax.axvline(best, color="gray", linestyle="--", linewidth=.8)
        ax.set(xlabel="Epoch", ylabel=label)
        ax.grid(alpha=.2)
    axes.flat[0].legend(frameon=False)
    fig.suptitle(f"Taxi single MLP, seed 42 — same-checkpoint evaluation\nInternal dev selection: {best}; last: {last}. Test locked.")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _fit_once(args):
    from paper.scripts import run_shared_history_mlp_control as old
    from paper.scripts.count_aware_tpp_backbone import core, training
    from simple_lab_test.search.common.runner import torch_load_checkpoint, canonical_state_dict_sha256
    prepared = _prepared(args)
    before = _inputs(args)
    run = args.root / "fit"
    require(not run.exists() and not (args.root / "FIT_CLAIM.json").exists(), "Fit attempt already exists; no resume/retry")
    write_json(args.root / "FIT_CLAIM.json", {"schema": SCHEMA, "seed": 42, "started_unix": time.time(),
                                           "science_sha256": sha(args.science_spec), "test_locked": True})
    run.mkdir(exist_ok=False)
    source, data = read_json(args.source_contract), read_json(prepared / "DATA.json")
    frame = pl.read_parquet(prepared / "fit_dev_frame.parquet")
    quantity, interface = read_json(prepared / "QUANTITY_CONTRACT.json"), read_json(prepared / "INTERFACE.json")
    data["inherited_data_identity"]["data"]["path"] = str(prepared / "fit_dev_frame.parquet")
    data["inherited_data_identity"]["split_manifest"]["path"] = str(prepared / "SPLIT.json")
    data["prepared_train_validation_path"] = str(prepared / "fit_dev_frame.parquet")
    fit_loader = _loader(frame, "train")
    snapshots = run / "snapshots"
    snapshots.mkdir()
    original_epoch, original_loader, original_evaluate = training.train_epoch_with_telemetry, training.make_loader, training.evaluate
    epoch_counter = 0
    captured_generators = []
    curves = []
    purged = pl.read_parquet(prepared / "purged_metadata.parquet").to_dicts()
    context_keys = {(str(row["entity_id"]), int(row["target_seq"])) for row in purged if row["label_time"] <= 411}
    try:
        with internal_grid_guard(context_keys), old._engine_hooks(data, "single_mlp"):
            original_save = training.atomic_torch_save

            def make_loader(*pos, **kwargs):
                generator = kwargs.get("generator")
                if generator is not None:
                    captured_generators.append(generator)
                return original_loader(*pos, **kwargs)

            def train_epoch(**kwargs):
                nonlocal epoch_counter
                telemetry = original_epoch(**kwargs)
                epoch_counter += 1
                measured = evaluate_preserving_rng(core.evaluate, generators=captured_generators,
                                                   model=kwargs["model"], loader=fit_loader, quantity_contract=quantity,
                                                   device=args.device, lambda_log_qty=1., max_batches=None, include_breakdowns=False)
                telemetry.update({"train_eval_" + key: value for key, value in measured.items()})
                return telemetry

            def evaluate(**kwargs):
                return evaluate_preserving_rng(original_evaluate, generators=captured_generators, **kwargs)

            def save(payload, path):
                original_save(payload, path)
                if payload.get("checkpoint_type") == "epoch_resume" and payload["epoch"] in GRID:
                    destination = snapshots / f"epoch_{payload['epoch']:03d}.pt"
                    require(not destination.exists(), "Snapshot already exists")
                    original_save(payload, destination)

            training.make_loader, training.train_epoch_with_telemetry, training.atomic_torch_save = make_loader, train_epoch, save
            training.evaluate = evaluate
            try:
                training_args = old._training_args(source, data, run, "single_mlp")
                training_args.device = args.device
                training_args.execution_role = SCHEMA
                training_args.source_revision = os.environ.get("SOURCE_REVISION", source["source"]["base_git_revision"])
                require(len(training_args.source_revision) == 40 and all(
                    character in "0123456789abcdef" for character in training_args.source_revision),
                    "Invalid diagnostic runner source revision")
                interface["base_source_revision"] = source["source"]["base_git_revision"]
                interface["diagnostic_runner_revision"] = training_args.source_revision
                interface["diagnostic_runner_revision_source"] = (
                    "SOURCE_REVISION_environment" if "SOURCE_REVISION" in os.environ else "frozen_parent_fallback")
                summary, _, _ = training.train_one(args=training_args, frame=frame, quantity_contract=quantity,
                                                   interface_meta=interface, backbone="single_mlp",
                                                   quantity_variant="count_only_log_regression", seed=42)
            finally:
                training.make_loader, training.train_epoch_with_telemetry = original_loader, original_epoch
                training.atomic_torch_save = original_save
                training.evaluate = original_evaluate
        best, last = int(summary["best_epoch"]), int(summary["completed_epochs"])
        selected = Path(summary["checkpoint_path"])
        last_path = selected.parent / "last_epoch_state.pt"
        write_json(run / "SELECTION_CLOSED.json", {"best_epoch": best, "last_epoch": last,
                                                  "selection_population": "internal_dev", "test_locked": True,
                                                  "best_checkpoint_sha256": sha(selected), "last_checkpoint_sha256": sha(last_path),
                                                  "authorized_evaluation_pairs": authorized_evaluation_pairs(last, best)})
        history = read_json(selected.parent / "history.json")["history"]
        require([row["epoch"] for row in history] == list(range(1, last + 1)), "Training epoch history is incomplete")
        earliest = min(history, key=lambda row: row["val_qty_rmse"])["epoch"]
        require(best == earliest, "Selection does not match the earliest strict dev RMSE minimum")
        for row in history:
            for split, prefix in (("fit", "train_eval_"), ("dev", "")):
                curves.append({"split": split, "epoch": row["epoch"],
                               "count": COUNTS[split],
                               "qty_rmse": row[prefix + ("qty_rmse" if split == "fit" else "val_qty_rmse")],
                               "qty_mae": row[prefix + ("qty_mae" if split == "fit" else "val_qty_mae")],
                               "log_qty_mse": row[prefix + "val_log_qty_mse"],
                               "time_nll": row[prefix + "val_time_nll"],
                               "joint_objective": row[prefix + "val_joint_objective"]})
        full = safe_load_input(args)
        original_train, original_val = _loader(full, "train").dataset, _loader(full, "validation").dataset
        metadata = {name: pl.read_parquet(prepared / f"{name}_metadata.parquet").to_dicts()
                    for name in ("fit", "dev", "audit", "validation")}
        loaders = {name: _subset_loader(original_val if name == "validation" else original_train,
                                       [r["target_index"] for r in rows]) for name, rows in metadata.items()}
        model, _ = old._build_model(data, "single_mlp")
        model.to(args.device)
        evaluations = run / "post_selection"
        evaluations.mkdir()
        summaries, group_summaries = [], []
        for epoch in audit_epochs(last, best):
            path = selected if epoch == best else last_path if epoch == last else snapshots / f"epoch_{epoch:03d}.pt"
            payload = torch_load_checkpoint(path, map_location="cpu")
            model.load_state_dict(payload["model_state_dict"], strict=True)
            digest = canonical_state_dict_sha256(model.state_dict())
            require(digest == payload["model_state_sha256"], "Checkpoint state changed")
            for split in ("fit", "dev", "audit", "validation"):
                if epoch not in authorized_evaluation_pairs(last, best)[split]:
                    continue
                rows, metrics = _evaluate_events(model, loaders[split], metadata[split], args.device)
                if split in ("fit", "dev"):
                    recorded = next(row for row in curves if row["split"] == split and row["epoch"] == epoch)
                    require(metrics["count"] == recorded["count"] and all(
                        math.isclose(metrics[key], recorded[key], rel_tol=1e-8, abs_tol=1e-10)
                        for key in ("qty_rmse", "qty_mae", "log_qty_mse", "time_nll", "joint_objective")),
                        "Same-checkpoint event metrics differ from the recorded learning curve")
                filename = evaluations / f"{split}_epoch_{epoch:03d}.parquet"
                require(not filename.exists(), "Post-selection output already exists")
                pl.DataFrame(rows).write_parquet(filename)
                record = {"split": split, "epoch": epoch, **metrics, "checkpoint_sha256": sha(path),
                          "state_sha256": digest, "events_sha256": sha(filename)}
                summaries.append(record)
                group_summaries.extend({"split": split, "epoch": epoch, **group}
                                       for group in group_event_metrics(rows))
                if split in ("audit", "validation"):
                    curves.append({key: record[key] for key in ("split", "epoch", "count", "qty_rmse", "qty_mae", "log_qty_mse", "time_nll", "joint_objective")})
                print(f"[post-selection] {split} epoch={epoch} count={metrics['count']} rmse={metrics['qty_rmse']:.8f} nll={metrics['time_nll']:.8f}", flush=True)
        require(_inputs(args) == before, "Original inputs changed during fitting")
        write_json(evaluations / "SUMMARY.json", {"evaluations": summaries, "test_locked": True})
        with (evaluations / "support_group_metrics.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(group_summaries[0]))
            writer.writeheader()
            writer.writerows(group_summaries)
        with (run / "learning_curves.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(curves[0]))
            writer.writeheader()
            writer.writerows(curves)
        _plot_curves(run / "learning_curves", curves, best, last)
        write_json(run / "COMPLETE.json", {"schema": SCHEMA, "status": "computation_complete_pending_independent_QA",
                                          "best_epoch": best, "last_epoch": last, "inputs_unchanged": True,
                                          "test_locked": True, "completed_unix": time.time()})
        return {"status": "computation_complete_pending_independent_QA", "best_epoch": best, "last_epoch": last}
    except BaseException as exc:
        path = run / "FAILURE.json"
        if not path.exists():
            write_json(path, {"error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
                              "test_locked": True, "automatic_retry": False})
        raise
    finally:
        training.make_loader, training.train_epoch_with_telemetry = original_loader, original_epoch
        training.evaluate = original_evaluate


def fit(args):
    """Preserve even setup failures after an exclusive attempt has been claimed."""
    run, claim = args.root / "fit", args.root / "FIT_CLAIM.json"
    require(not run.exists() and not claim.exists(), "Fit attempt already exists; no resume/retry")
    try:
        return _fit_once(args)
    except BaseException as exc:
        if claim.exists():
            run.mkdir(exist_ok=True)
            failure = run / "FAILURE.json"
            if not failure.exists():
                write_json(failure, {"error_type": type(exc).__name__, "error": str(exc),
                                     "traceback": traceback.format_exc(), "test_locked": True,
                                     "automatic_retry": False})
        raise


def synthetic():
    def row(index, origin, label):
        return {"split": "train", "target_index": index, "entity_id": "x", "target_seq": label,
                "raw_quantity": 1., "target_gap": label-origin, "history_length": 1,
                "origin_time": origin, "label_time": label}
    groups = partition_rows([row(0, 309, 310), row(1, 310, 311), row(2, 311, 312), row(3, 411, 412), row(4, 412, 413)])
    require({key: len(value) for key, value in groups.items()} == {"fit": 1, "dev": 1, "audit": 1, "purged": 2}, "Boundary synthetic failed")
    require(audit_epochs(45, 27) == [1, 5, 10, 20, 27, 40, 45], "Epoch grid synthetic failed")
    state = torch.get_rng_state().clone()
    with preserved_rng():
        torch.rand(3)
    require(torch.equal(state, torch.get_rng_state()), "RNG restore synthetic failed")
    return {"synthetic": "passed", "real_training": False, "real_input_access": False, "checks": 3}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "synthetic", "fit"), required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--source-contract", type=Path)
    parser.add_argument("--input-frame", type=Path)
    parser.add_argument("--train-metadata", type=Path)
    parser.add_argument("--validation-metadata", type=Path)
    parser.add_argument("--science-spec", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.mode != "synthetic":
        require(all(getattr(args, key) is not None for key in ("root", "source_contract", "input_frame", "train_metadata", "validation_metadata", "science_spec")), "All input and artifact paths required")
    result = synthetic() if args.mode == "synthetic" else prepare(args) if args.mode == "prepare" else fit(args)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
