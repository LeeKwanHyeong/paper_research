"""Bind existing Q3-capped inputs and derive shared initialization from Train only."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
CAP = ROOT / "benchmark_data/Cap"
MANIFEST_SHA = "a2104f03b744f56ade795f1008c5f85be93ce9de0f1421905a092d374e906514"
PARENT = ROOT / "search_artifacts/titantpp_time_quantity_cross_precheck6_20261016_v1/execution_contract.json"
DATASETS = {"yellow_trip_hourly": ("taxi", 65.), "raf_spare_parts": ("raf", 10.),
            "intermittent_frozen_5000": ("intermittent", 20.)}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def train_statistics(train_values, train_targets):
    import numpy as np
    from paper.scripts.quantity_comparison_data import _all_row_quantity_identity
    q, targets = np.asarray(train_values, dtype=np.float64), np.asarray(train_targets, dtype=np.float64)
    require(q.size > 0 and targets.size > 0 and np.isfinite(q).all()
            and np.isfinite(targets).all() and (q >= 0).all() and (targets >= 0).all(), "Invalid Train quantity")
    logq = np.log1p(q)
    require(logq.std() > 0, "Degenerate capped Train quantities")
    return {"train_log_mean": float(logq.mean()), "train_log_std": float(logq.std()),
            "raw_scale": max(1., float(np.sqrt(np.mean(targets ** 2)))),
            "all_train_quantity_sha256": _all_row_quantity_identity(q),
            "initialization_policy": "derive once from capped Train only; identical for all arms and seeds",
            "recomputation_absolute_tolerance": 1e-12}


def _dataset(frame, definition, split):
    from data_loader.event_seq_data_module import RMTPPWeekLookbackDataset
    return RMTPPWeekLookbackDataset(frame, lookback_weeks=definition["loader"]["lookback_weeks"],
        max_seq_len=definition["loader"]["max_seq_len"], val_ratio=.2, mode="all",
        split_col="chronological_split", target_splits={split})


def derive_definition(parent, trainval_path, full_path, manifest_path, cap_value):
    import numpy as np
    import polars as pl
    import torch
    from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
    from paper.scripts.run_count_aware_tpp_backbone_control import derive_train_time_contract
    from paper.scripts.run_hard_lmm_frozen_lognormal_duration import target_dt_sha256
    from paper.scripts.run_titantpp_mlp_cap36 import population_identity
    from paper.scripts.quantity_comparison_data import _validate_frame

    raw = pl.read_parquet(trainval_path).sort(["oper_part_no", "seq"])
    _validate_frame(raw, allowed_splits={"train", "validation"})
    full_raw = pl.read_parquet(full_path).sort(["oper_part_no", "seq"])
    _validate_frame(full_raw, allowed_splits={"train", "validation", "test"})
    require(full_raw.filter(pl.col("chronological_split") != "test").equals(raw), "Cap split snapshot differs")
    require(float(full_raw["demand_qty"].max()) <= cap_value, "Quantity exceeds existing Cap")
    train = raw.filter(pl.col("chronological_split") == "train")
    frame, train_frame, full_frame = map(prepare_count_frame, (raw, train, full_raw))
    d = deepcopy(parent)
    populations, quantity_targets, time_hashes = {}, {}, {}
    for split, admitted in (("train", train_frame), ("validation", frame), ("test", full_frame)):
        loader = _dataset(admitted, d, split)
        q, populations[split] = population_identity(loader, split)
        quantity_targets[split] = q
        dt = torch.tensor(np.asarray([max(1., float(loader.dt_lists[p][i + 1])) for p, i in loader.index], dtype=np.float64))
        time_hashes[split] = target_dt_sha256(dt)
        if split != "test":
            previous = parent["inherited_data_identity"]["populations"][split]
            require(all(populations[split][k] == previous[k] for k in ("target_count", "target_identity_sha256")),
                    "Cap changed target identity or count")
            require(time_hashes[split] == d["time_statistics"][f"expected_{split}_target_dt_sha256"],
                    "Cap changed target duration")
    d["statistics"] = train_statistics(train["demand_qty"].to_numpy(), quantity_targets["train"])
    d["mu_all_train_rows"] = d["statistics"]["train_log_mean"]
    # Existing reporting quantiles; coincident capped boundaries remain coincident.
    d["quantity_boundaries_all_train_rows"] = [float(train["demand_qty"].quantile(p, interpolation="nearest"))
                                             for p in (.5, .9, .95, .99)]
    require(d["quantity_boundaries_all_train_rows"][-1] == cap_value, "Unexpected Cap p99")
    d["cap_value"] = cap_value
    times = derive_train_time_contract(frame, lookback_weeks=d["loader"]["lookback_weeks"],
                                      max_seq_len=d["loader"]["max_seq_len"])
    for observed, expected in (("time_scale", "train_time_scale"), ("target_log_scaled_mean", "train_log_scaled_mean"),
                               ("target_log_scaled_std", "train_log_scaled_std")):
        require(math.isclose(times[observed], d["time_statistics"][expected], rel_tol=0, abs_tol=1e-12), "Time initialization changed")
    relative = Path("inputs/data") / d["dataset_id"]
    manifest = {"path": str(relative / "split_manifest.json"), "sha256": sha(manifest_path)}
    d["inherited_data_identity"] = {"data": {"path": str(relative / "train_validation.parquet"), "sha256": sha(trainval_path)},
        "split_manifest": manifest, "populations": {s: populations[s] for s in ("train", "validation")}}
    d["parent_data_identity"] = {"data": {"path": str(relative / "full_capped.parquet"), "sha256": sha(full_path)},
        "split_manifest": dict(manifest), "populations": populations}
    d["prepared_train_validation_path"] = d["inherited_data_identity"]["data"]["path"]
    d["data_sha256"] = sha(trainval_path)
    evidence = {"dataset_id": d["dataset_id"], "cap_value": cap_value, "statistics": d["statistics"],
        "quantity_boundaries": d["quantity_boundaries_all_train_rows"], "target_populations": populations,
        "target_dt_sha256": time_hashes, "time_statistics_unchanged": True,
        "train_rows": train.height, "train_validation_rows": raw.height, "full_rows": full_raw.height,
        "strict_tail_count_by_split": {s: int((q > cap_value).sum()) for s, q in quantity_targets.items()},
        "cap_hit_count_by_split": {s: int((q == cap_value).sum()) for s, q in quantity_targets.items()},
        "test_use_at_preparation": "identity/count/Cap verification only; no prediction or fitting"}
    return d, evidence


def prepare_datasets(output_dir):
    output_dir = Path(output_dir)
    require(not output_dir.exists(), "Input snapshot already exists")
    require(sha(CAP / "manifest.json") == MANIFEST_SHA, "Existing Cap manifest changed")
    manifest = json.loads((CAP / "manifest.json").read_text())
    files = {record["path"]: record["sha256"] for record in manifest["files"]}
    parent = json.loads(PARENT.read_text())
    definitions, evidence, inventory = [], [], {}
    output_dir.mkdir(parents=True)
    for prior in parent["datasets"]:
        ds = prior["dataset_id"]
        folder, cap = DATASETS[ds]
        require(manifest["datasets"][folder]["cap_value"] == cap, "Cap threshold changed")
        inputs = {"train_validation.parquet": CAP / "data" / folder / f"{ds}_train_validation.parquet",
                  "full_capped.parquet": CAP / "data" / folder / f"{ds}_with_split.parquet",
                  "split_manifest.json": CAP / "data" / folder / f"{ds}_split_manifest.json"}
        for name, source in inputs.items():
            require(str(source) in files and sha(source) == files[str(source)], "Cap input SHA mismatch")
            relative = Path("inputs/data") / ds / name
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as src, destination.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            inventory[str(relative)] = sha(destination)
        d, item = derive_definition(prior, inputs["train_validation.parquet"], inputs["full_capped.parquet"],
                                    inputs["split_manifest.json"], cap)
        definitions.append(d)
        evidence.append(item)
    return {"datasets": definitions, "input_files": inventory,
            "data_receipt": {"status": "passed", "cap_manifest_sha256": MANIFEST_SHA,
                "raw_parent_contract_sha256": sha(PARENT), "dataset_count": len(definitions), "datasets": evidence,
                "raw_originals_modified": False, "held_out_test_evaluated": False,
                "strict_tail_policy": "q > capped Train p99 is empty; metrics N/A; q == cap reported separately"}}
