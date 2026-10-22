"""Local-only evidence validation and A--D aggregation for the Shared15 study."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile

import torch

SCHEMA = "shared_component_ablation_comparison_v1"
ARMS = ("full", "no_prototype", "no_history", "encoder_only")
SPLITS = ("validation", "test")


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), "JSON object required")
    return value


def sealed(path: Path, digest: str) -> dict:
    require(sha(path) == digest, "Contract raw SHA differs")
    return read(path)


def write_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


class MetricAccumulator:
    """Raw quantity and recorded TimeNLL recomputation from stored predictions."""
    def __init__(self, cutoff: float):
        require(math.isfinite(cutoff) and cutoff >= 0, "Invalid large-quantity cutoff")
        self.cutoff, self.all, self.large = cutoff, [0, 0., 0., 0.], [0, 0., 0., 0.]

    def add(self, truth: float, prediction: float, time_nll: float) -> None:
        require(all(math.isfinite(float(x)) for x in (truth, prediction, time_nll)) and truth >= 0, "Nonfinite prediction row")
        for values in (self.all, self.large if truth > self.cutoff else None):
            if values is None: continue
            error = prediction-truth; values[0] += 1; values[1] += abs(error); values[2] += error*error; values[3] += time_nll

    @staticmethod
    def _finish(values: list[float]) -> dict:
        n, ae, se, nll = values
        return {"count": n, "qty_mae": ae/n if n else None, "qty_rmse": math.sqrt(se/n) if n else None,
                "time_nll": nll/n if n else None, "status": "available" if n else "N/A_empty"}

    def finish(self) -> dict:
        require(self.all[0] > 0, "Empty prediction inventory")
        return {"overall": self._finish(self.all), "large": self._finish(self.large), "large_rule": "true_qty > quantity_boundaries_all_train_rows[-1]", "large_cutoff": self.cutoff}


def prediction_metrics(path: Path, cutoff: float, split: str) -> dict:
    accumulator = MetricAccumulator(cutoff); rows = 0; targets = []; seen = set(); quantities = []
    with gzip.open(path, "rt") as stream:
        for line in stream:
            value = json.loads(line); require(value.get("split") == split, "Prediction split differs")
            target = (value["target_index"], value["entity_id"], value["target_seq"]); entity_target = target[1:]
            require(entity_target not in seen and value["target_index"] == rows, "Prediction target identity/order differs")
            accumulator.add(value["raw_quantity"], value["predicted_raw_quantity"], value["time_nll"]); rows += 1
            targets.append(target); seen.add(entity_target); quantities.append(float(value["raw_quantity"]))
    require(rows > 0, "Empty prediction file")
    import numpy as np
    result = accumulator.finish(); result["prediction_rows"] = rows; result["prediction_sha256"] = sha(path); result["target_tuples_sha256"] = canonical(targets); result["target_quantity_sha256"] = hashlib.sha256(b"hard_lmm_target_quantity_v1\0" + np.asarray(quantities, dtype="<f8").tobytes()).hexdigest(); return result


def mapped(root: Path, remote_root: str, foreign: str) -> Path:
    remote, foreign_path = Path(remote_root), Path(foreign)
    try: relative = foreign_path.relative_to(remote)
    except ValueError as error: raise ValueError("Artifact path escaped remote root") from error
    local = (root / relative).resolve(); require(local.is_relative_to(root.resolve()) and local.is_file() and not local.is_symlink(), "Missing or symbolic retrieved artifact")
    return local


def relative_file(root: Path, value: str) -> Path:
    path = Path(value); require(not path.is_absolute() and ".." not in path.parts, "Relative artifact escaped host root")
    local = (root / path).resolve(); require(local.is_relative_to(root.resolve()) and local.is_file() and not local.is_symlink(), "Missing or symbolic relative artifact")
    return local


def _manifest(host_dir: Path) -> tuple[Path, dict, Path]:
    for base in (host_dir, host_dir / "originals"):
        candidate = base / "collection_test_cost.manifest.json"
        if candidate.is_file(): return candidate, read(candidate), base / "originals" if (base / "originals").is_dir() else base
    raise ValueError("Missing collection_test_cost.manifest.json")


def verify_collection(retrieval: Path, contract: dict) -> dict[str, Path]:
    """Validate each retrieved archive once, returning its verified extracted tree."""
    from paper.scripts import advance_shared_component_ablation as controller
    roots = {}
    for host in ("5080", "5090"):
        host_dir = retrieval / "test_cost" / host if (retrieval / "test_cost").is_dir() else retrieval / host
        roots[host] = controller.validate_collection(contract, host, "test_cost", host_dir, extract=False)
    return roots

def _receipt(root: Path, contract: dict, host: str, job: dict, stage: str) -> dict:
    path = root / "run" / job["id"] / f"{stage}_receipt.json"; value = read(path)
    require(value.get("status") == "complete" and value.get("stage") == stage and value.get("host") == host and value.get("job_id") == job["id"], "Stage receipt identity differs")
    require(value.get("contract_sha256") == canonical(contract), "Stage contract differs")
    return value


def _exit(root: Path, contract: dict, job: dict, stage: str) -> dict:
    candidates = [root / "control" / f"{job['id']}.{stage}.exit.json", root / "run" / job["id"] / f"{stage}.exit.json", root / "run" / job["id"] / f"{stage}_exit.json"]
    path = next((candidate for candidate in candidates if candidate.is_file()), None); require(path is not None, "Missing scientific actual-exit receipt")
    value = read(path); require(value.get("actual_os_returncode") == 0 and value.get("timeout") is False and value.get("supervisor_error") is None
        and value.get("resource_incident") is None and value.get("job_id") == job["id"] and value.get("stage") == stage
        and value.get("host") == job["host"] and value.get("contract_sha256") == canonical(contract), "Scientific stage did not exit successfully")
    return value


def _mean_sd(values: list[float | None]) -> dict:
    cleaned = [float(value) for value in values if value is not None]
    return {"mean": statistics.fmean(cleaned) if cleaned else None, "sample_sd_ddof1": statistics.stdev(cleaned) if len(cleaned) > 1 else None, "n": len(cleaned)}


def _diff(left: float | None, right: float | None) -> dict:
    if left is None or right is None: return {"absolute": None, "relative": None}
    return {"absolute": left-right, "relative": None if right == 0 else (left-right)/right}


def _verify_cost(root: Path, contract: dict, host: str, job: dict, train: dict) -> list[dict]:
    batch = root / "cost" / "batches" / f"{job['dataset']}.json"; manifest = read(batch)
    data = next(item for item in contract["datasets"] if item["dataset_id"] == job["dataset"])
    batch_file = batch.with_suffix(".pt")
    require(sha(batch_file) == manifest.get("file_sha256"), "Cost batch binary differs from manifest")
    require(manifest.get("dataset") == job["dataset"] and manifest.get("split") == "train" and manifest.get("synthetic") is False
        and manifest.get("contract_sha256") == canonical(contract) and manifest.get("preparation_seed") == 42
        and manifest.get("input_sha256") == data["inherited_data_identity"]["data"]["sha256"], "Cost cache is not real fixed Train evidence")
    cache_rows = manifest.get("batches", [])
    require(len(cache_rows) == 37 and all(row.get("batch") == index and row.get("phase") == ("warmup" if index < 5 else "measured")
        and row.get("target_split") == "train" and isinstance(row.get("targets"), list) and len(row["targets"]) == 128
        and isinstance(row.get("tensor_sha256"), str) and len(row["tensor_sha256"]) == 64 for index, row in enumerate(cache_rows)), "Cost batch manifest rows differ")
    cache_targets = [tuple(target) for row in cache_rows for target in row["targets"]]
    require(len(cache_targets) == 4736 and len(set(cache_targets)) == 4736 and manifest.get("target_ids_sha256") == canonical([[str(entity), int(seq)] for entity, seq in cache_targets]), "Cost cache target identity differs")
    values = []
    for mode in ("train", "eval"):
        for rep in (1, 2, 3):
            value = read(root / "cost" / "results" / job["id"] / f"{mode}_rep{rep}.json")
            require(value.get("status") == "complete" and value.get("host") == host and value.get("mode") == mode and value.get("timing_rep") == rep, "Cost receipt identity differs")
            require(value.get("contract_sha256") == canonical(contract) and value.get("batch_manifest_sha256") == sha(batch), "Cost contract/cache differs")
            require(value.get("target_ids_sha256") == manifest["target_ids_sha256"] and len(value.get("batch_tensor_sha256", [])) == 37, "Cost target/tensor evidence differs")
            require(value["batch_tensor_sha256"] == [row["tensor_sha256"] for row in manifest["batches"]], "Cost batch tensors differ")
            require(value.get("checkpoint_binary_unchanged") is True and value.get("checkpoint_saved") is False and value.get("input_split") == "train" and value.get("synthetic") is False, "Cost altered checkpoint or input split")
            require(value.get("runtime", {}).get("gpu", {}).get("uuid") == contract["hosts"][host]["gpu_uuid"], "Cost GPU identity differs")
            selected = value.get("selected", {})
            require(selected.get("checkpoint_sha256") == train["checkpoint_sha256"] and selected.get("state_sha256") == train["state_sha256"] and selected.get("selected_epoch") == train["selected_epoch"], "Cost selected checkpoint identity differs")
            require(value.get("job_id") == job["id"] and value.get("dataset") == job["dataset"] and value.get("arm") == job["arm"] and value.get("training_seed") == job["seed"], "Cost job identity differs")
            for key in ("median_compute_seconds", "median_transfer_compute_seconds", "throughput_targets_per_second", "throughput_including_transfer_targets_per_second"):
                require(math.isfinite(float(value[key])) and float(value[key]) > 0, "Cost timing is not finite positive")
            rows = value.get("rows", []); require(len(rows) == 37 and all(row.get("batch") == index and row.get("phase") == ("warmup" if index < 5 else "measured") and row.get("targets") == 128 for index, row in enumerate(rows)), "Cost raw timing rows differ")
            for row in rows:
                transfer, compute, total = (row.get(key) for key in ("transfer_seconds", "compute_seconds", "transfer_compute_seconds"))
                require(all(isinstance(item, (int, float)) and math.isfinite(float(item)) and float(item) > 0 for item in (transfer, compute, total))
                    and math.isclose(float(total), float(transfer) + float(compute), rel_tol=1e-8, abs_tol=1e-10), "Cost raw timing values differ")
            measured = rows[5:]
            require(value["median_compute_seconds"] == statistics.median(row["compute_seconds"] for row in measured) and value["median_transfer_compute_seconds"] == statistics.median(row["transfer_compute_seconds"] for row in measured), "Cost median differs from raw timing rows")
            require(math.isclose(value["throughput_targets_per_second"], sum(row["targets"] for row in measured)/sum(row["compute_seconds"] for row in measured)) and math.isclose(value["throughput_including_transfer_targets_per_second"], sum(row["targets"] for row in measured)/sum(row["transfer_compute_seconds"] for row in measured)), "Cost throughput differs from raw timing rows")
            require(value.get("optimizer_steps") == (37 if mode == "train" else 0), "Cost optimizer steps differ")
            for group in ("memory_after_warmup_baseline_bytes", "warmup_peak_bytes", "measured_peak_bytes", "total_peak_bytes", "measured_incremental_peak_bytes"):
                require(all(isinstance(value[group][key], int) and value[group][key] >= 0 for key in ("allocated", "reserved")), "Cost memory evidence differs")
            for group in ("memory_after_warmup_baseline_bytes", "warmup_peak_bytes", "measured_peak_bytes", "total_peak_bytes"):
                require(value[group]["reserved"] >= value[group]["allocated"], "Reserved GPU memory is below allocated memory")
            for key in ("allocated", "reserved"):
                require(value["measured_peak_bytes"][key] >= value["memory_after_warmup_baseline_bytes"][key], "Measured peak is below post-warmup baseline")
                require(value["total_peak_bytes"][key] == max(value["warmup_peak_bytes"][key], value["measured_peak_bytes"][key]) and value["measured_incremental_peak_bytes"][key] == max(0, value["measured_peak_bytes"][key]-value["memory_after_warmup_baseline_bytes"][key]), "Cost memory derivation differs")
            require(isinstance(value.get("parameters"), int) and value["parameters"] > 0, "Cost parameter count differs")
            values.append(value)
    return values


def aggregate(contract: dict, retrieval: Path) -> dict:
    from paper.scripts import manage_shared_component_ablation as manager
    manager.validate(contract); roots = verify_collection(retrieval, contract)
    for host, root in roots.items():
        gate = manager.verify_global_gate(contract, root)
        for job_id, evidence in gate["jobs"].items():
            job = next(job for job in contract["jobs"] if job["id"] == job_id)
            if job["host"] != host: continue
            for stage in ("train", "validation"):
                receipt_item = evidence[f"{stage}_receipt"]
                receipt_local = root / "run" / job_id / f"{stage}_receipt.json"
                require(receipt_local.is_file() and sha(receipt_local) == receipt_item["sha256"], "Global gate receipt differs from collected run receipt")
                exit_item = evidence[f"{stage}_exit"]
                exit_stage = "reuse" if stage == "train" and job.get("reuse") else stage
                exit_local = root / "control" / f"{job_id}.{exit_stage}.exit.json"
                require(exit_local.is_file() and sha(exit_local) == exit_item["sha256"], "Global gate exit differs from collected control exit")
        manager.verify_phase_receipt(contract, host, "test_cost", root)
    records, costs, fits = [], [], []
    for job in contract["jobs"]:
        host, root = job["host"], roots[job["host"]]
        train = _receipt(root, contract, host, job, "train"); _exit(root, contract, job, "reuse" if job.get("reuse") else "train")
        checkpoint = relative_file(root, train["selected_checkpoint"])
        require(sha(checkpoint) == train["checkpoint_sha256"], "Selected checkpoint binary differs")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        from simple_lab_test.search.common.runner import canonical_state_dict_sha256
        require(canonical_state_dict_sha256(payload["model_state_dict"]) == train["state_sha256"], "Selected checkpoint tensor digest differs")
        summary = train.get("summary", {})
        if train.get("training_origin") == "fresh_fit":
            elapsed, epochs = summary.get("elapsed_seconds"), summary.get("completed_epochs")
            require(isinstance(elapsed, (int, float)) and math.isfinite(float(elapsed)) and elapsed > 0
                and isinstance(epochs, int) and epochs > 0 and train.get("new_training_epochs") == epochs, "Fresh fit elapsed/epoch evidence differs")
            fits.append({"job_id": job["id"], "dataset": job["dataset"], "arm": job["arm"], "seed": job["seed"], "host": host,
                "training_origin": "fresh_fit", "elapsed_seconds": float(elapsed), "completed_epochs": epochs,
                "amortized_elapsed_seconds_per_epoch": float(elapsed) / epochs})
        else:
            fits.append({"job_id": job["id"], "dataset": job["dataset"], "arm": job["arm"], "seed": job["seed"], "host": host,
                "training_origin": "reused_verified_fit", "elapsed_seconds": None, "completed_epochs": None,
                "amortized_elapsed_seconds_per_epoch": None})
        if job.get("reuse"):
            require(train.get("training_origin") == "reused_verified_fit" and train.get("reuse") == job["reuse"], "Reuse training provenance differs")
            for key, item in job["reuse"].items():
                if not isinstance(item, dict) or "path" not in item: continue
                legacy = (root / "source" / item["path"]).resolve(); require(legacy.is_relative_to(root.resolve()) and sha(legacy) == item["sha256"], "Reused legacy evidence SHA differs")
                if key.startswith("old_"):
                    exit_value = read(legacy); require(exit_value.get("actual_os_returncode") == 0 and exit_value.get("timeout") is False, "Reused original actual exit differs")
        for split in SPLITS:
            receipt = _receipt(root, contract, host, job, split); _exit(root, contract, job, split)
            require(all(receipt.get(key) == train.get(key) for key in ("selected_checkpoint", "selected_epoch", "checkpoint_sha256", "state_sha256")), "Evaluation selected checkpoint differs from training")
            prediction = mapped(root, contract["hosts"][host]["root"], receipt["prediction_path"])
            metrics = prediction_metrics(prediction, job and next(d for d in contract["datasets"] if d["dataset_id"] == job["dataset"])["quantity_boundaries_all_train_rows"][-1], split)
            require(metrics["prediction_rows"] == receipt["prediction_rows"] == receipt["metrics"]["overall"]["count"], "Prediction row count differs")
            identity = receipt.get("target_identity", {})
            require(identity.get("target_count") == metrics["prediction_rows"] and identity.get("target_quantity_sha256") == metrics["target_quantity_sha256"], "Frozen target population differs from predictions")
            data = next(data for data in contract["datasets"] if data["dataset_id"] == job["dataset"])
            identity_source = data["inherited_data_identity"] if split == "validation" else data["full_data_identity"]
            expected = identity_source["populations"][split]
            require(identity == expected, "Receipt target population differs from frozen contract")
            for scope, receipt_scope in (("overall", "overall"), ("large", "tail")):
                for key in ("qty_rmse", "qty_mae", "time_nll"):
                    left, right = metrics[scope][key], receipt["metrics"][receipt_scope][key]
                    require((left is None and right is None) or (left is not None and math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)), "Receipt metrics differ from stored predictions")
            records.append({"job_id": job["id"], "dataset": job["dataset"], "arm": job["arm"], "seed": job["seed"], "host": host, "split": split, "metrics": metrics, "checkpoint_sha256": train["checkpoint_sha256"], "state_sha256": train["state_sha256"]})
        costs.extend(_verify_cost(root, contract, host, job, train))
    for dataset in sorted({record["dataset"] for record in records}):
        cache = [read(roots[host] / "cost" / "batches" / f"{dataset}.json") for host in ("5080", "5090")]
        require(cache[0].get("target_ids_sha256") == cache[1].get("target_ids_sha256")
            and [row.get("tensor_sha256") for row in cache[0].get("batches", [])] == [row.get("tensor_sha256") for row in cache[1].get("batches", [])],
            "Hosts do not share identical fixed Train cost batches")
        for split in SPLITS:
            for seed in (42, 52, 62):
                matched = [record["metrics"] for record in records if (record["dataset"], record["split"], record["seed"]) == (dataset, split, seed)]
                require(len(matched) == 4 and len({record["target_tuples_sha256"] for record in matched}) == 1, "Arms do not share the same ordered prediction targets")
    table = []
    for dataset in sorted({r["dataset"] for r in records}):
        for split in SPLITS:
            for arm in ARMS:
                rows = [r for r in records if (r["dataset"], r["split"], r["arm"]) == (dataset, split, arm)]
                require(len(rows) == 3, "Missing matched seed metrics")
                table.append({"dataset": dataset, "split": split, "arm": arm, "overall": {key: _mean_sd([r["metrics"]["overall"][key] for r in rows]) for key in ("qty_rmse", "qty_mae", "time_nll")}, "large": {key: _mean_sd([r["metrics"]["large"][key] for r in rows]) for key in ("qty_rmse", "qty_mae", "time_nll")}})
    deltas = []
    paired, adverse = [], []
    for left, right in (("full", "no_history"), ("full", "no_prototype"), ("no_prototype", "encoder_only"), ("no_history", "encoder_only")):
        for row in table:
            if row["arm"] != left: continue
            peer = next(item for item in table if item["dataset"] == row["dataset"] and item["split"] == row["split"] and item["arm"] == right)
            deltas.append({"dataset": row["dataset"], "split": row["split"], "comparison": left + "-" + right, "overall": {key: _diff(row["overall"][key]["mean"], peer["overall"][key]["mean"]) for key in row["overall"]}})
        for seed in (42, 52, 62):
            for dataset in sorted({r["dataset"] for r in records}):
                for split in SPLITS:
                    a = next(r for r in records if (r["dataset"], r["split"], r["arm"], r["seed"]) == (dataset, split, left, seed)); b = next(r for r in records if (r["dataset"], r["split"], r["arm"], r["seed"]) == (dataset, split, right, seed))
                    for scope in ("overall", "large"):
                        for metric in ("qty_rmse", "qty_mae", "time_nll"):
                            item = {"dataset": dataset, "split": split, "seed": seed, "comparison": left + "-" + right, "scope": scope, "metric": metric, "left": a["metrics"][scope][metric], "right": b["metrics"][scope][metric]}
                            item.update(_diff(item["left"], item["right"])); item["adverse"] = item["absolute"] is not None and item["absolute"] > 0
                            paired.append(item)
                            if item["adverse"]: adverse.append(item)
    return {"schema": SCHEMA, "status": "complete", "contract_sha256": canonical(contract), "records": records, "table": table, "component_deltas": deltas, "paired_seed_deltas": paired, "adverse_seed_results": adverse, "cost_records": costs, "fit_records": fits, "cost_note": "Timing repetitions are stratified by host/GPU; 5080 and 5090 absolute timings are not pooled. Historical A multi-worker fit times are not directly comparable with new single-worker fits."}


def _formatted(summary: dict) -> str:
    """Render nullable large-quantity metrics without converting N/A into zero."""
    mean, deviation = summary.get("mean"), summary.get("sample_sd_ddof1")
    if mean is None:
        return "N/A"
    if deviation is None:
        return f"{mean:.6g} (n={summary.get('n', 0)})"
    return f"{mean:.6g} ± {deviation:.3g}"


def _memory(value: dict, field: str) -> int | None:
    block = value.get("total_peak_bytes", {})
    item = block.get(field)
    return item if isinstance(item, int) else None


def render_markdown(result: dict) -> str:
    lines = [
        "# Shared15 component ablation", "",
        "Validation/Test summaries use three matched seeds; SD is sample SD (ddof=1). Large-quantity rows use the strict `truth > cutoff` rule. `N/A` preserves an empty tail rather than treating it as zero.", "",
        "## Overall quantity and time accuracy", "",
        "| Dataset | Split | Arm | RMSE | MAE | TimeNLL |", "|---|---|---|---:|---:|---:|",
    ]
    for row in result["table"]:
        values = row["overall"]
        lines.append("| {dataset} | {split} | {arm} | {rmse} | {mae} | {nll} |".format(
            dataset=row["dataset"], split=row["split"], arm=row["arm"],
            rmse=_formatted(values["qty_rmse"]), mae=_formatted(values["qty_mae"]), nll=_formatted(values["time_nll"])))
    lines.extend(["", "## Large-quantity accuracy", "", "| Dataset | Split | Arm | RMSE | MAE | TimeNLL |", "|---|---|---:|---:|---:|---:|"])
    for row in result["table"]:
        values = row["large"]
        lines.append("| {dataset} | {split} | {arm} | {rmse} | {mae} | {nll} |".format(
            dataset=row["dataset"], split=row["split"], arm=row["arm"],
            rmse=_formatted(values["qty_rmse"]), mae=_formatted(values["qty_mae"]), nll=_formatted(values["time_nll"])))
    lines.extend(["", "## Paired adverse seed results", "", "Positive absolute deltas mean the left arm had a higher error or TimeNLL than the right arm.", "", "| Dataset | Split | Seed | Comparison | Scope | Metric | Left | Right | Absolute | Relative |", "|---|---|---:|---|---|---|---:|---:|---:|---:|"])
    for item in result.get("adverse_seed_results", []):
        relative = "N/A" if item.get("relative") is None else f"{item['relative']:.6g}"
        lines.append("| {dataset} | {split} | {seed} | {comparison} | {scope} | {metric} | {left:.6g} | {right:.6g} | {absolute:.6g} | {relative} |".format(**{**item, "relative": relative}))
    if not result.get("adverse_seed_results"):
        lines.append("| — | — | — | — | — | — | — | — | — | — |")
    lines.extend(["", "## Host-stratified cost evidence", "", "Each row is an independent timing process. GPU models are not pooled. Allocated/reserved values are this profiler's allocator peaks, not total device reservations; existing paused-process reservations are separate.", "", "| Host | Dataset | Arm | Seed | Mode | Rep | Median compute (s) | Throughput (targets/s) | Allocated peak (bytes) | Reserved peak (bytes) | Parameters |", "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|"])
    for item in result.get("cost_records", []):
        selected = item.get("selected", {})
        lines.append("| {host} | {dataset} | {arm} | {seed} | {mode} | {rep} | {seconds:.6g} | {throughput:.6g} | {allocated} | {reserved} | {parameters} |".format(
            host=item["host"], dataset=item["dataset"], arm=item["arm"], seed=item["training_seed"], mode=item["mode"], rep=item["timing_rep"],
            seconds=item["median_compute_seconds"], throughput=item["throughput_targets_per_second"], allocated=_memory(item, "allocated"), reserved=_memory(item, "reserved"), parameters=item["parameters"]))
    lines.extend(["", "## Fit wall-clock evidence", "", "These are amortized `elapsed_seconds / completed_epochs` values from the train receipt, including validation and checkpoint storage. They are not per-epoch timing measurements. Reused A has no fabricated zero-time value.", "", "| Host | Dataset | Arm | Seed | Origin | Elapsed (s) | Completed epochs | Amortized elapsed/epoch (s) |", "|---|---|---|---:|---|---:|---:|---:|"])
    for item in result.get("fit_records", []):
        elapsed = "N/A" if item["elapsed_seconds"] is None else f"{item['elapsed_seconds']:.6g}"
        epochs = "N/A" if item["completed_epochs"] is None else str(item["completed_epochs"])
        amortized = "N/A" if item["amortized_elapsed_seconds_per_epoch"] is None else f"{item['amortized_elapsed_seconds_per_epoch']:.6g}"
        lines.append("| {host} | {dataset} | {arm} | {seed} | {training_origin} | {elapsed} | {epochs} | {amortized} |".format(**{**item, "elapsed": elapsed, "epochs": epochs, "amortized": amortized}))
    lines.extend(["", "Historical reused-A parallel fit time is not treated as zero or compared directly with new single-worker fit time."])
    return "\n".join(lines) + "\n"

def render_csv(result: dict) -> str:
    rows = ["dataset,split,arm,metric,mean,sample_sd_ddof1,n"]
    for row in result["table"]:
        for scope in ("overall", "large"):
            for metric, value in row[scope].items():
                rows.append(",".join(str(part) for part in (row["dataset"], row["split"], row["arm"], scope + "." + metric, value["mean"], value["sample_sd_ddof1"], value["n"])))
    return "\n".join(rows) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--contract", type=Path, required=True); parser.add_argument("--contract-sha", required=True); parser.add_argument("--retrieval", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--check-only", action="store_true"); args = parser.parse_args(argv)
    contract = sealed(args.contract, args.contract_sha); result = aggregate(contract, args.retrieval)
    if args.output.exists():
        cached = read(args.output / "REPORT.json"); require(cached == result, "Cached aggregation differs from pure recomputation")
        require((args.output / "REPORT.md").read_text(encoding="utf-8") == render_markdown(result), "Cached Markdown differs")
        require((args.output / "TABLE.csv").read_text(encoding="utf-8") == render_csv(result), "Cached CSV differs")
        print(json.dumps({"status": "verified", "output": str(args.output)}, ensure_ascii=False)); return 0
    if args.check_only:
        print(json.dumps({"status": "ready_outputs_not_written", "output": str(args.output)}, ensure_ascii=False)); return 0
    args.output.mkdir(parents=True, exist_ok=False); write_exclusive(args.output / "REPORT.json", result); (args.output / "REPORT.md").write_text(render_markdown(result), encoding="utf-8"); (args.output / "TABLE.csv").write_text(render_csv(result), encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(args.output)}, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
