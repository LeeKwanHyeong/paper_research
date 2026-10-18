"""Saved Cap Validation predictions grouped only by observable last quantity.

No model is constructed, checkpoint loaded, inference performed, or Test opened.
"""
from __future__ import annotations

import csv
import datetime as dt
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
import polars as pl

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
BUNDLE = ROOT / "search_artifacts/titantpp_mlp_cap36_20261017_v2"
CONTRACT_SHA = "48bfbfeb75085af687b9055d9fe31f6cf1d7c6709ba3d2f37662360380133f4d"
FINAL_SHA = "a4d2cab80f9e57224460e8146675f6d64a0ac6b7be81f3aae7f33177080e5c15"
ARMS = ("titantpp_history_mlp", "titantpp_history_mlp_width16",
        "titantpp_all_available_history_mlp", "titantpp_current_only_param_matched")
SEEDS = (42, 52, 62)
INPUTS = {}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bound(path, expected):
    path = Path(path).resolve()
    assert path.is_relative_to(ROOT)
    assert "test_predictions" not in path.name and "test_receipt" not in path.name
    assert "THREE_SEED_TEST" not in path.name
    digest = sha(path)
    assert digest == expected, str(path)
    INPUTS[str(path.relative_to(ROOT))] = digest
    return path


def read(path, expected):
    return json.loads(bound(path, expected).read_text())


def metrics(truth, prediction):
    error = prediction - truth
    n = len(error)
    assert n > 0
    sse = float(np.square(error).sum())
    ae = float(np.abs(error).sum())
    return {"rmse": math.sqrt(sse / n), "mae": ae / n,
            "bias": float(error.mean()), "sse": sse, "absolute_error_sum": ae,
            "prediction_mean": float(prediction.mean())}


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    # The existing dataset loader constructs observed histories; no model is built.
    from data_loader.event_seq_data_module import RMTPPWeekLookbackDataset
    from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
    from paper.scripts.run_titantpp_mlp_cap36 import population_identity

    source_sha = sha(__file__)
    final = read(BUNDLE / "retrieval/final_receipt.json", FINAL_SHA)
    assert final["status"] == "complete" and final["candidate_fit_count"] == 36
    contract = read(BUNDLE / "execution_contract.json", CONTRACT_SHA)
    canonical = hashlib.sha256(json.dumps(contract, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert canonical == final["contract_sha256"]
    assert len(contract["jobs"]) == 36
    assert {j["arm"] for j in contract["jobs"]} == set(ARMS)

    datasets = {}
    for d in contract["datasets"]:
        binding = d["inherited_data_identity"]["data"]
        input_path = BUNDLE / "input_snapshot" / binding["path"]
        raw = pl.read_parquet(bound(input_path, binding["sha256"]))
        assert set(raw["chronological_split"].unique().to_list()) == {"train", "validation"}
        cap = float(d["cap_value"])
        assert raw["demand_qty"].is_finite().all()
        assert raw["demand_qty"].min() >= 0 and raw["demand_qty"].max() <= cap
        frame = prepare_count_frame(raw.sort(["oper_part_no", "seq"]))
        ds = RMTPPWeekLookbackDataset(frame, lookback_weeks=d["loader"]["lookback_weeks"],
            max_seq_len=d["loader"]["max_seq_len"], mode="all",
            split_col="chronological_split", target_splits={"validation"})
        truth, identity = population_identity(ds, "validation")
        assert identity == d["inherited_data_identity"]["populations"]["validation"]
        last = np.array([ds.val_lists[p][i] for p, i in ds.index], dtype=np.float64)
        entity = [str(ds.parts[p]) for p, i in ds.index]
        sequence = [int(ds.seq_lists[p][i + 1]) for p, i in ds.index]
        assert len(last) == len(truth) == len(entity) == len(sequence)
        assert np.isfinite(last).all() and (last >= 0).all() and (last <= cap).all()
        assert len(set(zip(entity, sequence))) == len(truth)
        datasets[d["dataset_id"]] = dict(cap=cap, truth=truth, last=last, entity=entity,
            sequence=sequence, identity=identity)

    manifests, originals = {}, {}
    for host in ("5080", "5090"):
        directory = BUNDLE / "retrieval" / host
        collection = read(directory / "receipt.json", final["files"][str(directory / "receipt.json")])
        assert collection["status"] == "complete"
        manifest = read(directory / "manifest.json", collection["manifest_sha256"])
        assert manifest["files"]
        manifests[host], originals[host] = manifest["files"], directory / "originals"

    per_seed, bindings, checks = [], [], []
    for job in contract["jobs"]:
        host, job_id = job["host"], job["id"]
        original, manifest = originals[host], manifests[host]
        receipt_rel = f"run/{job_id}/validation_receipt.json"
        receipt_path = original / receipt_rel
        assert manifest[receipt_rel] == final["files"][str(receipt_path)]
        receipt = read(receipt_path, manifest[receipt_rel])
        assert receipt["schema"] == "titantpp_mlp_cap36_stage_v1"
        assert receipt["status"] == "complete" and receipt["stage"] == "validation"
        assert receipt["held_out_test_evaluated"] is False
        assert receipt["validation_replay_passed"] is True and receipt["parameters_unchanged"] is True
        assert (receipt["arm"], receipt["dataset"], receipt["seed"], receipt["host"], receipt["job_id"]) == (
            job["arm"], job["dataset"], job["seed"], host, job_id)
        assert receipt["contract_sha256"] == canonical and job["seed"] in SEEDS
        for relative, expected in receipt["files"].items():
            assert manifest[relative] == final["files"][str(original / relative)] == expected
            bound(original / relative, expected)
        pred_rel = f"run/{job_id}/validation_predictions.jsonl.gz"
        pred_path = original / pred_rel
        assert manifest[pred_rel] == receipt["files"][pred_rel]
        cohort = datasets[job["dataset"]]
        assert receipt["target_identity"] == cohort["identity"]
        truth, last, cap = cohort["truth"], cohort["last"], cohort["cap"]
        predicted = np.empty(len(truth), dtype=np.float64)
        count = 0
        with gzip.open(pred_path, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                assert row["split"] == "validation" and row["target_index"] == count
                assert row["entity_id"] == cohort["entity"][count]
                assert row["target_seq"] == cohort["sequence"][count]
                assert row["raw_quantity"] == truth[count]
                predicted[count] = row["predicted_raw_quantity"]
                count += 1
        assert count == len(truth) == receipt["prediction_rows"]
        assert np.isfinite(predicted).all()
        overall = metrics(truth, predicted)
        for own, stored in (("rmse", "qty_rmse"), ("mae", "qty_mae"), ("bias", "qty_bias")):
            assert math.isclose(overall[own], receipt["metrics"]["overall"][stored], rel_tol=1e-12, abs_tol=1e-10)
        group_metrics = []
        for name, mask in (("last_at_cap", last == cap), ("last_below_cap", last < cap)):
            n = int(mask.sum())
            assert n > 0
            model = metrics(truth[mask], predicted[mask])
            reference = metrics(truth[mask], last[mask])
            r = {"dataset": job["dataset"], "model": job["arm"], "seed": job["seed"],
                "host": host, "split": "validation", "condition": name, "cap_value": cap,
                "selected_epoch": receipt["selected_epoch"], "count": n, "total_count": len(truth),
                "population_fraction": n / len(truth),
                "entity_count": len({entity for entity, take in zip(cohort["entity"], mask) if take}),
                "truth_mean": float(truth[mask].mean()), "last_mean": float(last[mask].mean()),
                "next_at_cap_fraction": float((truth[mask] == cap).mean())}
            for k in model:
                r["model_" + k], r["last_reference_" + k] = model[k], reference[k]
            for k in ("rmse", "mae", "bias"):
                r["model_minus_last_" + k] = model[k] - reference[k]
            for k in ("rmse", "mae"):
                r["model_minus_last_" + k + "_percent"] = 100 * (model[k] - reference[k]) / reference[k] if reference[k] else None
                r["model_" + k + "_strictly_better_than_last"] = model[k] < reference[k]
            r["model_both_rmse_mae_strictly_better"] = model["rmse"] < reference["rmse"] and model["mae"] < reference["mae"]
            per_seed.append(r)
            group_metrics.append((n, model))
        assert sum(n for n, _ in group_metrics) == len(truth)
        for k in ("sse", "absolute_error_sum"):
            assert math.isclose(sum(m[k] for _, m in group_metrics), overall[k], rel_tol=1e-12, abs_tol=1e-10)
        assert math.isclose(sum(n * m["bias"] for n, m in group_metrics) / len(truth), overall["bias"], rel_tol=1e-12, abs_tol=1e-10)
        checks.append({"job_id": job_id, "exact_target_index_entity_seq_truth": True,
            "saved_overall_metrics_reproduced": True, "partition_additivity": True,
            "target_count": count})
        bindings.append({"job_id": job_id, "dataset": job["dataset"], "model": job["arm"],
            "seed": job["seed"], "host": host, "selected_epoch": receipt["selected_epoch"],
            "checkpoint_sha256": receipt["checkpoint_sha256"], "state_sha256": receipt["state_sha256"],
            "receipt_path": str(receipt_path.relative_to(ROOT)), "receipt_sha256": manifest[receipt_rel],
            "prediction_path": str(pred_path.relative_to(ROOT)), "prediction_sha256": manifest[pred_rel]})

    assert len(per_seed) == 72 and len(bindings) == 36
    summaries = []
    for dataset in datasets:
        for model in ARMS:
            for condition in ("last_at_cap", "last_below_cap"):
                rows = sorted([r for r in per_seed if (r["dataset"], r["model"], r["condition"]) == (dataset, model, condition)], key=lambda r: r["seed"])
                assert [r["seed"] for r in rows] == list(SEEDS)
                fixed = ("cap_value", "count", "total_count", "population_fraction", "entity_count", "truth_mean", "last_mean", "next_at_cap_fraction")
                summary = {"dataset": dataset, "model": model, "split": "validation",
                    "condition": condition, "seed_count": 3}
                for k in fixed:
                    assert all(r[k] == rows[0][k] for r in rows)
                    summary[k] = rows[0][k]
                for k in ("model_rmse", "model_mae", "model_bias", "model_prediction_mean",
                    "last_reference_rmse", "last_reference_mae", "last_reference_bias",
                    "model_minus_last_rmse", "model_minus_last_mae", "model_minus_last_bias",
                    "model_minus_last_rmse_percent", "model_minus_last_mae_percent"):
                    values = [r[k] for r in rows]
                    summary[k + "_mean"] = statistics.mean(values) if all(v is not None for v in values) else None
                    summary[k + "_sample_sd"] = statistics.stdev(values) if all(v is not None for v in values) else None
                for k in ("rmse", "mae"):
                    summary["model_" + k + "_better_seeds"] = ";".join(str(r["seed"]) for r in rows if r["model_" + k + "_strictly_better_than_last"])
                    summary["model_" + k + "_worse_seeds"] = ";".join(str(r["seed"]) for r in rows if r["model_minus_last_" + k] > 0)
                    summary["model_" + k + "_tied_seeds"] = ";".join(str(r["seed"]) for r in rows if r["model_minus_last_" + k] == 0)
                summary["model_both_metrics_better_seeds"] = ";".join(str(r["seed"]) for r in rows if r["model_both_rmse_mae_strictly_better"])
                summaries.append(summary)

    outputs = {"CAP_OBSERVABLE_PREVIOUS_CAP_PER_SEED.csv": per_seed,
        "CAP_OBSERVABLE_PREVIOUS_CAP_SUMMARY.csv": summaries}
    for filename, rows in outputs.items():
        write_csv(OUT / filename, rows)
    assert sha(__file__) == source_sha
    result = {"status": "complete", "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "official_Q3_Cap_36_saved_Validation_predictions_only",
        "conditions": ["observed_last_quantity_equal_to_fixed_cap", "observed_last_quantity_below_fixed_cap"],
        "not_conditioned_on_target_quantity": True, "dataset_count": 3, "model_count": 4,
        "seed_count": 3, "fit_count": 36, "per_seed_group_rows": len(per_seed),
        "summary_group_rows": len(summaries), "input_file_count": len(INPUTS),
        "prediction_rows_exactly_joined": sum(c["target_count"] for c in checks),
        "new_model_training": False, "new_model_inference": False, "checkpoint_loading": False,
        "remote_calls": 0, "held_out_rows_opened": 0,
        "reference": "Arithmetic prediction equals the observed last quantity, without fitting or model substitution.",
        "bias_sign": "prediction minus truth", "delta_sign": "model metric minus last-quantity arithmetic reference; negative RMSE/MAE favors model",
        "uncertainty": "Mean and sample SD(ddof=1) over exactly seeds42/52/62; no significance claim",
        "population_identities": {name: c["identity"] for name, c in datasets.items()},
        "checks": checks, "bindings": bindings, "inputs": INPUTS,
        "source_sha256": source_sha,
        "output_sha256": {filename: sha(OUT / filename) for filename in outputs},
        "summaries": summaries,
        "interpretation_limits": [
            "These conditions are available from observed history before the target, avoiding a target-at-cap selection as the grouping rule.",
            "A worse saved model than this arithmetic reference demonstrates loss on the stated observable cohort only; it does not isolate log1p, head, loss, capacity, or backbone as the cause.",
            "No proposed replacement or new model was trained. Full Validation selection remains unchanged. This is a descriptive follow-up on previously analyzed Validation data.",
            "Capped quantity defines this official dataset and merges distinct quantities at its ceiling; results do not estimate uncapped prediction performance.",
            "Intermittent seeds are hosted on both 5080/cuDNN91900 and5090/cuDNN92000 as recorded; no server adjustment or causal attribution is made."]}
    destination = OUT / "CAP_OBSERVABLE_PREVIOUS_CAP_RESULT.json"
    with destination.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "inputs": len(INPUTS), "joined_rows": result["prediction_rows_exactly_joined"],
        "result_sha256": sha(destination), "summaries": summaries}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
