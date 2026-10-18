"""Stored official Q3-Cap Validation diagnostics; no model or Test imports."""
from __future__ import annotations

from collections import defaultdict
import csv
import datetime as dt
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
BUNDLE = ROOT / "search_artifacts/titantpp_mlp_cap36_20261017_v2"
CONTRACT_SHA = "48bfbfeb75085af687b9055d9fe31f6cf1d7c6709ba3d2f37662360380133f4d"
FINAL_SHA = "a4d2cab80f9e57224460e8146675f6d64a0ac6b7be81f3aae7f33177080e5c15"
ARMS = ("titantpp_history_mlp", "titantpp_history_mlp_width16",
        "titantpp_all_available_history_mlp", "titantpp_current_only_param_matched")
SEEDS = (42, 52, 62)
INPUTS = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def recorded(path, expected=None):
    path = Path(path).resolve()
    assert path.is_relative_to(ROOT)
    # Binary archive hashing is permitted; Test result files/rows are not read.
    assert "test_predictions" not in path.name and "test_receipt" not in path.name
    assert "/runs/test/" not in str(path) and "THREE_SEED_TEST" not in path.name
    actual = sha(path)
    if expected is not None:
        assert actual == expected, str(path)
    INPUTS[str(path.relative_to(ROOT))] = actual
    return path


def read(path, expected=None):
    return json.loads(recorded(path, expected).read_text())


def stats(q, prediction, temporal, history, cap, selected, total_sse, total_ae):
    n = int(selected.sum())
    base = {"count": n, "status": "available" if n else "N/A_empty"}
    if not n:
        return base
    truth, pred = q[selected], prediction[selected]
    error = pred - truth
    sse = float(np.square(error).sum())
    ae = float(np.abs(error).sum())
    mse, mae, bias = sse / n, ae / n, float(error.mean())
    variance = float(np.square(error - bias).mean())
    assert math.isclose(mse, variance + bias * bias, rel_tol=1e-12, abs_tol=1e-12)
    positive = truth > 0
    return {**base, "truth_mean": float(truth.mean()), "truth_median": float(np.median(truth)),
            "prediction_mean": float(pred.mean()), "prediction_median": float(np.median(pred)),
            "prediction_to_cap_mean": float(pred.mean()) / cap,
            "prediction_to_truth_mean_ratio": float(pred.mean() / truth.mean()) if truth.mean() else None,
            "bias_prediction_minus_truth": bias, "bias_divided_by_cap": bias / cap,
            "underprediction_fraction": float((error < 0).mean()),
            "overprediction_fraction": float((error > 0).mean()),
            "exact_prediction_fraction": float((error == 0).mean()),
            "qty_rmse": math.sqrt(mse), "qty_mae": mae,
            "rmse_divided_by_cap": math.sqrt(mse) / cap, "mae_divided_by_cap": mae / cap,
            "mape_positive_truth_count": int(positive.sum()),
            "mean_absolute_relative_error_positive_truth": float((abs(error[positive]) / truth[positive]).mean()) if positive.any() else None,
            "qty_sse": sse, "qty_absolute_error_sum": ae,
            "sse_share_of_total": sse / total_sse if total_sse else None,
            "absolute_error_share_of_total": ae / total_ae if total_ae else None,
            "mse": mse, "squared_bias": bias * bias, "centered_error_variance": variance,
            "squared_bias_share_of_mse": bias * bias / mse if mse else None,
            "time_nll": float(temporal[selected].mean()),
            "history_length_mean": float(history[selected].mean()),
            "history_length_median": float(np.median(history[selected]))}


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    source_before = sha(Path(__file__))
    final = read(BUNDLE / "retrieval/final_receipt.json", FINAL_SHA)
    assert final["status"] == "complete" and final["candidate_fit_count"] == final["validation_replay_count"] == 36
    assert final["dataset_count"] == final["seed_count"] == 3
    contract = read(BUNDLE / "execution_contract.json", CONTRACT_SHA)
    canonical = hashlib.sha256(json.dumps(contract, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert canonical == final["contract_sha256"]
    assert {j["arm"] for j in contract["jobs"]} == set(ARMS)
    assert len(contract["jobs"]) == 36
    data = {d["dataset_id"]: d for d in contract["datasets"]}
    manifests, local_roots = {}, {}
    for host in ("5080", "5090"):
        dest = BUNDLE / "retrieval" / host
        collection = read(dest / "receipt.json", final["files"][str(dest / "receipt.json")])
        assert collection["status"] == "complete"
        recorded(dest / "original.tar.gz", collection["archive_sha256"])
        manifest = read(dest / "manifest.json", collection["manifest_sha256"])
        assert manifest["files"]
        manifests[host] = manifest["files"]
        local_roots[host] = dest / "originals"
    per_seed, bindings, cohort_keys, additive = [], [], {}, []
    for job in contract["jobs"]:
        host, job_id = job["host"], job["id"]
        original = local_roots[host]
        relative = "run/" + job_id + "/validation_receipt.json"
        rp = original / relative
        expected_sha = manifests[host][relative]
        assert final["files"][str(rp)] == expected_sha
        receipt = read(rp, expected_sha)
        assert receipt["schema"] == "titantpp_mlp_cap36_stage_v1"
        assert receipt["status"] == "complete" and receipt["stage"] == "validation"
        assert receipt["held_out_test_evaluated"] is False and receipt["parameters_unchanged"] is True
        assert receipt["validation_replay_passed"] is True
        assert (receipt["arm"], receipt["dataset"], receipt["seed"], receipt["host"], receipt["job_id"]) == (
            job["arm"], job["dataset"], job["seed"], host, job_id)
        assert receipt["contract_sha256"] == canonical and job["seed"] in SEEDS
        d = data[job["dataset"]]
        population = d["inherited_data_identity"]["populations"]["validation"]
        assert receipt["target_identity"] == population
        predrel = "run/" + job_id + "/validation_predictions.jsonl.gz"
        predp = original / predrel
        assert manifests[host][predrel] == receipt["files"][predrel] == final["files"][str(predp)]
        recorded(predp, manifests[host][predrel])
        # Verify only Validation input/selected-binary dependencies, not Test files.
        for dependency, h in receipt["files"].items():
            assert manifests[host][dependency] == final["files"][str(original / dependency)] == h
            recorded(original / dependency, h)
        rows = []
        with gzip.open(predp, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                assert row["split"] == "validation" and row["target_index"] == len(rows)
                rows.append(row)
        assert len(rows) == receipt["prediction_rows"] == population["target_count"]
        keys = [(r["entity_id"], r["target_seq"]) for r in rows]
        assert len(set(keys)) == len(keys)
        identity = hashlib.sha256(json.dumps(keys, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        q = np.array([r["raw_quantity"] for r in rows], dtype=np.float64)
        pred = np.array([r["predicted_raw_quantity"] for r in rows], dtype=np.float64)
        temporal = np.array([r["time_nll"] for r in rows], dtype=np.float64)
        history = np.array([r["history_length"] for r in rows], dtype=np.float64)
        qsha = hashlib.sha256(b"hard_lmm_target_quantity_v1\0" + q.astype("<f8").tobytes()).hexdigest()
        assert qsha == population["target_quantity_sha256"]
        pair = (identity, qsha)
        if job["dataset"] not in cohort_keys:
            cohort_keys[job["dataset"]] = pair
        assert cohort_keys[job["dataset"]] == pair
        assert all(np.isfinite(a).all() for a in (q, pred, temporal, history))
        cap = float(d["cap_value"])
        assert cap > 0 and (q >= 0).all() and (q <= cap).all() and (history >= 1).all()
        err = pred - q
        sse, ae = float(np.square(err).sum()), float(np.abs(err).sum())
        groups = {"overall": np.ones(len(rows), bool), "target_at_cap": q == cap,
                  "target_below_cap": q < cap,
                  "prediction_at_or_above_cap": pred >= cap, "prediction_below_cap": pred < cap,
                  "target_cap_prediction_under": (q == cap) & (pred < cap),
                  "target_below_cap_prediction_at_or_above_cap": (q < cap) & (pred >= cap)}
        calculated = {}
        for group, mask in groups.items():
            m = stats(q, pred, temporal, history, cap, mask, sse, ae)
            calculated[group] = m
            per_seed.append(dict(dataset=job["dataset"], model=job["arm"], seed=job["seed"], host=host,
                split="validation", group=group, cap_value=cap, total_count=len(rows),
                population_fraction=m["count"] / len(rows), selected_epoch=receipt["selected_epoch"], **m))
        for own, stored in (("overall", "overall"), ("target_at_cap", "cap_hit")):
            c, s = calculated[own], receipt["metrics"][stored]
            assert c["count"] == s["count"]
            for k in ("qty_rmse", "qty_mae", "qty_sse", "qty_absolute_error_sum", "time_nll"):
                assert math.isclose(c[k], s[k], rel_tol=1e-12, abs_tol=1e-10), (job_id, k)
            assert math.isclose(c["bias_prediction_minus_truth"], s["qty_bias"], rel_tol=1e-12, abs_tol=1e-10)
        capm, body = calculated["target_at_cap"], calculated["target_below_cap"]
        for k in ("count", "qty_sse", "qty_absolute_error_sum"):
            assert math.isclose(capm[k] + body[k], calculated["overall"][k], rel_tol=1e-12, abs_tol=1e-10)
        additive.append({"job_id": job_id, "cap_body_additivity": True, "saved_metrics_reproduced": True})
        bindings.append({"job_id": job_id, "dataset": job["dataset"], "model": job["arm"], "seed": job["seed"],
                         "host": host, "selected_epoch": receipt["selected_epoch"],
                         "checkpoint_sha256": receipt["checkpoint_sha256"], "state_sha256": receipt["state_sha256"],
                         "receipt_path": str(rp.relative_to(ROOT)), "receipt_sha256": expected_sha,
                         "prediction_path": str(predp.relative_to(ROOT)), "prediction_sha256": manifests[host][predrel],
                         "population_identity": population, "independent_entity_seq_identity_sha256": identity})
    assert len(bindings) == 36 and len({(x["dataset"], x["model"], x["seed"]) for x in bindings}) == 36
    numerical = [k for k in per_seed[0] if isinstance(per_seed[0][k], (float, int))
                 and k not in {"seed", "host", "selected_epoch", "cap_value", "total_count"}]
    summaries, contrasts = [], []
    for dataset in data:
        for model in ARMS:
            for group in groups:
                rr = sorted([r for r in per_seed if (r["dataset"], r["model"], r["group"]) == (dataset, model, group)], key=lambda r: r["seed"])
                assert [r["seed"] for r in rr] == list(SEEDS)
                summary = {"dataset": dataset, "model": model, "split": "validation", "group": group,
                           "cap_value": rr[0]["cap_value"], "seed_count": 3,
                           "available_seed_count": sum(r["count"] > 0 for r in rr),
                           "empty_seeds": ";".join(str(r["seed"]) for r in rr if not r["count"])}
                for k in numerical:
                    values = [r.get(k) for r in rr]
                    if any(v is None for v in values):
                        summary[k + "_mean"] = summary[k + "_sample_sd"] = None
                    else:
                        summary[k + "_mean"] = statistics.mean(values)
                        summary[k + "_sample_sd"] = statistics.stdev(values)
                summaries.append(summary)
            for seed in SEEDS:
                high = next(r for r in per_seed if (r["dataset"], r["model"], r["seed"], r["group"]) == (dataset, model, seed, "target_at_cap"))
                low = next(r for r in per_seed if (r["dataset"], r["model"], r["seed"], r["group"]) == (dataset, model, seed, "target_below_cap"))
                c = {"dataset": dataset, "model": model, "seed": seed, "split": "validation",
                     "comparison": "target_at_cap minus target_below_cap", "cap_count": high["count"], "body_count": low["count"]}
                for k in ("qty_rmse", "qty_mae", "bias_prediction_minus_truth", "underprediction_fraction"):
                    c["cap_minus_body_" + k] = high[k] - low[k]
                contrasts.append(c)
    for dataset in data:
        for model in ARMS:
            rr = [r for r in contrasts if r["dataset"] == dataset and r["model"] == model]
            s = {"dataset": dataset, "model": model, "split": "validation", "comparison": "target_at_cap minus target_below_cap", "seed": "summary", "seed_count": 3}
            for k in ("qty_rmse", "qty_mae", "bias_prediction_minus_truth", "underprediction_fraction"):
                v = [r["cap_minus_body_" + k] for r in rr]
                s["cap_minus_body_" + k + "_mean"] = statistics.mean(v)
                s["cap_minus_body_" + k + "_sample_sd"] = statistics.stdev(v)
                s["cap_worse_seeds_" + k] = ";".join(str(r["seed"]) for r in rr if r["cap_minus_body_" + k] > 0)
            contrasts.append(s)
    outputs = {"CAP_VALIDATION_PER_SEED.csv": per_seed, "CAP_VALIDATION_SUMMARY.csv": summaries,
               "CAP_VALIDATION_CAP_BODY_CONTRASTS.csv": contrasts}
    for name, rows in outputs.items():
        write_csv(OUT / name, rows)
    assert sha(Path(__file__)) == source_before
    result = {"status": "complete_validation_only_saved_prediction_analysis",
              "created_kst": dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(),
              "data_role": "User-designated official Q3-Cap baseline; previous raw data are pre-cap data.",
              "scope": "36 saved full Validation cohorts only; no Test prediction/metric files read",
              "input_sha256": INPUTS, "input_sha_verified_count": len(INPUTS), "cohort_bindings": bindings,
              "source_script_sha256": source_before, "population_keys": cohort_keys,
              "cohort_count": len(bindings), "prediction_rows_read": sum(x["population_identity"]["target_count"] for x in bindings),
              "per_seed_metric_rows": len(per_seed), "summary_rows": len(summaries),
              "additive_and_saved_metric_checks": additive,
              "definitions": {"cap_group": "target q equals fixed Train-Q3 cap; all capped targets, not original strict tail only",
                              "body_group": "target q strictly below fixed cap",
                              "underprediction": "prediction strictly less than saved target; no tolerance or clipping",
                              "relative_error": "mean absolute error divided per positive target; zero targets excluded with count retained",
                              "normalized_error": "RMSE/MAE/bias divided by cap; not pooled across datasets",
                              "prediction_cap_condition": "prediction >= cap, without post-hoc output clipping",
                              "mse_decomposition": "MSE = bias^2 + mean centered squared error within cohort",
                              "summary": "3seed arithmetic mean and sample SD(ddof1), not pooled predictions or an ensemble"},
              "limitations": ["Conditioning on the realized target q==cap is retrospective; it alone proves neither model bias conditional on input nor learnability or causality.",
                              "The cap group combines original q at the threshold and all above-threshold q; original magnitude cannot be recovered from these saved capped rows.",
                              "Larger absolute errors at larger target values need normalized and signed-error context; all seeds and below-cap exceptions are retained.",
                              "A cap point-mass target is paired with a head trained by squared log1p error; this analysis does not change that objective or prove it causes any observed pattern.",
                              "Saved rows provide history length, not preceding quantity/time values; no observed-transition conditioning or input-matched causal comparison is claimed.",
                              "Intermittent seeds42/52/62 run on5080/5090/5090 with cuDNN91900/92000/92000; Taxi5080 andRAF5090. Runtime provenance is inherited, not changed.",
                              "Archive bytes were hashed without extracting/reading Test rows; only metadata manifests and Validation prediction files were opened."],
              "outputs": {name: sha(OUT / name) for name in outputs},
              "new_training_count": 0, "new_inference_count": 0, "remote_calls": 0,
              "test_prediction_or_metric_files_opened": False, "historical_cnn_changed": False}
    with (OUT / "CAP_VALIDATION_RESULT.json").open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "cohort_count", "prediction_rows_read", "per_seed_metric_rows", "summary_rows", "input_sha_verified_count")}))


if __name__ == "__main__":
    main()
