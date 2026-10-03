"""Integrate three completed datasets from frozen validation JSON; no model execution."""

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
SNAPSHOT = ROOT / "monitor/20260910T233614Z_snapshot.json"
ARMS = ("joint", "quantity_only", "time_only")
DATASETS = ((1, "intermittent_frozen_5000", "Intermittent"),
            (2, "yellow_trip_hourly", "Taxi"),
            (3, "insta_market_basket", "Instacart"))
EXPOSURE = ("epoch", "global_step", "train_batch_order_sha256", "train_count",
            "train_batches", "validation_count", "validation_batches")
KST = timezone(timedelta(hours=9))


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return True


def comparison(joint, single, relative=True):
    return {"J": joint, "single_task": single, "single_minus_J": single - joint,
            "change_percent": (single / joint - 1) * 100 if relative else None}


def change_text(value):
    delta = value["single_minus_J"]
    direction = "악화" if delta > 0 else "개선" if delta < 0 else "동일"
    if value["change_percent"] is None:
        return f"{delta:+.6f} · {direction}"
    return f"{value['change_percent']:+.2f}% · {direction}"


def main():
    raw = SNAPSHOT.read_bytes()
    envelope = json.loads(raw)
    files = envelope["snapshot"]["files"]
    suite = files["suite_contract.json"]["json"]
    status = files["suite_run/status.json"]["json"]
    assert all(envelope["identity_checks"].values())
    assert suite == json.loads((ROOT / "remote_receipts/suite_contract.json").read_text())
    assert status["held_out_test_evaluated"] is False
    assert status["status"] == "complete"
    assert len(status["jobs"]) == 3
    datasets = []
    for index, dataset_id, label in DATASETS:
        prefix = f"suite_run/{index:02d}_{dataset_id}/"
        contract = files[f"contracts/{dataset_id}.json"]["json"]
        job = next(job for job in status["jobs"] if job["dataset_id"] == dataset_id)
        manifest = files[prefix + "wrapper_manifest.json"]["json"]
        paired = files[prefix + "paired_comparison_epoch_000120.json"]["json"]
        summaries = {arm: files[prefix + arm + "/summary.json"]["json"] for arm in ARMS}
        histories = {arm: files[prefix + arm + "/history.json"]["json"]["history"] for arm in ARMS}
        train_n = contract["populations"]["train"]["target_count"]
        val_n = contract["populations"]["validation"]["target_count"]
        train_batches = math.ceil(train_n / 128)
        val_batches = math.ceil(val_n / 128)
        assert job["status"] == "complete" and job["returncode"] == 0
        assert job["recomputed_equal_exposure_verified"] is True
        assert contract["seed"] == 42 and contract["epochs"] == 120
        assert manifest["evaluation_scope"] == "validation_only"
        assert manifest["held_out_materialized"] is False
        assert manifest["contract"] == contract
        assert paired["equal_training_exposure_verified"] is True
        assert paired["global_step"] == train_batches * 120
        assert job["wrapper_manifest_sha256"] == files[prefix + "wrapper_manifest.json"]["sha256"]
        assert job["paired_comparison_sha256"] == files[prefix + "paired_comparison_epoch_000120.json"]["sha256"]
        for arm in ARMS:
            summary, history = summaries[arm], histories[arm]
            assert summary["status"] == "complete"
            assert summary["epochs_completed"] == summary["epochs_budget"] == len(history) == 120
            assert summary["global_step"] == train_batches * 120
            assert summary["history"] == history and finite(history)
            assert summary["evaluation_scope"] == "validation_only"
            assert summary["held_out_test_evaluated"] is False
            assert summary["initial_state_sha256"] == manifest["initial_state_sha256"]
            assert job["arms"][arm]["summary_sha256"] == files[prefix + arm + "/summary.json"]["sha256"]
            assert files[prefix + arm + "/contract.json"]["json"]["identity"]["contract"] == contract
            for epoch, row in enumerate(history, 1):
                assert (row["epoch"], row["global_step"], row["train_count"], row["train_batches"],
                        row["validation_count"], row["validation_batches"]) == (
                    epoch, epoch * train_batches, train_n, train_batches, val_n, val_batches)
                assert all(row[field] == histories["joint"][epoch - 1][field] for field in EXPOSURE)
            for metric, selector in summary["selectors"].items():
                if selector["applicable"]:
                    best = min(history, key=lambda row: (row[metric], row["epoch"]))
                    assert (selector["best_value"], selector["best_epoch"], selector["global_step"]) == (
                        best[metric], best["epoch"], best["global_step"])
                else:
                    assert all(row[metric] is None for row in history)
                    assert selector["best_epoch"] is None and selector["best_value"] is None
        selected = {}
        for task, other, metric in (("quantity", "quantity_only", "raw_quantity_rmse"),
                                     ("time", "time_only", "legacy_time_loss")):
            pair = paired["paired_comparisons"][task]
            for key, arm in (("joint", "joint"), ("single_task", other)):
                assert all(pair[key][field] == value for field, value in summaries[arm]["selectors"][metric].items())
            j_epoch = pair["joint"]["best_epoch"]
            s_epoch = pair["single_task"]["best_epoch"]
            j_row, s_row = histories["joint"][j_epoch - 1], histories[other][s_epoch - 1]
            value = comparison(j_row[metric], s_row[metric], task == "quantity")
            assert value["single_minus_J"] == pair["single_task_minus_joint"]
            selected[task] = {"J_epoch": j_epoch, "single_epoch": s_epoch, "metric": metric,
                              "comparison": value, "J_state_sha256": pair["joint"]["state_sha256"],
                              "single_state_sha256": pair["single_task"]["state_sha256"]}
            if task == "quantity":
                selected[task]["mae_at_rmse_selector"] = comparison(j_row["quantity_mae"], s_row["quantity_mae"])
        same_epoch = []
        for i in range(120):
            j, q, t = (histories[arm][i] for arm in ARMS)
            same_epoch.append({"epoch": i + 1, "global_step_per_arm": j["global_step"],
                               "raw_quantity_rmse": comparison(j["raw_quantity_rmse"], q["raw_quantity_rmse"]),
                               "quantity_mae": comparison(j["quantity_mae"], q["quantity_mae"]),
                               "legacy_time_loss": comparison(j["legacy_time_loss"], t["legacy_time_loss"], False)})
        clipping = {}
        for arm, history in histories.items():
            count = sum(row["train_clipped_batch_count"] for row in history)
            clipping[arm] = {"count": count, "total_batches": train_batches * 120,
                             "percent": count / (train_batches * 120) * 100,
                             "first_10_epoch_percent": sum(row["train_clipped_batch_count"] for row in history[:10]) / (train_batches * 10) * 100,
                             "last_10_epoch_percent": sum(row["train_clipped_batch_count"] for row in history[-10:]) / (train_batches * 10) * 100,
                             "by_epoch": [row["train_clipped_batch_count"] for row in history]}
        trajectories = []
        phases = {}
        for arm, history in histories.items():
            for row in history:
                trajectories.append({
                    "dataset": label, "arm": arm, "epoch": row["epoch"],
                    "global_step": row["global_step"],
                    "validation_raw_rmse": row["raw_quantity_rmse"],
                    "validation_mae": row["quantity_mae"],
                    "validation_legacy_time_loss": row["legacy_time_loss"],
                    "validation_quantity_log_mse": row["quantity_train_loss"],
                    "train_quantity_log_mse": row["train"]["quantity_train_loss"],
                    "train_legacy_time_loss": row["train"]["legacy_time_loss"],
                    "clipping_percent": row["train_clipped_batch_count"] / row["train_batches"] * 100,
                    "preclip_norm_mean": row["train_global_preclip_norm_mean"],
                })
            phases[arm] = {}
            for start, end in [(1, 10), (81, 100), (101, 120), (111, 120)]:
                rows = [x for x in trajectories if x["arm"] == arm and start <= x["epoch"] <= end]
                metric_names = ["validation_raw_rmse", "validation_mae", "validation_legacy_time_loss",
                                "validation_quantity_log_mse", "train_quantity_log_mse",
                                "train_legacy_time_loss", "clipping_percent", "preclip_norm_mean"]
                phases[arm][f"{start}-{end}"] = {
                    metric: (sum(x[metric] for x in rows) / len(rows)
                             if all(x[metric] is not None for x in rows) else None)
                    for metric in metric_names}
        learning_late_change = {}
        for arm in ARMS:
            a, b = phases[arm]["81-100"], phases[arm]["101-120"]
            learning_late_change[arm] = {
                key: ({"mean_epoch81_100": a[key], "mean_epoch101_120": b[key],
                       "difference": b[key] - a[key],
                       "change_percent": (b[key] / a[key] - 1) * 100 if a[key] != 0 and "time_loss" not in key else None}
                      if a[key] is not None and b[key] is not None else None)
                for key in a}
        selector_to_final = {}
        for arm, history in histories.items():
            selector_to_final[arm] = {}
            for metric, selector in summaries[arm]["selectors"].items():
                if selector["applicable"]:
                    selector_to_final[arm][metric] = {
                        "best_epoch": selector["best_epoch"], "best": selector["best_value"],
                        "final": history[-1][metric], "delta": history[-1][metric] - selector["best_value"],
                        "relative_percent": (history[-1][metric] / selector["best_value"] - 1) * 100 if metric == "raw_quantity_rmse" else None}
        datasets.append({"dataset_id": dataset_id, "label": label,
                         "finished_at_kst": datetime.fromtimestamp(job["finished_at_unix"], KST).isoformat(),
                         "train_targets_per_epoch": train_n, "validation_targets": val_n,
                         "train_batches_per_epoch": train_batches, "validation_batches": val_batches,
                         "steps_per_arm": train_batches * 120, "audit_status": "passed",
                         "initial_state_sha256": manifest["initial_state_sha256"],
                         "selected_checkpoint_comparison": selected,
                         "same_epoch_comparison": same_epoch, "clipping": clipping,
                         "trajectories": trajectories, "phases": phases,
                         "late_phase_change": learning_late_change, "selector_to_final": selector_to_final,
                         "evidence_sha256": {path: value["sha256"] for path, value in files.items()
                                             if path.startswith(prefix) and "sha256" in value}})
    result = {"assessment": "Share with caveats", "scope": "seed42 validation-only, static B J/Q/T, three completed datasets",
              "source_revision": suite["source"]["revision"], "source_snapshot": str(SNAPSHOT),
              "source_snapshot_sha256": hashlib.sha256(raw).hexdigest(),
              "suite_finished_at_kst": datetime.fromtimestamp(status["finished_at_unix"], KST).isoformat(),
              "source_evidence_root": str(ROOT / "monitor/20260910T233614Z_terminal"),
              "total_arms": 9, "total_optimizer_steps": sum(d["steps_per_arm"] * 3 for d in datasets),
              "observed_at_kst": datetime.fromisoformat(envelope["snapshot"]["observed_at_utc"]).astimezone(KST).isoformat(),
              "independent_review": "Original completion audit passed; integration review receipt is recorded separately after independent recomputation.",
              "datasets": datasets, "new_model_evaluation": False, "held_out_evaluated": False,
              "metric_notes": {"quantity": "original quantity scale, lower is better; percentage=(single/J-1)*100",
                               "time": "legacy clamped loss, not normalized NLL; report absolute delta only",
                               "selected_mae": "MAE at each raw-RMSE-selected checkpoint, not MAE-selected",
                               "same_epoch": "all 120 paired boundaries retained; final epoch120 is separate from primary selectors",
                               "phase_means": "Descriptive post-hoc unweighted means over stated epochs; no statistical replicates. Fixed per-epoch population.",
                               "train_validation_names": "history row quantity_train_loss is validation log-MSE; nested train.quantity_train_loss is actual training log-MSE.",
                               "clipping": "Percent of training batches with preclip gradient norm >1. Not parameter-update suppression or evidence of a causal mechanism."}}
    (OUT / "validation_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "verified extraction", "datasets": len(datasets),
                      "completed_arms": 9, "output": str(OUT / "validation_results.json"),
                      "source_snapshot_sha256": result["source_snapshot_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
