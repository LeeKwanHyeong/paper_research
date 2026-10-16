#!/usr/bin/env python3
"""Local-only, frozen-checkpoint CNN/GRU comparison; never runs inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / "reports/titantpp_three_dataset_final_comparison_20261005_v1"
DATA = {
    "yellow_trip_hourly": {"label": "Taxi", "threshold": 3449, "validation": (8268, 79), "test": (8327, 83)},
    "intermittent_frozen_5000": {"label": "Intermittent", "threshold": 187, "validation": (86285, 1141), "test": (88019, 243)},
    "raf_spare_parts": {"label": "RAF", "threshold": 200, "validation": (6690, 50), "test": (5226, 39)},
}
NEW = {"titantpp_cnn_mlp16", "titantpp_gru54", "titantpp_cnn_gru54"}
EXTERNAL = {"rmtpp", "thp", "nhp", "sahp", "s2p2_matched_head", "attnhp_matched_head"}
QUANTITY_ONLY = {"last_observed_quantity", "mean_quantity_in_same_observed_window"}
NATIVE = {"deep_renewal_event_native_nb"}
LABEL = {
    "titantpp_history_mlp": "MLP4", "titantpp_history_mlp_width8": "MLP8",
    "titantpp_history_mlp_width12": "MLP12", "titantpp_history_mlp_width16": "MLP16",
    "titantpp": "TitanTPP B", "titantpp_all_available_history_mlp": "모든 가용 분기 MLP",
    "titantpp_current_only_param_matched": "현재 상태만 · 파라미터 대조",
    "rmtpp": "RMTPP", "thp": "THP", "nhp": "NHP", "sahp": "SAHP",
    "s2p2_matched_head": "S2P2 · 공통 출력부", "attnhp_matched_head": "AttNHP · 공통 출력부",
    "deep_renewal_event_native_nb": "Deep Renewal native NB · 원고 제외",
    "last_observed_quantity": "마지막 관측 수량", "mean_quantity_in_same_observed_window": "관측 이력 평균",
    "titantpp_cnn_mlp16": "CNN+MLP16", "titantpp_gru54": "GRU54", "titantpp_cnn_gru54": "CNN+GRU54",
}
METRICS = ("qty_rmse", "qty_mae", "time_nll")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def num(value):
    if value in (None, ""):
        return None
    v = float(value)
    require(math.isfinite(v), "Nonfinite metric")
    return v


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5)


def family(model):
    if model in NATIVE:
        return "native_head_research_only_excluded_from_primary_rank"
    if model in QUANTITY_ONLY:
        return "deterministic_quantity_only_separate_rank"
    if model in EXTERNAL:
        return "external_matched_head"
    if model in NEW:
        return "new_frozen_seed42_single_seed_mixed_GPU"
    return "internal_capacity_or_structure_control"


def source_aggregate(obj, split):
    """Return direct aggregates when present; legacy prediction receipts have none."""
    if "selected" in obj:
        require(split == "validation", "Endpoint cannot supply Test")
        aggregate = obj["selected"]
        require(aggregate["evaluation_scope"] == "validation_only"
                and aggregate["held_out_test_evaluated"] is False, "Endpoint split differs")
        return aggregate
    require(obj["status"] == "complete" and obj["split"] == split
            and obj["full_population"] is True, "Full population receipt required")
    return obj.get("metrics")


def prepare_baseline():
    receipt = read(BASE / "comparison_receipt.json")
    inputs = {name: sha(BASE / name) for name in ("metrics_per_seed.csv", "metrics_aggregated.csv", "selected_epochs.csv")}
    require(all(receipt["outputs"][name] == digest for name, digest in inputs.items()), "Baseline output SHA changed")
    with (BASE / "metrics_per_seed.csv").open() as f:
        original = list(csv.DictReader(f))
    with (BASE / "metrics_aggregated.csv").open() as f:
        supplied_aggregates = list(csv.DictReader(f))
    with (BASE / "selected_epochs.csv").open() as f:
        selected_epochs = list(csv.DictReader(f))
    require(len(original) == 516 and len(selected_epochs) == 123, "Baseline coverage differs")
    sources = {}
    rows = []
    seen = set()
    direct_cells = 0
    for original_row in original:
        row = dict(original_row)
        key = tuple(row[k] for k in ("dataset", "model", "seed", "split", "view"))
        require(key not in seen, "Duplicate baseline metric row")
        seen.add(key)
        d, split, view = row["dataset"], row["split"], row["view"]
        require(d in DATA and split in ("validation", "test") and view in ("overall", "tail"), "Unexpected scope")
        count = int(row["count"])
        require(count == DATA[d][split][view == "tail"], "Baseline target count differs")
        if view == "tail":
            require(float(row["threshold"]) == DATA[d]["threshold"], "Train tail threshold differs")
        p = path(row["source"])
        if str(p) not in sources:
            obj = read(p)
            digest = sha(p)
            sources[str(p)] = {"sha256": digest, "object": obj}
        source = sources[str(p)]
        require(source["sha256"] == row["source_sha256"], "Baseline source SHA differs: " + str(p))
        aggregate = source_aggregate(source["object"], split)
        if "selected" not in source["object"]:
            population = receipt["population_references"][d][split]
            require(all(source["object"].get(k) == population.get(k)
                        for k in ("target_identity_sha256", "truth_sha256", "data_file_sha256", "loader")),
                    "Baseline full-population identity/truth/loader differs")
        row.update({"seed": int(row["seed"]) if row["seed"] else None, "count": count,
                    "selected_epoch": int(row["selected_epoch"]) if row["selected_epoch"] else None,
                    "threshold": DATA[d]["threshold"] if view == "tail" else None,
                    "family": family(row["model"]), "source": str(p), "new_model": False})
        row.update({k: num(row[k]) for k in METRICS})
        require(row["qty_rmse"] >= 0 and row["qty_mae"] >= 0, "Negative quantity error")
        if aggregate is not None:
            scope_aggregate = aggregate if view == "overall" else aggregate["tail"]
            require(scope_aggregate["count"] == count, "Direct aggregate count differs")
            if row["model"] not in QUANTITY_ONLY:
                require(all(close(row[k], scope_aggregate[k]) for k in METRICS), "Direct baseline metric differs")
                direct_cells += 3
        else:
            require(source["object"]["expected_target_count"] == DATA[d][split][0], "Legacy full receipt target differs")
        rows.append(row)
    models = sorted({r["model"] for r in rows})
    require(len(models) == 16 and EXTERNAL <= set(models), "Baseline model coverage differs")
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in ("dataset", "model", "split", "view"))].append(row)
    aggregates = []
    for supplied in supplied_aggregates:
        key = tuple(supplied[k] for k in ("dataset", "model", "split", "view"))
        group = groups[key]
        require(int(supplied["seed_count"]) == (0 if group[0]["seed"] is None else len(group)), "Seed count differs")
        agg = {k: supplied[k] for k in ("dataset", "model", "split", "view")}
        agg.update({"seed_count": len(group) if group[0]["seed"] is not None else 0,
                    "count_per_seed": group[0]["count"], "family": family(group[0]["model"])})
        if group[0]["seed"] is not None:
            require({r["seed"] for r in group} == {42, 52, 62}, "Incomplete 3seed baseline")
        for metric in METRICS:
            values = [r[metric] for r in group if r[metric] is not None]
            mean = statistics.mean(values) if values else None
            sd = statistics.stdev(values) if len(values) > 1 else None
            expected_mean = num(supplied[metric + "_mean"])
            expected_sd = num(supplied[metric + "_sample_sd"])
            require((mean is None and expected_mean is None) or mean is not None and close(mean, expected_mean), "Mean differs")
            require((sd is None and expected_sd is None) or sd is not None and close(sd, expected_sd), "Sample SD differs")
            agg[metric + "_mean"], agg[metric + "_sample_sd"] = mean, sd
        aggregates.append(agg)
    inventory = {
        "schema": "cnn_gru_baseline_inventory_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT), "baseline_report": str(BASE), "baseline_comparison_receipt_sha256": sha(BASE / "comparison_receipt.json"),
        "input_files_sha256": inputs, "row_count": len(rows), "unique_model_count": len(models), "models": models,
        "selected_epoch_condition_count": len(selected_epochs), "source_file_count": len(sources),
        "source_receipt_SHA_all_verified": True, "direct_metric_cells_checked": direct_cells,
        "full_population_receipt_identity_truth_data_loader_all_verified": True,
        "population_references": receipt["population_references"],
        "Validation_endpoint_limit": "Frozen endpoints omit per-target truth/identity digests; preserve original reference linkage, no new raw prediction reaggregation claimed",
        "legacy_without_aggregate": "Metrics reused from SHA-bound published aggregate; raw predictions not recomputed here",
        "populations": DATA, "tail_rule": "original TRAIN threshold, raw quantity strictly greater",
        "aggregation": "within-seed event aggregate; 42/52/62 arithmetic mean and sample SD ddof=1; no ensemble",
        "all_per_seed_rows": rows,
        "seed42_rows": [r for r in rows if r["seed"] in (42, None)], "three_seed_reference": aggregates,
        "selected_epochs": selected_epochs,
        "source_files": [{"path": p, "sha256": v["sha256"]} for p, v in sorted(sources.items())],
        "missing": [{"dataset": "raf_spare_parts", "model": "titantpp", "reason": "No comparable original; preserve missing"}],
        "evaluation_or_training_executed": False,
    }
    write("baseline_inventory.json", inventory)
    return inventory


def record_job(record):
    job = record.get("job", record)
    return job["dataset"], job.get("arm", job.get("model")), int(job["seed"])


def load_new(results_path):
    """Supports exact source paths or explicitly SHA-bound embedded aggregates."""
    result = read(results_path)
    records = result["records"]
    require(len(records) == 9, "All 9 new frozen conditions are required; no favorable subset")
    rows, bindings = [], []
    seen = set()
    for record in records:
        dataset, model, seed = record_job(record)
        require((dataset, model, seed) not in seen, "Duplicate new scientific condition")
        seen.add((dataset, model, seed))
        require(dataset in DATA and model in NEW and seed == 42, "New scientific scope differs")
        split_info = {}
        for split in ("validation", "test"):
            info = record[split]
            if isinstance(info, str):
                info = {"path": info, "sha256": record[split + "_sha256"]}
            source_path = info.get("path", info.get("source_path", record.get(split + "_path")))
            expected = info.get("sha256", info.get("source_sha256", record.get(split + "_sha256")))
            require(source_path and expected, "New result must bind exact source JSON and SHA")
            p = path(source_path)
            require(sha(p) == expected, "New source SHA differs")
            obj = read(p)
            require(obj.get("status") == "complete", "New result is not complete")
            if "job" in obj:
                actual_dataset, actual_model, actual_seed = record_job(obj)
            else:
                actual_dataset, actual_model, actual_seed = obj["dataset"], obj["model"], int(obj["seed"])
            require((actual_dataset, actual_model, actual_seed) == (dataset, model, seed),
                    "New source scientific condition differs")
            aggregate = source_aggregate(obj, split)
            require(aggregate is not None, "New exact aggregate missing")
            epoch = obj.get("best_epoch", obj.get("selected_epoch"))
            state = aggregate.get("state_sha256", obj.get("state_tensor_sha256"))
            require(epoch is not None and state, "New selected checkpoint binding missing")
            require(aggregate["count"] == DATA[dataset][split][0]
                    and aggregate["tail"]["count"] == DATA[dataset][split][1], "New full population or tail count differs")
            if "quantity_boundaries" in aggregate:
                require(aggregate["quantity_boundaries"][-1] == DATA[dataset]["threshold"],
                        "New Train tail threshold differs")
            if "checkpoint_file_sha256" in obj:
                require(record.get("checkpoint_file_sha256", obj["checkpoint_file_sha256"]) == obj["checkpoint_file_sha256"],
                        "New source selected checkpoint binary differs")
            if record.get("selected_epoch") is not None:
                require(epoch == record["selected_epoch"], "New source selected epoch changed")
            if record.get("state_tensor_sha256") is not None:
                require(state == record["state_tensor_sha256"], "New source selected tensor changed")
            split_info[split] = {"path": str(p), "sha256": expected, "selected_epoch": epoch, "state_tensor_sha256": state}
            for view in ("overall", "tail"):
                v = aggregate if view == "overall" else aggregate["tail"]
                row = {"dataset": dataset, "model": model, "seed": 42, "split": split, "view": view,
                       "count": v["count"], "threshold": DATA[dataset]["threshold"] if view == "tail" else None,
                       "selected_epoch": epoch, "source": str(p), "source_sha256": expected,
                       "family": family(model), "new_model": True}
                row.update({k: num(v[k]) for k in METRICS})
                require(all(row[k] is not None for k in METRICS), "New metric missing")
                rows.append(row)
        require(split_info["validation"]["selected_epoch"] == split_info["test"]["selected_epoch"]
                and split_info["validation"]["state_tensor_sha256"] == split_info["test"]["state_tensor_sha256"],
                "Validation/Test selected checkpoint differs")
        bindings.append({"dataset": dataset, "model": model, "seed": 42, **split_info})
    require(seen == {(d, m, 42) for d in DATA for m in NEW}, "New condition coverage differs")
    return rows, bindings


def dominates(a, b, metrics):
    return all(a[k] <= b[k] for k in metrics) and any(a[k] < b[k] for k in metrics)


def factorial_contrasts(rows):
    """Descriptive 2x2 contrasts of four whole structures, never causal attribution."""
    index = {(r["dataset"], r["split"], r["view"], r["model"]): r for r in rows}
    contrasts = []
    for dataset in DATA:
        for split in ("validation", "test"):
            for view in ("overall", "tail"):
                arms = {name: index[(dataset, split, view, model)] for name, model in {
                    "mlp16": "titantpp_history_mlp_width16", "cnn_mlp16": "titantpp_cnn_mlp16",
                    "gru54": "titantpp_gru54", "cnn_gru54": "titantpp_cnn_gru54"}.items()}
                require(all(r["seed"] == 42 for r in arms.values()), "2x2 requires paired seed42")
                for metric in METRICS:
                    base, cnn, gru, combined = (arms[k][metric] for k in ("mlp16", "cnn_mlp16", "gru54", "cnn_gru54"))
                    contrasts.append({"dataset": dataset, "split": split, "view": view, "seed": 42, "metric": metric,
                        "mlp16": base, "cnn_mlp16": cnn, "gru54": gru, "cnn_gru54": combined,
                        "cnn_given_mlp_delta": cnn - base, "gru_without_cnn_delta": gru - base,
                        "cnn_given_gru_delta": combined - gru, "gru_given_cnn_delta": combined - cnn,
                        "combined_vs_mlp16_delta": combined - base,
                        "additive_error_reference": cnn + gru - base,
                        "interaction": combined - cnn - gru + base,
                        "interpretation": "descriptive_whole_structure_single_seed_interaction_not_causal_or_confirmed_synergy"})
    return contrasts


def analyze_rows(rows):
    """No cross-dataset normalization, weighting, significance or model refitting."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["split"], row["view"])].append(row)
    ranks, deltas, pareto, recommendations = [], [], [], []
    for (dataset, split, view), group in sorted(groups.items()):
        primary = [r for r in group if r["model"] not in NATIVE | QUANTITY_ONLY]
        external = [r for r in group if r["model"] in EXTERNAL]
        by_model = {r["model"]: r for r in group}
        for metric in METRICS:
            ordered = sorted(primary, key=lambda r: (r[metric], r["model"]))
            ranks.append({"dataset": dataset, "split": split, "view": view, "metric": metric,
                          "scope": "learned_primary_matched_head", "ordered": [{"model": r["model"], "value": r[metric],
                           "rank": 1 + sum(x[metric] < r[metric] for x in primary)} for r in ordered]})
            best_external = min(external, key=lambda r: (r[metric], r["model"]))
            for model in sorted(NEW):
                new = by_model[model]
                for ref in ("titantpp_history_mlp_width16", "s2p2_matched_head", "metric_best_external"):
                    baseline = best_external if ref == "metric_best_external" else by_model[ref]
                    delta = new[metric] - baseline[metric]
                    deltas.append({"dataset": dataset, "split": split, "view": view, "metric": metric,
                                   "model": model, "reference": ref, "reference_model": baseline["model"],
                                   "new_value": new[metric], "reference_value": baseline[metric], "absolute_delta": delta,
                                   "relative_percent": 100 * delta / baseline[metric] if metric != "time_nll" and baseline[metric] > 0 else None,
                                   "improved": delta < 0, "comparison": "same_seed42_frozen_selected_full_population"})
        for dimensions in (("qty_rmse", "qty_mae"), METRICS):
            pareto.append({"dataset": dataset, "split": split, "view": view, "metrics": list(dimensions),
                           "scope": "learned_primary_matched_head", "models": [r["model"] for r in primary
                            if not any(dominates(other, r, dimensions) for other in primary)]})
    # A four-objective front combines overall and tail RMSE, overall MAE and NLL.
    for dataset in DATA:
        split_fronts = {}
        for split in ("validation", "test"):
            by_view = {view: {r["model"]: r for r in groups[(dataset, split, view)]} for view in ("overall", "tail")}
            vectors = [{"model": m, "overall_rmse": o["qty_rmse"], "tail_rmse": by_view["tail"][m]["qty_rmse"],
                        "overall_mae": o["qty_mae"], "overall_time_nll": o["time_nll"]}
                       for m, o in by_view["overall"].items() if m not in NATIVE | QUANTITY_ONLY]
            dimensions = ("overall_rmse", "tail_rmse", "overall_mae", "overall_time_nll")
            front = [v["model"] for v in vectors if not any(dominates(other, v, dimensions) for other in vectors)]
            split_fronts[split] = front
            pareto.append({"dataset": dataset, "split": split, "view": "overall_plus_tail", "metrics": list(dimensions),
                           "scope": "learned_primary_matched_head", "models": front})
        val_overall = groups[(dataset, "validation", "overall")]
        test_overall = groups[(dataset, "test", "overall")]
        candidate = min((r for r in val_overall if r["model"] in NEW), key=lambda r: (r["qty_rmse"], r["model"]))
        test_winner = min((r for r in test_overall if r["model"] not in NATIVE | QUANTITY_ONLY), key=lambda r: (r["qty_rmse"], r["model"]))
        test_new_winner = min((r for r in test_overall if r["model"] in NEW), key=lambda r: (r["qty_rmse"], r["model"]))
        direction = {}
        for model in sorted(NEW):
            paired = [d for d in deltas if d["dataset"] == dataset and d["model"] == model
                      and d["reference"] == "titantpp_history_mlp_width16"]
            val = {(d["view"], d["metric"]): d["absolute_delta"] for d in paired if d["split"] == "validation"}
            test = {(d["view"], d["metric"]): d["absolute_delta"] for d in paired if d["split"] == "test"}
            direction[model] = [{"view": k[0], "metric": k[1], "validation_delta": v, "test_delta": test[k],
                                 "test_relation": "same_improvement_direction" if v < 0 and test[k] < 0 else
                                  "same_worsening_direction" if v > 0 and test[k] > 0 else "split_direction_differs_or_tie"}
                                for k, v in sorted(val.items())]
        recommendations.append({"dataset": dataset, "validation_quantity_candidate": candidate["model"],
                                "validation_four_objective_front": split_fronts["validation"],
                                "test_four_objective_front": split_fronts["test"],
                                "observed_test_overall_RMSE_winner": test_winner["model"],
                                "observed_test_overall_RMSE_best_new_candidate": test_new_winner["model"],
                                "MLP16_direction_consistency": direction,
                                "status": "posthoc_single_seed_exploratory_followup_only",
                                "not_confirmed": ["new3seed_stability", "statistical_superiority", "unaccessed_independent_holdout", "scientific_efficiency"],
                                "selection_consequence": "If Test informs further architecture selection, this Test is development-exposed; require new independent evaluation for confirmatory claims"})
    return {"rankings": ranks, "paired_deltas": deltas, "pareto_fronts": pareto, "recommendations": recommendations,
            "factorial_contrasts": factorial_contrasts(rows)}


def fmt(value):
    if value is None:
        return "—"
    return f"{value:.4e}" if value and abs(value) < 0.001 else f"{value:.6f}"


def write_csv(name, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rank_csv_rows(rows, analysis):
    rank = {(r["dataset"], r["split"], r["view"], r["metric"], entry["model"]): entry["rank"]
            for r in analysis["rankings"] for entry in r["ordered"]}
    output = []
    for row in rows:
        r = {k: row[k] for k in ("dataset", "model", "seed", "split", "view", "count", "selected_epoch", "family", *METRICS, "source", "source_sha256")}
        for metric in METRICS:
            r["rank_" + metric] = rank.get((row["dataset"], row["split"], row["view"], metric, row["model"]))
        r["display_rank_overall_qty_rmse"] = rank.get((row["dataset"], row["split"], "overall", "qty_rmse", row["model"]))
        output.append(r)
    return sorted(output, key=lambda r: (r["dataset"], r["split"], r["view"],
                  r["display_rank_overall_qty_rmse"] is None, r["display_rank_overall_qty_rmse"] or 0, r["model"]))


def render(rows, inventory, analysis):
    for split in ("validation", "test"):
        lines = [f"# {split.title()} · 동결 checkpoint seed42 비교", "",
                 "새 CNN·GRU 세 후보는 단일 seed42다. 기존 동일 seed42와 비교하며 3seed 참고 평균과 혼합하지 않는다.", "",
                 "모든 수치는 낮을수록 좋다. 주비교 표는 전체 수량 RMSE 오름차순이다. tail 표도 같은 전체 RMSE 순서를 유지하며 tail 자체 순위는 순위 CSV에 별도로 저장했다.", "",
                 "Deep Renewal native NB와 수량만 예측하는 통계 기준선은 별도 표에 보존하며 주순위/Pareto에 포함하지 않는다.", ""]
        for dataset in DATA:
            lines += [f"## {DATA[dataset]['label']}", ""]
            overall = {r["model"]: r for r in rows if r["dataset"] == dataset and r["split"] == split and r["view"] == "overall"}
            primary = sorted((r for r in overall.values() if r["model"] not in NATIVE | QUANTITY_ONLY),
                             key=lambda r: (r["qty_rmse"], r["model"]))
            primary_order = {r["model"]: 1 + sum(other["qty_rmse"] < r["qty_rmse"] for other in primary) for r in primary}
            for view in ("overall", "tail"):
                lines += [f"### {view} · target {DATA[dataset][split][view == 'tail']:,}", "",
                          "| 전체 RMSE 순위 | 모델 | 선택 epoch | RMSE | MAE | Time NLL |",
                          "|---:|---|---:|---:|---:|---:|"]
                group = sorted((r for r in rows if r["dataset"] == dataset and r["split"] == split and r["view"] == view),
                               key=lambda r: (r["model"] not in primary_order, primary_order.get(r["model"], 0), r["model"]))
                for r in (row for row in group if row["model"] in primary_order):
                    lines.append(f"| {primary_order[r['model']]} | {LABEL[r['model']]} | {r['selected_epoch'] or '—'} | {fmt(r['qty_rmse'])} | {fmt(r['qty_mae'])} | {fmt(r['time_nll'])} |")
                lines += ["", "다른 출력부 또는 수량 전용 참고 기준선:", "",
                          "| 모델 | RMSE | MAE | Time NLL |", "|---|---:|---:|---:|"]
                for r in (row for row in group if row["model"] not in primary_order):
                    lines.append(f"| {LABEL[r['model']]} | {fmt(r['qty_rmse'])} | {fmt(r['qty_mae'])} | {fmt(r['time_nll'])} |")
                lines.append("")
        (OUT / (split.upper() + "_SEED42.md")).write_text("\n".join(lines) + "\n")
    lines = ["# 기존 3seed 평균 ± 표본 SD · 참고 대조", "", "신규9조건은 여기에 포함되지 않는다. seed42 주비교와 독립적인 참고표다.", ""]
    for dataset in DATA:
        for split in ("validation", "test"):
            for view in ("overall", "tail"):
                lines += [f"## {DATA[dataset]['label']} · {split} · {view}", "", "| 모델 | seed수 | RMSE | MAE | Time NLL |", "|---|---:|---:|---:|---:|"]
                for r in inventory["three_seed_reference"]:
                    if (r["dataset"], r["split"], r["view"]) != (dataset, split, view):
                        continue
                    values = [fmt(r[m + "_mean"]) + (" ± " + fmt(r[m + "_sample_sd"]) if r[m + "_sample_sd"] is not None else "") for m in METRICS]
                    lines.append(f"| {LABEL[r['model']]} | {r['seed_count'] or '결정적1회'} | " + " | ".join(values) + " |")
                lines.append("")
    (OUT / "BASELINE_THREE_SEED_REFERENCE.md").write_text("\n".join(lines) + "\n")
    lines = ["# 신규 후보의 이득·손해와 후속 연구 판단", "", "선택 epoch는 Validation에서 동결됐다. Test 순위는 사후 탐색 결과이며 이후 모델 선택에 사용하면 해당 Test의 독립 평가 자격을 유지할 수 없다.", "", "## 데이터별 관측", "", "| 데이터 | 신규 중 Validation RMSE 최저 | 전체 Test RMSE 관측1위 | 신규 중 Test RMSE 최저 | Validation 4목적 Pareto의 신규 후보 |", "|---|---|---|---|---|"]
    for r in analysis["recommendations"]:
        candidates = [LABEL[m] for m in r["validation_four_objective_front"] if m in NEW]
        lines.append(f"| {DATA[r['dataset']]['label']} | {LABEL[r['validation_quantity_candidate']]} | {LABEL[r['observed_test_overall_RMSE_winner']]} | {LABEL[r['observed_test_overall_RMSE_best_new_candidate']]} | {', '.join(candidates) or '없음'} |")
    lines += ["", "## 동일 seed42에서 MLP16·S2P2·지표별 최선 외부 모델과 비교", "",
              "NLL은 절대 차이만 표시한다. 큰 수량 기준은 원래 TRAIN의 엄격 초과이며 Test에서 다시 정하지 않았다.", "",
              "| 데이터 | split | 범위 | 신규 | 지표 | 기준 | 절대 Δ | 수량 변화율 |", "|---|---|---|---|---|---|---:|---:|"]
    for d in analysis["paired_deltas"]:
        ref = LABEL[d["reference_model"]] + (" · 지표최선외부" if d["reference"] == "metric_best_external" else "")
        lines.append(f"| {DATA[d['dataset']]['label']} | {d['split']} | {d['view']} | {LABEL[d['model']]} | {d['metric']} | {ref} | {fmt(d['absolute_delta'])} | {fmt(d['relative_percent']) + '%' if d['relative_percent'] is not None else '—'} |")
    lines += ["", "## 해석 범위", "", "- 신규3seed 안정성과 통계적 우월성은 미확인이다. 모든 불리한 지표와 비교군을 보존했다.",
              "- CNN은 Encoder1 Q/K/V 국소 경로 추가이며 GRU는 중간 보정 구조 전체 변경이다. 연산 하나의 독립 기여로 단정하지 않는다.",
              "- 수량·tail·시간의 목적이 다르므로 지표를 임의 가중 합산하거나 데이터 간 NLL을 합산하지 않는다.",
              "- Test 관측1위를 후속 탐색 후보로 고를 수 있으나, 이를 최종 모델 우월성이나 독립 Test 검증으로 주장할 수 없다.",
              "- 후속 구조의 기여는 고정 용량/출력부/손실/선택 규칙의 CNN만·GRU만·결합 대조와 추가 seed에서 검증해야 한다.",
              "- 원본 binary 회수·CPU 감사·속도 효율 측정은 이 로컬 보고서 생성과 별도다."]
    (OUT / "RECOMMENDATION.md").write_text("\n".join(lines) + "\n")
    lines = ["# MLP16·CNN·GRU·결합의 동일 seed42 구조 대조", "",
             "MLP16, CNN+MLP16, GRU54, CNN+GRU54를 같은 split·사건 집합에서 비교했다. 각 checkpoint의 선택은 Validation에서 동결되어 있다.", "",
             "Interaction = 결합 − CNN+MLP16 − GRU54 + MLP16. 낮을수록 좋은 오차에서 음수는 산술적 가산 기준보다 결합 오차가 낮다는 뜻이다. 이 수치는 단일 seed의 기술적 대조이며 확정 synergy나 recurrence만의 독립 인과효과가 아니다.", "",
             "GRU는 중간 보정 전체·참조 범위·활성화·잔차 크기를 바꾸며, 두 보정의 파라미터 수는 근접하지만 동일하지 않다. 이종 GPU와 사전 허용오차 안의 replay 차이도 유지한다.", "",
             "| 데이터 | split | 범위 | 지표 | MLP16 | CNN+MLP16 | GRU54 | CNN+GRU54 | CNN Δ | GRU Δ | 결합 Δ | Interaction |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in analysis["factorial_contrasts"]:
        values = [fmt(r[k]) for k in ("mlp16", "cnn_mlp16", "gru54", "cnn_gru54", "cnn_given_mlp_delta", "gru_without_cnn_delta", "combined_vs_mlp16_delta", "interaction")]
        lines.append(f"| {DATA[r['dataset']]['label']} | {r['split']} | {r['view']} | {r['metric']} | " + " | ".join(values) + " |")
    lines += ["", "전체/큰 수량 RMSE·MAE·Time NLL을 각자 유지하며 임의 합산하지 않았다. 조건부 CNN/GRU 차이와 가산 기준값은 [CSV](factorial_contrasts_seed42.csv)에 모두 보존했다.",
              "", "Test 대조를 이후 모델 선택에 활용하면 이 Test는 개발 과정에 노출된 자료이며 독립 confirmatory holdout으로 주장하지 않는다."]
    (OUT / "FACTORIAL_CONTRASTS.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-results", type=Path)
    args = parser.parse_args()
    inventory = prepare_baseline()
    if args.new_results is None:
        print(json.dumps({"status": "baseline_prepared", "models": inventory["unique_model_count"], "rows": inventory["row_count"], "source_SHA_verified": inventory["source_file_count"]}))
        return
    new_rows, bindings = load_new(args.new_results)
    rows = inventory["seed42_rows"] + new_rows
    analysis = analyze_rows(rows)
    analysis.update({"schema": "cnn_gru_frozen_seed42_comparison_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                     "model_count": len({r["model"] for r in rows}), "new_conditions": 9, "new_seed_count": 1,
                     "primary_comparison": "same_seed42", "baseline_inventory_sha256": sha(OUT / "baseline_inventory.json"),
                     "new_results_path": str(args.new_results.resolve()), "new_results_sha256": sha(args.new_results),
                     "new_checkpoint_bindings": bindings, "rows": rows, "inference_or_training_executed": False})
    write("analysis.json", analysis)
    combined = inventory["all_per_seed_rows"] + new_rows
    require(len(combined) == 552, "Combined per-seed row coverage differs")
    write_csv("combined_metrics_per_seed.csv", combined)
    write_csv("seed42_metric_ranks.csv", rank_csv_rows(rows, analysis))
    write_csv("factorial_contrasts_seed42.csv", analysis["factorial_contrasts"])
    render(rows, inventory, analysis)
    print(json.dumps({"status": "complete", "model_count": analysis["model_count"], "new_conditions": 9, "rows": len(rows)}))


if __name__ == "__main__":
    main()
