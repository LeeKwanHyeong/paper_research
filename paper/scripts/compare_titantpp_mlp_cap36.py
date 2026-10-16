"""Pure local three-seed comparison of completed, separately capped MLP fits."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics

from paper.scripts import manage_titantpp_mlp_cap36 as manager

GROUPS = ("overall", "tail", "cap_hit")
METRICS = ("qty_rmse", "qty_mae", "time_nll")
SEEDS = (42, 52, 62)
ARMS = manager.runtime.ARMS
LABELS = dict(zip(ARMS, ("MLP4", "MLP16", "AllAvailable", "CurrentOnly")))
CONTRASTS = ((ARMS[1], ARMS[0]), (ARMS[2], ARMS[0]), (ARMS[3], ARMS[0]),
             (ARMS[2], ARMS[1]), (ARMS[3], ARMS[1]))
require, read, sha = manager.require, manager.read, manager.sha


def summarize(values):
    require(len(values) == 3, "Exactly three seeds are required")
    if all(v is None for v in values):
        return {"mean": None, "sample_sd": None, "seed_count": 3, "status": "N/A"}
    require(all(v is not None and math.isfinite(v) for v in values), "Partial or nonfinite metric support")
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "seed_count": 3, "status": "available"}


def differences(candidate, reference):
    if candidate is None or reference is None:
        require(candidate is None and reference is None, "Asymmetric missing comparison")
        return {"absolute": None, "relative_percent": None, "status": "N/A_empty"}
    delta = candidate - reference
    return {"absolute": delta, "relative_percent": 100. * delta / abs(reference) if reference != 0 else None,
            "status": "available" if reference != 0 else "relative_N/A_zero_reference"}


def verify_metric_groups(metrics, expected_count, cap_value):
    require(metrics["overall"]["count"] == expected_count > 0, "Overall target count changed")
    require(metrics["cap_value"] == cap_value and metrics["tail_rule"] == "q > cap_value"
            and metrics["cap_hit_rule"] == "q == cap_value", "Cap grouping contract changed")
    require(metrics["tail"]["count"] == 0, "A Q3-capped target cannot exceed its cap")
    require(metrics["time_metric"] == "recorded_positive_integer_time_nll", "Time metric meaning changed")
    for group in GROUPS:
        value = metrics[group]
        require(type(value["count"]) is int and 0 <= value["count"] <= expected_count, "Invalid metric support")
        for key in METRICS:
            v = value[key]
            require(v is None if value["count"] == 0 else isinstance(v, (float, int)) and math.isfinite(v),
                    "Invalid empty/nonfinite group metric")
            if key != "time_nll" and v is not None:
                require(v >= 0, "Negative quantity error")
        if value["count"]:
            for total, expected in (("qty_sse", value["qty_rmse"] ** 2 * value["count"]),
                    ("qty_absolute_error_sum", value["qty_mae"] * value["count"]),
                    ("time_nll_sum", value["time_nll"] * value["count"])):
                if total in value:
                    require(math.isclose(value[total], expected, rel_tol=1e-10, abs_tol=1e-8), "Metric sum inconsistency")


def load_verified(contract):
    """Verify both archives and every collected file before using metric receipts."""
    rows, evidence = [], {}
    for host in ("5080", "5090"):
        dest = manager.BUNDLE / "retrieval" / host
        collection = read(dest / "receipt.json")
        count = sum(j["host"] == host for j in contract["jobs"])
        require(collection["status"] == "complete" and collection["host"] == host
                and collection["new_fit_count"] == collection["validation_replay_count"] == collection["test_count"] == count
                and collection["held_out_test_evaluated"] is True, "Host collection incomplete")
        require(sha(dest / "original.tar.gz") == collection["archive_sha256"], "Original archive SHA changed")
        require(sha(dest / "manifest.json") == collection["manifest_sha256"], "Collection manifest SHA changed")
        inventory = read(dest / "manifest.json")["files"]
        require(inventory and len(inventory) == collection["file_count"], "Empty/incomplete collection inventory")
        originals = dest / "originals"
        actual_names = {str(p.relative_to(originals)) for p in originals.rglob("*") if p.is_file()}
        require(actual_names == set(inventory), "Collection inventory differs from original files")
        for name, digest in inventory.items():
            path = manager.runtime.local_path(originals, name)
            require(sha(path) == digest, "Original file SHA changed: " + name)
            evidence[str(path.resolve())] = digest
        manager.verify_collected(contract, host, originals)
        for path in (dest / "receipt.json", dest / "manifest.json", dest / "original.tar.gz", dest / "transport.json"):
            evidence[str(path.resolve())] = sha(path)
        for job in [j for j in contract["jobs"] if j["host"] == host]:
            folder = originals / "run" / job["id"]
            selected = read(folder / "train_receipt.json")
            for split in ("validation", "test"):
                path = folder / (split + "_receipt.json")
                value = read(path)
                require(all(value[k] == job[k] for k in ("host", "dataset", "arm", "seed")), "Evaluation job identity changed")
                require(all(value[k] == selected[k] for k in ("selected_checkpoint", "selected_epoch", "checkpoint_sha256", "state_sha256")),
                        "Validation/Test checkpoint differs from selected training checkpoint")
                require(value["parameters_unchanged"] is True, "Evaluation state changed")
                data = next(d for d in contract["datasets"] if d["dataset_id"] == job["dataset"])
                identity = data["parent_data_identity" if split == "test" else "inherited_data_identity"]["populations"][split]
                require(value["target_identity"] == identity and value["prediction_rows"] == identity["target_count"],
                        "Evaluation target identity changed")
                verify_metric_groups(value["metrics"], identity["target_count"], data["cap_value"])
                rows.append({"dataset": job["dataset"], "arm": job["arm"], "seed": job["seed"], "host": host,
                    "split": split, "selected_epoch": selected["selected_epoch"], "checkpoint_sha256": selected["checkpoint_sha256"],
                    "state_sha256": selected["state_sha256"], "target_identity": value["target_identity"],
                    "metrics": value["metrics"], "receipt_path": str(path.resolve()), "receipt_sha256": sha(path)})
    require(len(rows) == 72, "Expected 36 Validation and 36 Test results")
    return rows, evidence


def aggregate(rows, contract, split):
    require(split in ("validation", "test"), "Unknown split")
    chosen = [r for r in rows if r["split"] == split]
    expected = {(d["dataset_id"], a, s) for d in contract["datasets"] for a in ARMS for s in SEEDS}
    require(len(chosen) == len(expected) == 36 and {(r["dataset"], r["arm"], r["seed"]) for r in chosen} == expected,
            "Three-seed result matrix incomplete or duplicated")
    datasets = []
    for data in contract["datasets"]:
        subset = [r for r in chosen if r["dataset"] == data["dataset_id"]]
        identity = subset[0]["target_identity"]
        require(all(r["target_identity"] == identity for r in subset), "Models/seeds have different targets")
        require(all(all(r["metrics"][g]["count"] == subset[0]["metrics"][g]["count"] for g in GROUPS) for r in subset),
                "Group support differs across models/seeds")
        arms = []
        for arm in ARMS:
            seeds = sorted((r for r in subset if r["arm"] == arm), key=lambda r: r["seed"])
            groups = {g: {"count": seeds[0]["metrics"][g]["count"],
                "metrics": {k: summarize([r["metrics"][g][k] for r in seeds]) for k in METRICS}} for g in GROUPS}
            arms.append({"arm": arm, "label": LABELS[arm], "seeds": seeds, "groups": groups})
        comparisons = []
        lookup = {(r["arm"], r["seed"]): r for r in subset}
        for candidate, reference in CONTRASTS:
            groups = {}
            for group in GROUPS:
                seed_differences = [{"seed": seed, "metrics": {key: differences(
                    lookup[candidate, seed]["metrics"][group][key], lookup[reference, seed]["metrics"][group][key])
                    for key in METRICS}} for seed in SEEDS]
                summaries = {}
                for key in METRICS:
                    absolute = [r["metrics"][key]["absolute"] for r in seed_differences]
                    relative = [r["metrics"][key]["relative_percent"] for r in seed_differences]
                    summaries[key] = {"absolute": summarize(absolute),
                        "relative_percent": summarize(relative) if all(v is None for v in relative) or all(v is not None for v in relative)
                            else {"mean": None, "sample_sd": None, "seed_count": 3, "status": "N/A_some_zero_references"},
                        "improved_seed_count": sum(v is not None and v < 0 for v in absolute),
                        "worsened_seed_count": sum(v is not None and v > 0 for v in absolute),
                        "tied_seed_count": sum(v == 0 for v in absolute),
                        "unavailable_seed_count": sum(v is None for v in absolute)}
                groups[group] = {"seed_differences": seed_differences, "summary": summaries}
            comparisons.append({"candidate": candidate, "reference": reference,
                                "label": LABELS[candidate] + "−" + LABELS[reference], "groups": groups})
        datasets.append({"dataset": data["dataset_id"], "cap_value": data["cap_value"], "target_identity": identity,
                         "arms": arms, "comparisons": comparisons})
    return {"schema": "titantpp_mlp_cap36_three_seed_v1", "split": split, "seed_count": 3, "sample_sd_ddof": 1,
        "model_fit_count": 36, "datasets": datasets,
        "difference_definition": "candidate minus reference; negative means lower error/NLL",
        "relative_percent_definition": "100 * (candidate-reference) / abs(reference); zero reference is N/A",
        "cross_dataset_raw_metric_average_computed": False, "raw_data_baselines_used": False,
        "limitations": ["Separate Q3-capped task; these metrics do not establish performance on original uncapped quantities.",
            "Tail q>cap is empty by construction; cap_hit q==cap is reported separately.",
            "Dataset/seed comparisons use four models on the same GPU; Intermittent seed42 uses5080 and seeds52/62 use5090.",
            "5080 cuDNN91900 and5090 cuDNN92000 differ; these are exploratory results, not statistical significance."]}


def number(value, *, signed=False):
    return "N/A" if value is None else format(value, "+.6f" if signed else ".6f")


def markdown(result):
    lines = [f"# Cap MLP 3seed {result['split'].title()} 비교", "",
        "수량 상한은 Train Q3에서 정했고 입력 이력과 정답에 함께 적용했다. 기존 raw 데이터 모델의 점수는 비교하지 않는다.",
        "평균±표본SD는 seed42·52·62를 사용했다(ddof=1). 서로 다른 데이터셋의 수량 오류는 평균내지 않는다.",
        "음수 차이는 개선, 양수 차이는 악화다. 상대차이는 100×(후보−대조)/|대조|이며 대조가0이면 N/A다.", ""]
    for dataset in result["datasets"]:
        lines += [f"## {dataset['dataset']} — cap {dataset['cap_value']}", "",
            f"동일 평가 대상 {dataset['target_identity']['target_count']}개; target SHA `{dataset['target_identity']['target_identity_sha256']}`.", "",
            "| 구조 | 집단 | 개수 | RMSE 평균±SD | MAE 평균±SD | TimeNLL 평균±SD |",
            "|---|---|---:|---:|---:|---:|"]
        for arm in dataset["arms"]:
            for group in GROUPS:
                values = arm["groups"][group]
                cells = []
                for key in METRICS:
                    s = values["metrics"][key]
                    cells.append("N/A" if s["mean"] is None else f"{number(s['mean'])} ± {number(s['sample_sd'])}")
                lines.append(f"| {arm['label']} | {group} | {values['count']} | " + " | ".join(cells) + " |")
        lines += ["", "tail은 q>cap으로 빈 집합이며 N/A를 유지한다. cap_hit은 q==cap이다.", "",
            "| 구조 | seed | GPU | 선택 epoch | 전체 RMSE | 전체 MAE |", "|---|---:|---:|---:|---:|---:|"]
        for arm in dataset["arms"]:
            for row in arm["seeds"]:
                overall = row["metrics"]["overall"]
                lines.append(f"| {arm['label']} | {row['seed']} | {row['host']} | {row['selected_epoch']} | {number(overall['qty_rmse'])} | {number(overall['qty_mae'])} |")
        lines += ["", "| 비교 | 집단 | seed | ΔRMSE (상대%) | ΔMAE (상대%) | ΔTimeNLL (상대%) |",
                  "|---|---|---:|---:|---:|---:|"]
        for comparison in dataset["comparisons"]:
            for group in GROUPS:
                for row in comparison["groups"][group]["seed_differences"]:
                    cells = [f"{number(row['metrics'][k]['absolute'], signed=True)} ({number(row['metrics'][k]['relative_percent'], signed=True)}%)"
                             if row["metrics"][k]["absolute"] is not None else "N/A" for k in METRICS]
                    lines.append(f"| {comparison['label']} | {group} | {row['seed']} | " + " | ".join(cells) + " |")
        lines += ["", "| 비교 | 전체 RMSE 악화 seed 수 | 전체 MAE 악화 seed 수 | 전체 TimeNLL 악화 seed 수 |",
                  "|---|---:|---:|---:|"]
        for comparison in dataset["comparisons"]:
            summaries = comparison["groups"]["overall"]["summary"]
            lines.append(f"| {comparison['label']} | " + " | ".join(str(summaries[k]["worsened_seed_count"]) for k in METRICS) + " |")
        lines.append("")
    lines += ["5080은 cuDNN91900,5090은 cuDNN92000이다. Intermittent는 seed42가5080,52·62가5090이지만 같은 dataset·seed의 네 구조는 같은 GPU에서 평가했다.",
        "반복 실험의 탐색적 결과다. 원래 수량의 일반화 개선이나 통계적 유의성을 주장하지 않는다.", ""]
    return "\n".join(lines)


def compare():
    contract = manager.verify_local()
    final_path = manager.BUNDLE / "retrieval/final_receipt.json"
    if final_path.exists():
        existing = read(final_path)
        require(existing["status"] == "complete" and existing["files"]
                and existing["new_fit_count"] == existing["candidate_test_count"] == existing["validation_replay_count"] == 36
                and existing["seed_count"] == 3 and existing["held_out_test_evaluated"] is True
                and existing["contract_sha256"] == manager.canonical(contract)
                and len(existing["report_files"]) == 4, "Invalid comparison cache")
        for path, digest in existing["files"].items():
            require(sha(path) == digest, "Completed comparison evidence changed")
        return existing
    manager.write(manager.BUNDLE / "retrieval/comparison_attempt.json", {"started_at": manager.now(), "automatic_retry": False})
    rows, evidence = load_verified(contract)
    outputs = {}
    for split in ("validation", "test"):
        result = aggregate(rows, contract, split)
        stem = manager.REPORT / ("THREE_SEED_" + split.upper())
        json_path, md_path = stem.with_suffix(".json"), stem.with_suffix(".md")
        manager.write(json_path, result)
        with md_path.open("x", encoding="utf-8") as out:
            out.write(markdown(result))
        for path in (json_path, md_path):
            evidence[str(path.resolve())] = sha(path)
            outputs[str(path.resolve())] = sha(path)
    for path in (manager.BUNDLE / "execution_contract.json", manager.REPORT / "PREPARED.json"):
        evidence[str(path.resolve())] = sha(path)
    final = {"status": "complete", "completed_at": manager.now(), "contract_sha256": manager.canonical(contract),
        "new_fit_count": 36, "candidate_test_count": 36, "validation_replay_count": 36,
        "candidate_fit_count": 36, "comparison_model_count": 36, "baseline_reference_count": 0,
        "seed_count": 3, "dataset_count": 3, "arm_count": 4, "actual_training_worker_exit0_count": 36,
        "actual_evaluation_worker_exit0_count": 72, "actual_dispatcher_exit0_count": 2,
        "held_out_test_evaluated": True, "historical_cnn_retrained": False, "historical_cnn_reevaluated": False,
        "raw_data_baselines_used": False, "report_files": outputs, "files": evidence,
        "supervisor_own_exit_directly_observed": False}
    manager.write(final_path, final)
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    import json
    result = compare()
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
