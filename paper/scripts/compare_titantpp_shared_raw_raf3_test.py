"""Compare immutable RAF Shared15 Test3 with stored MLP16/external Test21."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
NAME = "titantpp_shared_raw_raf3_test_20261020_v1"
SHARED = "titantpp_history_mlp_shared15_v1"
MLP16 = "titantpp_history_mlp_width16"
EXTERNAL = ("rmtpp", "thp", "nhp", "sahp", "s2p2_matched_head", "attnhp_matched_head")
SEEDS = (42, 52, 62)
GROUPS, METRICS = ("overall", "tail"), ("qty_rmse", "qty_mae", "time_nll")
LABELS = {SHARED: "공유형 Shared15", MLP16: "기존 MLP16", "rmtpp": "RMTPP",
          "thp": "THP", "nhp": "NHP", "sahp": "SAHP",
          "s2p2_matched_head": "S2P2 (공통 head)", "attnhp_matched_head": "AttNHP (공통 head)"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False,
            indent=2, allow_nan=False) + "\n")


def simple_quantity_baselines(receipt):
    baselines = receipt.get("simple_quantity_baselines")
    require(isinstance(baselines, dict) and set(baselines) == {"last", "history_mean"},
            "Deterministic quantity baselines missing")
    for baseline in baselines.values():
        require(baseline.get("time_metric") == "N/A_no_time_model", "Baseline TimeNLL must be N/A")
        for group in GROUPS:
            require(baseline[group]["time_nll"] is None
                and baseline[group]["count"] == receipt["metrics"][group]["count"],
                "Baseline time score/support differs")
            require(all(type(baseline[group][metric]) in (float, int)
                and math.isfinite(baseline[group][metric]) for metric in METRICS[:2]),
                "Invalid deterministic quantity errors")
    return deepcopy(baselines)


def aggregate(records):
    """Pure complete-matrix arithmetic. Never selects a checkpoint using Test."""
    indexed = {}
    for row in records:
        model, seed = row["model"], row["seed"]
        require(model in (SHARED, MLP16, *EXTERNAL) and type(seed) is int and seed in SEEDS,
                "Foreign model or seed")
        require((model, seed) not in indexed, "Duplicate endpoint")
        require(row.get("dataset") == "raf_spare_parts" and row.get("split") == "test"
            and row.get("data_variant") == "original_uncapped", "Foreign dataset/split/Cap endpoint")
        require(type(row.get("selected_epoch")) is int and row["selected_epoch"] > 0,
                "Missing Validation-selected epoch")
        identity = row["target_identity"]
        require(identity["target_count"] == 5226 and row["metrics"]["overall"]["count"] == 5226,
                "Wrong RAF Test population")
        for group in GROUPS:
            values = row["metrics"][group]
            require(type(values["count"]) is int and 0 <= values["count"] <= 5226, "Invalid support")
            for metric in METRICS:
                value = values[metric]
                require(value is None if not values["count"] else type(value) in (float, int)
                    and math.isfinite(value) and (metric == "time_nll" or value >= 0), "Invalid metric")
        indexed[model, seed] = row
    require(set(indexed) == {(m, s) for m in (SHARED, MLP16, *EXTERNAL) for s in SEEDS},
            "Expected candidate3 and stored references21")
    first = indexed[SHARED, 42]
    for row in indexed.values():
        require(row["target_identity"] == first["target_identity"], "Target/truth identity mismatch")
        for group in GROUPS:
            require(row["metrics"][group]["count"] == first["metrics"][group]["count"], "Cohort mismatch")
    summaries, differences = {}, []
    for model in (SHARED, MLP16, *EXTERNAL):
        summaries[model] = {}
        for group in GROUPS:
            summaries[model][group] = {}
            for metric in METRICS:
                vals = [indexed[model, seed]["metrics"][group][metric] for seed in SEEDS]
                available = vals[0] is not None
                summaries[model][group][metric] = {"mean": statistics.mean(vals) if available else None,
                    "sample_sd_ddof1": statistics.stdev(vals) if available else None}
        if model == SHARED:
            continue
        for seed in SEEDS:
            for group in GROUPS:
                for metric in METRICS:
                    a = indexed[SHARED, seed]["metrics"][group][metric]
                    b = indexed[model, seed]["metrics"][group][metric]
                    delta = None if a is None else a - b
                    differences.append({"reference_model": model, "seed": seed, "group": group,
                        "metric": metric, "shared": a, "reference": b, "absolute_difference": delta,
                        "relative_difference_percent": None if delta is None or b == 0 else 100 * delta / abs(b),
                        "worsened": None if delta is None else delta > 0})
    best = {}
    for metric in METRICS[:2]:
        model = min(EXTERNAL, key=lambda m: summaries[m]["overall"][metric]["mean"])
        old, new = summaries[model]["overall"][metric]["mean"], summaries[SHARED]["overall"][metric]["mean"]
        best[metric] = {"external_model": model, "external_mean": old, "shared_mean": new,
            "absolute_difference": new - old,
            "relative_difference_percent": None if old == 0 else 100 * (new - old) / old,
            "shared_strictly_better": new < old}
    simple = [indexed[SHARED, seed].get("simple_quantity_baselines") for seed in SEEDS]
    if any(value is not None for value in simple):
        require(all(value is not None for value in simple) and simple[0] == simple[1] == simple[2],
                "Deterministic baseline differs across seeds")
    return {"schema": "shared_raw_raf3_three_seed_test_v1", "dataset": "raf_spare_parts",
        "split": "test", "data_variant": "original_uncapped", "records": deepcopy(records),
        "summary": summaries, "seed_differences": differences, "external_metric_bests": best,
        "external_both_quantity_means_improved": all(v["shared_strictly_better"] for v in best.values()),
        "held_out_test_evaluated": True, "checkpoint_reselected": False,
        "historical_baselines_reevaluated": False, "independent_new_test": False,
        "simple_quantity_baselines": simple[0]}


def markdown(result):
    lines = ["# 원본 RAF 공유형의 고정 checkpoint Test 비교", "",
        "Validation RMSE로 선택한 checkpoint를 고정했다. 평균±표본SD(ddof1), 낮을수록 좋다.", "",
        "| 모델 | RMSE | MAE | 큰수량 RMSE | 큰수량 MAE | 전체 TimeNLL | 큰수량 TimeNLL |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for model, values in result["summary"].items():
        cells = []
        for group, metric in (("overall", "qty_rmse"), ("overall", "qty_mae"), ("tail", "qty_rmse"),
                ("tail", "qty_mae"), ("overall", "time_nll"), ("tail", "time_nll")):
            x = values[group][metric]
            cells.append("N/A" if x["mean"] is None else f'{x["mean"]:.6f} ± {x["sample_sd_ddof1"]:.6f}')
        lines.append("| " + LABELS[model] + " | " + " | ".join(cells) + " |")
    lines += ["", "## 전체 수량의 seed별 차이", "", "차이는 공유형−비교군이며 음수는 개선이다.", "",
        "| 비교군 | seed | 지표 | 공유형 | 비교군 | 절대 차이 | 상대 차이(%) |",
        "|---|---:|---|---:|---:|---:|---:|"]
    for row in result["seed_differences"]:
        if row["group"] == "overall" and row["metric"] in METRICS[:2]:
            pct = row["relative_difference_percent"]
            rel = "N/A" if pct is None else f"{pct:+.4f}"
            lines.append(f'| {LABELS[row["reference_model"]]} | {row["seed"]} | {row["metric"]} | '
                f'{row["shared"]:.6f} | {row["reference"]:.6f} | {row["absolute_difference"]:+.6f} | {rel} |')
    if result.get("simple_quantity_baselines"):
        lines += ["", "## 동일 관측의 단순 기준선", "",
            "세 seed에 동일한 결정적 수량 예측이며 TimeNLL은 N/A다.", "",
            "| 기준선 | 전체 RMSE | 전체 MAE | 큰수량 RMSE | 큰수량 MAE | TimeNLL |",
            "|---|---:|---:|---:|---:|---|"]
        for name, baseline in result["simple_quantity_baselines"].items():
            label = "마지막 관측 수량" if name == "last" else "관측 이력 평균"
            lines.append(f'| {label} | {baseline["overall"]["qty_rmse"]:.6f} | '
                f'{baseline["overall"]["qty_mae"]:.6f} | {baseline["tail"]["qty_rmse"]:.6f} | '
                f'{baseline["tail"]["qty_mae"]:.6f} | N/A |')
    lines += ["", "## 해석의 범위", "",
        "- 큰수량은 기존 Train 고정 q>200이다. 전체/큰수량 recorded positive-integer TimeNLL과 불리한 결과도 보존했다.",
        "- 외부6종은 프로젝트의 공통 수량·시간 head에 적응한 구현이며 원논문 고유 head의 모든 튜닝을 대표하지 않는다.",
        "- 신규5090/cuDNN92000과 기존5080/cuDNN91900 및 일부 CPU 재현 결과의 차이가 있다.",
        "- 이력 보정 파라미터는 MLP16 24576, 공유형9600이다. 공유와 용량 감소 효과를 분리한 실험은 아니다.",
        "- 기존 Test를 본 뒤 개발한 탐색적 후속 평가이며 새로운 독립 평가·인과관계·통계적 유의성으로 확대하지 않는다.",
        "- 큰수량/시간을 포함한 모든 seed별 절대·상대 차이는 JSON에 보존했다. 단순 기준선의 TimeNLL은 N/A다.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args(argv)
    bundle = args.bundle.resolve()
    # Collection verifies owner absence, transport, actual exits and the remote
    # inventory before writing this local receipt. No remote call is made here.
    receipt_path = bundle / "retrieval/receipt.json"
    collected = read(receipt_path)
    require(collected.get("status") in ("verified", "complete")
        and collected.get("owner_absence_verified") is True
        and collected.get("candidate_test_count") == collected.get("validation_replay_count") == 3
        and collected.get("actual_pipeline_exit0_count") == 1, "Verified collection required")
    originals = bundle / "retrieval/originals"
    archive, manifest_path = bundle / "retrieval/original.tar.gz", bundle / "retrieval/manifest.json"
    require(sha(archive) == collected["archive_sha256"]
        and sha(manifest_path) == collected["manifest_sha256"], "Collection archive/manifest changed")
    inventory = read(manifest_path)["files"]
    require(bool(inventory) and len(inventory) == collected["file_count"], "Empty/incomplete collection inventory")
    for relative, digest in inventory.items():
        path = (originals / relative).resolve()
        require(path.is_relative_to(originals.resolve()) and not path.is_symlink()
            and sha(path) == digest, "Collected original changed: " + relative)
    c = read(originals / "evaluation_contract.json")
    contract_sha = canonical(c)
    pipeline, pipeline_exit = read(originals / "pipeline_receipt.json"), read(originals / "pipeline_exit.json")
    require(pipeline["status"] == "complete" and pipeline["contract_sha256"] == contract_sha
        and pipeline.get("qualification_exit0_count") == 3 and pipeline.get("owned_pids_absent") is True
        and pipeline_exit["actual_os_returncode"] == 0, "Incomplete native pipeline")
    source_binding = c["references"]
    ref_path = originals / source_binding["path"]
    require(sha(ref_path) == source_binding["sha256"], "Stored reference registry changed")
    refs = read(ref_path)
    require(refs["reference_count"] == 21 and refs["seed_count"] == 3, "Incomplete stored references")
    for relative, digest in refs["source_bindings"].items():
        require(sha(ROOT / relative) == digest, "Existing reference changed: " + relative)
    records = deepcopy(refs["references"])
    for job in c["jobs"]:
        folder = originals / "results" / job["id"]
        item = read(folder / "test_receipt.json")
        actual = read(folder / "test_exit.json")
        require(actual["actual_os_returncode"] == 0 and actual["timeout"] is False
            and actual["actual_child_exit_observed"] is True, "Test actual exit0 missing")
        require(item["status"] == "complete" and item["split"] == "test"
            and item["contract_sha256"] == contract_sha and item["job_id"] == job["id"]
            and item["selected_epoch"] == job["selected_epoch"]
            and item["checkpoint_sha256"] == job["checkpoint"]["sha256"], "Foreign candidate receipt")
        population = item["target_identity"]
        records.append({"dataset": "raf_spare_parts", "model": SHARED, "seed": job["seed"],
            "split": "test", "data_variant": "original_uncapped", "selected_epoch": item["selected_epoch"],
            "metrics": item["metrics"], "target_identity": population,
            "simple_quantity_baselines": simple_quantity_baselines(item),
            "receipt_sha256": sha(folder / "test_receipt.json")})
    result = aggregate(records)
    result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    report = ROOT / "reports" / NAME
    report.mkdir(parents=True, exist_ok=True)
    paths = [report / "THREE_SEED_TEST.json", report / "THREE_SEED_TEST.md"]
    write(paths[0], result)
    write(paths[1], markdown(result))
    final = {"status": "complete", "candidate_test_count": 3, "baseline_reference_count": 21,
        "seed_count": 3, "validation_replay_count": 3, "qualification_exit0_count": 3,
        "actual_data_worker_exit0_count": 6, "actual_native_worker_exit0_count": 9,
        "actual_pipeline_exit0_count": 1,
        "held_out_test_evaluated": True, "checkpoint_reselected": False,
        "historical_baselines_reevaluated": False, "collection_receipt_sha256": sha(receipt_path),
        "result_files": {str(p): sha(p) for p in paths}}
    write(bundle / "final_receipt.json", final)
    print(json.dumps(final, ensure_ascii=False))


if __name__ == "__main__":
    main()
