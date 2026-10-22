"""Local Validation-only interim analysis; no inference or cost-result claims."""

from __future__ import annotations

import csv
import datetime
import io
import json
import math
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "TEST_SESSION_PROTOCOL.md").is_file())
sys.path.insert(0, str(ROOT))

from paper.scripts import compare_shared_history_mlp_control as frozen  # noqa: E402
from paper.scripts import advance_shared_history_mlp_control as collection  # noqa: E402
from reports.shared_history_mlp_control_20261022_v1.execution_v2 import advance  # noqa: E402

MANAGER = frozen.manager
BUNDLE = MANAGER.BUNDLE / "execution_v2"
OUT = Path(__file__).resolve().parent
RAW_CONTRACT = "56721f98f69c962fee1cdb8c4331d6dd4c4373bb5745f302fc2164a1b9819673"
LABELS = {"shared15": "B", "single_mlp": "단일 MLP", "encoder_only": "D"}
DATA_LABELS = {"raf_spare_parts": "RAF", "yellow_trip_hourly": "Taxi", "intermittent_frozen_5000": "Intermittent"}
DATASETS = tuple(DATA_LABELS)
METRICS = frozen.METRICS


def formatted(value):
    mean, sd = value["mean"], value["sample_sd_ddof1"]
    if mean is None:
        return "N/A"
    if mean != 0 and abs(mean) < 1e-4:
        return f"{mean:.5e} ± {sd:.5e}"
    return f"{mean:.6f} ± {sd:.6f}"


def csv_text(rows, fields):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main():
    MANAGER.require(MANAGER.sha(BUNDLE / "execution_contract.json") == RAW_CONTRACT, "Contract changed")
    state = advance.check_only(
        "3f61a06bdac3b53815771ccf9cf24f7285db14f9624876972dbeb316915283af",
        "5e631f4a2ba71cd49da2b8ef134ece59457c76d14994ae9c83c8e31d60152ef2",
    )
    c = MANAGER.read(BUNDLE / "execution_contract.json")
    gate = MANAGER.verify_global_gate(c, BUNDLE / "global_validation_v1")
    roots = {
        host: collection.validate_collection(c, host, "fit_validation", BUNDLE / "retrieval/fit_validation" / host)
        for host in frozen.HOSTS
    }
    evidence, records, errors = {}, [], []
    evidence[str(BUNDLE / "execution_contract.json")] = RAW_CONTRACT
    gate_path = BUNDLE / "global_validation_v1/global_validation_gate.json"
    evidence[str(gate_path)] = MANAGER.sha(gate_path)
    for job in c["jobs"]:
        root, host = roots[job["host"]], job["host"]
        data = next(d for d in c["datasets"] if d["dataset_id"] == job["dataset"])
        train = frozen._receipt(root, c, host, job, "train")
        val = frozen._receipt(root, c, host, job, "validation")
        frozen._exit(root, c, job, "reuse" if job.get("reuse") else "train")
        frozen._exit(root, c, job, "validation")
        for key in ("selected_checkpoint", "selected_epoch", "checkpoint_sha256", "state_sha256", "training_origin"):
            MANAGER.require(train[key] == val[key], "Selected checkpoint differs")
        MANAGER.require(val["validation_replay_passed"] and val["individual_predictions_match"], "Replay missing")
        prediction = frozen.mapped(root, c["hosts"][host]["root"], val["prediction_path"])
        metrics = frozen.prediction_metrics(prediction, data["quantity_boundaries_all_train_rows"][-1], "validation")
        MANAGER.require(val["target_identity"] == data["inherited_data_identity"]["populations"]["validation"], "Population differs")
        MANAGER.require(metrics["target_quantity_sha256"] == val["target_identity"]["target_quantity_sha256"], "Target quantities differ")
        MANAGER.require(metrics["prediction_rows"] == val["prediction_rows"], "Prediction count differs")
        for scope, original_scope in (("overall", "overall"), ("large", "tail")):
            MANAGER.require(metrics[scope]["count"] == val["metrics"][original_scope]["count"], "Scope count differs")
            for metric in METRICS:
                a, b = metrics[scope][metric], val["metrics"][original_scope][metric]
                MANAGER.require(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), "Stored metric differs")
                errors.append(abs(a - b))
        if job.get("reuse"):
            frozen.verify_legacy(c, job, train, val, root, prediction)
        for stage in ("train", "validation"):
            path = root / "run" / job["id"] / (stage + "_receipt.json")
            evidence[str(path)] = MANAGER.sha(path)
            MANAGER.require(evidence[str(path)] == gate["jobs"][job["id"]][stage + "_receipt"]["sha256"], "Global gate original differs")
        evidence[str(prediction)] = MANAGER.sha(prediction)
        records.append({
            "job_id": job["id"], "dataset": job["dataset"], "arm": job["arm"], "seed": job["seed"],
            "host": host, "split": "validation", "metrics": metrics,
            "selected_epoch": val["selected_epoch"], "checkpoint_sha256": val["checkpoint_sha256"],
            "state_sha256": val["state_sha256"], "training_origin": val["training_origin"],
            "prediction_path": str(prediction), "validation_receipt_path": str(root / "run" / job["id"] / "validation_receipt.json"),
        })
    for dataset in DATASETS:
        same = {(r["metrics"]["target_tuples_sha256"], r["metrics"]["target_quantity_sha256"]) for r in records if r["dataset"] == dataset}
        MANAGER.require(len(same) == 1, "Matched target population/order differs")
    table, paired = frozen.summarize_records(records), frozen.paired_differences(records)
    index = {(r["dataset"], r["arm"]): r for r in table}
    means, worsening, flat = [], [], []
    for dataset in DATASETS:
        for left, right in frozen.PAIRS:
            for scope in frozen.SCOPES:
                for metric in METRICS:
                    a, b = index[dataset, left]["metrics"][scope][metric]["mean"], index[dataset, right]["metrics"][scope][metric]["mean"]
                    means.append({"dataset": dataset, "scope": scope, "metric": metric, "left_arm": left, "right_arm": right, **frozen._diff(a, b)})
                    rows = [p for p in paired if (p["dataset"], p["left_arm"], p["right_arm"], p["scope"], p["metric"]) == (dataset, left, right, scope, metric)]
                    worsening.append({"dataset": dataset, "scope": scope, "metric": metric, "left_arm": left, "right_arm": right, "seeds_left_worse": [p["seed"] for p in rows if p["left_worse"]]})
        for arm in frozen.ARMS:
            for scope in frozen.SCOPES:
                for metric in METRICS:
                    flat.append({"dataset": dataset, "split": "validation", "arm": arm, "scope": scope, "metric": metric, **index[dataset, arm]["metrics"][scope][metric]})
    result = {
        "status": "VALIDATION_ONLY_INTERIM", "verification_status": "ANALYZED",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "contract_file_sha256": RAW_CONTRACT,
        "cost_results_included": False, "held_out_test_evaluated": False, "new_model_forward_passes": 0, "remote_calls": 0,
        "numeric_recomputation": {"status": "PASS", "prediction_files": 27, "prediction_rows": sum(r["metrics"]["prediction_rows"] for r in records), "max_absolute_receipt_difference": max(errors), "time_nll": "mean of recorded per-target TimeNLL; not recomputed from distribution parameters"},
        "sample_sd_ddof": 1, "seeds": list(frozen.SEEDS), "checkpoint_selection": "first strict minimum full Validation raw quantity RMSE",
        "records": records, "table": table, "mean_differences": means, "paired_differences": paired,
        "worsened_seeds": worsening, "evidence_sha256": evidence, "controller_local_only_status": state,
    }
    lines = ["# B·단일 MLP·D Validation 비교", "", "## Material Passport", "", "- 범위: execution_v2, Validation-only 중간 분석. 비용 결과와 Test는 포함하지 않는다.", "- Verification Status: ANALYZED. 원예측 27개의 수량 지표와 recorded TimeNLL 집계 재계산 PASS; 새로운 모델 추론은 0회다.", "- seed42/52/62의 산술평균 ± 표본SD(ddof=1). 모든 오차·NLL은 낮을수록 좋다.", "- B=공유15차원 이력 보정, MLP=단일 보정 MLP, D=공통 encoder. B·MLP 보정 파라미터 각9600이며 활성 수와 함수 집합까지 같지는 않다.", "- 선택: 전체 Validation raw 수량 RMSE의 첫 strict minimum checkpoint. 시간 NLL 최적 checkpoint의 비교가 아니다."]
    for scope, label in (("overall", "전체 Validation"), ("large", "큰 수량 Validation")):
        lines += ["", "## " + label, "", "| 데이터 | 모델 | 수량 RMSE | 수량 MAE | recorded TimeNLL |", "|---|---|---:|---:|---:|"]
        for dataset in DATASETS:
            for arm in frozen.ARMS:
                m = index[dataset, arm]["metrics"][scope]
                lines.append(f"| {DATA_LABELS[dataset]} | {LABELS[arm]} | " + " | ".join(formatted(m[k]) for k in METRICS) + " |")
    lines += ["", "## 큰 수량 정의", "", "| 데이터 | 전체 수 | 큰 수량 조건 | 큰 수량 수 |", "|---|---:|---|---:|"]
    for dataset in DATASETS:
        m = next(r["metrics"] for r in records if r["dataset"] == dataset)
        lines.append(f"| {DATA_LABELS[dataset]} | {m['overall']['count']} | q > {m['large_cutoff']:g} | {m['large']['count']} |")
    lines += ["", "## 모든 seed의 악화 위치", "", "아래는 왼쪽 모델의 원래 값이 오른쪽보다 큰 seed다. 극소 TimeNLL 차이까지 산술적으로 포함하므로 이 표를 승패 점수로 합산하지 않는다.", "", "| 데이터·범위·지표 | B−MLP 악화 seed | B−D 악화 seed | MLP−D 악화 seed |", "|---|---|---|---|"]
    for dataset in DATASETS:
        for scope in frozen.SCOPES:
            for metric in METRICS:
                cells = []
                for left, right in frozen.PAIRS:
                    w = next(w for w in worsening if (w["dataset"], w["scope"], w["metric"], w["left_arm"], w["right_arm"]) == (dataset, scope, metric, left, right))
                    cells.append(", ".join(map(str, w["seeds_left_worse"])) or "없음")
                lines.append(f"| {DATA_LABELS[dataset]}·{scope}·{metric} | " + " | ".join(cells) + " |")
    lines += ["", "## 원본 연결", "", "모든 27개 원 receipt/prediction 경로·SHA·split·선택 epoch는 REPORT.json의 records/evidence_sha256에 저장한다. PAIRED_SEED_DELTAS.csv는 162개 모든 seed·지표 차이를 포함한다. 차이는 왼쪽−오른쪽, relative는 그 차이/오른쪽이며 양수는 악화다. 평균 차이는 평균들의 차이로 계산하며 seed별 상대차이의 평균과 구분한다.", "", "- 전체 읽은 예측 행: " + str(result["numeric_recomputation"]["prediction_rows"]), "- receipt 대비 최대 절대 차이: " + str(max(errors)), "- Taxi 큰 수량의 극소 TimeNLL을 반올림된 정확한 0이나 완벽한 시간 예측으로 해석하지 않는다.", "- 같은 Validation으로 checkpoint를 선택하고 구조를 비교했다. held-out 일반화·통계적 유의성·동등성은 이 표로 확정하지 않는다.", ""]
    outputs = {
        "REPORT.json": json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        "REPORT.md": "\n".join(lines),
        "TABLE.csv": csv_text(flat, ("dataset", "split", "arm", "scope", "metric", "mean", "sample_sd_ddof1", "n")),
        "PAIRED_SEED_DELTAS.csv": csv_text(paired, tuple(paired[0])),
        "MEAN_DELTAS.csv": csv_text(means, tuple(means[0])),
    }
    for name in outputs:
        MANAGER.require(not (OUT / name).exists(), "Preserve existing analysis output: " + name)
    for name, digest in evidence.items():
        MANAGER.require(MANAGER.sha(Path(name)) == digest, "Original evidence changed during analysis")
    for name, content in outputs.items():
        with (OUT / name).open("x") as stream:
            stream.write(content)
    print(json.dumps({"status": "PASS", **result["numeric_recomputation"], "paired_seed_metrics": len(paired), "new_forward_passes": 0, "remote_calls": 0}, allow_nan=False))


if __name__ == "__main__":
    main()
