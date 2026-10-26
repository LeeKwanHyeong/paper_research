"""Local-only final comparison; no results are written before verified completion.

Only the pilot's Train/Validation retrieval and the original fit_validation
5080 RAF/Taxi full-seed42 records are admitted.  No model loading, network,
held-out predictions, or mixed Validation/Test comparison tables are used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
BASE = PROJECT / "search_artifacts/shared_component_ablation_20261022_v1"
BASE_RETRIEVAL = BASE / "retrieval/fit_validation/5080"
BASE_ORIGINALS = BASE_RETRIEVAL / "originals"
BASE_MANIFEST_SHA = "61aab5c0f0d94288a32eb891d0ff2e18ce903c7d6cc7274730bac833f2699f9e"
DATASETS = ("raf_spare_parts", "yellow_trip_hourly")
NAMES = {"raf_spare_parts": "RAF", "yellow_trip_hourly": "Taxi"}
METRICS = ("qty_rmse", "qty_mae", "log_qty_mse", "time_nll", "joint_objective")
HISTORY_KEYS = {"qty_rmse": "val_qty_rmse", "qty_mae": "val_qty_mae", "log_qty_mse": "val_log_qty_mse",
                "time_nll": "val_time_nll", "joint_objective": "val_joint_objective"}
SUMMARY_KEYS = {metric: "best_" + key for metric, key in HISTORY_KEYS.items()}
LABELS = {"qty_rmse": "수량 RMSE", "qty_mae": "수량 MAE", "log_qty_mse": "log1p 수량 MSE",
          "time_nll": "기록 시간 NLL", "joint_objective": "결합 손실"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe(root, relative):
    p = Path(relative)
    require(not p.is_absolute() and bool(p.parts) and ".." not in p.parts, "Unsafe evidence path: " + str(p))
    result = Path(root) / p
    require(not any(item.is_symlink() for item in (result, *result.parents)), "Symlink evidence refused")
    require(result.resolve().is_relative_to(Path(root).resolve()), "Evidence escapes root")
    return result


def verify_file(root, relative, record):
    path = safe(root, relative)
    require(path.is_file(), "Missing evidence: " + str(path))
    expected_sha = record if isinstance(record, str) else record["sha256"]
    require(sha(path) == expected_sha, "Evidence SHA mismatch: " + str(path))
    if isinstance(record, dict) and "bytes" in record:
        require(path.stat().st_size == record["bytes"], "Evidence byte mismatch: " + str(path))
    return path


def verify_exit(root, relative, *, legacy=False):
    receipt = read(safe(root, relative))
    require(receipt["actual_os_returncode"] == 0, "Actual process failure: " + relative)
    require(not receipt.get("timeout", False) and not receipt.get("supervisor_error")
            and not receipt.get("resource_incident"), "Incomplete process termination: " + relative)
    if not legacy:
        for stream in ("stdout", "stderr"):
            verify_file(root, receipt[stream]["path"], receipt[stream])
    return receipt


def finite(value):
    require(isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value),
            "Nonfinite/non-numeric metric")
    return float(value)


def close(a, b, context):
    require(math.isclose(finite(a), finite(b), rel_tol=1e-5, abs_tol=1e-5), "Metric mismatch: " + context)


def epoch_metrics(row):
    return {metric: finite(row[key]) for metric, key in HISTORY_KEYS.items()}


def validate_history(summary, history):
    require(history and [row["epoch"] for row in history] == list(range(1, len(history) + 1)),
            "Incomplete epoch sequence")
    require(summary["completed_epochs"] == len(history) and len(history) <= 300, "Completed epoch mismatch")
    require(summary["checkpoint_monitor"] == "validation_raw_quantity_rmse", "Selection criterion changed")
    for row in history:
        values = epoch_metrics(row)
        close(values["joint_objective"], values["log_qty_mse"] + values["time_nll"], "joint objective")
    selected = min(history, key=lambda row: finite(row["val_qty_rmse"]))
    require(selected["epoch"] == summary["best_epoch"], "Checkpoint is not first strict RMSE minimum")
    for metric, key in SUMMARY_KEYS.items():
        close(epoch_metrics(selected)[metric], summary[key], key)
    exhausted = len(history) >= 40 and len(history) - selected["epoch"] >= 40
    require(len(history) == 300 or exhausted, "Fit has neither exhausted native patience nor reached max epochs")
    return selected, "maximum_epochs" if len(history) == 300 else "early_stopping"


def signed_change(new, old):
    new, old = finite(new), finite(old)
    delta = new - old
    return {"absolute": delta, "relative_percent": None if old == 0 else delta / abs(old) * 100,
            "direction": "lower" if delta < 0 else "higher" if delta > 0 else "equal"}


def slope(values):
    """Ordinary least-squares slope per epoch; descriptive, not a test."""
    values = [finite(v) for v in values]
    if len(values) < 2:
        return None
    xbar = (len(values) - 1) / 2
    mean = statistics.mean(values)
    return sum((i - xbar) * (v - mean) for i, v in enumerate(values)) / sum((i - xbar) ** 2 for i in range(len(values)))


def late_changes(rows):
    # The last ten observed epochs are a fixed descriptive window, not a
    # hand-picked maximum/minimum or a smoothed replacement for the raw curves.
    window = rows[-min(10, len(rows)):]
    result = {"start_epoch": window[0]["epoch"], "end_epoch": window[-1]["epoch"], "epochs": len(window),
              "window_rule": "last_min_10_completed_epochs", "metrics": {}}
    for metric in METRICS:
        train = [r["train_eval"][metric] for r in window]
        validation = [r["validation"][metric] for r in window]
        gaps = [v - t for t, v in zip(train, validation)]
        result["metrics"][metric] = {
            "train_end_minus_start": train[-1] - train[0], "validation_end_minus_start": validation[-1] - validation[0],
            "gap_end_minus_start": gaps[-1] - gaps[0], "train_slope_per_epoch": slope(train),
            "validation_slope_per_epoch": slope(validation), "gap_slope_per_epoch": slope(gaps),
            "train_decreases_validation_increases": train[-1] < train[0] and validation[-1] > validation[0]}
    return result


def epoch_comparison(row):
    return {"epoch": row["epoch"], "model_state_sha256": row["model_state_sha256"],
            "train_eval": row["train_eval"], "validation": row["validation"],
            "validation_minus_train": {metric: row["validation"][metric] - row["train_eval"][metric] for metric in METRICS},
            "train_online": row["train_online"]}


def attempt_directory(retrieval):
    root = Path(retrieval).resolve()
    admitted = {(HERE / "retrieval").resolve(): HERE.resolve(),
                (HERE / "attempt2/retrieval").resolve(): (HERE / "attempt2").resolve()}
    require(root in admitted, "Only the original or attempt2 campaign retrieval is admitted")
    return admitted[root]


def verify_pilot(root):
    attempt = attempt_directory(root)
    # Fail before reading any scientific metrics or creating any output when
    # collection or process completion is absent.
    require(root.is_dir() and (root / "COMPLETE.json").is_file(), "No verified completed retrieval; no report written")
    done = read(root / "COMPLETE.json")
    require(done.get("status") == "complete" and done.get("all_children_actual_exit_zero") is True
            and done.get("dispatcher_actual_exit") == 0 and done.get("plots_actual_exit") == 0,
            "Pilot completion gate failed")
    require(sha(root / "RESULT_MANIFEST.json") == done["result_manifest_sha256"], "Result manifest SHA mismatch")
    manifest = read(root / "RESULT_MANIFEST.json")
    require(manifest.get("held_out_test_evaluated") is False and manifest.get("files"), "Invalid result inventory")
    for name, record in manifest["files"].items():
        require(not any(part.lower() in ("test", "test_cost") for part in Path(name).parts), "Held-out artifact rejected")
        verify_file(root, name, record)
    allowed = set(manifest["files"]) | {"COMPLETE.json", "RESULT_MANIFEST.json", "parent.stdout", "parent.stderr"}
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    require(actual == allowed or actual == allowed - {"parent.stdout", "parent.stderr"}, "Uninventoried or missing retrieval files")
    require("FAILURE.json" not in actual, "Preserved scientific failure prevents final report")
    contract = read(root / "execution_contract.json")
    prepared = read(attempt / "PREPARED.json")
    require(sha(root / "execution_contract.json") == prepared["contract_raw_sha256"]
            == sha(attempt / "EXECUTION_CONTRACT.json"), "Execution contract changed")
    require(canonical(contract) == prepared["contract_canonical_sha256"], "Canonical contract changed")
    require(contract["evaluation_scope"] == "validation_only" and contract["seeds"] == [42]
            and set(contract["hosts"]) == {"5080"}, "Pilot evaluation scope changed")
    for label in ("source_check", "dispatcher", "plots", *(d + "." + mode for d in DATASETS for mode in ("synthetic", "capacity"))):
        require(label + ".exit.json" in manifest["files"], "Missing actual exit inventory")
        verify_exit(root, label + ".exit.json")
    for dataset in DATASETS:
        data = next(d for d in contract["datasets"] if d["dataset_id"] == dataset)
        for mode, shape in (("synthetic", [3, min(8, data["loader"]["max_seq_len"])]),
                            ("capacity", [128, data["loader"]["max_seq_len"]])):
            qualification = read(root / "qualification" / dataset / (mode + ".json"))
            require(qualification["status"] == "passed" and qualification["dataset_id"] == dataset
                    and qualification["host"] == "5080" and qualification["runtime"]["device"] == "cuda:0"
                    and qualification["shape"] == shape and qualification["contract_sha256"] == canonical(contract)
                    and qualification["source_files_sha256"] == contract["source"]["files_sha256"],
                    "Native qualification identity changed")
    dispatcher = read(root / "dispatch_receipt.json")
    require(dispatcher["status"] == "complete" and dispatcher["contract_sha256"] == canonical(contract)
            and len(dispatcher["exits"]) == 2, "Dispatcher receipt mismatch")
    for dataset, expected in zip(DATASETS, dispatcher["exits"]):
        name = dataset + "__fixed_sinusoidal__seed42"
        actual_exit = verify_exit(root, name + ".exit.json")
        require(expected == actual_exit and expected["dataset_id"] == dataset and expected["seed"] == 42,
                "Dispatcher/child actual exit disagreement")
    return contract, {"attempt_directory": str(attempt), "contract_raw_sha256": prepared["contract_raw_sha256"],
                      "manifest_sha256": sha(root / "RESULT_MANIFEST.json"), "verified_files": len(manifest["files"]),
                      "completion_sha256": sha(root / "COMPLETE.json"), "dispatcher_actual_exit": 0,
                      "supervisor_own_exit_directly_observed": False}


def verify_baseline(contract):
    record = contract["baseline_contract"]
    require(Path(record["path"]).resolve() == (BASE / "execution_contract.json").resolve()
            and sha(BASE / "execution_contract.json") == record["sha256"], "Original contract identity changed")
    original = read(BASE / "execution_contract.json")
    collection = read(BASE_RETRIEVAL / "collection_receipt.json")
    manifest_path = BASE_ORIGINALS / "collection_fit_validation.manifest.json"
    require(collection["status"] == "complete" and collection["phase"] == "fit_validation"
            and collection["host"] == "5080", "Original collection identity changed")
    require(sha(manifest_path) == BASE_MANIFEST_SHA == collection["manifest_sha256"], "Original manifest changed")
    manifest = read(manifest_path)
    require(manifest["contract_sha256"] == canonical(original) and manifest["phase"] == "fit_validation"
            and manifest["all_source_input_legacy_actual_sha_verified"] is True, "Original source/input validation missing")
    # Hash only approved fit_validation evidence. This collection contains no
    # held-out Test phase, and no test_cost or comparison_v1 path is traversed.
    for name, value in manifest["files"].items():
        require(not any(part.lower() in ("test", "test_cost") for part in Path(name).parts), "Held-out baseline path rejected")
        verify_file(BASE_ORIGINALS, name, value)
    verify_exit(BASE_ORIGINALS, "fit_validation.supervisor_exit.json", legacy=True)
    for dataset in DATASETS:
        for phase in ("train", "validation"):
            verify_exit(BASE_ORIGINALS, f"control/{dataset}__full__seed42.{phase}.exit.json", legacy=True)
        old = next(d for d in original["datasets"] if d["dataset_id"] == dataset)
        new = next(d for d in contract["datasets"] if d["dataset_id"] == dataset)
        for key in ("loader", "model", "statistics", "inherited_data_identity", "quantity_boundaries_all_train_rows"):
            require(new[key] == old[key], "Comparison data/model/loader changed: " + dataset + "/" + key)
    return {"manifest_sha256": sha(manifest_path), "verified_files": len(manifest["files"]),
            "contract_raw_sha256": record["sha256"], "dispatcher_actual_exit": 0,
            "train_eval_same_ending_weights_available": False}


def compare_dataset(dataset, contract, root):
    job = root / "jobs" / (dataset + "__fixed_sinusoidal__seed42")
    receipt = read(job / "receipt.json")
    require(receipt["status"] == "complete" and receipt["dataset_id"] == dataset and receipt["seed"] == 42
            and receipt["held_out_test_evaluated"] is False and receipt["individual_predictions_match"] is True
            and receipt["contract_sha256"] == canonical(contract), "Pilot fit receipt mismatch")
    require(receipt.get("files"), "Empty pilot fit inventory")
    for path, digest in receipt["files"].items(): verify_file(root, path, digest)
    summary = receipt["summary"]
    checkpoint = verify_file(root, receipt["selected_checkpoint"], receipt["checkpoint_sha256"])
    history = read(checkpoint.parent / "history.json")["history"]
    selected, stopping = validate_history(summary, history)
    require(receipt["selected_epoch"] == selected["epoch"] and receipt["completed_epochs"] == len(history), "Receipt epoch mismatch")
    paths = sorted((job / "curve_epochs").glob("epoch_*.json"))
    rows = [read(p) for p in paths]
    jsonl = [json.loads(line) for line in (job / "learning_curves.jsonl").read_text().splitlines()]
    require(rows == jsonl and len(rows) == len(history) == receipt["curve_epoch_count"], "Curve/history inventory mismatch")
    data = next(d for d in contract["datasets"] if d["dataset_id"] == dataset)
    for index, (row, hist) in enumerate(zip(rows, history), 1):
        require(row["epoch"] == index and row["dataset_id"] == dataset and row["seed"] == 42
                and row["evaluation_mode"] == "eval" and row["monitoring_rng_preserved"] is True
                and row["model_state_unchanged"] is True, "Invalid matched-weight curve row")
        for split, key in (("train", "train_eval"), ("validation", "validation")):
            require(row[key]["count"] == data["inherited_data_identity"]["populations"][split]["target_count"], "Curve population changed")
            for metric in METRICS: finite(row[key][metric])
            close(row[key]["joint_objective"], row[key]["log_qty_mse"] + row[key]["time_nll"], "curve objective")
        for metric, history_key in HISTORY_KEYS.items(): close(row["validation"][metric], hist[history_key], "curve/history " + metric)
        for key, value in row["train_online"].items():
            if key in hist: require(value == hist[key], "Online telemetry changed")
    best, last = rows[selected["epoch"] - 1], rows[-1]
    require(best["model_state_sha256"] == receipt["state_sha256"] == summary["checkpoint_state_sha256"], "Selected curve/checkpoint state mismatch")
    expected_population = data["inherited_data_identity"]["populations"]["validation"]
    for replay in ("validation", "validation_replay"):
        require(receipt[replay]["target_identity"] == expected_population, "Selected replay population mismatch")
        for metric in ("qty_rmse", "qty_mae", "time_nll"):
            close(receipt[replay]["metrics"]["overall"][metric], best["validation"][metric], "selected replay " + metric)
    old_job = BASE_ORIGINALS / "run" / (dataset + "__full__seed42")
    old_train = read(old_job / "train_receipt.json")
    old_validation = read(old_job / "validation_receipt.json")
    for item in (old_train, old_validation):
        require(item["status"] == "complete" and item["held_out_test_evaluated"] is False and item["arm"] == "full"
                and item["seed"] == 42 and item["dataset"] == dataset, "Baseline scope/receipt mismatch")
    old_run = old_job / "runs/full/count_only_log_regression/seed_42"
    old_summary, old_history = read(old_run / "summary.json"), read(old_run / "history.json")["history"]
    old_best, old_stopping = validate_history(old_summary, old_history)
    require(old_validation["target_identity"] == expected_population and old_validation["individual_predictions_match"] is True,
            "Baseline target/replay changed")
    for metric in ("qty_rmse", "qty_mae", "time_nll"):
        close(old_validation["metrics"]["overall"][metric], old_summary[SUMMARY_KEYS[metric]], "baseline replay " + metric)
    differences = {metric: signed_change(best["validation"][metric], old_summary[key]) for metric, key in SUMMARY_KEYS.items()}
    postselection = {metric: {"train": signed_change(last["train_eval"][metric], best["train_eval"][metric]),
                             "validation": signed_change(last["validation"][metric], best["validation"][metric])} for metric in METRICS}
    return {"dataset_id": dataset, "seed": 42, "split": "validation", "original_learned_position": {
            "best_epoch": old_best["epoch"], "completed_epochs": len(old_history), "stopping": old_stopping,
            "best_validation": epoch_metrics(old_best), "last_validation": epoch_metrics(old_history[-1]),
            "full_train_eval_available": False},
        "fixed_sinusoidal": {"best_epoch": selected["epoch"], "completed_epochs": len(history), "stopping": stopping,
            "best_validation": best["validation"], "selected_epoch": epoch_comparison(best), "last_epoch": epoch_comparison(last),
            "last_minus_selected": postselection, "last_ten_epochs": late_changes(rows),
            "minimum_epoch_by_validation_metric": {metric: min(rows, key=lambda row: row["validation"][metric])["epoch"] for metric in METRICS}},
        "fixed_minus_original_validation": differences, "adverse_metrics": [m for m in METRICS if differences[m]["absolute"] > 0],
        "sources": {"original_history": str(old_run / "history.json"), "original_summary": str(old_run / "summary.json"),
            "original_validation_receipt": str(old_job / "validation_receipt.json"), "fixed_receipt": str(job / "receipt.json"),
            "fixed_history": str(checkpoint.parent / "history.json"), "fixed_curves": str(job / "learning_curves.jsonl")}}


def fmt(value):
    return "산출 불가" if value is None else f"{value:.6g}"


def render(summary):
    lines = ["# PHEN 고정 위치 인코딩 시험: RAF·Taxi seed 42", "",
        "실제 학습 종료와 선택 checkpoint의 Validation 재현을 확인한 뒤 작성한 비교이다. 모든 성능 수치는 Validation 기준이며 Test는 사용하지 않았다.", "",
        "## 1. 기존 PHEN과 고정 위치 PHEN의 선택 결과", "",
        "각 실행은 Validation 수량 RMSE가 처음 최저가 된 epoch를 선택했다. 아래 차이는 고정 위치 − 기존 학습형 위치이며, 양수는 해당 오차·손실의 악화다.", "",
        "| 데이터 | 기존 선택 / 종료 epoch | 고정 선택 / 종료 epoch |",
        "|---|---:|---:|"]
    for row in summary["datasets"]:
        old, new = row["original_learned_position"], row["fixed_sinusoidal"]
        lines.append(f"| {NAMES[row['dataset_id']]} | {old['best_epoch']} / {old['completed_epochs']} | {new['best_epoch']} / {new['completed_epochs']} |")
    lines += ["", "| 데이터 | Validation 지표 | 기존 PHEN | 고정 위치 PHEN | 절대차 | 상대차 (%) |", "|---|---|---:|---:|---:|---:|"]
    for row in summary["datasets"]:
        for metric in METRICS:
            change = row["fixed_minus_original_validation"][metric]
            lines.append(f"| {NAMES[row['dataset_id']]} | {LABELS[metric]} | {fmt(row['original_learned_position']['best_validation'][metric])} | {fmt(row['fixed_sinusoidal']['best_validation'][metric])} | {fmt(change['absolute'])} | {fmt(change['relative_percent'])} |")
    lines += ["", "상대차는 `(고정 − 기존) / |기존| × 100`이다. 시간 NLL은 오차율이 아니므로 상대차를 정확도 개선율로 해석하지 않는다.", "",
              "## 2. 같은 epoch 종료 가중치의 Train·Validation 차이", "",
              "고정 위치 실행의 Train·Validation은 모두 전체 대상에 대한 eval 모드 평가다. 학습 중 변화하는 가중치와 dropout으로 누적한 online Train 손실과 구분한다.", "",
              "| 데이터 | 시점 | 지표 | Train eval | Validation | Validation − Train |", "|---|---|---|---:|---:|---:|"]
    for row in summary["datasets"]:
        for kind, label in (("selected_epoch", "선택"), ("last_epoch", "종료")):
            point = row["fixed_sinusoidal"][kind]
            for metric in METRICS:
                lines.append(f"| {NAMES[row['dataset_id']]} | {label} epoch {point['epoch']} | {LABELS[metric]} | {fmt(point['train_eval'][metric])} | {fmt(point['validation'][metric])} | {fmt(point['validation_minus_train'][metric])} |")
    lines += ["", "## 3. 학습 후반의 변화", "",
              "마지막 10개 완료 epoch(10개 미만이면 전체)를 고정 구간으로 사용했다. 기울기는 원시 epoch 지표에 대한 단순 최소제곱 기울기이며 통계적 유의성 검정이 아니다.", "",
              "| 데이터 / epoch 구간 | 지표 | Train 마지막−처음 | Validation 마지막−처음 | Train 기울기 / epoch | Validation 기울기 / epoch |", "|---|---|---:|---:|---:|---:|"]
    notes = []
    for row in summary["datasets"]:
        late = row["fixed_sinusoidal"]["last_ten_epochs"]
        for metric, value in late["metrics"].items():
            lines.append(f"| {NAMES[row['dataset_id']]} {late['start_epoch']}–{late['end_epoch']} | {LABELS[metric]} | {fmt(value['train_end_minus_start'])} | {fmt(value['validation_end_minus_start'])} | {fmt(value['train_slope_per_epoch'])} | {fmt(value['validation_slope_per_epoch'])} |")
        adverse = row["adverse_metrics"]
        notes += ["", f"{NAMES[row['dataset_id']]}의 선택 Validation에서 기존보다 높아진 지표: " + (", ".join(LABELS[m] for m in adverse) if adverse else "없음") + "."]
        opposing = [LABELS[m] for m, value in late["metrics"].items() if value["train_decreases_validation_increases"]]
        notes.append("후반 구간에서 Train은 낮아지고 Validation은 높아진 지표: " + (", ".join(opposing) if opposing else "없음") + ". 이 패턴의 존재·부재만으로 과적합 원인을 확정하지 않는다.")
        minima = row["fixed_sinusoidal"]["minimum_epoch_by_validation_metric"]
        notes.append(f"고정 위치 실행의 수량 RMSE 선택은 epoch {minima['qty_rmse']}, 시간 NLL 최저는 epoch {minima['time_nll']}, log1p 수량 MSE 최저는 epoch {minima['log_qty_mse']}다.")
    lines += notes
    lines += ["", "## 4. 이 비교에서 판단할 수 있는 범위", "",
        "- 기존 PHEN에는 동일한 epoch 종료 가중치로 측정한 전체 Train eval 기록이 없다. 기존 online Train 손실을 이번 Train eval과 직접 비교하거나 일반화 격차가 얼마나 줄었다고 주장하지 않는다.",
        "- 바뀐 것은 위치 표현 방식이다. 학습형 위치 테이블을 base 10000, 진폭 1의 고정 sin/cos로 바꿨으므로 위치 함수·값의 스케일·학습 가능성이 함께 달라진다. 차이를 미학습 위치 행 하나의 인과 효과로 분리할 수 없다.",
        "- 두 데이터셋 각각 seed 42 한 번의 대조다. 다른 seed·Intermittent·새 데이터에 대한 일반화나 정식 모델 교체를 확정하지 않는다.",
        "- 선택 epoch 이후의 손실 변화는 수량 RMSE 선택과 다른 지표의 최저점이 일치하는지 함께 읽는다. 모든 지표에 유리한 checkpoint가 존재한다고 가정하지 않는다.",
        "- 추가 Train 평가 시간을 포함한 wall time을 순수 epoch 학습 속도로 보고하지 않는다. 이번 실행으로 속도·메모리 우월성을 주장하지 않는다.", "",
        "## 5. 완료 및 원본 연결", "",
        f"회수 manifest {summary['verification']['pilot']['verified_files']}개 파일과 기존 fit_validation manifest {summary['verification']['baseline']['verified_files']}개 파일의 byte/SHA를 확인했다. 두 학습 child와 dispatcher의 실제 종료 코드는 0이다. 상위 supervisor 자신의 OS 종료를 직접 관측했다고 주장하지 않는다."]
    for row in summary["datasets"]:
        lines += ["", f"**{NAMES[row['dataset_id']]} 원본 (Train/Validation 전용)**"]
        for label, path in row["sources"].items(): lines.append(f"- {label}: `{path}`")
    lines += ["", "세부 수치와 선택→종료 변화는 같은 디렉터리의 SUMMARY.json에 보존했다.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, default=HERE / "attempt2/retrieval",
                        help="Defaults to attempt2; the original campaign retrieval can be selected explicitly")
    parser.add_argument("--output", type=Path,
                        help="Defaults to analysis_v1 inside the selected attempt directory")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        assert signed_change(12, 10) == {"absolute": 2., "relative_percent": 20., "direction": "higher"}
        assert signed_change(8, 10)["relative_percent"] == -20.
        assert signed_change(-2, -1)["relative_percent"] == -100.
        assert signed_change(1, 0)["relative_percent"] is None
        assert slope([2, 4, 6]) == 2. and slope([1]) is None
        assert attempt_directory(HERE / "retrieval") == HERE.resolve()
        assert attempt_directory(HERE / "attempt2/retrieval") == (HERE / "attempt2").resolve()
        try:
            attempt_directory(HERE / "attempt3/retrieval")
        except ValueError:
            pass
        else:
            raise AssertionError("An unapproved attempt was admitted")
        print("Pure calculation self-check passed; no scientific result files generated")
        return 0
    root = args.retrieval.resolve()
    attempt = attempt_directory(root)
    output = args.output if args.output is not None else attempt / "analysis_v1"
    contract, pilot_proof = verify_pilot(root)
    baseline_proof = verify_baseline(contract)
    rows = [compare_dataset(dataset, contract, root) for dataset in DATASETS]
    result = {"schema": "phen_fixed_position_pilot_comparison_v1", "status": "complete", "split": "validation",
        "held_out_test_used": False, "seed": 42, "created_at": datetime.now(timezone.utc).isoformat(),
        "position_policy": contract["position_encoding"], "verification": {"pilot": pilot_proof, "baseline": baseline_proof},
        "datasets": rows, "analysis_source_sha256": sha(__file__)}
    if args.check_only:
        print(json.dumps({"status": "verified", "datasets": list(DATASETS), "output_written": False}))
        return 0
    # Exclusive directory creation prevents silent regeneration/overwrites.
    output.mkdir(parents=True, exist_ok=False)
    (output / "SUMMARY.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (output / "REPORT.md").write_text(render(result))
    print(json.dumps({"status": "complete", "report": str(output / "REPORT.md"), "summary": str(output / "SUMMARY.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
