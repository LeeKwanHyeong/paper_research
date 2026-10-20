"""Local evidence-only five-seed aggregation; no remote, model or inference route.

Missing completion receipts return waiting without creating any file. Checkpoint
loads, after every archive gate passes, read CPU metadata/tensors only. The
sealed extra42 source, contracts, old results and manuscripts remain immutable.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
NAME = "titantpp_shared_external_extra42_20261020_v1"
RAW9 = "titantpp_history_mlp_shared_raw9_20261020_v1"
RAW9_EXECUTION = "execution_5090_v1"
LEGACY = ("titantpp_shared_raf_taxi6_test_20261020_v1", "titantpp_shared_inter3_interim_test_20261020_v1")
SCHEMA = "titantpp_shared_external_five_seed_v1"
SHARED = "titantpp_history_mlp_shared15_v1"
EXTERNAL = ("rmtpp", "thp", "nhp", "sahp", "s2p2_matched_head", "attnhp_matched_head")
MODELS = (SHARED, *EXTERNAL)
DATASETS = ("raf_spare_parts", "yellow_trip_hourly", "intermittent_frozen_5000")
OLD_SEEDS, NEW_SEEDS = (42, 52, 62), (72, 82)
SEEDS = (*OLD_SEEDS, *NEW_SEEDS)
SPLITS, GROUPS = ("validation", "test"), ("overall", "tail")
METRICS = ("qty_rmse", "qty_mae", "time_nll")
NA = "N/A_unverified_original_support"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def local(root, value):
    base = Path(root).resolve()
    path = Path(value)
    candidate = path if path.is_absolute() else base / path
    require(not candidate.is_symlink(), "Symlink evidence is not admitted")
    result = candidate.resolve()
    require(result.is_relative_to(base), "Evidence escaped its local root")
    return result


def verify_bindings(root, bindings):
    require(isinstance(bindings, dict) and bindings, "Empty evidence inventory")
    for path, digest in bindings.items():
        require(sha(local(root, path)) == digest, "Evidence SHA changed: " + path)


def verify_collection(root, folder):
    folder = local(root, folder)
    receipt = read(folder / "receipt.json")
    require(receipt.get("status") in ("complete", "verified"), "Collection is incomplete")
    require(sha(folder / "original.tar.gz") == receipt["archive_sha256"]
            and sha(folder / "manifest.json") == receipt["manifest_sha256"], "Archive/manifest SHA changed")
    inventory = read(folder / "manifest.json")["files"]
    require(len(inventory) == receipt["file_count"], "Collection file count changed")
    verify_bindings(folder / "originals", inventory)
    return folder / "originals", receipt


def identity(receipt, dataset, model, seed, stage):
    require(receipt.get("status") == "complete" and receipt.get("dataset") == dataset
            and receipt.get("arm") == model and receipt.get("seed") == seed
            and type(receipt.get("seed")) is int, "Receipt scientific identity differs")
    require(receipt.get("stage", receipt.get("split")) == stage
            and receipt.get("held_out_test_evaluated") is (stage == "test"), "Receipt split differs")
    require(type(receipt.get("selected_epoch")) is int and 0 < receipt["selected_epoch"] <= 300,
            "Selected epoch is invalid")


def expected_matrix(seeds, splits=SPLITS):
    return {(d, m, s, split) for d in DATASETS for m in MODELS for s in seeds for split in splits}


def index_rows(rows, expected):
    indexed = {}
    for row in rows:
        key = row["dataset"], row["model"], row["seed"], row["split"]
        require(key not in indexed, "Duplicate scientific endpoint")
        indexed[key] = row
    require(set(indexed) == expected, "Incomplete or foreign scientific matrix")
    return indexed


def checked_metric(value, *, quantity=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (not quantity or value >= 0), "Nonfinite or invalid metric")
    return float(value)


def normalize(dataset, model, seed, split, receipt, provenance):
    metrics = deepcopy(receipt["metrics"])
    for group in GROUPS:
        metrics.setdefault(group, {"count": None, "status": NA})
        for metric in METRICS:
            value = metrics[group].get(metric)
            if value is not None:
                checked_metric(value, quantity=metric != "time_nll")
            else:
                metrics[group][metric] = None
    for metric in METRICS[:2]:
        checked_metric(metrics["overall"][metric], quantity=True)
    return {"dataset": dataset, "model": model, "seed": seed, "split": split,
        "selected_epoch": receipt["selected_epoch"], "metrics": metrics,
        "target_identity": deepcopy(receipt.get("target_identity")),
        "checkpoint_sha256": receipt.get("checkpoint_sha256"),
        "state_sha256": receipt.get("state_sha256", receipt.get("state_tensor_sha256")),
        "provenance": provenance, "data_variant": "original_uncapped"}


def verify_new(root, bundle, report, contract):
    final = read(bundle / "retrieval/final_receipt.json")
    require(final.get("status") == "complete" and final.get("new_fit_count") == 42
        and final.get("validation_replay_count") == final.get("candidate_test_count") == 42
        and final.get("actual_scientific_worker_exit0_count") == 126
        and final.get("actual_dispatcher_exit0_count") == 1
        and final.get("held_out_test_evaluated") is True, "Extra42 completion is incomplete")
    verify_bindings(root, final["report_files"])
    originals, collected = verify_collection(root, bundle / "retrieval/5080")
    require(all(final.get(k) == collected.get(k) for k in ("archive_sha256", "manifest_sha256", "file_count")),
            "Extra42 final/collection evidence differs")
    require(sha(originals / "execution_contract.json") == sha(bundle / "execution_contract.json"),
            "Collected contract differs")
    native = read(originals / "final_receipt.json")
    require(native.get("status") == "complete" and native.get("contract_sha256") == canonical(contract)
            and native.get("completed_fit_count") == native.get("validation_replay_count") == native.get("test_count") == 42
            and native.get("owned_worker_pids_absent") is True, "Native extra42 completion differs")
    dispatcher = read(originals / "dispatcher_exit.json")
    require(type(dispatcher.get("actual_os_returncode")) is int and dispatcher["actual_os_returncode"] == 0
            and dispatcher.get("timeout") is False and dispatcher.get("actual_child_exit_observed") is True,
            "Extra42 dispatcher actual exit0 missing")
    jobs = contract["jobs"]
    require(len(jobs) == 42 and {(j["dataset"], j["arm"], j["seed"]) for j in jobs}
            == {(d, m, s) for d in DATASETS for m in MODELS for s in NEW_SEEDS}, "Extra42 job matrix differs")
    export = read(report / "NEW_SEED_RESULTS.json")
    require(export.get("data_variant") == "original_uncapped" and export.get("extra_seeds") == list(NEW_SEEDS)
            and export.get("five_seed_merged") is False, "New-seed export identity differs")
    exported = {(r["dataset"], r["model"], r["seed"], r["stage"]): r["receipt"] for r in export["records"]}
    require(len(export["records"]) == len(exported) == 126, "Duplicate or incomplete new export")
    rows = []
    for stage in ("train", *SPLITS):
        terminal = native["stages"][stage]
        require(terminal["status"] == "complete" and set(terminal["receipts"]) == set(terminal["exits"])
                == {j["id"] for j in jobs}, "Native terminal stage matrix differs")
        for j in jobs:
            key = j["dataset"], j["arm"], j["seed"], stage
            bound = terminal["receipts"][j["id"]]
            path = local(originals, bound["path"])
            require(sha(path) == bound["sha256"], "Native stage receipt changed")
            r = read(path)
            require(exported.get(key) == r, "Exported receipt differs from collected original")
            identity(r, *key)
            require(r.get("contract_sha256") == canonical(contract)
                    and r.get("time_alpha") == 1. and r.get("loss_mode") == "log_mse"
                    and r.get("time_input") == "log1p" and r.get("native_runtime_checked") is True,
                    "New common input/objective/runtime differs")
            exited = terminal["exits"][j["id"]]
            require(type(exited.get("actual_os_returncode")) is int and exited["actual_os_returncode"] == 0
                    and exited.get("timeout") is False, "Native worker actual exit0 missing")
            verify_bindings(originals, r["files"])
            if stage != "train":
                train = read(path.with_name("train_receipt.json"))
                require(all(r.get(k) == train.get(k) for k in ("selected_epoch", "checkpoint_sha256", "state_sha256")),
                        "New endpoint changed Validation-selected checkpoint")
                require(r.get("parameters_unchanged") is True and r.get("context_unchanged") is True,
                        "New evaluation changed model state")
                if stage == "validation":
                    require(r.get("validation_replay_passed") is True and r.get("individual_predictions_match") is True,
                            "New native Validation replay missing")
                data = next(d for d in contract["datasets"] if d["dataset_id"] == j["dataset"])
                expected = (data["inherited_data_identity"] if stage == "validation" else data["full_data_identity"])["populations"][stage]
                require(r["target_identity"] == expected, "New target population differs")
                rows.append(normalize(*key, r, {"kind": "new_native_receipt", "path": str(path), "sha256": sha(path)}))
    index_rows(rows, expected_matrix(NEW_SEEDS))
    return rows


def legacy_tests(root, science):
    rows, refs = [], {}
    for operation in LEGACY:
        bundle = root / "search_artifacts" / operation
        final = read(bundle / "final_receipt.json")
        require(final.get("status") == "complete" and final.get("checkpoint_reselected") is False
                and final.get("historical_baselines_reevaluated") is False, "Legacy Test completion differs")
        verify_bindings(root, final["result_files"])
        originals, collection = verify_collection(root, bundle / "retrieval")
        require(sha(bundle / "retrieval/receipt.json") == final["collection_receipt_sha256"], "Legacy receipt changed")
        ec = read(originals / "evaluation_contract.json")
        registry_path = local(originals, ec["references"]["path"])
        require(sha(registry_path) == ec["references"]["sha256"], "Reference registry changed")
        registry = read(registry_path)
        verify_bindings(root, registry["source_bindings"])
        for ref in registry["references"]:
            if ref["model"] in EXTERNAL:
                key = ref["dataset"], ref["model"], ref["seed"], ref["split"]
                require(key not in refs, "Duplicate historical reference")
                refs[key] = ref
        result = read(root / "reports" / operation / "THREE_SEED_TEST.json")
        datasets = result.get("datasets", {result.get("dataset"): result})
        for dataset, group in datasets.items():
            for r in group["records"]:
                if r["model"] not in MODELS:
                    continue  # Stored MLP16 ablation is not a five-seed external model.
                model, seed = r["model"], r["seed"]
                require(seed in OLD_SEEDS and r["split"] == "test", "Legacy endpoint scope differs")
                if model == SHARED:
                    path = originals / "results" / f"{dataset}__{model}__seed{seed}__losslog_mse" / "test_receipt.json"
                    original = read(path)
                    identity(original, dataset, model, seed, "test")
                    require(original["metrics"] == r["metrics"] and original["selected_epoch"] == r["selected_epoch"]
                            and original["parameters_unchanged"] is True, "Legacy candidate Test differs")
                else:
                    original = refs[dataset, model, seed, "test"]
                    require(original["metrics"] == r["metrics"] and original["selected_epoch"] == r["selected_epoch"],
                            "Legacy Test/reference metrics differ")
                require(r["target_identity"] == science["datasets"][dataset]["populations"]["test"],
                        "Legacy Test target population differs")
                rows.append(normalize(dataset, model, seed, "test", original,
                    {"kind": "stored_test_reference", "report_path": str(root / "reports" / operation / "THREE_SEED_TEST.json")}))
    index_rows(rows, expected_matrix(OLD_SEEDS, ("test",)))
    return rows, refs


def checkpoint_payload(path):
    import torch  # Only reached after completion/archive gates; CPU evidence read, no model.
    return torch.load(path, map_location="cpu", weights_only=False)


def state_sha(state):
    import torch
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        metadata = json.dumps({"name": name, "dtype": str(value.dtype), "shape": list(value.shape)},
                              sort_keys=True, separators=(",", ":")).encode()
        digest.update(len(metadata).to_bytes(8, "big")); digest.update(metadata)
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes() if value.numel() else b""
        digest.update(len(raw).to_bytes(8, "big")); digest.update(raw)
    return digest.hexdigest()


def selected_summary(summary, history, payload, dataset, model, seed):
    require(summary.get("status") == "success" and summary.get("backbone") == model
        and summary.get("seed") == seed and summary.get("variant") == "count_only_log_regression"
        and summary.get("evaluation_scope") == "validation_only"
        and summary.get("held_out_test_evaluated") is False
        and summary.get("checkpoint_monitor") == "validation_raw_quantity_rmse", "Historical summary scope differs")
    finite = [r for r in history if type(r.get("val_qty_rmse")) in (float, int) and math.isfinite(r["val_qty_rmse"])]
    require(finite, "No finite historical selection")
    selected = min(finite, key=lambda r: (r["val_qty_rmse"], r["epoch"]))
    require(summary["best_epoch"] == selected["epoch"] == payload["best_epoch"]
        and payload["backbone"] == model and payload["seed"] == seed
        and payload["checkpoint_monitor"] == "validation_raw_quantity_rmse"
        and summary["resume_identity"]["arguments"]["dataset_contract"] == dataset,
        "Historical checkpoint selection differs")
    require(summary["encoder_config"] == payload["encoder_config"]
        and summary["checkpoint_state_sha256"] == payload["model_state_sha256"] == state_sha(payload["model_state_dict"]),
        "Historical checkpoint metadata/state differs")
    for key in ("selected_metric_value", "best_val_qty_rmse"):
        require(math.isclose(summary[key], selected["val_qty_rmse"], rel_tol=1e-10, abs_tol=1e-8),
                "Historical RMSE is not the selected-checkpoint metric")
    require(math.isclose(payload["selected_metric_value"], selected["val_qty_rmse"], rel_tol=1e-10, abs_tol=1e-8)
        and math.isclose(summary["best_val_qty_mae"], selected["val_qty_mae"], rel_tol=1e-10, abs_tol=1e-8),
        "Historical MAE/RMSE does not use the identical selected epoch")
    return selected


def external_validation(root, conditions, science, refs):
    rows, bindings = [], []
    for model in EXTERNAL:
        model_bindings = conditions["models"][model]["historical_summary_bindings"]
        require(len(model_bindings) == 9 and {(b["dataset"], b["seed"]) for b in model_bindings}
                == {(d, s) for d in DATASETS for s in OLD_SEEDS}, "Historical external summary54 matrix differs")
        for binding in model_bindings:
            dataset, seed = binding["dataset"], binding["seed"]
            path = local(root, binding["path"])
            require(sha(path) == binding["sha256"], "Frozen external summary changed")
            summary, history = read(path), read(path.with_name("history.json"))["history"]
            ref = refs[dataset, model, seed, "test"]
            checkpoint = local(root, ref["checkpoint_path"])
            require(sha(checkpoint) == ref["checkpoint_sha256"], "Historical selected checkpoint binary changed")
            require(sha(path.with_name("history.json")) == ref["history_sha256"], "Historical selection history changed")
            payload = checkpoint_payload(checkpoint)
            selected = selected_summary(summary, history, payload, dataset, model, seed)
            require(summary["best_epoch"] == ref["selected_epoch"]
                    and payload["model_state_sha256"] == ref["state_tensor_sha256"], "Test and Validation selected checkpoints differ")
            metrics = {"overall": {"count": None, "status": "selected_checkpoint_summary_not_individual_replay",
                "qty_rmse": summary["best_val_qty_rmse"], "qty_mae": summary["best_val_qty_mae"],
                "time_nll": None}, "tail": {"count": None, "status": NA}}
            # Overall time is admitted only when the identical selected history supports it.
            if type(summary.get("best_val_time_nll")) in (float, int) and type(selected.get("val_time_nll")) in (float, int):
                require(math.isclose(summary["best_val_time_nll"], selected["val_time_nll"], rel_tol=1e-10, abs_tol=1e-8),
                        "Selected time metric differs")
                metrics["overall"]["time_nll"] = summary["best_val_time_nll"]
            item = {"metrics": metrics, "selected_epoch": summary["best_epoch"],
                "checkpoint_sha256": ref["checkpoint_sha256"], "state_sha256": ref["state_tensor_sha256"]}
            rows.append(normalize(dataset, model, seed, "validation", item, {"kind": "original_selected_checkpoint_summary",
                "path": str(path), "sha256": binding["sha256"], "history_sha256": sha(path.with_name("history.json")),
                "checkpoint_metadata_read_on": "cpu", "individual_validation_predictions_verified": False,
                "tail_support": NA, "expected_population": science["datasets"][dataset]["populations"]["validation"]}))
            bindings.append({**binding, "checkpoint_path": str(checkpoint), "checkpoint_sha256": ref["checkpoint_sha256"]})
    return rows, bindings


def verify_paper_gate(root, report, raw_final, old_final, science):
    gate = read(report / "PAPER_START_GATE.json")
    require(gate.get("status") == "PASS" and gate.get("paper_may_start") is True
            and gate.get("canonical_tensors_compared") is True and gate.get("remote_calls") == 0,
            "Intermittent final/temporary checkpoint gate did not pass; no aggregation")
    require(gate["original_final_receipt_sha256"] == sha(raw_final)
            and gate["frozen_test_final_receipt_sha256"] == sha(old_final), "Paper gate evidence changed")
    expected = science["paper_start_gate"]["frozen_selected_states"]
    require(len(gate["rows"]) == 3 and {r["seed"] for r in gate["rows"]} == set(OLD_SEEDS), "Incomplete paper gate")
    for row in gate["rows"]:
        value = expected[str(row["seed"])]
        require(row.get("matches") is True and row.get("validation_replay_passed") is True
            and row["final_epoch"] == row["frozen_epoch"] == value["epoch"]
            and row["final_state_sha256"] == row["frozen_state_sha256"] == value["state_sha256"],
            "Final Intermittent selected state differs from frozen Test")
    return gate


def shared_validation(root, raw_bundle, science, gate):
    final = read(raw_bundle / "retrieval/final_receipt.json")
    require(final.get("status") == "complete" and final.get("new_fit_count") == final.get("validation_replay_count") == 9
        and final.get("actual_scientific_worker_exit0_count") == 18
        and final.get("all_training_terminal_verified") is True and final.get("held_out_test_evaluated") is False,
        "Original5090 final Validation9 incomplete")
    for key in ("source_bindings", "archive_bindings", "report_files"):
        verify_bindings(root, final[key])
    originals, _ = verify_collection(root, raw_bundle / "retrieval/5090")
    c = read(raw_bundle / "execution_contract.json")
    report_paths = [local(root, p) for p in final["report_files"] if p.endswith("THREE_SEED_VALIDATION.json")]
    require(len(report_paths) == 1, "Original final Validation report is not uniquely bound")
    saved = read(report_paths[0])
    rows = []
    for job in c["jobs"]:
        dataset, seed = job["dataset"], job["seed"]
        path = originals / "run" / job["id"] / "validation_receipt.json"
        r = read(path)
        identity(r, dataset, SHARED, seed, "validation")
        require(r["validation_replay_passed"] is True
                and r["target_identity"] == science["datasets"][dataset]["populations"]["validation"], "Final native Validation target/replay differs")
        stored = [x for x in saved["records"] if x["dataset"] == dataset and x["seed"] == seed and x["model"] == SHARED]
        require(len(stored) == 1 and stored[0]["metrics"] == r["metrics"]
                and stored[0]["selected_epoch"] == r["selected_epoch"], "Final Validation report differs from original receipt")
        if dataset == "intermittent_frozen_5000":
            g = next(x for x in gate["rows"] if x["seed"] == seed)
            require(r["selected_epoch"] == g["final_epoch"] and r["state_sha256"] == g["final_state_sha256"], "Gate/final Validation state differs")
        rows.append(normalize(dataset, SHARED, seed, "validation", r,
                    {"kind": "original5090_final_validation", "path": str(path), "sha256": sha(path)}))
    index_rows(rows, {(d, SHARED, s, "validation") for d in DATASETS for s in OLD_SEEDS})
    return rows


def aggregate(records):
    indexed = index_rows(records, expected_matrix(SEEDS))
    for row in records:
        for metric in METRICS[:2]:
            checked_metric(row["metrics"]["overall"].get(metric), quantity=True)
        for group in GROUPS:
            for metric in METRICS:
                value = row["metrics"][group].get(metric)
                if value is not None:
                    checked_metric(value, quantity=metric != "time_nll")
    summaries, differences = {}, []
    for name, seeds in (("historical_three", OLD_SEEDS), ("additional_two", NEW_SEEDS), ("all_five", SEEDS)):
        entries = []
        for dataset in DATASETS:
            for split in SPLITS:
                for model in MODELS:
                    for group in GROUPS:
                        for metric in METRICS:
                            values = [indexed[dataset, model, seed, split]["metrics"][group].get(metric) for seed in seeds]
                            available = [s for s, v in zip(seeds, values) if v is not None]
                            complete = len(available) == len(seeds)
                            entries.append({"dataset": dataset, "split": split, "model": model, "group": group,
                                "metric": metric, "seeds": list(seeds), "available_seeds": available,
                                "mean": statistics.mean(values) if complete else None,
                                "sample_sd_ddof1": statistics.stdev(values) if complete else None,
                                "status": "available" if complete else "N/A_incomplete_evidence"})
        summaries[name] = entries
    for dataset in DATASETS:
        for split in SPLITS:
            for seed in SEEDS:
                for external in EXTERNAL:
                    for group in GROUPS:
                        for metric in METRICS:
                            left = indexed[dataset, SHARED, seed, split]["metrics"][group].get(metric)
                            right = indexed[dataset, external, seed, split]["metrics"][group].get(metric)
                            delta = None if left is None or right is None else left - right
                            differences.append({"dataset": dataset, "split": split, "seed": seed,
                                "candidate": SHARED, "reference": external, "group": group, "metric": metric,
                                "shared_value": left, "external_value": right, "absolute_difference": delta,
                                "relative_difference_percent": None if delta is None or right == 0 else 100 * delta / abs(right),
                                "relative_status": "N/A_reference_zero" if right == 0 else ("available" if delta is not None else "N/A_incomplete_evidence"),
                                "worsened": None if delta is None else delta > 0})
    return {"schema": SCHEMA, "status": "complete", "data_variant": "original_uncapped",
        "models": list(MODELS), "seeds": list(SEEDS), "records": deepcopy(records), "summaries": summaries,
        "seed_differences": differences, "standard_deviation": "sample ddof=1", "raw_errors_pooled": False,
        "historical_models_retrained_or_reevaluated": False, "mlp16_fits_or_evaluations": 0,
        "model_inference_calls": 0, "remote_calls": 0, "paper_text_generated": False,
        "limitations": ["Repeated Test-informed exploratory follow-up; five seeds do not create a new independent Test set.",
            "Common-head encoder adaptations do not reproduce every native paper head or tuning protocol.",
            "Three historical seeds and two new seeds retain distinct runtime/provenance tables.",
            "Historical external Validation uses same-selected-checkpoint summary/history metadata; individual rows and tail support are unverified.",
            "Unsupported five-seed tail/time cells are N/A, never partial-seed means.",
            "Mean and sample SD are descriptive; no significance or causal claim."]}


def markdown(result):
    lines = ["# Shared15·외부6종의 원본 데이터 5seed 근거표", "",
        "Validation/Test를 구분한 평균±표본SD(ddof1). 기존3seed·추가2seed 표를 함께 보존한다.",
        "자동 원고 본문이 아닌 수치 근거 자료다. 반복 Test, 공통 head 및 Runtime 차이의 한계를 유지한다.", ""]
    for cohort, values in result["summaries"].items():
        lines += ["## " + cohort, "", "| 데이터셋 | split | 모델 | RMSE | MAE | 전체 TimeNLL | 큰수량 RMSE | 큰수량 MAE | 큰수량 TimeNLL |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for dataset in DATASETS:
            for split in SPLITS:
                for model in MODELS:
                    cells = []
                    for group, metric in (("overall", "qty_rmse"), ("overall", "qty_mae"), ("overall", "time_nll"),
                                          ("tail", "qty_rmse"), ("tail", "qty_mae"), ("tail", "time_nll")):
                        r = next(x for x in values if (x["dataset"], x["split"], x["model"], x["group"], x["metric"])
                                 == (dataset, split, model, group, metric))
                        cells.append("N/A" if r["mean"] is None else f"{r['mean']:.6g} ± {r['sample_sd_ddof1']:.6g}")
                    lines.append(f"| {dataset} | {split} | {model} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## 전체 seed별 Shared15−외부 비교군 차이", "",
        "양수는 악화다. 상대 차이는 비교군 절대 값으로 나누며 비교군0이면 N/A다.", "",
        "| 데이터셋 | split | seed | 비교군 | cohort | 지표 | 절대 차이 | 상대 차이(%) |",
        "|---|---|---:|---|---|---|---:|---:|"]
    for r in result["seed_differences"]:
        absolute = "N/A" if r["absolute_difference"] is None else f"{r['absolute_difference']:+.6g}"
        relative = "N/A" if r["relative_difference_percent"] is None else f"{r['relative_difference_percent']:+.6g}"
        lines.append(f"| {r['dataset']} | {r['split']} | {r['seed']} | {r['reference']} | {r['group']} | {r['metric']} | {absolute} | {relative} |")
    lines += ["", "## 근거의 한계", "", *["- " + x for x in result["limitations"]], ""]
    return "\n".join(lines)


def run(root=ROOT, output_dir=None, *, check_only=False):
    root = Path(root).resolve(); bundle = root / "search_artifacts" / NAME; report = root / "reports" / NAME
    new_final = bundle / "retrieval/final_receipt.json"
    if not new_final.exists():
        return {"status": "waiting_for_extra42_completion", "remote_calls": 0, "output_files": []}
    raw_bundle = root / "search_artifacts" / RAW9 / RAW9_EXECUTION
    if not (raw_bundle / "retrieval/final_receipt.json").exists():
        return {"status": "waiting_for_original5090_final_validation", "remote_calls": 0, "output_files": []}
    if not (report / "PAPER_START_GATE.json").exists():
        return {"status": "waiting_for_paper_start_gate", "remote_calls": 0, "output_files": []}
    contract = read(bundle / "execution_contract.json"); science = read(report / "SCIENCE_SPEC.json")
    require(contract["science_spec_sha256"] == sha(report / "SCIENCE_SPEC.json"), "Science specification changed")
    conditions = read(report / "FROZEN_CONDITIONS.json")
    require(contract["frozen_conditions_sha256"] == sha(report / "FROZEN_CONDITIONS.json")
            and contract["frozen_conditions"] == conditions, "Frozen historical conditions changed")
    verify_bindings(root, science["prior_completion_bindings"])
    gate = verify_paper_gate(root, report, raw_bundle / "retrieval/final_receipt.json",
        root / "search_artifacts" / LEGACY[1] / "final_receipt.json", science)
    new = verify_new(root, bundle, report, contract)
    old_test, refs = legacy_tests(root, science)
    old_shared = shared_validation(root, raw_bundle, science, gate)
    old_external, bindings = external_validation(root, conditions, science, refs)
    rows = old_test + old_shared + old_external + new
    # The original Test3 Shared15 state must still equal the final Validation-selected state.
    by_key = index_rows(rows, expected_matrix(SEEDS))
    for dataset in DATASETS:
        for seed in SEEDS:
            left, right = (by_key[dataset, SHARED, seed, split] for split in SPLITS)
            require(left["selected_epoch"] == right["selected_epoch"] and left["state_sha256"] == right["state_sha256"],
                    "Shared Validation/Test selected state differs")
    result = aggregate(rows)
    result["source_bindings"] = {str(p): sha(p) for p in (new_final, raw_bundle / "retrieval/final_receipt.json",
        report / "PAPER_START_GATE.json", report / "FROZEN_CONDITIONS.json", report / "SCIENCE_SPEC.json")}
    result["historical_external_summary_bindings"] = bindings
    if check_only:
        return {"status": "verified_ready", "record_count": len(rows), "remote_calls": 0,
                "model_inference_calls": 0, "output_files": []}
    output = local(root, output_dir or report / "five_seed_aggregation_v1")
    require(not output.exists(), "Preserve existing aggregation; output directory must be new")
    serialized = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    rendered = markdown(result)
    output.mkdir(parents=True, exist_ok=False)
    (output / "FIVE_SEED_RESULTS.json").write_text(serialized)
    (output / "FIVE_SEED_RESULTS.md").write_text(rendered)
    return {"status": "complete", "record_count": len(rows), "remote_calls": 0, "model_inference_calls": 0,
            "output_files": [str(output / n) for n in ("FIVE_SEED_RESULTS.json", "FIVE_SEED_RESULTS.md")]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run(args.root, args.output_dir, check_only=args.check_only)
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(json.dumps({"status": "failed", "error": str(error), "remote_calls": 0, "output_files": []}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
