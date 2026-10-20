"""Local, immutable three-seed Shared15 versus completed matched-head MAC comparison."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

from paper.scripts import manage_titans_mac_accuracy9 as manager
from paper.scripts import prepare_titans_mac_accuracy9 as prep

ROOT, BUNDLE, REPORT = prep.ROOT, prep.BUNDLE, prep.REPORT
SHARED = "titantpp_history_mlp_shared15_v1"
MAC = "titantpp_titans_mac"
DATASETS, SEEDS, SPLITS = manager.runtime.DATASETS, (42, 52, 62), ("validation", "test")
METRICS = (("overall", "qty_rmse"), ("overall", "qty_mae"), ("tail", "qty_rmse"),
           ("tail", "qty_mae"), ("overall", "time_nll"), ("tail", "time_nll"))
PAPER = ROOT / "reports/titantpp_shared_paper_preparation_20261020_v1"
PAPER_MANIFEST_SHA = "b87d3e7f22a44717c8a0ba58286ce1e737361bc1ccec9642be3afb5908dd47d7"
read, sha, require = prep.read, prep.sha, prep.require


def verify_shared_evidence(directory=PAPER, *, expected_manifest_sha=PAPER_MANIFEST_SHA):
    directory = Path(directory)
    require(sha(directory / "MANIFEST.json") == expected_manifest_sha, "Shared evidence manifest changed")
    files = read(directory / "MANIFEST.json")["files"]
    require(files and {"RESULT_TABLES.json", "SOURCE_VERIFICATION.json", "REPRODUCIBILITY.json"} <= set(files), "Missing Shared evidence inventory")
    for name, digest in files.items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory.resolve()) and sha(path) == digest, "Shared evidence file changed or escaped directory")
    table, verified, reproduction = (read(directory / name) for name in (
        "RESULT_TABLES.json", "SOURCE_VERIFICATION.json", "REPRODUCIBILITY.json"))
    require(table["schema"] == "shared_final_three_seed_paper_evidence_v1"
        and table["status"] == "verified_existing_results_reaggregated"
        and table["intermittent_final_best_verified"] is True and table["seeds"] == list(SEEDS)
        and table["data_variant"] == "original_uncapped", "Shared final selection is not verified")
    require(verified["status"] == "PASS" and verified["remote_calls"] == verified["new_training"] == verified["new_inference"] == 0,
        "Shared source verification differs")
    bindings = {**verified["file_bindings"], **verified["final_receipts"]}
    require(bindings and verified["final_receipts"], "Empty Shared original bindings")
    for name, digest in bindings.items():
        path = Path(name)
        require(path.is_absolute() and path.resolve().is_relative_to(ROOT.resolve()) and sha(path) == digest,
            "Shared bound original changed or escaped project")
    return table, reproduction, {"directory": str(directory), "manifest_sha256": expected_manifest_sha,
        "files_sha256": files, "verified_original_bindings": len(bindings)}


def expected_population(reproduction, dataset, split):
    data = reproduction["datasets"][dataset]
    return data["test_population"] if split == "test" else data["training_validation_identity"]["populations"][split]


def verify_common_conditions(contract, reproduction):
    require(reproduction["data_variant"] == "original_uncapped" and reproduction["quantity_capping_applied"] is False
        and reproduction["training_condition"] == contract["common_condition"]
        and reproduction["budget"] == contract["training"], "Shared/MAC data, loss or selection budget differs")
    require(reproduction["shared_training_runtime"] == contract["hosts"]["5090"]["runtime_expected"], "Shared/MAC training Runtime differs")
    for definition in contract["datasets"]:
        dataset = definition["dataset_id"]; old = reproduction["datasets"][dataset]
        require(definition["loader"] == old["loader"] and all(definition["model"].get(k) == v
            for k, v in old["model_head"].items()), "Common loader or quantity/time head differs")
        require(definition["quantity_boundaries_all_train_rows"][-1] == old["tail_threshold"], "Frozen Train tail threshold differs")
        identity = old["training_validation_identity"]
        for kind, suffix in (("data", "train_validation.parquet"), ("split_manifest", "split_manifest.json")):
            require(contract["input_files"][f"inputs/data/{dataset}/{suffix}"] == identity[kind]["sha256"], "Train/Validation input SHA differs")
        require(contract["input_files"][f"inputs/full/{dataset}.parquet"] == old["full_data"]["sha256"], "Full original input SHA differs")
        require(definition["full_data_identity"]["populations"]["test"] == expected_population(reproduction, dataset, "test"), "Test population differs")
    return True


def evaluation_runtime_context(contract, reproduction):
    """Preserve the known reference-evaluation difference without rerunning references."""
    native = contract["hosts"]["5090"]["runtime_expected"]
    return {
        "shared_training": {"host": "5090", "runtime": reproduction["shared_training_runtime"]},
        "shared_validation_reference": {"host": "5090", "runtime": reproduction["shared_training_runtime"]},
        "shared_test_reference": {"host": "5080", "runtime": reproduction["shared_test_runtime"], "stored_results_reused": True},
        "mac_training_validation_test": {"host": "5090", "runtime": native},
        "all_accuracy_endpoints_same_gpu_runtime": False,
        "limitation": "Shared Test reused fixed 5080/cuDNN91900 results; Shared training/Validation and new MAC use 5090/cuDNN92000. This is not an all-endpoint same-GPU accuracy comparison.",
    }


def _key(row):
    return row["dataset"], row["split"], row["seed"]


def _indexed(records, model):
    selected = [r for r in records if r.get("model") == model]
    keys = {_key(r) for r in selected}
    expected = {(d, s, n) for d in DATASETS for s in SPLITS for n in SEEDS}
    require(len(selected) == len(keys) == 18 and keys == expected, "Exactly 18 distinct model endpoints required")
    return {_key(r): r for r in selected}


def finite_metric(metrics, group, metric):
    value = metrics[group][metric]
    count = metrics[group]["count"]
    require(type(count) is int and count >= 0, "Invalid metric population count")
    if count == 0:
        require(value is None, "Empty cohort cannot have a numeric metric")
        return None
    require(type(value) in (int, float) and math.isfinite(value), "Missing or nonfinite metric")
    require(value >= 0, "Negative quantity error or recorded discrete TimeNLL")
    return float(value)


def _stats(values):
    require(len(values) == 3, "Three seeds required for sample SD")
    if all(value is None for value in values):
        return {"mean": None, "sample_sd": None, "seed_count": 3, "status": "N/A_empty_cohort"}
    require(all(value is not None for value in values), "Partial metric availability across seeds")
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "seed_count": 3, "status": "available"}


def compare_records(shared_records, mac_records, populations):
    shared, mac = _indexed(shared_records, SHARED), _indexed(mac_records, MAC)
    differences, aggregates = [], []
    for dataset in DATASETS:
        for split in SPLITS:
            for seed in SEEDS:
                a, b = shared[dataset, split, seed], mac[dataset, split, seed]
                require(type(a["selected_epoch"]) is int and 0 < a["selected_epoch"] <= 300
                    and type(b["selected_epoch"]) is int and 0 < b["selected_epoch"] <= 300, "Invalid selected epoch")
                expected = populations[dataset, split]
                require(b["target_identity"] == expected, "MAC target identities/quantity truth differ from Shared")
                for group in ("overall", "tail"):
                    require(a["metrics"][group]["count"] == b["metrics"][group]["count"], "Shared/MAC cohort count differs")
                require(a["metrics"]["overall"]["count"] == expected["target_count"], "Shared overall population differs")
                require(a["metrics"].get("tail_threshold") == b["metrics"].get("tail_threshold")
                    and a["metrics"].get("tail_rule") == b["metrics"].get("tail_rule")
                    and a["metrics"].get("time_metric") == b["metrics"].get("time_metric")
                    == "recorded_positive_integer_time_nll", "Tail or recorded-time metric definition differs")
                for group, metric in METRICS:
                    av, bv = finite_metric(a["metrics"], group, metric), finite_metric(b["metrics"], group, metric)
                    require((av is None) == (bv is None), "Shared/MAC availability differs")
                    delta = None if av is None else av - bv
                    differences.append({"dataset": dataset, "split": split, "seed": seed, "group": group, "metric": metric,
                        "shared": av, "mac": bv, "shared_minus_mac": delta,
                        "relative_percent_of_mac": 100. * delta / bv if bv is not None and bv > 0 else None,
                        "relative_status": "available" if bv is not None and bv > 0 else "N/A_zero_reference" if bv == 0 else "N/A_empty_cohort",
                        "shared_worse": delta > 0 if delta is not None else None,
                        "shared_selected_epoch": a["selected_epoch"], "mac_selected_epoch": b["selected_epoch"]})
            for group, metric in METRICS:
                rows = [r for r in differences if (r["dataset"], r["split"], r["group"], r["metric"]) == (dataset, split, group, metric)]
                aggregates.append({"dataset": dataset, "split": split, "group": group, "metric": metric,
                    "shared": _stats([r["shared"] for r in rows]), "mac": _stats([r["mac"] for r in rows]),
                    "shared_minus_mac": _stats([r["shared_minus_mac"] for r in rows]),
                    "shared_worse_seeds": [r["seed"] for r in rows if r["shared_worse"] is True]})
    return {"aggregates": aggregates, "seed_differences": differences,
        "endpoint_count_per_model": 18, "aggregate_comparison_count": 36, "seed_metric_comparison_count": 108,
        "delta_definition": "Shared15 minus MAC; negative means Shared15 has lower error or TimeNLL",
        "relative_definition": "100*(Shared15-MAC)/MAC, N/A for zero MAC or empty cohorts", "sample_sd_ddof": 1}


def _format(value):
    return "N/A" if value["mean"] is None else f"{value['mean']:.6f} ± {value['sample_sd']:.6f}"


def render_markdown(result):
    lines = ["# Shared15–MAC 동일 조건 3seed 비교", "", "고정 Validation 선택 checkpoint의 저장 결과만 재집계했습니다. 새 학습·추론·원격 호출은 없습니다.",
        "평균 ± 표본 표준편차(ddof=1), seed42·52·62입니다. 차이는 공유형−MAC이며 음수는 공유형 개선입니다.", "",
        "## 전체 수량 RMSE·MAE", "", "| 데이터 | Split | 지표 | 공유형 | MAC | 공유형−MAC | 악화 seed |", "|---|---|---|---:|---:|---:|---|"]
    def row(a):
        worse = ", ".join(map(str, a["shared_worse_seeds"])) or "없음"
        return f"| {a['dataset']} | {a['split']} | {a['group']}/{a['metric']} | {_format(a['shared'])} | {_format(a['mac'])} | {_format(a['shared_minus_mac'])} | {worse} |"
    lines += [row(a) for a in result["aggregates"] if a["group"] == "overall" and a["metric"] != "time_nll"]
    lines += ["", "## 큰 수량·시간 지표", "", "| 데이터 | Split | 지표 | 공유형 | MAC | 공유형−MAC | 악화 seed |", "|---|---|---|---:|---:|---:|---|"]
    lines += [row(a) for a in result["aggregates"] if a["group"] == "tail" or a["metric"] == "time_nll"]
    lines += ["", "## 모든 seed별 절대·상대 차이", "", "| 데이터 | Split | seed | 지표 | 공유형 | MAC | 차이 | 상대차(%) | 악화 |", "|---|---|---:|---|---:|---:|---:|---:|---|"]
    for row_ in result["seed_differences"]:
        vals = ["N/A" if row_[k] is None else f"{row_[k]:.6f}" for k in ("shared", "mac", "shared_minus_mac", "relative_percent_of_mac")]
        lines.append(f"| {row_['dataset']} | {row_['split']} | {row_['seed']} | {row_['group']}/{row_['metric']} | " + " | ".join(vals) + f" | {row_['shared_worse']} |")
    lines += ["", "## 해석 범위", "", "- 원본 데이터·공통 head·손실·선택 기준·학습 예산을 맞춘 구현 비교입니다.",
        "- MAE·큰 수량·TimeNLL도 같은 RMSE 선택 checkpoint의 결과이며 불리한 seed를 제외하지 않았습니다.",
        "- 타 모델 결과나 내부 MLP16을 합치지 않았고, 5seed 결과·통계적 유의성·원인 확정으로 확대하지 않습니다.",
        "- 공유형의 저장 Test는 5080/cuDNN91900, 공유형 학습·Validation과 신규 MAC 학습·Validation/Test는 5090/cuDNN92000입니다. 모든 정확도 평가를 같은 GPU·Runtime에서 수행한 비교는 아닙니다.",
        "- 효율 측정은 별도 5090 비용 실험이며 정확도 학습의 수렴 비용과 구분합니다."]
    return "\n".join(lines) + "\n"


def run(integrity_sha, *, check_only=False):
    final_path = BUNDLE / "retrieval/final_receipt.json"
    if not final_path.exists():
        require(check_only, "MAC9 is incomplete; no comparison output is allowed")
        return {"status": "waiting_for_complete_MAC9", "remote_calls": 0, "outputs_created": False}
    final = read(final_path)
    require(final["status"] == "complete" and final["new_fit_count"] == final["validation_replay_count"]
        == final["candidate_test_count"] == 9 and final["actual_scientific_worker_exit0_count"] == 27
        and final["actual_dispatcher_exit0_count"] == 1 and final["held_out_test_evaluated"] is True, "MAC9 receipt is incomplete")
    c = prep.verify(integrity_sha)
    retrieval = BUNDLE / "retrieval/5090"
    require(sha(retrieval / "original.tar.gz") == final["archive_sha256"]
        and sha(retrieval / "manifest.json") == final["manifest_sha256"], "MAC archive/manifest changed")
    manifest = read(retrieval / "manifest.json")["files"]
    require(manifest and len(manifest) == final["file_count"], "MAC immutable original inventory is empty or incomplete")
    for name, digest in manifest.items():
        require(sha(manager.runtime.local_path(retrieval / "originals", name)) == digest, "MAC collected original changed")
    manager.verify_collected(c, retrieval / "originals")
    source = REPORT / "NEW_MAC_RESULTS.json"
    require(final["report_files"].get(str(source)) == sha(source), "MAC result table differs from final receipt")
    mac_result = read(source)
    require(mac_result["data_variant"] == "original_uncapped" and mac_result["seeds"] == list(SEEDS), "MAC result identity differs")
    records = mac_result["records"]
    require(len(records) == 27 and {(r["dataset"], r["model"], r["seed"], r["stage"]) for r in records}
        == {(d, MAC, n, s) for d in DATASETS for n in SEEDS for s in ("train", *SPLITS)}, "MAC result endpoint matrix differs")
    mac_records = []
    for record in records:
        path = retrieval / "originals/run" / f"{record['dataset']}__{MAC}__seed{record['seed']}" / f"{record['stage']}_receipt.json"
        receipt = read(path)
        require(record["receipt"] == receipt, "MAC result row differs from bound original receipt")
        if record["stage"] in SPLITS:
            mac_records.append({"dataset": record["dataset"], "model": MAC, "seed": record["seed"], "split": record["stage"],
                "metrics": receipt["metrics"], "selected_epoch": receipt["selected_epoch"], "target_identity": receipt["target_identity"]})
    table, reproduction, shared_proof = verify_shared_evidence()
    verify_common_conditions(c, reproduction)
    populations = {(d, s): expected_population(reproduction, d, s) for d in DATASETS for s in SPLITS}
    result = {"schema": "titans_mac_shared_three_seed_comparison_v1", "status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
        "data_variant": "original_uncapped", "models": [SHARED, MAC], "seeds": list(SEEDS), "remote_calls": 0,
        "new_training": 0, "new_inference": 0, "held_out_test_evaluated": True,
        "runtime_context": evaluation_runtime_context(c, reproduction),
        "source_proof": {"mac_final_receipt_sha256": sha(final_path), "mac_results_sha256": sha(source),
            "mac_integrity_sha256": integrity_sha, "mac_archive_sha256": final["archive_sha256"], "mac_manifest_sha256": final["manifest_sha256"],
            "shared": shared_proof}, **compare_records(table["records"], mac_records, populations)}
    if check_only:
        return {"status": "complete_verified_outputs_not_written", "remote_calls": 0, "outputs_created": False,
            "seed_metric_comparison_count": result["seed_metric_comparison_count"]}
    destination = REPORT / "comparison_v1"
    require(not destination.exists(), "Existing comparison must be preserved; no overwrite")
    destination.mkdir(parents=True, exist_ok=False)
    prep.write(destination / "MAC_SHARED_THREE_SEED.json", result)
    with (destination / "MAC_SHARED_THREE_SEED.md").open("x") as stream: stream.write(render_markdown(result))
    prep.write(destination / "MANIFEST.json", {"status": "complete", "files": {
        name: sha(destination / name) for name in ("MAC_SHARED_THREE_SEED.json", "MAC_SHARED_THREE_SEED.md")}})
    return {"status": "complete", "report": str(destination), "seed_metric_comparison_count": 108, "remote_calls": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integrity-sha", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.integrity_sha, check_only=args.check_only), ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
