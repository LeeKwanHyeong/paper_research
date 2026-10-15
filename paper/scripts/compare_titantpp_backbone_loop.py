"""Freeze RAF references and compare completed, fixed-checkpoint loop results.

No model loading, training, inference, SSH, or checkpoint selection occurs here.
The external cohort is a historical common-head encoder comparison, not native
paper heads or freshly retrained baselines. Test has already informed this loop.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path("reports/titantpp_backbone_nonlinear_raf9_20261015_v1")
DATASET = "raf_spare_parts"
SEEDS = (42, 52, 62)
SPLITS = ("validation", "test")
METRICS = ("qty_rmse", "qty_mae")
EXTERNAL = ("rmtpp", "thp", "nhp", "sahp", "s2p2_matched_head", "attnhp_matched_head")
DIAG = Path("reports/titantpp_rank_reversal_diagnostics_20261014_v1")
COMPONENT = Path("search_artifacts/titantpp_components_taxi_raf24_test_20261014_v1")
EXTERNAL_TRAIN = Path("search_artifacts/titantpp_raf_5080_20260930_v1/retrieved/terminal_20261001_v2/original")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stats(values):
    require(len(values) == 3, "Exactly three seed metrics are required")
    require(all(type(x) in (float, int) and math.isfinite(x) and x >= 0 for x in values),
            "Nonfinite or negative error metric")
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "ddof": 1}


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["split"])].append(row)
    result = {}
    for (model, split), group in grouped.items():
        require(sorted(r["seed"] for r in group) == list(SEEDS), "Duplicate or missing seed")
        result.setdefault(model, {})[split] = {
            metric: stats([r["metrics"][metric] for r in group]) for metric in METRICS
        }
    return result


def first_best(history):
    require([r["epoch"] for r in history] == list(range(1, len(history) + 1)), "Incomplete epoch history")
    finite = [r for r in history if math.isfinite(r["val_qty_rmse"])]
    require(bool(finite), "No finite Validation selection metric")
    return min(finite, key=lambda r: (r["val_qty_rmse"], r["epoch"]))


def build_registry(root=ROOT):
    """Reverify local source evidence and freeze references before new evaluation."""
    root = Path(root).resolve()
    references = {}

    def checked(path, expected=None):
        path = path if Path(path).is_absolute() else root / path
        actual = sha(path)
        require(expected is None or actual == expected, f"Source SHA mismatch: {path}")
        references[str(path.relative_to(root))] = actual
        return path

    def read(path, expected=None):
        return json.loads(checked(path, expected).read_text())

    checked(Path(__file__).resolve())

    old = read(DIAG / "EXTERNAL_ANALYSIS.json", "cb1f5dab5de826e52f2125c7b732e60d2dc88b072e75d0c24298ab8ab1f2e533")
    code = checked(DIAG / "external_analysis.py", "86714f9262b3c8615d0aa287aaead239d484cb16d2e1d6465acd0af15b4d6556")
    spec = importlib.util.spec_from_file_location("_frozen_external_reaggregation", code)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    audit.ROOT = root
    audit.SOURCES.clear()
    binding = read(COMPONENT / "bindings.json", "88d4e6388cb58b1bd9598698a0f2cb1ac8505fb4be5e05b2785d3c2d27b0dba0")
    final = read(COMPONENT / "final_receipt.json", "9b4db5b90700910d6dc650966fb15753e4036f9ae2b83068cc71d073c0e478f6")
    require(final["status"] == "complete", "Historical CNN evaluation incomplete")
    native = read(COMPONENT / "native_input_receipt.json", "c3a9c2ebaa1c00e50fd41d20c999c93ae4b0762f3b00cfffbf766c8d6a4101f5")
    results_manifest = read(COMPONENT / "evaluation_results_manifest.json", "bf3c3ae028bbf9a6c6f1f2de1ca3881a5b8d2b13feab84d3601616187acf0a68")
    originals_manifest = read(COMPONENT / "originals_manifest.json", "f0cd36eb5cffabab76b25755353d16d48ff9bbc17380cb32359718559666385f")
    definition = binding["datasets"][DATASET]
    for key, hashkey in (("data_path", "data_sha256"), ("training_data_path", "training_data_sha256"),
                         ("split_manifest_path", "split_manifest_sha256")):
        checked(COMPONENT / definition[key], definition[hashkey])
    external_contract = read(EXTERNAL_TRAIN / "execution_contract.json", "7b83022702d244688d13e8afbd404ed38975b9f09f533de69995b4bd7fd3d9fc")
    historical_definition = external_contract["datasets"][0]
    for key in ("loader", "model", "optimizer", "statistics", "time_statistics", "epochs", "steps_per_epoch"):
        require(historical_definition[key] == definition["definition"][key], "Historical common condition differs: " + key)
    for path, expected in external_contract["source"]["files"].items():
        checked(EXTERNAL_TRAIN / "source" / path, expected)
    for path, expected in binding["sources"]["components"]["files"].items():
        checked(COMPONENT / "source" / path, expected)

    cnn_rows = []
    cnn_lookup = {}
    jobs = {j["id"]: j for j in binding["jobs"]}
    for oldrow in old["CNN_rows"]:
        if oldrow["dataset"] != DATASET:
            continue
        row = audit.component_row(oldrow)
        job = jobs[row["id"]]
        for pathkey, hashkey in (("terminal_manifest_path", "terminal_manifest_sha256"),
                                 ("worker_exit_path", "worker_exit_sha256"),
                                 ("history_path", "history_sha256"),
                                 ("checkpoint_metadata_path", "checkpoint_metadata_sha256")):
            checked(COMPONENT / job[pathkey], job[hashkey])
        checked(COMPONENT / job["checkpoint"]["relative_path"], job["checkpoint"]["sha256"])
        best = first_best(read(COMPONENT / job["history_path"], job["history_sha256"])["history"])
        require(best["epoch"] == job["selected_epoch"], "CNN selection is not first finite minimum")
        receipt_rel = Path(row["source_path"]).relative_to(COMPONENT).as_posix()
        receipt = read(Path(row["source_path"]), results_manifest["files"][receipt_rel])
        input_rel = f"originals/run/{row['id']}/input_receipt.json"
        input_receipt = read(COMPONENT / input_rel, originals_manifest["files"][input_rel])
        require(input_receipt["populations"] == definition["definition"]["inherited_data_identity"]["populations"],
                "CNN original Train/Validation targets changed")
        require(receipt["selected_epoch"] == job["selected_epoch"] and
                receipt["checkpoint_sha256"] == job["checkpoint"]["sha256"], "CNN checkpoint binding differs")
        expected = definition[row["split"] + "_population"]
        require(receipt["prediction_rows"] == expected["count"], "CNN population count differs")
        for key in ("target_identity_sha256", "truth_sha256", "time_sha256"):
            require(receipt[key] == expected[key] == native["populations"][DATASET][row["split"]][key],
                    "CNN native input fingerprint differs")
        cnn_lookup[(row["split"], row["seed"])] = row
        cnn_rows.append({"dataset": DATASET, "model": "historical_CNN", "split": row["split"], "seed": row["seed"],
            "selected_epoch": row["selected_epoch"], "checkpoint_sha256": row["checkpoint_sha256"],
            "tensor_sha256": row["state_sha256"], "metrics": {k: receipt["metrics"][k] for k in METRICS},
            "population": row["population"], "count": row["count"], "source_path": row["source_path"],
            "source_sha256": row["receipt_sha256"], "training_host": "5080", "evaluation_host": "5080",
            "evaluation_cudnn": receipt["runtime"]["cudnn"], "source_closure": receipt["source_closure"]})
    require(len(cnn_rows) == 6, "Expected historical CNN3 in both splits")
    external_rows = []
    terminal_file_count = 0
    for arm in EXTERNAL:
        for seed in SEEDS:
            pair = {r["split"]: r for r in old["external_rows"]
                    if r["dataset"] == DATASET and r["arm"] == arm and r["seed"] == seed}
            require(set(pair) == set(SPLITS), "External reference pair missing")
            test, test_receipt = audit.external_test(pair["test"], cnn_lookup[("test", seed)])
            validation = audit.external_validation(pair["validation"], test_receipt)
            run = EXTERNAL_TRAIN / "run" / f"{DATASET}__{seed}__{arm}"
            terminal = read(run / "terminal_manifest.json")
            require(terminal["scientific_success"] is True and terminal["status"] == "complete", "External training incomplete")
            require(terminal["job"] == {"arm": arm, "dataset": DATASET, "host": "5080", "id": run.name, "seed": seed},
                    "External job identity differs")
            for path, expected in terminal["files"].items():
                checked(run / path, expected)
                terminal_file_count += 1
            inp = read(run / "input_receipt.json", terminal["files"]["input_receipt.json"])
            require(inp["input_identity"]["data_sha256"] == definition["training_data_sha256"] and
                    inp["input_identity"]["split_manifest_sha256"] == definition["split_manifest_sha256"],
                    "External training data/split differ")
            require(inp["populations"] == definition["definition"]["inherited_data_identity"]["populations"],
                    "External Train/Validation target identity or quantity SHA differs")
            checked(test_receipt["checkpoint_path"], test_receipt["checkpoint_file_sha256"])
            hist_rel = f"runs/{arm}/count_only_log_regression/seed_{seed}/history.json"
            best = first_best(read(run / hist_rel, terminal["files"][hist_rel])["history"])
            require(best["epoch"] == test_receipt["selected_epoch"], "External selection is not first finite minimum")
            require(math.isclose(best["val_qty_rmse"], validation["metrics"]["overall_qty_rmse"], rel_tol=1e-10, abs_tol=1e-8),
                    "Selected Validation endpoint differs from training history")
            require(validation["selected_tensor_sha256"] == test_receipt["state_tensor_sha256"], "External Val/Test tensor mismatch")
            for split, verified in (("validation", validation), ("test", test)):
                pop = definition[split + "_population"]
                require(verified["count"] == pop["count"], "External full population count differs")
                external_rows.append({"dataset": DATASET, "model": arm, "split": split, "seed": seed,
                    "metrics": {k: verified["metrics"]["overall_" + k] for k in METRICS},
                    "selected_epoch": verified["selected_epoch"], "checkpoint_sha256": test_receipt["checkpoint_file_sha256"],
                    "tensor_sha256": test_receipt["state_tensor_sha256"], "count": verified["count"],
                    "source_path": verified["source_path"], "source_sha256": verified["source_raw_sha256"],
                    "training_host": "5080", "evaluation_host": "5080", "evaluation_cudnn": test_receipt["runtime"]["cudnn"],
                    "source_closure": test_receipt["source_closure_sha256"],
                    "population_evidence": ("input_receipt Train/Validation target identity+quantity SHA and same data/loader/endpoint count"
                        if split == "validation" else "prediction part SHA, target order, common truth and recorded time rehashed"),
                    "raw_validation_predictions_available": False if split == "validation" else None,
                    "original_training_population": inp["populations"]["validation"] if split == "validation" else None})
    references.update(audit.SOURCES)
    aggregates = summarize(external_rows + cnn_rows)
    frontier = {}
    for split in SPLITS:
        frontier[split] = {}
        for metric in METRICS:
            best = min(aggregates[m][split][metric]["mean"] for m in EXTERNAL)
            winners = [m for m in EXTERNAL if aggregates[m][split][metric]["mean"] == best]
            frontier[split][metric] = {"models": winners, "mean": best,
                "sample_sd": {m: aggregates[m][split][metric]["sample_sd"] for m in winners}}
    return {"schema": "raf_backbone_loop_frozen_references_v1", "status": "verified", "dataset": DATASET,
        "created_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(), "seeds": list(SEEDS),
        "candidate_results_read": False, "new_training": False, "new_inference": False, "remote_calls": 0,
        "prediction_part_reaggregation": "existing external Test predictions only", "checkpoint_deserialization": False,
        "first_strict_finite_selected_epoch_verified_from_history": True,
        "selection": "first strict finite minimum fullValidation raw quantity RMSE; identical checkpoint for both splits and MAE",
        "statistics": "arithmetic mean of seed metrics; sample SD ddof=1; no pooling across targets, seeds, or datasets",
        "metrics": {"qty_rmse": "sqrt(mean((predicted_raw_quantity - raw_quantity)^2))",
                    "qty_mae": "mean(abs(predicted_raw_quantity - raw_quantity))"},
        "user_performance_gate": "candidate 3seed means strictly lower for both metrics on both splits than historical CNN and per-metric external frontier",
        "same_host_control_gate": "reported separately: candidate means below fresh same-host CNN for both metrics and both splits",
        "capacity_attribution": "reported separately; not silently added to user's performance gate",
        "expected_population": {s: definition[s + "_population"] for s in SPLITS},
        "data": {k: definition[k] for k in ("data_sha256", "training_data_sha256", "split_manifest_sha256")},
        "common_conditions": {k: definition["definition"][k] for k in ("loader", "model", "optimizer", "statistics", "epochs", "steps_per_epoch")},
        "rows": external_rows + cnn_rows, "aggregates": aggregates, "external_frontier": frontier,
        "historical_cnn": aggregates["historical_CNN"], "verified_source_count": len(references),
        "external_terminal_files_verified": terminal_file_count, "external_fit_count": 18,
        "historical_cnn_fit_count": 3, "source_references": dict(sorted(references.items())),
        "limitations": ["Historical common-head numerical-input external adaptations, not native original-paper model heads.",
            "Historical external/CNN evaluation was 5080 cuDNN91900; new RAF loop is planned on 5090 cuDNN92000. Whole source and runtime differ.",
            "External Validation raw prediction rows were not available or regenerated. Population identity is verified through original input receipts, data SHA, loader and full endpoint counts.",
            "Historical Test informed the new design. Repeated Test comparisons are exploratory, not independent generalization evidence or statistical significance."]}


def verify_registry(registry_path, expected_sha, root=ROOT):
    require(sha(registry_path) == expected_sha, "Frozen registry SHA mismatch")
    registry = json.loads(Path(registry_path).read_text())
    require(registry["status"] == "verified" and registry["dataset"] == DATASET, "Unsupported or incomplete registry")
    for relative, expected in registry["source_references"].items():
        p = Path(relative)
        require(not p.is_absolute() and ".." not in p.parts, "Unsafe source reference")
        require(sha(Path(root) / p) == expected, "Frozen source mismatch: " + relative)
    return registry


def _validated_group(rows, registry):
    require(len(rows) == 6, "Exactly 3seed x 2split completed rows required")
    require(len({r.get("model") for r in rows}) == 1 and isinstance(rows[0].get("model"), str),
            "Result group mixes model identities")
    indexed = {}
    for row in rows:
        key = (row["split"], row["seed"])
        require(key not in indexed and row["dataset"] == registry["dataset"] and key[0] in SPLITS and key[1] in SEEDS,
                "Unexpected/duplicate result identity")
        require(row["status"] == "complete" and row["validation_selected_checkpoint_verified"] is True,
                "Unverified evaluation or checkpoint selection")
        expected = registry["expected_population"][row["split"]]
        require(row["count"] == expected["count"], "Population count mismatch")
        for k in ("target_identity_sha256", "truth_sha256", "time_sha256"):
            require(row["population"][k] == expected[k], "Population identity mismatch")
        for metric in METRICS:
            x = row["metrics"][metric]
            require(type(x) in (float, int) and math.isfinite(x) and x >= 0, "Invalid error metric")
        require(type(row["selected_epoch"]) is int and row["selected_epoch"] >= 1 and
                isinstance(row["checkpoint_sha256"], str) and len(row["checkpoint_sha256"]) == 64 and
                all(c in "0123456789abcdef" for c in row["checkpoint_sha256"]),
                "Missing checkpoint identity")
        indexed[key] = row
    require(set(indexed) == {(s, seed) for s in SPLITS for seed in SEEDS}, "Incomplete group")
    for seed in SEEDS:
        v, t = indexed[("validation", seed)], indexed[("test", seed)]
        require((v["checkpoint_sha256"], v["selected_epoch"]) == (t["checkpoint_sha256"], t["selected_epoch"]),
                "Test checkpoint differs from Validation")
    return indexed


def compare_rows(registry, candidate_rows, same_host_cnn_rows, capacity_rows=None):
    """Compare receipt-normalized rows after caller verifies native result evidence.

Each row's runtime must be a stable identity dict, not per-run telemetry: host,
gpu_uuid, python, torch, cudnn and threads. Caller must bind native receipt SHA,
worker exits and Validation replay gate before granting status/selection flags.
"""
    candidate = _validated_group(candidate_rows, registry)
    cnn = _validated_group(same_host_cnn_rows, registry)
    capacity = _validated_group(capacity_rows, registry) if capacity_rows is not None else None
    for indexed in (candidate, cnn) + ((capacity,) if capacity is not None else ()):
        require(all(set(row["runtime"]) == {"host", "gpu_uuid", "python", "torch", "cudnn", "threads"}
                    and all(v is not None for v in row["runtime"].values()) for row in indexed.values()),
                "Stable Runtime identity missing")
    summaries, per_seed, user_checks, same_checks, capacity_checks = {}, [], [], [], []
    history = {(r["split"], r["seed"]): r for r in registry["rows"] if r["model"] == "historical_CNN"}
    for split in SPLITS:
        summaries[split] = {}
        for metric in METRICS:
            cs = stats([candidate[(split, seed)]["metrics"][metric] for seed in SEEDS])
            ns = stats([cnn[(split, seed)]["metrics"][metric] for seed in SEEDS])
            historic = registry["historical_cnn"][split][metric]
            ext = registry["external_frontier"][split][metric]
            user_pass = cs["mean"] < historic["mean"] and cs["mean"] < ext["mean"]
            same_pass = cs["mean"] < ns["mean"]
            item = {"candidate": cs, "historical_CNN": historic, "same_host_CNN": ns, "external_frontier": ext,
                    "user_performance_pass": user_pass, "same_host_control_pass": same_pass}
            user_checks.append(user_pass); same_checks.append(same_pass)
            if capacity is not None:
                cap = stats([capacity[(split, seed)]["metrics"][metric] for seed in SEEDS])
                item["capacity"] = cap
                item["capacity_attribution_pass"] = cs["mean"] < cap["mean"]
                capacity_checks.append(item["capacity_attribution_pass"])
            summaries[split][metric] = item
            for seed in SEEDS:
                key = (split, seed)
                require(candidate[key]["runtime"] == cnn[key]["runtime"], "Fresh CNN/candidate runtime mismatch")
                refs = {"historical_CNN": history[key]["metrics"][metric], "same_host_CNN": cnn[key]["metrics"][metric]}
                if capacity is not None:
                    require(candidate[key]["runtime"] == capacity[key]["runtime"], "Capacity/candidate runtime mismatch")
                    refs["capacity"] = capacity[key]["metrics"][metric]
                value = candidate[key]["metrics"][metric]
                per_seed.append({"split": split, "seed": seed, "metric": metric, "candidate": value,
                    "differences": {name: {"reference": ref, "absolute_candidate_minus_reference": value - ref,
                        "relative_percent_candidate_minus_reference": 100 * (value - ref) / ref if ref else None}
                        for name, ref in refs.items()}})
    return {"status": "complete", "dataset": registry["dataset"], "summaries": summaries,
        "seed_differences": per_seed, "user_performance_gate": all(user_checks),
        "same_host_control_gate": all(same_checks), "capacity_attribution_gate": all(capacity_checks) if capacity_checks else None,
        "test_independent": False, "statistical_significance_established": False,
        "interpretation": "Performance gate and attribution control are distinct; all adverse seeds are retained."}


def registry_markdown(registry):
    lines = ["# RAF backbone 반복 실험의 고정 비교 기준", "",
        "후보 평가 전에 외부 6구조 × 3seed와 기존 CNN3의 저장 증적을 검증했습니다. 새 학습·추론·원격 호출은 없습니다.", "",
        "| 평가 | 지표 | 외부 최저값 모델 | 3seed 평균 | 기존 CNN 평균±표본SD |", "|---|---|---|---:|---:|"]
    for split in SPLITS:
        for metric in METRICS:
            e = registry["external_frontier"][split][metric]; c = registry["historical_cnn"][split][metric]
            lines.append(f"| {split} | {metric} | {', '.join(e['models'])} | {e['mean']:.9f} | {c['mean']:.9f} ± {c['sample_sd']:.9f} |")
    lines += ["", "- 사용자 성능 기준: 위 네 칸에서 모두 외부 최저값과 기존 CNN보다 후보 평균이 엄격히 낮아야 합니다.",
        "- 같은 5090에서 새로 학습한 CNN과의 대조 결과는 별도 판정합니다. 용량 대조군 대비 효과 역시 별도로 기록합니다.",
        "- Validation 6,690개, Test 5,226개 대상입니다. Validation은 원 input_receipt의 대상·수량 SHA와 같은 입력/loader를 확인했고, Test는 외부 저장 예측의 대상 순서·공통 truth·시간 SHA와 RMSE/MAE를 다시 계산했습니다.",
        "- 외부 Validation 개별 예측은 새로 생성하지 않았습니다. 저장 endpoint와 원본 입력 신원으로 비교하며 signed-error 분석을 했다고 주장하지 않습니다.",
        "- 기존 CNN·외부 평가: 5080/cuDNN91900. 새 RAF 실험: 5090/cuDNN92000. source와 전체 Runtime이 동일한 새 외부 재학습은 아닙니다.",
        "- 외부 모델은 공통 수량·시간 head를 사용한 구현입니다. 원 논문 고유 head의 성능과 동일시하지 않습니다.",
        "- 기존 Test를 본 뒤 설계된 반복입니다. 독립적인 일반화 확인이나 통계적 유의성으로 해석하지 않습니다.", "",
        f"검증한 고유 원본/코드/입력 파일: {registry['verified_source_count']}개. 외부 18fit의 terminal manifest 파일 {registry['external_terminal_files_verified']}개를 검증했습니다.",
        "상세 seed별 수치·선택 epoch·checkpoint SHA·원본 경로·검증 SHA는 EXTERNAL_BASELINES.json에 보존합니다.", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("freeze", "compare"), required=True)
    p.add_argument("--registry", type=Path, default=ROOT / REPORT / "EXTERNAL_BASELINES.json")
    p.add_argument("--registry-sha")
    p.add_argument("--results", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if args.mode == "freeze":
        require(not args.registry.exists() and not args.registry.with_suffix(".md").exists(), "Existing registry is immutable")
        registry = build_registry()
        args.registry.parent.mkdir(parents=True, exist_ok=True)
        with args.registry.open("x") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")
        with args.registry.with_suffix(".md").open("x") as f:
            f.write(registry_markdown(registry))
        print(json.dumps({"status": "verified", "registry": str(args.registry), "sha256": sha(args.registry),
            "markdown_sha256": sha(args.registry.with_suffix(".md")), "verified_source_count": registry["verified_source_count"]}))
    else:
        require(args.registry_sha and args.results and args.out, "Registry SHA, results and output required")
        registry = verify_registry(args.registry, args.registry_sha)
        results = json.loads(args.results.read_text())
        result = compare_rows(registry, results["candidate"], results["same_host_cnn"], results.get("capacity"))
        result["registry_sha256"] = args.registry_sha
        result["input_results_sha256"] = sha(args.results)
        with args.out.open("x") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")
        print(json.dumps({"status": result["status"], "output": str(args.out), "sha256": sha(args.out)}))


if __name__ == "__main__":
    main()
