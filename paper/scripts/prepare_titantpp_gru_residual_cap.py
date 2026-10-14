"""고정 GRU 보정 상한 후보의 로컬 계약을 준비한다. 학습·원격·Test는 실행하지 않는다."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

from models.TPPs.CountAwareTitanCNNGRUResidualCap import metadata as model_metadata

PROJECT = Path(__file__).resolve().parents[2]
SCHEMA = "titantpp_gru_residual_cap_validation_20261014_v1"
CANDIDATE = "titantpp_cnn_gru54_residual_cap025"
CNN, FULL = "titantpp_cnn_gru54_cnn_only", "titantpp_cnn_gru54"
DATASETS = ("yellow_trip_hourly", "raf_spare_parts", "intermittent_frozen_5000")
SEEDS = (42, 52, 62)
PARENT = PROJECT / "search_artifacts/titantpp_cnn_gru_components_20261012_v5"
PARENT_RAW_SHA = "591a31ed862148c1f7343fef33a65da9d263ec9b80769a240d1ff43cdb6b49f4"
PARENT_CANONICAL = "e60badde6d9d902500aa67764f6834213ea1775797508f31d204ae67a5e63074"
SOURCE131 = "15f6d2b482c5f76c0488d017f5c923c76bc54fd75543aab838c477266a608b53"
OPERATION = "12c836dc5fe1d9146b83a352ba77abc297f4d69c6f68d83ad00935a2bc442903"
LOCAL40 = "16e317c3c16dddede95c6aa6174026dca0ab772879bf438c3d1e6b752eff7074"
BASELINE_CONTRACTS = (
    ("titantpp_components_taxi_raf24_test_20261014_v1", "918d4d5ae5c2440171bb4929d6049c60fe31a500b4a115b5a37dd42059498d39"),
    ("titantpp_components_inter8_test_20261014_v1", "b81978694cd19a40663cd0340d8e2f892fa11a4e377ce15b6ffaaaf1df15a8f6"),
    ("titantpp_components_inter62_test_20261014_v1", "39a54933b345f64167d0f57defd0852fdc82873619a64b270d4e3d5bbd40b803"),
)
BASELINE_MANIFESTS = {
    "titantpp_components_taxi_raf24_test_20261014_v1": "bf3c3ae028bbf9a6c6f1f2de1ca3881a5b8d2b13feab84d3601616187acf0a68",
    "titantpp_components_inter8_test_20261014_v1": "062b00d3e31139503aa485413a2089764f3ea231c360238e5a3a642f688a1a8b",
    "titantpp_components_inter62_test_20261014_v1": "856fa4e6646bf9ccb8350970e4e6f55321949027544793799b3e950e98ef550b",
}
EXTENSION_PATHS = (
    "models/TPPs/CountAwareTitanCNNGRUResidualCap.py",
    "paper/scripts/run_titantpp_gru_residual_cap.py",
    "paper/scripts/prepare_titantpp_gru_residual_cap.py",
)
CAP = {
    "arm": CANDIDATE, "rho": 0.25, "sweep": False,
    "contract_id": "titantpp_cnn_gru_residual_cap025_v1",
    "identity_buffer": "cnn_gru_residual_cap_identity",
    "identity_buffer_dtype": "uint8", "identity_buffer_bytes": 32,
    "model_role": "observed_time_cnn_gru54_residual_cap025_v1",
    "reference": "block1_base_hidden.detach()", "norm": "per_batch_position_last_dimension_L2",
    "norm_dtype": "float64", "output_dtype": "original_residual_dtype",
    "zero_base": "zero_correction", "zero_residual": "zero_correction_with_valid_finite_gradient",
    "epsilon_floor": False, "new_trainable_parameters": 0,
    "inactive_cap": "unsaturated_residual_bitwise_identity",
    "preserved": ["FULL parameter objects", "constructor RNG", "CNN correction", "prototype", "quantity head", "time head"],
}


def candidate_definition():
    """추가 버퍼 신원과 상한 동작은 실제 모델 metadata로 결속한다."""
    metadata = model_metadata(64, arm=CANDIDATE, rho=0.25)
    identity_hex = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    return {**deepcopy(CAP), "metadata": metadata, "identity_buffer_hex": identity_hex}
ANALYSIS = {
    "scope": "validation_only", "primary": "whole_qty_rmse",
    "selection": "first_strict_finite_minimum_full_validation_raw_quantity_rmse",
    "tie": "earliest_epoch", "reporting": "three_seed_mean_sample_sd_and_each_paired_seed_difference",
    "sample_sd_ddof": 1,
    "metrics": ["whole_qty_rmse", "whole_qty_mae", "tail_qty_rmse", "tail_qty_mae", "whole_time_nll", "tail_time_nll"],
    "time_metric": "recorded_positive_integer_TimeNLL",
    "tail": "raw target quantity strictly above original all-Train upper boundary",
    "paired_comparisons": ["candidate_minus_original_FULL", "candidate_minus_original_CNN"],
    "same_selected_checkpoint_for_all_metrics": True,
    "adverse_metrics_and_seed_exceptions_retained": True,
    "formal_significance": False, "Test_evaluation": False,
    "post_Test_exploratory_followup": True, "independent_confirmatory_claim": False,
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def confined(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "경로가 승인된 원본 루트를 벗어납니다")
    return path


class Evidence:
    def __init__(self):
        self.files = {}

    def verify(self, path, expected):
        path = Path(path).resolve()
        require(sha(path) == expected, "원본 SHA가 다릅니다: " + str(path))
        self.files[str(path)] = expected
        return path

    def read(self, path, expected):
        return json.loads(self.verify(path, expected).read_text())

    def recheck(self):
        for path, expected in self.files.items():
            require(sha(path) == expected, "준비 중 원본이 변경됐습니다: " + path)


def planned_host(dataset, seed):
    return "5090" if dataset == "intermittent_frozen_5000" and seed in (42, 52) else "5080"


def read_parent(evidence):
    parent = evidence.read(PARENT / "execution_contract.json", PARENT_RAW_SHA)
    require(canonical(parent) == PARENT_CANONICAL, "원래 v5 canonical 계약이 다릅니다")
    pointer_path = PARENT / "current.json"
    pointer = evidence.read(pointer_path, sha(pointer_path))
    require(pointer.get("contract_sha256") == PARENT_CANONICAL
            and Path(pointer.get("bundle", "")).resolve() == PARENT.resolve(), "원래 current 포인터가 다릅니다")
    require(canonical(parent["source"]["files"]) == SOURCE131 and len(parent["source"]["files"]) == 131,
            "원래 과학 source131 봉인이 다릅니다")
    require(canonical(parent["operation"]["files"]) == OPERATION, "원래 운영 봉인이 다릅니다")
    local = evidence.read(PARENT / "local_tools_integrity.json", LOCAL40)
    require(len(local["files"]) == 40, "원래 로컬 도구 봉인이 다릅니다")
    for relative, expected in parent["source"]["files"].items():
        evidence.verify(confined(PARENT / "source", relative), expected)
    for relative, expected in parent["operation"]["files"].items():
        evidence.verify(confined(PARENT, relative), expected)
    for relative, expected in local["files"].items():
        evidence.verify(confined(PROJECT, relative), expected)
    return parent


def earliest_selection(history):
    rows = history["history"]
    require(rows and [r["epoch"] for r in rows] == sorted({r["epoch"] for r in rows}),
            "Validation history epoch 순서 또는 고유성이 다릅니다")
    best = None
    for row in rows:
        value = row["val_qty_rmse"]
        if math.isfinite(value) and (best is None or value < best[1]):
            best = (row["epoch"], value)
    require(best is not None, "선택 가능한 finite Validation RMSE가 없습니다")
    return best


def collect_baselines(parent, evidence):
    references = []
    definitions = {d["dataset_id"]: d for d in parent["datasets"]}
    for name, expected in BASELINE_CONTRACTS:
        root = PROJECT / "search_artifacts" / name
        contract = evidence.read(root / "contract.json", expected)
        # 결과 수치를 포함하지 않는 파일/SHA 목록이며 Validation 항목만 사용한다.
        manifest_path = root / "evaluation_results_manifest.json"
        manifest_sha = BASELINE_MANIFESTS[name]
        manifest = evidence.read(manifest_path, manifest_sha)
        require(manifest["contract_sha256"] == contract["contract_sha256"], "기존 평가 manifest의 계약 결속이 다릅니다")
        for job in contract["jobs"]:
            if job["arm"] not in (CNN, FULL):
                continue
            receipt_path = root / "results" / job["id"] / "validation_replay.json"
            # 이 경로는 Validation-only receipt이며 Test 파일은 읽지 않는다.
            receipt_sha = manifest["files"][str(receipt_path.relative_to(root))]
            receipt = evidence.read(receipt_path, receipt_sha)
            require(receipt.get("status") == "complete" and receipt.get("split") == "validation"
                    and receipt.get("held_out_test_evaluated") is False and receipt.get("full_population") is True
                    and receipt.get("retrained") is False and receipt.get("reselected") is False,
                    "기존 Validation receipt의 범위가 다릅니다")
            require(receipt.get("selected_epoch") == job["selected_epoch"]
                    and receipt.get("checkpoint_sha256") == job["checkpoint"]["sha256"]
                    and receipt.get("state_sha256") == job["checkpoint_state_sha256"]
                    and receipt.get("source_closure") == SOURCE131, "기존 선택 모델 결속이 다릅니다")
            history_path = confined(root, job["selection_history_path"])
            history = evidence.read(history_path, job["selection_history_sha256"])
            selected = earliest_selection(history)
            require(selected == (job["selected_epoch"], job["selected_metric_value"]), "기존 최초 strict finite 최소 선택이 다릅니다")
            checkpoint = confined(root, job["checkpoint"]["relative_path"])
            evidence.verify(checkpoint, job["checkpoint"]["sha256"])
            definition = definitions[job["dataset"]]
            require(receipt["original_population_identity"] == definition["inherited_data_identity"]["populations"]["validation"],
                    "기존 Validation 모집단이 원래 Train/Validation 계약과 다릅니다")
            require(receipt["metrics"]["quantity_boundaries"] == definition["quantity_boundaries_all_train_rows"],
                    "기존 Train 수량 구간이 다릅니다")
            require(job["training_host"] == planned_host(job["dataset"], job["seed"]), "같은 seed의 원래 학습 호스트가 다릅니다")
            metrics = receipt["metrics"]
            metric_values = {"whole_qty_rmse": metrics["qty_rmse"], "whole_qty_mae": metrics["qty_mae"],
                             "whole_time_nll": metrics["time_nll"], "tail_qty_rmse": metrics["tail"]["qty_rmse"],
                             "tail_qty_mae": metrics["tail"]["qty_mae"], "tail_time_nll": metrics["tail"]["time_nll"]}
            references.append({"id": job["id"], "dataset": job["dataset"], "seed": job["seed"], "arm": job["arm"],
                               "campaign": job["cohort"], "root": str(root), "parent_contract_path": str(root / "contract.json"),
                               "parent_contract_raw_sha256": expected, "selected_epoch": job["selected_epoch"],
                               "results_manifest_path": str(manifest_path), "results_manifest_sha256": manifest_sha,
                               "checkpoint_path": str(checkpoint), "checkpoint_sha256": job["checkpoint"]["sha256"],
                               "state_tensor_sha256": job["checkpoint_state_sha256"], "source_closure": SOURCE131,
                               "history_path": str(history_path), "history_sha256": job["selection_history_sha256"],
                               "validation_receipt_path": str(receipt_path), "validation_receipt_sha256": receipt_sha,
                               "validation_population": receipt["original_population_identity"],
                               "train_input_sha256": definition["data_sha256"], "validation_input_sha256": receipt["data_sha256"],
                               "metrics": metric_values, "training_host": job["training_host"],
                               "runtime": {"python": receipt["runtime"]["python"], "torch": receipt["runtime"]["torch"],
                                           "cuda": receipt["runtime"]["cuda"], "cudnn": receipt["runtime"]["cudnn"],
                                           "gpu_uuid": receipt["runtime"]["gpu"]["uuid"]},
                               "physical_parameter_counts": receipt["physical_parameter_counts"],
                               "retraining": False, "replacement_of_original_FULL": False})
    require(len(references) == 18 and len({r["id"] for r in references}) == 18, "기존 CNN/FULL Validation18개가 아닙니다")
    return sorted(references, key=lambda r: r["id"])


def make_contract(parent, references, extension_files, bundle):
    inherited_files = parent["source"]["files"]
    require(set(extension_files) == set(EXTENSION_PATHS), "신규 source 모듈 목록이 다릅니다")
    files = {**inherited_files, **extension_files}
    full_refs = {(r["dataset"], r["seed"]): r for r in references if r["arm"] == FULL}
    jobs = [{"id": f"{d}__{s}__{CANDIDATE}", "dataset": d, "seed": s, "arm": CANDIDATE,
             "host": planned_host(d, s), "state": "planned_training_not_started",
             "expected_registered_parameters": full_refs[(d, s)]["physical_parameter_counts"]["actual_registered_parameters"],
             "expected_active_parameters": full_refs[(d, s)]["physical_parameter_counts"]["actual_active_parameters_before_evaluation_freeze"]}
            for d in DATASETS for s in SEEDS]
    hosts = {}
    for host in ("5080", "5090"):
        runtime = deepcopy(parent["hosts"][host])
        native_root = "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/" + Path(bundle).name + "_" + host
        hosts[host] = {"planned_root": native_root, "planned_source_root": native_root + "/source",
                       "python": runtime["python"], "gpu_uuid": runtime["gpu_uuid"],
                       "runtime_expected": runtime["runtime_expected"],
                       "live_runtime_queried": False, "native_qualification": "required_before_any_training",
                       "training_authorized": False, "training_permit": None}
    return {"schema": SCHEMA, "stage": "local_preparation_only", "status": "prepared_training_not_started",
            "candidate": candidate_definition(), "arms": [CANDIDATE], "seeds": list(SEEDS),
            "datasets": deepcopy(parent["datasets"]), "input_files": deepcopy(parent["input_files"]),
            "inherited_input_root": str(PARENT / "source"), "input_rows_read_during_preparation": False,
            "input_bytes_rechecked_during_preparation": False,
            "training": deepcopy(parent["training"]), "objective_weights": {"log_quantity": 1.0, "time": 1.0, "tail": 0.0},
            "limits": deepcopy(parent["limits"]), "jobs": jobs, "planned_fit_count": 9, "started_fit_count": 0,
            "hosts": hosts, "evaluation_scope": "validation_only", "held_out_test_evaluated": False, "test_evaluation": False,
            "analysis": deepcopy(ANALYSIS), "baseline_references": deepcopy(references),
            "baseline_references_sha256": canonical(references),
            "source": {"parent_science_closure": SOURCE131, "parent_files": deepcopy(inherited_files),
                       "extension_files": dict(extension_files), "files": files, "files_sha256": canonical(files),
                       "frozen_source_directory": "source", "parent_revision": parent["source"]["git_revision"],
                       "base_git_revision": parent["source"]["base_git_revision"]},
            "parent": {"bundle": str(PARENT), "contract_raw_sha256": PARENT_RAW_SHA,
                       "contract_canonical_sha256": PARENT_CANONICAL, "science_closure": SOURCE131,
                       "operation_closure": OPERATION, "local40_raw_sha256": LOCAL40},
            "runner_module": "paper.scripts.run_titantpp_gru_residual_cap",
            "authorization": {"scope": "contract_implementation_and_local_synthetic_checks_only",
                              "execution_authorized": False,
                              "execution_approval": None, "training_permits": [], "remote_calls": 0,
                              "new_5090_training_authorized": False},
            "policy": {"automatic_retry": False, "resume": False, "Test": False, "rho_sweep": False,
                       "dataset_expansion": False, "seed_expansion": False, "external_model_addition": False,
                       "overwrite_originals": False, "shared_runtime_changes": False, "manuscript_changes": False},
            "created_utc": datetime.now(timezone.utc).isoformat()}


def validate_contract(contract, parent, *, expected_references=None):
    """형태뿐 아니라 원래 계약과의 불변 조건을 검사한다."""
    require(contract.get("schema") == SCHEMA and contract.get("status") == "prepared_training_not_started"
            and contract.get("stage") == "local_preparation_only", "준비 상태 또는 schema가 다릅니다")
    require(contract.get("candidate") == candidate_definition() and contract.get("arms") == [CANDIDATE], "단일 cap025 후보 계약 또는 모델 metadata가 다릅니다")
    require(contract.get("seeds") == list(SEEDS) and contract.get("datasets") == parent["datasets"], "dataset/seed 또는 head/통계/loader가 다릅니다")
    require(contract.get("input_files") == parent["input_files"] and contract.get("inherited_input_root") == str(PARENT / "source"), "입력 결속이 다릅니다")
    require(contract.get("training") == parent["training"] and contract.get("limits") == parent["limits"], "학습 예산 또는 종료 조건이 다릅니다")
    require(contract.get("objective_weights") == {"log_quantity": 1.0, "time": 1.0, "tail": 0.0}, "손실 가중치가 다릅니다")
    require(contract.get("evaluation_scope") == "validation_only" and contract.get("held_out_test_evaluated") is False
            and contract.get("test_evaluation") is False and contract.get("input_rows_read_during_preparation") is False
            and contract.get("input_bytes_rechecked_during_preparation") is False
            and contract.get("analysis") == ANALYSIS, "평가 split 또는 선택/분석 기준이 다릅니다")
    require(contract.get("planned_fit_count") == 9 and contract.get("started_fit_count") == 0, "예정 또는 실제 fit 수가 다릅니다")
    require(contract.get("authorization") == {"scope": "contract_implementation_and_local_synthetic_checks_only",
            "execution_authorized": False,
            "execution_approval": None, "training_permits": [], "remote_calls": 0, "new_5090_training_authorized": False}, "실행 승인을 준비 승인으로 확대했습니다")
    require(contract.get("policy") == {"automatic_retry": False, "resume": False, "Test": False, "rho_sweep": False,
            "dataset_expansion": False, "seed_expansion": False, "external_model_addition": False,
            "overwrite_originals": False, "shared_runtime_changes": False, "manuscript_changes": False}, "허용 범위가 다릅니다")
    require(contract["parent"] == {"bundle": str(PARENT), "contract_raw_sha256": PARENT_RAW_SHA,
            "contract_canonical_sha256": PARENT_CANONICAL, "science_closure": SOURCE131,
            "operation_closure": OPERATION, "local40_raw_sha256": LOCAL40}, "원본 봉인 결속이 다릅니다")
    source = contract["source"]
    require(source["parent_science_closure"] == SOURCE131 and source["parent_files"] == parent["source"]["files"], "원래 source131을 변경했습니다")
    require(set(source["extension_files"]) == set(EXTENSION_PATHS)
            and source["files"] == {**parent["source"]["files"], **source["extension_files"]}
            and source["files_sha256"] == canonical(source["files"])
            and all(re.fullmatch(r"[0-9a-f]{64}", value) for value in source["files"].values())
            and source["frozen_source_directory"] == "source"
            and source["parent_revision"] == parent["source"]["git_revision"]
            and source["base_git_revision"] == parent["source"]["base_git_revision"], "신규 source SHA 또는 모듈 목록이 다릅니다")
    expected_ids = {f"{d}__{s}__{CANDIDATE}": (d, s) for d in DATASETS for s in SEEDS}
    jobs = contract["jobs"]
    require(len(jobs) == 9 and {j["id"] for j in jobs} == set(expected_ids), "9개 고유 새 fit 구성이 다릅니다")
    refs = contract["baseline_references"]
    require(contract.get("baseline_references_sha256") == canonical(refs), "기준선 참조 내용의 SHA가 다릅니다")
    if expected_references is not None:
        require(refs == expected_references, "검증한 원래 기준선 참조가 변경됐습니다")
    require(len(refs) == 18 and {(r["dataset"], r["seed"], r["arm"]) for r in refs}
            == {(d, s, arm) for d in DATASETS for s in SEEDS for arm in (CNN, FULL)}, "기존 기준선18개 구성이 다릅니다")
    full = {(r["dataset"], r["seed"]): r for r in refs if r["arm"] == FULL}
    for job in jobs:
        d, s = expected_ids[job["id"]]
        require(job["dataset"] == d and job["seed"] == s and job["arm"] == CANDIDATE
                and job["host"] == planned_host(d, s) and job["state"] == "planned_training_not_started", "fit 호스트/상태/후보가 다릅니다")
        counts = full[(d, s)]["physical_parameter_counts"]
        require(job["expected_registered_parameters"] == counts["actual_registered_parameters"]
                and job["expected_active_parameters"] == counts["actual_active_parameters_before_evaluation_freeze"], "FULL과 파라미터 수가 다릅니다")
    for ref in refs:
        require(ref["source_closure"] == SOURCE131 and ref["training_host"] == planned_host(ref["dataset"], ref["seed"])
                and ref["retraining"] is False and ref["replacement_of_original_FULL"] is False, "기준선 source/host/보존 조건이 다릅니다")
        require(set(ref["metrics"]) == set(ANALYSIS["metrics"]), "기준선 여섯 Validation 지표가 다릅니다")
        require(all(math.isfinite(value) for value in ref["metrics"].values()), "기준선 지표가 finite가 아닙니다")
    require(set(contract["hosts"]) == {"5080", "5090"}, "호스트 목록이 다릅니다")
    for host, plan in contract["hosts"].items():
        old = parent["hosts"][host]
        require(plan["runtime_expected"] == old["runtime_expected"] and plan["python"] == old["python"]
                and plan["gpu_uuid"] == old["gpu_uuid"] and plan["live_runtime_queried"] is False
                and plan["training_authorized"] is False and plan["training_permit"] is None
                and plan["native_qualification"] == "required_before_any_training", "native Runtime 계획 또는 승인 상태가 다릅니다")
    return contract


def validate_output_path(output):
    output = Path(output)
    require(not output.is_symlink(), "출력 경로가 symbolic link입니다")
    output = output.resolve()
    require(output.parent == (PROJECT / "search_artifacts").resolve()
            and re.fullmatch(r"titantpp_gru_residual_cap_validation_[A-Za-z0-9_]+", output.name), "출력은 별도 cap 준비 bundle이어야 합니다")
    require(not output.exists(), "기존 bundle을 덮어쓰거나 다시 준비하지 않습니다")
    return output


def prepare(output):
    output = validate_output_path(output)
    evidence = Evidence()
    parent = read_parent(evidence)
    references = collect_baselines(parent, evidence)
    extensions = {p: sha(confined(PROJECT, p)) for p in EXTENSION_PATHS}
    for relative, expected in extensions.items():
        evidence.verify(confined(PROJECT, relative), expected)
    contract = validate_contract(make_contract(parent, references, extensions, output), parent, expected_references=references)
    evidence.recheck()
    output.mkdir(exist_ok=False)
    for relative, expected in contract["source"]["files"].items():
        original = confined(PROJECT if relative in extensions else PARENT / "source", relative)
        target = confined(output / "source", relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        with original.open("rb") as source, target.open("xb") as destination:
            shutil.copyfileobj(source, destination)
        require(sha(target) == expected, "복제된 source bytes가 다릅니다")
    evidence.recheck()
    contract_path = output / "execution_contract.json"
    with contract_path.open("x", encoding="utf-8") as stream:
        json.dump(contract, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    receipt = {"schema": SCHEMA, "status": "prepared_training_not_started", "contract_raw_sha256": sha(contract_path),
               "contract_canonical_sha256": canonical(contract), "planned_fit_count": 9, "started_fit_count": 0,
               "native_qualification": "not_run", "Test": False, "remote_calls": 0,
               "training_permits_created": 0, "execution_approvals_created": 0,
               "verified_reference_files": evidence.files, "input_rows_read": False,
               "input_bytes_rechecked": False, "source_file_count": len(contract["source"]["files"])}
    with (output / "preparation_receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = prepare(args.output)
    print(json.dumps({"status": receipt["status"], "bundle": str(args.output.resolve()),
                      "contract_raw_sha256": receipt["contract_raw_sha256"],
                      "contract_canonical_sha256": receipt["contract_canonical_sha256"],
                      "planned_fit_count": 9, "started_fit_count": 0, "remote_calls": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
