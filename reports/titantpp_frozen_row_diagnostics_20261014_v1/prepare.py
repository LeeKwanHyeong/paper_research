"""Bind approved selected models without deserializing any checkpoint."""
import hashlib
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAME = "titantpp_frozen_row_diagnostics_20261014_v1"
B = ROOT / "search_artifacts" / NAME
ARMS = {"titantpp_cnn_gru54", "titantpp_cnn_gru54_cnn_only"}
EXTERNAL = {"rmtpp", "thp", "nhp", "sahp", "s2p2_matched_head", "attnhp_matched_head"}
DATASETS = {"yellow_trip_hourly", "raf_spare_parts", "intermittent_frozen_5000"}
USER_TEXT = "ㅇㅇ 그럼 새 학습 없이 선택 checkpoint를 고정해서 Train·Validation·Test의 raw 오류와 개별 예측부터 뽑자. CNN+GRU 대 CNN, CNN 대 외부 비교군을 따로 분석해서 반전 원인을 더 좁혀보자."


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1024**2), b""):
            h.update(b)
    return h.hexdigest()


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def prepare():
    if B.exists():
        assert read(B / "approval.json")["user_text"] == USER_TEXT
        assert not (B / "bindings.json").exists()
    else:
        B.mkdir()
    approval = {"user_text": USER_TEXT, "new_training": False,
        "checkpoint_reselection": False, "component_fits": 18, "external_fits": 54,
        "fresh_inference_jobs": 162, "external_Test_reused_conditions": 54,
        "allowed_remote_hosts": ["5080", "5090"], "originals_overwritten": False,
        "manuscript_changes": False, "Runtime_changes": False, "recorded_unix": time.time()}
    if not (B / "approval.json").exists():
        write(B / "approval.json", approval)
    jobs, sources, runtimes, reused = [], {}, {}, []
    prior = ROOT / "reports/titantpp_rank_reversal_diagnostics_20261014_v1/EXTERNAL_ANALYSIS.json"
    ext_analysis = read(prior)
    for cohort in ["taxi_raf24", "inter8", "inter62"]:
        p = ROOT / f"search_artifacts/titantpp_components_{cohort}_test_20261014_v1"
        c = read(p / "contract.json")
        runtimes.update(c["runtimes"])
        sources[str(p / "contract.json")] = sha(p / "contract.json")
        for old in c["jobs"]:
            if old["arm"] not in ARMS:
                continue
            refs, populations = {}, {"train": c["datasets"][old["dataset"]]["definition"]["inherited_data_identity"]["populations"]["train"]}
            for split, filename in [("validation", "validation_replay.json"), ("test", "test_metrics.json")]:
                rel = f"results/{old['id']}/{filename}"
                receipt = read(p / rel)
                refs[split] = {"path": rel, "sha256": sha(p / rel)}
                populations[split] = {"target_count": receipt["expected_target_count"],
                    "target_identity_sha256": receipt["target_identity_sha256"],
                    "truth_sha256": receipt["truth_sha256"], "time_sha256": receipt["time_sha256"],
                    "original_population_identity": receipt["original_population_identity"]}
            jobs.append({"id": old["id"], "family": "component", "dataset": old["dataset"],
                "arm": old["arm"], "seed": old["seed"], "host": old["evaluation_host"],
                "local_parent_root": str(p), "parent_root": c["evaluation_root_mapping"]["authorized_remote_root"],
                "parent_contract_raw_sha256": sha(p / "contract.json"),
                "parent_evaluator_relative_path": "operation/evaluate.py", "parent_evaluator_sha256": sha(p / "operation/evaluate.py"),
                "parent_job_id": old["id"], "expected_populations": populations, "reference_receipts": refs,
                "runtime": c["runtimes"][old["evaluation_host"]], "allowed_splits": ["train", "validation", "test"]})

    registry_rel = "reports/titantpp_final_eval_checkpoint_binding_20261003_v1/evaluation_registry.json"
    registry = read(ROOT / registry_rel)
    manifest = read(ROOT / "reports/titantpp_legacy_evaluation_20261003_v1/dataset_manifest.json")
    external_parent = B / "external_parent"
    copied = {}

    def copy_file(rel, expected=None):
        if rel in copied:
            return
        source, target = ROOT / rel, external_parent / rel
        actual = sha(source)
        if expected is not None:
            assert actual == expected, rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        assert sha(target) == actual
        copied[rel] = actual

    copy_file(registry_rel, "8b715bd360d895bb2ab433072333f9603cb7de2be5de081c0bf0243f0e86fe2b")
    sources[str(prior)] = sha(prior)
    for old in registry["rows"]:
        if old["model"] not in EXTERNAL or old["dataset"] not in DATASETS:
            continue
        bundle = registry["bundles"][old["evaluator_source_bundle"]]
        definition = next(d for d in bundle["datasets"] if d["dataset_id"] == old["dataset"])
        data = next(d for d in manifest["datasets"] if d["dataset"] == old["dataset"])
        copy_file(old["checkpoint_path"], old["checkpoint_file_sha256"])
        for rel, expected in bundle["source_files"].items():
            copy_file(bundle["source_root"] + "/" + rel, expected)
        copy_file(data["path"], data["sha256"])
        refs = {}
        for split in ["validation", "test"]:
            row = next(r for r in ext_analysis["external_rows"] if (r["dataset"], r["arm"], r["seed"], r["split"]) == (old["dataset"], old["model"], old["seed"], split))
            copy_file(row["source_path"], row["source_raw_sha256"])
            refs[split] = {"path": row["source_path"], "sha256": row["source_raw_sha256"]}
            if split == "test":
                reused.append({"id": f"{old['dataset']}__{old['seed']}__{old['model']}",
                    "dataset": old["dataset"], "arm": old["model"], "seed": old["seed"],
                    "receipt": row["source_path"], "receipt_sha256": row["source_raw_sha256"],
                    "prediction_parts": row["prediction_parts"]})
        jobs.append({"id": f"{old['dataset']}__{old['seed']}__{old['model']}", "family": "external",
            "dataset": old["dataset"], "arm": old["model"], "seed": old["seed"], "host": "5080",
            "local_parent_root": str(external_parent), "parent_root": "{REMOTE_ROOT}/external_parent",
            "legacy_registry": {"path": registry_rel, "sha256": sha(ROOT / registry_rel)},
            "legacy_registry_row": old, "source_bundle": bundle, "definition": definition,
            "data": data, "runtime": runtimes["5080"], "expected_populations": data["populations"],
            "reference_receipts": refs, "allowed_splits": ["train", "validation"]})
    assert len(jobs) == len({j["id"] for j in jobs}) == 72
    assert sum(len(j["allowed_splits"]) for j in jobs) == 162 and len(reused) == 54
    write(B / "external_parent_manifest.json", copied)
    write(B / "bindings.json", {"jobs": jobs, "runtimes": runtimes, "sources": sources,
        "external_Test_reuse": reused, "metric_tolerance": {"relative": 1e-5, "absolute": 1e-5},
        "expected_counts": {"train": {"yellow_trip_hourly": 38393, "raf_spare_parts": 25779, "intermittent_frozen_5000": 393824}}})
    print(json.dumps({"status": "bound", "fits": 72, "new_inference_jobs": 162,
        "external_parent_files": len(copied), "packet_bytes": sum((external_parent / rel).stat().st_size for rel in copied)}))


if __name__ == "__main__":
    prepare()
