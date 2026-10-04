"""Seal the explicitly approved width8/12 study without any remote action.

Historic training/data/loss sources are copied from the immutable width16
bundle. Only the width model, its routing/evaluation and the new supervisor
are replaced. Importing this file does not create a bundle or execute a fit.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.scripts import observed_slot_parallel_common as common

NAME = "titantpp_history_width8_12_dual_20261004_v1"
ARMS = ("titantpp_history_mlp_width8", "titantpp_history_mlp_width12")
ASSIGNMENTS = {"5080": ["yellow_trip_hourly", "raf_spare_parts"],
               "5090": ["intermittent_frozen_5000"]}
SEEDS = (42, 52, 62)
CHANGED = ("models/TPPs/CountAwareTitanHistoryWidth.py",
           "paper/scripts/run_titantpp_history_width.py",
           "paper/scripts/evaluate_titantpp_history_width.py",
           "paper/scripts/run_titantpp_history_capacity_campaign.py")
USER_INSTRUCTION = "이거 진행하되, instacart는 후속 실험으로 고정하고 taxi intermittent raf만 실험하자 5080 5090에서 동시에 실험하자"


def read(path):
    return json.loads(Path(path).read_text())


def seal(revision):
    common.require(len(revision) == 40 and revision == subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "Seal the current source commit")
    for rel in CHANGED:
        committed = subprocess.check_output(["git", "show", revision + ":" + rel], cwd=ROOT)
        common.require(committed == (ROOT / rel).read_bytes(), "Uncommitted source: " + rel)
    bundle = ROOT / "search_artifacts" / NAME
    common.require(not bundle.exists(), "No overwrite/retry of an existing bundle")
    dual = ROOT / "search_artifacts/titantpp_history_width16_dual_20261003_v1"
    replication = ROOT / "search_artifacts/titantpp_history_width16_replication_20261003_v1"
    old42, old = read(dual / "execution_contract.json"), read(replication / "execution_contract.json")
    common.require(common.sha_json(old42) == "7b0681473fcde54a16b564821448490a5d6a5561c9d51b60903c9b7f2981e548", "Seed42 parent changed")
    common.require(common.sha_json(old) == "dea8ccb93ccfdd3b9c76319dc7a0f894e67e02fbd2ea546d668843dd71251a7a", "Replication parent changed")
    source = bundle / "frozen_source"
    for rel, digest in old["source"]["files"].items():
        path = replication / "frozen_source" / rel
        common.require(common.sha_file(path) == digest, "Historical source changed: " + rel)
        target = source / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for rel in CHANGED:
        target = source / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    (source / "sample_data").mkdir(exist_ok=True)
    (source / "sample_data/.keep").write_text("Root marker; no data.\n")
    closure = common.source_manifest(source, entrypoints=(CHANGED[-1],), extra_files=())
    files = {**old["source"]["files"], **closure}
    for rel in CHANGED:
        files[rel] = common.sha_file(source / rel)
    design = {"schema": "titantpp_history_capacity_design_v1", "widths": [8, 12],
        "reference_baseline": "titantpp_history_mlp_width16", "preserved_comparator": "titantpp_history_mlp",
        "branches": 8, "branch_input_dim": 128, "branch_output_dim": 64, "activation": "GELU",
        "correction_parameters": {"4": 6144, "8": 12288, "12": 18432, "16": 24576},
        "main_hidden_dimension": 64, "branch_availability": "original observed-history thresholds",
        "residual_divisor": 8, "initialization": "forked CPU RNG, ordinary inputs, zero outputs",
        "heads_loss_input_transform_optimizer_selection": "unchanged frozen parent sources",
        "cnn_gru_included": False, "warm_start": False, "hpo": False,
        "full_model_parameters": "measured separately for each frozen maximum sequence length",
        "GPU_assignment": "Both new widths for each dataset/seed run on the same GPU; Intermittent moves to5090; no cross-GPU efficiency claim"}
    c = deepcopy(old)
    c.pop("queue", None)
    c.pop("operational_files", None)
    datasets = {d for ds in ASSIGNMENTS.values() for d in ds}
    c.update(schema=NAME, approved_scope=USER_INSTRUCTION, arms=list(ARMS),
        roles={a: f"observed_time_history_mlp_width{w}_v1" for a, w in zip(ARMS, (8, 12))},
        quantity_variants={a: "count_only_log_regression" for a in ARMS},
        architecture=design, design_sha256=common.sha_json(design),
        reference_baseline="titantpp_history_mlp_width16", deferred_datasets=["insta_market_basket"],
        jobs=[{"id": f"{d}__{s}__{a}", "host": h, "dataset": d, "seed": s, "arm": a}
              for h, ds in ASSIGNMENTS.items() for d in ds for s in SEEDS for a in ARMS],
        datasets=[d for d in c["datasets"] if d["dataset_id"] in datasets],
        nonlearned_baselines=[], lineage={"seed42_parent": common.sha_json(old42),
            "replication_parent": common.sha_json(old), "changed_sources": list(CHANGED),
            "unchanged_scientific_sources": [p for p in old["source"]["files"] if p not in CHANGED]})
    for h, spec in c["hosts"].items():
        remote = "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/" + NAME + "_" + h
        spec.update(root=remote, source_root=remote + "/source", output_dir=remote + "/run",
                    tmux=NAME + "_" + h, assigned_datasets=ASSIGNMENTS[h])
        spec["environment"]["SOURCE_REVISION"] = revision
    c["source"] = {"base_git_revision": revision, "git_revision": revision, "working_branch": "master",
        "files": files, "files_sha256": common.sha_json(files),
        "revision_note": "Historic model/trainer/loader/head bytes retained except explicit width routing; source closure authoritative; sources archived for replay."}
    c["baseline_replays"], c["reuse"] = {}, []
    registry = read(ROOT / "reports/titantpp_final_eval_checkpoint_binding_20261003_v1/evaluation_registry.json")
    for h, ds in ASSIGNMENTS.items():
        for d in ds:
            c["baseline_replays"][d] = {}
            for seed in SEEDS:
                parent = old42 if seed == 42 else old
                ref = deepcopy(parent["baseline_replays"][d] if seed == 42 else parent["baseline_replays"][d][str(seed)])
                reuse = deepcopy(next(r for r in parent["reuse"] if r["dataset"] == d and r["seed"] == seed and r["arm"] == "titantpp_history_mlp"))
                reuse["host"] = h
                if h == "5090":
                    row = next(r for r in registry["rows"] if r["dataset"] == d and r["model"] == "titantpp_history_mlp" and r["seed"] == seed)
                    cp = ROOT / row["checkpoint_path"]
                    common.require(common.sha_file(cp) == ref["checkpoint_sha256"], "Reference checkpoint changed")
                    exposure = cp.parent / "exposure.json"
                    common.require(common.sha_file(exposure) == reuse["file_sha256"]["exposure.json"], "Reference exposure changed")
                    dest = bundle / "references/width4" / d / str(seed)
                    dest.mkdir(parents=True)
                    shutil.copy2(cp, dest / cp.name)
                    shutil.copy2(exposure, dest / exposure.name)
                    remote_dir = c["hosts"][h]["root"] + "/" + str(dest.relative_to(bundle))
                    reuse["remote_run"] = remote_dir
                    reuse["file_sha256"] = {cp.name: common.sha_file(cp), exposure.name: common.sha_file(exposure)}
                    ref["checkpoint"] = remote_dir + "/" + cp.name
                c["baseline_replays"][d][str(seed)] = ref
                c["reuse"].append(reuse)
    shutil.copytree(dual / "data", bundle / "data")
    for d in c["datasets"]:
        if d["dataset_id"] == "raf_spare_parts":
            identity = d["inherited_data_identity"]
            identity["data"]["path"] = c["hosts"]["5080"]["root"] + "/data/raf_spare_parts/train_validation.parquet"
            identity["split_manifest"]["path"] = c["hosts"]["5080"]["root"] + "/data/raf_spare_parts/split_manifest.json"
    c["dataset_sha256"] = {d["dataset_id"]: common.sha_json(d) for d in c["datasets"]}
    c["diagnostic"]["tail_thresholds"].pop("insta_market_basket", None)
    c["reporting"].update(interpretation="3seed capacity ablation; train stratified estimates separate from full validation; no test selection", baseline_retrained=False)
    c["reference_binding"] = {"baseline_pointer_path": "paper/reproducibility/current_baseline.json",
        "baseline_pointer_sha256": common.sha_file(ROOT / "paper/reproducibility/current_baseline.json"),
        "width16_registry_path": "reports/titantpp_completed_validation_test_20261004_v1/width16/evaluation_registry.json",
        "width16_registry_sha256": common.sha_file(ROOT / "reports/titantpp_completed_validation_test_20261004_v1/width16/evaluation_registry.json")}
    for name, value in (("design.json", design), ("execution_contract.json", c)):
        common.write_json(bundle / name, value, exclusive=True)
    approval = {"approved": True, "contract_sha256": common.sha_json(c), "hosts": ["5080", "5090"],
        "user_instruction": USER_INSTRUCTION, "scope": "18 fresh validation-only width8/12 fits; no Instacart, CNN/GRU, baseline retraining or rental; isolated native directories"}
    common.write_json(bundle / "approval.json", approval, exclusive=True)
    now = time.time()
    common.write_json(bundle / "start_permit.json", {"contract_sha256": common.sha_json(c),
        "approval_sha256": common.sha_json(approval), "started_at_unix": now,
        "started_at_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "deadline_unix": now + c["limits"]["total_wall_seconds"]}, exclusive=True)
    common.write_json(bundle / "current.json", {"bundle": str(bundle.relative_to(ROOT)),
        "contract_sha256": common.sha_json(c), "source_closure_sha256": c["source"]["files_sha256"],
        "hosts": {h: s["root"] for h, s in c["hosts"].items()}}, exclusive=True)
    (bundle / "README.md").write_text("# 폭8·12 용량 대조\n\nTaxi·Intermittent·RAF × width8/12 × seed42/52/62 =18조건. Instacart 후속.\n폭16 운영 기준선과 폭4 대조군은 재학습하지 않습니다. 손실·출력부·선택기준 고정.\n5080 Taxi/RAF12조건,5090 Intermittent6조건; 각 서버1worker, 두 서버 동시.\n전체168h/조건36h 상한, 자동 retry 없음. Train 층화 진단과 full validation, Test 미평가.\n")
    print(json.dumps({"bundle": str(bundle), "jobs": len(c["jobs"]), "contract_sha256": common.sha_json(c), "source_files": len(files)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-revision", required=True)
    seal(parser.parse_args().source_revision)
