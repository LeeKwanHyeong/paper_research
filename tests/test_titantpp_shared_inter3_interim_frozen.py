"""CPU fabricated-only interim snapshot checks; no research Test data is read."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
import torch

from paper.scripts import evaluate_titantpp_shared_inter3_interim_frozen as evaluator
from paper.scripts import run_titantpp_shared_history_raw9_5090 as frozen


def jobs():
    return [{"id": f"{evaluator.DATASET}__{evaluator.ARM}__seed{seed}__losslog_mse",
             "dataset": evaluator.DATASET, "arm": evaluator.ARM, "seed": seed,
             "selected_epoch": best, "snapshot_epoch": saved, "state_sha256": "b" * 64}
            for seed, best, saved in ((42, 5, 41), (52, 5, 39), (62, 11, 37))]


def contract_fixture():
    return {"schema": evaluator.SCHEMA, "host": "5080", "evaluation_authorized": True, "jobs": jobs()}


def bind(root, path):
    return {"path": str(path.relative_to(root)), "sha256": evaluator.sha(path)}


@pytest.mark.parametrize("seed", evaluator.SEEDS)
def test_snapshot_epoch_comes_from_frozen_contract_not_past_status(seed):
    contract = contract_fixture()
    job = next(job for job in contract["jobs"] if job["seed"] == seed)
    job["snapshot_epoch"] += 7
    job["selected_epoch"] += 2
    assert evaluator.selected_job(contract, job["id"]) == job


@pytest.mark.parametrize("key,value", [("dataset", "raf_spare_parts"), ("arm", "titantpp_history_mlp_width16"),
    ("seed", 32), ("selected_epoch", 0), ("selected_epoch", 42), ("snapshot_epoch", True)])
def test_foreign_model_or_impossible_interim_epoch_rejected(key, value):
    contract = contract_fixture()
    contract["jobs"][0][key] = value
    with pytest.raises(ValueError):
        evaluator.selected_job(contract, contract["jobs"][0]["id"])


@pytest.fixture(scope="module")
def fabricated(tmp_path_factory):
    from paper.scripts.count_aware_tpp_backbone import training, core
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    configure_runtime("cpu", 4)
    root = tmp_path_factory.mktemp("interim_inter3_fabricated")
    args, frame, definition, interface, quantity = frozen.prepared._engine_fixture(root, evaluator.ARM)
    args.dataset_contract = evaluator.DATASET
    definition["model"]["time_observation_contract"]["unit"] = "week"
    for key, value in frozen.input_identity().items():
        setattr(args, key, value)
    interface.update(frozen.input_identity(), source_files_sha256="a" * 64)
    definition["quantity_boundaries_all_train_rows"] = [2, 30, 60, 187.0]
    definition["loader"].update(batch_size=128, lookback_weeks=84)
    definition["inherited_data_identity"] = {
        "data": {"path": str(args.data), "sha256": args.data_sha256},
        "split_manifest": {"path": str(args.split_manifest), "sha256": args.split_manifest_sha256},
        "populations": {}}
    for split in ("train", "validation"):
        loader = make_loader(frame, target_split=split, batch_size=128, lookback_weeks=84,
            max_seq_len=84, shuffle=False, generator=None)
        definition["inherited_data_identity"]["populations"][split] = frozen.population_identity(loader.dataset, split)[1]
    with frozen._engine_hooks(evaluator.ARM):
        summary, _, _ = training.train_one(args=args, frame=frame, quantity_contract=quantity,
            interface_meta=interface, backbone=evaluator.ARM, quantity_variant=frozen.VARIANT, seed=42)
    selected = Path(summary["checkpoint_path"])
    checkpoint = selected.parent / "last_epoch_state.pt"
    payload = torch_load_checkpoint(checkpoint, map_location="cpu")
    raw = pl.read_parquet(args.data)
    test_rows = []
    for entity in raw["oper_part_no"].unique().sort().to_list():
        for seq, q in ((11, 187.0), (12, 188.0), (13, 10000.0)):
            test_rows.append({"oper_part_no": entity, "seq": seq, "delta_t": 2,
                "demand_qty": q, "chronological_split": "test"})
    full = pl.concat([raw, pl.DataFrame(test_rows)]).sort(["oper_part_no", "seq"])
    populations = {}
    for split in ("validation", "test"):
        loader = make_loader(core.prepare_count_frame(full), target_split=split, batch_size=128,
            lookback_weeks=84, max_seq_len=84, shuffle=False, generator=None)
        populations[split] = frozen.population_identity(loader.dataset, split)[1]
    job = {"id": f"{evaluator.DATASET}__{evaluator.ARM}__seed42__losslog_mse",
        "dataset": evaluator.DATASET, "arm": evaluator.ARM, "seed": 42,
        "selected_epoch": payload["best_epoch"], "snapshot_epoch": payload["epoch"],
        "state_sha256": payload["best_state_sha256"]}
    metadata = root / "checkpoint_metadata.json"
    metadata.write_text(json.dumps(evaluator.checkpoint_snapshot_metadata(payload), allow_nan=False))
    row = payload["history"][payload["best_epoch"] - 1]
    reference = {"split": "validation", "selected_epoch": payload["best_epoch"],
        "source": "embedded_checkpoint_history", "metrics": {"overall":
            {metric: row[field] for metric, field in evaluator.HISTORY_METRICS.items()}}}
    snapshot = {"schema": evaluator.SNAPSHOT_SCHEMA, "status": "frozen_interim_selected",
        "job_id": job["id"], "dataset": evaluator.DATASET, "arm": evaluator.ARM, "seed": 42,
        "selected_epoch": job["selected_epoch"], "snapshot_epoch": job["snapshot_epoch"],
        "checkpoint_sha256": evaluator.sha(checkpoint), "state_sha256": payload["best_state_sha256"],
        "source_files_sha256": "a" * 64, "target_identity": populations["validation"],
        "history_reference": reference, "original_training_terminal_required": False,
        "held_out_test_evaluated": False, "historical_tail_metrics_available": False,
        "historical_individual_predictions_available": False, "predictions_available": False}
    snapshot_path = root / "snapshot_receipt.json"
    snapshot_path.write_text(json.dumps(snapshot))
    for key, path in (("checkpoint", checkpoint), ("checkpoint_metadata", metadata),
        ("selection_history", checkpoint.parent / "history.json"), ("snapshot_receipt", snapshot_path)):
        job[key] = bind(root, path)
    training_contract = {"source": {"files_sha256": "a" * 64}}
    contract = {"data": {evaluator.DATASET: {"populations": populations}}, "jobs": [job]}
    return root, raw, full, definition, job, payload, snapshot, training_contract, contract, populations


def test_raw_resume_and_best_tensor_identity_are_bound_without_completed_receipt(fabricated):
    root, _, _, definition, job, payload, _, training, contract, _ = fabricated
    with patch.object(frozen, "select_job", return_value=(job, definition, None, None)):
        data, relocated, reference = evaluator.verify_job_artifacts(root, contract, training, job)
    assert data == definition and reference["source"] == "embedded_checkpoint_history"
    assert relocated["state_sha256"] == payload["best_state_sha256"]
    assert relocated["time_head_snapshot_verification"]["original_current_state_proof_verified"]
    assert not relocated["time_head_snapshot_verification"]["selected_epoch_individual_gradient_proof_available"]
    assert not any(key in job for key in ("train_receipt", "train_exit", "validation_predictions"))


@pytest.mark.parametrize("mutation", ("best_tensor", "current_tensor", "proof", "best_epoch", "history", "objective", "source"))
def test_raw_resume_best_state_history_and_current_proof_tampering_rejected(fabricated, mutation):
    _, _, _, definition, job, original, _, _, _, _ = fabricated
    payload = deepcopy(original)
    if mutation in ("best_tensor", "current_tensor"):
        name = "best_state_dict" if mutation == "best_tensor" else "model_state_dict"
        payload[name]["v_t.weight"].add_(1)
    elif mutation == "proof":
        payload["time_head_training_verification"]["checkpoint_time_head_sha256"] = "0" * 64
    elif mutation == "best_epoch":
        payload["best_epoch"] += 1
    elif mutation == "history":
        payload["history"][payload["best_epoch"] - 1]["val_qty_rmse"] += 1
    elif mutation == "objective":
        payload["time_alpha"] = 0
    else:
        payload["interface_meta"]["source_files_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        evaluator.verify_snapshot_payload(payload, job, definition, "a" * 64)


def test_snapshot_metadata_preserves_identity_but_excludes_optimizer_and_tensors(fabricated):
    _, _, _, _, job, payload, _, _, _, _ = fabricated
    metadata = evaluator.checkpoint_snapshot_metadata(payload)
    assert metadata["metadata"]["best_state_sha256"] == payload["best_state_sha256"]
    assert metadata["history"] == evaluator._json_safe(payload["history"])
    assert not any(key in metadata["metadata"] for key in ("model_state_dict", "best_state_dict", "optimizer_state_dict", "rng_state"))
    evaluator.validate_checkpoint_metadata(payload, metadata, job)
    metadata["metadata"]["encoder_config"]["time_head"]["time_scale"] += 1
    with pytest.raises(ValueError):
        evaluator.validate_checkpoint_metadata(payload, metadata, job)


def test_capture_history_may_advance_but_selection_stays_with_raw_snapshot(fabricated):
    root, _, _, definition, original_job, payload, _, training, original_contract, _ = fabricated
    job, contract = deepcopy(original_job), deepcopy(original_contract)
    path = root / "advanced_history.json"
    history = deepcopy(payload["history"])
    history.append({**deepcopy(history[-1]), "epoch": payload["epoch"] + 1, "val_qty_rmse": 0.0})
    path.write_text(json.dumps({"history": history}))
    job["selection_history"] = bind(root, path)
    with patch.object(frozen, "select_job", return_value=(job, definition, None, None)):
        _, relocated, _ = evaluator.verify_job_artifacts(root, contract, training, job)
        assert relocated["selected_epoch"] == payload["best_epoch"]
        history[0]["val_qty_mae"] += 1
        path.write_text(json.dumps({"history": history}))
        job["selection_history"] = bind(root, path)
        with pytest.raises(ValueError, match="prefix"):
            evaluator.verify_job_artifacts(root, contract, training, job)


def test_actual_best_state_inference_produces_six_metrics_and_strict_q187_tail(fabricated):
    root, raw, full, definition, job, payload, snapshot, _, _, populations = fabricated
    checkpoint = root / job["checkpoint"]["path"]
    before = evaluator.sha(checkpoint)
    relocated = {**snapshot, "selected_checkpoint": str(checkpoint)}
    val_file, test_file = root / "fabricated_validation.gz", root / "fabricated_test.gz"
    validation = evaluator.evaluate_frame(definition, job, relocated, root, raw, "validation", val_file,
        populations["validation"], device="cpu")
    evaluator.compare_recorded_history(validation["metrics"], snapshot["history_reference"])
    evaluated = evaluator.evaluate_frame(definition, job, relocated, root, full, "test", test_file,
        populations["test"], device="cpu")
    assert evaluated["state_sha256"] == payload["best_state_sha256"]
    assert evaluated["metrics"]["tail_threshold"] == 187
    assert evaluated["metrics"]["overall"]["count"] == 12
    assert evaluated["metrics"]["tail"]["count"] == 8
    assert all(evaluated["qualification"].values())
    assert evaluated["parameters_unchanged"] and evaluated["context_unchanged"]
    assert evaluator.sha(checkpoint) == before
    with gzip.open(test_file, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    for group, selected in (("overall", rows), ("tail", [row for row in rows if row["raw_quantity"] > 187])):
        q = np.asarray([row["raw_quantity"] for row in selected])
        p = np.asarray([row["predicted_raw_quantity"] for row in selected])
        t = np.asarray([row["time_nll"] for row in selected])
        metrics = evaluated["metrics"][group]
        assert metrics["qty_rmse"] == pytest.approx(np.sqrt(np.mean((q - p) ** 2)))
        assert metrics["qty_mae"] == pytest.approx(np.abs(q - p).mean())
        assert metrics["time_nll"] == pytest.approx(t.mean())
    assert max(row["raw_quantity"] for row in rows) == 10000
    for baseline in evaluated["simple_quantity_baselines"].values():
        assert baseline["overall"]["time_nll"] is None and baseline["tail"]["time_nll"] is None


def test_history_replay_never_claims_historical_tail_or_individual_predictions(fabricated):
    _, _, _, _, _, _, snapshot, _, _, _ = fabricated
    reference = snapshot["history_reference"]
    metrics = {"tail_threshold": 187.0, "tail_rule": "q > frozen_Train_large_quantity_threshold",
        "time_metric": "recorded_positive_integer_time_nll", "overall": {"count": 8, **reference["metrics"]["overall"]},
        "tail": {"count": 1, "qty_rmse": 999., "qty_mae": 555., "time_nll": 3.}}
    evaluator.compare_recorded_history(metrics, reference)
    metrics["overall"]["time_nll"] += .1
    with pytest.raises(ValueError):
        evaluator.compare_recorded_history(metrics, reference)


def test_test_gate_precedes_any_checkpoint_or_test_data_access(tmp_path):
    contract = contract_fixture()
    with patch.object(evaluator, "verify_contract", return_value=(contract, {})), \
         patch.object(evaluator, "verify_job_artifacts", side_effect=AssertionError("premature checkpoint access")), \
         patch.object(pl, "scan_parquet", side_effect=AssertionError("premature Test access")):
        with pytest.raises(FileNotFoundError):
            evaluator.evaluate_stage(tmp_path, contract["jobs"][0]["id"], "test")


def test_validation_cannot_materialize_test_rows(fabricated):
    with pytest.raises(ValueError):
        evaluator.validate_frame(fabricated[2], "validation")


def test_bound_checkpoint_and_source_paths_cannot_escape_root(tmp_path):
    with pytest.raises(ValueError):
        evaluator.bound_path(tmp_path, {"path": "../outside", "sha256": "0" * 64})


def test_reviewed_workers_remain_immutable():
    import hashlib
    assert hashlib.sha256(evaluator._INTERIM_REVIEWED_PATH.read_bytes()).hexdigest() == evaluator.INTERIM_REVIEWED_WORKER_SHA256
    assert evaluator.INTERIM_REVIEWED_WORKER_SHA256 == "3d796732555f2d97b0a8e74f1b30cbe22310a49e0b380ad061d3f92e983f70a9"


def test_loader_passes_best_dictionary_to_model_without_loading_optimizer(fabricated):
    root, _, _, definition, job, payload, snapshot, _, _, _ = fabricated
    captured = {}
    class FrozenModel:
        def load_state_dict(self, state, strict):
            captured["state"] = state
            assert strict
        def to(self, device):
            return self
        def eval(self):
            return self
        def requires_grad_(self, enabled):
            assert not enabled
            return self
    relocated = {**snapshot, "selected_checkpoint": str(root / job["checkpoint"]["path"])}
    with patch("simple_lab_test.search.common.runner.torch_load_checkpoint", return_value=payload), \
         patch.object(frozen, "build_model", return_value=(FrozenModel(), payload["encoder_config"])):
        evaluator._load_interim_selected(definition, job, relocated, root, "cpu")
    assert captured["state"] is payload["best_state_dict"]
    assert captured["state"] is not payload["model_state_dict"]


def test_all_three_synthetic_qualifications_are_required(tmp_path):
    contract = contract_fixture()
    for job in contract["jobs"]:
        path = tmp_path / "qualification" / (job["id"] + ".json")
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"schema": evaluator.SCHEMA, "status": "passed",
            "contract_sha256": evaluator.canonical(contract), "job_id": job["id"], "seed": job["seed"],
            "real_data_loaded": False, "checkpoint_loaded": False, "held_out_test_evaluated": False,
            "checks": {key: True for key in evaluator.QUALIFICATION_KEYS}}))
    evaluator.verify_synthetic_qualifications(tmp_path, contract)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        evaluator.verify_synthetic_qualifications(tmp_path, contract)


def test_validation_stage_records_limited_replay_and_preserves_raw_files(fabricated):
    root, raw, full, definition, job, _, _, training, original_contract, populations = fabricated
    raw_path, full_path = root / "stage_training.parquet", root / "stage_full.parquet"
    raw.write_parquet(raw_path)
    full.write_parquet(full_path)
    contract = {**deepcopy(original_contract), "data": {evaluator.DATASET: {"populations": populations,
        "training": bind(root, raw_path), "full": bind(root, full_path)}}}
    before = {key: evaluator.sha(root / job[key]["path"]) for key in
        ("checkpoint", "checkpoint_metadata", "selection_history", "snapshot_receipt")}
    with patch.object(evaluator, "verify_contract", return_value=(contract, training)), \
         patch.object(evaluator, "selected_job", return_value=job), \
         patch.object(frozen, "select_job", return_value=(job, definition, None, None)), \
         patch.object(evaluator, "verify_synthetic_qualifications"), \
         patch.object(evaluator, "runtime_check", return_value={"fabricated_cpu_runtime": True}):
        receipt = evaluator.evaluate_stage(root, job["id"], "validation", device="cpu")
        evaluator.verify_validation_receipt(root, contract, job, receipt)
        assert receipt["validation_replay_passed"] and receipt["recorded_history_metric_replay_count"] == 3
        assert receipt["individual_predictions_match"] is None
        assert not receipt["historical_tail_metrics_available"]
        assert not receipt["historical_individual_predictions_available"]
        assert not receipt["held_out_test_evaluated"]
        altered = {**receipt, "historical_tail_metrics_available": True}
        with pytest.raises(ValueError):
            evaluator.verify_validation_receipt(root, contract, job, altered)
        with pytest.raises(ValueError):
            evaluator.evaluate_stage(root, job["id"], "validation", device="cpu")
    assert all(evaluator.sha(root / job[key]["path"]) == digest for key, digest in before.items())
