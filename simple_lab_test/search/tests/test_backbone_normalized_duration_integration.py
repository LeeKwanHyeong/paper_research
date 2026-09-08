"""One synthetic CPU run through the normalized-duration command's glue."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from paper.scripts import run_backbone_normalized_duration as route
from paper.scripts.run_hard_lmm_frozen_lognormal_duration import (
    build_candidate_from_selected_checkpoint,
)
from simple_lab_test.search.common.runner import canonical_state_dict_sha256
from simple_lab_test.search.tests.test_backbone_normalized_duration import (
    ALIGNED_PATH, CONTRACT_PATH, SHA40, load, source_artifacts, source_manifest,
)


def test_bounded_run_extracts_own_cache_fits_restores_and_replays(tmp_path, monkeypatch):
    """Use real loaders/encoder/fit/I/O, with only synthetic admission metadata."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _exercise_run(tmp_path, monkeypatch)
    finally:
        torch.set_num_threads(previous_threads)


def _exercise_run(tmp_path, monkeypatch):
    # The negative quantity would fail prepare_count_frame if the held-out
    # sentinel were materialized into the admitted training/evaluation frame.
    frame = pl.DataFrame({
        "oper_part_no": ["synthetic_series"] * 8,
        "seq": list(range(8)),
        "delta_t": [0., 1., 2., 4., 3., 2., 5., 999999.],
        "demand_qty": [1., 2., 3., 7., 2., 9., 5., -9e12],
        "chronological_split": ["train"] * 5 + ["validation"] * 2 + ["test"],
    })
    data_path = tmp_path / "synthetic.parquet"
    frame.write_parquet(data_path)
    split_path = tmp_path / "synthetic_split.json"
    split_path.write_text(json.dumps({"synthetic_fixture": True, "held_out": "excluded"}))
    dataset = copy.deepcopy(load(ALIGNED_PATH)["datasets"][0])
    dataset.update(
        data_path=data_path.name,
        data_sha256=route.sha256_file(data_path),
        split_manifest_path=split_path.name,
        split_manifest_sha256=route.sha256_file(split_path),
        lookback=8,
        max_sequence_length=8,
    )
    admitted = route.prepare_count_frame(route.load_admitted_frame(data_path))
    assert admitted.height == 7
    for split in ("train", "validation"):
        population = route.exact_target_population_contract(
            admitted, target_split=split, lookback=8, max_seq_len=8,
        )
        dataset.update({
            f"expected_{split}_targets": population["target_count"],
            f"expected_{split}_target_identity_sha256": population["target_identity_sha256"],
            f"expected_{split}_target_quantity_sha256": population["target_quantity_sha256"],
        })
    train_statistics = route.derive_train_time_contract(admitted, lookback_weeks=8, max_seq_len=8)
    dataset.update(
        train_time_scale=train_statistics["time_scale"],
        train_log_scaled_mean=train_statistics["target_log_scaled_mean"],
        train_log_scaled_std=train_statistics["target_log_scaled_std"],
        reporting_body_max_train_p95=float(np.quantile([2., 3., 7., 2.], .95)),
        reporting_tail_min_exclusive_train_p99=float(np.quantile([2., 3., 7., 2.], .99)),
    )

    source, source_summary, source_spec = source_artifacts("BOUNDED")
    arguments = source["resume_identity"]["arguments"]
    arguments.update(
        data_sha256=dataset["data_sha256"],
        split_manifest_sha256=dataset["split_manifest_sha256"],
        lookback_weeks=8,
        max_seq_len=8,
    )
    source_summary["resume_identity"] = copy.deepcopy(source["resume_identity"])
    selected_epoch = int(source["best_epoch"])
    selected_rmse = float(source["selected_metric_value"])
    stop_epoch = max(40, selected_epoch + 40)
    source_history = {"history": [
        {
            "epoch": epoch,
            "val_qty_rmse": selected_rmse if epoch == selected_epoch else selected_rmse + 1.,
        }
        for epoch in range(1, stop_epoch + 1)
    ]}
    source_summary.update(
        completed_epochs=stop_epoch,
        stopped_early=True,
        best_val_qty_rmse=selected_rmse,
    )
    checkpoint_path = tmp_path / "synthetic_bounded_source.pt"
    source_summary_path = tmp_path / "synthetic_source_summary.json"
    source_history_path = tmp_path / "synthetic_source_history.json"
    torch.save(source, checkpoint_path)
    source_summary_path.write_text(json.dumps(source_summary))
    source_history_path.write_text(json.dumps(source_history))
    source_spec.update(
        checkpoint_path=checkpoint_path.name,
        checkpoint_file_sha256=route.sha256_file(checkpoint_path),
        summary_path=source_summary_path.name,
        summary_sha256=route.sha256_file(source_summary_path),
        history_path=source_history_path.name,
        history_sha256=route.sha256_file(source_history_path),
    )
    manifest = source_manifest("BOUNDED", contract_sha=route.sha256_file(CONTRACT_PATH))
    manifest["evaluation_runner_file_sha256"] = route.sha256_file(Path(route.__file__))
    manifest["runtime"] = route.runtime_identity(torch.device("cpu"))
    manifest["source"] = source_spec
    manifest["data"] = {
        "path": data_path.name, "sha256": dataset["data_sha256"],
        "split_manifest_path": split_path.name,
        "split_manifest_sha256": dataset["split_manifest_sha256"],
    }
    manifest_path = tmp_path / "job.json"
    manifest_path.write_text(json.dumps(manifest))

    # Preserve the approved contract bytes and their digest checks. Only the
    # dataset admission fixtures and current-source provider are substituted.
    design_relative = load(CONTRACT_PATH)["backbone_design_contract"]["path"]
    design_path = tmp_path / design_relative
    design_path.parent.mkdir(parents=True)
    design_path.write_bytes((route.PROJECT_ROOT / design_relative).read_bytes())
    monkeypatch.setattr(route, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(route, "current_source_revision", lambda: SHA40)
    monkeypatch.setattr(route, "validate_aligned_contract", lambda _: {dataset["dataset"]: dataset})
    output_dir = tmp_path / "fit"
    args = route.parse_args([
        "--job-manifest", str(manifest_path),
        "--expected-job-manifest-sha256", route.sha256_file(manifest_path),
        "--output-dir", str(output_dir),
        "--contract", str(CONTRACT_PATH),
        "--aligned-contract", str(ALIGNED_PATH),
        "--source-revision", SHA40,
        "--device", "cpu", "--allow-partial-contract",
        "--max-train-batches", "1", "--max-validation-batches", "1",
        "--max-epochs", "1",
    ])
    summary = route.run(args)
    assert summary["status"] == "success"
    assert summary["model_role"] == "BOUNDED"
    assert summary["qualified_full_data"] is False
    assert summary["qualified_full_fit"] is False
    assert summary["completed_epochs"] == 1
    assert summary["best_epoch"] in (0, 1)
    assert summary["quantity_prediction_bitwise_identical"] is True
    assert summary["source_quantity_prediction_sha256"] == summary["selected_quantity_prediction_sha256"]
    assert summary["held_out_test_evaluated"] is False
    assert summary["trainable_parameter_count"] == 130

    for split, expected_count, expected_durations in (
        ("train", 4, [1., 2., 4., 3.]),
        ("validation", 2, [2., 5.]),
    ):
        cache = route.torch_load_checkpoint(output_dir / "cache" / f"{split}_features.pt", map_location="cpu")
        assert cache["identity"]["model_role"] == "BOUNDED"
        assert cache["identity"]["source_state_sha256"] == source["model_state_sha256"]
        assert cache["identity"]["max_batches"] == 1
        assert cache["time_hidden"].shape == (expected_count, 64)
        assert torch.equal(cache["target_dt"], torch.tensor(expected_durations))
    selected_path = output_dir / route.SELECTED_CHECKPOINT_NAME
    selected = route.torch_load_checkpoint(selected_path, map_location="cpu")
    restored = build_candidate_from_selected_checkpoint(selected)
    assert canonical_state_dict_sha256(restored.state_dict()) == selected["model_state_sha256"]
    assert route.state_partition_sha256(restored.state_dict(), time_head=False) == source_spec["source_non_time_state_sha256"]

    # Completed-run replay must reuse the matching caches and selected state.
    selected_sha = route.sha256_file(selected_path)
    replay = route.run(args)
    assert replay["selected_state_sha256"] == summary["selected_state_sha256"]
    assert route.sha256_file(selected_path) == selected_sha
    assert replay["best_validation_proper_time_nll"] == summary["best_validation_proper_time_nll"]
