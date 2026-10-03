"""Exercise the native adapter's full 120-epoch path using fabricated CPU data."""
from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import polars as pl
import pytest
import torch

from data_loader.event_seq_data_module import RMTPPWeekLookbackDataset
from paper.scripts import quantity_comparison_data as qdata
from paper.scripts import run_count_aware_tpp_backbone_control as cli
from paper.scripts import run_pair_message_execution as native
from paper.scripts.run_hard_lmm_frozen_lognormal_duration import target_dt_sha256
from simple_lab_test.search.tests.test_pair_message_preparation import contract


def test_real_three_arm_production_and_endpoint_replay(contract, monkeypatch, tmp_path):
    rows = [{"oper_part_no": str(part), "seq": i,
             "delta_t": [0, 1, 2, 3, 4, 7][i % 6],
             "demand_qty": float(1 + i + part * 20),
             "chronological_split": "train" if i < 8 else "validation"}
            for part in range(3) for i in range(12)]
    raw = pl.DataFrame(rows)
    path, manifest = tmp_path / "synthetic.parquet", tmp_path / "synthetic_split.json"
    raw.write_parquet(path)
    manifest.write_text('{"synthetic": true}')
    data = contract["datasets"][0]
    identity = data["inherited_data_identity"]
    identity["data"] = {"path": str(path), "sha256": cli.sha256_file(path)}
    identity["split_manifest"] = {"path": str(manifest), "sha256": cli.sha256_file(manifest)}
    frame = cli.prepare_count_frame(raw)
    loader = {k: data["loader"][k] for k in ("lookback_weeks", "max_seq_len")}
    identity["populations"] = {split: qdata._population_view(
        qdata.exact_target_population(frame, target_split=split, **loader)[1])
        for split in ("train", "validation")}
    quantities = raw.filter(pl.col("chronological_split") == "train")["demand_qty"].to_numpy()
    data["mu_all_train_rows"] = float(np.log1p(quantities).mean())
    stats = qdata.prepare_dataset_statistics(data)
    data["statistics"] = {"train_log_mean": stats["all_train_rows"]["log1p_mean"],
        "train_log_std": stats["all_train_rows"]["log1p_std"],
        "raw_scale": stats["canonical_train_targets"]["raw_scale"],
        "all_train_quantity_sha256": stats["all_train_rows"]["quantity_sha256"]}
    data["quantity_boundaries_all_train_rows"] = np.quantile(quantities, [.5, .9, .95, .99]).tolist()
    data["expected_global_steps"] = 120
    time_stats = cli.derive_train_time_contract(frame, **loader)
    frozen = data["time_statistics"]
    for name, key in (("time_scale", "train_time_scale"),
                      ("target_log_scaled_mean", "train_log_scaled_mean"),
                      ("target_log_scaled_std", "train_log_scaled_std")):
        frozen[key] = time_stats[name]
    for split in ("train", "validation"):
        ds = RMTPPWeekLookbackDataset(frame, **loader, val_ratio=.2, mode="all",
            split_col="chronological_split", target_splits={split})
        values = torch.tensor([max(1., float(ds.dt_lists[p][i + 1])) for p, i in ds.index], dtype=torch.float64)
        frozen[f"expected_{split}_targets"] = values.numel()
        frozen[f"expected_{split}_target_dt_sha256"] = target_dt_sha256(values)
    data["model"].update(time_scale=time_stats["time_scale"],
        time_initial_location=time_stats["target_log_scaled_mean"],
        time_initial_scale=time_stats["target_log_scaled_std"])
    output = tmp_path / "execution"
    output.mkdir()
    contract["hosts"]["5080"]["output_dir"] = str(output)
    runtime = {"synthetic_cpu": True}
    # Only deployment identity and device selection are substituted. Data
    # checks, initialization, loaders, optimizer, selector and replays are real.
    monkeypatch.setattr(native.common, "runtime_check", lambda *a: runtime)
    monkeypatch.setattr(native.proposal, "validate_contract", lambda *a, **k: None)
    real_args, real_replay = native.training_args, native.replay_checkpoint
    def cpu_args(*args):
        parsed = real_args(*args)
        parsed.device = "cpu"
        return parsed
    monkeypatch.setattr(native, "training_args", cpu_args)
    monkeypatch.setattr(native, "replay_checkpoint", lambda *a, **k: real_replay(*a, **{**k, "device": "cpu"}))
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = native.production(contract, {"deadline_unix": 1e20}, "5080",
            {"runtime": runtime}, lambda: None)
    finally:
        torch.set_num_threads(threads)
    assert result["status"] == "complete"
    assert result["total_optimizer_steps"] == 360 and result["endpoint_replays"] == 6
    dataset = result["datasets"][data["dataset_id"]]
    assert dataset["exposure_equal"] is True
    assert set(dataset["arms"]) == set(native.ARMS)
    for arm, outcome in dataset["arms"].items():
        assert outcome["global_steps"] == 120
        assert outcome["selected"]["time_nll"] >= 0
        assert outcome["last30"]["count"] == 30
        run = output / data["dataset_id"] / "runs" / arm / native.VARIANT / "seed_42"
        history = json.loads((run / "history.json").read_text())["history"]
        assert len(history) == 120
        assert outcome["best_epoch"] == min(history, key=lambda row: row["val_qty_rmse"])["epoch"]
    assert result["held_out_test_evaluated"] is False
    # Rejected artifacts must fail before constructing a model or a loader.
    path = run / "last_epoch_state.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    for key, value in (("seed", 7), ("checkpoint_monitor", "validation_time_nll"),
                       ("interface_meta", {**saved["interface_meta"], "execution_contract_sha256": "foreign"})):
        foreign = deepcopy(saved)
        foreign[key] = value
        corrupted = tmp_path / (key + ".pt")
        torch.save(foreign, corrupted)
        with pytest.raises(ValueError, match="Replay"):
            real_replay(corrupted, data, frame, lambda: None, device="cpu",
                expected_arm=arm, expected_identity=saved["resume_identity"],
                expected_initial=saved["initial_state_sha256"], expected_epoch=120)
