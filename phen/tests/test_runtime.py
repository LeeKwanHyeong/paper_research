"""Fabricated-data integration only; no research dataset or GPU is read."""

import json
import subprocess
import sys

import numpy as np
import polars as pl
import pytest
import torch

from phen.data import EventWindows, load_frame, target_outputs, validate_frame
from phen.experiment import ROOT, prepare_data
from phen.registry import MODELS, build_model
from phen.run import main


@pytest.fixture
def fixture_data(tmp_path):
    rows = [{"oper_part_no": part, "seq": i * 2, "delta_t": (i % 3) + 1,
             "demand_qty": float(i + 1),
             "chronological_split": "train" if i < 4 else "validation" if i < 6 else "test"}
            for part in ("a", "b") for i in range(8)]
    frame = pl.DataFrame(rows)
    path = tmp_path / "events.parquet"
    frame.write_parquet(path)
    config = {"schema": "phen_reproduction_v1", "dataset_id": "fabricated", "data_path": str(path),
              "loader": {"lookback": 9, "max_seq_len": 8, "batch_size": 4},
              "model": {"hidden_dim": 64, "time_scale": 2.,
                        "time_initial_location": float(np.log(np.array([2., 3., 1.]) / 2).mean()),
                        "time_initial_scale": float(np.log(np.array([2., 3., 1.]) / 2).std()),
                        "time_observation_contract": {"mode": "positive_integer_round_clamp_v1", "unit": "week", "top_code": None}},
              "statistics": {"train_log_mean": float(np.log1p([1., 2., 3., 4.]).mean()),
                             "train_log_std": float(np.log1p([1., 2., 3., 4.]).std())},
              "training": {"epochs": 1, "min_epochs": 1, "patience": 1, "lr": .001,
                           "weight_decay": .01, "grad_clip": 1., "lambda_log_qty": 1.},
              "seeds": [42]}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return frame, path, config, config_path


def test_windows_and_population_match_legacy(fixture_data):
    pytest.importorskip("data_loader.event_seq_data_module", reason="Legacy loader is not included in a source-only archive")
    from data_loader.event_seq_data_module import RMTPPWeekLookbackDataset
    from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame, right_pad_batch
    from paper.scripts.run_count_aware_tpp_backbone_control import exact_target_population
    frame, path, config, _ = fixture_data
    admitted = load_frame(path)
    for split in ("train", "validation"):
        new = EventWindows(admitted, target_split=split, lookback=9, max_seq_len=8)
        legacy_frame = prepare_count_frame(admitted)
        old = RMTPPWeekLookbackDataset(legacy_frame, lookback_weeks=9, max_seq_len=8,
                                       mode="all", target_splits={split})
        _, population = exact_target_population(legacy_frame, target_split=split,
                                                lookback_weeks=9, max_seq_len=8)
        assert new.population() == {k: population[k] for k in new.population()}
        assert len(new) == len(old)
        for index in range(len(new)):
            before, after = old[index], new[index]
            dts, qty, mask, _ = right_pad_batch(before["dts"][None], before["values"][None], before["mask"][None])
            for actual, expected in ((after["dts"], dts[0]), (after["quantities"], qty[0]), (after["mask"], mask[0])):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_filter_and_frozen_population_checks(fixture_data):
    frame, path, config, _ = fixture_data
    corrupt_test = frame.with_columns(pl.when(pl.col("chronological_split") == "test")
                                     .then(float("nan")).otherwise(pl.col("demand_qty")).alias("demand_qty"))
    corrupt_test.write_parquet(path)
    datasets, info = prepare_data(config, path)
    assert info["held_out_materialized"] is False
    assert set(datasets) == {"train", "validation"}
    config["expected_populations"] = {"train": {**datasets["train"].population(), "target_count": 999}}
    with pytest.raises(ValueError, match="population"):
        prepare_data(config, path)
    with pytest.raises(ValueError, match="demand_qty"):
        load_frame(path, include_test=True)


def test_frozen_reporting_boundaries_are_not_refitted(fixture_data):
    _, path, config, _ = fixture_data
    config["quantity_boundaries"] = [2., 3., 4., 5.]
    _, info = prepare_data(config, path)
    assert info["quantity_boundaries"] == config["quantity_boundaries"]
    assert info["quantity_boundaries_source"] == "frozen_configuration"


@pytest.mark.parametrize("name", MODELS)
def test_train_and_selected_checkpoint_replay(fixture_data, tmp_path, name):
    _, _, _, config_path = fixture_data
    args = ["--config", str(config_path), "--model", name, "--output-root", str(tmp_path / "runs")]
    main(args)
    output = tmp_path / "runs/fabricated" / name / "seed_42"
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "complete" and summary["held_out_test_evaluated"] is False
    assert (output / "source.tar.gz").is_file() and (output / "pip_freeze.txt").is_file()
    main([*args, "--mode", "evaluate", "--checkpoint", str(output / "best.pt")])
    replay = json.loads((output / "evaluation_validation/summary.json").read_text())
    assert replay["metrics"] == summary["selected_validation"]
    with pytest.raises(FileExistsError):
        main(args)


def test_test_gate_fails_before_config_or_data_read():
    with pytest.raises(PermissionError):
        main(["--config", "/missing/config.json", "--mode", "evaluate", "--split", "test"])


def test_authorized_fabricated_test_evaluation(fixture_data, tmp_path):
    _, _, _, config_path = fixture_data
    args = ["--config", str(config_path), "--output-root", str(tmp_path / "heldout")]
    main(args)
    checkpoint = tmp_path / "heldout/fabricated/phen/seed_42/best.pt"
    main([*args, "--mode", "evaluate", "--checkpoint", str(checkpoint), "--split", "test", "--allow-held-out"])
    summary = json.loads((checkpoint.parent / "evaluation_test/summary.json").read_text())
    assert summary["held_out_test_evaluated"] is True
    assert summary["metrics"]["count"] == 4


def test_chronology_and_duplicate_rejection(fixture_data):
    frame, _, _, _ = fixture_data
    with pytest.raises(ValueError, match="Duplicate"):
        validate_frame(pl.concat([frame, frame.head(1)]))
    reversed_split = frame.with_columns(pl.when(pl.col("seq") == 0).then(pl.lit("test"))
                                       .otherwise(pl.col("chronological_split")).alias("chronological_split"))
    with pytest.raises(ValueError, match="chronological"):
        validate_frame(reversed_split)


@pytest.mark.parametrize("name", MODELS)
def test_target_values_are_withheld(name):
    from phen.tests.test_equivalence import HEADS
    model = build_model(name, hidden_dim=64, train_log_mean=1.2, max_seq_len=8, **HEADS).eval()
    batch = {"dts": torch.ones(1, 8), "quantities": torch.ones(1, 8),
             "mask": torch.ones(1, 8, dtype=torch.bool)}
    before = target_outputs(model, batch)
    batch["dts"][0, -1], batch["quantities"][0, -1] = 27, 10000
    after = target_outputs(model, batch)
    torch.testing.assert_close(before["pred_qty"], after["pred_qty"], rtol=0, atol=0)
    torch.testing.assert_close(before["pred_dt"], after["pred_dt"], rtol=0, atol=0)


@pytest.mark.parametrize("name", MODELS)
def test_shell_entrypoint_dry_run(name):
    import os
    env = {**os.environ, "SEEDS": "42", "PYTHON_BIN": sys.executable}
    result = subprocess.run(["bash", str(ROOT / f"scripts/phen/run_{name}.sh"),
                             "yellow_trip_hourly", "--data", "/missing/data.parquet", "--dry-run"],
                            cwd="/tmp", env=env, check=True, capture_output=True, text=True)
    plan = json.loads(result.stdout)
    assert plan["model"] == name and plan["data_read"] is False and plan["training_executed"] is False


def test_runtime_has_no_legacy_experiment_imports():
    result = subprocess.run([sys.executable, "-c",
        "from phen.registry import MODELS, build_model; import sys; "
        "from phen.tests.test_equivalence import HEADS; "
        "[build_model(name, hidden_dim=64, train_log_mean=1.2, max_seq_len=8, **HEADS) for name in MODELS]; "
        "assert not any(name.startswith(('paper.scripts', 'simple_lab_test', 'models.TPPs', 'models.Titan')) for name in sys.modules)"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_repeated_seed_reproduces_training(fixture_data, tmp_path):
    _, _, _, config_path = fixture_data
    states = []
    histories = []
    for name in ("first", "second"):
        main(["--config", str(config_path), "--output-root", str(tmp_path / name)])
        output = tmp_path / name / "fabricated/phen/seed_42"
        states.append(torch.load(output / "last.pt", weights_only=True)["model_state_dict"])
        histories.append((output / "history.jsonl").read_text())
    assert histories[0] == histories[1]
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)


def test_all_frozen_configs_and_shell_syntax():
    from phen.experiment import validate_config
    for path in (ROOT / "phen/configs").glob("*.json"):
        config = validate_config(json.loads(path.read_text()))
        assert config["model"]["hidden_dim"] == 64
        assert config["training"] == {"epochs": 300, "min_epochs": 40, "patience": 40,
                                      "lr": .001, "weight_decay": .01, "grad_clip": 1, "lambda_log_qty": 1}
    for path in (ROOT / "scripts/phen").glob("*.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_source_archive_can_train_without_legacy_checkout(fixture_data, tmp_path):
    import tarfile
    _, _, _, config_path = fixture_data
    original = tmp_path / "original"
    main(["--config", str(config_path), "--output-root", str(original)])
    archive_path = original / "fabricated/phen/seed_42/source.tar.gz"
    extracted = tmp_path / "extracted"
    with tarfile.open(archive_path) as archive:
        archive.extractall(extracted, filter="data")
    assert not (extracted / "models").exists()
    result = subprocess.run([sys.executable, "-m", "phen.run", "--config", str(config_path),
                             "--output-root", str(tmp_path / "standalone")],
                            cwd=extracted, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "standalone/fabricated/phen/seed_42/manifest.json").read_text())
    assert manifest["git_revision"] is None


def test_test_rejects_nonselected_checkpoint(fixture_data, tmp_path):
    _, _, _, config_path = fixture_data
    output = tmp_path / "selection"
    main(["--config", str(config_path), "--output-root", str(output)])
    checkpoint = output / "fabricated/phen/seed_42/last.pt"
    payload = torch.load(checkpoint, weights_only=True)
    payload["epoch"] = payload["best_epoch"] + 1
    diagnostic = tmp_path / "diagnostic.pt"
    torch.save(payload, diagnostic)
    with pytest.raises(ValueError, match="selected"):
        main(["--config", str(config_path), "--mode", "evaluate", "--checkpoint", str(diagnostic),
              "--split", "test", "--allow-held-out", "--output-root", str(output)])


def test_shell_default_seed_matrix_comes_from_configuration():
    import os
    env = {**os.environ, "PYTHON_BIN": sys.executable}
    env.pop("SEEDS", None)
    result = subprocess.run(["bash", str(ROOT / "scripts/phen/run_phen.sh"),
                             "yellow_trip_hourly", "--dry-run"], env=env,
                            capture_output=True, text=True, check=True)
    decoder = json.JSONDecoder()
    remaining, seeds = result.stdout.strip(), []
    while remaining:
        value, stop = decoder.raw_decode(remaining)
        seeds.append(value["seed"])
        remaining = remaining[stop:].lstrip()
    config = json.loads((ROOT / "phen/configs/yellow_trip_hourly.json").read_text())
    assert seeds == config["seeds"]


def test_outputs_cannot_pollute_source_snapshots(fixture_data):
    _, _, _, config_path = fixture_data
    with pytest.raises(ValueError, match="source directories"):
        main(["--config", str(config_path), "--output-root", str(ROOT / "phen/runs"), "--dry-run"])
