from __future__ import annotations

import copy
import json
import sys
import types
from pathlib import Path

import polars as pl
import pytest
import torch

from paper.scripts import run_time_quantity_diagnostic as cli


@pytest.fixture
def frozen_contract(tmp_path: Path, monkeypatch):
    """Tiny synthetic registry substitution; never opens a research dataset."""
    rows = []
    for part in ("A", "B"):
        for seq in range(1, 11):
            split = "train" if seq <= 6 else "validation" if seq <= 9 else "test"
            rows.append({
                "oper_part_no": part, "seq": seq, "delta_t": seq % 3,
                "demand_qty": float(seq) if split != "test" else float("nan"),
                "chronological_split": split,
            })
    data = tmp_path / "observations.parquet"
    pl.DataFrame(rows).write_parquet(data)
    split_path = tmp_path / "splits.json"
    split_path.write_text(json.dumps({"format": "synthetic immutable split identity"}))
    registry = copy.deepcopy(cli.DATASET_CONTRACTS["intermittent_frozen_5000"])
    registry.update(data_sha256=cli.sha256_file(data),
                    split_manifest_sha256=cli.sha256_file(split_path))
    monkeypatch.setitem(cli.DATASET_CONTRACTS, "intermittent_frozen_5000", registry)
    revision = "a" * 40
    sources = {"paper/scripts/run_time_quantity_diagnostic.py": "b" * 64}
    monkeypatch.setattr(cli, "current_revision", lambda: revision)
    monkeypatch.setattr(cli, "source_file_hashes", lambda: sources.copy())
    cli.configure_runtime("cpu", 4)
    cli.reset_seed(42)
    raw = cli.load_train_validation_frame(data)
    frame = cli.prepare_count_frame(raw)
    populations = {}
    for split in ("train", "validation"):
        _, population = cli.exact_target_population(frame, target_split=split,
                                                    lookback_weeks=520, max_seq_len=256)
        populations[split] = {key: population[key] for key in cli.POPULATION_KEYS}
    contract = {
        "schema_version": 1, "track": cli.TRACK, "dataset_id": "intermittent_frozen_5000",
        "seed": 42, "epochs": 2, "objectives": list(cli.OBJECTIVES),
        "data": {"path": data.name, "sha256": registry["data_sha256"]},
        "split_manifest": {"path": split_path.name, "sha256": registry["split_manifest_sha256"]},
        "source": {"revision": revision, "files": sources.copy()},
        "runtime": cli.current_runtime(),
        "loader": {**cli.FIXED_LOADER, "batch_size": 3, "lookback_weeks": 520, "max_seq_len": 256},
        "model": cli.FIXED_MODEL.copy(),
        "optimizer": {"lr": 0.001, "weight_decay": 0.01, "grad_clip": 1.0},
        "populations": populations,
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    return contract_path, contract, tmp_path / "output"


def _save(path, contract):
    path.write_text(json.dumps(contract))


def _mock_summary(objective, initial, epochs_budget=2, completed=2):
    history = [
        {"epoch": epoch, "global_step": epoch * 4,
         "train_batch_order_sha256": "e" * 64,
         "train_count": 10, "train_batches": 4, "validation_count": 6, "validation_batches": 2,
         "raw_quantity_rmse": 3.0 - epoch if objective != "time_only" else None,
         "legacy_time_loss": -3.0 + epoch if objective != "quantity_only" else None}
        for epoch in range(1, completed + 1)
    ]
    selectors = {}
    for key in ("raw_quantity_rmse", "legacy_time_loss"):
        applicable = history[0][key] is not None
        best = min(history, key=lambda row: row[key]) if applicable else None
        selectors[key] = {
            "applicable": applicable, "best_epoch": best["epoch"] if best else None,
            "best_value": best[key] if best else None,
            "global_step": best["global_step"] if best else None,
            "state_sha256": "d" * 64 if best else None,
        }
    return {
        "objective": objective, "initial_state_sha256": initial,
        "epochs_budget": epochs_budget, "epochs_completed": completed,
        "status": "complete" if completed == epochs_budget else "paused_at_epoch_boundary",
        "global_step": completed * 4, "history": history, "selectors": selectors,
    }


def test_default_validates_without_training_and_excludes_heldout(frozen_contract, monkeypatch):
    path, contract, output = frozen_contract
    # Any attempted engine call in the default path is a test failure.
    fake = types.ModuleType("paper.scripts.time_quantity_diagnostic")
    fake.run_arm = lambda **kwargs: pytest.fail("validate-only attempted training")
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    result = cli.run_contract(contract_path=path, output_dir=output)
    assert result["status"] == "validated_only" and result["training_executed"] is False
    assert sorted(item.name for item in output.iterdir()) == ["wrapper_manifest.json"]
    manifest = json.loads((output / "wrapper_manifest.json").read_text())
    assert manifest["held_out_materialized"] is False
    assert manifest["populations"]["train"]["target_count"] == 10
    assert manifest["populations"]["validation"]["target_count"] == 6
    assert manifest["resolved_inputs"]["data"] == str((path.parent / "observations.parquet").resolve())
    assert manifest["loader_contract"]["delta_t_transform"] == "Int32 then clamp_min(1) then Float32"
    assert manifest["initial_state_sha256"] == result["initial_state_sha256"]
    # An identical validation is idempotent and does not rewrite evidence.
    before = (output / "wrapper_manifest.json").stat().st_mtime_ns
    cli.run_contract(contract_path=path, output_dir=output)
    assert (output / "wrapper_manifest.json").stat().st_mtime_ns == before


def test_jqt_dispatch_has_identical_fresh_states_loader_order_and_resume(frozen_contract, monkeypatch):
    path, contract, output = frozen_contract
    calls = []
    fake = types.ModuleType("paper.scripts.time_quantity_diagnostic")

    def run_arm(**kwargs):
        loader = kwargs["train_loader"]
        batch = next(iter(loader))
        calls.append({
            "objective": kwargs["objective"], "resume": kwargs["resume"],
            "initial": cli.canonical_state_dict_sha256(kwargs["model"].state_dict()),
            "batch_values": batch[4].clone(), "generator": loader.generator.get_state().clone(),
            "identity": kwargs["identity"], "budget": kwargs["epochs"],
            "stop": kwargs["stop_after_epochs"],
        })
        kwargs["output_dir"].mkdir(exist_ok=True)
        (kwargs["output_dir"] / "last_epoch_state.pt").write_bytes(b"mock checkpoint")
        return _mock_summary(kwargs["objective"], calls[-1]["initial"], kwargs["epochs"],
                             kwargs["stop_after_epochs"] or kwargs["epochs"])

    fake.run_arm = run_arm
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    result = cli.run_contract(contract_path=path, output_dir=output, execute=True, stop_after_epochs=1)
    assert result["training_executed"] is True
    assert [call["objective"] for call in calls] == list(cli.OBJECTIVES)
    assert len({call["initial"] for call in calls}) == 1
    assert all(torch.equal(call["generator"], calls[0]["generator"]) for call in calls)
    assert all(torch.equal(call["batch_values"], calls[0]["batch_values"]) for call in calls)
    assert all(call["identity"] == calls[0]["identity"] for call in calls)
    assert all(call["budget"] == 2 and call["stop"] == 1 and not call["resume"] for call in calls)
    with pytest.raises(ValueError, match="require --resume"):
        cli.run_contract(contract_path=path, output_dir=output, execute=True)
    assert len(calls) == 3
    cli.run_contract(contract_path=path, output_dir=output, execute=True, resume=True)
    assert all(call["resume"] for call in calls[3:])
    assert all(call["initial"] == calls[0]["initial"] for call in calls)
    assert (output / "paired_comparison_epoch_000001.json").exists()
    assert (output / "paired_comparison_epoch_000002.json").exists()


@pytest.mark.parametrize("mutate,match", [
    (lambda c: c["model"].update(time_intercept_limit=30.0), "cap300"),
    (lambda c: c["model"].update(backbone="thp"), "Static B"),
    (lambda c: c["loader"].update(max_seq_len=128), "context window"),
    (lambda c: c["loader"].update(train_shuffle=False), "loader setting"),
    (lambda c: c["runtime"].update(torch="drift"), "Runtime identity"),
    (lambda c: c["source"].update(revision="c" * 40), "Source revision"),
    (lambda c: c["source"]["files"].update({"extra.py": "c" * 64}), "Source file closure"),
    (lambda c: c["data"].update(sha256="c" * 64), "frozen registry"),
    (lambda c: c["populations"]["validation"].update(target_count=7), "population mismatch"),
    (lambda c: c["populations"]["train"].update(target_quantity_sha256="c" * 64), "population mismatch"),
    (lambda c: c.update(epochs=True), "positive integer"),
    (lambda c: c.update(objectives=["quantity_only"]), "All J/Q/T"),
    (lambda c: c["optimizer"].update(lr=float("nan")), "optimizer.lr"),
])
def test_contract_drift_fails_before_creating_output(frozen_contract, mutate, match):
    path, contract, output = frozen_contract
    mutate(contract)
    _save(path, contract)
    with pytest.raises(ValueError, match=match):
        cli.run_contract(contract_path=path, output_dir=output)
    assert not output.exists()


def test_actual_input_bytes_and_source_content_are_checked(frozen_contract, monkeypatch):
    path, contract, output = frozen_contract
    data_path = path.parent / "observations.parquet"
    with data_path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="data file digest mismatch"):
        cli.run_contract(contract_path=path, output_dir=output)
    assert not output.exists()


def test_existing_evidence_is_not_repurposed(frozen_contract):
    path, contract, output = frozen_contract
    cli.run_contract(contract_path=path, output_dir=output)
    contract["epochs"] = 3
    _save(path, contract)
    with pytest.raises(ValueError, match="refusing overwrite"):
        cli.run_contract(contract_path=path, output_dir=output)
    original = json.loads((output / "wrapper_manifest.json").read_text())
    assert original["contract"]["epochs"] == 2


def test_cli_defaults_and_explicit_control_flags(frozen_contract, capsys):
    path, _, output = frozen_contract
    assert cli.main(["--contract", str(path), "--output-dir", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["training_executed"] is False
    for kwargs, match in (({"resume": True}, "resume requires"),
                          ({"stop_after_epochs": 1}, "stop-after-epochs requires")):
        with pytest.raises(ValueError, match=match):
            cli.run_contract(contract_path=path, output_dir=output, **kwargs)


def test_source_hash_closure_includes_transitive_and_relative_imports(tmp_path, monkeypatch):
    root = tmp_path / "source"
    files = {
        "paper/scripts/run_time_quantity_diagnostic.py": "from models.toy import Model\n",
        "paper/scripts/time_quantity_diagnostic.py": "from .helper import run\n",
        "paper/scripts/helper.py": "from data_loader.loader import load\n",
        "models/toy.py": "class Model: pass\n",
        "data_loader/loader.py": "def load(): pass\n",
        "paper/scripts/unrelated.py": "unrelated = True\n",
    }
    for name, contents in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    monkeypatch.setattr(cli, "PROJECT_ROOT", root)
    hashes = cli.source_file_hashes()
    assert set(hashes) == set(files) - {"paper/scripts/unrelated.py"}
    (root / "data_loader/loader.py").write_text("def load(): return 1\n")
    assert cli.source_file_hashes()["data_loader/loader.py"] != hashes["data_loader/loader.py"]


def test_cli_and_actual_engine_complete_tiny_synthetic_jqt(frozen_contract, monkeypatch):
    path, contract, output = frozen_contract
    # This is an in-memory/synthetic integration test, not research training.
    registry = copy.deepcopy(cli.DATASET_CONTRACTS["intermittent_frozen_5000"])
    registry["max_seq_len"] = 8
    monkeypatch.setitem(cli.DATASET_CONTRACTS, "intermittent_frozen_5000", registry)
    contract["loader"]["max_seq_len"] = 8
    contract["epochs"] = 1
    _save(path, contract)
    result = cli.run_contract(contract_path=path, output_dir=output, execute=True)
    assert set(result["arms"]) == set(cli.OBJECTIVES)
    summaries = {}
    for objective in cli.OBJECTIVES:
        summaries[objective] = json.loads((output / objective / "summary.json").read_text())
        assert (output / objective / "last_epoch_state.pt").is_file()
    # Resume a completed fixed-budget experiment without extending its budget.
    before = {objective: (output / objective / "history.json").read_bytes() for objective in cli.OBJECTIVES}
    cli.run_contract(contract_path=path, output_dir=output, execute=True, resume=True)
    assert all((output / objective / "history.json").read_bytes() == before[objective]
               for objective in cli.OBJECTIVES)


def test_seed_reset_preserves_strict_determinism(frozen_contract):
    path, contract, output = frozen_contract
    torch.use_deterministic_algorithms(True, warn_only=True)
    cli.run_contract(contract_path=path, output_dir=output)
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    frame, metadata = cli.validate_inputs(path, contract, "cpu")
    torch.use_deterministic_algorithms(True, warn_only=True)
    cli.build_arm_inputs(contract, frame, metadata)
    assert not torch.is_deterministic_algorithms_warn_only_enabled()


def test_external_first_party_import_is_rejected(frozen_contract, monkeypatch):
    path, _, output = frozen_contract
    shadow = types.ModuleType("models.external_shadow")
    shadow.__file__ = str(path.parent / "other_checkout" / "model.py")
    monkeypatch.setitem(sys.modules, shadow.__name__, shadow)
    with pytest.raises(ValueError, match="import origin escapes"):
        cli.run_contract(contract_path=path, output_dir=output)
    assert not output.exists()


def test_external_namespace_search_path_is_rejected(frozen_contract, monkeypatch):
    path, _, output = frozen_contract
    shadow = types.ModuleType("paper.external_namespace")
    shadow.__path__ = [str(path.parent / "other_checkout")]
    monkeypatch.setitem(sys.modules, shadow.__name__, shadow)
    with pytest.raises(ValueError, match="namespace origin escapes"):
        cli.run_contract(contract_path=path, output_dir=output)
    assert not output.exists()


def test_paired_comparison_uses_corresponding_task_selectors():
    initial = "a" * 64
    summaries = {objective: _mock_summary(objective, initial) for objective in cli.OBJECTIVES}
    result = cli.audit_arm_comparison(summaries, initial_hash=initial, epochs_budget=2)
    pairs = result["paired_comparisons"]
    assert result["equal_training_exposure_verified"] is True
    assert pairs["quantity"]["joint"]["best_epoch"] == 2
    assert pairs["time"]["joint"]["best_epoch"] == 1
    assert pairs["quantity"]["single_task"]["objective"] == "quantity_only"
    assert pairs["time"]["single_task"]["objective"] == "time_only"
    assert "not combined" in result["joint_checkpoint_policy"]


@pytest.mark.parametrize("mutate,match", [
    (lambda s: s["quantity_only"].update(initial_state_sha256="b" * 64), "initialization mismatch"),
    (lambda s: s["time_only"].update(global_step=9), "global steps differ"),
    (lambda s: s["quantity_only"]["history"][0].update(train_batch_order_sha256="f" * 64), "sample order/count/step"),
    (lambda s: s["quantity_only"]["history"][0].update(train_count=9), "sample order/count/step"),
    (lambda s: s["time_only"]["history"][0].update(validation_count=5), "sample order/count/step"),
    (lambda s: s["joint"]["selectors"]["legacy_time_loss"].update(best_epoch=2), "earliest strict minimum"),
])
def test_paired_comparison_rejects_exposure_and_selector_drift(mutate, match):
    initial = "a" * 64
    summaries = {objective: _mock_summary(objective, initial) for objective in cli.OBJECTIVES}
    mutate(summaries)
    with pytest.raises(ValueError, match=match):
        cli.audit_arm_comparison(summaries, initial_hash=initial, epochs_budget=2)
