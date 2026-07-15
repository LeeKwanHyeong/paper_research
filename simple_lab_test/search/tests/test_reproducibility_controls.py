from dataclasses import replace
from types import SimpleNamespace

import polars as pl
import pytest
import torch

from data_loader.event_seq_data_module import RMTPPWeekLookbackDataset
from models.RMTPPs.config import RMTPPConfig
from simple_lab_test.search.common.configs import ExperimentConfig, RunConfig
from simple_lab_test.search.common.experiment_utils import DatasetSpec
from simple_lab_test.search.common.runner import (
    build_dataset_source_manifest,
    build_run_paths,
    canonical_state_dict_sha256,
    configure_reproducibility,
    restore_train_loader_generator_state,
    save_epoch_resume_checkpoint,
    torch_load_checkpoint,
)
from simple_lab_test.search.tpp_experiment import build_long_epoch_config, build_parser
from utils.training import TrainingConfig, make_fixed_split_week_lookback_loaders


@pytest.fixture(autouse=True)
def restore_torch_determinism():
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only_getter = getattr(
        torch,
        "is_deterministic_algorithms_warn_only_enabled",
        None,
    )
    warn_only = bool(warn_only_getter()) if callable(warn_only_getter) else False
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    yield
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark


def strict_environment(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "42")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv(
        "SOURCE_REVISION",
        "0123456789abcdef0123456789abcdef01234567",
    )


def make_run_config() -> RunConfig:
    return RunConfig(
        dataset_name="intermittent",
        dataset_kind="marked_target",
        model_name="titantpp",
        candidate_name="small_lmm",
        candidate=SimpleNamespace(name="small_lmm"),
        seed=42,
        epochs=3,
        scale_base=2.0,
        titan_profile="dataset_best",
    )


def make_fixed_split_frame() -> pl.DataFrame:
    rows = []
    for part in ("z_part", "a_part", "m_part"):
        for seq in range(4):
            rows.append({
                "oper_part_no": part,
                "seq": seq,
                "delta_t": seq + 1,
                "mark": seq % 2,
                "scale_residual": float(seq) / 10.0,
                "chronological_split": "train" if seq < 3 else "validation",
            })
    return pl.DataFrame(list(reversed(rows)))


def loader_part_order(loader) -> list[int]:
    order: list[int] = []
    for _, _, _, part_indices, _ in loader:
        order.extend(int(value) for value in part_indices.tolist())
    return order


def test_standard_mode_preserves_legacy_path_and_strict_path_is_distinct(tmp_path) -> None:
    args = build_parser().parse_args([
        "long-epoch",
        "--datasets",
        "intermittent",
        "--models",
        "titantpp",
    ])
    standard_cfg = replace(build_long_epoch_config(args), base_dir=str(tmp_path))
    strict_args = build_parser().parse_args([
        "long-epoch",
        "--datasets",
        "intermittent",
        "--models",
        "titantpp",
        "--reproducibility-mode",
        "strict",
    ])
    strict_cfg = replace(build_long_epoch_config(strict_args), base_dir=str(tmp_path))

    standard_path = str(build_run_paths(standard_cfg, make_run_config()).run_dir)
    strict_path = str(build_run_paths(strict_cfg, make_run_config()).run_dir)

    assert standard_cfg.reproducibility_mode == "standard"
    assert "repro_strict" not in standard_path
    assert "repro_strict" in strict_path
    assert standard_path != strict_path


def test_strict_mode_rejects_missing_launcher_environment(monkeypatch) -> None:
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.delenv("SOURCE_REVISION", raising=False)

    assert configure_reproducibility("standard")["mode"] == "standard"
    with pytest.raises(RuntimeError, match="PYTHONHASHSEED"):
        configure_reproducibility("strict")

    monkeypatch.setenv("PYTHONHASHSEED", "42")
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        configure_reproducibility("strict")

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with pytest.raises(RuntimeError, match="SOURCE_REVISION"):
        configure_reproducibility("strict")

    monkeypatch.setenv("SOURCE_REVISION", "0123456")
    with pytest.raises(RuntimeError, match="full 40- or 64-character"):
        configure_reproducibility("strict")


def test_strict_mode_enables_and_records_deterministic_runtime(monkeypatch) -> None:
    strict_environment(monkeypatch)

    manifest = configure_reproducibility("strict")

    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
    assert manifest["mode"] == "strict"
    assert manifest["source_revision"] == "0123456789abcdef0123456789abcdef01234567"
    assert manifest["python_hash_seed"] == "42"
    assert manifest["cublas_workspace_config"] == ":4096:8"
    assert manifest["torch_deterministic_algorithms"] is True
    assert manifest["torch_deterministic_warn_only"] is False
    assert manifest["grouped_series_order"] == "oper_part_no_ascending"
    assert manifest["train_loader_num_workers"] == 0


def test_grouped_series_and_dedicated_loader_shuffle_are_exact() -> None:
    marked_df = make_fixed_split_frame()
    dataset = RMTPPWeekLookbackDataset(
        marked_df,
        lookback_weeks=52,
        max_seq_len=8,
        mode="all",
        target_splits={"train"},
    )
    assert dataset.parts == ["a_part", "m_part", "z_part"]

    training_cfg = TrainingConfig(
        lookback=52,
        max_seq_len=8,
        batch_size=2,
        device="cpu",
    )
    generator_a = torch.Generator().manual_seed(42)
    generator_b = torch.Generator().manual_seed(42)
    loader_a, _, _ = make_fixed_split_week_lookback_loaders(
        marked_df,
        training_cfg,
        train_generator=generator_a,
    )
    torch.rand(100)
    loader_b, _, _ = make_fixed_split_week_lookback_loaders(
        marked_df,
        training_cfg,
        train_generator=generator_b,
    )

    assert loader_a.num_workers == 0
    assert loader_b.num_workers == 0
    assert loader_a.generator is generator_a
    assert loader_part_order(loader_a) == loader_part_order(loader_b)


def test_train_loader_generator_state_restores_next_shuffle() -> None:
    generator = torch.Generator().manual_seed(42)
    torch.randperm(19, generator=generator)
    saved_state = generator.get_state()
    expected = torch.randperm(19, generator=generator)

    restored = torch.Generator().manual_seed(999)
    restore_train_loader_generator_state(restored, saved_state)
    actual = torch.randperm(19, generator=restored)

    assert torch.equal(actual, expected)
    with pytest.raises(ValueError, match="no train loader generator state"):
        restore_train_loader_generator_state(restored, None)


def test_resume_checkpoint_persists_loader_state_and_model_digest(tmp_path) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(42)
    torch.randperm(11, generator=generator)
    checkpoint_path = tmp_path / "last_epoch_state.pt"
    cfg = ExperimentConfig(
        base_dir=str(tmp_path),
        device="cpu",
        reproducibility_mode="strict",
    )
    run_cfg = replace(make_run_config(), candidate={"name": "small_lmm"})

    save_epoch_resume_checkpoint(
        path=checkpoint_path,
        epoch=1,
        model=model,
        optimizer=optimizer,
        history=[{"epoch": 1, "train_loss": 1.0}],
        best_score=0.0,
        best_val_nll=1.0,
        best_score_state=None,
        best_val_nll_state=None,
        cfg=cfg,
        run_cfg=run_cfg,
        training_cfg=TrainingConfig(device="cpu"),
        rmtpp_cfg=RMTPPConfig(num_marks=3),
        encoder_cfg=None,
        train_loader_generator=generator,
    )
    expected_next_shuffle = torch.randperm(11, generator=generator)
    payload = torch_load_checkpoint(checkpoint_path, map_location="cpu")
    restored_generator = torch.Generator().manual_seed(999)
    restore_train_loader_generator_state(
        restored_generator,
        payload["train_loader_generator_state"],
    )

    assert payload["model_state_sha256"] == canonical_state_dict_sha256(
        payload["model_state_dict"]
    )
    assert torch.equal(
        torch.randperm(11, generator=restored_generator),
        expected_next_shuffle,
    )


def test_dataset_source_manifest_records_exact_sha256(tmp_path) -> None:
    source_path = tmp_path / "events.parquet"
    source_path.write_bytes(b"deterministic-dataset")
    spec = DatasetSpec(
        name="tiny",
        parquet_path=str(source_path),
        kind="marked_target",
    )

    manifest = build_dataset_source_manifest(
        [spec],
        split_mode="internal",
        include_sha256=True,
    )

    assert manifest["tiny"] == [{
        "role": "event_table",
        "path": str(source_path.resolve()),
        "size_bytes": len(b"deterministic-dataset"),
        "sha256": "16446366104ec46803cdb15442fe4cb5fbf5ac74b5fcd346eace45f7b93f227f",
    }]


def test_canonical_state_digest_ignores_mapping_order_but_detects_tensor_change() -> None:
    left = {
        "weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "step": torch.tensor(7, dtype=torch.int64),
    }
    reordered = {
        "step": left["step"].clone(),
        "weight": left["weight"].clone(),
    }
    changed = {
        **reordered,
        "weight": reordered["weight"].clone(),
    }
    changed["weight"][0, 0] += 1.0

    left_digest = canonical_state_dict_sha256(left)

    assert len(left_digest) == 64
    assert canonical_state_dict_sha256(reordered) == left_digest
    assert canonical_state_dict_sha256(changed) != left_digest
