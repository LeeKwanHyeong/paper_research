import json
from pathlib import Path

import torch

from simple_lab_test.search.common.runner import canonical_state_dict_sha256
from simple_lab_test.search.compare_titantpp_strict_repro import compare_strict_runs


SOURCE_REVISION = "0123456789abcdef0123456789abcdef01234567"
DATA_HASHES = [character * 64 for character in "abcde"]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(value, file_obj, ensure_ascii=True, indent=2)


def runtime_identity(*, loader: bool) -> dict:
    runtime = {
        "mode": "strict",
        "source_revision": SOURCE_REVISION,
        "python_hash_seed": "42",
        "cublas_workspace_config": ":4096:8",
        "python_version": "3.12.0",
        "numpy_version": "2.0.0",
        "polars_version": "1.0.0",
        "torch_version": "2.11.0+cu130",
        "torch_cuda_version": "13.0",
        "cudnn_version": 91000,
        "torch_deterministic_algorithms": True,
        "torch_deterministic_warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_available": True,
        "cuda_devices": [{
            "index": 0,
            "name": "NVIDIA GeForce RTX 5090",
            "compute_capability": "12.0",
            "total_memory_bytes": 34000000000,
        }],
        "train_loader_seed": 42 if loader else None,
        "train_loader_generator": (
            "dedicated_torch_generator" if loader else "process_global_rng"
        ),
        "train_loader_num_workers": 0,
        "grouped_series_order": "oper_part_no_ascending",
    }
    if loader:
        runtime["dataset_sources"] = [
            {"role": f"source_{index}", "sha256": sha256}
            for index, sha256 in enumerate(DATA_HASHES)
        ]
    else:
        runtime["dataset_sources"] = {
            "intermittent": [
                {"role": f"source_{index}", "sha256": sha256}
                for index, sha256 in enumerate(DATA_HASHES)
            ]
        }
    return runtime


def build_fake_run(
    base_dir: Path,
    *,
    history_value: float = 1.0,
    state_offset: float = 0.0,
) -> None:
    run_dir = base_dir / "runs" / "fake_q2"
    experiment_config = {
        "base_dir": str(base_dir),
        "reproducibility_mode": "strict",
        "datasets": ["intermittent"],
        "models": ["titantpp"],
        "epochs": 3,
        "seeds": [42],
        "split_mode": "fixed",
    }
    write_json(
        base_dir / "experiment_manifest.json",
        {
            "experiment_config": experiment_config,
            "reproducibility": runtime_identity(loader=False),
        },
    )

    state = {
        "weight": torch.tensor([[1.0 + state_offset, 2.0], [3.0, 4.0]]),
        "step": torch.tensor(3, dtype=torch.int64),
    }
    state_digest = canonical_state_dict_sha256(state)
    summary = {
        "status": "success",
        "reproducibility_mode": "strict",
        "source_revision": SOURCE_REVISION,
        "train_loader_seed": 42,
        "epochs": 3,
        "best_score_epoch": 2,
        "best_val_nll_epoch": 3,
        "final_epoch": 3,
        "best_score_state_sha256": state_digest,
        "best_val_nll_state_sha256": state_digest,
        "final_state_sha256": state_digest,
    }
    write_json(run_dir / "metrics" / "summary.json", summary)
    write_json(
        run_dir / "metrics" / "history.json",
        {
            "history": [
                {"epoch": 1, "train_loss": history_value, "inactive": float("nan")},
                {"epoch": 2, "train_loss": 0.8, "inactive": float("nan")},
                {"epoch": 3, "train_loss": 0.7, "inactive": float("nan")},
            ]
        },
    )
    write_json(
        run_dir / "manifest" / "run_config.json",
        {
            "experiment_config": experiment_config,
            "run_config": {
                "dataset_name": "intermittent",
                "model_name": "titantpp",
                "candidate_name": "small_lmm",
                "seed": 42,
                "epochs": 3,
            },
            "training_config": {
                "epochs": 3,
                "batch_size": 128,
                "lookback": 52,
                "max_seq_len": 16,
                "lr": 0.001,
            },
            "rmtpp_config": {
                "qty_decoder_mode": "direct_raw_qty",
                "magnitude_norm_mode": "causal_shrinkage_revin",
                "magnitude_encoder_gradient_mode": "coupled",
                "magnitude_aux_loss_mode": "none",
                "train_loss_scope": "target_only",
                "loss_mode": "hybrid",
            },
            "encoder_config": {"d_model": 64, "memory_mode": "static_lmm"},
            "loader_sample_counts": {"train": 6, "validation": 2, "test": 2},
            "reproducibility": runtime_identity(loader=True),
        },
    )
    for selection in ("best_score", "best_val_nll", "final"):
        checkpoint_path = run_dir / "checkpoints" / f"{selection}_model.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "selection": selection,
                "model_state_dict": state,
                "model_state_sha256": state_digest,
            },
            checkpoint_path,
        )


def compare_fake_runs(run_a: Path, run_b: Path) -> dict:
    return compare_strict_runs(
        run_a,
        run_b,
        expected_epochs=3,
        expected_seed=42,
        expected_sample_counts={"train": 6, "validation": 2, "test": 2},
    )


def test_exact_comparator_passes_identical_strict_artifacts(tmp_path) -> None:
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    build_fake_run(run_a)
    build_fake_run(run_b)

    report = compare_fake_runs(run_a, run_b)

    assert report["status"] == "pass"
    assert report["mismatch_count"] == 0
    assert report["history_sha256"]["run_a"] == report["history_sha256"]["run_b"]
    assert report["state_digests"]["run_a"] == report["state_digests"]["run_b"]
    assert "held-out artifacts were not read" in report["scope"]


def test_exact_comparator_rejects_history_byte_mismatch(tmp_path) -> None:
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    build_fake_run(run_a)
    build_fake_run(run_b, history_value=1.1)

    report = compare_fake_runs(run_a, run_b)

    assert report["status"] == "fail"
    assert "exact_history_json" in report["mismatch_names"]
    assert report["history_sha256"]["run_a"] != report["history_sha256"]["run_b"]


def test_exact_comparator_rejects_tensor_state_mismatch(tmp_path) -> None:
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    build_fake_run(run_a)
    build_fake_run(run_b, state_offset=0.5)

    report = compare_fake_runs(run_a, run_b)

    assert report["status"] == "fail"
    assert "exact_selection_state_digests" in report["mismatch_names"]
    integrity_checks = [
        check
        for check in report["checks"]
        if check["name"].endswith("checkpoint_digest_integrity")
    ]
    assert all(check["passed"] for check in integrity_checks)
