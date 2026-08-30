from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from paper.scripts.validate_count_aware_titantpp_mac_three_seed_validation import (
    finalize_shard,
    finalize_split,
    validate_split_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "paper/contracts/count_aware_titantpp_mac_three_seed_validation_v1.json"
)
SPLIT_CONTRACT_PATH = (
    PROJECT_ROOT
    / "paper/contracts/count_aware_titantpp_mac_three_seed_split_v1.json"
)
RUNNER_PATH = (
    PROJECT_ROOT
    / "simple_lab_test/search/scripts/"
    "run_count_aware_titantpp_mac_validation_shard_20260830.sh"
)
HANDOFF_PATH = (
    PROJECT_ROOT
    / "simple_lab_test/search/scripts/"
    "handoff_count_aware_titantpp_mac_seed52_5090_20260830.sh"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_fixture_run(
    output_root: Path,
    *,
    dataset: str,
    seed: int,
    contract: dict,
) -> None:
    dataset_contract = contract["datasets"][dataset]
    run_root = output_root / "shards" / dataset / f"seed_{seed}"
    leaf = (
        run_root
        / "runs/titantpp_titans_mac/count_only_log_regression"
        / f"seed_{seed}"
    )
    leaf.mkdir(parents=True)
    checkpoint = leaf / "best_val_joint_objective_model.pt"
    checkpoint.write_bytes(b"fixture")
    launch = {
        "backbones": ["titantpp_titans_mac"],
        "batch_size": 128,
        "completed_run_count": 1,
        "dataset": dataset,
        "epochs": 300,
        "evaluation_scope": "validation_only",
        "expected_run_count": 1,
        "grad_clip": 1.0,
        "held_out_test_evaluated": False,
        "hidden_dim": 64,
        "lambda_log_qty": 1.0,
        "lambda_tail": 0.0,
        "lookback_weeks": dataset_contract["lookback"],
        "lr": 0.001,
        "max_seq_len": dataset_contract["max_sequence_length"],
        "quantity_variants": ["count_only_log_regression"],
        "seeds": [seed],
        "source_revision": contract["training_source_revision"],
        "status": "complete",
        "interfaces": {
            "count_only_log_regression": {
                "quantity_loss": "mse_on_log1p_quantity"
            }
        },
        "time_head": {"mode": "legacy_clamped_rmtpp"},
        "early_stopping": {
            "min_epochs": 40,
            "patience": 40,
            "monitor": "validation_joint_objective",
        },
    }
    summary = {
        "backbone": "titantpp_titans_mac",
        "variant": "count_only_log_regression",
        "seed": seed,
        "source_revision": contract["training_source_revision"],
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "status": "success",
        "checkpoint_state_sha256": "a" * 64,
        "checkpoint_path": str(checkpoint),
        "completed_epochs": 2,
        "best_epoch": 1,
        "elapsed_seconds": 3.5,
    }
    history = {"history": [{"epoch": 1}, {"epoch": 2}]}
    (run_root / "launch_contract.json").write_text(json.dumps(launch))
    (leaf / "summary.json").write_text(json.dumps(summary))
    (leaf / "history.json").write_text(json.dumps(history))


def test_split_contract_is_disjoint_and_covers_parent_grid() -> None:
    contract = load_json(CONTRACT_PATH)
    split_contract = load_json(SPLIT_CONTRACT_PATH)
    evidence = validate_split_contract(
        contract=contract,
        split_contract=split_contract,
        contract_path=CONTRACT_PATH,
    )

    assert evidence["partition_is_disjoint"] is True
    assert evidence["partition_union_matches_parent"] is True
    assert evidence["canonical_run_count"] == 9
    assert split_contract["shards"]["seed52_5090"]["run_count"] == 5
    assert split_contract["shards"]["seed62_5080"]["run_count"] == 4


def test_split_contract_rejects_duplicate_partition() -> None:
    contract = load_json(CONTRACT_PATH)
    split_contract = load_json(SPLIT_CONTRACT_PATH)
    split_contract["shards"]["seed62_5080"]["run_order"][0] = [
        "raf_spare_parts",
        52,
    ]

    with pytest.raises(ValueError, match="duplicate"):
        validate_split_contract(
            contract=contract,
            split_contract=split_contract,
        )


@pytest.mark.parametrize("script", [RUNNER_PATH, HANDOFF_PATH])
def test_split_launcher_has_valid_shell_syntax(script: Path) -> None:
    completed = subprocess.run(
        ["bash", "-n", str(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_shard_and_canonical_finalization(tmp_path: Path) -> None:
    contract = load_json(CONTRACT_PATH)
    split_contract = load_json(SPLIT_CONTRACT_PATH)
    shard_roots = {
        shard_id: tmp_path / shard_id
        for shard_id in split_contract["shards"]
    }
    for shard_id, shard in split_contract["shards"].items():
        for dataset, seed in shard["run_order"]:
            write_fixture_run(
                shard_roots[shard_id],
                dataset=dataset,
                seed=seed,
                contract=contract,
            )
        summary = finalize_shard(
            output_root=shard_roots[shard_id],
            contract=contract,
            split_contract=split_contract,
            shard_id=shard_id,
        )
        assert summary["validated_run_count"] == shard["run_count"]

    canonical = finalize_split(
        output_root=tmp_path / "canonical",
        contract=contract,
        split_contract=split_contract,
        shard_roots=shard_roots,
    )
    assert canonical["validated_run_count"] == 9
    assert canonical["execution_servers"] == ["5080", "5090"]
    assert canonical[
        "mixed_server_runtime_is_not_a_model_compute_comparison"
    ] is True
    assert canonical["held_out_test_evaluated"] is False
