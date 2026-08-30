from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from paper.scripts.validate_count_aware_titantpp_mac_three_seed_validation import (
    finalize,
    validate_run,
    verify_source,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "paper/contracts/count_aware_titantpp_mac_three_seed_validation_v1.json"
)
RUNNER_PATH = (
    PROJECT_ROOT
    / "simple_lab_test/search/scripts/"
    "run_count_aware_titantpp_mac_three_seed_validation_20260830.sh"
)


def load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_contract_uses_titantpp_mac_name_and_exact_nine_run_grid() -> None:
    contract = load_contract()

    assert contract["short_model_name"] == "TitanTPP-MAC"
    assert contract["run_count"] == 9
    assert len(contract["run_order"]) == 9
    assert len({tuple(run) for run in contract["run_order"]}) == 9
    assert sum(
        len(dataset["seeds"])
        for dataset in contract["datasets"].values()
    ) == 9
    assert contract["execution"]["semantic_optimization_adapter_used"] is False
    assert contract["execution"]["held_out_test"] == "locked"


def test_frozen_training_file_digests_match_source_revision() -> None:
    contract = load_contract()
    evidence = verify_source(PROJECT_ROOT, contract)

    assert evidence["status"] == "complete"
    assert evidence["verified_file_count"] == len(
        contract["frozen_training_file_sha256"]
    )
    for relative_path, expected in contract[
        "frozen_training_file_sha256"
    ].items():
        assert hashlib.sha256(
            (PROJECT_ROOT / relative_path).read_bytes()
        ).hexdigest() == expected


def test_launcher_has_valid_shell_syntax() -> None:
    completed = subprocess.run(
        ["bash", "-n", str(RUNNER_PATH)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def write_fixture_run(
    output_root: Path,
    *,
    dataset: str,
    seed: int,
    contract: dict,
) -> Path:
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
    return run_root


def test_validator_accepts_complete_validation_only_grid(tmp_path: Path) -> None:
    contract = load_contract()
    for dataset, seed in contract["run_order"]:
        run_root = write_fixture_run(
            tmp_path,
            dataset=dataset,
            seed=seed,
            contract=contract,
        )
        evidence = validate_run(
            run_root=run_root,
            dataset=dataset,
            seed=seed,
            contract=contract,
        )
        assert evidence["all_metrics_finite"] is True
        assert evidence["held_out_test_evaluated"] is False

    summary = finalize(tmp_path, contract)
    assert summary["validated_run_count"] == 9
    assert summary["held_out_test_evaluated"] is False
