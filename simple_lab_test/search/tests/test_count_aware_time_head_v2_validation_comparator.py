from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from paper.scripts.compare_count_aware_time_head_v2_validation import (
    evaluate_safety_gate,
    validate_stability_decision,
)


def metrics(**updates: float | int) -> dict[str, float | int]:
    base: dict[str, float | int] = {
        "joint_objective": 0.82,
        "time_nll": 0.81,
        "log_qty_mse": 0.01,
        "qty_mae": 0.64,
        "qty_rmse": 1.70,
        "le_p95_qty_mae": 0.46,
        "best_epoch": 20,
        "completed_epochs": 60,
        "parameter_count": 90_000,
    }
    base.update(updates)
    return base


def test_gate_accepts_bounded_time_and_quantity_regressions() -> None:
    gate = evaluate_safety_gate(
        metrics(),
        metrics(
            time_nll=0.82,
            qty_mae=0.6528,
            qty_rmse=1.734,
            le_p95_qty_mae=0.4692,
        ),
    )

    assert gate["status"] == "pass"
    assert all(gate["checks"].values())


@pytest.mark.parametrize(
    ("updates", "failed_check"),
    [
        ({"time_nll": 0.8201}, "time_nll_regression_at_most_0_01"),
        ({"qty_mae": 0.653}, "qty_mae_regression_at_most_2pct"),
        ({"qty_rmse": 1.735}, "qty_rmse_regression_at_most_2pct"),
        (
            {"le_p95_qty_mae": 0.4693},
            "le_p95_qty_mae_regression_at_most_2pct",
        ),
    ],
)
def test_gate_rejects_each_guardrail_violation(
    updates: dict[str, float],
    failed_check: str,
) -> None:
    gate = evaluate_safety_gate(metrics(), metrics(**updates))

    assert gate["status"] == "fail"
    assert not gate["checks"][failed_check]


def test_stability_decision_requires_train_only_h1_selection() -> None:
    valid = {
        "selected_variant": "H1",
        "h2_executed": False,
        "selection_source": "train_stability_only",
        "validation_evaluated": False,
        "held_out_test_evaluated": False,
        "variant_gates": {"H1": {"passed": True}},
    }
    validate_stability_decision(valid)

    invalid = {**valid, "selected_variant": "H2"}
    with pytest.raises(ValueError, match="train-only selection mismatch"):
        validate_stability_decision(invalid)


def test_comparator_cli_imports_from_outside_project_root(tmp_path: Path) -> None:
    script = (
        Path(__file__).resolve().parents[3]
        / "paper"
        / "scripts"
        / "compare_count_aware_time_head_v2_validation.py"
    )

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
