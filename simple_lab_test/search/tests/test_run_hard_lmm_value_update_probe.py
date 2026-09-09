from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = ROOT / "paper/scripts/run_hard_lmm_value_update_probe.py"
CONTRACT_PATH = ROOT / "paper/contracts/hard_lmm_value_update_probe_v1.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_module("run_hard_lmm_value_update_probe_test", RUNNER_PATH)


def test_contract_freezes_train_only_no_update_scope_and_all_controls() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    runner.validate_contract(contract)
    assert contract["authorization"] == {
        "candidate_implementation": False,
        "gpu_execution": False,
        "held_out_test": False,
        "local_train_only_diagnostic": True,
        "model_parameter_updates": False,
        "validation_rows": False,
    }
    assert contract["diagnostic_scope"]["batch_size"] == 128
    assert contract["diagnostic_scope"]["maximum_targets_per_dataset"] == 4096
    assert "63" in contract["controls"]["cyclic_row_shuffles"]


def test_scope_gate_requires_improvement_over_b_and_shuffle_controls() -> None:
    hard = torch.tensor([1.0, 0.0], dtype=torch.float64)
    arms = torch.zeros(64, 2, dtype=torch.float64)
    arms[0] = torch.tensor([2.0, 0.0])
    arms[1:] = torch.tensor([0.0, 1.0])
    held = {
        metric: torch.tensor([1.0, 0.0], dtype=torch.float64)
        for metric in ("log", "raw", "body", "time")
    }
    passing = runner.scope_decision(hard, arms, held)
    assert passing["passed"] is True
    assert passing["metrics"]["raw"]["checks"]["incremental_over_b"] is True
    assert passing["metrics"]["raw"]["checks"]["exceeds_shuffle_p95"] is True

    opposed = dict(held)
    opposed["raw"] = -held["raw"]
    failing = runner.scope_decision(hard, arms, opposed)
    assert failing["passed"] is False
    assert failing["metrics"]["raw"]["checks"]["absolute_positive"] is False


def test_global_clip_matches_training_rule() -> None:
    vector = torch.tensor([3.0, 4.0], dtype=torch.float64)
    scale = runner.clip_scale(vector)
    assert scale == 1.0 / (5.0 + 1e-6)
    assert torch.linalg.vector_norm(vector * scale) < 1.0
    assert runner.clip_scale(torch.tensor([0.2, 0.1], dtype=torch.float64)) == 1.0


def test_descriptor_records_exact_selection_and_metric_conflict() -> None:
    descriptor = runner.empty_descriptor(4, 2)
    indices = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)
    quantity = torch.tensor([1.0, 7.0])
    credits = {
        "log": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        "raw": torch.tensor([[-1.0, 0.0], [-1.0, 0.0]]),
        "body": torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        "time": torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
    }
    runner.add_descriptor_batch(
        descriptor, indices, credits, quantity, boundaries=[2.0, 5.0, 8.0, 10.0]
    )
    summary = runner.summarize_descriptor(descriptor)
    assert summary["selection_counts"] == [2.0, 1.0, 1.0, 0.0]
    assert summary["active_rows"] == 3
    assert summary["metric_conflicts"]["log_vs_raw"]["conflicting_rows"] == 3
    assert summary["metric_conflicts"]["log_vs_raw"][
        "conflicting_slot_ids"
    ] == [0, 1, 2]
    assert summary["metric_conflicts"]["log_vs_raw"][
        "eligible_slot_ids"
    ] == [0, 1, 2]
    assert summary["metric_conflicts"]["log_vs_raw"]["slot_cosines"] == [
        -1.0,
        -1.0,
        -1.0,
        None,
    ]
    assert summary["quantity_strata"][0]["target_count"] == 1
    assert summary["quantity_strata"][2]["target_count"] == 1
