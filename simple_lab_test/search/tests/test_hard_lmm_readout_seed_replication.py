import copy
import statistics

import pytest
import torch

from paper.scripts.run_hard_lmm_frozen_probe import load_json
from paper.scripts.run_hard_lmm_readout_seed_replication import (
    CONTRACT_PATH, inherited_contract, read_reference, require_frozen, verify_alignment,
)
from paper.scripts.validate_hard_lmm_readout_seed_replication import paired_summary, replication_decision
from simple_lab_test.search.common.runner import canonical_state_dict_sha256


def test_contract_keeps_discovery_optimizer_and_selector():
    contract = load_json(CONTRACT_PATH)
    parent = inherited_contract(contract, {})
    assert parent["training"]["seed"] == parent["training"]["shuffle_seed"] == 42
    assert parent["training"]["maximum_epochs"] == 40
    assert parent["training"]["batch_size"] == 1024
    assert parent["training"]["learning_rate"] == .001
    assert contract["hypotheses"] == ["constant_log_mse", "linear_raw_mae"]
    changed = copy.deepcopy(contract)
    changed["adapter_initialization_and_shuffle_seed"] = 52
    with pytest.raises(ValueError, match="seed"):
        inherited_contract(changed, {})
    changed = copy.deepcopy(contract)
    changed["heads"].append("mlp")
    with pytest.raises(ValueError, match="grid"):
        inherited_contract(changed, {})


def synthetic_cache():
    return {"features": torch.randn(5, 138), "target_index": torch.arange(5),
        "series_index": torch.arange(5), "context_end": torch.arange(5),
        "history_length": torch.full((5,), 3), "quantity": torch.arange(5).float()}


@pytest.mark.parametrize("key", ["target_index", "series_index", "context_end", "history_length", "quantity"])
def test_alignment_rejects_changed_ids_or_labels(key):
    a = synthetic_cache()
    b = copy.deepcopy(a)
    b[key][0] += 1
    with pytest.raises(ValueError, match="alignment"):
        verify_alignment(a, b)


def test_alignment_allows_checkpoint_specific_features():
    a, b = synthetic_cache(), synthetic_cache()
    verify_alignment(a, b)
    b["features"] = b["features"][:, :-1]
    with pytest.raises(ValueError, match="schema/shape"):
        verify_alignment(a, b)


def test_frozen_base_mutation_or_training_is_rejected():
    model = torch.nn.Linear(2, 1).requires_grad_(False)
    row = {"checkpoint_state_sha256": canonical_state_dict_sha256(model.state_dict())}
    require_frozen(model, row)
    model.requires_grad_(True)
    with pytest.raises(AssertionError, match="modified"):
        require_frozen(model, row)
    model.requires_grad_(False)
    model.bias.add_(.1)
    with pytest.raises(AssertionError, match="modified"):
        require_frozen(model, row)


def test_replication_cannot_average_away_failed_seed():
    rows = [{"seed": s, "gate_pass": s != 62} for s in (42, 52, 62)]
    result = replication_decision(rows)
    assert result["new_seeds_passed"] == 1 and not result["replicated_on_both_new_seeds"]
    assert not result["all_three_pass"] and not result["fresh_training_authorized"]
    for invalid in (rows[:2], rows + [rows[0]], [rows[0], rows[0], rows[2]]):
        with pytest.raises(ValueError):
            replication_decision(invalid)
    rows = [{"seed": s, "gate_pass": s != 42} for s in (42, 52, 62)]
    assert replication_decision(rows)["replicated_on_both_new_seeds"]
    assert not replication_decision(rows)["all_three_pass"]


def test_paired_aggregate_uses_sample_std_and_keeps_seed_pairs():
    rows = []
    for i, seed in enumerate((42, 52, 62)):
        row = {"seed": seed, "gate_pass": True}
        for metric in ("qty_mae", "qty_rmse", "body_mae", "p99_mae", "time_nll", "joint_objective"):
            row[f"baseline_{metric}"] = 10 + i
            row[metric] = 9 + 2*i
        rows.append(row)
    result = paired_summary(rows)
    assert result["qty_mae_mean"] == 11
    assert result["qty_mae_std"] == 2
    assert result["paired_qty_mae_delta_mean"] == 0
    assert result["paired_qty_mae_delta_std"] == statistics.stdev([-1, 0, 1])


def test_held_out_cache_rejected_before_access():
    with pytest.raises(ValueError, match="forbidden"):
        read_reference({}, "test", {})
