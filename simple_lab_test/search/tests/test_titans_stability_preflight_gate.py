"""A diagnostic exit must not be mistaken for a complete matched experiment."""

import copy

import pytest

from paper.scripts.validate_titantpp_mac_stability_preflight import check_header


def evidence():
    contract = {"status": "complete", "completed_run_count": 1,
                "source_revision": "a"*40, "dataset": "insta_market_basket",
                "seeds": [42], "backbones": ["titantpp_titans_mac"],
                "epochs": 1, "batch_size": 128, "lr": .001,
                "lookback_weeks": 52, "max_seq_len": 64, "hidden_dim": 64,
                "lambda_log_qty": 1., "lambda_tail": 0., "grad_clip": 1.,
                "titans_memory_gradient_clip": 1., "partial_smoke": False,
                "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
                "time_head": {"mode": "legacy_clamped_rmtpp"}}
    summary = {"status": "success", "completed_epochs": 1,
               "source_revision": "a"*40, "source_revision_history": ["a"*40],
               "held_out_test_evaluated": False,
               "encoder_config": {"titans_memory_gradient_clip": 1.}}
    history = [{"epoch": 1, "train_all_finite": True, "val_joint_objective": 1.}]
    return contract, summary, history


def test_complete_evidence_accepted():
    check_header(*evidence(), seed=42, revision="a"*40)


@pytest.mark.parametrize("key,value", [("partial_smoke", True), ("status", "running"),
                                      ("titans_memory_gradient_clip", None),
                                      ("source_revision", "b"*40),
                                      ("held_out_test_evaluated", True)])
def test_partial_stale_or_mismatched_contract_rejected(key, value):
    contract, summary, history = evidence()
    contract[key] = value
    with pytest.raises(ValueError):
        check_header(contract, summary, history, seed=42, revision="a"*40)


def test_nonfinite_history_and_mixed_source_rejected():
    contract, summary, history = evidence()
    bad = copy.deepcopy(history)
    bad[0]["val_joint_objective"] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        check_header(contract, summary, bad, seed=42, revision="a"*40)
    summary["source_revision_history"].append("b"*40)
    with pytest.raises(ValueError, match="Mixed source"):
        check_header(contract, summary, history, seed=42, revision="a"*40)
