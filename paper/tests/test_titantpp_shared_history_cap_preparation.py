"""Preparation boundaries and the complete, immutable Cap comparison matrix."""
from copy import deepcopy
import subprocess
import sys

import pytest

from paper.scripts import prepare_titantpp_shared_history_cap as p


def fixture_contract():
    return {
        "schema": "titantpp_shared_history_cap_preparation_v1", "scope": "preparation_only",
        "execution_authorization": deepcopy(p.AUTHORIZATION), "common_condition": deepcopy(p.CONDITION),
        "arms": list(p.ARMS), "seeds": [42, 52, 62],
        "shared_width": 15, "independent_widths": [15, 5, 5, 5, 5, 5, 5, 5],
        "correction_parameter_count": 9600,
        "active_prefix_parameter_counts": [0] + [960 * (k + 2) for k in range(1, 9)],
        "planned_jobs": [{"dataset": d, "arm": a, "seed": s, **p.CONDITION}
                         for d in p.DATASETS for a in p.ARMS for s in (42, 52, 62)],
        "held_out_test_evaluated": False, "native_qualification_performed": False,
        "real_data_training_started": False, "current_training_modified": False,
    }


def test_preparation_matrix_and_prefix_budgets_match_independent_algebra():
    c = p.validate_contract(fixture_contract())
    assert len(c["planned_jobs"]) == 18
    independent = (15, 5, 5, 5, 5, 5, 5, 5)
    for k in range(9):
        shared = 128 * 15 + k * 64 * 15 if k else 0
        unshared = 192 * sum(independent[:k])
        assert shared == unshared == c["active_prefix_parameter_counts"][k]


@pytest.mark.parametrize("permission", p.AUTHORIZATION)
def test_preparation_does_not_grant_real_data_or_server_permission(permission):
    c = fixture_contract()
    c["execution_authorization"][permission] = True
    with pytest.raises(ValueError, match="not authorized"):
        p.validate_contract(c)


@pytest.mark.parametrize("field,value", [
    ("scope", "execution"), ("held_out_test_evaluated", True),
    ("native_qualification_performed", True), ("real_data_training_started", True),
    ("current_training_modified", True), ("shared_width", 12),
    ("independent_widths", [5] * 8), ("correction_parameter_count", 7680),
    ("active_prefix_parameter_counts", [9600] * 9),
])
def test_changed_design_or_execution_state_is_rejected(field, value):
    c = fixture_contract()
    c[field] = value
    with pytest.raises(ValueError):
        p.validate_contract(c)


@pytest.mark.parametrize("field,value", [("time_input", "raw"), ("loss_mode", "raw_mae"), ("time_alpha", 1.)])
def test_changed_common_condition_is_rejected(field, value):
    c = fixture_contract()
    c["common_condition"][field] = value
    with pytest.raises(ValueError):
        p.validate_contract(c)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "seed", "loss"])
def test_matrix_cannot_silently_drop_duplicate_or_change_a_seed(mutation):
    c = fixture_contract()
    if mutation == "missing":
        c["planned_jobs"].pop()
    elif mutation == "duplicate":
        c["planned_jobs"][-1] = deepcopy(c["planned_jobs"][0])
    elif mutation == "seed":
        c["planned_jobs"][0]["seed"] = 32
    else:
        c["planned_jobs"][0]["loss_mode"] = "raw_mse"
    with pytest.raises(ValueError):
        p.validate_contract(c)


def test_actual_parent_sources_inputs_references_and_cap_models_unchanged():
    parent, bindings, refs = p.verify_parent()
    assert len(parent["source"]["files"]) == 963 and len(bindings) == 6 and len(refs) == 9
    assert {d["dataset_id"]: d["cap_value"] for d in parent["datasets"]} == {
        "yellow_trip_hourly": 65., "raf_spare_parts": 10., "intermittent_frozen_5000": 20.}
    raf = next(d for d in parent["datasets"] if d["dataset_id"] == "raf_spare_parts")
    assert raf["loader"]["max_seq_len"] == 84 and 128 not in raf["potentially_active_lags"]
    assert all(r["retrained"] is False and r["reevaluated"] is False for r in refs)
    assert p.sha(p.SPEC) == p.SPEC_SHA


@pytest.mark.parametrize("mode", ["train", "deploy", "start", "test"])
def test_cli_has_no_execution_entrypoint(mode):
    result = subprocess.run([sys.executable, "-B", "-m", "paper.scripts.prepare_titantpp_shared_history_cap",
                             "--mode", mode], capture_output=True, text=True, cwd=p.ROOT)
    assert result.returncode == 2 and "invalid choice" in result.stderr
