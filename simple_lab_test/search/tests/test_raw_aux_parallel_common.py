from copy import deepcopy

import pytest

from paper.scripts import raw_aux_parallel_common as common


def contract():
    hosts = {}
    for alias, arms in common.ASSIGNMENTS.items():
        root = "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/raw_aux_gradient_seed42_parallel_test"
        hosts[alias] = {
            "root": root, "source_root": root + "/source", "output_dir": root + "/run",
            "python": ("/opt/miniconda3" if alias == "5090" else "/home/leekwanhyeong/miniconda3") + "/envs/ai_env/bin/python3.12",
            "assigned_arms": list(arms), "environment": deepcopy(common.ENVIRONMENT),
            "gpu_uuid": {"5090": "GPU-c9adc246-ef66-7906-bc07-6e152fa9952f", "5080": "GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6"}[alias],
        }
    return {"schema": common.SCHEMA, "design_sha256": common.DESIGN_SHA256,
            "limits": deepcopy(common.LIMITS), "hosts": hosts, "dataset": common.data_entry(),
            "source": {"files": {}, "files_sha256": common.sha_json({})},
            "authorization": {"user_instruction": "5080이랑 병렬로 진행하자", "scope": "fresh_three_arm_parallel_training_after_two_host_qualification"},
            "policy": {"resume": False, "retry": False, "held_out": False, "coefficient_refit": False,
                       "additional_arms": False, "scheduler_changes": False, "runtime_mutation": False, "automatic_transfer": False}}


@pytest.mark.parametrize("mutation", [
    lambda c: c["hosts"]["5080"]["assigned_arms"].append("B_log_original"),
    lambda c: c["limits"].update(training_seconds=86400),
    lambda c: c["dataset"]["model"].update(hidden_dim=128),
    lambda c: c["policy"].update(resume=True),
    lambda c: c["hosts"]["5090"].update(source_root="/home/leekwanhyeong/workspace/paper_research"),
    lambda c: c["authorization"].update(user_instruction="old approval"),
])
def test_execution_scope_tampering_rejected(tmp_path, mutation):
    c = contract()
    p = tmp_path / "contract.json"
    common.write_json(p, c)
    assert common.read_contract(p, check_source=False) == c
    mutation(c)
    common.write_json(p, c)
    with pytest.raises(ValueError):
        common.read_contract(p, check_source=False)


def test_missing_or_changed_source_cannot_authorize_runtime(tmp_path):
    c = contract()
    p = tmp_path / "contract.json"
    common.write_json(p, c)
    with pytest.raises(ValueError):
        common.read_contract(p)


def test_minimal_passed_receipts_are_insufficient():
    c = contract()
    p = {"schema": "raw_aux_parallel_start_permit_v1", "execution_contract_sha256": common.sha_json(c),
         "authorized_arms": common.ASSIGNMENTS, "started_at_unix": 1000, "deadline_unix": 47800,
         "qualifications": {h: {"receipt": {"passed": True}, "sha256": common.sha_json({"passed": True})}
                            for h in common.ASSIGNMENTS}}
    p["permit_sha256"] = common.sha_json(p)
    with pytest.raises(ValueError):
        common.validate_permit(c, p, "5090", now=1001)


def test_latest_scope_does_not_modify_historical_design_authorization():
    assert common.load_design()["authorization"]["new_training"] is False
    assert common.data_entry()["expected_global_steps"] == 369240
