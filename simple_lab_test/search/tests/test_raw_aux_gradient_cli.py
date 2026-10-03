"""Boundary and proof checks for the CPU-only preparation entrypoint."""
import copy
import json

import pytest

from paper.scripts import run_raw_aux_gradient_comparison as runner


def test_design_change_is_rejected(tmp_path):
    design = runner.load_design()
    design["control"]["cap_ratio"] = .5
    changed = tmp_path / "design.json"
    changed.write_text(json.dumps(design))
    with pytest.raises(ValueError, match="design changed"):
        runner.load_design(changed)


def test_three_arm_controller_mapping_and_frozen_coefficients():
    configs = [runner.objective_for_arm(name) for name in runner.ARMS]
    assert configs[0][0].alpha == 0 and configs[0][1] is None
    assert configs[1][0] == configs[2][0]
    assert configs[1][1] is None and configs[2][1].cap_ratio == 1
    assert configs[1][0].alpha == runner.load_design()["losses"]["alpha"]


def test_source_closure_includes_new_evaluator_dependencies():
    sources = runner.source_hashes()
    for name in ("raw_aux_gradient_control.py", "raw_aux_gradient_acceptance.py",
                 "intermittent_cell_gradients.py", "intermittent_cell_statistics.py"):
        assert "paper/scripts/" + name in sources


def test_pair_audit_rejects_equal_step_with_different_batches():
    row = {"epoch": 1, "global_step": 2, "train_count": 8, "train_batches": 2,
           "validation_count": 8, "validation_batches": 2, "train_batch_order_sha256": "same"}
    summary = {"status": "complete", "initial_state_sha256": "init", "global_step": 2, "history": [row]}
    arms = {name: copy.deepcopy(summary) for name in runner.ARMS}
    assert runner.audit_pairs(arms)["passed"]
    arms[runner.ARMS[2]]["history"][0]["train_batch_order_sha256"] = "changed"
    with pytest.raises(ValueError, match="Paired exposure"):
        runner.audit_pairs(arms)


def test_research_execution_disabled_even_with_user_written_approval():
    sources = runner.source_hashes()
    identity = {"raw_aux_design_sha256": runner.DESIGN_SHA256,
                "source": {"files": sources, "files_sha256": runner.sha_json(sources)},
                "execution_approval": {"status": "approved_by_user"}}
    with pytest.raises(ValueError, match="Research execution is disabled"):
        runner.run_arm(arm_id=runner.ARMS[2], model=None, train_loader=None,
                       validation_loader=None, statistics=None, output_dir="unused",
                       identity=identity, synthetic=False)


def test_minimal_success_json_cannot_qualify_source(tmp_path):
    sources = runner.source_hashes()
    receipt = tmp_path / "fake.json"
    receipt.write_text(json.dumps({"passed": True, "qualifies_cuda": False,
        "design_sha256": runner.DESIGN_SHA256, "source_files_sha256": runner.sha_json(sources)}))
    with pytest.raises(ValueError, match="Incomplete CPU qualification"):
        runner.freeze_binding(receipt, tmp_path / "snapshot")
    assert not (tmp_path / "snapshot").exists()
