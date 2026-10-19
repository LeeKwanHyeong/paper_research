"""Only root/entrypoint wiring changes after the pre-fit process query error."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from paper.scripts import manage_titantpp_shared_history_cap_execution as v1
from paper.scripts import manage_titantpp_shared_history_cap_execution_v3 as v2
from paper.scripts import compare_titantpp_shared_history_cap_execution as comparison1
from paper.scripts import compare_titantpp_shared_history_cap_execution_v3 as comparison2


def contracts():
    p = v1.preparation
    prepared = v1.read(p.BUNDLE / "preparation_contract.json")
    parent = v1.read(p.PARENT / "execution_contract.json")
    observed = v1.read(v1.REPORT / "INITIAL_5080.json")
    source = deepcopy(prepared["source"]);source["base_git_revision"] = source["git_revision"]
    return v1.build_contract(prepared, parent, observed, source), v2.build_contract(prepared, parent, observed, source)


def test_isolated_new_root_and_same_scientific_contract():
    a, b = contracts()
    assert len(v2.runtime.validate_contract(b, "5080")) == 18
    for key in ("schema", "scope", "source", "architecture", "design_sha256", "common_condition", "jobs", "datasets",
                "training", "input_files", "baseline_validation_references", "limits", "execution_authorization"):
        assert a[key] == b[key], key
    h1, h2 = a["hosts"]["5080"], b["hosts"]["5080"]
    assert h1["root"] != h2["root"] and h2["root"].endswith("_execution_v3_5080")
    assert h1["tmux"] != h2["tmux"] and h2["tmux"].endswith("_v3")
    assert h1["paused_processes"] == h2["paused_processes"]
    assert h1["absolute_host_deadline_unix"] == h2["absolute_host_deadline_unix"]
    assert v1.BUNDLE.name == "execution_v1" and v2.BUNDLE.name == "execution_v3"
    assert v2.MODULE.endswith("execution_v3_runtime")
    assert len(v2._adapter.NEW_SOURCE) == 7


def test_corrected_collection_and_finite_completion_routes():
    _, c = contracts()
    program = v2.collection_program(c, "5080")
    compile(program, "<collector>", "exec")
    assert "import titantpp_shared_history_cap_execution_v3_runtime as r" in program
    assert "paper.scripts.compare_titantpp_shared_history_cap_execution_v3" in v2.finish.__code__.co_names
    assert "paper.scripts.manage_titantpp_shared_history_cap_execution_v3" in v2.start_finisher.__code__.co_consts
    assert comparison1.aggregate is not comparison2.aggregate
    assert "paper.scripts" in comparison2.main.__code__.co_names


def test_new_finite_process_command_without_changing_old_process(tmp_path):
    report = tmp_path / "new";report.mkdir()
    class Process:pid=123456
    with patch.object(v2._adapter,"REPORT",report), patch.object(v2._adapter,"verify_local",return_value={}), \
         patch.object(v2._adapter.subprocess,"Popen",return_value=Process()) as run:
        result=v2.start_finisher()
    assert result["command"][-3:] == ["paper.scripts.manage_titantpp_shared_history_cap_execution_v3","--mode","finish"]
    assert result["recurring_automation"] is False and result["pid"] == 123456
    assert run.call_count == 1
