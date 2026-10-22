"""Independent CPU-only oracles; no remote work or real dataset evaluation."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from models.TPPs.CountAwareHistoryMLPControl import SingleHistoryCorrection
from paper.scripts import manage_shared_history_mlp_control as manager


@pytest.fixture(autouse=True)
def one_cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def reference_inputs(hidden, valid, write):
    """Python history walk independent of production lag_source_indices."""
    result = torch.zeros((*hidden.shape[:2], 136), dtype=hidden.dtype)
    thresholds = (1, 2, 4, 8, 16, 32, 64, 128)
    for batch in range(len(hidden)):
        previous = None
        count = 0
        for index in range(hidden.shape[1]):
            if not valid[batch, index]:
                continue
            if not write[batch, index]:
                previous, count = None, 0
                continue
            count += 1
            result[batch, index, :64] = hidden[batch, index]
            if previous is not None:
                result[batch, index, 64:128] = hidden[batch, previous]
            result[batch, index, 128:] = torch.tensor([count > t for t in thresholds])
            previous = index
    return result


@pytest.mark.parametrize("seed", (42, 52, 62))
def test_random_padding_and_observation_reset_match_independent_history_walk(seed):
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn((3, 280, 64), generator=generator)
    valid = torch.rand((3, 280), generator=generator) > 0.15
    write = torch.ones_like(valid)
    write[0, 141] = False
    write[1, 17:20] = False
    valid[:, 0] = True
    hidden[~(valid & write)] = float("nan")
    correction = SingleHistoryCorrection()
    captured = []
    handle = correction.input_projection.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )
    output = correction(hidden, valid, memory_write_mask=write)
    handle.remove()
    assert torch.equal(captured[0], reference_inputs(hidden, valid, write))
    assert torch.isfinite(output).all() and torch.count_nonzero(output) == 0


def test_single_mlp_can_use_length_without_hidden_signal_and_does_not_claim_equivalence():
    correction = SingleHistoryCorrection()
    with torch.no_grad():
        correction.input_projection.weight.zero_()
        correction.output_projection.weight.zero_()
        correction.input_projection.weight[0, 128] = 1
        correction.output_projection.weight[0, 0] = 1
    hidden = torch.zeros(1, 3, 64)
    result = correction(hidden, torch.ones(1, 3, dtype=torch.bool))
    assert result[0, 0].count_nonzero() == 0
    assert result[0, 1, 0] > 0
    assert torch.equal(result[0, 1], result[0, 2])


def guard_contract():
    paused = {
        "pid": 123,
        "state": "T",
        "start_ticks": 456,
        "pgid": 123,
        "argv": ["old-worker"],
        "cwd": "/old",
    }
    return {
        "hosts": {
            "5080": {
                "boot_id": "boot-fixed",
                "absolute_host_deadline_unix": 100,
                "paused_processes": [paused],
            }
        }
    }


@pytest.mark.parametrize(
    "fault",
    (
        "deadline",
        "boot",
        "paused_missing",
        "paused_running",
        "paused_reused_pid",
        "unknown_gpu",
        "gpu_query_error",
    ),
)
def test_guard_rejects_environment_changes_before_work(fault):
    contract = guard_contract()
    identity = deepcopy(contract["hosts"]["5080"]["paused_processes"][0])
    if fault == "paused_running":
        identity["state"] = "R"
    if fault == "paused_reused_pid":
        identity["start_ticks"] += 1
    query = SimpleNamespace(stdout="123\n999\n" if fault == "unknown_gpu" else "123\n")
    with (
        patch.object(
            manager.time, "time", return_value=100 if fault == "deadline" else 99
        ),
        patch.object(
            Path, "read_text", return_value="other" if fault == "boot" else "boot-fixed"
        ),
        patch.object(
            manager,
            "proc_identity",
            return_value=None if fault == "paused_missing" else identity,
        ),
        patch.object(
            manager.subprocess,
            "run",
            return_value=query,
            side_effect=RuntimeError("query failed")
            if fault == "gpu_query_error"
            else None,
        ) as run,
    ):
        with pytest.raises(RuntimeError):
            manager.guard(contract, "5080")
        if fault in (
            "deadline",
            "boot",
            "paused_missing",
            "paused_running",
            "paused_reused_pid",
        ):
            run.assert_not_called()


def test_guard_accepts_only_pinned_paused_and_explicit_owned_worker():
    contract = guard_contract()
    identity = deepcopy(contract["hosts"]["5080"]["paused_processes"][0])
    with (
        patch.object(manager.time, "time", return_value=99),
        patch.object(Path, "read_text", return_value="boot-fixed"),
        patch.object(manager, "proc_identity", return_value=identity),
        patch.object(
            manager.subprocess, "run", return_value=SimpleNamespace(stdout="123\n789\n")
        ) as run,
    ):
        assert manager.guard(contract, "5080", active=(789,)) == {
            "compute_pids": [123, 789],
            "paused_preserved": 1,
        }
        assert run.call_args.kwargs["timeout"] == 15
        assert run.call_args.kwargs["check"] is True


@pytest.mark.parametrize("phase", ("test", "test_cost", "validation_test"))
def test_manager_cli_cannot_request_held_out_execution(phase):
    with patch.object(manager, "dispatch") as dispatch:
        with pytest.raises(SystemExit) as error:
            manager.main(["--mode", "dispatch", "--phase", phase])
        assert error.value.code == 2
        dispatch.assert_not_called()


def test_exclusive_evidence_writer_preserves_first_raw_record(tmp_path):
    evidence = tmp_path / "receipt.json"
    manager.write(evidence, {"status": "failed", "raw": "original\n"})
    before = evidence.read_bytes()
    with pytest.raises(FileExistsError):
        manager.write(evidence, {"status": "complete"})
    assert evidence.read_bytes() == before


@pytest.mark.parametrize("relative", ("../outside", "/tmp/outside", "a/../../outside"))
def test_evidence_traversal_is_rejected(tmp_path, relative):
    with pytest.raises(RuntimeError):
        manager.below(tmp_path, relative)


def test_evidence_symlink_escape_is_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "escape").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(RuntimeError):
        manager.below(root, "escape/result.json")


def test_synthetic_qualification_rejects_nonfinite_loss_even_with_finite_gradients(
    tmp_path,
):
    from paper.scripts import run_shared_history_mlp_control as runner
    from paper.scripts.count_aware_tpp_backbone import core
    from paper.tests.test_run_shared_history_mlp_control import contract

    value = contract(tmp_path)
    job = next(item for item in value["jobs"] if item["arm"] == "single_mlp")
    target = core.target_outputs

    def corrupt_loss(*args, **kwargs):
        output = target(*args, **kwargs)
        return {**output, "joint_loss": output["joint_loss"] + float("nan")}

    with patch.object(core, "target_outputs", corrupt_loss):
        with pytest.raises((RuntimeError, ValueError)):
            runner.synthetic_check(value, job["host"], job["id"], device="cpu")


@pytest.mark.parametrize(
    "member", ("../evidence", "/evidence", "a/../b", "a//b", "./a", "a/./b", "")
)
def test_controller_archive_member_rejects_noncanonical_names(member):
    from paper.scripts import advance_shared_history_mlp_control as controller

    with pytest.raises(RuntimeError):
        controller.safe_member(member)


@pytest.mark.parametrize("host", ("5080", "5090"))
@pytest.mark.parametrize("phase", ("fit_validation", "cost"))
def test_generated_remote_programs_compile_without_execution(host, phase):
    from paper.scripts import advance_shared_history_mlp_control as controller

    contract = {
        "hosts": {
            host: {"root": "/fixture/root with space", "tmux_binary": "/usr/bin/tmux"}
        }
    }
    terminal = controller.terminal_program(
        contract, host, phase, {"identity": {"pid": 123}}
    )
    compile(terminal, "<terminal read-only template>", "exec")
    compile(
        controller.cost_launch_program(contract, host), "<cost launch template>", "exec"
    )
    assert "subprocess" not in terminal
    assert "kill(" not in terminal and "killpg(" not in terminal


def test_controller_failure_preserves_initial_cause_and_never_retries(tmp_path):
    from paper.scripts import advance_shared_history_mlp_control as controller

    with (
        patch.object(controller, "REPORT", tmp_path),
        patch.object(controller, "transport") as transport,
    ):
        first = controller.preserve_failure(RuntimeError("first transport failed"))
        original = (tmp_path / "FAILURE.json").read_bytes()
        second = controller.preserve_failure(RuntimeError("later error"))
        assert second["error"] == first["error"]
        assert (tmp_path / "FAILURE.json").read_bytes() == original
        assert controller.cached_failure()["remote_calls"] == 0
        assert (
            controller.cached_failure()["training_failure_inferred_from_transport"]
            is False
        )
        transport.assert_not_called()


def test_controller_failure_cache_detects_corrupted_original(tmp_path):
    from paper.scripts import advance_shared_history_mlp_control as controller

    with patch.object(controller, "REPORT", tmp_path):
        controller.preserve_failure(RuntimeError("immutable first failure"))
        with (tmp_path / "FAILURE.json").open("ab") as stream:
            stream.write(b" ")
        with pytest.raises(RuntimeError, match="Cached failure changed"):
            controller.cached_failure()


def test_native_training_admission_requires_actual_dispatcher_exit(tmp_path):
    from paper.tests.test_manage_shared_history_mlp_control import contract, native_gate

    value = contract.__wrapped__()
    native_gate(value, tmp_path)
    terminal = tmp_path / "qualification.supervisor_exit.json"
    if terminal.exists():
        terminal.unlink()  # Synthetic tmp fixture only; never production evidence.
    with pytest.raises((RuntimeError, FileNotFoundError)):
        manager.verify_native_qualification(value, "5080", tmp_path)


@pytest.mark.parametrize(
    "fault",
    (
        "nonzero_dispatcher",
        "dispatcher_stream",
        "child_stream",
        "live_child",
        "live_dispatcher",
    ),
)
def test_native_training_admission_rejects_raw_exit_tampering_and_live_identities(
    tmp_path, fault
):
    import json
    from paper.tests.test_manage_shared_history_mlp_control import contract, native_gate

    value = contract.__wrapped__()
    complete = native_gate(value, tmp_path)
    live = None
    if fault == "nonzero_dispatcher":
        path = tmp_path / "qualification.supervisor_exit.json"
        receipt = json.loads(path.read_text())
        receipt["actual_os_returncode"] = 1
        path.write_text(json.dumps(receipt))
    elif fault == "dispatcher_stream":
        (tmp_path / "qualification.stdout.log").write_bytes(b"changed\n")
    elif fault == "child_stream":
        child = complete["actual_child_exits"][0]
        path = (
            tmp_path
            / "control"
            / (child["job_id"] + "." + child["stage"] + ".stdout.log")
        )
        path.write_bytes(b"changed\n")
    elif fault == "live_child":
        live = complete["actual_child_exits"][0]["identity"]
    else:
        live = manager.read(tmp_path / "qualification.claim.json")["identity"]
    with patch.object(
        manager,
        "proc_identity",
        side_effect=lambda pid: live if live and pid == live["pid"] else None,
    ):
        with pytest.raises(RuntimeError):
            manager.verify_native_qualification(value, "5080", tmp_path)


def analytical_records():
    records = []
    for dataset in ("fixture_a", "fixture_b", "fixture_c"):
        for arm, offset in (("shared15", 1), ("single_mlp", 0), ("encoder_only", 2)):
            for seed, value in zip((42, 52, 62), (1.0, 2.0, 3.0)):
                records.append(
                    {
                        "dataset": dataset,
                        "arm": arm,
                        "seed": seed,
                        "host": "5080" if seed == 42 else "5090",
                        "split": "validation",
                        "prediction_path": "/fixture/"
                        + dataset
                        + "/"
                        + arm
                        + str(seed),
                        "metrics": {
                            scope: {
                                metric: value + offset
                                for metric in ("qty_rmse", "qty_mae", "time_nll")
                            }
                            for scope in ("overall", "large")
                        },
                    }
                )
    return records


def test_comparator_ddof1_and_all_adverse_pairs_use_original_seed_not_pooled_metrics():
    from paper.scripts import compare_shared_history_mlp_control as comparator

    records = analytical_records()
    rows = comparator.summarize_records(records)
    assert len(rows) == 9
    row = next(
        r for r in rows if r["dataset"] == "fixture_a" and r["arm"] == "single_mlp"
    )
    assert row["metrics"]["overall"]["qty_rmse"] == {
        "mean": 2.0,
        "sample_sd_ddof1": 1.0,
        "n": 3,
    }
    pairs = comparator.paired_differences(records)
    assert len(pairs) == 162
    positive = [
        r
        for r in pairs
        if r["left_arm"] == "shared15" and r["right_arm"] == "single_mlp"
    ]
    assert len(positive) == 54 and all(
        r["left_worse"] and r["absolute"] == 1 for r in positive
    )
    assert all(
        r["left_prediction_path"] != r["right_prediction_path"]
        and r["split"] == "validation"
        for r in pairs
    )
    one = next(r for r in positive if r["seed"] == 42)
    assert one["relative"] == 1.0


def test_comparator_does_not_turn_empty_large_scope_into_zero():
    from paper.scripts import compare_shared_history_mlp_control as comparator

    records = analytical_records()
    for record in records:
        record["metrics"]["large"] = {key: None for key in record["metrics"]["large"]}
    summaries = comparator.summarize_records(records)
    assert all(
        row["metrics"]["large"]["qty_rmse"]
        == {"mean": None, "sample_sd_ddof1": None, "n": 0}
        for row in summaries
    )
    assert all(
        row["absolute"] is None and not row["left_worse"]
        for row in comparator.paired_differences(records)
        if row["scope"] == "large"
    )


@pytest.mark.parametrize(
    "fault", ("held_out_split", "missing_seed", "duplicate_condition")
)
def test_comparator_rejects_unmatched_or_held_out_record_sets(fault):
    from paper.scripts import compare_shared_history_mlp_control as comparator

    records = analytical_records()
    if fault == "held_out_split":
        records[0]["split"] = "test"
    elif fault == "missing_seed":
        records.pop()
    else:
        records[-1] = deepcopy(records[0])
    with pytest.raises(RuntimeError):
        comparator.summarize_records(records)


def test_comparator_refuses_cross_gpu_pairs():
    from paper.scripts import compare_shared_history_mlp_control as comparator

    records = analytical_records()
    records[0]["host"] = "5090"
    with pytest.raises(RuntimeError, match="different GPUs"):
        comparator.paired_differences(records)


def test_generated_deployment_checker_has_its_own_imports_and_checks_before_writes(
    tmp_path, monkeypatch
):
    import ast
    from paper.tests.test_history_mlp_initial_execution import (
        d,
        fixture,
        archive_fixture,
    )

    c, integrity = fixture(tmp_path, monkeypatch)
    program = d.deployment_program(c, "5080", integrity)
    tree = ast.parse(program)
    definitions = []
    for statement in tree.body:
        if not isinstance(statement, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            break
        definitions.append(statement)
    namespace = {}
    # Execute only imports and a pure archive checker, never the deployment body.
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), "<checker>", "exec"),
        namespace,
    )
    raw, digest, canonical, expected = archive_fixture()
    assert namespace["check_archive"](raw, digest, canonical) == expected
    assert program.index("c=check_archive(payload") < program.index("root.mkdir(")


@pytest.mark.parametrize("fault", ("missing_proof", "changed_stream", "unbound_extra"))
def test_local_native_pass_revalidates_actual_raw_inventory(
    tmp_path, monkeypatch, fault
):
    from paper.tests.test_history_mlp_initial_execution import d
    from paper.tests.test_manage_shared_history_mlp_control import contract, native_gate

    c = contract.__wrapped__()
    state = tmp_path / "state"
    snapshot = state / "5080" / "snapshot"
    snapshot.mkdir(parents=True)
    native_gate(c, snapshot)
    monkeypatch.setattr(d, "STATE", state)
    d.write(
        state / "5080/QUALIFICATION_PASS.json",
        {
            "status": "PASS",
            "host": "5080",
            "contract_sha256": d.canonical(c),
            "snapshot": "snapshot",
            "files": {
                str(p.relative_to(snapshot)): d.sha(p)
                for p in snapshot.rglob("*")
                if p.is_file()
            },
        },
    )
    if fault == "missing_proof":
        (snapshot / "qualification.supervisor_exit.json").unlink()
    elif fault == "changed_stream":
        (snapshot / "qualification.stdout.log").write_bytes(b"changed")
    else:
        (snapshot / "extra.txt").write_bytes(b"not in admitted inventory")
    with patch.object(d, "transport") as transport:
        with pytest.raises(RuntimeError):
            d.verify_pass(c, "5080")
        transport.assert_not_called()
