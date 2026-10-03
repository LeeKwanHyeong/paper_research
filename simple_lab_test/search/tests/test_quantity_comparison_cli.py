import copy
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

import pytest

from paper.scripts import run_quantity_comparison as cli


def contract_fixture(tmp_path):
    datasets = []
    for name, count, steps in zip(cli.DATASETS, (393824, 38393, 1991192), (369240, 36000, 1866840), strict=True):
        datasets.append({"dataset_id": name, "loader": {"batch_size": 128},
                         "statistics": {"train_log_mean": 1.0, "train_log_std": 1.0, "raw_scale": 1.0},
                         "inherited_data_identity": {"populations": {"train": {"target_count": count}}},
                         "expected_global_steps": steps})
    return {"schema": cli.SCHEMA, "status": "frozen_pending_explicit_approval", "cases": list(cli.CASES),
            "datasets": datasets, "seed": 42, "epochs": 120, "total_optimizer_steps": 9088320,
            "source": {"files": {}, "files_sha256": cli.sha_json({})},
            "limits": {"max_wall_seconds": 172800, "qualification_seconds": 900,
                       "max_concurrent_gpu_jobs": 1, "output_bytes": 2 * 1024**3,
                       "minimum_free_bytes": 5 * 1024**3, "per_file_bytes": 64 * 1024**2},
            "policy": {"automatic_retry": False, "automatic_resume": False, "held_out": False, "budget_extension": False},
            "execution": {"output_dir": str(tmp_path / "run")}}


def test_contract_budget_and_source_drift_rejected(tmp_path):
    contract = contract_fixture(tmp_path)
    assert cli.validate_contract(contract, check_source=False)["gpu_started"] is False
    for alter in (
        lambda c: c.update(epochs=121),
        lambda c: c["cases"].reverse(),
        lambda c: c["limits"].update(max_wall_seconds=259200),
        lambda c: c["policy"].update(automatic_resume=True),
        lambda c: c["datasets"][0].update(expected_global_steps=1),
        lambda c: c["datasets"][0]["statistics"].update(raw_scale=float("nan")),
    ):
        changed = copy.deepcopy(contract)
        alter(changed)
        with pytest.raises(ValueError):
            cli.validate_contract(changed, check_source=False)
    with pytest.raises(ValueError, match="Source closure"):
        cli.validate_contract(contract)


def test_approval_binds_scope_and_exact_execution_contract(tmp_path):
    contract = contract_fixture(tmp_path)
    approval = {"status": "approved_by_user", "execution_contract_sha256": cli.sha_json(contract),
                "scope": "synthetic_cuda_qualification_and_12_fresh_joint_arms", "user_instruction": "synthetic example only"}
    cli.validate_approval(contract, approval)
    for key, value in (("status", "pending"), ("scope", "old_jqt_approval"),
                       ("execution_contract_sha256", "0" * 64), ("user_instruction", "")):
        changed = {**approval, key: value}
        with pytest.raises(ValueError):
            cli.validate_approval(contract, changed)


def test_budget_clock_rollback_and_storage_reserve(tmp_path):
    contract = contract_fixture(tmp_path)
    contract["limits"]["max_wall_seconds"] = 10
    wall, mono = [100.0], [5.0]
    budget = cli.Budget(contract, 100.0, clock=lambda: wall[0], monotonic=lambda: mono[0])
    budget("train_batch")
    wall[0], mono[0] = 90.0, 15.0
    with pytest.raises(TimeoutError):
        budget("validation_batch")
    contract["limits"]["output_bytes"] = 64 * 1024**2 + 8
    output = Path(contract["execution"]["output_dir"])
    output.mkdir()
    (output / "sample.bin").write_bytes(b"12345678")
    with pytest.raises(RuntimeError, match="storage"):
        cli.Budget(contract, time.time())("before_epoch_commit")


def test_paired_exposure_and_replay_checks_are_not_metric_comparisons():
    row = {key: 1 for key in ("epoch", "global_step", "train_count", "train_batches",
                             "validation_count", "validation_batches")}
    row["train_batch_order_sha256"] = "same"
    summary = {"status": "complete", "initial_state_sha256": "initial", "global_step": 1,
               "history": [row], "last_state_sha256": "last", "selectors": {}}
    summaries = [copy.deepcopy(summary) for _ in range(4)]
    assert cli.audit_pairs(summaries)["status"] == "passed"
    summaries[3]["history"][0]["train_batch_order_sha256"] = "different"
    with pytest.raises(ValueError, match="Paired exposure"):
        cli.audit_pairs(summaries)
    with pytest.raises(ValueError, match="replay mismatch"):
        cli.compare_replay(summary, {**summary, "last_state_sha256": "different"})


@pytest.mark.parametrize("mode", ["_suite", "execute"])
def test_worker_cannot_use_cpu_flag_to_start_research_cuda_suite(tmp_path, mode):
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract_fixture(tmp_path)))
    with pytest.raises(ValueError, match="requires explicit CUDA"):
        cli.main([mode, "--contract", str(path), "--started", str(time.time()), "--device", "cpu"])


def test_owned_group_cleanup_after_group_leader_exits():
    script = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); print(p.pid,flush=True)"
    process = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True, start_new_session=True)
    child = int(process.stdout.readline())
    process.wait(timeout=10)
    try:
        os.kill(child, 0)
        cli.stop_owned_process_group(process)
        # The descendant holds the pipe open after its parent exits. EOF
        # proves cleanup without requiring system-wide process inspection.
        readable, _, _ = select.select([process.stdout], [], [], 2)
        assert readable, "Owned descendant still holds the pipe open"
        assert process.stdout.read() == ""
        assert process.returncode == 0
    finally:
        try:
            os.kill(child, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_disk_scan_tolerates_only_disappearing_atomic_temporaries(tmp_path, monkeypatch):
    contract = contract_fixture(tmp_path)

    class Temporary:
        def stat(self):
            raise FileNotFoundError("renamed")

    monkeypatch.setattr(Path, "rglob", lambda *args: iter([Temporary()]))
    cli.Budget(contract, time.time())("before_epoch_commit")

    class Unreadable:
        def stat(self):
            raise PermissionError("unreadable")

    monkeypatch.setattr(Path, "rglob", lambda *args: iter([Unreadable()]))
    with pytest.raises(PermissionError):
        cli.Budget(contract, time.time())("before_epoch_commit")


def test_metadata_validation_does_not_import_or_start_runtime(tmp_path, monkeypatch, capsys):
    contract = contract_fixture(tmp_path)
    monkeypatch.setattr(cli, "source_hashes", lambda: {})
    monkeypatch.setattr(cli, "configured_runtime", lambda *args: pytest.fail("Metadata must not initialize runtime"))
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    assert cli.main(["validate", "--contract", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["gpu_started"] is False


def test_recomputed_roundoff_preserves_frozen_B_initialization():
    frozen = {"train_log_mean": 1.821123902075507, "train_log_std": 1.2853917394914367,
              "raw_scale": 36.878211774114746, "all_train_quantity_sha256": "same"}
    metadata = {"train_log_mean": 1.8211239020755068, "train_log_std": 1.2853917394914365,
                "raw_scale": frozen["raw_scale"], "all_train_rows": {"quantity_sha256": "same"}}
    bound = cli.bind_frozen_statistics({"statistics": frozen}, metadata)
    assert bound["train_log_mean"] == frozen["train_log_mean"]
    assert bound["train_log_std"] == frozen["train_log_std"]
    assert metadata["train_log_mean"] != frozen["train_log_mean"]
    with pytest.raises(ValueError, match="Train statistic drift"):
        cli.bind_frozen_statistics({"statistics": frozen}, {**metadata, "raw_scale": frozen["raw_scale"] + 1e-8})
    with pytest.raises(ValueError, match="quantity identity drift"):
        cli.bind_frozen_statistics({"statistics": frozen}, {**metadata, "all_train_rows": {"quantity_sha256": "other"}})


def test_runtime_rejects_validator_from_an_ancestor_checkout(tmp_path, monkeypatch):
    from paper.scripts import run_time_quantity_diagnostic as diagnostic
    from paper.scripts import quantity_comparison_runtime as runtime
    monkeypatch.setattr(diagnostic, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runtime, "configure_runtime", lambda *args, **kwargs: pytest.fail("Wrong source must fail before runtime setup"))
    with pytest.raises(ValueError, match="validator loaded from outside"):
        cli.configured_runtime({}, device="cpu")


@pytest.mark.parametrize("name", ["../old_run", "quantity_comparison_seed42_v2/../v1", "quantity_comparison_seed42_v0"])
def test_freeze_rejects_unsafe_run_identity_before_accessing_files(tmp_path, name):
    with pytest.raises(ValueError, match="Run name must be"):
        cli.freeze_contract(tmp_path / "missing.json", tmp_path / "stats.json",
                            tmp_path / "contract.json", run_name=name)
    assert list(tmp_path.iterdir()) == []
