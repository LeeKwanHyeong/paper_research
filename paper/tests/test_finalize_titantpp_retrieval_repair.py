"""Offline archive, failure preservation, and exact execution admission tests."""
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile
from unittest.mock import patch

import pytest

PROJECT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("retrieval_repair_finalizer_tested",
    PROJECT / "paper/scripts/finalize_titantpp_retrieval_repair.py")
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value))


def no_remote(*args, **kwargs):
    raise AssertionError("Pending/already-finalized must not make remote calls")


def fixture(tmp_path):
    parent_bundle = PROJECT / "search_artifacts/titantpp_contextual_retrieval_pilot_dual_20261007_v1"
    parent = f.read(parent_bundle / "execution_contract.json")
    old = f.read(parent_bundle / "hourly_monitor/20261007T050852850406Z/5090/snapshot.json")
    parent_path = tmp_path / "evidence/parent_5090_snapshot.json"
    write(parent_path, old)
    jobs = [j for j in parent["jobs"] if j["host"] == "5080" or j["arm"] not in parent["arms"][:2]]
    c = {"parent_contract": parent, "jobs": jobs,
         "reused_jobs": [j for j in parent["jobs"] if j not in jobs],
         "hosts": {h: {**s, "root": str(tmp_path / ("native_" + h))}
                   for h, s in parent["hosts"].items()},
         "limits": {"root_bytes": 8 * 1024 ** 3},
         "operation": {"files": {}},
         "qualification_bindings": {host: {"receipt_sha256": "d" * 64} for host in ("5090", "5080")},
         "resume": {f.FIXED: {"files": next(r for r in old["rows"] if r["job"]["id"] == f.FIXED)["checkpoint_receipt"]["files"]}},
         "parent_terminal_evidence": {"snapshot": str(parent_path), "snapshot_sha256": f.sha(parent_path)}}
    analyses = {}
    for host in ("5090", "5080"):
        rows = []
        for job in [j for j in jobs if j["host"] == host]:
            prefix = f.run_prefix(job)
            if job["id"] == f.FIXED:
                history = deepcopy(next(r for r in old["rows"] if r["job"]["id"] == f.FIXED)["history"]["history"])
                for epoch in range(149, 161):
                    history.append({**deepcopy(history[-1]), "epoch": epoch,
                                    "val_qty_rmse": 500.0, "val_qty_mae": 100.0, "val_time_nll": 2.0})
            else:
                history = [{"epoch": epoch, "train_all_finite": True,
                            "val_qty_rmse": 1.0 if epoch == 1 else 2.0,
                            "val_qty_mae": 0.5, "val_time_nll": 1.0} for epoch in range(1, 42)]
            best = min(history, key=lambda row: row["val_qty_rmse"])
            endpoint = {"status": "complete", "job": job, "best_epoch": best["epoch"],
                        "completed_epochs": len(history), "evaluation_scope": "validation_only",
                        "held_out_test_evaluated": False}
            for label, epoch, binary in (("selected", best["epoch"], f.BINARIES[0]),
                                         ("last", len(history), f.BINARIES[1])):
                row = history[epoch - 1]
                endpoint[label] = {"count": 8268, "quantity_boundaries": parent["datasets"][0]["quantity_boundaries_all_train_rows"],
                    "qty_rmse": row["val_qty_rmse"], "qty_mae": row["val_qty_mae"], "time_nll": row["val_time_nll"],
                    "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
                    "checkpoint_path": c["hosts"][host]["root"] + "/run/" + job["id"] + "/" + prefix + binary,
                    "tail": {"count": 79, "qty_rmse": 10.0, "qty_mae": 5.0, "time_nll": 0.01}}
            terminal = {"status": "complete", "scientific_success": True, "job": job,
                        "contract_sha256": f.canonical(c), "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
                        "files": {prefix + name: "a" * 64 for name in f.REQUIRED}}
            row = {"job": job, "claim": {"job": job}, "terminal": terminal, "SHA_verified": True,
                   "manifest_sha256": "b" * 64, "endpoint": endpoint, "history": {"history": history}}
            if job["id"] == f.FIXED:
                endpoint["resume_parent_epoch"] = 148
                row["resume_startup_gate"] = {"status": "passed", "epoch": 148, "best_epoch": 120,
                    "native_identity_validated": True, "optimizer_rng_loader_and_selected_state_validated": True,
                    "optimizer_updates": 0, "parent_checkpoint_sha256": c["resume"][f.FIXED]["files"]["last_epoch_state.pt"]}
                row["resume_copy_receipt"] = {"copied_raw_sha256": c["resume"][f.FIXED]["files"],
                    "last_epoch": 148, "best_epoch": 120, "original_files_modified": False, "optimizer_updates": 0}
            rows.append(row)
        qualification = deepcopy(old["qualification"])
        qualification["host"] = host
        qualification["runtime"]["gpu"]["uuid"] = c["hosts"][host]["gpu_uuid"]
        snap = {"host": host, "processes": [], "gpu_pids": "", "gpu_uuid_verified": True,
                "source_SHA_verified": True, "operation_SHA_verified": True, "parent_operation_SHA_verified": True,
                "actual_observed_utc": "2026-10-07T10:00:00+00:00", "rows": rows,
                "server_status": {"status": "complete", "completed": {r["job"]["id"]:
                    {"terminal_manifest_sha256": r["manifest_sha256"]} for r in rows}},
                "supervisor_exit": {"returncode": 0}, "qualification": qualification}
        analysis = {"terminal_verified": True, "host": host}
        analyses[host] = analysis
        folder = tmp_path / "hourly_monitor/final" / host
        write(folder / "snapshot.json", snap)
        write(folder / "analysis.json", analysis)
        write(tmp_path / ("hourly_monitor/terminal_cache_" + host + ".json"),
            {"snapshot": str(folder / "snapshot.json"), "snapshot_sha256": f.sha(folder / "snapshot.json"),
             "analysis_path": str(folder / "analysis.json"), "analysis_sha256": f.sha(folder / "analysis.json")})
    qs = importlib.util.spec_from_file_location("offline_parent_qualify",
        parent_bundle / "operation/qualify.py")
    qualification_module = importlib.util.module_from_spec(qs)
    qs.loader.exec_module(qualification_module)
    return c, SimpleNamespace(analyze=lambda c, s: analyses[s["host"]],
        profile_eligibility=qualification_module.profile_eligibility)


def alter_snapshot(tmp_path, host, mutate):
    cache_path = tmp_path / ("hourly_monitor/terminal_cache_" + host + ".json")
    cache = f.read(cache_path)
    snap = f.read(cache["snapshot"])
    mutate(snap)
    write(cache["snapshot"], snap)
    cache["snapshot_sha256"] = f.sha(cache["snapshot"])
    write(cache_path, cache)


def test_exact_ten_ids_reuse_two_and_preserve_failed_parent(tmp_path):
    c, observer = fixture(tmp_path)
    evidence = f.terminal_gate(c, tmp_path, observer)
    assert set(evidence) == {"repair_5090", "repair_5080", "parent_5090"}
    assert sum(len(v["jobs"]) for v in evidence.values()) == 10
    assert evidence["parent_5090"]["snapshot"]["supervisor_exit"]["returncode"] == 1
    assert f.read(c["parent_terminal_evidence"]["snapshot"])["failure"]["active_job"]["id"] == f.FIXED


def test_pending_has_zero_remote_calls(tmp_path):
    c, observer = fixture(tmp_path)
    (tmp_path / "hourly_monitor/terminal_cache_5080.json").unlink()
    with patch.object(f, "load_bundle", return_value=(c, observer, {})):
        result = f.run(tmp_path, runner=no_remote)
    assert result["status"] == "pending_verified_all10" and result["remote_calls"] == 0
    assert not (tmp_path / "analysis/finalize.claim").exists()


@pytest.mark.parametrize("mutator,match", [
    (lambda s: s["supervisor_exit"].update(returncode=1), "shutdown"),
    (lambda s: s["processes"].append("123 1 owned worker"), "shutdown"),
    (lambda s: s["rows"][0]["resume_startup_gate"].update(optimizer_rng_loader_and_selected_state_validated=False), "startup"),
    (lambda s: s["rows"][0]["history"]["history"][0].update(val_qty_rmse=12345), "prefix"),
    (lambda s: s["rows"][1]["endpoint"]["selected"].update(time_nll=99), "same-epoch"),
    (lambda s: s["rows"][1]["terminal"]["files"].update({"predictions_test.json": "c" * 64}), "held-out"),
])
def test_invalid_terminal_never_reaches_recovery(tmp_path, mutator, match):
    c, observer = fixture(tmp_path)
    alter_snapshot(tmp_path, "5090", mutator)
    with pytest.raises(ValueError, match=match):
        f.terminal_gate(c, tmp_path, observer)


def test_wrong_parent_failure_does_not_become_completed(tmp_path):
    c, observer = fixture(tmp_path)
    path = c["parent_terminal_evidence"]["snapshot"]
    snap = f.read(path)
    snap["failure"]["message"] = "NaN failure"
    snap["server_status"] = deepcopy(snap["failure"])
    write(path, snap)
    c["parent_terminal_evidence"]["snapshot_sha256"] = f.sha(path)
    # Refresh the modified contract SHA on the mock terminal records.
    for host in ("5090", "5080"):
        alter_snapshot(tmp_path, host, lambda s: [r["terminal"].update(contract_sha256=f.canonical(c)) for r in s["rows"]])
    with pytest.raises(ValueError, match="exact approved fixed timeout"):
        f.terminal_gate(c, tmp_path, observer)


def test_claim_and_failure_are_preserved_without_retry(tmp_path):
    c, observer = fixture(tmp_path)
    write(tmp_path / "analysis/finalize.claim", {"existing": "claim"})
    write(tmp_path / "analysis/failure.json", {"existing": "failure"})
    with patch.object(f, "load_bundle", return_value=(c, observer, {})):
        result = f.run(tmp_path, runner=no_remote)
    assert result["status"] == "manual_action_required_preserved_partial"
    assert f.read(tmp_path / "analysis/finalize.claim") == {"existing": "claim"}
    assert f.read(tmp_path / "analysis/failure.json") == {"existing": "failure"}


def test_first_recovery_failure_claim_prevents_later_retry(tmp_path):
    c, observer = fixture(tmp_path)
    with patch.object(f, "load_bundle", return_value=(c, observer, {})), \
         patch.object(f, "retrieve", side_effect=ValueError("Transport failed")):
        with pytest.raises(ValueError, match="Transport failed"):
            f.run(tmp_path, runner=no_remote)
        result = f.run(tmp_path, runner=no_remote)
    assert result["status"] == "manual_action_required_preserved_partial"
    assert (tmp_path / "analysis/finalize.claim").exists()
    assert f.read(tmp_path / "analysis/failure.json")["automatic_retry"] is False


def archive(tmp_path, members):
    path = tmp_path / "training_originals.tar.gz"
    with tarfile.open(path, "w:gz") as stream:
        for name, data, kind in members:
            item = tarfile.TarInfo(name)
            if kind == "symlink":
                item.type = tarfile.SYMTYPE
                item.linkname = "/outside"
                stream.addfile(item)
            else:
                item.size = len(data)
                stream.addfile(item, io.BytesIO(data))
    return path


@pytest.mark.parametrize("member", [("../escape", b"x", "regular"), ("link", b"", "symlink")])
def test_archive_cannot_escape_or_use_symlinks(tmp_path, member):
    path = archive(tmp_path, [member])
    with pytest.raises(ValueError, match="Unsafe"):
        f.extract_verified(path, tmp_path / "original", {"archive_sha256": f.sha(path)}, 1024)
    assert not (tmp_path / "original").exists()


def test_summary_preserves_ten_unique_rows_and_same_epoch_metrics(tmp_path):
    c, observer = fixture(tmp_path)
    evidence = f.terminal_gate(c, tmp_path, observer)
    originals = {key: {"binary_originals": []} for key in evidence}
    result = f.summarize(c, evidence, originals)
    assert len(result["selected_metrics"]) == len(result["owner_mapping"]) == 10
    assert result["scientific_seed_count"] == 1 and result["execution_count"] == 10
    assert result["owner_mapping"][f.FIXED] == "repair_5090"
    assert result["selection_changed"] is False and result["independent_evaluation"] is False
    assert result["held_out_test_evaluated"] is False


def test_remote_recovery_code_has_no_model_or_heldout_execution(tmp_path):
    c, observer = fixture(tmp_path)
    evidence = f.terminal_gate(c, tmp_path, observer)
    script = f.remote_script(c, "parent_5090", evidence["parent_5090"])
    compile(script, "remote_archive", "exec")
    assert "torch.load" not in script and "optimizer.step" not in script
    assert "'failure.json'" in script and "'EXIT_CODE': 1" not in script
    assert "EXIT_CODE=1" in script
