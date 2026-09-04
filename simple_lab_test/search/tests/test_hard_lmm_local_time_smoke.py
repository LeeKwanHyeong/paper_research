"""Fail-closed smoke orchestration and production entry-point checks."""

import argparse
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from models.TPPs.CountAwareFactory import build_count_aware_model
from simple_lab_test.search.common.runner import canonical_state_dict_sha256

from paper.scripts import run_hard_lmm_local_time_smoke as smoke
from paper.scripts.count_aware_tpp_backbone.training import train_one


def test_command_is_full_e1_only_and_uses_the_dedicated_role(tmp_path):
    registry = smoke.read(smoke.ROOT / smoke.read(smoke.CONTRACT)["baseline"]["registry"])
    for row in registry["datasets"]:
        if row["dataset"] not in smoke.DATASETS:
            with pytest.raises(ValueError, match="Only Taxi/RAF"):
                smoke.command(row, tmp_path, tmp_path / "output", "a" * 40)
            continue
        cmd = smoke.command(row, tmp_path, tmp_path / "output", "a" * 40)
        for key, value in {"epochs": "1", "min-epochs": "1", "early-stopping-patience": "1",
                           "seeds": "42", "backbones": smoke.BACKBONE, "model-role": smoke.ROLE,
                           "batch-size": "128", "lr": "0.001", "lambda-tail": "0.0",
                           "lookback-weeks": str(row["lookback"]), "max-seq-len": str(row["max_seq_len"])}.items():
            assert cmd[cmd.index(f"--{key}") + 1] == value
        assert not any(flag in cmd for flag in ("--force-rerun", "--max-train-batches", "--max-val-batches"))


def test_existing_artifact_never_written_or_resumed(tmp_path, monkeypatch):
    sentinel = tmp_path / "status.json"
    sentinel.write_text('{"status":"complete"}')
    run = Mock()
    monkeypatch.setattr(smoke, "run_logged", run)
    with pytest.raises(FileExistsError):
        smoke.execute(argparse.Namespace(output_root=tmp_path, project_root=tmp_path, source_revision="a" * 40))
    assert sentinel.read_text() == '{"status":"complete"}'
    run.assert_not_called()


@pytest.mark.parametrize("boundary", ["source", "preflight", "cuda", "training"])
def test_failure_records_failed_and_does_not_advance(tmp_path, monkeypatch, boundary):
    out = tmp_path / "fresh"
    monkeypatch.setattr(smoke, "verify_source", lambda _: "verified")
    monkeypatch.setattr(smoke, "baseline", lambda *_: ({}, {}))
    monkeypatch.setattr(smoke, "reference_route_parity", lambda *_: {"status": "passed"})
    monkeypatch.setattr(smoke, "preflight", lambda *_: {"free_vram_mib": 16000})
    original_read = smoke.read
    monkeypatch.setattr(smoke, "read", lambda p: {} if Path(p).name == "source_manifest.json" else original_read(p))
    run = Mock()
    monkeypatch.setattr(smoke, "run_logged", run)
    error = RuntimeError(f"synthetic {boundary} failure")
    if boundary == "source":
        monkeypatch.setattr(smoke, "verify_source", Mock(side_effect=error))
    elif boundary == "preflight":
        monkeypatch.setattr(smoke, "preflight", Mock(side_effect=error))
    elif boundary == "cuda":
        run.side_effect = error
    else:
        run.side_effect = [None, error]
    with pytest.raises(RuntimeError, match="synthetic"):
        smoke.execute(argparse.Namespace(output_root=out, project_root=tmp_path, source_revision="a" * 40))
    status = smoke.read(out / "status.json")
    assert status["status"] == "failed" and status["completed"] == []
    assert boundary in status["error"]
    assert run.call_count == {"source": 0, "preflight": 0, "cuda": 1, "training": 2}[boundary]
    assert not list(out.glob("**/*test_summary*"))


def test_cache_cannot_silently_relabel_an_original_checkpoint(tmp_path):
    run = tmp_path / smoke.SUMMARY.parent
    run.mkdir(parents=True)
    smoke.save(run / "summary.json", {"backbone": "titantpp", "variant": smoke.VARIANT})
    (run / "best_val_joint_objective_model.pt").touch()
    with pytest.raises(ValueError, match="routing metadata"):
        train_one(args=argparse.Namespace(output_dir=tmp_path, force_rerun=False),
                  frame=None, quantity_contract={}, interface_meta={},
                  backbone=smoke.BACKBONE, quantity_variant=smoke.VARIANT, seed=42)


def test_candidate_entrypoint_filters_before_materializing():
    # The filtering function itself has a synthetic held-out regression test.
    source = (smoke.ROOT / "paper/scripts/run_count_aware_tpp_backbone_control.py").read_text()
    assert "{MODEL_ROLE_WEIGHTED_STATIC, MODEL_ROLE_HARD_LOCAL_TIME}" in source
    assert "raw_frame = load_train_validation_frame(args.data)" in source


@pytest.mark.parametrize("violation", [None, "nonfinite", "scope", "optimizer", "route", "digest", "history"])
@pytest.mark.parametrize("screening", [False, True])
def test_artifact_audit_accepts_complete_e1_and_rejects_drift(tmp_path, violation, screening):
    model, meta = build_count_aware_model(smoke.BACKBONE, hidden_dim=16, train_log_mean=1.5,
        max_seq_len=8, quantity_variant=smoke.VARIANT, lambda_tail=0., time_head_mode="legacy_clamped_rmtpp")
    revision = "a" * 40
    count = sum(p.numel() for p in model.parameters())
    row = dict(dataset="yellow_trip_hourly", data_sha256="data", split_manifest_sha256="split", lookback=168, max_seq_len=8)
    spec = dict(train_targets=3, validation_targets=2, parameter_count=count)
    time = dict(mode="legacy_clamped_rmtpp", time_scale=3., time_w_max=10. / 3.,
                time_intercept_limit=30., time_head_lr_multiplier=1., train_time_statistics={"target_count": 3})
    launch = dict(status="complete", completed_run_count=1, backbones=[smoke.BACKBONE], seeds=[42],
        model_role=smoke.ROLE, epochs=1, batch_size=128, lr=.001, hidden_dim=64, grad_clip=1.,
        lambda_log_qty=1., lambda_tail=0., data_sha256="data", split_manifest_sha256="split",
        source_revision=revision, partial_smoke=False, split_rows={"train": 3, "validation": 2},
        evaluation_scope="validation_only", held_out_test_evaluated=False, time_head=time,
        early_stopping=dict(min_epochs=1, patience=1, monitor="validation_joint_objective"), lookback_weeks=168, max_seq_len=8)
    summary = dict(status="success", backbone=smoke.BACKBONE, variant=smoke.VARIANT, epochs=1, best_epoch=1,
        source_revision=revision, source_revision_history=[revision], evaluation_scope="validation_only",
        held_out_test_evaluated=False, quantity_rows=[dict(stratum="le_p50", count=2, qty_mae=1.)],
        history_rows=[dict(stratum="short", count=2)], parameter_count=count,
        interface_meta=dict(train_target_mean=1.5, train_target_std=.5), encoder_config=meta,
        checkpoint_state_sha256=canonical_state_dict_sha256(model.state_dict()))
    checkpoint = dict(backbone=smoke.BACKBONE, variant=smoke.VARIANT, encoder_config=meta,
                      source_revision=revision, model_state_dict=model.state_dict())
    history = [{"epoch": 1}]
    if screening:
        launch.update(epochs=300, early_stopping=dict(min_epochs=40, patience=40, monitor="validation_joint_objective"))
        summary.update(epochs=300, completed_epochs=41)
        history = [{"epoch": epoch, "val_joint_objective": 2., "train_event_count": 3} for epoch in range(1, 42)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001).state_dict()
    if violation == "nonfinite":
        summary["quantity_rows"][0]["qty_mae"] = float("nan")
    if violation == "scope":
        launch["held_out_test_evaluated"] = True
    if violation == "optimizer":
        optimizer["param_groups"][0]["weight_decay"] = 0.
    if violation == "route":
        meta["time_memory_route"] = "hard_local_memory_matcher"
    if violation == "digest":
        summary["checkpoint_state_sha256"] = "wrong"
    run = tmp_path / smoke.SUMMARY.parent
    run.mkdir(parents=True)
    smoke.save(tmp_path / "launch_contract.json", launch)
    # A malformed external artifact can contain NaN despite our writer rejecting it.
    import json
    (run / "summary.json").write_text(json.dumps(summary))
    smoke.save(run / "history.json", history if violation == "history" else {"history": history})
    torch.save(checkpoint, run / "best_val_joint_objective_model.pt")
    last = checkpoint | {"optimizer_state_dict": optimizer, "best_state_dict": model.state_dict()}
    if screening:
        last["model_state_dict"] = {key: value + .01 for key, value in model.state_dict().items()}
    torch.save(last, run / "last_epoch_state.pt")
    if violation is None:
        assert smoke.audit_run(tmp_path, row, summary, revision, spec,
                               screening=screening, write_audit=False)["status"] == "passed"
        assert not (tmp_path / "audit.json").exists()
        assert smoke.audit_run(tmp_path, row, summary, revision, spec, screening=screening)["status"] == "passed"
        if screening:
            summary["best_epoch"] = 2
            smoke.save(run / "summary.json", summary)
            with pytest.raises(ValueError, match="checkpoint selection"):
                smoke.audit_run(tmp_path, row, summary, revision, spec, screening=True, write_audit=False)
            summary["best_epoch"] = 1
            summary["completed_epochs"] = 40
            smoke.save(run / "summary.json", summary)
            smoke.save(run / "history.json", {"history": history[:-1]})
            with pytest.raises(ValueError, match="Premature completion"):
                smoke.audit_run(tmp_path, row, summary, revision, spec, screening=True, write_audit=False)
    else:
        with pytest.raises(ValueError):
            smoke.audit_run(tmp_path, row, summary, revision, spec, screening=screening)
        assert not (tmp_path / "audit.json").exists()
