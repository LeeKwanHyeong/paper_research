"""CPU-only admission, frozen reuse, and generic-engine regression tests."""

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from paper.scripts import run_shared_history_mlp_control as r
from paper.scripts import run_titantpp_shared_history_cap as fixture


def data_for(dataset):
    value = fixture.fabricated_data()
    value["model"].update(
        lambda_log_qty=1.0, lambda_tail=0.0, time_head_lr_multiplier=1.0
    )
    value["loader"].update(
        batch_size=128,
        num_workers=0,
        drop_last=False,
        train_shuffle=True,
        validation_shuffle=False,
    )
    value.update(
        dataset_id=dataset,
        optimizer={
            "name": "AdamW",
            "lr": 0.001,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
        },
    )
    value["inherited_data_identity"] = {"populations": {"train": {}, "validation": {}}}
    return value


def contract(tmp_path):
    jobs = []
    for dataset in r.DATASETS:
        for seed in r.SEEDS:
            for arm in r.ARMS:
                item = {
                    "id": f"{dataset}__{arm}__seed{seed}",
                    "dataset": dataset,
                    "arm": arm,
                    "seed": seed,
                    "host": "5080" if seed == 42 else "5090",
                }
                if arm != "single_mlp":
                    underlying = r.UNDERLYING_ARMS[arm]
                    item["reuse"] = {
                        "underlying_arm": underlying,
                        "parent_job_id": f"{dataset}__{underlying}__seed{seed}",
                    }
                jobs.append(item)
    files = {"fixture.py": "0" * 64}
    return {
        "schema": r.SCHEMA,
        "campaign": r.CAMPAIGN,
        "scope": "execution",
        "execution_scope": "validation_only",
        "evaluation_scope": "validation_only",
        "held_out_test_authorized": False,
        "hosts": {
            h: {"root": str(tmp_path / h), "source_root": str(tmp_path / h / "source")}
            for h in ("5080", "5090")
        },
        "implementation_qualification": {"status": "passed", "training_allowed": True},
        "training": {
            "maximum_epochs": 300,
            "minimum_epochs": 40,
            "patience": 40,
            "batch_size": 128,
            "monitor": r.MONITOR,
        },
        "common_condition": {
            "time_input": "log1p",
            "loss_mode": "log_mse",
            "time_alpha": 1.0,
        },
        "source": {"files": files, "files_sha256": r._canonical(files)},
        "input_files": {str(i): "0" * 64 for i in range(9)},
        "datasets": [data_for(d) for d in r.DATASETS],
        "jobs": jobs,
    }


def test_exact_27_job_matrix_and_fixed_hosts(tmp_path):
    c = contract(tmp_path)
    assert len(c["jobs"]) == 27
    for job in c["jobs"]:
        selected, data, root, folder = r.select_job(c, job["host"], job["id"])
        assert selected == job and data["dataset_id"] == job["dataset"]
        assert folder == root / "run" / job["id"]
    assert sum(bool(j.get("reuse")) for j in c["jobs"]) == 18


@pytest.mark.parametrize(
    "fault",
    [
        "schema",
        "campaign",
        "execution_scope",
        "evaluation_scope",
        "test_auth",
        "qualification",
        "matrix",
        "duplicate",
        "host",
        "reuse_fresh",
        "reuse_arm",
        "reuse_parent",
        "training",
        "source",
        "input",
    ],
)
def test_contract_mutations_fail_closed(tmp_path, fault):
    c = contract(tmp_path)
    if fault in ("schema", "campaign", "execution_scope", "evaluation_scope"):
        c[fault] = "changed"
    elif fault == "test_auth":
        c["held_out_test_authorized"] = True
    elif fault == "qualification":
        c["implementation_qualification"]["training_allowed"] = False
    elif fault == "matrix":
        c["jobs"].pop()
    elif fault == "duplicate":
        c["jobs"][1]["id"] = c["jobs"][0]["id"]
    elif fault == "host":
        c["jobs"][0]["host"] = "5090"
    elif fault == "reuse_fresh":
        next(j for j in c["jobs"] if j["arm"] == "single_mlp")["reuse"] = {"bad": True}
    elif fault == "reuse_arm":
        c["jobs"][0]["reuse"]["underlying_arm"] = "full"
    elif fault == "reuse_parent":
        c["jobs"][0]["reuse"]["parent_job_id"] = "foreign"
    elif fault == "training":
        c["training"]["patience"] = 39
    elif fault == "source":
        c["source"]["files"]["fixture.py"] = "1" * 64
    else:
        c["input_files"].pop("0")
    with pytest.raises(ValueError):
        r._validate_contract(c)


@pytest.mark.parametrize("entry", ["frame", "evaluation", "stage", "cli"])
def test_test_is_rejected_before_dataset_or_checkpoint_loading(
    tmp_path, monkeypatch, entry
):
    def forbidden(*args, **kwargs):
        raise AssertionError("No held-out entry may reach loading")

    monkeypatch.setattr(r, "_load_model", forbidden)
    monkeypatch.setattr(r, "select_job", forbidden)
    if entry == "frame":
        with pytest.raises(ValueError, match="Test"):
            r._evaluation_frame({}, "test")
    elif entry == "evaluation":
        with pytest.raises(ValueError, match="Test"):
            r.evaluate_selected(
                {}, {}, {}, {}, tmp_path, None, "test", tmp_path / "prediction.gz"
            )
    elif entry == "stage":
        with pytest.raises(ValueError, match="Test"):
            r.run_stage({}, "5080", "job", "test")
    else:
        with pytest.raises(SystemExit) as error:
            r.main(
                [
                    "--contract",
                    str(tmp_path / "missing.json"),
                    "--contract-sha",
                    "x",
                    "--host",
                    "5080",
                    "--job",
                    "job",
                    "--stage",
                    "test",
                ]
            )
        assert error.value.code == 2
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("arm", r.ARMS)
def test_real_factory_common_joint_target_interface(arm):
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs

    model, metadata = r._build_model(fixture.fabricated_data(), arm)
    dt, qty, mask = fixture.fabricated_batch()
    value = target_outputs(model, dt, mask, qty, lambda_log_qty=1.0)
    assert isinstance(metadata, dict)
    assert torch.isfinite(value["joint_loss"]).all()
    assert torch.equal(value["joint_loss"], value["time_loss"] + value["log_qty_loss"])
    assert not any(name.startswith("lmm.") for name in model.state_dict())


def test_actual_generic_cpu_training_two_epochs_and_fresh_identity(tmp_path):
    from paper.scripts.count_aware_tpp_backbone import training

    args, frame, data, interface, quantity = fixture._engine_fixture(
        tmp_path, "titantpp_history_mlp_shared15_v1"
    )
    args.device = "cpu"
    args.epochs = args.min_epochs = args.early_stopping_patience = 2
    args.batch_size = 2
    args.time_alpha = 1.0
    args.max_val_batches = None
    source = "a" * 64
    interface["source_files_sha256"] = source
    with r._engine_hooks(data, "single_mlp"):
        summary, _, _ = training.train_one(
            args=args,
            frame=frame,
            quantity_contract=quantity,
            interface_meta=interface,
            backbone="single_mlp",
            quantity_variant=r.VARIANT,
            seed=42,
        )
    assert summary["completed_epochs"] == 2
    checkpoint = torch.load(
        summary["checkpoint_path"], map_location="cpu", weights_only=False
    )
    assert checkpoint["backbone"] == "single_mlp"
    assert checkpoint["interface_meta"]["source_files_sha256"] == source
    assert checkpoint["held_out_test_evaluated"] is False
    assert r._model_module().validate_checkpoint(checkpoint, "single_mlp") is True


def test_construction_hooks_restore_all_generic_globals_on_exception():
    from paper.scripts.count_aware_tpp_backbone import training, core

    names = (
        "build_model",
        "build_optimizer",
        "atomic_torch_save",
        "save_json",
        "BACKBONE_LABELS",
        "validate_training_identity",
        "target_outputs",
    )
    before = {name: getattr(training, name) for name in names}
    target = core.target_outputs
    with pytest.raises(RuntimeError, match="fixture interruption"):
        with r._engine_hooks(fixture.fabricated_data(), "single_mlp"):
            raise RuntimeError("fixture interruption")
    assert all(getattr(training, name) is value for name, value in before.items())
    assert core.target_outputs is target


def reuse_fixture(tmp_path, monkeypatch):
    c = contract(tmp_path)
    job = c["jobs"][0]
    root = Path(c["hosts"]["5080"]["root"])
    source = root / "source"
    source.mkdir(parents=True)
    p_contract, p_source = "1" * 64, "2" * 64
    payload = {
        "seed": 42,
        "best_epoch": 1,
        "backbone": "no_prototype",
        "checkpoint_monitor": r.MONITOR,
        "model_state_dict": {"weight": torch.tensor([3.0])},
        "interface_meta": {"source_files_sha256": p_source},
        "time_head_training_verification": {},
    }
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256

    state_sha = canonical_state_dict_sha256(payload["model_state_dict"])
    payload["model_state_sha256"] = state_sha
    torch.save(payload, source / "checkpoint.pt")
    checkpoint_sha = r.sha_file(source / "checkpoint.pt")
    history = {
        "history": [
            {"epoch": 1, "val_qty_rmse": 2.0},
            {"epoch": 2, "val_qty_rmse": 2.0},
        ]
    }
    (source / "history.json").write_text(json.dumps(history))
    reuse = {
        "selected_epoch": 1,
        "canonical_tensor_sha256": state_sha,
        "parent_job_id": job["reuse"]["parent_job_id"],
        "parent_contract_sha256": p_contract,
        "parent_source_files_sha256": p_source,
        "underlying_arm": "no_prototype",
    }
    for stage in ("train", "validation"):
        receipt = {
            "status": "complete",
            "stage": stage,
            "job_id": reuse["parent_job_id"],
            "host": "5080",
            "arm": "no_prototype",
            "dataset": job["dataset"],
            "seed": 42,
            "contract_sha256": p_contract,
            "source_files_sha256": p_source,
            "held_out_test_evaluated": False,
            "selected_epoch": 1,
            "checkpoint_sha256": checkpoint_sha,
            "state_sha256": state_sha,
            "training_origin": "fresh_fit",
            "validation_replay_passed": True,
            "individual_predictions_match": True,
            "time_head_training_verification": {},
        }
        (source / f"{stage}_receipt.json").write_text(json.dumps(receipt))
        for stream in ("stdout", "stderr"):
            (source / f"{stage}_{stream}.txt").write_text(stage + stream)
        exit_value = {
            "job_id": reuse["parent_job_id"],
            "host": "5080",
            "stage": stage,
            "contract_sha256": p_contract,
            "actual_os_returncode": 0,
            "timeout": False,
            "supervisor_error": None,
            "resource_incident": None,
            "stdout_sha256": r.sha_file(source / f"{stage}_stdout.txt"),
            "stderr_sha256": r.sha_file(source / f"{stage}_stderr.txt"),
        }
        (source / f"{stage}_exit.json").write_text(json.dumps(exit_value))
    with gzip.open(source / "validation_predictions.gz", "wt") as f:
        f.write("{}\n")
    filenames = {
        "checkpoint": "checkpoint.pt",
        "history": "history.json",
        "train_receipt": "train_receipt.json",
        "validation_receipt": "validation_receipt.json",
        "train_exit": "train_exit.json",
        "validation_exit": "validation_exit.json",
        "validation_predictions": "validation_predictions.gz",
        "train_stdout": "train_stdout.txt",
        "train_stderr": "train_stderr.txt",
        "validation_stdout": "validation_stdout.txt",
        "validation_stderr": "validation_stderr.txt",
    }
    reuse.update(
        {
            key: {"path": name, "sha256": r.sha_file(source / name)}
            for key, name in filenames.items()
        }
    )
    job["reuse"] = reuse
    monkeypatch.setattr(
        r,
        "_model_module",
        lambda: SimpleNamespace(validate_checkpoint=lambda value, arm: True),
    )
    monkeypatch.setattr(r.raw9, "verify_time_head_proof", lambda *args: None)
    monkeypatch.setattr(r.raw9, "verify_checkpoint_fit_proof", lambda *args: None)
    monkeypatch.setattr(r.raw9.prepared, "select_best_epoch", lambda rows: 1)
    return c, job, root, source


def test_parent_checkpoint_is_loaded_without_retagging(tmp_path, monkeypatch):
    c, job, root, source = reuse_fixture(tmp_path, monkeypatch)
    before = {p.name: r.sha_file(p) for p in source.iterdir()}
    payload = r._verify_reuse(c, c["datasets"][0], job, root)
    assert payload["backbone"] == "no_prototype"
    assert (
        payload["interface_meta"]["source_files_sha256"]
        == job["reuse"]["parent_source_files_sha256"]
    )
    assert before == {p.name: r.sha_file(p) for p in source.iterdir()}


@pytest.mark.parametrize(
    "fault",
    [
        "byte_tamper",
        "parent_job",
        "underlying_arm",
        "source_relabel",
        "nonzero_exit",
        "exit_stage",
        "selected_epoch",
        "receipt_source",
        "replay_false",
        "stream_digest",
    ],
)
def test_parent_reuse_rejects_changed_scientific_evidence(tmp_path, monkeypatch, fault):
    c, job, root, source = reuse_fixture(tmp_path, monkeypatch)
    if fault == "byte_tamper":
        (source / "history.json").write_text("tampered")
    elif fault in ("parent_job", "underlying_arm"):
        job["reuse"]["parent_job_id" if fault == "parent_job" else "underlying_arm"] = (
            "foreign"
        )
    elif fault == "source_relabel":
        value = torch.load(source / "checkpoint.pt", weights_only=False)
        value["interface_meta"]["source_files_sha256"] = c["source"]["files_sha256"]
        torch.save(value, source / "checkpoint.pt")
        new_sha = r.sha_file(source / "checkpoint.pt")
        job["reuse"]["checkpoint"]["sha256"] = new_sha
        for stage in ("train", "validation"):
            p = source / f"{stage}_receipt.json"
            v = json.loads(p.read_text())
            v["checkpoint_sha256"] = new_sha
            p.write_text(json.dumps(v))
            job["reuse"][stage + "_receipt"]["sha256"] = r.sha_file(p)
    else:
        key = (
            "validation_receipt"
            if fault in ("selected_epoch", "receipt_source", "replay_false")
            else "train_exit"
        )
        p = source / job["reuse"][key]["path"]
        v = json.loads(p.read_text())
        field, value = {
            "nonzero_exit": ("actual_os_returncode", 1),
            "exit_stage": ("stage", "reuse"),
            "selected_epoch": ("selected_epoch", 2),
            "receipt_source": ("source_files_sha256", "3" * 64),
            "replay_false": ("individual_predictions_match", False),
            "stream_digest": ("stdout_sha256", "4" * 64),
        }[fault]
        v[field] = value
        p.write_text(json.dumps(v))
        job["reuse"][key]["sha256"] = r.sha_file(p)
    with pytest.raises((ValueError, KeyError)):
        r._verify_reuse(c, c["datasets"][0], job, root)


def test_path_cannot_escape_isolated_root(tmp_path):
    with pytest.raises(ValueError):
        r._under(tmp_path, "../foreign")
    with pytest.raises(ValueError):
        r._under(tmp_path, "/absolute")


@pytest.mark.parametrize("outside", [False, True])
def test_symbolic_artifacts_rejected_even_when_hash_would_match(tmp_path, outside):
    source = (
        (tmp_path.parent / (tmp_path.name + "_outside"))
        if outside
        else tmp_path / "regular"
    )
    source.write_text("identical bytes")
    (tmp_path / "alias").symlink_to(source)
    with pytest.raises(ValueError, match="Symbolic"):
        r._under(tmp_path, "alias")


@pytest.mark.parametrize("arm", r.ARMS)
def test_synthetic_qualification_accepts_actual_finite_updates(tmp_path, arm):
    c = contract(tmp_path)
    job = next(job for job in c["jobs"] if job["arm"] == arm and job["host"] == "5080")
    result = r.synthetic_check(c, job["host"], job["id"])
    assert result["status"] == "qualified"
    assert result["real_data_loaded"] is False
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_synthetic_qualification_rejects_nonfinite_loss_before_update(
    tmp_path, monkeypatch, nonfinite
):
    from paper.scripts.count_aware_tpp_backbone import core

    c = contract(tmp_path)
    job = next(
        job for job in c["jobs"] if job["arm"] == "single_mlp" and job["host"] == "5080"
    )
    original_outputs = core.target_outputs
    calls = {"step": 0}

    def corrupt_loss(*args, **kwargs):
        output = original_outputs(*args, **kwargs)
        assert torch.isfinite(output["joint_loss"]).all()
        # Detached contamination leaves finite gradients, so a gradient-only gate misses it.
        output["joint_loss"] = output["joint_loss"] + torch.tensor(nonfinite)
        return output

    def forbidden_step(*args, **kwargs):
        calls["step"] += 1
        raise AssertionError("A nonfinite loss must not reach the optimizer")

    monkeypatch.setattr(core, "target_outputs", corrupt_loss)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden_step)
    with pytest.raises(ValueError, match="Nonfinite synthetic loss"):
        r.synthetic_check(c, job["host"], job["id"])
    assert calls["step"] == 0


def test_synthetic_qualification_rejects_non_time_head_nonfinite_gradient(
    tmp_path, monkeypatch
):
    c = contract(tmp_path)
    job = next(
        job for job in c["jobs"] if job["arm"] == "single_mlp" and job["host"] == "5080"
    )
    original_build = r._build_model
    observed = {}

    def corrupt_encoder_gradient(*args, **kwargs):
        model, metadata = original_build(*args, **kwargs)
        time_head_ids = {
            id(parameter) for _, parameter in model.time_head_named_parameters()
        }
        encoder = next(
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in time_head_ids
        )
        encoder.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
        observed["model"] = model
        return model, metadata

    monkeypatch.setattr(r, "_build_model", corrupt_encoder_gradient)
    with pytest.raises(ValueError, match="Nonfinite synthetic gradient"):
        r.synthetic_check(c, job["host"], job["id"])
    head_gradients = [
        parameter.grad
        for _, parameter in observed["model"].time_head_named_parameters()
    ]
    assert any(gradient is not None for gradient in head_gradients)
    assert all(
        gradient is None or torch.isfinite(gradient).all()
        for gradient in head_gradients
    )


def test_synthetic_qualification_rejects_nonfinite_parameter_after_update(
    tmp_path, monkeypatch
):
    c = contract(tmp_path)
    job = next(
        job for job in c["jobs"] if job["arm"] == "single_mlp" and job["host"] == "5080"
    )
    original_step = torch.optim.AdamW.step
    calls = {"step": 0}

    def corrupt_after_step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        calls["step"] += 1
        with torch.no_grad():
            optimizer.param_groups[0]["params"][0].fill_(float("nan"))
        return result

    monkeypatch.setattr(torch.optim.AdamW, "step", corrupt_after_step)
    with pytest.raises(ValueError, match="Nonfinite synthetic parameter"):
        r.synthetic_check(c, job["host"], job["id"])
    assert calls["step"] == 1
