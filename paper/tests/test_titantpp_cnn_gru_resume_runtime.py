"""Offline operational continuation guards; no SSH, GPU or research data."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/titantpp_cnn_gru_resume_runtime.py"
spec = importlib.util.spec_from_file_location("resume_runtime_tested", SCRIPT)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def mappings():
    return {k: {"origin_path": "/a100/" + k, "target_path": "/5080/" + k,
        "sha256": k + "sha", "independently_verified": True} for k in ("data", "split_manifest")}


def identity():
    return {"arguments": {"data": "/a100/data", "data_sha256": "datasha", "split_manifest": "/a100/split_manifest",
        "split_manifest_sha256": "split_manifestsha", "execution_role": "fresh_cnn_gru_validation", "device": "cuda:0"}}


def prefix_fixture(epoch=2):
    history = [{"epoch": i, "train_event_count": 7, "train_batch_count": 1, "val_qty_rmse": float(i),
        "val_qty_mae": .4, "val_time_nll": .2} for i in range(1, epoch + 1)]
    exposure = {split: [{"epoch": i, "split": split, "count": count, "batches": 1,
        "batch_order_sha256": "a" * 64} for i in range(1, epoch + 1)] for split, count in (("train", 7), ("validation", 3))}
    payload = {"epoch": epoch, "history": history, "best_epoch": 1, "model_state_sha256": "current",
        "best_state_sha256": "best", "model_state_dict": "current", "best_state_dict": "best", "resume_identity": identity(),
        "optimizer_state_dict": {"state": {0: {"step": epoch}}}, "evaluation_scope": "validation_only", "held_out_test_evaluated": False}
    artifact = {"source_epoch": epoch, "source_best_epoch": 1, "source_best_state_sha256": "best",
        "source_current_state_sha256": "current", "source_resume_identity_sha256": r.canonical(payload["resume_identity"])}
    timing = {"epochs": [{"epoch": i, "elapsed_seconds": 2.} for i in range(1, epoch + 1)]}
    data = {"inherited_data_identity": {"populations": {"train": {"target_count": 7}, "validation": {"target_count": 3}}},
        "loader": {"batch_size": 128}}
    return payload, history, exposure, timing, data, artifact


def test_identity_maps_only_two_paths_without_payload_mutation():
    original = identity()
    relocated = deepcopy(original)
    for key, m in mappings().items():
        relocated["arguments"][key] = m["target_path"]
    frozen = deepcopy(relocated)
    assert r.canonicalize_identity(relocated, mappings()) == original
    assert relocated == frozen
    training = SimpleNamespace(_resume_identity=lambda **kwargs: relocated)
    original_function = training._resume_identity
    with r.identity_mapping(training, mappings(), original):
        assert training._resume_identity() == original
    assert training._resume_identity is original_function
    relocated["arguments"]["execution_role"] = "unapproved"
    with r.identity_mapping(training, mappings(), original):
        with pytest.raises(ValueError, match="beyond input paths"):
            training._resume_identity()


@pytest.mark.parametrize("change", ["path", "data_sha", "device"])
def test_identity_rejects_unapproved_drift(change):
    value = identity()
    if change == "path": value["arguments"]["data"] = "/foreign/data"
    if change == "data_sha": value["arguments"]["data_sha256"] = "different"
    if change == "device": value["arguments"]["device"] = "cpu"
    if change == "device":
        training = SimpleNamespace(_resume_identity=lambda **kwargs: value)
        with r.identity_mapping(training, mappings(), identity()):
            with pytest.raises(ValueError, match="beyond input paths"): training._resume_identity()
    else:
        with pytest.raises(ValueError): r.canonicalize_identity(value, mappings())


def test_prefix_preserves_records_steps_and_source_history():
    values = prefix_fixture()
    before = deepcopy(values)
    records, times, steps = r.validate_prefix(*values, state_digest=lambda x: x)
    assert records == values[2] and times == values[3]["epochs"] and steps == 2
    assert values == before


@pytest.mark.parametrize("change", ["history", "epoch", "exposure", "steps", "timing", "best_SHA", "current_SHA", "identity", "test"])
def test_corrupted_original_prefix_is_rejected(change):
    p, h, e, t, d, a = prefix_fixture()
    if change == "history": h[0]["val_qty_mae"] = 99.; p["history"] = deepcopy(prefix_fixture()[1])
    if change == "epoch": p["epoch"] = 1
    if change == "exposure": e["train"] = e["train"][:1]
    if change == "steps": p["optimizer_state_dict"]["state"][0]["step"] = 0
    if change == "timing": t["epochs"][-1]["epoch"] = 3
    if change == "best_SHA": p["best_state_dict"] = "corrupted"
    if change == "current_SHA": p["model_state_dict"] = "corrupted"
    if change == "identity": p["resume_identity"]["arguments"]["device"] = "cpu"
    if change == "test": p["held_out_test_evaluated"] = True
    with pytest.raises(ValueError): r.validate_prefix(p, h, e, t, d, a, state_digest=lambda x: x)


def test_once_claim_is_exclusive_and_preserves_failed_claim(tmp_path):
    job = {"id": "job", "host": "5080"}
    c = {"artifacts": {"job": {"source_checkpoint_sha256": "checkpoint"}}}
    first = r.exclusive_claim(c, tmp_path, "5080", job, time.time() + 60)
    before = (tmp_path / "claims/job.json").read_bytes()
    with pytest.raises(FileExistsError): r.exclusive_claim(c, tmp_path, "5080", job, time.time() + 90)
    assert (tmp_path / "claims/job.json").read_bytes() == before
    assert first["manual_resume_once"] is True and first["automatic_retry"] is False


def test_selected_cross_gpu_gate_is_scoped_to_exact_train_one_frame():
    q = {"status": "passed", "selected": {"state_sha256": "qualified", "qty_rmse": 1.000004}}
    source = {"best_state_sha256": "qualified", "best_epoch": 3, "best_selection_value": 1.}
    events = []
    training = SimpleNamespace(math=math)

    def train_one(digest="qualified", epoch=3, replay=1.000004):
        state_digest, best_epoch = digest, epoch
        selected_metric_value, reevaluated_metric_value = 1., replay
        return training.math.isclose(reevaluated_metric_value, selected_metric_value, rel_tol=1e-10, abs_tol=1e-8)

    training.train_one = train_one
    with r.selected_replay_gate(training, q, source, events):
        assert train_one() is True
        assert train_one(digest="foreign") is False
        assert train_one(epoch=4) is False
        assert train_one(replay=1.00003) is False
        # Identical values in another function cannot relax another scientific gate.
        assert training.math.isclose(1.000004, 1., rel_tol=1e-10, abs_tol=1e-8) is False
        assert math.isclose(1.000004, 1., rel_tol=1e-10, abs_tol=1e-8) is False
    assert training.math is math
    assert len(events) == 1 and events[0]["source_history_overwritten"] is False


def test_validation_replay_compares_same_epoch_all_three_metrics():
    replay = {"evaluation_scope": "validation_only", "held_out_test_evaluated": False,
        "qty_rmse": 1.000004, "qty_mae": .4, "time_nll": .2}
    row = {"val_qty_rmse": 1., "val_qty_mae": .4, "val_time_nll": .2}
    assert all(r.compare_replay(replay, row).values())
    replay["time_nll"] = .3
    assert not r.compare_replay(replay, row)["time_nll"]
    replay["held_out_test_evaluated"] = True
    with pytest.raises(ValueError, match="scope"): r.compare_replay(replay, row)


@pytest.fixture
def sealed_contract(tmp_path, monkeypatch):
    job = {"id": "intermittent_frozen_5000__42__titantpp_gru54", "dataset": "intermittent_frozen_5000",
        "arm": "titantpp_gru54", "seed": 42, "host": "5080"}
    source_files = {"file" + str(i): "sha" for i in range(117)}
    worker_files = {"worker/runtime.py": "worker"}
    monkeypatch.setattr(r, "SOURCE_SHA", r.canonical(source_files))
    monkeypatch.setattr(r, "WORKER_SHA", r.canonical(worker_files))
    data = {"dataset_id": job["dataset"], "inherited_data_identity": {k: {"path": m["origin_path"], "sha256": m["sha256"]}
        for k, m in mappings().items()}}
    parent = {"source": {"files": source_files, "files_sha256": r.SOURCE_SHA},
        "worker": {"files": worker_files, "files_sha256": r.WORKER_SHA},
        "training": {"batch_size": 128, "maximum_epochs": 300, "minimum_epochs": 40, "patience": 40},
        "jobs": [{**job, "host": "a100"}], "datasets": [data]}
    monkeypatch.setattr(r, "PARENT_SHA", r.canonical(parent))
    c = {"schema": r.SCHEMA, "parent": {"contract": "parent_execution_contract.json", "contract_sha256": r.PARENT_SHA,
        "source_files_sha256": r.SOURCE_SHA, "worker_files_sha256": r.WORKER_SHA},
        **{key: deepcopy(parent[key]) for key in ("source", "worker", "training")},
        "evaluation_scope": "validation_only", "held_out_test_evaluated": False, "replay_tolerance": r.TOLERANCE,
        "limits": {"automatic_retry": False, "manual_resume_once": True, "workers_per_host": 1,
            "per_condition_seconds": 129600, "total_wall_seconds": 604800, "max_device_memory_fraction": .8},
        "hosts": {h: {"root": str(tmp_path / h)} for h in ("5080", "5090")}, "jobs": [job],
        "artifacts": {job["id"]: {"files": dict.fromkeys(r.REQUIRED_ARTIFACTS, "original"),
            "origin_contract_sha256": r.PARENT_SHA, "terminal_scientific_success": False, "source_checkpoint_sha256": "original"}},
        "path_mappings": {job["id"]: mappings()}}
    return parent, c, job


def test_contract_preserves_every_scientific_setting(sealed_contract):
    parent, c, _ = sealed_contract
    before = deepcopy(parent)
    r.validate_contract(c, parent)
    assert parent == before
    for key in ("source", "worker", "training"):
        bad = deepcopy(c); bad[key]["new"] = True
        with pytest.raises(ValueError, match="changed"): r.validate_contract(bad, parent)


@pytest.mark.parametrize("change", ["retry", "resume", "workers", "tolerance", "Test", "mapping", "completed"])
def test_unsafe_scope_and_permits_rejected(sealed_contract, change):
    parent, c, job = sealed_contract
    if change == "retry": c["limits"]["automatic_retry"] = True
    if change == "resume": c["limits"]["manual_resume_once"] = False
    if change == "workers": c["limits"]["workers_per_host"] = 2
    if change == "tolerance": c["replay_tolerance"] = {"rtol": .1, "atol": .1}
    if change == "Test": c["held_out_test_evaluated"] = True
    if change == "mapping": c["path_mappings"][job["id"]]["data"]["independently_verified"] = False
    if change == "completed": c["artifacts"][job["id"]]["terminal_scientific_success"] = True
    with pytest.raises(ValueError): r.validate_contract(c, parent)


def test_approval_deadline_cannot_be_bypassed(sealed_contract, tmp_path):
    parent, c, job = sealed_contract
    approval = {"approved": True, "manual_resume_once": True, "contract_sha256": r.canonical(c),
        "parent_contract_sha256": r.PARENT_SHA, "hosts": ["5080", "5090"], "job_ids": [job["id"]],
        "user_instruction": "Continue the approved checkpoints"}
    start = {"contract_sha256": r.canonical(c), "approval_sha256": r.canonical(approval),
        "started_at_unix": 10, "deadline_unix": 60}
    r.write(tmp_path / "approval.json", approval); r.write(tmp_path / "start_permit.json", start)
    assert r.authorize(c, tmp_path, "5080", now=30) == start
    with pytest.raises(ValueError, match="deadline"): r.authorize(c, tmp_path, "5080", now=60)
    approval["manual_resume_once"] = False; r.write(tmp_path / "approval.json", approval)
    with pytest.raises(ValueError, match="approval"): r.authorize(c, tmp_path, "5080", now=30)


def test_real_optimizer_exposure_wrapper_accumulates_after_prefix(tmp_path):
    torch = pytest.importorskip("torch")
    from paper.scripts import run_observed_slot_partition as shared
    p, h, exposure, timing, data, _ = prefix_fixture()
    data["inherited_data_identity"]["populations"]["train"]["target_count"] = 1
    data["inherited_data_identity"]["populations"]["validation"]["target_count"] = 1
    for split in exposure:
        for row in exposure[split]: row["count"] = 1
    for row in h: row["train_event_count"] = 1
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    for _ in range(2):
        optimizer.zero_grad(); model(torch.ones(1, 1)).sum().backward(); optimizer.step()
    batch = (None, torch.ones(1, 1), torch.ones(1, 1, dtype=torch.bool), None, torch.ones(1, 1))
    saves, statuses = [], []
    training = SimpleNamespace(make_loader=lambda *a, **kw: [batch], atomic_torch_save=lambda p, path: saves.append(p),
        _titantpp_resume_source_history=h)
    old_loader, old_save = training.make_loader, training.atomic_torch_save
    with r.audited_resume(training, shared, data, lambda: None, lambda e, rec: statuses.append((e, deepcopy(rec))),
            exposure, timing["epochs"], 2) as records:
        train_loader = training.make_loader(target_split="train")
        val_loader = training.make_loader(target_split="validation")
        for _, x, *_ in train_loader:
            optimizer.zero_grad(); model(x).sum().backward(); optimizer.step()
        list(val_loader)
        resumed_history = h + [{"epoch": 3, "train_event_count": 1, "train_batch_count": 1}]
        training.atomic_torch_save({"epoch": 3, "history": resumed_history,
            "optimizer_state_dict": optimizer.state_dict()}, tmp_path / "last_epoch_state.pt")
    assert [x["epoch"] for x in records["train"]] == [1, 2, 3]
    assert float(optimizer.state_dict()["state"][0]["step"]) == 3
    assert records["train"][:2] == exposure["train"]
    assert r.read(tmp_path / "epoch_timing.json")["epochs"][:2] == timing["epochs"]
    assert statuses[0][0] == 3 and saves[0]["history"][:2] == h
    assert training.make_loader is old_loader and training.atomic_torch_save is old_save


def test_wrapper_refuses_restarted_optimizer_step(tmp_path):
    from paper.scripts import run_observed_slot_partition as shared
    torch = pytest.importorskip("torch")
    _, h, exposure, timing, data, _ = prefix_fixture()
    batch = (None, torch.ones(7, 1), None, None, None)
    training = SimpleNamespace(make_loader=lambda *a, **kw: [batch], atomic_torch_save=lambda *a: None,
        _titantpp_resume_source_history=h)
    with r.audited_resume(training, shared, data, lambda: None, lambda *a: None, exposure, timing["epochs"], 2):
        list(training.make_loader(target_split="train")); list(training.make_loader(target_split="validation"))
        with pytest.raises(ValueError, match="steps restarted"):
            training.atomic_torch_save({"epoch": 3, "history": h + [{"epoch": 3}],
                "optimizer_state_dict": {"state": {0: {"step": 1}}}}, tmp_path / "last_epoch_state.pt")


def test_strict_seed_keeps_rng_and_loader_generator_unchanged():
    torch = pytest.importorskip("torch")
    from paper.scripts.run_taxi_quantity_interface_ablation import set_seed
    old_enabled = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    expected_generator = set_seed(42).get_state().clone()
    expected_rng = torch.get_rng_state().clone()
    training = SimpleNamespace(set_seed=set_seed)
    try:
        with r.strict_seed(training):
            actual = training.set_seed(42)
            assert torch.equal(actual.get_state(), expected_generator)
            assert torch.equal(torch.get_rng_state(), expected_rng)
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert training.set_seed is set_seed
    finally:
        torch.use_deterministic_algorithms(old_enabled, warn_only=old_warn)


def test_no_phantom_gpu_training_segment_when_only_finalizing():
    assert r.gpu_segments(41, 41, "5080") == [{"host": "a100", "first_epoch": 1, "last_epoch": 41}]
    assert r.gpu_segments(39, 41, "5090")[-1] == {"host": "5090", "first_epoch": 40, "last_epoch": 41}


@pytest.mark.parametrize("source_epoch, expected_new_epochs", [(39, 2), (41, 0)])
def test_frozen_train_one_preserves_source_best_and_honors_original_patience(tmp_path, monkeypatch, source_epoch, expected_new_epochs):
    """Run the actual frozen lifecycle with a tiny model and real AdamW state."""
    torch = pytest.importorskip("torch")
    from paper.scripts import run_observed_slot_partition as shared
    from simple_lab_test.search.common.runner import capture_rng_state, canonical_state_dict_sha256, torch_load_checkpoint
    frozen = Path(__file__).resolve().parents[2] / "search_artifacts/titantpp_cnn_gru_a100_seed42_20261004_v1/source/paper/scripts/count_aware_tpp_backbone/training.py"
    spec = importlib.util.spec_from_file_location("frozen_resume_lifecycle_test", frozen)
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    monkeypatch.setattr(training, "validate_training_identity", lambda *a: None)
    monkeypatch.setattr(training, "validate_checkpoint_route", lambda *a: None)
    monkeypatch.setattr(training, "time_model_kwargs", lambda *a: {})
    monkeypatch.setitem(training.BACKBONE_LABELS, "offline_toy", "offline toy")
    generator = training.set_seed(42)
    model = torch.nn.Linear(1, 1)
    initial = canonical_state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    best_state = None
    for epoch in range(1, source_epoch + 1):
        optimizer.zero_grad(); model(torch.ones(1, 1)).sum().backward(); optimizer.step()
        torch.rand(1, generator=generator)  # Genuine loader-generator advance.
        if epoch == 1:
            best_state = training.clone_state_dict(model)
    history = [{"epoch": e, "val_qty_rmse": 1. if e == 1 else 2., "val_joint_objective": .5,
        "val_qty_mae": .4, "val_time_nll": .2, "train_event_count": 1, "train_batch_count": 1,
        "train_all_finite": True} for e in range(1, source_epoch + 1)]
    args = SimpleNamespace(**{k: None for k in training._RESUME_IDENTITY_ARGUMENTS})
    args.__dict__.update(output_dir=tmp_path, force_rerun=False, epochs=300, min_epochs=40, early_stopping_patience=40,
        device="cpu", checkpoint_monitor="validation_raw_quantity_rmse", source_revision="a" * 40,
        batch_size=128, max_seq_len=2, lookback_weeks=1, hidden_dim=1, lr=.001, time_head_lr_multiplier=1.,
        lambda_log_qty=1., max_val_batches=None, max_train_batches=None, grad_clip=1., model_role=None,
        quantity_sigma_floor=.001, lambda_location_huber=1., location_huber_delta=.25, lambda_tail=0.,
        tail_threshold=1., tail_normalization_scale=1., tail_clip_cap=1., tail_huber_delta=1.,
        time_head_mode="legacy_clamped_rmtpp", time_scale=1., time_w_max=1., time_intercept_limit=300.,
        time_wd_safety_limit=40., time_sigma_floor=.001)
    interface = {"train_target_mean": 1., "train_target_std": 1., "time_head": {}}
    quantity = {"boundaries": [1, 2, 3, 4], "strata": []}
    ident = training._resume_identity(args=args, backbone="offline_toy", quantity_variant=r.VARIANT,
        seed=42, monitor="validation_raw_quantity_rmse", quantity_contract=quantity, interface_meta=interface)
    selection = training.checkpoint_monitor_spec("validation_raw_quantity_rmse")
    payload = {"checkpoint_type": "epoch_resume", "checkpoint_schema_version": 2, "epoch": source_epoch,
        "backbone": "offline_toy", "variant": r.VARIANT, "seed": 42, "resume_identity": ident,
        "checkpoint_monitor": "validation_raw_quantity_rmse", "checkpoint_monitor_history_key": "val_qty_rmse",
        "checkpoint_selection": selection["selection"], "initial_state_sha256": initial,
        "model_state_dict": training.clone_state_dict(model), "model_state_sha256": canonical_state_dict_sha256(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(), "history": history, "best_selection_value": 1., "best_epoch": 1,
        "best_state_dict": best_state, "best_state_sha256": canonical_state_dict_sha256(best_state),
        "rng_state": capture_rng_state(), "train_loader_generator_state": generator.get_state(),
        "source_revision": args.source_revision, "source_revision_history": [args.source_revision]}
    run = tmp_path / "runs/offline_toy" / r.VARIANT / "seed_42"
    run.mkdir(parents=True)
    training.atomic_torch_save(payload, run / "last_epoch_state.pt")
    r.write(run / "history.json", {"history": history})
    source_bytes = (run / "last_epoch_state.pt").read_bytes()
    batch = (None, torch.ones(1, 1), torch.ones(1, 1, dtype=torch.bool), None, torch.ones(1, 1))
    captured_generator = {}

    def make_loader(*a, **kwargs):
        if kwargs["target_split"] == "train":
            captured_generator["instance"] = kwargs["generator"]
        return [batch]

    monkeypatch.setattr(training, "make_loader", make_loader)
    monkeypatch.setattr(training, "build_model", lambda *a, **kw: (torch.nn.Linear(1, 1), {}))
    new_epochs = []

    def train_epoch(**kwargs):
        assert torch.equal(captured_generator["instance"].get_state(), payload["train_loader_generator_state"])
        for _, x, *_ in kwargs["loader"]:
            optim = kwargs["optimizer"]
            optim.zero_grad(); kwargs["model"](x).sum().backward(); optim.step()
        new_epochs.append(True)
        return {"train_joint_objective": .5, "train_time_nll": .2, "train_batch_joint_p99": .5,
            "train_pre_clip_grad_norm_mean": 1., "train_gradient_clip_fraction": 0., "train_event_count": 1,
            "train_batch_count": 1, "train_all_finite": True}

    monkeypatch.setattr(training, "train_epoch_with_telemetry", train_epoch)

    def evaluate(**kwargs):
        list(kwargs["loader"])
        is_source_best = canonical_state_dict_sha256(kwargs["model"].state_dict()) == payload["best_state_sha256"]
        return {"val_joint_objective": .5, "val_time_nll": .2, "val_quantity_train_loss": .5,
            "val_log_qty_mse": .5, "val_quantity_distribution_nll": 0., "val_quantity_location_huber": 0.,
            "val_tail_aux_loss": 0., "val_tail_count": 0, "val_quantity_scale_mean": 0., "qty_mae": .4,
            "qty_rmse": 1.000004 if is_source_best else 2.}

    monkeypatch.setattr(training, "evaluate", evaluate)
    exposure = {s: [{"epoch": e, "split": s, "count": 1, "batches": 1, "batch_order_sha256": "f" * 64}
        for e in range(1, source_epoch + 1)] for s in ("train", "validation")}
    times = [{"epoch": e, "elapsed_seconds": 1.} for e in range(1, source_epoch + 1)]
    training._titantpp_resume_source_history = deepcopy(history)
    events = []
    qualification = {"status": "passed", "selected": {"state_sha256": payload["best_state_sha256"], "qty_rmse": 1.000004}}
    with r.strict_seed(training), r.audited_resume(training, shared, {}, lambda: None, lambda *a: None,
            exposure, times, source_epoch) as records, r.selected_replay_gate(training, qualification, payload, events):
        summary, _, _ = training.train_one(args=args, frame=None, quantity_contract=quantity,
            interface_meta=interface, backbone="offline_toy", quantity_variant=r.VARIANT, seed=42)
    assert len(new_epochs) == expected_new_epochs
    assert summary["completed_epochs"] == 41 and summary["best_epoch"] == 1
    assert summary["selected_metric_value"] == 1. and summary["best_val_qty_rmse"] == 1.000004
    assert summary["checkpoint_state_sha256"] == payload["best_state_sha256"]
    assert r.read(run / "history.json")["history"][:source_epoch] == history
    last = torch_load_checkpoint(run / "last_epoch_state.pt", map_location="cpu")
    assert all(float(v["step"]) == 41 for v in last["optimizer_state_dict"]["state"].values())
    assert len(records["train"]) == 41 and len(records["validation"]) == 42
    assert len(events) == 1
    if expected_new_epochs == 0:
        assert (run / "last_epoch_state.pt").read_bytes() == source_bytes


def test_sealed_source_originals_and_inputs_cannot_drift(sealed_contract, tmp_path, monkeypatch):
    parent, c, job = sealed_contract
    root = Path(c["hosts"]["5080"]["root"])
    c["hosts"]["5080"].update(source_root=str(root / "source"), worker_root=str(root / "worker"))
    for name in c["source"]["files"]:
        p = root / "source" / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(name)
        c["source"]["files"][name] = r.sha(p)
    for name in c["worker"]["files"]:
        p = root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(name)
        c["worker"]["files"][name] = r.sha(p)
    monkeypatch.setattr(r, "SOURCE_SHA", r.canonical(c["source"]["files"]))
    monkeypatch.setattr(r, "WORKER_SHA", r.canonical(c["worker"]["files"]))
    c["source"]["files_sha256"] = r.SOURCE_SHA; c["worker"]["files_sha256"] = r.WORKER_SHA
    parent["source"] = deepcopy(c["source"]); parent["worker"] = deepcopy(c["worker"])
    for key in ("data", "split_manifest"):
        p = root / "data" / key; p.parent.mkdir(exist_ok=True); p.write_text(key)
        parent["datasets"][0]["inherited_data_identity"][key]["sha256"] = r.sha(p)
        c["path_mappings"][job["id"]][key].update(target_path=str(p), sha256=r.sha(p))
    monkeypatch.setattr(r, "PARENT_SHA", r.canonical(parent))
    c["parent"].update(contract_sha256=r.PARENT_SHA, source_files_sha256=r.SOURCE_SHA, worker_files_sha256=r.WORKER_SHA)
    original = root / "originals" / job["id"]
    original.mkdir(parents=True)
    for name in c["artifacts"][job["id"]]["files"]:
        p = original / name; p.write_text(name)
        c["artifacts"][job["id"]]["files"][name] = r.sha(p)
    c["artifacts"][job["id"]].update(origin_root=str(original), origin_contract_sha256=r.PARENT_SHA,
        source_checkpoint_sha256=r.sha(original / "last_epoch_state.pt"))
    c["operation_adapter_sha256"] = r.sha(SCRIPT)
    r.write(root / "parent_execution_contract.json", parent)
    r.write(root / "execution_contract.json", c)
    r.load(root / "execution_contract.json", "5080", job["id"])
    source_file = root / "source/file0"
    prior = source_file.read_bytes(); source_file.write_text("changed")
    with pytest.raises(ValueError, match="Scientific source changed"):
        r.load(root / "execution_contract.json", "5080", job["id"])
    source_file.write_bytes(prior)
    (original / "last_epoch_state.pt").write_text("changed")
    with pytest.raises(ValueError, match="Immutable original SHA changed"):
        r.load(root / "execution_contract.json", "5080", job["id"])


def test_target_host_lock_prevents_concurrent_workers(tmp_path):
    import fcntl
    first = (tmp_path / "host_worker.lock").open("a")
    second = (tmp_path / "host_worker.lock").open("a")
    try:
        fcntl.flock(first.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            fcntl.flock(second.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        first.close(); second.close()


def test_synthetic_resume_probe_restores_dropout_optimizer_and_data_rng(monkeypatch):
    torch = pytest.importorskip("torch")
    from paper.scripts.count_aware_tpp_backbone import core
    engine = SimpleNamespace(build_model=lambda data, arm: (torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.Dropout(.2), torch.nn.Linear(8, 1)), {}))
    monkeypatch.setattr(core, "target_outputs", lambda model, dt, mask, qty, **kw:
        {"joint_loss": (model(dt).reshape(-1) - qty.mean(1)).square()})
    result = r.synthetic_resume_check(engine, {"loader": {"max_seq_len": 4}}, "offline_toy", lambda: None, device="cpu")
    assert result["uninterrupted_state_sha256"] == result["resumed_state_sha256"]
    assert result["uninterrupted_losses"][1:] == result["resumed_losses"]
    assert result["real_data_loaded"] is False and result["held_out_test_evaluated"] is False


def test_frozen_import_root_requires_empty_sample_data_sentinel(tmp_path):
    """The 117-file closure alone omits a deliberately empty bootstrap folder."""
    frozen = Path(__file__).resolve().parents[2] / "search_artifacts/titantpp_cnn_gru_a100_seed42_20261004_v1/source/simple_lab_test/common/pathing.py"
    spec = importlib.util.spec_from_file_location("frozen_pathing_layout_test", frozen)
    pathing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pathing)
    source = tmp_path / "isolated/source"
    (source / "models").mkdir(parents=True)
    (source / "utils").mkdir()
    anchor = source / "simple_lab_test/search/common"
    anchor.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Could not locate"):
        pathing.resolve_project_root(anchor)
    (source / "sample_data").mkdir()
    (source / "sample_data/.keep").write_bytes(b"")
    assert pathing.resolve_project_root(anchor) == source
    assert list((source / "sample_data").iterdir()) == [source / "sample_data/.keep"]
    assert (source / "sample_data/.keep").read_bytes() == b""


def test_tiny_train_statistics_difference_cannot_silently_relax_resume_identity():
    expected = identity()
    expected["interface_meta"] = {"time_head": {"train_time_statistics": {
        "target_log_scaled_mean": -0.21389036220110552}}}
    rebuilt = deepcopy(expected)
    rebuilt["interface_meta"]["time_head"]["train_time_statistics"]["target_log_scaled_mean"] = -0.2138903622011054
    assert math.isclose(rebuilt["interface_meta"]["time_head"]["train_time_statistics"]["target_log_scaled_mean"],
        expected["interface_meta"]["time_head"]["train_time_statistics"]["target_log_scaled_mean"], abs_tol=1e-12)
    training = SimpleNamespace(_resume_identity=lambda **kw: rebuilt)
    with r.identity_mapping(training, mappings(), expected):
        with pytest.raises(ValueError, match="beyond input paths"):
            training._resume_identity()
