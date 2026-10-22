"""Train-only cost protocol and selected-copy tests without any GPU work."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from paper.scripts import profile_shared_history_mlp_control as p
from paper.scripts import profile_shared_component_ablation as frozen
from paper.tests.test_run_shared_history_mlp_control import contract


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))


class Hooks:
    def __init__(self):
        self.tick = 0
        self.resets = 0
        self.events = []

    def sync(self):
        self.events.append("sync")

    def clock(self):
        self.tick += 1
        return float(self.tick)

    def transfer(self, batch):
        self.events.append("transfer")
        return batch

    def reset_peak(self):
        self.resets += 1
        self.events.append("reset")

    def memory(self):
        return {"allocated": 100, "reserved": 200}

    def peak(self):
        return {"allocated": 130 if self.resets else 150, "reserved": 250}


def batch(size=128):
    return {
        "dts": torch.ones(size, 2),
        "quantities": torch.ones(size, 2),
        "mask": torch.ones(size, 2, dtype=torch.bool),
    }


def output(model, value):
    return {"joint_loss": (model.weight * value["quantities"][:, -1]).square()}


def cache(data=None, c=None):
    data = data or {"dataset_id": "fixture"}
    batches = [batch() for _ in range(37)]
    rows, targets = [], []
    for i, b in enumerate(batches):
        ids = [["part", i * 128 + n] for n in range(128)]
        targets.extend(ids)
        rows.append(
            {
                "batch": i,
                "phase": "warmup" if i < 5 else "measured",
                "target_split": "train",
                "targets": ids,
                "tensor_sha256": p.tensor_sha(b),
            }
        )
    manifest = {
        "dataset": data["dataset_id"],
        "split": "train",
        "synthetic": False,
        "preparation_seed": 42,
        "batches": rows,
        "target_ids_sha256": p.canonical_sha(targets),
    }
    if c is not None:
        manifest.update(contract_sha256=p.canonical_sha(c), contract_file_sha256=None)
    return batches, manifest


def test_frozen_measurement_target_and_cache_functions_are_exact_dependencies():
    assert p.measure_batches is frozen.measure_batches
    assert p.CudaHooks is frozen.CudaHooks
    assert p._target_outputs is frozen._target_outputs
    assert p._optimizer is frozen._optimizer
    assert p.validate_batch_cache is frozen.validate_batch_cache


def test_all_27_cost_jobs_and_fixed_protocol_admitted(tmp_path, monkeypatch):
    from paper.scripts import run_shared_history_mlp_control as r

    c = contract(tmp_path)
    c["cost"] = deepcopy(p.PROTOCOL)
    monkeypatch.setattr(
        p, "_manager", lambda: SimpleNamespace(validate=r._validate_contract)
    )
    for job in c["jobs"]:
        selected, _, root, folder = p.select_job(c, job["host"], job["id"])
        assert selected == job and folder == root / "run" / job["id"]


@pytest.mark.parametrize("key", p.PROTOCOL)
def test_changed_cost_protocol_is_rejected(tmp_path, monkeypatch, key):
    from paper.scripts import run_shared_history_mlp_control as r

    c = contract(tmp_path)
    c["cost"] = deepcopy(p.PROTOCOL)
    c["cost"][key] = "changed"
    monkeypatch.setattr(
        p, "_manager", lambda: SimpleNamespace(validate=r._validate_contract)
    )
    with pytest.raises(ValueError):
        p.validate_cost_contract(c)


def test_raw_contract_hash_is_of_the_actual_parsed_bytes(tmp_path):
    path = tmp_path / "contract.json"
    path.write_text('{"schema":"fixture"}\n')
    digest = p.sha_file(path)
    assert p.read_sealed_contract(path, digest) == {"schema": "fixture"}
    path.write_text('{ "schema":"fixture"}\n')
    with pytest.raises(ValueError, match="raw SHA"):
        p.read_sealed_contract(path, digest)


@pytest.mark.parametrize("mode,rep", [("test", 1), ("train", 0), ("eval", 4)])
def test_invalid_mode_or_repetition_cannot_load_models(
    tmp_path, monkeypatch, mode, rep
):
    def forbidden(*args):
        raise AssertionError("Selection must not run")

    monkeypatch.setattr(p, "select_job", forbidden)
    with pytest.raises(ValueError):
        p.profile({}, "5080", "job", mode, rep)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("entry", ["profile", "prepare"])
def test_global_validation_gate_precedes_real_input_and_measurement(
    tmp_path, monkeypatch, entry
):
    job = {
        "id": "job",
        "host": "5080",
        "arm": "single_mlp",
        "dataset": "fixture",
        "seed": 42,
    }
    c = {
        "hosts": {
            "5080": {"root": str(tmp_path), "source_root": str(tmp_path / "source")}
        }
    }

    def fail(*args):
        raise ValueError("global gate absent")

    def forbidden(*args):
        raise AssertionError("Real cache/model must not be loaded")

    manager = SimpleNamespace(verify_global_gate=fail, verify_files=lambda *args: None)
    monkeypatch.setattr(p, "_manager", lambda: manager)
    monkeypatch.setattr(
        p, "select_job", lambda *args: (job, {}, tmp_path, tmp_path / "run/job")
    )
    monkeypatch.setattr(p, "validate_cost_contract", lambda *args: None)
    monkeypatch.setattr(p, "_load_selected", forbidden)
    with pytest.raises(ValueError, match="global gate absent"):
        if entry == "profile":
            p.profile(c, "5080", "job", "eval", 1)
        else:
            p.prepare_batches(c, "5080")
    assert not list(tmp_path.iterdir())


def test_selected_copy_uses_complete_runner_validation_and_restores_gradients(
    tmp_path, monkeypatch
):
    model = Tiny().eval().requires_grad_(False)
    checkpoint = tmp_path / "original.pt"
    torch.save(model.state_dict(), checkpoint)
    original_sha = p.sha_file(checkpoint)
    payload = {
        "model_state_dict": {k: v.clone() for k, v in model.state_dict().items()}
    }
    selected = {
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": original_sha,
        "checkpoint_payload": payload,
    }
    folder = tmp_path / "run/job"
    folder.mkdir(parents=True)
    (folder / "train_receipt.json").write_text('{"fixture":true}')
    calls = []

    def complete(*args):
        calls.append(args)
        return model, selected

    runner = SimpleNamespace(
        _check_source=lambda *args: None,
        select_job=lambda *args: (
            {"id": "job"},
            {"dataset_id": "fixture"},
            tmp_path,
            folder,
        ),
        _load_model=complete,
    )
    monkeypatch.setattr(p, "_runner", lambda: runner)
    loaded, identity, hashes, data = p._load_selected({}, "5080", "job")
    assert len(calls) == 1 and calls[0][-1] == "cpu"
    assert loaded is model and loaded.weight.requires_grad
    assert identity is selected and data["dataset_id"] == "fixture"
    assert hashes["binary_sha256"] == original_sha == p.sha_file(checkpoint)


def test_only_ram_training_changes_and_peak_protocol_is_retained(tmp_path):
    model, hooks = Tiny(), Hooks()
    checkpoint = tmp_path / "original.pt"
    torch.save(model.state_dict(), checkpoint)
    before = p.sha_file(checkpoint)
    result = p.measure_batches(
        model,
        [batch(2)] * 5,
        "train",
        torch.optim.AdamW(model.parameters(), lr=0.01),
        hooks,
        output,
        grad_clip=1.0,
        warmup=2,
        measured=3,
    )
    assert result["optimizer_steps"] == 5 and model.weight.item() < 2.0
    assert result["throughput_targets_per_second"] == 2.0
    assert result["total_peak_bytes"] == {"allocated": 150, "reserved": 250}
    assert result["measured_incremental_peak_bytes"] == {
        "allocated": 30,
        "reserved": 50,
    }
    assert hooks.events[: hooks.events.index("reset")].count("transfer") == 2
    assert before == p.sha_file(checkpoint)


def test_eval_is_gradient_free_and_state_is_unchanged():
    model, seen = Tiny(), []
    before = p.tensor_sha(model.state_dict())

    def checked(m, b):
        seen.append((m.training, torch.is_grad_enabled()))
        return output(m, b)

    result = p.measure_batches(
        model,
        [batch(2)] * 3,
        "eval",
        None,
        Hooks(),
        checked,
        grad_clip=1.0,
        warmup=1,
        measured=2,
    )
    assert result["optimizer_steps"] == 0
    assert seen == [(False, False)] * 3
    assert before == p.tensor_sha(model.state_dict())


@pytest.mark.parametrize(
    "fault", ["split", "target_split", "duplicate", "tensor", "contract"]
)
def test_non_train_or_changed_batch_evidence_is_rejected(fault):
    c = {"fixture": True}
    data = {"dataset_id": "fixture"}
    batches, manifest = cache(data, c)
    p.validate_batch_cache(
        manifest, batches, data, contract=c, contract_file_sha256=None
    )
    if fault == "split":
        manifest["split"] = "test"
    elif fault == "target_split":
        manifest["batches"][0]["target_split"] = "validation"
    elif fault == "duplicate":
        manifest["batches"][1]["targets"][0] = manifest["batches"][0]["targets"][0]
    elif fault == "tensor":
        batches[0]["dts"][0, 0] = 17.0
    else:
        manifest["contract_sha256"] = "changed"
    with pytest.raises(ValueError):
        p.validate_batch_cache(
            manifest, batches, data, contract=c, contract_file_sha256=None
        )


def test_cpu_profile_exact_37_batches_and_exclusive_result_preserves_checkpoint(
    tmp_path, monkeypatch
):
    job = {
        "id": "job",
        "host": "5080",
        "arm": "single_mlp",
        "dataset": "fixture",
        "seed": 42,
    }
    c = {
        "source": {"files_sha256": "source"},
        "hosts": {"5080": {"root": str(tmp_path)}},
    }
    data = {"dataset_id": "fixture", "optimizer": {"grad_clip": 1.0}}
    model = Tiny()
    checkpoint = tmp_path / "selected.pt"
    torch.save(model.state_dict(), checkpoint)
    before = p.sha_file(checkpoint)
    selected = {
        "checkpoint_path": checkpoint,
        "selected_epoch": 1,
        "training_origin": "fresh_fit",
        "provenance": None,
    }
    identity = {
        "binary_sha256": before,
        "tensor_sha256": p.tensor_sha(model.state_dict()),
    }
    batches, manifest = cache(data, c)
    path = tmp_path / "cost/batches/fixture.pt"
    path.parent.mkdir(parents=True)
    torch.save(batches, path)
    manifest["file_sha256"] = p.sha_file(path)
    path.with_suffix(".json").write_text(json.dumps(manifest))
    manager = SimpleNamespace(verify_global_gate=lambda *args: None)
    monkeypatch.setattr(p, "_manager", lambda: manager)
    monkeypatch.setattr(
        p, "select_job", lambda *args: (job, data, tmp_path, tmp_path / "run/job")
    )
    monkeypatch.setattr(
        p, "_load_selected", lambda *args: (model, selected, identity, data)
    )
    monkeypatch.setattr(p, "_runtime", lambda *args: {"gpu": {"name": "CPU fixture"}})
    monkeypatch.setattr(p, "_guard", lambda *args: None)
    monkeypatch.setattr(p, "CudaHooks", Hooks)
    monkeypatch.setattr(model, "to", lambda *args, **kwargs: model)
    monkeypatch.setattr(p, "_target_outputs", output)
    monkeypatch.setattr(
        p, "_optimizer", lambda m, d: torch.optim.AdamW(m.parameters(), lr=0.01)
    )
    result = p.profile(c, "5080", "job", "train", 1)
    assert result["input_split"] == "train" and result["scientific_fit"] is False
    assert result["checkpoint_saved"] is False and result["optimizer_steps"] == 37
    assert len(result["rows"]) == 37 and result["checkpoint_binary_unchanged"] is True
    assert before == p.sha_file(checkpoint)
    output_path = tmp_path / "cost/results/job/train_rep1.json"
    result_sha = p.sha_file(output_path)
    with pytest.raises(ValueError, match="immutable"):
        p.profile(c, "5080", "job", "train", 1)
    assert result_sha == p.sha_file(output_path)
