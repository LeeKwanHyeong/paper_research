"""CPU-only contract and generic-engine coverage for the A--D runner."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from paper.scripts import run_shared_component_ablation as runner
from paper.scripts import run_titantpp_shared_history_cap as fixture


def _data(dataset):
    value = fixture.fabricated_data()
    value["model"].update(lambda_log_qty=1.0, lambda_tail=0.0, time_head_lr_multiplier=1.0)
    value["loader"].update(batch_size=128, num_workers=0, drop_last=False,
                            train_shuffle=True, validation_shuffle=False)
    value.update(dataset_id=dataset, optimizer={"name": "AdamW", "lr": .001, "weight_decay": .01, "grad_clip": 1.0})
    value["inherited_data_identity"] = {"populations": {"train": {}, "validation": {}}}
    return value


def contract(tmp_path):
    datasets = [_data(name) for name in runner.DATASETS]
    jobs = [{"id": f"{dataset}__{arm}__seed{seed}", "dataset": dataset, "arm": arm,
             "seed": seed, "host": "5080" if seed == 42 else "5090",
             **({"reuse": {"placeholder": True}} if arm == "full" and seed in (52, 62) else {})}
            for dataset in runner.DATASETS for arm in runner.ARMS for seed in runner.SEEDS]
    files = {"placeholder": "0" * 64}
    return {"schema": runner.SCHEMA, "scope": "execution", "hosts": {"5080": {"root": str(tmp_path)}, "5090": {"root": str(tmp_path)}},
            "common_condition": {"time_input": "log1p", "loss_mode": "log_mse", "time_alpha": 1.0},
            "implementation_qualification": {"status": "passed", "training_allowed": True},
            "training": {"maximum_epochs": 300, "minimum_epochs": 40, "patience": 40, "batch_size": 128, "monitor": runner.MONITOR},
            "source": {"files": files, "files_sha256": runner._canonical(files)}, "input_files": {str(i): "0" * 64 for i in range(9)},
            "datasets": datasets, "jobs": jobs}


def test_exact_matrix_and_id_binding(tmp_path):
    value = contract(tmp_path)
    assert len(value["jobs"]) == 36
    for job in value["jobs"]:
        selected, data, root, folder = runner.select_job(value, job["host"], job["id"])
        assert selected == job and data["dataset_id"] == job["dataset"]
        assert folder == root / "run" / job["id"]


@pytest.mark.parametrize("arm", runner.ARMS)
def test_component_builder_preserves_joint_target_interface(arm):
    data = fixture.fabricated_data()
    model, encoder = runner._build_model(data, arm)
    assert isinstance(encoder, dict)
    dt = torch.ones((2, 8)); qty = torch.ones((2, 8)); mask = torch.ones((2, 8), dtype=torch.bool)
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    value = target_outputs(model, dt, mask, qty, lambda_log_qty=1.0)
    assert torch.isfinite(value["joint_loss"]).all()
    assert torch.equal(value["joint_loss"], value["time_loss"] + value["quantity_train_loss"])


@pytest.mark.parametrize("arm", runner.ARMS)
def test_generic_train_one_cpu_two_epochs_uses_each_arm(tmp_path, arm):
    """Actual generic training engine with multiple Validation batches is retained as CPU evidence."""
    from paper.scripts.count_aware_tpp_backbone import training
    (tmp_path / arm).mkdir()
    args, frame, data, interface, quantity = fixture._engine_fixture(tmp_path / arm, "titantpp_history_mlp_shared15_v1")
    args.device = "cpu"; args.epochs = args.min_epochs = args.early_stopping_patience = 2
    args.batch_size = 2; args.time_alpha = 1.0; args.max_val_batches = None
    with runner._engine_hooks(data, arm):
        summary, _, _ = training.train_one(args=args, frame=frame, quantity_contract=quantity,
            interface_meta=interface, backbone=arm, quantity_variant=runner.VARIANT, seed=42)
    assert summary["completed_epochs"] == 2
    assert summary["best_epoch"] in (1, 2)
    assert Path(summary["checkpoint_path"]).exists()


def test_contract_rejects_extra_test_population_requirement(tmp_path):
    value = contract(tmp_path)
    value["jobs"][0]["id"] = "tampered"
    with pytest.raises(ValueError):
        runner.select_job(value, "5080", "tampered")
