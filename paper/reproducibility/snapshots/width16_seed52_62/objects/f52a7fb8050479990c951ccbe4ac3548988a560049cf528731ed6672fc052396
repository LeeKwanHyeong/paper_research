"""Original-B quantity graph with a train-calibrated raw-error auxiliary loss.

Calibration is a bounded, optimizer-free read of training targets. It matches
initial quantity-head gradient energy only; it does not claim to match later
encoder interference, time gradients, or the effect of global clipping.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
import re
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset, default_collate

from paper.scripts.quantity_objective_comparison import QuantityStatistics
from paper.scripts.time_quantity_diagnostic import (
    _batch_tensors,
    _validate_model,
    task_outputs as b_task_outputs,
)
from simple_lab_test.search.common.runner import canonical_state_dict_sha256


NAMES = ("B_log_original", "mixed_original", "mixed_matched_original")
CALIBRATION_SCHEMA = "mixed_quantity_calibration_v1"
POLICY = {
    "rho": 0.25,
    "max_samples": 8192,
    "batch_size": 128,
    "seed": 1042,
    "aggregation": "mean_batch_head_gradient_squared_norms",
    "calibration_head": "quantity_head",
    "dropout_mode": "train",
    "selection": "isolated_cpu_randperm_without_replacement",
}
CALIBRATION_POLICY = dict(POLICY)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _number(value: Any, name: str) -> float:
    _require(type(value) in {int, float} and math.isfinite(value), f"{name} must be a finite number")
    return float(value)


def _digest(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


@dataclass(frozen=True)
class MixedQuantityObjective:
    name: str
    alpha: float
    quantity_scale: float
    calibration_sha256: str

    def __post_init__(self) -> None:
        _require(self.name in NAMES, "Unknown mixed quantity objective")
        alpha = _number(self.alpha, "alpha")
        scale = _number(self.quantity_scale, "quantity_scale")
        _require(_digest(self.calibration_sha256), "Calibration hash must be 64 lowercase hexadecimal characters")
        if self.name == NAMES[0]:
            _require(alpha == 0.0 and scale == 1.0, "B requires alpha=0 and quantity_scale=1")
        else:
            _require(alpha > 0.0 and scale > 0.0, "Mixed objectives require positive alpha and quantity_scale")
            if self.name == NAMES[1]:
                _require(scale == 1.0, "Unmatched mixed objective requires quantity_scale=1")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "quantity_scale", scale)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MixedQuantityObjective:
        _require(isinstance(value, dict) and set(value) == {
            "name", "alpha", "quantity_scale", "calibration_sha256",
        }, "Mixed quantity objective has an unexpected schema")
        return cls(**value)


def mixed_joint_causal_batch_objective(
    model, dts, mask, quantities, *, statistics: QuantityStatistics,
    objective: MixedQuantityObjective,
) -> dict[str, torch.Tensor]:
    """Call B's stochastic graph once and preserve its control arithmetic."""
    _require(isinstance(objective, MixedQuantityObjective), "A frozen MixedQuantityObjective is required")
    _require(isinstance(statistics, QuantityStatistics), "Frozen QuantityStatistics are required")
    outputs = b_task_outputs(model, dts, mask, quantities, "joint")
    raw = ((outputs["pred_qty"] - outputs["true_qty"]) / statistics.raw_scale).square()
    outputs["raw_qty_loss"] = raw
    if objective.name != NAMES[0]:
        quantity = objective.quantity_scale * (outputs["log_qty_loss"] + objective.alpha * raw)
        outputs.update(quantity_train_loss=quantity, loss_quantity=quantity)
        joint = outputs["time_loss"] + quantity
        outputs.update(objective_loss=joint, joint_loss=joint)
    _require(all(bool(torch.isfinite(outputs[key]).all()) for key in (
        "pred_qty", "true_qty", "raw_qty_loss", "log_qty_loss", "quantity_train_loss",
        "time_loss", "objective_loss",
    )), "Mixed quantity objective is nonfinite")
    return outputs


def _coefficients(a: float, r: float, d: float) -> tuple[float, float]:
    a, r, d = (_number(value, name) for value, name in ((a, "A"), (r, "R"), (d, "D")))
    _require(a > 0.0 and r > 0.0, "Degenerate calibration gradient energy")
    # sqrt(A)/sqrt(R) avoids an avoidable overflow in A/R.
    alpha = POLICY["rho"] * math.sqrt(a) / math.sqrt(r)
    energy = a + 2.0 * alpha * d + alpha * alpha * r
    _require(math.isfinite(energy) and energy > 0.0, "Degenerate mixed gradient energy")
    scale = math.sqrt(a / energy)
    _require(math.isfinite(alpha) and alpha > 0.0 and math.isfinite(scale), "Nonfinite calibration coefficient")
    tolerance = 1e-10
    _require(abs(d) <= math.sqrt(a) * math.sqrt(r) * (1.0 + tolerance), "Invalid gradient inner product")
    _require(0.8 - tolerance <= scale <= 4.0 / 3.0 + tolerance, "Calibration scale violates rho bound")
    return alpha, scale


def _collate(samples):
    batch = default_collate(samples)
    if isinstance(batch, dict) and "quantities" not in batch and "values" in batch:
        batch["quantities"] = batch["values"]
    return batch


def calibrate_mixed_objective(
    model, train_dataset, *, statistics: QuantityStatistics, identity: dict[str, Any],
    device="cpu", synthetic=False, budget_check=None,
) -> dict[str, Any]:
    """Freeze coefficients from mean-batch gradients of the initial B head.

The model must already be on ``device``. No optimizer is instantiated, and a
private data loader leaves the later training loader's RNG untouched.
"""
    _validate_model(model, "joint")
    _require(isinstance(statistics, QuantityStatistics), "Frozen QuantityStatistics are required")
    _require(type(synthetic) is bool, "synthetic must be a boolean")
    _require(isinstance(identity, dict) and {"source", "data", "runtime", "execution_contract_sha256"} <= set(identity),
             "Calibration requires source/data/runtime/execution identity")
    identity = json.loads(json.dumps(identity, sort_keys=True, ensure_ascii=False, allow_nan=False))
    if synthetic:
        _require(type(train_dataset) is TensorDataset and 0 < len(train_dataset) <= 8,
                 "Synthetic calibration accepts only TensorDataset with at most 8 rows")
        splits = getattr(train_dataset, "target_splits", None)
        _require(splits is None or set(splits) == {"train"}, "Synthetic calibration cannot override non-train target provenance")
    else:
        splits = getattr(train_dataset, "target_splits", None)
        _require(splits is not None and set(splits) == {"train"}, "Calibration targets must be train only")
    _require(len(train_dataset) > 0, "Calibration dataset is empty")
    device = torch.device(device)
    _require(device.type in {"cpu", "cuda"}, "Calibration supports CPU or CUDA only")
    parameters = list(model.named_parameters())
    _require(all(p.device.type == device.type and (device.index is None or p.device.index == device.index)
                 for _, p in parameters), "Model must already be on the calibration device")
    _require(isinstance(model.quantity_head, torch.nn.Linear) and model.quantity_head.out_features == 1,
             "Calibration requires the original linear quantity head")
    _require(bool((model.quantity_head.weight == 0).all()), "Calibration requires initial zero quantity-head weights")
    _require(model.quantity_head.bias is not None and bool(torch.isfinite(model.quantity_head.bias).all()),
             "Calibration requires the original finite quantity-head bias")
    expected_bias = torch.full_like(model.quantity_head.bias, statistics.z0)
    _require(torch.allclose(model.quantity_head.bias, expected_bias, rtol=1e-6, atol=1e-7),
             "Calibration quantity-head bias differs from frozen train initialization")

    rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_initialized():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    initial_hash = canonical_state_dict_sha256(before)
    modes = [(module, module.training) for module in model.modules()]
    parameter_flags = [(p, p.requires_grad, p.grad, None if p.grad is None else p.grad.detach().clone()) for _, p in parameters]
    state_mutated = False
    try:
        if budget_check is not None:
            budget_check("mixed_calibration_start")
        model.train()
        for _, p in parameters:
            p.requires_grad_(True)
        random.seed(POLICY["seed"])
        np.random.seed(POLICY["seed"])
        torch.random.default_generator.manual_seed(POLICY["seed"])
        if "cuda" in rng:
            torch.cuda.manual_seed_all(POLICY["seed"])
        index_generator = torch.Generator(device="cpu").manual_seed(POLICY["seed"])
        indices = torch.randperm(len(train_dataset), generator=index_generator)[:POLICY["max_samples"]]
        loader = DataLoader(Subset(train_dataset, indices.tolist()), batch_size=POLICY["batch_size"],
                            shuffle=False, drop_last=False, num_workers=0, collate_fn=_collate,
                            generator=torch.Generator(device="cpu").manual_seed(POLICY["seed"]))
        head = [(name, p) for name, p in parameters if name.startswith("quantity_head.")]
        _require(len(head) == 2, "Calibration requires exactly original quantity-head weight and bias")
        batch_hash = hashlib.sha256()
        energies = []
        sample_count = 0
        for batch in loader:
            if budget_check is not None:
                budget_check("mixed_calibration_batch")
            dts, mask, quantities = _batch_tensors(batch, device)
            digest = canonical_state_dict_sha256({"dts": dts, "mask": mask, "quantities": quantities})
            batch_hash.update(digest.encode("ascii"))
            with torch.enable_grad():
                outputs = b_task_outputs(model, dts, mask, quantities, "joint")
                log_loss = outputs["log_qty_loss"].mean()
                raw_loss = ((outputs["pred_qty"] - outputs["true_qty"]) / statistics.raw_scale).square().mean()
                _require(bool(torch.isfinite(log_loss)) and bool(torch.isfinite(raw_loss))
                         and bool(torch.isfinite(outputs["time_loss"]).all()), "Nonfinite calibration loss")
                log_grad = torch.autograd.grad(log_loss, [p for _, p in head], retain_graph=True)
                raw_grad = torch.autograd.grad(raw_loss, [p for _, p in head])
            _require(all(bool(torch.isfinite(g).all()) for g in (*log_grad, *raw_grad)), "Nonfinite calibration gradient")
            a = sum(float(g.detach().double().square().sum().item()) for g in log_grad)
            r = sum(float(g.detach().double().square().sum().item()) for g in raw_grad)
            d = sum(float((gl.detach().double() * gr.detach().double()).sum().item())
                    for gl, gr in zip(log_grad, raw_grad, strict=True))
            energies.append((a, r, d))
            sample_count += int(outputs["true_qty"].numel())
        _require(sample_count == len(indices), "Calibration target count differs from selected canonical rows")
        a, r, d = (math.fsum(row[i] for row in energies) / len(energies) for i in range(3))
        alpha, scale = _coefficients(a, r, d)
        if budget_check is not None:
            budget_check("mixed_calibration_complete")
        receipt = {
            "schema_version": CALIBRATION_SCHEMA, "policy": dict(POLICY),
            "identity": identity, "statistics": {"mu": statistics.mu, "raw_scale": statistics.raw_scale},
            "synthetic": synthetic, "train_dataset_count": len(train_dataset),
            "selected_sample_count": sample_count, "batch_count": len(energies),
            "selected_indices_sha256": canonical_state_dict_sha256({"indices": indices}),
            "batch_sha256": batch_hash.hexdigest(), "initial_model_state_sha256": initial_hash,
            "head_parameter_names": [name for name, _ in head], "moments": {"A": a, "R": r, "D": d},
            "alpha": alpha, "quantity_scale": scale, "optimizer_updates": 0,
            "validation_accessed": False, "held_out_accessed": False,
        }
        receipt["calibration_sha256"] = _json_sha(receipt)
    finally:
        state_mutated = canonical_state_dict_sha256(model.state_dict()) != initial_hash
        if state_mutated:
            model.load_state_dict(before, strict=True)
        for p, enabled, grad_reference, grad_value in parameter_flags:
            p.requires_grad_(enabled)
            p.grad = grad_reference
            if grad_value is not None:
                with torch.no_grad():
                    grad_reference.copy_(grad_value)
        for module, training in modes:
            module.training = training
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if "cuda" in rng:
            torch.cuda.set_rng_state_all(rng["cuda"])
    _require(not state_mutated, "Calibration forward mutated model state; state restored")
    return receipt


def objectives_from_calibration(receipt: dict[str, Any]) -> list[MixedQuantityObjective]:
    """Validate a self-bound receipt and reconstruct the three fixed arms."""
    fields = {"schema_version", "policy", "identity", "statistics", "synthetic", "train_dataset_count",
              "selected_sample_count", "batch_count", "selected_indices_sha256", "batch_sha256",
              "initial_model_state_sha256", "head_parameter_names", "moments", "alpha", "quantity_scale",
              "optimizer_updates", "validation_accessed", "held_out_accessed", "calibration_sha256"}
    _require(isinstance(receipt, dict) and set(receipt) == fields, "Unexpected calibration receipt schema")
    digest = receipt["calibration_sha256"]
    _require(_digest(digest) and digest == _json_sha({k: v for k, v in receipt.items() if k != "calibration_sha256"}),
             "Calibration receipt checksum mismatch")
    _require(receipt["schema_version"] == CALIBRATION_SCHEMA and receipt["policy"] == POLICY, "Calibration policy mismatch")
    _require(receipt["optimizer_updates"] == 0 and receipt["validation_accessed"] is False
             and receipt["held_out_accessed"] is False, "Calibration scope mismatch")
    _require(isinstance(receipt["identity"], dict) and {"source", "data", "runtime", "execution_contract_sha256"} <= set(receipt["identity"]),
             "Calibration identity is incomplete")
    _require(isinstance(receipt["statistics"], dict) and set(receipt["statistics"]) == {"mu", "raw_scale"},
             "Calibration statistics schema mismatch")
    QuantityStatistics(**receipt["statistics"])
    _require(type(receipt["synthetic"]) is bool, "Calibration synthetic flag is invalid")
    for key in ("train_dataset_count", "selected_sample_count", "batch_count"):
        _require(type(receipt[key]) is int and receipt[key] > 0, f"Invalid calibration {key}")
    _require(receipt["selected_sample_count"] == min(POLICY["max_samples"], receipt["train_dataset_count"]), "Calibration sample count mismatch")
    _require(receipt["batch_count"] == math.ceil(receipt["selected_sample_count"] / POLICY["batch_size"]), "Calibration batch count mismatch")
    _require(not receipt["synthetic"] or receipt["train_dataset_count"] <= 8, "Synthetic calibration exceeds bound")
    _require(all(_digest(receipt[key]) for key in ("selected_indices_sha256", "batch_sha256", "initial_model_state_sha256")), "Calibration fingerprint is invalid")
    _require(receipt["head_parameter_names"] == ["quantity_head.weight", "quantity_head.bias"], "Calibration head mismatch")
    _require(isinstance(receipt["moments"], dict) and set(receipt["moments"]) == {"A", "R", "D"}, "Calibration moments schema mismatch")
    alpha, scale = _coefficients(receipt["moments"]["A"], receipt["moments"]["R"], receipt["moments"]["D"])
    _require(math.isclose(_number(receipt["alpha"], "alpha"), alpha, rel_tol=1e-12, abs_tol=0.0)
             and math.isclose(_number(receipt["quantity_scale"], "quantity_scale"), scale, rel_tol=1e-12, abs_tol=0.0),
             "Calibration coefficients differ from gradient moments")
    return [MixedQuantityObjective(NAMES[0], 0.0, 1.0, digest),
            MixedQuantityObjective(NAMES[1], alpha, 1.0, digest),
            MixedQuantityObjective(NAMES[2], alpha, scale, digest)]
