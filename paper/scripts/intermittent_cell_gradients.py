"""Same-forward, cell-level gradient attribution for immutable saved weights.

Only batch summaries are serializable. Detached component vectors are transient
inputs to the streaming accumulator; they are never per-example gradients.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Callable

import numpy as np
import torch


COMPONENTS = ("time", "log", "raw_scaled")
LOSS_KEYS = ("time_loss", "log_qty_loss", "raw_qty_loss")
GROUPS = ("all", "encoder", "time_head", "quantity_head")
ABS_L2_TOLERANCE = 1e-6
REL_L2_TOLERANCE = 1e-4
_MISSING = object()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value, label):
    _require(bool(torch.isfinite(value).all()), f"Nonfinite {label}")


def _state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        metadata = json.dumps({"name": name, "dtype": str(value.dtype),
                               "shape": list(value.shape)}, sort_keys=True,
                              separators=(",", ":")).encode()
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


@contextmanager
def preserved_rng():
    """Restore process RNGs, without initializing CUDA in a CPU process."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state().clone()
    cuda_initialized = torch.cuda.is_initialized()
    cuda_states = torch.cuda.get_rng_state_all() if cuda_initialized else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        _require(cuda_initialized == torch.cuda.is_initialized(),
                 "CUDA initialization changed inside RNG preservation scope")


@contextmanager
def preserved_model_state(model):
    """Restore modes/flags/RNG and reject any persistent-state or .grad write.

Context memory is an ordinary attribute, absent from state_dict, so its empty
static configuration is checked separately. Synthetic modules may omit it.
"""
    modules = list(model.modules())
    modes = [(module, module.training) for module in modules]
    memory = []
    for module in modules:
        for name, expected in (("_ctx_mem", None), ("contextual_mem_size", 0),
                               ("use_context_update", False)):
            actual = getattr(module, name, _MISSING)
            if actual is not _MISSING:
                ok = actual is None if name == "_ctx_mem" else actual == expected
                _require(ok, f"Static probe requires {name}={expected!r}")
            memory.append((module, name, actual))
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    for name, value in state.items():
        _finite(value, f"model state {name}")
    state_hash = _state_hash(state)
    parameters = [(parameter, parameter.requires_grad, parameter.grad,
                   None if parameter.grad is None else parameter.grad.detach().clone())
                  for parameter in model.parameters()]
    with preserved_rng():
        try:
            yield {"state_sha256": state_hash}
        finally:
            mutations = []
            try:
                if _state_hash(model.state_dict()) != state_hash:
                    mutations.append("state_dict")
                for module, name, before in memory:
                    after = getattr(module, name, _MISSING)
                    if before is _MISSING:
                        unchanged = after is _MISSING
                    elif name == "_ctx_mem":
                        unchanged = after is None
                    else:
                        unchanged = after == before
                    if not unchanged:
                        mutations.append(name)
                    if before is _MISSING:
                        if after is not _MISSING:
                            delattr(module, name)
                    else:
                        setattr(module, name, before)
                for parameter, flag, gradient, saved_gradient in parameters:
                    same = parameter.grad is gradient
                    if saved_gradient is not None:
                        same = same and torch.equal(parameter.grad, saved_gradient)
                    if not same:
                        mutations.append("parameter.grad")
                    parameter.requires_grad_(flag)
                    parameter.grad = gradient
                    if gradient is not None:
                        with torch.no_grad():
                            gradient.copy_(saved_gradient)
                if "state_dict" in mutations:
                    model.load_state_dict(state, strict=True)
            finally:
                for module, mode in modes:
                    module.training = mode
            _require(not mutations, "Probe mutated model state; restored: " + ", ".join(mutations))


def _layout(named):
    offset = 0
    result = []
    for name, parameter in named:
        group = ("time_head" if name in {"b_t", "w_raw"} or name.startswith("v_t.")
                 else "quantity_head" if name.startswith("quantity_head.") else "encoder")
        result.append((name, tuple(parameter.shape), offset, offset + parameter.numel(), group))
        offset += parameter.numel()
    return tuple(result)


def _group_vectors(vector, layout):
    groups = {"all": vector}
    for group in GROUPS[1:]:
        slices = [vector[start:end] for _, _, start, end, owner in layout if owner == group]
        groups[group] = torch.cat(slices) if slices else vector[:0]
    return groups


def _tasks(components, alpha, scale):
    time, log, raw = components.unbind(0)
    weighted_log, weighted_raw = scale * log, (scale * alpha) * raw
    quantity = weighted_log + weighted_raw
    return {"time": time, "log": log, "raw_scaled": raw, "weighted_log": weighted_log,
            "weighted_raw": weighted_raw, "quantity": quantity, "joint": time + quantity}


def _inner(left, right):
    return float(torch.dot(left, right).item())


def _alignment(left, right):
    dot = _inner(left, right)
    denominator = math.sqrt(_inner(left, left) * _inner(right, right))
    return {"dot": dot, "cosine": dot / denominator if denominator > 0 else None}


def _statistics(components, reference, layout, alpha, scale, *, empty=False):
    tasks = {name: _group_vectors(vector, layout) for name, vector in _tasks(components, alpha, scale).items()}
    reference = _group_vectors(_tasks(reference, alpha, scale)["joint"], layout)
    result = {}
    for group in GROUPS:
        vectors = {name: values[group] for name, values in tasks.items()}
        squared = {name: _inner(vector, vector) for name, vector in vectors.items()}
        denominator = _inner(reference[group], reference[group])
        dots = {name: _inner(vectors[name], reference[group])
                for name in ("joint", "raw_scaled", "weighted_raw")}
        result[group] = {
            "norms": {name: math.sqrt(value) for name, value in squared.items()},
            "squared_norms": squared,
            "time_quantity": _alignment(vectors["time"], vectors["quantity"]),
            "log_raw": _alignment(vectors["log"], vectors["raw_scaled"]),
            "joint_all_dot": dots["joint"],
            "raw_joint_all_dot": dots["raw_scaled"],
            "weighted_raw_joint_all_dot": dots["weighted_raw"],
            "joint_projection": dots["joint"] / denominator if denominator > 0 and not empty else None,
            "raw_joint_projection": dots["raw_scaled"] / denominator if denominator > 0 and not empty else None,
            "weighted_raw_joint_projection": dots["weighted_raw"] / denominator if denominator > 0 and not empty else None,
        }
    return result


def _audit(actual, expected, label):
    absolute = float(torch.linalg.vector_norm(actual - expected).item())
    reference = float(torch.linalg.vector_norm(expected).item())
    relative = absolute / reference if reference > 0 else None
    passed = absolute <= ABS_L2_TOLERANCE or (relative is not None and relative <= REL_L2_TOLERANCE)
    _require(passed, f"Gradient reconstruction failed: {label}: absolute={absolute}, relative={relative}")
    return {"passed": True, "absolute_l2_residual": absolute, "relative_l2_residual": relative}


def _loss_record(sums, count, alpha, scale):
    time, log, raw = (float(x) for x in sums)
    values = {"time": time, "log": log, "raw_scaled": raw,
              "weighted_log": scale * log, "weighted_raw": scale * alpha * raw,
              "quantity": scale * (log + alpha * raw),
              "joint": time + scale * (log + alpha * raw)}
    return {"loss_sums": values,
            "loss_means": {name: value / count if count else None for name, value in values.items()}}


def _rows(vectors, clipped, counts, losses, layout, alpha, scale):
    rows = []
    for index, (components, post_components, count, sums) in enumerate(zip(vectors, clipped, counts, losses, strict=True)):
        rows.append({"cell_index": None if index == 0 else index - 1, "count": int(count),
                     **_loss_record(sums, int(count), alpha, scale),
                     "preclip": _statistics(components, vectors[0], layout, alpha, scale, empty=count == 0),
                     "postclip": _statistics(post_components, clipped[0], layout, alpha, scale, empty=count == 0)})
    return rows


@dataclass
class ProbeResult:
    summary: dict
    _vectors: torch.Tensor
    _layout: tuple
    _loss_sums: torch.Tensor
    _counts: tuple
    _alpha: float
    _scale: float


def probe_batch(model, forward_fn: Callable, *, alpha, quantity_scale,
                cell_indices, cell_count=15, budget_callback=None):
    """One whole-batch graph, at most 3*(15+1)+1 autograd calls.

    ``cell_indices`` may be an integer tensor or a pure outputs->indices callable.
    Caller sets train mode and the prescribed dropout seed before this call.
    """
    _require(model.training, "Train-mode gradient probe required")
    _require(all(module.training for module in model.modules()
                 if isinstance(module, torch.nn.modules.dropout._DropoutNd)),
             "All dropout modules must remain in train mode")
    _require(type(cell_count) is int and 0 < cell_count <= 15, "Cell count must be 1..15")
    alpha, quantity_scale = float(alpha), float(quantity_scale)
    _require(math.isfinite(alpha) and alpha >= 0 and math.isfinite(quantity_scale) and quantity_scale > 0,
             "Invalid frozen objective coefficients")
    budget = budget_callback or (lambda: None)
    named = list(model.named_parameters())
    _require(bool(named), "Probe model has no parameters")
    layout = _layout(named)
    parameters = [parameter for _, parameter in named]
    calls = 0
    with preserved_model_state(model) as preservation, torch.enable_grad():
        for parameter in parameters:
            parameter.requires_grad_(True)
        budget()
        outputs = forward_fn()
        budget()
        _require(isinstance(outputs, dict), "Forward must return frozen output mapping")
        losses = [outputs[key] for key in LOSS_KEYS]
        actual_objective = outputs["objective_loss"]
        count = losses[0].numel()
        _require(count > 0 and all(value.ndim == 1 and value.numel() == count for value in (*losses, actual_objective)),
                 "Losses must be aligned nonempty per-target vectors")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                _finite(value, f"forward output {key}")
        expected_objective = losses[0] + quantity_scale * (losses[1] + alpha * losses[2])
        _require(torch.allclose(actual_objective, expected_objective, atol=ABS_L2_TOLERANCE, rtol=REL_L2_TOLERANCE),
                 "Frozen mixed objective values disagree with coefficients")
        indices = cell_indices(outputs) if callable(cell_indices) else cell_indices
        _require(isinstance(indices, torch.Tensor) and indices.ndim == 1 and indices.numel() == count
                 and indices.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8),
                 "Cell indices must be aligned integer tensor")
        indices = indices.to(losses[0].device)
        _require(bool(((indices >= 0) & (indices < cell_count)).all()), "Cell index outside fixed partition")

        def gradient(loss):
            nonlocal calls
            budget()
            calls += 1
            _require(calls <= 49, "Autograd call budget exceeded")
            grads = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
            vector = torch.cat([(torch.zeros_like(parameter) if grad is None else grad).detach().reshape(-1).to(device="cpu", dtype=torch.float64)
                                for parameter, grad in zip(parameters, grads, strict=True)])
            _finite(vector, "gradient")
            budget()
            return vector

        full = torch.stack([gradient(loss.mean()) for loss in losses])
        actual_joint = gradient(actual_objective.mean())
        vectors = torch.zeros((cell_count + 1, 3, full.shape[-1]), dtype=torch.float64)
        vectors[0] = full
        counts, sums = [count], [torch.tensor([float(loss.detach().double().sum()) for loss in losses], dtype=torch.float64)]
        for cell in range(cell_count):
            mask = indices == cell
            n = int(mask.sum().item())
            counts.append(n)
            sums.append(torch.tensor([float(loss[mask].detach().double().sum()) for loss in losses], dtype=torch.float64))
            if n:
                vectors[cell + 1] = torch.stack([gradient(loss[mask].sum() / count) for loss in losses])
        conservation = {name: _audit(vectors[1:, component].sum(0), full[component], name)
                        for component, name in enumerate(COMPONENTS)}
        conservation["actual_objective"] = _audit(_tasks(full, alpha, quantity_scale)["joint"], actual_joint, "actual objective")
        joint_norm = float(torch.linalg.vector_norm(_tasks(full, alpha, quantity_scale)["joint"]).item())
        clip_factor = min(1.0, 1.0 / (joint_norm + 1e-6))
        losses_cpu = torch.stack(sums)
        rows = _rows(vectors, vectors * clip_factor, counts, losses_cpu, layout, alpha, quantity_scale)
        summary = {"count": count, "cell_count": cell_count, "autograd_calls": calls, "forward_calls": 1,
                   "model_state_sha256": preservation["state_sha256"],
                   "alpha": alpha, "quantity_scale": quantity_scale,
                   "global_joint_preclip_norm": joint_norm, "common_clip_factor": clip_factor,
                   "clipped": clip_factor < 1.0, "norm_above_max": joint_norm > 1.0,
                   "overall": rows[0], "cells": rows[1:],
                   "conservation": {"passed": True, "checks": conservation},
                   "gradient_normalization": "cell loss sum / original batch target count",
                   "clipping_meaning": "hypothetical common full-joint coefficient; clipped means k<1, norm_above_max means norm>1; zero optimizer updates"}
        budget()
        json.dumps(summary, allow_nan=False)
    summary["immutability"] = {"passed": True, "model_state": True, "context": True,
                               "existing_gradients": True, "modes_flags": True, "rng": True}
    return ProbeResult(summary, vectors, layout, losses_cpu, tuple(counts), alpha, quantity_scale)


class GradientAccumulator:
    """Sample-weighted running vectors and gradient energies, never graph storage."""

    def __init__(self):
        self.count = 0
        self.batches = 0
        self._signature = None
        self._pre = self._post = self._losses = self._energy = self._post_energy = None
        self._counts = None
        self.autograd_calls = 0
        self.clipped_batches = 0

    def add(self, result: ProbeResult, count=None):
        n = result.summary["count"]
        _require(count is None or type(count) is int and count == n, "Accumulator batch count mismatch")
        _require(type(n) is int and n > 0 and result.summary["conservation"]["passed"]
                 and result.summary["immutability"]["passed"], "Invalid batch probe receipt")
        signature = (result._layout, result._alpha, result._scale, result._vectors.shape,
                     result.summary["model_state_sha256"])
        if self._signature is None:
            self._signature = signature
            self._pre = torch.zeros_like(result._vectors)
            self._post = torch.zeros_like(result._vectors)
            self._losses = torch.zeros_like(result._loss_sums)
            self._counts = [0] * len(result._counts)
            self._energy = self._energies(result._vectors, result)
            self._energy.zero_()
            self._post_energy = torch.zeros_like(self._energy)
        _require(signature == self._signature, "Accumulator parameter/objective/partition identity drift")
        _require(result._vectors.device.type == "cpu" and not result._vectors.requires_grad, "Only detached CPU batch vectors allowed")
        _finite(result._vectors, "accumulator input")
        k = result.summary["common_clip_factor"]
        self._pre.add_(result._vectors, alpha=n)
        self._post.add_(result._vectors, alpha=n * k)
        self._losses.add_(result._loss_sums)
        self._counts = [a + b for a, b in zip(self._counts, result._counts, strict=True)]
        energy = self._energies(result._vectors, result)
        self._energy.add_(energy, alpha=n)
        self._post_energy.add_(energy, alpha=n * k * k)
        self.count += n
        self.batches += 1
        self.autograd_calls += result.summary["autograd_calls"]
        self.clipped_batches += int(result.summary["clipped"])

    @staticmethod
    def _energies(vectors, result):
        return torch.tensor([[[float(values[group].square().sum()) for group in GROUPS]
                              for values in (_group_vectors(v, result._layout)
                                             for v in _tasks(components, result._alpha, result._scale).values())]
                             for components in vectors], dtype=torch.float64)

    def finalize(self):
        _require(self.count > 0, "Cannot finalize an empty gradient accumulator")
        layout, alpha, scale, shape, state_hash = self._signature
        pre, post = self._pre / self.count, self._post / self.count
        _finite(pre, "mean gradient")
        _finite(post, "postclip mean gradient")
        checks = {}
        for tag, values in (("preclip", pre), ("postclip", post)):
            for component, name in enumerate(COMPONENTS):
                checks[f"{tag}_{name}"] = _audit(values[1:, component].sum(0), values[0, component], f"{tag} {name}")
        rows = _rows(pre, post, self._counts, self._losses, layout, alpha, scale)
        names = tuple(_tasks(pre[0], alpha, scale))
        for index, row in enumerate(rows):
            row["gradient_energy"] = {}
            for tag, energy in (("preclip", self._energy), ("postclip", self._post_energy)):
                row["gradient_energy"][tag] = {
                    group: {"mean_batch_squared_norm": {name: float(energy[index, task, g] / self.count) for task, name in enumerate(names)},
                            "squared_norm_of_mean_gradient": row[tag][group]["squared_norms"]}
                    for g, group in enumerate(GROUPS)}
        result = {"count": self.count, "batches": self.batches, "autograd_calls": self.autograd_calls,
                  "model_state_sha256": state_hash,
                  "clipped_batches": self.clipped_batches, "alpha": alpha, "quantity_scale": scale,
                  "overall": rows[0], "cells": rows[1:],
                  "parameter_groups": {group: [name for name, _, _, _, owner in layout if group == "all" or group == owner] for group in GROUPS},
                  "conservation": {"passed": True, "checks": checks},
                  "aggregation": {"gradient": "sum(batch_count * batch_gradient) / total_count",
                                  "postclip_gradient": "sum(batch_count * common_batch_clip_factor * batch_gradient) / total_count",
                                  "energy": "mean_batch_squared_norm is E[||g||^2]; squared_norm_of_mean_gradient is ||E[g]||^2",
                                  "projections": "computed from running vector sums, never averages of batch projections"}}
        json.dumps(result, allow_nan=False)
        return result
