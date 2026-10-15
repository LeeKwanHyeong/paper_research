"""Read-only frozen-CNN memory attenuation instrumentation.

This module neither loads datasets/checkpoints nor trains/evaluates a model.
The caller must separately bind the original source, checkpoint and split.  It
intercepts one *existing* HardLocalMemoryMatcher instance, calls its original
retrieval exactly once, and changes only the final additive residual multiplier.
No targets, predictions, learned gates, normalizations or checkpoint choices are
inputs to this API. Telemetry positions include padding; masking/alignment is the
caller's responsibility.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Real
from types import MethodType
from typing import Callable, Iterator
from weakref import WeakSet

import torch
from torch import nn


ALLOWED_ALPHAS = (0.0, 0.25, 0.5, 1.0)
_ACTIVE: WeakSet = WeakSet()
_HOOK_NAMES = (
    "_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks",
)
Telemetry = dict[str, torch.Tensor | float | int]


@dataclass
class MemoryAttenuationAudit:
    alpha: float
    records: list[Telemetry] = field(default_factory=list)
    call_count: int = 0
    state_sha_before: str | None = None
    state_sha_after: str | None = None
    restored: bool = False


def memory_state_sha256(module: nn.Module) -> str:
    """Hash parameters and all buffers, including non-persistent buffers."""
    digest = hashlib.sha256()
    for kind, entries in (
        ("parameter", module.named_parameters(recurse=True)),
        ("buffer", module.named_buffers(recurse=True)),
    ):
        for name, value in entries:
            metadata = [kind, name, str(value.dtype), list(value.shape), value.requires_grad]
            digest.update(json.dumps(metadata, separators=(",", ":")).encode())
            raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
    return digest.hexdigest()


def _require_no_hooks(module: nn.Module) -> None:
    if any(getattr(module, name, {}) for name in _HOOK_NAMES):
        raise ValueError("memory matcher hooks must be absent for frozen diagnostics")


def _validate_trace(encoded: torch.Tensor, residual: torch.Tensor, info: dict) -> None:
    if residual.shape != encoded.shape or residual.dtype != encoded.dtype:
        raise ValueError("retrieval residual shape/dtype does not match encoded history")
    indices, scores = info["prototype_indices"], info["topk_similarity"]
    if indices.shape != scores.shape or indices.shape[:2] != encoded.shape[:2]:
        raise ValueError("retrieval trace is not aligned with encoded history")
    if indices.ndim != 3 or indices.dtype != torch.long:
        raise ValueError("retrieval prototype indices must be [batch, position, topk] longs")
    if not all(torch.isfinite(t).all().item() for t in (encoded, residual, scores)):
        raise ValueError("non-finite frozen memory diagnostic input/trace")


def _telemetry(encoded: torch.Tensor, residual: torch.Tensor, info: dict,
               alpha: float, call_index: int) -> Telemetry:
    # Float64 CPU telemetry cannot influence the actual model arithmetic.
    h = encoded.detach().to(device="cpu", dtype=torch.float64)
    m = residual.detach().to(device="cpu", dtype=torch.float64)
    h_norm = torch.linalg.vector_norm(h, dim=-1)
    m_norm = torch.linalg.vector_norm(m, dim=-1)
    denom = h_norm * m_norm
    cosine = torch.where(denom > 0, (h * m).sum(-1) / denom.clamp_min(1e-300), 0.0)
    return {
        "alpha": alpha,
        "call_index": call_index,
        "encoded_norm": h_norm,
        "memory_norm": m_norm,
        "scaled_memory_norm": alpha * m_norm,
        "cosine_h_memory": cosine,
        "prototype_indices": info["prototype_indices"].detach().cpu().clone(),
        "topk_similarity": info["topk_similarity"].detach().cpu().clone(),
    }


@contextmanager
def frozen_memory_attenuation(
    matcher: nn.Module,
    alpha: float,
    telemetry_sink: Callable[[Telemetry], None] | None = None,
) -> Iterator[MemoryAttenuationAudit]:
    """Temporarily attenuate the exact residual in a frozen matcher.

    Use inside ``torch.no_grad()``/``torch.inference_mode()`` with an eval model.
    ``alpha=1`` returns the original forward result without recomputation;
    ``alpha=0`` returns its encoded input, after observing the same retrieval.
    Other allowed alphas compute ``encoded + alpha * exact_residual``. Replacing
    methods instead of subtracting the rounded forward result avoids loss of the
    residual through cancellation. No hooks or state tensors are installed.

    A sink can immediately reduce/write CPU telemetry, avoiding unbounded trace
    retention; otherwise ``audit.records`` keeps each call. The context checks
    matcher parameter/buffer bytes on exit. A mutation is reported, not reverted.
    The caller should also fingerprint the full model before and after inference.
    This per-instance instrumentation is not for concurrent model calls.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise ValueError("alpha must be a finite scalar from the fixed diagnostic grid")
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha not in ALLOWED_ALPHAS:
        raise ValueError(f"alpha must be one of {ALLOWED_ALPHAS}")
    if type(matcher).__name__ != "HardLocalMemoryMatcher":
        raise TypeError("only the original HardLocalMemoryMatcher is supported")
    if matcher.training or torch.is_grad_enabled():
        raise ValueError("frozen diagnostics require eval mode and disabled gradients")
    if matcher in _ACTIVE:
        raise ValueError("nested attenuation on the same matcher is forbidden")
    if "forward" in matcher.__dict__ or "retrieve" in matcher.__dict__:
        raise ValueError("pre-existing matcher method overrides are forbidden")
    _require_no_hooks(matcher)
    original_forward, original_retrieve = matcher.forward, matcher.retrieve
    original_config = (matcher.training, matcher.d_model, matcher.mem_size, matcher.topk)
    audit = MemoryAttenuationAudit(alpha, state_sha_before=memory_state_sha256(matcher))
    pending: list[tuple[torch.Tensor, torch.Tensor, dict]] = []
    inside_forward = False

    def retrieve(this, encoded, memory=None):
        if not inside_forward:
            raise ValueError("diagnostic retrieval must occur inside the original forward")
        residual, info = original_retrieve(encoded, memory=memory)
        pending.append((encoded, residual, info))
        return residual, info

    def forward(this, encoded, memory=None):
        nonlocal inside_forward
        if matcher.training or torch.is_grad_enabled():
            raise ValueError("eval mode and disabled gradients must remain active")
        _require_no_hooks(matcher)
        if inside_forward:
            raise ValueError("recursive/concurrent matcher calls are forbidden")
        if not isinstance(encoded, torch.Tensor) or encoded.ndim != 3:
            raise ValueError("encoded must be an observed-history [batch, position, width] tensor")
        inside_forward = True
        pending.clear()
        try:
            original_output = original_forward(encoded, memory=memory)
            if len(pending) != 1 or pending[0][0] is not encoded:
                raise ValueError("original forward must retrieve exactly once from its input")
            _, residual, info = pending[0]
            _validate_trace(encoded, residual, info)
            if not torch.equal(original_output, encoded + residual):
                raise ValueError("original matcher is not the expected additive residual path")
            record = _telemetry(encoded, residual, info, alpha, audit.call_count)
            if telemetry_sink is None:
                audit.records.append(record)
            else:
                telemetry_sink(record)
            audit.call_count += 1
            if alpha == 1.0:
                return original_output
            if alpha == 0.0:
                return encoded
            return encoded + alpha * residual
        finally:
            pending.clear()
            inside_forward = False

    _ACTIVE.add(matcher)
    try:
        matcher.retrieve = MethodType(retrieve, matcher)
        matcher.forward = MethodType(forward, matcher)
        yield audit
    finally:
        # Restore class-method resolution even after inference or sink exceptions.
        matcher.__dict__.pop("forward", None)
        matcher.__dict__.pop("retrieve", None)
        _ACTIVE.discard(matcher)
        audit.restored = True
        audit.state_sha_after = memory_state_sha256(matcher)
        if audit.state_sha_before != audit.state_sha_after:
            raise RuntimeError("frozen matcher parameters/buffers changed during diagnostics")
        if original_config != (matcher.training, matcher.d_model, matcher.mem_size, matcher.topk):
            raise RuntimeError("frozen matcher mode/configuration changed during diagnostics")
        _require_no_hooks(matcher)
