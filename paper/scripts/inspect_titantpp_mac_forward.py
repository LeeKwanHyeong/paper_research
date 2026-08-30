#!/usr/bin/env python3
"""Locate the first non-finite stage in a trusted, locally captured forward."""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper.scripts.count_aware_tpp_backbone.core import target_outputs


def tensors(value, prefix=""):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif is_dataclass(value):
        for field in fields(value):
            yield from tensors(getattr(value, field.name), prefix + "." + field.name)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensors(item, prefix + "." + str(key))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensors(item, prefix + "." + str(index))


def summarize(value):
    result = []
    for name, tensor in tensors(value):
        if not tensor.numel() or not tensor.is_floating_point():
            continue
        finite = torch.isfinite(tensor)
        bad_rows = (~finite).reshape(tensor.size(0), -1).any(dim=1) if tensor.ndim else ~finite.reshape(1)
        result.append({"name": name, "shape": list(tensor.shape),
                       "nonfinite_count": int((~finite).sum()),
                       "bad_rows": bad_rows.nonzero().flatten().tolist(),
                       "finite_abs_max": float(tensor[finite].detach().abs().max()) if finite.any() else None})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--inner-grad-clip", type=float)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    # Only use fixtures emitted by our own diagnostic wrapper.
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=False)
    model = fixture["model"].to(args.device)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.accumulated_recompile_limit = 512
    memory = model.titans_mac_encoder.neural_memory
    memory.gradient_max_norm = args.inner_grad_clip
    if args.eager:
        memory.compile_cuda_scan = False
    events = []

    def check(name, value):
        stats = summarize(value)
        events.append({"stage": name, "tensors": stats})
        if any(item["nonfinite_count"] for item in stats):
            raise FloatingPointError("First non-finite stage: " + name)

    for name, module in model.named_modules():
        if not list(module.children()):
            module.register_forward_hook(lambda mod, inputs, output, name=name: check(name, output))

    for name in ("read", "_project_write", "associative_gradients", "write_token", "write_sequence"):
        original = getattr(memory, name)
        def observed(*positional, _name=name, _original=original, **kwargs):
            value = _original(*positional, **kwargs)
            check("memory." + _name, value)
            return value
        setattr(memory, name, observed)
    torch.set_rng_state(fixture["cpu_rng"])
    if args.device == "cuda":
        torch.cuda.set_rng_state_all(fixture["cuda_rng"])
    status = "finite"
    error = None
    try:
        outputs = target_outputs(model, fixture["dts"].to(args.device),
                                 fixture["mask"].to(args.device),
                                 fixture["quantities"].to(args.device), **fixture["kwargs"])
        check("target_outputs", outputs)
        if args.backward:
            outputs["joint_loss"].mean().backward()
            check("backward", {name: p.grad for name, p in model.named_parameters()})
    except FloatingPointError as exc:
        status, error = "nonfinite", str(exc)
    report = {"status": status, "error": error, "eager": args.eager,
              "batch_index": fixture["batch_index"], "events": events}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({**report, "events": events[-3:]}), flush=True)


if __name__ == "__main__":
    main()
