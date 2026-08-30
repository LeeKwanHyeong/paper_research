#!/usr/bin/env python3
"""Capture an exact train-only failing forward without changing training math."""

from __future__ import annotations

import argparse
import copy
import json
import runpy
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paper.scripts.count_aware_tpp_backbone import training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--probe-fresh-batch", type=int)
    parser.add_argument("runner", type=Path)
    args, runner_args = parser.parse_known_args()
    args.snapshot_dir.mkdir(parents=True, exist_ok=False)
    status_path = args.snapshot_dir / "diagnostic_status.json"
    def status(state, error=None):
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"state": state, "error": error,
                                         "runner_args": runner_args}, indent=2) + "\n")
        temporary.replace(status_path)
    status("running")
    original = training.target_outputs
    batch_index = 0
    started = time.monotonic()

    def capture(model, dts, mask, quantities, **kwargs):
        nonlocal batch_index
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all()
        outputs = original(model, dts, mask, quantities, **kwargs)
        bad = [name for name in ("joint_loss", "time_loss", "quantity_train_loss")
               if not torch.isfinite(outputs[name]).all()]
        if bad or args.probe_fresh_batch is not None:
            # The trusted local fixture includes the class and pre-forward RNG,
            # allowing replay of dropout as well as weights on the same host.
            torch.save({"model": copy.deepcopy(model).cpu(),
                        "dts": dts.cpu(), "mask": mask.cpu(),
                        "quantities": quantities.cpu(), "kwargs": kwargs,
                        "cpu_rng": cpu_rng, "cuda_rng": cuda_rng,
                        "batch_index": batch_index},
                       args.snapshot_dir / "forward_fixture.pt")
            report = {"batch_index": batch_index, "nonfinite_outputs": bad,
                      "elapsed_seconds": time.monotonic() - started,
                      "training_mode": model.training,
                      "input_finite": bool(torch.isfinite(dts).all()
                                           and torch.isfinite(quantities).all()),
                      "parameters_finite": all(bool(torch.isfinite(p).all())
                                               for p in model.parameters()),
                      "argv": runner_args}
            (args.snapshot_dir / "capture.json").write_text(
                json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
        if batch_index % 250 == 0:
            print(f"[diagnostic] batch={batch_index} elapsed={time.monotonic()-started:.1f}s",
                  flush=True)
        batch_index += 1
        return outputs

    training.target_outputs = capture
    if args.probe_fresh_batch is not None:
        def probe(**kw):
            nonlocal batch_index
            loader = kw["loader"]
            # Creating the actual iterator consumes DataLoader's base seed
            # before RandomSampler, exactly as the real train loop does.
            iterator = iter(loader)
            for _ in range(args.probe_fresh_batch + 1):
                indices = next(iterator._sampler_iter)
            _, dts, mask, _, quantities = loader.collate_fn(
                [loader.dataset[index] for index in indices])
            batch_index = args.probe_fresh_batch
            kw["model"].train()
            capture(kw["model"], dts.to(kw["device"]), mask.to(kw["device"]),
                    quantities.to(kw["device"]), lambda_log_qty=kw["lambda_log_qty"])
            raise SystemExit(0)
        training.train_epoch_with_telemetry = probe
    sys.argv = [str(args.runner), *runner_args]
    try:
        runpy.run_path(str(args.runner), run_name="__main__")
    except SystemExit as exc:
        status("complete" if exc.code in (None, 0) else "failed", str(exc))
        raise
    except BaseException as exc:
        status("failed", repr(exc))
        raise
    else:
        status("complete")


if __name__ == "__main__":
    main()
