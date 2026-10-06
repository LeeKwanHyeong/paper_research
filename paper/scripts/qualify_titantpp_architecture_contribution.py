"""Local/native synthetic qualification of the six architecture interventions.

No dataset, checkpoint, Test split, SSH or training campaign is opened here.
The two optimizer steps are synthetic diagnostic updates in this process only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time


def tensor_sha(tensors):
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def qualify(device="cpu", seeds=(42, 52, 62)):
    import torch
    from torch.nn import functional as F
    from models.TPPs.CountAwareTitanArchitectureContribution import (
        ARMS, MODULES, CountAwareTitanArchitectureContribution, CurrentBudgetCorrection,
        expected_intervention_shapes, metadata,
    )
    from models.TPPs.CountAwareTitanHistoryWidth import CountAwareTitanHistoryWidth

    torch.set_num_threads(1)
    device = torch.device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Native CUDA requested but unavailable")
        torch.cuda.reset_peak_memory_stats(device)
    kwargs = dict(time_head_mode="heteroscedastic_lognormal_duration", time_scale=1.,
                  time_initial_location=.2, time_initial_scale=.6,
                  time_observation_contract={"mode": "positive_integer_round_clamp_v1", "unit": "hour", "top_code": None})
    def model(arm, seed):
        torch.manual_seed(seed)
        return CountAwareTitanArchitectureContribution(64, 2., 256, architecture_arm=arm, **kwargs).to(device)
    def inputs(length=160):
        dt = (torch.arange(length, device=device).float() % 7 + 1).repeat(2, 1)
        q = ((torch.arange(length, device=device).float() * 13) % 197 + 1).repeat(2, 1)
        q[1] = q[1].flip(0)
        return dt, q, torch.ones_like(dt, dtype=torch.bool)
    x, q, valid = inputs()
    parity = []
    for seed in seeds:
        hashes, rngs = [], []
        for arm in ARMS:
            m = model(arm, seed)
            common = {k: v for k, v in m.state_dict().items()
                      if not k.startswith(("lmm.", "multilag_detail.")) and k != "architecture_contribution_identity"}
            hashes.append(tensor_sha(common))
            rngs.append(torch.random.get_rng_state().clone())
            if device.type == "cuda":
                rngs[-1] = torch.cat([rngs[-1], torch.cuda.get_rng_state(device).cpu()])
            m.train()
            torch.manual_seed(seed + 1000)
            m.encode_task_states(x, q, valid)
            after = torch.random.get_rng_state().clone()
            if device.type == "cuda":
                after = torch.cat([after, torch.cuda.get_rng_state(device).cpu()])
            if arm == ARMS[0]:
                dropout_rng = after
            elif not torch.equal(dropout_rng, after):
                raise AssertionError("Optional intervention changed common dropout RNG stream")
            del m
        if len(set(hashes)) != 1 or not all(torch.equal(rngs[0], r) for r in rngs):
            raise AssertionError("Common tensor initialization or post-construction RNG mismatch")
        parity.append({"seed": seed, "common_tensor_sha256": hashes[0], "common_initialization_equal": True,
                       "post_initialization_rng_equal": True, "common_dropout_rng_equal": True})

    arms = []
    for arm in ARMS:
        m = model(arm, seeds[0])
        named = dict(m.named_parameters())
        shapes = expected_intervention_shapes(arm)
        actual_optional = {k: tuple(p.shape) for k, p in named.items() if k.startswith(("lmm.", "multilag_detail."))}
        if actual_optional != shapes:
            raise AssertionError("Actual optional parameter presence/shape mismatch")
        budget = sum(named[k].numel() for k in shapes)
        if budget != metadata(arm=arm)["intervention_parameter_count"]:
            raise AssertionError("Intervention budget mismatch")
        persistent = sum(p.numel() for k, p in named.items() if "persistent" in k)
        if persistent != 2048:
            raise AssertionError("Persistent encoder memory changed")
        optimizer = torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=.01)
        participating = set()
        participation_norms = {}
        prototype_rows = set()
        for update in range(2):
            torch.manual_seed(2000 + update)
            optimizer.zero_grad(set_to_none=True)
            m.train()
            t, h = m.encode_task_states(x, q, valid)
            log_q, _ = m.predict_quantity(h)
            loss = -m.log_observation_dt(t.flatten(0, 1), x.flatten()).mean() + F.mse_loss(log_q, torch.log1p(q))
            if not bool(torch.isfinite(loss)):
                raise AssertionError("Nonfinite synthetic combined loss")
            loss.backward()
            for name in shapes:
                grad = named[name].grad
                if grad is None or not bool(torch.isfinite(grad).all()):
                    raise AssertionError("Missing/nonfinite intervention gradient: " + name)
                norm = float(grad.norm().item())
                participation_norms[name] = max(participation_norms.get(name, 0.), norm)
                if norm > 0:
                    participating.add(name)
                if name == "lmm.mem":
                    prototype_rows.update(torch.nonzero(grad.abs().sum(-1)[0] > 0).flatten().cpu().tolist())
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
            optimizer.step()
        if participating != set(shapes):
            raise AssertionError("Unused intervention parameters after two synthetic updates: " + repr(set(shapes) - participating))
        m.eval()
        with torch.no_grad():
            before = {k: v.clone() for k, v in m.state_dict().items()}
            t, h = m.encode_task_states(x, q, valid)
            t2, h2 = m.encode_task_states(x, q, valid)
            if not torch.equal(t, t2) or not torch.equal(h, h2):
                raise AssertionError("Native repeated-eval nondeterminism")
            changed_x, changed_q = x.clone(), q.clone()
            changed_x[:, 81:] *= 3
            changed_q[:, 81:] += 97
            ft, fh = m.encode_task_states(changed_x, changed_q, valid)
            if not torch.equal(h[:, :81], fh[:, :81]) or not torch.equal(t[:, :81], ft[:, :81]):
                raise AssertionError("Future event affected an earlier state")
            mask = valid.clone()
            mask[:, -7:] = False
            xt, qt = x.clone(), q.clone()
            xt[:, -7:] = float("nan")
            qt[:, -7:] = float("nan")
            p_t, p_h = m.encode_task_states(xt, qt, mask)
            plain_t, plain_h = m.encode_task_states(x, q, mask)
            if not torch.equal(p_h, plain_h) or not torch.equal(p_t, plain_t):
                raise AssertionError("Padding influenced observed states")
            if not bool(torch.isfinite(p_h).all()) or bool(p_h[:, -7:].count_nonzero()):
                raise AssertionError("Invalid padding output")
            if any(not torch.equal(before[k], v) for k, v in m.state_dict().items()):
                raise AssertionError("Evaluation mutated a weight or buffer")
            if arm == "P1_H1":
                torch.manual_seed(seeds[0])
                reference = CountAwareTitanHistoryWidth(64, 2., 256, history_mlp_width=8, **kwargs).to(device).eval()
                state = reference.state_dict()
                for key in state:
                    if key in m.state_dict():
                        state[key] = m.state_dict()[key].clone()
                reference.load_state_dict(state)
                rt, rh = reference.encode_task_states(x, q, valid)
                if not torch.equal(h, rh) or not torch.equal(t, rt):
                    raise AssertionError("P1_H1 differs from frozen width8 after nonzero updates")
        arms.append({"arm": arm, "total_parameters": sum(p.numel() for p in named.values()),
                     "persistent_parameters": persistent, "intervention_parameters": budget,
                     "participating_parameter_tensors_after_two_updates": len(participating),
                     "intervention_gradient_max_norms": participation_norms,
                     "prototype_rows_with_nonzero_gradient": sorted(prototype_rows),
                     "finite": True, "causal": True, "deterministic_eval": True,
                     "padding_invariant": True, "eval_state_unchanged": True,
                     "P1_H1_frozen_width8_exact_after_updates": arm == "P1_H1"})
        del m, optimizer

    # Eligibility/reset are inherited identically by H1 and current-state HC.
    from models.TPPs.CountAwareTitanHistoryWidth import HistoryWidthCorrection
    reset_results = []
    for correction in (HistoryWidthCorrection(64, width=8), CurrentBudgetCorrection()):
        correction = correction.to(device)
        with torch.no_grad():
            for p in correction.output_projections:
                p.weight.fill_(.01)
        z = torch.arange(2 * 132 * 64, device=device).float().reshape(2, 132, 64) / 8192
        mask = torch.ones((2, 132), device=device, dtype=torch.bool)
        write = mask.clone()
        write[:, 65] = False
        perturbed = z.clone()
        perturbed[:, :66] += 5
        a = correction(z, mask, memory_write_mask=write)
        b = correction(perturbed, mask, memory_write_mask=write)
        if not torch.equal(a[:, 66:], b[:, 66:]) or bool(a[:, 65:67].count_nonzero()):
            raise AssertionError("Correction reset permits a pre-reset predecessor")
        raf_mask = torch.ones((2, 128), device=device, dtype=torch.bool)
        correction.zero_grad(set_to_none=True)
        correction(z[:, :128], raf_mask).square().sum().backward()
        if bool(correction.input_projections[7].weight.grad.count_nonzero()) or bool(correction.output_projections[7].weight.grad.count_nonzero()):
            raise AssertionError("Inherited RAF branch128 should be ineligible")
        reset_results.append({"correction": type(correction).__name__, "withheld_segment_reset": True,
                              "RAF_branch128_ineligible": True, "RAF_nominal_eligible_parameters": 10752})
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return {"schema": "titantpp_architecture_synthetic_qualification_v1", "status": "passed",
            "device": str(device), "torch_version": torch.__version__, "initialization": parity,
            "arms": arms, "correction_reset": reset_results, "peak_cuda_allocated_bytes": peak,
            "process_max_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "synthetic_optimizer_updates_per_arm": 2, "real_training_updates": 0,
            "held_out_test_evaluated": False, "data_population_gate": "separate_required"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="42,52,62")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    started = time.time()
    result = qualify(args.device, tuple(int(x) for x in args.seeds.split(",")))
    result["elapsed_seconds"] = time.time() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output), "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
