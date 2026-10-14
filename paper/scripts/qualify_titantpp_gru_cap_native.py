"""Fabricated-input native checks; no dataset reads or scientific fits."""
from __future__ import annotations


def require(ok, message):
    if not ok:
        raise ValueError(message)


def synthetic_check(runner, data, seed, *, device="cuda:0", batch_size=128):
    import torch
    from models.TPPs.CountAwareTitanCNNGRUResidualCap import cap_gru_residual
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs

    require(seed in (42, 52, 62) and (device == "cpu" or device == "cuda:0"), "Foreign qualification input")
    length = data["loader"]["max_seq_len"]
    require(batch_size == 128 or device == "cpu", "Native qualification must use actual batch128")
    if device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    devices = [] if device == "cpu" else [0]
    with torch.random.fork_rng(devices=devices):
        # The full initial-state comparison also checks constructor RNG and
        # every learned tensor. Only the dedicated identity buffer differs.
        initials = runner.initial_states(data, seed)
        generator = torch.Generator().manual_seed(831)
        dt = torch.randint(1, 8, (batch_size, length), generator=generator).float().to(device)
        qty = torch.randint(0, 40, (batch_size, length), generator=generator).float().to(device)
        dt[:, 0] = 0.
        mask = torch.ones_like(dt, dtype=torch.bool)
        torch.manual_seed(seed)
        full, _ = runner.build_model(data, runner.FULL)
        full.to(device).eval()
        full_parameters = {k: p.detach().clone() for k, p in full.named_parameters()}
        with torch.no_grad():
            reference = target_outputs(full, dt, mask, qty, lambda_log_qty=1.)
        torch.manual_seed(seed)
        candidate, _ = runner.build_model(data, runner.CAPPED)
        candidate.to(device).eval()
        require(set(dict(candidate.named_parameters())) == set(full_parameters), "Candidate parameter names differ")
        require(all(torch.equal(p, full_parameters[k]) for k, p in candidate.named_parameters()),
                "Candidate initial learned tensors differ from FULL")
        with torch.no_grad():
            output = target_outputs(candidate, dt, mask, qty, lambda_log_qty=1.)
        require(reference.keys() == output.keys() and all(torch.equal(reference[k], output[k]) for k in output),
                "Native initial FULL/CAPPED output differs")
        parameter_count = sum(p.numel() for p in candidate.parameters())
        active_count = sum(p.numel() for p in candidate.parameters() if p.requires_grad)
        require(active_count == sum(p.numel() for p in full.parameters() if p.requires_grad), "FULL/candidate active parameter mismatch")
        del full, full_parameters, reference, output

        # Exercise the cap directly beyond its zero-initialized projection.
        for level in (0., .01, 10.):
            base = torch.ones((2, 3, 64), device=device)
            residual = torch.full_like(base, level, requires_grad=True)
            limited = cap_gru_residual(base, residual)
            require(bool((torch.linalg.vector_norm(limited.double(), dim=-1)
                    <= .25 * torch.linalg.vector_norm(base.double(), dim=-1)).all()), "Native cap bound violated")
            if level <= .01:
                require(torch.equal(limited, residual), "Unsaturated residual changed")
            limited.sum().backward()
            require(residual.grad is not None and bool(torch.isfinite(residual.grad).all()), "Native cap gradient nonfinite")
        base.zero_()
        require(bool((cap_gru_residual(base, residual.detach()) == 0).all()), "Zero base cap nonzero")

        candidate.train()
        corrections = runner.correction_parameters(candidate)
        before = {k: p.detach().clone() for k, p in corrections.items()}
        active = set()
        optimizer = torch.optim.AdamW(candidate.parameters(), lr=.001, weight_decay=.01)
        for step in range(4):
            optimizer.zero_grad(set_to_none=True)
            torch.manual_seed(seed + 100 + step)
            loss = target_outputs(candidate, dt, mask, qty, lambda_log_qty=1.)["joint_loss"].mean()
            require(bool(torch.isfinite(loss)), "Nonfinite native synthetic loss")
            loss.backward()
            require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in candidate.parameters()),
                    "Nonfinite native synthetic gradient")
            active.update(k for k, p in corrections.items() if p.grad is not None and bool(p.grad.ne(0).any()))
            norm = torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.)
            require(bool(torch.isfinite(norm)), "Nonfinite native clipped gradient")
            optimizer.step()
        require(active == set(corrections), "Inactive native correction parameter")
        require(all(bool(torch.isfinite(p).all()) for p in candidate.parameters()), "Nonfinite native parameter")
        require(all(not torch.equal(before[k], p) for k, p in corrections.items()), "Native correction did not update")

        # Same-shape masking and causal prefix checks after nonzero updates.
        candidate.eval()
        short_dt, short_qty = dt[:1, :6].clone(), qty[:1, :6].clone()
        valid = torch.ones_like(short_dt, dtype=torch.bool)
        observed = valid.clone(); observed[:, -1] = False
        with torch.no_grad():
            state = candidate.encode_task_states(short_dt, short_qty, valid, memory_write_mask=observed)[0]
            altered_dt, altered_qty = short_dt.clone(), short_qty.clone()
            altered_dt[:, -1], altered_qty[:, -1] = float("nan"), float("inf")
            withheld = candidate.encode_task_states(altered_dt, altered_qty, valid, memory_write_mask=observed)[0]
            require(torch.equal(state, withheld) and bool((state[:, -1] == 0).all()), "Withheld target leaked")
            altered_dt, altered_qty = short_dt.clone(), short_qty.clone()
            altered_dt[:, 4], altered_qty[:, 4] = 1000., 9999.
            future = candidate.encode_task_states(altered_dt, altered_qty, valid, memory_write_mask=observed)[0]
            require(torch.equal(state[:, :4], future[:, :4]), "Future input changed causal prefix")
        peak = None
        if device != "cpu":
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            require(peak <= .8 * torch.cuda.get_device_properties(0).total_memory, "Native GPU allocation exceeds80percent")
        result = {"status": "passed", "device": device, "seed": seed,
            "shape": [batch_size, length], "initial_state_sha256": initials,
            "initial_FULL_candidate_tensors_outputs": "exact", "registered_parameters": parameter_count,
            "active_parameters": active_count, "cap_bound_zero_unsaturated_saturated": "passed",
            "all_gradients_finite": True, "correction_parameter_names_with_gradient": sorted(active),
            "withheld_target": "invariant", "future_prefix": "invariant",
            "synthetic_optimizer_updates": 4, "scientific_optimizer_updates": 0,
            "peak_allocated_bytes": peak, "real_data_loaded": False, "held_out_test_evaluated": False}
        del candidate, optimizer, loss, before
    if device != "cpu":
        torch.cuda.empty_cache()
    return result
