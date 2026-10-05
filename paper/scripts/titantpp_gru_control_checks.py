"""Native A100 checks on synthetic inputs and three existing MLP16 checkpoints."""
from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

from paper.scripts import observed_slot_parallel_common as r


def synthetic_data(max_length):
    return {'model': {'hidden_dim': 64, 'quantity_variant': 'count_only_log_regression',
        'time_head_mode': 'heteroscedastic_lognormal_duration', 'time_scale': 7.,
        'time_initial_location': .2, 'time_initial_scale': .8,
        'time_observation_contract': {'mode': 'positive_integer_round_clamp_v1', 'unit': 'week', 'top_code': None}},
        'statistics': {'train_log_mean': 1.2, 'train_log_std': .8}, 'loader': {'max_seq_len': max_length}}


def synthetic_repeat(engine, data, arm, budget, *, device='cuda:0', batch_size=128, steps=3):
    """Actual joint loss/gradients and AdamW; fabricated targets only."""
    import torch
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    torch.manual_seed(42)
    if str(device).startswith('cuda'):
        torch.cuda.manual_seed_all(42)
    model, _ = engine.build_model(data, arm)
    model = model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    length = data['loader']['max_seq_len']
    # Data RNG is explicit and consumes no model/dropout RNG stream.
    generator = torch.Generator(device=device).manual_seed(20261004)
    dt = torch.randint(1, 8, (batch_size, length), device=device, generator=generator).float()
    qty = torch.randint(1, 20, dt.shape, device=device, generator=generator).float()
    mask = torch.ones_like(dt, dtype=torch.bool)
    mask[:max(1, batch_size // 4), :length // 3] = False
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times, losses, gradient_digests, new_module_gradients = [], [], [], {}
    for step in range(steps):
        budget()
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        outputs = target_outputs(model, dt, mask, qty, lambda_log_qty=1.)
        loss = outputs['joint_loss'].mean()
        r.require(bool(torch.isfinite(loss)), 'Nonfinite actual synthetic joint loss')
        loss.backward()
        gradients = {name: p.grad.detach().cpu() for name, p in model.named_parameters() if p.grad is not None}
        r.require(gradients and all(bool(torch.isfinite(value).all()) for value in gradients.values()),
            'Nonfinite actual synthetic gradient')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        r.require(bool(torch.isfinite(norm)) and float(norm) > 0., 'Invalid actual joint gradient norm')
        gradient_digests.append(canonical_state_dict_sha256(gradients))
        for name, value in gradients.items():
            if name.startswith('multilag_detail.') or name.endswith('_kernel') and 'causal_' in name:
                new_module_gradients[name] = new_module_gradients.get(name, False) or bool(value.abs().sum() > 0)
        optimizer.step()
        r.require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Nonfinite synthetic parameter')
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
        losses.append(float(loss.detach().cpu()))
    r.require(new_module_gradients and any(new_module_gradients.values()), 'Correction lacks active joint gradients')
    if arm in engine.GRU_ARMS:
        r.require(all(active for name, active in new_module_gradients.items() if name.startswith('multilag_detail.')),
            'GRU parameters lack active gradients after three updates')
    if arm in engine.CNN_ARMS:
        kernels = [active for name, active in new_module_gradients.items() if name.endswith('_kernel') and 'causal_' in name]
        r.require(len(kernels) == 3 and all(kernels), 'CNN Q/K/V kernels lack active joint gradients')
    model.eval()
    with torch.no_grad():
        output = target_outputs(model, dt, mask, qty, lambda_log_qty=1.)
    fingerprint = {'losses': losses, 'gradient_sha256': gradient_digests,
        'final_state_sha256': canonical_state_dict_sha256(model.state_dict()),
        'output_sha256': canonical_state_dict_sha256(output)}
    result = {'arm': arm, 'batch_size': batch_size, 'sequence_length': length, 'synthetic_steps': steps,
        'fingerprint': fingerprint, 'step_seconds': times, 'median_step_seconds': statistics.median(times),
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'active_new_module_gradients': new_module_gradients, 'actual_joint_gradient_finite': True,
        'real_data_loaded': False, 'held_out_test_evaluated': False}
    if str(device).startswith('cuda'):
        result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            total_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    del model, optimizer, gradients, output, outputs, loss
    gc.collect()
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    return result


def prediction_outputs(model, dt, qty, valid):
    """Expose actual shared head predictions; these are not target-dependent losses."""
    import torch
    from paper.scripts.count_aware_tpp_backbone.core import right_pad_batch, target_outputs
    expected_keys = ('pred_qty', 'pred_log_qty', 'pred_time_mu', 'pred_time_sigma')
    ordinary = target_outputs(model, dt, valid, qty, lambda_log_qty=1.)
    r.require('pred_qty' in ordinary and 'time_loss' in ordinary, 'Actual quantity/time outputs missing')
    dt, qty, valid, lengths = right_pad_batch(dt, qty, valid)
    batch_ids = torch.arange(dt.size(0), device=dt.device)
    targets, histories = lengths - 1, lengths - 2
    history_qty = qty.clone()
    history_qty[batch_ids, targets] = 0.
    observed = valid.clone()
    observed[batch_ids, targets] = False
    time_states, quantity_states = model.encode_task_states(dt, history_qty, valid, memory_write_mask=observed)
    time_hidden = time_states[batch_ids, histories]
    quantity_hidden = quantity_states[batch_ids, histories]
    log_qty, raw_qty = model.predict_quantity(quantity_hidden)
    result = {'pred_qty': raw_qty, 'pred_log_qty': log_qty,
        'pred_time_mu': model.time_location(time_hidden),
        'pred_time_sigma': model.positive_time_sigma(time_hidden)}
    r.require(set(result) == set(expected_keys) and torch.equal(result['pred_qty'], ordinary['pred_qty']),
        'Shared forecast head/ordinary trainer prediction differs')
    r.require(all(bool(torch.isfinite(result[key]).all()) for key in expected_keys)
        and bool((result['pred_time_sigma'] > 0).all()), 'Invalid shared forecast prediction')
    return result


def causality_check(engine, arm, budget, *, device='cuda:0'):
    """Exercise nonzero modules: target, future, padding, prefix and call-local state."""
    import torch
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    torch.manual_seed(62)
    model, _ = engine.build_model(synthetic_data(256), arm)
    model = model.to(device).eval()
    for name, parameter in model.named_parameters():
        if name.endswith('_kernel') and 'causal_' in name:
            torch.nn.init.normal_(parameter, std=.015)
    correction = model.multilag_detail
    if hasattr(correction, 'output_projection'):
        torch.nn.init.normal_(correction.output_projection.weight, std=.02)
    else:
        for projection in correction.output_projections:
            torch.nn.init.normal_(projection.weight, std=.02)
    generator = torch.Generator(device=device).manual_seed(62)
    dt = torch.randint(1, 8, (2, 14), device=device, generator=generator).float()
    qty = torch.randint(1, 20, dt.shape, device=device, generator=generator).float()
    valid = torch.ones_like(dt, dtype=torch.bool)
    observed = valid.clone()
    observed[:, -1] = False
    with torch.no_grad():
        budget()
        original = prediction_outputs(model, dt, qty, valid)
        changed_dt, changed_qty = dt.clone(), qty.clone()
        changed_dt[:, -1], changed_qty[:, -1] = 999., 4000.
        changed = prediction_outputs(model, changed_dt, changed_qty, valid)
        expected_keys = ('pred_qty', 'pred_log_qty', 'pred_time_mu', 'pred_time_sigma')
        r.require(set(original) == set(changed) == set(expected_keys), 'Mandatory forecast outputs missing')
        for key in expected_keys:
            r.require(torch.equal(original[key], changed[key]), 'Target leakage ' + key)
        states = model.encode_task_states(dt, qty, valid, memory_write_mask=observed)
        changed_states = model.encode_task_states(changed_dt, changed_qty, valid, memory_write_mask=observed)
        r.require(all(torch.equal(a, b) for a, b in zip(states, changed_states)), 'Withheld target leakage')
        future_dt, future_qty = dt.clone(), qty.clone()
        future_dt[:, 8:12] += 400.
        future_qty[:, 8:12] += 400.
        future = model.encode_task_states(future_dt, future_qty, valid, memory_write_mask=observed)
        prefix = model.encode_task_states(dt[:, :8], qty[:, :8], valid[:, :8], memory_write_mask=observed[:, :8])
        for old, changed, shortened in zip(states, future, prefix):
            torch.testing.assert_close(old[:, :8], changed[:, :8], rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(old[:, :8], shortened, rtol=1e-5, atol=1e-5)
        padding = torch.full((2, 2), float('nan'), device=device)
        padded = prediction_outputs(model, torch.cat((padding, dt), 1), torch.cat((padding, qty), 1),
            torch.cat((torch.zeros((2, 2), device=device, dtype=torch.bool), valid), 1))
        r.require(set(padded) == set(expected_keys), 'Mandatory padded forecast outputs missing')
        for key in expected_keys:
            torch.testing.assert_close(original[key], padded[key], rtol=1e-5, atol=1e-5)
        # Intermediate unrelated calls cannot retain a recurrent state.
        model.encode_task_states(future_dt, future_qty, valid, memory_write_mask=observed)
        repeated = model.encode_task_states(dt, qty, valid, memory_write_mask=observed)
        r.require(all(torch.equal(a, b) for a, b in zip(states, repeated)), 'Cross-call state leakage')
    budget()
    del model
    gc.collect()
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    return {'status': 'passed', 'nonzero_modules': True, 'target_invariance': True, 'future_invariance': True,
        'padding_invariance': True, 'prefix_invariance': True, 'call_local_state': True,
        'mandatory_forecast_keys': list(expected_keys), 'quantity_and_time_heads_checked': True,
        'comparison_tolerance': {'rtol': 1e-5, 'atol': 1e-5}, 'real_data_loaded': False}
