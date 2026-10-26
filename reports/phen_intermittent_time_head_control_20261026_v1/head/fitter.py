"""Time-head-only control on frozen 64-dimensional inputs.

No file is loaded or training performed at import. The orchestration layer must
verify its frozen science contract, input hashes, and GO before real-data calls.
Only the standalone frozen positive-integer probability helper is imported;
no encoder, quantity head, dataset, or full model is constructed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
SOURCE_MATH = (PROJECT / 'reports/phen_intermittent_full_observations_20261026_v1'
               / 'execution_v6/bundle/source42/models/TPPs/positive_integer_time.py')
SOURCE_MATH_SHA256 = '783113642d897aaed3da1213fc91fb993ee71af380f953a1159142a1cda8ae59'
HEAD_SHAPES = {'v_t.weight': (1, 64), 'b_t': (1,),
               'time_scale_weight.weight': (1, 64), 'w_raw': (1,)}
WIDTH_KEYS = ('time_scale_weight.weight', 'w_raw')
ARMS = ('original', 'widthonly', 'joint')


def require(value, message):
    if not value:
        raise ValueError(message)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode())
    h.update(memoryview(value).cast('B'))
    return h.hexdigest()


def state_sha(state):
    h = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


@lru_cache(maxsize=1)
def probability_helper():
    """Standalone differentiable arithmetic with no full model package import."""
    require(file_sha(SOURCE_MATH) == SOURCE_MATH_SHA256, 'Frozen probability source hash differs')
    spec = importlib.util.spec_from_file_location('_frozen_headcontrol_integer_math', SOURCE_MATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.positive_integer_log_mass


def recorded_nll(target, mu, sigma):
    return -probability_helper()(target, mu, sigma, time_scale=3.0, top_code=None)


def extract_head_state(checkpoint_path, state_key=None):
    """Read a local checkpoint safely; unsupported pickle globals fail closed."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    require(isinstance(checkpoint, dict), 'Checkpoint must be a mapping')
    if state_key is not None:
        require(state_key in checkpoint, 'Explicit state_dict key not present')
        state = checkpoint[state_key]
    elif all(key in checkpoint for key in HEAD_SHAPES):
        state = checkpoint
    else:
        candidates = [value for key, value in checkpoint.items()
                      if key in ('state_dict', 'model_state_dict', 'model')
                      and isinstance(value, dict) and all(k in value for k in HEAD_SHAPES)]
        require(len(candidates) == 1, 'Ambiguous/missing state_dict; supply exact state_key')
        state = candidates[0]
    return validate_head_state(state)


def validate_head_state(state):
    require(isinstance(state, dict), 'Head state must be a mapping')
    result = {}
    for name, shape in HEAD_SHAPES.items():
        require(name in state and isinstance(state[name], torch.Tensor), 'Missing head tensor: ' + name)
        value = state[name].detach().cpu()
        require(tuple(value.shape) == shape and value.is_floating_point()
                and bool(torch.isfinite(value).all()), 'Invalid head tensor: ' + name)
        result[name] = value.to(torch.float64).clone()
    require(sum(value.numel() for value in result.values()) == 130, 'Head parameter count differs')
    return result


@dataclass(frozen=True)
class FitConfig:
    learning_rate: float = .001
    weight_decay: float = 0.
    batch_size: int = 2048
    maximum_epochs: int = 50
    patience: int = 8
    minimum_improvement: float = 1e-8
    gradient_clip_norm: float = 5.
    shuffle_seed: int = 20261026
    evaluation_batch_size: int = 8192

    def validate(self):
        require(math.isfinite(self.learning_rate) and self.learning_rate >= 0, 'Invalid learning rate')
        require(math.isfinite(self.weight_decay) and self.weight_decay >= 0, 'Invalid weight decay')
        require(self.batch_size >= 1 and self.evaluation_batch_size >= 1, 'Invalid batch size')
        require(self.maximum_epochs >= 0 and self.patience >= 1, 'Invalid stopping counts')
        require(self.minimum_improvement >= 0 and math.isfinite(self.minimum_improvement),
                'Invalid improvement tolerance')
        require(self.gradient_clip_norm > 0 and math.isfinite(self.gradient_clip_norm), 'Invalid clip norm')


class TimeHead(nn.Module):
    """Exactly 130 float64 scalars; widths use .001 + softplus(raw)."""
    def __init__(self, state, arm):
        super().__init__()
        require(arm in ARMS, 'Unknown control arm')
        state = validate_head_state(state)
        self.arm = arm
        self.location_weight = nn.Parameter(state['v_t.weight'], requires_grad=arm == 'joint')
        self.location_bias = nn.Parameter(state['b_t'], requires_grad=arm == 'joint')
        self.width_weight = nn.Parameter(state['time_scale_weight.weight'], requires_grad=arm != 'original')
        self.width_bias = nn.Parameter(state['w_raw'], requires_grad=arm != 'original')

    def forward(self, hidden):
        mu = F.linear(hidden, self.location_weight).squeeze(-1) + self.location_bias
        raw = F.linear(hidden, self.width_weight).squeeze(-1) + self.width_bias
        return mu, .001 + F.softplus(raw)

    def export(self):
        return {name: value.detach().cpu().clone() for name, value in (
            ('v_t.weight', self.location_weight), ('b_t', self.location_bias),
            ('time_scale_weight.weight', self.width_weight), ('w_raw', self.width_bias))}


def chronological_masks(entity_id, target_seq, target_index, fraction=.8):
    """Per-entity strict 80/20 chronology; fewer than two events fails closed."""
    require(fraction == .8, 'Only the predeclared 80/20 split is supported')
    entity_id, target_seq, target_index = map(np.asarray, (entity_id, target_seq, target_index))
    n = len(entity_id)
    require(target_seq.shape == target_index.shape == entity_id.shape == (n,), 'Split shape mismatch')
    require(len(np.unique(target_index)) == n, 'Duplicate target index')
    require(pd.notna(entity_id).all() and np.isfinite(target_seq).all(), 'Missing split identity')
    frame = pd.DataFrame({'entity_id': entity_id, 'target_seq': target_seq,
                          'target_index': target_index, 'position': np.arange(n)})
    require(not frame.duplicated(['entity_id', 'target_seq']).any(), 'Ambiguous within-entity chronology')
    fit = np.zeros(n, dtype=bool)
    dev = np.zeros(n, dtype=bool)
    counts = []
    for entity, part in frame.sort_values(['entity_id', 'target_seq', 'target_index']).groupby('entity_id', sort=False):
        count = len(part)
        require(count >= 2, 'Entity has fewer than two Train events')
        cut = math.floor(fraction * count)
        fit[part.position.iloc[:cut].to_numpy()] = True
        dev[part.position.iloc[cut:].to_numpy()] = True
        counts.append({'entity_id': entity, 'events': count, 'fit_events': cut, 'dev_events': count - cut,
                       'fit_last_seq': part.target_seq.iloc[cut - 1],
                       'dev_first_seq': part.target_seq.iloc[cut]})
    require(np.all(fit ^ dev), 'Split is not a disjoint exhaustive partition')
    return fit, dev, pd.DataFrame(counts)


def prepare_data(hidden, target, fit_mask=None, dev_mask=None):
    hidden, target = np.asarray(hidden), np.asarray(target)
    require(hidden.ndim == 2 and hidden.shape[1] == 64 and target.shape == (len(hidden),),
            'Expected N×64 hidden and N duration targets')
    require(np.isfinite(hidden).all() and np.isfinite(target).all()
            and (target >= 1).all() and (target == np.rint(target)).all(), 'Invalid frozen input/target')
    if fit_mask is not None or dev_mask is not None:
        fit_mask, dev_mask = np.asarray(fit_mask), np.asarray(dev_mask)
        require(fit_mask.dtype == dev_mask.dtype == bool and fit_mask.shape == dev_mask.shape == target.shape,
                'Fit/dev masks must be boolean N-vectors')
        require(np.all(fit_mask ^ dev_mask) and fit_mask.any() and dev_mask.any(),
                'Fit/dev must be disjoint, exhaustive, and nonempty')
    return torch.as_tensor(hidden, dtype=torch.float64).detach(), torch.as_tensor(target, dtype=torch.float64).detach()


def evaluate(head, hidden, target, indices=None, batch_size=8192):
    """Event-weighted NLL on supplied frozen inputs; never backpropagates."""
    if indices is None:
        indices = torch.arange(len(hidden))
    count, total = 0, 0.
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            mu, sigma = head(hidden[batch])
            loss = recorded_nll(target[batch], mu, sigma)
            total += float(loss.sum())
            count += len(batch)
    require(count > 0 and math.isfinite(total), 'Empty or nonfinite evaluation')
    return total / count


def predict(state, hidden, target, batch_size=8192):
    x, y = prepare_data(hidden, target)
    head = TimeHead(state, 'original')
    values = {'mu': [], 'sigma': [], 'time_nll': []}
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            mu, sigma = head(x[start:start + batch_size])
            for key, value in (('mu', mu), ('sigma', sigma),
                               ('time_nll', recorded_nll(y[start:start + batch_size], mu, sigma))):
                values[key].append(value.numpy())
    return {key: np.concatenate(parts) if parts else np.empty(0) for key, parts in values.items()}


def assert_head_scope(initial, fitted, arm):
    require(set(fitted) == set(HEAD_SHAPES), 'Output must contain exactly four time-head tensors')
    fixed = HEAD_SHAPES if arm == 'original' else ('v_t.weight', 'b_t') if arm == 'widthonly' else ()
    for name in fixed:
        require(torch.equal(initial[name], fitted[name]), 'Frozen parameter changed: ' + name)


def fit_arm(hidden, target_gap, fit_mask, dev_mask, initial_head_state, arm,
            config=None, output_dir=None, epoch_callback=None):
    """Fit one arm using only head-fit; select only head-dev, including epoch 0.

    ``epoch_callback(row)`` is for progress/evidence only. Validation inputs are
    absent from this function's interface. A new output directory retains the
    original, selected, and last copies plus every completed epoch and failures.
    """
    config = config or FitConfig()
    config.validate()
    require(arm in ARMS, 'Unknown arm')
    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        require(not output_dir.exists(), 'Refusing to overwrite an existing arm directory')
        output_dir.mkdir(parents=True)
    curves = []
    try:
        x, y = prepare_data(hidden, target_gap, fit_mask, dev_mask)
        hidden_sha = array_sha(hidden)
        initial = validate_head_state(initial_head_state)
        initial_sha = state_sha(initial)
        head = TimeHead(initial, arm)
        fit_ids, dev_ids = (torch.as_tensor(np.flatnonzero(mask), dtype=torch.long) for mask in (fit_mask, dev_mask))
        parameters = [value for value in head.parameters() if value.requires_grad]
        trainable = sum(p.numel() for p in parameters)
        require(trainable == {'original': 0, 'widthonly': 65, 'joint': 130}[arm], 'Incorrect trainable scope')

        def record(row):
            curves.append(row)
            if output_dir is not None:
                with (output_dir / 'curves.jsonl').open('a') as handle:
                    handle.write(json.dumps(row, allow_nan=False) + '\n')
            if epoch_callback is not None:
                epoch_callback(dict(row))

        selected, selected_epoch = head.export(), 0
        best_dev = evaluate(head, x, y, dev_ids, config.evaluation_batch_size)
        patience_best_dev = best_dev
        record({'epoch': 0, 'fit_nll': evaluate(head, x, y, fit_ids, config.evaluation_batch_size),
                'dev_nll': best_dev, 'selected': True, 'stale_epochs': 0,
                'significant_improvement': True, 'patience_reference_nll': patience_best_dev,
                'maximum_preclip_gradient_norm': None})
        if output_dir is not None:
            torch.save(initial, output_dir / 'initial_head.pt')
            torch.save(selected, output_dir / 'selected_head.pt')
            torch.save(selected, output_dir / 'last_head.pt')
        stop_reason, stale = 'original_no_training' if arm == 'original' else 'maximum_epochs', 0
        if parameters:
            optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate,
                                          weight_decay=config.weight_decay, foreach=False)
            generator = torch.Generator(device='cpu').manual_seed(config.shuffle_seed)
            for epoch in range(1, config.maximum_epochs + 1):
                order = fit_ids[torch.randperm(len(fit_ids), generator=generator)]
                max_norm = 0.
                for start in range(0, len(order), config.batch_size):
                    ids = order[start:start + config.batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    mu, sigma = head(x[ids])
                    loss = recorded_nll(y[ids], mu, sigma).mean()
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip_norm, error_if_nonfinite=True)
                    max_norm = max(max_norm, float(norm))
                    optimizer.step()
                fit_loss = evaluate(head, x, y, fit_ids, config.evaluation_batch_size)
                dev_loss = evaluate(head, x, y, dev_ids, config.evaluation_batch_size)
                improved = dev_loss < best_dev
                significant = dev_loss < patience_best_dev - config.minimum_improvement
                if improved:
                    best_dev, selected_epoch, selected = dev_loss, epoch, head.export()
                    if output_dir is not None:
                        torch.save(selected, output_dir / 'selected_head.pt')
                if significant:
                    patience_best_dev, stale = dev_loss, 0
                else:
                    stale += 1
                assert_head_scope(initial, head.export(), arm)
                record({'epoch': epoch, 'fit_nll': fit_loss, 'dev_nll': dev_loss,
                        'selected': improved, 'stale_epochs': stale,
                        'significant_improvement': significant, 'patience_reference_nll': patience_best_dev,
                        'maximum_preclip_gradient_norm': max_norm})
                if output_dir is not None:
                    torch.save(head.export(), output_dir / 'last_head.pt')
                if stale >= config.patience:
                    stop_reason = 'early_stopping'
                    break
        require(state_sha(validate_head_state(initial_head_state)) == initial_sha, 'Input checkpoint mutated')
        require(array_sha(hidden) == hidden_sha, 'Frozen hidden input mutated')
        require(file_sha(SOURCE_MATH) == SOURCE_MATH_SHA256, 'Frozen probability source changed during fit')
        assert_head_scope(initial, selected, arm)
        result = {'arm': arm, 'selected_epoch': selected_epoch, 'selected_dev_nll': best_dev,
                  'epochs_completed': len(curves) - 1, 'stop_reason': stop_reason,
                  'trainable_parameters': trainable, 'config': asdict(config),
                  'fit_events': len(fit_ids), 'dev_events': len(dev_ids),
                  'hidden_sha256': hidden_sha, 'initial_head_sha256': initial_sha,
                  'selected_head_sha256': state_sha(selected),
                  'source_math_sha256': file_sha(SOURCE_MATH),
                  'input_hidden_unchanged': True, 'input_checkpoint_unchanged': True,
                  'validation_used_for_selection': False}
        if output_dir is not None:
            pd.DataFrame(curves).to_csv(output_dir / 'curves.csv', index=False)
            (output_dir / 'FIT_RECEIPT.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        return {'selected_state': selected, 'last_state': head.export(), 'curves': curves, 'receipt': result}
    except Exception:
        if output_dir is not None:
            (output_dir / 'FAILURE.json').write_text(json.dumps({
                'arm': arm, 'completed_curve_rows': len(curves), 'traceback': traceback.format_exc()}, indent=2) + '\n')
        raise
