"""Evaluate frozen, internally selected time heads only after all four fits finish."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC_SHA = '8bc478500eb57ff402e1c7a65b779c19ef0b26691c16c975d479a59ddb0d03b7'
FITTER_SHA = 'e19c5d059cf4877493fcd725051b9736ba2c414bf5131b22526cbdf1b39f8857'
TIME_HELPER = ROOT.parent / 'phen_intermittent_conditional_mechanisms_20261026_v1/time/analyze_time.py'
TIME_HELPER_SHA = '713dda33f1793398dc0ffd3c7e2d407b524afbeca654cfefa5cd4d1afdf12777'
ARMS = ('original', 'widthonly', 'joint')
CONDITION_KEYS = ['previous_quantity', 'previous_gap', 'previous2_quantity_key', 'history_length']
METRICS = ['time_nll', 'sigma', 'signed_log_bin_distance', 'absolute_log_bin_distance',
           'absolute_standardized_distance', 'center_below_bin', 'center_inside_bin',
           'center_above_bin', 'observed_bin_disjoint_from_predicted_central_95']


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def record(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def verify(ref):
    path = Path(ref['path'])
    require(path.is_file() and path.stat().st_size == ref['bytes'] and sha(path) == ref['sha256'],
            'File binding differs: ' + str(path))
    return path


def module(path, expected_sha, name):
    require(sha(path) == expected_sha, 'Helper code changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def selection_check(curves, receipt, arm, spec):
    require(curves and curves[0]['epoch'] == 0, 'Epoch zero candidate missing')
    require([r['epoch'] for r in curves] == list(range(len(curves))), 'Epoch curve is not consecutive')
    values = np.asarray([row['dev_nll'] for row in curves], dtype=np.float64)
    require(np.isfinite(values).all(), 'Nonfinite selection curve')
    chosen = int(np.argmin(values))
    require(receipt['selected_epoch'] == chosen and receipt['selected_dev_nll'] == values[chosen],
            'Selected head does not use earliest absolute internal-dev minimum')
    require(receipt['config'] == spec['fit_config'], 'Fit config changed')
    require(receipt['epochs_completed'] == len(curves) - 1, 'Completed epoch count differs')
    require(receipt['trainable_parameters'] == spec['arms'][arm]['trainable_parameters'], 'Trainable scope differs')
    require(receipt['validation_used_for_selection'] is False, 'Validation selected the head')
    require(receipt['input_hidden_unchanged'] and receipt['input_checkpoint_unchanged'], 'Frozen input changed')
    require(receipt['fit_events'] == spec['split']['fit_rows'] and receipt['dev_events'] == spec['split']['dev_rows'],
            'Internal split counts changed')
    if arm == 'original':
        require(chosen == 0 and len(curves) == 1, 'Original arm was trained')
    return chosen


def completed_fits(spec):
    """Require every seed/checkpoint technical completion before selected evaluation."""
    cohorts, bindings = {}, []
    # This first pass reads only completion receipts/bindings, never Validation.
    for seed in spec['seed_order']:
        for kind in ('best', 'last'):
            directory = ROOT / 'fits' / f'seed{seed}_{kind}'
            success_path = directory / 'COHORT_COMPLETE.json'
            require(success_path.is_file(), 'All four fit cohorts must complete before evaluation')
            success = json.loads(success_path.read_text())
            require((success['seed'], success['checkpoint'], success['status']) == (seed, kind, 'COMPLETE'),
                    'Wrong technical completion')
            for ref in success['arm_bindings']:
                verify(ref)
            verify(success['checkpoint_file'])
            exit_path = ROOT / 'fit_processes' / f'seed{seed}_{kind}' / 'ACTUAL_EXIT.json'
            actual = json.loads(exit_path.read_text())
            require(actual['actual_exit'] == 0, 'Fit subprocess did not exit zero')
            for key in ('stdout', 'stderr'):
                verify(actual[key])
            bindings.extend([record(success_path), record(exit_path), actual['stdout'], actual['stderr']])
            cohorts[(seed, kind)] = {'directory': directory, 'success': success, 'actual': actual}
    return cohorts, bindings


def load_selected(cohort, seed, kind, spec, fit):
    states, receipts, curves, bindings = {}, {}, [], []
    logged = []
    for line in Path(cohort['actual']['stdout']['path']).read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get('seed') == seed and row.get('checkpoint') == kind and 'epoch' in row and 'dev_nll' in row:
            logged.append(row)
    for arm in ARMS:
        directory = cohort['directory'] / arm
        receipt_path, binding_path = directory / 'FIT_RECEIPT.json', directory / 'INPUT_BINDING.json'
        receipt, binding = (json.loads(p.read_text()) for p in (receipt_path, binding_path))
        for key in ('contract', 'feature_receipt', 'checkpoint', 'split', 'fitter', 'native_math_qa'):
            verify(binding[key])
        for ref in binding['feature_files'].values():
            verify(ref)
        require(binding['quantity_head_unchanged'] is True, 'Quantity head was not frozen')
        curve_path = directory / 'curves.jsonl'
        arm_curves = [json.loads(line) for line in curve_path.read_text().splitlines()]
        logged_curves = [{k: v for k, v in row.items() if k not in ('seed', 'checkpoint', 'arm')}
                         for row in logged if row['arm'] == arm]
        require(arm_curves == logged_curves, 'Curves differ from SHA-bound original process log')
        selection_check(arm_curves, receipt, arm, spec)
        state_path = directory / 'selected_head.pt'
        raw = torch.load(state_path, weights_only=True, map_location='cpu')
        require(set(raw) == set(fit.HEAD_SHAPES), 'Selected artifact contains non-time-head parameters')
        state = fit.validate_head_state(raw)
        require(fit.state_sha(state) == receipt['selected_head_sha256'], 'Selected head hash differs')
        states[arm], receipts[arm] = state, receipt
        curves.extend([{'seed': seed, 'checkpoint': kind, 'arm': arm, **row} for row in arm_curves])
        bindings.extend(record(p) for p in (receipt_path, binding_path, curve_path, state_path))
    for arm in ARMS:
        fit.assert_head_scope(states['original'], states[arm], arm)
    return states, receipts, curves, bindings


def load_metadata(spec):
    metadata = pd.read_parquet(verify(spec['bindings']['metadata']))
    require(set(metadata.split) == {'train', 'validation'}, 'Out-of-scope metadata split')
    require(not metadata.duplicated(['split', 'target_index']).any(), 'Duplicate event identity')
    cells = pd.read_csv(verify(spec['bindings']['support_cells'])).sort_values(CONDITION_KEYS).reset_index(drop=True)
    cells['condition_id'] = np.arange(len(cells), dtype=np.int64)
    require(cells.two_route_feasibility.dtype == bool and cells.two_route_feasibility.sum() == 532,
            'Fixed 532-condition eligibility changed')
    metadata = metadata.merge(cells[CONDITION_KEYS + ['condition_id', 'two_route_feasibility']],
                              on=CONDITION_KEYS, how='left', validate='many_to_one')
    require(metadata.condition_id.notna().all(), 'Condition membership missing')
    roles = pd.read_parquet(verify(spec['bindings']['train_roles']))
    role_by_id = roles.set_index('target_index').role
    require(role_by_id.index.is_unique and set(role_by_id) == {'fit', 'dev'}, 'Invalid frozen head roles')
    metadata['head_role'] = 'validation'
    train = metadata.split.eq('train')
    metadata.loc[train, 'head_role'] = metadata.loc[train, 'target_index'].map(role_by_id).to_numpy()
    require(metadata.head_role.notna().all(), 'Frozen Train roles missing')
    require((metadata.head_role == 'fit').sum() == spec['split']['fit_rows']
            and (metadata.head_role == 'dev').sum() == spec['split']['dev_rows'], 'Head-role count changed')
    return metadata


def feature_data(seed, kind, split, meta, spec, fit):
    directory = ROOT / 'features' / f'seed{seed}_{kind}_{split}'
    completion = record(directory / 'COMPLETE.json')
    value = json.loads(verify(completion).read_text())
    require(value['contract_sha256'] == SPEC_SHA and
            (value['seed'], value['checkpoint'], value['split'], value['rows']) == (seed, kind, split, len(meta)),
            'Feature cohort differs')
    for ref in value['files'].values():
        verify(ref)
    verify(value['checkpoint_file'])
    x = np.load(directory / 'hidden.npy', allow_pickle=False)
    ids = np.load(directory / 'target_index.npy', allow_pickle=False)
    require(x.dtype == np.float32 and x.shape == (len(meta), 64) and np.isfinite(x).all(), 'Invalid frozen hidden')
    require(np.array_equal(ids, meta.target_index), 'Feature/metadata event order differs')
    native = pd.read_parquet(directory / 'native_predictions.parquet')
    require(np.array_equal(native.target_index, ids), 'Native prediction order differs')
    head = fit.validate_head_state(torch.load(directory / 'original_time_head.pt', weights_only=True, map_location='cpu'))
    require(fit.state_sha(head) == value['head_state_sha256'], 'Original head identity differs')
    quantity = torch.load(directory / 'original_quantity_head.pt', weights_only=True, map_location='cpu')
    require(set(quantity) == {'quantity_head.weight', 'quantity_head.bias'}, 'Quantity artifact key set differs')
    return x, native, head, quantity, [completion, value['checkpoint_file'], *value['files'].values()]


def direct_math(state, hidden, batch_size=8192):
    mu, sigma = [], []
    with torch.no_grad():
        for start in range(0, len(hidden), batch_size):
            x = torch.from_numpy(hidden[start:start + batch_size]).to(torch.float64)
            mu.append((F.linear(x, state['v_t.weight']).squeeze(-1) + state['b_t']).numpy())
            raw = F.linear(x, state['time_scale_weight.weight']).squeeze(-1) + state['w_raw']
            sigma.append((.001 + F.softplus(raw)).numpy())
    return np.concatenate(mu), np.concatenate(sigma)


def parity(actual, expected, tolerance):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape and np.isfinite(actual).all() and np.isfinite(expected).all(),
            'Parity inputs are nonfinite or differently shaped')
    error = np.abs(actual - expected)
    violations = int((error > tolerance['atol'] + tolerance['rtol'] * np.abs(expected)).sum())
    require(violations == 0, 'Frozen native parity gate failed')
    return {'maximum_absolute_error': float(error.max()), 'violations': violations, **tolerance}


def group_masks(meta, split):
    """Every fixed group is emitted even when no events are present."""
    all_rows = np.ones(len(meta), dtype=bool)
    scopes = [('whole', all_rows)]
    if split == 'train':
        scopes += [('head_fit', meta.head_role.eq('fit').to_numpy()),
                   ('head_dev', meta.head_role.eq('dev').to_numpy())]
    h = meta.history_length.to_numpy()
    matched = meta.two_route_feasibility.to_numpy(dtype=bool)
    groups = [('all', all_rows), ('history_1_63', (h >= 1) & (h <= 63)),
              ('history_64_127', (h >= 64) & (h <= 127)),
              ('history_128_192', (h >= 128) & (h <= 192)), ('history_193_plus', h >= 193),
              ('matched_532', matched), ('matched_532_history_128_192', matched & (h >= 128) & (h <= 192)),
              ('quantity_gt_187_posthoc', meta.raw_quantity.to_numpy() > 187)]
    for scope, scope_mask in scopes:
        for group, group_mask in groups:
            yield scope, group, scope_mask & group_mask


def event_frame(meta, prediction, diagnostics, quantity, original_nll, native_nll, arm):
    frame = meta[['target_index', 'entity_id', 'target_seq', 'raw_quantity', 'target_gap',
                  'history_length', 'condition_id', 'two_route_feasibility', 'head_role']].copy()
    for key in ('mu', 'sigma', 'time_nll'):
        frame[key] = prediction[key]
    for key in METRICS[2:]:
        frame[key] = diagnostics[key]
    frame['native_original_time_nll'] = native_nll
    frame['nll_change_vs_original'] = frame.time_nll.to_numpy() - original_nll
    frame['pred_quantity'] = quantity
    frame['arm'] = arm
    require(np.isfinite(frame[METRICS + ['pred_quantity', 'nll_change_vs_original']].to_numpy(dtype=np.float64)).all(),
            'Nonfinite selected evaluation; no imputation or clipping')
    return frame


def summarize(frame):
    count = len(frame)
    row = {'events': count, 'entities': frame.entity_id.nunique(), 'conditions': frame.condition_id.nunique(),
           'status': 'available' if count else 'empty'}
    for metric in METRICS:
        row[metric + '_mean'] = float(frame[metric].mean()) if count else None
    error = frame.pred_quantity.to_numpy() - frame.raw_quantity.to_numpy()
    row['quantity_mae'] = float(np.abs(error).mean()) if count else None
    row['quantity_rmse'] = float(np.sqrt(np.square(error).mean())) if count else None
    row['native_original_time_nll_mean'] = float(frame.native_original_time_nll.mean()) if count else None
    return row


def plots(output, summary, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'original': '#475569', 'widthonly': '#D29420', 'joint': '#127D88'}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (seed, kind) in zip(axes.flat, [(52, 'best'), (52, 'last'), (42, 'best'), (42, 'last')]):
        for arm in ARMS:
            c = curves[(curves.seed == seed) & (curves.checkpoint == kind) & (curves.arm == arm)]
            if arm == 'original':
                ax.axhline(c.dev_nll.iloc[0], color=colors[arm], ls='--', label=arm)
            else:
                ax.plot(c.epoch, c.dev_nll, color=colors[arm], label=arm)
                selected = c.loc[c.dev_nll.idxmin()]
                ax.scatter([selected.epoch], [selected.dev_nll], color=colors[arm], s=30)
        ax.set(title=f'Seed {seed} / original {kind}', xlabel='Head-only epoch', ylabel='Internal-dev mean time NLL')
        ax.grid(alpha=.2)
    axes[0, 0].legend(frameon=False)
    for extension in ('png', 'pdf'):
        fig.savefig(output / ('internal_dev_curves.' + extension), dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    groups = ['all', 'history_128_192']
    for ax, (seed, kind) in zip(axes.flat, [(52, 'best'), (52, 'last'), (42, 'best'), (42, 'last')]):
        for number, arm in enumerate(ARMS):
            s = summary[(summary.seed == seed) & (summary.checkpoint == kind) & (summary.split == 'validation')
                        & (summary.scope == 'whole') & (summary.arm == arm)].set_index('group')
            ax.bar(np.arange(2) + (number - 1) * .24, s.loc[groups].time_nll_mean,
                   width=.22, color=colors[arm], label=arm)
        ax.set_xticks([0, 1], ['Whole Validation', 'History 128–192'])
        ax.set(title=f'Seed {seed} / original {kind}', ylabel='Mean recorded-bin NLL')
        ax.grid(axis='y', alpha=.2)
    axes[0, 0].legend(frameon=False)
    for extension in ('png', 'pdf'):
        fig.savefig(output / ('validation_arm_comparison.' + extension), dpi=170)
    plt.close(fig)


def run(output, go_receipt):
    require(not output.exists(), 'Existing evaluation preserved; use a new output directory')
    output.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    try:
        require(sha(ROOT / 'SCIENCE_SPEC.json') == SPEC_SHA, 'Frozen science contract changed')
        gate = json.loads(go_receipt.read_text())
        require(gate.get('approved') is True and gate.get('science_spec_sha256') == SPEC_SHA
                and gate.get('analysis_code_sha256') == sha(__file__), 'Evaluation GO does not bind current code/spec')
        spec = json.loads((ROOT / 'SCIENCE_SPEC.json').read_text())
        for ref in spec['bindings'].values():
            verify(ref)
        fit = module(ROOT / 'head/fitter.py', FITTER_SHA, 'selected_head_fitter')
        helper = module(TIME_HELPER, TIME_HELPER_SHA, 'selected_head_diagnostics')
        cohorts, inputs = completed_fits(spec)
        metadata = load_metadata(spec)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        summaries, paired, curves, diagnostics, exclusions, selections = ([] for _ in range(6))
        for seed in spec['seed_order']:
            for kind in ('best', 'last'):
                states, receipts, these_curves, refs = load_selected(cohorts[(seed, kind)], seed, kind, spec, fit)
                curves.extend(these_curves)
                inputs.extend(refs)
                for arm in ARMS:
                    selections.append({'seed': seed, 'checkpoint': kind, 'arm': arm, **receipts[arm]})
                for split in ('train', 'validation'):
                    meta = metadata[metadata.split == split].sort_values('target_index').reset_index(drop=True)
                    hidden, native, initial, quantity_state, refs = feature_data(seed, kind, split, meta, spec, fit)
                    inputs.extend(refs)
                    require(fit.state_sha(initial) == fit.state_sha(states['original']), 'Original arm is not native head')
                    with torch.no_grad():
                        q = torch.expm1(F.softplus(F.linear(torch.from_numpy(hidden),
                            quantity_state['quantity_head.weight'], quantity_state['quantity_head.bias']).squeeze(-1))).numpy()
                    q_gate = parity(q, native.pred_quantity.to_numpy(), spec['parity']['quantity'])
                    predictions = {arm: fit.predict(states[arm], hidden, meta.target_gap.to_numpy(),
                                                   spec['fit_config']['evaluation_batch_size']) for arm in ARMS}
                    nll_gate = parity(predictions['original']['time_nll'], native.time_nll.to_numpy(), spec['parity']['time_nll'])
                    frames = {}
                    for arm in ARMS:
                        prediction = predictions[arm]
                        mu, sigma = direct_math(states[arm], hidden, spec['fit_config']['evaluation_batch_size'])
                        require(np.array_equal(mu, prediction['mu']) and np.array_equal(sigma, prediction['sigma']),
                                'Independent selected-head formula differs')
                        diag = helper.time_diagnostics(meta.target_gap, mu, sigma)
                        nll_math = parity(diag['recalculated_time_nll'], prediction['time_nll'], spec['parity']['time_nll'])
                        frames[arm] = event_frame(meta, prediction, diag, native.pred_quantity.to_numpy(),
                            predictions['original']['time_nll'], native.time_nll.to_numpy(), arm)
                        diagnostics.append({'seed': seed, 'checkpoint': kind, 'split': split, 'arm': arm,
                                            'independent_mu_sigma_exact': True, 'quantity_native_parity': q_gate,
                                            'original_native_nll_parity': nll_gate, 'independent_nll_parity': nll_math})
                    for arm in ARMS:
                        require(np.array_equal(frames[arm].target_index, frames['original'].target_index)
                                and np.array_equal(frames[arm].pred_quantity, frames['original'].pred_quantity),
                                'Paired event or frozen quantity output changed')
                    compact = pd.concat(frames.values(), ignore_index=True)
                    compact['seed'], compact['checkpoint'], compact['split'] = seed, kind, split
                    compact.to_parquet(output / f'events_seed{seed}_{kind}_{split}.parquet', index=False, compression='zstd')
                    changes = np.column_stack([*[frames[arm].nll_change_vs_original for arm in ARMS],
                        frames['joint'].time_nll.to_numpy() - frames['widthonly'].time_nll.to_numpy()])
                    for scope, group, mask in group_masks(meta, split):
                        ci = helper.cluster_matrix_ci(changes[mask], meta.entity_id.to_numpy()[mask], seed_offset=0)
                        for number, arm in enumerate(ARMS):
                            tag = {'seed': seed, 'checkpoint': kind, 'split': split, 'scope': scope,
                                   'group': group, 'arm': arm, 'selected_epoch': receipts[arm]['selected_epoch']}
                            summaries.append({**tag, **summarize(frames[arm][mask])})
                            paired.append({**tag, 'events': int(mask.sum()), 'entities': ci['entities'],
                                'reference_arm': 'original', 'contrast': arm + '_minus_original',
                                'nll_change_mean': ci['estimate'][number], 'ci_lower': ci['lower'][number],
                                'ci_upper': ci['upper'][number], 'ci_status': ci['ci_status'],
                                'bootstrap_requested_replicates': ci['requested_replicates'],
                                'bootstrap_executed_replicates': ci['executed_replicates']})
                        paired.append({'seed': seed, 'checkpoint': kind, 'split': split, 'scope': scope,
                            'group': group, 'arm': 'joint', 'selected_epoch': receipts['joint']['selected_epoch'],
                            'reference_arm': 'widthonly', 'contrast': 'joint_minus_widthonly',
                            'events': int(mask.sum()), 'entities': ci['entities'],
                            'nll_change_mean': ci['estimate'][3], 'ci_lower': ci['lower'][3], 'ci_upper': ci['upper'][3],
                            'ci_status': ci['ci_status'], 'bootstrap_requested_replicates': ci['requested_replicates'],
                            'bootstrap_executed_replicates': ci['executed_replicates']})
                        exclusions.append({'seed': seed, 'checkpoint': kind, 'split': split, 'scope': scope,
                                           'group': group, 'full_split_events': len(meta),
                                           'included_events': int(mask.sum()), 'excluded_events': int((~mask).sum())})
                    del hidden, predictions, frames, compact
        summary, paired_frame, curve_frame = pd.DataFrame(summaries), pd.DataFrame(paired), pd.DataFrame(curves)
        summary.to_csv(output / 'summary.csv', index=False)
        paired_frame.to_csv(output / 'paired_nll.csv', index=False)
        curve_frame.to_csv(output / 'curves.csv', index=False)
        pd.DataFrame(exclusions).to_csv(output / 'excluded_groups.csv', index=False)
        (output / 'selected_heads.json').write_text(json.dumps(selections, indent=2, allow_nan=False) + '\n')
        (output / 'native_math_and_quantity_parity.json').write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + '\n')
        plots(output, summary, curve_frame)
        unique_inputs = {ref['path']: ref for ref in inputs}
        for ref in unique_inputs.values():
            verify(ref)
        receipt = {'status': 'complete', 'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
            'contract': record(ROOT / 'SCIENCE_SPEC.json'), 'evaluator': record(__file__),
            'fitter': record(ROOT / 'head/fitter.py'), 'diagnostic_helper': record(TIME_HELPER), 'go_receipt': record(go_receipt),
            'contract_input_bindings': spec['bindings'], 'input_bindings': list(unique_inputs.values()),
            'all_four_fit_cohorts_complete_before_selected_evaluation': True,
            'head_fitting_and_evaluation': 'CPU float64 time-output-head fitting/evaluation on frozen float32 hidden inputs; original time-head native arithmetic likewise casts hidden and weights to float64. This is not deployable float32 full-model retraining.',
            'quantity_metrics_source': 'unchanged_saved_native_original_predictions',
            'quantity_replay': 'Independent frozen float32 quantity-head arithmetic checked at declared native tolerance; same saved predictions copied exactly across all arms.',
            'uncertainty': '1000 whole-entity paired resamples, RNG20261026, minimum20 entities; conditional on selected heads, not training or selection uncertainty.',
            'original_nll_reference': 'Recomputed original time head on identical saved inputs; native NLL stored separately.',
            'model_training': False, 'full_model_inference': False, 'test_materialized': False,
            'outputs': [record(p) for p in sorted(output.iterdir()) if p.is_file()]}
        (output / 'RECEIPT.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
        return receipt
    except Exception:
        (output / 'FAILURE.json').write_text(json.dumps({'status': 'failed', 'started_utc': started,
                'evaluator': record(__file__), 'traceback': traceback.format_exc()}, indent=2) + '\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--go-receipt', type=Path, required=True)
    args = parser.parse_args()
    receipt = run(args.output, args.go_receipt)
    print(json.dumps({'status': receipt['status'], 'outputs': len(receipt['outputs'])}))
