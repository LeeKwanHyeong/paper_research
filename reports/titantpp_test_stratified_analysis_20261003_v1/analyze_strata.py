"""Additive descriptive strata for already sealed full-population Test predictions.

No fitting, inference, target intersection, seed pooling, or outcome-driven cuts.
The separate analysis contract must exist and bind all source metadata before use.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

METRICS = ('qty_mae', 'qty_rmse', 'time_nll', 'bias')
SUMS = ('absolute_error_sum', 'squared_error_sum', 'time_nll_sum', 'signed_error_sum')
RTOL = ATOL = 1e-10


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_legacy(path):
    spec = importlib.util.spec_from_file_location('_frozen_legacy_strata_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_source_hashes(contract):
    hashes = contract.get('source_hashes')
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('Missing frozen source hashes')
    for raw, expected in hashes.items():
        path = Path(raw)
        if not path.is_absolute() or len(expected) != 64 or sha256_file(path) != expected:
            raise ValueError('Frozen source SHA mismatch: ' + raw)
    required = [contract['source_run_manifest'], contract['source_analysis'],
                str(Path(contract['source_bundle']) / 'analyze.py')]
    for raw in required:
        if raw not in hashes:
            raise ValueError('Required source is not bound: ' + raw)


def validate_dataset_spec(spec):
    for key in ('quantity_boundaries', 'history_boundaries', 'history_detail_boundaries'):
        if key not in spec and key == 'history_detail_boundaries':
            continue
        values = np.asarray(spec[key], dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
            raise ValueError('Boundaries must be finite and strictly increasing: ' + key)
    axis = spec['time_axis']
    if axis['kind'] not in ('calendar_hour', 'calendar_week', 'calendar_month', 'relative_day'):
        raise ValueError('Unsupported time axis')
    if any(type(axis[key]) is not int for key in ('min_tick', 'max_tick')) or axis['max_tick'] < axis['min_tick']:
        raise ValueError('Invalid frozen time range')
    if not axis.get('semantics') or spec['target_count'] < 1:
        raise ValueError('Missing time semantics or population')


def time_ticks(dates, kind):
    if kind == 'relative_day':
        if not dates.str.contains(r'^-?\d+$').all():
            raise ValueError('Relative days must be recorded integers')
        return dates.cast(pl.Int64, strict=True).to_numpy()
    format_ = '%Y%m%d%H%M%S' if kind == 'calendar_hour' else '%Y%m%d'
    parsed = dates.str.strptime(pl.Datetime if kind == 'calendar_hour' else pl.Date,
                               format_, strict=True, exact=True)
    if kind == 'calendar_hour':
        milliseconds = parsed.dt.timestamp('ms').to_numpy()
        if np.any(milliseconds % 3600000):
            raise ValueError('Taxi recorded timestamps must align to hours')
        return milliseconds // 3600000
    if kind == 'calendar_week':
        return parsed.cast(pl.Int32).to_numpy().astype(np.int64) // 7
    if kind == 'calendar_month':
        if parsed.dt.day().ne(1).any():
            raise ValueError('RAF monthly recorded timestamps must be first of month')
        return parsed.dt.year().to_numpy().astype(np.int64) * 12 + parsed.dt.month().to_numpy() - 1
    raise ValueError('Unsupported time axis')


def assign_strata(frame, spec):
    axis = spec['time_axis']
    ticks = time_ticks(frame['recorded_date'], axis['kind'])
    low, high = axis['min_tick'], axis['max_tick']
    if np.any(ticks < low) or np.any(ticks > high):
        raise ValueError('Observed time is outside frozen range')
    result = {'all': np.zeros(len(frame), dtype=np.int64),
              'time': np.minimum(2, 3 * (ticks - low) // (high - low + 1))}
    for view, field in [('quantity', 'raw_quantity'), ('history', 'history_length'),
                        ('history_detail', 'history_length')]:
        key = view + '_boundaries'
        if key in spec:
            result[view] = np.searchsorted(spec[key], frame[field].to_numpy(), side='left')
    return result


def view_sizes(spec):
    result = {'all': 1, 'time': 3, 'quantity': len(spec['quantity_boundaries']) + 1,
              'history': len(spec['history_boundaries']) + 1}
    if 'history_detail_boundaries' in spec:
        result['history_detail'] = len(spec['history_detail_boundaries']) + 1
    return result


def bin_definition(spec, view, index):
    if view == 'all':
        return {'label': 'all', 'rule': 'every frozen target'}
    if view == 'time':
        axis = spec['time_axis']
        span = axis['max_tick'] - axis['min_tick'] + 1
        return {'label': ['early', 'middle', 'late'][index],
                'min_tick_inclusive': axis['min_tick'] + (index * span + 2) // 3,
                'max_tick_inclusive': axis['min_tick'] + ((index + 1) * span + 2) // 3 - 1,
                'semantics': axis['semantics'], 'rule': 'equal_duration_native_tick_grid_thirds'}
    boundaries = spec[view + '_boundaries']
    return {'label': str(index), 'lower_exclusive': boundaries[index - 1] if index else None,
            'upper_inclusive': boundaries[index] if index < len(boundaries) else None,
            'rule': 'searchsorted_side_left_equality_in_lower_bin'}


class StratumSums:
    def __init__(self, spec, slots):
        self.counts = {view: np.zeros((slots, size), dtype=np.int64)
                       for view, size in view_sizes(spec).items()}
        self.sums = {view: np.zeros((slots, size, 4), dtype=np.float64)
                     for view, size in view_sizes(spec).items()}

    def add(self, frame, slot, groups, prediction='predicted_raw_quantity', deterministic=False):
        error = frame[prediction].to_numpy() - frame['raw_quantity'].to_numpy()
        losses = [np.abs(error), error * error,
                  np.zeros(len(frame)) if deterministic else frame['time_nll'].to_numpy(), error]
        if not all(np.isfinite(value).all() for value in losses):
            raise ValueError('Nonfinite sufficient loss')
        for view, group in groups.items():
            size = self.counts[view].shape[1]
            self.counts[view][slot] += np.bincount(group, minlength=size)
            for i, loss in enumerate(losses):
                self.sums[view][slot, :, i] += np.bincount(group, weights=loss, minlength=size)

    def verify_partitions(self):
        checks = {}
        for view in self.counts:
            if not np.array_equal(self.counts[view], np.broadcast_to(self.counts[view][0], self.counts[view].shape)):
                raise ValueError('Different stratum population across models or seeds: ' + view)
            if not np.array_equal(self.counts[view].sum(axis=1), self.counts['all'][:, 0]):
                raise ValueError('Stratum counts do not conserve population: ' + view)
            totals = self.sums[view].sum(axis=1)
            if not np.allclose(totals, self.sums['all'][:, 0], rtol=RTOL, atol=ATOL):
                raise ValueError('Stratum sufficient sums do not conserve all: ' + view)
            checks[view] = {'counts_equal_across_conditions': True, 'counts_conserve_all': True,
                            'sufficient_sums_conserve_all': True,
                            'max_absolute_sum_difference': float(np.max(np.abs(totals - self.sums['all'][:, 0])))}
        return checks


def sufficient_metrics(count, values, deterministic=False):
    result = {'count': int(count), **{name: float(value) for name, value in zip(SUMS, values)}}
    result.update({key: None for key in METRICS})
    if count:
        result.update(qty_mae=float(values[0] / count), qty_rmse=math.sqrt(values[1] / count),
                      time_nll=None if deterministic else float(values[2] / count),
                      bias=float(values[3] / count))
    if deterministic:
        result['time_nll_sum'] = None
    return result


def summarize(accumulator, spec, models, seeds, simple_models):
    results = {}
    learned_slots = len(models) * len(seeds)
    for view, counts in accumulator.counts.items():
        results[view] = {}
        for index in range(counts.shape[1]):
            seed_metrics, means, sd = {}, {}, {}
            for mi, model in enumerate(models):
                rows = [sufficient_metrics(counts[mi * len(seeds) + si, index],
                                          accumulator.sums[view][mi * len(seeds) + si, index])
                        for si in range(len(seeds))]
                seed_metrics[model] = {str(seed): row for seed, row in zip(seeds, rows)}
                means[model] = {metric: float(np.mean([r[metric] for r in rows])) if counts[0, index] else None
                                for metric in METRICS}
                sd[model] = {metric: float(np.std([r[metric] for r in rows], ddof=1))
                             if counts[0, index] and len(rows) > 1 else None for metric in METRICS}
            for si, model in enumerate(simple_models):
                slot = learned_slots + si
                row = sufficient_metrics(counts[slot, index], accumulator.sums[view][slot, index], True)
                seed_metrics[model] = {'deterministic': row}
                means[model] = {key: row[key] for key in METRICS}
                sd[model] = {key: None for key in METRICS}
            results[view][str(index)] = {'count': int(counts[0, index]),
                                        'definition': bin_definition(spec, view, index),
                                        'seed_metrics': seed_metrics, 'means': means, 'sample_sd': sd}
    return results


def verify_receipt_identity(receipt, condition, reference, spec):
    if receipt.get('status') != 'complete' or any(receipt.get(k) != condition[k] for k in ('dataset', 'model', 'seed')):
        raise ValueError('Receipt condition/status mismatch')
    for key in ('dataset', 'split', 'evaluation_scope', 'full_population', 'data_file_sha256',
                'expected_target_count', 'prediction_rows', 'target_identity_sha256', 'truth_sha256',
                'registry_sha256', 'dataset_manifest_sha256'):
        if receipt.get(key) != reference.get(key):
            raise ValueError('Receipt identity/truth mismatch: ' + key)
    if receipt['split'] != 'test' or receipt.get('full_population') is not True or receipt.get('evaluation_scope') != 'full_population':
        raise ValueError('Only full-population Test predictions are authorized')
    if receipt['data_file_sha256'] != spec['data_file_sha256'] or receipt['expected_target_count'] != spec['target_count']:
        raise ValueError('Receipt differs from frozen stratification population')


def stream_dataset(legacy, conditions, receipts, spec, models, seeds, simple_models):
    dataset = spec['dataset']
    expected = [(dataset, model, seed) for model in models for seed in seeds]
    if [(r['dataset'], r['model'], r['seed']) for r in conditions] != expected or len(receipts) != len(expected):
        raise ValueError('Incomplete or reordered model/seed panel')
    acc = StratumSums(spec, len(expected) + len(simple_models))
    reference_fingerprints, evidence, reference_weekdays = None, [], set()
    for slot, (condition, receipt) in enumerate(zip(conditions, receipts)):
        verify_receipt_identity(receipt, condition, receipts[0], spec)
        root = Path(condition['output_dir']).resolve()
        if (root / 'failure.json').exists():
            raise ValueError('Condition includes failure evidence; cannot silently aggregate')
        receipt_sha = sha256_file(root / 'receipt.json')
        fingerprints = legacy.ColumnFingerprints()
        identity_hash, truth_hash, population_hash = [hashlib.sha256() for _ in range(3)]
        previous, previous_entity, entity_count, rows, seen, parts = None, None, 0, 0, set(), []
        if not receipt.get('parts'):
            raise ValueError('Complete receipt has no prediction parts')
        for part in receipt['parts']:
            path = (root / part['path']).resolve()
            if not path.is_relative_to(root) or path in seen:
                raise ValueError('Unsafe or duplicate prediction part path')
            seen.add(path)
            if sha256_file(path) != part['sha256']:
                raise ValueError('Prediction part SHA256 mismatch')
            if type(part['rows']) is not int or not 0 < part['rows'] <= 65536:
                raise ValueError('Prediction part exceeds bounded streaming contract')
            frame = pl.read_parquet(path)
            if len(frame) != part['rows']:
                raise ValueError('Prediction part row-count mismatch')
            if part.get('target_index_start', rows) != rows or part.get('target_index_end', rows + len(frame) - 1) != rows + len(frame) - 1:
                raise ValueError('Prediction part target range mismatch')
            previous = legacy._validate_frame(frame, rows, dataset, 'test', receipt['data_file_sha256'], slot == 0, previous)
            fingerprints.update(frame)
            if slot == 0:
                if spec['time_axis']['kind'] == 'calendar_week':
                    reference_weekdays.update(frame['recorded_date'].str.strptime(
                        pl.Date, '%Y%m%d', strict=True, exact=True).dt.weekday().unique().to_list())
                    if len(reference_weekdays) != 1:
                        raise ValueError('Intermittent weekly recorded dates must share a fixed weekday')
                truth_bytes = b''.join((json.dumps(row, separators=(',', ':'), ensure_ascii=False,
                                                  allow_nan=False) + '\n').encode()
                                      for row in frame.select(legacy.TRUTH_FIELDS).iter_rows())
                truth_hash.update(truth_bytes)
                if part.get('truth_sha256') != hashlib.sha256(truth_bytes).hexdigest():
                    raise ValueError('Prediction part truth digest mismatch')
                for entity, seq, date in frame.select('entity_id', 'seq', 'recorded_date').iter_rows():
                    population_hash.update((json.dumps([entity, seq, date], separators=(',', ':')) + '\n').encode())
                    if entity != previous_entity:
                        entity_count += 1
                        previous_entity = entity
            raw_ids = ('\n'.join(frame['target_id'].to_list()) + '\n').encode()
            identity_hash.update(raw_ids)
            if 'target_identity_sha256' in part and hashlib.sha256(raw_ids).hexdigest() != part['target_identity_sha256']:
                raise ValueError('Prediction part target identity digest mismatch')
            groups = assign_strata(frame, spec)
            acc.add(frame, slot, groups)
            if slot == 0:
                for i, prediction in enumerate(('last_quantity', 'mean_quantity')):
                    acc.add(frame, len(expected) + i, groups, prediction, True)
            rows += len(frame)
            parts.append({'path': str(path), 'sha256': part['sha256'], 'rows': part['rows']})
        if rows != receipt['prediction_rows'] or rows != spec['target_count']:
            raise ValueError('Incomplete target population; no intersection fallback')
        if identity_hash.hexdigest() != receipt['target_identity_sha256']:
            raise ValueError('Receipt target identity digest mismatch')
        current = fingerprints.result()
        if slot == 0:
            if truth_hash.hexdigest() != receipt['truth_sha256']:
                raise ValueError('Receipt truth digest mismatch')
            population = spec['population']
            if rows != population['target_count'] or entity_count != population['entity_count']:
                raise ValueError('Frozen population count mismatch')
            if population_hash.hexdigest() != population['ordered_entity_seq_recorded_date_sha256']:
                raise ValueError('Frozen population identity mismatch')
            reference_fingerprints = current
            population_digest = population_hash.hexdigest()
        elif current != reference_fingerprints:
            changed = [name for name in legacy.TRUTH_FIELDS if current[name] != reference_fingerprints[name]]
            raise ValueError('Target identity/truth/history mismatch: ' + ', '.join(changed))
        if sha256_file(root / 'receipt.json') != receipt_sha:
            raise ValueError('Receipt changed during analysis')
        evidence.append({'model': condition['model'], 'seed': condition['seed'], 'prediction_rows': rows,
                         'receipt_path': str(root / 'receipt.json'), 'receipt_sha256': receipt_sha,
                         'checkpoint_file_sha256': receipt.get('checkpoint_file_sha256'),
                         'state_tensor_sha256': receipt.get('state_tensor_sha256'),
                         'selected_epoch': receipt.get('selected_epoch'),
                         'source_closure_sha256': receipt.get('source_closure_sha256'),
                         'parts': parts, 'independent_column_fingerprints': current})
    checks = acc.verify_partitions()
    return summarize(acc, spec, models, seeds, simple_models), {
        'dataset': dataset, 'target_count': spec['target_count'], 'exact_common_population': True,
        'independent_column_fingerprints': reference_fingerprints,
        'ordered_entity_seq_recorded_date_sha256': population_digest,
        'truth_sha256': receipts[0]['truth_sha256'], 'target_identity_sha256': receipts[0]['target_identity_sha256'],
        'recorded_weekdays_iso_1_monday': sorted(reference_weekdays) or None,
        'week_grid': 'Unix_epoch_anchored_fixed_7_day_bins_not_ISO_weeks'
        if spec['time_axis']['kind'] == 'calendar_week' else None,
        'conditions': evidence, 'partition_checks': checks}


def reconcile_all(current, previous, models, seeds, simple_models):
    if current['count'] != previous['count']:
        raise ValueError('All-population count differs from original analysis')
    comparisons = 0
    for field in ('means', 'sample_sd'):
        for model in models + simple_models:
            for metric in ('qty_mae', 'qty_rmse', 'time_nll'):
                a, b = current[field][model][metric], previous[field][model][metric]
                if (a is None) != (b is None) or (a is not None and not np.isclose(a, b, rtol=RTOL, atol=ATOL)):
                    raise ValueError(f'Original primary metric mismatch: {field}/{model}/{metric}')
                comparisons += 1
    for model in models + simple_models:
        for seed in (['deterministic'] if model in simple_models else [str(s) for s in seeds]):
            for metric in ('qty_mae', 'qty_rmse', 'time_nll'):
                a, b = current['seed_metrics'][model][seed][metric], previous['seed_metrics'][model][seed][metric]
                if (a is None) != (b is None) or (a is not None and not np.isclose(a, b, rtol=RTOL, atol=ATOL)):
                    raise ValueError(f'Original per-seed metric mismatch: {model}/{seed}/{metric}')
                comparisons += 1
    return {'verified': True, 'comparisons': comparisons, 'rtol': RTOL, 'atol': ATOL}


def analyze_contract(path):
    contract_sha = sha256_file(path)
    contract = json.loads(Path(path).read_text())
    verify_source_hashes(contract)
    legacy = load_legacy(Path(contract['source_bundle']) / 'analyze.py')
    models, seeds, simple = contract['models'], contract['seeds'], contract['simple_models']
    if models != legacy.MODELS or seeds != legacy.SEEDS or simple != legacy.SIMPLE:
        raise ValueError('Model/seed/reference panel differs from original frozen analysis')
    specs = contract['datasets']
    if [d['dataset'] for d in specs] != legacy.DATASETS:
        raise ValueError('Dataset panel differs from original frozen analysis')
    for spec in specs:
        validate_dataset_spec(spec)
    manifest, conditions, receipts, pending = legacy.inspect_manifest(contract['source_run_manifest'])
    if pending:
        raise ValueError('Original full-population evaluation is incomplete')
    if any(r['split'] != 'test' for r in receipts.values()):
        raise ValueError('Analysis requires original Test receipts only')
    for spec in specs:
        frozen = manifest['_verified_frozen_datasets'][spec['dataset']]
        if spec['population'] != frozen['populations']['test'] or spec['data_file_sha256'] != frozen['sha256']:
            raise ValueError('Stratification changed the frozen population')
    original = json.loads(Path(contract['source_analysis']).read_text())
    if original.get('status') != 'complete' or original.get('split') != 'test' or original.get('learned_conditions') != 108:
        raise ValueError('Original analysis is incomplete or not full Test')
    if original['run_manifest_sha256'] != sha256_file(contract['source_run_manifest']):
        raise ValueError('Original analysis references another run manifest')
    results, evidence = {}, {}
    for spec in specs:
        dataset = spec['dataset']
        keys = [(dataset, model, seed) for model in models for seed in seeds]
        result, proof = stream_dataset(legacy, [conditions[k] for k in keys], [receipts[k] for k in keys],
                                       spec, models, seeds, simple)
        primary = next(iter(legacy.analysis_specs(dataset)))
        proof['original_primary_reconciliation'] = reconcile_all(result['all']['0'], original['results'][dataset][primary],
                                                                 models, seeds, simple)
        results[dataset], evidence[dataset] = result, proof
    verify_source_hashes(contract)
    if sha256_file(path) != contract_sha:
        raise ValueError('Analysis contract changed during execution')
    return {'status': 'complete', 'split': 'test', 'scope': 'additive_descriptive_stratification_existing_test',
            'interpretation': legacy.INTERPRETATION, 'analysis_contract_sha256': contract_sha,
            'analysis_script_sha256': sha256_file(__file__), 'source_hashes': contract['source_hashes'],
            'learned_conditions': 108, 'deterministic_reference_vectors': 8,
            'metric_aggregation': 'arithmetic_mean_and_sample_sd_of_per_seed_metrics',
            'prediction_ensemble': False, 'new_inference': False, 'new_training': False,
            'cuts_outcome_adaptive': False, 'bias_definition': 'predicted_raw_quantity_minus_raw_quantity',
            'intervals': None, 'uncertainty': 'seed_sample_sd_only_no_new_significance_claim',
            'results': results}, {'status': 'complete', 'analysis_contract_sha256': contract_sha,
                                   'datasets': evidence, 'all_108_checkpoint_receipt_bindings_verified': True}


def write_outputs(directory, result, verification):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    mean_rows, seed_rows = [], []
    for dataset, views in result['results'].items():
        for view, bins in views.items():
            for bin_id, bin_ in bins.items():
                definition = bin_['definition']
                context = {'dataset': dataset, 'split': 'test', 'view': view, 'bin_id': bin_id,
                           'bin_label': definition['label'],
                           'lower_exclusive': definition.get('lower_exclusive'),
                           'upper_inclusive': definition.get('upper_inclusive'),
                           'time_min_tick': definition.get('min_tick_inclusive'),
                           'time_max_tick': definition.get('max_tick_inclusive')}
                for model, means in bin_['means'].items():
                    base = {**context, 'model': model,
                            'count': bin_['count'], 'seed_count': 0 if 'deterministic' in bin_['seed_metrics'][model] else len(bin_['seed_metrics'][model])}
                    mean_rows.append({**base, **{f'{m}_mean': means[m] for m in METRICS},
                                      **{f'{m}_sample_sd': bin_['sample_sd'][model][m] for m in METRICS}})
                    for seed, metrics in bin_['seed_metrics'][model].items():
                        seed_rows.append({**context, 'model': model, 'seed': '' if seed == 'deterministic' else seed,
                                          'deterministic': seed == 'deterministic', **metrics})
    for filename, rows in [('metrics.csv', mean_rows), ('seed_metrics.csv', seed_rows)]:
        with (directory / filename).open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for filename, content in [('analysis.json', result), ('verification.json', verification)]:
        with (directory / filename).open('x') as handle:
            json.dump(content, handle, indent=2, allow_nan=False)
            handle.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output directory must be new; existing evidence is never overwritten')
    result, verification = analyze_contract(args.contract)
    write_outputs(args.output_dir, result, verification)
    print(json.dumps({'status': 'complete', 'output_dir': str(args.output_dir), 'learned_conditions': 108}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
