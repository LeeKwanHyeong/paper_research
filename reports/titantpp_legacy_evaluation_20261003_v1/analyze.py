"""Frozen legacy reevaluation: strict streaming pairing and sufficient-sum inference.

This file does not train, select, drop, intersect, or pool populations. Production
CLI settings are fixed. Missing receipts yield a pending report without reading
prediction parts. Only a complete manifest can reach performance aggregation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

DATASETS = ['yellow_trip_hourly', 'intermittent_frozen_5000',
            'insta_market_basket', 'raf_spare_parts']
MODELS = ['titantpp_history_mlp', 'rmtpp', 'thp', 'nhp', 'sahp',
          's2p2_matched_head', 'attnhp_matched_head',
          'titantpp_current_only_param_matched', 'titantpp_all_available_history_mlp']
SEEDS = [42, 52, 62]
SIMPLE = ['last_observed_quantity', 'mean_quantity_in_same_observed_window']
METRICS = ('qty_mae', 'qty_rmse', 'time_nll')
TRUTH_FIELDS = ['target_index', 'target_id', 'entity_id', 'seq', 'split',
                'recorded_date', 'time_bucket', 'site_id', 'raw_quantity',
                'recorded_gap', 'history_length', 'last_quantity', 'mean_quantity']
SCHEMA = {name: pl.String for name in ['target_id', 'entity_id', 'split',
                                      'recorded_date', 'time_bucket', 'site_id']}
SCHEMA.update({name: pl.Int64 for name in ['target_index', 'seq', 'history_length']})
SCHEMA.update({name: pl.Float64 for name in ['raw_quantity', 'predicted_raw_quantity',
                                           'recorded_gap', 'time_nll',
                                           'last_quantity', 'mean_quantity']})
OPTIONAL = {'time_bucket', 'site_id'}
INTERPRETATION = ('retrospective_locked_test_reevaluation; existing legacy test '
                  'population, not an untouched independent confirmation')


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def analysis_specs(dataset):
    if dataset == DATASETS[0]:
        return {'primary_168_hour': ('calendar', 'time_bucket', 168),
                'sensitivity_24_hour': ('calendar', 'time_bucket', 24),
                'sensitivity_336_hour': ('calendar', 'time_bucket', 336),
                'sensitivity_whole_cell': ('cluster', 'entity_id', None)}
    if dataset == DATASETS[1]:
        return {'primary_site': ('cluster', 'site_id', None),
                'sensitivity_series': ('cluster', 'entity_id', None)}
    return {'primary_entity': ('cluster', 'entity_id', None)}


class ColumnFingerprints:
    """SHA256 values/nulls/lengths independently, invariant to part boundaries.

    Primitive columns have fixed-width little-endian encoding. String lengths,
    concatenated UTF8 contents, and per-row null flags use separate streams;
    this prevents ambiguous concatenation without Python JSON per prediction.
    """
    def __init__(self):
        self.hashes = {name: [hashlib.sha256() for _ in range(3)]
                       for name in TRUTH_FIELDS}

    def update(self, frame):
        for name in TRUTH_FIELDS:
            values = frame[name]
            h, lengths, nulls = self.hashes[name]
            if values.dtype == pl.String:
                nulls.update(values.is_null().to_numpy().astype(np.uint8).tobytes())
                clean = values.fill_null('')
                lengths.update(clean.str.len_bytes().to_numpy().astype('<u8').tobytes())
                h.update(clean.str.join('').item().encode('utf-8'))
            else:
                h.update(values.to_numpy().astype('<i8' if values.dtype == pl.Int64 else '<f8').tobytes())

    def result(self):
        return {name: digest([h.hexdigest() for h in streams])
                for name, streams in self.hashes.items()}


class UnitSums:
    """One loss vector per learned condition or deterministic reference."""
    def __init__(self, kind, key, slots):
        self.kind, self.key, self.slots = kind, key, slots
        self.initial = {}
        self.labels = self.counts = self.sums = self.lookup = None

    def add(self, frame, slot, reference=False):
        if self.kind == 'calendar':
            times = np.array(frame[self.key].to_list(), dtype='datetime64[ns]')
            hours = times.astype('datetime64[h]')
            if np.isnat(times).any() or not np.array_equal(times, hours.astype('datetime64[ns]')):
                raise ValueError('Taxi timestamps must be recorded aligned hours; no fallback')
            frame = frame.with_columns(pl.Series('_unit', hours.astype(np.int64)))
        else:
            if frame[self.key].null_count() or frame[self.key].eq('').any():
                raise ValueError('Missing cluster mapping: ' + self.key)
            frame = frame.with_columns(pl.col(self.key).alias('_unit'))
        predictions = ['predicted_raw_quantity', 'last_quantity', 'mean_quantity'] if reference else ['predicted_raw_quantity']
        expressions = [pl.len().alias('_count')]
        for i, pred in enumerate(predictions):
            err = pl.col(pred) - pl.col('raw_quantity')
            expressions += [err.abs().sum().alias(f'a{i}'), err.pow(2).sum().alias(f's{i}')]
        expressions.append(pl.col('time_nll').sum().alias('nll'))
        grouped = frame.group_by('_unit').agg(expressions)
        for row in grouped.iter_rows(named=True):
            label = row['_unit']
            if reference:
                count, total = self.initial.setdefault(label, [0, np.zeros((3, 3))])
                self.initial[label][0] = count + row['_count']
                for i in range(3):
                    total[i] += [row[f'a{i}'], row[f's{i}'], row['nll'] if i == 0 else 0]
            else:
                if label not in self.lookup:
                    raise ValueError('Population introduced a new resampling unit')
                self.sums[self.lookup[label], slot] += [row['a0'], row['s0'], row['nll']]

    def finalize_reference(self):
        if not self.initial:
            raise ValueError('Empty target population')
        self.labels = (list(range(min(self.initial), max(self.initial) + 1))
                       if self.kind == 'calendar' else sorted(self.initial))
        self.lookup = {label: i for i, label in enumerate(self.labels)}
        self.counts = np.zeros(len(self.labels), dtype=np.int64)
        self.sums = np.zeros((len(self.labels), self.slots, 3), dtype=np.float64)
        for label, (count, sums) in self.initial.items():
            i = self.lookup[label]
            self.counts[i] = count
            self.sums[i, 0] = sums[0]
            self.sums[i, -2:] = sums[1:]
        self.initial.clear()


def _validate_frame(frame, start, dataset, split, data_sha, reference, previous):
    if dict(frame.schema) != SCHEMA:
        raise ValueError('Prediction parquet schema mismatch')
    if not len(frame):
        raise ValueError('Empty prediction part')
    for name in SCHEMA:
        if name not in OPTIONAL and frame[name].null_count():
            raise ValueError('Null required prediction field: ' + name)
    if not np.array_equal(frame['target_index'].to_numpy(), np.arange(start, start + len(frame))):
        raise ValueError('Target index order mismatch; no drop/intersection fallback')
    if frame['split'].ne(split).any():
        raise ValueError('Prediction split mismatch')
    for name, dtype in SCHEMA.items():
        if dtype == pl.Float64 and not frame[name].is_finite().all():
            raise ValueError('Nonfinite prediction/truth: ' + name)
    for name in ['raw_quantity', 'recorded_gap', 'last_quantity', 'mean_quantity']:
        if frame[name].lt(0).any():
            raise ValueError('Negative observed quantity/gap')
    if frame['history_length'].lt(1).any():
        raise ValueError('Empty admitted history')
    err = frame['predicted_raw_quantity'] - frame['raw_quantity']
    if not err.pow(2).is_finite().all():
        raise ValueError('Nonfinite squared loss')
    if reference:
        for row in frame.select('entity_id', 'seq', 'recorded_date', 'target_id', 'site_id').iter_rows(named=True):
            key = (row['entity_id'], row['seq'])
            if previous is not None and key <= previous:
                raise ValueError('Duplicate or unordered entity/sequence target identity')
            expected = digest([dataset, data_sha, row['entity_id'], row['seq'], split, row['recorded_date']])
            if row['target_id'] != expected:
                raise ValueError('Target identity does not match frozen flat-tuple contract')
            if dataset == DATASETS[1] and row['site_id'] != row['entity_id'].split('::', 1)[0]:
                raise ValueError('Intermittent site mapping mismatch')
            previous = key
    return previous


def stream_dataset(conditions, receipts, dataset, expected_population=None):
    """Read at most one bounded parquet part, retain only unit-level sums."""
    expected = [(dataset, model, seed) for model in MODELS for seed in SEEDS]
    actual = [(row['dataset'], row['model'], row['seed']) for row in conditions]
    if actual != expected or len(receipts) != len(expected):
        raise ValueError('Dataset panel must contain all 27 conditions in frozen order')
    learned_slots = len(MODELS) * len(SEEDS)
    specs = analysis_specs(dataset)
    units = {key: UnitSums(kind, key, learned_slots + len(SIMPLE))
             for kind, key, _ in specs.values()}
    reference_fingerprints = None
    population_hash = hashlib.sha256()
    first_receipt = receipts[0]
    evidence = []
    for slot, (condition, receipt) in enumerate(zip(conditions, receipts)):
        if receipt.get('status') != 'complete' or any(receipt.get(key) != condition[key] for key in ['dataset', 'model', 'seed']):
            raise ValueError('Receipt condition/status mismatch')
        for key in ['dataset', 'split', 'evaluation_scope', 'full_population', 'data_file_sha256', 'expected_target_count',
                    'prediction_rows', 'target_identity_sha256', 'truth_sha256']:
            if receipt.get(key) != first_receipt.get(key):
                raise ValueError('Receipt identity/truth mismatch: ' + key)
        for key in ['registry_sha256', 'dataset_manifest_sha256']:
            if key in first_receipt and receipt.get(key) != first_receipt[key]:
                raise ValueError('Receipt provenance mismatch: ' + key)
        root = Path(condition['output_dir']).resolve()
        fingerprints = ColumnFingerprints()
        identity_hash = hashlib.sha256()
        truth_hash = hashlib.sha256()
        rows, previous = 0, None
        seen_parts = set()
        if not receipt.get('parts'):
            raise ValueError('Complete receipt has no prediction parts')
        for part in receipt['parts']:
            path = (root / part['path']).resolve()
            if not path.is_relative_to(root) or path in seen_parts:
                raise ValueError('Unsafe or duplicate prediction part path')
            seen_parts.add(path)
            if sha256_file(path) != part['sha256']:
                raise ValueError('Prediction part SHA256 mismatch')
            # Producer seals fixed 8,192-row parts. Enforce bounded reads.
            if not 0 < part['rows'] <= 65536:
                raise ValueError('Prediction part exceeds bounded streaming contract')
            frame = pl.read_parquet(path)
            if len(frame) != part['rows']:
                raise ValueError('Prediction part row-count mismatch')
            if part.get('target_index_start', rows) != rows or part.get('target_index_end', rows + len(frame) - 1) != rows + len(frame) - 1:
                raise ValueError('Prediction part target range mismatch')
            previous = _validate_frame(frame, rows, dataset, receipt['split'],
                                       receipt['data_file_sha256'], slot == 0, previous)
            fingerprints.update(frame)
            if slot == 0:
                truth_bytes = b''.join((json.dumps(row, separators=(',', ':'), ensure_ascii=False,
                                                   allow_nan=False) + '\n').encode()
                                      for row in frame.select(TRUTH_FIELDS).iter_rows())
                truth_hash.update(truth_bytes)
                if part.get('truth_sha256') != hashlib.sha256(truth_bytes).hexdigest():
                    raise ValueError('Prediction part truth digest mismatch')
                for entity, seq, date in frame.select('entity_id', 'seq', 'recorded_date').iter_rows():
                    population_hash.update((json.dumps([entity, seq, date], separators=(',', ':')) + '\n').encode())
            raw_ids = ('\n'.join(frame['target_id'].to_list()) + '\n').encode()
            identity_hash.update(raw_ids)
            if 'target_identity_sha256' in part and hashlib.sha256(raw_ids).hexdigest() != part['target_identity_sha256']:
                raise ValueError('Prediction part target identity digest mismatch')
            for accumulator in units.values():
                accumulator.add(frame, slot, reference=slot == 0)
            rows += len(frame)
        if rows != receipt['prediction_rows'] or rows != receipt['expected_target_count'] or rows < 1:
            raise ValueError('Incomplete target population; no intersection fallback')
        if identity_hash.hexdigest() != receipt['target_identity_sha256']:
            raise ValueError('Receipt target identity digest mismatch')
        current = fingerprints.result()
        if slot == 0:
            if truth_hash.hexdigest() != receipt['truth_sha256']:
                raise ValueError('Receipt truth digest mismatch')
            reference_fingerprints = current
            for accumulator in units.values():
                accumulator.finalize_reference()
            if expected_population is not None:
                if rows != expected_population['target_count'] or len(units['entity_id'].labels) != expected_population['entity_count']:
                    raise ValueError('Target population count differs from frozen dataset manifest')
                if population_hash.hexdigest() != expected_population['ordered_entity_seq_recorded_date_sha256']:
                    raise ValueError('Target population order/identity differs from frozen dataset manifest')
        elif current != reference_fingerprints:
            changed = [key for key in TRUTH_FIELDS if current[key] != reference_fingerprints[key]]
            raise ValueError('Target identity/truth/history mismatch: ' + ', '.join(changed))
        evidence.append({'model': condition['model'], 'seed': condition['seed'],
                         'prediction_rows': rows, 'receipt_sha256': sha256_file(root / 'receipt.json')})
    return units, {'dataset': dataset, 'target_count': first_receipt['prediction_rows'],
                   'data_file_sha256': first_receipt['data_file_sha256'],
                   'target_identity_sha256': first_receipt['target_identity_sha256'],
                   'truth_sha256': first_receipt['truth_sha256'],
                   'truth_sha256_independently_verified': True,
                   'ordered_entity_seq_recorded_date_sha256': population_hash.hexdigest(),
                   'frozen_population_identity_verified': expected_population is not None,
                   'independent_column_fingerprints': reference_fingerprints,
                   'conditions': evidence, 'exact_common_population': True,
                   'deterministic_vectors_per_model': 1}


def draw_weights(rng, size, kind, length=None):
    # Exact predecessor RNG call order and circular trimming, intentionally fixed.
    if kind == 'cluster':
        index = rng.integers(0, size, size=size)
    else:
        starts = rng.integers(0, size, size=math.ceil(size / length))
        index = ((starts[:, None] + np.arange(length)[None, :]) % size).ravel()[:size]
    return np.bincount(index, minlength=size)


def _model_metrics(slot_metrics):
    learned = slot_metrics[..., :-2, :].reshape(slot_metrics.shape[:-2] + (len(MODELS), len(SEEDS), 3))
    deterministic = slot_metrics[..., -2:, :].copy()
    deterministic[..., 2] = np.nan
    means = np.concatenate([learned.mean(axis=-2), deterministic], axis=-2)
    return learned, deterministic, means


def bootstrap_sums(counts, sums, *, dataset, kind, cluster_key, block_hours=None,
                   draws=10000, rng_seed=None, scope='legacy_test_reevaluation'):
    """Fixed estimator; configurable draw count is solely for synthetic tests."""
    if draws < 2 or (kind == 'calendar' and (not isinstance(block_hours, int) or block_hours < 1)):
        raise ValueError('Invalid bootstrap specification')
    if len(counts) == 0 or int(counts.sum()) <= 0 or sums.shape != (len(counts), len(MODELS) * len(SEEDS) + 2, 3):
        raise ValueError('Invalid sufficient sums')
    if np.any(counts < 0) or not np.isfinite(sums).all():
        raise ValueError('Nonfinite or invalid sufficient sums')
    rng_seed = 20261001 + DATASETS.index(dataset) if rng_seed is None else rng_seed
    values = sums.sum(axis=0) / counts.sum()
    values[:, 1] = np.sqrt(values[:, 1])
    learned, deterministic, means = _model_metrics(values)
    names = MODELS + SIMPLE
    metric_dict = lambda array: {key: float(value) if np.isfinite(value) else None
                                 for key, value in zip(METRICS, array)}
    result = {'scope': scope, 'count': int(counts.sum()), 'units': len(counts),
              'kind': kind, 'cluster_key': cluster_key if kind == 'cluster' else None,
              'seed_metrics': {model: {str(seed): metric_dict(learned[mi, si]) for si, seed in enumerate(SEEDS)} for mi, model in enumerate(MODELS)},
              'means': {model: metric_dict(means[mi]) for mi, model in enumerate(names)},
              'sample_sd': {model: metric_dict(learned[mi].std(axis=0, ddof=1)) for mi, model in enumerate(MODELS)},
              'deterministic_models': SIMPLE, 'deterministic_prediction_vectors_per_model': 1,
              'uncertainty_scope': 'conditional_on_fixed_trained_seeds',
              'draws_requested': draws, 'rng': 'PCG64', 'rng_seed': rng_seed,
              'numpy': np.__version__, 'block_hours': block_hours,
              'estimate': 'mean_of_per_seed_metrics_not_prediction_ensemble',
              'time_nll_definition': 'frozen_recorded_integer_bin_heteroscedastic_lognormal_NLL',
              'interpretation': INTERPRETATION}
    for i, model in enumerate(SIMPLE):
        result['seed_metrics'][model] = {'deterministic': metric_dict(deterministic[i])}
        result['sample_sd'][model] = {key: None for key in METRICS}
    effective = len(counts) if kind == 'cluster' else len(counts) // block_hours
    result.update(nominal_nonoverlapping_units=effective,
                  small_sample_descriptive_only=effective < 20)
    if len(counts) < 2 or (kind == 'calendar' and len(counts) < 2 * block_hours):
        return {**result, 'interval_status': 'not_estimable_insufficient_units_or_span',
                'intervals': None, 'draws_completed': 0}
    rng = np.random.Generator(np.random.PCG64(rng_seed))
    differences = np.full((draws, len(names), 3), np.nan)
    ratios = np.full((draws, len(names)), np.nan)
    weights_hash = hashlib.sha256()
    invalid = 0
    # Limit the transient weight matrix to about 16 MiB. Batched BLAS keeps
    # large entity-cluster populations practical without retaining predictions.
    batch_size = max(1, min(128, (16 << 20) // (8 * len(counts))))
    for start in range(0, draws, batch_size):
        stop = min(draws, start + batch_size)
        weights = np.empty((stop - start, len(counts)), dtype=np.float64)
        for i in range(stop - start):
            weight = draw_weights(rng, len(counts), kind, block_hours)
            weights_hash.update(weight.astype('<i8').tobytes())
            weights[i] = weight
        totals = weights @ counts
        good = totals > 0
        invalid += int((~good).sum())
        batch = np.full((len(weights), sums.shape[1], 3), np.nan)
        batch[good] = (weights[good] @ sums.reshape(len(counts), -1)).reshape((-1, sums.shape[1], 3)) / totals[good, None, None]
        batch[..., 1] = np.sqrt(batch[..., 1])
        avg = _model_metrics(batch)[2]
        differences[start:stop] = avg[:, :1] - avg
        np.divide(100 * (avg[..., 1] - avg[:, :1, 1]), avg[..., 1],
                  out=ratios[start:stop], where=avg[..., 1] != 0)
    result.update(draws_completed=draws, invalid_empty_draws=invalid,
                  resampling_weights_sha256=weights_hash.hexdigest())
    if invalid:
        return {**result, 'interval_status': 'not_finalized_invalid_draws', 'intervals': None}
    anchor = 'rmtpp' if DATASETS.index(dataset) < 2 else 's2p2_matched_head'
    intervals = {}
    for mi, model in enumerate(names[1:], 1):
        entry = {metric: ({'difference': float(means[0, k] - means[mi, k]),
                           'pointwise_95_percentile': np.quantile(differences[:, mi, k], [.025, .975], method='linear').tolist()}
                          if not (model in SIMPLE and metric == 'time_nll') else None)
                 for k, metric in enumerate(METRICS)}
        valid = np.isfinite(ratios[:, mi]).all() and means[mi, 1] != 0
        entry['relative_RMSE_improvement_percent'] = {
            'point': float(100 * (means[mi, 1] - means[0, 1]) / means[mi, 1]) if means[mi, 1] != 0 else None,
            'pointwise_95_percentile': np.quantile(ratios[:, mi], [.025, .975], method='linear').tolist() if valid else None,
            'status': 'defined' if valid else 'undefined_zero_denominator'}
        if model == anchor:
            entry['qty_rmse']['four_anchor_bonferroni_98_75_percentile'] = np.quantile(differences[:, mi, 1], [.00625, .99375], method='linear').tolist()
        intervals[model] = entry
    return {**result, 'interval_status': 'computed_descriptive' if effective < 20 else 'computed_conditional',
            'intervals': intervals}


def load_frozen_bindings(manifest, manifest_path):
    """Bind analysis to the separately frozen contract, registry, and population."""
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    base = Path(manifest_path).resolve().parent
    if any(not isinstance(manifest.get(field), str) for field in
           ['execution_contract_path', 'dataset_manifest_path', 'registry_path']):
        raise ValueError('Run manifest is missing frozen artifact paths')
    contract_path = (base / manifest['execution_contract_path']).resolve()
    data_path = (base / manifest['dataset_manifest_path']).resolve()
    if contract_path != here / 'execution_contract.json' or data_path != here / 'dataset_manifest.json':
        raise ValueError('Run manifest must reference this frozen campaign contract and dataset manifest')
    if manifest.get('execution_contract_sha256') != sha256_file(contract_path):
        raise ValueError('Run manifest execution contract mismatch')
    contract = json.loads(contract_path.read_text())
    registry_path = (base / manifest['registry_path']).resolve()
    if registry_path != (root / contract['registry_path']).resolve():
        raise ValueError('Run manifest references a different frozen registry')
    registry = json.loads(registry_path.read_text())
    data = json.loads(data_path.read_text())
    for field, artifact in [('registry_sha256', registry_path), ('dataset_manifest_sha256', data_path)]:
        actual = sha256_file(artifact)
        if contract.get(field) != actual or manifest.get(field) != actual:
            raise ValueError('Frozen/run manifest identity mismatch: ' + field)
    seal_path = here / 'code_seal.json'
    seal = json.loads(seal_path.read_text())
    if manifest.get('code_seal_sha256') != sha256_file(seal_path):
        raise ValueError('Run manifest code seal mismatch')
    for path in [Path(__file__).resolve(), here / 'evaluate.py']:
        if seal['files'].get(str(path.relative_to(root))) != sha256_file(path):
            raise ValueError('Analysis/evaluator code differs from frozen seal')
    datasets = {row['dataset']: row for row in data['datasets']}
    rows = {(row['dataset'], row['model'], row['seed']): row for row in registry['rows']}
    expected = {(d, m, s) for d in DATASETS for m in MODELS for s in SEEDS}
    if set(rows) != expected or set(datasets) != set(DATASETS):
        raise ValueError('Frozen registry/dataset coverage mismatch')
    if contract.get('route') != 'retrospective_locked_test_reevaluation' or contract['approval'].get('retraining_authorized') is not False:
        raise ValueError('Invalid frozen legacy evaluation route')
    for dataset, row in datasets.items():
        if contract['dataset_sha256'].get(dataset) != row['sha256']:
            raise ValueError('Frozen dataset identity mismatch')
    return {'contract': contract, 'contract_sha256': sha256_file(contract_path),
            'registry': registry, 'rows': rows, 'datasets': datasets,
            'runner_sha256': sha256_file(here / 'evaluate.py')}


def verify_receipt_binding(receipt, key, bindings):
    contract, registry, rows = bindings['contract'], bindings['registry'], bindings['rows']
    dataset, model, seed = key
    selected = rows[key]
    bundle = registry['bundles'][selected['evaluator_source_bundle']]
    split = receipt['split']
    expected = {'registry_sha256': contract['registry_sha256'],
                'dataset_manifest_sha256': contract['dataset_manifest_sha256'],
                'contract_sha256': bindings['contract_sha256'],
                'runner_sha256': bindings['runner_sha256'],
                'data_file_sha256': bindings['datasets'][dataset]['sha256'],
                'source_closure_sha256': bundle['source_closure_sha256'],
                'full_population': True, 'evaluation_scope': 'full_population',
                'retrained': False, 'max_batches': None}
    expected.update({field: selected[field] for field in ['checkpoint_path', 'checkpoint_file_sha256',
                     'state_tensor_sha256', 'selected_epoch', 'evaluator_source_bundle']})
    specs = [row for row in bundle['datasets'] if row['dataset_id'] == dataset]
    if len(specs) != 1:
        raise ValueError('Frozen source dataset specification mismatch')
    expected['loader'] = specs[0]['loader']
    population = bindings['datasets'][dataset]['populations'][split]
    expected['prediction_rows'] = expected['expected_target_count'] = population['target_count']
    for field, value in expected.items():
        if field not in receipt or receipt[field] != value:
            raise ValueError('Receipt differs from frozen binding: ' + field)
    for field, value in [('datasets', dataset), ('models', model), ('seeds', seed), ('allowed_splits', split)]:
        if value not in contract['approval'][field]:
            raise ValueError('Receipt falls outside frozen approval')
    if split == 'test' and contract['approval'].get('test_inference_authorized') is not True:
        raise ValueError('Frozen contract does not authorize test evaluation')
    if receipt.get('truth_digest_fields') != TRUTH_FIELDS or receipt.get('schema') != {key: str(value) for key, value in SCHEMA.items()}:
        raise ValueError('Receipt serialization contract mismatch')
    if not receipt.get('qualification', {}).get('parameters_unchanged') or not receipt.get('qualification', {}).get('finite_outputs'):
        raise ValueError('Unqualified evaluation receipt')


def inspect_manifest(path):
    """Validate the full 108-condition plan before opening any prediction part."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    expected = {(d, m, s) for d in DATASETS for m in MODELS for s in SEEDS}
    conditions = manifest.get('conditions', [])
    keys = [(row['dataset'], row['model'], row['seed']) for row in conditions]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Run manifest must declare exactly all 108 learned conditions')
    by_key, receipts, missing, failures = {}, {}, [], []
    for key, raw in zip(keys, conditions):
        condition = dict(raw)
        root = Path(condition['output_dir'])
        if not root.is_absolute():
            root = path.parent / root
        condition['output_dir'] = str(root.resolve())
        by_key[key] = condition
        receipt_path = root / 'receipt.json'
        if not receipt_path.exists():
            entry = {'dataset': key[0], 'model': key[1], 'seed': key[2], 'output_dir': str(root)}
            if (root / 'failure.json').exists():
                entry['failure'] = json.loads((root / 'failure.json').read_text())
                failures.append(entry)
            else:
                missing.append(entry)
            continue
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('status') != 'complete' or tuple(receipt.get(k) for k in ['dataset', 'model', 'seed']) != key:
            raise ValueError('Receipt condition/status mismatch: ' + str(key))
        if receipt.get('split') not in ('test', 'validation', 'synthetic'):
            raise ValueError('Unspecified evaluation split')
        for field in ['data_file_sha256', 'target_identity_sha256', 'truth_sha256']:
            if not isinstance(receipt.get(field), str) or len(receipt[field]) != 64:
                raise ValueError('Missing receipt identity: ' + field)
        if receipt['split'] == 'test' and (receipt.get('full_population') is not True or receipt.get('evaluation_scope') != 'full_population'):
            raise ValueError('Test analysis requires full-population receipts')
        receipts[key] = receipt
    if missing or failures:
        return manifest, by_key, receipts, {
            'status': 'blocked_failed_conditions' if failures else 'pending_conditions',
            'expected_conditions': 108, 'completed_conditions': len(receipts),
            'missing_conditions': missing, 'failed_conditions': failures,
            'performance_aggregation_started': False, 'results': None,
            'interpretation': INTERPRETATION}
    if len({row['split'] for row in receipts.values()}) != 1:
        raise ValueError('Mixed evaluation splits are forbidden')
    if next(iter(receipts.values()))['split'] != 'synthetic':
        bindings = load_frozen_bindings(manifest, path)
        for key, receipt in receipts.items():
            verify_receipt_binding(receipt, key, bindings)
        manifest['_verified_frozen_datasets'] = bindings['datasets']
    return manifest, by_key, receipts, None


def analyze_manifest(path):
    manifest, conditions, receipts, pending = inspect_manifest(path)
    if pending:
        return pending
    results, validation = {}, {}
    for dataset in DATASETS:
        keys = [(dataset, model, seed) for model in MODELS for seed in SEEDS]
        split = receipts[keys[0]]['split']
        population = manifest.get('_verified_frozen_datasets', {}).get(dataset, {}).get('populations', {}).get(split)
        units, evidence = stream_dataset([conditions[key] for key in keys], [receipts[key] for key in keys], dataset, population)
        validation[dataset] = evidence
        scope = 'legacy_test_reevaluation' if receipts[keys[0]]['split'] == 'test' else 'validation_qualification'
        results[dataset] = {name: bootstrap_sums(units[key].counts, units[key].sums,
                                                dataset=dataset, kind=kind, cluster_key=key,
                                                block_hours=block, scope=scope)
                            for name, (kind, key, block) in analysis_specs(dataset).items()}
    return {'status': 'complete', 'run_manifest_sha256': sha256_file(path),
            'analysis_script_sha256': sha256_file(__file__), 'interpretation': INTERPRETATION,
            'split': next(iter(receipts.values()))['split'], 'learned_conditions': 108,
            'deterministic_reference_vectors': 8, 'prediction_ensemble': False,
            'v0_7_validation_baseline_replaced': False,
            'validation': validation, 'results': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = analyze_manifest(args.run_manifest)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        result = {'status': 'failed_contract_or_artifact', 'error_type': type(exc).__name__,
                  'error': str(exc), 'results': None, 'interpretation': INTERPRETATION}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output)}))
    return 0 if result['status'] in ('complete', 'pending_conditions') else 2


if __name__ == '__main__':
    raise SystemExit(main())
