"""Synthetic-only contract, boundary, conservation and pairing regression checks."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('analyze_strata', HERE / 'analyze_strata.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
legacy = a.load_legacy(HERE.parent / 'titantpp_legacy_evaluation_20261003_v1' / 'analyze.py')
MODELS = ['first', 'second']
SEEDS = [42, 52, 62]
SIMPLE = legacy.SIMPLE


def dataset_spec():
    return {'dataset': 'insta_market_basket', 'target_count': 6,
            'data_file_sha256': 'a' * 64, 'population': {},
            'quantity_boundaries': [1, 4, 10], 'history_boundaries': [1, 3, 7, 15, 31],
            'history_detail_boundaries': [1, 3, 7, 15, 31, 63, 127],
            'time_axis': {'kind': 'relative_day', 'min_tick': 3, 'max_tick': 8,
                          'semantics': 'synthetic user-relative days'}}


def frame(errors=1):
    ds = dataset_spec()
    entities, seqs, dates = ['a'] * 3 + ['b'] * 3, [1, 2, 3, 1, 2, 3], ['3', '4', '4', '5', '7', '8']
    quantities = np.array([0., 1., 3., 4., 10., 20.])
    data = {'target_index': list(range(6)), 'target_id': [legacy.digest([ds['dataset'], ds['data_file_sha256'], e, s, 'test', d])
                                                       for e, s, d in zip(entities, seqs, dates)],
            'entity_id': entities, 'seq': seqs, 'split': ['test'] * 6, 'recorded_date': dates,
            'time_bucket': [None] * 6, 'site_id': [None] * 6,
            'raw_quantity': quantities, 'predicted_raw_quantity': quantities + errors,
            'recorded_gap': [1.] * 6, 'time_nll': [1., 2., 3., 4., 5., 6.],
            'history_length': [1, 2, 4, 8, 16, 32], 'last_quantity': quantities + 2,
            'mean_quantity': quantities + 3}
    return pl.DataFrame(data, schema=legacy.SCHEMA)


def build_panel(tmp_path):
    ds = dataset_spec()
    canonical = frame()
    popbytes = b''.join((json.dumps(list(row), separators=(',', ':')) + '\n').encode()
                        for row in canonical.select('entity_id', 'seq', 'recorded_date').iter_rows())
    ds['population'] = {'target_count': 6, 'entity_count': 2,
                        'ordered_entity_seq_recorded_date_sha256': hashlib.sha256(popbytes).hexdigest()}
    conditions, receipts = [], []
    for mi, model in enumerate(MODELS):
        for si, seed in enumerate(SEEDS):
            root = tmp_path / f'{model}_{seed}'
            root.mkdir()
            f = frame(errors=mi + si + 1)
            parts, truth_global, identity_global = [], hashlib.sha256(), hashlib.sha256()
            size = 2 + si  # Different physical part boundaries must still pair exactly.
            for start in range(0, len(f), size):
                piece = f.slice(start, size)
                path = root / f'part_{start}.parquet'
                piece.write_parquet(path)
                truth = b''.join((json.dumps(row, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n').encode()
                                 for row in piece.select(legacy.TRUTH_FIELDS).iter_rows())
                ids = ('\n'.join(piece['target_id'].to_list()) + '\n').encode()
                truth_global.update(truth)
                identity_global.update(ids)
                parts.append({'path': path.name, 'rows': len(piece), 'sha256': a.sha256_file(path),
                              'target_index_start': start, 'target_index_end': start + len(piece) - 1,
                              'truth_sha256': hashlib.sha256(truth).hexdigest(),
                              'target_identity_sha256': hashlib.sha256(ids).hexdigest()})
            condition = {'dataset': ds['dataset'], 'model': model, 'seed': seed, 'output_dir': str(root)}
            receipt = {**condition, 'status': 'complete', 'split': 'test', 'full_population': True,
                       'evaluation_scope': 'full_population', 'data_file_sha256': ds['data_file_sha256'],
                       'expected_target_count': 6, 'prediction_rows': 6,
                       'truth_sha256': truth_global.hexdigest(), 'target_identity_sha256': identity_global.hexdigest(),
                       'registry_sha256': 'b' * 64, 'dataset_manifest_sha256': 'c' * 64, 'parts': parts}
            (root / 'receipt.json').write_text(json.dumps(receipt))
            conditions.append(condition)
            receipts.append(receipt)
    return ds, conditions, receipts


def test_ties_and_empty_architecture_bins():
    ds = dataset_spec()
    f = frame().with_columns(pl.Series('raw_quantity', [1., 1., 4., 4., 10., 10.]))
    g = a.assign_strata(f, ds)
    assert g['quantity'].tolist() == [0, 0, 1, 1, 2, 2]
    assert g['time'].tolist() == [0, 0, 0, 1, 2, 2]
    assert g['history'].tolist() == [0, 1, 2, 3, 4, 5]
    assert g['history_detail'].tolist() == [0, 1, 2, 3, 4, 5]
    acc = a.StratumSums(ds, 3)
    for slot in range(3):
        acc.add(f, slot, g, deterministic=slot > 0)
    out = a.summarize(acc, ds, ['model'], [42], SIMPLE)
    for index in ('6', '7'):
        b = out['history_detail'][index]
        assert b['count'] == 0
        assert all(v is None for v in b['means']['model'].values())
        assert b['seed_metrics']['model']['42']['squared_error_sum'] == 0
    assert out['quantity']['3']['count'] == 0
    assert out['history_detail']['0']['sample_sd'][SIMPLE[0]]['qty_rmse'] is None
    assert out['history_detail']['0']['seed_metrics'][SIMPLE[0]]['deterministic']['time_nll_sum'] is None


@pytest.mark.parametrize('kind,dates,expected', [
    ('calendar_hour', ['19700101000000', '19700101010000'], [0, 1]),
    ('calendar_week', ['19700101', '19700108'], [0, 1]),
    ('calendar_month', ['20011201', '20020101'], [24023, 24024]),
    ('relative_day', ['3', '365'], [3, 365]),
])
def test_native_time_ticks(kind, dates, expected):
    assert a.time_ticks(pl.Series(dates), kind).tolist() == expected


@pytest.mark.parametrize('dates,kind', [(['19700101003000'], 'calendar_hour'), (['3.5'], 'relative_day'),
                                       (['20011301'], 'calendar_month'), (['20011202'], 'calendar_month'),
                                       (['3'], 'unknown')])
def test_invalid_time_rejected(dates, kind):
    with pytest.raises((ValueError, pl.exceptions.InvalidOperationError)):
        a.time_ticks(pl.Series(dates), kind)


def test_frozen_time_range_and_integer_thirds():
    ds = dataset_spec()
    ds['time_axis'].update(min_tick=3, max_tick=7)
    with pytest.raises(ValueError, match='outside frozen'):
        a.assign_strata(frame(), ds)
    ds['time_axis'].update(min_tick=3, max_tick=8)
    definitions = [a.bin_definition(ds, 'time', i) for i in range(3)]
    assert [(d['min_tick_inclusive'], d['max_tick_inclusive']) for d in definitions] == [(3, 4), (5, 6), (7, 8)]


def test_seed_mean_rmse_is_not_pooled_rmse_and_conservation():
    ds = dataset_spec()
    acc = a.StratumSums(ds, 5)
    for slot, error in enumerate((0, 2, 10)):
        f = frame(error)
        acc.add(f, slot, a.assign_strata(f, ds))
    for slot, column in [(3, 'last_quantity'), (4, 'mean_quantity')]:
        f = frame()
        acc.add(f, slot, a.assign_strata(f, ds), column, True)
    out = a.summarize(acc, ds, ['model'], SEEDS, SIMPLE)
    assert out['all']['0']['means']['model']['qty_rmse'] == 4.
    assert not np.isclose(4., np.sqrt((0 + 4 + 100) / 3))
    assert out['all']['0']['sample_sd']['model']['qty_rmse'] == pytest.approx(np.std([0, 2, 10], ddof=1))
    assert acc.verify_partitions()['time']['sufficient_sums_conserve_all']
    assert acc.counts['time'][0].tolist() == [3, 1, 2]
    acc.sums['time'][0, 0, 1] += 1
    with pytest.raises(ValueError, match='sums do not conserve'):
        acc.verify_partitions()


def test_different_part_boundaries_exactly_pair(tmp_path):
    ds, conditions, receipts = build_panel(tmp_path)
    out, proof = a.stream_dataset(legacy, conditions, receipts, ds, MODELS, SEEDS, SIMPLE)
    assert out['all']['0']['means']['first']['qty_rmse'] == 2.
    assert out['all']['0']['means']['second']['qty_rmse'] == 3.
    assert len(proof['conditions']) == 6
    assert proof['exact_common_population']
    assert proof['partition_checks']['history_detail']['counts_conserve_all']
    assert len({json.dumps(r['independent_column_fingerprints'], sort_keys=True) for r in proof['conditions']}) == 1


@pytest.mark.parametrize('field,value', [('seed', 99), ('split', 'validation'), ('full_population', False),
    ('truth_sha256', 'z' * 64), ('target_identity_sha256', 'z' * 64), ('data_file_sha256', 'z' * 64),
    ('prediction_rows', 5), ('dataset_manifest_sha256', 'z' * 64)])
def test_receipt_metadata_mismatch(tmp_path, field, value):
    ds, conditions, receipts = build_panel(tmp_path)
    receipts[1][field] = value
    with pytest.raises(ValueError, match='Receipt'):
        a.stream_dataset(legacy, conditions, receipts, ds, MODELS, SEEDS, SIMPLE)


def test_altered_history_cannot_hide_behind_receipt_digest(tmp_path):
    ds, conditions, receipts = build_panel(tmp_path)
    part = receipts[1]['parts'][0]
    path = Path(conditions[1]['output_dir']) / part['path']
    f = pl.read_parquet(path).with_columns((pl.col('history_length') + 1).alias('history_length'))
    f.write_parquet(path)
    part['sha256'] = a.sha256_file(path)
    with pytest.raises(ValueError, match='identity/truth/history mismatch'):
        a.stream_dataset(legacy, conditions, receipts, ds, MODELS, SEEDS, SIMPLE)


@pytest.mark.parametrize('mutation,match', [('sha', 'SHA256'), ('rows', 'row-count'),
                                          ('indices', 'target range'), ('duplicate', 'duplicate'),
                                          ('traversal', 'Unsafe'), ('population', 'population identity')])
def test_part_and_population_integrity(tmp_path, mutation, match):
    ds, conditions, receipts = build_panel(tmp_path)
    part = receipts[0]['parts'][0]
    if mutation == 'sha':
        part['sha256'] = '0' * 64
    elif mutation == 'rows':
        part['rows'] += 1
    elif mutation == 'indices':
        part['target_index_start'] = 1
    elif mutation == 'duplicate':
        receipts[0]['parts'].insert(1, copy.deepcopy(part))
    elif mutation == 'traversal':
        part['path'] = '../outside.parquet'
    else:
        ds['population']['ordered_entity_seq_recorded_date_sha256'] = '0' * 64
    with pytest.raises(ValueError, match=match):
        a.stream_dataset(legacy, conditions, receipts, ds, MODELS, SEEDS, SIMPLE)


def test_reconcile_rejects_seed_sd_and_mean_changes(tmp_path):
    ds, conditions, receipts = build_panel(tmp_path)
    out, _ = a.stream_dataset(legacy, conditions, receipts, ds, MODELS, SEEDS, SIMPLE)
    current = out['all']['0']
    assert a.reconcile_all(current, copy.deepcopy(current), MODELS, SEEDS, SIMPLE)['verified']
    for field in ('means', 'sample_sd', 'seed_metrics'):
        previous = copy.deepcopy(current)
        row = previous[field]['first']
        if field == 'seed_metrics':
            row = row['42']
        row['qty_rmse'] += 0.01
        with pytest.raises(ValueError, match='metric mismatch'):
            a.reconcile_all(current, previous, MODELS, SEEDS, SIMPLE)


def test_frozen_model_selected_epoch_binding_rejected():
    key = ('insta_market_basket', 'first', 42)
    selected = {'checkpoint_path': 'weights.pt', 'checkpoint_file_sha256': 'a', 'state_tensor_sha256': 'b',
                'selected_epoch': 7, 'evaluator_source_bundle': 'bundle'}
    contract = {'registry_sha256': 'c', 'dataset_manifest_sha256': 'd', 'approval': {
        'datasets': [key[0]], 'models': [key[1]], 'seeds': [42], 'allowed_splits': ['test'], 'test_inference_authorized': True}}
    bindings = {'contract': contract, 'contract_sha256': 'e', 'runner_sha256': 'f', 'rows': {key: selected},
                'registry': {'bundles': {'bundle': {'source_closure_sha256': 'g',
                              'datasets': [{'dataset_id': key[0], 'loader': 'frozen_loader'}]}}},
                'datasets': {key[0]: {'sha256': 'h', 'populations': {'test': {'target_count': 6}}}}}
    receipt = {**selected, 'registry_sha256': 'c', 'dataset_manifest_sha256': 'd', 'contract_sha256': 'e',
               'runner_sha256': 'f', 'source_closure_sha256': 'g', 'data_file_sha256': 'h',
               'full_population': True, 'evaluation_scope': 'full_population', 'retrained': False,
               'max_batches': None, 'loader': 'frozen_loader', 'prediction_rows': 6, 'expected_target_count': 6,
               'split': 'test', 'truth_digest_fields': legacy.TRUTH_FIELDS,
               'schema': {k: str(v) for k, v in legacy.SCHEMA.items()},
               'qualification': {'parameters_unchanged': True, 'finite_outputs': True}}
    legacy.verify_receipt_binding(receipt, key, bindings)
    receipt['selected_epoch'] = 8
    with pytest.raises(ValueError, match='selected_epoch'):
        legacy.verify_receipt_binding(receipt, key, bindings)


def test_source_hash_binding_and_outputs_preserve_existing(tmp_path):
    paths = [tmp_path / n for n in ['run_manifest.json', 'analysis.json', 'analyze.py']]
    for path in paths:
        path.write_text('{}')
    contract = {'source_bundle': str(tmp_path), 'source_run_manifest': str(paths[0]),
                'source_analysis': str(paths[1]), 'source_hashes': {str(p): a.sha256_file(p) for p in paths}}
    a.verify_source_hashes(contract)
    paths[2].write_text('changed')
    with pytest.raises(ValueError, match='SHA mismatch'):
        a.verify_source_hashes(contract)
    with pytest.raises(FileExistsError):
        a.write_outputs(tmp_path, {}, {})


def test_csv_has_split_bin_definitions_and_no_fake_reference_seeds(tmp_path):
    ds, conditions, receipts = build_panel(tmp_path)
    out, _ = a.stream_dataset(legacy, conditions, receipts, ds, MODELS, SEEDS, SIMPLE)
    destination = tmp_path / 'output'
    a.write_outputs(destination, {'results': {ds['dataset']: out}}, {'status': 'synthetic'})
    mean = pl.read_csv(destination / 'metrics.csv')
    assert mean['split'].unique().to_list() == ['test']
    assert {'bin_label', 'lower_exclusive', 'upper_inclusive', 'time_min_tick', 'time_max_tick'} <= set(mean.columns)
    seed = pl.read_csv(destination / 'seed_metrics.csv')
    refs = seed.filter(pl.col('deterministic'))
    assert refs['seed'].null_count() == len(refs)
    assert refs['time_nll_sum'].null_count() == len(refs)
