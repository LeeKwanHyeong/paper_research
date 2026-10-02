"""Synthetic qualification only: no held-out data, inference, or checkpoint I/O."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


new = load('legacy_analysis', HERE / 'analyze.py')
old = load('validation_statistics', HERE.parent / 'titantpp_final_eval_runner_validation_20261003_v1' / 'paired_statistics.py')


def write_part(root, rows, number=0):
    path = root / 'predictions' / f'part-{number:05d}.parquet'
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(rows, schema=new.SCHEMA)
    frame.write_parquet(path)
    raw_ids = ('\n'.join(frame['target_id'].to_list()) + '\n').encode()
    truth = b''.join((json.dumps([row[key] for key in new.TRUTH_FIELDS],
                                separators=(',', ':'), ensure_ascii=False) + '\n').encode() for row in rows)
    return {'path': str(path.relative_to(root)), 'sha256': new.sha256_file(path),
            'rows': len(rows), 'target_index_start': rows[0]['target_index'],
            'target_index_end': rows[-1]['target_index'],
            'target_identity_sha256': new.hashlib.sha256(raw_ids).hexdigest(),
            'truth_sha256': new.hashlib.sha256(truth).hexdigest()}


def fixture_dataset(root, dataset):
    conditions, receipts, records = [], [], []
    metadata = []
    for entity_index in range(8):
        entity = f's{entity_index // 2}::e{entity_index}'
        for seq in range(4):
            i = len(metadata)
            date = str(np.datetime64('2020-01-01T00', 'h') + np.timedelta64(i * 12, 'h'))
            q = float(i % 7 + 2)
            row = {'target_index': i, 'target_id': new.digest([dataset, '1' * 64, entity, seq, 'synthetic', date]),
                   'entity_id': entity, 'seq': seq, 'split': 'synthetic', 'recorded_date': date,
                   'time_bucket': date if dataset == new.DATASETS[0] else None,
                   'site_id': entity.split('::')[0] if dataset == new.DATASETS[1] else None,
                   'raw_quantity': q, 'recorded_gap': float(seq + 1), 'history_length': seq + 1,
                   'last_quantity': q - 1, 'mean_quantity': q - .5}
            metadata.append(row)
    for mi, model in enumerate(new.MODELS):
        for si, seed in enumerate(new.SEEDS):
            output = root / model / str(seed)
            rows = [{**row, 'predicted_raw_quantity': row['raw_quantity'] + (mi + 1) * .12 + (si - 1) * .3 * (row['seq'] + 1),
                     'time_nll': .5 + mi * .1 + si * .05 + row['seq'] * .02} for row in metadata]
            parts = [write_part(output, rows[:17]), write_part(output, rows[17:], 1)]
            ids = ('\n'.join(row['target_id'] for row in rows) + '\n').encode()
            truth = b''.join((json.dumps([row[key] for key in new.TRUTH_FIELDS], separators=(',', ':'), ensure_ascii=False) + '\n').encode() for row in rows)
            receipt = {'status': 'complete', 'dataset': dataset, 'model': model, 'seed': seed,
                       'split': 'synthetic', 'evaluation_scope': 'validation_qualification',
                       'full_population': False, 'data_file_sha256': '1' * 64,
                       'expected_target_count': len(rows), 'prediction_rows': len(rows),
                       'target_identity_sha256': new.hashlib.sha256(ids).hexdigest(),
                       'truth_sha256': new.hashlib.sha256(truth).hexdigest(), 'parts': parts}
            (output / 'receipt.json').write_text(json.dumps(receipt))
            conditions.append({'dataset': dataset, 'model': model, 'seed': seed, 'output_dir': str(output)})
            receipts.append(receipt)
            records += [{**row, 'dataset': dataset, 'model': model, 'seed': seed,
                         'evaluation_scope': 'synthetic', 'data_file_sha256': '1' * 64,
                         'recorded_event_key': [row['entity_id'], row['seq'], row['recorded_date']]} for row in rows]
    for model, column in zip(new.SIMPLE, ['last_quantity', 'mean_quantity']):
        records += [{**row, 'dataset': dataset, 'model': model, 'seed': None,
                     'predicted_raw_quantity': row[column], 'time_nll': None,
                     'evaluation_scope': 'synthetic', 'data_file_sha256': '1' * 64,
                     'recorded_event_key': [row['entity_id'], row['seq'], row['recorded_date']]} for row in metadata]
    return conditions, receipts, records


def assert_recursive_equal(left, right):
    if isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            assert_recursive_equal(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_recursive_equal(a, b)
    elif isinstance(left, (float, int)) and not isinstance(left, bool):
        assert left == pytest.approx(right, abs=1e-12, rel=1e-12)
    else:
        assert left == right


@pytest.mark.parametrize('dataset', new.DATASETS)
def test_streaming_bootstrap_matches_frozen_validation_math(tmp_path, dataset):
    conditions, receipts, records = fixture_dataset(tmp_path, dataset)
    units, evidence = new.stream_dataset(conditions, receipts, dataset)
    assert evidence['target_count'] == 32
    assert evidence['exact_common_population']
    assert len(evidence['conditions']) == 27
    for name, (kind, key, block) in new.analysis_specs(dataset).items():
        common = {'draws': 96, 'rng_seed': 20261001 + new.DATASETS.index(dataset)}
        current = new.bootstrap_sums(units[key].counts, units[key].sums, dataset=dataset,
                                     kind=kind, cluster_key=key, block_hours=block,
                                     scope='synthetic', **common)
        expected = old.paired_bootstrap(records, new.MODELS + new.SIMPLE, new.SEEDS,
                                        kind=kind, cluster_key=key, block_hours=block,
                                        anchor='rmtpp' if new.DATASETS.index(dataset) < 2 else 's2p2_matched_head',
                                        deterministic_models=new.SIMPLE, **common)
        for field in ['seed_metrics', 'means', 'sample_sd', 'interval_status', 'intervals',
                      'draws_completed', 'resampling_weights_sha256']:
            if field in expected:
                assert_recursive_equal(current[field], expected[field])
        for model in new.SIMPLE:
            assert set(current['seed_metrics'][model]) == {'deterministic'}
            assert current['means'][model]['time_nll'] is None
            assert set(current['sample_sd'][model].values()) == {None}
            if current['intervals']:
                assert current['intervals'][model]['time_nll'] is None
        assert current['estimate'] == 'mean_of_per_seed_metrics_not_prediction_ensemble'


@pytest.mark.parametrize('field,value', [('raw_quantity', 99.), ('recorded_gap', 5.),
                                        ('history_length', 99), ('last_quantity', 7.),
                                        ('mean_quantity', 8.), ('recorded_date', '2021-01-01'),
                                        ('entity_id', 'different'), ('site_id', 'other')])
def test_cross_condition_metadata_mismatch_is_rejected(tmp_path, field, value):
    conditions, receipts, _ = fixture_dataset(tmp_path, new.DATASETS[1])
    root = Path(conditions[1]['output_dir'])
    rows = pl.read_parquet(root / receipts[1]['parts'][0]['path']).to_dicts()
    rows[0][field] = value
    receipts[1]['parts'][0] = write_part(root, rows)
    with pytest.raises(ValueError, match='identity/truth/history mismatch|new resampling unit'):
        new.stream_dataset(conditions, receipts, new.DATASETS[1])


def test_receipt_data_identity_mismatch_is_rejected(tmp_path):
    conditions, receipts, _ = fixture_dataset(tmp_path, new.DATASETS[3])
    receipts[1]['data_file_sha256'] = '2' * 64
    with pytest.raises(ValueError, match='data_file_sha256'):
        new.stream_dataset(conditions, receipts, new.DATASETS[3])


def test_missing_rows_never_intersect(tmp_path):
    conditions, receipts, _ = fixture_dataset(tmp_path, new.DATASETS[3])
    receipts[1]['parts'].pop()
    with pytest.raises(ValueError, match='Incomplete target population'):
        new.stream_dataset(conditions, receipts, new.DATASETS[3])


def test_nonfinite_predictions_reject_affected_analysis(tmp_path):
    conditions, receipts, _ = fixture_dataset(tmp_path, new.DATASETS[3])
    root = Path(conditions[1]['output_dir'])
    rows = pl.read_parquet(root / receipts[1]['parts'][0]['path']).to_dicts()
    rows[0]['predicted_raw_quantity'] = float('nan')
    receipts[1]['parts'][0] = write_part(root, rows)
    with pytest.raises(ValueError, match='Nonfinite'):
        new.stream_dataset(conditions, receipts, new.DATASETS[3])


def test_column_fingerprints_do_not_depend_on_part_boundaries(tmp_path):
    conditions, receipts, _ = fixture_dataset(tmp_path, new.DATASETS[3])
    root = Path(conditions[1]['output_dir'])
    rows = pl.concat([pl.read_parquet(root / part['path']) for part in receipts[1]['parts']]).to_dicts()
    receipts[1]['parts'] = [write_part(root, rows)]
    units, evidence = new.stream_dataset(conditions, receipts, new.DATASETS[3])
    assert evidence['exact_common_population']
    assert units['entity_id'].counts.sum() == len(rows)


def test_short_taxi_span_remains_nonestimable():
    sums = np.ones((100, 29, 3), dtype=np.float64)
    result = new.bootstrap_sums(np.ones(100, dtype=np.int64), sums,
                                dataset=new.DATASETS[0], kind='calendar',
                                cluster_key='time_bucket', block_hours=168)
    assert result['intervals'] is None
    assert result['interval_status'] == 'not_estimable_insufficient_units_or_span'
    assert result['draws_completed'] == 0
    assert result['small_sample_descriptive_only']


def test_empty_calendar_draw_is_not_silently_redrawn():
    counts = np.zeros(48, dtype=np.int64)
    counts[0] = 1
    sums = np.zeros((48, 29, 3), dtype=np.float64)
    sums[0] = 1
    result = new.bootstrap_sums(counts, sums, dataset=new.DATASETS[0], kind='calendar',
                                cluster_key='time_bucket', block_hours=24, draws=100)
    assert result['interval_status'] == 'not_finalized_invalid_draws'
    assert result['invalid_empty_draws'] > 0
    assert result['draws_completed'] == 100
    assert result['intervals'] is None


def test_pending_manifest_reads_no_prediction_parts(tmp_path, monkeypatch):
    manifest = {'conditions': [{'dataset': d, 'model': m, 'seed': s,
                                'output_dir': str(tmp_path / d / m / str(s))}
                               for d in new.DATASETS for m in new.MODELS for s in new.SEEDS]}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(pl, 'read_parquet', lambda *args, **kwargs: pytest.fail('pending manifest accessed prediction parts'))
    result = new.analyze_manifest(path)
    assert result['status'] == 'pending_conditions'
    assert len(result['missing_conditions']) == 108
    assert result['results'] is None
    root = Path(manifest['conditions'][0]['output_dir'])
    root.mkdir(parents=True)
    (root / 'failure.json').write_text(json.dumps({'reason': 'synthetic failure'}))
    result = new.analyze_manifest(path)
    assert result['status'] == 'blocked_failed_conditions'
    assert len(result['failed_conditions']) == 1


def test_partial_or_duplicate_manifest_is_rejected(tmp_path):
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'conditions': []}))
    with pytest.raises(ValueError, match='exactly all 108'):
        new.analyze_manifest(path)


def test_seed_metric_average_is_not_an_ensemble():
    sums = np.ones((2, 29, 3))
    sums[:, 0, 1], sums[:, 1, 1], sums[:, 2, 1] = 1, 4, 16
    result = new.bootstrap_sums(np.ones(2, dtype=np.int64), sums,
                                dataset=new.DATASETS[3], kind='cluster', cluster_key='entity_id', draws=8)
    assert result['means'][new.MODELS[0]]['qty_rmse'] == pytest.approx((1 + 2 + 4) / 3)
    assert result['sample_sd'][new.MODELS[0]]['qty_rmse'] == pytest.approx(np.std([1, 2, 4], ddof=1))


def test_target_order_and_file_integrity_rejections(tmp_path):
    conditions, receipts, _ = fixture_dataset(tmp_path, new.DATASETS[3])
    root = Path(conditions[0]['output_dir'])
    part = receipts[0]['parts'][0]
    rows = pl.read_parquet(root / part['path']).to_dicts()
    rows[0]['target_index'] = 99
    frame = pl.DataFrame(rows, schema=new.SCHEMA)
    frame.write_parquet(root / part['path'])
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        new.stream_dataset(conditions, receipts, new.DATASETS[3])
    receipts[0]['parts'][0]['sha256'] = new.sha256_file(root / part['path'])
    with pytest.raises(ValueError, match='Target index order mismatch'):
        new.stream_dataset(conditions, receipts, new.DATASETS[3])


@pytest.mark.parametrize('mismatch', ['count', 'population_hash', 'receipt_truth', 'part_truth'])
def test_identically_wrong_panel_or_unverified_truth_is_rejected(tmp_path, mismatch):
    conditions, receipts, _ = fixture_dataset(tmp_path, new.DATASETS[3])
    population = {'target_count': 32, 'entity_count': 8,
                  'ordered_entity_seq_recorded_date_sha256': 'f' * 64}
    if mismatch == 'count':
        population['target_count'] = 33
    elif mismatch == 'receipt_truth':
        for receipt in receipts:
            receipt['truth_sha256'] = 'f' * 64
        population = None
    elif mismatch == 'part_truth':
        receipts[0]['parts'][0]['truth_sha256'] = 'f' * 64
        population = None
    with pytest.raises(ValueError, match='frozen dataset manifest|truth digest mismatch'):
        new.stream_dataset(conditions, receipts, new.DATASETS[3], population)


def test_receipts_bind_exact_checkpoint_source_and_population():
    dataset, model, seed = new.DATASETS[3], new.MODELS[0], new.SEEDS[0]
    key = dataset, model, seed
    selected = {'checkpoint_path': 'frozen.pt', 'checkpoint_file_sha256': 'a' * 64,
                'state_tensor_sha256': 'b' * 64, 'selected_epoch': 12,
                'evaluator_source_bundle': 'bundle'}
    loader = {'batch_size': 128, 'max_seq_len': 32}
    contract = {'registry_sha256': 'c' * 64, 'dataset_manifest_sha256': 'd' * 64,
                'approval': {'datasets': [dataset], 'models': [model], 'seeds': [seed],
                             'allowed_splits': ['test'], 'test_inference_authorized': True}}
    bindings = {'contract': contract, 'contract_sha256': 'e' * 64,
                'runner_sha256': 'f' * 64, 'rows': {key: selected},
                'datasets': {dataset: {'sha256': '1' * 64, 'populations': {'test': {'target_count': 32}}}},
                'registry': {'bundles': {'bundle': {'source_closure_sha256': '2' * 64,
                              'datasets': [{'dataset_id': dataset, 'loader': loader}]}}}}
    receipt = {**selected, 'split': 'test', 'registry_sha256': 'c' * 64,
               'dataset_manifest_sha256': 'd' * 64, 'contract_sha256': 'e' * 64,
               'runner_sha256': 'f' * 64, 'data_file_sha256': '1' * 64,
               'source_closure_sha256': '2' * 64, 'loader': loader,
               'prediction_rows': 32, 'expected_target_count': 32,
               'full_population': True, 'evaluation_scope': 'full_population',
               'retrained': False, 'max_batches': None, 'truth_digest_fields': new.TRUTH_FIELDS,
               'schema': {key: str(value) for key, value in new.SCHEMA.items()},
               'qualification': {'parameters_unchanged': True, 'finite_outputs': True}}
    new.verify_receipt_binding(receipt, key, bindings)
    for field in ['prediction_rows', 'expected_target_count', 'data_file_sha256',
                  'registry_sha256', 'contract_sha256', 'checkpoint_file_sha256',
                  'state_tensor_sha256', 'source_closure_sha256', 'selected_epoch']:
        with pytest.raises(ValueError, match='frozen binding: ' + field):
            new.verify_receipt_binding({**receipt, field: 'incorrect'}, key, bindings)


def test_production_manifest_cannot_skip_external_identity_binding(tmp_path):
    with pytest.raises(ValueError, match='missing frozen artifact paths'):
        new.load_frozen_bindings({}, tmp_path / 'manifest.json')
    contract_path = HERE / 'execution_contract.json'
    contract = json.loads(contract_path.read_text())
    manifest = {'execution_contract_path': str(contract_path),
                'execution_contract_sha256': new.sha256_file(contract_path),
                'dataset_manifest_path': str(HERE / 'dataset_manifest.json'),
                'registry_path': str(HERE.parents[1] / contract['registry_path']),
                'registry_sha256': 'wrong'}
    with pytest.raises(ValueError, match='Frozen/run manifest identity mismatch'):
        new.load_frozen_bindings(manifest, tmp_path / 'manifest.json')
