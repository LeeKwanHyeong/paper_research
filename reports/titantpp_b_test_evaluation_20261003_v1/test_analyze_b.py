"""Small synthetic tests only; no real predictions, inference, or remote I/O."""
import copy
import hashlib
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


b = load('b_analysis_test', HERE / 'analyze_b.py')
old = load('independent_row_bootstrap', HERE.parent /
           'titantpp_final_eval_runner_validation_20261003_v1' / 'paired_statistics.py')
renderer = load('b_renderer_test', HERE / 'render_b.py')


def write_part(root, rows, index=0):
    path = root / 'predictions' / f'part-{index:05d}.parquet'
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(rows, schema=b.legacy.SCHEMA)
    frame.write_parquet(path)
    truth = b''.join((json.dumps([row[key] for key in b.legacy.TRUTH_FIELDS],
                                separators=(',', ':'), ensure_ascii=False) + '\n').encode() for row in rows)
    ids = ('\n'.join(row['target_id'] for row in rows) + '\n').encode()
    return {'path': str(path.relative_to(root)), 'rows': len(rows),
            'sha256': b.sha256_file(path), 'target_index_start': rows[0]['target_index'],
            'target_index_end': rows[-1]['target_index'],
            'truth_sha256': hashlib.sha256(truth).hexdigest(),
            'target_identity_sha256': hashlib.sha256(ids).hexdigest()}


def fixture_panel(root, dataset):
    metadata, conditions, receipts, records = [], [], [], []
    for entity in range(8):
        for seq in range(4):
            i = len(metadata)
            identity = f's{entity // 2}::e{entity}'
            date = str(np.datetime64('2020-01-01T00', 'h') + np.timedelta64(i * 175 // 31, 'h'))
            quantity = float(2 + i % 7)
            metadata.append({'target_index': i, 'target_id': b.legacy.digest(
                [dataset, '1' * 64, identity, seq, 'synthetic', date]),
                'entity_id': identity, 'seq': seq, 'split': 'synthetic', 'recorded_date': date,
                'time_bucket': date if dataset == b.DATASETS[0] else None,
                'site_id': identity.split('::')[0] if dataset == b.DATASETS[1] else None,
                'raw_quantity': quantity, 'recorded_gap': float(seq + 1),
                'history_length': seq + 1, 'last_quantity': quantity - 1,
                'mean_quantity': quantity - .5})
    for mi, model in enumerate(b.MODELS):
        for si, seed in enumerate(b.SEEDS):
            output = root / dataset / model / str(seed)
            rows = [{**row, 'predicted_raw_quantity': row['raw_quantity'] + (mi + 1) * .4 +
                     (si - 1) * .2 * (row['seq'] + 1),
                     'time_nll': .5 + mi * .1 + si * .05 + row['seq'] * .02} for row in metadata]
            # Different part boundaries must not change population fingerprints.
            boundary = 13 + si
            parts = [write_part(output, rows[:boundary]), write_part(output, rows[boundary:], 1)]
            truth = b''.join((json.dumps([row[key] for key in b.legacy.TRUTH_FIELDS],
                                        separators=(',', ':'), ensure_ascii=False) + '\n').encode() for row in rows)
            ids = ('\n'.join(row['target_id'] for row in rows) + '\n').encode()
            receipt = {'status': 'complete', 'dataset': dataset, 'model': model, 'seed': seed,
                       'split': 'synthetic', 'evaluation_scope': 'full_population', 'full_population': True,
                       'data_file_sha256': '1' * 64, 'expected_target_count': len(rows),
                       'prediction_rows': len(rows), 'target_identity_sha256': hashlib.sha256(ids).hexdigest(),
                       'truth_sha256': hashlib.sha256(truth).hexdigest(), 'parts': parts,
                       'registry_sha256': ('2' if model == b.B else '3') * 64,
                       'dataset_manifest_sha256': ('4' if model == b.B else '5') * 64}
            (output / 'receipt.json').write_text(json.dumps(receipt))
            conditions.append({'dataset': dataset, 'model': model, 'seed': seed, 'output_dir': str(output)})
            receipts.append(receipt)
            records.extend({**row, 'dataset': dataset, 'model': model, 'seed': seed,
                            'evaluation_scope': 'synthetic', 'data_file_sha256': '1' * 64,
                            'recorded_event_key': [row['entity_id'], row['seq'], row['recorded_date']]}
                           for row in rows)
    digest = hashlib.sha256()
    for row in metadata:
        digest.update((json.dumps([row['entity_id'], row['seq'], row['recorded_date']],
                                  separators=(',', ':')) + '\n').encode())
    population = {'target_count': 32, 'entity_count': 8,
                  'ordered_entity_seq_recorded_date_sha256': digest.hexdigest()}
    return conditions, receipts, records, population


def assert_equal(left, right):
    if isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            assert_equal(first, second)
    elif isinstance(left, (int, float)) and not isinstance(left, bool):
        assert left == pytest.approx(right, rel=1e-12, abs=1e-12)
    else:
        assert left == right


@pytest.mark.parametrize('dataset', b.DATASETS)
def test_stream_and_bootstrap_match_independent_row_estimator(tmp_path, dataset):
    conditions, receipts, records, population = fixture_panel(tmp_path, dataset)
    units, evidence = b.stream_paired_dataset(conditions, receipts, dataset, population)
    assert evidence['exact_common_population'] and evidence['frozen_population_identity_verified']
    assert len(evidence['conditions']) == 24
    assert len({row['registry_sha256'] for row in evidence['conditions']}) == 2
    assert 'not independent reconstruction' in evidence['paired_history_check_scope']
    for _, (kind, key, block) in b.paired.analysis_specs(dataset).items():
        actual = b.bootstrap_b(units[key].counts, units[key].sums, dataset=dataset,
                               kind=kind, cluster_key=key, block_hours=block, draws=64)
        expected = old.paired_bootstrap(records, b.MODELS, b.SEEDS, representative=b.B,
                                       kind=kind, cluster_key=key, block_hours=block,
                                       draws=64, rng_seed=20261001 + b.DATASETS.index(dataset))
        for field in ['seed_metrics', 'means', 'sample_sd', 'interval_status', 'intervals', 'draws_completed']:
            assert_equal(actual[field], expected[field])
        assert actual.get('resampling_weights_sha256') == expected.get('resampling_weights_sha256')
        assert actual['representative_model'] == b.REPRESENTATIVE
        assert actual['representative_replaced'] is False
        assert actual['exploratory'] is True
        assert 'four_anchor' not in json.dumps(actual)
        for model in b.MODELS[1:]:
            for metric in b.paired.METRICS:
                assert actual['point_differences'][model][metric] == pytest.approx(
                    actual['means'][b.B][metric] - actual['means'][model][metric])
        if dataset == b.DATASETS[0] and block in (168, 336):
            assert actual['intervals'] is None and actual['draws_completed'] == 0


@pytest.mark.parametrize('field,value', [('raw_quantity', 20.), ('recorded_gap', 20.),
                                        ('history_length', 20), ('last_quantity', 20.),
                                        ('mean_quantity', 20.), ('recorded_date', '2021-01-01'),
                                        ('target_id', '9' * 64), ('site_id', 'other')])
def test_changed_paired_truth_history_or_identity_is_rejected(tmp_path, field, value):
    dataset = b.DATASETS[1]
    conditions, receipts, _, population = fixture_panel(tmp_path, dataset)
    output = Path(conditions[3]['output_dir'])
    rows = pl.read_parquet(output / receipts[3]['parts'][0]['path']).to_dicts()
    rows[0][field] = value
    receipts[3]['parts'][0] = write_part(output, rows)
    with pytest.raises(ValueError, match='identity|truth|history|resampling'):
        b.stream_paired_dataset(conditions, receipts, dataset, population)


def test_part_digest_and_frozen_population_are_not_optional(tmp_path):
    dataset = b.DATASETS[2]
    conditions, receipts, _, population = fixture_panel(tmp_path, dataset)
    receipts[2]['parts'][0]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='SHA256'):
        b.stream_paired_dataset(conditions, receipts, dataset, population)
    receipts[2]['parts'][0]['sha256'] = b.sha256_file(
        Path(conditions[2]['output_dir']) / receipts[2]['parts'][0]['path'])
    with pytest.raises(ValueError, match='frozen dataset manifest'):
        b.stream_paired_dataset(conditions, receipts, dataset, {**population, 'target_count': 31})


def manifest_rows(tmp_path):
    return [{'dataset': dataset, 'model': b.B, 'seed': seed,
             'output_dir': str(tmp_path / dataset / str(seed))}
            for dataset in b.DATASETS for seed in b.SEEDS]


def test_missing_manifest_conditions_do_not_open_predictions(tmp_path, monkeypatch):
    path = tmp_path / 'run_manifest.json'
    path.write_text(json.dumps({'conditions': manifest_rows(tmp_path)}))
    monkeypatch.setattr(b.paired, 'stream_dataset', lambda *a: pytest.fail('read predictions before completion'))
    result = b.analyze_manifest(path, 'not-opened', 'not-opened', 'not-opened', 'not-opened', 'not-opened')
    assert result['status'] == 'pending_conditions'
    assert result['performance_aggregation_started'] is False
    assert result['results'] is None


@pytest.mark.parametrize('mutation', ['duplicate', 'extra', 'wrong_model'])
def test_manifest_scope_cannot_expand_or_duplicate(tmp_path, mutation):
    rows = manifest_rows(tmp_path)
    if mutation == 'duplicate':
        rows[-1] = dict(rows[0])
    elif mutation == 'extra':
        rows.append({**rows[0], 'seed': 999})
    else:
        rows[0]['model'] = b.REPRESENTATIVE
    path = tmp_path / 'run_manifest.json'
    path.write_text(json.dumps({'conditions': rows}))
    with pytest.raises(ValueError, match='nine unique'):
        b.inspect_b_manifest(path)


def test_failure_record_blocks_even_if_receipt_exists(tmp_path):
    rows = manifest_rows(tmp_path)
    output = Path(rows[0]['output_dir'])
    output.mkdir(parents=True)
    (output / 'failure.json').write_text('{"error":"retained failure"}')
    path = tmp_path / 'run_manifest.json'
    path.write_text(json.dumps({'conditions': rows}))
    assert b.inspect_b_manifest(path)[3]['status'] == 'blocked_failed_conditions'


def test_legacy_draw_hash_and_metric_identity_are_checked(tmp_path):
    dataset = b.DATASETS[1]
    conditions, receipts, _, population = fixture_panel(tmp_path, dataset)
    units, _ = b.stream_paired_dataset(conditions, receipts, dataset, population)
    panel = b.bootstrap_b(units['site_id'].counts, units['site_id'].sums, dataset=dataset,
                          kind='cluster', cluster_key='site_id', draws=8)
    prior = {'results': {dataset: {'primary_site': copy.deepcopy(panel)}}}
    b.verify_legacy_results(panel, prior, dataset, 'primary_site')
    prior['results'][dataset]['primary_site']['resampling_weights_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='resampling differs'):
        b.verify_legacy_results(panel, prior, dataset, 'primary_site')
    prior['results'][dataset]['primary_site'] = copy.deepcopy(panel)
    prior['results'][dataset]['primary_site']['seed_metrics'][b.REPRESENTATIVE]['42']['time_nll'] += .01
    with pytest.raises(ValueError, match='metric changed'):
        b.verify_legacy_results(panel, prior, dataset, 'primary_site')


def test_outputs_are_never_overwritten(tmp_path):
    output = tmp_path / 'analysis.json'
    output.write_text('preserved')
    args = [item for name in ['run-manifest', 'legacy-run-manifest', 'contract', 'registry',
                             'dataset-manifest', 'legacy-analysis'] for item in ['--' + name, 'missing']]
    with pytest.raises(SystemExit):
        b.main(args + ['--output', str(output)])
    assert output.read_text() == 'preserved'


def test_private_model_override_does_not_mutate_legacy_binding_module():
    assert b.legacy.MODELS[0] == b.REPRESENTATIVE
    assert len(b.legacy.MODELS) == 9 and b.B not in b.legacy.MODELS
    assert b.paired.MODELS == b.MODELS


def fixture_binding(tmp_path, monkeypatch):
    here = tmp_path / 'reports' / 'b'
    here.mkdir(parents=True)
    runner = here / 'evaluate.py'
    runner.write_bytes((b.LEGACY / 'evaluate.py').read_bytes())
    script = here / 'analyze_b.py'
    script.write_bytes((HERE / 'analyze_b.py').read_bytes())
    monkeypatch.setattr(b, 'HERE', here)
    monkeypatch.setattr(b, '__file__', str(script))
    paths = {key: here / (key + '.json') for key in ['execution_contract', 'evaluation_registry', 'dataset_manifest']}
    registry = {'rows': [{'dataset': d, 'model': b.B, 'seed': s} for d in b.DATASETS for s in b.SEEDS]}
    data = {'datasets': [{'dataset': d, 'sha256': '1' * 64} for d in b.DATASETS]}
    paths['evaluation_registry'].write_text(json.dumps(registry))
    paths['dataset_manifest'].write_text(json.dumps(data))
    contract = {'registry_sha256': b.sha256_file(paths['evaluation_registry']),
                'dataset_manifest_sha256': b.sha256_file(paths['dataset_manifest']),
                'dataset_sha256': {d: '1' * 64 for d in b.DATASETS},
                'approval': {'test_inference_authorized': True, 'retraining_authorized': False,
                             'datasets': b.DATASETS, 'models': [b.B], 'seeds': b.SEEDS,
                             'allowed_splits': ['validation', 'test']}}
    paths['execution_contract'].write_text(json.dumps(contract))
    seal = {'files': {str(path.relative_to(tmp_path)): b.sha256_file(path) for path in [runner, script]}}
    (here / 'code_seal.json').write_text(json.dumps(seal))
    manifest = {'execution_contract_path': 'execution_contract.json',
                'registry_path': 'evaluation_registry.json', 'dataset_manifest_path': 'dataset_manifest.json',
                'execution_contract_sha256': b.sha256_file(paths['execution_contract']),
                'contract_sha256': b.sha256_file(paths['execution_contract']),
                'registry_sha256': contract['registry_sha256'],
                'dataset_manifest_sha256': contract['dataset_manifest_sha256'],
                'code_seal_sha256': b.sha256_file(here / 'code_seal.json')}
    args = (manifest, here / 'run_manifest.json', paths['execution_contract'],
            paths['evaluation_registry'], paths['dataset_manifest'])
    return args, paths, contract


def test_campaign_bindings_and_seal_reject_stale_artifacts(tmp_path, monkeypatch):
    args, paths, _ = fixture_binding(tmp_path, monkeypatch)
    assert len(b.load_b_bindings(*args)['rows']) == 9
    paths['dataset_manifest'].write_text('{}')
    with pytest.raises(ValueError, match='SHA mismatch'):
        b.load_b_bindings(*args)


def test_edited_analyzer_rejected_by_b_seal(tmp_path, monkeypatch):
    args, _, _ = fixture_binding(tmp_path, monkeypatch)
    Path(b.__file__).write_text('edited after seal')
    with pytest.raises(ValueError, match='differs from seal'):
        b.load_b_bindings(*args)


def test_retraining_approval_or_extra_scope_is_rejected(tmp_path, monkeypatch):
    args, paths, contract = fixture_binding(tmp_path, monkeypatch)
    contract['approval']['retraining_authorized'] = True
    paths['execution_contract'].write_text(json.dumps(contract))
    args[0]['contract_sha256'] = args[0]['execution_contract_sha256'] = b.sha256_file(paths['execution_contract'])
    with pytest.raises(ValueError, match='approval scope'):
        b.load_b_bindings(*args)


@pytest.mark.parametrize('field,value', [('selected_epoch', 8), ('checkpoint_file_sha256', 'x' * 64),
                                        ('state_tensor_sha256', 'x' * 64), ('source_closure_sha256', 'x' * 64),
                                        ('loader', {}), ('retrained', True), ('max_batches', 1)])
def test_selected_checkpoint_and_frozen_loader_are_bound(field, value):
    dataset = b.DATASETS[0]
    key = (dataset, b.B, 42)
    selected = {'checkpoint_path': 'frozen/cp.pt', 'checkpoint_file_sha256': 'a' * 64,
                'state_tensor_sha256': 'b' * 64, 'selected_epoch': 7, 'evaluator_source_bundle': 'original'}
    loader = {'max_seq_len': 64, 'lookback_days': 7}
    contract = {'registry_sha256': 'c' * 64, 'dataset_manifest_sha256': 'd' * 64,
                'approval': {'datasets': b.DATASETS, 'models': [b.B], 'seeds': b.SEEDS,
                             'allowed_splits': ['test'], 'test_inference_authorized': True}}
    bindings = {'contract': contract, 'contract_sha256': 'e' * 64, 'runner_sha256': 'f' * 64,
                'registry': {'bundles': {'original': {'source_closure_sha256': '0' * 64,
                             'datasets': [{'dataset_id': dataset, 'loader': loader}]}}},
                'rows': {key: selected},
                'datasets': {dataset: {'sha256': '1' * 64, 'populations': {'test': {'target_count': 32}}}}}
    receipt = {**selected, 'split': 'test', 'registry_sha256': 'c' * 64,
               'dataset_manifest_sha256': 'd' * 64, 'contract_sha256': 'e' * 64,
               'runner_sha256': 'f' * 64, 'data_file_sha256': '1' * 64,
               'source_closure_sha256': '0' * 64, 'full_population': True,
               'evaluation_scope': 'full_population', 'retrained': False, 'max_batches': None,
               'loader': loader, 'expected_target_count': 32, 'prediction_rows': 32,
               'truth_digest_fields': b.legacy.TRUTH_FIELDS,
               'schema': {key: str(dtype) for key, dtype in b.legacy.SCHEMA.items()},
               'qualification': {'parameters_unchanged': True, 'finite_outputs': True}}
    b.legacy.verify_receipt_binding(receipt, key, bindings)
    receipt[field] = value
    with pytest.raises(ValueError, match='frozen binding'):
        b.legacy.verify_receipt_binding(receipt, key, bindings)


@pytest.fixture(scope='module')
def synthetic_analysis(tmp_path_factory):
    root = tmp_path_factory.mktemp('small_b_synthetic_panel')
    results, validation = {}, {}
    for dataset in b.DATASETS:
        conditions, receipts, _, population = fixture_panel(root, dataset)
        units, validation[dataset] = b.stream_paired_dataset(conditions, receipts, dataset, population)
        results[dataset] = {
            name: b.bootstrap_b(units[key].counts, units[key].sums, dataset=dataset, kind=kind,
                                cluster_key=key, block_hours=block)
            for name, (kind, key, block) in b.paired.analysis_specs(dataset).items()}
    return {'status': 'complete', 'split': 'test', 'scope': b.SCOPE, 'exploratory': True,
            'interpretation': b.INTERPRETATION, 'new_b_conditions': 9,
            'legacy_comparison_conditions': 63, 'learned_conditions': 72,
            'representative_model': b.REPRESENTATIVE, 'representative_replaced': False,
            'prediction_ensemble': False, 'retrained': False,
            'validation': validation, 'results': results}


def test_renderer_covers_all_comparisons_and_preserves_outputs(tmp_path, synthetic_analysis):
    path = tmp_path / 'synthetic_analysis.json'
    path.write_text(json.dumps(synthetic_analysis))
    output = tmp_path / 'review'
    receipt = renderer.write_outputs(path, output)
    assert receipt['analysis_sha256'] == b.sha256_file(path)
    assert receipt['bootstrap_performed'] is False
    assert receipt['prediction_files_read'] is False
    for filename, digest in receipt['output_sha256'].items():
        assert b.sha256_file(output / filename) == digest
    markdown = (output / 'summary.md').read_text()
    latex = (output / 'tables.tex').read_text()
    assert latex.count('\\begin{table}') == 8
    assert '자동 교체하지 않습니다' in markdown
    assert 'Taxi의 168시간' in markdown and '미산출' in markdown
    assert '다중 비교 보정' in markdown
    assert '98.75' not in latex
    with pytest.raises(FileExistsError):
        renderer.write_outputs(path, output)
    assert (output / 'summary.md').read_text() == markdown


@pytest.mark.parametrize('field,value', [('status', 'pending_conditions'),
                                        ('split', 'validation'), ('learned_conditions', 71),
                                        ('new_b_conditions', 8), ('exploratory', False),
                                        ('representative_replaced', True), ('retrained', True)])
def test_renderer_rejects_incomplete_or_reinterpreted_results(synthetic_analysis, field, value):
    analysis = copy.deepcopy(synthetic_analysis)
    analysis[field] = value
    with pytest.raises(ValueError):
        renderer.validate_analysis(analysis)


def test_renderer_rejects_legacy_familywise_correction(synthetic_analysis):
    analysis = copy.deepcopy(synthetic_analysis)
    panel = analysis['results'][b.DATASETS[1]]['primary_site']
    panel['intervals']['rmtpp']['qty_rmse']['four_anchor_bonferroni_98_75_percentile'] = [-1., 1.]
    with pytest.raises(ValueError, match='pointwise exploratory'):
        renderer.validate_analysis(analysis)


def test_renderer_rejects_mismatched_seed_mean_and_population(synthetic_analysis):
    analysis = copy.deepcopy(synthetic_analysis)
    analysis['results'][b.DATASETS[1]]['primary_site']['means'][b.B]['time_nll'] += .1
    with pytest.raises(ValueError, match='Mean/sample SD'):
        renderer.validate_analysis(analysis)
    analysis = copy.deepcopy(synthetic_analysis)
    analysis['validation'][b.DATASETS[1]]['conditions'].pop()
    with pytest.raises(ValueError, match='24 learned conditions'):
        renderer.validate_analysis(analysis)
