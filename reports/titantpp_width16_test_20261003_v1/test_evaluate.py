"""Synthetic contract, streaming-output, leakage and isolation regressions."""
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
import torch

SPEC = importlib.util.spec_from_file_location('legacy_evaluate', Path(__file__).with_name('evaluate.py'))
e = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e)


def approval():
    return {'schema_version': 1, 'approval': {
        'test_inference_authorized': True, 'retraining_authorized': False,
        'allowed_splits': ['validation', 'test'], 'datasets': ['synthetic'],
        'models': ['rmtpp'], 'seeds': [42]}, 'registry_sha256': 'r',
        'dataset_manifest_sha256': 'm', 'dataset_sha256': {'synthetic': 'd'}}


def contract_arguments():
    return dict(split='test', dataset='synthetic', model='rmtpp', seed=42,
                registry_sha='r', manifest_sha='m', data_sha='d')


def test_test_requires_explicit_identity_bound_approval():
    args = contract_arguments()
    with pytest.raises(ValueError, match='requires --contract'):
        e.validate_contract(None, **args)
    e.validate_contract(approval(), **args)
    for key in ('registry_sha', 'manifest_sha', 'data_sha', 'dataset', 'model', 'seed'):
        altered = {**args, key: 'other'}
        with pytest.raises(ValueError):
            e.validate_contract(approval(), **altered)
    for key, value in [('test_inference_authorized', False), ('test_inference_authorized', 'true'),
                       ('retraining_authorized', True), ('allowed_splits', ['validation'])]:
        contract = approval()
        contract['approval'][key] = value
        with pytest.raises(ValueError):
            e.validate_contract(contract, **args)
    e.validate_contract(None, **{**args, 'split': 'validation'})


def meta(index=0, split='validation', date='2025-01-01'):
    row = {'oper_part_no': 'site::item', 'seq': index + 1,
           'chronological_split': split, 'demand_dt': date}
    return e.target_metadata('intermittent_frozen_5000', 'd' * 64, index,
                             'site::item', index + 1, split, row)


def prediction(index=0):
    return {**meta(index), 'raw_quantity': 3., 'predicted_raw_quantity': 2.,
            'recorded_gap': 1., 'time_nll': .5, 'history_length': 2,
            'last_quantity': 1., 'mean_quantity': 1.5}


def test_identity_includes_dataset_bytes_split_recorded_date():
    m = meta()
    expected = hashlib.sha256(e.json_bytes(['intermittent_frozen_5000', 'd' * 64,
                'site::item', 1, 'validation', '2025-01-01'])).hexdigest()
    assert m['target_id'] == expected
    assert m['site_id'] == 'site' and m['time_bucket'] is None
    assert len({m['target_id'], meta(split='test')['target_id'],
                meta(date='2025-01-02')['target_id']}) == 3
    with pytest.raises(ValueError, match='Wrong target split'):
        e.target_metadata('d', 'd' * 64, 0, 'x', 1, 'test', {
            'oper_part_no': 'x', 'seq': 1, 'demand_dt': 'date', 'chronological_split': 'validation'})


def test_streamed_parts_roundtrip_nullable_schema_and_incremental_digest(tmp_path):
    writer = e.PredictionWriter(tmp_path, part_rows=2)
    rows = [prediction(i) for i in range(5)]
    writer.add(rows[:3])
    assert len(writer.parts) == 1 and len(writer.buffer) == 1
    assert not (tmp_path / 'receipt.json').exists()
    writer.add(rows[3:])
    writer.flush()
    assert [p['rows'] for p in writer.parts] == [2, 2, 1]
    assert all(p['schema'] == e.SCHEMA for p in writer.parts)
    combined = pl.concat([pl.read_parquet(tmp_path / p['path']) for p in writer.parts])
    assert combined.to_dicts() == rows
    expected_targets, expected_truth = e.row_digests(rows)
    assert writer.targets.hexdigest() == expected_targets
    assert writer.truth.hexdigest() == expected_truth
    for part in writer.parts:
        data = pl.read_parquet(tmp_path / part['path']).to_dicts()
        assert e.row_digests(data) == (part['target_identity_sha256'], part['truth_sha256'])
        assert e.sha256_file(tmp_path / part['path']) == part['sha256']
    assert json.loads((tmp_path / 'progress.json').read_text())['prediction_rows'] == 5


@pytest.mark.parametrize('field,value', [('target_index', 2), ('raw_quantity', float('nan')),
                                      ('predicted_raw_quantity', -1.), ('history_length', 0)])
def test_writer_rejects_gap_nonfinite_and_invalid_history(tmp_path, field, value):
    writer = e.PredictionWriter(tmp_path)
    row = prediction()
    row[field] = value
    with pytest.raises(ValueError):
        writer.add([row])
    assert writer.rows == 0


def test_select_condition_rejects_missing_or_duplicate_binding():
    row = {'dataset': 'd', 'model': 'm', 'seed': 42, 'endpoint': 'selected',
           'evaluator_source_bundle': 'b'}
    registry = {'rows': [row], 'bundles': {'b': {'datasets': [{'dataset_id': 'd'}]}}}
    assert e.select_condition(registry, 'd', 'm', 42)[0] == row
    for invalid in ([], [row, row]):
        with pytest.raises(ValueError, match='exactly one'):
            e.select_condition({**registry, 'rows': invalid}, 'd', 'm', 42)


def test_frozen_source_hash_and_import_escape_are_rejected(tmp_path, monkeypatch):
    source = tmp_path / 'frozen'
    source.mkdir()
    file = source / 'original.py'
    file.write_text('original')
    bundle = {'source_root': 'frozen', 'source_files': {'original.py': e.sha256_file(file)}}
    assert e.verify_source(tmp_path, bundle) == source
    file.write_text('changed')
    with pytest.raises(ValueError, match='Frozen source changed'):
        e.verify_source(tmp_path, bundle)
    monkeypatch.setitem(sys.modules, 'models.current_fake', SimpleNamespace(__file__=str(tmp_path / 'current.py')))
    with pytest.raises(ValueError, match='current/nonfrozen source'):
        e.verify_imports(source, bundle)


def test_qualification_detects_target_quantity_leakage():
    dts = torch.tensor([[0., 1., 2.], [1., 1., 3.]])
    mask = torch.tensor([[False, True, True], [True, True, True]])
    quantities = torch.tensor([[0., 2., 4.], [1., 2., 5.]])
    def causal(model, dts, mask, quantities, **kwargs):
        return {'pred_qty': quantities[:, -2], 'time_loss': dts[:, -1] ** 2}
    value = causal(None, dts, mask, quantities)
    assert all(e.qualify(torch, None, causal, dts, mask, quantities, value, 30).values())
    def leaky(model, dts, mask, quantities, **kwargs):
        return {'pred_qty': quantities[:, -1], 'time_loss': dts[:, -1] ** 2}
    with pytest.raises(ValueError, match='Qualification failed'):
        e.qualify(torch, None, leaky, dts, mask, quantities,
                  leaky(None, dts, mask, quantities), None)


def test_worker_failure_persists_and_existing_output_cannot_be_reused(tmp_path):
    output = tmp_path / 'new_attempt'
    args = e.parser().parse_args(['--root', str(tmp_path), '--registry', 'missing.json',
        '--dataset-manifest', 'missing-data.json', '--dataset', 'synthetic', '--model', 'rmtpp',
        '--seed', '42', '--split', 'validation', '--output', str(output)])
    with pytest.raises(FileNotFoundError):
        e.run_worker(args)
    failure = e.read_json(output / 'failure.json')
    assert failure['status'] == 'failed' and not (output / 'receipt.json').exists()
    saved = (output / 'failure.json').read_bytes()
    with pytest.raises(FileExistsError):
        e.run_worker(args)
    assert (output / 'failure.json').read_bytes() == saved


def test_manifest_cannot_substitute_unbound_dataset_bytes(tmp_path):
    registry = {'rows': [{'dataset': 'synthetic', 'model': 'rmtpp', 'seed': 42,
        'endpoint': 'selected', 'evaluator_source_bundle': 'b'}],
        'bundles': {'b': {'datasets': [{'dataset_id': 'synthetic',
        'loader': {'batch_size': 128}, 'inherited_data_identity': {'data': {'sha256': 'a' * 64}}}]}}}
    (tmp_path / 'registry.json').write_text(json.dumps(registry))
    (tmp_path / 'data.json').write_text(json.dumps({'datasets': [
        {'dataset': 'synthetic', 'path': 'not-read.parquet', 'sha256': 'b' * 64}]}))
    args = e.parser().parse_args(['--root', str(tmp_path), '--registry', 'registry.json',
        '--dataset-manifest', 'data.json', '--dataset', 'synthetic', '--model', 'rmtpp',
        '--seed', '42', '--split', 'validation', '--output', str(tmp_path / 'output')])
    with pytest.raises(ValueError, match='not a frozen'):
        e.validate_inputs(args)


def test_deadline_fails_before_model_import_and_preserves_failure(tmp_path, monkeypatch):
    args = SimpleNamespace(output=tmp_path / 'timed-out', deadline_seconds=1e-12,
                           root=tmp_path, dataset='synthetic', model='rmtpp', seed=42)
    checkpoint = tmp_path / 'checkpoint'
    dataset = tmp_path / 'dataset'
    checkpoint.write_text('immutable weights')
    dataset.write_text('immutable data')
    row = {'checkpoint_path': 'checkpoint', 'checkpoint_file_sha256': e.sha256_file(checkpoint)}
    data = {'path': 'dataset', 'sha256': e.sha256_file(dataset)}
    monkeypatch.setattr(e, 'validate_inputs', lambda _: (tmp_path, row, {}, {}, data, {}))
    monkeypatch.setattr(e, 'verify_source', lambda *_: tmp_path)
    with pytest.raises(TimeoutError, match='Deadline exceeded'):
        e.run_worker(args)
    failure = e.read_json(args.output / 'failure.json')
    assert failure['exception'] == 'TimeoutError' and failure['prediction_rows'] == 0
    assert not (args.output / 'receipt.json').exists()
