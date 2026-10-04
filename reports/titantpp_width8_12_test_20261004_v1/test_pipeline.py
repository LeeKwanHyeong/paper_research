"""Focused synthetic contract/gate tests; no research data, remote, or GPU access."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


p = module('width8_12_pipeline', 'pipeline.py')
pre = module('width8_12_prepare', 'prepare.py')
ev = module('width8_12_evaluate', 'evaluate.py')


def setup(tmp_path):
    registry = {'rows': [{'dataset': 'synthetic', 'model': 'model', 'seed': seed} for seed in (42, 52)]}
    for name, value in [('evaluation_registry.json', registry), ('dataset_manifest.json', {}), ('selected_binding.json', {}), ('code_seal.json', {'files': {}})]:
        (tmp_path / name).write_text(json.dumps(value))
    contract = {'registry_sha256': p.sha(tmp_path / 'evaluation_registry.json'),
        'dataset_manifest_sha256': p.sha(tmp_path / 'dataset_manifest.json'),
        'selected_binding_sha256': p.sha(tmp_path / 'selected_binding.json'),
        'resources': {'campaign_timeout_seconds': 60, 'root': str(tmp_path)}, 'environment': {}, 'validation_gate': {'conditions': 2}}
    (tmp_path / 'execution_contract.json').write_text(json.dumps(contract))
    return SimpleNamespace(campaign=tmp_path, root=tmp_path, device='cpu')


def test_validation_failure_stops_before_test(tmp_path, monkeypatch):
    args = setup(tmp_path)
    calls = []
    def fail(row, split, *rest):
        calls.append(split)
        raise ValueError('frozen validation mismatch')
    monkeypatch.setattr(p, 'run_condition', fail)
    with pytest.raises(ValueError, match='frozen validation mismatch'):
        p.main(args)
    assert calls == ['validation']
    assert not (tmp_path / 'validation_gate.json').exists()
    assert p.read(tmp_path / 'pipeline_status.json')['status'] == 'failed'


def test_all_validation_precedes_test_and_existing_attempt_preserved(tmp_path, monkeypatch):
    args = setup(tmp_path)
    calls = []
    def complete(row, split, *rest):
        if split == 'test':
            gate = p.read(tmp_path / 'validation_gate.json')
            assert gate['passed'] and gate['conditions'] == 2
        calls.append((split, row['seed']))
        return {'condition': p.identity(row), 'split': split}
    monkeypatch.setattr(p, 'run_condition', complete)
    p.main(args)
    assert calls == [('validation', 42), ('validation', 52), ('test', 42), ('test', 52)]
    before = (tmp_path / 'pipeline_status.json').read_bytes()
    with pytest.raises(ValueError, match='Never overwrite'):
        p.main(args)
    assert (tmp_path / 'pipeline_status.json').read_bytes() == before


@pytest.mark.parametrize('name', ['evaluation_registry.json', 'dataset_manifest.json', 'selected_binding.json'])
def test_modified_input_stops_before_worker(tmp_path, monkeypatch, name):
    args = setup(tmp_path)
    with (tmp_path / name).open('a') as stream:
        stream.write(' ')
    monkeypatch.setattr(p, 'run_condition', lambda *args: pytest.fail('Worker must not start'))
    with pytest.raises(ValueError, match='SHA mismatch'):
        p.main(args)


def test_first_minimum_keeps_earliest_tie_and_rejects_nonfinite_history():
    history = {'history': [{'epoch': 1, 'val_qty_rmse': 4., 'train_all_finite': True}, {'epoch': 2, 'val_qty_rmse': 3., 'train_all_finite': True}, {'epoch': 3, 'val_qty_rmse': 3., 'train_all_finite': True}]}
    assert pre.first_minimum(history)['epoch'] == 2
    with pytest.raises(ValueError, match='contiguous'):
        pre.first_minimum({'history': list(reversed(history['history']))})
    history['history'][0]['val_qty_rmse'] = float('nan')
    with pytest.raises(ValueError, match='all finite'):
        pre.first_minimum(history)


def test_test_requires_identity_bound_contract():
    kwargs = {'split': 'test', 'dataset': 'synthetic', 'model': 'model', 'seed': 42,
              'registry_sha': 'registry', 'manifest_sha': 'manifest', 'data_sha': 'data'}
    with pytest.raises(ValueError, match='requires --contract'):
        ev.validate_contract(None, **kwargs)
    contract = {'schema_version': 1, 'approval': {'retraining_authorized': False, 'test_inference_authorized': True,
        'allowed_splits': ['validation', 'test'], 'datasets': ['synthetic'], 'models': ['model'], 'seeds': [42]},
        'registry_sha256': 'registry', 'dataset_manifest_sha256': 'manifest', 'dataset_sha256': {'synthetic': 'data'}}
    ev.validate_contract(contract, **kwargs)
    with pytest.raises(ValueError, match='Registry contract SHA mismatch'):
        ev.validate_contract({**contract, 'registry_sha256': 'changed'}, **kwargs)


def test_aggregate_writer_records_only_metrics_and_digests(tmp_path):
    writer = ev.AggregateWriter(tmp_path, [1., 2., 3., 4.])
    rows = []
    for index, (quantity, prediction) in enumerate([(4., 1.), (5., 3.)]):
        row = {key: None for key in ev.SCHEMA}
        row.update(target_index=index, target_id=str(index), entity_id='entity', seq=index,
                   split='test', recorded_date='date', raw_quantity=quantity,
                   predicted_raw_quantity=prediction, recorded_gap=1., time_nll=.5,
                   history_length=1, last_quantity=1., mean_quantity=1.)
        rows.append(row)
    writer.add(rows)
    writer.flush()
    result = writer.metrics.finish()
    assert result['count'] == 2 and result['tail']['count'] == 1
    assert result['tail']['qty_rmse'] == 2. and result['tail']['qty_mae'] == 2.
    assert writer.rows == 2 and writer.parts == [] and list(tmp_path.iterdir()) == []
    assert result['tail_definition'] == 'raw_quantity > quantity_boundaries[-1]'
